"""
tests/test_checkpoint_and_smoke.py

Unit and integration tests for:
1. save_checkpoint and load_checkpoint lifecycle
2. Head weights and training state fidelity after reload
3. Smoke test sanity verification logic
"""

import os
import tempfile
import pytest
import torch
import torch.nn as nn
from transformers import AutoTokenizer

from fine_tuning_engine import (
    save_checkpoint,
    load_checkpoint,
    run_smoke_test,
)


class MockBackbone(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, input_ids, attention_mask=None, output_hidden_states=True, return_dict=True):
        hidden = self.linear(self.embed(input_ids))
        return type("BackboneOutput", (), {"last_hidden_state": hidden, "hidden_states": [hidden]})()


class MockHybridModel(nn.Module):
    """Lightweight mock model for fast unit testing of checkpointing and smoke test."""
    def __init__(self, hidden_dim: int = 64, proj_dim: int = 32, vocab_size: int = 200000):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.proj_dim = proj_dim
        self.base_model_name = "mock_model"
        self.backbone = MockBackbone(vocab_size, hidden_dim)
        self.w_noul = nn.Linear(hidden_dim, 2)
        self.w_score = nn.Linear(hidden_dim, 10)
        self.q_proj = nn.Linear(hidden_dim, proj_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, proj_dim, bias=False)
        self.scale = 1.0 / (proj_dim ** 0.5)
        self.tau_choice = nn.Parameter(torch.ones(1) * 1.0)
        self.tau_noul = nn.Parameter(torch.ones(1) * 1.0)
        self.tau_score = nn.Parameter(torch.ones(1) * 1.0)
        self.tokenizer = None

    def forward(self, input_ids, attention_mask, decide_token_idx, task_types, candidate_spans=None):
        backbone_out = self.backbone(input_ids, attention_mask)
        hidden_states = backbone_out.last_hidden_state
        batch_size = input_ids.size(0)
        batch_idx = torch.arange(batch_size, device=input_ids.device)
        h_decide = hidden_states[batch_idx, decide_token_idx]

        outputs = {
            "noul_logits": (self.w_noul(h_decide).float() / self.tau_noul),
            "score_logits": (self.w_score(h_decide).float() / self.tau_score),
            "choice_logits": [],
        }

        for b in range(batch_size):
            if task_types[b] == "choice" and candidate_spans is not None:
                spans = candidate_spans[b]
                cand_reps = [
                    hidden_states[b, start:end, :].mean(dim=0)
                    for (start, end) in spans
                    if end > start
                ]
                if cand_reps:
                    h_cands = torch.stack(cand_reps, dim=0)
                    q = self.q_proj(h_decide[b]).unsqueeze(0)
                    k = self.k_proj(h_cands)
                    logits = (torch.mm(q, k.t()).squeeze(0) * self.scale).float() / self.tau_choice
                    outputs["choice_logits"].append(logits)
                else:
                    outputs["choice_logits"].append(torch.empty(0, device=input_ids.device))
            else:
                outputs["choice_logits"].append(torch.empty(0, device=input_ids.device))

        return outputs


def test_save_and_load_checkpoint_lifecycle():
    """Verifies complete save_checkpoint -> load_checkpoint state restoration."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model = MockHybridModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)

        # Alter parameters to test restoration fidelity
        with torch.no_grad():
            model.w_noul.weight.fill_(3.1415)
            model.tau_choice.fill_(2.7182)

        ckpt_dir = os.path.join(tmpdir, "checkpoint-step-100")
        save_checkpoint(
            checkpoint_dir=ckpt_dir,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=2,
            step_count=100,
        )

        assert os.path.exists(os.path.join(ckpt_dir, "hybrid_head.safetensors"))
        assert os.path.exists(os.path.join(ckpt_dir, "training_state.pt"))

        # Create a new model and restore
        new_model = MockHybridModel()
        new_optimizer = torch.optim.AdamW(new_model.parameters(), lr=1e-3)
        new_scheduler = torch.optim.lr_scheduler.StepLR(new_optimizer, step_size=1)

        resumed_epoch, resumed_step = load_checkpoint(
            checkpoint_dir=ckpt_dir,
            model=new_model,
            optimizer=new_optimizer,
            scheduler=new_scheduler,
            device="cpu",
        )

        assert resumed_epoch == 2
        assert resumed_step == 100
        assert torch.allclose(new_model.w_noul.weight, torch.tensor(3.1415))
        assert torch.allclose(new_model.tau_choice, torch.tensor(2.7182))


def test_run_smoke_test_execution():
    """Verifies that run_smoke_test executes all 5 stages without errors."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model = MockHybridModel(hidden_dim=1536, proj_dim=512, vocab_size=200000)
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

        smoke_output = os.path.join(tmpdir, "smoke_run")
        success = run_smoke_test(
            model=model,
            tokenizer=tokenizer,
            device="cpu",
            output_dir=smoke_output,
        )
        assert success is True
        # Verify temporary directory cleaned up
        assert not os.path.exists(smoke_output)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA for bitsandbytes")
def test_bnb_adamw8bit_checkpoint_resume_and_step():
    """Verifies that bitsandbytes 8-bit AdamW can resume from checkpoint and execute step without C++ invalid argument crashes."""
    import bitsandbytes as bnb
    with tempfile.TemporaryDirectory() as tmpdir:
        p = nn.Parameter(torch.randn(8960, 32, device="cuda"))
        opt = bnb.optim.AdamW8bit([p], lr=1e-4)

        # Do a step and save
        p.grad = torch.randn_like(p)
        opt.step()

        state = {
            "epoch": 1,
            "step_count": 10,
            "optimizer_state_dict": opt.state_dict(),
        }
        ckpt_file = os.path.join(tmpdir, "training_state.pt")
        torch.save(state, ckpt_file)

        # Restore into a new optimizer
        new_p = nn.Parameter(torch.randn(8960, 32, device="cuda"))
        new_opt = bnb.optim.AdamW8bit([new_p], lr=1e-4)
        loaded = torch.load(ckpt_file, map_location="cuda")
        new_opt.load_state_dict(loaded["optimizer_state_dict"])

        # Execute step after resumption
        new_p.grad = torch.randn_like(new_p)
        new_opt.step()
        assert new_opt.state[new_p]["step"] == 2

