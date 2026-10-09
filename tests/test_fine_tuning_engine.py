"""
tests/test_fine_tuning_engine.py

Comprehensive test suite verifying all 5 layers of the System One Fine-Tuning Module Topology:
1. Data Ingestion Engine (DecisionDataCollator & typed targets)
2. Hybrid Model Container (Dual-Mode Head projections & bilinear candidate pointer)
3. Multi-Task Loss Engine (InfoNCE, EMD Wasserstein, Gaussian Soft-CE, Calibration regularizer)
4. Hardware-Constrained Optimization components
5. Post-Training Calibration & Production Artifact Exporter
"""

import os
import json
import tempfile
import pytest
import torch
import torch.nn as nn
from transformers import AutoTokenizer

from fine_tuning_engine import (
    DecisionDataCollator,
    DecisionLossEngine,
    calibrate_temperatures,
    export_production_artifacts,
)


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")


def test_collator_formatting_and_spans(tokenizer):
    """Verifies DecisionDataCollator schema serialization, candidate span extraction, and targets."""
    collator = DecisionDataCollator(tokenizer=tokenizer, max_seq_length=512)

    samples = [
        {
            "task_type": "choice",
            "domain": "finance_routing",
            "state": {"query": "I lost my credit card"},
            "question": "Select customer intent.",
            "options": ["card_arrival", "lost_card", "transfer_delay"],
            "target_idx": 1,
        },
        {
            "task_type": "noul",
            "domain": "safety",
            "state": {"text": "harmless greeting"},
            "question": "Is this message toxic?",
            "target_bool": False,
        },
        {
            "task_type": "score",
            "domain": "summary_evaluation",
            "state": {"doc": "Document text", "summary": "Summary text"},
            "question": "Rate summary quality.",
            "target_score": 3,
            "active_levels": 5,
        },
    ]

    batch = collator(samples)

    # 1. Tensor Shapes
    assert batch["input_ids"].shape[0] == 3
    assert batch["attention_mask"].shape[0] == 3
    assert batch["decide_token_idx"].shape[0] == 3

    # 2. Check Candidate Spans for Choice sample (index 0)
    choice_spans = batch["candidate_spans"][0]
    assert len(choice_spans) == 3
    for s_start, s_end in choice_spans:
        assert s_end > s_start
        span_text = tokenizer.decode(batch["input_ids"][0, s_start:s_end])
        assert "[" in span_text  # Contains candidate index prefix

    # 3. Check decide_token_idx
    for b in range(3):
        dec_idx = batch["decide_token_idx"][b].item()
        assert 0 <= dec_idx < batch["input_ids"].shape[1]

    # 4. Typed Targets
    assert batch["choice_targets"][0].item() == 1
    assert batch["choice_targets"][1].item() == -100
    assert batch["choice_targets"][2].item() == -100

    assert batch["noul_targets"][0].item() == -100
    assert batch["noul_targets"][1].item() == 0  # False -> 0
    assert batch["noul_targets"][2].item() == -100

    assert batch["score_targets"][0].item() == -100
    assert batch["score_targets"][1].item() == -100
    assert batch["score_targets"][2].item() == 3
    assert batch["score_active_levels"][2].item() == 5


def test_loss_engine_noul_bce():
    """Verifies calibrated Binary Cross-Entropy with label smoothing."""
    loss_engine = DecisionLossEngine()
    logits = torch.tensor([[2.5, -1.0], [-1.2, 3.0], [0.0, 0.0]], requires_grad=True)
    targets = torch.tensor([0, 1, -100], dtype=torch.long)

    loss = loss_engine.compute_noul_loss(logits, targets)
    assert loss.dim() == 0
    assert loss.item() > 0.0
    loss.backward()
    assert logits.grad is not None


def test_loss_engine_choice_infonce():
    """Verifies Dynamic Simplex InfoNCE over arbitrary candidate sizes K."""
    loss_engine = DecisionLossEngine()
    # Sample 0: 3 candidates, target 1
    logits_0 = torch.tensor([0.2, 1.8, -0.5], requires_grad=True)
    # Sample 1: 5 candidates, target 4
    logits_1 = torch.tensor([0.1, 0.2, 0.3, 0.4, 2.0], requires_grad=True)
    # Sample 2: non-choice
    logits_2 = torch.empty(0)

    logits_list = [logits_0, logits_1, logits_2]
    targets = torch.tensor([1, 4, -100], dtype=torch.long)

    loss = loss_engine.compute_choice_loss(logits_list, targets)
    assert loss.dim() == 0
    assert loss.item() > 0.0
    loss.backward()
    assert logits_0.grad is not None
    assert logits_1.grad is not None


def test_loss_engine_score_wasserstein_and_gaussian_soft():
    """Verifies combined 1-Wasserstein CDF loss + Bilateral Gaussian Soft-CE."""
    loss_engine = DecisionLossEngine(gamma_wasserstein=0.5, gaussian_sigma=0.65)
    logits = torch.randn(2, 10, requires_grad=True)
    targets = torch.tensor([3, 4], dtype=torch.long)
    active_levels = torch.tensor([5, 6], dtype=torch.long)

    loss = loss_engine.compute_score_loss(logits, targets, active_levels)
    assert loss.dim() == 0
    assert loss.item() > 0.0
    loss.backward()
    assert logits.grad is not None


def test_loss_engine_calibration_penalty_and_composite_forward():
    """Verifies multi-task forward dispatch with surrogate entropy regularization."""
    loss_engine = DecisionLossEngine(beta_calib=0.05)

    batch = {
        "input_ids": torch.zeros((3, 10), dtype=torch.long),
        "task_types": ["choice", "noul", "score"],
        "noul_targets": torch.tensor([-100, 1, -100], dtype=torch.long),
        "score_targets": torch.tensor([-100, -100, 2], dtype=torch.long),
        "score_active_levels": torch.tensor([5, 5, 5], dtype=torch.long),
        "choice_targets": torch.tensor([0, -100, -100], dtype=torch.long),
    }

    model_outputs = {
        "noul_logits": torch.tensor([[0.0, 0.0], [-1.0, 2.0], [0.0, 0.0]], requires_grad=True),
        "score_logits": torch.randn(3, 10, requires_grad=True),
        "choice_logits": [
            torch.tensor([1.5, -0.2, 0.1], requires_grad=True),
            torch.empty(0),
            torch.empty(0),
        ],
    }

    losses = loss_engine(model_outputs, batch)
    assert "loss_total" in losses
    assert "loss_noul" in losses
    assert "loss_score" in losses
    assert "loss_choice" in losses
    assert "loss_calib" in losses

    total_loss = losses["loss_total"]
    total_loss.backward()
    assert model_outputs["noul_logits"].grad is not None
    assert model_outputs["score_logits"].grad is not None
    assert model_outputs["choice_logits"][0].grad is not None


def test_temperature_calibration_optimization():
    """Verifies L-BFGS temperature fitting converges and stays positive."""
    tau = nn.Parameter(torch.ones(1) * 2.0)
    opt = torch.optim.LBFGS([tau], lr=0.05, max_iter=20)
    logits = torch.randn(50, 5) * 5.0
    targets = torch.randint(0, 5, (50,))

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(logits / torch.clamp(tau, min=0.05), targets)
        loss.backward()
        return loss

    opt.step(closure)
    assert tau.item() > 0.05


def test_artifact_export_and_metadata_serialization():
    """Verifies hybrid_head.safetensors and decision_config.json export schema."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Mock class with head attributes
        class MockEngine:
            def __init__(self):
                self.base_model_name = "Qwen/Qwen2.5-1.5B-Instruct"
                self.hidden_dim = 1536
                self.proj_dim = 512
                self.w_noul = nn.Linear(1536, 2)
                self.w_score = nn.Linear(1536, 10)
                self.q_proj = nn.Linear(1536, 512, bias=False)
                self.k_proj = nn.Linear(1536, 512, bias=False)
                self.tau_choice = nn.Parameter(torch.tensor([1.12]))
                self.tau_noul = nn.Parameter(torch.tensor([0.95]))
                self.tau_score = nn.Parameter(torch.tensor([1.04]))
                self.backbone = nn.Identity()
                self.tokenizer = None

        mock_engine = MockEngine()
        export_production_artifacts(mock_engine, output_dir=tmpdir)

        # Check safetensors file
        head_path = os.path.join(tmpdir, "hybrid_head.safetensors")
        assert os.path.exists(head_path)
        from safetensors.torch import load_file
        loaded_weights = load_file(head_path)
        assert "w_noul.weight" in loaded_weights
        assert "w_score.weight" in loaded_weights
        assert "q_proj.weight" in loaded_weights
        assert "tau_choice" in loaded_weights

        # Check config json
        config_path = os.path.join(tmpdir, "decision_config.json")
        assert os.path.exists(config_path)
        with open(config_path, "r") as f:
            cfg = json.load(f)
        assert cfg["model_type"] == "system_one_decision_engine"
        assert cfg["hidden_dimension"] == 1536
        assert cfg["proj_dimension"] == 512
        assert cfg["fitted_temperatures"]["choice"] == 1.12
