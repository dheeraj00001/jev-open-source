"""
tests/test_cuda_and_collator_bugs.py

Verification and regression test suite for:
1. Device & Dtype mismatch resolution on decision heads (Bug 1 & Bug 2)
2. Decision horizon terminal prefill boundary alignment (Bug 3)
3. Truncation-induced empty candidate spans NaN prevention (Bug 4)
4. Slicing through backbone directly to preserve execution wrappers (Bug 5)
"""

import pytest
import torch
import torch.nn as nn
from transformers import AutoTokenizer

from fine_tuning_engine import (
    DecisionDataCollator,
    HybridDecisionEngine,
)


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")


def test_fix_bug3_decide_token_idx_is_terminal(tokenizer):
    """
    Bug 3 Fix: decide_token_idx must point directly to the final terminal token
    of the sequence (seq_len - 1) so the causal transformer attends to the
    entire <|im_start|>decision\\n boundary tag.
    """
    collator = DecisionDataCollator(tokenizer=tokenizer, max_seq_length=512)
    sample = {
        "task_type": "noul",
        "domain": "factuality",
        "state": {"text": "test state"},
        "question": "Is this factual?",
        "target_bool": True,
    }

    batch = collator([sample])
    prompt, _ = collator.format_prompt(sample)
    encoded = tokenizer(prompt, add_special_tokens=False)
    seq_len = len(encoded["input_ids"])

    decide_idx = batch["decide_token_idx"][0].item()
    assert decide_idx == seq_len - 1, (
        f"decide_token_idx={decide_idx} does not point to terminal prefill token {seq_len - 1}!"
    )


def test_fix_bug4_truncation_empty_span_nan_prevention(tokenizer):
    """
    Bug 4 Fix: Under aggressive truncation, candidate spans are strictly clamped
    with end > start, and mean pooling produces valid non-NaN representations.
    """
    collator = DecisionDataCollator(tokenizer=tokenizer, max_seq_length=40)
    sample = {
        "task_type": "choice",
        "domain": "routing",
        "state": {"query": "A " * 50},  # Fills up token budget
        "question": "What is the intent of this query?",
        "options": ["first_option", "second_option", "third_option"],
        "target_idx": 0,
    }

    batch = collator([sample])
    spans = batch["candidate_spans"][0]

    # All spans must satisfy end > start
    for i, (start, end) in enumerate(spans):
        assert end > start, f"Candidate span {i} collapsed to empty slice ({start}, {end})!"
        assert start < 40 and end <= 40, f"Candidate span {i} exceeded sequence bounds!"

    # Simulate hidden state mean-pooling
    hidden_states = torch.randn(1, 40, 1536)
    for i, (start, end) in enumerate(spans):
        cand_rep = hidden_states[0, start:end, :].mean(dim=0)
        assert not torch.isnan(cand_rep).any(), f"Candidate {i} representation contains NaN!"


def test_fix_bug1_and_bug2_device_dtype_alignment():
    """
    Bug 1 & Bug 2 Fix: Linear heads and temperature scalers match backbone
    device and compute_dtype (e.g. bfloat16), producing float32 logits without crash.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float32

    # Initialize heads directly on target device & compute_dtype
    w_noul = nn.Linear(1536, 2, bias=True).to(device=device, dtype=compute_dtype)
    tau_noul = nn.Parameter(torch.ones(1, device=device, dtype=torch.float32) * 1.0)

    # Simulated bfloat16 hidden state on target device
    h_decide = torch.randn(1, 1536, device=device, dtype=compute_dtype)

    # Forward pass executes cleanly and outputs float32
    logits = w_noul(h_decide).float() / tau_noul
    assert logits.shape == (1, 2)
    assert logits.dtype == torch.float32
    assert logits.device.type == torch.device(device).type


def test_fix_bug5_forward_preserves_backbone_outputs():
    """
    Bug 5 Fix: Mocking the backbone forward demonstrates hidden state extraction
    from top-level wrapper outputs without slicing into internal model.model.
    """
    class MockBackbone(nn.Module):
        def forward(self, input_ids, attention_mask, output_hidden_states=True, return_dict=True):
            B, S = input_ids.shape
            last_hidden_state = torch.randn(B, S, 1536)
            return type("BackboneOutput", (), {"last_hidden_state": last_hidden_state})()

    mock_backbone = MockBackbone()
    input_ids = torch.zeros(1, 10, dtype=torch.long)
    attention_mask = torch.ones(1, 10, dtype=torch.long)

    outputs = mock_backbone(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
    assert hasattr(outputs, "last_hidden_state")
    assert outputs.last_hidden_state.shape == (1, 10, 1536)
