import pytest
import torch
import torch.nn as nn
from fine_tuning_engine import calibrate_temperatures

class DummyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_A = nn.Parameter(torch.randn(10, 10))
        self.lora_B = nn.Parameter(torch.randn(10, 10))

    def forward(self, input_ids, **kwargs):
        class Out:
            hidden_states = [torch.randn(1, 4, 10)]
        return Out()

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = DummyBackbone()
        self.w_noul = nn.Linear(10, 2)
        self.w_score = nn.Linear(10, 5)
        self.q_proj = nn.Linear(10, 4)
        self.k_proj = nn.Linear(10, 4)
        self.tau_choice = nn.Parameter(torch.ones(1))
        self.tau_noul = nn.Parameter(torch.ones(1))
        self.tau_score = nn.Parameter(torch.ones(1))
        self.scale = 1.0

def test_calibrate_temperatures_does_not_freeze_backbone():
    """
    Guarantees that calibrate_temperatures does NOT set requires_grad = False
    on model parameters, which previously caused the backbone LoRA adapters
    to be excluded from the optimizer.
    """
    model = DummyModel()
    assert model.backbone.lora_A.requires_grad is True
    assert model.backbone.lora_B.requires_grad is True

    dummy_batch = {
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1]]),
        "decide_token_idx": torch.tensor([3]),
        "task_types": ["noul"],
        "candidate_spans": [[]],
        "noul_targets": torch.tensor([1]),
    }

    calibrate_temperatures(model, [dummy_batch], device="cpu", max_iter=2)

    # Must remain True!
    assert model.backbone.lora_A.requires_grad is True
    assert model.backbone.lora_B.requires_grad is True
