import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from fine_tuning_engine import train_system_one_model


class MockBackboneWithLora(nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_A = nn.Parameter(torch.randn(10, 10))


class MockHybridModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = MockBackboneWithLora()
        self.w_noul = nn.Linear(10, 2)
        self.w_score = nn.Linear(10, 10)
        self.q_proj = nn.Linear(10, 10, bias=False)
        self.k_proj = nn.Linear(10, 10, bias=False)
        self.tau_choice = nn.Parameter(torch.ones(1))
        self.tau_noul = nn.Parameter(torch.ones(1))
        self.tau_score = nn.Parameter(torch.ones(1))


def test_train_system_one_model_initialization(capsys):
    model = MockHybridModel()
    dataset = TensorDataset(torch.zeros(1))
    dataloader = DataLoader(dataset)
    train_system_one_model(
        model=model,
        train_dataloader=dataloader,
        num_epochs=0,
        device="cpu",
    )
    captured = capsys.readouterr()
    assert "Verified trainable parameter groups: Backbone LoRA=" in captured.out
    assert "Decision Heads=" in captured.out
