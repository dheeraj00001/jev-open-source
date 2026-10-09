"""
tests/test_early_stopping.py

TDD test suite for EarlyStopping integration into System One fine-tuning engine:
1. EarlyStopping class unit tests (min/max modes, patience, min_delta, dict input, state_dict)
2. Checkpoint integration (state persistence and resumption)
3. End-to-end training loop early termination
4. CLI argument parsing in train.py
"""

import os
import tempfile
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from fine_tuning_engine import (
    EarlyStopping,
    save_checkpoint,
    load_checkpoint,
    train_system_one_model,
)
from train import parse_args


# ==============================================================================
# 1. UNIT TESTS: EarlyStopping Class
# ==============================================================================

def test_early_stopping_initialization():
    es = EarlyStopping(patience=3, min_delta=0.01, mode="min", metric_name="val_loss")
    assert es.patience == 3
    assert es.min_delta == 0.01
    assert es.mode == "min"
    assert es.metric_name == "val_loss"
    assert es.counter == 0
    assert es.best_score is None
    assert es.early_stop is False
    assert es.is_enabled is True


def test_early_stopping_invalid_mode():
    with pytest.raises(ValueError, match="mode must be 'min' or 'max'"):
        EarlyStopping(mode="unsupported")


def test_early_stopping_disabled_when_patience_zero_or_negative():
    es_zero = EarlyStopping(patience=0)
    assert es_zero.is_enabled is False
    assert es_zero.step(1.0) is False
    assert es_zero.counter == 0

    es_neg = EarlyStopping(patience=-1)
    assert es_neg.is_enabled is False
    assert es_neg.step(1.0) is False


def test_early_stopping_mode_min():
    es = EarlyStopping(patience=2, min_delta=0.0, mode="min")

    # Step 1: initial best
    assert es.step(1.0, epoch=1, step=10) is False
    assert es.counter == 0
    assert es.best_score == 1.0
    assert es.best_epoch == 1
    assert es.best_step == 10

    # Step 2: improvement
    assert es.step(0.8, epoch=2, step=20) is False
    assert es.counter == 0
    assert es.best_score == 0.8
    assert es.best_epoch == 2

    # Step 3: no improvement (counter = 1)
    assert es.step(0.9, epoch=3, step=30) is False
    assert es.counter == 1
    assert es.best_score == 0.8

    # Step 4: no improvement again (counter = 2 >= patience), early_stop triggered
    assert es.step(0.85, epoch=4, step=40) is True
    assert es.counter == 2
    assert es.early_stop is True


def test_early_stopping_mode_max():
    es = EarlyStopping(patience=2, min_delta=0.0, mode="max", metric_name="choice_acc")

    # Step 1: initial score
    assert es.step(80.0) is False
    assert es.counter == 0
    assert es.best_score == 80.0

    # Step 2: improvement
    assert es.step(85.0) is False
    assert es.counter == 0
    assert es.best_score == 85.0

    # Step 3: no improvement (82.0 < 85.0)
    assert es.step(82.0) is False
    assert es.counter == 1

    # Step 4: triggers early stop
    assert es.step(84.0) is True
    assert es.counter == 2
    assert es.early_stop is True


def test_early_stopping_min_delta():
    es = EarlyStopping(patience=2, min_delta=0.05, mode="min")

    es.step(1.0)
    assert es.best_score == 1.0

    # 0.98 is lower than 1.0, but not by min_delta 0.05 (threshold is 0.95)
    es.step(0.98)
    assert es.counter == 1
    assert es.best_score == 1.0

    # 0.94 satisfies min_delta improvement
    es.step(0.94)
    assert es.counter == 0
    assert es.best_score == 0.94


def test_early_stopping_dict_input():
    es = EarlyStopping(patience=2, metric_name="val_loss_choice", mode="min")

    metrics_step1 = {"val_loss": 2.5, "val_loss_choice": 1.2, "choice_acc": 85.0}
    es.step(metrics_step1)
    assert es.best_score == 1.2
    assert es.counter == 0

    metrics_step2 = {"val_loss": 2.1, "val_loss_choice": 1.3, "choice_acc": 84.0}
    es.step(metrics_step2)
    assert es.counter == 1

    # Missing metric in dict raises KeyError
    with pytest.raises(KeyError, match="not found in validation metrics"):
        es.step({"unrelated_metric": 10.0})


def test_early_stopping_state_dict_and_load():
    es1 = EarlyStopping(patience=3, min_delta=0.01, mode="min", metric_name="val_loss")
    es1.step(2.0, epoch=1, step=100)
    es1.step(2.1, epoch=2, step=200)
    assert es1.counter == 1
    assert es1.best_score == 2.0

    state = es1.state_dict()
    assert state["counter"] == 1
    assert state["best_score"] == 2.0
    assert state["best_epoch"] == 1
    assert state["best_step"] == 100

    es2 = EarlyStopping(patience=5, min_delta=0.0, mode="max")
    es2.load_state_dict(state)
    assert es2.patience == 3
    assert es2.min_delta == 0.01
    assert es2.mode == "min"
    assert es2.counter == 1
    assert es2.best_score == 2.0
    assert es2.best_epoch == 1
    assert es2.best_step == 100


# ==============================================================================
# 2. INTEGRATION TESTS: Checkpoint Persistence with EarlyStopping
# ==============================================================================

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.w_noul = nn.Linear(10, 2)
        self.w_score = nn.Linear(10, 10)
        self.q_proj = nn.Linear(10, 10, bias=False)
        self.k_proj = nn.Linear(10, 10, bias=False)
        self.tau_choice = nn.Parameter(torch.ones(1))
        self.tau_noul = nn.Parameter(torch.ones(1))
        self.tau_score = nn.Parameter(torch.ones(1))
        self.backbone = nn.Linear(10, 10)


def test_checkpoint_saves_and_restores_early_stopping():
    model = DummyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    es_save = EarlyStopping(patience=4, min_delta=0.02, mode="min", metric_name="val_loss")
    es_save.step(1.5, epoch=2, step=400)
    es_save.step(1.6, epoch=3, step=600)  # counter = 1

    with tempfile.TemporaryDirectory() as tmpdir:
        save_checkpoint(
            checkpoint_dir=tmpdir,
            model=model,
            optimizer=optimizer,
            scheduler=None,
            epoch=3,
            step_count=600,
            early_stopping=es_save,
        )

        es_load = EarlyStopping(patience=2, min_delta=0.0, mode="min", metric_name="val_loss")
        load_checkpoint(
            checkpoint_dir=tmpdir,
            model=model,
            optimizer=optimizer,
            scheduler=None,
            device="cpu",
            early_stopping=es_load,
        )

        assert es_load.patience == 4
        assert es_load.min_delta == 0.02
        assert es_load.counter == 1
        assert es_load.best_score == 1.5
        assert es_load.best_epoch == 2
        assert es_load.best_step == 400


def test_checkpoint_backward_compatibility_without_early_stopping():
    model = DummyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save without early_stopping argument
        save_checkpoint(
            checkpoint_dir=tmpdir,
            model=model,
            optimizer=optimizer,
            scheduler=None,
            epoch=1,
            step_count=100,
        )

        # Load with early_stopping passed: should not fail even if state is missing
        es_load = EarlyStopping(patience=3)
        load_checkpoint(
            checkpoint_dir=tmpdir,
            model=model,
            optimizer=optimizer,
            scheduler=None,
            device="cpu",
            early_stopping=es_load,
        )
        assert es_load.counter == 0
        assert es_load.best_score is None


# ==============================================================================
# 3. INTEGRATION TESTS: Training Loop Early Termination
# ==============================================================================

class MockTrainDataset(Dataset):
    def __len__(self):
        return 8

    def __getitem__(self, idx):
        return {
            "input_ids": torch.randint(0, 100, (16,)),
            "attention_mask": torch.ones(16, dtype=torch.long),
            "decide_token_idx": torch.tensor(15),
            "task_types": "choice",
            "candidate_spans": [(0, 4), (4, 8)],
            "noul_targets": torch.tensor(0),
            "score_targets": torch.zeros(10),
            "score_active_levels": torch.tensor(1),
            "choice_targets": torch.tensor(0),
        }


def mock_collator(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "decide_token_idx": torch.stack([b["decide_token_idx"] for b in batch]),
        "task_types": [b["task_types"] for b in batch],
        "candidate_spans": [b["candidate_spans"] for b in batch],
        "noul_targets": torch.stack([b["noul_targets"] for b in batch]),
        "score_targets": torch.stack([b["score_targets"] for b in batch]),
        "score_active_levels": torch.stack([b["score_active_levels"] for b in batch]),
        "choice_targets": torch.stack([b["choice_targets"] for b in batch]),
    }


class MockFullModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(nn.Embedding(100, 16), nn.Linear(16, 16))
        self.backbone.lora_A = nn.Parameter(torch.randn(16, 16))
        self.w_noul = nn.Linear(16, 2)
        self.w_score = nn.Linear(16, 10)
        self.q_proj = nn.Linear(16, 8, bias=False)
        self.k_proj = nn.Linear(16, 8, bias=False)
        self.tau_choice = nn.Parameter(torch.ones(1))
        self.tau_noul = nn.Parameter(torch.ones(1))
        self.tau_score = nn.Parameter(torch.ones(1))

    def forward(self, input_ids, attention_mask, decide_token_idx, task_types, candidate_spans=None):
        hidden = self.backbone(input_ids)
        batch_size = input_ids.size(0)
        batch_idx = torch.arange(batch_size)
        h_decide = hidden[batch_idx, decide_token_idx]
        choice_logits = []
        for b in range(batch_size):
            num_spans = len(candidate_spans[b]) if candidate_spans and candidate_spans[b] else 1
            choice_logits.append(self.w_score(h_decide[b])[:num_spans])
        return {
            "noul_logits": self.w_noul(h_decide),
            "score_logits": self.w_score(h_decide),
            "choice_logits": choice_logits,
        }


def test_train_system_one_model_triggers_early_stopping(monkeypatch):
    """
    Simulates training where validation loss fails to improve,
    verifying that early stopping stops the loop before all epochs run.
    """
    model = MockFullModel()
    train_loader = DataLoader(MockTrainDataset(), batch_size=1, collate_fn=mock_collator)
    val_loader = DataLoader(MockTrainDataset(), batch_size=1, collate_fn=mock_collator)

    # Mock evaluate_model to return worsening validation loss
    eval_call_count = 0
    def fake_evaluate_model(**kwargs):
        nonlocal eval_call_count
        eval_call_count += 1
        # Returns increasing val_loss: 1.0, 1.5, 2.0...
        loss_val = 1.0 + (eval_call_count * 0.5)
        return {
            "val_loss": loss_val,
            "val_loss_choice": 0.5,
            "val_loss_noul": 0.5,
            "val_loss_score": 0.5,
            "choice_acc": 50.0,
            "noul_acc": 50.0,
            "score_mae": 1.0,
        }

    monkeypatch.setattr("fine_tuning_engine.evaluate_model", fake_evaluate_model)

    with tempfile.TemporaryDirectory() as tmpdir:
        es = EarlyStopping(patience=2, min_delta=0.0, mode="min", metric_name="val_loss")
        result = train_system_one_model(
            model=model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            num_epochs=10,
            gradient_accumulation_steps=1,
            device="cpu",
            save_steps=1000,
            eval_steps=2,
            output_dir=tmpdir,
            early_stopping=es,
        )

        assert es.early_stop is True
        # eval_call_count should have stopped at patience=2 (initial + 2 failures = 3 evaluations)
        assert eval_call_count == 3
        # Result should indicate early stopping state
        assert result is not None
        assert result.get("early_stopped") is True
        assert result.get("stopped_epoch") is not None


def test_train_system_one_model_epoch_end_early_stopping(monkeypatch):
    """
    Simulates early stopping triggered during epoch-end evaluation (eval_steps=0).
    """
    model = MockFullModel()
    train_loader = DataLoader(MockTrainDataset(), batch_size=1, collate_fn=mock_collator)
    val_loader = DataLoader(MockTrainDataset(), batch_size=1, collate_fn=mock_collator)

    epoch_val_count = 0
    def fake_evaluate_model(**kwargs):
        nonlocal epoch_val_count
        epoch_val_count += 1
        # Loss does not improve: 1.0, 1.2, 1.4...
        return {
            "val_loss": 1.0 + (epoch_val_count * 0.2),
            "val_loss_choice": 0.5,
            "val_loss_noul": 0.5,
            "val_loss_score": 0.5,
            "choice_acc": 50.0,
            "noul_acc": 50.0,
            "score_mae": 1.0,
        }

    monkeypatch.setattr("fine_tuning_engine.evaluate_model", fake_evaluate_model)

    with tempfile.TemporaryDirectory() as tmpdir:
        result = train_system_one_model(
            model=model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            num_epochs=10,
            gradient_accumulation_steps=1,
            device="cpu",
            save_steps=1000,
            eval_steps=0,  # Only epoch-end evaluation
            output_dir=tmpdir,
            early_stopping_patience=2,
            early_stopping_metric="val_loss",
            early_stopping_mode="min",
        )

        assert result["early_stopped"] is True
        # Initial epoch + 2 failed epochs = stopped after epoch 3
        assert result["stopped_epoch"] == 3
        assert epoch_val_count == 3


def test_train_system_one_model_save_and_restore_best_weights(monkeypatch):
    """
    Tests that checkpoint-best is saved on improvement and restored when restore_best_weights=True.
    """
    model = MockFullModel()
    train_loader = DataLoader(MockTrainDataset(), batch_size=1, collate_fn=mock_collator)
    val_loader = DataLoader(MockTrainDataset(), batch_size=1, collate_fn=mock_collator)

    # First eval is 2.0, second eval is 1.0 (best!), subsequent evals are 3.0, 4.0
    eval_call_count = 0
    def fake_evaluate_model(**kwargs):
        nonlocal eval_call_count
        eval_call_count += 1
        losses = [2.0, 1.0, 3.0, 4.0]
        val = losses[eval_call_count - 1] if eval_call_count <= len(losses) else 5.0
        return {
            "val_loss": val,
            "val_loss_choice": 0.5,
            "val_loss_noul": 0.5,
            "val_loss_score": 0.5,
            "choice_acc": 50.0,
            "noul_acc": 50.0,
            "score_mae": 1.0,
        }

    monkeypatch.setattr("fine_tuning_engine.evaluate_model", fake_evaluate_model)

    with tempfile.TemporaryDirectory() as tmpdir:
        result = train_system_one_model(
            model=model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            num_epochs=10,
            gradient_accumulation_steps=1,
            device="cpu",
            save_steps=1000,
            eval_steps=2,
            output_dir=tmpdir,
            early_stopping_patience=2,
            early_stopping_metric="val_loss",
            early_stopping_mode="min",
            save_best_checkpoint=True,
            restore_best_weights=True,
        )

        assert result["early_stopped"] is True
        best_ckpt = os.path.join(tmpdir, "checkpoint-best")
        assert os.path.exists(best_ckpt)
        assert os.path.exists(os.path.join(best_ckpt, "hybrid_head.safetensors"))
        assert os.path.exists(os.path.join(best_ckpt, "training_state.pt"))
        assert result["early_stopping"].best_score == 1.0


# ==============================================================================
# 4. CLI ARGUMENT PARSING TESTS
# ==============================================================================

def test_train_cli_early_stopping_args(monkeypatch):
    test_args = [
        "train.py",
        "--early_stopping_patience", "4",
        "--early_stopping_min_delta", "0.005",
        "--early_stopping_metric", "choice_acc",
        "--early_stopping_mode", "max",
        "--restore_best_weights",
    ]
    monkeypatch.setattr("sys.argv", test_args)
    parsed = parse_args()

    assert parsed.early_stopping_patience == 4
    assert parsed.early_stopping_min_delta == 0.005
    assert parsed.early_stopping_metric == "choice_acc"
    assert parsed.early_stopping_mode == "max"
    assert parsed.restore_best_weights is True

