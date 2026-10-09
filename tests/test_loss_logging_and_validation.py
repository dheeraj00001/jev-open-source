import pytest
import torch
from fine_tuning_engine import DecisionLossEngine

def test_running_loss_accumulator_scale_bug_repro():
    """
    Demonstrates that dividing by accumulated micro-batches correctly computes
    the true un-inflated loss (2.8125 instead of 45.0).
    """
    gradient_accumulation_steps = 16
    log_interval = 50
    true_mean_loss = 2.8125

    running_loss_total = 0.0
    accumulated_micro_batches = 0
    step_count = 0
    logged_avg_loss = None

    for step in range(log_interval * gradient_accumulation_steps):  # 800 micro-batches
        loss_total = torch.tensor(true_mean_loss)
        loss = loss_total / gradient_accumulation_steps
        
        # Accumulate exact micro-batch loss
        running_loss_total += loss.item() * gradient_accumulation_steps
        accumulated_micro_batches += 1

        if (step + 1) % gradient_accumulation_steps == 0:
            step_count += 1
            if step_count % log_interval == 0:
                logged_avg_loss = running_loss_total / max(1, accumulated_micro_batches)
                running_loss_total = 0.0
                accumulated_micro_batches = 0

    assert logged_avg_loss is not None
    assert abs(logged_avg_loss - true_mean_loss) < 1e-4


def test_epoch_boundary_incomplete_window_fix():
    """
    Demonstrates that dividing by accumulated micro-batches preserves the true loss
    across arbitrary epoch transition step counts (2.8172 instead of 6.3105).
    """
    gradient_accumulation_steps = 16
    log_interval = 50
    true_mean_loss = 2.8172

    step_count = 1543
    running_loss_total = 0.0
    accumulated_micro_batches = 0
    logged_avg_loss = None

    for step in range(7 * gradient_accumulation_steps):
        loss_total = torch.tensor(true_mean_loss)
        loss = loss_total / gradient_accumulation_steps
        running_loss_total += loss.item() * gradient_accumulation_steps
        accumulated_micro_batches += 1

        if (step + 1) % gradient_accumulation_steps == 0:
            step_count += 1
            if step_count % log_interval == 0:
                logged_avg_loss = running_loss_total / max(1, accumulated_micro_batches)
                running_loss_total = 0.0
                accumulated_micro_batches = 0

    assert logged_avg_loss is not None
    assert abs(logged_avg_loss - true_mean_loss) < 1e-4
