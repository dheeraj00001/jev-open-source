"""
train.py

Production entry point to train, calibrate, and export the System One Hybrid Decision Model.
Supports:
- Checkpoint saving (save_steps) and seamless resumption (resume_from_checkpoint).
- Pre-flight smoke test sanity verification before starting training.

Usage:
    # Full training with checkpoints every 500 steps
    python train.py --data_path decision_train_26k.jsonl --output_dir ./system_one_model --epochs 3 --grad_accum 16

    # Resume from checkpoint
    python train.py --data_path decision_train_26k.jsonl --output_dir ./system_one_model --resume_from_checkpoint ./system_one_model/checkpoint-step-1000

    # Sanity smoke test only
    python train.py --smoke_test
"""

import os
import argparse
import torch
from torch.utils.data import DataLoader, random_split

from fine_tuning_engine import (
    DecisionDataset,
    DecisionDataCollator,
    HybridDecisionEngine,
    EarlyStopping,
    train_system_one_model,
    calibrate_temperatures,
    export_production_artifacts,
    run_smoke_test,
)


def parse_args():
    parser = argparse.ArgumentParser(description="System One Decision Engine Fine-Tuning")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    default_data_path = "data/decision_train_26k.jsonl" if os.path.exists("data/decision_train_26k.jsonl") else "decision_train_26k.jsonl"
    parser.add_argument("--data_path", type=str, default=default_data_path)
    parser.add_argument("--output_dir", type=str, default="./system_one_model")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--lr_backbone", type=float, default=2.0e-4)
    parser.add_argument("--lr_head", type=float, default=1.5e-4)
    parser.add_argument("--val_split", type=float, default=0.05)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--load_in_4bit", action="store_true", default=True)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_steps", type=int, default=500, help="Save a checkpoint every N optimizer steps.")
    parser.add_argument("--eval_steps", type=int, default=250, help="Run validation evaluation every N optimizer steps.")
    parser.add_argument("--eval_samples", type=int, default=100, help="Number of validation samples to evaluate on periodic checks.")
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=0,
        help="Number of evaluations with no improvement before early stopping (0 to disable).",
    )
    parser.add_argument(
        "--early_stopping_min_delta",
        type=float,
        default=0.0,
        help="Minimum change in monitored metric to qualify as an improvement.",
    )
    parser.add_argument(
        "--early_stopping_metric",
        type=str,
        default="val_loss",
        help="Validation metric to monitor for early stopping (e.g. val_loss, choice_acc, score_mae).",
    )
    parser.add_argument(
        "--early_stopping_mode",
        type=str,
        default="min",
        choices=["min", "max"],
        help="Mode for early stopping ('min' for loss/mae, 'max' for accuracy).",
    )
    parser.add_argument(
        "--restore_best_weights",
        action="store_true",
        help="Restore model weights from best checkpoint upon early stopping or completion.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint directory (e.g. ./system_one_model/checkpoint-step-1000) to resume from.",
    )
    parser.add_argument(
        "--smoke_test",
        action="store_true",
        help="Run only the 5-layer sanity smoke test to verify hardware, CUDA, forward/backward, and export.",
    )
    parser.add_argument(
        "--skip_smoke_test",
        action="store_true",
        help="Skip the pre-flight smoke test sanity check before full training.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 60)
    print("SYSTEM ONE DECISION ENGINE FINE-TUNING PIPELINE")
    print(f"Base Model : {args.base_model}")
    print(f"Dataset    : {args.data_path}")
    print(f"Device     : {args.device}")
    print(f"Output Dir : {args.output_dir}")
    print("=" * 60)

    # 1. Initialize Hybrid Model Container
    print("\n[1/5] Loading Hybrid Model Container (Excising LM Head & Initializing Dual-Mode Heads)...")
    model = HybridDecisionEngine(
        base_model_name=args.base_model,
        hidden_dim=1536,
        proj_dim=512,
        max_seq_length=args.max_seq_len,
        load_in_4bit=args.load_in_4bit if args.device.startswith("cuda") else False,
        target_device=args.device,
    )

    # Pre-Flight Smoke Test Sanity Check
    if args.smoke_test:
        print("\n[Smoke Test Mode Requested] Executing sanity verification...")
        smoke_dir = os.path.join(args.output_dir, "smoke_test_verify")
        success = run_smoke_test(
            model=model,
            tokenizer=model.tokenizer,
            device=args.device,
            output_dir=smoke_dir,
        )
        if success:
            print("Smoke test completed successfully. System is ready for training.")
        return

    if not args.skip_smoke_test and not args.resume_from_checkpoint:
        smoke_dir = os.path.join(args.output_dir, "preflight_smoke_check")
        run_smoke_test(
            model=model,
            tokenizer=model.tokenizer,
            device=args.device,
            output_dir=smoke_dir,
        )

    # 2. Data Ingestion & Collation
    print("\n[2/5] Initializing Data Ingestion & DecisionDataCollator...")
    dataset = DecisionDataset(args.data_path)
    total_samples = len(dataset)
    val_size = int(args.val_split * total_samples)
    train_size = total_samples - val_size

    generator = torch.Generator().manual_seed(42)
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=generator)

    collator = DecisionDataCollator(tokenizer=model.tokenizer, max_seq_length=args.max_seq_len)

    train_loader = DataLoader(
        train_ds,
        batch_size=1,  # Micro-batch size B=1
        shuffle=True,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=(args.device.startswith("cuda")),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=(args.device.startswith("cuda")),
    )
    print(f"Train samples: {train_size:,} | Val calibration samples: {val_size:,}")

    # 3. Train Model with Checkpointing
    print("\n[3/5] Starting Optimization Loop...")
    train_system_one_model(
        model=model,
        train_dataloader=train_loader,
        num_epochs=args.epochs,
        gradient_accumulation_steps=args.grad_accum,
        device=args.device,
        learning_rate_backbone=args.lr_backbone,
        learning_rate_head=args.lr_head,
        save_steps=args.save_steps,
        output_dir=args.output_dir,
        resume_checkpoint_path=args.resume_from_checkpoint,
        val_dataloader=val_loader,
        eval_steps=args.eval_steps,
        eval_samples=args.eval_samples,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        early_stopping_metric=args.early_stopping_metric,
        early_stopping_mode=args.early_stopping_mode,
        restore_best_weights=args.restore_best_weights,
    )

    # 4. Temperature Calibration
    print("\n[4/5] Fitting Task Temperature Scalers on Validation Split...")
    calibrate_temperatures(model=model, val_dataloader=val_loader, device=args.device)

    # 5. Export Production Artifacts
    print(f"\n[5/5] Exporting Production Artifacts to {args.output_dir}...")
    export_production_artifacts(model=model, output_dir=args.output_dir)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()
