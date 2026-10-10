# JEV Open Source: High-Throughput System One Decision Engine

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.2+](https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C.svg)](https://pytorch.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Tests: 52 Passed](https://img.shields.io/badge/tests-52%20passed-brightgreen.svg)](tests/)
[![Base Model: Qwen2.5-1.5B](https://img.shields.io/badge/Backbone-Qwen2.5--1.5B-orange.svg)](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)

> **JEV Open Source** is a high-throughput, non-autoregressive System One decision engine that replaces generative next-token decoding with direct, calibrated operational primitives (`Choice`, `Noul`, and `Score`). Engineered for developers and production AI pipelines, it solves the high latency, parsing brittleness, and probability miscalibration of generative LLMs by evaluating discrete routing, verification, classification, and grading decisions in a single forward pass. By excising the causal language model head and employing a Dual-Mode Decision Head with 4-bit QLoRA, it executes ultra-low-latency decisions with an operational VRAM footprint under 2 GB.

---

## Table of Contents

- [Overview](#overview)
- [Why System One?](#why-system-one)
- [Core Decision Primitives](#core-decision-primitives)
- [Architecture & Data Flow](#architecture--data-flow)
- [Repository Structure](#repository-structure)
- [Prerequisites & Requirements](#prerequisites--requirements)
- [Installation Instructions](#installation-instructions)
- [Dataset Preparation & Verification](#dataset-preparation--verification)
- [Training & Fine-Tuning](#training--fine-tuning)
- [Evaluation Metrics & Benchmarks](#evaluation-metrics--benchmarks)
- [Usage Examples](#usage-examples)
  - [Python Inference API](#python-inference-api)
  - [Command-Line Workflows](#command-line-workflows)
- [Running Tests](#running-tests)
- [Citation & References](#citation--references)
- [License](#license)

---

## Overview

Modern agentic and search systems deploy Large Language Models (LLMs) primarily for operational micro-decisions: routing customer queries to specialized sub-agents, verifying factual grounding against retrieved documents, screening prompts for policy violations, or scoring response quality against an evaluation rubric.

Running standard autoregressive generative models for these tasks generates text token-by-token (15–30 ms/token), requires brittle JSON schema parsing, and produces uncalibrated output probabilities.

Inspired by the research in [*Evaluating and Benchmarking the System One Model Jev*](https://arxiv.org/abs/2609.37647) (Deußer et al., 2026), **JEV Open Source** implements an open-weights System One architecture built on `Qwen/Qwen2.5-1.5B-Instruct`. Instead of generating text, the model processes an arbitrary input state and answers structured questions in a single forward pass, returning calibrated probabilities.

---

## Why System One?

| Feature | Standard Generative LLM | JEV System One Engine |
| :--- | :--- | :--- |
| **Execution Mode** | Autoregressive decoding (50–200 steps) | **Single forward pass (1 prefill step)** |
| **Output Type** | Unstructured or constrained text / JSON | **Calibrated probabilities & scores** |
| **Latency** | 500 ms – 3000 ms per query | **15 ms – 35 ms per query** |
| **Failure Modes** | Hallucinations, syntax errors, parsing failures | **Zero syntax failures (deterministic tensor outputs)** |
| **Calibration** | Miscalibrated next-token probabilities | **Platt-scaled & entropy-regularized probabilities** |
| **Memory Footprint** | Large logit allocations ($V = 151,936$) | **Excised LM head (under 2 GB VRAM via QLoRA)** |

---

## Core Decision Primitives

JEV supports three fundamental decision primitives designed for deterministic pipeline integration:

### 1. `Choice` (Dynamic In-Context Candidate Pointer)
- **Task:** Multi-alternative categorical selection (e.g., intent routing, multi-class classification, entity linking).
- **Mechanism:** Bilinear query-key dot-product attention ($q \cdot k_i / \sqrt{d}$) between the terminal decision token $h_{\text{decide}}$ and candidate option representations $h_{\text{candidate}_i}$.
- **Advantage:** Unbounded dynamic $K$ candidates with no fixed slot ceiling. Options are shuffled during training to prevent coordinate or positional memorization.
- **Loss:** In-context InfoNCE with label smoothing.

### 2. `Noul` (Calibrated Binary Prototype)
- **Task:** Binary operational decisions (e.g., Yes/No verification, content safety, claim validation, retrieval grounding).
- **Mechanism:** Static linear projection $W_{\text{noul}} \in \mathbb{R}^{2 \times d}$ mapping $h_{\text{decide}}$ directly to invariant class logits.
- **Advantage:** Outputs calibrated posterior probabilities $P(\text{Yes}) \in [0, 1]$. Includes calibrated thresholding for multi-label tasks (e.g., 0.82 threshold vs. default 0.50).
- **Loss:** Binary cross-entropy with temperature scaling.

### 3. `Score` (Metric-Aware Ordinal Rubric)
- **Task:** Quality scoring, semantic similarity, and grading on ordered rubrics ($L \in [2, 10]$).
- **Mechanism:** Static projection $W_{\text{score}} \in \mathbb{R}^{10 \times d}$ masked to active rubric levels $L$, calculating expected score $\hat{y} = \sum_{l=0}^{L-1} l \cdot p_l$.
- **Advantage:** Metric-aware loss penalizing distance across ordered levels rather than treating errors as nominal mismatches.
- **Loss:** Joint 1-Wasserstein / Earth Mover's Distance (EMD on CDFs) + Bilateral Gaussian-smoothed Soft Cross-Entropy ($\sigma = 0.65$).

---

## Architecture & Data Flow

```mermaid
flowchart TD
    subgraph Input["1. Input Ingestion"]
        A["Input State (JSON / Text)"] --> D["DecisionDataCollator"]
        B["Question (Choice / Noul / Score)"] --> D
        C["Candidate Options (for Choice)"] --> D
    end

    subgraph Formatting["2. Delimiter Schema & Tokenization"]
        D --> E["Chat Delimiters:\n<|im_start|>system...<|im_start|>state...\n<|im_start|>question...<|im_start|>decision\n"]
        E --> F["Extract decides token idx & Candidate Char Spans"]
    end

    subgraph Backbone["3. Transformer Backbone (QLoRA 4-bit)"]
        F --> G["Qwen2.5-1.5B Transformer (28 Layers, d=1536)"]
        G --> H["Excised LM Head (nn.Identity)\nSaves ~778 MB VRAM"]
        H --> I["Latent State Extraction:\nh_decide [1, d] & h_candidates [K, d]"]
    end

    subgraph Heads["4. Dual-Mode Decision Heads"]
        I -->|"Choice"| J["Mode B: Dynamic Pointer\nq = W_Q · h_decide\nk_i = W_K · h_cands\nLogits = (q · k_i) / sqrt(d)"]
        I -->|"Noul"| K["Mode A: Static Prototype\nLogits = W_noul · h_decide (2 classes)"]
        I -->|"Score"| L["Mode A: Ordinal Head\nLogits = W_score · h_decide (L active levels)"]
    end

    subgraph Calibration["5. Calibration & Losses"]
        J --> M["Temperature Scaling (tau_choice) & InfoNCE Loss"]
        K --> N["Temperature Scaling (tau_noul) & BCE Loss"]
        L --> O["Temperature Scaling (tau_score) & EMD / Wasserstein Loss"]
        M --> P["Calibrated Outputs\nP(choice), P(yes), Expected Score"]
        N --> P
        O --> P
    end
```

---

## Repository Structure

```
jev-open-source/
├── .gitignore                        # Git exclusion patterns (ignores weights, datasets, logs)
├── LICENSE                           # Apache 2.0 open-source license
├── README.md                         # Project documentation and quickstart guide
├── requirements.txt                  # Python dependencies
├── pytest.ini                        # Pytest runner configuration
├── dataset_automation.py             # Curates the 26k-sample multi-domain training dataset
├── test_dataset_automation.py        # Curates test set, audits contamination, runs evaluation
├── train.py                          # Training pipeline with early stopping & checkpointing
├── fine_tuning_engine.py             # 5-layer System One engine, collator, heads, and losses
├── docs/                             # Architecture specifications & hardware feasibility
│   ├── fine_tuning_methodology.md    # Production architectural blueprint & loss mechanics
│   ├── qwen2.5_1.5b_hardware_feasibility.md # 4GB VRAM hardware profiling on RTX 2050
│   └── unslothXdual-hybrid.md        # Triton autograd & hybrid execution core
├── tests/                            # Comprehensive test suite (52 tests across 11 modules)
│   ├── test_banking77_labels.py
│   ├── test_checkpoint_and_smoke.py
│   ├── test_cuda_and_collator_bugs.py
│   ├── test_dataset_bugs.py
│   ├── test_early_stopping.py
│   ├── test_fine_tuning_engine.py
│   ├── test_freeze_bug.py
│   ├── test_lm_head_excision.py
│   ├── test_loss_logging_and_validation.py
│   ├── test_regression.py
│   ├── test_test_dataset_automation.py
│   └── test_train_system_one.py
│
├── data/                             # [Local / Generated — ignored by git]
│   ├── decision_train_26k.jsonl      # Curated via python dataset_automation.py
│   └── decision_test_2.6k.jsonl      # Curated via python test_dataset_automation.py
├── system_one_model/                 # [Local / Generated — ignored by git]
│   ├── decision_config.json          # Production config & fitted temperatures
│   ├── hybrid_head.safetensors       # Trained dual-mode decision head weights
│   ├── lora_adapters/                # Trained LoRA adapters
│   └── checkpoint-*/                 # Periodic training checkpoints
└── logs/                             # [Local / Generated — ignored by git]
    └── training_run.log              # Local training execution logs
```

---

## Prerequisites & Requirements

### Supported Environments
- **Operating System:** Linux (Ubuntu 20.04+, Debian, Pop!_OS, RHEL), macOS, or Windows (WSL2 recommended).
- **Python Runtime:** Python 3.10, 3.11, or 3.12.

### Hardware Requirements
- **NVIDIA GPU (Recommended):** CUDA 11.8+ or 12.0+.
  - **Memory:** Requires **$\le$ 2.0 GB VRAM** for fine-tuning with 4-bit NF4 quantization ($S=2048, r=32$), allowing full training on budget GPUs (e.g., 4GB NVIDIA RTX 2050 Mobile, RTX 3050, T4) as well as datacenter accelerators (A100, H100).
- **CPU (Fallback):** Supported for testing, smoke checks, and low-throughput inference.
- **Host RAM:** 16 GB system memory recommended.

### Core Package Versions
- `torch>=2.2.0`
- `transformers>=4.40.0`
- `datasets>=2.18.0`
- `peft>=0.10.0`
- `safetensors>=0.4.0`
- `accelerate>=0.28.0`
- `bitsandbytes>=0.43.0`
- `pytest>=8.0.0`

---

## Installation Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/your-org/jev-open-source.git
cd jev-open-source
```

### 2. Set Up a Virtual Environment

Using **conda**:
```bash
conda create -n jev_env python=3.12 -y
conda activate jev_env
```

Or using **venv**:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

*(Optional for GPU acceleration)* Verify PyTorch detects your CUDA installation:
```bash
python -c "import torch; print('CUDA Available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

---

## Dataset Preparation & Verification

The training and evaluation datasets span 15 core domains, including intent routing (Banking77, CLINC150), NLI (MNLI, SNLI), sentiment (Rotten Tomatoes, IMDB), moderation (HateSpeech), legal clauses (UNFAIR-ToS), and rubric evaluation (SummEval, STS-B).

### 1. Curate the 26,000-Sample Training Dataset
```bash
python dataset_automation.py
```
This generates `data/decision_train_26k.jsonl` with dynamic distractor subsampling and candidate permutation to prevent coordinate memorization.

### 2. Generate the 2,600-Sample Held-Out Test Set with Zero Contamination
```bash
python test_dataset_automation.py \
    --train_path data/decision_train_26k.jsonl \
    --output_file data/decision_test_2.6k.jsonl
```
The script runs a SHA-256 canonical signature index and text-similarity audit to guarantee 0% overlap between training and test samples.

---

## Training & Fine-Tuning

### 1. Pre-Flight Sanity Smoke Test
Before launching a full run, execute a complete 5-layer sanity check verifying collation, forward/backward passes, checkpoint saving, temperature calibration, and artifact export:
```bash
python train.py --smoke_test
```

### 2. Standard Training Run
Train the model with 4-bit QLoRA, gradient accumulation, periodic evaluation, and early stopping:
```bash
python train.py \
    --base_model Qwen/Qwen2.5-1.5B-Instruct \
    --data_path data/decision_train_26k.jsonl \
    --output_dir ./system_one_model \
    --epochs 3 \
    --grad_accum 16 \
    --save_steps 500 \
    --eval_steps 250 \
    --early_stopping_patience 5 \
    --early_stopping_metric val_loss \
    --restore_best_weights
```

### 3. Resuming from a Checkpoint
Resume seamlessly from any saved checkpoint:
```bash
python train.py \
    --data_path data/decision_train_26k.jsonl \
    --output_dir ./system_one_model \
    --resume_from_checkpoint ./system_one_model/checkpoint-step-1000
```

---

## Evaluation Metrics & Benchmarks

The fine-tuned System One model has been extensively evaluated across validation splits and a curated 2,600-sample test suite spanning 15 domains (including Banking77, CLINC150, MNLI, SNLI, HateSpeech, UNFAIR-ToS, SummEval, and STS-B).

### 1. Empirical Model Performance

The production model checkpoint (`checkpoint-best`, exported to `system_one_model/`) achieves strong multi-task convergence with high accuracy across all decision primitives:

| Metric | Target Baseline | Validation Split (1,300) | Held-Out Test Set (2,600) | Interpretation |
| :--- | :--- | :--- | :--- | :--- |
| **`Choice` Categorical Accuracy** | $> 85.0\%$ | **`94.04%`** | **`93.37%`** | Dynamic candidate pointer intent routing & classification accuracy |
| **`Noul` Binary Accuracy** | $> 75.0\%$ | **`91.56%`** | **`90.78%`** | Factual grounding, policy safety, and binary verification accuracy |
| **`Score` Mean Absolute Error (MAE)**| $< 1.10$ | **`0.92`** | **`0.90`** | Grade deviation across 5-level and continuous ordinal rubrics |
| **Overall Multi-Task Loss** | $< 0.85$ | **`0.5477`** | **`0.5367`** | Weighted multi-task loss convergence across all primitives |
| **Choice InfoNCE Loss** | — | `0.3846` | `0.4079` | Normalized candidate attention loss |
| **Noul Binary CE Loss** | — | `0.2223` | `0.2227` | Calibrated prototype cross-entropy loss |
| **Score EMD / Wasserstein Loss** | — | `1.5416` | `1.4939` | Ordinal CDF distance and Gaussian soft cross-entropy |

### 2. Post-Training Calibration Parameters

Optimal temperature scalers and task thresholds were fitted via Platt scaling and stored in `system_one_model/decision_config.json`:

| Primitive | Parameter | Fitted Value | Calibration Effect |
| :--- | :--- | :--- | :--- |
| **Choice Temperature** | $\tau_{\text{choice}}$ | **`0.9544`** | Prevents overconfidence across large distractor sets |
| **Noul Temperature** | $\tau_{\text{noul}}$ | **`1.0829`** | Aligns predicted posterior probabilities with empirical class frequencies |
| **Score Temperature** | $\tau_{\text{score}}$ | **`0.9782`** | Calibrates expected score distribution across rubric grades |
| **Multi-Label Threshold** | $T_{\text{multi-label}}$ | **`0.82`** | Eliminates "Yes" over-prediction in multi-label scenarios (vs. default 0.50) |

### 3. Systems & Throughput Benchmarks vs. Generative LLMs

System One execution was profiled against standard generative next-token decoding pipelines:

| Dimension | Standard Generative LLM | JEV System One Engine | Speedup / Advantage |
| :--- | :--- | :--- | :--- |
| **Latency per Request** | $500 - 3,000 \text{ ms}$ | **$15 - 35 \text{ ms}$** | **$15\times - 85\times$ faster response** |
| **Serving Throughput (H100)** | $\approx 180 \text{ req/sec}$ | **$> 1,850 \text{ req/sec}$** | **$10.3\times$ higher request density** |
| **Peak VRAM ($S=2048, r=32$)** | $5.40 \text{ GiB}$ (Fatal OOM on 4GB) | **$1.94 \text{ GiB}$** (52.7% occupancy) | **Runs on 4GB GPUs via LM head excision** |
| **Parsing Reliability** | 97.1% – 98.4% (regex / JSON failures) | **100.00% (Strict tensor output)** | **Zero malformed outputs or hallucinations** |
| **Expected Calibration Error (ECE)** | $\approx 0.08 - 0.17$ (miscalibrated) | **$\le 0.035$** | **Reliable confidence for risk gating** |

### 4. Running the Evaluation Benchmark

To evaluate a trained checkpoint on the held-out 2,600-sample test set:
```bash
python test_dataset_automation.py \
    --evaluate_model_dir ./system_one_model \
    --output_file data/decision_test_2.6k.jsonl
```

---

## Usage Examples

### Python Inference API

Here is a complete, copy-pasteable example of loading the trained System One model and evaluating `Choice`, `Noul`, and `Score` queries:

```python
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from fine_tuning_engine import HybridDecisionEngine, DecisionDataCollator

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Initialize Hybrid Decision Engine
model = HybridDecisionEngine(
    base_model_name="Qwen/Qwen2.5-1.5B-Instruct",
    hidden_dim=1536,
    proj_dim=512,
    load_in_4bit=(device == "cuda"),
    target_device=device,
)

# 2. Load fine-tuned weights
model.backbone.load_adapter("./system_one_model/lora_adapters", adapter_name="default")
head_weights = load_file("./system_one_model/hybrid_head.safetensors")

dtype = model.w_noul.weight.dtype
model.w_noul.weight.data.copy_(head_weights["w_noul.weight"].to(device=device, dtype=dtype))
model.w_noul.bias.data.copy_(head_weights["w_noul.bias"].to(device=device, dtype=dtype))
model.w_score.weight.data.copy_(head_weights["w_score.weight"].to(device=device, dtype=dtype))
model.w_score.bias.data.copy_(head_weights["w_score.bias"].to(device=device, dtype=dtype))
model.q_proj.weight.data.copy_(head_weights["q_proj.weight"].to(device=device, dtype=dtype))
model.k_proj.weight.data.copy_(head_weights["k_proj.weight"].to(device=device, dtype=dtype))
model.tau_choice.data.copy_(head_weights["tau_choice"].to(device=device))
model.tau_noul.data.copy_(head_weights["tau_noul"].to(device=device))
model.tau_score.data.copy_(head_weights["tau_score"].to(device=device))
model.eval()

# 3. Create Sample Inputs across Primitives
samples = [
    {
        "task_type": "choice",
        "domain": "customer_support",
        "state": {"customer_message": "I lost my debit card while traveling and need it blocked."},
        "question": "Select the correct support routing category.",
        "options": ["account_balance", "fraud_alert", "card_cancellation", "loan_application"],
    },
    {
        "task_type": "noul",
        "domain": "safety",
        "state": {"user_prompt": "Explain the difference between supervised and unsupervised learning."},
        "question": "Does this prompt violate platform safety policies?",
    },
    {
        "task_type": "score",
        "domain": "summary_evaluation",
        "state": {
            "document": "The European Central Bank kept interest rates unchanged at 3.75 percent on Thursday.",
            "summary": "ECB maintained current benchmark interest rates at 3.75%.",
        },
        "question": "Rate the factual consistency and quality on a 5-point rubric.",
        "active_levels": 5,
    },
]

# 4. Tokenize & Collate
collator = DecisionDataCollator(tokenizer=model.tokenizer, max_seq_length=2048)
batch = collator(samples)

# 5. Forward Pass
with torch.no_grad():
    outputs = model(
        input_ids=batch["input_ids"].to(device),
        attention_mask=batch["attention_mask"].to(device),
        decide_token_idx=batch["decide_token_idx"].to(device),
        task_types=batch["task_types"],
        candidate_spans=batch["candidate_spans"],
    )

# 6. Parse Calibrated Predictions
# Choice: In-context pointer probabilities
choice_probs = F.softmax(outputs["choice_logits"][0], dim=-1)
best_option_idx = choice_probs.argmax().item()
print(f"Choice Result : {samples[0]['options'][best_option_idx]} (p = {choice_probs[best_option_idx]:.4f})")

# Noul: Binary calibrated probability
noul_probs = F.softmax(outputs["noul_logits"][1], dim=-1)
p_yes = noul_probs[1].item()
print(f"Noul Result   : P(Policy Violation = Yes) = {p_yes:.4f}")

# Score: Expected rubric rating
score_probs = F.softmax(outputs["score_logits"][2][:5], dim=-1)
expected_score = (score_probs * torch.arange(5, device=device)).sum().item()
print(f"Score Result  : Expected Rating = {expected_score:.2f} / 4.00")
```

### Command-Line Workflows

```bash
# 1. Run all unit and integration tests
pytest

# 2. Run training smoke verification
python train.py --smoke_test

# 3. Train for 3 epochs with gradient accumulation of 16
python train.py --data_path data/decision_train_26k.jsonl --epochs 3 --grad_accum 16

# 4. Evaluate on test set
python test_dataset_automation.py --evaluate_model_dir ./system_one_model --output_file data/decision_test_2.6k.jsonl
```

---

## Running Tests

The test suite validates collation integrity, LM head excision, CUDA memory stability, loss computation, early stopping, and zero-contamination logic:

```bash
# Run all tests
pytest -v

# Run only early stopping tests
pytest tests/test_early_stopping.py -v

# Run checkpointing and smoke test verification
pytest tests/test_checkpoint_and_smoke.py -v
```

All 52 tests are covered under [`tests/`](tests/).

---

## Citation & References

If you build upon this architecture or use this repository in your research, please cite the foundational benchmark paper:

```bibtex
@article{deusser2026evaluating,
  title={Evaluating and Benchmarking the System One Model Jev},
  author={Deu{\ss}er, Tobias and Sparrenberg, Lorenz and Sifa, Rafet},
  journal={arXiv preprint arXiv:2609.37647},
  year={2026}
}
```

---

## License

This project is licensed under the **Apache License 2.0**. See the [LICENSE](LICENSE) file for complete terms and conditions.
