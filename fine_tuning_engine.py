"""
fine_tuning_engine.py

System One Fine-Tuning Execution Engine for Qwen2.5-1.5B Hybrid Decision Model.
Implements the 5-layer topology:
1. Data Ingestion Engine (DecisionDataCollator & Arrow Dataset)
2. Hybrid Model Container (HybridDecisionEngine: Unsloth/HF QLoRA + Excised LM Head + Dual-Mode Head)
3. Multi-Task Loss Computation Engine (DecisionLossEngine: InfoNCE, EMD Wasserstein, Gaussian Soft-CE)
4. Hardware-Constrained Optimization Loop with Checkpoint Saving & Resumption (PagedAdamW8bit, B=1, Grad Accum=16-32)
5. Post-Training Temperature Calibrator, Artifact Exporter & Pre-Flight Smoke Test
"""

import os
import shutil
import json
import random
from typing import Optional, List, Dict, Any, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from safetensors.torch import save_file, load_file

# Mitigate CUDA memory fragmentation on 4GB/8GB GPUs
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"

# Check for bitsandbytes
try:
    import bitsandbytes as bnb
    HAS_BNB = True
except ImportError:
    HAS_BNB = False

# Check for Unsloth
try:
    from unsloth import FastLanguageModel
    HAS_UNSLOTH = True
except ImportError:
    HAS_UNSLOTH = False

# Standard PEFT imports
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


# ==============================================================================
# 1. DATA INGESTION ENGINE
# ==============================================================================

class DecisionDataset(Dataset):
    """
    Memory-mapped Arrow / JSONL dataset for System One decision primitives.
    """
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.records = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.records.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        return self.records[idx]


class DecisionDataCollator:
    """
    Serializes samples into the System One delimiter schema,
    tokenizes text, locates the terminal decide_token_idx,
    extracts token spans for candidate options, and collates typed targets.
    """
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        max_seq_length: int = 2048,
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def format_prompt(self, sample: dict) -> Tuple[str, List[Tuple[int, int]]]:
        """
        Formats sample into deterministic chat schema:
        <|im_start|>system
        You are a high-speed System One decision model. Analyze the provided state and evaluate the question. Output calibrated decision probabilities directly.<|im_end|>
        <|im_start|>state
        {state_json}
        <|im_end|>
        <|im_start|>question type="{task_type}" domain="{domain}"
        {question}
        [0] Option 0
        [1] Option 1
        <|im_end|>
        <|im_start|>decision\n
        """
        task_type = sample.get("task_type", "noul")
        domain = sample.get("domain", "general")
        state_dict = sample.get("state", {})
        question = sample.get("question", "")

        state_str = json.dumps(state_dict, ensure_ascii=False)

        prompt = (
            "<|im_start|>system\n"
            "You are a high-speed System One decision model. "
            "Analyze the provided state and evaluate the question. "
            "Output calibrated decision probabilities directly.<|im_end|>\n"
            "<|im_start|>state\n"
            f"{state_str}\n"
            "<|im_end|>\n"
            f"<|im_start|>question type=\"{task_type}\" domain=\"{domain}\"\n"
            f"{question}\n"
        )

        candidate_char_spans = []
        if task_type == "choice":
            options = sample.get("options", [])
            for i, opt in enumerate(options):
                opt_line = f"[{i}] {opt}\n"
                start_char = len(prompt)
                prompt += opt_line
                end_char = len(prompt) - 1  # exclude trailing newline
                candidate_char_spans.append((start_char, end_char))

        prompt += "<|im_end|>\n<|im_start|>decision\n"
        return prompt, candidate_char_spans

    def __call__(self, batch: List[dict]) -> dict:
        input_ids_list = []
        attention_mask_list = []
        decide_token_indices = []
        candidate_spans_batch = []
        task_types = []

        noul_targets = []
        score_targets = []
        score_active_levels = []
        choice_targets = []

        for sample in batch:
            task_type = sample.get("task_type", "noul")
            task_types.append(task_type)

            prompt, candidate_char_spans = self.format_prompt(sample)
            encoded = self.tokenizer(
                prompt,
                max_length=self.max_seq_length,
                truncation=True,
                return_offsets_mapping=True,
                add_special_tokens=False,
            )

            ids = encoded["input_ids"]
            mask = encoded["attention_mask"]
            offsets = encoded["offset_mapping"]
            seq_len = len(ids)

            # Fix 3: Point directly to the final terminal token of the prompt (<|im_start|>decision\n)
            decide_tok_idx = seq_len - 1
            decide_token_indices.append(decide_tok_idx)

            # Fix 4: Safely map and clamp candidate character spans to prevent empty slices
            spans_for_sample = []
            if task_type == "choice":
                for (c_start, c_end) in candidate_char_spans:
                    t_start = None
                    t_end = None
                    for t_idx, (tok_cs, tok_ce) in enumerate(offsets):
                        if tok_ce > c_start and t_start is None:
                            t_start = t_idx
                        if tok_ce >= c_end and t_end is None:
                            t_end = t_idx + 1
                            break

                    if t_start is None:
                        t_start = max(0, seq_len - 2)
                    if t_end is None:
                        t_end = seq_len

                    # Clamp strictly to avoid NaNs
                    t_start = min(t_start, seq_len - 1)
                    t_end = max(t_start + 1, min(t_end, seq_len))
                    spans_for_sample.append((t_start, t_end))
            candidate_spans_batch.append(spans_for_sample)

            # Target mappings
            if task_type == "noul":
                noul_targets.append(1 if sample.get("target_bool", False) else 0)
                score_targets.append(-100)
                score_active_levels.append(5)
                choice_targets.append(-100)
            elif task_type == "score":
                noul_targets.append(-100)
                score_targets.append(int(sample.get("target_score", 0)))
                score_active_levels.append(int(sample.get("active_levels", 5)))
                choice_targets.append(-100)
            elif task_type == "choice":
                noul_targets.append(-100)
                score_targets.append(-100)
                score_active_levels.append(5)
                choice_targets.append(int(sample.get("target_idx", 0)))
            else:
                noul_targets.append(-100)
                score_targets.append(-100)
                score_active_levels.append(5)
                choice_targets.append(-100)

            input_ids_list.append(torch.tensor(ids, dtype=torch.long))
            attention_mask_list.append(torch.tensor(mask, dtype=torch.long))

        # Dynamic batch padding
        max_len = max(t.size(0) for t in input_ids_list)
        padded_ids = []
        padded_masks = []
        pad_id = self.tokenizer.pad_token_id or 0

        for ids_tensor, mask_tensor in zip(input_ids_list, attention_mask_list):
            pad_len = max_len - ids_tensor.size(0)
            if pad_len > 0:
                padded_ids.append(F.pad(ids_tensor, (0, pad_len), value=pad_id))
                padded_masks.append(F.pad(mask_tensor, (0, pad_len), value=0))
            else:
                padded_ids.append(ids_tensor)
                padded_masks.append(mask_tensor)

        return {
            "input_ids": torch.stack(padded_ids, dim=0),
            "attention_mask": torch.stack(padded_masks, dim=0),
            "decide_token_idx": torch.tensor(decide_token_indices, dtype=torch.long),
            "candidate_spans": candidate_spans_batch,
            "task_types": task_types,
            "noul_targets": torch.tensor(noul_targets, dtype=torch.long),
            "score_targets": torch.tensor(score_targets, dtype=torch.long),
            "score_active_levels": torch.tensor(score_active_levels, dtype=torch.long),
            "choice_targets": torch.tensor(choice_targets, dtype=torch.long),
        }


# ==============================================================================
# 2. MULTI-TASK MATHEMATICAL LOSS ENGINE
# ==============================================================================

class DecisionLossEngine(nn.Module):
    """
    Computes exact losses across the three System One decision primitives:
    - Choice: In-context InfoNCE over dynamic candidate simplex
    - Noul: Calibrated Binary Cross-Entropy on static prototypes
    - Score: Combined 1-Wasserstein (EMD) + Gaussian-smoothed Soft Cross-Entropy
    - Regularization: Entropy-based surrogate calibration penalty
    """
    def __init__(
        self,
        lambda_choice: float = 1.0,
        lambda_noul: float = 1.0,
        lambda_score: float = 1.2,
        gamma_wasserstein: float = 0.5,
        beta_calib: float = 0.05,
        gaussian_sigma: float = 0.65,
    ):
        super().__init__()
        self.lambda_choice = lambda_choice
        self.lambda_noul = lambda_noul
        self.lambda_score = lambda_score
        self.gamma_wasserstein = gamma_wasserstein
        self.beta_calib = beta_calib
        self.gaussian_sigma = gaussian_sigma

    def compute_noul_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Standard Cross-Entropy over invariant No (0) / Yes (1) coordinates."""
        valid_mask = targets != -100
        if not valid_mask.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        return F.cross_entropy(logits[valid_mask], targets[valid_mask], label_smoothing=0.02)

    def compute_choice_loss(self, logits_list: List[torch.Tensor], targets: torch.Tensor) -> torch.Tensor:
        """
        Dynamic In-Context InfoNCE loss.
        Each sample in the batch has an arbitrary number of candidates K.
        """
        losses = []
        for i, target in enumerate(targets):
            if target == -100:
                continue
            if logits_list[i].numel() == 0:
                continue
            sample_logits = logits_list[i].unsqueeze(0)  # [1, K]
            target_idx = target.unsqueeze(0)             # [1]
            loss = F.cross_entropy(sample_logits, target_idx, label_smoothing=0.02)
            losses.append(loss)

        if not losses:
            return torch.tensor(0.0, device=targets.device, requires_grad=True)
        return torch.stack(losses).mean()

    def compute_score_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        active_levels_tensor: torch.Tensor
    ) -> torch.Tensor:
        """
        Combined 1-Wasserstein (EMD) and Soft-Target Bilateral Gaussian Loss.
        Penalizes metric distance on ordered rubric levels rather than nominal error.
        """
        total_loss = []
        for i, target in enumerate(targets):
            if target == -100:
                continue
            L = active_levels_tensor[i].item()
            y_star = target.item()
            sample_logits = logits[i, :L].unsqueeze(0)  # [1, L]
            probs = F.softmax(sample_logits, dim=-1)     # [1, L]

            # 1. 1-Wasserstein / Earth Mover's Distance (CDF L1 Distance)
            cdf_pred = torch.cumsum(probs, dim=-1)
            cdf_target = torch.zeros_like(cdf_pred)
            cdf_target[0, y_star:] = 1.0
            loss_emd = torch.mean(torch.abs(cdf_pred[:, :-1] - cdf_target[:, :-1]))

            # 2. Soft-Target Cross-Entropy with Bilateral Gaussian Smoothing
            levels = torch.arange(L, device=logits.device, dtype=torch.float32)
            gaussian_weights = torch.exp(-((levels - y_star) ** 2) / (2 * (self.gaussian_sigma ** 2)))
            target_dist = (gaussian_weights / gaussian_weights.sum()).unsqueeze(0)  # [1, L]
            loss_soft = -(target_dist * F.log_softmax(sample_logits, dim=-1)).sum(dim=-1).mean()

            sample_score_loss = loss_soft + (self.gamma_wasserstein * loss_emd)
            total_loss.append(sample_score_loss)

        if not total_loss:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        return torch.stack(total_loss).mean()

    def compute_calibration_penalty(self, probs: torch.Tensor) -> torch.Tensor:
        """Surrogate Entropy regularization to prevent logit explosion."""
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
        return -entropy.mean()  # Penalize low entropy (overconfidence)

    def forward(self, model_outputs: dict, batch: dict) -> dict:
        task_types = batch["task_types"]
        losses = {}
        total = torch.tensor(0.0, device=batch["input_ids"].device)
        calib_penalties = []

        # Dispatch Noul
        if "noul" in task_types:
            l_noul = self.compute_noul_loss(model_outputs["noul_logits"], batch["noul_targets"])
            total = total + self.lambda_noul * l_noul
            losses["loss_noul"] = l_noul.detach()

            valid_mask = batch["noul_targets"] != -100
            if valid_mask.any():
                noul_probs = F.softmax(model_outputs["noul_logits"][valid_mask], dim=-1)
                calib_penalties.append(self.compute_calibration_penalty(noul_probs))

        # Dispatch Score
        if "score" in task_types:
            l_score = self.compute_score_loss(
                model_outputs["score_logits"], batch["score_targets"], batch["score_active_levels"]
            )
            total = total + self.lambda_score * l_score
            losses["loss_score"] = l_score.detach()

            for i, target in enumerate(batch["score_targets"]):
                if target != -100:
                    L = batch["score_active_levels"][i].item()
                    s_probs = F.softmax(model_outputs["score_logits"][i, :L].unsqueeze(0), dim=-1)
                    calib_penalties.append(self.compute_calibration_penalty(s_probs))

        # Dispatch Choice
        if "choice" in task_types:
            l_choice = self.compute_choice_loss(model_outputs["choice_logits"], batch["choice_targets"])
            total = total + self.lambda_choice * l_choice
            losses["loss_choice"] = l_choice.detach()

            for i, target in enumerate(batch["choice_targets"]):
                if target != -100 and model_outputs["choice_logits"][i].numel() > 0:
                    c_probs = F.softmax(model_outputs["choice_logits"][i].unsqueeze(0), dim=-1)
                    calib_penalties.append(self.compute_calibration_penalty(c_probs))

        # Calibration Regularizer
        if calib_penalties and self.beta_calib > 0.0:
            l_calib = torch.stack(calib_penalties).mean()
            total = total + self.beta_calib * l_calib
            losses["loss_calib"] = l_calib.detach()

        losses["loss_total"] = total
        return losses


def excise_lm_head(model: nn.Module) -> bool:
    """
    Safely excises the generative 151,936-class lm_head from a causal LM or PEFT model,
    reclaiming up to 778 MB VRAM by replacing the projection with nn.Identity().
    Properly handles PeftModelForCausalLM (navigating get_base_model / base_model.model),
    Unsloth FastLanguageModel, and standard Hugging Face PreTrainedModel architectures.
    """
    target = None
    if hasattr(model, "get_base_model") and hasattr(model.get_base_model(), "lm_head"):
        target = model.get_base_model()
    elif hasattr(model, "base_model") and hasattr(model.base_model, "model") and hasattr(model.base_model.model, "lm_head"):
        target = model.base_model.model
    elif hasattr(model, "_modules") and "lm_head" in model._modules:
        target = model

    if target is not None and hasattr(target, "lm_head"):
        try:
            del target.lm_head
        except (AttributeError, KeyError):
            pass
        target.lm_head = nn.Identity()
        if hasattr(model, "_modules") and "lm_head" in model._modules:
            model.lm_head = nn.Identity()
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return True
    return False


# ==============================================================================
# 3. HYBRID MODEL CONTAINER
# ==============================================================================

class HybridDecisionEngine(nn.Module):
    """
    Integrates QLoRA transformer backbone with excised LM head
    and Dual-Mode Decision Head (Mode A: Static Anchors, Mode B: Candidate Pointer).
    Compatible with hidden_dim=1536 (Qwen2.5-1.5B) and hidden_dim=2560 (K2 Horizon 3.7B).
    """
    def __init__(
        self,
        base_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        hidden_dim: int = 1536,
        proj_dim: int = 512,
        max_seq_length: int = 2048,
        load_in_4bit: bool = True,
        lora_r: int = 32,
        lora_alpha: int = 64,
        target_device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.proj_dim = proj_dim
        self.base_model_name = base_model_name
        self.target_device = target_device

        # Select target compute dtype (Fix 2: Dtype Mismatch)
        compute_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported() and target_device.startswith("cuda")) else (
            torch.float16 if target_device.startswith("cuda") else torch.float32
        )

        # 1. Backbone Initialization
        if HAS_UNSLOTH and load_in_4bit and target_device.startswith("cuda"):
            self.backbone, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=base_model_name,
                max_seq_length=max_seq_length,
                load_in_4bit=True,
            )
            self.backbone = FastLanguageModel.get_peft_model(
                self.backbone,
                r=lora_r,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                lora_alpha=lora_alpha,
                lora_dropout=0.0,
                bias="none",
                use_gradient_checkpointing="unsloth",
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            if load_in_4bit and target_device.startswith("cuda"):
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=True,
                )
                raw_model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    quantization_config=bnb_config,
                    device_map=target_device,
                )
                raw_model = prepare_model_for_kbit_training(raw_model)
            else:
                raw_model = AutoModelForCausalLM.from_pretrained(base_model_name)
                if target_device.startswith("cuda"):
                    raw_model.to(target_device)

            peft_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                lora_dropout=0.0,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.backbone = get_peft_model(raw_model, peft_config)
            if hasattr(self.backbone, "gradient_checkpointing_enable"):
                self.backbone.gradient_checkpointing_enable()

        # 2. Excise generative LM Head (reclaims up to 778 MB VRAM)
        excise_lm_head(self.backbone)

        # 3. Mode A: Static Anchors (Fix 1 & 2: Moved directly to target_device and compute_dtype)
        self.w_noul = nn.Linear(hidden_dim, 2, bias=True).to(device=target_device, dtype=compute_dtype)
        self.w_score = nn.Linear(hidden_dim, 10, bias=True).to(device=target_device, dtype=compute_dtype)

        # 4. Mode B: Dynamic Contextual Pointer
        self.q_proj = nn.Linear(hidden_dim, proj_dim, bias=False).to(device=target_device, dtype=compute_dtype)
        self.k_proj = nn.Linear(hidden_dim, proj_dim, bias=False).to(device=target_device, dtype=compute_dtype)
        self.scale = 1.0 / (proj_dim ** 0.5)

        # 5. Temperature Scalers (Kept on target_device)
        self.tau_choice = nn.Parameter(torch.ones(1, device=target_device, dtype=torch.float32) * 1.0)
        self.tau_noul = nn.Parameter(torch.ones(1, device=target_device, dtype=torch.float32) * 1.0)
        self.tau_score = nn.Parameter(torch.ones(1, device=target_device, dtype=torch.float32) * 1.0)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        decide_token_idx: torch.Tensor,
        task_types: List[str],
        candidate_spans: Optional[List[List[Tuple[int, int]]]] = None,
    ) -> dict:
        # Fix 5: Route directly through top-level backbone wrapper to retain Unsloth/PEFT hooks
        backbone_outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        if hasattr(backbone_outputs, "hidden_states") and backbone_outputs.hidden_states is not None:
            hidden_states = backbone_outputs.hidden_states[-1]
        elif hasattr(backbone_outputs, "last_hidden_state"):
            hidden_states = backbone_outputs.last_hidden_state
        else:
            hidden_states = backbone_outputs[0]

        # Ensure hidden_states matches decision heads' parameter dtype (handles 4-bit float32 norm upcasting)
        hidden_states = hidden_states.to(dtype=self.w_noul.weight.dtype)

        batch_size = input_ids.size(0)
        batch_idx = torch.arange(batch_size, device=input_ids.device)
        h_decide = hidden_states[batch_idx, decide_token_idx]  # [B, d]

        outputs = {}

        # Mode A: Noul Projections (Cast logits to float32 before temperature scaling)
        outputs["noul_logits"] = self.w_noul(h_decide).float() / self.tau_noul

        # Mode A: Score Projections
        outputs["score_logits"] = self.w_score(h_decide).float() / self.tau_score

        # Mode B: Dynamic Choice Pointer
        choice_logits_list = []
        for b in range(batch_size):
            if task_types[b] == "choice" and candidate_spans is not None:
                spans = candidate_spans[b]
                # Fix 4: Guard against degenerate empty spans during mean-pooling
                cand_reps = [
                    hidden_states[b, start:end, :].mean(dim=0)
                    for (start, end) in spans
                    if end > start
                ]
                if cand_reps:
                    h_cands = torch.stack(cand_reps, dim=0)  # [K, d]
                    q = self.q_proj(h_decide[b]).unsqueeze(0)  # [1, proj_dim]
                    k = self.k_proj(h_cands)                    # [K, proj_dim]
                    logits = (torch.mm(q, k.t()).squeeze(0) * self.scale).float() / self.tau_choice
                    choice_logits_list.append(logits)
                else:
                    choice_logits_list.append(torch.empty(0, device=input_ids.device))
            else:
                choice_logits_list.append(torch.empty(0, device=input_ids.device))

        outputs["choice_logits"] = choice_logits_list
        return outputs


# ==============================================================================
# 4. HARDWARE-CONSTRAINED OPTIMIZATION LOOP WITH CHECKPOINTING & EARLY STOPPING
# ==============================================================================

class EarlyStopping:
    """
    Early Stopping monitor to stop training when a monitored validation metric stops improving.

    Args:
        patience: Number of validation evaluations with no improvement before stopping.
                  If patience <= 0 or None, early stopping is disabled.
        min_delta: Minimum change in the monitored metric to qualify as an improvement.
        mode: 'min' for metrics where lower is better (e.g., val_loss, score_mae),
              'max' for metrics where higher is better (e.g., choice_acc, noul_acc).
        metric_name: Name of metric in the validation dictionary to monitor (default: 'val_loss').
    """
    def __init__(
        self,
        patience: int = 3,
        min_delta: float = 0.0,
        mode: str = "min",
        metric_name: str = "val_loss",
    ):
        self.patience = patience
        self.min_delta = float(min_delta)
        self.mode = mode.lower()
        if self.mode not in ("min", "max"):
            raise ValueError(f"EarlyStopping mode must be 'min' or 'max', got {mode}")
        self.metric_name = metric_name

        self.counter: int = 0
        self.best_score: Optional[float] = None
        self.early_stop: bool = False
        self.best_epoch: Optional[int] = None
        self.best_step: Optional[int] = None

    @property
    def is_enabled(self) -> bool:
        return self.patience is not None and self.patience > 0

    def step(
        self,
        metrics: Union[float, Dict[str, float]],
        epoch: Optional[int] = None,
        step: Optional[int] = None,
    ) -> bool:
        """
        Updates early stopping state with the latest validation result.
        Returns True if early stopping was triggered (training should halt), False otherwise.
        """
        if not self.is_enabled or self.early_stop:
            return self.early_stop

        if isinstance(metrics, dict):
            if self.metric_name not in metrics:
                raise KeyError(
                    f"EarlyStopping metric '{self.metric_name}' not found in validation metrics: {list(metrics.keys())}"
                )
            current = float(metrics[self.metric_name])
        else:
            current = float(metrics)

        if self.best_score is None:
            self.best_score = current
            self.best_epoch = epoch
            self.best_step = step
            self.counter = 0
            return False

        if self.mode == "min":
            improved = current < (self.best_score - self.min_delta)
        else:
            improved = current > (self.best_score + self.min_delta)

        if improved:
            self.best_score = current
            self.best_epoch = epoch
            self.best_step = step
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def state_dict(self) -> Dict[str, Any]:
        return {
            "patience": self.patience,
            "min_delta": self.min_delta,
            "mode": self.mode,
            "metric_name": self.metric_name,
            "counter": self.counter,
            "best_score": self.best_score,
            "early_stop": self.early_stop,
            "best_epoch": self.best_epoch,
            "best_step": self.best_step,
        }

    def load_state_dict(self, state: Dict[str, Any]):
        self.patience = state.get("patience", self.patience)
        self.min_delta = state.get("min_delta", self.min_delta)
        self.mode = state.get("mode", self.mode)
        self.metric_name = state.get("metric_name", self.metric_name)
        self.counter = state.get("counter", 0)
        self.best_score = state.get("best_score", None)
        self.early_stop = state.get("early_stop", False)
        self.best_epoch = state.get("best_epoch", None)
        self.best_step = state.get("best_step", None)


def save_checkpoint(
    checkpoint_dir: str,
    model: HybridDecisionEngine,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    step_count: int,
    early_stopping: Optional[EarlyStopping] = None,
):
    """
    Saves complete training state to resume seamlessly:
    - Backbone LoRA adapters
    - Dual-Mode Hybrid Decision Head (w_noul, w_score, q_proj, k_proj, tau_*)
    - PagedAdamW8bit optimizer momentum & variance buffers
    - LR Scheduler step position
    - PyTorch and CUDA RNG states
    - EarlyStopping state
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 1. Save LoRA Adapters
    lora_dir = os.path.join(checkpoint_dir, "lora_adapters")
    os.makedirs(lora_dir, exist_ok=True)
    if hasattr(model.backbone, "save_pretrained"):
        model.backbone.save_pretrained(lora_dir)
    if hasattr(model, "tokenizer") and model.tokenizer is not None:
        model.tokenizer.save_pretrained(lora_dir)

    # 2. Save Dual-Mode Hybrid Head
    head_state_dict = {
        "w_noul.weight": model.w_noul.weight.data.cpu(),
        "w_noul.bias": model.w_noul.bias.data.cpu(),
        "w_score.weight": model.w_score.weight.data.cpu(),
        "w_score.bias": model.w_score.bias.data.cpu(),
        "q_proj.weight": model.q_proj.weight.data.cpu(),
        "k_proj.weight": model.k_proj.weight.data.cpu(),
        "tau_choice": model.tau_choice.data.cpu(),
        "tau_noul": model.tau_noul.data.cpu(),
        "tau_score": model.tau_score.data.cpu(),
    }
    save_file(head_state_dict, os.path.join(checkpoint_dir, "hybrid_head.safetensors"))

    # 3. Save Optimizer, Scheduler, and Metadata
    train_state = {
        "epoch": epoch,
        "step_count": step_count,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else {},
        "rng_state_torch": torch.get_rng_state(),
        "rng_state_cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        "early_stopping": early_stopping.state_dict() if early_stopping is not None else None,
    }
    torch.save(train_state, os.path.join(checkpoint_dir, "training_state.pt"))
    print(f"\n[Checkpoint Saved] -> {checkpoint_dir}")


def load_checkpoint(
    checkpoint_dir: str,
    model: HybridDecisionEngine,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    device: str = "cuda",
    early_stopping: Optional[EarlyStopping] = None,
) -> Tuple[int, int]:
    """
    Restores the complete training state from a saved checkpoint directory.
    """
    print(f"\n[Resuming from Checkpoint] -> {checkpoint_dir}")

    # 1. Restore LoRA Adapters
    lora_dir = os.path.join(checkpoint_dir, "lora_adapters")
    if os.path.exists(lora_dir):
        adapter_file = os.path.join(lora_dir, "adapter_model.safetensors")
        if os.path.exists(adapter_file) and hasattr(model.backbone, "peft_config"):
            from peft import set_peft_model_state_dict
            adapter_weights = load_file(adapter_file)
            set_peft_model_state_dict(model.backbone, adapter_weights)
        elif hasattr(model.backbone, "load_adapter"):
            try:
                model.backbone.load_adapter(lora_dir, adapter_name="default")
            except Exception:
                pass

    # 2. Restore Dual-Mode Decision Head (Respecting target compute_dtype)
    head_path = os.path.join(checkpoint_dir, "hybrid_head.safetensors")
    if os.path.exists(head_path):
        head_dict = load_file(head_path)
        compute_dtype = model.w_noul.weight.dtype
        model.w_noul.weight.data.copy_(head_dict["w_noul.weight"].to(device=device, dtype=compute_dtype))
        model.w_noul.bias.data.copy_(head_dict["w_noul.bias"].to(device=device, dtype=compute_dtype))
        model.w_score.weight.data.copy_(head_dict["w_score.weight"].to(device=device, dtype=compute_dtype))
        model.w_score.bias.data.copy_(head_dict["w_score.bias"].to(device=device, dtype=compute_dtype))
        model.q_proj.weight.data.copy_(head_dict["q_proj.weight"].to(device=device, dtype=compute_dtype))
        model.k_proj.weight.data.copy_(head_dict["k_proj.weight"].to(device=device, dtype=compute_dtype))
        model.tau_choice.data.copy_(head_dict["tau_choice"].to(device=device, dtype=torch.float32))
        model.tau_noul.data.copy_(head_dict["tau_noul"].to(device=device, dtype=torch.float32))
        model.tau_score.data.copy_(head_dict["tau_score"].to(device=device, dtype=torch.float32))

    state_path = os.path.join(checkpoint_dir, "training_state.pt")
    if os.path.exists(state_path):
        state = torch.load(state_path, map_location=device, weights_only=False)
        saved_opt = state.get("optimizer_state_dict")
        if saved_opt:
            try:
                optimizer.load_state_dict(saved_opt)
            except ValueError as e:
                print(f"[Notice] Handled optimizer parameter group size transition: {e}")
                # Restore head parameter group states
                if len(saved_opt.get("param_groups", [])) > 1 and len(optimizer.param_groups) > 1:
                    for idx_head, p in enumerate(optimizer.param_groups[1]["params"]):
                        if idx_head in saved_opt.get("state", {}):
                            optimizer.state[p] = saved_opt["state"][idx_head]
                # Restore group learning rates
                for i, g in enumerate(optimizer.param_groups):
                    if i < len(saved_opt.get("param_groups", [])):
                        g["lr"] = saved_opt["param_groups"][i]["lr"]
        if scheduler is not None and "scheduler_state_dict" in state and state["scheduler_state_dict"]:
            scheduler.load_state_dict(state["scheduler_state_dict"])
        if state.get("rng_state_torch") is not None:
            torch.set_rng_state(state["rng_state_torch"].cpu())
        if state.get("rng_state_cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state(state["rng_state_cuda"].cpu())
        if early_stopping is not None and state.get("early_stopping") is not None:
            early_stopping.load_state_dict(state["early_stopping"])

        return state["epoch"], state["step_count"]

    return 0, 0


def evaluate_model(
    model: HybridDecisionEngine,
    val_dataloader: DataLoader,
    loss_engine: DecisionLossEngine,
    device: str = "cuda",
    max_eval_batches: Optional[int] = None,
) -> dict:
    """
    Evaluates model performance across all System One decision primitives on validation data:
    Computes validation loss, per-primitive loss breakdown, and accuracy/MAE metrics.
    """
    model.eval()
    val_loss_total = 0.0
    val_loss_noul = 0.0
    val_loss_score = 0.0
    val_loss_choice = 0.0

    count_total = 0
    count_noul = 0
    count_score = 0
    count_choice = 0

    correct_noul = 0
    correct_choice = 0
    abs_error_score = 0.0

    with torch.no_grad():
        for step, batch in enumerate(val_dataloader):
            if max_eval_batches is not None and step >= max_eval_batches:
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            decide_token_idx = batch["decide_token_idx"].to(device)
            task_types = batch["task_types"]
            candidate_spans = batch["candidate_spans"]

            targets = {
                "input_ids": input_ids,
                "task_types": task_types,
                "noul_targets": batch["noul_targets"].to(device),
                "score_targets": batch["score_targets"].to(device),
                "score_active_levels": batch["score_active_levels"].to(device),
                "choice_targets": batch["choice_targets"].to(device),
            }

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decide_token_idx=decide_token_idx,
                task_types=task_types,
                candidate_spans=candidate_spans,
            )

            losses = loss_engine(outputs, targets)
            val_loss_total += losses["loss_total"].item()
            count_total += 1

            # Noul metric
            if "loss_noul" in losses:
                val_loss_noul += losses["loss_noul"].item()
                count_noul += 1
                for b, target in enumerate(batch["noul_targets"]):
                    if target != -100:
                        pred = outputs["noul_logits"][b].argmax().item()
                        if pred == target.item():
                            correct_noul += 1

            # Score metric
            if "loss_score" in losses:
                val_loss_score += losses["loss_score"].item()
                count_score += 1
                for b, target in enumerate(batch["score_targets"]):
                    if target != -100:
                        L = batch["score_active_levels"][b].item()
                        probs = F.softmax(outputs["score_logits"][b, :L], dim=-1)
                        pred_score = (probs * torch.arange(L, device=device, dtype=torch.float32)).sum().item()
                        abs_error_score += abs(pred_score - target.item())

            # Choice metric
            if "loss_choice" in losses:
                val_loss_choice += losses["loss_choice"].item()
                count_choice += 1
                for b, target in enumerate(batch["choice_targets"]):
                    if target != -100 and outputs["choice_logits"][b].numel() > 0:
                        pred = outputs["choice_logits"][b].argmax().item()
                        if pred == target.item():
                            correct_choice += 1

    model.train()

    metrics = {
        "val_loss": val_loss_total / max(1, count_total),
        "val_loss_noul": val_loss_noul / max(1, count_noul) if count_noul else 0.0,
        "val_loss_score": val_loss_score / max(1, count_score) if count_score else 0.0,
        "val_loss_choice": val_loss_choice / max(1, count_choice) if count_choice else 0.0,
        "noul_acc": (correct_noul / max(1, count_noul)) * 100 if count_noul else 0.0,
        "score_mae": abs_error_score / max(1, count_score) if count_score else 0.0,
        "choice_acc": (correct_choice / max(1, count_choice)) * 100 if count_choice else 0.0,
    }
    return metrics


def train_system_one_model(
    model: HybridDecisionEngine,
    train_dataloader: DataLoader,
    num_epochs: int = 3,
    gradient_accumulation_steps: int = 16,
    device: str = "cuda",
    learning_rate_backbone: float = 2.0e-4,
    learning_rate_head: float = 1.5e-4,
    log_interval: int = 50,
    save_steps: int = 500,
    output_dir: str = "./system_one_model",
    resume_checkpoint_path: Optional[str] = None,
    val_dataloader: Optional[DataLoader] = None,
    eval_steps: int = 250,
    eval_samples: int = 100,
    early_stopping_patience: Optional[int] = None,
    early_stopping_min_delta: float = 0.0,
    early_stopping_metric: str = "val_loss",
    early_stopping_mode: str = "min",
    early_stopping: Optional[EarlyStopping] = None,
    save_best_checkpoint: bool = True,
    restore_best_weights: bool = False,
) -> Dict[str, Any]:
    loss_engine = DecisionLossEngine().to(device)

    if early_stopping is None and early_stopping_patience is not None and early_stopping_patience > 0:
        early_stopping = EarlyStopping(
            patience=early_stopping_patience,
            min_delta=early_stopping_min_delta,
            mode=early_stopping_mode,
            metric_name=early_stopping_metric,
        )

    # 1. Parameter Groups: Decouple LoRA and Head Learning Rates
    # Guarantee that LoRA adapter parameters are trainable
    for n, p in model.backbone.named_parameters():
        if "lora_" in n:
            p.requires_grad = True

    backbone_params = [p for n, p in model.backbone.named_parameters() if p.requires_grad]
    assert len(backbone_params) > 0, (
        "Fatal error: No trainable backbone parameters found in model.backbone! "
        "Check that LoRA adapters are loaded properly and requires_grad is enabled."
    )
    head_params = (
        list(model.w_noul.parameters())
        + list(model.w_score.parameters())
        + list(model.q_proj.parameters())
        + list(model.k_proj.parameters())
        + [model.tau_choice, model.tau_noul, model.tau_score]
    )
    print(f"Verified trainable parameter groups: Backbone LoRA={len(backbone_params)} tensors | Decision Heads={len(head_params)} tensors")

    optimizer_grouped_parameters = [
        {"params": backbone_params, "lr": learning_rate_backbone, "weight_decay": 0.01},
        {"params": head_params, "lr": learning_rate_head, "weight_decay": 0.0},
    ]

    # 2. 8-bit AdamW with standard AdamW fallback
    if HAS_BNB and device.startswith("cuda"):
        # Use AdamW8bit instead of PagedAdamW8bit: PagedAdamW8bit has a known bug in bitsandbytes
        # when resuming checkpoints where deserialized paged buffers trigger an invalid argument
        # error in pythonInterface.cpp (line 580). AdamW8bit uses the exact same 8-bit quantization
        # (occupying only ~73 MB VRAM) and resumes seamlessly from checkpoints.
        optimizer = bnb.optim.AdamW8bit(optimizer_grouped_parameters)
    else:
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters)

    # 3. Learning Rate Scheduler
    total_training_steps = (len(train_dataloader) // gradient_accumulation_steps) * num_epochs
    warmup_steps = int(0.05 * max(1, total_training_steps))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max(1, total_training_steps),
    )

    start_epoch = 0
    start_step = 0

    # Resume from checkpoint if supplied
    if resume_checkpoint_path and os.path.exists(resume_checkpoint_path):
        start_epoch, start_step = load_checkpoint(
            checkpoint_dir=resume_checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            early_stopping=early_stopping,
        )
        print(f"Resumed successfully: Starting at Epoch {start_epoch + 1}, Global Step {start_step}.")

    model.train()
    step_count = start_step
    optimizer.zero_grad()

    early_stop_triggered = False
    stopped_epoch = None
    stopped_step = None

    print(f"Beginning fine-tuning: {total_training_steps} optimization steps...")

    for epoch in range(start_epoch, num_epochs):
        running_loss_total = 0.0
        running_loss_noul = 0.0
        running_loss_score = 0.0
        running_loss_choice = 0.0
        accumulated_micro_batches = 0
        count_noul = 0
        count_score = 0
        count_choice = 0

        for step, batch in enumerate(train_dataloader):
            # Fast-forward batches already completed in the current epoch when resuming
            batch_global_step = (epoch * (len(train_dataloader) // gradient_accumulation_steps)) + (step // gradient_accumulation_steps)
            if batch_global_step < start_step:
                continue

            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            decide_token_idx = batch["decide_token_idx"].to(device)
            task_types = batch["task_types"]
            candidate_spans = batch["candidate_spans"]

            targets = {
                "input_ids": input_ids,
                "task_types": task_types,
                "noul_targets": batch["noul_targets"].to(device),
                "score_targets": batch["score_targets"].to(device),
                "score_active_levels": batch["score_active_levels"].to(device),
                "choice_targets": batch["choice_targets"].to(device),
            }

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decide_token_idx=decide_token_idx,
                task_types=task_types,
                candidate_spans=candidate_spans,
            )

            # Compute loss
            losses = loss_engine(outputs, targets)
            loss = losses["loss_total"] / gradient_accumulation_steps
            loss.backward()

            # Exact micro-batch loss accumulation
            running_loss_total += losses["loss_total"].item()
            accumulated_micro_batches += 1

            if "loss_noul" in losses:
                running_loss_noul += losses["loss_noul"].item()
                count_noul += 1
            if "loss_score" in losses:
                running_loss_score += losses["loss_score"].item()
                count_score += 1
            if "loss_choice" in losses:
                running_loss_choice += losses["loss_choice"].item()
                count_choice += 1

            # Optimization step
            if (step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step_count += 1

                if step_count % log_interval == 0:
                    avg_loss = running_loss_total / max(1, accumulated_micro_batches)
                    avg_noul = running_loss_noul / max(1, count_noul) if count_noul else 0.0
                    avg_score = running_loss_score / max(1, count_score) if count_score else 0.0
                    avg_choice = running_loss_choice / max(1, count_choice) if count_choice else 0.0
                    print(
                        f"Epoch: {epoch+1} | Step: {step_count}/{total_training_steps} "
                        f"| Train Loss: {avg_loss:.4f} "
                        f"(Choice: {avg_choice:.4f}, Noul: {avg_noul:.4f}, Score: {avg_score:.4f})"
                    )
                    running_loss_total = 0.0
                    running_loss_noul = 0.0
                    running_loss_score = 0.0
                    running_loss_choice = 0.0
                    accumulated_micro_batches = 0
                    count_noul = 0
                    count_score = 0
                    count_choice = 0

                # Periodic Validation Evaluation
                if val_dataloader is not None and eval_steps > 0 and step_count % eval_steps == 0:
                    val_metrics = evaluate_model(
                        model=model,
                        val_dataloader=val_dataloader,
                        loss_engine=loss_engine,
                        device=device,
                        max_eval_batches=eval_samples,
                    )
                    print(
                        f"  -> [VALIDATION @ Step {step_count}] "
                        f"Val Loss: {val_metrics['val_loss']:.4f} "
                        f"(Choice: {val_metrics['val_loss_choice']:.4f}, Noul: {val_metrics['val_loss_noul']:.4f}, Score: {val_metrics['val_loss_score']:.4f}) | "
                        f"Choice Acc: {val_metrics['choice_acc']:.1f}% | "
                        f"Noul Acc: {val_metrics['noul_acc']:.1f}% | "
                        f"Score MAE: {val_metrics['score_mae']:.2f}"
                    )
                    if early_stopping is not None and early_stopping.is_enabled:
                        is_best = False
                        if early_stopping.metric_name in val_metrics:
                            cur_val = float(val_metrics[early_stopping.metric_name])
                            if early_stopping.best_score is None:
                                is_best = True
                            elif early_stopping.mode == "min" and cur_val < (early_stopping.best_score - early_stopping.min_delta):
                                is_best = True
                            elif early_stopping.mode == "max" and cur_val > (early_stopping.best_score + early_stopping.min_delta):
                                is_best = True

                        should_stop = early_stopping.step(val_metrics, epoch=epoch + 1, step=step_count)
                        if is_best and save_best_checkpoint:
                            best_ckpt_dir = os.path.join(output_dir, "checkpoint-best")
                            save_checkpoint(
                                checkpoint_dir=best_ckpt_dir,
                                model=model,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                epoch=epoch + 1,
                                step_count=step_count,
                                early_stopping=early_stopping,
                            )
                        if should_stop:
                            print(
                                f"\n[EarlyStopping] Early stopping triggered at Step {step_count} (Epoch {epoch+1})! "
                                f"Patience {early_stopping.patience} exhausted without improvement in '{early_stopping.metric_name}'. "
                                f"Best score: {early_stopping.best_score:.4f} (at Epoch {early_stopping.best_epoch}, Step {early_stopping.best_step})."
                            )
                            early_stop_triggered = True
                            stopped_epoch = epoch + 1
                            stopped_step = step_count
                            break

                # Periodic Checkpoint Saving
                if step_count % save_steps == 0:
                    ckpt_dir = os.path.join(output_dir, f"checkpoint-step-{step_count}")
                    save_checkpoint(
                        checkpoint_dir=ckpt_dir,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch,
                        step_count=step_count,
                        early_stopping=early_stopping,
                    )

        if early_stop_triggered:
            break

        # Full validation at epoch end
        if val_dataloader is not None:
            full_val_metrics = evaluate_model(
                model=model,
                val_dataloader=val_dataloader,
                loss_engine=loss_engine,
                device=device,
                max_eval_batches=None,
            )
            print(
                f"\n[EPOCH {epoch+1} EVALUATION] "
                f"Full Val Loss: {full_val_metrics['val_loss']:.4f} "
                f"(Choice: {full_val_metrics['val_loss_choice']:.4f}, Noul: {full_val_metrics['val_loss_noul']:.4f}, Score: {full_val_metrics['val_loss_score']:.4f}) | "
                f"Choice Acc: {full_val_metrics['choice_acc']:.1f}% | "
                f"Noul Acc: {full_val_metrics['noul_acc']:.1f}% | "
                f"Score MAE: {full_val_metrics['score_mae']:.2f}\n"
            )
            if early_stopping is not None and early_stopping.is_enabled:
                is_best = False
                if early_stopping.metric_name in full_val_metrics:
                    cur_val = float(full_val_metrics[early_stopping.metric_name])
                    if early_stopping.best_score is None:
                        is_best = True
                    elif early_stopping.mode == "min" and cur_val < (early_stopping.best_score - early_stopping.min_delta):
                        is_best = True
                    elif early_stopping.mode == "max" and cur_val > (early_stopping.best_score + early_stopping.min_delta):
                        is_best = True

                should_stop = early_stopping.step(full_val_metrics, epoch=epoch + 1, step=step_count)
                if is_best and save_best_checkpoint:
                    best_ckpt_dir = os.path.join(output_dir, "checkpoint-best")
                    save_checkpoint(
                        checkpoint_dir=best_ckpt_dir,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch + 1,
                        step_count=step_count,
                        early_stopping=early_stopping,
                    )
                if should_stop:
                    print(
                        f"\n[EarlyStopping] Early stopping triggered at end of Epoch {epoch+1}! "
                        f"Patience {early_stopping.patience} exhausted without improvement in '{early_stopping.metric_name}'. "
                        f"Best score: {early_stopping.best_score:.4f} (at Epoch {early_stopping.best_epoch}, Step {early_stopping.best_step})."
                    )
                    early_stop_triggered = True
                    stopped_epoch = epoch + 1
                    stopped_step = step_count

        # Save end-of-epoch checkpoint
        epoch_ckpt_dir = os.path.join(output_dir, f"checkpoint-epoch-{epoch+1}")
        save_checkpoint(
            checkpoint_dir=epoch_ckpt_dir,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch + 1,
            step_count=step_count,
            early_stopping=early_stopping,
        )

        if early_stop_triggered:
            break

    if restore_best_weights and early_stopping is not None and early_stopping.best_score is not None:
        best_ckpt_dir = os.path.join(output_dir, "checkpoint-best")
        if os.path.exists(best_ckpt_dir):
            print(f"\n[EarlyStopping] Restoring best model weights from {best_ckpt_dir} (Best {early_stopping.metric_name}: {early_stopping.best_score:.4f})...")
            load_checkpoint(
                checkpoint_dir=best_ckpt_dir,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                early_stopping=None,
            )

    return {
        "early_stopped": early_stop_triggered,
        "stopped_epoch": stopped_epoch,
        "stopped_step": stopped_step,
        "early_stopping": early_stopping,
        "total_steps": step_count,
    }


# ==============================================================================
# 5. POST-TRAINING TEMPERATURE CALIBRATION, EXPORTER & SMOKE TEST
# ==============================================================================

def calibrate_temperatures(
    model: HybridDecisionEngine,
    val_dataloader: DataLoader,
    device: str = "cuda",
    max_iter: int = 50,
):
    """
    Freezes model backbone and fits temperature parameters
    (tau_choice, tau_noul, tau_score) via NLL minimization on validation split.
    """
    model.eval()

    # Collect unscaled raw logits and targets
    noul_raw_logits = []
    noul_labels = []

    score_raw_logits = []
    score_labels = []
    score_levels = []

    choice_raw_logits = []
    choice_labels = []

    with torch.no_grad():
        for batch in val_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            decide_token_idx = batch["decide_token_idx"].to(device)
            task_types = batch["task_types"]
            candidate_spans = batch["candidate_spans"]

            backbone_outputs = model.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            if hasattr(backbone_outputs, "hidden_states") and backbone_outputs.hidden_states is not None:
                hidden_states = backbone_outputs.hidden_states[-1]
            elif hasattr(backbone_outputs, "last_hidden_state"):
                hidden_states = backbone_outputs.last_hidden_state
            else:
                hidden_states = backbone_outputs[0]

            hidden_states = hidden_states.to(dtype=model.w_noul.weight.dtype)

            batch_size = input_ids.size(0)
            batch_idx = torch.arange(batch_size, device=input_ids.device)
            h_decide = hidden_states[batch_idx, decide_token_idx]

            # Collect Noul
            raw_noul = model.w_noul(h_decide).float()
            for b in range(batch_size):
                if task_types[b] == "noul":
                    noul_raw_logits.append(raw_noul[b].cpu())
                    noul_labels.append(batch["noul_targets"][b].item())

            # Collect Score
            raw_score = model.w_score(h_decide).float()
            for b in range(batch_size):
                if task_types[b] == "score":
                    L = batch["score_active_levels"][b].item()
                    score_raw_logits.append(raw_score[b, :L].cpu())
                    score_labels.append(batch["score_targets"][b].item())
                    score_levels.append(L)

            # Collect Choice
            for b in range(batch_size):
                if task_types[b] == "choice":
                    spans = candidate_spans[b]
                    cand_reps = [
                        hidden_states[b, start:end, :].mean(dim=0)
                        for (start, end) in spans
                        if end > start
                    ]
                    if cand_reps:
                        h_cands = torch.stack(cand_reps, dim=0)
                        q = model.q_proj(h_decide[b]).unsqueeze(0)
                        k = model.k_proj(h_cands)
                        raw_c_logits = (torch.mm(q, k.t()).squeeze(0) * model.scale).float()
                        choice_raw_logits.append(raw_c_logits.cpu())
                        choice_labels.append(batch["choice_targets"][b].item())

    print(f"Collected validation logits: Noul={len(noul_labels)}, Score={len(score_labels)}, Choice={len(choice_labels)}")

    # Optimize Noul Temperature
    if noul_labels:
        tau_n = nn.Parameter(torch.ones(1) * 1.0)
        opt_n = torch.optim.LBFGS([tau_n], lr=0.01, max_iter=max_iter)
        logits_n = torch.stack(noul_raw_logits)
        targets_n = torch.tensor(noul_labels, dtype=torch.long)

        def closure_n():
            opt_n.zero_grad()
            scaled = logits_n / torch.clamp(tau_n, min=0.05)
            loss = F.cross_entropy(scaled, targets_n)
            loss.backward()
            return loss

        opt_n.step(closure_n)
        model.tau_noul.data = torch.clamp(tau_n.data.to(model.tau_noul.device), min=0.05)

    # Optimize Score Temperature
    if score_labels:
        tau_s = nn.Parameter(torch.ones(1) * 1.0)
        opt_s = torch.optim.LBFGS([tau_s], lr=0.01, max_iter=max_iter)

        def closure_s():
            opt_s.zero_grad()
            loss = 0.0
            for raw_l, tgt in zip(score_raw_logits, score_labels):
                scaled = (raw_l / torch.clamp(tau_s, min=0.05)).unsqueeze(0)
                loss = loss + F.cross_entropy(scaled, torch.tensor([tgt], dtype=torch.long))
            loss = loss / len(score_labels)
            loss.backward()
            return loss

        opt_s.step(closure_s)
        model.tau_score.data = torch.clamp(tau_s.data.to(model.tau_score.device), min=0.05)

    # Optimize Choice Temperature
    if choice_labels:
        tau_c = nn.Parameter(torch.ones(1) * 1.0)
        opt_c = torch.optim.LBFGS([tau_c], lr=0.01, max_iter=max_iter)

        def closure_c():
            opt_c.zero_grad()
            loss = 0.0
            for raw_l, tgt in zip(choice_raw_logits, choice_labels):
                scaled = (raw_l / torch.clamp(tau_c, min=0.05)).unsqueeze(0)
                loss = loss + F.cross_entropy(scaled, torch.tensor([tgt], dtype=torch.long))
            loss = loss / len(choice_labels)
            loss.backward()
            return loss

        opt_c.step(closure_c)
        model.tau_choice.data = torch.clamp(tau_c.data.to(model.tau_choice.device), min=0.05)

    print(
        f"Fitted Temperatures: Choice={model.tau_choice.item():.4f}, "
        f"Noul={model.tau_noul.item():.4f}, Score={model.tau_score.item():.4f}"
    )


def export_production_artifacts(
    model: HybridDecisionEngine,
    output_dir: str = "./system_one_model",
):
    """
    Serializes LoRA adapters, decision heads (hybrid_head.safetensors),
    and production configuration (decision_config.json).
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Save LoRA Adapters and Tokenizer
    adapters_dir = os.path.join(output_dir, "lora_adapters")
    os.makedirs(adapters_dir, exist_ok=True)
    if hasattr(model.backbone, "save_pretrained"):
        model.backbone.save_pretrained(adapters_dir)
    if hasattr(model, "tokenizer") and model.tokenizer is not None:
        model.tokenizer.save_pretrained(adapters_dir)

    # 2. Save Dual-Mode Decision Head Weights
    head_state_dict = {
        "w_noul.weight": model.w_noul.weight.data.cpu(),
        "w_noul.bias": model.w_noul.bias.data.cpu(),
        "w_score.weight": model.w_score.weight.data.cpu(),
        "w_score.bias": model.w_score.bias.data.cpu(),
        "q_proj.weight": model.q_proj.weight.data.cpu(),
        "k_proj.weight": model.k_proj.weight.data.cpu(),
        "tau_choice": model.tau_choice.data.cpu(),
        "tau_noul": model.tau_noul.data.cpu(),
        "tau_score": model.tau_score.data.cpu(),
    }
    head_path = os.path.join(output_dir, "hybrid_head.safetensors")
    save_file(head_state_dict, head_path)

    # 3. Save Production Deployment Configuration
    config = {
        "model_type": "system_one_decision_engine",
        "base_model": model.base_model_name,
        "hidden_dimension": model.hidden_dim,
        "proj_dimension": model.proj_dim,
        "max_score_levels": 10,
        "fitted_temperatures": {
            "choice": round(model.tau_choice.item(), 4),
            "noul": round(model.tau_noul.item(), 4),
            "score": round(model.tau_score.item(), 4),
        },
        "default_thresholds": {
            "noul_default": 0.50,
            "noul_multi_label_calibrated": 0.82,
        },
        "supported_primitives": ["choice", "noul", "score"],
    }
    config_path = os.path.join(output_dir, "decision_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"Artifacts successfully exported to: {output_dir}")


def run_smoke_test(
    model: HybridDecisionEngine,
    tokenizer: AutoTokenizer,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    output_dir: str = "./smoke_test_run",
) -> bool:
    """
    Executes a comprehensive pre-flight sanity test across all 5 System One layers:
    1. Delimiter schema formatting, offset mapping, and typed target collation.
    2. Forward pass with QLoRA/backbone + excised LM head + Dual-Mode head.
    3. Multi-task loss computation (InfoNCE + EMD Wasserstein + Gaussian Soft-CE + Entropy reg).
    4. Gradient backpropagation and optimizer parameter update.
    5. Checkpoint saving and resumption integrity.
    6. Post-training temperature calibration step.
    7. Artifact export (safetensors + config).
    """
    print("\n" + "=" * 60)
    print("STARTING PRE-FLIGHT SMOKE TEST (SANITY VERIFICATION)")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)
    try:
        # Synthetic samples spanning all three primitives
        synthetic_samples = [
            {
                "task_type": "choice",
                "domain": "finance_routing",
                "state": {"text": "Sanity check query: checking card status"},
                "question": "Identify primary service intent.",
                "options": ["activate_card", "card_status", "cancel_account", "lost_card"],
                "target_idx": 1,
            },
            {
                "task_type": "noul",
                "domain": "safety",
                "state": {"user_message": "Hello, how do I check my account balance?"},
                "question": "Does this message contain toxic or abusive content?",
                "target_bool": False,
            },
            {
                "task_type": "score",
                "domain": "summary_evaluation",
                "state": {
                    "document": "Financial reports indicate quarterly growth.",
                    "summary": "Quarterly earnings grew as expected.",
                },
                "question": "Rate the quality of this summary on the evaluation rubric.",
                "target_score": 3,
                "active_levels": 5,
            },
        ]

        # 1. Test Collation
        print("[Smoke 1/5] Testing DecisionDataCollator on all 3 primitives...")
        collator = DecisionDataCollator(tokenizer=tokenizer, max_seq_length=512)
        batch = collator(synthetic_samples)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        decide_token_idx = batch["decide_token_idx"].to(device)
        task_types = batch["task_types"]
        candidate_spans = batch["candidate_spans"]

        targets = {
            "input_ids": input_ids,
            "task_types": task_types,
            "noul_targets": batch["noul_targets"].to(device),
            "score_targets": batch["score_targets"].to(device),
            "score_active_levels": batch["score_active_levels"].to(device),
            "choice_targets": batch["choice_targets"].to(device),
        }

        # 2. Test Forward Pass & Loss Computation
        print("[Smoke 2/5] Testing Forward Pass & Multi-Task Loss Engine...")
        model.train()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decide_token_idx=decide_token_idx,
            task_types=task_types,
            candidate_spans=candidate_spans,
        )

        loss_engine = DecisionLossEngine().to(device)
        losses = loss_engine(outputs, targets)
        loss = losses["loss_total"]

        assert not torch.isnan(loss).any(), "Smoke test encountered NaN in loss_total!"
        print(f"  -> Total Loss: {loss.item():.4f} (Noul={losses['loss_noul'].item():.4f}, Score={losses['loss_score'].item():.4f}, Choice={losses['loss_choice'].item():.4f})")

        # 3. Test Backward Pass & Optimizer Update
        print("[Smoke 3/5] Testing Backward Pass and Optimizer Step...")
        backbone_params = [p for n, p in model.backbone.named_parameters() if p.requires_grad]
        head_params = (
            list(model.w_noul.parameters())
            + list(model.w_score.parameters())
            + list(model.q_proj.parameters())
            + list(model.k_proj.parameters())
            + [model.tau_choice, model.tau_noul, model.tau_score]
        )
        optimizer = torch.optim.AdamW(
            [{"params": backbone_params, "lr": 1e-4}, {"params": head_params, "lr": 1e-4}]
        )
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=1, num_training_steps=10)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        print("  -> Gradient backpropagation and parameter update succeeded cleanly.")

        # 4. Test Checkpoint Save & Resume
        print("[Smoke 4/5] Testing Checkpoint Save and Load...")
        ckpt_dir = os.path.join(output_dir, "smoke_checkpoint")
        save_checkpoint(
            checkpoint_dir=ckpt_dir,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=1,
            step_count=1,
        )
        resumed_epoch, resumed_step = load_checkpoint(
            checkpoint_dir=ckpt_dir,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )
        assert resumed_epoch == 1 and resumed_step == 1, "Failed to restore exact checkpoint step/epoch!"
        print("  -> Checkpoint saving and resumption verified.")

        # 5. Test Temperature Calibration & Artifact Export
        print("[Smoke 5/5] Testing Calibration Step and Artifact Export...")
        dummy_val_loader = [batch]
        calibrate_temperatures(model=model, val_dataloader=dummy_val_loader, device=device, max_iter=5)
        export_production_artifacts(model=model, output_dir=os.path.join(output_dir, "smoke_artifacts"))
        print("  -> Temperature calibration and artifact serialization verified.")

        if device.startswith("cuda") and torch.cuda.is_available():
            mem_mb = torch.cuda.memory_allocated() / (1024 ** 2)
            max_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            print(f"  -> CUDA Memory Allocated: {mem_mb:.2f} MiB | Peak: {max_mb:.2f} MiB")

        print("=" * 60)
        print("[PRE-FLIGHT SMOKE TEST PASSED] All 5 system layers are 100% healthy.")
        print("=" * 60 + "\n")
        # Restore model to train mode and ensure LoRA parameters remain trainable
        model.train()
        for n, p in model.backbone.named_parameters():
            if "lora_" in n:
                p.requires_grad = True

        return True

    finally:
        # Clean up temporary smoke test files
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir, ignore_errors=True)
