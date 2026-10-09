import pytest
import torch
import torch.nn as nn
from transformers import Qwen2Config, Qwen2ForCausalLM
from peft import get_peft_model, LoraConfig
from fine_tuning_engine import excise_lm_head, HybridDecisionEngine


def test_peft_lm_head_direct_delattr_fails():
    """
    Confirms the root cause: PeftModelForCausalLM forwards getattr to base_model,
    but does NOT override __delattr__, so 'del peft_model.lm_head' raises AttributeError.
    """
    config = Qwen2Config(
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
    )
    raw_model = Qwen2ForCausalLM(config)
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj"],
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(raw_model, peft_config)

    assert hasattr(peft_model, "lm_head")
    with pytest.raises(AttributeError, match="has no attribute 'lm_head'"):
        del peft_model.lm_head


def test_excise_lm_head_on_peft_model():
    """
    Validates that excise_lm_head properly navigates the PeftModel module hierarchy,
    replaces lm_head with nn.Identity(), and preserves forward execution.
    """
    config = Qwen2Config(
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
    )
    raw_model = Qwen2ForCausalLM(config)
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj"],
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(raw_model, peft_config)

    success = excise_lm_head(peft_model)
    assert success is True
    assert isinstance(peft_model.lm_head, nn.Identity)

    # Test forward pass with Identity lm_head
    input_ids = torch.randint(0, 1000, (2, 8))
    out = peft_model(input_ids, output_hidden_states=True, return_dict=True)
    # With lm_head as Identity, logits match the hidden state shape [B, S, H]
    assert out.logits.shape == (2, 8, 64)
    assert out.hidden_states[-1].shape == (2, 8, 64)


def test_excise_lm_head_on_raw_model():
    """
    Validates that excise_lm_head works on standard non-PEFT models as well.
    """
    config = Qwen2Config(
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
    )
    raw_model = Qwen2ForCausalLM(config)
    success = excise_lm_head(raw_model)
    assert success is True
    assert isinstance(raw_model.lm_head, nn.Identity)
