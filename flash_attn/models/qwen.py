# Copyright (c) 2023, GGGGGGXY

import json
import math
import re
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn.functional as F
from einops import rearrange
from transformers import AutoConfig, GPT2Config, PretrainedConfig


def remap_state_dict_hf_qwen(state_dict, config):
    # Word embedding
    def key_mapping_layers(key):
        return re.sub(r"^transformer.h.", "transformer.layers.", key)

    state_dict = OrderedDict((key_mapping_layers(k), v) for k, v in state_dict.items())

    def key_mapping_emb(key):
        return re.sub(r"^transformer.wte.", "transformer.embeddings.word_embeddings.", key)

    state_dict = OrderedDict((key_mapping_emb(k), v) for k, v in state_dict.items())
    word_embeddings = state_dict.pop("transformer.embeddings.word_embeddings.weight")
    # It's possible that vocab_size is padded to be a multiple of 8, for example.
    pad_vocab_size_multiple = getattr(config, "pad_vocab_size_multiple", 1)
    vocab_size = math.ceil(config.vocab_size / pad_vocab_size_multiple) * pad_vocab_size_multiple
    state_dict["transformer.embeddings.word_embeddings.weight"] = F.pad(
        word_embeddings, (0, 0, 0, vocab_size - word_embeddings.shape[0])
    )
    if getattr(config, "tie_word_embeddings"):
        state_dict["lm_head.weight"] = state_dict["transformer.embeddings.word_embeddings.weight"]
    else:
        output_embeddings = state_dict.pop("lm_head.weight")
        # Need to recompute vocab_size since Baichuan shards the word embeddings and output embeddings
        # differently.
        vocab_size = (
            math.ceil(config.vocab_size / pad_vocab_size_multiple) * pad_vocab_size_multiple
        )
        # It's possible that vocab_size is padded to be a multiple of 8, for example.
        state_dict["lm_head.weight"] = F.pad(
            output_embeddings, (0, 0, 0, vocab_size - output_embeddings.shape[0])
        )

    # LayerNorm
    def key_mapping_ln(key):
        key = re.sub(r"^transformer.layers.(\d+).ln_1.", r"transformer.layers.\1.norm1.", key)
        key = re.sub(r"^transformer.layers.(\d+).ln_2.", r"transformer.layers.\1.norm2.", key)
        return key

    state_dict = OrderedDict((key_mapping_ln(k), v) for k, v in state_dict.items())

    # MLP
    for l in range(config.n_layer):
        w1 = state_dict.pop(f"transformer.layers.{l}.mlp.w1.weight")
        w2 = state_dict.pop(f"transformer.layers.{l}.mlp.w2.weight")
        # Our ordering is different
        state_dict[f"transformer.layers.{l}.mlp.fc1.weight"] = torch.cat([w1, w2], dim=0)

    def key_mapping_mlp(key):
        return re.sub(
            r"^transformer.layers.(\d+).mlp.c_proj.", r"transformer.layers.\1.mlp.fc2.", key
        )

    state_dict = OrderedDict((key_mapping_mlp(k), v) for k, v in state_dict.items())

    # Attention
    def key_mapping_attn(key):
        key = re.sub(
            r"^transformer.layers.(\d+).attn.c_attn.", r"transformer.layers.\1.mixer.Wqkv.", key
        )
        key = re.sub(
            r"^transformer.layers.(\d+).attn.c_proj.", r"transformer.layers.\1.mixer.out_proj.", key
        )
        return key

    state_dict = OrderedDict((key_mapping_attn(k), v) for k, v in state_dict.items())
    return state_dict


def config_from_checkpoint(checkpoint_path: str, model_name: str) -> PretrainedConfig:
    """Load a QwenConfig from a checkpoint path."""
    config = AutoConfig.from_pretrained(Path(checkpoint_path) / model_name, trust_remote_code=True)
    return config


def state_dicts_from_checkpoint(checkpoint_path: str, model_name: str) -> dict:
    # huggingface checkpoint splited layer by layer, so we just merge to one state_dict is fine
    layer_state_dicts = [
        torch.load(path, map_location="cpu")
        for path in sorted((Path(checkpoint_path) / model_name).glob("pytorch_model*.bin"))
    ]
    merged_state_dict = OrderedDict()
    for sd in layer_state_dicts:
        for k, v in sd.items():
            merged_state_dict[k] = v
    return [merged_state_dict]


def qwen_config_to_gpt2_config(qwen_config: PretrainedConfig) -> GPT2Config:
    return GPT2Config(
        vocab_size=qwen_config.vocab_size,
        n_positions=0,  # No absolute position embedding
        n_embd=qwen_config.hidden_size,
        n_layer=qwen_config.num_hidden_layers,
        n_head=qwen_config.num_attention_heads,
        n_inner=qwen_config.intermediate_size // 2,  # qwen already double the size
        activation_function="swiglu",  # Hardcode since HF calls it 'silu'
        # qwen doesn't have dropout, idk if it's because they only release the inference code
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        layer_norm_epsilon=qwen_config.layer_norm_epsilon,
        initializer_range=qwen_config.initializer_range,
        bos_token_id=qwen_config.bos_token_id,
        eos_token_id=qwen_config.eos_token_id,
        # These are new arguments not in the original GPT2Config
        pad_token_id=qwen_config.pad_token_id,  # Idk if this does anything
        rms_norm=True,
        rotary_emb_fraction=1.0,
        rotary_emb_interleaved=False,
        tie_word_embeddings=False,
        qkv_proj_bias=True,
        out_proj_bias=False,
        mlp_fc1_bias=False,
        mlp_fc2_bias=False,
    )
