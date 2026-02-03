"""Attention builder registry for transformer factory."""

from __future__ import annotations

from typing import Callable, Dict

from software_model.utils import DataType
from software_model.transformer_components import (
    build_tp_init_block,
    build_tp_autoreg_block,
    TPInitStageGQA,
    TPAutoregStageGQA,
    TPFeedForwardStage,
    ModularTransformerBlockInitTP,
    ModularTransformerBlockAutoTP,
)
from software_model.llm_spec import AttentionSpec, FeedForwardSpec


AttentionBuilder = Callable[[AttentionSpec, FeedForwardSpec, DataType], object]


REGISTRY: Dict[str, AttentionBuilder] = {}


def register_attention(kind: str, builder: AttentionBuilder):
    if kind in REGISTRY:
        raise ValueError(f"Attention kind {kind} already registered")
    REGISTRY[kind] = builder


def get_attention_builder(kind: str) -> AttentionBuilder:
    if kind not in REGISTRY:
        raise KeyError(f"Attention kind {kind} is not registered")
    return REGISTRY[kind]


def _default_mha_builder(
    attn: AttentionSpec, ffn: FeedForwardSpec, data_type: DataType
):
    if attn.mode == "init":
        return build_tp_init_block(
            d_model=attn.d_model,
            n_heads=attn.num_heads,
            device_count=attn.device_count,
            data_type=data_type,
            ffn_multiplier=ffn.expansion,
            use_attention_allreduce=attn.use_allreduce,
            use_ffn_allreduce=ffn.use_allreduce,
        )
    if attn.mode == "autoregressive":
        return build_tp_autoreg_block(
            d_model=attn.d_model,
            n_heads=attn.num_heads,
            device_count=attn.device_count,
            data_type=data_type,
            ffn_multiplier=ffn.expansion,
            use_attention_allreduce=attn.use_allreduce,
            use_ffn_allreduce=ffn.use_allreduce,
        )
    raise ValueError(f"Unknown attention mode {attn.mode}")


def _gqa_builder(attn: AttentionSpec, ffn: FeedForwardSpec, data_type: DataType):
    """GQA builder for init and autoregressive workloads."""

    if attn.num_kv_heads is None:
        raise ValueError("GQA attention requires num_kv_heads")

    if attn.num_heads % attn.num_kv_heads != 0:
        raise ValueError("num_heads must be divisible by num_kv_heads for GQA")

    if attn.mode == "init":
        attention = TPInitStageGQA(
            d_model=attn.d_model,
            n_heads=attn.num_heads,
            device_count=attn.device_count,
            data_type=data_type,
            num_kv_heads=attn.num_kv_heads,
            use_allreduce=attn.use_allreduce,
        )
        feedforward = TPFeedForwardStage(
            d_model=attn.d_model,
            device_count=attn.device_count,
            data_type=data_type,
            ffn_multiplier=ffn.expansion,
            use_allreduce=ffn.use_allreduce,
        )
        return ModularTransformerBlockInitTP(attention, feedforward)

    if attn.mode != "autoregressive":
        raise ValueError(f"GQA unsupported attention mode {attn.mode}")

    attention = TPAutoregStageGQA(
        d_model=attn.d_model,
        n_heads=attn.num_heads,
        device_count=attn.device_count,
        data_type=data_type,
        num_kv_heads=attn.num_kv_heads,
        use_allreduce=attn.use_allreduce,
    )
    feedforward = TPFeedForwardStage(
        d_model=attn.d_model,
        device_count=attn.device_count,
        data_type=data_type,
        ffn_multiplier=ffn.expansion,
        use_allreduce=ffn.use_allreduce,
    )
    return ModularTransformerBlockAutoTP(attention, feedforward)


def register_default_attention_builders():
    if REGISTRY:
        return
    register_attention("mha", _default_mha_builder)
    register_attention("gqa", _gqa_builder)
