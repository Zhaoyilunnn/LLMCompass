"""Factory that instantiates modular transformer operators from specs."""

from __future__ import annotations

from typing import Literal, Mapping, Any

from software_model.llm_spec import (
    TransformerBlockSpec,
    AttentionSpec,
    FeedForwardSpec,
    make_transformer_block_spec,
)
from software_model.attention_registry import (
    get_attention_builder,
    register_default_attention_builders,
)
from software_model.utils import DataType


def build_transformer_block_operator(
    spec: TransformerBlockSpec,
    data_type: DataType,
):
    """Return an Operator instance matching the provided spec."""

    register_default_attention_builders()
    attn = spec.attention
    ffn = spec.feedforward
    builder = get_attention_builder(attn.kind)
    return builder(attn, ffn, data_type)


def transformer_block_from_dict(
    config: Mapping[str, Any],
    *,
    data_type: DataType,
):
    """Convenience helper that builds a spec then operator from a python dict."""

    mode: Literal["init", "autoregressive"] = config.get("mode", "init")
    d_model = int(config["d_model"])
    num_heads = int(config["num_heads"])
    device_count = int(config.get("device_count", 1))
    expansion = int(config.get("expansion", 4))
    attention_kind = str(config.get("attention_kind", "mha"))
    num_kv_heads = config.get("num_kv_heads", None)
    if num_kv_heads is not None:
        num_kv_heads = int(num_kv_heads)
    raw_use_allreduce = config.get("use_allreduce", None)

    def _resolve_bool(name: str, fallback: bool) -> bool:
        if name in config:
            return bool(config[name])
        if raw_use_allreduce is not None:
            return bool(raw_use_allreduce)
        return fallback

    use_attention_allreduce = _resolve_bool("use_attention_allreduce", True)
    use_ffn_allreduce = _resolve_bool("use_ffn_allreduce", True)

    spec = make_transformer_block_spec(
        mode=mode,
        d_model=d_model,
        num_heads=num_heads,
        device_count=device_count,
        attention_kind=attention_kind,
        num_kv_heads=num_kv_heads,
        expansion=expansion,
        use_attention_allreduce=use_attention_allreduce,
        use_ffn_allreduce=use_ffn_allreduce,
    )
    return build_transformer_block_operator(spec, data_type)
