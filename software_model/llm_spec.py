from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class AttentionSpec:
    mode: Literal["init", "autoregressive"]
    d_model: int
    num_heads: int
    device_count: int
    kind: str = "mha"
    num_kv_heads: int | None = None
    use_allreduce: bool = True


@dataclass(frozen=True)
class FeedForwardSpec:
    expansion: int = 4
    use_allreduce: bool = True


@dataclass(frozen=True)
class TransformerBlockSpec:
    attention: AttentionSpec
    feedforward: FeedForwardSpec


def make_transformer_block_spec(
    *,
    mode: Literal["init", "autoregressive"],
    d_model: int,
    num_heads: int,
    device_count: int,
    attention_kind: str = "mha",
    num_kv_heads: int | None = None,
    expansion: int = 4,
    use_attention_allreduce: bool = True,
    use_ffn_allreduce: bool = True,
) -> TransformerBlockSpec:
    attention = AttentionSpec(
        mode=mode,
        d_model=d_model,
        num_heads=num_heads,
        device_count=device_count,
        kind=attention_kind,
        num_kv_heads=num_kv_heads,
        use_allreduce=use_attention_allreduce,
    )
    feedforward = FeedForwardSpec(
        expansion=expansion,
        use_allreduce=use_ffn_allreduce,
    )
    return TransformerBlockSpec(
        attention=attention,
        feedforward=feedforward,
    )
