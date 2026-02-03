"""Simple driver for modular transformer (MHA vs GQA).

This script mirrors the structure of ``ae/figure5/ijkl/test_transformer.py`` but
uses the new spec/factory path in ``software_model.llm_factory``. It is
intended mainly for sanity-checking the GQA attention wiring and to compare the
reported memory requirements between MHA and GQA in autoregressive mode.
"""

from __future__ import annotations

import argparse

from hardware_model.system import system_dict
from software_model.llm_factory import transformer_block_from_dict
from software_model.utils import data_type_dict, Tensor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["init", "autoregressive"],
        default="autoregressive",
        help="Workload mode",
    )
    parser.add_argument(
        "--attention-kind",
        choices=["mha", "gqa"],
        default="mha",
        help="Attention kind to instantiate",
    )
    parser.add_argument(
        "--num-kv-heads",
        type=int,
        default=None,
        help="Number of KV heads for GQA (required if --attention-kind gqa)",
    )
    parser.add_argument(
        "--d-model",
        type=int,
        default=12288,
        help="Model dimension d_model",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=96,
        help="Number of attention heads",
    )
    parser.add_argument(
        "--device-count",
        type=int,
        default=4,
        help="Number of tensor-parallel devices",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=2048,
        help="Sequence length for init / prefill",
    )
    parser.add_argument(
        "--output-tokens",
        type=int,
        default=1024,
        help="Number of autoregressive decoding steps",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size",
    )
    parser.add_argument(
        "--system",
        default="A100_4_fp16",
        help="System key in hardware_model.system.system_dict",
    )
    parser.add_argument(
        "--roofline",
        action="store_true",
        help="Run roofline model instead of compile-and-simulate",
    )
    args = parser.parse_args()

    if args.attention_kind == "gqa" and args.num_kv_heads is None:
        parser.error("--num-kv-heads is required when --attention-kind gqa")

    config = {
        "mode": args.mode,
        "d_model": args.d_model,
        "num_heads": args.num_heads,
        "device_count": args.device_count,
        "attention_kind": args.attention_kind,
        "num_kv_heads": args.num_kv_heads,
    }

    data_type = data_type_dict["fp16"]
    block = transformer_block_from_dict(config, data_type=data_type)
    system = system_dict[args.system]

    bs = args.batch_size
    s = args.seq_len

    if args.mode == "init":
        print(f"Running init mode with attention={args.attention_kind}")
        inp = Tensor([bs, s, args.d_model], data_type)
        _ = block(inp)
        if args.roofline:
            latency = block.roofline_model(system)
        else:
            latency = block.compile_and_simulate(
                system,
                compile_mode="heuristic-GPU",
            )
        print(f"Latency: {latency}")
    else:
        print(
            f"Running autoregressive mode with attention={args.attention_kind}, "
            f"output_tokens={args.output_tokens}",
        )
        inp = Tensor([bs, 1, args.d_model], data_type)
        seq_len_total = s + args.output_tokens
        _ = block(inp, seq_len_total)
        if args.roofline:
            latency = block.roofline_model(system)
        else:
            latency = block.compile_and_simulate(
                system,
                compile_mode="heuristic-GPU",
            )
        print(f"Latency: {latency}")

    # Expose the attention stage memory requirement when available.
    attn_stage = getattr(block, "attention_stage", None)
    mem_req = getattr(attn_stage, "memory_requirement", None)
    if mem_req is not None:
        print(f"Attention memory_requirement: {mem_req}")


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
