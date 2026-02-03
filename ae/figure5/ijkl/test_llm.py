from software_model.utils import data_type_dict, Tensor
from software_model.llm_factory import transformer_block_from_dict
from hardware_model.system import system_dict
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", action="store_true", help="initial computation")
    parser.add_argument("--gpu", action="store_true", help="Enable GPU")
    parser.add_argument("--simgpu", action="store_true", help="Enable simulation")
    parser.add_argument("--simgpu-hbf", action="store_true", help="Enable simulation")
    parser.add_argument("--simtpu", action="store_true", help="Enable simulation")
    parser.add_argument("--roofline", action="store_true", help="use roofline")
    parser.add_argument(
        "--exclude-fixed-latency",
        action="store_true",
        help="Exclude fixed IO latency overheads",
    )
    parser.add_argument(
        "--bandwidth",
        type=float,
        default=None,
        help="Override IO bandwidth (B/s) for A100_HBF systems",
    )
    parser.add_argument(
        "--fixed-io-write-coeff",
        type=float,
        default=1.0,
        help="Coefficient x for write-side fixed IO latency (total fixed IO = read + x * read)",
    )
    parser.add_argument(
        "--fixed-latency",
        type=float,
        default=None,
        help="Override IO fixed latency (s) for A100_HBF systems",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=None,
        help="Sequence length s used for transformer inputs (default: 2048)",
    )
    parser.add_argument(
        "--attention-kind",
        choices=["mha", "gqa"],
        default="mha",
        help="Attention kind (mha or gqa)",
    )
    parser.add_argument(
        "--num-kv-heads",
        type=int,
        default=None,
        help="Number of KV heads for GQA (required if --attention-kind gqa)",
    )
    args = parser.parse_args()

    bs = 8
    s = 2048 if args.seq_len is None else args.seq_len
    seq_len_tag = "" if args.seq_len is None else f"_seq{args.seq_len}"

    # Attention kind / KV head tags for filenames
    attn_tag = f"_{args.attention_kind}"
    kv_tag = f"_kv{args.num_kv_heads}" if args.attention_kind == "gqa" else ""

    if args.attention_kind == "gqa" and args.num_kv_heads is None:
        parser.error("--num-kv-heads is required when --attention-kind gqa")

    if args.init:
        config = {
            "mode": "init",
            "d_model": 12288,
            "num_heads": 96,
            "device_count": 4,
            "attention_kind": args.attention_kind,
            "num_kv_heads": args.num_kv_heads,
        }
        model = transformer_block_from_dict(config, data_type=data_type_dict["fp16"])
        print("Initial computation")
        if args.simgpu:
            A100_system = system_dict["A100_4_fp16"]
            # from design_space_exploration.dse import read_architecture_template, template_to_system
            # arch_specs = read_architecture_template("configs/template.json")
            # A100_system = template_to_system(arch_specs)
            _ = model(Tensor([bs, s, 12288], data_type_dict["fp16"]))
            if args.roofline:
                model.roofline_model(A100_system)
                file_name = (
                    f"transformer_A100_roofline{attn_tag}{kv_tag}{seq_len_tag}.csv"
                )
            else:
                model.compile_and_simulate(
                    A100_system,
                    compile_mode="heuristic-GPU",
                    include_fixed_io_latency=(not args.exclude_fixed_latency),
                )
                file_name = (
                    f"transformer_A100_sim_excl{attn_tag}{kv_tag}{seq_len_tag}.csv"
                    if args.exclude_fixed_latency
                    else f"transformer_A100_sim{attn_tag}{kv_tag}{seq_len_tag}.csv"
                )

        if args.simgpu_hbf:
            A100_system = system_dict["A100_4_fp16_HBF"]
            # from design_space_exploration.dse import read_architecture_template, template_to_system
            # arch_specs = read_architecture_template("configs/template.json")
            # A100_system = template_to_system(arch_specs)
            _ = model(Tensor([bs, s, 12288], data_type_dict["fp16"]))
            if args.bandwidth is not None:
                A100_system.device.io_module.bandwidth = args.bandwidth
            if args.fixed_latency is not None:
                A100_system.device.io_module.latency = args.fixed_latency
            bw_tag = (
                f"_bw{int(args.bandwidth / 1e9)}GBs"
                if args.bandwidth is not None
                else ""
            )
            coeff_tag = (
                f"_coeff{args.fixed_io_write_coeff:g}"
                if args.fixed_io_write_coeff != 1.0
                else ""
            )
            latency_tag = (
                f"_lat{args.fixed_latency:g}" if args.fixed_latency is not None else ""
            )
            if args.roofline:
                model.roofline_model(A100_system)
                file_name = (
                    "transformer_A100_HBF_roofline"
                    f"{attn_tag}{kv_tag}{bw_tag}{coeff_tag}{latency_tag}{seq_len_tag}.csv"
                )
            else:
                model.compile_and_simulate(
                    A100_system,
                    compile_mode="heuristic-GPU",
                    include_fixed_io_latency=(not args.exclude_fixed_latency),
                    fixed_io_write_coeff=args.fixed_io_write_coeff,
                )
                file_name = (
                    "transformer_A100_HBF_sim_HBF_excl"
                    f"{attn_tag}{kv_tag}{bw_tag}{coeff_tag}{latency_tag}{seq_len_tag}.csv"
                    if args.exclude_fixed_latency
                    else "transformer_A100_HBF_sim_HBF"
                    f"{attn_tag}{kv_tag}{bw_tag}{coeff_tag}{latency_tag}{seq_len_tag}.csv"
                )
    else:
        print("Auto-regression")
        config = {
            "mode": "autoregressive",
            "d_model": 12288,
            "num_heads": 96,
            "device_count": 4,
            "attention_kind": args.attention_kind,
            "num_kv_heads": args.num_kv_heads,
        }
        model = transformer_block_from_dict(config, data_type=data_type_dict["fp16"])
        output_token_length = 1024
        if args.simgpu:
            A100_system = system_dict["A100_4_fp16"]
            _ = model(
                Tensor([bs, 1, 12288], data_type_dict["fp16"]), s + output_token_length
            )
            if args.roofline:
                model.roofline_model(A100_system)
                file_name = (
                    f"transformerAR_A100_roofline{attn_tag}{kv_tag}{seq_len_tag}.csv"
                )
            else:
                model.compile_and_simulate(
                    A100_system,
                    compile_mode="heuristic-GPU",
                    include_fixed_io_latency=(not args.exclude_fixed_latency),
                )
                file_name = (
                    f"transformerAR_A100_sim_excl{attn_tag}{kv_tag}{seq_len_tag}.csv"
                    if args.exclude_fixed_latency
                    else f"transformerAR_A100_sim{attn_tag}{kv_tag}{seq_len_tag}.csv"
                )
        if args.simgpu_hbf:
            print("Simulating on A100 HBF")
            A100_system = system_dict["A100_4_fp16_HBF"]
            _ = model(
                Tensor([bs, 1, 12288], data_type_dict["fp16"]), s + output_token_length
            )
            if args.bandwidth is not None:
                A100_system.device.io_module.bandwidth = args.bandwidth
            if args.fixed_latency is not None:
                A100_system.device.io_module.latency = args.fixed_latency
            bw_tag = (
                f"_bw{int(args.bandwidth / 1e9)}GBs"
                if args.bandwidth is not None
                else ""
            )
            coeff_tag = (
                f"_coeff{args.fixed_io_write_coeff:g}"
                if args.fixed_io_write_coeff != 1.0
                else ""
            )
            latency_tag = (
                f"_lat{args.fixed_latency:g}" if args.fixed_latency is not None else ""
            )
            if args.roofline:
                model.roofline_model(A100_system)
                file_name = (
                    "transformerAR_A100_HBF_roofline"
                    f"{attn_tag}{kv_tag}{bw_tag}{coeff_tag}{latency_tag}{seq_len_tag}.csv"
                )
            else:
                model.compile_and_simulate(
                    A100_system,
                    compile_mode="heuristic-GPU",
                    include_fixed_io_latency=(not args.exclude_fixed_latency),
                    fixed_io_write_coeff=args.fixed_io_write_coeff,
                )
                file_name = (
                    "transformerAR_A100_HBF_sim_excl"
                    f"{attn_tag}{kv_tag}{bw_tag}{coeff_tag}{latency_tag}{seq_len_tag}.csv"
                    if args.exclude_fixed_latency
                    else "transformerAR_A100_HBF_sim"
                    f"{attn_tag}{kv_tag}{bw_tag}{coeff_tag}{latency_tag}{seq_len_tag}.csv"
                )
    with open(f"ae/figure5/ijkl/{file_name}", "w") as f:
        if args.roofline:
            f.write(model.roofline_log)
        else:
            f.write(model.simulate_log)
