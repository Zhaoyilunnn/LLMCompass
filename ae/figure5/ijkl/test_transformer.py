from software_model.transformer import (
    TransformerBlockInitComputationTP,
    TransformerBlockAutoRegressionTP,
)
from software_model.utils import data_type_dict, Tensor
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
    args = parser.parse_args()

    bs = 8
    s = 2048
    if args.init:
        print("Initial computation")
        if args.simgpu:
            model = TransformerBlockInitComputationTP(
                d_model=12288,
                n_heads=96,
                device_count=4,
                data_type=data_type_dict["fp16"],
            )
            A100_system = system_dict["A100_4_fp16"]
            # from design_space_exploration.dse import read_architecture_template, template_to_system
            # arch_specs = read_architecture_template("configs/template.json")
            # A100_system = template_to_system(arch_specs)
            _ = model(Tensor([bs, s, 12288], data_type_dict["fp16"]))
            if args.roofline:
                model.roofline_model(A100_system)
                file_name = "transformer_A100_roofline.csv"
            else:
                model.compile_and_simulate(
                    A100_system,
                    compile_mode="heuristic-GPU",
                    include_fixed_io_latency=(not args.exclude_fixed_latency),
                )
                file_name = (
                    "transformer_A100_sim_excl.csv"
                    if args.exclude_fixed_latency
                    else "transformer_A100_sim.csv"
                )

        if args.simgpu_hbf:
            model = TransformerBlockInitComputationTP(
                d_model=12288,
                n_heads=96,
                device_count=4,
                data_type=data_type_dict["fp16"],
            )
            A100_system = system_dict["A100_4_fp16_HBF"]
            # from design_space_exploration.dse import read_architecture_template, template_to_system
            # arch_specs = read_architecture_template("configs/template.json")
            # A100_system = template_to_system(arch_specs)
            _ = model(Tensor([bs, s, 12288], data_type_dict["fp16"]))
            if args.bandwidth is not None:
                A100_system.device.io_module.bandwidth = args.bandwidth
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
            if args.roofline:
                model.roofline_model(A100_system)
                file_name = f"transformer_A100_HBF_roofline{bw_tag}{coeff_tag}.csv"
            else:
                model.compile_and_simulate(
                    A100_system,
                    compile_mode="heuristic-GPU",
                    include_fixed_io_latency=(not args.exclude_fixed_latency),
                    fixed_io_write_coeff=args.fixed_io_write_coeff,
                )
                file_name = (
                    f"transformer_A100_HBF_sim_HBF_excl{bw_tag}{coeff_tag}.csv"
                    if args.exclude_fixed_latency
                    else f"transformer_A100_HBF_sim_HBF{bw_tag}{coeff_tag}.csv"
                )
        if args.simtpu:
            model = TransformerBlockInitComputationTP(
                d_model=12288,
                n_heads=96,
                device_count=8,
                data_type=data_type_dict["fp16"],
            )
            TPU_system = system_dict["TPUv3_8"]
            _ = model(Tensor([bs, s, 12288], data_type_dict["fp16"]))
            if args.roofline:
                model.roofline_model(TPU_system)
                file_name = "transformer_TPUv3_roofline.csv"
            else:
                model.compile_and_simulate(TPU_system, compile_mode="heuristic-TPU")
                file_name = "transformer_TPUv3_sim.csv"
        if args.gpu:
            model = TransformerBlockInitComputationTP(
                d_model=12288,
                n_heads=96,
                device_count=4,
                data_type=data_type_dict["fp16"],
            )
            _ = model(Tensor([bs, s, 12288], data_type_dict["fp16"]))
            model.run_on_gpu()
    else:
        print("Auto-regression")
        output_token_length = 1024
        if args.simgpu:
            model = TransformerBlockAutoRegressionTP(
                d_model=12288,
                n_heads=96,
                device_count=4,
                data_type=data_type_dict["fp16"],
            )
            A100_system = system_dict["A100_4_fp16"]
            _ = model(
                Tensor([bs, 1, 12288], data_type_dict["fp16"]), s + output_token_length
            )
            if args.roofline:
                model.roofline_model(A100_system)
                file_name = "transformerAR_A100_roofline.csv"
            else:
                model.compile_and_simulate(
                    A100_system,
                    compile_mode="heuristic-GPU",
                    include_fixed_io_latency=(not args.exclude_fixed_latency),
                )
                file_name = (
                    "transformerAR_A100_sim_excl.csv"
                    if args.exclude_fixed_latency
                    else "transformerAR_A100_sim.csv"
                )
        if args.simgpu_hbf:
            print("Simulating on A100 HBF")
            model = TransformerBlockAutoRegressionTP(
                d_model=12288,
                n_heads=96,
                device_count=4,
                data_type=data_type_dict["fp16"],
            )
            A100_system = system_dict["A100_4_fp16_HBF"]
            _ = model(
                Tensor([bs, 1, 12288], data_type_dict["fp16"]), s + output_token_length
            )
            if args.bandwidth is not None:
                A100_system.device.io_module.bandwidth = args.bandwidth
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
            if args.roofline:
                model.roofline_model(A100_system)
                file_name = f"transformerAR_A100_HBF_roofline{bw_tag}{coeff_tag}.csv"
            else:
                model.compile_and_simulate(
                    A100_system,
                    compile_mode="heuristic-GPU",
                    include_fixed_io_latency=(not args.exclude_fixed_latency),
                    fixed_io_write_coeff=args.fixed_io_write_coeff,
                )
                file_name = (
                    f"transformerAR_A100_HBF_sim_excl{bw_tag}{coeff_tag}.csv"
                    if args.exclude_fixed_latency
                    else f"transformerAR_A100_HBF_sim{bw_tag}{coeff_tag}.csv"
                )
        if args.simtpu:
            model = TransformerBlockAutoRegressionTP(
                d_model=12288,
                n_heads=96,
                device_count=8,
                data_type=data_type_dict["fp16"],
            )
            TPU_system = system_dict["TPUv3_8"]
            _ = model(
                Tensor([bs, 1, 12288], data_type_dict["fp16"]), s + output_token_length
            )
            if args.roofline:
                model.roofline_model(TPU_system)
                file_name = "transformerAR_TPUv3_roofline.csv"
            else:
                model.compile_and_simulate(TPU_system, compile_mode="heuristic-TPU")
                file_name = "transformerAR_TPUv3_sim.csv"
        if args.gpu:
            model = TransformerBlockAutoRegressionTP(
                d_model=12288,
                n_heads=96,
                device_count=4,
                data_type=data_type_dict["fp16"],
            )
            _ = model(
                Tensor([bs, 1, 12288], data_type_dict["fp16"]), s + output_token_length
            )
            model.run_on_gpu()
    with open(f"ae/figure5/ijkl/{file_name}", "w") as f:
        if args.roofline:
            f.write(model.roofline_log)
        else:
            f.write(model.simluate_log)
