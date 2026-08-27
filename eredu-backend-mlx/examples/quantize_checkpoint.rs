//! Execute an architecture-authored checkpoint quantization plan with MLX.

use std::{path::PathBuf, str::FromStr};

use clap::{Parser, ValueEnum};
use eredu_architectures::checkpoint_conversion::{
    SafetensorsQuantizationPlan, SafetensorsQuantizationTarget,
};
use eredu_backend_mlx::native::{
    quantize_checkpoint, CheckpointQuantizationOptions, Device, DeviceType, ExecutionContext,
};
use eredu_checkpoint::{AffineQuantization, WeightQuantization};

#[derive(Debug, Clone, Copy, ValueEnum)]
enum Mode {
    Affine,
    Mxfp4,
}

#[derive(Debug, Parser)]
#[command(about = "Quantize a safetensors model directory using MLX affine or MXFP4 packing")]
struct Args {
    /// Source Hugging Face/MLX model directory.
    source: PathBuf,
    /// New output directory. It must not already exist.
    output: PathBuf,
    /// Architecture-authored output config.json.
    #[arg(long)]
    output_config: PathBuf,
    /// Exact SOURCE,WEIGHT,SCALES[,BIASES] target (repeatable).
    #[arg(long, value_parser = parse_target)]
    target: Vec<SafetensorsQuantizationTarget>,
    /// Quantized weight encoding.
    #[arg(long, value_enum, default_value_t = Mode::Affine)]
    mode: Mode,
    /// Number of values sharing each scale (and affine bias).
    #[arg(long)]
    group_size: Option<i32>,
    /// Packed bits per weight.
    #[arg(long)]
    bits: Option<i32>,
    /// Approximate maximum output shard size in MiB.
    #[arg(long, default_value_t = 512)]
    shard_size_mib: usize,
}

fn parse_target(value: &str) -> Result<SafetensorsQuantizationTarget, String> {
    let fields = value.split(',').collect::<Vec<_>>();
    match fields.as_slice() {
        [source, weight, scales] => Ok(SafetensorsQuantizationTarget::new(
            *source,
            *weight,
            *scales,
            None::<String>,
        )),
        [source, weight, scales, biases] => Ok(SafetensorsQuantizationTarget::new(
            *source,
            *weight,
            *scales,
            Some(*biases),
        )),
        _ => Err("expected SOURCE,WEIGHT,SCALES[,BIASES]".into()),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let context = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
    let stream = context.stream();
    let quantization = match args.mode {
        Mode::Affine => WeightQuantization::Affine(AffineQuantization::new(
            args.group_size.unwrap_or(64),
            args.bits.unwrap_or(4),
        )?),
        Mode::Mxfp4 => {
            let group_size = args.group_size.unwrap_or(32);
            let bits = args.bits.unwrap_or(4);
            if group_size != 32 || bits != 4 {
                return Err(format!(
                    "MXFP4 requires --group-size 32 and --bits 4, got {group_size}/{bits}"
                )
                .into());
            }
            WeightQuantization::MxFp4
        }
    };
    let output_config = serde_json::Value::from_str(&std::fs::read_to_string(args.output_config)?)?;
    let plan = SafetensorsQuantizationPlan::new(quantization, args.target, output_config)?;
    let mut options = CheckpointQuantizationOptions::new(plan);
    options.shard_size_bytes = args.shard_size_mib * 1024 * 1024;
    let report = quantize_checkpoint(args.source, args.output, &options, stream)?;
    println!("quantized_tensors={}", report.quantized_tensors);
    println!("copied_tensors={}", report.copied_tensors);
    println!("shards={}", report.shards);
    println!("total_size_bytes={}", report.total_size);
    Ok(())
}
