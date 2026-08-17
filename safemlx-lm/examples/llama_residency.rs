//! Compare Llama residency policies with an opt-in local checkpoint benchmark.

use std::{path::PathBuf, time::Instant};

use clap::Parser;
use safemlx::{Array, Device, DeviceType, ExecutionContext};
use safemlx_lm::{
    architectures::llama::model as llama,
    core::residency::{MemoryTier, OffloadConfig, TransferDirection},
    core::{Backend as _, BackendSession as _},
    load_model_with_options,
    runtime::execution::layerwise::LayerwiseLoadOptions,
    runtime::media::input,
    runtime::residency::dense_stream::DenseDiskStreamLoadOptions,
    MlxBackend, ModelLoadOptions, WeightResidency,
};

#[derive(Debug, Parser)]
#[command(about = "Measure synchronous Llama decoder-layer host transfers")]
struct Args {
    /// Directory containing config.json and Llama-compatible safetensors.
    model_dir: PathBuf,
    /// Comma-separated prompt token ids.
    #[arg(long, default_value = "1,2,3,4")]
    prompt_tokens: String,
    /// Number of greedy decode tokens.
    #[arg(long, default_value_t = 8)]
    decode_tokens: usize,
    /// Maximum decoder layers resident on the execution device.
    #[arg(long, default_value_t = 1)]
    device_layer_window: usize,
    /// Optional device parameter budget in bytes.
    #[arg(long)]
    device_budget_bytes: Option<u64>,
    /// Optional host parameter budget in bytes.
    #[arg(long)]
    host_budget_bytes: Option<u64>,
    /// Maximum simultaneously mapped checkpoint shards.
    #[arg(long, default_value_t = 4)]
    mapped_shards: usize,
    /// Enable experimental dense disk streaming instead of eager host layers.
    #[arg(long)]
    dense_disk_stream: bool,
    /// Finite logical host-layer budget for dense disk streaming.
    #[arg(long, default_value_t = 8 << 30)]
    stream_host_budget: u64,
    /// Finite logical device parameter budget for dense disk streaming.
    #[arg(long, default_value_t = 4 << 30)]
    stream_device_budget: u64,
    /// Protected host-layer lookahead for dense disk streaming.
    #[arg(long, default_value_t = 2)]
    stream_host_lookahead: usize,
    /// Bounded background host-prefetch queue capacity.
    #[arg(long, default_value_t = 2)]
    stream_queue_capacity: usize,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let prompt = args
        .prompt_tokens
        .split(',')
        .map(|value| value.trim().parse::<u32>())
        .collect::<Result<Vec<_>, _>>()?;
    anyhow::ensure!(!prompt.is_empty(), "prompt token list must not be empty");

    let execution = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
    let weights = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = execution.stream();
    anyhow::ensure!(
        !args.dense_disk_stream
            || (args.device_budget_bytes.is_none() && args.host_budget_bytes.is_none()),
        "--device-budget-bytes and --host-budget-bytes apply only to layerwise host residency; use --stream-device-budget and --stream-host-budget with --dense-disk-stream"
    );
    let weight_residency = if args.dense_disk_stream {
        let mut dense = DenseDiskStreamLoadOptions::new(
            args.stream_device_budget,
            args.stream_host_budget,
            args.stream_host_lookahead,
            args.stream_queue_capacity,
        )?;
        dense.max_mapped_shards = args.mapped_shards;
        dense.sample_mlx_memory = true;
        dense.sample_process_memory = true;
        WeightResidency::dense_disk_stream(dense)
    } else {
        let config = OffloadConfig::new(
            args.device_budget_bytes,
            args.host_budget_bytes,
            args.device_layer_window,
        )?;
        let layerwise = LayerwiseLoadOptions {
            offload: config,
            max_mapped_shards: args.mapped_shards,
            sample_mlx_memory: true,
            sample_process_memory: true,
            ..LayerwiseLoadOptions::default()
        };
        WeightResidency::layerwise_host(layerwise)
    };
    let mut model = load_model_with_options(
        &args.model_dir,
        ModelLoadOptions::default().with_weight_residency(weight_residency),
        stream,
        weights.stream(),
    )?;
    anyhow::ensure!(
        model.model_type() == "llama",
        "the Llama residency benchmark requires a Llama-compatible checkpoint, got {}",
        model.model_type()
    );
    let backend = MlxBackend::new(stream);
    let mut session = backend.create_session(&model)?;

    if let Some(report) = model.dense_stream_report()? {
        let offload = report.residency().offload();
        println!("experimental dense disk streaming enabled");
        println!(
            "planned cold layer count/bytes: {}/{}",
            report.planned_layer_count(),
            report.planned_layer_bytes()
        );
        println!(
            "load-time logical host/device layer bytes: {}/{}",
            offload.resident_bytes().get(MemoryTier::Host),
            offload.resident_bytes().get(MemoryTier::Device)
        );
        println!(
            "pinned static device bytes: {}",
            report.pinned_static_device_bytes()
        );
    }

    stream.synchronize()?;
    let prompt_array = Array::from_slice(&prompt, &[1, prompt.len() as i32]);
    let prompt_parts = [input::InputPart::text_token_ids(&prompt_array)];
    let prefill_started = Instant::now();
    let _ = session
        .prefill(
            &backend,
            &mut model,
            input::ModelInput::new(&prompt_parts).into(),
        )?
        .wait()?;
    stream.synchronize()?;
    let prefill = prefill_started.elapsed();
    let time_to_first_token = prefill;

    session.reset(&model)?;
    let repeated_prefill_started = Instant::now();
    let mut logits = session
        .prefill(
            &backend,
            &mut model,
            input::ModelInput::new(&prompt_parts).into(),
        )?
        .wait()?
        .into_logits()
        .expect("replicated model session returns logits");
    stream.synchronize()?;
    let repeated_prefill = repeated_prefill_started.elapsed();

    let decode_started = Instant::now();
    for _ in 0..args.decode_tokens {
        let token = llama::sample(&logits, 0.0, None, stream)?;
        stream.synchronize()?;
        let token = token.reshape(&[1, 1], stream)?;
        logits = session
            .decode(&backend, &mut model, token)?
            .wait()?
            .into_logits()
            .expect("replicated model session returns logits");
        stream.synchronize()?;
    }
    let decode = decode_started.elapsed();
    let report = model
        .residency_report()?
        .expect("layerwise residency was selected");
    let offload = report.offload();
    let observed_window = report
        .units()
        .iter()
        .filter(|unit| unit.id().as_str().starts_with("llama.layer.") && unit.device_resident())
        .count();

    println!("model type: {}", model.model_type());
    println!(
        "configured/observed device-layer window: {}/{}",
        args.device_layer_window, observed_window
    );
    println!(
        "current logical host/device parameter bytes: {}/{}",
        offload.resident_bytes().get(MemoryTier::Host),
        offload.resident_bytes().get(MemoryTier::Device)
    );
    println!(
        "peak logical host/device parameter bytes: {}/{}",
        offload.peak_resident_bytes().get(MemoryTier::Host),
        offload.peak_resident_bytes().get(MemoryTier::Device)
    );
    for direction in TransferDirection::ALL {
        let transfer = offload.transfer(direction);
        if transfer.count() > 0 {
            println!(
                "synchronous {direction:?}: {} transfers, {} bytes, {:?}",
                transfer.count(),
                transfer.bytes(),
                transfer.duration()
            );
        }
    }
    println!("prefill latency: {prefill:?}");
    println!("first-process prefill latency: {prefill:?}");
    println!("repeated-process prefill latency after cache reset: {repeated_prefill:?}");
    println!(
        "prefill throughput: {:.2} tokens/s",
        prompt.len() as f64 / prefill.as_secs_f64()
    );
    println!("decode latency: {decode:?}");
    if args.decode_tokens > 0 {
        println!(
            "decode throughput: {:.2} tokens/s",
            args.decode_tokens as f64 / decode.as_secs_f64()
        );
    }
    println!("time to first token: {time_to_first_token:?}");
    println!("Backend allocator memory: {:?}", offload.allocator_memory());
    println!("weight-store diagnostics: {:?}", report.weight_store());
    if let Some(dense) = model.dense_stream_report()? {
        println!("background host-prefetch: {:?}", dense.background());
        println!(
            "streamed host layers current/peak: {}/{} ({} peak bytes)",
            dense.host_layers().current_layer_count(),
            dense.host_layers().peak_layer_count(),
            dense.host_layers().peak_layer_bytes()
        );
        println!(
            "streamed device layers current/peak: {}/{} ({} peak bytes)",
            dense.device_layers().current_layer_count(),
            dense.device_layers().peak_layer_count(),
            dense.device_layers().peak_layer_bytes()
        );
        println!("host cache: {:?}", dense.host_layers().cache());
        println!("device cache: {:?}", dense.device_layers().cache());
        println!("execution groups: {:?}", dense.execution_groups());
        println!("prefill dense activity: {:?}", dense.prefill());
        println!("decode dense activity: {:?}", dense.decode());
        println!("exact physical disk I/O is not inferred from mmap or page-fault observations");
    }
    println!("KV cache and activations are not included in logical parameter residency totals.");
    #[cfg(target_os = "macos")]
    println!(
        "Metal uses unified physical memory; this run validates scheduling and logical residency, not expanded model capacity."
    );
    Ok(())
}
