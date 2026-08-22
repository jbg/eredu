//! Save, drop, lazily reopen, and continue a deterministic text prompt cache.

use std::path::PathBuf;

use clap::Parser;
use eredu::{
    api::LoadedModel,
    backend::mlx::runtime::media::input::{InputPart, ModelInput},
    AttentionPolicy, CacheResidencyPolicy, PagedCacheOptions, PromptCacheDescriptor,
    PromptCacheOptions, PromptCacheTopology,
};
use eredu_backend_mlx::native::{
    transforms::async_eval_with_event, Array, Device, DeviceType, ExecutionContext,
};

#[derive(Debug, Parser)]
#[command(about = "Verify reusable paged prompt-cache parity")]
struct Args {
    /// Directory containing a supported text model.
    model_dir: PathBuf,
    /// Persistent prompt-cache destination.
    cache_dir: PathBuf,
    /// Deterministic text prefix.
    #[arg(long)]
    prompt: String,
    /// One token appended after the restored prefix.
    #[arg(long)]
    suffix_token: u32,
    /// Stable content-based checkpoint identity.
    #[arg(long)]
    checkpoint_fingerprint: String,
    /// Token count per immutable cache block.
    #[arg(long, default_value_t = 128)]
    block_tokens: i32,
    /// Finite logical execution-device cache bytes.
    #[arg(long, default_value_t = 512 << 20)]
    device_cache_bytes: u64,
    /// Finite logical host cache bytes.
    #[arg(long, default_value_t = 2 << 30)]
    host_cache_bytes: u64,
    /// Recent device blocks protected per layer.
    #[arg(long, default_value_t = 1)]
    recent_device_blocks: usize,
    /// Optional explicit live-cache backing directory.
    #[arg(long)]
    live_disk_dir: Option<PathBuf>,
    /// Finite logical live-cache disk bytes.
    #[arg(long, default_value_t = 8 << 30)]
    live_disk_bytes: u64,
    /// Replace an existing persistent cache directory atomically.
    #[arg(long)]
    replace: bool,
    /// Use the ordinary device-resident cache and skip persistence.
    #[arg(long)]
    device_cache: bool,
}

fn prefill_tokens(
    tokens: &Array,
    model: &mut LoadedModel<eredu::backend::mlx::MlxBackend<'static>>,
) -> anyhow::Result<Array> {
    let parts = [InputPart::text_token_ids(tokens)];
    let input = eredu::composition::mlx::MlxModelInput::from(ModelInput::new(&parts));
    model
        .runtime_mut()
        .prefill(input)?
        .wait()?
        .into_logits()
        .ok_or_else(|| anyhow::anyhow!("selected MLX rank does not own prefill logits"))
}

fn decode_tokens(
    tokens: &Array,
    model: &mut LoadedModel<eredu::backend::mlx::MlxBackend<'static>>,
) -> anyhow::Result<Array> {
    model
        .runtime_mut()
        .decode(tokens.clone())?
        .wait()?
        .into_logits()
        .ok_or_else(|| anyhow::anyhow!("selected MLX rank does not own decode logits"))
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let execution = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
    let weights = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = execution.stream();
    let mut model = LoadedModel::load(
        eredu::backend::mlx::MlxBackend::new(stream, weights.stream()),
        &args.model_dir,
        Default::default(),
    )?;
    let prefix_ids = model.encode(&args.prompt, false)?;
    anyhow::ensure!(
        !prefix_ids.is_empty(),
        "prompt must encode to at least one token"
    );
    let prefix = Array::from_slice(&prefix_ids, &[1, prefix_ids.len() as i32]);
    let suffix = Array::from_slice(&[args.suffix_token], &[1, 1]);

    if args.device_cache {
        let _ = prefill_tokens(&prefix, &mut model)?;
        let logits = decode_tokens(&suffix, &mut model)?;
        async_eval_with_event([&logits])?.synchronize()?;
        println!(
            "ordinary device cache suffix logits shape: {:?}",
            logits.shape()
        );
        return Ok(());
    }

    let layer_layout = model.runtime().session().prompt_cache_layer_layout()?;
    let has_full_attention = layer_layout
        .iter()
        .any(|policy| matches!(policy.attention(), Some(AttentionPolicy::Full)));
    let mut paged = PagedCacheOptions::new(
        args.block_tokens,
        args.device_cache_bytes,
        args.host_cache_bytes,
        args.recent_device_blocks,
    )?
    .with_full_attention(has_full_attention)
    .with_persistence_retention(true)
    .with_process_sampling(true);
    if let Some(directory) = &args.live_disk_dir {
        paged = paged.with_live_disk(directory, args.live_disk_bytes, 2)?;
    }

    model
        .runtime_mut()
        .session_mut()
        .configure_cache(CacheResidencyPolicy::Paged(paged.clone()))?;
    let _ = prefill_tokens(&prefix, &mut model)?;
    let uninterrupted_logits = decode_tokens(&suffix, &mut model)?;
    async_eval_with_event([&uninterrupted_logits])?.synchronize()?;
    println!(
        "uninterrupted report: {:#?}",
        model.runtime().session().cache_residency_report()?
    );

    model
        .runtime_mut()
        .session_mut()
        .configure_cache(CacheResidencyPolicy::Paged(paged.clone()))?;
    let _ = prefill_tokens(&prefix, &mut model)?;
    let model_type = model.model_type().to_owned();
    let model_family = if model_type.contains("deepseek") {
        "deepseek_v3"
    } else if model_type.contains("qwen") {
        "qwen"
    } else if model_type.contains("gpt_oss") {
        "gpt_oss"
    } else {
        "llama"
    };
    let layer_count = layer_layout.len();
    let descriptor = PromptCacheDescriptor {
        model_family: model_family.into(),
        effective_model_type: model_type,
        checkpoint_fingerprint: args.checkpoint_fingerprint,
        prefix_content_fingerprint: format!("tokens:{prefix_ids:?}"),
        architecture_fingerprint: model
            .runtime()
            .session()
            .prompt_cache_architecture_fingerprint()?,
        layer_count,
        global_layer_start: 0,
        global_layer_end: layer_count,
        batch_size: 1,
        layer_prefix_offsets: model
            .runtime()
            .session()
            .prompt_cache_layer_prefix_offsets()?,
        layer_layout,
        sink_tokens: 0,
        topology: PromptCacheTopology::default(),
    };
    let manifest = {
        let (backend, session) = model.runtime_mut().parts_mut();
        session.save_prompt_cache(
            backend,
            &args.cache_dir,
            descriptor.clone(),
            &prefix_ids,
            &PromptCacheOptions {
                application_namespace: Some("paged-prompt-cache-example".into()),
                replace_existing: args.replace,
            },
        )?
    };
    println!("saved blocks: {}", manifest.blocks.len());
    println!(
        "save report: {:#?}",
        model.runtime().session().cache_residency_report()?
    );

    let inspected = {
        let (backend, session) = model.runtime_mut().parts_mut();
        session.load_prompt_cache(backend, &args.cache_dir, &descriptor, &prefix_ids, paged)?
    };
    println!("cataloged blocks: {}", inspected.blocks.len());
    println!(
        "load report: {:#?}",
        model.runtime().session().cache_residency_report()?
    );
    let restored_logits = decode_tokens(&suffix, &mut model)?;
    async_eval_with_event([&restored_logits])?.synchronize()?;
    let equal = restored_logits
        .all_close(&uninterrupted_logits, 1e-4, 1e-4, None, stream)?
        .item::<bool>(stream);
    println!("restored suffix logits match uninterrupted execution: {equal}");
    println!(
        "continued report: {:#?}",
        model.runtime().session().cache_residency_report()?
    );
    anyhow::ensure!(
        equal,
        "restored suffix logits differ from uninterrupted execution"
    );
    Ok(())
}
