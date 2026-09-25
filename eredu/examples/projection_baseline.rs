//! Native diagnostic baseline; collection/replay is separate from ordinary timings.
use anyhow::{ensure, Context};
use eredu::api::{local_device_plan, LoadedModel, LocalDevice};
use eredu::api::{reset_local_allocator_peak, set_local_allocator_cache_limit};
use eredu_backend_mlx::allocator_memory;
use eredu_backend_mlx::{backend::nn::projection_profile::ProjectionCapture, MlxBackendFactory};
use eredu_core::{
    residency::ParameterConversionRetentionPolicy, ExecutionPlan, GenerationConfigOverrides,
    TextGenerationConfig,
};
use serde_json::json;
use std::{path::PathBuf, time::Instant};

fn main() -> anyhow::Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    ensure!(args.len() == 6, "usage: projection_baseline MODEL OUTPUT_JSON baseline|capture|capture-prototype disabled|BYTES|unlimited POSITIONS SAMPLES");
    let path = PathBuf::from(&args[0]);
    let mode = args[2].as_str();
    ensure!(
        matches!(mode, "baseline" | "capture" | "capture-prototype"),
        "unknown mode"
    );
    let policy = match args[3].as_str() {
        "disabled" => ParameterConversionRetentionPolicy::Disabled,
        "unlimited" => ParameterConversionRetentionPolicy::Unlimited,
        bytes => ParameterConversionRetentionPolicy::Bounded {
            max_bytes: bytes.parse()?,
        },
    };
    let positions: usize = args[4].parse()?;
    let samples: usize = args[5].parse()?;
    ensure!(
        positions > 0 && samples > 0,
        "positions/samples must be positive"
    );
    set_local_allocator_cache_limit(0)?;
    let plan = ExecutionPlan::fully_resident(local_device_plan(LocalDevice::Accelerator(0))?)
        .with_parameter_conversion_retention(Some(policy));
    let factory = MlxBackendFactory::default();
    let (mut model, _) = LoadedModel::load_execution_plan(&factory, &path, &plan)?.into_parts();
    let config = model.resolve_generation_config(GenerationConfigOverrides {
        temperature: Some(0.0),
        max_new_tokens: Some(8),
        ..Default::default()
    })?;
    let mut ordinary = Vec::new();
    let mut capture_cases = Vec::new();
    let repeats = if mode == "baseline" { samples + 1 } else { 1 };
    let mut reference_tokens = None;
    for iteration in 0..repeats {
        model.reset()?;
        model.synchronize()?;
        let before = allocator_memory()?.active_bytes();
        let retained_before = model.parameter_conversion_retention()?;
        reset_local_allocator_peak()?;
        let capture = if mode != "baseline" {
            Some(ProjectionCapture::begin(128)?)
        } else {
            None
        };
        let start = Instant::now();
        let mut generation =
            model.generate_tokens(vec![1; positions], TextGenerationConfig::new(config))?;
        let first = generation.next().context("missing first prediction")??;
        let first_id = first.token_id()?;
        generation.synchronize()?;
        let first_ms = start.elapsed().as_secs_f64() * 1000.0;
        let prefill_peak = allocator_memory()?.peak_bytes();
        if let Some(capture) = capture {
            capture_cases = capture.finish()?;
        }
        let mut tokens = vec![first_id];
        for token in generation {
            tokens.push(token?.token_id()?);
        }
        model.synchronize()?;
        ensure!(tokens.len() == 8, "baseline requires eight predictions");
        let total_ms = start.elapsed().as_secs_f64() * 1000.0;
        if let Some(reference) = &reference_tokens {
            ensure!(reference == &tokens, "reset replay token mismatch");
        } else {
            reference_tokens = Some(tokens.clone());
        }
        ordinary.push(json!({"iteration": iteration, "first_request": iteration == 0,
            "tokens": tokens, "first_token_ms": first_ms, "generation_ms": total_ms,
            "baseline_active_bytes": before, "prefill_active_peak_bytes": prefill_peak,
            "generation_active_peak_bytes": allocator_memory()?.peak_bytes(),
            "retention_before": retained_before, "retention_after": model.parameter_conversion_retention()?,
            "timing_and_peak_valid_for_ordinary_inference": mode == "baseline",
        }));
    }
    let mut classes = Vec::new();
    let mut replays = Vec::new();
    if mode != "baseline" {
        ensure!(
            !capture_cases.is_empty(),
            "no mixed-width projections observed"
        );
        for (index, case) in capture_cases.iter().enumerate() {
            classes.push(case.describe()?);
            let mut rows = vec![case.rows()];
            // Leading real prefill activations exercise the small row counts used
            // by verification. This is a projection replay, not a speculative run.
            if positions == 128 {
                rows.extend([2, 4, 8, 16].into_iter().filter(|&rows| rows < case.rows()));
            }
            rows.sort_unstable();
            rows.dedup();
            for rows in rows {
                let label = format!("case-{index}");
                replays.push(if mode == "capture-prototype" {
                    case.replay_with_mixed_storage(rows, samples, &label)?
                } else {
                    case.replay(rows, samples, &label)?
                });
            }
        }
    }
    let result = json!({"schema_version": 1, "model": path, "mode": mode, "positions": positions,
        "retention_policy": args[3], "allocator_cache_bytes": 0, "samples": samples,
        "ordinary": ordinary, "classes": classes, "replays": replays,
        "notes": ["Capture retains actual array owners and changes graph retention; its timings and peaks are not ordinary inference metrics.",
        "Replay uses settled real arrays in an idle process. Peak growth is MLX active allocation above its per-sample baseline, not process RSS.",
        "First replay samples are separate; capture and reference construction may already warm compiled kernels.",
        "Short-row cases slice actual prefill inputs; they are verification-shaped projection workloads, not end-to-end speculative decoding.",
        "eredu_dispatch inherits conversion retention from collection and preceding replays; cast/native_mixed/preconverted_gemm isolate explicit conversion states.",
        "GPU evaluation timing excludes eager work during graph construction. Total wall timing includes it; per-phase times are not additive end-to-end latency."]});
    std::fs::write(&args[1], serde_json::to_vec_pretty(&result)?)?;
    Ok(())
}
