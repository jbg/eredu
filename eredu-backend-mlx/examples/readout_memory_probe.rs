//! Compares ordinary last-row readout with a sequence observer in a fresh process.
//! Use `/usr/bin/time -l` for the independent process RSS high-water mark.
//! This probe does not claim working-memory enforcement.
use anyhow::{bail, ensure, Context};
use eredu_backend_mlx::{backend::MlxBackend, MlxBackendFactory};
use eredu_core::{
    DevicePlan, ExecutionPlan, ObservationRequest, ObservationSelector, ObservationValue,
    TensorObservationData, TextGenerationBackend, MODEL_LOGITS_OBSERVATION_PATH,
};
use serde::Deserialize;
use serde_json::json;
use std::{path::PathBuf, time::Instant};

#[derive(Deserialize)]
struct Input {
    prompt_ids: Vec<u32>,
    decode_ids: Vec<u32>,
}

fn process_rss_bytes() -> anyhow::Result<u64> {
    let output = std::process::Command::new("/bin/ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()?;
    ensure!(output.status.success(), "process RSS observation failed");
    Ok(std::str::from_utf8(&output.stdout)?.trim().parse::<u64>()? * 1024)
}

fn main() -> anyhow::Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    ensure!(
        args.len() == 4 || args.len() == 5,
        "usage: readout_memory_probe CHECKPOINT INPUT.json last|sequence OUTPUT.json [TEXT_CHUNK_POSITIONS]"
    );
    let input: Input = serde_json::from_slice(&std::fs::read(&args[1])?)?;
    ensure!(!input.prompt_ids.is_empty(), "prompt must not be empty");
    let chunk_positions = args
        .get(4)
        .map(|value| value.parse::<std::num::NonZeroU64>())
        .transpose()?;
    let sequence = match args[2].as_str() {
        "last" => false,
        "sequence" => true,
        _ => bail!("readout mode must be last or sequence"),
    };
    let path = PathBuf::from(&args[0]);
    let plan = ExecutionPlan::fully_resident(DevicePlan::new("mlx", "metal:0")?);
    let factory = MlxBackendFactory::default();
    let inspection = eredu_architectures::configuration::inspect_artifact(&path)?;
    let selected = eredu_core::select_execution_plan_target(&factory, &plan, inspection)?;
    let mut runtime =
        eredu_core::realize_execution_plan_target(&factory, &plan, selected)?.into_runtime()?;
    runtime.backend().synchronize()?;
    // Preserve full-sequence demand without retaining hidden observations or
    // copying full scores to the host. Both modes expose the same final row.
    let sequence_observer = ObservationRequest::selected([ObservationSelector::Exact(
        "__readout_probe_no_capture__".into(),
    )]);
    let baseline = eredu_backend_mlx::allocator_memory()?;
    let baseline_rss = process_rss_bytes()?;
    eredu_backend_mlx::reset_allocator_peak()?;
    let mut passes = Vec::new();
    for step in 0..=input.decode_ids.len() {
        let started = Instant::now();
        let output = if step == 0 {
            let mut prompt =
                MlxBackend::prepare_text_prompt(runtime.backend(), input.prompt_ids.clone())?;
            if let Some(positions) = chunk_positions {
                prompt = prompt.with_prefill_chunk_positions(positions);
            }
            if sequence {
                let result = runtime.inspect_prefill(prompt, &sequence_observer)?;
                ensure!(
                    result.observations.iter().next().is_none(),
                    "unexpected captured values"
                );
                result.output
            } else {
                runtime.prefill(prompt)?.wait()?
            }
        } else {
            let token = safemlx::Array::from_slice(&[input.decode_ids[step - 1]], &[1, 1]);
            runtime.decode(token)?.wait()?
        };
        runtime.backend().synchronize()?;
        let elapsed = started.elapsed().as_secs_f64();
        let memory = eredu_backend_mlx::allocator_memory()?;
        let rss = process_rss_bytes()?;
        let native_dtype = format!(
            "{:?}",
            output
                .logits()
                .context("missing native scores")?
                .as_array()
                .dtype()
        );
        let observed = runtime.observe_output(&output)?;
        let ObservationValue::Tensor(scores) = observed
            .get(MODEL_LOGITS_OBSERVATION_PATH)
            .context("missing score observation")?
        else {
            bail!("scores were not a tensor")
        };
        let TensorObservationData::F32(values) = scores.data() else {
            bail!("scores were not F32")
        };
        ensure!(
            scores.shape().len() == 2 && scores.shape()[0] == 1,
            "expected one vocabulary row"
        );
        ensure!(values.iter().all(|v| v.is_finite()), "non-finite logits");
        let argmax = values
            .iter()
            .enumerate()
            .max_by(|(ia, a), (ib, b)| a.total_cmp(b).then_with(|| ib.cmp(ia)))
            .unwrap()
            .0;
        passes.push(json!({"step":step,"shape":scores.shape(),"native_dtype":native_dtype,"scores":values,"argmax":argmax,
            "seconds":elapsed,"mlx_active_bytes":memory.active_bytes(),"mlx_peak_bytes":memory.peak_bytes(),
            "mlx_cache_bytes":memory.cached_bytes(),"process_rss_bytes":rss}));
    }
    let report = json!({"checkpoint":path,"readout":args[2],"prompt_ids":input.prompt_ids,
        "decode_ids":input.decode_ids,"requested_text_chunk_positions":chunk_positions.map(std::num::NonZeroU64::get),
        "baseline_mlx_active_bytes":baseline.active_bytes(),
        "baseline_process_rss_bytes":baseline_rss,"passes":passes});
    std::fs::write(&args[3], serde_json::to_vec(&report)?)?;
    eprintln!("wrote {} passes to {}", input.decode_ids.len() + 1, args[3]);
    Ok(())
}
