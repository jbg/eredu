//! Teacher-forced released scores through cold selection and ordinary public sessions.
//! Inputs are token IDs, without tokenizer, host-media or assistant-session claims.
use clap::Parser;
use eredu_backend_mlx::{
    backend::runtime::media::input::{token_ids_part, ModelInput},
    native::MlxModelInput,
    MlxBackendFactory,
};
use eredu_core::{
    DevicePlan, ExecutionPlan, ObservationValue, ResidencyPlan, TensorObservationData,
};
use safemlx::Array;
use serde_json::{json, Value};
use std::{error::Error, path::PathBuf, time::Instant};

use eredu_evaluation::checkpoint_reference::{sha256, validator_sha256, verify_payload};

#[derive(Parser)]
struct Args {
    checkpoint: PathBuf,
    reference: PathBuf,
    output: PathBuf,
    /// Frozen publisher revisions, complete payload names and file hashes.
    #[arg(long)]
    inventory: PathBuf,
    #[arg(long)]
    atol: f64,
    #[arg(long)]
    rtol: f64,
    #[arg(long, default_value = "cpu:0")]
    device: String,
    #[arg(long, default_value_t = 64)]
    device_budget_gib: u64,
    #[arg(long, default_value_t = 64)]
    host_budget_gib: u64,
    #[arg(long, default_value_t = 12)]
    readers: usize,
    /// Validate provenance and cold selection without realizing native resources.
    #[arg(long)]
    select_only: bool,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    if !args.atol.is_finite() || args.atol < 0. || !args.rtol.is_finite() || args.rtol < 0. {
        return Err("tolerances must be finite and nonnegative".into());
    }
    let reference_bytes = std::fs::read(args.reference.join("reference-evidence.json"))?;
    let reference: Value = serde_json::from_slice(&reference_bytes)?;
    if !reference["media"].is_null() {
        return Err("this runner accepts text-token reference inputs only".into());
    }
    let inventory: Value = serde_json::from_slice(&std::fs::read(&args.inventory)?)?;
    let repository = reference["repository"]
        .as_str()
        .ok_or("missing repository")?;
    let revision = reference["revision"].as_str().ok_or("missing revision")?;
    let proof_hash = verify_payload(
        &args.checkpoint,
        &reference,
        &inventory,
        repository,
        revision,
    )?;
    let steps = reference["steps"]
        .as_array()
        .ok_or("missing reference steps")?;
    if steps.len() < 4 {
        return Err("reference requires prefill and at least three cached decode steps".into());
    }
    let plan = ExecutionPlan::fully_resident(DevicePlan::new("mlx", args.device.clone())?)
        .with_residency(ResidencyPlan::DenseDiskStream {
            device_budget_bytes: args
                .device_budget_gib
                .checked_mul(1 << 30)
                .ok_or("device budget overflow")?,
            host_budget_bytes: args
                .host_budget_gib
                .checked_mul(1 << 30)
                .ok_or("host budget overflow")?,
            host_lookahead: 1,
            background_queue: 1,
        })
        .with_max_cached_shards(args.readers);
    let started = Instant::now();
    println!("Inspecting and selecting the complete checkpoint through the execution-plan API");
    let inspection = eredu_architectures::configuration::inspect_artifact(&args.checkpoint)?;
    let factory = MlxBackendFactory::default();
    let selected = eredu_core::select_execution_plan_target(&factory, &plan, inspection)?;
    let mut report = json!({
        "repository": repository, "revision": revision, "plan": plan,
        "payload_proof_sha256": proof_hash,
        "reference_evidence_sha256": sha256(&reference_bytes),
        "native_runner_sha256": sha256(include_bytes!("execution_plan_reference.rs")),
        "payload_validator_sha256": validator_sha256(),
        "selection_verified": true, "execution_verified": false,
        "comparison_scope": "public execution-plan target and ordinary text prefill/cached decode; final vocabulary row of each invocation",
        "atol": args.atol, "rtol": args.rtol,
        "native_build": {"debug_assertions": cfg!(debug_assertions), "accelerate": cfg!(feature="accelerate"), "metal": cfg!(feature="metal")},
        "selection_seconds": started.elapsed().as_secs_f64(), "steps": []
    });
    std::fs::create_dir_all(&args.output)?;
    let evidence = args.output.join("comparison-evidence.json");
    std::fs::write(&evidence, serde_json::to_vec_pretty(&report)?)?;
    if args.select_only {
        println!("Cold selection passed; no native execution requested");
        return Ok(());
    }
    println!("Realizing the retained target with bounded weight residency");
    let mut runtime =
        eredu_core::realize_execution_plan_target(&factory, &plan, selected)?.into_runtime()?;
    report["realization_seconds"] =
        json!(started.elapsed().as_secs_f64() - report["selection_seconds"].as_f64().unwrap());
    let mut frontier = 0_u64;
    let mut passed = true;
    for (step, record) in steps.iter().enumerate() {
        let ids = record["input_ids"]
            .as_array()
            .ok_or("missing input IDs")?
            .iter()
            .map(|v| u32::try_from(v.as_u64().ok_or("invalid token")?).map_err(Into::into))
            .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
        if ids.is_empty() || record["cache_before"] != frontier {
            return Err("reference input or cached frontier differs".into());
        }
        let shape = record["shape"]
            .as_array()
            .ok_or("missing reference shape")?;
        if shape.len() != 3 || shape[0] != 1 || shape[1] != ids.len() {
            return Err("expected batch-one sequence-major vocabulary scores".into());
        }
        let vocabulary = usize::try_from(
            shape[2]
                .as_u64()
                .filter(|v| *v > 0)
                .ok_or("invalid vocabulary")?,
        )?;
        let name = record["file"]
            .as_str()
            .ok_or("missing reference score file")?;
        if std::path::Path::new(name)
            .file_name()
            .and_then(|p| p.to_str())
            != Some(name)
        {
            return Err("reference scores must be a direct child".into());
        }
        let bytes = std::fs::read(args.reference.join(name))?;
        let row_bytes = vocabulary.checked_mul(4).ok_or("score extent overflow")?;
        if record["sha256"] != sha256(&bytes)
            || bytes.len()
                != ids
                    .len()
                    .checked_mul(row_bytes)
                    .ok_or("score extent overflow")?
        {
            return Err("reference score digest or shape differs".into());
        }
        let expected = &bytes[bytes.len() - row_bytes..];
        let tokens = Array::from_slice(&ids, &[1, i32::try_from(ids.len())?]);
        println!(
            "Starting {} step {step} with {} input positions",
            if step == 0 {
                "prefill"
            } else {
                "cached decode"
            },
            ids.len()
        );
        let step_started = Instant::now();
        let output = if step == 0 {
            let parts = [token_ids_part(&tokens)?];
            let input = MlxModelInput::from(ModelInput::new(&parts))
                .with_semantic_content_fingerprint(
                    eredu_core::cache::prompt_cache_token_fingerprint(&ids),
                )?;
            runtime.prefill(input)?.wait()?
        } else {
            runtime.decode(tokens)?.wait()?
        };
        let observations = runtime.observe_output(&output)?;
        let Some(ObservationValue::Tensor(logits)) =
            observations.get(eredu_core::MODEL_LOGITS_OBSERVATION_PATH)
        else {
            return Err("target owns no output logits".into());
        };
        let actual_shape = logits.shape();
        if actual_shape != [1, vocabulary]
            && actual_shape != [1, 1, vocabulary]
            && actual_shape != [1, ids.len(), vocabulary]
        {
            return Err(format!("unexpected public output geometry {actual_shape:?}").into());
        }
        let TensorObservationData::F32(values) = logits.data() else {
            return Err("logit observation is not floating point".into());
        };
        let actual = &values[values.len() - vocabulary..];
        let mut maximum = 0_f64;
        let mut squared = 0_f64;
        let mut outside = 0_usize;
        for (&a, b) in actual.iter().zip(expected.chunks_exact(4)) {
            let b = f32::from_le_bytes(b.try_into()?);
            if !a.is_finite() || !b.is_finite() {
                return Err("nonfinite reference or native scores".into());
            }
            let delta = (f64::from(a) - f64::from(b)).abs();
            maximum = maximum.max(delta);
            squared += delta * delta;
            outside += usize::from(delta > args.atol + args.rtol * f64::from(b).abs());
        }
        frontier = frontier
            .checked_add(ids.len() as u64)
            .ok_or("frontier overflow")?;
        if record["cache_after"] != frontier {
            return Err("reference cache progression differs".into());
        }
        let cache = runtime
            .session()
            .cache_residency_report()?
            .map(|r| r.logical_cached_tokens);
        if cache.is_some_and(|observed| observed != frontier) {
            return Err("observed cache frontier differs".into());
        }
        let actual_bytes: Vec<u8> = actual.iter().flat_map(|v| v.to_le_bytes()).collect();
        let filename = format!("step-{step}-last-row.f32");
        std::fs::write(args.output.join(&filename), &actual_bytes)?;
        passed &= outside == 0;
        report["steps"].as_array_mut().unwrap().push(json!({
            "input_ids": ids, "native_shape": actual_shape, "compared_reference_row": ids.len() - 1,
            "reference_cache_after": frontier, "observed_cache_tokens": cache,
            "max_abs": maximum, "rms": (squared / vocabulary as f64).sqrt(),
            "outside_tolerance": outside, "file": filename, "sha256": sha256(&actual_bytes),
            "reference_last_row_sha256": sha256(expected), "elapsed_seconds": step_started.elapsed().as_secs_f64()
        }));
        report["elapsed_seconds"] = json!(started.elapsed().as_secs_f64());
        std::fs::write(&evidence, serde_json::to_vec_pretty(&report)?)?;
        println!("Completed step {step}: max_abs={maximum:.8}, outside_tolerance={outside}");
    }
    report["execution_verified"] = json!(passed);
    std::fs::write(&evidence, serde_json::to_vec_pretty(&report)?)?;
    if !passed {
        return Err(
            "public execution-plan comparison exceeds declared tolerances; evidence retained"
                .into(),
        );
    }
    Ok(())
}
