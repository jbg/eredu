//! Bounded released NVFP4 text and multimodal comparison using the ordinary layered runtime.
//!
//! Run `validation/muse_spark_reference_run.py` first. This deliberately bypasses
//! application family registration: numerical comparison does not establish
//! facade, controlled-session, or assistant support.
use std::{cell::RefCell, error::Error, path::PathBuf, sync::Arc, time::Instant};

use clap::Parser;
use eredu_architectures::muse::{self, DecoderConfig, LayeredModel, MuseIdentity};
use eredu_backend_mlx::{
    backend::{
        nn::shared::MlxNeuralBackend,
        runtime::{
            cache::state::MlxHybridState,
            checkpoint::binding::{
                build_neutral_module_bindings_with_recipes,
            },
            execution::generic::prepare_layerwise_policy_with_bindings,
        },
        ExecutionContext,
    },
    MlxTensor,
};
use eredu_checkpoint::{
    safetensors::SafetensorsMetadataCatalog,
    store::{open_prepared_safetensors_source, RestrictedCheckpointSource, SharedCheckpointSource},
    validation::resolve_safetensors_plan,
};
use eredu_evaluation::checkpoint_reference::{self, sha256};
use eredu_nn::{AttentionCache, Tensor};
use eredu_runtime::{
    DenseDiskStreamLoadOptions, LayerRuntimeState, LayerWeightResidency, LayeredArchitecture,
    LayerwiseRuntime,
};
use safemlx::{Array, Device, DeviceType, Dtype, Stream};
use serde_json::{json, Value};

const REPOSITORY: &str = "meta-models/Muse-Spark-Open-816B-A42B-NVFP4";
const REVISION: &str = "4239e4db19376904f2e82c167e78f70abb5455fe";
// The pinned catalog needs at most nine backing shards across two device
// blocks plus one host lookahead, and three conservative static source readers.
const SOURCE_READER_BOUND: usize = 12;
#[path = "support/muse_spark_media.rs"]
mod released_media;
type Backend = MlxNeuralBackend;
type Model = LayeredModel<Backend>;
type State = MlxHybridState;

#[derive(Parser)]
struct Args {
    /// Complete authenticated, digest-verified official checkpoint directory.
    checkpoint: PathBuf,
    /// Output directory from the pinned Transformers/ModelOpt reference run.
    reference: PathBuf,
    /// Output directory for native scores and comparison evidence.
    output: PathBuf,
    /// Absolute comparison tolerance, chosen before executing the comparison.
    #[arg(long)]
    atol: f64,
    /// Relative comparison tolerance, chosen before executing the comparison.
    #[arg(long)]
    rtol: f64,
    /// Native device; CPU works without concrete accelerator features.
    #[arg(long, default_value = "cpu", value_parser = ["cpu", "gpu"])]
    device: String,
    /// Compare and retain each decoder output against reference layer traces.
    #[arg(long)]
    trace_layers: bool,
    /// Compare mixed image/audio/video prefill followed by three cached steps.
    #[arg(long)]
    multimodal: bool,
}

fn read_json(path: impl AsRef<std::path::Path>) -> Result<Value, Box<dyn Error>> {
    Ok(serde_json::from_slice(&std::fs::read(path)?)?)
}

fn layer_trace(
    value: &MlxTensor,
    reference: &Value,
    step: usize,
    layer: usize,
    reference_dir: &std::path::Path,
    output_dir: &std::path::Path,
    stream: &Stream,
) -> Result<Value, eredu_nn::Error> {
    if reference["step"] != step
        || reference["layer"] != layer
        || reference["shape"] != json!(value.shape())
    {
        return Err(eredu_nn::Error::backend(
            "reference decoder trace geometry differs",
        ));
    }
    let filename = format!("step-{step}-layer-{layer}.f32");
    if reference["file"] != filename {
        return Err(eredu_nn::Error::backend(
            "reference decoder trace path differs",
        ));
    }
    let expected =
        std::fs::read(reference_dir.join(&filename)).map_err(eredu_nn::Error::backend_source)?;
    if reference["sha256"] != sha256(&expected) || expected.len() % 4 != 0 {
        return Err(eredu_nn::Error::backend(
            "reference decoder trace digest differs",
        ));
    }
    let actual = value
        .as_array()
        .as_dtype(Dtype::Float32, stream)
        .map_err(eredu_nn::Error::backend_source)?
        .into_evaluated()
        .map_err(eredu_nn::Error::backend_source)?
        .try_to_vec::<f32>()
        .map_err(eredu_nn::Error::backend_source)?;
    if actual.len() * 4 != expected.len() {
        return Err(eredu_nn::Error::backend(
            "reference decoder trace length differs",
        ));
    }
    let mut maximum = 0.0_f64;
    let mut squared = 0.0_f64;
    let mut different = 0;
    for (&actual, expected) in actual.iter().zip(expected.chunks_exact(4)) {
        let expected = f32::from_le_bytes(expected.try_into().unwrap());
        if !actual.is_finite() || !expected.is_finite() {
            return Err(eredu_nn::Error::backend("nonfinite decoder trace"));
        }
        let delta = (f64::from(actual) - f64::from(expected)).abs();
        maximum = maximum.max(delta);
        squared += delta * delta;
        different += usize::from(actual != expected);
    }
    let bytes = actual
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    std::fs::write(output_dir.join(&filename), &bytes).map_err(eredu_nn::Error::backend_source)?;
    println!("native trace step {step} layer {layer}: max_abs={maximum:.8} different={different}");
    Ok(
        json!({"step":step,"layer":layer,"shape":value.shape(),"file":filename,"sha256":sha256(&bytes),
        "max_abs":maximum,"rms":(squared / actual.len() as f64).sqrt(),"different":different}),
    )
}



fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    if !args.atol.is_finite() || args.atol < 0.0 || !args.rtol.is_finite() || args.rtol < 0.0 {
        return Err("tolerances must be finite and nonnegative".into());
    }
    let reference = read_json(args.reference.join("reference-evidence.json"))?;
    let inventory: Value = serde_json::from_str(include_str!(
        "../../validation/fixtures/muse-spark/inventory.json"
    ))?;
    let proof_hash = checkpoint_reference::verify_payload(
        &args.checkpoint,
        &reference,
        &inventory,
        REPOSITORY,
        REVISION,
    )?;

    // Admit the complete selected artifact before deriving a restricted text view.
    // The media towers retain their actual provenance and are simply unused here.
    let raw = read_json(args.checkpoint.join("config.json"))?;
    let catalog = SafetensorsMetadataCatalog::discover(&args.checkpoint)?;
    let full =
        muse::resolve_safetensors_config(&DecoderConfig::from_hf_value(&raw)?, &raw, &catalog)?;
    if full.identity != MuseIdentity::Spark {
        return Err("expected Spark identity".into());
    }
    let plan = muse::safetensors_plan(&full)?;
    let resolution = resolve_safetensors_plan(&catalog, &plan)
        .map_err(|issues| std::io::Error::other(format!("admission: {issues:?}")))?;
    // Calibration-only inputs are validated by the complete source schema.
    // They have no runtime parameter destination; all learned outputs remain
    // subject to the ordinary policy's strict source-consumption check.
    let recipe_inputs = plan
        .common_tensors
        .iter()
        .chain(
            plan.layout_groups
                .iter()
                .flat_map(|group| &group.variants)
                .flat_map(|variant| &variant.tensors),
        )
        .filter(|tensor| tensor.role == eredu_checkpoint::schema::TensorRole::RecipeInput)
        .flat_map(|tensor| std::iter::once(tensor.key.clone()).chain(tensor.aliases.clone()))
        .filter(|key| resolution.source_keys().contains(key))
        .collect::<std::collections::BTreeSet<_>>();
    let source = open_prepared_safetensors_source(
        catalog.admitted_shards(),
        catalog.tensors().clone(),
        resolution,
        SOURCE_READER_BOUND,
    )?;


    let mut config = full;
    let source = if args.multimodal {
        if reference["media"].is_null() {
            return Err("multimodal comparison requires retained mixed-media inputs".into());
        }
        source
    } else {
        let denied = source
            .source_keys()
            .into_iter()
            .filter(|key| !key.starts_with("model.language_model.") && key != "lm_head.weight")
            .collect();
        config.vision_config = None;
        config.audio_config = None;
        Arc::new(RestrictedCheckpointSource::excluding(
            source,
            "official Spark NVFP4 text validation view",
            denied,
        )?) as SharedCheckpointSource
    };
    let context = ExecutionContext::new(Device::new(
        if args.device == "cpu" {
            DeviceType::Cpu
        } else {
            DeviceType::Gpu
        },
        0,
    ));
    let stream = context.stream();
    let weights = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));

    let mut model = Model::new(config.clone(), stream)?;
    // The binder consumes each module's declared destinations from one retained
    // architecture recipe map; no per-unit source policy is reconstructed here.
    let recipes = RefCell::new(muse::safetensors_recipes(&config, source.as_ref())?);
    // Retain packed expert banks in one decoder-block window. The ordinary
    // selected-expert operator converts only bounded rows of selected experts.
    let residency = LayerWeightResidency::DenseDiskStream(
        DenseDiskStreamLoadOptions::new(64 << 30, 64 << 30, 1, 1)?
            .with_max_cached_shards(SOURCE_READER_BOUND),
    );
    let (policy, metadata) = prepare_layerwise_policy_with_bindings(
        source,
        &mut model,
        (),
        std::marker::PhantomData::<State>,
        residency,
        stream,
        &weights,
        |key| recipe_inputs.contains(key),
        |modules, source| {
            build_neutral_module_bindings_with_recipes(modules, source, &mut recipes.borrow_mut())
                .map_err(Into::into)
        },
        |_, _, _, unit, source, _| {
            build_neutral_module_bindings_with_recipes(&unit, source, &mut recipes.borrow_mut())
                .map_err(Into::into)
        },
    )?;
    if !recipes.borrow().is_empty() {
        return Err("native validation left unowned architecture recipe outputs".into());
    }
    println!("selected bounded runtime: {metadata:?}");
    let mut runtime = LayerwiseRuntime::<Model, Backend, State, _>::new(model, policy);
    let mut state = State::device(muse::state_layout(&config)?, &[])?;
    std::fs::create_dir_all(&args.output)?;
    let started = Instant::now();
    let mut records = Vec::new();
    let mut passed = true;
    let mut traces = Vec::new();
    let reference_traces = if args.trace_layers {
        let traces = reference["layer_traces"]
            .as_array()
            .ok_or("reference has no layer traces")?;
        if traces.len() != config.num_hidden_layers as usize * 4 {
            return Err("reference must trace every layer in all four steps".into());
        }
        Some(traces)
    } else {
        None
    };
    let prompt: &[i32] = if args.multimodal {
        &[
            200000, 1234, 200092, 200092, 5678, 200038, 200038, 21, 200091, 200091,
        ]
    } else {
        &[200000, 1234, 5678]
    };
    let cases: [&[i32]; 4] = [prompt, &[20], &[21], &[22]];
    let media = if args.multimodal {
        let manifest = &reference["media"];
        if manifest["grid"] != json!([[1, 2, 4], [2, 2, 2]])
            || manifest["sample_rate_hz"] != 24000
            || manifest["samples_per_token"] != 1920
            || manifest["parts"]
                != json!([
                {"kind":"text","tokens":2}, {"kind":"image","tokens":2},
                {"kind":"text","tokens":1}, {"kind":"audio","tokens":2},
                {"kind":"text","tokens":1}, {"kind":"video","tokens":2}])
        {
            return Err("mixed-media fixture geometry or framing differs".into());
        }
        let (pixels, shape) = released_media::read_tensor(&args, &manifest["pixels"])?;
        let (waveform, audio_shape) = released_media::read_tensor(&args, &manifest["waveform"])?;
        if shape != [16, 1176] || audio_shape != [1, 3840] {
            return Err("mixed-media fixture source geometry differs".into());
        }
        Some((
            MlxTensor::from_array(Array::from_slice(&pixels, &shape)),
            MlxTensor::from_array(Array::from_slice(&waveform, &audio_shape)),
        ))
    } else {
        None
    };
    let grid = [(1, 2, 4), (2, 2, 2)];
    let reference_steps = reference["steps"]
        .as_array()
        .ok_or("missing reference steps")?;
    if reference_steps.len() != cases.len() {
        return Err("expected prefill and three cached decode steps".into());
    }
    for (step, ids) in cases.into_iter().enumerate() {
        if reference_steps[step]["input_ids"] != json!(ids) {
            return Err("reference token sequence differs".into());
        }
        let tokens = MlxTensor::from_array(Array::from_slice(ids, &[1, ids.len() as i32]));
        let spans = if args.multimodal && step == 0 {
            [2_usize, 2, 1, 2, 1, 2]
                .into_iter()
                .scan(0, |first, count| {
                    let value = MlxTensor::from_array(Array::from_slice(
                        &ids[*first..*first + count],
                        &[1, count as i32],
                    ));
                    *first += count;
                    Some(value)
                })
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        let parts = if spans.is_empty() {
            vec![muse::DecoderInputPart::Text(&tokens)]
        } else {
            vec![
                muse::DecoderInputPart::Text(&spans[0]),
                muse::DecoderInputPart::Media(&spans[1]),
                muse::DecoderInputPart::Text(&spans[2]),
                muse::DecoderInputPart::Audio(&spans[3]),
                muse::DecoderInputPart::Text(&spans[4]),
                muse::DecoderInputPart::Media(&spans[5]),
            ]
        };
        let active_media = media.as_ref().filter(|_| step == 0);
        let (actual, _) = runtime.forward_with_unit_executor_and_context_hook(
            muse::ModelInput {
                parts: &parts,
                vision: active_media.map(|(pixels, _)| muse::VisionInput {
                    pixels,
                    grid: &grid,
                }),
                audio: active_media.map(|(_, waveform)| {
                    muse::audio::execution::AudioInput::Continuation(waveform)
                }),
                mask: None,
            },
            &mut state,
            stream,
            |architecture, group, index, unit, hidden, state, forward, context| {
                let output = architecture
                    .forward_unit(group, index, unit, hidden, state, forward, context)?;
                if group == 1 {
                    if let Some(reference_traces) = reference_traces {
                        traces.push(layer_trace(
                            &output,
                            &reference_traces[step * config.num_hidden_layers as usize + index],
                            step,
                            index,
                            &args.reference,
                            &args.output,
                            context,
                        )?);
                    }
                }
                Ok(output)
            },
            |group, index, _| {
                if group == 1
                    && ((index + 1) % 8 == 0 || index + 1 == config.num_hidden_layers as usize)
                {
                    println!("native step {step}: decoder layer {index} completed");
                }
                Ok(())
            },
        )?;
        let actual_shape = actual.shape();
        if json!(actual_shape) != reference_steps[step]["shape"] {
            return Err("logit geometry differs".into());
        }
        let actual = actual
            .as_array()
            .as_dtype(Dtype::Float32, stream)?
            .into_evaluated()?
            .try_to_vec::<f32>()?;
        let bytes = std::fs::read(args.reference.join(format!("step-{step}.f32")))?;
        if sha256(&bytes)
            != reference_steps[step]["sha256"]
                .as_str()
                .ok_or("missing reference digest")?
            || bytes.len() != actual.len() * 4
        {
            return Err("reference score digest or length differs".into());
        }
        let expected: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        let mut maximum = 0.0_f64;
        let mut squared = 0.0_f64;
        let mut outside = 0_usize;
        for (&a, &b) in actual.iter().zip(&expected) {
            if !a.is_finite() || !b.is_finite() {
                return Err("nonfinite released-model scores".into());
            }
            let error = (f64::from(a) - f64::from(b)).abs();
            maximum = maximum.max(error);
            squared += error * error;
            outside += usize::from(error > args.atol + args.rtol * f64::from(b).abs());
        }
        let positions: Vec<_> = (0..config.num_hidden_layers as usize)
            .map(|layer| state.layer(layer).map(|state| state.offset()))
            .collect::<Result<_, _>>()?;
        if positions
            .iter()
            .any(|position| json!(position) != reference_steps[step]["cache_after"])
        {
            return Err("native cached decode frontiers differ from the reference".into());
        }
        let filename = format!("step-{step}.f32");
        let bytes: Vec<u8> = actual
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        std::fs::write(args.output.join(&filename), &bytes)?;
        let rms = (squared / actual.len() as f64).sqrt();
        let top1 = actual
            .chunks(config.vocab_size as usize)
            .map(|row| {
                row.iter()
                    .enumerate()
                    .fold((0, f32::NEG_INFINITY), |best, (index, value)| {
                        if *value > best.1 {
                            (index, *value)
                        } else {
                            best
                        }
                    })
                    .0
            })
            .collect::<Vec<_>>();
        println!("step {step}: max_abs={maximum:.8} rms={rms:.8} outside_tolerance={outside}");
        passed &= outside == 0;
        records.push(
            json!({"input_ids": ids, "shape": actual_shape, "cache_after": positions[0],
            "max_abs": maximum, "rms": rms, "outside_tolerance": outside,
            "top1": [&top1], "top1_matches": json!([&top1]) == reference_steps[step]["top1"],
            "file": filename, "sha256": sha256(&bytes)}),
        );
    }
    let report = json!({"repository": REPOSITORY, "revision": REVISION, "payload_proof_sha256": proof_hash,
        "architecture_fingerprint": config.architecture_fingerprint(),
        "native_build": {"debug_assertions":cfg!(debug_assertions), "accelerate":cfg!(feature="accelerate"), "metal":cfg!(feature="metal")},
        "package_version": env!("CARGO_PKG_VERSION"),
        "reader_bound": SOURCE_READER_BOUND, "device_budget_bytes": 64_u64 << 30, "host_budget_bytes":64_u64 << 30,
        "native_runner_sha256": sha256(include_bytes!("muse_spark_reference.rs")),
        "payload_validator_sha256": checkpoint_reference::validator_sha256(),
        "layer_traces": traces,
        "comparison_kind": reference["comparison_kind"],
        "media": reference["media"],
        "reference_evidence_sha256": sha256(&std::fs::read(args.reference.join("reference-evidence.json"))?),
        "device": args.device, "residency": "ordinary dense disk stream, packed banks, selected-expert bounded conversion",
        "atol": args.atol, "rtol": args.rtol, "eredu_comparison_verified": passed,
        "elapsed_seconds": started.elapsed().as_secs_f64(), "steps": records});
    std::fs::write(
        args.output.join("comparison-evidence.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    if !passed {
        return Err(
            "released-model comparison exceeds declared tolerances; evidence retained".into(),
        );
    }
    Ok(())
}
