//! Released assistant neural comparison through the shared unit-policy runtime.
//! This does not register a target family or establish speculative session parity.
use clap::Parser;
use eredu_architectures::muse::{
    assistant::{
        integration::{MuseAssistantModule, MuseAssistantProposal},
        spark::{SparkAssistantConfig, SparkAssistantLoadConfig},
    },
    DecoderConfig,
};
use eredu_backend_mlx::{
    backend::{
        nn::shared::MlxNeuralBackend,
        runtime::{
            checkpoint::binding::{
                build_neutral_module_bindings_with_recipes, materialize_module_bindings,
            },
            execution::generic::prepare_parameter_unit_policy,
        },
    },
    MlxTensor,
};
use eredu_checkpoint::{
    safetensors::SafetensorsMetadataCatalog,
    schema::{
        CatalogPolicy, SafetensorsCheckpointPlan, SafetensorsTensorConstraint,
        StoredDtypeConstraint,
    },
    store::open_prepared_safetensors_source,
    validation::resolve_safetensors_plan,
};
use eredu_nn::{Linear, LinearFormatSpec, LinearSpec, ParameterSpec, Tensor};
use eredu_runtime::LayerWeightResidency;
use safemlx::{Array, Device, DeviceType, Dtype, Stream};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    error::Error,
    io::Read,
    path::{Path, PathBuf},
    time::Instant,
};

#[derive(Parser)]
struct Args {
    checkpoint: PathBuf,
    target_config: PathBuf,
    reference: PathBuf,
    output: PathBuf,
    /// Comparison thresholds fixed before the native run.
    #[arg(long)]
    atol: f64,
    #[arg(long)]
    rtol: f64,
    /// Compare intermediate block publications to locate numerical divergence.
    #[arg(long)]
    trace_layers: bool,
    /// Exact target component source required by assistants borrowing its head.
    #[arg(long)]
    target_checkpoint: Option<PathBuf>,
}
fn sha256(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}
fn read_json(path: impl AsRef<Path>) -> Result<Value, Box<dyn Error>> {
    Ok(serde_json::from_slice(&std::fs::read(path)?)?)
}
fn record_data(root: &Path, record: &Value) -> Result<Vec<f32>, Box<dyn Error>> {
    let name = record["file"].as_str().ok_or("missing tensor file")?;
    if Path::new(name).file_name().and_then(|p| p.to_str()) != Some(name) {
        return Err("tensor file must be a direct child".into());
    }
    let bytes = std::fs::read(root.join(name))?;
    if record["sha256"] != sha256(&bytes) || bytes.len() % 4 != 0 {
        return Err("reference tensor digest or byte extent differs".into());
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect())
}
fn load(root: &Path, record: &Value, stream: &Stream) -> Result<MlxTensor, Box<dyn Error>> {
    let shape = record["shape"]
        .as_array()
        .ok_or("missing tensor shape")?
        .iter()
        .map(|v| i32::try_from(v.as_u64().ok_or("invalid shape")?).map_err(Into::into))
        .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
    let values = record_data(root, record)?;
    let dtype = match record["dtype"].as_str() {
        Some("bfloat16") => Dtype::Bfloat16,
        Some("float32") => Dtype::Float32,
        _ => return Err("unexpected reference input dtype".into()),
    };
    Ok(MlxTensor::from_array(
        Array::from_slice(&values, &shape).as_dtype(dtype, stream)?,
    ))
}
fn compare(
    value: &MlxTensor,
    record: &Value,
    args: &Args,
    stream: &Stream,
) -> Result<Value, Box<dyn Error>> {
    if json!(value.shape()) != record["shape"] {
        return Err(format!(
            "reference shape differs for {}: actual {:?}, expected {}",
            record["file"],
            value.shape(),
            record["shape"]
        )
        .into());
    }
    let expected = record_data(&args.reference, record)?;
    let actual = value.to_f32_vec(stream)?;
    if actual.len() != expected.len() {
        return Err("reference tensor length differs".into());
    }
    let (mut max_abs, mut squared, mut outside, mut support_mismatches) =
        (0f64, 0f64, 0usize, 0usize);
    for (&a, &e) in actual.iter().zip(&expected) {
        if a == f32::NEG_INFINITY && e == f32::NEG_INFINITY {
            continue;
        }
        if a.is_nan() || e.is_nan() || a == f32::INFINITY || e == f32::INFINITY {
            return Err("nonfinite neural comparison".into());
        }
        if a == f32::NEG_INFINITY || e == f32::NEG_INFINITY {
            support_mismatches += 1;
            outside += 1;
            continue;
        }
        let delta = (f64::from(a) - f64::from(e)).abs();
        max_abs = max_abs.max(delta);
        squared += delta * delta;
        outside += usize::from(delta > args.atol + args.rtol * f64::from(e).abs());
    }
    let bytes = actual
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<_>>();
    std::fs::write(args.output.join(record["file"].as_str().unwrap()), &bytes)?;
    Ok(
        json!({"file":record["file"],"shape":record["shape"],"sha256":sha256(&bytes),
        "max_abs":max_abs,"rms":(squared/actual.len() as f64).sqrt(),"outside_tolerance":outside,
        "support_mismatches":support_mismatches}),
    )
}
struct Trace<'a> {
    args: &'a Args,
    expected: &'a Value,
    stream: &'a Stream,
    records: Vec<Value>,
}
impl eredu_runtime::ActivationObserver<MlxTensor, eredu_nn::Error> for Trace<'_> {
    fn observe(&mut self, path: &str, value: &MlxTensor) -> Result<(), eredu_nn::Error> {
        for (layer, records) in self
            .expected
            .as_array()
            .ok_or_else(|| eredu_nn::Error::backend("missing layer traces"))?
            .iter()
            .enumerate()
        {
            for (suffix, record) in records
                .as_object()
                .ok_or_else(|| eredu_nn::Error::backend("missing trace records"))?
            {
                if path == format!("draft.layers.{layer}.{suffix}") {
                    let record = compare(value, record, self.args, self.stream)
                        .map_err(|error| eredu_nn::Error::backend(error.to_string()))?;
                    println!(
                        "{path}: max_abs={} outside={}",
                        record["max_abs"], record["outside_tolerance"]
                    );
                    self.records.push(record);
                }
            }
        }
        Ok(())
    }
}
fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    if !args.atol.is_finite() || !args.rtol.is_finite() || args.atol < 0. || args.rtol < 0. {
        return Err("comparison thresholds must be finite and nonnegative".into());
    }
    std::fs::create_dir_all(&args.output)?;
    let reference = read_json(args.reference.join("reference-evidence.json"))?;
    if reference["steps"].as_array().map(Vec::len) != Some(4) {
        return Err("released comparison requires four context updates".into());
    }
    let proof_bytes = std::fs::read(args.checkpoint.join("payload-evidence.json"))?;
    let config_bytes = std::fs::read(args.checkpoint.join("config.json"))?;
    let proof: Value = serde_json::from_slice(&proof_bytes)?;
    if ![
        "6f7aed1cac8d8b1c02a902988876172b39e49fc0",
        "46c90f412549c472941a68382d91a99e5e6372a0",
    ]
    .iter()
    .any(|revision| reference["revision"] == *revision)
        || reference["revision"] != proof["revision"]
        || reference["payload_proof_sha256"] != sha256(&proof_bytes)
        || reference["config_sha256"] != sha256(&config_bytes)
        || proof["all_weight_payload_sha256_verified"] != true
    {
        return Err("released assistant reference lineage differs".into());
    }
    let target = DecoderConfig::from_hf_value(&read_json(&args.target_config)?)?;
    let config = SparkAssistantLoadConfig::new(SparkAssistantConfig::from_hf_value(
        &serde_json::from_slice(&config_bytes)?,
    )?);
    let definition = config.construction(&target)?.execution_definition()?;
    let plan = config.safetensors_plan()?;
    let catalog = SafetensorsMetadataCatalog::discover(&args.checkpoint)?;
    let resolution = resolve_safetensors_plan(&catalog, &plan)
        .map_err(|issues| std::io::Error::other(format!("admission: {issues:?}")))?;
    let source = open_prepared_safetensors_source(
        catalog.admitted_shards(),
        catalog.tensors().clone(),
        resolution,
        1,
    )?;
    let device = Device::new(DeviceType::Cpu, 0);
    let stream = Stream::new_with_device(&device);
    let weights = Stream::new_with_device(&device);
    let target_head = if let Some(component) = reference.get("target_head") {
        let root = args
            .target_checkpoint
            .as_ref()
            .ok_or("borrowed head requires --target-checkpoint")?;
        let target_proof = read_json(root.join("payload-evidence.json"))?;
        if target_proof["revision"] != reference["target_revision"]
            || target_proof["all_weight_payload_sha256_verified"] != true
            || reference["target_payload_proof_sha256"]
                != sha256(&std::fs::read(root.join("payload-evidence.json"))?)
        {
            return Err("target component lineage differs".into());
        }
        let name = component["name"]
            .as_str()
            .ok_or("missing borrowed head name")?;
        let file = component["file"]
            .as_str()
            .ok_or("missing borrowed head member")?;
        if Path::new(file).file_name().and_then(|p| p.to_str()) != Some(file) {
            return Err("invalid target component member".into());
        }
        let physical = root.join(file);
        let metadata = std::fs::metadata(&physical)?;
        let proof_file = target_proof["files"]
            .as_array()
            .ok_or("missing target payload inventory")?
            .iter()
            .find(|record| record["file"] == file)
            .ok_or("missing head payload proof")?;
        if proof_file["bytes"].as_u64() != Some(metadata.len())
            || metadata.modified()?
                > std::fs::metadata(root.join("payload-evidence.json"))?.modified()?
        {
            return Err("verified target head source changed".into());
        }
        let retained_header = read_json(root.join(format!("{file}.header.json")))?;
        let bytes = retained_header["header_bytes"]
            .as_u64()
            .filter(|n| *n <= 16 << 20)
            .ok_or("invalid head header extent")?;
        let mut header = vec![0; bytes as usize];
        std::fs::File::open(&physical)?.read_exact(&mut header)?;
        if sha256(&header) != component["header"] {
            return Err("borrowed head header differs".into());
        }
        let catalog =
            SafetensorsMetadataCatalog::discover_members_matching(root, |key| key == name)?;
        let plan = SafetensorsCheckpointPlan::new(
            "validation borrowed target head",
            vec![SafetensorsTensorConstraint::required(
                name,
                vec![target.vocab_size as usize, target.hidden_size as usize],
                StoredDtypeConstraint::Exact(eredu_checkpoint::StoredDtype::BF16),
            )],
            Vec::new(),
            CatalogPolicy::non_strict(),
        )?;
        let resolution = resolve_safetensors_plan(&catalog, &plan)
            .map_err(|issues| std::io::Error::other(format!("head admission: {issues:?}")))?;
        let source = open_prepared_safetensors_source(
            catalog.admitted_shards(),
            catalog.tensors().clone(),
            resolution,
            1,
        )?;
        let linear = Linear::<MlxTensor>::unloaded(
            LinearSpec {
                input: target.hidden_size,
                output: target.vocab_size,
                weight: ParameterSpec::trainable(name)?,
                bias: None,
                format: LinearFormatSpec::unscaled(eredu_checkpoint::LinearFormat::Dense)?,
            },
            &stream,
        )?;
        let bindings = build_neutral_module_bindings_with_recipes(
            &linear,
            source.as_ref(),
            &mut BTreeMap::new(),
        )?;
        let mut arrays =
            materialize_module_bindings(source.as_ref(), &bindings, &weights, &stream)?;
        if arrays.len() != 1 {
            return Err("borrowed head must materialize one exact parameter".into());
        }
        Some(MlxTensor::from_array(
            arrays.remove(name).ok_or("missing borrowed head")?,
        ))
    } else {
        None
    };
    let units = definition.build_units::<MlxNeuralBackend>(&stream)?;
    let bindings = units
        .iter()
        .map(|unit| {
            build_neutral_module_bindings_with_recipes(unit, source.as_ref(), &mut BTreeMap::new())
        })
        .collect::<Result<Vec<_>, _>>()?;
    let (policy, _) = prepare_parameter_unit_policy::<_, _, _, _>(
        source,
        &mut Vec::<
            eredu_architectures::muse::assistant::execution::MuseAssistantUnit<MlxNeuralBackend>,
        >::new(),
        (),
        LayerWeightResidency::FullyResident,
        &stream,
        &weights,
        |_| false,
        definition.unit_layout().clone(),
        Vec::new(),
        bindings,
        Vec::new(),
    )?;
    let mut module = MuseAssistantModule::<MlxNeuralBackend, _>::new(
        definition,
        policy.into_resident_units(units, &stream)?,
    );
    let (mut previous, mut records, mut passed) = (None, Vec::new(), true);
    let started = Instant::now();
    for (step, expected) in reference["steps"]
        .as_array()
        .ok_or("missing reference updates")?
        .iter()
        .enumerate()
    {
        let pending = load(&args.reference, &expected["pending"], &stream)?;
        let committed = module.update_context(
            previous.take(),
            &pending,
            i32::try_from(expected["end"].as_i64().ok_or("missing context end")?)?,
            &stream,
        )?;
        let mut comparisons = vec![compare(
            &committed.encoded,
            &expected["encoded"],
            &args,
            &stream,
        )?];
        if expected["layers"].as_array().map(Vec::len) != Some(committed.layers.len())
            || expected["rows"].as_array().map(Vec::len)
                != Some(config.released().proposal_capacity())
        {
            return Err("incomplete released context or proposal inventory".into());
        }
        for (layer, expected) in committed.layers.iter().zip(
            expected["layers"]
                .as_array()
                .ok_or("missing context layers")?,
        ) {
            comparisons.push(compare(&layer.keys, &expected["keys"], &args, &stream)?);
            comparisons.push(compare(&layer.values, &expected["values"], &args, &stream)?);
        }
        let embeddings = load(&args.reference, &expected["embeddings"], &stream)?;
        let mut trace = Trace {
            args: &args,
            expected: &expected["traces"],
            stream: &stream,
            records: Vec::new(),
        };
        let end = i32::try_from(expected["end"].as_i64().unwrap())?;
        let hidden = if args.trace_layers {
            module.proposal_states_observed(
                &embeddings,
                &committed,
                end,
                &stream,
                &mut eredu_architectures::decoder::ComponentInstrumentation::new(
                    "draft", &mut trace,
                ),
            )?
        } else {
            module.proposal_states(&embeddings, &committed, end, &stream)?
        };
        if args.trace_layers
            && trace.records.len()
                != expected["traces"].as_array().ok_or("missing traces")?.len() * 4
        {
            return Err("missing native intermediate publications".into());
        }
        comparisons.extend(trace.records);
        comparisons.push(compare(&hidden, &expected["hidden"], &args, &stream)?);
        let raw = target_head
            .as_ref()
            .map(|weight| MlxTensor::linear(&hidden, weight, None, &stream))
            .transpose()?;
        if let Some(raw) = &raw {
            comparisons.push(compare(raw, &expected["raw"], &args, &stream)?);
        }
        let proposal =
            module.prepare_proposal(&hidden, raw.as_ref(), target.vocab_size, &stream)?;
        let candidate_ids = if let MuseAssistantProposal::DFlash2 { ids, .. } = &proposal {
            let actual = ids.to_i32_vec(&stream)?;
            let expected_ids = expected["candidate_ids"]
                .as_array()
                .ok_or("missing candidates")?
                .iter()
                .flat_map(|batch| batch.as_array().unwrap())
                .flat_map(|row| row.as_array().unwrap())
                .map(|id| id.as_i64().unwrap() as i32)
                .collect::<Vec<_>>();
            passed &= actual == expected_ids;
            Some(actual)
        } else {
            None
        };
        let mut top1 = Vec::new();
        for (row, expected) in expected["rows"]
            .as_array()
            .ok_or("missing reference score rows")?
            .iter()
            .enumerate()
        {
            let scores = module.proposal_row(
                &proposal,
                row,
                u32::try_from(expected["previous"].as_u64().ok_or("missing predecessor")?)?,
                &stream,
            )?;
            comparisons.push(compare(
                &scores.logits,
                &expected["scores"],
                &args,
                &stream,
            )?);
            match (scores.confidence.as_ref(), expected.get("confidence")) {
                (Some(actual), Some(expected)) => {
                    comparisons.push(compare(actual, expected, &args, &stream)?)
                }
                (None, None) => (),
                _ => return Err("assistant confidence contract differs".into()),
            }
            let values = scores.logits.to_f32_vec(&stream)?;
            let best = values
                .iter()
                .enumerate()
                .max_by(|(ai, a), (bi, b)| a.total_cmp(b).then_with(|| bi.cmp(ai)))
                .ok_or("empty score row")?
                .0;
            passed &= expected["top1"].as_u64() == Some(best as u64);
            top1.push(best);
        }
        passed &= comparisons.iter().all(|r| r["outside_tolerance"] == 0);
        println!("native released assistant step {step}: top1={top1:?}, cumulative_pass={passed}");
        records.push(json!({"step":step,"top1":top1,"candidate_ids":candidate_ids,"comparisons":comparisons}));
        previous = Some(committed);
    }
    let report = json!({"repository":reference["repository"],"revision":reference["revision"],
        "reference_evidence_sha256":sha256(&std::fs::read(args.reference.join("reference-evidence.json"))?),
        "payload_proof_sha256":sha256(&proof_bytes),"atol":args.atol,"rtol":args.rtol,
        "comparison_verified":passed,"comparison_kind":"released BF16 neural body/head composition through ordinary resident units; no speculative session claim",
        "elapsed_seconds":started.elapsed().as_secs_f64(),"steps":records});
    std::fs::write(
        args.output.join("native-comparison.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    if !passed {
        return Err("released assistant neural comparison differs".into());
    }
    Ok(())
}
