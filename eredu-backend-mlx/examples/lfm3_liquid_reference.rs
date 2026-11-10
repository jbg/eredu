//! Released LiquidCodec spectral/FSQ component acceptance; not full codec support.
use clap::Parser;
use eredu_backend_mlx::{
    backend::{
        nn::shared::MlxNeuralBackend,
        runtime::checkpoint::binding::{
            build_neutral_module_bindings_with_recipes, materialize_module_bindings,
            populate_module_from_arrays_excluding,
        },
        ExecutionContext,
    },
    MlxTensor,
};
use eredu_checkpoint::{
    safetensors::SafetensorsMetadataCatalog, store::open_prepared_safetensors_source,
    validation::resolve_safetensors_plan,
};
use eredu_codec::{
    liquid::{Analysis, Config, Quantizer, QuantizerSpec, Synthesis},
    spectral::OverlapAddStream,
};
use eredu_nn::{Index, Tensor};
use safemlx::{Array, Device, DeviceType};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    error::Error,
    io::Read,
    path::{Path, PathBuf},
};
type Backend = MlxNeuralBackend;

#[derive(Parser)]
struct Args {
    /// F32 learned scalar-projection checkpoint produced by the reference runner.
    checkpoint: PathBuf,
    /// Directory containing reference.json and reference.safetensors.
    reference: PathBuf,
    /// Compact numerical results outside the repository.
    output: PathBuf,
    #[arg(long,default_value="cpu",value_parser=["cpu","gpu"])]
    device: String,
}

fn sha256(path: &Path) -> Result<String, Box<dyn Error>> {
    let mut input = std::fs::File::open(path)?;
    let mut hash = Sha256::new();
    let mut buffer = [0u8; 65536];
    loop {
        let n = input.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
    }
    Ok(hash.finalize().iter().map(|b| format!("{b:02x}")).collect())
}
fn compare(actual: &MlxTensor, expected: &Array, atol: f64) -> Result<Value, Box<dyn Error>> {
    if actual.shape() != expected.shape() {
        return Err("LiquidCodec reference shape differs".into());
    }
    let a = actual.as_array().evaluated()?.try_to_vec::<f32>()?;
    let b = expected.evaluated()?.try_to_vec::<f32>()?;
    let mut maximum = 0f64;
    let mut delta = 0f64;
    let mut norm = 0f64;
    let mut finite = true;
    for (a, b) in a.iter().zip(&b) {
        finite &= a.is_finite() && b.is_finite();
        let d = f64::from(*a) - f64::from(*b);
        maximum = maximum.max(d.abs());
        delta += d * d;
        norm += f64::from(*b) * f64::from(*b);
    }
    Ok(
        json!({"shape":actual.shape(),"maximum_absolute_error":maximum,"relative_l2":delta.sqrt()/norm.sqrt().max(1e-12),
        "finite":finite,"passed":finite&&maximum<=atol}),
    )
}
fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    let config = Config::from_json(&std::fs::read(args.checkpoint.join("config.json"))?)?;
    let spec = QuantizerSpec::new(&config, "quantizer")?;
    let metadata: Value =
        serde_json::from_slice(&std::fs::read(args.reference.join("reference.json"))?)?;
    if metadata["derived_sha256"] != sha256(&args.checkpoint.join("model.safetensors"))? {
        return Err("native scalar projection weights differ from the reference artifact".into());
    }
    let catalog = SafetensorsMetadataCatalog::discover(&args.checkpoint)?;
    let resolution = resolve_safetensors_plan(&catalog, &spec.safetensors_plan()?)
        .map_err(|e| format!("scalar projection admission: {e:?}"))?;
    let source = open_prepared_safetensors_source(
        catalog.admitted_shards(),
        catalog.tensors().clone(),
        resolution,
        2,
    )?;
    let io = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let context = ExecutionContext::new(Device::new(
        if args.device == "gpu" {
            DeviceType::Gpu
        } else {
            DeviceType::Cpu
        },
        0,
    ));
    let stream = context.stream();
    let reference =
        Array::load_safetensors(args.reference.join("reference.safetensors"), io.stream())?;
    let mut quantizer = Quantizer::<Backend>::new(&spec, stream)?;
    let bindings = build_neutral_module_bindings_with_recipes(
        &quantizer,
        source.as_ref(),
        &mut BTreeMap::new(),
    )?;
    let arrays = materialize_module_bindings(source.as_ref(), &bindings, io.stream(), stream)?;
    populate_module_from_arrays_excluding(&mut quantizer, &arrays, |_| false)?;
    let analysis = Analysis::<MlxTensor>::new(&config, stream)?;
    let synthesis = Synthesis::<Backend>::new(&config, stream)?;
    let atol = metadata["maximum_absolute_tolerance"]
        .as_f64()
        .ok_or("missing comparison tolerance")?;
    let mut results = Vec::new();
    for case in metadata["cases"].as_array().ok_or("missing cases")? {
        let name = case["id"].as_str().ok_or("missing case identity")?;
        let tensor = |suffix: &str| -> Result<MlxTensor, Box<dyn Error>> {
            Ok(MlxTensor::from_array(
                reference[&format!("{name}.{suffix}")].copy(stream)?,
            ))
        };
        let features = analysis.features(&tensor("pcm")?, stream)?;
        let features = compare(&features, &reference[&format!("{name}.features")], atol)?;
        let codes = quantizer.encode(&tensor("latents")?, stream)?;
        let codes_match = codes.to_i32_vec(stream)?
            == reference[&format!("{name}.codes")]
                .evaluated()?
                .try_to_vec::<i32>()?;
        let decoded = quantizer.decode(&tensor("codes")?, stream)?;
        let decoded = compare(&decoded, &reference[&format!("{name}.decoded")], atol)?;
        let log = tensor("log_magnitude")?;
        let angle = tensor("angle")?;
        let raw = synthesis.overlap_add(&log, &angle, stream)?;
        let raw_audio = compare(&raw.audio, &reference[&format!("{name}.raw_audio")], atol)?;
        let raw_envelope = compare(
            &raw.envelope,
            &reference[&format!("{name}.raw_envelope")],
            atol,
        )?;
        let pcm = synthesis.reconstruct(&log, &angle, stream)?;
        let pcm = compare(&pcm, &reference[&format!("{name}.full_pcm")], atol)?;
        let mut state = OverlapAddStream::new(1, config.stft_n_fft, config.stft_hop_length)?;
        let mut start = 0i32;
        let mut chunks = Vec::new();
        for count in case["spectral_chunks"]
            .as_array()
            .ok_or("missing spectral chunks")?
        {
            let count = count.as_i64().ok_or("invalid spectral chunk")? as i32;
            let slice = |t: &MlxTensor| {
                t.index(
                    &[Index::Full, Index::Full, Index::Range(start, start + count)],
                    stream,
                )
            };
            let raw = synthesis.overlap_add(&slice(&log)?, &slice(&angle)?, stream)?;
            chunks.push(state.push(Some(raw), false, stream)?);
            start += count;
        }
        chunks.push(state.push(None, true, stream)?);
        let lengths = chunks.iter().map(|t| t.dim(2)).collect::<Vec<_>>();
        let lengths_match = json!(lengths) == case["streamed_samples"];
        let streamed = compare(
            &MlxTensor::concatenate(&chunks, 2, stream)?,
            &reference[&format!("{name}.streamed_pcm")],
            atol,
        )?;
        results.push(json!({"id":name,"features":features,"codes_match":codes_match,"decoded":decoded,
            "raw_audio":raw_audio,"raw_envelope":raw_envelope,"full_pcm":pcm,"streamed_pcm":streamed,
            "streamed_samples":lengths,"streamed_sample_counts_match":lengths_match}));
    }
    let passed = results.iter().all(|r| {
        r["codes_match"] == true
            && r["streamed_sample_counts_match"] == true
            && [
                "features",
                "decoded",
                "raw_audio",
                "raw_envelope",
                "full_pcm",
                "streamed_pcm",
            ]
            .iter()
            .all(|k| r[k]["passed"] == true)
    });
    std::fs::write(
        &args.output,
        serde_json::to_vec_pretty(&json!({"device":args.device,"dtype":"float32",
        "revision":metadata["revision"],"derived_sha256":metadata["derived_sha256"],
        "maximum_absolute_tolerance":atol,"cases":results,"passed":passed}))?,
    )?;
    println!("output={} passed={passed}", args.output.display());
    if !passed {
        return Err("LiquidCodec component parity failed".into());
    }
    Ok(())
}
