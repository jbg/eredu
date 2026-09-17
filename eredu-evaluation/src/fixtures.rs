//! Synthetic checkpoint writers shared by native and facade conformance.
//! These preserve released tensor encodings but are not released checkpoints.
use std::{io::Write, path::Path};
use eredu_checkpoint::{schema::StoredDtypeConstraint, StoredDtype};

/// Failure while writing the retained synthetic checkpoint source.
#[derive(Debug, thiserror::Error)]
pub enum FixtureError {
    /// The architecture rejected its exact fixed fixture configuration.
    #[error("invalid checkpoint fixture: {0}")]
    Configuration(String),
    /// Artifact serialization failed.
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    /// Artifact I/O failed.
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

/// Writes the existing two-layer GPT-OSS Ring fixture with its exact published
/// MXFP4 block/scale geometry and independent gate/up and down biases.
///
/// `component_patterns` selects the existing signed, nonzero component pattern;
/// false preserves the original constant-row fixture. The intermediate width
/// of 96 yields unequal 32-aligned shards when split across two tensor ranks.
/// Callers own the directory and tokenizer/application configuration.
pub fn write_gpt_oss_checkpoint(directory: &Path, component_patterns: bool) -> Result<(), FixtureError> {
    let config=serde_json::json!({
        "model_type":"gpt_oss", "hidden_size":64, "intermediate_size":96,
        "num_hidden_layers":2, "num_attention_heads":4, "num_key_value_heads":2,
        "head_dim":32, "vocab_size":64, "num_local_experts":2, "num_experts_per_tok":1,
        "rms_norm_eps":0.00001, "sliding_window":3, "max_position_embeddings":128,
        "rope_theta":150000.0, "layer_types":["sliding_attention","full_attention"],
        "quantization_config":{"quant_method":"mxfp4"}, "swiglu_limit":7.0
    });
    let args=eredu_architectures::gpt_oss::model_args_from_config_value(&config)
        .map_err(|e| FixtureError::Configuration(e.to_string()))?;
    let plan=eredu_architectures::gpt_oss::safetensors_plan(&args)
        .map_err(FixtureError::Configuration)?;
    let mut header=serde_json::Map::new();let mut data=Vec::new();
    for tensor in &plan.common_tensors {
        let start=data.len();let count=tensor.shape.iter().product::<usize>();
        let dtype=if matches!(&tensor.dtype,StoredDtypeConstraint::Exact(StoredDtype::U8)) {
            // Published expert blocks store two FP4 nibbles per byte; scales
            // use the E8M0 exponent byte. Never serialize them as floating data.
            data.resize(start+count,if tensor.key.ends_with("_scales") {127} else {0x11});
            "U8"
        } else {
            let ordinal=tensor.key.bytes().map(u32::from).sum::<u32>()%17;
            for index in 0..count {
                let value=if tensor.key.ends_with("norm.weight") {1.0f32}
                    else if component_patterns {(((index*17+ordinal as usize*11)%29) as f32-14.0)*0.003}
                    else {0.002+ordinal as f32*0.0003};
                data.extend_from_slice(&value.to_le_bytes());
            }
            "F32"
        };
        header.insert(tensor.key.clone(),serde_json::json!({"dtype":dtype,"shape":tensor.shape,
            "data_offsets":[start,data.len()]}));
    }
    let mut header=serde_json::to_vec(&header)?;
    while !header.len().is_multiple_of(8) {header.push(b' ');}
    let mut weights=std::fs::File::create(directory.join("model.safetensors"))?;
    weights.write_all(&(header.len() as u64).to_le_bytes())?;
    weights.write_all(&header)?;weights.write_all(&data)?;
    std::fs::write(directory.join("config.json"),serde_json::to_vec_pretty(&config)?)?;
    Ok(())
}
