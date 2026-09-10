//! Neutral normalization of published block-FP8 checkpoint metadata.
use crate::{BlockFp8Format, BlockFp8ScaleEncoding};
use std::collections::BTreeSet;

/// Exact block-FP8 encoding and parameters explicitly retained in dense storage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockFp8Metadata {
    format: BlockFp8Format,
    excluded: BTreeSet<String>,
    scale_suffix: &'static str,
}
impl BlockFp8Metadata {
    /// Normalizes dynamic E4M3 block formats without allocating native tensors.
    pub fn parse(value: &serde_json::Value) -> Result<Self, String> {
        let invalid = || "unsupported block-FP8 checkpoint metadata".to_owned();
        let method = value["quant_method"].as_str().ok_or_else(invalid)?;
        let (block, ignore, scale_suffix) = match method {
            "fp8" => {
                if value["activation_scheme"] != "dynamic"
                    || value.get("fmt").is_some_and(|v| v != "e4m3")
                {
                    return Err(invalid());
                }
                if let (Some(first), Some(second)) = (
                    value.get("ignored_layers"),
                    value.get("modules_to_not_convert"),
                ) {
                    if first != second {
                        return Err("conflicting block-FP8 module exclusions".into());
                    }
                }
                let exclusions = value
                    .get("ignored_layers")
                    .or_else(|| value.get("modules_to_not_convert"));
                (&value["weight_block_size"], exclusions, "weight_scale_inv")
            }
            "compressed-tensors" => {
                if value["format"] != "float-quantized"
                    || value["quantization_status"] != "compressed"
                    || value.get("kv_cache_scheme").is_some_and(|v| !v.is_null())
                {
                    return Err(invalid());
                }
                let groups = value["config_groups"].as_object().ok_or_else(invalid)?;
                if groups.len() != 1 {
                    return Err(
                        "block-FP8 metadata requires one homogeneous Linear configuration group"
                            .into(),
                    );
                }
                let group = groups.values().next().expect("one group");
                let weights = &group["weights"];
                let input = &group["input_activations"];
                if group["targets"] != serde_json::json!(["Linear"])
                    || weights["type"] != "float"
                    || weights["num_bits"] != 8
                    || weights["strategy"] != "block"
                    || weights["symmetric"] != true
                    || weights["dynamic"] != false
                    || input["type"] != "float"
                    || input["num_bits"] != 8
                    || input["dynamic"] != true
                    || input["symmetric"] != true
                    || input["strategy"] != "group"
                    || input["group_size"] != 128
                    || group
                        .get("output_activations")
                        .is_some_and(|v| !v.is_null())
                {
                    return Err(invalid());
                }
                (
                    &weights["block_structure"],
                    value.get("ignore"),
                    "weight_scale",
                )
            }
            _ => return Err(invalid()),
        };
        if block != &serde_json::json!([128, 128]) {
            return Err("block-FP8 metadata requires 128 by 128 weight blocks".into());
        }
        let excluded = match ignore {
            None => BTreeSet::new(),
            Some(value) => value
                .as_array()
                .ok_or_else(invalid)?
                .iter()
                .map(|v| {
                    v.as_str()
                        .filter(|s| !s.trim().is_empty())
                        .map(str::to_owned)
                        .ok_or_else(invalid)
                })
                .collect::<Result<_, _>>()?,
        };
        Ok(Self {
            format: BlockFp8Format::new(128, 128, BlockFp8ScaleEncoding::FloatingPoint)
                .map_err(|e| e.to_string())?,
            excluded,
            scale_suffix,
        })
    }
    /// Native E4M3 block geometry; scales are dequantization multipliers.
    pub const fn format(&self) -> BlockFp8Format {
        self.format
    }
    /// Whether a named Linear module is selected by this policy.
    pub fn includes(&self, weight: &str) -> bool {
        !self.excluded.iter().any(|module| {
            weight
                .strip_prefix(module)
                .is_some_and(|tail| tail.is_empty() || tail.starts_with('.'))
        })
    }
    /// Source companion name, preserving the publisher's naming convention.
    pub fn source_scale_name(&self, weight: &str) -> Result<String, String> {
        let prefix = weight
            .strip_suffix(".weight")
            .ok_or_else(|| "FP8 matrix identity must end in .weight".to_owned())?;
        Ok(format!("{prefix}.{}", self.scale_suffix))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exclusions_end_at_module_boundaries_and_keep_source_scale_semantics() {
        let policy=BlockFp8Metadata::parse(&serde_json::json!({"quant_method":"fp8","activation_scheme":"dynamic","weight_block_size":[128,128],"ignored_layers":["model.layers.1","lm_head"]})).unwrap();
        assert!(!policy.includes("model.layers.1.mlp.gate_proj.weight"));
        assert!(policy.includes("model.layers.10.mlp.gate_proj.weight"));
        assert!(!policy.includes("lm_head.weight"));
        assert_eq!(
            policy
                .source_scale_name("model.layers.10.mlp.gate_proj.weight")
                .unwrap(),
            "model.layers.10.mlp.gate_proj.weight_scale_inv"
        );
        for malformed in [
            serde_json::json!({"quant_method":"fp8","activation_scheme":"static","weight_block_size":[128,128]}),
            serde_json::json!({"quant_method":"fp8","activation_scheme":"dynamic","weight_block_size":[64,128]}),
            serde_json::json!({"quant_method":"fp8","activation_scheme":"dynamic","weight_block_size":[128,128],"ignored_layers":[""]}),
        ] {
            assert!(BlockFp8Metadata::parse(&malformed).is_err());
        }
    }
}
