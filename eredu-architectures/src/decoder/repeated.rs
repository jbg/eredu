//! Portable lowering of shared-weight stacks to logical decoder invocations.
//!
//! Checkpoint recipes retain the single physical source of each repeated weight.
//! Logical execution units have independent residency and state ownership, so
//! ordinary decoder drivers handle pipeline cuts, tensor sharding and offloading.

use eredu_checkpoint::{
    recipe::DerivedWeightRecipe,
    store::{CheckpointSource, TensorSelection},
};
use eredu_core::{AttentionPolicy, LayerSchedule};
use std::collections::BTreeMap;

/// Expands the physical attention schedule in forward execution order.
pub fn attention_schedule(
    physical: &LayerSchedule<AttentionPolicy>,
    repetitions: usize,
) -> Result<LayerSchedule<AttentionPolicy>, String> {
    let count = physical
        .len()
        .checked_mul(repetitions)
        .filter(|n| *n > 0 && *n <= i32::MAX as usize)
        .ok_or("invalid repeated decoder depth")?;
    LayerSchedule::new(
        count,
        (0..repetitions)
            .flat_map(|_| physical.iter().cloned())
            .collect(),
    )
    .map_err(|e| e.to_string())
}

/// Maps a logical block parameter back to its shared physical checkpoint name.
pub fn source_name(root: &str, physical_layers: usize, name: &str) -> String {
    source_name_view(root, physical_layers, name).to_string()
}

pub(crate) enum RepeatedParameterName<'a> {
    Original(&'a str),
    FinalNorm(&'a str),
    Layer {
        root: &'a str,
        layer: usize,
        suffix: &'a str,
    },
}
impl std::fmt::Display for RepeatedParameterName<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Original(name) => f.write_str(name),
            Self::FinalNorm(root) => write!(f, "{root}.norm.weight"),
            Self::Layer {
                root,
                layer,
                suffix,
            } => write!(f, "{root}.layers.{layer}.{suffix}"),
        }
    }
}
pub(crate) fn source_name_view<'a>(
    root: &'a str,
    physical_layers: usize,
    name: &'a str,
) -> RepeatedParameterName<'a> {
    if let Some((layer, suffix)) = name
        .strip_prefix(root)
        .and_then(|s| s.strip_prefix(".layers."))
        .and_then(|s| s.split_once('.'))
    {
        if let Ok(layer) = layer.parse::<usize>() {
            if suffix == "output_norm.weight" {
                return RepeatedParameterName::FinalNorm(root);
            }
            return RepeatedParameterName::Layer {
                root,
                layer: layer % physical_layers,
                suffix,
            };
        }
    }
    RepeatedParameterName::Original(name)
}

/// Exact aliases for every invocation, including the normalization between passes.
/// Self aliases keep shared originals in the executable catalog when another
/// invocation consumes them through a recipe.
pub fn recipes(
    source: &dyn CheckpointSource,
    root: &str,
    physical_layers: usize,
    repetitions: usize,
    normalize_between: bool,
) -> Result<BTreeMap<String, DerivedWeightRecipe>, String> {
    physical_layers
        .checked_mul(repetitions)
        .filter(|&count| physical_layers > 0 && repetitions > 0 && count <= i32::MAX as usize)
        .ok_or("invalid repeated decoder depth")?;
    if repetitions == 1 {
        return Ok(BTreeMap::new());
    }
    let prefix = format!("{root}.layers.");
    let mut result = BTreeMap::new();
    for key in source.source_keys() {
        let Some((layer, suffix)) = key.strip_prefix(&prefix).and_then(|s| s.split_once('.'))
        else {
            continue;
        };
        let Ok(layer) = layer.parse::<usize>() else {
            continue;
        };
        if layer >= physical_layers {
            return Err(format!("physical block {layer} exceeds {physical_layers}"));
        }
        for pass in 0..repetitions {
            result.insert(
                format!("{prefix}{}.{suffix}", pass * physical_layers + layer),
                DerivedWeightRecipe::source(&key, TensorSelection::Full),
            );
        }
    }
    if normalize_between && repetitions > 1 {
        let norm = format!("{root}.norm.weight");
        result.insert(
            norm.clone(),
            DerivedWeightRecipe::source(&norm, TensorSelection::Full),
        );
        for pass in 1..repetitions {
            result.insert(
                format!("{prefix}{}.output_norm.weight", pass * physical_layers - 1),
                DerivedWeightRecipe::source(&norm, TensorSelection::Full),
            );
        }
    }
    Ok(result)
}
