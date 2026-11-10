//! Portable lowering of shared-weight stacks to logical decoder invocations.
//!
//! Checkpoint recipes retain the single physical source of each repeated weight.
//! Logical invocations retain independent state while declared canonical owners
//! determine residency, pipeline replicas and materialization.

use eredu_checkpoint::{recipe::DerivedWeightRecipe, store::TensorSelection};
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
    let prefix = format!("{root}.layers.");
    if let Some((layer, suffix)) = name.strip_prefix(&prefix).and_then(|s| s.split_once('.')) {
        if let Ok(layer) = layer.parse::<usize>() {
            if suffix == "output_norm.weight" {
                return format!("{root}.norm.weight");
            }
            return format!("{prefix}{}.{suffix}", layer % physical_layers);
        }
    }
    name.to_owned()
}

/// Exact recipes for canonical block owners, including normalization between passes.
/// Self aliases keep shared originals in the executable catalog when another
/// invocation consumes them through a recipe.
pub fn recipes(
    parameters: &eredu_runtime::ArchitectureParameterDescription,
) -> BTreeMap<String, DerivedWeightRecipe> {
    let mut result = BTreeMap::new();
    if parameters.invocations().logical_addresses().len()
        == parameters.invocations().storage_layout().len()
    {
        return result;
    }
    for group in parameters.groups() {
        if !matches!(
            group.owner(),
            eredu_runtime::ParameterGroupOwner::ExecutionUnit { .. }
        ) {
            continue;
        }
        for member in group.members() {
            let source = member.alias_of().map_or(member.target(), |id| id.as_str());
            result.insert(
                member.target().to_owned(),
                DerivedWeightRecipe::source(source, TensorSelection::Full),
            );
            // Keep each shared original visible when an inter-pass normalization aliases it.
            if member.alias_of().is_some() {
                result.insert(
                    source.to_owned(),
                    DerivedWeightRecipe::source(source, TensorSelection::Full),
                );
            }
        }
    }
    result
}
