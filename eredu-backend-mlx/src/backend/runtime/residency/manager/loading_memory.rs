//! Cold payload accounting for the manager's exact retained source and bindings.

use super::*;
use eredu_runtime::memory_estimation::MemoryBytes;
use eredu_runtime::memory_forecast::ParameterLoadingMemoryProfile;

/// No payload acquisition, native allocation, source reopening or evaluation.
/// The ordinary recipe envelope also covers the direct-read optimization.
/// Sources with deferred converters need their own physical-group workspace contract.
pub(crate) fn parameter_loading_memory_profile(
    source: &dyn eredu_checkpoint::store::CheckpointSource,
    units: &[Vec<WeightBinding>],
    residency: eredu_runtime::LayerWeightResidency,
) -> Result<ParameterLoadingMemoryProfile, crate::backend::error::Error> {
    use crate::backend::runtime::checkpoint::{
        binding::binding_bytes, recipe::native_recipe_workspace,
    };
    let source_payload = match source.retained_payload_bytes_upper_bound() {
        Some(0) => MemoryBytes::exact(0),
        Some(bytes) => MemoryBytes::estimated(0, bytes, "retained checkpoint-source payload backing; shared child owners can be conservatively counted more than once"),
        None => MemoryBytes::unknown("retained checkpoint-source payload ownership unavailable"),
    };
    let unknown = |detail: &str| ParameterLoadingMemoryProfile {
        source_payload: source_payload.clone(),
        host_workspace: MemoryBytes::unknown(detail),
        execution_workspace: MemoryBytes::unknown(detail),
    };
    if !source.has_direct_encoded_leases() {
        return Ok(unknown(
            "source has no direct encoded-lease workspace contract",
        ));
    }
    if units.is_empty() {
        return Ok(unknown(
            "parameter loading has no retained atomic-unit bindings",
        ));
    }
    let mut workspaces = Vec::with_capacity(units.len());
    for bindings in units {
        let mut workspace = 0u64;
        for binding in bindings.iter().filter(|binding| !binding.is_alias()) {
            let direct;
            let recipe = match binding.recipe() {
                Some(recipe) => recipe,
                None => {
                    direct = binding.source_recipe();
                    &direct
                }
            };
            let bytes = native_recipe_workspace(recipe, source)
                .map_err(crate::backend::error::Error::ArchitectureModel)?;
            workspace = workspace.checked_add(bytes).ok_or_else(|| {
                crate::backend::error::Error::ArchitectureModel(
                    "native loading workspace overflow".into(),
                )
            })?;
        }
        workspaces.push((
            binding_bytes(bindings)?,
            host_capacity_upper_bound_for_bindings(bindings)?,
            workspace,
            workspace,
        ));
    }
    // Each envelope includes the complete source/read/recipe graph, regardless
    // of which pool a particular intermediary uses. This deliberately allows
    // duplication across pools. This query materializes no parameter payload.
    ParameterLoadingMemoryProfile::from_unit_workspaces(workspaces, residency, source_payload)
        .map_err(|error| crate::backend::error::Error::ArchitectureModel(error.to_string()))
}
