//! Reuse the existing validator's success only for its exact immutable source.

use super::{
    config_source, validate_store_handoff, ReplicatedTextConstructionSource,
    ReplicatedTextDispatchError, SelectedReplicatedTextRealization,
};
use eredu_checkpoint::store::RetainedCheckpointSource;
use eredu_nn::{
    workspace::{WorkspaceContext, WorkspaceMetadataError},
    NeuralBackend, Tensor,
};

pub(crate) fn validate<B: NeuralBackend, E>(
    source: &impl ReplicatedTextConstructionSource,
    selected: &SelectedReplicatedTextRealization,
    store: &RetainedCheckpointSource,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<(), ReplicatedTextDispatchError<E>> {
    let metadata = B::construction_metadata(context);
    let prepared = source.prepared_sources();
    if prepared.is_some_and(|prepared| !config_source::exact_selection(prepared, selected)) {
        return Err(ReplicatedTextDispatchError::Metadata(
            WorkspaceMetadataError::Unqualified.into(),
        ));
    }
    // `recipe_cache` already promises immutable catalog metadata/provenance and
    // checks every lease/read against that catalog. We retain no cache entry,
    // infer no recipes here, and call no allocating ordinary metadata adapter.
    let immutable_target = prepared
        .filter(|prepared| prepared.target().same_source(store) && store.recipe_cache().is_some());
    if let Some(metadata) = metadata {
        let controls = [
            2 * std::mem::size_of::<Option<&crate::prepared_sources::PreparedModelSources>>(),
            std::mem::size_of::<Option<&WorkspaceContext>>(),
            std::mem::size_of::<Option<&eredu_checkpoint::recipe::RecipeInferenceCache>>(),
            std::mem::size_of::<Result<(), ReplicatedTextDispatchError<E>>>(),
            std::mem::size_of::<Result<(), ()>>(),
            std::mem::size_of::<Option<&()>>(),
            std::mem::size_of::<bool>(),
        ];
        let bytes = controls
            .into_iter()
            .try_fold(std::mem::size_of_val(&controls), usize::checked_add)
            .ok_or_else(|| {
                ReplicatedTextDispatchError::Metadata(WorkspaceMetadataError::Overflow.into())
            })?;
        metadata
            .charge_metadata(bytes)
            .map_err(|cause| ReplicatedTextDispatchError::Metadata(cause.into()))?;
    }
    if immutable_target.is_some_and(|prepared| {
        prepared
            .construction_semantics()
            .store_handoff
            .get()
            .is_some()
    }) {
        return Ok(());
    }
    // This is an exact missing-source-validation witness. Request-funded
    // reconstruction cannot create a replacement witness using the allocating
    // ordinary validator. Standalone/ordinary callers keep the same validator.
    if metadata.is_some_and(|metadata| metadata.uses_checked_metadata()) {
        return Err(ReplicatedTextDispatchError::Metadata(
            WorkspaceMetadataError::Unqualified.into(),
        ));
    }
    validate_store_handoff(selected.requirements(), store.as_ref())
        .map_err(ReplicatedTextDispatchError::Architecture)?;
    if let Some(prepared) = immutable_target {
        // Publish success only. The retained prepared graph keeps this exact
        // target alive; failed validation never changes this one-time slot.
        let _ = prepared.construction_semantics().store_handoff.set(());
    }
    Ok(())
}
