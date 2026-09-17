//! Exact typed handoff retained by the real external assistant materializer.
//! These are existing cold construction values, moved after successful binding.
//! They supply provenance and borrowed parameters, never original execution fit.
use super::*;
use eredu_architectures::{
    ExternalAssistantArchitecture, external_assistant::ExternalAssistantCheckpoint,
};
use eredu_checkpoint::store::RetainedCheckpointSource;
use eredu_core::artifact::DeferredArtifactIdentity;
use eredu_runtime::{ReplicatedTextMaterializationTask, WeightBinding};
use std::collections::BTreeMap;

pub(crate) struct MlxExternalAssistantSource<A: ExternalAssistantArchitecture> {
    // Actual materialized values precede their source/configuration retention.
    values: BTreeMap<String, Array>,
    bindings: Vec<WeightBinding>,
    tasks: Vec<ReplicatedTextMaterializationTask>,
    source_config: A::Config,
    checkpoint: ExternalAssistantCheckpoint,
    artifact_identity: DeferredArtifactIdentity,
    store: RetainedCheckpointSource,
    // Release actual values and source aliases before their shared accounting.
    // The registration also retains physical roots through ordinary retirement.
    _registered_storage: Option<crate::backend::runtime::residency::storage::RetainedStorageReservation>,
}
impl<A: ExternalAssistantArchitecture> MlxExternalAssistantSource<A> {
    pub(super) fn new(
        store: RetainedCheckpointSource,
        checkpoint: ExternalAssistantCheckpoint,
        artifact_identity: DeferredArtifactIdentity,
        source_config: A::Config,
        tasks: Vec<ReplicatedTextMaterializationTask>,
        bindings: Vec<WeightBinding>,
        values: BTreeMap<String, Array>,
        registered_storage: Option<crate::backend::runtime::residency::storage::RetainedStorageReservation>,
    ) -> Self {
        Self {
            values,
            bindings,
            tasks,
            source_config,
            checkpoint,
            artifact_identity,
            store,
            _registered_storage: registered_storage,
        }
    }
    pub(crate) fn store(&self) -> &dyn eredu_checkpoint::store::CheckpointSource {
        self.store.as_ref()
    }
    pub(crate) fn checkpoint(&self) -> &ExternalAssistantCheckpoint {
        &self.checkpoint
    }
    pub(crate) fn artifact_identity(&self) -> &DeferredArtifactIdentity {
        &self.artifact_identity
    }
    pub(crate) fn source_config(&self) -> &A::Config {
        &self.source_config
    }
    pub(crate) fn tasks(&self) -> &[ReplicatedTextMaterializationTask] {
        &self.tasks
    }
    pub(crate) fn bindings(&self) -> &[WeightBinding] {
        &self.bindings
    }
    pub(crate) fn values(&self) -> &BTreeMap<String, Array> {
        &self.values
    }
}
