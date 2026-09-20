//! Host metadata admission around the shared CPU conversion worker.
use super::*;
use eredu_checkpoint::{
    recipe::RecipeSourceVisitor,
    store::{MetadataCloneLayout, RetainedCheckpointSource},
};
use eredu_runtime::working_memory::{
    DependencyMemoryPolicy, SharedNativeInitializationCustody, SharedNativeInitializationFailure,
    SharedNativeInitializer, WorkingMemoryError, WorkingMemoryPool,
};
use std::{cell::Cell, mem::size_of};

/// Borrows declarations until admission. Source/header construction and the
/// native resource owner have their own independent admission contracts.
pub(super) struct ColdConversion<'a> {
    pub(super) source: &'a RetainedCheckpointSource,
    pub(super) plan: &'a BoundedQuantizationPlan,
    pub(super) pool: &'a WorkingMemoryPool,
    pub(super) resources: &'a cpu_resources::CpuTileResources,
    pub(super) metadata_policy: DependencyMemoryPolicy,
}

#[derive(Debug, thiserror::Error)]
pub(super) enum ConstructionError {
    #[error("cold conversion resources: {0}")]
    Resources(#[source] cpu_resources::ResourceError),
    #[error("cold conversion preparation: {0}")]
    Preparation(#[from] Error),
    #[error("cold conversion: {0}")]
    Conversion(#[from] PipelineAdmissionError<encoded_affine::TileError<()>>),
}

struct MetadataInputs<'a> {
    source: &'a RetainedCheckpointSource,
    bytes: usize,
}
impl RecipeSourceVisitor for MetadataInputs<'_> {
    type Error = WorkingMemoryError;
    fn source(&mut self, key: &str, _: &TensorSelection) -> Result<(), Self::Error> {
        let metadata = self
            .source
            .source_metadata_borrowed(key)
            .map_err(|_| WorkingMemoryError::UnknownBound)?;
        let layout = MetadataCloneLayout::of(metadata).ok_or(WorkingMemoryError::Overflow)?;
        self.bytes = self
            .bytes
            .checked_add(layout.value.size())
            .and_then(|n| n.checked_add(layout.payload_bytes()?))
            .ok_or(WorkingMemoryError::Overflow)?;
        Ok(())
    }
}

impl ColdConversion<'_> {
    pub(super) fn prepare(
        self,
    ) -> Result<
        ConvertedQuantization,
        SharedNativeInitializationFailure<Cell<Option<ConvertedQuantization>>, ConstructionError>,
    > {
        let initialized = self
            .pool
            .initialize_shared_native(self)
            .map_err(|error| error.into_parts().1)?;
        // This private output independently retains its account through both
        // the plan and source root. The generic initialized owner stays borrowed.
        Ok(initialized
            .output()
            .take()
            .expect("completed cold conversion"))
    }
}
impl SharedNativeInitializer for ColdConversion<'_> {
    type Output = Cell<Option<ConvertedQuantization>>;
    type Error = ConstructionError;

    fn required_storage_bytes(&self) -> Result<usize, WorkingMemoryError> {
        let mut inputs = MetadataInputs {
            source: self.source,
            bytes: self
                .source
                .materialization_input_bytes()
                .ok_or(WorkingMemoryError::UnknownBound)?,
        };
        for target in self.plan.targets() {
            inputs.bytes = inputs
                .bytes
                .checked_add(size_of::<BoundedQuantizationTarget>())
                .and_then(|n| n.checked_add(target.weight_name().len()))
                .and_then(|n| n.checked_add(target.scales_name().len()))
                .and_then(|n| n.checked_add(target.biases_name().map_or(0, str::len)))
                .and_then(|n| n.checked_add(target.source().metadata_input_bytes()?))
                .ok_or(WorkingMemoryError::Overflow)?;
            target.source().visit_sources(&mut inputs)?;
        }
        // Estimate cold declarations, inference/collision scratch, provenance,
        // retained plan and outer sharing. Output-buffer publication headroom
        // is already reserved by allocate_memory_tensor_buffer; do not repeat it.
        self.metadata_policy
            .estimate(inputs.bytes)
            .and_then(|n| n.checked_add(size_of::<ConvertedQuantization>()))
            .and_then(|n| n.checked_add(size_of::<ConstructionError>()))
            .ok_or(WorkingMemoryError::Overflow)
    }

    fn initialize(
        self,
        custody: SharedNativeInitializationCustody,
    ) -> Result<Self::Output, Self::Error> {
        self.resources
            .validate_pool(self.pool)
            .map_err(ConstructionError::Resources)?;
        let custody = Arc::new(custody);
        let prepared = super::super::super::preparation::ColdQuantization::prepare(
            self.source.clone(),
            self.plan.clone(),
        )?
        .allocate_original(
            self.pool,
            self.metadata_policy,
            self.resources.streams()[0],
        )?;
        let (store, plan) = prepared.materialize_cpu_encoded(self.pool, self.resources)?;
        Ok(Cell::new(Some(ConvertedQuantization::new(
            self.source.clone(),
            plan,
            store,
            Some(custody),
        ))))
    }
}
