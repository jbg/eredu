//! Encoded input and fixed owner construction inside one cold affine invocation.
use super::*;
use crate::backend::runtime::checkpoint::store::{
    ColdMaterializationSlot, ColdMaterializationSlotError, EncodedInputConstructionError,
    MaterializationPayloadShape, PreparedEncodedInputPlan,
};
use crate::backend::submission_recovery::native_role::{NativeRoleCapacity, NativeRoleContext};
use eredu_runtime::working_memory::{SharedNativeInitializationFailure, WorkingMemoryPool};
use safemlx::{
    CpuAffineQuantizeSubmissionLayout, ImmutableHostTransferBuffer, PreparedInputRuntime,
};

/// Every constructor retains its actual prefix/account. Some slot prefixes own
/// thread-local recovery state, so this error is intentionally not erased into
/// the backend's Send + Sync error envelope.
#[derive(Debug, thiserror::Error)]
pub(super) enum ConstructionError {
    #[error("affine tile materialization: {0}")]
    Backend(#[from] Error),
    #[error("affine tile owner: {0}")]
    Slot(#[from] ColdMaterializationSlotError),
    #[error("affine tile input alias: {0}")]
    Alias(#[source] safemlx::PreparedInputCause),
    #[error("affine tile input: {0}")]
    Input(
        #[source]
        SharedNativeInitializationFailure<
            ImmutableHostTransferBuffer,
            EncodedInputConstructionError,
        >,
    ),
}

/// Failure retains its actual constructor or native invocation prefix. Native
/// recovery remains thread-local and is not erased into a Send error envelope.
#[derive(Debug, thiserror::Error)]
pub(super) enum TileError<I: 'static> {
    #[error("tile resources: {0}")]
    Resources(#[from] cpu_resources::ResourceError),
    #[error("tile pipeline: {0}")]
    Backend(#[from] Error),
    #[error("tile policy: {0}")]
    Policy(#[from] eredu_runtime::working_memory::WorkingMemoryError),
    #[error("tile native allocation layout: {0}")]
    Layout(#[source] safemlx::OriginalBufferCause),
    #[error("tile source read: {0}")]
    Read(#[from] eredu_runtime::working_memory::EncodedRecipeSourceError),
    #[error("tile source requires numerical or unsupported read construction")]
    ReadUnavailable,
    #[error("tile output metadata: {0}")]
    Metadata(
        #[source]
        SharedNativeInitializationFailure<metadata::Metadata, metadata::ConstructionError>,
    ),
    #[error("cold tile admission: {0}")]
    Admission(
        #[source]
        SharedNativeInitializationFailure<
            cold::Output<I, WeightMaterialization, ConstructionError>,
            eredu_core::BackendFailure,
        >,
    ),
    #[error("cold tile invocation: {0}")]
    Invocation(#[source] cold::FailedSubmission<I, ConstructionError>),
}
impl<I: 'static> From<eredu_checkpoint::recipe::RecipeError> for TileError<I> {
    fn from(cause: eredu_checkpoint::recipe::RecipeError) -> Self {
        Self::Backend(cause.into())
    }
}

/// Admitted host prerequisites for one CPU affine submission. Its compiled read
/// and native shape remain alive until the synchronous input constructor returns;
/// native aliases then retain their own input accounts through completion.
pub(super) struct Tile<'a> {
    pool: &'a WorkingMemoryPool,
    runtime: &'a PreparedInputRuntime,
    target: &'a BoundedQuantizationTarget,
    quantization: eredu_checkpoint::AffineQuantization,
    metadata: eredu_runtime::working_memory::InitializedSharedNative<metadata::Metadata>,
    read: eredu_checkpoint::recipe::EncodedRecipeRead<
        eredu_runtime::working_memory::CompiledRecipeCustody<
            eredu_runtime::working_memory::SharedNativeInitializationCustody,
        >,
    >,
    dtype: Dtype,
    layout: CpuAffineQuantizeSubmissionLayout,
    capacity: NativeRoleCapacity,
}
impl<'a> Tile<'a> {
    pub(super) fn prepare<I: 'static>(
        pool: &'a WorkingMemoryPool,
        runtime: &'a PreparedInputRuntime,
        source: &eredu_checkpoint::store::RetainedCheckpointSource,
        recipe: &DerivedWeightRecipe,
        target: &'a BoundedQuantizationTarget,
        quantization: WeightQuantization,
    ) -> Result<Self, TileError<I>> {
        use eredu_runtime::working_memory::WorkingMemoryError as W;
        let WeightQuantization::Affine(quantization) = quantization else {
            return Err(W::UnknownBound.into());
        };
        let metadata = metadata::MetadataPlan::new(recipe, source.as_ref())?
            .prepare(pool)
            .map_err(|error| TileError::Metadata(error.into_parts().1))?;
        metadata.validate_pool(pool)?;
        let read = pool
            .prepare_encoded_recipe(source, recipe)?
            .ok_or(TileError::ReadUnavailable)?;
        if read.output() != metadata.output().inferred() {
            return Err(W::IdentityMismatch.into());
        }
        let native_dtype = |dtype: &RecipeDtype| match dtype {
            RecipeDtype::F16 => Ok(Dtype::Float16),
            RecipeDtype::BF16 => Ok(Dtype::Bfloat16),
            RecipeDtype::F32 => Ok(Dtype::Float32),
            _ => Err(W::UnknownBound),
        };
        let dtype = native_dtype(read.output().dtype())?;
        let companion = native_dtype(&target.affine_companion_dtype)?;
        let shape = metadata.output().shape();
        let (&columns, leading) = shape.split_last().ok_or(W::UnknownBound)?;
        let rows = leading
            .iter()
            .try_fold(1usize, |rows, &dimension| {
                rows.checked_mul(usize::try_from(dimension).ok()?)
            })
            .ok_or(W::Overflow)?;
        let columns = usize::try_from(columns).map_err(|_| W::UnknownBound)?;
        let layout = safemlx::OperationEvent::cpu_affine_quantize_submission_layout(
            dtype,
            companion,
            shape.len(),
            rows,
            columns,
            quantization.group_size,
            quantization.bits,
        )
        .ok_or(W::UnknownBound)?;
        let capacity = NativeRoleCapacity {
            graph: layout.graph_capacity(),
            records: layout.record_capacity(),
            backing: layout
                .physical_capacity(runtime)
                .map_err(TileError::Layout)?,
        };
        Ok(Self {
            pool,
            runtime,
            target,
            quantization,
            metadata,
            read,
            dtype,
            layout,
            capacity,
        })
    }
    pub(super) fn source_bytes(&self) -> u64 {
        self.read.output().byte_len()
    }
    pub(super) fn input(
        &self,
    ) -> Result<
        PreparedEncodedInputPlan<
            '_,
            eredu_runtime::working_memory::CompiledRecipeCustody<
                eredu_runtime::working_memory::SharedNativeInitializationCustody,
            >,
        >,
        eredu_runtime::working_memory::WorkingMemoryError,
    > {
        PreparedEncodedInputPlan::new(
            &self.read,
            self.runtime,
            self.metadata.output().shape(),
            self.dtype,
        )
    }
    pub(super) fn plan<'b, I: 'static>(
        &'b self,
        invocation: I,
        stream: &'b Stream,
    ) -> Result<
        cold::Plan<
            'b,
            I,
            impl FnOnce(
                    &I,
                    &NativeRoleContext<'_>,
                )
                    -> Result<Result<WeightMaterialization, ConstructionError>, Error>
                + 'b,
        >,
        TileError<I>,
    > {
        let input = self.input()?;
        let pool = self.pool;
        let quantization = self.quantization;
        let target = self.target;
        let layout = self.layout;
        Ok(cold::Plan::new(
            self.runtime,
            self.capacity,
            None,
            invocation,
            move |_, context| {
                Ok((|| {
                    let mut slot = ColdMaterializationSlot::prepare(
                        pool,
                        MaterializationPayloadShape {
                            inputs: 1,
                            outputs: 3,
                            pending_sources: 0,
                        },
                    )?;
                    let ready = slot.take(pool).map_err(Error::PrefillControl)?;
                    let mut owner =
                        WeightMaterialization::prepare_original_slot(ready, context.observer())
                            .map_err(Error::from)?;
                    owner.prepare_input_capacity(1).map_err(Error::from)?;
                    let input = input.prepare(pool).map_err(|failure| {
                        let (uncalled, failure) = failure.into_parts();
                        // This plan only borrows prerequisites. No constructor ran if
                        // it is returned; the owned failure retains any actual prefix.
                        drop(uncalled);
                        ConstructionError::Input(failure)
                    })?;
                    let alias = input
                        .output()
                        .try_prepared_source_array()
                        .map_err(ConstructionError::Alias)?;
                    owner.retain_input(alias).map_err(Error::from)?;
                    drop(input);
                    submit_original_affine_tile(owner, quantization, target, stream, layout)
                        .map_err(ConstructionError::Backend)
                })())
            },
        ))
    }
}

/// Producer over the actual admitted runtime and stream identities. Source/header
/// birth, cold declarations and final destinations belong to preparation; the
/// shared driver admits its queue and cache controls.
pub(super) struct CpuEncodedAffineProducer<'a> {
    pub(super) pool: &'a WorkingMemoryPool,
    pub(super) resources: &'a cpu_resources::CpuTileResources,
}
impl TileProducer for CpuEncodedAffineProducer<'_> {
    type Completion = cold::Submission<(), WeightMaterialization, Infallible>;
    type Error = TileError<()>;
    fn submit(
        &mut self,
        source: &eredu_checkpoint::store::RetainedCheckpointSource,
        recipe: &DerivedWeightRecipe,
        target: &BoundedQuantizationTarget,
        quantization: WeightQuantization,
        slot: usize,
    ) -> Result<(Self::Completion, u64), Self::Error> {
        let tile = Tile::prepare(
            self.pool,
            self.resources.runtime(),
            source,
            recipe,
            target,
            quantization,
        )?;
        let source_bytes = tile.source_bytes();
        let streams = self.resources.streams();
        let stream = streams
            .get(slot)
            .ok_or(eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch)?;
        let submission = tile
            .plan((), stream)?
            .submit(self.pool)
            .map_err(|error| TileError::Admission(error.into_parts().1))?;
        submission
            .into_result()
            .map(|completion| (completion, source_bytes))
            .map_err(TileError::Invocation)
    }
}
