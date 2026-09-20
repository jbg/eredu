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

/// The caller owns and funds runtime, streams and retained read metadata. The
/// selected layout supplies native capacity; inputs and fixed completion slots
/// are admitted from the same pool inside the invocation. No output or failed
/// prefix is detached from its owner when the callback returns.
#[allow(clippy::too_many_arguments)]
pub(super) fn plan<'a, I: 'static>(
    pool: &'a WorkingMemoryPool,
    runtime: &'a PreparedInputRuntime,
    capacity: NativeRoleCapacity,
    invocation: I,
    input: PreparedEncodedInputPlan<'a>,
    quantization: eredu_checkpoint::AffineQuantization,
    target: &'a BoundedQuantizationTarget,
    stream: &'a Stream,
    layout: CpuAffineQuantizeSubmissionLayout,
) -> cold::Plan<
    'a,
    I,
    impl FnOnce(
            &I,
            &NativeRoleContext<'_>,
        ) -> Result<Result<WeightMaterialization, ConstructionError>, Error>
        + 'a,
> {
    cold::Plan::new(runtime, capacity, None, invocation, move |_, context| {
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
            let mut owner = WeightMaterialization::prepare_original_slot(ready, context.observer())
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
    })
}
