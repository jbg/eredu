//! Final native input backing filled directly from a retained encoded read.
use eredu_checkpoint::{
    recipe::EncodedRecipeRead,
    store::{EncodedReadFailure, EncodedReadLayout},
};
use eredu_runtime::working_memory::{
    InitializedSharedNative, MemoryLedger, SharedNativeInitializationCustody,
    SharedNativeInitializationError, SharedNativeInitializer, WorkingMemoryError,
};
use safemlx::{
    Dtype, ImmutableHostTransferBuffer, PreparedHostTransferPlan, PreparedInputArena,
    PreparedInputCause, PreparedInputRuntime, PreparedSubmissionGraphQuota,
    SubmissionGraphQuotaCause,
};
use std::mem::{size_of, size_of_val};

/// Borrows the selected read and runtime; neither source metadata nor runtime
/// birth is covered here. The caller retains and funds those prerequisites.
/// The native source, its one array alias slot and synchronous read scratch are
/// admitted together before allocating a destination or performing payload I/O.
pub(crate) struct PreparedEncodedInputPlan<'a, C> {
    read: &'a EncodedRecipeRead<C>,
    native: PreparedHostTransferPlan<'a>,
    scratch: EncodedReadLayout,
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum EncodedInputConstructionError {
    #[error("encoded input arena: {0}")]
    Arena(#[source] SubmissionGraphQuotaCause),
    #[error("encoded input destination: {0}")]
    Native(#[source] PreparedInputCause),
    #[error("encoded input byte access: {0}")]
    Bytes(#[source] safemlx::error::Exception),
    #[error("encoded input payload: {0}")]
    Read(#[source] EncodedReadFailure),
}

impl<'a, C> PreparedEncodedInputPlan<'a, C> {
    pub(crate) fn new(
        read: &'a EncodedRecipeRead<C>,
        runtime: &'a PreparedInputRuntime,
        shape: &'a [i32],
        dtype: Dtype,
    ) -> Result<Self, WorkingMemoryError> {
        let output = read.output();
        if output.shape().len() != shape.len()
            || !output
                .shape()
                .iter()
                .zip(shape)
                .all(|(&expected, &actual)| usize::try_from(actual).ok() == Some(expected))
            || *output.dtype() != super::super::recipe::recipe_dtype_from_mlx(dtype)
        {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        let native = PreparedHostTransferPlan::new(runtime, shape, dtype, 1)
            .map_err(|_| WorkingMemoryError::UnknownBound)?;
        if u64::try_from(native.logical_bytes()).ok() != Some(output.byte_len()) {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        let scratch = EncodedRecipeRead::borrowed_read_layout(std::iter::once(read))
            .ok_or(WorkingMemoryError::UnknownBound)?;
        Ok(Self {
            read,
            native,
            scratch,
        })
    }

    /// Includes this producer's original account and constructor-result controls.
    pub(crate) fn required_bytes(&self) -> Result<u64, WorkingMemoryError> {
        MemoryLedger::shared_native_initialization_required_bytes(self)
    }

    /// A rejected comparison retains this uncalled borrowed plan. Native/source
    /// failures retain the original account through the existing error owner.
    /// Successful array aliases independently retain the same source account.
    pub(crate) fn prepare(
        self,
        pool: &MemoryLedger,
    ) -> Result<
        InitializedSharedNative<ImmutableHostTransferBuffer>,
        SharedNativeInitializationError<Self>,
    > {
        pool.initialize_shared_native(self)
    }
}

impl<C> SharedNativeInitializer for PreparedEncodedInputPlan<'_, C> {
    type Output = ImmutableHostTransferBuffer;
    type Error = EncodedInputConstructionError;

    fn required_storage_bytes(&self) -> Result<usize, WorkingMemoryError> {
        let arena = PreparedInputArena::layout::<SharedNativeInitializationCustody>(
            self.native.metadata_bytes(),
        )
        .map_err(|_| WorkingMemoryError::UnknownBound)?;
        let controls = [
            self.scratch.required_bytes(),
            arena.total_bytes().ok_or(WorkingMemoryError::Overflow)?,
            self.native.backing_bytes(),
            self.native
                .control_bytes()
                .ok_or(WorkingMemoryError::Overflow)?,
            size_of::<[&mut [u8]; 1]>(),
            size_of::<std::iter::Once<&EncodedRecipeRead<C>>>(),
            size_of::<EncodedInputConstructionError>(),
        ];
        controls
            .into_iter()
            .try_fold(size_of_val(&controls), usize::checked_add)
            .ok_or(WorkingMemoryError::Overflow)
    }

    fn initialize(
        self,
        custody: SharedNativeInitializationCustody,
    ) -> Result<Self::Output, Self::Error> {
        let quota = PreparedSubmissionGraphQuota::try_new(self.native.metadata_bytes(), custody)
            .map_err(|error| EncodedInputConstructionError::Arena(error.cause()))?;
        let arena = PreparedInputArena::try_allocate(quota)
            .map_err(|error| EncodedInputConstructionError::Arena(error.cause()))?;
        let mut buffer = self
            .native
            .construct(&arena)
            .map_err(EncodedInputConstructionError::Native)?;
        EncodedRecipeRead::read_many_borrowed_into(
            std::iter::once(self.read),
            &mut [buffer
                .as_bytes_mut()
                .map_err(EncodedInputConstructionError::Bytes)?],
        )
        .map_err(EncodedInputConstructionError::Read)?;
        // No array alias or partially initialized destination is published on
        // failure. The immutable source and its aliases retain the arena.
        Ok(buffer.freeze())
    }
}

#[cfg(test)]
mod tests;
