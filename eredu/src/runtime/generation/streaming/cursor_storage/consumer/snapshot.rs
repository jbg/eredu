//! Host portion of the same retained cursor; native snapshot joining is separate.
use super::*;
use eredu_runtime::{
    execution_control::{PreparedTextHostCopy, TextHostCopyError},
    working_memory::{WorkingMemoryError, WorkingMemoryPool},
};

struct CursorHostCopy<P, S, D, const O: bool, const PLAIN: bool, const TEXT: bool> {
    provider: P,
    layout: GenerationSequenceConsumerLayout,
    finish_reason: Option<FinishReason>,
    failed: bool,
    types: PhantomData<fn() -> (S, D)>,
}
impl<P, S, D, const O: bool, const PLAIN: bool, const TEXT: bool> PreparedTextHostCopy
    for CursorHostCopy<P, S, D, O, PLAIN, TEXT>
where
    P: PreparedTextHostCopy<Copied = RetainedGenerationSequence>,
{
    type Copied = RetainedConsumerCursor<S, D, O, PLAIN, TEXT>;
    fn storage_bytes(&self) -> Option<u64> {
        self.provider.storage_bytes()
    }
    fn original_control_bytes(&self) -> Option<usize> {
        self.provider.original_control_bytes()
    }
    fn original_preparation_bytes(&self) -> Option<u64> {
        self.provider.original_preparation_bytes()
    }

    fn copy_original(
        self,
        retained_bytes: u64,
    ) -> Result<(Self::Copied, eredu_core::HostPreparationAuthority), TextHostCopyError> {
        let (sequence, custody) = self.provider.copy_original(retained_bytes)?;
        Ok((
            RetainedConsumerCursor {
                layout: self.layout,
                types: PhantomData,
                cursor: Cursor {
                    sequence: Some(sequence),
                    finish_reason: self.finish_reason,
                    failed: self.failed,
                },
            },
            custody,
        ))
    }
    fn copy_original_resume(
        self,
        retained_bytes: u64,
        reservation: eredu_runtime::execution_control::PendingSnapshotResumeRetention,
    ) -> Result<(Self::Copied, eredu_core::HostPreparationAuthority), TextHostCopyError> {
        let (sequence, custody) = self
            .provider
            .copy_original_resume(retained_bytes, reservation)?;
        Ok((
            RetainedConsumerCursor {
                layout: self.layout,
                types: PhantomData,
                cursor: Cursor {
                    sequence: Some(sequence),
                    finish_reason: self.finish_reason,
                    failed: self.failed,
                },
            },
            custody,
        ))
    }
    fn copy(self, retained_bytes: u64) -> Result<Self::Copied, TextHostCopyError> {
        let sequence = self.provider.copy(retained_bytes)?;
        Ok(RetainedConsumerCursor {
            layout: self.layout,
            types: PhantomData,
            cursor: Cursor {
                sequence: Some(sequence),
                finish_reason: self.finish_reason,
                failed: self.failed,
            },
        })
    }
}
impl<
    S: Error + Send + Sync + 'static,
    D: Error + Send + Sync + 'static,
    const O: bool,
    const PLAIN: bool,
    const TEXT: bool,
> RetainedConsumerCursor<S, D, O, PLAIN, TEXT>
{
    /// Prepares this same cursor provider with the shared snapshot controls and
    /// the facade's concrete enclosing result/error representations.
    pub(crate) fn prepare_snapshot_host_copy<'a, B, C, E>(
        &'a self,
        pool: &WorkingMemoryPool,
        capacity: u64,
        native_preparation_bytes: u64,
    ) -> Result<impl PreparedTextHostCopy<Copied = Self> + 'a, WorkingMemoryError>
    where
        B: eredu_runtime::execution_control::TextSnapshotBackend,
        C: eredu_runtime::execution_control::SnapshotTokenController,
    {
        let sequence = self
            .cursor
            .sequence
            .as_ref()
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        let provider = pool.prepare_generation_snapshot_host_copy::<Self, E, B, C>(
            sequence,
            capacity,
            native_preparation_bytes,
        )?;
        Ok(CursorHostCopy::<_, S, D, O, PLAIN, TEXT> {
            provider,
            layout: self.layout,
            finish_reason: self.cursor.finish_reason,
            failed: self.cursor.failed,
            types: PhantomData,
        })
    }

    pub(crate) fn prepare_resume_host_copy<'a, B, C, E>(
        &'a self, pool: &WorkingMemoryPool, capacity: u64, native_preparation_bytes: u64,
    ) -> Result<impl PreparedTextHostCopy<Copied = Self> + 'a, WorkingMemoryError>
    where
        B: eredu_runtime::execution_control::TextSnapshotBackend + eredu_core::TextResumeBackend<ResumeSource = <B as eredu_runtime::execution_control::TextSnapshotBackend>::SavedTextComponents>,
        C: eredu_runtime::execution_control::SnapshotTokenController,
    {
        let sequence = self
            .cursor
            .sequence
            .as_ref()
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        let provider = pool.prepare_generation_resume_host_copy::<Self, E, B, C>(
            sequence,
            capacity,
            native_preparation_bytes,
        )?;
        Ok(CursorHostCopy::<_, S, D, O, PLAIN, TEXT> {
            provider,
            layout: self.layout,
            finish_reason: self.cursor.finish_reason,
            failed: self.cursor.failed,
            types: PhantomData,
        })
    }

    /// A resumed backend may have a smaller output allowance than its saved
    /// source. Keep the paid provider intact and shorten only token termination.
    pub(crate) fn restrict_remaining(&mut self, remaining: usize) {
        let sequence = self.cursor.sequence.as_mut().expect("copied live sequence");
        sequence.restrict_remaining(remaining);
        self.cursor.finish_reason = sequence.finish_reason();
    }

    /// The actual prepared-host-copy provider for a managed snapshot boundary.
    /// This cannot advance/restore native state and is never a complete snapshot.
    /// Caller must pass it through the shared nonrefundable snapshot transaction.
    pub(crate) fn prepare_host_copy<'a>(
        &'a self,
        pool: &WorkingMemoryPool,
        capacity: u64,
    ) -> Result<impl PreparedTextHostCopy<Copied = Self> + 'a, WorkingMemoryError> {
        let sequence = self
            .cursor
            .sequence
            .as_ref()
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        let provider =
            pool.prepare_generation_host_copy::<Self, TextHostCopyError>(sequence, capacity)?;
        Ok(CursorHostCopy::<_, S, D, O, PLAIN, TEXT> {
            provider,
            layout: self.layout,
            finish_reason: self.cursor.finish_reason,
            failed: self.cursor.failed,
            types: PhantomData,
        })
    }
}
