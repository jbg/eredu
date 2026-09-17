//! Once-only creator-thread freeze; the I/O output keeps every failed prefix.
use super::*;
type BodyLoan<'a> = MutexGuard<'a, Option<ReadBody>>;

#[derive(Debug, thiserror::Error)]
pub(super) enum FinishCause {
    #[error("prepared disk read completion or creator-thread source differs")]
    Identity,
    #[error("prepared disk read output is already borrowed")]
    Busy,
    #[error("prepared disk read output is poisoned")]
    Poisoned,
    #[error("prepared disk read failed; its output retains the exact cause")]
    SettledRead,
    #[error(transparent)]
    Read(DiskReadFailure),
    #[error(transparent)]
    Input(safemlx::PreparedInputCause),
    #[error(transparent)]
    Freeze(filled_host::SourceCause),
    #[error(transparent)]
    Publication(filled_host::SourceError),
    #[error(transparent)]
    Descriptor(safemlx::HostTransferMetadataError),
}
/// The actual buffers and completed file source retire before pin/occupancy/H.
/// Canonical installation still needs its exact pending-read transition; this
/// completed producer grants no manager commit or native promotion by itself.
pub(crate) struct CompletedDiskRead {
    pub(super) body: ReadBody,
}
/// Shared completion and optional moved-out native prefix precede their H.
/// Mutex safely transports existing Send-only writers, without unsafe traits.
pub(crate) struct DiskReadFinishFailure {
    cause: FinishCause,
    retained: Option<Mutex<ReadBody>>,
    output: PreparedDiskReadOutput,
    funding: WorkspaceMetadataFunding,
}
impl std::fmt::Debug for DiskReadFinishFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DiskReadFinishFailure")
            .field("cause", &self.cause)
            .field("retains_prefix", &self.retained.is_some())
            .finish_non_exhaustive()
    }
}
impl std::fmt::Display for DiskReadFinishFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.cause, f)
    }
}
impl std::error::Error for DiskReadFinishFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        if matches!(self.cause, FinishCause::SettledRead) {
            if let Some(Err(cause)) = self.output.result() {
                return Some(cause);
            }
        }
        Some(&self.cause)
    }
}
impl DiskReadFinishFailure {
    pub(super) fn retain(
        cause: FinishCause,
        body: ReadBody,
        output: PreparedDiskReadOutput,
    ) -> Self {
        let funding = output.funding.clone();
        Self {
            cause,
            retained: Some(Mutex::new(body)),
            output,
            funding,
        }
    }
    fn before_take(cause: FinishCause, output: &PreparedDiskReadOutput) -> Self {
        Self {
            cause,
            retained: None,
            output: output.clone(),
            funding: output.funding.clone(),
        }
    }
}
impl PreparedDiskReadOutput {
    pub(crate) fn result(&self) -> Option<Result<(), &DiskReadFailure>> {
        self.inner
            .get()
            .map(|completion| completion.result.as_ref().map(|()| ()))
    }
    /// Runs only after the existing task has settled. A second call cannot
    /// recreate native publication or consume another source-bank receipt.
    pub(crate) fn finish(&self) -> Result<CompletedDiskRead, DiskReadFinishFailure> {
        let failed = |cause| DiskReadFinishFailure::before_take(cause, self);
        let completion = self
            .inner
            .get()
            .ok_or_else(|| failed(FinishCause::Identity))?;
        let mut body = {
            let mut slot = completion.body.try_lock().map_err(|cause| {
                failed(match cause {
                    TryLockError::WouldBlock => FinishCause::Busy,
                    TryLockError::Poisoned(_) => FinishCause::Poisoned,
                })
            })?;
            slot.take().ok_or_else(|| failed(FinishCause::Identity))?
        };
        if completion.result.is_err() {
            return Err(DiskReadFinishFailure::retain(
                FinishCause::SettledRead,
                body,
                self.clone(),
            ));
        }
        match body.finish() {
            Ok(()) => Ok(CompletedDiskRead { body }),
            Err(cause) => Err(DiskReadFinishFailure::retain(cause, body, self.clone())),
        }
    }
}
impl ReadBody {
    fn finish(&mut self) -> Result<(), FinishCause> {
        for index in 0..2 {
            self.filling[index]
                .as_mut()
                .ok_or(FinishCause::Identity)?
                .completed_mut()
                .freeze()
                .map_err(FinishCause::Freeze)?;
            let (buffer, allocation) =
                filled_host::finish(self.filling[index].take().expect("frozen destination"))
                    .map_err(FinishCause::Publication)?;
            // Two exact immutable Arc shells were priced before native entry.
            self.ready[index] = Some(Arc::new(buffer));
            let descriptor = self.ready[index]
                .as_ref()
                .expect("retained immutable destination")
                .try_fixed_descriptor::<4>()
                .map_err(FinishCause::Descriptor)?;
            if descriptor.shape() != self.shapes[index]
                || descriptor.dtype() != self.dtypes[index]
                || descriptor.allocation() != allocation
                || allocation.bytes() != self.capacities[index]
                || descriptor.policy() != HostTransferPolicy::Transfer
                || descriptor.storage_kind() != safemlx::HostTransferStorageKind::MetalShared
            {
                return Err(FinishCause::Identity);
            }
        }
        let first = self.ready[0]
            .as_ref()
            .expect("complete first component")
            .clone();
        let second = self.ready[1]
            .as_ref()
            .expect("complete second component")
            .clone();
        self.host = Some(match self.id.representation {
            CacheRepresentation::KeyValue => HostCacheBlock::KeyValue {
                keys: first,
                values: second,
            },
            CacheRepresentation::CompressedLatentRotary => HostCacheBlock::CompressedLatentRotary {
                latent: first,
                rotary_key: second,
            },
        });
        Ok(())
    }
}
impl CompletedDiskRead {
    pub(crate) fn id(&self) -> &CacheBlockId {
        &self.body.id
    }
    pub(crate) fn manager(&self) -> &CacheResidencyManager {
        &self.body.manager
    }
    pub(crate) fn generation(&self) -> u64 {
        self.body.generation
    }
    pub(crate) fn host(&self) -> &HostCacheBlock {
        self.body.host.as_ref().expect("completed Host publication")
    }
}
pub(super) fn control_bytes() -> Option<usize> {
    let frames = [
        size_of::<FinishCause>(),
        size_of::<CompletedDiskRead>(),
        size_of::<Result<CompletedDiskRead, DiskReadFinishFailure>>(),
        size_of::<Result<(), FinishCause>>(),
        size_of::<ReadBody>(),
        size_of::<Mutex<Option<ReadBody>>>(),
        size_of::<Mutex<ReadBody>>(),
        size_of::<MutexGuard<'_, Option<ReadBody>>>(),
        size_of::<Result<BodyLoan<'_>, TryLockError<BodyLoan<'_>>>>(),
        size_of::<Option<ReadBody>>(),
        size_of::<Option<&ReadCompletion>>(),
        size_of::<(&PreparedDiskReadOutput,)>(),
        size_of::<(FinishCause, ReadBody, PreparedDiskReadOutput)>(),
        size_of::<(FinishCause, &PreparedDiskReadOutput)>(),
        size_of::<(&mut ReadBody, usize, safemlx::AllocationInfo)>(),
        size_of::<
            Result<
                (ImmutableHostTransferBuffer, safemlx::AllocationInfo),
                filled_host::SourceError,
            >,
        >(),
        size_of::<Result<(), filled_host::SourceCause>>(),
        size_of::<HostCacheBlock>(),
        size_of::<[Arc<ImmutableHostTransferBuffer>; 2]>(),
        safemlx::HostTransferDescriptor::<4>::control_bytes()?.checked_mul(2)?,
    ];
    frames
        .into_iter()
        .try_fold(size_of_val(&frames), usize::checked_add)
}
