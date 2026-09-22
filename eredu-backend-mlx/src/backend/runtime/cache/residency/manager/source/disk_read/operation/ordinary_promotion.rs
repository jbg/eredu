//! Actual completed ordinary file read, moved into one promotion handoff.
use super::*;
use crate::backend::nn::workspace::OrdinaryPagedCause;
use crate::backend::runtime::cache::residency::PreparedHostPromotion;
use eredu_nn::workspace::WorkspaceMetadataAllocation;
use safemlx::{HostTransferDescriptor, error::Exception};

/// Creator-side context and a closed Send brand carried by the actual I/O body.
/// The context never crosses the worker boundary and labels cannot mint a brand.
pub(crate) struct OrdinaryDiskReadSource {
    identity: Arc<()>,
    context: WorkspaceContext,
}
impl OrdinaryDiskReadSource {
    pub(in super::super) fn prepare(
        context: &WorkspaceContext,
    ) -> Result<Self, CacheSourceFailure> {
        Ok(Self {
            identity: context
                .metadata_arc(())
                .map_err(|cause| CacheSourceFailure::metadata(cause.into(), context))?,
            context: context.clone(),
        })
    }
    pub(in super::super) fn from_prepared_identity(
        identity: Arc<()>,
        context: &WorkspaceContext,
    ) -> Self {
        Self {
            identity,
            context: context.clone(),
        }
    }
    pub(in super::super) fn identity(&self) -> Arc<()> {
        self.identity.clone()
    }
}

/// Minted only after the shared read operation committed these exact buffers.
/// Moving its completed body empties the operation's staging slot, so successful
/// promotion can retire Host staging independently of Device outputs and backing.
pub(crate) struct OrdinaryReadCacheHostSource {
    host: HostCacheBlock,
    _completed: Mutex<CompletedDiskRead>,
    backing: DiskLocation,
    file: CacheFileSource,
    descriptors: [HostTransferDescriptor<4>; 2],
    manager: CacheResidencyManager,
    id: CacheBlockId,
    generation: u64,
    context: WorkspaceContext,
    _funding: HostMetadataFunding,
}
impl OrdinaryReadCacheHostSource {
    pub(crate) fn host(&self) -> &HostCacheBlock {
        &self.host
    }
    pub(crate) fn buffers(&self) -> [&ImmutableHostTransferBuffer; 2] {
        self.host.buffers()
    }
    pub(in super::super::super::super) fn backing(&self) -> &DiskLocation {
        &self.backing
    }
    pub(crate) fn validate<P: PreparedHostPromotion>(
        &self,
        proof: &P,
        id: &CacheBlockId,
        actual: [&ImmutableHostTransferBuffer; 2],
    ) -> Result<(), Exception> {
        proof.validate_context(&self.context)?;
        proof.validate_manager(&self.manager, self.generation)?;
        proof.validate_id(id)?;
        if id != &self.id
            || self
                .backing
                .file_source()
                .is_none_or(|file| !file.same_source(&self.file))
        {
            return Err(proof.error(CacheSourceError::Identity.into()));
        }
        for (index, (expected, actual)) in self.buffers().into_iter().zip(actual).enumerate() {
            if !std::ptr::eq(expected, actual) {
                return Err(proof.error(CacheSourceError::Identity.into()));
            }
            let descriptor = actual
                .try_fixed_descriptor::<4>()
                .map_err(|cause| proof.error(CacheSourceError::HostDescriptor(cause).into()))?;
            if descriptor.shape() != self.descriptors[index].shape()
                || descriptor.dtype() != self.descriptors[index].dtype()
                || descriptor.allocation() != self.descriptors[index].allocation()
                || descriptor.policy() != HostTransferPolicy::Transfer
            {
                return Err(proof.error(CacheSourceError::Identity.into()));
            }
        }
        Ok(())
    }
    pub(crate) fn validation_control_bytes<P: PreparedHostPromotion>() -> Option<usize> {
        let frames = [
            size_of::<(&Self, &P, &CacheBlockId, [&ImmutableHostTransferBuffer; 2])>(),
            size_of::<Result<(), Exception>>(),
            size_of::<[&ImmutableHostTransferBuffer; 2]>(),
            size_of::<
                std::iter::Enumerate<
                    std::iter::Zip<
                        std::array::IntoIter<&ImmutableHostTransferBuffer, 2>,
                        std::array::IntoIter<&ImmutableHostTransferBuffer, 2>,
                    >,
                >,
            >(),
            size_of::<HostTransferDescriptor<4>>(),
            HostTransferDescriptor::<4>::control_bytes()?.checked_mul(2)?,
        ];
        frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)
    }
    pub(crate) fn handoff_control_bytes<P: PreparedHostPromotion>(
        publication_controls: usize,
    ) -> Option<usize> {
        let frames = [
            Self::validation_control_bytes::<P>()?,
            publication_controls,
            size_of::<Self>(),
            size_of::<Mutex<CompletedDiskRead>>(),
            initialized_mutex_control_bytes::<CompletedDiskRead>()?,
            size_of::<(&mut DiskReadOperation, &P, &OrdinaryDiskReadSource)>(),
            size_of::<Result<Self, Exception>>(),
            size_of::<BodyLoan<'_>>(),
            size_of::<Result<BodyLoan<'_>, TryLockError<BodyLoan<'_>>>>(),
            size_of::<[HostTransferDescriptor<4>; 2]>(),
            HostTransferDescriptor::<4>::control_bytes()?.checked_mul(2)?,
            size_of::<(
                &P,
                &CompletedDiskRead,
                &WorkspaceContext,
                &DiskReadOperation,
            )>(),
            size_of::<DiskLocation>(),
            size_of::<CacheBlockSelection>(),
            size_of::<CacheBlockSourceLoan<'_>>(),
            size_of::<Result<DiskLocation, Exception>>(),
            size_of::<HostCacheBlock>(),
            size_of::<CacheBlockId>(),
        ];
        frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)
    }
}
type BodyLoan<'a> = MutexGuard<'a, Option<CompletedDiskRead>>;
impl DiskReadOperation {
    pub(crate) fn take_ordinary_host_source<
        P: PreparedHostPromotion<Cause = OrdinaryPagedCause>,
    >(
        &mut self,
        proof: &P,
        source: &OrdinaryDiskReadSource,
    ) -> Result<OrdinaryReadCacheHostSource, Exception> {
        let context = &source.context;
        let fail = |cause: CacheSourceError| proof.error(cause.into());
        proof.validate_context(context)?;
        proof.validate_manager(&self.manager, self.key.generation)?;
        proof.validate_id(&self.key.id)?;
        if !self.committed
            || self.armed
            || self.promotion_attempted
            || context
                .metadata_funding()
                .as_ref()
                .is_none_or(|owner| !owner.same_account(&self.funding))
        {
            return Err(fail(CacheSourceError::Identity));
        }
        let mut finished = self
            .finished
            .try_lock()
            .map_err(|cause| fail(lock_cause(cause)))?;
        let completed = finished
            .as_ref()
            .ok_or_else(|| fail(CacheSourceError::Identity))?;
        let body = &completed.body;
        let origin = body
            .ordinary_identity
            .as_ref()
            .ok_or_else(|| fail(CacheSourceError::Identity))?;
        if body.source_custody.is_some()
            || !Arc::ptr_eq(origin, &source.identity)
            || body.id != self.key.id
            || body.generation != self.key.generation
            || !body.manager.same_catalog(&self.manager)
            || !body.source.same_source(&self.source)
        {
            return Err(fail(CacheSourceError::Identity));
        }
        let buffers = completed.host().buffers();
        let descriptors = [
            buffers[0]
                .try_fixed_descriptor::<4>()
                .map_err(|cause| fail(cause.into()))?,
            buffers[1]
                .try_fixed_descriptor::<4>()
                .map_err(|cause| fail(cause.into()))?,
        ];
        let selection = CacheBlockSelection::new(
            self.key.id.global_layer,
            self.key.id.representation,
            self.key.id.start,
            self.key.id.end,
            0,
        );
        let backing = self.manager.with_source_loan_inner(
            selection,
            Some(proof.publication_controls()),
            fail,
            |loan| {
                proof.validate_manager(loan.manager(), loan.generation())?;
                let row = loan
                    .blocks()
                    .find(|row| row.id() == &self.key.id)
                    .ok_or_else(|| fail(CacheSourceError::Identity))?;
                let disk = row.disk().ok_or_else(|| fail(CacheSourceError::Identity))?;
                if row.phase() != CacheStoragePhase::HostBacked
                    || disk
                        .file_source()
                        .is_none_or(|file| !file.same_source(&self.source))
                    || row.host().is_none_or(|actual| {
                        actual
                            .into_iter()
                            .zip(buffers)
                            .any(|(actual, expected)| !std::ptr::eq(actual.as_ref(), expected))
                    })
                {
                    return Err(fail(CacheSourceError::Identity));
                }
                Ok(disk.location.clone())
            },
        )?;
        let host = completed.host().clone();
        self.promotion_attempted = true;
        self.staging_transferred = true;
        let completed = Mutex::new(finished.take().expect("validated completed read"));
        drop(
            completed
                .lock()
                .expect("new unshared read-completion mutex"),
        );
        Ok(OrdinaryReadCacheHostSource {
            host,
            _completed: completed,
            backing,
            file: self.source.clone(),
            descriptors,
            manager: self.manager.clone(),
            id: self.key.id.clone(),
            generation: self.key.generation,
            context: context.clone(),
            _funding: self.funding.clone(),
        })
    }
}
