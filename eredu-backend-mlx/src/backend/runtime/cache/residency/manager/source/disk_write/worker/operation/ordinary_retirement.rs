//! Successful ordinary writes detach completed staging from durable file custody.
use super::*;
use crate::backend::nn::workspace::{OrdinaryCallControls, OrdinaryNativeControls};

impl PreparedDiskWriteHostRetirement {
    pub(crate) fn prepare(
        context: &WorkspaceContext,
        publication_controls: usize,
    ) -> Result<Self, CacheSourceFailure> {
        context
            .charge_metadata(size_of::<(
                Self,
                &WorkspaceContext,
                usize,
                Result<Self, CacheSourceFailure>,
            )>())
            .map_err(|cause| CacheSourceFailure::metadata(cause.into(), context))?;
        Ok(Self {
            retirement:
                super::super::super::super::device_retirement::DeviceRetirement::prepare_host(
                    context,
                )?,
            context: context.clone(),
            consumed: false,
            publication_controls,
        })
    }

    pub(crate) fn reclaim(&mut self) {
        self.retirement.reclaim();
    }

    pub(crate) fn source_controls<P: PreparedCacheDiskWriteSource>(
        &self,
    ) -> Option<OrdinaryCallControls> {
        let frames = [
            size_of::<(&mut DiskWriteOperation, &P, &mut Self)>(),
            size_of::<Result<OrdinaryWrittenCacheHostSource, DiskWriteOperationFailure>>(),
            size_of::<Result<OrdinaryWrittenCacheHostSource, Cause>>(),
            size_of::<OrdinaryWrittenCacheHostSource>(),
            size_of::<CacheBlockSelection>(),
            size_of::<HostCacheBlock>(),
            size_of::<MutexGuard<'_, Option<HostCacheBlock>>>(),
            size_of::<
                Result<
                    MutexGuard<'_, Option<HostCacheBlock>>,
                    TryLockError<MutexGuard<'_, Option<HostCacheBlock>>>,
                >,
            >(),
            size_of::<MutexGuard<'_, Option<CachePoolReservation>>>(),
            size_of::<
                Result<
                    MutexGuard<'_, Option<CachePoolReservation>>,
                    TryLockError<MutexGuard<'_, Option<CachePoolReservation>>>,
                >,
            >(),
            CacheResidencyManager::source_loan_control_bytes::<()>(size_of::<(
                &DiskWriteOperation,
                &P,
                &LiveCacheBlockSource,
            )>())?,
            self.publication_controls,
            OrdinaryWrittenCacheHostSource::validation_control_bytes(),
        ];
        Some(OrdinaryCallControls {
            metadata_bytes: u64::try_from(
                frames
                    .into_iter()
                    .try_fold(size_of_val(&frames), usize::checked_add)?,
            )
            .ok()?,
            observed: OrdinaryNativeControls::default(),
        })
    }
}

impl OrdinaryWrittenCacheHostSource {
    pub(crate) fn buffers(&self) -> [&ImmutableHostTransferBuffer; 2] {
        self.host.buffers()
    }
    pub(crate) fn file(&self) -> &LiveCacheBlockSource {
        &self.file
    }
    pub(crate) fn validate_buffers(
        &self,
        id: &CacheBlockId,
        buffers: [&ImmutableHostTransferBuffer; 2],
        context: &WorkspaceContext,
    ) -> Result<(), CacheSourceError> {
        if id != &self.id
            || !self.context.shares_trace(context)
            || context
                .metadata_funding()
                .as_ref()
                .is_none_or(|funding| !funding.same_account(&self.funding))
            || self
                .buffers()
                .into_iter()
                .zip(buffers)
                .any(|(expected, actual)| !std::ptr::eq(expected, actual))
        {
            return Err(CacheSourceError::Identity);
        }
        Ok(())
    }
    pub(crate) fn validation_control_bytes() -> usize {
        size_of::<(
            &Self,
            &CacheBlockId,
            [&ImmutableHostTransferBuffer; 2],
            &WorkspaceContext,
            Option<HostMetadataFunding>,
            Result<(), CacheSourceError>,
            std::iter::Zip<
                std::array::IntoIter<&ImmutableHostTransferBuffer, 2>,
                std::array::IntoIter<&ImmutableHostTransferBuffer, 2>,
            >,
        )>()
    }
}

impl DiskWriteOperation {
    /// Completion, canonical file identity and the actual source proof all
    /// precede releasing any Host alias. Cancellation and polling errors cannot
    /// produce this witness.
    pub(crate) fn retire_ordinary_host_source<P: PreparedCacheDiskWriteSource>(
        &mut self,
        proof: &P,
        retirement: &mut PreparedDiskWriteHostRetirement,
    ) -> Result<OrdinaryWrittenCacheHostSource, DiskWriteOperationFailure> {
        self.retire_ordinary_host_source_inner(proof, retirement)
            .map_err(|cause| self.failure(cause))
    }

    fn retire_ordinary_host_source_inner<P: PreparedCacheDiskWriteSource>(
        &mut self,
        proof: &P,
        retirement: &mut PreparedDiskWriteHostRetirement,
    ) -> Result<OrdinaryWrittenCacheHostSource, Cause> {
        let invalid = || Cause::Source(CacheSourceError::Identity);
        let context = proof.context();
        if !self.committed
            || self.armed
            || retirement.consumed
            || proof.id() != &self.key.id
            || !retirement.context.shares_trace(context)
            || context
                .metadata_funding()
                .as_ref()
                .is_none_or(|funding| !funding.same_account(&self.funding))
        {
            return Err(invalid());
        }
        let completion = self.output.inner.get().ok_or(Cause::Completion)?;
        let location = completion.result.as_ref().map_err(|_| Cause::Write)?;
        let file = location.live_source.as_ref().ok_or_else(invalid)?;
        if completion.body.id != self.key.id
            || completion.body.generation != self.key.generation
            || !completion.body.manager.same_catalog(&self.manager)
            || !Arc::ptr_eq(&completion.body.transfer.inner, &self.occupancy.inner)
        {
            return Err(invalid());
        }
        let mut completed_host = completion
            .host
            .try_lock()
            .map_err(|cause| Cause::Source(lock_cause(cause)))?;
        let host = completed_host.as_ref().ok_or_else(invalid)?;
        let retained = self.host.as_ref().ok_or_else(invalid)?;
        if host
            .buffers()
            .into_iter()
            .zip(retained.buffers())
            .any(|(expected, actual)| !std::ptr::eq(expected, actual))
        {
            return Err(invalid());
        }
        let selection = CacheBlockSelection::new(
            self.key.id.global_layer,
            self.key.id.representation,
            self.key.id.start,
            self.key.id.end,
            0,
        );
        self.manager.with_source_loan_inner(
            selection,
            Some(retirement.publication_controls),
            Cause::Source,
            |loan| {
                let pins = proof.validate(&loan).map_err(Cause::SourceProof)?;
                if loan.generation() != self.key.generation {
                    return Err(invalid());
                }
                let (actual_pins, demand) =
                    loan.source_ownership(&self.key.id).map_err(Cause::Source)?;
                if demand || pins != actual_pins {
                    return Err(Cause::Source(CacheSourceError::Busy));
                }
                let row = loan
                    .blocks()
                    .find(|row| row.id() == &self.key.id)
                    .ok_or_else(invalid)?;
                if row.phase() != CacheStoragePhase::DiskReady
                    || row
                        .disk()
                        .and_then(|disk| disk.live_file())
                        .is_none_or(|actual| !actual.same_source(file))
                {
                    return Err(invalid());
                }
                Ok(())
            },
        )?;
        let mut occupancy = self
            .occupancy
            .inner
            .reservation
            .try_lock()
            .map_err(|cause| Cause::Source(lock_cause(cause)))?;
        if occupancy.is_none() {
            return Err(invalid());
        }
        retirement.consumed = true;
        retirement
            .retirement
            .attach_host(host.buffers())
            .map_err(Cause::Source)?;
        // Attached backing owners retain the exact removed Host and completed
        // transfer occupancy. Their lifetime is independent of the file/pin.
        retirement.retirement.publish(&mut occupancy);
        let host = completed_host.take().expect("validated completed source");
        drop(occupancy);
        drop(completed_host);
        drop(self.host.take());
        Ok(OrdinaryWrittenCacheHostSource {
            host,
            file: file.clone(),
            id: self.key.id.clone(),
            context: context.clone(),
            funding: self.funding.clone(),
        })
    }
}
