//! Completed file reads enter the same exact-source Host promotion worker.
use super::*;
use crate::backend::nn::workspace::OriginalPagedScanSource;
use crate::backend::runtime::cache::residency::PreparedCacheHostPromotion;
use crate::backend::runtime::cache::residency::PreparedCacheHostPromotionSlots;
use safemlx::{HostTransferDescriptor, error::Exception};

/// Constructed only from a positively completed and canonically published read.
/// Native buffers precede the exact read/file/pin/occupancy and source account.
/// No raw buffer, path, shape or caller-created ID can construct this witness.
pub(crate) struct ReadCacheHostSource {
    host: HostCacheBlock,
    completed: Finished,
    file: CacheFileSource,
    descriptors: [HostTransferDescriptor<4>; 2],
    id: CacheBlockId,
    manager: CacheResidencyManager,
    generation: u64,
    custody: OriginalHostSourceCustody,
    context: WorkspaceContext,
    funding: HostMetadataFunding,
    selection: ReadSelection,
}
// Only the completed-read handoff below can select this private origin. The
// prepared arm must still bind its exact declaration's paid promotion slots.
enum ReadSelection {
    Retained,
    PreparedSlots,
}

impl ReadCacheHostSource {
    pub(crate) fn validate(
        &self,
        proof: &OriginalPagedScanSource<'_>,
        id: &CacheBlockId,
        actual: [&ImmutableHostTransferBuffer; 2],
    ) -> Result<(), Exception> {
        proof.validate_manager(&self.manager, self.generation)?;
        if id != &self.id
            || !self.custody.same_source(proof.host_source_custody())
            || !self.context.shares_trace(proof.context())
        {
            return Err(proof.error(CacheSourceError::Identity));
        }
        if matches!(self.selection, ReadSelection::Retained) {
            proof.validate_retained_read_geometry(id, &self.descriptors, &self.context)?;
        }
        if self
            .host
            .buffers()
            .into_iter()
            .zip(actual)
            .any(|(expected, actual)| !std::ptr::eq(expected, actual))
        {
            return Err(proof.error(CacheSourceError::Identity));
        }
        Ok(())
    }
}
impl DiskReadOperation {
    /// Uses the same canonical Host preparation worker after this exact file
    /// read has committed. The caller supplies the currently accepted scan
    /// role; merely completing a read cannot grant inference or a Device copy.
    ///
    /// This compatibility handoff covers retained selected rows. The finite
    /// itinerary uses the same worker with its already prepared declaration slots.
    pub(crate) fn prepare_host_promotion(
        &mut self,
        proof: &OriginalPagedScanSource<'_>,
    ) -> Result<PreparedCacheHostPromotion, CacheSourceFailure> {
        let context = proof.context();
        let fail = |cause| CacheSourceFailure::source(cause, context);
        context
            .charge_metadata(
                Self::promotion_control_bytes().ok_or_else(|| fail(CacheSourceError::Overflow))?,
            )
            .map_err(|cause| CacheSourceFailure::metadata(cause.into(), context))?;
        self.prepare_host_promotion_inner(proof, None)
    }
    /// Consumes the actual paid Device destination from the same selected load.
    /// Its controls were paid with that destination before source construction.
    pub(crate) fn prepare_host_promotion_from_slots(
        &mut self,
        proof: &OriginalPagedScanSource<'_>,
        slots: PreparedCacheHostPromotionSlots,
    ) -> Result<PreparedCacheHostPromotion, CacheSourceFailure> {
        self.prepare_host_promotion_inner(proof, Some(slots))
    }
    fn prepare_host_promotion_inner(
        &mut self,
        proof: &OriginalPagedScanSource<'_>,
        slots: Option<PreparedCacheHostPromotionSlots>,
    ) -> Result<PreparedCacheHostPromotion, CacheSourceFailure> {
        let context = proof.context();
        let fail = |cause| CacheSourceFailure::source(cause, context);
        if !self.committed
            || self.armed
            || self.promotion_attempted
            || context
                .metadata_funding()
                .as_ref()
                .is_none_or(|funding| !funding.same_account(&self.funding))
        {
            return Err(fail(CacheSourceError::Identity));
        }
        self.promotion_attempted = true;
        proof
            .validate_manager(&self.manager, self.key.generation)
            .map_err(|cause| {
                CacheSourceFailure::metadata(context.metadata_source(cause), context)
            })?;
        let source = {
            let finished = self
                .finished
                .try_lock()
                .map_err(|cause| fail(lock_cause(cause)))?;
            let completed = finished
                .as_ref()
                .ok_or_else(|| fail(CacheSourceError::Identity))?;
            let body = &completed.body;
            let custody = body
                .source_custody
                .as_ref()
                .ok_or_else(|| fail(CacheSourceError::Identity))?;
            if body.id != self.key.id
                || body.generation != self.key.generation
                || !body.manager.same_catalog(&self.manager)
                || !body.source.same_source(&self.source)
                || !custody.same_source(proof.host_source_custody())
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
            let source = ReadCacheHostSource {
                host: completed.host().clone(),
                completed: self.finished.clone(),
                file: self.source.clone(),
                descriptors,
                id: self.key.id.clone(),
                manager: self.manager.clone(),
                generation: self.key.generation,
                custody: custody.clone(),
                context: context.clone(),
                funding: self.funding.clone(),
                selection: if slots.is_some() {
                    ReadSelection::PreparedSlots
                } else {
                    ReadSelection::Retained
                },
            };
            source
                .validate(proof, &self.key.id, buffers)
                .map_err(|cause| {
                    CacheSourceFailure::metadata(context.metadata_source(cause), context)
                })?;
            source
        };
        let selection = CacheBlockSelection::new(
            self.key.id.global_layer,
            self.key.id.representation,
            self.key.id.start,
            self.key.id.end,
            0,
        );
        self.manager.with_source_loan_inner(
            selection,
            Some(proof.publication_controls()),
            fail,
            |mut loan| {
                proof.validate_loan(&loan).map_err(|cause| {
                    CacheSourceFailure::metadata(context.metadata_source(cause), context)
                })?;
                let row = loan
                    .blocks()
                    .find(|row| row.id() == &self.key.id)
                    .ok_or_else(|| fail(CacheSourceError::Identity))?;
                if row.phase() != CacheStoragePhase::HostBacked
                    || row
                        .disk()
                        .and_then(|disk| disk.file_source())
                        .is_none_or(|file| !file.same_source(&source.file))
                {
                    return Err(fail(CacheSourceError::Identity));
                }
                source
                    .validate(
                        proof,
                        &self.key.id,
                        row.host()
                            .ok_or_else(|| fail(CacheSourceError::Identity))?
                            .map(AsRef::as_ref),
                    )
                    .map_err(|cause| {
                        CacheSourceFailure::metadata(context.metadata_source(cause), context)
                    })?;
                // All fallible source construction/validation precedes the existing
                // worker's final pin. Nothing after that pin can fail under the loan.
                match slots {
                    Some(slots) => slots.bind_read(&mut loan, proof, source),
                    None => {
                        let mut promotion = loan.prepare_host_promotion(&self.key.id, context)?;
                        promotion.read_source = Some(source);
                        Ok(promotion)
                    }
                }
            },
        )
    }
    pub(crate) fn promotion_control_bytes() -> Option<usize> {
        let frames = [
            size_of::<ReadCacheHostSource>(),
            size_of::<ReadSelection>(),
            size_of::<Option<PreparedCacheHostPromotionSlots>>(),
            size_of::<(
                &mut Self,
                &OriginalPagedScanSource<'_>,
                Option<PreparedCacheHostPromotionSlots>,
            )>(),
            size_of::<(&mut Self, &OriginalPagedScanSource<'_>)>(),
            size_of::<(
                &ReadCacheHostSource,
                &OriginalPagedScanSource<'_>,
                &CacheBlockId,
                [&ImmutableHostTransferBuffer; 2],
            )>(),
            size_of::<Result<PreparedCacheHostPromotion, CacheSourceFailure>>(),
            size_of::<Result<(), Exception>>(),
            size_of::<Result<HostTransferDescriptor<4>, safemlx::HostTransferMetadataError>>(),
            size_of::<MutexGuard<'_, Option<CompletedDiskRead>>>(),
            size_of::<
                Result<
                    MutexGuard<'_, Option<CompletedDiskRead>>,
                    TryLockError<MutexGuard<'_, Option<CompletedDiskRead>>>,
                >,
            >(),
            size_of::<Option<HostMetadataFunding>>(),
            size_of::<CacheBlockSelection>(),
            size_of::<(&CacheBlockSourceLoan<'_>, Option<CacheDiskSource<'_>>, bool)>(),
            size_of::<
                std::iter::Zip<
                    std::array::IntoIter<&ImmutableHostTransferBuffer, 2>,
                    std::array::IntoIter<&ImmutableHostTransferBuffer, 2>,
                >,
            >(),
            HostTransferDescriptor::<4>::control_bytes()?.checked_mul(2)?,
            CacheResidencyManager::source_loan_control_bytes::<PreparedCacheHostPromotion>(
                size_of::<(
                    &Self,
                    &OriginalPagedScanSource<'_>,
                    ReadCacheHostSource,
                    Option<PreparedCacheHostPromotionSlots>,
                )>(),
            )?,
        ];
        frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)
    }
}

use eredu_nn::workspace::WorkspaceMetadataAllocation;
