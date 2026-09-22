//! Source-derived host construction, separate from numerical copy admission.
use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PagedArrayCopyLayout {
    source: IndependentCacheManagerPlan,
    blocks: usize,
    operands: usize,
    retained_file_bytes: u64,
    tails: usize,
    bytes: usize,
}
impl PagedArrayCopyLayout {
    pub(crate) fn retained_file_bytes(self) -> u64 {
        self.retained_file_bytes
    }
    pub(crate) fn control_bytes(self) -> usize {
        self.bytes
    }
    pub(crate) fn page_operands(self) -> Option<usize> {
        Some(self.operands)
    }
}
impl CacheResidencyManager {
    pub(crate) fn inspect_paged_array_copy(
        &self,
    ) -> Result<PagedArrayCopyLayout, CacheSourceError> {
        self.with_source_loan_inner(
            CacheBlockSelection::new(0, CacheRepresentation::KeyValue, 0, i64::MAX, 0),
            None,
            |cause| cause,
            |source| source.paged_copy_layout(),
        )
    }
}
impl CacheBlockSourceLoan<'_> {
    pub(super) fn paged_copy_layout(&self) -> Result<PagedArrayCopyLayout, CacheSourceError> {
        let source = self.independent_plan()?;
        let (blocks, tails) = self.catalog_population();
        if self.all_blocks().count() != blocks {
            return Err(CacheSourceError::Identity);
        }
        let mut rows = 0usize;
        let mut operands = 0usize;
        let mut retained_file_bytes = 0u64;
        for block in self.all_blocks() {
            let dtypes = source_types(block)?;
            if block.device().is_some() || block.host().is_some() {
                operands = operands.checked_add(2).ok_or(CacheSourceError::Overflow)?;
            }
            if let Some(file) = block.disk().and_then(|disk| disk.file_source()) {
                retained_file_bytes = retained_file_bytes
                    .checked_add(
                        u64::try_from(file.file_bytes().ok_or(CacheSourceError::Identity)?)
                            .map_err(|_| CacheSourceError::Overflow)?,
                    )
                    .ok_or(CacheSourceError::Overflow)?;
            }
            if block.host().is_some() {
                rows = rows
                    .checked_add(
                        PreparedHostArrayCopy::control_bytes()
                            .and_then(|n| n.checked_mul(2))
                            .ok_or(CacheSourceError::Overflow)?,
                    )
                    .ok_or(CacheSourceError::Overflow)?;
            }
            rows = rows
                .checked_add(row_controls().ok_or(CacheSourceError::Overflow)?)
                .and_then(|n| {
                    n.checked_add(CacheBlockMetadata::floating_control_bytes(
                        block.shapes(),
                        dtypes,
                    )?)
                })
                .ok_or(CacheSourceError::Overflow)?;
        }
        let publication = self
            .copy_publication_control_bytes()
            .ok_or(CacheSourceError::Overflow)?;
        let frames = [
            Some(source.control_bytes()),
            controls(),
            host::control_bytes(),
            Some(rows),
            WorkspaceContext::metadata_vec_bytes::<Row>(blocks),
            WorkspaceContext::metadata_vec_bytes::<HostWork>(blocks),
            WorkspaceContext::metadata_vec_bytes::<Option<[PreparedArrayClone; 2]>>(blocks),
            PreparedPagedArrayCopy::failure_control_bytes(),
            Some(self.copy_occupancy_control_bytes()?),
            publication.checked_mul(
                blocks
                    .checked_add(tails)
                    .ok_or(CacheSourceError::Overflow)?,
            ),
            // Final manager validation. Every actual layer claim, including an
            // empty tail, is priced separately by the state source plan.
            PreparedIndependentCacheManager::copy_tail_control_bytes(),
            CacheResidencyManager::source_loan_control_bytes::<()>(size_of::<(
                &mut Pending,
                &PagedArrayCopyLayout,
                &WorkspaceContext,
            )>()),
            WorkspaceContext::metadata_source_bytes::<CacheSourceFailure>(),
            WorkspaceContext::metadata_source_bytes::<PublicationFailure>(),
            WorkspaceContext::metadata_source_bytes::<
                eredu_runtime::cache::CachePoolReservationPreparationFailure,
            >(),
            WorkspaceContext::metadata_source_bytes::<
                eredu_runtime::cache::CachePoolReservationAdmissionFailure,
            >(),
            Some(
                size_of::<Self>()
                    + size_of::<PagedArrayCopyLayout>()
                    + size_of::<Result<PagedArrayCopyLayout, CacheSourceError>>(),
            ),
        ];
        let bytes = frames
            .into_iter()
            .try_fold(std::mem::size_of_val(&frames), |sum, n| sum.checked_add(n?))
            .ok_or(CacheSourceError::Overflow)?;
        Ok(PagedArrayCopyLayout {
            source,
            blocks,
            operands,
            retained_file_bytes,
            tails,
            bytes,
        })
    }
}
