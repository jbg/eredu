//! Bind only an exact paid scan row through the existing lexical source loan.
use super::*;
use crate::backend::nn::workspace::{OriginalPagedBlockSource, OriginalPagedScanSource};
use eredu_runtime::CacheBlockSelection;
use safemlx::error::Exception;

impl CacheResidencyManager {
    pub(crate) fn original_scan_control_bytes() -> Option<usize> {
        use std::mem::size_of;
        let frames = [
            Self::source_loan_control_bytes::<()>(size_of::<(
                &mut OriginalPagedBlockSource<'_, '_>,
                &OriginalPagedScanSource<'_>,
            )>())?,
            size_of::<(&Self, &mut OriginalPagedBlockSource<'_, '_>)>(),
            size_of::<(&Self, &OriginalPagedScanSource<'_>, bool, u64, u64, u64)>(),
            size_of::<MutexGuard<'_, CacheManagerState>>(),
            size_of::<
                Result<
                    MutexGuard<'_, CacheManagerState>,
                    std::sync::TryLockError<MutexGuard<'_, CacheManagerState>>,
                >,
            >(),
            size_of::<Result<(), Exception>>(),
            size_of::<CacheBlockSelection>(),
            size_of::<std::iter::Once<usize>>(),
            size_of::<(u64, bool)>(),
            PinnedCacheBlock::fixed_controls()?,
        ];
        frames
            .into_iter()
            .try_fold(std::mem::size_of_val(&frames), usize::checked_add)
    }
    pub(crate) fn record_original_attention_scan(
        &self,
        source: &OriginalPagedScanSource<'_>,
        prefill: bool,
        blocks: u64,
        bytes: u64,
        scratch: u64,
    ) -> Result<(), Exception> {
        let mut state = self.inner.state.try_lock().map_err(|cause| {
            source.error(match cause {
                std::sync::TryLockError::WouldBlock => CacheSourceError::Busy,
                std::sync::TryLockError::Poisoned(_) => CacheSourceError::Poisoned,
            })
        })?;
        source.validate_manager(self, state.generation)?;
        state
            .telemetry
            .validate_prepared_layers(std::iter::once(source.layer()))
            .map_err(|cause| source.error(CacheSourceError::Lifecycle(cause.into())))?;
        reporting::record_scan(&mut state, source.layer(), prefill, blocks, bytes, scratch);
        Ok(())
    }
    pub(crate) fn bind_original_scan_block(
        &self,
        target: &mut OriginalPagedBlockSource<'_, '_>,
    ) -> Result<(), Exception> {
        let proof = target.proof;
        if !self.same_catalog(proof.manager()) {
            return Err(proof.error(CacheSourceError::Identity));
        }
        let id = target.id();
        let selection =
            CacheBlockSelection::new(id.global_layer, id.representation, id.start, id.end, 0);
        self.with_source_loan_inner(
            selection,
            Some(proof.publication_controls()),
            |cause| proof.error(cause),
            |loan| target.bind(loan),
        )
    }
}
