//! The same five completed loans populate a paid sparse edit destination.
use super::*;
use eredu_core::intervention::RoutedUnitLocation;
use eredu_runtime::intervention::{
    PreparedRoutedInterventionRows, RoutedInterventionLoweringError,
};

#[derive(Debug, thiserror::Error)]
pub(crate) enum RoutedInterventionReadError {
    #[error(transparent)]
    Native(#[from] Error),
    #[error(transparent)]
    Destination(#[from] RoutedInterventionLoweringError),
}
impl CompletedRoutedCaptureSource<'_> {
    /// The original capture/intervention observer lends its own source scope,
    /// roots and paid row destination. Validation uses actual borrowed layouts;
    /// the shared settlement worker keeps all five graphs before any Eval.
    pub(crate) fn settle_intervention_rows(
        source: &RoutedUnitCaptureSource<'_, Array>,
        target: &mut PreparedRoutedInterventionRows,
        stream: &safemlx::Stream,
        roots: &std::cell::RefCell<Vec<Array>>,
        observer: &safemlx::OriginalScopeObserver,
    ) -> Result<(), RoutedInterventionReadError> {
        let result = (|| {
            let bank = target.geometry();
            let tokens = target.source_tokens();
            let (rows, chunk) = Self::validate_borrowed(source, bank, tokens)?;
            if rows != target.expected_rows()
                || source.token_offset.checked_add(chunk).map(|end| [source.token_offset, end])
                    != Some(target.source_range()) {
                return Err(Error::SourceChanged.into());
            }
            let completion = crate::backend::array_copy::CaptureCompletion::Original(observer);
            let inputs = [source.values, source.token_indices, source.selection_indices,
                source.coefficients, source.source_groups];
            let completed = completion.settle_retained(inputs, stream, roots)?;
            CompletedRoutedCaptureSource::new(source, completed, bank, tokens)?.copy_intervention_rows(target)
        })();
        if result.is_err() { target.reject_source(); }
        result
    }
    /// This is scalar source transport only. The caller authenticates and funds
    /// the source, and retains both the completed loans and destination on error.
    /// No row/index native tensor, ordinary readback vector or cast is created.
    pub(crate) fn copy_intervention_rows(
        self,
        target: &mut PreparedRoutedInterventionRows,
    ) -> Result<(), RoutedInterventionReadError> {
        let result: Result<(), RoutedInterventionReadError> = (|| {
            if self.bank != target.geometry()
                || self.source_tokens != target.source_tokens()
                || self.rows != target.expected_rows()
                || self
                    .token_offset
                    .checked_add(self.chunk_tokens)
                    .map(|end| [self.token_offset, end])
                    != Some(target.source_range())
            {
                return Err(Error::SourceChanged.into());
            }
            for index in 0..self.rows {
                let route = self.route(index)?;
                target.push_row(RoutedUnitLocation {
                    source_peer: None,
                    token: route.token,
                    slot: route.slot,
                    expert: route.expert,
                })?;
            }
            Ok(())
        })();
        if result.is_err() {
            target.reject_source();
        }
        result
    }
    pub(crate) fn intervention_read_control_bytes() -> Option<usize> {
        use std::mem::{size_of, size_of_val};
        let frames = [
            size_of::<(Self, &mut PreparedRoutedInterventionRows)>(),
            size_of::<(&RoutedUnitCaptureSource<'_, Array>, &mut PreparedRoutedInterventionRows,
                &safemlx::Stream, &std::cell::RefCell<Vec<Array>>, &safemlx::OriginalScopeObserver)>(),
            crate::backend::array_copy::CaptureCompletion::retained_settlement_control_bytes::<5>()?,
            size_of::<RoutedInterventionReadError>(),
            size_of::<Result<(), RoutedInterventionReadError>>(),
            size_of::<RoutedUnitLocation>(),
            size_of::<Result<(), RoutedInterventionLoweringError>>(),
            size_of::<std::ops::Range<usize>>(),
            size_of::<Option<[u64; 2]>>(),
        ];
        frames.into_iter().try_fold(
            Self::control_bytes()?.checked_add(size_of_val(&frames))?,
            usize::checked_add,
        )
    }
}
