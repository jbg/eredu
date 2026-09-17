//! Failure-only shape copies from the actual prepared grouped-observer bank.
use super::*;
use crate::backend::error::Error;
use eredu_nn::{Error as ComputeError, GroupedUnitError};
use eredu_runtime::working_memory::WorkingMemoryError;
use std::{alloc::Layout, mem::size_of};

#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
struct OriginalUnitFailure {
    #[source]
    cause: GroupedUnitError,
    // Both copied shape buffers and the erased source retire before this account.
    _custody: TokenValidationCustody,
}

pub(crate) struct PreparedGroupedUnitError {
    expected: Vec<i32>,
    actual: Vec<i32>,
    // Directory extraction moves the already-admitted custody; no new allowance.
    custody: TokenValidationCustody,
}
impl PreparedGroupedUnitError {
    pub(super) fn control_bytes(rank: usize) -> Option<usize> {
        let bytes = Layout::array::<i32>(2)
            .ok()?
            .size()
            .checked_add(Layout::array::<i32>(rank).ok()?.size())?;
        [
            size_of::<Self>(),
            size_of::<Result<Self, Error>>(),
            size_of::<Result<Option<Self>, Exception>>(),
            size_of::<Result<ComputeError, Exception>>(),
            size_of::<OriginalUnitFailure>(),
            size_of::<GroupedUnitError>(),
            size_of::<Option<safemlx::OriginalScopeObserver>>(),
            size_of::<std::collections::TryReserveError>(),
            ComputeError::retained_source_control_bytes::<OriginalUnitFailure>()?,
        ]
        .into_iter()
        .try_fold(bytes, usize::checked_add)
    }
    pub(super) fn new(rank: usize, custody: TokenValidationCustody) -> Result<Self, Error> {
        let mut value = Self {
            expected: Vec::new(),
            actual: Vec::new(),
            custody,
        };
        value.expected.try_reserve_exact(2).map_err(reserve_error)?;
        value
            .actual
            .try_reserve_exact(rank)
            .map_err(reserve_error)?;
        if value.expected.capacity() != 2 || value.actual.capacity() != rank {
            return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch));
        }
        Ok(value)
    }
    pub(crate) fn take_current() -> Result<Option<Self>, Exception> {
        let Some(observer) = safemlx::OriginalScopeObserver::try_current()? else {
            return Ok(None);
        };
        TOKEN_VALIDATION_SCOPE.with(|slot| {
            let mut slot = slot
                .try_borrow_mut()
                .map_err(|_| observer.capacity_error())?;
            slot.as_mut()
                .filter(|active| {
                    active.remaining.is_some()
                        && active
                            .observer
                            .as_ref()
                            .is_some_and(|owner| owner.same_scope(&observer))
                })
                .and_then(|active| active.grouped_outputs.take_unit_error())
                .map(Some)
                .ok_or_else(|| observer.capacity_error())
        })
    }
    pub(crate) fn shape(
        mut self,
        expected: &[i32],
        actual: &[i32],
    ) -> Result<ComputeError, Exception> {
        if expected.len() > self.expected.capacity() || actual.len() > self.actual.capacity() {
            // An unplanned foreign rank cannot grow the accepted error slot.
            // Refuse before copying either shape; preserve the real native cause.
            return Err(safemlx::OriginalScopeObserver::require_current()?.capacity_error());
        }
        self.expected.extend_from_slice(expected);
        self.actual.extend_from_slice(actual);
        Ok(ComputeError::backend_retained_source(OriginalUnitFailure {
            cause: GroupedUnitError::ReplacementShape {
                expected: self.expected,
                actual: self.actual,
            },
            _custody: self.custody,
        }))
    }
    pub(crate) fn dtype(self) -> ComputeError {
        let Self {
            expected,
            actual,
            custody,
        } = self;
        // Retire the unused buffers while the local custody still exists,
        // including if the following erased-error allocation unwinds.
        drop((expected, actual));
        ComputeError::backend_retained_source(OriginalUnitFailure {
            cause: GroupedUnitError::ReplacementDtype,
            _custody: custody,
        })
    }
}
fn reserve_error(cause: std::collections::TryReserveError) -> Error {
    Error::PrefillControl(WorkingMemoryError::ControlStorageReserve(cause))
}
