//! Explicit original completion and finite capture host controls.
use super::*;
use crate::backend::{
    error::Error,
    runtime::residency::storage::{RetainedStorage, RetainedStoragePublication},
};
use eredu_runtime::working_memory::OriginalTextMetadataCustody;
use safemlx::{EvaluatedArray, OperationEvent, OriginalScopeObserver, PreparedArrayClone};
use std::{alloc::Layout, mem::size_of};

/// The original observer must come from the actual accepted model/prefill role.
/// Current TLS only checks that retained identity; it never chooses a grant.
#[derive(Clone, Copy)]
pub(crate) enum CaptureCompletion<'a> {
    Ordinary,
    Original(&'a OriginalScopeObserver),
}
impl CaptureCompletion<'_> {
    /// Authenticate the retained and currently entered owner only. Completed
    /// readout retirement has no future submission and must not require another
    /// unused nested frontier after consuming the exact accepted population.
    pub(crate) fn validate_identity(self) -> Result<(), CaptureTensorNativeError> {
        if let Self::Original(expected) = self {
            let current = OriginalScopeObserver::require_current()?;
            if !expected.same_scope(&current) {
                return Err(CaptureTensorNativeError::ClaimMismatch);
            }
        }
        Ok(())
    }
    pub(crate) fn validate(self) -> Result<(), CaptureTensorNativeError> {
        self.validate_identity()?;
        if matches!(self, Self::Original(_)) {
            OperationEvent::validate_nested_completion(1)?;
        }
        Ok(())
    }
    pub(crate) fn settle<'v>(
        self,
        value: &'v Array,
        stream: &Stream,
    ) -> Result<EvaluatedArray<'v>, CaptureTensorNativeError> {
        match self {
            Self::Ordinary => Ok(value.evaluated()?),
            Self::Original(observer) => {
                self.validate()?;
                OperationEvent::complete_nested([value], stream)?;
                Ok(value.completed_in_original_scope(observer)?)
            }
        }
    }
    /// Keep the complete input graph set in the caller's existing recovery
    /// collector before the first settlement. Partial clone/settlement failure
    /// never releases the earlier roots or substitutes an ordinary readback.
    pub(crate) fn settle_retained<'v, const N: usize>(
        self, inputs: [&'v Array; N], stream: &Stream, roots: &RefCell<Vec<Array>>,
    ) -> Result<[EvaluatedArray<'v>; N], CaptureTensorNativeError> {
        self.validate_identity()?;
        self.reserve_roots(roots, N)?;
        for value in inputs {
            let retained = self.clone_array(value)?;
            let mut destination = roots.try_borrow_mut()
                .map_err(|_| CaptureTensorNativeError::CollectorBusy)?;
            if destination.len() == destination.capacity() {
                return Err(WorkingMemoryError::UnknownBound.into());
            }
            destination.push(retained);
        }
        let mut completed: [Option<EvaluatedArray<'v>>; N] = std::array::from_fn(|_| None);
        for (slot, value) in completed.iter_mut().zip(inputs) {
            *slot = Some(self.settle(value, stream)?);
        }
        Ok(completed.map(|value| value.expect("every source settled")))
    }
    pub(crate) fn retained_settlement_control_bytes<const N: usize>() -> Option<usize> {
        let frames = [
            size_of::<[&Array; N]>(), size_of::<[Option<EvaluatedArray<'_>>; N]>(),
            size_of::<[EvaluatedArray<'_>; N]>(),
            size_of::<Result<[EvaluatedArray<'_>; N], CaptureTensorNativeError>>(),
            size_of::<(Self, [&Array; N], &Stream, &RefCell<Vec<Array>>)>(),
            size_of::<std::cell::RefMut<'_, Vec<Array>>>(),
            size_of::<std::cell::BorrowMutError>(), size_of::<Array>(),
            size_of::<Result<Array, CaptureTensorNativeError>>(),
            size_of::<Result<EvaluatedArray<'_>, CaptureTensorNativeError>>(),
            PreparedArrayClone::control_bytes()?.checked_add(Array::inspection_clone_handle_bytes())?.checked_mul(N)?,
            OperationEvent::nested_completion_control_bytes::<1>()?.checked_mul(N)?,
            OriginalScopeObserver::control_bytes()?,
        ];
        frames.into_iter().try_fold(std::mem::size_of_val(&frames), usize::checked_add)
    }
    pub(crate) fn clone_array(self, value: &Array) -> Result<Array, CaptureTensorNativeError> {
        match self {
            Self::Ordinary => Ok(value.try_clone_for_inspection()?),
            Self::Original(observer) => {
                let current = OriginalScopeObserver::require_current()?;
                if !observer.same_scope(&current) {
                    return Err(CaptureTensorNativeError::ClaimMismatch);
                }
                let mut clone = PreparedArrayClone::try_prepare_for_inspection()?;
                Ok(clone.fill_in_original_scope(value, observer)?)
            }
        }
    }
    pub(crate) fn reserve_roots(
        self,
        roots: &RefCell<Vec<Array>>,
        additional: usize,
    ) -> Result<(), CaptureTensorNativeError> {
        let mut roots = roots
            .try_borrow_mut()
            .map_err(|_| CaptureTensorNativeError::CollectorBusy)?;
        match self {
            Self::Ordinary => roots.try_reserve_exact(additional)?,
            Self::Original(_) if roots.capacity() - roots.len() < additional => {
                return Err(CaptureTensorNativeError::Memory(
                    WorkingMemoryError::UnknownBound,
                ));
            }
            Self::Original(_) => (),
        }
        Ok(())
    }
}
#[derive(Debug)]
struct Failure {
    cause: Error,
    // Only the closed account; no Work, native graph or source backedge.
    _custody: OriginalTextMetadataCustody,
}
impl fmt::Display for Failure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.cause, f)
    }
}
impl std::error::Error for Failure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.cause)
    }
}
pub(crate) fn retain_error(cause: Error, custody: OriginalTextMetadataCustody) -> Error {
    Error::with_original_control_source(
        eredu_core::BackendFailure::from_error(Failure {
            cause,
            _custody: custody,
        }),
        false,
    )
}

pub(crate) fn error_control_bytes() -> Option<usize> {
    [
        size_of::<Failure>(),
        size_of::<Error>(),
        eredu_core::BackendFailure::source_retention_peak_bytes::<Failure>()?,
    ]
    .into_iter()
    .try_fold(0usize, usize::checked_add)
}

/// Fixed closed Slice/optional Preview/optional AsType profile, rank at most32.
/// Graph/P are in the actual shared transform trace. Publication nodes/entries
/// use the selected bank's separately counted attempts. These are the extra
/// per-transfer collectors, clone handles, metadata snapshots and call frames.
pub(crate) fn control_bytes() -> Option<usize> {
    let clone =
        PreparedArrayClone::control_bytes()?.checked_add(Array::inspection_clone_handle_bytes())?;
    let snapshots = Layout::array::<i32>(32)
        .ok()?
        .size()
        .checked_add(size_of::<ArrayMetadataSnapshot>())?
        .checked_mul(3)?;
    let values = [
        OperationEvent::nested_completion_control_bytes::<1>()?.checked_mul(2)?,
        RetainedStorage::original_collector_control_bytes(1)?,
        // Source/carrier plus two selected-source clones and four possible
        // retained intermediates: Slice, reshape, Preview Slice and AsType.
        clone.checked_mul(7)?,
        snapshots,
        Stream::device_type_control_bytes()?.checked_mul(2)?,
        Array::descriptor_control_bytes()?.checked_mul(3)?,
        EvaluatedArray::iteration_control_bytes::<f32>()?,
        size_of::<Selection>(),
        size_of::<Selection>(),
        size_of::<PreparedCaptureTensor<'static>>(),
        size_of::<PreparedCaptureFragment<'static, 'static, 'static, 'static>>(),
        size_of::<CaptureCompletion<'static>>(),
        size_of::<Native<'static>>(),
        size_of::<RetainedStoragePublication>(),
        size_of::<Result<RetainedStoragePublication, Error>>(),
        size_of::<WorkingMemoryFundingScope>(),
        size_of::<Result<OriginalScopeObserver, Error>>(),
        crate::backend::nn::tensor::TokenValidationScope::capture_observer_control_bytes()?,
        OriginalScopeObserver::control_bytes()?,
        // The final nested event has already completed. Each selected retained
        // descriptor reuses this one borrowed validation frame, including its
        // fixed refusal transport; no per-descriptor allocation is performed.
        OriginalScopeObserver::control_bytes()?,
        size_of::<usize>(), // start of this invocation's retained roots
        size_of::<std::cell::Ref<'static, Vec<Array>>>(),
        size_of::<&'static [Array]>(),
        size_of::<std::slice::Iter<'static, Array>>(),
        size_of::<&Array>(),
        size_of::<Result<(), Exception>>(),
        size_of::<WorkingMemoryError>(),
        size_of::<Box<WorkingMemoryError>>(),
        size_of::<std::cell::BorrowError>(),
        size_of::<std::cell::BorrowMutError>(),
        size_of::<std::collections::TryReserveError>(),
        size_of::<CaptureTensorHostPlan<'static>>(),
        size_of::<CaptureTensorGeometry<'static>>(),
        size_of::<Result<CaptureTensorHostPlan<'static>, WorkingMemoryError>>(),
        size_of::<CaptureTensorNativeError>(),
        size_of::<Box<CaptureTensorNativeError>>(),
        size_of::<Error>(),
        size_of::<Error>(),
        size_of::<Result<ClaimedCaptureTensor, Error>>(),
        // Leaf error, shared observer callback and final scheduled transport
        // can each wrap the preceding source; every shell retains Q last.
        error_control_bytes()?.checked_mul(3)?,
    ];
    values
        .into_iter()
        .try_fold(std::mem::size_of_val(&values), usize::checked_add)
}
