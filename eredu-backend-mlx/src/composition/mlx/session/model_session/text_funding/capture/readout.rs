//! Source completion, publication and retained roots shared by scalar readout claims.
use super::*;
use crate::backend::array_copy::CaptureTensorNativeError;
use crate::backend::runtime::residency::storage::StorageIdentity;
use eredu_runtime::working_memory::{
    CaptureCandidateClaim, CaptureSourceSegment, CaptureTokenScoreClaim, WorkingMemoryError,
    WorkingMemoryStorage,
};
fn error<E: std::error::Error + Send + Sync + 'static>(cause: E) -> Error {
    Error::Other(Box::new(cause))
}
pub(super) trait ReadoutClaim {
    fn validate_scope(&self, scope: &WorkingMemoryFundingScope) -> Result<(), WorkingMemoryError>;
}
impl ReadoutClaim for eredu_runtime::working_memory::CaptureSummaryClaim<'_, '_> {
    fn validate_scope(&self, scope: &WorkingMemoryFundingScope) -> Result<(), WorkingMemoryError> {
        self.validate_native_scope(scope)
    }
}
impl ReadoutClaim for eredu_runtime::working_memory::CaptureHistogramClaim<'_, '_> {
    fn validate_scope(&self, scope: &WorkingMemoryFundingScope) -> Result<(), WorkingMemoryError> {
        self.validate_native_scope(scope)
    }
}
impl ReadoutClaim for CaptureCandidateClaim<'_, '_> {
    fn validate_scope(&self, scope: &WorkingMemoryFundingScope) -> Result<(), WorkingMemoryError> {
        self.validate_native_scope(scope)
    }
}
impl ReadoutClaim for CaptureTokenScoreClaim<'_, '_> {
    fn validate_scope(&self, scope: &WorkingMemoryFundingScope) -> Result<(), WorkingMemoryError> {
        self.validate_native_scope(scope)
    }
}
impl FundedWork {
    /// C is one of the existing closed scheduled claims, not a new authority.
    /// Claim and segment authenticate before completion or publication. The
    /// callback consumes the same claim/pin with its exact typed destination.
    pub(super) fn with_capture_readout<C: ReadoutClaim, R>(
        &self,
        source: &Array,
        claim: C,
        stream: &Stream,
        completion: CaptureCompletion<'_>,
        roots: &RefCell<Vec<Array>>,
        publications: &RefCell<Vec<RetainedStoragePublication>>,
        segment: Option<&mut CaptureSourceSegment>,
        retained_roots: usize,
        operation: impl for<'s> FnOnce(
            C,
            &'s mut WorkingMemoryFundingScope,
            Option<&'s mut CaptureSourceSegment>,
            WorkingMemoryStorage<StorageIdentity>,
            &mut dyn FnMut(&Array) -> Result<(), CaptureTensorNativeError>,
        ) -> Result<R, Error>,
    ) -> Result<R, Error> {
        self.validate_capture_completion(completion)?;
        let _activity = publication_scope::Activity::begin(&self.publishing)?;
        PreparedCaptureTensor::validate_stream(stream).map_err(error)?;
        {
            let mut values = roots.try_borrow_mut().map_err(error)?;
            if values
                .capacity()
                .checked_sub(values.len())
                .is_none_or(|remaining| remaining < retained_roots)
            {
                return Err(error(WorkingMemoryError::UnknownBound));
            }
        }
        let mut scope_owner = publication_scope::OwnedScope::take(&self.scope)?;
        let scope = scope_owner.get_mut();
        claim.validate_scope(scope).map_err(error)?;
        if let Some(segment) = segment.as_ref() {
            segment.validate_native_scope(scope).map_err(error)?;
        }
        let mut retain = |value: &Array| -> Result<(), CaptureTensorNativeError> {
            let array = completion.clone_array(value)?;
            let mut values = roots
                .try_borrow_mut()
                .map_err(|_| CaptureTensorNativeError::CollectorBusy)?;
            if values.len() == values.capacity() {
                return Err(WorkingMemoryError::UnknownBound.into());
            }
            values.push(array);
            Ok(())
        };
        retain(source).map_err(error)?;
        self.published.set(false);
        drop(completion.settle(source, stream).map_err(error)?);
        let publication = self.publish_capture_source(source, scope, completion)?;
        // The source pin protects attached native charges through the callback.
        // This receipt has its own paid one-row collector and stays lexical.
        let _publication = publication;
        let observed = source.try_metadata_snapshot().map_err(error)?;
        let allocation = observed
            .allocation()
            .ok_or_else(|| error(WorkingMemoryError::UnknownBound))?;
        let prepared = eredu_runtime::working_memory::StoragePublicationLayout::new(2)
            .and_then(|layout| layout.fund_from(scope))
            .map_err(error)?;
        let rows = [
            (allocation.bytes() != 0).then_some((
                StorageIdentity::Native(allocation.identity()),
                u64::try_from(allocation.bytes())
                    .map_err(|_| error(WorkingMemoryError::Overflow))?,
            )),
            (allocation.host_control_bytes() != 0).then_some((
                StorageIdentity::NativeControl(allocation.identity()),
                u64::try_from(allocation.host_control_bytes())
                    .map_err(|_| error(WorkingMemoryError::Overflow))?,
            )),
        ];
        let pin = prepared
            .pin_registered_storage(rows.into_iter().flatten())
            .map_err(error)?;
        operation(claim, scope, segment, pin, &mut retain)
    }
}

/// Exact shared helper parameter/local/return representations. Invocations are
/// serialized by Activity; these lexical controls are reused between callbacks.
/// Per-callback clone, snapshot payload, publication and completion allocations
/// remain in CaptureNativePopulation and their existing native owner queries.
pub(super) fn control_bytes<C: ReadoutClaim, R>() -> Option<usize> {
    use std::mem::{size_of, size_of_val};
    type Pin = WorkingMemoryStorage<StorageIdentity>;
    let frames = [
        // Actual by-value claim and helper arguments, alongside the caller.
        size_of::<(
            C,
            &FundedWork,
            &Array,
            &Stream,
            CaptureCompletion<'static>,
            &RefCell<Vec<Array>>,
            &RefCell<Vec<RetainedStoragePublication>>,
            Option<&mut CaptureSourceSegment>,
            usize,
        )>(),
        // The retain closure borrows completion and roots while the callback runs.
        size_of::<(&CaptureCompletion<'static>, &RefCell<Vec<Array>>)>(),
        size_of::<&mut dyn FnMut(&Array) -> Result<(), CaptureTensorNativeError>>(),
        size_of::<std::cell::RefMut<'static, Vec<Array>>>(),
        size_of::<std::cell::RefMut<'static, Vec<RetainedStoragePublication>>>(),
        size_of::<Result<(), CaptureTensorNativeError>>(),
        size_of::<safemlx::ArrayMetadataSnapshot>(),
        size_of::<Result<safemlx::ArrayMetadataSnapshot, safemlx::ArrayMetadataError>>(),
        size_of::<Option<RetainedStoragePublication>>(),
        size_of::<Result<RetainedStoragePublication, Error>>(),
        size_of::<Pin>(),
        size_of::<Result<Pin, WorkingMemoryError>>(),
        size_of::<[(StorageIdentity, u64); 1]>(),
        size_of::<Result<R, Error>>(), // callback result handed through this frame
        size_of::<safemlx::ArrayMetadataError>(),
        size_of::<Box<safemlx::ArrayMetadataError>>(),
        publication_scope::control_bytes()?,
    ];
    frames
        .into_iter()
        .try_fold(size_of_val(&frames), usize::checked_add)
}

#[cfg(test)]
#[allow(unused_imports)]
use crate::memory_fixture::{FundingFixture as _, StorageFixture as _};
