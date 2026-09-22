//! Lazy activation ingress under the already active original text work scope.
use super::*;
mod candidates;
mod carrier;
mod histogram;
mod observer;
mod ordinary;
pub(in crate::composition::mlx::session::model_session) use ordinary::control_bytes as ordinary_publication_control_bytes;
mod readout;
mod summary;
mod token_scores;
mod transient;
use crate::backend::array_copy::{
    CaptureCompletion, PreparedCaptureTensor, ScheduledCaptureTensorExecutionError,
};
pub(super) use carrier::{active_control_bytes, common_control_bytes, CaptureCarrierOwner};
use eredu_core::capture::CaptureTensorGeometry;
use eredu_runtime::working_memory::{
    CaptureTensorClaim, CaptureTensorHostPlan, ClaimedCaptureTensor,
};
pub(super) fn replica_control_bytes() -> Option<usize> {
    observer::replica_control_bytes()
}
pub(super) fn observer_error_control_bytes() -> Option<usize> {
    observer::error_control_bytes()?
        .checked_add(token_scores::control_bytes()?)?
        .checked_add(transient::control_bytes()?)?
        .checked_add(candidates::control_bytes()?)?
        .checked_add(summary::control_bytes()?)?
        .checked_add(histogram::control_bytes()?)?
        .checked_add(readout::control_bytes::<
            eredu_runtime::working_memory::CaptureTokenScoreClaim<'static, 'static>,
            eredu_runtime::working_memory::ClaimedCaptureTokenScores,
        >()?)
}

impl FundedWork {
    fn validate_capture_identity(&self, completion: CaptureCompletion<'_>) -> Result<(), Error> {
        match (self._controls.as_ref(), completion) {
            (_, CaptureCompletion::Ordinary) if self.native_storage.is_none() => Ok(()),
            (Some(_), CaptureCompletion::Original(_)) if self.native_storage.is_some() => {
                completion
                    .validate_identity()
                    .map_err(|e| Error::Other(Box::new(e)))
            }
            _ => Err(Error::PrefillScopeUnavailable),
        }
    }
    fn validate_capture_completion(&self, completion: CaptureCompletion<'_>) -> Result<(), Error> {
        self.validate_capture_identity(completion)?;
        completion.validate().map_err(|e| Error::Other(Box::new(e)))
    }
    fn capture_error(&self, cause: Error) -> Error {
        match self.original_metadata_custody() {
            Some(custody) => crate::backend::array_copy::capture_original_error(cause, custody),
            None => cause,
        }
    }
    fn publish_capture_source(
        &self,
        source: &Array,
        scope: &mut WorkingMemoryFundingScope,
        completion: CaptureCompletion<'_>,
    ) -> Result<RetainedStoragePublication, Error> {
        match completion {
            CaptureCompletion::Ordinary => {
                let mut storage = ordinary::prepare(scope)?;
                storage.include_array(source)?;
                storage.publish_funded(scope)
            }
            CaptureCompletion::Original(_) => {
                let controls = self
                    ._controls
                    .as_ref()
                    .ok_or(Error::PrefillScopeUnavailable)?;
                controls
                    .validate_native_scope(scope)
                    .map_err(Error::PrefillControl)?;
                let bank = self
                    .native_storage
                    .as_ref()
                    .ok_or(Error::PrefillScopeUnavailable)?;
                let mut storage =
                    RetainedStorage::prepare_original(1, controls.metadata_custody())?;
                storage.include_array(source)?;
                self.native_publications
                    .publish(storage, bank, scope, controls, None, None)
            }
        }
    }
    /// Settles and captures one actual activation inside this original work.
    ///
    /// The enclosing session must already hold the matching core text permit
    /// and original observer-aware native quote. The claim protects whole-run
    /// host destinations, not this activation's equations or native transforms.
    /// Keep this FundedWork in ordinary recovery through all failures/unwind.
    /// This method never opens another scope, retires unrelated roots, marks the
    /// full work published, or certifies completion.
    pub(in crate::composition::mlx::session::model_session) fn capture_tensor<'a, 'c>(
        &self,
        source: &Array,
        claim: CaptureTensorClaim<'a, 'c>,
        stream: &Stream,
    ) -> Result<ClaimedCaptureTensor, Error> {
        self.capture_tensor_with_completion(source, claim, stream, CaptureCompletion::Ordinary)
    }
    pub(in crate::composition::mlx::session::model_session) fn capture_tensor_with_completion<
        'a,
        'c,
    >(
        &self,
        source: &Array,
        claim: CaptureTensorClaim<'a, 'c>,
        stream: &Stream,
        completion: CaptureCompletion<'_>,
    ) -> Result<ClaimedCaptureTensor, Error> {
        self.validate_capture_completion(completion)?;
        let _activity = publication_scope::Activity::begin(&self.publishing)?;
        let mut scope_owner = publication_scope::OwnedScope::take(&self.scope)?;
        let scope = scope_owner.get_mut();
        claim
            .validate_native_scope(scope)
            .map_err(|error| Error::Other(Box::new(error)))?;
        PreparedCaptureTensor::validate_source_geometry(source, claim.geometry())
            .map_err(|error| Error::Other(Box::new(error)))?;
        PreparedCaptureTensor::validate_stream(stream)
            .map_err(|error| Error::Other(Box::new(error)))?;
        // Reserve descriptor/publication slots before retaining or evaluating.
        // No native root is dropped if a later fallible step rejects. These are
        // existing native recovery bookkeeping, not numerical allocation grants.
        self.retain(source);
        if let Some(cause) = self.take_collection_failure() {
            return Err(cause);
        }
        #[cfg(test)]
        tests::before_evaluation()?;
        drop(
            completion
                .settle(source, stream)
                .map_err(|e| Error::Other(Box::new(e)))?,
        );
        claim
            .validate_native_scope(scope)
            .map_err(|error| Error::Other(Box::new(error)))?;
        let publication = self.publish_capture_source(source, scope, completion)?;
        // Native charges attach to the backing. Keep the paid collector receipt
        // through the source pin and complete transfer without another Work slot.
        let _publication = publication;
        // Source publication alone must not hide unsettled unrelated work or
        // clear its recovery roots. Source pins attach to this very same scope
        // in the scheduled leaf and survive all subsequent transfer failures.
        #[cfg(test)]
        tests::after_source_publication()?;
        let geometry = claim.geometry();
        // The claim already binds the actual local axes and selection. Rebuilding
        // from its global admission would discard partition/window coordinates.
        // This fixed geometry copy borrows that same admission; the existing
        // claim remains the only destination and native-scope authority.
        let host = CaptureTensorHostPlan::prepare(geometry.clone())
            .map_err(|error| Error::Other(Box::new(error)))?;
        let prepared = PreparedCaptureTensor::new(source, host)
            .map_err(|error| Error::Other(Box::new(error)))?;
        prepared
            .transfer_scheduled_with_completion(
                claim,
                scope,
                stream,
                &self.roots,
                completion,
                self.original_metadata_custody().as_ref(),
            )
            .map_err(|error| match error {
                ScheduledCaptureTensorExecutionError::Mechanism(error) => {
                    Error::Other(Box::new(error))
                }
                ScheduledCaptureTensorExecutionError::Finish(error) => {
                    Error::Other(Box::new(error.into_owned_error()))
                }
            })
    }
}

#[cfg(test)]
mod tests;
