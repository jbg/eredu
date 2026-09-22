//! Original request controls around the unchanged session mutation/recovery guard.
use super::*;
use crate::backend::submission_recovery::PreparedRecovery;
use eredu_nn::workspace::{HostMetadataFunding, HostMetadataFundingError};
use eredu_runtime::working_memory::WorkingMemoryError;
use std::mem::{size_of, size_of_val};

type Prepared = PreparedRecovery<ScopeRetention, HostMetadataFunding>;

#[derive(Debug, thiserror::Error)]
enum Cause {
    #[error("original model scope preparation failed: {0}")]
    Scope(#[source] safemlx::SubmissionScopeOwnerCause),
    #[error(transparent)]
    Operation(Error),
    #[error(transparent)]
    Authority(eredu_core::SessionAuthorityError),
}

#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
struct Failure {
    #[source]
    cause: Cause,
    // BackendFailure's closed source retirement frees its shell before this account.
    funding: HostMetadataFunding,
}

fn failure(cause: Cause, funding: HostMetadataFunding) -> Error {
    failure_with_preservation(cause, funding, false)
}

fn failure_with_preservation(cause: Cause, funding: HostMetadataFunding, preserved: bool) -> Error {
    let kind = match &cause {
        Cause::Operation(error) => error
            .retained_backend_failure_kind()
            .unwrap_or(eredu_core::BackendFailureKind::Other),
        Cause::Scope(_) | Cause::Authority(_) => eredu_core::BackendFailureKind::Other,
    };
    Error::with_original_control_source(
        eredu_core::BackendFailure::new(kind, Failure { cause, funding }),
        preserved,
    )
}

fn control_bytes<T>() -> Option<usize> {
    let controls = [
        usize::try_from(submission_owner::control_bytes()?).ok()?,
        usize::try_from(Prepared::control_bytes()?).ok()?,
        size_of::<SubmissionLease>(),
        size_of::<Result<SubmissionLease, eredu_core::SessionAuthorityError>>(),
        size_of::<SessionOperation<'_>>(),
        size_of::<Result<T, Error>>(),
        size_of::<(T, SubmissionResourcesOwner, Recovery<ScopeRetention>)>(),
        size_of::<Result<(T, SubmissionResourcesOwner, Recovery<ScopeRetention>), Error>>(),
        size_of::<HostMetadataFunding>(),
        size_of::<Cause>(),
        size_of::<Failure>(),
        size_of::<eredu_core::BackendFailureKind>(),
        size_of::<eredu_core::BackendFailure>(),
        size_of::<Result<(), HostMetadataFundingError>>(),
        size_of::<bool>(),
        eredu_core::BackendFailure::source_retention_peak_bytes::<Failure>()?,
    ];
    controls
        .into_iter()
        .try_fold(size_of_val(&controls), usize::checked_add)
}

impl MlxModelSession {
    /// Unpublished parameter inspection may return a callback refusal after
    /// native work succeeds. Carry that refusal through the same completion
    /// worker before returning it; failed or unobservable native work retains
    /// the existing recovery fencing. This entry must not publish parameters,
    /// mutate decoder state or perform another semantic session transition.
    pub(in crate::composition::mlx::session) fn with_model_inspection_funded<T>(
        &mut self,
        funding: HostMetadataFunding,
        operation: impl FnOnce(&mut Executable) -> Result<T, Error>,
    ) -> Result<T, Error> {
        funding
            .reserve_metadata(size_of::<(
                HostMetadataFunding,
                Result<T, Error>,
                Result<Result<T, Error>, Error>,
            )>())
            .map_err(Error::WorkspacePlanning)?;
        let callback =
            self.with_model_operation_funded(funding.clone(), |model| Ok(operation(model)))?;
        callback.map_err(|cause| failure_with_preservation(Cause::Operation(cause), funding, true))
    }

    /// Cold loan of the actual idle executable for paired-source authentication.
    /// This neither acquires an ordinary lease nor reconstructs source storage.
    pub(crate) fn original_model_source(&self) -> Result<&Executable, WorkingMemoryError> {
        if self.poison.get() {
            return Err(WorkingMemoryError::ExecutionFenced);
        }
        self.authority
            .try_borrow()
            .map_err(|_| WorkingMemoryError::ReservedWorkActive)?
            .require_idle()
            .map_err(|_| WorkingMemoryError::ReservedWorkActive)?;
        if !self.payload.is_exclusive() {
            return Err(WorkingMemoryError::ReservedWorkActive);
        }
        Ok(&self.payload.model)
    }

    /// Begin outside native phase arenas. The enclosing scope retains the actual
    /// executable and move-only session lease; each child phase owns its numerical
    /// reservation, native quotas, publication and independent completion proof.
    pub(crate) fn with_model_operation_funded<T>(
        &mut self,
        funding: HostMetadataFunding,
        operation: impl FnOnce(&mut Executable) -> Result<T, Error>,
    ) -> Result<T, Error> {
        crate::backend::submission_recovery::reap();
        crate::backend::ordinary_retirement::reclaim();
        self.original_model_source()
            .map_err(Error::PrefillControl)?;
        funding
            .reserve_metadata(
                control_bytes::<T>()
                    .ok_or(Error::WorkspacePlanning(HostMetadataFundingError::Overflow))?,
            )
            .map_err(Error::WorkspacePlanning)?;
        // From this point one returned typed cause shell and all native/host
        // scope construction are prepaid. Refusal never enters ordinary memory.
        let result = (|| {
            let lease = self
                .authority
                .try_borrow_mut()
                .map_err(|_| {
                    Cause::Operation(Error::PrefillControl(
                        WorkingMemoryError::ReservedWorkActive,
                    ))
                })?
                .begin_submission()
                .map_err(Cause::Authority)?;
            let owner = SubmissionResources::with_operation_funding(
                lease,
                Rc::clone(&self.poison),
                SubmissionPurpose::OriginalModel,
                Some(funding.clone()),
            );
            let prepared = Prepared::new(owner.ticket(), funding.clone())
                .map_err(|error| Cause::Scope(error.cause))?;
            let recovery = prepared
                .try_begin()
                .map_err(|error| Cause::Scope(error.cause))?;
            let mut guard = SessionOperation {
                session: self,
                owner,
                recovery: Some(recovery),
                handed_off: false,
                token_validations: Default::default(),
            };
            let result = operation(guard.model());
            let (value, owner, recovery) = guard.finish(result).map_err(Cause::Operation)?;
            complete_model_operation(value, owner, recovery).map_err(Cause::Operation)
        })();
        result.map_err(|cause| failure(cause, funding))
    }
}
