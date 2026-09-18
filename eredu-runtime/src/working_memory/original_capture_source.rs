//! Fresh immutable capture declarations under the shared original source compiler.
use super::original_declaration_source::Account;
use super::{WorkingMemoryError, WorkingMemoryPool, loaded_decode_source::Allowance};
use eredu_core::{
    HostPreparationAuthority,
    capture::{CapturePlanCopyError, PreparedCapturePlanCopy, SharedCapturePlan},
};
use std::mem::{size_of, size_of_val};

/// The freshly copied immutable declaration and its actual original C account.
/// Every shared-plan alias retains that account internally; neither the caller's
/// source buffers nor a caller-supplied host token can construct this proof.
#[derive(Debug, Clone)]
pub struct OriginalCaptureSource {
    source: SharedCapturePlan,
    account: Account,
}
impl OriginalCaptureSource {
    /// Borrow the exact freshly constructed shared plan, with its real custody.
    pub fn plan(&self) -> &SharedCapturePlan {
        &self.source
    }
    /// Fixed borrowed source/account validation controls; no allocation is made.
    pub fn validation_control_bytes() -> Option<usize> {
        [
            size_of::<&Self>(),
            size_of::<&WorkingMemoryPool>(),
            size_of::<std::sync::MutexGuard<'_, Allowance>>(),
            size_of::<WorkingMemoryError>(),
            size_of::<Result<(), WorkingMemoryError>>(),
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)
    }
    /// Check the exact source domain and its private owner mutex. Request/pool
    /// admission health is validated separately by the consuming request. This
    /// grants no publication, quote or execution.
    pub fn validate_pool(&self, pool: &WorkingMemoryPool) -> Result<(), WorkingMemoryError> {
        self.account.validate(pool)
    }
    /// Identity of the physical immutable source, never a semantic digest.
    pub fn same_source(&self, other: &Self) -> bool {
        self.source.same_storage(&other.source)
    }
}
#[derive(Debug, thiserror::Error)]
enum Cause {
    #[error("{0}")]
    Memory(#[from] WorkingMemoryError),
    #[error("{0}")]
    Copy(#[from] CapturePlanCopyError),
    #[error(transparent)]
    Admission(#[from] eredu_core::capture::CaptureError),
    #[error(transparent)]
    Funding(#[from] eredu_core::HostMetadataFundingError),
    #[error(transparent)]
    Backend(#[from] eredu_core::BackendFailure),
}
/// A typed refusal or failed copy retaining its full original compiler charge.
/// No partial source or account can be extracted or reused to retry a copy.
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
pub struct OriginalCaptureSourceError {
    #[source]
    cause: Cause,
    settlement: Option<WorkingMemoryError>,
    completed: Option<SharedCapturePlan>,
    account: Option<Account>,
    admission_funding: Option<eredu_core::HostMetadataFunding>,
}
impl OriginalCaptureSourceError {
    /// Preserve a neutral mechanism failure after its enclosing caller paid
    /// error retention controls. This constructs no diagnostic or source box.
    pub fn backend(
        cause: eredu_core::BackendFailure,
        funding: &eredu_core::HostMetadataFunding,
    ) -> Self {
        let mut error = Self::refused(cause);
        error.admission_funding = Some(funding.clone());
        error
    }
    /// Fixed rejection before entering any source producer.
    pub fn rejected(cause: WorkingMemoryError) -> Self {
        Self::refused(cause)
    }

    fn refused(cause: impl Into<Cause>) -> Self {
        Self {
            cause: cause.into(),
            settlement: None,
            completed: None,
            account: None,
            admission_funding: None,
        }
    }
}
impl WorkingMemoryPool {
    /// Compile a borrowed raw declaration using the same paid semantic validator,
    /// then construct its independent original C owner. Temporary admission
    /// storage remains charged to `funding` until the enclosing account retires.
    pub fn compile_capture_declaration(
        &self,
        plan: &eredu_core::capture::CapturePlan,
        catalog: &eredu_core::ObservationCatalog,
        support: &eredu_core::ObservationSupportReport,
        request: eredu_core::capture::CaptureRequestShape,
        origin: eredu_core::capture::CaptureTextOrigin,
        funding: &eredu_core::HostMetadataFunding,
    ) -> Result<OriginalCaptureSource, OriginalCaptureSourceError> {
        let result = (|| -> Result<_, OriginalCaptureSourceError> {
            let controls = [
                size_of::<eredu_core::capture::AdmittedCapturePlan>(),
                size_of::<OriginalCaptureSource>(),
                size_of::<OriginalCaptureSourceError>(),
                size_of::<Result<OriginalCaptureSource, OriginalCaptureSourceError>>(),
                PreparedCapturePlanCopy::inspection_control_bytes().ok_or_else(|| {
                    Self::declaration_failure(WorkingMemoryError::Overflow, funding)
                })?,
            ];
            let bytes = controls
                .into_iter()
                .try_fold(size_of_val(&controls), usize::checked_add)
                .ok_or_else(|| Self::declaration_failure(WorkingMemoryError::Overflow, funding))?;
            funding
                .reserve_metadata(bytes)
                .map_err(OriginalCaptureSourceError::refused)?;
            let owned = plan
                .copy_with_funding(funding)
                .map_err(OriginalCaptureSourceError::refused)?;
            let admitted = owned
                .admit_with_text_origin_and_funding(
                    catalog,
                    support,
                    &support.capture,
                    request,
                    origin,
                    funding,
                )
                .map_err(OriginalCaptureSourceError::refused)?;
            let copy = PreparedCapturePlanCopy::inspect(&admitted)
                .map_err(OriginalCaptureSourceError::refused)?;
            self.compile_capture_source(copy)
        })();
        result.map_err(|mut error| {
            error.admission_funding = Some(funding.clone());
            error
        })
    }
    fn declaration_failure(
        cause: WorkingMemoryError,
        funding: &eredu_core::HostMetadataFunding,
    ) -> OriginalCaptureSourceError {
        let mut error = OriginalCaptureSourceError::refused(cause);
        error.admission_funding = Some(funding.clone());
        error
    }

    /// Exact source destination and closed-account controls. The borrowed plan
    /// determines every copied String/Vec; no caller amount is accepted.
    pub fn capture_source_required_bytes(
        plan: &PreparedCapturePlanCopy<'_>,
    ) -> Result<u64, WorkingMemoryError> {
        let controls = [
            Account::control_bytes().ok_or(WorkingMemoryError::Overflow)?,
            size_of::<OriginalCaptureSource>(),
            size_of::<OriginalCaptureSourceError>(),
            size_of::<Result<OriginalCaptureSource, OriginalCaptureSourceError>>(),
            size_of::<Result<SharedCapturePlan, CapturePlanCopyError>>(),
        ];
        controls
            .into_iter()
            .try_fold(
                plan.required_bytes()
                    .checked_add(size_of_val(&controls))
                    .ok_or(WorkingMemoryError::Overflow)?,
                usize::checked_add,
            )
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or(WorkingMemoryError::Overflow)
    }
    /// Reserve before the first destination/account allocation, then consume
    /// the same closed copy worker. Existing active request ceilings apply.
    /// This source compiler creates no native value and no numerical occurrence.
    pub fn compile_capture_source(
        &self,
        plan: PreparedCapturePlanCopy<'_>,
    ) -> Result<OriginalCaptureSource, OriginalCaptureSourceError> {
        let bytes = Self::capture_source_required_bytes(&plan)
            .map_err(OriginalCaptureSourceError::refused)?;
        let account = Account::admit(self, bytes).map_err(OriginalCaptureSourceError::refused)?;
        let host = HostPreparationAuthority::retain(account.clone());
        match plan.copy(host) {
            Err(cause) => {
                let settlement = account.finish().err();
                Err(OriginalCaptureSourceError {
                    cause: cause.into(),
                    settlement,
                    completed: None,
                    account: Some(account),
                    admission_funding: None,
                })
            }
            Ok(source) => match account.finish() {
                Ok(()) => Ok(OriginalCaptureSource { source, account }),
                Err(cause) => Err(OriginalCaptureSourceError {
                    cause: cause.into(),
                    settlement: None,
                    completed: Some(source),
                    account: Some(account),
                    admission_funding: None,
                }),
            },
        }
    }
}

#[cfg(test)]
mod tests;
