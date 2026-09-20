//! One original encoding operation, independent of source C and generation input I.
use super::{Allowance, OriginalTokenizer, WorkingMemoryError, WorkingMemoryPool};
use eredu_core::{BackendFailure, BackendFailureKind};
use eredu_text::tokenizer_storage::{
    EncodeIdsError, EncodeIdsFailure, EncodeIdsPlan, EncodedTokenIds,
};
use std::{fmt, mem::size_of};

/// Move-only original encoding output. It lends IDs to the existing TokenIds input
/// plan; that request admits its own destination while this owner remains charged.
/// No storage adoption, raw Vec, refill or clone operation exists.
/// ```compile_fail
/// use eredu_runtime::working_memory::OriginalEncodedTokenIds;
/// fn copy(ids: OriginalEncodedTokenIds) { let _ = ids.clone(); }
/// ```
#[derive(Debug)]
pub struct OriginalEncodedTokenIds {
    ids: EncodedTokenIds,
    source: OriginalTokenizer,
    allowance: Allowance,
}
impl OriginalEncodedTokenIds {
    /// Borrows completed IDs without allocating or transferring their destination.
    pub fn ids(&self) -> &[u32] {
        self.ids.ids()
    }
    /// Original E only; source C is retained separately and is never requoted.
    pub fn original_bytes(&self) -> u64 {
        self.allowance.bytes()
    }
    /// Exact source identity, without exporting the retained source owner.
    pub fn matches_source(&self, source: &OriginalTokenizer) -> bool {
        self.source.same_source(source)
    }
}
#[derive(Debug)]
enum Cause {
    Profile(EncodeIdsError),
    Accounting(WorkingMemoryError),
    Encoding(EncodeIdsFailure),
}
/// One actual operation failure. Its partial/completed destinations retire before
/// the original source lease and E. Preflight/admission rejections retain neither.
/// ```compile_fail
/// use eredu_runtime::working_memory::OriginalTokenizerEncodeError;
/// fn retry(error: OriginalTokenizerEncodeError) { let _ = error.into_plan(); }
/// ```
#[derive(Debug)]
pub struct OriginalTokenizerEncodeError {
    cause: Cause,
    settlement: Option<WorkingMemoryError>,
    completed: Option<EncodedTokenIds>,
    source: Option<OriginalTokenizer>,
    allowance: Option<Allowance>,
}
impl OriginalTokenizerEncodeError {
    fn rejected(cause: Cause) -> Self {
        Self {
            cause,
            settlement: None,
            completed: None,
            source: None,
            allowance: None,
        }
    }
    /// Original E retained here, or zero if rejection preceded admission.
    pub fn retained_bytes(&self) -> u64 {
        self.allowance.as_ref().map_or(0, Allowance::bytes)
    }
    /// Actual profile/geometry diagnostic before any reserve.
    pub fn profile_failure(&self) -> Option<&EncodeIdsError> {
        match &self.cause {
            Cause::Profile(error) => Some(error),
            _ => None,
        }
    }
    /// Original upstream encoding failure; the operation reservation stays held.
    pub fn encoding_failure(&self) -> Option<&EncodeIdsFailure> {
        match &self.cause {
            Cause::Encoding(error) => Some(error),
            _ => None,
        }
    }
    /// Original admission or terminal conservative accounting failure.
    pub fn accounting_failure(&self) -> Option<&WorkingMemoryError> {
        self.settlement.as_ref().or_else(|| match &self.cause {
            Cause::Accounting(error) => Some(error),
            _ => None,
        })
    }
    /// Whether failure retains the exact original source lease.
    pub fn matches_source(&self, source: &OriginalTokenizer) -> bool {
        self.source
            .as_ref()
            .is_some_and(|actual| actual.same_source(source))
    }
    /// Completed IDs retained when terminal accounting rejects successful encoding.
    pub fn completed_ids(&self) -> Option<&[u32]> {
        self.completed.as_ref().map(EncodedTokenIds::ids)
    }
    /// Direct neutral erasure: the core source allocation retires before this owner.
    pub fn into_backend_failure(self) -> BackendFailure {
        let kind = match self.accounting_failure() {
            Some(WorkingMemoryError::IdentityMismatch | WorkingMemoryError::Poisoned) => {
                BackendFailureKind::InvalidSession
            }
            Some(WorkingMemoryError::UnknownBound) => BackendFailureKind::Unsupported,
            _ if matches!(self.profile_failure(), Some(EncodeIdsError::Overflow)) => {
                BackendFailureKind::ResourceExhausted
            }
            _ if self.profile_failure().is_some() => BackendFailureKind::Unsupported,
            _ if self
                .encoding_failure()
                .is_some_and(|e| matches!(e.cause(), EncodeIdsError::Upstream(_))) =>
            {
                BackendFailureKind::InvalidInput
            }
            _ => BackendFailureKind::ResourceExhausted,
        };
        BackendFailure::new(kind, self)
    }
}
impl fmt::Display for OriginalTokenizerEncodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.cause {
            Cause::Profile(e) => fmt::Display::fmt(e, f),
            Cause::Accounting(e) => fmt::Display::fmt(e, f),
            Cause::Encoding(e) => fmt::Display::fmt(e, f),
        }
    }
}
impl std::error::Error for OriginalTokenizerEncodeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.cause {
            Cause::Profile(e) => Some(e),
            Cause::Accounting(e) => Some(e),
            Cause::Encoding(e) => Some(e),
        }
    }
}
fn required(plan: &EncodeIdsPlan<'_>) -> Result<u64, WorkingMemoryError> {
    let bytes = [
        plan.required_bytes(),
        super::OriginalTextSourceError::encode_controls().ok_or(WorkingMemoryError::Overflow)?,
        size_of::<OriginalTokenizer>(),
        size_of::<Option<OriginalTokenizer>>(),
        size_of::<Allowance>(),
        size_of::<Option<Allowance>>(),
        size_of::<Result<Allowance, WorkingMemoryError>>(),
        size_of::<Result<(), WorkingMemoryError>>(),
        size_of::<Cause>(),
        size_of::<Option<WorkingMemoryError>>(),
        size_of::<OriginalEncodedTokenIds>(),
        size_of::<Option<EncodedTokenIds>>(),
        size_of::<OriginalTokenizerEncodeError>(),
        size_of::<Result<OriginalEncodedTokenIds, OriginalTokenizerEncodeError>>(),
        size_of::<Result<OriginalEncodedTokenIds, BackendFailure>>(),
        size_of::<Result<u64, WorkingMemoryError>>(),
        size_of::<(
            Result<EncodedTokenIds, EncodeIdsFailure>,
            Result<(), WorkingMemoryError>,
        )>(),
        BackendFailure::source_retention_peak_bytes::<OriginalTokenizerEncodeError>()
            .ok_or(WorkingMemoryError::Overflow)?,
    ]
    .into_iter()
    .try_fold(0usize, usize::checked_add)
    .ok_or(WorkingMemoryError::Overflow)?;
    u64::try_from(bytes).map_err(|_| WorkingMemoryError::Overflow)
}
impl WorkingMemoryPool {
    /// Borrows the exact source and text to derive E without allocating or granting it.
    pub fn tokenizer_encode_required_bytes(
        source: &OriginalTokenizer,
        input: &str,
        add_special_tokens: bool,
    ) -> Result<u64, OriginalTokenizerEncodeError> {
        let plan = EncodeIdsPlan::prepare(&source.payload().model, input, add_special_tokens)
            .map_err(|e| OriginalTokenizerEncodeError::rejected(Cause::Profile(e)))?;
        required(&plan).map_err(|e| OriginalTokenizerEncodeError::rejected(Cause::Accounting(e)))
    }
    /// Estimates dependency workspace, admits once, and performs upstream encoding.
    /// The borrowed input may retire on return; the immutable source stays under its C lease.
    pub fn encode_tokenizer_ids(
        &self,
        source: &OriginalTokenizer,
        input: &str,
        add_special_tokens: bool,
    ) -> Result<OriginalEncodedTokenIds, OriginalTokenizerEncodeError> {
        self.encode_tokenizer_ids_with(source, input, add_special_tokens, |plan| plan, || {}, || {})
    }
    fn encode_tokenizer_ids_with<'a>(
        &self,
        source: &'a OriginalTokenizer,
        input: &'a str,
        add_special_tokens: bool,
        configure: impl FnOnce(EncodeIdsPlan<'a>) -> EncodeIdsPlan<'a>,
        after_admission: impl FnOnce(),
        after_encoding: impl FnOnce(),
    ) -> Result<OriginalEncodedTokenIds, OriginalTokenizerEncodeError> {
        source
            .validate_pool(self)
            .map_err(|e| OriginalTokenizerEncodeError::rejected(Cause::Accounting(e)))?;
        let plan = EncodeIdsPlan::prepare(&source.payload().model, input, add_special_tokens)
            .map_err(|e| OriginalTokenizerEncodeError::rejected(Cause::Profile(e)))?;
        // Private test configuration cannot change source ownership.
        let plan = configure(plan);
        let bytes = required(&plan)
            .map_err(|e| OriginalTokenizerEncodeError::rejected(Cause::Accounting(e)))?;
        let mut allowance = self
            .admit_source_compiler(bytes)
            .map_err(|e| OriginalTokenizerEncodeError::rejected(Cause::Accounting(e)))?;
        // No fallible operation or allocation separates acceptance from the guard.
        let held_source = source.clone();
        after_admission();
        // Upstream owns temporary workspace retirement. The complete Encoding
        // output or original error stays under the operation reservation; that
        // reservation is an estimate and remains held through settlement errors.
        let encoded = plan.encode();
        after_encoding();
        let settlement = allowance.end_compilation();
        match (encoded, settlement) {
            (Ok(ids), Ok(())) => Ok(OriginalEncodedTokenIds {
                ids,
                source: held_source,
                allowance,
            }),
            (Ok(ids), Err(error)) => Err(OriginalTokenizerEncodeError {
                cause: Cause::Accounting(error),
                settlement: None,
                completed: Some(ids),
                source: Some(held_source),
                allowance: Some(allowance),
            }),
            (Err(error), settlement) => Err(OriginalTokenizerEncodeError {
                cause: Cause::Encoding(error),
                settlement: settlement.err(),
                completed: None,
                source: Some(held_source),
                allowance: Some(allowance),
            }),
        }
    }
}
#[cfg(test)]
mod tests;
