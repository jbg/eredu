//! Original exact file bytes coexist with a fresh aggregate in the same pool.
use super::{
    OriginalTokenizer, OriginalTokenizerError, TokenizerPlan, WorkingMemoryError, WorkingMemoryPool,
};
use eredu_checkpoint::artifact::{ArtifactFileReadFailure, PreparedArtifactFileRead};
use eredu_core::{BackendFailure, BackendFailureKind};
use eredu_text::tokenizer_storage::TokenizerSourceError;
use std::{collections::TryReserveError, fmt, mem::size_of};

use crate::working_memory::original_file::{self, Input};
#[derive(Debug)]
enum Cause {
    Accounting(WorkingMemoryError),
    Reserve(TryReserveError),
    Read(ArtifactFileReadFailure),
    Plan(TokenizerSourceError),
    Compile(OriginalTokenizerError),
}
/// Terminal file/aggregate failure retaining the actual input and compiler prefix.
/// Opening/path preparation precedes this boundary. There is no consuming cause,
/// file, byte-buffer or allowance extraction and no replayable read capability.
///
/// ```compile_fail
/// # use eredu_runtime::working_memory::OriginalTokenizerFileError;
/// fn retry(error: OriginalTokenizerFileError) { let _ = error.into_bytes(); }
/// ```
#[derive(Debug)]
pub struct OriginalTokenizerFileError {
    cause: Cause,
    settlement: Option<WorkingMemoryError>,
    input: Option<Input>,
}
impl OriginalTokenizerFileError {
    fn from_read(error: original_file::Failure) -> Self {
        let cause = match error.cause {
            original_file::Cause::Accounting(error) => Cause::Accounting(error),
            original_file::Cause::Reserve(error) => Cause::Reserve(error),
            original_file::Cause::Read(error) => Cause::Read(error),
        };
        Self {
            cause,
            settlement: error.settlement,
            input: error.input,
        }
    }
    /// Complete original input allowance, including its error/control envelope.
    pub fn input_bytes(&self) -> u64 {
        self.input
            .as_ref()
            .map_or(0, |input| input.allowance.bytes())
    }
    /// Actual file bytes written before failure (not initialized trailing bytes).
    pub fn filled_bytes(&self) -> usize {
        self.input.as_ref().map_or(0, |input| input.filled)
    }
    /// Actual backing capacity, including a successfully reserved partial input.
    pub fn input_capacity(&self) -> usize {
        self.input
            .as_ref()
            .map_or(0, |input| input.bytes.capacity())
    }
    /// Borrows the unchanged failed consuming read.
    pub fn read_failure(&self) -> Option<&ArtifactFileReadFailure> {
        match &self.cause {
            Cause::Read(error) => Some(error),
            _ => None,
        }
    }
    /// Borrows the root/source planning diagnostic before C admission.
    pub fn planning_failure(&self) -> Option<&TokenizerSourceError> {
        match &self.cause {
            Cause::Plan(error) => Some(error),
            _ => None,
        }
    }
    /// Borrows the actual fresh aggregate failure, with its independent C custody.
    pub fn compiler_failure(&self) -> Option<&OriginalTokenizerError> {
        match &self.cause {
            Cause::Compile(error) => Some(error),
            _ => None,
        }
    }
    /// Original input or aggregate accounting failure, if present.
    pub fn accounting_failure(&self) -> Option<&WorkingMemoryError> {
        self.settlement.as_ref().or_else(|| match &self.cause {
            Cause::Accounting(error) => Some(error),
            Cause::Compile(error) => error.accounting_failure(),
            _ => None,
        })
    }
    /// Classifies the actual cause without constructing another owning wrapper.
    pub fn kind(&self) -> BackendFailureKind {
        if let Some(error) = self.accounting_failure() {
            return match error {
                WorkingMemoryError::Poisoned | WorkingMemoryError::IdentityMismatch => {
                    BackendFailureKind::InvalidSession
                }
                WorkingMemoryError::UnknownBound => BackendFailureKind::Unsupported,
                WorkingMemoryError::ReservedWorkActive => BackendFailureKind::Busy,
                _ => BackendFailureKind::ResourceExhausted,
            };
        }
        match &self.cause {
            Cause::Read(_) => BackendFailureKind::Io,
            Cause::Plan(_) => BackendFailureKind::Unsupported,
            _ => BackendFailureKind::ResourceExhausted,
        }
    }
    /// Moves the one actual failure into core's allocation-retiring envelope.
    pub fn into_backend_failure(self) -> BackendFailure {
        BackendFailure::new(self.kind(), self)
    }
}
impl fmt::Display for OriginalTokenizerFileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.cause {
            Cause::Accounting(e) => fmt::Display::fmt(e, f),
            Cause::Reserve(e) => fmt::Display::fmt(e, f),
            Cause::Read(e) => fmt::Display::fmt(e, f),
            Cause::Plan(e) => fmt::Display::fmt(e, f),
            Cause::Compile(e) => fmt::Display::fmt(e, f),
        }
    }
}
impl std::error::Error for OriginalTokenizerFileError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(match &self.cause {
            Cause::Accounting(e) => e,
            Cause::Reserve(e) => e,
            Cause::Read(e) => e,
            Cause::Plan(e) => e,
            Cause::Compile(e) => e,
        })
    }
}

impl WorkingMemoryPool {
    /// Original input bytes and concrete finite controls, before any read reserve.
    /// This is I only; the root-derived C comparison occurs with I still retained.
    pub fn tokenizer_file_required_bytes(
        read: &PreparedArtifactFileRead,
    ) -> Result<u64, WorkingMemoryError> {
        let controls = [
            super::OriginalTextSourceError::tokenizer_controls()
                .ok_or(WorkingMemoryError::Overflow)?,
            original_file::control_bytes().ok_or(WorkingMemoryError::Overflow)?,
            size_of::<bool>(), // closed generation-domain constructor selection
            size_of::<TokenizerPlan<'_>>(),
            size_of::<Result<TokenizerPlan<'_>, TokenizerSourceError>>(),
            size_of::<Result<OriginalTokenizer, OriginalTokenizerError>>(),
            size_of::<Cause>(),
            size_of::<OriginalTokenizerFileError>(),
            size_of::<Result<OriginalTokenizer, OriginalTokenizerFileError>>(),
            size_of::<Result<OriginalTokenizer, BackendFailure>>(),
            BackendFailure::source_retention_peak_bytes::<OriginalTokenizerFileError>()
                .ok_or(WorkingMemoryError::Overflow)?,
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)
        .ok_or(WorkingMemoryError::Overflow)?;
        read.byte_len()
            .checked_add(controls)
            .and_then(|n| u64::try_from(n).ok())
            .ok_or(WorkingMemoryError::Overflow)
    }
    /// Consumes an exact prepared file, admitting before its one buffer reserve.
    /// The read becomes idle before fresh construction, while its entire I charge
    /// still reduces C's available capacity. Successful construction destroys the
    /// input before returning the unchanged closed aggregate owner.
    pub fn compile_tokenizer_file(
        &self,
        read: PreparedArtifactFileRead,
    ) -> Result<OriginalTokenizer, OriginalTokenizerFileError> {
        self.compile_tokenizer_file_with(
            read,
            |_| {},
            || {},
            |pool, plan| pool.compile_tokenizer(plan),
        )
    }
    fn compile_tokenizer_file_with(
        &self,
        read: PreparedArtifactFileRead,
        after_admission: impl FnOnce(&mut usize),
        after_read: impl FnOnce(),
        compile: impl FnOnce(
            &Self,
            TokenizerPlan<'_>,
        ) -> Result<OriginalTokenizer, OriginalTokenizerError>,
    ) -> Result<OriginalTokenizer, OriginalTokenizerFileError> {
        self.compile_tokenizer_file_inner(read, false, after_admission, after_read, compile)
    }
    /// Fresh generation source from exact original file bytes. The selected
    /// canonical domain is planned before C admission; no existing C is promoted.
    pub fn compile_tokenizer_file_for_generation(
        &self,
        read: PreparedArtifactFileRead,
    ) -> Result<OriginalTokenizer, OriginalTokenizerFileError> {
        self.compile_tokenizer_file_inner(
            read,
            true,
            |_| {},
            || {},
            |pool, plan| pool.compile_tokenizer(plan),
        )
    }
    fn compile_tokenizer_file_inner(
        &self,
        read: PreparedArtifactFileRead,
        generation: bool,
        after_admission: impl FnOnce(&mut usize),
        after_read: impl FnOnce(),
        compile: impl FnOnce(
            &Self,
            TokenizerPlan<'_>,
        ) -> Result<OriginalTokenizer, OriginalTokenizerError>,
    ) -> Result<OriginalTokenizer, OriginalTokenizerFileError> {
        let input = original_file::read(
            self,
            read,
            original_file::Destination::Tokenizer,
            after_admission,
        )
        .map_err(OriginalTokenizerFileError::from_read)?;
        after_read();
        let plan = match TokenizerPlan::prepare_json(&input.bytes).and_then(|plan| {
            if generation {
                plan.with_generation_domain()
            } else {
                Ok(plan)
            }
        }) {
            Ok(plan) => plan,
            Err(error) => {
                return Err(OriginalTokenizerFileError {
                    cause: Cause::Plan(error),
                    settlement: None,
                    input: Some(input),
                })
            }
        };
        match compile(self, plan) {
            Ok(source) => {
                drop(input);
                Ok(source)
            }
            Err(error) => Err(OriginalTokenizerFileError {
                cause: Cause::Compile(error),
                settlement: None,
                input: Some(input),
            }),
        }
    }
}

#[cfg(all(test, unix))]
mod tests;
