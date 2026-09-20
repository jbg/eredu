//! Original exact file bytes coexist with a fresh aggregate in the same pool.
use super::{
    OriginalTokenizer, OriginalTokenizerError, TokenizerPlan, WorkingMemoryError, WorkingMemoryPool,
};
use eredu_checkpoint::artifact::{ArtifactFileReadFailure, PreparedArtifactFileRead};
use eredu_core::{BackendFailure, BackendFailureKind};
use eredu_text::tokenizer_storage::TokenizerSourceError;
use std::{collections::TryReserveError, fmt, mem::size_of};

use crate::working_memory::original_file::{self, Input};
/// Actual immutable source form for fresh aggregate compilation.
/// A retained configuration is serialized completely; no prior graph is adopted.
pub enum OriginalTokenizerInput<'a> {
    /// A consumed exact file, read once under its original input allowance.
    File(PreparedArtifactFileRead),
    /// Complete selected configuration already retained by the loaded model.
    Configuration(&'a eredu_text::tokenizer::Tokenizer),
}

impl fmt::Debug for OriginalTokenizerInput<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::File(read) => f.debug_tuple("File").field(read).finish(),
            Self::Configuration(_) => f.write_str("Configuration(retained complete tokenizer)"),
        }
    }
}

#[derive(Debug)]
enum Cause {
    Accounting(WorkingMemoryError),
    Reserve(TryReserveError),
    Read(ArtifactFileReadFailure),
    Plan(TokenizerSourceError),
    Compile(OriginalTokenizerError),
    Serialize(serde_json::Error, Option<WorkingMemoryError>),
}
/// Terminal file/aggregate failure retaining the actual input and compiler prefix.
/// Opening/path preparation precedes this boundary. There is no consuming cause,
/// file, byte-buffer or allowance extraction and no replayable read capability.
///
/// ```compile_fail
/// # use eredu_runtime::working_memory::OriginalTokenizerInputError;
/// fn retry(error: OriginalTokenizerInputError) { let _ = error.into_bytes(); }
/// ```
#[derive(Debug)]
pub struct OriginalTokenizerInputError {
    cause: Cause,
    settlement: Option<WorkingMemoryError>,
    input: Option<Input>,
}
impl OriginalTokenizerInputError {
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
            Cause::Serialize(_, accounting) => accounting.as_ref(),
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
            Cause::Serialize(_, _) => BackendFailureKind::InvalidInput,
            _ => BackendFailureKind::ResourceExhausted,
        }
    }
    /// Moves the one actual failure into core's allocation-retiring envelope.
    pub fn into_backend_failure(self) -> BackendFailure {
        BackendFailure::new(self.kind(), self)
    }
}
impl fmt::Display for OriginalTokenizerInputError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.cause {
            Cause::Accounting(e) => fmt::Display::fmt(e, f),
            Cause::Reserve(e) => fmt::Display::fmt(e, f),
            Cause::Read(e) => fmt::Display::fmt(e, f),
            Cause::Plan(e) => fmt::Display::fmt(e, f),
            Cause::Compile(e) => fmt::Display::fmt(e, f),
            Cause::Serialize(_, Some(e)) => fmt::Display::fmt(e, f),
            Cause::Serialize(e, None) => fmt::Display::fmt(e, f),
        }
    }
}
impl std::error::Error for OriginalTokenizerInputError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(match &self.cause {
            Cause::Accounting(e) => e,
            Cause::Reserve(e) => e,
            Cause::Read(e) => e,
            Cause::Plan(e) => e,
            Cause::Compile(e) => e,
            Cause::Serialize(_, Some(e)) => e,
            Cause::Serialize(e, None) => e,
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
            super::OriginalTokenizerSourceError::tokenizer_controls()
                .ok_or(WorkingMemoryError::Overflow)?,
            original_file::control_bytes().ok_or(WorkingMemoryError::Overflow)?,
            size_of::<bool>(), // closed generation-domain constructor selection
            size_of::<TokenizerPlan<'_>>(),
            size_of::<Result<TokenizerPlan<'_>, TokenizerSourceError>>(),
            size_of::<Result<OriginalTokenizer, OriginalTokenizerError>>(),
            size_of::<Cause>(),
            size_of::<OriginalTokenizerInputError>(),
            size_of::<Result<OriginalTokenizer, OriginalTokenizerInputError>>(),
            size_of::<Result<OriginalTokenizer, BackendFailure>>(),
            BackendFailure::source_retention_peak_bytes::<OriginalTokenizerInputError>()
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
    ) -> Result<OriginalTokenizer, OriginalTokenizerInputError> {
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
    ) -> Result<OriginalTokenizer, OriginalTokenizerInputError> {
        self.compile_tokenizer_file_inner(read, false, after_admission, after_read, compile)
    }
    /// Construct a fresh generation source from its actual file or retained
    /// complete configuration. Both converge on the same aggregate compiler.
    pub fn compile_tokenizer_source_for_generation(
        &self,
        input: OriginalTokenizerInput<'_>,
    ) -> Result<OriginalTokenizer, OriginalTokenizerInputError> {
        self.compile_tokenizer_source_with_headroom(input, None)
    }
    /// Uses explicit headroom for upstream serialization scratch. The output
    /// buffer is funded separately as it grows. This is an estimate, not a
    /// dependency memory ceiling; file input does not use serialization scratch.
    pub fn compile_tokenizer_source_for_generation_with_serialization_headroom(
        &self,
        input: OriginalTokenizerInput<'_>,
        headroom: usize,
    ) -> Result<OriginalTokenizer, OriginalTokenizerInputError> {
        self.compile_tokenizer_source_with_headroom(input, Some(headroom))
    }
    fn compile_tokenizer_source_with_headroom(
        &self,
        input: OriginalTokenizerInput<'_>,
        headroom: Option<usize>,
    ) -> Result<OriginalTokenizer, OriginalTokenizerInputError> {
        match input {
            OriginalTokenizerInput::File(read) => self.compile_tokenizer_file_inner(
                read,
                true,
                |_| {},
                || {},
                |pool, plan| pool.compile_tokenizer(plan),
            ),
            OriginalTokenizerInput::Configuration(source) => {
                let input = self.serialize_tokenizer_configuration(source, headroom)?;
                self.compile_tokenizer_input_bytes(
                    input,
                    true,
                    source.get_encode_special_tokens(),
                    |pool, plan| pool.compile_tokenizer(plan),
                )
            }
        }
    }
    fn serialize_tokenizer_configuration(
        &self,
        source: &eredu_text::tokenizer::Tokenizer,
        headroom: Option<usize>,
    ) -> Result<Input, OriginalTokenizerInputError> {
        use std::io;
        struct Destination {
            input: Input,
            accounting: Option<WorkingMemoryError>,
            reserve: Option<TryReserveError>,
        }
        impl io::Write for Destination {
            fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
                if self.accounting.is_some() || self.reserve.is_some() {
                    return Err(io::ErrorKind::OutOfMemory.into());
                }
                let needed = self
                    .input
                    .bytes
                    .len()
                    .checked_add(bytes.len())
                    .ok_or_else(|| {
                        self.accounting = Some(WorkingMemoryError::Overflow);
                        io::Error::from(io::ErrorKind::OutOfMemory)
                    })?;
                if needed > self.input.bytes.capacity() {
                    let capacity = needed
                        .max(self.input.bytes.capacity().saturating_mul(2))
                        .max(1024);
                    // During reallocation the old allocation can coexist with
                    // the replacement. Reserve the full new capacity and keep
                    // that conservative charge until the input owner retires.
                    let charge = u64::try_from(capacity)
                        .map_err(|_| WorkingMemoryError::Overflow)
                        .and_then(|bytes| self.input.allowance.reserve_more(bytes));
                    if let Err(error) = charge {
                        self.accounting = Some(error);
                        return Err(io::ErrorKind::OutOfMemory.into());
                    }
                    if let Err(error) = self
                        .input
                        .bytes
                        .try_reserve_exact(capacity - self.input.bytes.len())
                    {
                        self.reserve = Some(error);
                        return Err(io::ErrorKind::OutOfMemory.into());
                    }
                }
                self.input.bytes.extend_from_slice(bytes);
                self.input.filled = self.input.bytes.len();
                Ok(bytes.len())
            }
            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }
        let rejected = |error| OriginalTokenizerInputError {
            cause: Cause::Accounting(error),
            settlement: None,
            input: None,
        };
        // Public serialization may copy merge spellings and model tables before
        // writing output. The default is 64 KiB plus 256 bytes per vocabulary
        // entry; arbitrary token lengths can exceed this heuristic. Callers can
        // select headroom explicitly, and output growth is admitted separately.
        let headroom = headroom
            .or_else(|| {
                source
                    .get_vocab_size(true)
                    .checked_mul(256)?
                    .checked_add(64 * 1024)
            })
            .ok_or_else(|| rejected(WorkingMemoryError::Overflow))?;
        let bytes = [
            headroom,
            size_of::<Destination>(),
            size_of::<Input>(),
            size_of::<Cause>(),
            size_of::<OriginalTokenizerInputError>(),
            size_of::<TokenizerPlan<'_>>(),
            size_of::<Result<Vec<u8>, serde_json::Error>>(),
            BackendFailure::source_retention_peak_bytes::<OriginalTokenizerInputError>()
                .ok_or_else(|| rejected(WorkingMemoryError::Overflow))?,
            super::OriginalTokenizerSourceError::tokenizer_controls()
                .ok_or_else(|| rejected(WorkingMemoryError::Overflow))?,
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)
        .and_then(|n| u64::try_from(n).ok())
        .ok_or_else(|| rejected(WorkingMemoryError::Overflow))?;
        let allowance = self.admit_source_compiler(bytes).map_err(rejected)?;
        let mut destination = Destination {
            input: Input {
                read: None,
                bytes: Vec::new(),
                filled: 0,
                allowance,
            },
            accounting: None,
            reserve: None,
        };
        let serialized = serde_json::to_writer(&mut destination, &**source);
        let settlement = destination.input.allowance.end_compilation().err();
        if let Some(error) = destination.reserve {
            return Err(OriginalTokenizerInputError {
                cause: Cause::Reserve(error),
                settlement,
                input: Some(destination.input),
            });
        }
        match serialized {
            Err(error) => Err(OriginalTokenizerInputError {
                cause: Cause::Serialize(error, destination.accounting),
                settlement,
                input: Some(destination.input),
            }),
            Ok(()) => {
                if let Some(error) = destination.accounting.or(settlement) {
                    return Err(OriginalTokenizerInputError {
                        cause: Cause::Accounting(error),
                        settlement: None,
                        input: Some(destination.input),
                    });
                }
                Ok(destination.input)
            }
        }
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
    ) -> Result<OriginalTokenizer, OriginalTokenizerInputError> {
        let input = original_file::read(
            self,
            read,
            original_file::Destination::Tokenizer,
            after_admission,
        )
        .map_err(OriginalTokenizerInputError::from_read)?;
        after_read();
        self.compile_tokenizer_input_bytes(input, generation, false, compile)
    }
    fn compile_tokenizer_input_bytes(
        &self,
        input: Input,
        generation: bool,
        encode_special_tokens: bool,
        compile: impl FnOnce(
            &Self,
            TokenizerPlan<'_>,
        ) -> Result<OriginalTokenizer, OriginalTokenizerError>,
    ) -> Result<OriginalTokenizer, OriginalTokenizerInputError> {
        let plan = match TokenizerPlan::prepare_json(&input.bytes)
            .map(|plan| plan.with_encode_special_tokens(encode_special_tokens))
            .and_then(|plan| {
                if generation {
                    plan.with_generation_domain()
                } else {
                    Ok(plan)
                }
            }) {
            Ok(plan) => plan,
            Err(error) => {
                return Err(OriginalTokenizerInputError {
                    cause: Cause::Plan(error),
                    settlement: None,
                    input: Some(input),
                });
            }
        };
        match compile(self, plan) {
            Ok(source) => {
                drop(input);
                Ok(source)
            }
            Err(error) => Err(OriginalTokenizerInputError {
                cause: Cause::Compile(error),
                settlement: None,
                input: Some(input),
            }),
        }
    }
}

#[cfg(all(test, unix))]
mod tests;
