//! Original exact config bytes coexist with the fresh J compiler in one pool.
use super::*;
use crate::working_memory::original_file::{self, Input};
use eredu_checkpoint::artifact::{ArtifactFileReadFailure, PreparedArtifactFileRead};
use eredu_text::chat_storage::ChatSourceError;

#[derive(Debug)]
enum Cause {
    Input(original_file::Cause),
    Plan(ChatSourceError),
    Compile(OriginalChatTemplateError),
}
/// Consumed config-file failure retaining its original I and any fresh J prefix.
/// Path opening is outside this boundary; no file/bytes/retry extraction exists.
#[derive(Debug)]
pub struct OriginalChatFileError {
    cause: Cause,
    settlement: Option<WorkingMemoryError>,
    input: Option<Input>,
}
impl OriginalChatFileError {
    fn from_read(error: original_file::Failure) -> Self {
        Self {
            cause: Cause::Input(error.cause),
            settlement: error.settlement,
            input: error.input,
        }
    }
    /// Complete retained original file-byte allowance.
    pub fn input_bytes(&self) -> u64 {
        self.input.as_ref().map_or(0, |i| i.allowance.bytes())
    }
    /// Bytes actually filled by the consuming read, excluding initialized tail.
    pub fn filled_bytes(&self) -> usize {
        self.input.as_ref().map_or(0, |i| i.filled)
    }
    /// Actual retained input vector capacity, including partial success.
    pub fn input_capacity(&self) -> usize {
        self.input.as_ref().map_or(0, |i| i.bytes.capacity())
    }
    /// Actual OS/version/read failure, without a replayable file capability.
    pub fn read_failure(&self) -> Option<&ArtifactFileReadFailure> {
        match &self.cause {
            Cause::Input(original_file::Cause::Read(error)) => Some(error),
            _ => None,
        }
    }
    /// Actual checked config/template selection error before J admission.
    pub fn planning_failure(&self) -> Option<&ChatSourceError> {
        match &self.cause {
            Cause::Plan(error) => Some(error),
            _ => None,
        }
    }
    /// Actual J constructor error, retaining its independent original charge.
    pub fn compiler_failure(&self) -> Option<&OriginalChatTemplateError> {
        match &self.cause {
            Cause::Compile(error) => Some(error),
            _ => None,
        }
    }
    /// Actual I/J admission or settlement cause.
    pub fn accounting_failure(&self) -> Option<&WorkingMemoryError> {
        self.settlement.as_ref().or_else(|| match &self.cause {
            Cause::Input(original_file::Cause::Accounting(error)) => Some(error),
            Cause::Compile(error) => error.accounting_failure(),
            _ => None,
        })
    }
}
impl fmt::Display for OriginalChatFileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.cause {
            Cause::Input(original_file::Cause::Accounting(error)) => fmt::Display::fmt(error, f),
            Cause::Input(original_file::Cause::Reserve(error)) => fmt::Display::fmt(error, f),
            Cause::Input(original_file::Cause::Read(error)) => fmt::Display::fmt(error, f),
            Cause::Plan(error) => fmt::Display::fmt(error, f),
            Cause::Compile(error) => fmt::Display::fmt(error, f),
        }
    }
}
impl std::error::Error for OriginalChatFileError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(match &self.cause {
            Cause::Input(original_file::Cause::Accounting(error)) => error,
            Cause::Input(original_file::Cause::Reserve(error)) => error,
            Cause::Input(original_file::Cause::Read(error)) => error,
            Cause::Plan(error) => error,
            Cause::Compile(error) => error,
        })
    }
}
impl MemoryLedger {
    /// Exact original config-file I population before the real byte reserve.
    /// This admits only I; its retained charge still constrains subsequent J.
    pub fn chat_template_file_required_bytes(
        read: &PreparedArtifactFileRead,
    ) -> Result<u64, WorkingMemoryError> {
        let controls = [
            original_file::control_bytes().ok_or(WorkingMemoryError::Overflow)?,
            OriginalChatSourceError::source_controls().ok_or(WorkingMemoryError::Overflow)?,
            size_of::<(&str, bool)>(), // actual borrowed model/name and request selection
            size_of::<ChatTemplatePlan<'_>>(),
            size_of::<Result<ChatTemplatePlan<'_>, ChatSourceError>>(),
            size_of::<Result<OriginalChatTemplate, OriginalChatTemplateError>>(),
            size_of::<Cause>(),
            size_of::<OriginalChatFileError>(),
            size_of::<Result<OriginalChatTemplate, OriginalChatFileError>>(),
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)
        .ok_or(WorkingMemoryError::Overflow)?;
        read.byte_len()
            .checked_add(controls)
            .and_then(|n| u64::try_from(n).ok())
            .ok_or(WorkingMemoryError::Overflow)
    }
    /// Consume the exact file, then compile its selected fresh J in the same pool.
    /// Success retires I before returning J; any later failure keeps both owners.
    pub fn compile_chat_template_file(
        &self,
        read: PreparedArtifactFileRead,
        model_id: &str,
        has_tools: bool,
    ) -> Result<OriginalChatTemplate, OriginalChatFileError> {
        self.compile_chat_template_file_with(
            read,
            model_id,
            has_tools,
            |_| {},
            || {},
            |pool, plan| pool.compile_chat_template(plan),
        )
    }
    fn compile_chat_template_file_with(
        &self,
        read: PreparedArtifactFileRead,
        model_id: &str,
        has_tools: bool,
        after_admission: impl FnOnce(&mut usize),
        after_read: impl FnOnce(),
        compile: impl FnOnce(
            &Self,
            ChatTemplatePlan<'_>,
        ) -> Result<OriginalChatTemplate, OriginalChatTemplateError>,
    ) -> Result<OriginalChatTemplate, OriginalChatFileError> {
        let input = original_file::read(
            self,
            read,
            original_file::Destination::Chat,
            after_admission,
        )
        .map_err(OriginalChatFileError::from_read)?;
        after_read();
        let plan = match ChatTemplatePlan::prepare_config(&input.bytes, model_id, has_tools) {
            Ok(plan) => plan,
            Err(error) => {
                return Err(OriginalChatFileError {
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
            Err(error) => Err(OriginalChatFileError {
                cause: Cause::Compile(error),
                settlement: None,
                input: Some(input),
            }),
        }
    }
}

#[cfg(all(test, unix))]
mod tests;
