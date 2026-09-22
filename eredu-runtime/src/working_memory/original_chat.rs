//! Original J/H custody over closed fresh chat compiler and shared VM storage.
mod file;
mod operation;
mod profile;
mod render;
pub use file::OriginalChatFileError;
pub use operation::{
    OriginalChatBackend, OriginalChatRenderOperationError, OriginalChatSourceError,
};
pub use profile::{OriginalChatProfileError, OriginalChatProfilePreparation};
mod controller;
pub use controller::{
    ControllerCompilationOutput, ControllerCompilationSources, OriginalControllerCompilation,
    OriginalControllerCompilationError, OriginalControllerCompiler,
};
pub use render::{
    OriginalChatConsumer, OriginalChatConsumerError, OriginalChatRenderError, OriginalRenderedChat,
};

use super::loaded_decode_source::Allowance;
use super::{MemoryLedger, WorkingMemoryError};
use eredu_text::chat_storage::{ChatTemplateFailure, ChatTemplatePlan, PreparedChatTemplate};
use std::{
    alloc::Layout,
    fmt,
    mem::size_of,
    sync::{Arc, atomic::AtomicUsize},
};

#[derive(Debug)]
struct SourcePayload {
    source: PreparedChatTemplate,
    allowance: Allowance,
}
/// Originally admitted fresh template source; no raw VM/Arc/Weak/serde escape.
pub struct OriginalChatTemplate(Option<Arc<SourcePayload>>);
impl OriginalChatTemplate {
    fn payload(&self) -> &SourcePayload {
        self.0.as_deref().expect("live original chat source")
    }
    /// Actual selected source name copied during original construction.
    pub fn name(&self) -> &str {
        self.payload().source.name()
    }
    /// Authenticates actual loaded template selection against this compiled
    /// source, preserving the original account and making no allocation.
    pub fn matches_configuration(
        &self,
        selected: &eredu_text::tokenizer::ModelChatTemplate,
        model_id: &str,
    ) -> bool {
        self.payload()
            .source
            .matches_configuration(selected, model_id)
    }
    /// Checks loaded defaults against the actual compiled image's external
    /// inputs. Unused values stay borrowed; no ordinary context map is copied.
    pub fn accepts_default_variables(
        &self,
        values: &serde_json::Map<String, serde_json::Value>,
    ) -> bool {
        self.payload().source.accepts_default_variables(values)
    }
    /// Authenticates the exact request's ordinary named-template selection.
    pub fn matches_selection(
        &self,
        template: &eredu_text::tokenizer::ModelChatTemplate,
        model_id: &str,
        has_tools: bool,
    ) -> bool {
        self.payload()
            .source
            .matches_selection(template, model_id, has_tools)
    }
    /// Entire original allowance, retained through final source/control retirement.
    pub fn original_bytes(&self) -> u64 {
        self.payload().allowance.bytes()
    }
    /// Exact source-owner identity, never a source digest or layout comparison.
    pub fn same_source(&self, other: &Self) -> bool {
        Arc::ptr_eq(
            self.0.as_ref().expect("live source"),
            other.0.as_ref().expect("live source"),
        )
    }
    /// Authenticate the same real original pool without adding a registry hold.
    pub fn validate_pool(&self, pool: &MemoryLedger) -> Result<(), WorkingMemoryError> {
        if self.payload().allowance.pool().same_ledger(pool) {
            Ok(())
        } else {
            Err(WorkingMemoryError::IdentityMismatch)
        }
    }
}
impl Clone for OriginalChatTemplate {
    fn clone(&self) -> Self {
        Self(Some(Arc::clone(self.0.as_ref().expect("live source"))))
    }
}
impl Drop for OriginalChatTemplate {
    fn drop(&mut self) {
        if let Some(owner) = self.0.take() {
            // Every strong alias participates. No Weak exists. Arc control
            // deallocation precedes payload return, and the allowance is last.
            drop(Arc::into_inner(owner));
        }
    }
}
impl fmt::Debug for OriginalChatTemplate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OriginalChatTemplate")
            .field("name", &self.name())
            .field("original_bytes", &self.original_bytes())
            .finish_non_exhaustive()
    }
}
#[derive(Debug)]
enum SourceCause {
    Admission(WorkingMemoryError),
    Compilation(ChatTemplateFailure),
}
/// By-value J error retaining all real compiler prefixes and original allowance.
pub struct OriginalChatTemplateError {
    cause: SourceCause,
    settlement: Option<WorkingMemoryError>,
    completed: Option<PreparedChatTemplate>,
    allowance: Option<Allowance>,
}
impl OriginalChatTemplateError {
    fn rejected(cause: WorkingMemoryError) -> Self {
        Self {
            cause: SourceCause::Admission(cause),
            settlement: None,
            completed: None,
            allowance: None,
        }
    }
    /// Entire retained original charge; zero for rejection before compilation.
    pub fn retained_bytes(&self) -> u64 {
        self.allowance.as_ref().map_or(0, Allowance::bytes)
    }
    /// Actual owning compiler reserve failure, when construction started.
    pub fn compiler_failure(&self) -> Option<&ChatTemplateFailure> {
        match &self.cause {
            SourceCause::Compilation(error) => Some(error),
            _ => None,
        }
    }
    /// Actual admission/settlement error, preserving the original typed cause.
    pub fn accounting_failure(&self) -> Option<&WorkingMemoryError> {
        self.settlement.as_ref().or_else(|| match &self.cause {
            SourceCause::Admission(error) => Some(error),
            _ => None,
        })
    }
}
impl fmt::Debug for OriginalChatTemplateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OriginalChatTemplateError")
            .field("cause", &self.cause)
            .field("settlement", &self.settlement)
            .field("completed", &self.completed.is_some())
            .field("retained_bytes", &self.retained_bytes())
            .finish()
    }
}
impl fmt::Display for OriginalChatTemplateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.cause {
            SourceCause::Admission(error) => fmt::Display::fmt(error, f),
            SourceCause::Compilation(error) => fmt::Display::fmt(error, f),
        }
    }
}
impl std::error::Error for OriginalChatTemplateError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.cause {
            SourceCause::Admission(error) => Some(error),
            SourceCause::Compilation(error) => Some(error),
        }
    }
}
fn arc_bytes<T>() -> Result<usize, WorkingMemoryError> {
    // Same actual pinned Rust ArcInner<T> layout as the existing C and cold
    // decode owners: two atomic counts followed by aligned T.
    Layout::new::<[AtomicUsize; 2]>()
        .extend(Layout::new::<T>())
        .map(|(layout, _)| layout.pad_to_align().size())
        .map_err(|_| WorkingMemoryError::Overflow)
}
impl MemoryLedger {
    /// Concrete J compiler/owner/error layouts before its single comparison.
    pub fn chat_template_required_bytes(
        plan: &ChatTemplatePlan<'_>,
    ) -> Result<u64, WorkingMemoryError> {
        let controls = [
            OriginalChatSourceError::source_controls().ok_or(WorkingMemoryError::Overflow)?,
            arc_bytes::<SourcePayload>()?,
            size_of::<SourcePayload>(),
            size_of::<Option<SourcePayload>>(),
            size_of::<Arc<SourcePayload>>(),
            size_of::<Option<Arc<SourcePayload>>>(),
            size_of::<OriginalChatTemplate>(),
            size_of::<Option<OriginalChatTemplate>>(),
            size_of::<Allowance>(),
            size_of::<Result<Allowance, WorkingMemoryError>>(),
            size_of::<SourceCause>(),
            size_of::<Option<WorkingMemoryError>>(),
            size_of::<Option<PreparedChatTemplate>>(),
            size_of::<Result<(), WorkingMemoryError>>(),
            size_of::<OriginalChatTemplateError>(),
            size_of::<Result<OriginalChatTemplate, OriginalChatTemplateError>>(),
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)
        .ok_or(WorkingMemoryError::Overflow)?;
        plan.requirements()
            .required_bytes()
            .checked_add(controls)
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or(WorkingMemoryError::Overflow)
    }
    /// Consume a genuine source plan once; admit before its first real reserve.
    /// Compiler completion ends only the active count, never retained byte custody.
    pub fn compile_chat_template(
        &self,
        plan: ChatTemplatePlan<'_>,
    ) -> Result<OriginalChatTemplate, OriginalChatTemplateError> {
        self.compile_chat_template_with(plan, || {})
    }
    fn compile_chat_template_with(
        &self,
        plan: ChatTemplatePlan<'_>,
        after_admission: impl FnOnce(),
    ) -> Result<OriginalChatTemplate, OriginalChatTemplateError> {
        let bytes = Self::chat_template_required_bytes(&plan)
            .map_err(OriginalChatTemplateError::rejected)?;
        let mut allowance = self
            .admit_source_compiler(bytes)
            .map_err(OriginalChatTemplateError::rejected)?;
        after_admission();
        // Compiler locals are below the guard, including during unwind.
        match plan.compile() {
            Err(error) => {
                let settlement = allowance.end_compilation().err();
                Err(OriginalChatTemplateError {
                    cause: SourceCause::Compilation(error),
                    settlement,
                    completed: None,
                    allowance: Some(allowance),
                })
            }
            Ok(source) => {
                let mut owner = Arc::new(SourcePayload { source, allowance });
                let settlement = Arc::get_mut(&mut owner)
                    .expect("unpublished source")
                    .allowance
                    .end_compilation();
                match settlement {
                    Ok(()) => Ok(OriginalChatTemplate(Some(owner))),
                    Err(error) => {
                        let SourcePayload { source, allowance } =
                            Arc::into_inner(owner).expect("unpublished source");
                        Err(OriginalChatTemplateError {
                            cause: SourceCause::Admission(error),
                            settlement: None,
                            completed: Some(source),
                            allowance: Some(allowance),
                        })
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests;
