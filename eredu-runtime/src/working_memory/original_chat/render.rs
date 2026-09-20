mod consumer;
use super::*;
use crate::working_memory::OriginalTokenizer;
pub use consumer::{OriginalChatConsumer, OriginalChatConsumerError};
use eredu_core::GenerationSequenceConsumerLayout;
use eredu_text::chat_storage::{
    ChatRenderContext, ChatRenderFailure, ChatRenderPlan, ChatRenderPlanError, PreparedChatRender,
};
use std::mem::size_of_val;

#[derive(Debug)]
struct Payload {
    render: PreparedChatRender,
    template: OriginalChatTemplate,
    tokenizer: OriginalTokenizer,
    allowance: Allowance,
}
/// Originally admitted renderings with exact original J/C source custody.
pub struct OriginalRenderedChat(Option<Arc<Payload>>);
impl OriginalRenderedChat {
    fn payload(&self) -> &Payload {
        self.0.as_deref().expect("live original chat render")
    }
    /// Identity of the actual immutable rendering, not equality of prompt bytes.
    pub fn same_render(&self, other: &Self) -> bool {
        Arc::ptr_eq(
            self.0.as_ref().expect("live render"),
            other.0.as_ref().expect("live render"),
        )
    }
    /// Selected immutable prompt inside the original H destination.
    pub fn prompt(&self, generation_prompt: bool) -> &str {
        self.payload().render.prompt(generation_prompt)
    }
    /// Borrowed generation suffix range; no third owned String.
    pub fn generation_suffix(&self) -> &str {
        self.payload().render.generation_suffix()
    }
    /// Same closed original tokenizer source selected before H admission.
    pub fn tokenizer_source(&self) -> &OriginalTokenizer {
        &self.payload().tokenizer
    }
    /// Actual retained J identity; no raw VM is exposed.
    pub fn template_source(&self) -> &OriginalChatTemplate {
        &self.payload().template
    }
    /// Entire render reservation, including dependency headroom.
    pub fn original_bytes(&self) -> u64 {
        self.payload().allowance.bytes()
    }
    /// Exact retained source owners, never equal-content or numeric evidence.
    pub fn has_sources(
        &self,
        template: &OriginalChatTemplate,
        tokenizer: &OriginalTokenizer,
    ) -> bool {
        self.payload().template.same_source(template)
            && self.payload().tokenizer.same_source(tokenizer)
    }
    /// Validate the genuine H/J/C pool before downstream source operations.
    pub fn validate_pool(&self, pool: &WorkingMemoryPool) -> Result<(), WorkingMemoryError> {
        if !self.payload().allowance.pool().same_domain(pool) {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        self.payload().template.validate_pool(pool)?;
        self.payload().tokenizer.validate_pool(pool)
    }
}
impl Clone for OriginalRenderedChat {
    fn clone(&self) -> Self {
        Self(Some(Arc::clone(self.0.as_ref().expect("live render"))))
    }
}
impl Drop for OriginalRenderedChat {
    fn drop(&mut self) {
        if let Some(owner) = self.0.take() {
            drop(Arc::into_inner(owner));
        }
    }
}
impl fmt::Debug for OriginalRenderedChat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OriginalRenderedChat")
            .field("original_bytes", &self.original_bytes())
            .finish_non_exhaustive()
    }
}
#[derive(Debug)]
enum Cause {
    Accounting(WorkingMemoryError),
    Plan(ChatRenderPlanError),
    Domain,
    Render(ChatRenderFailure),
}
/// Closed H failure with all real partial/completed storage and source owners.
pub struct OriginalChatRenderError {
    cause: Cause,
    settlement: Option<WorkingMemoryError>,
    completed: Option<PreparedChatRender>,
    template: Option<OriginalChatTemplate>,
    tokenizer: Option<OriginalTokenizer>,
    allowance: Option<Allowance>,
}
impl OriginalChatRenderError {
    fn rejected(cause: Cause) -> Self {
        Self {
            cause,
            settlement: None,
            completed: None,
            template: None,
            tokenizer: None,
            allowance: None,
        }
    }
    /// Full retained H charge; zero for rejection before H construction.
    pub fn retained_bytes(&self) -> u64 {
        self.allowance.as_ref().map_or(0, Allowance::bytes)
    }
    /// Actual admission or settlement failure.
    pub fn accounting_failure(&self) -> Option<&WorkingMemoryError> {
        self.settlement.as_ref().or_else(|| match &self.cause {
            Cause::Accounting(error) => Some(error),
            _ => None,
        })
    }
    /// Actual real-reserve or shared-dispatch failure with all partial buffers.
    pub fn render_failure(&self) -> Option<&ChatRenderFailure> {
        match &self.cause {
            Cause::Render(error) => Some(error),
            _ => None,
        }
    }
    /// Actual absent-callable semantics from the closed renderer, only when
    /// admission/settlement succeeded. Reading this flag releases no custody.
    pub fn is_unknown_function(&self) -> bool {
        self.accounting_failure().is_none()
            && match &self.cause {
                Cause::Render(cause) => cause.cause().is_unknown_function(),
                _ => false,
            }
    }
    /// Template input failure suitable for a behavioral probe nonmatch. All
    /// admission, settlement, allocation, geometry, and overflow failures remain
    /// hard errors; inspecting this classification does not release custody.
    pub fn is_template_rejection(&self) -> bool {
        self.accounting_failure().is_none()
            && match &self.cause {
                Cause::Render(cause) => cause.cause().is_template_rejection(),
                _ => false,
            }
    }
    /// Whether this operation rejected a decoder-only C source before H admission.
    pub fn missing_generation_domain(&self) -> bool {
        matches!(self.cause, Cause::Domain)
    }
}
impl fmt::Debug for OriginalChatRenderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OriginalChatRenderError")
            .field("cause", &self.cause)
            .field("settlement", &self.settlement)
            .field("completed", &self.completed.is_some())
            .field(
                "source_owners",
                &(self.template.is_some(), self.tokenizer.is_some()),
            )
            .field("retained_bytes", &self.retained_bytes())
            .finish()
    }
}
impl fmt::Display for OriginalChatRenderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.cause {
            Cause::Accounting(error) => fmt::Display::fmt(error, f),
            Cause::Plan(error) => fmt::Display::fmt(error, f),
            Cause::Render(error) => fmt::Display::fmt(error, f),
            Cause::Domain => f.write_str("chat requires an original generation tokenizer domain"),
        }
    }
}
impl std::error::Error for OriginalChatRenderError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.cause {
            Cause::Accounting(error) => Some(error),
            Cause::Plan(error) => Some(error),
            Cause::Render(error) => Some(error),
            Cause::Domain => None,
        }
    }
}
fn required(plan: &ChatRenderPlan<'_>) -> Result<u64, WorkingMemoryError> {
    let controls = [
        size_of::<OriginalChatConsumerError>(),
        OriginalChatRenderOperationError::render_controls().ok_or(WorkingMemoryError::Overflow)?,
        arc_bytes::<Payload>()?,
        size_of::<Payload>(),
        size_of::<Option<Payload>>(),
        size_of::<Arc<Payload>>(),
        size_of::<Option<Arc<Payload>>>(),
        size_of::<OriginalRenderedChat>(),
        size_of::<Result<OriginalRenderedChat, OriginalChatRenderError>>(),
        size_of::<OriginalChatRenderError>(),
        size_of::<Allowance>(),
        size_of::<Result<Allowance, WorkingMemoryError>>(),
        size_of::<Option<PreparedChatRender>>(),
        size_of::<Option<OriginalChatTemplate>>(),
        size_of::<Option<OriginalTokenizer>>(),
        size_of::<Option<WorkingMemoryError>>(),
        size_of::<Result<(), WorkingMemoryError>>(),
        size_of::<Cause>(),
        size_of::<ChatRenderContext<'_>>(),
        size_of::<(
            &OriginalChatTemplate,
            &OriginalTokenizer,
            ChatRenderContext<'_>,
        )>(),
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
impl WorkingMemoryPool {
    fn chat_render_plan<'a>(
        &self,
        template: &'a OriginalChatTemplate,
        tokenizer: &OriginalTokenizer,
        context: ChatRenderContext<'a>,
    ) -> Result<ChatRenderPlan<'a>, OriginalChatRenderError> {
        template
            .validate_pool(self)
            .map_err(|e| OriginalChatRenderError::rejected(Cause::Accounting(e)))?;
        tokenizer
            .validate_pool(self)
            .map_err(|e| OriginalChatRenderError::rejected(Cause::Accounting(e)))?;
        if tokenizer.generation_domain().is_none() {
            return Err(OriginalChatRenderError::rejected(Cause::Domain));
        }
        template
            .payload()
            .source
            .render_plan_with_context(context)
            .map_err(|error| OriginalChatRenderError::rejected(Cause::Plan(error)))
    }
    /// Estimates dependency work and exact output capacities before execution.
    pub fn chat_render_required_bytes(
        &self,
        template: &OriginalChatTemplate,
        tokenizer: &OriginalTokenizer,
        context: ChatRenderContext<'_>,
    ) -> Result<u64, OriginalChatRenderError> {
        let plan = self.chat_render_plan(template, tokenizer, context)?;
        required(&plan).map_err(|e| OriginalChatRenderError::rejected(Cause::Accounting(e)))
    }
    /// Same source validation, allowance and retained render/error worker with
    /// borrowed ordinary-precedence context. No caller borrow escapes this call.
    pub fn render_original_chat(
        &self,
        template: &OriginalChatTemplate,
        tokenizer: &OriginalTokenizer,
        context: ChatRenderContext<'_>,
    ) -> Result<OriginalRenderedChat, OriginalChatRenderError> {
        let plan = self.chat_render_plan(template, tokenizer, context)?;
        self.render_original_chat_plan(template, tokenizer, plan, || {})
    }
    pub(super) fn render_original_chat_plan(
        &self,
        template: &OriginalChatTemplate,
        tokenizer: &OriginalTokenizer,
        plan: ChatRenderPlan<'_>,
        after_admission: impl FnOnce(),
    ) -> Result<OriginalRenderedChat, OriginalChatRenderError> {
        if !plan.is_for(&template.payload().source) {
            return Err(OriginalChatRenderError::rejected(Cause::Accounting(
                WorkingMemoryError::IdentityMismatch,
            )));
        }
        let bytes =
            required(&plan).map_err(|e| OriginalChatRenderError::rejected(Cause::Accounting(e)))?;
        let mut allowance = self
            .admit_source_compiler(bytes)
            .map_err(|e| OriginalChatRenderError::rejected(Cause::Accounting(e)))?;
        after_admission();
        // These existing original aliases allocate no new Arc control. Their
        // charges remain in the same pool; H receives no adoption or credit.
        let template = template.clone();
        let tokenizer = tokenizer.clone();
        match plan.render() {
            Err(error) => {
                let settlement = allowance.end_compilation().err();
                Err(OriginalChatRenderError {
                    cause: Cause::Render(error),
                    settlement,
                    completed: None,
                    template: Some(template),
                    tokenizer: Some(tokenizer),
                    allowance: Some(allowance),
                })
            }
            Ok(render) => {
                let mut owner = Arc::new(Payload {
                    render,
                    template,
                    tokenizer,
                    allowance,
                });
                let settlement = Arc::get_mut(&mut owner)
                    .expect("unpublished render")
                    .allowance
                    .end_compilation();
                match settlement {
                    Ok(()) => Ok(OriginalRenderedChat(Some(owner))),
                    Err(error) => {
                        let Payload {
                            render,
                            template,
                            tokenizer,
                            allowance,
                        } = Arc::into_inner(owner).expect("unpublished render");
                        Err(OriginalChatRenderError {
                            cause: Cause::Accounting(error),
                            settlement: None,
                            completed: Some(render),
                            template: Some(template),
                            tokenizer: Some(tokenizer),
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
