//! Prepared chat sources and rendering through stock MiniJinja.
mod input;
mod source_identity;
pub use crate::tokenizer::clock::ChatClockSnapshot;
use crate::tokenizer::{
    ModelChatTemplate, chat_environment, install_chat_clock, normalize_chat_template,
};
pub use input::{
    ChatInputArray, ChatMessageError, ChatMessages, ChatRecordField, ChatRecordValue,
    ChatRenderContext, ChatScalarBinding, ChatScalarBindingValue, TextMessage,
};
use source_identity::SourceIdentity;
use std::{collections::TryReserveError, io};

/// Observes local calendar and offset facts once for a request.
pub fn chat_clock_snapshot() -> ChatClockSnapshot {
    ChatClockSnapshot::now()
}

/// Configurable dependency headroom. These coefficients are estimates, not
/// allocator measurements or enforceable process/dependency memory ceilings.
#[derive(Debug, Clone, Copy)]
pub struct ChatMemoryEstimate {
    /// Fixed overhead per independent construction or rendering operation.
    pub fixed_bytes: usize,
    /// Construction headroom per serialized config or template byte.
    pub bytes_per_source_byte: usize,
    /// Render headroom per serialized input byte.
    pub bytes_per_input_byte: usize,
}
impl Default for ChatMemoryEstimate {
    fn default() -> Self {
        Self {
            fixed_bytes: 64 * 1024,
            bytes_per_source_byte: 128,
            bytes_per_input_byte: 128,
        }
    }
}
/// Enforced input/output and upstream execution controls. Fuel bounds VM
/// instructions, not time or allocations within individual filters or methods.
#[derive(Debug, Clone, Copy)]
pub struct ChatRenderLimits {
    /// Maximum serialized size of borrowed input and variable layers.
    pub max_input_bytes: usize,
    /// Maximum nesting of input JSON or portable records.
    pub max_input_depth: usize,
    /// Maximum UTF-8 bytes in each of the two prompt variants.
    pub max_output_bytes: usize,
    /// MiniJinja instruction fuel for each prompt variant.
    pub fuel: u64,
    /// MiniJinja recursion limit for each prompt variant.
    pub recursion_limit: usize,
}
impl Default for ChatRenderLimits {
    fn default() -> Self {
        Self {
            max_input_bytes: 8 * 1024 * 1024,
            max_input_depth: 128,
            max_output_bytes: 1024 * 1024,
            fuel: 10_000_000,
            recursion_limit: 256,
        }
    }
}
/// Source selection, syntax, JSON or construction-estimate error.
#[derive(Debug, thiserror::Error)]
pub enum ChatSourceError {
    /// Complete ordinary JSON validation failure.
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    /// Invalid chat_template representation.
    #[error("invalid chat template selection profile")]
    TemplateProfile,
    /// The requested single/default/tool template is absent.
    #[error("no selected chat template")]
    MissingTemplate,
    /// Repeated named template.
    #[error("duplicate chat template name")]
    DuplicateName,
    /// Original MiniJinja compiler failure.
    #[error(transparent)]
    Source(#[from] minijinja::Error),
    /// Admission estimate overflow.
    #[error("chat construction estimate overflow")]
    Overflow,
}
#[derive(Debug)]
enum Input<'a> {
    Utf8 {
        source: &'a str,
        entry: Option<&'static str>,
    },
    Config {
        bytes: &'a [u8],
        has_tools: bool,
    },
}
/// Estimated source-construction footprint, including dependency allocations.
#[derive(Debug, Clone, Copy)]
pub struct ChatTemplateRequirements {
    total: usize,
}
impl ChatTemplateRequirements {
    /// Estimate to reserve before construction. Does not certify a heap bound.
    pub fn required_bytes(self) -> usize {
        self.total
    }
}
/// One borrowed source/configuration, consumed by a single admitted compilation.
#[derive(Debug)]
pub struct ChatTemplatePlan<'a> {
    input: Input<'a>,
    model_id: &'a str,
    estimate: ChatMemoryEstimate,
    limits: ChatRenderLimits,
    requirements: ChatTemplateRequirements,
}
impl<'a> ChatTemplatePlan<'a> {
    /// Plans a UTF-8 template without compiling it.
    pub fn prepare_utf8(source: &'a str, model_id: &'a str) -> Result<Self, ChatSourceError> {
        Self::new(
            Input::Utf8 {
                source,
                entry: None,
            },
            model_id,
        )
    }
    /// Selects the request's actual named template before construction.
    pub fn prepare_model(
        template: &'a ModelChatTemplate,
        model_id: &'a str,
        has_tools: bool,
    ) -> Result<Self, ChatSourceError> {
        let (source, entry) = template
            .selected_source(has_tools)
            .ok_or(ChatSourceError::MissingTemplate)?;
        Self::new(Input::Utf8 { source, entry }, model_id)
    }
    /// Defers complete JSON validation and selection until admitted compilation.
    pub fn prepare_config(
        config: &'a [u8],
        model_id: &'a str,
        has_tools: bool,
    ) -> Result<Self, ChatSourceError> {
        Self::new(
            Input::Config {
                bytes: config,
                has_tools,
            },
            model_id,
        )
    }
    fn new(input: Input<'a>, model_id: &'a str) -> Result<Self, ChatSourceError> {
        let mut plan = Self {
            input,
            model_id,
            estimate: ChatMemoryEstimate::default(),
            limits: ChatRenderLimits::default(),
            requirements: ChatTemplateRequirements { total: 0 },
        };
        plan.reestimate()?;
        Ok(plan)
    }
    fn reestimate(&mut self) -> Result<(), ChatSourceError> {
        let bytes = match self.input {
            Input::Utf8 { source, .. } => source.len(),
            Input::Config { bytes, .. } => bytes.len(),
        };
        self.requirements.total = bytes
            .checked_add(self.model_id.len())
            .and_then(|n| n.checked_mul(self.estimate.bytes_per_source_byte))
            .and_then(|n| n.checked_add(self.estimate.fixed_bytes))
            .ok_or(ChatSourceError::Overflow)?;
        Ok(())
    }
    /// Sets construction and per-request dependency headroom.
    pub fn with_memory_estimate(
        mut self,
        estimate: ChatMemoryEstimate,
    ) -> Result<Self, ChatSourceError> {
        self.estimate = estimate;
        self.reestimate()?;
        Ok(self)
    }
    /// Sets limits retained by all renders of this source.
    pub fn with_render_limits(mut self, limits: ChatRenderLimits) -> Self {
        self.limits = limits;
        self
    }
    /// Construction headroom before compilation.
    pub fn requirements(&self) -> ChatTemplateRequirements {
        self.requirements
    }
    /// Compiles through the same normalization, filters and environment settings
    /// as ordinary chat. Upstream owns retirement of its compiler temporaries.
    pub fn compile(self) -> Result<PreparedChatTemplate, ChatTemplateFailure> {
        let config;
        let (source, entry) = match self.input {
            Input::Utf8 { source, entry } => (source, entry),
            Input::Config { bytes, has_tools } => {
                config = serde_json::from_slice::<serde_json::Value>(bytes)
                    .map_err(ChatSourceError::Json)?;
                select_config(&config, has_tools)?
            }
        };
        let identity = SourceIdentity::new(source.bytes());
        let name = match entry {
            None => self.model_id.to_owned(),
            Some(entry) => format!("{}::chat_template::{entry}", self.model_id),
        };
        let normalized = normalize_chat_template(source);
        let mut env = chat_environment();
        env.add_template_owned(name.clone(), normalized)?;
        Ok(PreparedChatTemplate {
            env,
            name,
            identity,
            estimate: self.estimate,
            limits: self.limits,
        })
    }
}
fn select_config(
    value: &serde_json::Value,
    tools: bool,
) -> Result<(&str, Option<&'static str>), ChatSourceError> {
    let object = value.as_object().ok_or(ChatSourceError::TemplateProfile)?;
    let value = object
        .get("chat_template")
        .filter(|v| !v.is_null())
        .ok_or(ChatSourceError::MissingTemplate)?;
    if let Some(source) = value.as_str() {
        return Ok((source, None));
    }
    let entries = value
        .as_array()
        .filter(|v| !v.is_empty())
        .ok_or(ChatSourceError::TemplateProfile)?;
    let mut names = std::collections::BTreeSet::new();
    let (mut default, mut tool) = (None, None);
    for entry in entries {
        let fields = entry
            .as_object()
            .filter(|v| v.len() == 2)
            .ok_or(ChatSourceError::TemplateProfile)?;
        let name = fields
            .get("name")
            .and_then(|v| v.as_str())
            .filter(|n| !n.is_empty())
            .ok_or(ChatSourceError::TemplateProfile)?;
        let source = fields
            .get("template")
            .and_then(|v| v.as_str())
            .ok_or(ChatSourceError::TemplateProfile)?;
        if !names.insert(name) {
            return Err(ChatSourceError::DuplicateName);
        }
        match name {
            "default" => default = Some(source),
            "tool_use" => tool = Some(source),
            _ => {}
        }
    }
    let name = crate::tokenizer::selected_chat_template_name(tools, tool.is_some());
    let source =
        if name == "tool_use" { tool } else { default }.ok_or(ChatSourceError::MissingTemplate)?;
    Ok((source, Some(name)))
}
/// Immutable compiled environment with no raw VM or mutable template escape.
#[derive(Debug)]
pub struct PreparedChatTemplate {
    env: minijinja::Environment<'static>,
    name: String,
    identity: SourceIdentity,
    estimate: ChatMemoryEstimate,
    limits: ChatRenderLimits,
}
impl PreparedChatTemplate {
    /// Stable model/template name selected at construction.
    pub fn name(&self) -> &str {
        &self.name
    }
    /// Compares the original selected source identity and name without allocating.
    /// Normalization does not make distinct source documents interchangeable.
    pub fn matches_configuration(
        &self,
        selected: &crate::tokenizer::ModelChatTemplate,
        model_id: &str,
    ) -> bool {
        use crate::tokenizer::ModelChatTemplate;
        match selected {
            ModelChatTemplate::Single(source) => {
                self.identity == SourceIdentity::new(source.bytes()) && self.name() == model_id
            }
            ModelChatTemplate::Named(templates) => self
                .name()
                .strip_prefix(model_id)
                .and_then(|suffix| suffix.strip_prefix("::chat_template::"))
                .and_then(|name| templates.get(name))
                .is_some_and(|source| self.identity == SourceIdentity::new(source.bytes())),
        }
    }

    /// Checks both the exact source and request-dependent named selection.
    pub fn matches_selection(
        &self,
        template: &crate::tokenizer::ModelChatTemplate,
        model_id: &str,
        has_tools: bool,
    ) -> bool {
        template
            .selected_source(has_tools)
            .is_some_and(|(source, name)| {
                self.identity == SourceIdentity::new(source.bytes())
                    && match name {
                        None => self.name() == model_id,
                        Some(name) => {
                            self.name()
                                .strip_prefix(model_id)
                                .and_then(|suffix| suffix.strip_prefix("::chat_template::"))
                                == Some(name)
                        }
                    }
            })
    }

    /// Message-only render calls preserve these defaults because they do
    /// not replace either intrinsic input. General contexts must instead use
    /// `render_plan_with_context`, which qualifies every actually used value.
    pub fn accepts_default_variables(
        &self,
        values: &serde_json::Map<String, serde_json::Value>,
    ) -> bool {
        !values.contains_key("messages") && !values.contains_key("add_generation_prompt")
    }

    /// Estimates rendering and exact first-party output buffers without executing
    /// the template. Template errors are returned by the admitted render itself.
    pub fn render_plan<'a>(
        &'a self,
        messages: ChatMessages<'a>,
    ) -> Result<ChatRenderPlan<'a>, ChatRenderPlanError> {
        self.render_plan_with_context(ChatRenderContext::from_messages(messages))
    }
    /// Preserves the same source and context borrows through one rendering attempt.
    pub fn render_plan_with_context<'a>(
        &'a self,
        context: ChatRenderContext<'a>,
    ) -> Result<ChatRenderPlan<'a>, ChatRenderPlanError> {
        let input =
            context.input_bytes(self.limits.max_input_bytes, self.limits.max_input_depth)?;
        let buffers = self
            .limits
            .max_output_bytes
            .checked_mul(2)
            .ok_or(ChatRenderPlanError::Overflow)?;
        std::alloc::Layout::array::<u8>(self.limits.max_output_bytes)
            .map_err(|_| ChatRenderPlanError::Overflow)?;
        let total = input
            .checked_mul(self.estimate.bytes_per_input_byte)
            .and_then(|n| n.checked_add(self.estimate.fixed_bytes))
            .and_then(|n| n.checked_add(buffers))
            .ok_or(ChatRenderPlanError::Overflow)?;
        Ok(ChatRenderPlan {
            source: self,
            context,
            clock: context.clock.unwrap_or_else(chat_clock_snapshot),
            requirements: ChatRenderRequirements { buffers, total },
            #[cfg(any(test, feature = "tokenizer-compiler-test-support"))]
            fail: None,
        })
    }
}
/// Original source error; dependency allocation inventories are not exposed.
#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct ChatTemplateFailure(#[from] ChatSourceError);
impl From<minijinja::Error> for ChatTemplateFailure {
    fn from(error: minijinja::Error) -> Self {
        Self(ChatSourceError::Source(error))
    }
}
impl ChatTemplateFailure {
    /// Original compiler error when JSON and source selection succeeded.
    pub fn cause(&self) -> Option<&minijinja::Error> {
        match &self.0 {
            ChatSourceError::Source(e) => Some(e),
            _ => None,
        }
    }
    /// Original JSON or source selection error.
    pub fn source_error(&self) -> Option<&ChatSourceError> {
        Some(&self.0)
    }
}
/// Pre-admission input/size refusal. No template is executed during planning.
#[derive(Debug, thiserror::Error)]
pub enum ChatRenderPlanError {
    /// Checked footprint overflow.
    #[error("chat render estimate overflow")]
    Overflow,
    /// Serialized borrowed inputs exceed the configured input-byte limit.
    #[error("chat input byte limit exceeded")]
    InputBytes,
    /// A JSON/record tree exceeds the configured input depth.
    #[error("chat input nesting limit exceeded")]
    InputDepth,
}
/// Exact output destinations plus estimated dependency work.
#[derive(Debug, Clone, Copy)]
pub struct ChatRenderRequirements {
    buffers: usize,
    total: usize,
}
impl ChatRenderRequirements {
    /// Capacity reserved for the two first-party UTF-8 output buffers.
    pub fn buffer_bytes(self) -> usize {
        self.buffers
    }
    /// Output capacities plus configurable dependency headroom.
    pub fn required_bytes(self) -> usize {
        self.total
    }
}
/// One borrowed source/context pair consumed by one admitted render.
#[derive(Debug)]
pub struct ChatRenderPlan<'a> {
    source: &'a PreparedChatTemplate,
    context: ChatRenderContext<'a>,
    clock: ChatClockSnapshot,
    requirements: ChatRenderRequirements,
    #[cfg(any(test, feature = "tokenizer-compiler-test-support"))]
    fail: Option<ChatRenderBuffer>,
}
/// First-party destinations available to development-only reserve failure tests.
#[cfg(any(test, feature = "tokenizer-compiler-test-support"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatRenderBuffer {
    WithoutPrompt,
    WithPrompt,
}
impl ChatRenderPlan<'_> {
    /// Exact borrowed source-object identity.
    pub fn is_for(&self, source: &PreparedChatTemplate) -> bool {
        std::ptr::eq(self.source, source)
    }
    /// Estimate and actual first-party output capacities before execution.
    pub fn requirements(&self) -> ChatRenderRequirements {
        self.requirements
    }
    /// Forces overflow at an actual first-party output reserve.
    #[cfg(any(test, feature = "tokenizer-compiler-test-support"))]
    pub fn fail_reservation(mut self, buffer: ChatRenderBuffer) -> Self {
        self.fail = Some(buffer);
        self
    }
    /// Renders each generation-prompt variant once with shared clock and limits.
    /// All temporary context values retire before the owned output or error returns.
    pub fn render(self) -> Result<PreparedChatRender, ChatRenderFailure> {
        let mut output: [Vec<u8>; 2] = [Vec::new(), Vec::new()];
        let result = (|| -> Result<(), ChatRenderCause> {
            let mut env = self.source.env.clone();
            env.set_fuel(Some(self.source.limits.fuel));
            env.set_recursion_limit(self.source.limits.recursion_limit);
            install_chat_clock(&mut env, self.clock);
            let template = env
                .get_template(self.source.name())
                .map_err(ChatRenderCause::Template)?;
            for (index, destination) in output.iter_mut().enumerate() {
                let capacity = self.source.limits.max_output_bytes;
                #[cfg(any(test, feature = "tokenizer-compiler-test-support"))]
                let capacity = if self.fail
                    == Some(if index == 0 {
                        ChatRenderBuffer::WithoutPrompt
                    } else {
                        ChatRenderBuffer::WithPrompt
                    }) {
                    usize::MAX
                } else {
                    capacity
                };
                destination
                    .try_reserve_exact(capacity)
                    .map_err(ChatRenderCause::Allocation)?;
                let context = self
                    .context
                    .values(index == 1)
                    .map_err(ChatRenderCause::Json)?;
                let mut writer = Output {
                    bytes: destination,
                    limit: self.source.limits.max_output_bytes,
                    exceeded: false,
                };
                let result = template.render_captured_to(context, &mut writer);
                if writer.exceeded {
                    return Err(ChatRenderCause::OutputLimit);
                }
                result.map_err(ChatRenderCause::Template)?;
            }
            Ok(())
        })();
        if let Err(cause) = result {
            return Err(ChatRenderFailure { cause, output });
        }
        Ok(PreparedChatRender {
            output: output.map(|bytes| String::from_utf8(bytes).expect("template writes UTF-8")),
        })
    }
}
struct Output<'a> {
    bytes: &'a mut Vec<u8>,
    limit: usize,
    exceeded: bool,
}
impl io::Write for Output<'_> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if bytes.len() > self.limit - self.bytes.len() {
            self.exceeded = true;
            return Err(io::ErrorKind::FileTooLarge.into());
        }
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
/// Two completed immutable UTF-8 prompts. No caller or VM reference escapes.
#[derive(Debug)]
pub struct PreparedChatRender {
    output: [String; 2],
}
impl PreparedChatRender {
    /// Prompt with or without generation suffix policy.
    pub fn prompt(&self, generation_prompt: bool) -> &str {
        &self.output[usize::from(generation_prompt)]
    }
    /// Suffix when the generation rendering extends the ordinary rendering.
    pub fn generation_suffix(&self) -> &str {
        self.output[1].strip_prefix(&self.output[0]).unwrap_or("")
    }
    /// Actual capacity of the two retained first-party output strings.
    pub fn retained_buffer_bytes(&self) -> usize {
        self.output.iter().map(String::capacity).sum()
    }
}
/// Actual execution, limit or first-party allocation failure.
#[derive(Debug, thiserror::Error)]
pub enum ChatRenderCause {
    /// Original upstream error and its cause chain.
    #[error(transparent)]
    Template(minijinja::Error),
    /// Context serialization error.
    #[error(transparent)]
    Json(serde_json::Error),
    /// Failure of an actual output-buffer reserve.
    #[error(transparent)]
    Allocation(TryReserveError),
    /// No output is grown beyond its configured capacity.
    #[error("chat output byte limit exceeded")]
    OutputLimit,
}
impl ChatRenderCause {
    /// Unknown function semantics, distinct from limits and allocation failures.
    pub fn is_unknown_function(&self) -> bool {
        matches!(self, Self::Template(e) if e.kind() == minijinja::ErrorKind::UnknownFunction)
    }
    /// Input/template errors that can make a behavioral probe a nonmatch.
    /// Fuel, output and recursion limits remain hard failures.
    pub fn is_template_rejection(&self) -> bool {
        let Self::Template(error) = self else {
            return false;
        };
        if error
            .detail()
            .is_some_and(|d| d.contains("recursion limit"))
        {
            return false;
        }
        use minijinja::ErrorKind::*;
        matches!(
            error.kind(),
            NonPrimitive
                | NonKey
                | InvalidOperation
                | TooManyArguments
                | MissingArgument
                | UnknownFilter
                | UnknownTest
                | UnknownFunction
                | UnknownMethod
                | BadEscape
                | UndefinedError
                | CannotUnpack
        )
    }
}
/// Failure retaining partial first-party output under the operation reservation.
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
pub struct ChatRenderFailure {
    #[source]
    cause: ChatRenderCause,
    output: [Vec<u8>; 2],
}
impl ChatRenderFailure {
    /// Actual retained output capacity; upstream temporaries have retired.
    pub fn retained_buffer_bytes(&self) -> usize {
        self.output.iter().map(Vec::capacity).sum()
    }
    /// Actual failure without releasing output custody.
    pub fn cause(&self) -> &ChatRenderCause {
        &self.cause
    }
}
#[cfg(test)]
mod tests;
