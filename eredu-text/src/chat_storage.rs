//! Closed fresh chat source and rendering, without runtime accounting authority.
use minijinja::bounded as vm;
/// Immutable local calendar facts shared across chat render attempts.
pub use vm::clock::Snapshot as ChatClockSnapshot;
/// Observe the host clock once at request-context preparation, before measurement.
pub fn chat_clock_snapshot() -> ChatClockSnapshot {
    ChatClockSnapshot::from_local(chrono::Local::now().naive_local())
}
use std::{fmt, mem::size_of};
use tokenizers::utils::json_view::{Document, StringValue, Value};
mod numeric;
mod source_identity;
use numeric::NumericPlan;
pub use numeric::{ChatNumericError, ChatNumericPlanError};
use source_identity::SourceIdentity;

pub use vm::{
    InputError as ChatMessageError, JsonCapacity as ChatJsonCapacity, Messages as ChatMessages,
    RenderBuffer as ChatRenderBuffer, RenderContext as ChatRenderContext,
    RenderPlanError as ChatRenderPlanError, ScalarBinding as ChatScalarBinding,
    ScalarBindingValue as ChatScalarBindingValue, SourceBuffer as ChatSourceBuffer,
    TextCapacity as ChatTextCapacity, TextMessage, ValueCapacity as ChatValueCapacity,
};

/// Fixed source-selection or checked-construction planning failure.
#[derive(Debug)]
pub enum ChatSourceError {
    /// Actual complete-document UTF-8/JSON validation failure.
    Json(tokenizers::utils::json_view::Error),
    /// Valid JSON outside the dependency-owned numeric storage profile.
    Numeric(ChatNumericPlanError),
    /// Config/root/chat_template or named-entry representation is invalid.
    TemplateProfile,
    /// No default text template is selected by this actual config.
    MissingTemplate,
    /// Distinct named entries contain the same decoded name.
    DuplicateName,
    /// Actual template/settings exceed the checked compiler construction profile.
    Source(vm::SourceError),
    /// Concrete wrapper/layout addition overflowed.
    Overflow,
}
impl fmt::Display for ChatSourceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Json(_) => "invalid chat source JSON",
            Self::Numeric(_) => "chat config numeric construction profile is unfinished",
            Self::TemplateProfile => "invalid chat template selection profile",
            Self::MissingTemplate => "no default chat template",
            Self::DuplicateName => "duplicate chat template name",
            Self::Source(_) => "chat template construction profile is unfinished",
            Self::Overflow => "chat source layout overflow",
        })
    }
}
impl std::error::Error for ChatSourceError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Json(error) => Some(error),
            Self::Numeric(error) => Some(error),
            Self::Source(error) => Some(error),
            _ => None,
        }
    }
}
#[derive(Debug)]
enum Input<'a> {
    Utf8(&'a str),
    Config {
        document: Document<'a>,
        selected: Option<StringValue<'a>>,
    },
}

#[derive(Clone)]
enum SourceBytes<'a> {
    Utf8(std::str::Bytes<'a>),
    Json(tokenizers::utils::json_view::Bytes<'a>),
}
impl Iterator for SourceBytes<'_> {
    type Item = u8;
    fn next(&mut self) -> Option<u8> {
        match self {
            Self::Utf8(value) => value.next(),
            Self::Json(value) => value.next(),
        }
    }
}
#[derive(Debug)]
enum CompilerPlan<'a> {
    Image(vm::CompilePlan<'a>),
    Source(vm::SourceCompilePlan<'a, crate::generation_blocks::Bytes<SourceBytes<'a>>>),
}
impl CompilerPlan<'_> {
    fn requirements(&self) -> vm::CompileRequirements {
        match self {
            Self::Image(plan) => plan.requirements(),
            Self::Source(plan) => plan.requirements(),
        }
    }
    fn compile(self) -> Result<vm::PreparedTemplate, vm::CompileError> {
        match self {
            Self::Image(plan) => plan.compile(),
            Self::Source(plan) => plan.compile(),
        }
    }
    #[cfg(feature = "tokenizer-compiler-test-support")]
    fn fail_reservation(self, buffer: vm::SourceBuffer) -> Self {
        match self {
            Self::Image(plan) => Self::Image(plan.fail_reservation(buffer)),
            Self::Source(plan) => Self::Source(plan.fail_reservation(buffer)),
        }
    }
}

/// Concrete fresh source layouts and all text adapter controls.
#[derive(Clone, Copy, Debug)]
pub struct ChatTemplateRequirements {
    buffers: usize,
    controls: usize,
    total: usize,
}
impl ChatTemplateRequirements {
    /// Requested decoded source, syntax, compiler continuation and output allocations.
    pub fn buffer_bytes(self) -> usize {
        self.buffers
    }
    /// Fixed parser/adapter/construction/error/return representations.
    pub fn control_bytes(self) -> usize {
        self.controls
    }
    /// Checked sum before runtime wrapper/account controls.
    pub fn required_bytes(self) -> usize {
        self.total
    }
}
/// Move-only plan retaining the actual original source/config borrow.
#[derive(Debug)]
pub struct ChatTemplatePlan<'a> {
    input: Input<'a>,
    inner: Result<CompilerPlan<'a>, ChatSourceError>,
    numeric: Option<NumericPlan<'a>>,
    requirements: ChatTemplateRequirements,
}
impl<'a> ChatTemplatePlan<'a> {
    /// Borrow exact UTF-8 template source under the ordinary chat settings/name.
    pub fn prepare_utf8(source: &'a str, model_id: &'a str) -> Result<Self, ChatSourceError> {
        let name = vm::TemplateName::Single(model_id);
        let settings = vm::TemplateSettings::text_chat();
        let inner = if source == vm::supported_source() {
            CompilerPlan::Image(
                vm::CompilePlan::prepare(source, name, settings)
                    .map_err(ChatSourceError::Source)?,
            )
        } else {
            CompilerPlan::Source(
                vm::SourceCompilePlan::prepare(
                    crate::generation_blocks::Bytes::new(SourceBytes::Utf8(source.bytes())),
                    name,
                    settings,
                )
                .map_err(ChatSourceError::Source)?,
            )
        };
        Self::finish(Input::Utf8(source), Ok(inner), None)
    }
    /// Validate complete JSON grammar and plan its actual text default without
    /// owned config values. Numeric conversion uses the ordinary scalar parser
    /// only during admitted compilation; its error precedes a retained selection
    /// failure. No serde map, normalization String, Environment or previously
    /// compiled source is constructed or adopted.
    pub fn prepare_config(config: &'a [u8], model_id: &'a str) -> Result<Self, ChatSourceError> {
        let document = Document::parse(config).map_err(ChatSourceError::Json)?;
        let numeric = NumericPlan::prepare(document)?;
        let planned = (|| {
            let object = document
                .root()
                .object()
                .ok_or(ChatSourceError::TemplateProfile)?;
            let value = object
                .get("chat_template")
                .ok_or(ChatSourceError::MissingTemplate)?;
            if value.is_null() {
                return Err(ChatSourceError::MissingTemplate);
            }
            let (selected, named) = if let Some(source) = value.string() {
                (source, false)
            } else {
                (select_default(value)?, true)
            };
            let name = if named {
                vm::TemplateName::Default(model_id)
            } else {
                vm::TemplateName::Single(model_id)
            };
            let settings = vm::TemplateSettings::text_chat();
            // The generic path retains the actual validated decoded-byte iterator.
            // No decoded String, syntax node or bytecode reserve precedes J.
            let inner = if selected.is(vm::supported_source()) {
                CompilerPlan::Image(
                    vm::CompilePlan::prepare(vm::supported_source(), name, settings)
                        .map_err(ChatSourceError::Source)?,
                )
            } else {
                CompilerPlan::Source(
                    vm::SourceCompilePlan::prepare(
                        crate::generation_blocks::Bytes::new(SourceBytes::Json(selected.bytes())),
                        name,
                        settings,
                    )
                    .map_err(ChatSourceError::Source)?,
                )
            };
            Ok((selected, inner))
        })();
        let (selected, inner) = match planned {
            Ok((selected, inner)) => (Some(selected), Ok(inner)),
            Err(error) if numeric.has_numbers() => (None, Err(error)),
            Err(error) => return Err(error),
        };
        Self::finish(Input::Config { document, selected }, inner, Some(numeric))
    }
    fn finish(
        input: Input<'a>,
        inner: Result<CompilerPlan<'a>, ChatSourceError>,
        numeric: Option<NumericPlan<'a>>,
    ) -> Result<Self, ChatSourceError> {
        let facts = inner.as_ref().ok().map(CompilerPlan::requirements);
        let compiler_controls = match facts {
            Some(facts) => facts
                .required_bytes()
                .checked_sub(facts.buffer_bytes())
                .ok_or(ChatSourceError::Overflow)?,
            None => 0,
        };
        let buffers = facts
            .map_or(0, |facts| facts.buffer_bytes())
            .checked_add(numeric.as_ref().map_or(0, NumericPlan::buffers))
            .ok_or(ChatSourceError::Overflow)?;
        let controls = [
            compiler_controls,
            source_identity::control_bytes::<SourceBytes<'a>>().ok_or(ChatSourceError::Overflow)?,
            crate::generation_blocks::control_bytes::<SourceBytes<'a>>()
                .ok_or(ChatSourceError::Overflow)?,
            numeric.as_ref().map_or(0, NumericPlan::controls),
            Document::control_bytes().ok_or(ChatSourceError::Overflow)?,
            size_of::<Input<'a>>(),
            size_of::<Self>(),
            size_of::<Result<Self, ChatSourceError>>(),
            size_of::<ChatSourceError>(),
            size_of::<ChatTemplateRequirements>(),
            size_of::<PreparedChatTemplate>(),
            size_of::<ChatTemplateFailure>(),
            size_of::<Result<PreparedChatTemplate, ChatTemplateFailure>>(),
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)
        .ok_or(ChatSourceError::Overflow)?;
        let total = controls
            .checked_add(buffers)
            .ok_or(ChatSourceError::Overflow)?;
        Ok(Self {
            input,
            inner,
            numeric,
            requirements: ChatTemplateRequirements {
                buffers,
                controls,
                total,
            },
        })
    }
    /// Actual sealed source-derived requirements before any compiler reserve.
    pub fn requirements(&self) -> ChatTemplateRequirements {
        self.requirements
    }
    /// Development-only real requested-capacity overflow at one source target.
    #[cfg(feature = "tokenizer-compiler-test-support")]
    pub fn fail_reservation(mut self, buffer: vm::SourceBuffer) -> Self {
        self.inner = self.inner.map(|inner| inner.fail_reservation(buffer));
        self
    }
    /// Compile fresh source bytes under the consumer's original allowance.
    ///
    /// A source plan cannot authorize a second constructor attempt.
    /// ```compile_fail
    /// use eredu_text::chat_storage::ChatTemplatePlan;
    /// fn twice(plan: ChatTemplatePlan<'_>) {
    ///     let first = plan.compile();
    ///     let second = plan.compile();
    /// }
    /// ```
    pub fn compile(self) -> Result<PreparedChatTemplate, ChatTemplateFailure> {
        let Self {
            input,
            inner,
            numeric,
            ..
        } = self;
        if let Some(numeric) = numeric {
            numeric.validate().map_err(|error| ChatTemplateFailure {
                inner: Failure::Numeric(error),
            })?;
        }
        let inner = inner.map_err(|error| ChatTemplateFailure {
            inner: Failure::Source(error),
        })?;
        let identity = match &input {
            Input::Utf8(source) => SourceIdentity::new(source.bytes()),
            Input::Config { selected, .. } => {
                SourceIdentity::new(selected.expect("successful source selection").bytes())
            }
        };
        let result = inner.compile();
        // Keep the real input view live through construction; the successful
        // destination owns all future source bytes, and failures retain every
        // real completed destination prefix.
        match input {
            Input::Utf8(source) => {
                if let Ok(value) = &result {
                    debug_assert!(
                        crate::generation_blocks::Bytes::new(source.bytes())
                            .eq(value.source().bytes())
                    );
                }
            }
            Input::Config { document, selected } => {
                if let Ok(value) = &result {
                    debug_assert!(
                        crate::generation_blocks::Bytes::new(
                            selected.expect("successful source selection").bytes()
                        )
                        .eq(value.source().bytes())
                    );
                }
                let _ = document.source();
            }
        }
        result
            .map(|inner| PreparedChatTemplate { inner, identity })
            .map_err(|inner| ChatTemplateFailure {
                inner: Failure::Compiler(inner),
            })
    }
}
fn entry(value: Value<'_>) -> Result<(StringValue<'_>, StringValue<'_>), ChatSourceError> {
    let object = value.object().ok_or(ChatSourceError::TemplateProfile)?;
    if object.unique_len() != 2 {
        return Err(ChatSourceError::TemplateProfile);
    }
    let name = object
        .get("name")
        .and_then(Value::string)
        .ok_or(ChatSourceError::TemplateProfile)?;
    let template = object
        .get("template")
        .and_then(Value::string)
        .ok_or(ChatSourceError::TemplateProfile)?;
    if name.is_empty() {
        return Err(ChatSourceError::TemplateProfile);
    }
    Ok((name, template))
}
fn select_default(value: Value<'_>) -> Result<StringValue<'_>, ChatSourceError> {
    let array = value.array().ok_or(ChatSourceError::TemplateProfile)?;
    let mut selected = None;
    let mut count = 0usize;
    for (index, item) in array.enumerate() {
        let (name, template) = entry(item)?;
        for prior in value.array().expect("validated array").take(index) {
            let (prior, _) = entry(prior)?;
            if name.same(prior) {
                return Err(ChatSourceError::DuplicateName);
            }
        }
        if name.is("default") {
            selected = Some(template);
        }
        count = count.checked_add(1).ok_or(ChatSourceError::Overflow)?;
    }
    if count == 0 {
        return Err(ChatSourceError::TemplateProfile);
    }
    selected.ok_or(ChatSourceError::MissingTemplate)
}
/// Fresh closed source with named readonly projections, no raw VM escape.
#[derive(Debug)]
pub struct PreparedChatTemplate {
    inner: vm::PreparedTemplate,
    identity: SourceIdentity,
}
impl PreparedChatTemplate {
    /// Actual selected name retained by the compiler destination.
    pub fn name(&self) -> &str {
        self.inner.name()
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
            ModelChatTemplate::Named(templates) => {
                templates
                    .get("default")
                    .is_some_and(|source| self.identity == SourceIdentity::new(source.bytes()))
                    && self.name().strip_prefix(model_id) == Some("::chat_template::default")
            }
        }
    }

    /// Legacy message-only render calls preserve these defaults because they do
    /// not replace either intrinsic input. General contexts must instead use
    /// `render_plan_with_context`, which qualifies every actually used value.
    pub fn accepts_default_variables(
        &self,
        values: &serde_json::Map<String, serde_json::Value>,
    ) -> bool {
        !values.contains_key("messages") && !values.contains_key("add_generation_prompt")
    }
    /// Actual retained source destination capacities.
    pub fn retained_buffer_bytes(&self) -> usize {
        self.inner.retained_buffer_bytes()
    }
}
/// Owning source failure retaining real constructor partial allocations.
#[derive(Debug)]
pub struct ChatTemplateFailure {
    inner: Failure,
}
#[derive(Debug)]
enum Failure {
    Compiler(vm::CompileError),
    Numeric(ChatNumericError),
    Source(ChatSourceError),
}
impl ChatTemplateFailure {
    /// Actual retained compiler prefix or numeric error allocation.
    pub fn retained_buffer_bytes(&self) -> usize {
        match &self.inner {
            Failure::Compiler(error) => error.retained_buffer_bytes(),
            Failure::Numeric(error) => error.retained_bytes(),
            Failure::Source(_) => 0,
        }
    }
    /// Real compiler cause when numeric and source selection validation succeeded.
    pub fn cause(&self) -> Option<&vm::CompileCause> {
        match &self.inner {
            Failure::Compiler(error) => Some(error.cause()),
            _ => None,
        }
    }
    /// Actual ordinary numeric-parser failure, before template construction.
    pub fn numeric_error(&self) -> Option<&ChatNumericError> {
        match &self.inner {
            Failure::Numeric(error) => Some(error),
            _ => None,
        }
    }
    /// Source selection refusal after the actual numeric validation succeeded.
    pub fn source_error(&self) -> Option<&ChatSourceError> {
        match &self.inner {
            Failure::Source(error) => Some(error),
            _ => None,
        }
    }
}
impl fmt::Display for ChatTemplateFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.inner {
            Failure::Compiler(error) => error.fmt(f),
            Failure::Numeric(error) => error.fmt(f),
            Failure::Source(error) => error.fmt(f),
        }
    }
}
impl std::error::Error for ChatTemplateFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(match &self.inner {
            Failure::Compiler(error) => error,
            Failure::Numeric(error) => error,
            Failure::Source(error) => error,
        })
    }
}

/// Actual H destination layouts and concrete text adapter/error controls.
#[derive(Clone, Copy, Debug)]
pub struct ChatRenderRequirements {
    buffers: usize,
    controls: usize,
    total: usize,
}
impl ChatRenderRequirements {
    /// Sum of six requested render destination layouts.
    pub fn buffer_bytes(self) -> usize {
        self.buffers
    }
    /// Fixed measurement, adapter, error and return representations.
    pub fn control_bytes(self) -> usize {
        self.controls
    }
    /// Checked total before runtime original-owner wrappers.
    pub fn required_bytes(self) -> usize {
        self.total
    }
}
/// Move-only render plan borrowing the actual closed source and message slice.
#[derive(Debug)]
pub struct ChatRenderPlan<'a> {
    source: &'a PreparedChatTemplate,
    inner: vm::RenderPlan<'a>,
    requirements: ChatRenderRequirements,
}
impl PreparedChatTemplate {
    /// Pure layout of the shared measurement workspace and wrapper controls.
    /// Original callers reserve this before measuring the selected input.
    pub fn render_prefix_bytes(&self) -> Result<usize, vm::RenderPlanError> {
        self.render_prefix_bytes_with_json_capacity(ChatJsonCapacity::default())
    }
    /// Pure layout for the source and one explicit JSON scratch attempt.
    pub fn render_prefix_bytes_with_json_capacity(
        &self,
        json: ChatJsonCapacity,
    ) -> Result<usize, vm::RenderPlanError> {
        self.render_prefix_bytes_with_capacities(json, ChatTextCapacity::default())
    }
    /// Pure layout of a single reached JSON/generated-text prefix attempt.
    pub fn render_prefix_bytes_with_capacities(
        &self,
        json: ChatJsonCapacity,
        text: ChatTextCapacity,
    ) -> Result<usize, vm::RenderPlanError> {
        self.render_prefix_bytes_with_values(json, text, ChatValueCapacity::default())
    }
    /// Pure admitted prefix layouts including immutable generated values.
    pub fn render_prefix_bytes_with_values(
        &self,
        json: ChatJsonCapacity,
        text: ChatTextCapacity,
        values: ChatValueCapacity,
    ) -> Result<usize, vm::RenderPlanError> {
        let controls = [
            size_of::<ChatJsonCapacity>(),
            size_of::<ChatTextCapacity>(),
            size_of::<ChatValueCapacity>(),
            size_of::<ChatRenderPlan<'_>>(),
            size_of::<ChatRenderRequirements>(),
            size_of::<Result<ChatRenderPlan<'_>, vm::RenderPlanError>>(),
            size_of::<Result<usize, vm::RenderPlanError>>(),
        ];
        controls
            .into_iter()
            .try_fold(
                vm::RenderPlan::prefix_bytes_with_values(&self.inner, json, text, values)?,
                usize::checked_add,
            )
            .ok_or(vm::RenderPlanError::Overflow)
    }
    /// Measure both ordinary generation-prompt variants through the shared
    /// dispatcher with source-derived scratch; no ordinary input list is built.
    pub fn render_plan<'a>(
        &'a self,
        messages: ChatMessages<'a>,
    ) -> Result<ChatRenderPlan<'a>, vm::RenderPlanError> {
        self.render_plan_with_context(ChatRenderContext::from_messages(messages))
    }
    /// Use the exact borrowed external context for measurement and admitted
    /// execution; no merged map or caller-owned value enters retained buffers.
    pub fn render_plan_with_context<'a>(
        &'a self,
        context: ChatRenderContext<'a>,
    ) -> Result<ChatRenderPlan<'a>, vm::RenderPlanError> {
        let inner = vm::RenderPlan::prepare_context(&self.inner, context)?;
        self.finish_render_plan(inner)
    }
    /// Measure exactly one pre-admitted temporary destination. Capacity refusal
    /// returns the next actual source/input requirement without growing storage.
    pub fn render_plan_attempt<'a>(
        &'a self,
        context: ChatRenderContext<'a>,
        json: ChatJsonCapacity,
    ) -> Result<ChatRenderPlan<'a>, vm::RenderPlanError> {
        let inner = vm::RenderPlan::prepare_context_with_json_capacity(&self.inner, context, json)?;
        self.finish_render_plan(inner)
    }
    /// One admitted generated-prefix attempt, without destination growth.
    pub fn render_plan_attempt_with_text<'a>(
        &'a self,
        context: ChatRenderContext<'a>,
        json: ChatJsonCapacity,
        text: ChatTextCapacity,
    ) -> Result<ChatRenderPlan<'a>, vm::RenderPlanError> {
        let inner =
            vm::RenderPlan::prepare_context_with_capacities(&self.inner, context, json, text)?;
        self.finish_render_plan(inner)
    }
    /// One fixed value/text/JSON attempt; the caller settles before retrying.
    pub fn render_plan_attempt_with_values<'a>(
        &'a self,
        context: ChatRenderContext<'a>,
        json: ChatJsonCapacity,
        text: ChatTextCapacity,
        values: ChatValueCapacity,
    ) -> Result<ChatRenderPlan<'a>, vm::RenderPlanError> {
        let inner =
            vm::RenderPlan::prepare_context_with_values(&self.inner, context, json, text, values)?;
        self.finish_render_plan(inner)
    }
    fn finish_render_plan<'a>(
        &'a self,
        inner: vm::RenderPlan<'a>,
    ) -> Result<ChatRenderPlan<'a>, vm::RenderPlanError> {
        let facts = inner.requirements();
        let controls = [
            facts
                .required_bytes()
                .checked_sub(facts.buffer_bytes())
                .ok_or(vm::RenderPlanError::Overflow)?,
            size_of::<ChatRenderPlan<'a>>(),
            size_of::<ChatRenderRequirements>(),
            size_of::<Result<ChatRenderPlan<'a>, vm::RenderPlanError>>(),
            size_of::<PreparedChatRender>(),
            size_of::<ChatRenderFailure>(),
            size_of::<Result<PreparedChatRender, ChatRenderFailure>>(),
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)
        .ok_or(vm::RenderPlanError::Overflow)?;
        let total = controls
            .checked_add(facts.buffer_bytes())
            .ok_or(vm::RenderPlanError::Overflow)?;
        Ok(ChatRenderPlan {
            source: self,
            inner,
            requirements: ChatRenderRequirements {
                buffers: facts.buffer_bytes(),
                controls,
                total,
            },
        })
    }
}
impl ChatRenderPlan<'_> {
    /// Exact borrowed source-object match, not a spelling/digest/capacity test.
    /// Runtime still authenticates its actual original owner and pool.
    pub fn is_for(&self, source: &PreparedChatTemplate) -> bool {
        std::ptr::eq(self.source, source)
    }
    /// Actual measured layouts before any destination allocation.
    pub fn requirements(&self) -> ChatRenderRequirements {
        self.requirements
    }
    /// Development-only overflow at the selected actual destination reserve.
    #[cfg(feature = "tokenizer-compiler-test-support")]
    pub fn fail_reservation(mut self, buffer: vm::RenderBuffer) -> Self {
        self.inner = self.inner.fail_reservation(buffer);
        self
    }
    /// Consume both actual borrows through fresh construction and both renders.
    ///
    /// The same source/input plan cannot refill a second render destination.
    /// ```compile_fail
    /// use eredu_text::chat_storage::ChatRenderPlan;
    /// fn twice(plan: ChatRenderPlan<'_>) {
    ///     let first = plan.render();
    ///     let second = plan.render();
    /// }
    /// ```
    pub fn render(self) -> Result<PreparedChatRender, ChatRenderFailure> {
        self.inner
            .render()
            .map(|inner| PreparedChatRender { inner })
            .map_err(|inner| ChatRenderFailure { inner })
    }
}
/// Completed immutable prompt variants with retained fixed scratch capacities.
#[derive(Debug)]
pub struct PreparedChatRender {
    inner: vm::Rendered,
}
impl PreparedChatRender {
    /// Selected exact output with no copied String or mutable extraction.
    pub fn prompt(&self, generation_prompt: bool) -> &str {
        self.inner.prompt(generation_prompt)
    }
    /// Borrowed suffix range in the true rendering, or empty for a non-prefix.
    pub fn generation_suffix(&self) -> &str {
        self.inner.generation_suffix()
    }
    /// Actual retained destination capacities.
    pub fn retained_buffer_bytes(&self) -> usize {
        self.inner.retained_buffer_bytes()
    }
}
/// Closed render failure retaining all real partial destinations.
#[derive(Debug)]
pub struct ChatRenderFailure {
    inner: vm::RenderError,
}
impl ChatRenderFailure {
    /// Actual completed prefix capacity still held by this error.
    pub fn retained_buffer_bytes(&self) -> usize {
        self.inner.retained_buffer_bytes()
    }
    /// Actual reserve cause or fixed instruction/location failure.
    pub fn cause(&self) -> &vm::RenderCause {
        self.inner.cause()
    }
}
impl fmt::Display for ChatRenderFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.inner, f)
    }
}
impl std::error::Error for ChatRenderFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.inner)
    }
}

#[cfg(test)]
mod tests;

/// Immutable input records for source-owned profile probes; no object callbacks.
pub use minijinja::bounded::{
    RecordField as ChatRecordField, RecordFields as ChatRecordFields,
    RecordValue as ChatRecordValue,
};
