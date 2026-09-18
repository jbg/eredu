//! Immutable literal program and actual structural-token rows, admitted before copy.
use super::{
    OriginalForbiddenSource, OriginalTokenTrieSource, OriginalTokenizer, OriginalControllerCompilation, WorkingMemoryError, WorkingMemoryPool,
    original_declaration_source::Account,
};
use eredu_core::{HostPreparationAuthority, SpeculativeBuffer, SpeculativeBufferAllocationError,
    SharedTokenFilter, SharedControllerBytes, SharedControllerDeclaration};
use eredu_core::speculative::{PreparedGrammarController, PreparedGrammarSource};
use eredu_text::semantic_channels::{ChannelProgram, DelimitedChannel, JsonEnvelope, JsonToolLayout, JsonToolProgram, JsonToolShape};
use eredu_text::semantic_channels::tagged::{TaggedToolProgram, TaggedEncoding, ToolProgram};
use eredu_text::json_fragments::{JsonCallId, JsonFieldNames};
use std::{
    alloc::Layout,
    mem::{size_of, size_of_val},
    sync::{Arc, atomic::AtomicUsize},
};

/// Immutable full-schema completion source supplied by facade policy.
/// Implementations retain their exact compiled source and paid source account.
/// Validation uses the ordinary validator, paying every reached scratch worker;
/// it must refuse before an unqualified worker and retain its first failure.
pub trait OriginalToolValidation: std::fmt::Debug + Send + Sync {
    /// Whether this exact source retains tagged parameter semantics for a tool.
    fn contains_tagged_tool(&self, _name: &str) -> bool { false }
    /// Check the original required field list against actual consumed names.
    fn tagged_missing_required(&self, _name: &str, _parameters: &eredu_text::semantic_channels::tagged::TaggedParameters) -> bool { true }
    /// Interpret one tagged value through the shared text worker and its actual
    /// original parameter validator. JSON-only sources do not provide this capability.
    fn parse_tagged_parameter(&self, _name: &str, _parameter: &str, _declared: Option<&str>, _raw: &str,
        _allocation: &dyn serde_json::allocation::Allocation, _funding: &eredu_nn::workspace::HostMetadataFunding)
        -> Result<serde_json::Value, eredu_core::BackendFailure> {
        Err(eredu_core::TokenInputRejection::Unsupported.into_backend_failure())
    }
    /// Compare exact source/controller/pool identity without reconstruction or
    /// allocation. The source's retained census covers this fixed query.
    fn validate_source(&self, controller: OriginalSemanticControllerSource<'_>, pool: &WorkingMemoryPool)
        -> Result<(), WorkingMemoryError>;
    /// Exact error-owner and fixed callback transport population. The channel
    /// reserves it before calling validate, so even its first funding refusal
    /// can retain the actual source without an unfunded error allocation.
    fn failure_control_bytes(&self) -> Option<usize>;
    /// Validate one complete raw argument object under the supplied invocation
    /// payer. No grammar acceptance is a substitute for the original full schema.
    fn validate(&self, name: &str, arguments: &str, funding: &eredu_nn::workspace::HostMetadataFunding)
        -> Result<(), eredu_core::BackendFailure>;
}

/// Exact borrowed controller source paired with the facade's semantic program.
/// Grammar inputs remain distinct from forbidden triggers and carry no mutable
/// parser or native execution authority into the channel owner.
#[derive(Debug, Clone, Copy)]
pub enum OriginalSemanticControllerSource<'a> {
    /// Existing original forbidden-trigger input owner.
    Forbidden(&'a OriginalForbiddenSource),
    /// Actual prepared grammar source, authenticated against its original trie.
    Grammar(PreparedGrammarSource<'a>),
}
impl OriginalSemanticControllerSource<'_> {
    fn validate(self, pool: &WorkingMemoryPool) -> Result<(), WorkingMemoryError> {
        match self {
            Self::Forbidden(source) => source.validate_pool(pool),
            Self::Grammar(source) => source.tokenizer().downcast_ref::<OriginalTokenTrieSource>()
                .ok_or(WorkingMemoryError::IdentityMismatch)?.validate_grammar_source(source, pool),
        }
    }
    fn retain(self) -> Result<ControllerSource, WorkingMemoryError> {
        Ok(match self {
            Self::Forbidden(source) => ControllerSource::Forbidden(source.clone()),
            Self::Grammar(source) => ControllerSource::Grammar {
                trie: source.tokenizer().downcast_ref::<OriginalTokenTrieSource>()
                    .ok_or(WorkingMemoryError::IdentityMismatch)?.clone(),
                validity: source.validity().clone(), recipe: source.recipe().clone(),
                declaration: source.declaration().clone(),
                compilation: source.compilation().downcast_ref::<OriginalControllerCompilation>()
                    .ok_or(WorkingMemoryError::IdentityMismatch)?.clone(),
            },
        })
    }
}
#[derive(Debug)]
enum ControllerSource {
    Forbidden(OriginalForbiddenSource),
    Grammar { trie: OriginalTokenTrieSource, validity: SharedTokenFilter,
        recipe: SharedControllerBytes, declaration: SharedControllerDeclaration, compilation: OriginalControllerCompilation },
}
impl ControllerSource {
    fn validate(&self, pool: &WorkingMemoryPool) -> Result<(), WorkingMemoryError> {
        match self {
            Self::Forbidden(source) => source.validate_pool(pool),
            Self::Grammar { trie, validity, recipe, declaration, compilation } =>
                trie.validate_grammar_inputs(validity, recipe, declaration, compilation, pool),
        }
    }
    fn matches<C: eredu_core::SpeculativeTokenFilterController>(
        &self, controller: &C, pool: &WorkingMemoryPool,
    ) -> Result<(), WorkingMemoryError> {
        let source = match (controller.prepared_grammar(), controller.prepared_forbidden_source(), controller.prepared_plain_source()) {
            (Some(grammar), None, None) => eredu_core::PreparedControllerSource::Grammar(grammar.prepared_grammar_source()),
            (None, Some(source), None) => eredu_core::PreparedControllerSource::Forbidden(source),
            _ => return Err(WorkingMemoryError::IdentityMismatch),
        };
        self.matches_source(source, pool)
    }
    fn matches_source(&self, actual: eredu_core::PreparedControllerSource<'_>, pool: &WorkingMemoryPool)
        -> Result<(), WorkingMemoryError> {
        match (self, actual) {
            (Self::Forbidden(source), eredu_core::PreparedControllerSource::Forbidden(actual)) =>
                source.validate_controller(actual, pool),
            (Self::Grammar { trie, validity, recipe, declaration, compilation }, eredu_core::PreparedControllerSource::Grammar(actual)) => {
                trie.validate_grammar_source(actual, pool)?;
                if !actual.compilation().downcast_ref::<OriginalControllerCompilation>()
                    .is_some_and(|actual| compilation.same_compilation(actual)) {
                    return Err(WorkingMemoryError::IdentityMismatch);
                }
                if !validity.same_storage(actual.validity()) || !recipe.same_storage(actual.recipe())
                    || !declaration.same_storage(actual.declaration()) { return Err(WorkingMemoryError::IdentityMismatch); }
                Ok(())
            }
            _ => Err(WorkingMemoryError::IdentityMismatch),
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct Span {
    start: usize,
    end: usize,
}
#[derive(Debug, Clone, Copy)]
struct Channel {
    prefix: Span,
    suffix: Span,
    prefilled: bool,
}
#[derive(Debug, Clone, Copy)]
struct Structural {
    id: u32,
    spelling: Span,
    stop: bool,
}
#[derive(Debug, Clone, Copy)]
struct JsonTools {
    literals: [Span; 10],
    id_length: Option<Option<usize>>,
    shape: JsonToolShape,
    layout: JsonToolLayout,
}
#[derive(Debug, Clone, Copy)]
struct TaggedTools { literals: [Span; 13], has_type: bool, strip_framing: bool }
#[derive(Debug, Clone, Copy)]
enum Tools { Json(JsonTools), Tagged(TaggedTools) }
#[derive(Debug)]
struct Payload {
    bytes: SpeculativeBuffer<u8>,
    structural: SpeculativeBuffer<Structural>,
    reasoning: Option<Channel>,
    text: Option<Channel>,
    tool: Span,
    tool_is_json: bool,
    tools: Option<Tools>,
    validation: Option<Arc<dyn OriginalToolValidation>>,
    tokenizer: OriginalTokenizer,
    controller: ControllerSource,
    // Actual data and both independent source aliases retire before this payer.
    account: Account,
}
/// Closed immutable program produced by the original source compiler. Literal
/// protocol selection is supplied by the facade; this owner supplies no model,
/// parser-payload, tokenizer replacement, or native execution permission.
#[derive(Debug)]
pub struct OriginalSemanticChannelSource(Option<Arc<Payload>>);
impl Clone for OriginalSemanticChannelSource {
    fn clone(&self) -> Self {
        Self(Some(self.0.as_ref().expect("live channel source").clone()))
    }
}
impl Drop for OriginalSemanticChannelSource {
    fn drop(&mut self) {
        if let Some(owner) = self.0.take() {
            drop(Arc::into_inner(owner));
        }
    }
}
impl OriginalSemanticChannelSource {
    fn payload(&self) -> &Payload {
        self.0.as_deref().expect("live channel source")
    }
    fn text(&self, span: Span) -> &str {
        std::str::from_utf8(&self.payload().bytes[span.start..span.end])
            .expect("copied whole UTF-8 literals")
    }
    /// Borrows the actual copied delimiters; no descriptor can replace them.
    pub fn program(&self) -> ChannelProgram<'_> {
        let channel = |c: Channel| DelimitedChannel {
            prefix: self.text(c.prefix),
            suffix: self.text(c.suffix),
            prefix_in_prompt: c.prefilled,
        };
        let p = self.payload();
        ChannelProgram {
            reasoning_channel: p.reasoning.map(channel),
            text_channel: p.text.map(channel),
            tool_delimiter: self.text(p.tool),
            tool_is_json: p.tool_is_json,
        }
    }
    /// Exact field and framing bytes from this same paid immutable source.
    /// Schema validation and mutable event storage require separate consumers.
    pub fn json_tools(&self) -> Option<JsonToolProgram<'_>> {
        let Tools::Json(t) = self.payload().tools? else { return None; };
        let literal = |index| self.text(t.literals[index]);
        Some(JsonToolProgram {
            output: JsonEnvelope { prefix: literal(0), suffix: literal(1) },
            call: JsonEnvelope { prefix: literal(2), suffix: literal(3) },
            function: JsonEnvelope { prefix: literal(4), suffix: literal(5) },
            fields: JsonFieldNames { name: literal(6), arguments: literal(7),
                call_id: t.id_length.map(|length| JsonCallId { field: literal(8), length }) },
            shape: t.shape, separator: literal(9), layout: t.layout,
        })
    }
    /// Borrow the exact tagged declaration from this source's paid literals.
    pub fn tagged_tools(&self) -> Option<TaggedToolProgram<'_>> {
        let Tools::Tagged(t) = self.payload().tools? else { return None; };
        let l = |i| self.text(t.literals[i]);
        Some(TaggedToolProgram { output: JsonEnvelope { prefix:l(0), suffix:l(1) },
            call: JsonEnvelope { prefix:l(2), suffix:l(3) },
            encoding: TaggedEncoding { function_prefix:l(4), function_name_suffix:l(5),
                parameter_prefix:l(6), parameter_name_suffix:l(7),
                parameter_type:t.has_type.then(|| JsonEnvelope { prefix:l(8), suffix:l(9) }),
                parameter_value_prefix:l(10), parameter_suffix:l(11), function_suffix:l(12),
                strip_value_framing:t.strip_framing } })
    }
    pub(super) fn tool_validation(&self) -> Option<&Arc<dyn OriginalToolValidation>> {
        self.payload().validation.as_ref()
    }
    /// Actual canonical structural spelling and selected stop classification.
    pub fn structural(&self, id: u32) -> Option<(&str, bool)> {
        self.payload()
            .structural
            .iter()
            .find(|row| row.id == id)
            .map(|row| (self.text(row.spelling), row.stop))
    }
    /// Exact maximum structural spelling length, for mutable output geometry.
    pub fn maximum_structural_bytes(&self) -> usize {
        self.payload()
            .structural
            .iter()
            .map(|s| s.spelling.end - s.spelling.start)
            .max()
            .unwrap_or(0)
    }
    /// Actual source identity, independent of equal program bytes.
    pub fn same_source(&self, other: &Self) -> bool {
        Arc::ptr_eq(
            self.0.as_ref().expect("live source"),
            other.0.as_ref().expect("live source"),
        )
    }
    /// Validates the complete original source domain and the exact decoder owner.
    pub fn validate(
        &self,
        pool: &WorkingMemoryPool,
        tokenizer: &OriginalTokenizer,
    ) -> Result<(), WorkingMemoryError> {
        let p = self.payload();
        p.account.validate(pool)?;
        p.tokenizer.validate_pool(pool)?;
        p.controller.validate(pool)?;
        if !p.tokenizer.same_source(tokenizer) {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        Ok(())
    }
    /// Actual forbidden source when this program was paired with that branch.
    pub fn forbidden(&self) -> Option<&OriginalForbiddenSource> {
        match &self.payload().controller { ControllerSource::Forbidden(source) => Some(source), _ => None }
    }
    /// Authenticates the same immutable controller inputs after independent
    /// history copies; no mutable parser or original account is retained here.
    pub(in crate::working_memory) fn validate_controller_source(
        &self, source: eredu_core::PreparedControllerSource<'_>, pool: &WorkingMemoryPool,
    ) -> Result<(), WorkingMemoryError> {
        self.payload().controller.matches_source(source, pool)
    }
    /// Authenticates the same immutable inputs after an independent controller copy.
    pub fn validate_controller<C: eredu_core::SpeculativeTokenFilterController>(
        &self, controller: &C, pool: &WorkingMemoryPool,
    ) -> Result<(), WorkingMemoryError> {
        self.payload().controller.matches(controller, pool)
    }
    /// Fixed query/validation frames, excluding copied payloads already owned.
    pub fn control_bytes() -> Option<usize> {
        let parts = [
            eredu_text::semantic_channels::control_bytes()?,
            OriginalForbiddenSource::validation_control_bytes()?,
            OriginalTokenTrieSource::grammar_validation_control_bytes()?,
            size_of::<ControllerSource>(), size_of::<OriginalSemanticControllerSource<'_>>(),
            size_of::<eredu_core::PreparedControllerSource<'_>>(),
            size_of::<(&ControllerSource, eredu_core::PreparedControllerSource<'_>, &WorkingMemoryPool)>(),
            size_of::<Result<ControllerSource, WorkingMemoryError>>(),
            size_of::<(&ControllerSource, &WorkingMemoryPool)>(),
            size_of::<(&WorkingMemoryPool, &OriginalTokenizer, &OriginalForbiddenSource, ChannelProgram<'_>, &[(u32, &str, bool)])>(),
            size_of::<(&OriginalTokenizer, OriginalSemanticControllerSource<'_>, ChannelProgram<'_>, &[(u32, &str, bool)])>(),
            size_of::<(OriginalSemanticControllerSource<'_>, &WorkingMemoryPool)>(),
            size_of::<(PreparedGrammarSource<'_>, &OriginalTokenTrieSource)>(),
            size_of::<Self>(),
            size_of::<Option<Arc<dyn OriginalToolValidation>>>(),
            size_of::<(&dyn OriginalToolValidation, OriginalSemanticControllerSource<'_>, &WorkingMemoryPool)>(),
            size_of::<Result<(), WorkingMemoryError>>(),
            JsonFieldNames::control_bytes()?,
            size_of::<Option<ToolProgram<'_>>>(), size_of::<Option<Tools>>(),
            size_of::<[&str; 13]>(), size_of::<[Span; 13]>(),
            size_of::<std::array::IntoIter<&str, 13>>(),
            size_of::<std::iter::Enumerate<std::array::IntoIter<&str, 13>>>(),
            size_of::<Result<Option<Tools>, Cause>>(),
            size_of::<(&mut SpeculativeBuffer<u8>, Option<ToolProgram<'_>>)>(),
            size_of::<ChannelProgram<'static>>(),
            size_of::<Option<(&str, bool)>>(),
            size_of::<Result<(), WorkingMemoryError>>(),
            size_of::<(&Self, &WorkingMemoryPool, &OriginalTokenizer)>(),
            size_of::<std::slice::Iter<'_, Structural>>(),
            size_of::<Result<&str, std::str::Utf8Error>>(),
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
}
#[derive(Debug, thiserror::Error)]
enum Cause {
    #[error(transparent)]
    Memory(#[from] WorkingMemoryError),
    #[error(transparent)]
    Buffer(#[from] SpeculativeBufferAllocationError),
    #[error("invalid literal channel program or structural source")]
    Source,
}
/// Failed original compilation retains its completed byte/row prefix and payer.
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
pub struct OriginalSemanticChannelSourceError {
    #[source]
    cause: Cause,
    settlement: Option<WorkingMemoryError>,
    completed: Option<OriginalSemanticChannelSource>,
    bytes: Option<SpeculativeBuffer<u8>>,
    rows: Option<SpeculativeBuffer<Structural>>,
    account: Option<Account>,
}
impl OriginalSemanticChannelSourceError {
    fn rejected(cause: impl Into<Cause>) -> Self {
        Self {
            cause: cause.into(),
            settlement: None,
            completed: None,
            bytes: None,
            rows: None,
            account: None,
        }
    }
}
fn append(bytes: &mut SpeculativeBuffer<u8>, text: &str) -> Result<Span, Cause> {
    let start = bytes.len();
    bytes.try_extend(text.bytes()).map_err(|_| Cause::Source)?;
    Ok(Span {
        start,
        end: bytes.len(),
    })
}
fn append_channel(
    bytes: &mut SpeculativeBuffer<u8>,
    c: Option<DelimitedChannel<'_>>,
) -> Result<Option<Channel>, Cause> {
    c.map(|c| {
        Ok(Channel {
            prefix: append(bytes, c.prefix)?,
            suffix: append(bytes, c.suffix)?,
            prefilled: c.prefix_in_prompt,
        })
    })
    .transpose()
}
impl WorkingMemoryPool {
    /// Pays every destination and immutable owner before copying the facade's
    /// selected program. Structural spellings must be actual atomic added tokens
    /// of the retained decoder. No tokenizer reconstruction or recognition runs.
    pub fn compile_semantic_channel_source(
        &self, tokenizer: &OriginalTokenizer, forbidden: &OriginalForbiddenSource,
        program: ChannelProgram<'_>, structural: &[(u32, &str, bool)],
    ) -> Result<OriginalSemanticChannelSource, OriginalSemanticChannelSourceError> {
        self.compile_semantic_channel_source_for_controller(tokenizer,
            OriginalSemanticControllerSource::Forbidden(forbidden), program, structural)
    }
    /// Same original literal/structural-row compiler with the actual closed
    /// forbidden or grammar source; the facade retains protocol selection.
    pub fn compile_semantic_channel_source_for_controller(
        &self,
        tokenizer: &OriginalTokenizer,
        controller: OriginalSemanticControllerSource<'_>,
        program: ChannelProgram<'_>,
        structural: &[(u32, &str, bool)],
    ) -> Result<OriginalSemanticChannelSource, OriginalSemanticChannelSourceError> {
        self.compile_semantic_channel_source_with_tools(tokenizer, controller, program, structural, None)
    }
    /// Compiles field/envelope literals alongside the exact same decoder,
    /// controller and structural source. Descriptors supply no admission.
    pub fn compile_semantic_channel_source_with_tools(
        &self, tokenizer: &OriginalTokenizer, controller: OriginalSemanticControllerSource<'_>,
        program: ChannelProgram<'_>, structural: &[(u32, &str, bool)], tools: Option<ToolProgram<'_>>,
    ) -> Result<OriginalSemanticChannelSource, OriginalSemanticChannelSourceError> {
        self.compile_semantic_channel_source_with_validation(tokenizer, controller, program, structural, tools, None)
    }
    /// Same source compiler with a retained original full-schema callback.
    /// The callback authenticates its exact controller and source pool before use.
    pub fn compile_semantic_channel_source_with_validation(
        &self, tokenizer: &OriginalTokenizer, controller: OriginalSemanticControllerSource<'_>,
        program: ChannelProgram<'_>, structural: &[(u32, &str, bool)], tools: Option<ToolProgram<'_>>,
        validation: Option<Arc<dyn OriginalToolValidation>>,
    ) -> Result<OriginalSemanticChannelSource, OriginalSemanticChannelSourceError> {
        if let Some(source) = &validation {
            if tools.is_none() { return Err(OriginalSemanticChannelSourceError::rejected(Cause::Source)); }
            source.validate_source(controller, self).map_err(OriginalSemanticChannelSourceError::rejected)?;
        }
        tokenizer
            .validate_pool(self)
            .map_err(OriginalSemanticChannelSourceError::rejected)?;
        controller
            .validate(self)
            .map_err(OriginalSemanticChannelSourceError::rejected)?;
        if !program.is_valid()
            || matches!(tools, Some(ToolProgram::Tagged(t)) if !t.encoding.is_valid())
        {
            return Err(OriginalSemanticChannelSourceError::rejected(Cause::Source));
        }
        let overflow =
            || OriginalSemanticChannelSourceError::rejected(WorkingMemoryError::Overflow);
        let mut total = program.tool_delimiter.len();
        for channel in [program.reasoning_channel, program.text_channel]
            .into_iter()
            .flatten()
        {
            total = total
                .checked_add(channel.prefix.len())
                .and_then(|n| n.checked_add(channel.suffix.len()))
                .ok_or_else(overflow)?;
        }
        if let Some(tools) = tools {
            for literal in tools.literals() { total = total.checked_add(literal.len()).ok_or_else(overflow)?; }
        }
        let count = structural.len();
        for (index, (id, text, _)) in structural.iter().copied().enumerate() {
            if text.is_empty()
                || tokenizer.added_token_id(text) != Some(id)
                || tokenizer.spelling(id) != Some(text)
                || structural[..index].iter().any(|(prior, _, _)| *prior == id)
            {
                return Err(OriginalSemanticChannelSourceError::rejected(Cause::Source));
            }
            total = total.checked_add(text.len()).ok_or_else(overflow)?;
        }
        let arc = Layout::new::<[AtomicUsize; 2]>()
            .extend(Layout::new::<Payload>())
            .map_err(|_| overflow())?
            .0
            .pad_to_align()
            .size();
        let parts = [
            arc,
            Account::control_bytes().ok_or_else(overflow)?,
            OriginalSemanticChannelSource::control_bytes().ok_or_else(overflow)?,
            SpeculativeBuffer::<u8>::retained_control_bytes(total).ok_or_else(overflow)?,
            SpeculativeBuffer::<Structural>::retained_control_bytes(count).ok_or_else(overflow)?,
            size_of::<Payload>(),
            size_of::<Option<Payload>>(),
            size_of::<OriginalSemanticChannelSourceError>(),
            size_of::<Result<OriginalSemanticChannelSource, OriginalSemanticChannelSourceError>>(),
            size_of::<Result<OriginalSemanticChannelSource, Cause>>(),
            size_of::<Result<SpeculativeBuffer<u8>, SpeculativeBufferAllocationError>>(),
            size_of::<Result<SpeculativeBuffer<Structural>, SpeculativeBufferAllocationError>>(),
            size_of::<Result<(), eredu_core::generation::GenerationError>>(),
            size_of::<Result<Option<Channel>, Cause>>(),
            size_of::<Result<Span, Cause>>(),
            size_of::<(
                &WorkingMemoryPool,
                &OriginalTokenizer,
                OriginalSemanticControllerSource<'_>,
                ChannelProgram<'_>,
                &[(u32, &str, bool)],
                Option<ToolProgram<'_>>,
                Option<Arc<dyn OriginalToolValidation>>,
            )>(),
            size_of::<std::slice::Iter<'_, (u32, &str, bool)>>(),
            size_of::<
                std::iter::Enumerate<std::iter::Copied<std::slice::Iter<'_, (u32, &str, bool)>>>,
            >(),
            size_of::<Option<&(u32, &str, bool)>>(),
            size_of::<std::str::Bytes<'_>>(),
            size_of::<(usize, usize, usize)>(),
            size_of::<Option<SpeculativeBuffer<u8>>>(),
            size_of::<Option<SpeculativeBuffer<Structural>>>(),
            size_of::<Arc<Payload>>(),
            size_of::<Option<OriginalSemanticChannelSource>>(),
            size_of::<(&mut SpeculativeBuffer<u8>, &str)>(),
            size_of::<[Option<DelimitedChannel<'_>>; 2]>(),
        ];
        let charge = parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
            .and_then(|n| u64::try_from(n).ok())
            .ok_or_else(overflow)?;
        let account =
            Account::admit(self, charge).map_err(OriginalSemanticChannelSourceError::rejected)?;
        let host = HostPreparationAuthority::retain(account.clone());
        let (mut bytes, mut rows) = (None, None);
        let result = (|| -> Result<OriginalSemanticChannelSource, Cause> {
            bytes = Some(SpeculativeBuffer::try_new_retained(total, host.clone())?);
            rows = Some(SpeculativeBuffer::try_new_retained(count, host.clone())?);
            let destination = bytes.as_mut().expect("allocated bytes");
            let reasoning = append_channel(destination, program.reasoning_channel)?;
            let text = append_channel(destination, program.text_channel)?;
            let tool = append(destination, program.tool_delimiter)?;
            let tools = tools.map(|tools| {
                let mut literals = [Span { start: 0, end: 0 }; 13];
                for (index, value) in tools.literals().into_iter().enumerate() { literals[index] = append(destination, value)?; }
                Ok::<_, Cause>(match tools {
                    ToolProgram::Json(t) => Tools::Json(JsonTools { literals: literals[..10].try_into().expect("fixed JSON literals"),
                        id_length: t.fields.call_id.map(|id| id.length), shape:t.shape, layout:t.layout }),
                    ToolProgram::Tagged(t) => Tools::Tagged(TaggedTools { literals, has_type:t.encoding.parameter_type.is_some(),
                        strip_framing:t.encoding.strip_value_framing }),
                })
            }).transpose()?;
            for &(id, spelling, stop) in structural {
                let spelling = append(destination, spelling)?;
                rows.as_mut()
                    .expect("allocated rows")
                    .try_push(Structural { id, spelling, stop })
                    .map_err(|_| Cause::Source)?;
            }
            if destination.len() != total || rows.as_ref().expect("rows").len() != count {
                return Err(Cause::Source);
            }
            Ok(OriginalSemanticChannelSource(Some(Arc::new(Payload {
                bytes: bytes.take().expect("bytes"),
                structural: rows.take().expect("rows"),
                reasoning,
                text,
                tool,
                tool_is_json: program.tool_is_json,
                tools,
                validation: validation.clone(),
                tokenizer: tokenizer.clone(),
                controller: controller.retain()?,
                account: account.clone(),
            }))))
        })();
        let settlement = account.finish();
        match (result, settlement) {
            (Ok(source), Ok(())) => Ok(source),
            (Ok(source), Err(cause)) => Err(OriginalSemanticChannelSourceError {
                cause: cause.into(),
                settlement: None,
                completed: Some(source),
                bytes,
                rows,
                account: Some(account),
            }),
            (Err(cause), settlement) => Err(OriginalSemanticChannelSourceError {
                cause,
                settlement: settlement.err(),
                completed: None,
                bytes,
                rows,
                account: Some(account),
            }),
        }
    }
}

impl OriginalTokenizer {
    /// Compile the shared channel/field source with its full-schema callback.
    pub fn compile_semantic_channel_source_with_validation(
        &self, controller: OriginalSemanticControllerSource<'_>, program: ChannelProgram<'_>,
        structural: &[(u32, &str, bool)], tools: Option<ToolProgram<'_>>,
        validation: Option<Arc<dyn OriginalToolValidation>>,
    ) -> Result<OriginalSemanticChannelSource, OriginalSemanticChannelSourceError> {
        self.pool().compile_semantic_channel_source_with_validation(self, controller, program, structural, tools, validation)
    }
    /// Compiles exact JSON field/framing literals under this tokenizer's own
    /// original source pool, together with the selected controller and channels.
    pub fn compile_semantic_channel_source_with_tools(
        &self, controller: OriginalSemanticControllerSource<'_>, program: ChannelProgram<'_>,
        structural: &[(u32, &str, bool)], tools: Option<ToolProgram<'_>>,
    ) -> Result<OriginalSemanticChannelSource, OriginalSemanticChannelSourceError> {
        self.pool().compile_semantic_channel_source_with_tools(self, controller, program, structural, tools)
    }
    /// Runs the same literal program compiler with the actual prepared grammar
    /// or forbidden inputs in this tokenizer's original source domain.
    pub fn compile_semantic_channel_source_for_controller(
        &self, controller: OriginalSemanticControllerSource<'_>,
        program: ChannelProgram<'_>, structural: &[(u32, &str, bool)],
    ) -> Result<OriginalSemanticChannelSource, OriginalSemanticChannelSourceError> {
        self.pool().compile_semantic_channel_source_for_controller(self, controller, program, structural)
    }
    /// Runs the same original compiler in this tokenizer's actual source domain.
    /// Borrowed protocol rows are caller declarations, never a funding proof.
    pub fn compile_semantic_channel_source(
        &self,
        forbidden: &OriginalForbiddenSource,
        program: ChannelProgram<'_>,
        structural: &[(u32, &str, bool)],
    ) -> Result<OriginalSemanticChannelSource, OriginalSemanticChannelSourceError> {
        self.pool()
            .compile_semantic_channel_source(self, forbidden, program, structural)
    }
}
