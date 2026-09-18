//! Original source contracts shared by the neutral public conformance backends.
//! These fixtures use the runtime's actual account and source constructors.
use eredu_core::{ModelRuntime, TextGenerationBackend};
use eredu_runtime::working_memory::{InferenceExecutionIdentity, WorkingMemoryPool};

pub(super) struct Environment {
    pub pool: WorkingMemoryPool,
    pub execution: InferenceExecutionIdentity,
    pub output_width: std::cell::Cell<usize>,
}
impl Environment {
    pub fn new(pool: Option<WorkingMemoryPool>) -> Self {
        Self {
            pool: pool
                .unwrap_or_else(|| WorkingMemoryPool::new(32 * 1024 * 1024 * 1024, 0).unwrap()),
            execution: InferenceExecutionIdentity::default(),
            output_width: std::cell::Cell::new(0),
        }
    }
}
pub(super) trait SourceBackend: TextGenerationBackend {
    fn source_environment(runtime: &ModelRuntime<Self>) -> &Environment;
    fn capture_facts(_: &ModelRuntime<Self>) -> Option<(&eredu_core::ObservationCatalog, &eredu_core::ObservationSupportReport)> { None }
    fn intervention_facts(_: &ModelRuntime<Self>) -> Option<&eredu_core::intervention::InterventionDiscovery> { None }
    fn before_semantic_prompt(_: &ModelRuntime<Self>) -> Result<(), eredu_core::BackendFailure> {
        Ok(())
    }
}

#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
struct SourceFailure<E: std::error::Error + Send + Sync + 'static> {
    #[source]
    cause: E,
    _funding: eredu_core::HostMetadataFunding,
}
fn retained_error<E: std::error::Error + Send + Sync + 'static>(cause: E, funding: &eredu_core::HostMetadataFunding) -> eredu_core::BackendFailure {
    use eredu_core::{BackendFailure, HostMetadataFundingError};
    let Some(bytes) = BackendFailure::source_retention_peak_bytes::<SourceFailure<E>>() else { return HostMetadataFundingError::Overflow.into(); };
    if let Err(cause) = funding.reserve_metadata(bytes) { return cause.into(); }
    BackendFailure::from_error(SourceFailure { cause, _funding: funding.clone() })
}
pub(super) fn compile_intervention<B: SourceBackend>(
    runtime: &ModelRuntime<B>, raw: &eredu_core::intervention::InterventionPlan,
    capture: &eredu_core::capture::SharedCapturePlan, session: &str,
    funding: &eredu_core::HostMetadataFunding,
) -> Result<eredu_runtime::working_memory::OriginalInterventionSource, eredu_core::BackendFailure> {
    use eredu_core::{HostPreparationAuthority, HostMetadataFunding, HostMetadataFundingError,
        intervention::{PreparedInterventionAdmission, PreparedInterventionPlanCopy}};
    let facts = B::intervention_facts(runtime).ok_or_else(|| eredu_core::TokenInputRejection::Unsupported.into_backend_failure())?;
    let capture = capture.admission();
    let bytes = PreparedInterventionAdmission::inspection_control_bytes()
        .and_then(|n| n.checked_add(PreparedInterventionPlanCopy::inspection_control_bytes()?))
        .and_then(|n| n.checked_add(HostPreparationAuthority::retention_bytes::<HostMetadataFunding>()?))
        .ok_or(HostMetadataFundingError::Overflow)?;
    funding.reserve_metadata(bytes)?;
    let plan = PreparedInterventionAdmission::inspect(raw, facts, capture.request(), capture.invocation_bounds(),
        capture.text_origin().unwrap_or_default(), session).map_err(|e| retained_error(e, funding))?;
    funding.reserve_metadata(plan.required_bytes())?;
    let host = HostPreparationAuthority::retain(funding.clone());
    let admission = plan.construct(&host).map_err(|e| retained_error(e, funding))?;
    let copy = PreparedInterventionPlanCopy::inspect(&admission).map_err(|e| retained_error(e, funding))?;
    B::source_environment(runtime).pool.compile_intervention_source(copy).map_err(|e| retained_error(e, funding))
}

macro_rules! implement {
    ($backend:ty) => {
        mod original_source_backend {
            use super::original_sources::SourceBackend;
            use super::*;
            use eredu_core::*;
            use eredu_runtime::working_memory::*;
            use eredu_text::chat_storage::{ChatRenderContext, ChatTemplatePlan};
            impl OriginalTokenizerBackend for $backend {
                fn validate_semantic_source(
                    runtime: &ModelRuntime<Self>,
                    source: &PreparedSemanticSource,
                ) -> Result<(), TokenInputRejection> {
                    let env = Self::source_environment(runtime);
                    source
                        .validate(&env.pool, &env.execution)
                        .map_err(|_| TokenInputRejection::IdentityMismatch)
                }
                fn prepare_semantic_source(
                    runtime: &ModelRuntime<Self>,
                    source: &OriginalTokenizer,
                    capacity: u64,
                ) -> Result<PreparedSemanticSource, SpeculativeOutputError> {
                    let env = Self::source_environment(runtime);
                    source
                        .validate_pool(&env.pool)
                        .map_err(|_| SpeculativeOutputError::Storage("foreign source"))?;
                    PreparedSemanticSource::new(source, &env.execution, capacity)
                }
                fn prepare_semantic_prompt(
                    runtime: &ModelRuntime<Self>,
                    source: &PreparedSemanticSource,
                    input: &TokenIdsInputPlan<'_>,
                    _: Option<std::num::NonZeroU64>,
                ) -> Result<Self::Prompt, BackendFailure> {
                    let env = Self::source_environment(runtime);
                    source
                        .validate(&env.pool, &env.execution)
                        .map_err(BackendFailure::from_error)?;
                    let domain = source.tokenizer().generation_domain()
                        .ok_or_else(|| TokenInputRejection::IdentityMismatch.into_backend_failure())?;
                    if input.tokens().iter().any(|&id| !domain.allows(id)) {
                        return Err(TokenInputRejection::InvalidToken.into_backend_failure());
                    }
                    Self::before_semantic_prompt(runtime)?;
                    super::original_sources::Prompt::copy(input.tokens(), source.metadata_funding())
                }
                fn prepare_original_text_source_budget(
                    runtime: &ModelRuntime<Self>,
                    source: &OriginalTokenizer,
                    capacity: u64,
                ) -> Result<OriginalTextSourceBudget, OriginalTextSourceError> {
                    let env = Self::source_environment(runtime);
                    source
                        .validate_pool(&env.pool)
                        .map_err(|_| TokenInputRejection::IdentityMismatch)?;
                    source
                        .prepare_text_source_budget(&env.execution, capacity)
                        .map_err(Into::into)
                }
                fn validate_original_tokenizer_source(
                    runtime: &ModelRuntime<Self>,
                    source: &OriginalTokenizer,
                ) -> Result<(), BackendFailure> {
                    source
                        .validate_pool(&Self::source_environment(runtime).pool)
                        .map_err(BackendFailure::from_error)
                }
                fn compile_original_tokenizer(
                    runtime: &ModelRuntime<Self>,
                    plan: eredu_text::tokenizer_storage::TokenizerPlan<'_>,
                ) -> Result<OriginalTokenizer, BackendFailure> {
                    let source = Self::source_environment(runtime)
                        .pool
                        .compile_tokenizer(plan)
                        .map_err(BackendFailure::from_error)?;
                    Self::source_environment(runtime)
                        .output_width
                        .set(source.ids().max().map_or(0, |id| id as usize + 1));
                    Ok(source)
                }
                fn compile_original_text_stop_source(
                    runtime: &ModelRuntime<Self>,
                    plan: eredu_text::stop_storage::StopCompilePlan<'_>,
                ) -> Result<OriginalStopSource, OriginalTextSourceError> {
                    Self::source_environment(runtime)
                        .pool
                        .compile_stop_source(plan)
                        .map_err(Into::into)
                }
                fn encode_original_text_ids(
                    runtime: &ModelRuntime<Self>,
                    source: &OriginalTokenizer,
                    input: &str,
                    add_special: bool,
                ) -> Result<OriginalEncodedTokenIds, OriginalTextSourceError> {
                    Self::source_environment(runtime)
                        .pool
                        .encode_tokenizer_ids(source, input, add_special)
                        .map_err(Into::into)
                }
                fn encode_original_tokenizer_ids(
                    runtime: &ModelRuntime<Self>,
                    source: &OriginalTokenizer,
                    input: &str,
                    add_special: bool,
                ) -> Result<OriginalEncodedTokenIds, BackendFailure> {
                    Self::source_environment(runtime)
                        .pool
                        .encode_tokenizer_ids(source, input, add_special)
                        .map_err(OriginalTokenizerEncodeError::into_backend_failure)
                }
                fn compile_original_tokenizer_source_for_generation(
                    runtime: &ModelRuntime<Self>,
                    input: eredu_runtime::working_memory::OriginalTokenizerInput<'_>,
                ) -> Result<OriginalTokenizer, OriginalTokenizerSourceError> {
                    let source = Self::source_environment(runtime)
                        .pool
                        .compile_tokenizer_source_for_generation(input)
                        .map_err(OriginalTokenizerSourceError::from)?;
                    Self::source_environment(runtime)
                        .output_width
                        .set(source.ids().max().map_or(0, |id| id as usize + 1));
                    Ok(source)
                }
            }
            impl OriginalChatBackend for $backend {
                fn compile_original_capture_declaration(
                    runtime: &ModelRuntime<Self>, plan: &eredu_core::capture::CapturePlan,
                    request: eredu_core::capture::CaptureRequestShape, funding: &HostMetadataFunding,
                ) -> Result<OriginalCaptureSource, OriginalCaptureSourceError> {
                    let (catalog, support) = Self::capture_facts(runtime)
                        .ok_or_else(|| OriginalCaptureSourceError::rejected(WorkingMemoryError::UnknownBound))?;
                    Self::source_environment(runtime).pool.compile_capture_declaration(
                        plan, catalog, support, request, Default::default(), funding)
                }
                fn compile_original_intervention_declaration(
                    runtime: &ModelRuntime<Self>, raw: &eredu_core::intervention::InterventionPlan,
                    capture: &eredu_core::capture::SharedCapturePlan, session_id: &str,
                    funding: &HostMetadataFunding,
                ) -> Result<OriginalInterventionSource, BackendFailure> {
                    super::original_sources::compile_intervention::<Self>(runtime, raw, capture, session_id, funding)
                }
                fn compile_original_forbidden_source(
                    runtime: &ModelRuntime<Self>,
                    plan: eredu_core::speculative::PreparedForbiddenInputCopy<'_>,
                ) -> Result<OriginalForbiddenSource, OriginalForbiddenSourceError> {
                    Self::source_environment(runtime)
                        .pool
                        .compile_forbidden_source(plan)
                }
                fn compile_original_forbidden_tokenizer_source(
                    runtime: &ModelRuntime<Self>,
                    source: &OriginalTokenizer,
                    trigger: &[u8],
                ) -> Result<OriginalForbiddenSource, OriginalForbiddenSourceError> {
                    Self::source_environment(runtime)
                        .pool
                        .compile_forbidden_tokenizer_source(source, trigger)
                }
                fn prepare_original_chat_profile(
                    runtime: &ModelRuntime<Self>,
                    template: &OriginalChatTemplate,
                    tokenizer: &OriginalTokenizer,
                    capacity: u64,
                ) -> Result<OriginalChatProfilePreparation, OriginalChatProfileError> {
                    OriginalChatProfilePreparation::new(
                        template,
                        tokenizer,
                        &Self::source_environment(runtime).execution,
                        capacity,
                    )
                }
                fn validate_original_chat_sources(
                    runtime: &ModelRuntime<Self>,
                    template: &OriginalChatTemplate,
                    tokenizer: &OriginalTokenizer,
                ) -> Result<(), TokenInputRejection> {
                    let env = Self::source_environment(runtime);
                    template
                        .validate_pool(&env.pool)
                        .and_then(|_| tokenizer.validate_pool(&env.pool))
                        .map_err(|_| TokenInputRejection::IdentityMismatch)
                }
                fn validate_original_chat_render(
                    runtime: &ModelRuntime<Self>,
                    render: &OriginalRenderedChat,
                ) -> Result<(), TokenInputRejection> {
                    render
                        .validate_pool(&Self::source_environment(runtime).pool)
                        .map_err(|_| TokenInputRejection::IdentityMismatch)
                }
                fn compile_original_chat_template(
                    runtime: &ModelRuntime<Self>,
                    plan: ChatTemplatePlan<'_>,
                ) -> Result<OriginalChatTemplate, OriginalChatSourceError> {
                    Self::source_environment(runtime)
                        .pool
                        .compile_chat_template(plan)
                        .map_err(Into::into)
                }
                fn compile_original_chat_template_file(
                    runtime: &ModelRuntime<Self>,
                    read: eredu_checkpoint::artifact::PreparedArtifactFileRead,
                    model_id: &str,
                    has_tools: bool,
                ) -> Result<OriginalChatTemplate, OriginalChatSourceError> {
                    Self::source_environment(runtime)
                        .pool
                        .compile_chat_template_file(read, model_id, has_tools)
                        .map_err(Into::into)
                }
                fn render_original_chat(
                    runtime: &ModelRuntime<Self>,
                    template: &OriginalChatTemplate,
                    tokenizer: &OriginalTokenizer,
                    context: ChatRenderContext<'_>,
                ) -> Result<OriginalRenderedChat, OriginalChatRenderOperationError> {
                    Self::source_environment(runtime)
                        .pool
                        .render_original_chat(template, tokenizer, context)
                        .map_err(Into::into)
                }
            }
        }
    };
}
pub(super) use implement;

use eredu_core::{BackendFailure, HostMetadataFunding};
use eredu_runtime::working_memory::{OriginalTextMetadataCustody, OwnedPromptTokenIds};
use std::{
    alloc::Layout,
    sync::{Arc, atomic::AtomicUsize},
};

#[derive(Debug)]
enum Tokens {
    Supplied(Vec<u32>),
    Original(OwnedPromptTokenIds),
}
#[derive(Debug)]
enum Custody {
    Supplied,
    Text(OriginalTextMetadataCustody),
    Semantic(HostMetadataFunding),
}
#[derive(Debug)]
struct PromptInner {
    tokens: Tokens,
    _custody: Custody,
}
/// The same immutable prompt for ordinary input and originally admitted IDs.
/// Shared aliases allocate no headers; the last alias retires the shell before its payer.
#[derive(Debug)]
pub(super) struct Prompt {
    owner: Option<Arc<PromptInner>>,
    start: usize,
    end: usize,
    resume_token: Option<u32>,
}
impl Clone for Prompt {
    fn clone(&self) -> Self {
        Self {
            owner: self.owner.clone(),
            start: self.start,
            end: self.end,
            resume_token: self.resume_token,
        }
    }
}
impl Drop for Prompt {
    fn drop(&mut self) {
        if let Some(owner) = self.owner.take() {
            drop(Arc::into_inner(owner));
        }
    }
}
impl Prompt {
    pub fn construction_bytes() -> usize {
        Layout::new::<[AtomicUsize; 2]>()
            .extend(Layout::new::<PromptInner>())
            .unwrap()
            .0
            .pad_to_align()
            .size()
            + std::mem::size_of::<Self>()
            + std::mem::size_of::<Result<Self, BackendFailure>>()
    }
    fn publish(tokens: Tokens, custody: Custody) -> Self {
        let end = match &tokens {
            Tokens::Supplied(ids) => ids.len(),
            Tokens::Original(ids) => ids.tokens().len(),
        };
        Self {
            owner: Some(Arc::new(PromptInner {
                tokens,
                _custody: custody,
            })),
            start: 0,
            end,
            resume_token: None,
        }
    }
    /// Exact saved decode input, consumed by the existing decoder equation.
    pub fn with_resume_token(mut self, token: u32) -> Self {
        self.resume_token = Some(token);
        self
    }
    pub fn resume_token(&self) -> Option<u32> { self.resume_token }
    /// Called only after the fixture's text-control quote admits this exact shell.
    pub fn original(input: OwnedPromptTokenIds, custody: OriginalTextMetadataCustody) -> Self {
        Self::publish(Tokens::Original(input), Custody::Text(custody))
    }
    pub fn copy(ids: &[u32], funding: &HostMetadataFunding) -> Result<Self, BackendFailure> {
        let payload = Layout::array::<u32>(ids.len())
            .map_err(BackendFailure::from_error)?
            .size();
        funding
            .reserve_metadata(Self::construction_bytes().checked_add(payload).unwrap())
            .map_err(BackendFailure::from_error)?;
        let mut copied = Vec::new();
        copied
            .try_reserve_exact(ids.len())
            .map_err(BackendFailure::from_error)?;
        copied.extend_from_slice(ids);
        Ok(Self::publish(
            Tokens::Supplied(copied),
            Custody::Semantic(funding.clone()),
        ))
    }
    pub fn prefix(mut self, length: usize) -> Self {
        self.end = self.start + length.min(self.len());
        self
    }
}
impl From<Vec<u32>> for Prompt {
    fn from(ids: Vec<u32>) -> Self {
        Self::publish(Tokens::Supplied(ids), Custody::Supplied)
    }
}
impl std::ops::Deref for Prompt {
    type Target = [u32];
    fn deref(&self) -> &[u32] {
        let inner = self.owner.as_ref().unwrap();
        let ids = match &inner.tokens {
            Tokens::Supplied(ids) => ids.as_slice(),
            Tokens::Original(ids) => ids.tokens(),
        };
        &ids[self.start..self.end]
    }
}
impl PartialEq for Prompt {
    fn eq(&self, other: &Self) -> bool {
        self.as_ref() == other.as_ref()
    }
}
impl Eq for Prompt {}
impl PartialEq<Vec<u32>> for Prompt {
    fn eq(&self, other: &Vec<u32>) -> bool { self.as_ref() == other.as_slice() }
}
impl AsRef<[u32]> for Prompt {
    fn as_ref(&self) -> &[u32] {
        self
    }
}
pub(super) struct PromptIter(Prompt);
impl Iterator for PromptIter {
    type Item = u32;
    fn next(&mut self) -> Option<u32> {
        let value = self.0.first().copied()?;
        self.0.start += 1;
        Some(value)
    }
}
impl IntoIterator for Prompt {
    type Item = u32;
    type IntoIter = PromptIter;
    fn into_iter(self) -> PromptIter {
        PromptIter(self)
    }
}
impl FromIterator<u32> for Prompt {
    fn from_iter<I: IntoIterator<Item = u32>>(iter: I) -> Self {
        Vec::from_iter(iter).into()
    }
}

/// Test-side original artifacts remain available after loading metadata. This
/// object never recreates a tokenizer from vocabulary observations.
pub(super) struct Fixture<B: TextGenerationBackend> {
    model: eredu::api::LoadedModel<B>,
    tokenizer_json: Vec<u8>,
    template_kwargs: serde_json::Map<String, serde_json::Value>,
    template: Option<eredu_text::tokenizer::ModelChatTemplate>,
    tokenizer_source: Option<eredu::api::ManagedPlainTextSource>,
    original_pool: Option<eredu_runtime::working_memory::WorkingMemoryPool>,
}
impl<B: SourceBackend> Fixture<B> {
    /// Retains caller fixture artifacts alongside an independently loaded model.
    pub fn from_loaded(
        model: eredu::api::LoadedModel<B>,
        tokenizer_json: Vec<u8>,
        template: Option<eredu_text::tokenizer::ModelChatTemplate>,
        template_kwargs: serde_json::Map<String, serde_json::Value>,
    ) -> Self {
        Self { model, tokenizer_json, template, template_kwargs, tokenizer_source: None, original_pool: None }
    }
    pub fn from_runtime(
        runtime: ModelRuntime<B>,
        tokenizer: eredu_text::tokenizer::Tokenizer,
        config: eredu::api::LoadedTextModelConfig,
    ) -> Result<Self, BackendFailure> {
        // This serializes the actual fixture input before loading it. File I/O
        // and caller-owned fixture artifacts are outside inference admission.
        let template_kwargs = tokenizer.template_kwargs().clone();
        let tokenizer_json = serde_json::to_vec(&*tokenizer.snapshot()).unwrap();
        let width = eredu_text::tokenizer::token_id_vocabulary(&tokenizer)
            .last_key_value()
            .map_or(0, |(&id, _)| id as usize + 1);
        B::source_environment(&runtime).output_width.set(width);
        let original_pool = B::source_environment(&runtime).pool.clone();
        let template = config.chat_template.clone();
        let model = eredu::api::LoadedModel::from_runtime(runtime, tokenizer, config)?;
        Ok(Self {
            model,
            tokenizer_json,
            template,
            template_kwargs,
            tokenizer_source: None,
            original_pool: Some(original_pool),
        })
    }
    pub fn original_pool(&self) -> &eredu_runtime::working_memory::WorkingMemoryPool {
        self.original_pool.as_ref().expect("fixture original runtime pool")
    }
    pub fn tokenizer_source(&self) -> &eredu::api::ManagedPlainTextSource {
        self.tokenizer_source
            .as_ref()
            .expect("original chat source compiled")
    }
    pub fn into_model(self) -> eredu::api::LoadedModel<B> {
        self.model
    }
    pub fn replace_template(&mut self, template: Option<eredu_text::tokenizer::ModelChatTemplate>) {
        self.template = template.clone();
        self.model.set_chat_template(template);
    }
}
impl<B: eredu_runtime::working_memory::OriginalChatBackend> Fixture<B> {
    pub fn chat_source(
        &mut self,
        has_tools: bool,
        cancel: &eredu_core::GenerationCancellationToken,
    ) -> Result<Option<eredu::api::ManagedChatSource>, FixtureSourceError> {
        if cancel.is_cancelled() {
            return Ok(None);
        }
        if self.tokenizer_source.is_none() {
            self.tokenizer_source = Some(
                self.model
                    .compile_managed_plain_text_source(artifact_file(&self.tokenizer_json))?,
            );
        }
        let template = match &self.template {
            None => serde_json::Value::Null,
            Some(eredu_text::tokenizer::ModelChatTemplate::Single(s)) => {
                serde_json::Value::String(s.clone())
            }
            Some(eredu_text::tokenizer::ModelChatTemplate::Named(map)) => {
                serde_json::Value::Array(map.iter().map(|(name, template)| {
                    serde_json::json!({"name": name, "template": template})
                }).collect())
            }
        };
        let mut config = self.template_kwargs.clone();
        config.insert("chat_template".into(), template);
        let config = serde_json::to_vec(&config).unwrap();
        self.model
            .compile_managed_chat_source(
                self.tokenizer_source.as_ref().unwrap(),
                artifact_file(&config),
                has_tools,
                cancel,
            )
            .map_err(Into::into)
    }
}
#[derive(Debug, thiserror::Error)]
pub(super) enum FixtureSourceError {
    #[error(transparent)]
    Tokenizer(#[from] eredu::api::ManagedPlainTextSourceError),
    #[error(transparent)]
    Chat(#[from] eredu::api::ManagedChatSourceError),
}
impl<B: TextGenerationBackend> std::ops::Deref for Fixture<B> {
    type Target = eredu::api::LoadedModel<B>;
    fn deref(&self) -> &Self::Target {
        &self.model
    }
}
impl<B: TextGenerationBackend> std::ops::DerefMut for Fixture<B> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.model
    }
}
fn artifact_file(bytes: &[u8]) -> std::fs::File {
    use std::io::{Seek, Write};
    let mut file = tempfile::tempfile().unwrap();
    file.write_all(bytes).unwrap();
    file.rewind().unwrap();
    file
}

/// Caller policy ceiling; every actual source/session producer still admits its own exact demand.
pub(super) const CAPACITY: u64 = 4 * 1024 * 1024 * 1024;
pub(super) fn settings(
    mut settings: eredu::api::PreparedChatGenerationSettings,
) -> eredu::api::PreparedChatGenerationSettings {
    settings.inference.managed_memory_capacity_bytes.get_or_insert(CAPACITY);
    settings
}
