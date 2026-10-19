//! Shared-embedding channel fusion and independently materializable function readout.
//!
//! The backbone executes once and supplies one normalized hidden tensor to both
//! vocabulary heads. Tokenization, sampling of agent text and feedback commitment
//! remain outside these neural operations.

use super::config::ChannelPolicy;
use crate::decoder::StaticModules;
use eredu_nn::{
    DistributedNeuralBackend, EmbeddingLookupPolicy, EmbeddingOperator, LinearFormatSpec,
    LinearOperator, LinearSpec, NeuralBackend, ParameterSpec, Parameterized, Tensor,
    TensorElementType, VocabularyParallelRange,
};
use eredu_runtime::{
    module_parameter_group, MemberSharding, ParallelPlanError, ParameterGroupSpec, ParameterRole,
};

/// Released `<s>` ID used to frame system-prompt tokens.
pub const TEXT_BOS_ID: i32 = 1;
/// Exclusive upper bound of the released agent/function token domain.
pub const TEXT_VOCABULARY_SIZE: usize = 131_072;
/// Released `</s>` ID; distinct from the base tokenizer's unmodified EOS setting.
pub const TEXT_EOS_ID: i32 = 2;
/// Released `<SPECIAL_12>` ID, also used as initial agent/function feedback.
pub const TEXT_PAD_ID: i32 = 12;

/// Invalid channel geometry, precision or bounded prompt admission.
#[derive(Debug, thiserror::Error)]
pub enum ChannelError {
    /// Shared component declaration or invocation is invalid.
    #[error(transparent)]
    Component(#[from] eredu_runtime::ComponentDefinitionError),
    /// The configuration changes the released channel equations.
    #[error("{0}")]
    Configuration(&'static str),
    /// Channel payloads have incompatible logical geometry.
    #[error("{0}")]
    Shape(&'static str),
    /// The backend cannot establish the declared floating representation.
    #[error("{0}")]
    Precision(&'static str),
    /// The framed, repeated prompt exceeds its explicit bound.
    #[error("VoiceChat prompt requires {required} tokens, capacity is {capacity}")]
    PromptCapacity {
        /// Total framed tokens, including repetition.
        required: usize,
        /// Caller-declared maximum tokens.
        capacity: usize,
    },
    /// Portable execution graph is invalid.
    #[error(transparent)]
    Graph(#[from] eredu_runtime::ExecutionGraphError),
    /// Graph units cannot map to one stable residency order.
    #[error(transparent)]
    UnitLayout(#[from] eredu_runtime::ExecutionUnitLayoutError),
    /// Runtime state geometry or indexing is invalid.
    #[error(transparent)]
    State(#[from] eredu_runtime::StateError),
    /// Ordered layer policy is invalid.
    #[error(transparent)]
    Schedule(#[from] eredu_core::LayerScheduleError),
    /// Cold parameter dimensions or placement are invalid.
    #[error(transparent)]
    Parameters(#[from] eredu_runtime::ParallelPlanError),
    /// Exact parameter ownership is invalid.
    #[error(transparent)]
    Ownership(#[from] eredu_runtime::ArchitectureParameterError),
    /// Portable state identity is invalid.
    #[error(transparent)]
    Identity(#[from] eredu_core::cache::PromptCacheError),
    /// Parameter provider or submission failed with its original source retained.
    #[error(transparent)]
    Backend(#[from] eredu_core::BackendFailure),
    /// A neural mechanism failed, retaining its original source.
    #[error(transparent)]
    Neural(#[from] eredu_nn::Error),
}

/// Explicit repetition and allocation limit for an already tokenized system prompt.
#[derive(Debug, Clone, Copy)]
pub struct PromptPolicy {
    /// Repeat the complete BOS/text/EOS sequence this many times; upstream defaults to one.
    pub repeat: usize,
    /// Maximum tokens after framing and repetition.
    pub max_tokens: usize,
}

impl PromptPolicy {
    /// Validates framing and capacity before allocating prompt tensors.
    pub fn framed_length(self, prompt: &[i32]) -> Result<i32, ChannelError> {
        if self.repeat == 0 || prompt.iter().any(|&id| id < 0) {
            return Err(ChannelError::Configuration(
                "prompt repeat must be positive and token IDs nonnegative",
            ));
        }
        let count = prompt
            .len()
            .checked_add(2)
            .and_then(|n| n.checked_mul(self.repeat))
            .ok_or(ChannelError::Configuration("prompt token count overflow"))?;
        if count > self.max_tokens {
            return Err(ChannelError::PromptCapacity {
                required: count,
                capacity: self.max_tokens,
            });
        }
        i32::try_from(count)
            .map_err(|_| ChannelError::Configuration("prompt exceeds tensor indexing"))
    }
}

/// Framed token attribution and the fused embeddings supplied to the target backbone.
#[derive(Debug, Clone)]
pub struct PromptEmbeddings<T> {
    /// `[1, sequence]` BOS/prompt/EOS token IDs, including any repetition.
    pub tokens: T,
    /// `[1, sequence, width]` prompt embeddings plus both PAD feedback channels.
    pub embedded: T,
}

/// Released additive fusion, without learned per-channel parameters or normalization.
#[derive(Debug, Clone, Copy)]
pub struct ChannelFusion {
    agent_weight: f32,
    audio_weight: f32,
    function_weight: f32,
}
impl ChannelFusion {
    /// Validates the released agent/audio/function channel policy.
    pub fn from_config(policy: &ChannelPolicy) -> Result<Self, ChannelError> {
        if policy.predict_user_text
            || policy.use_separate_asr_head
            || !policy.use_function_head
            || policy.duplex_text_channel_weight != 1.0
            || policy.duplex_user_channel_weight != 1.0
            || policy.duplex_function_channel_weight != 2.0
            || policy.text_embedding_weight != 1.0
        {
            return Err(ChannelError::Configuration(
                "unsupported VoiceChat embedding fusion policy",
            ));
        }
        Ok(Self {
            agent_weight: 1.0,
            audio_weight: 1.0,
            function_weight: 2.0,
        })
    }

    /// Fuses already embedded channels in NVIDIA's operation order.
    ///
    /// Audio is cast to the embedding dtype before weighting and addition. Each
    /// operation retains that dtype, including BF16 rounding between additions.
    pub fn fuse_embeddings<T: Tensor>(
        &self,
        agent: &T,
        audio: &T,
        function: &T,
        context: &T::Context,
    ) -> Result<T, ChannelError> {
        if agent.shape().len() != 3
            || agent.shape().iter().any(|&n| n <= 0)
            || agent.shape() != audio.shape()
            || agent.shape() != function.shape()
        {
            return Err(ChannelError::Shape(
                "fusion requires matching nonempty [batch, sequence, width] channels",
            ));
        }
        let element = compute_type(agent)?;
        if compute_type(function)? != element {
            return Err(ChannelError::Precision(
                "agent and function embeddings must share compute precision",
            ));
        }
        compute_type(audio)?;
        let audio = if audio.element_type() == Some(element) {
            audio.clone()
        } else {
            audio.cast_float(element, context)?
        };
        let weighted_agent = agent.multiply_scalar(self.agent_weight, context)?;
        let weighted_audio = audio.multiply_scalar(self.audio_weight, context)?;
        let weighted_function = function.multiply_scalar(self.function_weight, context)?;
        Ok(weighted_agent
            .add(&weighted_audio, context)?
            .add(&weighted_function, context)?)
    }

    /// Looks up both feedback channels through the backbone's single embedding table.
    /// Projected perception enters in F32 and is cast at the fusion boundary.
    pub fn fuse<B: NeuralBackend>(
        &self,
        modules: &mut StaticModules<B>,
        projected_audio: &B::Tensor,
        agent_tokens: &B::Tensor,
        function_tokens: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, ChannelError> {
        self.fuse_with_lookup(
            projected_audio,
            agent_tokens,
            function_tokens,
            context,
            |tokens| {
                Ok(modules
                    .embeddings
                    .lookup(tokens, EmbeddingLookupPolicy::Strict, context)?)
            },
            &mut FusionAgreement(None),
        )
    }

    /// Shares the existing vocabulary lookup/reduction for tensor-parallel feedback.
    pub fn fuse_parallel<B: DistributedNeuralBackend>(
        &self,
        modules: &mut StaticModules<B>,
        projected_audio: &B::Tensor,
        agent_tokens: &B::Tensor,
        function_tokens: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, ChannelError> {
        self.fuse_with_lookup(
            projected_audio,
            agent_tokens,
            function_tokens,
            context,
            |tokens| {
                Ok(B::vocabulary_parallel_lookup(
                    &mut modules.embeddings,
                    tokens,
                    EmbeddingLookupPolicy::Strict,
                    parallel,
                    context,
                )?)
            },
            &mut FusionAgreement(None),
        )
    }

    /// Agrees feedback ingress and each local boundary between collective lookups.
    /// The backend guards its own local lookup preparation before collective work.
    pub fn fuse_parallel_guarded<B, F>(
        &self,
        modules: &mut StaticModules<B>,
        projected_audio: &B::Tensor,
        agent_tokens: &B::Tensor,
        function_tokens: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        mut agree: F,
    ) -> Result<B::Tensor, ChannelError>
    where
        B: DistributedNeuralBackend,
        F: FnMut([u64; 8], bool, &<B::Tensor as Tensor>::Context) -> Result<bool, ChannelError>,
    {
        let mut guard = FusionAgreement(Some(&mut agree));
        let mut words = self.invocation_words(0, B::parallel_size(parallel));
        for value in [projected_audio, agent_tokens, function_tokens] {
            fusion_tensor_words(&mut words, value);
        }
        guard.gate(
            0,
            &words,
            validate_feedback(projected_audio, agent_tokens, function_tokens),
            context,
        )?;
        self.fuse_with_lookup(
            projected_audio,
            agent_tokens,
            function_tokens,
            context,
            |tokens| {
                Ok(B::vocabulary_parallel_lookup(
                    &mut modules.embeddings,
                    tokens,
                    EmbeddingLookupPolicy::Strict,
                    parallel,
                    context,
                )?)
            },
            &mut guard,
        )
    }

    fn fuse_with_lookup<T, L>(
        &self,
        projected_audio: &T,
        agent_tokens: &T,
        function_tokens: &T,
        context: &T::Context,
        mut lookup: L,
        guard: &mut FusionAgreement<'_, T::Context>,
    ) -> Result<T, ChannelError>
    where
        T: Tensor,
        L: FnMut(&T) -> Result<T, ChannelError>,
    {
        guard.gate(
            10,
            &[],
            validate_feedback(projected_audio, agent_tokens, function_tokens),
            context,
        )?;
        let agent = guard.tensor(11, lookup(agent_tokens), context)?;
        let function = guard.tensor(12, lookup(function_tokens), context)?;
        guard.tensor(
            13,
            self.fuse_embeddings(&agent, projected_audio, &function, context),
            context,
        )
    }

    /// Frames an already tokenized system prompt with the released IDs.
    ///
    /// Callers decide whether an empty/whitespace system prompt requests prefill;
    /// this explicit operation always includes BOS and EOS. Tokenizer and ASCII
    /// validation belong to request preparation.
    pub fn prefill<B: NeuralBackend>(
        &self,
        modules: &mut StaticModules<B>,
        prompt_ids: &[i32],
        policy: PromptPolicy,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<PromptEmbeddings<B::Tensor>, ChannelError> {
        self.prefill_with_lookup(
            prompt_ids,
            policy,
            context,
            |tokens| {
                Ok(modules
                    .embeddings
                    .lookup(tokens, EmbeddingLookupPolicy::Strict, context)?)
            },
            &mut FusionAgreement(None),
        )
    }

    /// Builds the same prompt through shared tensor-parallel embedding lookup.
    pub fn prefill_parallel<B: DistributedNeuralBackend>(
        &self,
        modules: &mut StaticModules<B>,
        prompt_ids: &[i32],
        policy: PromptPolicy,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<PromptEmbeddings<B::Tensor>, ChannelError> {
        self.prefill_with_lookup(
            prompt_ids,
            policy,
            context,
            |tokens| {
                Ok(B::vocabulary_parallel_lookup(
                    &mut modules.embeddings,
                    tokens,
                    EmbeddingLookupPolicy::Strict,
                    parallel,
                    context,
                )?)
            },
            &mut FusionAgreement(None),
        )
    }

    /// Agrees exact prompt framing before allocation and both collective lookups.
    pub fn prefill_parallel_guarded<B, F>(
        &self,
        modules: &mut StaticModules<B>,
        prompt_ids: &[i32],
        policy: PromptPolicy,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        mut agree: F,
    ) -> Result<PromptEmbeddings<B::Tensor>, ChannelError>
    where
        B: DistributedNeuralBackend,
        F: FnMut([u64; 8], bool, &<B::Tensor as Tensor>::Context) -> Result<bool, ChannelError>,
    {
        let mut guard = FusionAgreement(Some(&mut agree));
        let mut words = self.invocation_words(1, B::parallel_size(parallel));
        words.extend([
            policy.repeat as u64,
            policy.max_tokens as u64,
            prompt_ids.len() as u64,
        ]);
        words.extend(prompt_ids.iter().map(|&id| id as u64));
        guard.gate(0, &words, policy.framed_length(prompt_ids), context)?;
        self.prefill_with_lookup(
            prompt_ids,
            policy,
            context,
            |tokens| {
                Ok(B::vocabulary_parallel_lookup(
                    &mut modules.embeddings,
                    tokens,
                    EmbeddingLookupPolicy::Strict,
                    parallel,
                    context,
                )?)
            },
            &mut guard,
        )
    }

    fn prefill_with_lookup<T, L>(
        &self,
        prompt_ids: &[i32],
        policy: PromptPolicy,
        context: &T::Context,
        mut lookup: L,
        guard: &mut FusionAgreement<'_, T::Context>,
    ) -> Result<PromptEmbeddings<T>, ChannelError>
    where
        T: Tensor,
        L: FnMut(&T) -> Result<T, ChannelError>,
    {
        let (tokens, pads) = guard.gate(
            20,
            &[],
            prompt_tensors::<T>(prompt_ids, policy, context),
            context,
        )?;
        let audio = guard.tensor(21, lookup(&tokens), context)?;
        let feedback = guard.tensor(22, lookup(&pads), context)?;
        let embedded = guard.tensor(
            23,
            self.fuse_embeddings(&feedback, &audio, &feedback, context),
            context,
        )?;
        Ok(PromptEmbeddings { tokens, embedded })
    }

    fn invocation_words(&self, kind: u64, degree: usize) -> Vec<u64> {
        vec![
            kind,
            degree as u64,
            self.agent_weight.to_bits() as u64,
            self.audio_weight.to_bits() as u64,
            self.function_weight.to_bits() as u64,
        ]
    }
}

/// Exact host descriptor and local-success agreement for channel execution.
/// The selected communication owner retains completion and failure fencing.
pub type ChannelAgreement<'a, C> = dyn FnMut([u64; 8], bool, &C) -> Result<bool, ChannelError> + 'a;

struct FusionAgreement<'a, C: ?Sized>(Option<&'a mut ChannelAgreement<'a, C>>);
impl<C: ?Sized> FusionAgreement<'_, C> {
    fn agree(
        &mut self,
        phase: u64,
        words: &[u64],
        success: bool,
        context: &C,
    ) -> Result<(), ChannelError> {
        let Some(agree) = self.0.as_deref_mut() else {
            return Ok(());
        };
        const PROTOCOL: u64 = 0x465553494f4e0001;
        if !agree(
            [PROTOCOL, phase, words.len() as u64, 0, 0, 0, 0, 0],
            success,
            context,
        )? {
            return Err(ChannelError::Configuration(
                "channel fusion participants disagree or a participant failed",
            ));
        }
        for (index, chunk) in words.chunks(6).enumerate() {
            let mut descriptor = [u64::MAX; 8];
            descriptor[0] = PROTOCOL;
            descriptor[1] = index as u64;
            descriptor[2..2 + chunk.len()].copy_from_slice(chunk);
            if !agree(descriptor, true, context)? {
                return Err(ChannelError::Configuration(
                    "channel fusion participant metadata differs",
                ));
            }
        }
        Ok(())
    }
    fn gate<V>(
        &mut self,
        phase: u64,
        words: &[u64],
        local: Result<V, ChannelError>,
        context: &C,
    ) -> Result<V, ChannelError> {
        let agreement = self.agree(phase, words, local.is_ok(), context);
        let value = local?;
        agreement?;
        Ok(value)
    }
    fn tensor<T: Tensor<Context = C>>(
        &mut self,
        phase: u64,
        local: Result<T, ChannelError>,
        context: &C,
    ) -> Result<T, ChannelError> {
        let mut words = Vec::new();
        if self.0.is_some() {
            if let Ok(value) = &local {
                fusion_tensor_words(&mut words, value);
            }
        }
        self.gate(phase, &words, local, context)
    }
}
fn fusion_tensor_words<T: Tensor>(words: &mut Vec<u64>, value: &T) {
    let dtype = match value.element_type() {
        None => 0,
        Some(TensorElementType::Bool) => 1,
        Some(TensorElementType::F16) => 2,
        Some(TensorElementType::Bf16) => 3,
        Some(TensorElementType::F32) => 4,
        Some(TensorElementType::F64) => 5,
        Some(TensorElementType::I8) => 6,
        Some(TensorElementType::I16) => 7,
        Some(TensorElementType::I32) => 8,
        Some(TensorElementType::I64) => 9,
        Some(TensorElementType::U8) => 10,
        Some(TensorElementType::U16) => 11,
        Some(TensorElementType::U32) => 12,
        Some(TensorElementType::U64) => 13,
        Some(TensorElementType::Complex64) => 14,
    };
    words.extend([dtype, value.shape().len() as u64]);
    words.extend(value.shape().iter().map(|&extent| extent as u64));
}

pub(super) fn compute_type<T: Tensor>(tensor: &T) -> Result<TensorElementType, ChannelError> {
    match tensor.element_type() {
        Some(
            element @ (TensorElementType::F16 | TensorElementType::Bf16 | TensorElementType::F32),
        ) => Ok(element),
        _ => Err(ChannelError::Precision(
            "VoiceChat channels require known F16, BF16 or F32 tensors",
        )),
    }
}
fn validate_feedback<T: Tensor>(audio: &T, agent: &T, function: &T) -> Result<(), ChannelError> {
    if audio.shape().len() != 3
        || agent.shape().len() != 2
        || agent.shape() != function.shape()
        || agent.shape() != &audio.shape()[..2]
    {
        return Err(ChannelError::Shape(
            "feedback token geometry must match projected audio batch and sequence",
        ));
    }
    if audio.element_type() != Some(TensorElementType::F32) {
        return Err(ChannelError::Precision(
            "projected perception must enter channel fusion in F32",
        ));
    }
    Ok(())
}
fn prompt_tensors<T: Tensor>(
    prompt: &[i32],
    policy: PromptPolicy,
    context: &T::Context,
) -> Result<(T, T), ChannelError> {
    let length = policy.framed_length(prompt)?;
    let mut tokens = Vec::with_capacity(length as usize);
    for _ in 0..policy.repeat {
        tokens.push(TEXT_BOS_ID);
        tokens.extend_from_slice(prompt);
        tokens.push(TEXT_EOS_ID);
    }
    Ok((
        T::from_i32_slice(&tokens, &[1, length], context)?,
        T::full_i32(TEXT_PAD_ID, &[1, length], context)?,
    ))
}

/// The independent function vocabulary projection; the backbone owns the agent head.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct FunctionHead<B: NeuralBackend> {
    head: B::Linear,
    #[parameter(skip)]
    width: i32,
    #[parameter(skip)]
    vocabulary: i32,
    #[parameter(skip)]
    parallel: Option<VocabularyParallelRange>,
    #[parameter(skip)]
    format: LinearFormatSpec,
}
impl<B: NeuralBackend> FunctionHead<B> {
    /// Declares the independently materializable canonical function head.
    pub fn new(
        width: i32,
        vocabulary: i32,
        format: LinearFormatSpec,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, ChannelError> {
        let spec = head_spec(width, vocabulary, format.clone())?;
        Ok(Self {
            head: B::linear(spec, context)?,
            width,
            vocabulary,
            parallel: None,
            format,
        })
    }
    /// Logical vocabulary projection geometry, independent of storage format.
    pub fn dimensions(&self) -> super::function_head_layered::FunctionHeadDimensions {
        super::function_head_layered::FunctionHeadDimensions {
            width: self.width,
            vocabulary: self.vocabulary,
        }
    }
    /// Exact physical matrix and companion format selected at construction.
    pub fn format(&self) -> &LinearFormatSpec {
        &self.format
    }
    /// Exact selected vocabulary rows, when this unit requires collective projection.
    pub fn parallel_range(&self) -> Option<&VocabularyParallelRange> {
        self.parallel.as_ref()
    }
    /// Projects the normalized hidden state through only the function head.
    /// The agent head may be released before materializing this parameter group.
    pub fn project(
        &mut self,
        normalized: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, ChannelError> {
        self.validate_hidden(normalized)?;
        if self.parallel.is_some() {
            return Err(ChannelError::Configuration(
                "rank-local function head requires collective projection",
            ));
        }
        Ok(self.head.forward(normalized, context)?)
    }
    /// Declares vocabulary-axis sharding, including physical encoding companions.
    pub fn parallel_parameter_group(&self) -> Result<ParameterGroupSpec, ParallelPlanError> {
        module_parameter_group::<B::Tensor, _>(
            "voicechat.function_head",
            ParameterRole::Vocabulary,
            &self.head,
            |_, shape| {
                if shape.is_empty() {
                    Err(ParallelPlanError::InvalidTensor(
                        "VoiceChat function-head parameter is scalar".into(),
                    ))
                } else {
                    Ok(MemberSharding::Balanced { axis: 0 })
                }
            },
        )
    }
    pub(super) fn validate_replicated_projection(
        &self,
        hidden: &B::Tensor,
    ) -> Result<(), ChannelError> {
        self.validate_hidden(hidden)?;
        if self.parallel.is_some() {
            return Err(ChannelError::Configuration(
                "rank-local function head requires collective projection",
            ));
        }
        Ok(())
    }
    fn validate_hidden(&self, hidden: &B::Tensor) -> Result<(), ChannelError> {
        self.dimensions().validate_hidden(hidden)
    }
}
impl<B: DistributedNeuralBackend> FunctionHead<B> {
    /// Constructs this rank's declared vocabulary rows using the shared projection mechanism.
    pub fn new_parallel(
        width: i32,
        vocabulary: i32,
        format: LinearFormatSpec,
        range: VocabularyParallelRange,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, ChannelError> {
        range.validate_global_rows(vocabulary)?;
        let spec = head_spec(width, vocabulary, format.clone())?;
        Ok(Self {
            head: B::vocabulary_parallel_linear(spec, range.clone(), context)?,
            width,
            vocabulary,
            parallel: Some(range),
            format,
        })
    }
    pub(super) fn validate_parallel_projection(
        &self,
        normalized: &B::Tensor,
        parallel: &B::ParallelContext,
    ) -> Result<(), ChannelError> {
        self.validate_hidden(normalized)?;
        self.parallel
            .as_ref()
            .ok_or(ChannelError::Configuration(
                "collective function projection requires rank-local construction",
            ))?
            .balanced_peer_widths(B::parallel_size(parallel), B::parallel_rank(parallel))?;
        Ok(())
    }
    /// Agrees exact input geometry and local ownership before vocabulary projection.
    /// The backend coordinates any fallible local work preceding its own gather.
    pub fn project_parallel_guarded<F>(
        &mut self,
        normalized: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        mut agree: F,
    ) -> Result<B::Tensor, ChannelError>
    where
        F: FnMut([u64; 8], bool, &<B::Tensor as Tensor>::Context) -> Result<bool, ChannelError>,
    {
        let valid = self.validate_parallel_projection(normalized, parallel);
        let agreement = agree_parallel_projection::<B, _>(
            self.dimensions(),
            normalized,
            parallel,
            valid.is_ok(),
            context,
            &mut agree,
        );
        valid?;
        agreement?;
        self.project_parallel(normalized, parallel, context)
    }
    /// Projects and gathers only the function vocabulary through shared collectives.
    pub fn project_parallel(
        &mut self,
        normalized: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, ChannelError> {
        self.validate_parallel_projection(normalized, parallel)?;
        Ok(B::vocabulary_parallel_project(
            &mut self.head,
            normalized,
            parallel,
            context,
        )?)
    }
    /// Projects both vocabulary shards and gathers full logits through existing collectives.
    pub fn project_channels_parallel(
        &mut self,
        normalized: &B::Tensor,
        backbone: &mut StaticModules<B>,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<DualChannelLogits<B::Tensor>, ChannelError> {
        self.validate_parallel_projection(normalized, parallel)?;
        let agent = backbone.project_normalized_parallel(normalized, parallel, context)?;
        let function = self.project_parallel(normalized, parallel, context)?;
        validate_logits(&agent, &function, self.vocabulary)?;
        Ok(DualChannelLogits { agent, function })
    }
}
pub(super) fn agree_parallel_projection<B, F>(
    dimensions: super::function_head_layered::FunctionHeadDimensions,
    normalized: &B::Tensor,
    parallel: &B::ParallelContext,
    success: bool,
    context: &<B::Tensor as Tensor>::Context,
    agree: &mut F,
) -> Result<(), ChannelError>
where
    B: DistributedNeuralBackend,
    F: FnMut([u64; 8], bool, &<B::Tensor as Tensor>::Context) -> Result<bool, ChannelError>,
{
    let dtype = match normalized.element_type() {
        Some(TensorElementType::F16) => 0,
        Some(TensorElementType::Bf16) => 1,
        Some(TensorElementType::F32) => 2,
        Some(TensorElementType::F64) => 3,
        _ => u64::MAX,
    };
    if !agree(
        [
            0x46554e4348454144,
            dimensions.width as u64,
            dimensions.vocabulary as u64,
            B::parallel_size(parallel) as u64,
            normalized.shape().len() as u64,
            dtype,
            0,
            0,
        ],
        success,
        context,
    )? {
        return Err(ChannelError::Configuration(
            "function-head participants disagree or a participant rejected projection",
        ));
    }
    // Rank agreement precedes variable metadata so even malformed input cannot
    // make participants issue different numbers of collective exchanges.
    for (index, chunk) in normalized.shape().chunks(6).enumerate() {
        let mut descriptor = [u64::MAX; 8];
        descriptor[0] = 0x46554e4348534850;
        descriptor[1] = index as u64;
        for (word, extent) in descriptor[2..].iter_mut().zip(chunk) {
            *word = *extent as u64;
        }
        if !agree(descriptor, true, context)? {
            return Err(ChannelError::Configuration(
                "function-head participant input shapes differ",
            ));
        }
    }
    Ok(())
}

pub(super) fn head_spec(
    width: i32,
    vocabulary: i32,
    format: LinearFormatSpec,
) -> Result<LinearSpec, ChannelError> {
    if width <= 0 || vocabulary <= 0 {
        return Err(ChannelError::Configuration(
            "function-head dimensions must be positive",
        ));
    }
    let mut weight = ParameterSpec::trainable("stt_model.function_head.weight")
        .map_err(eredu_nn::Error::backend)?;
    weight.group = Some("voicechat.function_head".into());
    format.validate_for_weight(&weight)?;
    Ok(LinearSpec {
        input: width,
        output: vocabulary,
        weight,
        bias: None,
        format,
    })
}
pub(super) fn validate_logits<T: Tensor>(
    agent: &T,
    function: &T,
    vocabulary: i32,
) -> Result<(), ChannelError> {
    if agent.shape() != function.shape() || function.shape().last() != Some(&vocabulary) {
        return Err(ChannelError::Shape(
            "agent and function heads must produce the complete shared vocabulary",
        ));
    }
    Ok(())
}

/// Agent and function logits from the same normalized backbone invocation.
#[derive(Debug, Clone)]
pub struct DualChannelLogits<T> {
    /// Agent text logits; agent sampling and logit policy are applied separately.
    pub agent: T,
    /// Function-channel logits; its released decoding rule is always greedy.
    pub function: T,
}
impl<T: Tensor> DualChannelLogits<T> {
    /// Returns function-channel argmax IDs without sampling or host publication.
    pub fn greedy_function(&self, context: &T::Context) -> Result<T, ChannelError> {
        if self.function.shape().is_empty()
            || self.function.shape().last().copied().unwrap_or(0) <= 0
        {
            return Err(ChannelError::Shape(
                "function logits require a nonempty vocabulary axis",
            ));
        }
        let negative = self.function.multiply_scalar(-1.0, context)?;
        Ok(T::argmin_axis(&negative, -1, false, context)?)
    }
}
