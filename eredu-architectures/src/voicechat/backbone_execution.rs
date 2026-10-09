//! Selected Nemotron execution, channel fusion and two shared-hidden readouts.
//!
//! Resident, bounded and collective invocations use the existing layered runtime.
//! State is an unpublished caller-owned branch; this operation neither establishes
//! native completion nor commits state or sampled feedback.

use eredu_nn::RetainedTensorRoots;

use eredu_nn::{
    AttentionCache, DistributedNeuralBackend, GroupedNeuralBackend, NeuralBackend, Tensor,
};
use eredu_runtime::{
    LayerRuntimeState, LayeredInvocation, ParallelLayeredInvocation, RuntimeStateComponents,
};

use super::channels::{
    validate_logits, ChannelAgreement, ChannelError, ChannelFusion, DualChannelLogits, PromptPolicy,
};
use super::function_head_layered::FunctionHeadExecution;
use crate::{
    decoder::StaticModules,
    nemotron_h::{
        EmbeddedInput, LayeredModel, TargetBoundary, TargetBoundarySchema, TargetReadout,
    },
};

/// One prompt or one explicit batch of projected-audio/feedback positions.
pub enum BackboneRequest<'a, T> {
    /// Already tokenized prompt, framed with the released BOS/EOS and PAD feedback.
    Prompt {
        /// Prompt IDs before framing and repetition.
        tokens: &'a [i32],
        /// Explicit repetition and capacity.
        policy: PromptPolicy,
    },
    /// Projected perception or an explicitly selected silence/retained feature.
    Audio {
        /// F32 `[batch, sequence, hidden]` audio channel.
        projected_audio: &'a T,
        /// Previous agent IDs, shaped `[batch, sequence]`.
        agent_tokens: &'a T,
        /// Previous function IDs with the same geometry.
        function_tokens: &'a T,
    },
}

/// Architecture-owned ingress usable with ordinary or partitioned Nemotron execution.
#[derive(Debug, Clone)]
pub struct BackboneInput<T> {
    tokens: T,
    embedded: T,
}
impl<T: Tensor> BackboneInput<T> {
    /// Fuses channels using the shared unpartitioned embedding table.
    pub fn prepare<B: NeuralBackend<Tensor = T>>(
        fusion: &ChannelFusion,
        modules: &mut StaticModules<B>,
        request: BackboneRequest<'_, T>,
        context: &T::Context,
    ) -> Result<Self, ChannelError> {
        match request {
            BackboneRequest::Prompt { tokens, policy } => {
                let prompt = fusion.prefill(modules, tokens, policy, context)?;
                Ok(Self {
                    tokens: prompt.tokens,
                    embedded: prompt.embedded,
                })
            }
            BackboneRequest::Audio {
                projected_audio,
                agent_tokens,
                function_tokens,
            } => {
                let embedded = fusion.fuse(
                    modules,
                    projected_audio,
                    agent_tokens,
                    function_tokens,
                    context,
                )?;
                Ok(Self {
                    tokens: agent_tokens.clone(),
                    embedded,
                })
            }
        }
    }

    /// Fuses channels with the existing vocabulary-parallel lookup/reduction.
    pub fn prepare_parallel<B: DistributedNeuralBackend<Tensor = T>>(
        fusion: &ChannelFusion,
        modules: &mut StaticModules<B>,
        request: BackboneRequest<'_, T>,
        parallel: &B::ParallelContext,
        context: &T::Context,
    ) -> Result<Self, ChannelError> {
        match request {
            BackboneRequest::Prompt { tokens, policy } => {
                let prompt = fusion.prefill_parallel(modules, tokens, policy, parallel, context)?;
                Ok(Self {
                    tokens: prompt.tokens,
                    embedded: prompt.embedded,
                })
            }
            BackboneRequest::Audio {
                projected_audio,
                agent_tokens,
                function_tokens,
            } => {
                let embedded = fusion.fuse_parallel(
                    modules,
                    projected_audio,
                    agent_tokens,
                    function_tokens,
                    parallel,
                    context,
                )?;
                Ok(Self {
                    tokens: agent_tokens.clone(),
                    embedded,
                })
            }
        }
    }

    /// Agrees channel preparation and lookup boundaries before supplied-embedding traversal.
    /// Tensor contents and prepared parameter identity must already match across
    /// participants. The backend guards failures inside each vocabulary lookup.
    pub fn prepare_parallel_guarded<B, F>(
        fusion: &ChannelFusion,
        modules: &mut StaticModules<B>,
        request: BackboneRequest<'_, T>,
        parallel: &B::ParallelContext,
        context: &T::Context,
        agree: F,
    ) -> Result<Self, ChannelError>
    where
        B: DistributedNeuralBackend<Tensor = T>,
        F: FnMut([u64; 8], bool, &T::Context) -> Result<bool, ChannelError>,
    {
        match request {
            BackboneRequest::Prompt { tokens, policy } => {
                let prompt = fusion
                    .prefill_parallel_guarded(modules, tokens, policy, parallel, context, agree)?;
                Ok(Self {
                    tokens: prompt.tokens,
                    embedded: prompt.embedded,
                })
            }
            BackboneRequest::Audio {
                projected_audio,
                agent_tokens,
                function_tokens,
            } => {
                let embedded = fusion.fuse_parallel_guarded(
                    modules,
                    projected_audio,
                    agent_tokens,
                    function_tokens,
                    parallel,
                    context,
                    agree,
                )?;
                Ok(Self {
                    tokens: agent_tokens.clone(),
                    embedded,
                })
            }
        }
    }

    /// Logical token attribution used by Nemotron's existing pipeline transport.
    pub fn tokens(&self) -> &T {
        &self.tokens
    }

    /// Fused hidden input before any Nemotron block executes.
    pub fn embedded(&self) -> &T {
        &self.embedded
    }

    /// Selects normalized hidden output, so the backbone does not evaluate either head.
    pub fn embedded_input(&self) -> EmbeddedInput<'_, T> {
        EmbeddedInput::target_embedded(
            &self.tokens,
            &self.embedded,
            None,
            TargetReadout::NormalizedHidden,
        )
    }

    /// Builds exact supplied-embedding ingress for the existing Nemotron pipeline.
    ///
    /// VoiceChat prompt and feedback IDs use I32 storage, so callers retain
    /// `TargetBoundarySchema::from_args(args).with_signed_tokens()` in every
    /// selected partition. A mismatched schema is rejected before layer execution.
    pub fn partition_input(
        &self,
        schema: &TargetBoundarySchema,
        context: &T::Context,
    ) -> Result<eredu_runtime::LayeredPartitionInput<'_, T, TargetBoundary<T>>, BackboneBoundaryError>
    {
        schema.validate_input(&self.tokens, &self.embedded)?;
        Ok(eredu_runtime::LayeredPartitionInput::Hidden {
            hidden: self.embedded.clone(),
            auxiliary: TargetBoundary::new(
                self.tokens.clone(),
                self.embedded.clone(),
                TargetReadout::NormalizedHidden.encode::<T>(context)?,
            ),
        })
    }
}

/// Exact pipeline ingress validation or neutral readout-control construction failure.
#[derive(Debug, thiserror::Error)]
pub enum BackboneBoundaryError {
    /// Fused input does not satisfy the selected target wire schema.
    #[error(transparent)]
    Input(#[from] crate::nemotron_h::model::TargetBoundaryInputError),
    /// Construction of the readout control failed without invoking any layer.
    #[error(transparent)]
    Neural(#[from] eredu_nn::Error),
}

/// Both readouts and their exact shared normalized input from one backbone traversal.
#[derive(Debug, Clone)]
pub struct BackboneOutput<T> {
    /// Normalized hidden states; normalization is not repeated by either head.
    pub normalized: T,
    /// Full agent and function vocabularies, without sampling or forced overrides.
    pub logits: DualChannelLogits<T>,
}

impl<T> RetainedTensorRoots<T> for BackboneOutput<T> {
    fn visit_retained_tensors<'a>(&'a self, visitor: &mut dyn FnMut(&'a T)) {
        visitor(&self.normalized);
        visitor(&self.logits.agent);
        visitor(&self.logits.function);
    }
}

/// Selected execution of fused channels and the independent agent vocabulary head.
///
/// Providers retain traversal, parameter residency and topology. Mutable neural
/// state stays with the caller's unpublished transaction. Prefill invokes hidden
/// execution alone, without materializing either vocabulary head.
pub trait BackboneExecution<B: NeuralBackend, S: ?Sized> {
    /// Neural composition or parameter-provider failure, preserving its source.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Fuses input and traverses the selected backbone once to normalized hidden states.
    fn execute_hidden(
        &mut self,
        fusion: &ChannelFusion,
        request: BackboneRequest<'_, B::Tensor>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error>;

    /// Projects normalized hidden states through the selected agent vocabulary head.
    fn project_agent(
        &mut self,
        normalized: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, ChannelError>;

    /// Reads the logical backbone frontier without advancing its state.
    fn position(&self, state: &mut S) -> Result<u64, BackboneStateError>;
}

/// Selected backbone state cannot describe one complete invocation frontier.
#[derive(Debug, thiserror::Error)]
pub enum BackboneStateError {
    /// Access to a selected state slot failed.
    #[error(transparent)]
    State(#[from] eredu_runtime::StateError),
    /// Selected local stateful layers have no common valid frontier.
    #[error("invalid VoiceChat backbone state: {0}")]
    Inconsistent(&'static str),
}

/// Reads the common frontier of the selected stateful layered slots.
/// Stateless slots are skipped; no remote slot is accessed. A partition with
/// no local stateful slot needs an explicit provider-owned logical frontier.
pub fn layered_position<B, S>(state: &mut S) -> Result<u64, BackboneStateError>
where
    B: NeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: RuntimeStateComponents<B>,
{
    let layers = state
        .optional_layout()
        .map(|layout| {
            layout
                .layers()
                .iter()
                .enumerate()
                .filter_map(|(index, policy)| {
                    (!matches!(policy, eredu_core::cache::LayerCachePolicy::NoState))
                        .then_some(index)
                })
                .collect::<Vec<_>>()
        })
        .filter(|layers| !layers.is_empty())
        .ok_or(BackboneStateError::Inconsistent(
            "no local backbone frontier",
        ))?;
    let mut frontier = None;
    for layer in layers {
        let position = u64::try_from(state.layer(layer)?.position())
            .map_err(|_| BackboneStateError::Inconsistent("negative component frontier"))?;
        if frontier.is_some_and(|previous| previous != position) {
            return Err(BackboneStateError::Inconsistent(
                "backbone layers have different frontiers",
            ));
        }
        frontier = Some(position);
    }
    Ok(frontier.expect("nonempty local layers were checked"))
}

/// Fusion or readout failure, or the unchanged selected execution failure.
#[derive(Debug, thiserror::Error)]
pub enum BackboneExecutionError<E> {
    /// A channel equation or vocabulary head failed.
    #[error(transparent)]
    Channel(#[from] ChannelError),
    /// The selected runtime failed, preserving its architecture or policy source.
    #[error("VoiceChat backbone invocation failed")]
    Runtime(#[source] E),
}

/// Selects ordinary fusion, shared Nemotron traversal and agent projection.
/// The borrowed runtime retains its resident or bounded parameter provider.
pub struct OrdinaryBackbone<'a, B: NeuralBackend, R> {
    runtime: &'a mut R,
    backend: std::marker::PhantomData<fn() -> B>,
}

impl<'a, B: NeuralBackend, R> OrdinaryBackbone<'a, B, R> {
    /// Borrows an already constructed ordinary layered runtime.
    pub fn new(runtime: &'a mut R) -> Self {
        Self {
            runtime,
            backend: std::marker::PhantomData,
        }
    }
}

impl<B, S, R> BackboneExecution<B, S> for OrdinaryBackbone<'_, B, R>
where
    B: GroupedNeuralBackend + DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
    R: LayeredInvocation<B, S, Architecture = LayeredModel<B>>,
    R::Error: std::error::Error + Send + Sync + 'static,
{
    type Error = BackboneExecutionError<R::Error>;

    fn execute_hidden(
        &mut self,
        fusion: &ChannelFusion,
        request: BackboneRequest<'_, B::Tensor>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        let input = BackboneInput::prepare(
            fusion,
            self.runtime.architecture_mut().static_modules_mut(),
            request,
            context,
        )?;
        self.runtime
            .invoke(input.embedded_input(), state, context)
            .map_err(BackboneExecutionError::Runtime)
    }

    fn project_agent(
        &mut self,
        normalized: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, ChannelError> {
        Ok(self
            .runtime
            .architecture_mut()
            .static_modules_mut()
            .project_normalized(normalized, context)?)
    }

    fn position(&self, state: &mut S) -> Result<u64, BackboneStateError> {
        layered_position::<B, S>(state)
    }
}

/// Selects collective fusion, shared Nemotron traversal and agent projection.
/// The borrowed runtime retains its existing parameter provider; the parallel
/// context is the same selected group for all three operations. Optional ingress
/// agreement guards channel preparation and fusion before traversal. The runtime
/// and block mechanisms retain responsibility for their internal collectives.
pub struct ParallelBackbone<'a, B: NeuralBackend, R> {
    runtime: &'a mut R,
    parallel: &'a B::ParallelContext,
    ingress_agreement: Option<&'a mut ChannelAgreement<'a, <B::Tensor as Tensor>::Context>>,
}

impl<'a, B: NeuralBackend, R> ParallelBackbone<'a, B, R> {
    /// Borrows a rank-local runtime and its collective group, optionally agreeing
    /// ingress through that group's retained failure authority.
    pub fn new(
        runtime: &'a mut R,
        parallel: &'a B::ParallelContext,
        ingress_agreement: Option<&'a mut ChannelAgreement<'a, <B::Tensor as Tensor>::Context>>,
    ) -> Self {
        Self {
            runtime,
            parallel,
            ingress_agreement,
        }
    }
}

impl<B, S, R> BackboneExecution<B, S> for ParallelBackbone<'_, B, R>
where
    B: GroupedNeuralBackend + DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
    R: ParallelLayeredInvocation<B, S, Architecture = LayeredModel<B>>,
    R::Error: std::error::Error + Send + Sync + 'static,
{
    type Error = BackboneExecutionError<R::Error>;

    fn execute_hidden(
        &mut self,
        fusion: &ChannelFusion,
        request: BackboneRequest<'_, B::Tensor>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        let modules = self.runtime.architecture_mut().static_modules_mut();
        let input = match self.ingress_agreement.as_deref_mut() {
            Some(agree) => BackboneInput::prepare_parallel_guarded(
                fusion,
                modules,
                request,
                self.parallel,
                context,
                agree,
            )?,
            None => {
                BackboneInput::prepare_parallel(fusion, modules, request, self.parallel, context)?
            }
        };
        self.runtime
            .invoke_parallel(input.embedded_input(), state, self.parallel, context)
            .map_err(BackboneExecutionError::Runtime)
    }

    fn project_agent(
        &mut self,
        normalized: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, ChannelError> {
        Ok(self
            .runtime
            .architecture_mut()
            .static_modules_mut()
            .project_normalized_parallel(normalized, self.parallel, context)?)
    }

    fn position(&self, state: &mut S) -> Result<u64, BackboneStateError> {
        layered_position::<B, S>(state)
    }
}

/// Executes fused channels once through the selected backbone and both readouts.
///
/// This is a neural composition contract, not a public session operation. The
/// caller supplies a branch state and must discard it on error; after success it
/// must retain native completion before committing that state or publishing output.
pub fn execute<B, S: ?Sized, R, F>(
    runtime: &mut R,
    function: &mut F,
    fusion: &ChannelFusion,
    request: BackboneRequest<'_, B::Tensor>,
    state: &mut S,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<BackboneOutput<B::Tensor>, BackboneExecutionError<R::Error>>
where
    B: NeuralBackend,
    R: BackboneExecution<B, S>,
    F: FunctionHeadExecution<B>,
{
    let normalized = runtime
        .execute_hidden(fusion, request, state, context)
        .map_err(BackboneExecutionError::Runtime)?;
    function.validate_projection(&normalized)?;
    let agent = runtime.project_agent(&normalized, context)?;
    let function_logits = function.project(&normalized, context)?;
    validate_logits(&agent, &function_logits, function.dimensions().vocabulary)?;
    Ok(BackboneOutput {
        normalized,
        logits: DualChannelLogits {
            agent,
            function: function_logits,
        },
    })
}
