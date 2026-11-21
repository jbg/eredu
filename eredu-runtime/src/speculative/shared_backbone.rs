//! Masked drafting and causal verification over one retained executable.

use eredu_core::refinement::RefinementBackend;
pub(super) mod execution;
pub use execution::SharedRefinementState;

use super::{
    autoregressive::{AutoregressiveMechanisms, AutoregressivePass},
    causal_verification, CausalVerification,
};
use crate::{
    executable_binding::retained_session::PreparedInputPartPlan,
    generation::refinement::architecture::BlockRefinementGeometry,
};
use eredu_core::{
    BoundedCompletion, SpeculativeCommit, SpeculativeExecutor, SpeculativePrefill, Submission,
};

mod transaction;
pub use transaction::{SharedBackboneTransaction, SharedBackboneTransactionError};

/// Draft-only additive projection geometry, independent of parameter backing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SharedDraftAdapterGeometry {
    /// Complete replicated low-rank width.
    pub rank: std::num::NonZeroU32,
    /// Full attention-channel input width.
    pub input: std::num::NonZeroU32,
    /// Full residual output width.
    pub output: std::num::NonZeroU32,
    /// Number of independently owned adapter-bearing layers.
    pub layers: std::num::NonZeroU32,
}

/// Family-declared numerical defaults and geometry retained at construction.
/// Native adapters consume these facts without interpreting model configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SharedBackboneExecutionPolicy {
    geometry: crate::generation::refinement::architecture::BlockRefinementGeometry,
    mask_token_id: u32,
    defaults: eredu_core::refinement::MaskedRefinementConfig,
    diffusion: eredu_core::refinement::MaskedRefinementConfig,
    seed_first: bool,
    adapter: Option<SharedDraftAdapterGeometry>,
    inherit_target_temperature: bool,
}

impl SharedBackboneExecutionPolicy {
    /// Retains validated family policy without allocating a model or random key.
    pub fn new(
        geometry: crate::generation::refinement::architecture::BlockRefinementGeometry,
        mask_token_id: u32,
        defaults: eredu_core::refinement::MaskedRefinementConfig,
        diffusion: eredu_core::refinement::MaskedRefinementConfig,
        seed_first: bool,
        adapter: Option<SharedDraftAdapterGeometry>,
        inherit_target_temperature: bool,
    ) -> Result<Self, eredu_core::decoding::DecodingModeError> {
        SharedBackboneDraftOptions::new(geometry, defaults, mask_token_id, false)?;
        Ok(Self {
            geometry,
            mask_token_id,
            defaults,
            diffusion,
            seed_first,
            adapter,
            inherit_target_temperature,
        })
    }
    /// Complete trained invocation geometry.
    pub const fn geometry(
        self,
    ) -> crate::generation::refinement::architecture::BlockRefinementGeometry {
        self.geometry
    }
    /// Exact unresolved token identity of the retained model.
    pub const fn mask_token_id(self) -> u32 {
        self.mask_token_id
    }
    /// Separately retained draft-only adapter availability.
    pub const fn adapter_available(self) -> bool {
        self.adapter.is_some()
    }
    /// Exact additive projection geometry; its parameters belong to the shared catalog.
    pub const fn adapter_geometry(self) -> Option<SharedDraftAdapterGeometry> {
        self.adapter
    }
    /// Whether ordinary refinement seeds its first position from context logits.
    pub const fn seed_first(self) -> bool {
        self.seed_first
    }
    /// Ordinary unmasking defaults, resolved against request temperature.
    pub fn diffusion_policy(
        self,
        temperature: f32,
    ) -> Result<eredu_core::refinement::RefinementPolicy, eredu_core::decoding::DecodingModeError>
    {
        let mut config = self.diffusion.options();
        config.temperature = temperature;
        Ok(eredu_core::refinement::RefinementPolicy::MaskedUnmasking {
            config: config.try_into().map_err(|_| {
                eredu_core::decoding::DecodingModeError::Unsupported("diffusion temperature")
            })?,
            mask_token_id: self.mask_token_id,
        })
    }
    /// Maximum proposals exclude the seeded committed position.
    pub fn proposal_capacity(self) -> std::num::NonZeroUsize {
        std::num::NonZeroUsize::new(self.geometry.canvas_length.get() as usize - 1).unwrap()
    }
    /// Admits a lane against the shared target's selected draft policy.
    pub fn prepare_lane(
        self,
        config: &eredu_core::generation::GenerationRequestConfig,
        generation: &eredu_core::TextGenerationConfig,
    ) -> Result<SharedBackboneDraftOptions, eredu_core::generation::GenerationError> {
        use eredu_core::{
            decoding::{DecodingMode, DecodingModeError, SpeculationOptions},
            generation::GenerationError,
        };

        super::batch::validate_lane_proposal_capacity(config, self.proposal_capacity().get())?;
        if generation
            .decoding_mode()
            .is_some_and(|mode| mode != config.decoding_mode)
        {
            return Err(GenerationError::DecodingMode(
                DecodingModeError::Unsupported("lane sampling and execution modes disagree"),
            ));
        }
        let options = match config.decoding_mode {
            DecodingMode::SelfSpeculation(options) => options,
            DecodingMode::Autoregressive | DecodingMode::Diffusion { .. } => {
                SpeculationOptions::default().with_proposal_capacity(self.proposal_capacity())
            }
        };
        self.prepare(options, config.temperature)
            .map_err(GenerationError::DecodingMode)
    }

    /// Resolves one request against the already retained numerical declaration.
    pub fn prepare(
        self,
        options: eredu_core::decoding::SpeculationOptions,
        target_temperature: f32,
    ) -> Result<SharedBackboneDraftOptions, eredu_core::decoding::DecodingModeError> {
        use eredu_core::{decoding::DecodingModeError, refinement::RefinementPolicy};
        let options = options.request();
        if options.max_draft_tokens > self.proposal_capacity() {
            return Err(DecodingModeError::Unsupported(
                "proposal count exceeds the seeded draft canvas",
            ));
        }
        if options.draft_adapter && self.adapter.is_none() {
            return Err(DecodingModeError::Unsupported(
                "no draft adapter was retained at load time",
            ));
        }
        let policy = match options.refinement {
            Some(RefinementPolicy::MaskedUnmasking {
                config,
                mask_token_id,
            }) if mask_token_id == self.mask_token_id => config,
            Some(_) => {
                return Err(DecodingModeError::Unsupported(
                    "masked draft numerical policy or mask identity",
                ));
            }
            None => {
                let mut defaults = self.defaults.options();
                if self.inherit_target_temperature {
                    defaults.temperature = target_temperature;
                }
                defaults
                    .try_into()
                    .map_err(|_| DecodingModeError::Unsupported("draft sampling temperature"))?
            }
        };
        SharedBackboneDraftOptions::new(
            self.geometry,
            policy,
            self.mask_token_id,
            options.draft_adapter,
        )
    }
}

/// Validated request-local masked draft policy over a complete admitted canvas.
#[derive(Debug, Clone, Copy)]
pub struct SharedBackboneDraftOptions {
    geometry: crate::generation::refinement::architecture::BlockRefinementGeometry,
    policy: eredu_core::refinement::MaskedRefinementConfig,
    mask_token_id: u32,
    adapter: bool,
}

impl SharedBackboneDraftOptions {
    /// Checks token and complete-canvas geometry before native allocation.
    pub fn new(
        geometry: crate::generation::refinement::architecture::BlockRefinementGeometry,
        policy: eredu_core::refinement::MaskedRefinementConfig,
        mask_token_id: u32,
        adapter: bool,
    ) -> Result<Self, eredu_core::decoding::DecodingModeError> {
        if geometry.canvas_length.get() < 2
            || geometry.canvas_length.get() > i32::MAX as u32
            || geometry.vocabulary_size.get() > i32::MAX as u32
            || mask_token_id >= geometry.vocabulary_size.get()
        {
            return Err(eredu_core::decoding::DecodingModeError::Unsupported(
                "masked draft geometry",
            ));
        }
        Ok(Self {
            geometry,
            policy,
            mask_token_id,
            adapter,
        })
    }
    /// Complete invocation extent, retained even for a shortened proposal budget.
    pub const fn geometry(
        self,
    ) -> crate::generation::refinement::architecture::BlockRefinementGeometry {
        self.geometry
    }
    /// Mask initialization and unmasking controls; causal target sampling is separate.
    pub const fn policy(self) -> eredu_core::refinement::RefinementPolicy {
        eredu_core::refinement::RefinementPolicy::MaskedUnmasking {
            config: self.policy,
            mask_token_id: self.mask_token_id,
        }
    }
    /// Immutable draft-only adapter and attention authority.
    pub const fn invocation(self) -> eredu_core::decoding::AttentionInvocation {
        if self.adapter {
            eredu_core::decoding::AttentionInvocation::DRAFT
        } else {
            eredu_core::decoding::AttentionInvocation::DIFFUSION
        }
    }
}

/// Additional operations for a masked assistant sharing the causal target's
/// parameters. Ordinary operations retain their causal, adapter-disabled equations.
pub trait SharedBackboneMechanisms: AutoregressiveMechanisms {
    /// Borrows the selected context for a shorter executable acquisition.
    fn reborrow_context<'a: 'b, 'b>(context: Self::Context<'a>) -> Self::Context<'b>;

    /// Installs a verified cache at the settled commitment boundary.
    fn publish_committed(
        model: &mut Self::Model,
        state: &mut Self::State,
    ) -> Result<(), Self::Error>;

    /// Native activation value, independent of scores and parameter ownership.
    type ActivationTensor;

    /// Native transforms and retained transport for the portable speculative observer.
    type Capture: for<'a> crate::capture::SpeculativeCaptureMechanisms<
            Context<'a> = Self::Context<'a>,
            Provider: crate::capture::CaptureBackendProvider<Tensor = Self::ActivationTensor>,
            Error = Self::Error,
        > + 'static;

    /// Describes decoder positions and media axes accepted by this exact prompt.
    fn activation_prefill_plans(
        model: &Self::Model,
        input: &Self::Input,
    ) -> Result<Vec<PreparedInputPartPlan>, Self::Error>;

    /// Invokes the bound prompt with an already admitted activation observer.
    fn prefill_with_activation(
        model: &mut Self::Model,
        input: &Self::Input,
        state: &mut Self::State,
        observer: &mut dyn crate::ActivationObserver<Self::ActivationTensor, Self::Error>,
    ) -> Result<(Self::Output, usize), Self::Error>;

    /// Invokes causal decode with an already admitted activation observer.
    fn decode_with_activation(
        model: &mut Self::Model,
        tokens: &[u32],
        state: &mut Self::State,
        observer: &mut dyn crate::ActivationObserver<Self::ActivationTensor, Self::Error>,
    ) -> Result<Self::Output, Self::Error>;

    /// Request-owned proposal randomness, independent of target sampling.
    type ProposalRandomness: Clone;
    /// Retained declaration for ordinary target refinement.
    fn ordinary_refinement_policy(
        model: &Self::Model,
    ) -> Option<crate::generation::refinement::architecture::RefinementExecutionPolicy>;

    /// Retained declaration for masked shared-backbone drafting.
    fn shared_backbone_policy(model: &Self::Model) -> Option<SharedBackboneExecutionPolicy>;

    /// Creates the native root for model randomness.
    fn model_randomness_root(seed: u64) -> Result<Self::ProposalRandomness, Self::Error>;

    /// Selects one logical substream from an immutable root.
    fn split_model_randomness(
        root: &Self::ProposalRandomness,
        substream: usize,
        context: Self::Context<'_>,
    ) -> Result<Self::ProposalRandomness, Self::Error>;

    /// Settles the selected native keys before transferring their ownership.
    fn settle_model_randomness(randomness: &[&Self::ProposalRandomness])
        -> Result<(), Self::Error>;

    /// Native numerical execution queue.
    type Queue: ?Sized + 'static;
    /// Native numerical operations over one single-sequence block.
    type Refinement: RefinementBackend<
        RandomState = Self::ProposalRandomness,
        Error = Self::Error,
        Context = Self::Queue,
    >;
    /// Native queue used by target refinement.
    fn target_refinement_context<'a>(context: Self::Context<'a>) -> &'a Self::Queue;
    /// Native queue available to the masked draft on the target device.
    fn draft_refinement_context<'a>(context: Self::Context<'a>) -> Option<&'a Self::Queue>;
    /// Maps validation errors to the execution adapter's error type.
    fn execution_error(error: impl std::error::Error + Send + Sync + 'static) -> Self::Error;
    /// Exact block geometry implemented by the retained executable.
    fn block_geometry(model: &Self::Model) -> Option<BlockRefinementGeometry>;
    /// Native score format of target logits.
    fn score_format(
        logits: &Self::Logits,
    ) -> <Self::Refinement as RefinementBackend>::ConditioningFormat;
    /// Native conditioning format for masked drafting.
    fn draft_score_format() -> <Self::Refinement as RefinementBackend>::ConditioningFormat;
    /// Borrows or shares a causal score row for the numerical backend.
    fn refinement_scores(logits: &Self::Logits) -> <Self::Refinement as RefinementBackend>::Logits;
    /// Invokes the selected workspace with already prepared native values.
    fn evaluate_refinement(
        model: &mut Self::Model,
        prefix: &mut Self::State,
        canvas: &<Self::Refinement as RefinementBackend>::Token,
        feedback: Option<&<Self::Refinement as RefinementBackend>::Logits>,
        geometry: BlockRefinementGeometry,
        invocation: eredu_core::decoding::AttentionInvocation,
        context: &Self::Queue,
        observer: Option<&mut dyn crate::ActivationObserver<Self::ActivationTensor, Self::Error>>,
    ) -> Result<<Self::Refinement as RefinementBackend>::Logits, Self::Error>
    where
        Self: Sized;
    /// Reads exactly the selected token interval from the completed canvas.
    fn read_refinement_tokens(
        tokens: &<Self::Refinement as RefinementBackend>::Token,
        range: std::ops::Range<usize>,
        context: &Self::Queue,
    ) -> Result<Vec<u32>, Self::Error>;
    /// Extracts one retained score row for target sampling.
    fn refinement_score_row(
        scores: &<Self::Refinement as RefinementBackend>::Logits,
        position: usize,
        context: &Self::Queue,
    ) -> Result<Self::Logits, Self::Error>;
    /// Settles native numerical values before publishing paused target state.
    fn settle_refinement(state: &SharedRefinementState<Self>) -> Result<(), Self::Error>
    where
        Self: Sized;
    /// Bounds retained numerical values and host policy in a paused iteration.
    fn refinement_state_estimate(
        state: &SharedRefinementState<Self>,
    ) -> Option<eredu_core::execution_control::SnapshotEstimate>
    where
        Self: Sized;
    /// Complete retained bytes of a cloned proposal stream, including native backing.
    fn proposal_randomness_bytes(randomness: &Self::ProposalRandomness) -> Option<u64>;
}

/// One lane's prepared context, immutable draft policy and independent stream.
pub struct SharedBackboneInput<I, O, R> {
    input: I,
    options: O,
    randomness: R,
    refinement_randomness: R,
}
impl<I, O, R> SharedBackboneInput<I, O, R> {
    /// Binds complete context and proposal state before entering the scheduler.
    pub const fn new(input: I, options: O, randomness: R, refinement_randomness: R) -> Self {
        Self {
            input,
            options,
            randomness,
            refinement_randomness,
        }
    }
}

/// Settled causal prefix and the proposal stream at its commitment boundary.
pub struct SharedBackboneSeed<C, O, R, F> {
    prefix: C,
    options: O,
    randomness: R,
    refinement_randomness: R,
    #[cfg(feature = "tensor-capture")]
    activations: Option<crate::capture::SpeculativeActivationCheckpoint>,
    refinement: Option<F>,
}
impl<C: Clone, O: Clone, R: Clone, F: Clone> Clone for SharedBackboneSeed<C, O, R, F> {
    fn clone(&self) -> Self {
        Self {
            prefix: self.prefix.clone(),
            options: self.options.clone(),
            randomness: self.randomness.clone(),
            refinement_randomness: self.refinement_randomness.clone(),
            #[cfg(feature = "tensor-capture")]
            activations: self.activations.clone(),
            refinement: self.refinement.clone(),
        }
    }
}

/// Selected masked tokens remain tentative until causal target matching.
#[derive(Clone)]
pub struct SharedBackboneProposal<O, R> {
    tokens: Vec<u32>,
    cursor: usize,
    preceding_token: u32,
    refinement_randomness: R,
    randomness: R,
    options: O,
}

impl<O, R> SharedBackboneProposal<O, R> {
    /// Spent proposal randomness survives accepted, rejected and interrupted prefixes.
    pub fn randomness(&self) -> &R {
        &self.randomness
    }
}

/// One borrowed model, one canonical cache, and the ordinary speculative scheduler.
pub struct SharedBackboneExecutor<'a, M: SharedBackboneMechanisms> {
    model: &'a mut M::Model,
    capacity: std::num::NonZeroUsize,
    #[cfg(feature = "tensor-capture")]
    observer: Option<
        Box<dyn crate::inspection::SpeculativeActivationObserver<M::ActivationTensor, M::Error>>,
    >,
    #[cfg(feature = "tensor-capture")]
    execution_started: bool,
    #[cfg(feature = "tensor-capture")]
    observation_configured: bool,
}

impl<'a, M: SharedBackboneMechanisms> SharedBackboneExecutor<'a, M> {
    /// Binds admitted proposal capacity over one parameter owner. Each input
    /// supplies its own immutable policy and independent proposal stream.
    pub fn new(model: &'a mut M::Model, capacity: std::num::NonZeroUsize) -> Self {
        Self {
            model,
            capacity,
            #[cfg(feature = "tensor-capture")]
            observer: None,
            #[cfg(feature = "tensor-capture")]
            execution_started: false,
            #[cfg(feature = "tensor-capture")]
            observation_configured: false,
        }
    }

    /// Resumes drafting from the retained canonical prefix without re-encoding context.
    pub fn resume_seed(
        &self,
        saved: M::Checkpoint,
        options: SharedBackboneDraftOptions,
        randomness: M::ProposalRandomness,
        refinement_randomness: M::ProposalRandomness,
    ) -> SharedBackboneSeed<
        M::Checkpoint,
        SharedBackboneDraftOptions,
        M::ProposalRandomness,
        SharedRefinementState<M>,
    > {
        SharedBackboneSeed {
            prefix: saved,
            options,
            randomness,
            refinement_randomness,
            #[cfg(feature = "tensor-capture")]
            activations: None,
            refinement: None,
        }
    }
}

impl<M: SharedBackboneMechanisms> eredu_core::speculative::SpeculativeCacheFactory
    for SharedBackboneExecutor<'_, M>
{
    fn create_cache<'context>(
        &mut self,
        _: &(),
        context: Self::Context<'context>,
    ) -> Result<Self::Cache, Self::Error>
    where
        Self: 'context,
    {
        M::empty(self.model, AutoregressivePass::TargetPrefill, context)
    }
}

impl<M: SharedBackboneMechanisms> SpeculativeExecutor for SharedBackboneExecutor<'_, M> {
    #[cfg(feature = "tensor-capture")]
    fn configure_activation_capture<'context>(
        &mut self,
        plan: eredu_core::speculative::AdmittedSpeculativeActivations,
        request: eredu_core::SpeculativeRequestId,
        context: Self::Context<'context>,
    ) -> Result<(), eredu_core::speculative::SpeculativeControlError>
    where
        Self: 'context,
    {
        if self.execution_started || self.observation_configured {
            return Err(eredu_core::speculative::SpeculativeControlError::Invalid(
                "shared activation authority must be installed once before execution",
            ));
        }
        self.observer =
            crate::capture::bind_speculative_observer::<M::Capture>(&plan, request, context)?;
        self.observation_configured = true;
        Ok(())
    }
    #[cfg(feature = "tensor-capture")]
    fn requires_activation_origin(&self) -> bool {
        self.observer.is_some()
    }
    #[cfg(feature = "tensor-capture")]
    fn set_activation_origin(
        &mut self,
        origin: Option<eredu_core::speculative::SpeculativeActivationOrigin>,
    ) {
        if let Some(observer) = &mut self.observer {
            observer.set_activation_origin(origin);
        }
    }
    #[cfg(feature = "tensor-capture")]
    fn take_activation_capture(
        &mut self,
    ) -> Option<eredu_core::speculative::SpeculativeActivationCapture> {
        self.observer.as_mut()?.take_activation_capture()
    }
    #[cfg(feature = "tensor-capture")]
    fn take_activation_error(
        &mut self,
    ) -> Option<eredu_core::speculative::SpeculativeControlError> {
        self.observer.as_mut()?.take_activation_error()
    }
    #[cfg(feature = "tensor-capture")]
    fn readmit_activation_interventions(
        &mut self,
        plan: eredu_core::speculative::AdmittedSpeculativeActivations,
    ) -> Result<(), eredu_core::speculative::SpeculativeControlError> {
        self.observer
            .as_mut()
            .ok_or(
                eredu_core::speculative::SpeculativeControlError::Unsupported(
                    "shared run has no internal authority",
                ),
            )?
            .readmit_activation_interventions(plan)
    }
    fn logits_snapshot_estimate(
        &self,
        logits: &Self::Logits,
    ) -> Option<eredu_core::execution_control::SnapshotEstimate> {
        M::logits_snapshot_estimate(logits)
    }
    fn coordinate_speculative_step<'context>(
        &mut self,
        local: Vec<eredu_core::SpeculativeScheduleState>,
        context: Self::Context<'context>,
    ) -> Result<Vec<eredu_core::SpeculativeScheduleState>, eredu_core::BackendFailure>
    where
        Self: 'context,
    {
        M::coordinate_speculative_step(local, context)
    }
    fn agree_text_preparation<'context>(
        &mut self,
        stage: eredu_core::run_preparation::TextPreparationStage,
        status: eredu_core::run_preparation::TextPreparationStatus,
        context: Self::Context<'context>,
    ) -> Result<eredu_core::run_preparation::TextPreparationOutcome, eredu_core::BackendFailure>
    where
        Self: 'context,
    {
        M::agree_text_preparation(stage, status, context)
    }
    fn coordinate_decoding_control<'context>(
        &mut self,
        local: eredu_core::decoding::DecodingCoordination,
        context: Self::Context<'context>,
    ) -> Result<eredu_core::decoding::DecodingCoordination, eredu_core::BackendFailure>
    where
        Self: 'context,
    {
        M::coordinate_decoding_control(local, context)
    }
    fn configure_decoding_mode(
        &mut self,
        state: &mut Self::TargetState,
        mode: eredu_core::decoding::DecodingMode,
        temperature: f32,
    ) -> Result<(), eredu_core::GenerationError> {
        if let eredu_core::decoding::DecodingMode::SelfSpeculation(options) = mode {
            state.options =
                execution::prepare_draft_options::<M>(self.model, options, temperature)?;
        }
        Ok(())
    }
    fn decoding_capabilities(&self) -> eredu_core::decoding::DecodingCapabilities {
        let mut capabilities = execution::draft_capabilities::<M>(self.model);
        capabilities.max_draft_tokens = Some(self.capacity);
        capabilities.with_completion_support(
            M::Completion::supports_cancellation(
                eredu_core::CompletionCancellationMode::NativeCancel,
            ),
            M::Completion::supports_cancellation(
                eredu_core::CompletionCancellationMode::QuarantineUntilComplete,
            ),
        )
    }
    fn refine_block<'a>(
        &mut self,
        state: &mut Self::TargetState,
        next_logits: &Self::Logits,
        committed: &[u32],
        policy: Option<eredu_core::refinement::RefinementPolicy>,
        temperature: f32,
        context: Self::Context<'a>,
    ) -> Result<Option<eredu_core::speculative::RefinementProgress<Self::Logits>>, Self::Error>
    where
        Self: 'a,
    {
        #[cfg(feature = "tensor-capture")]
        {
            self.execution_started = true;
        }
        execution::refine::<M>(
            self.model,
            &state.prefix,
            next_logits,
            committed,
            policy,
            temperature,
            &mut state.refinement_randomness,
            &mut state.refinement,
            context,
            #[cfg(feature = "tensor-capture")]
            self.observer.as_mut().map(|observer| {
                &mut **observer as &mut dyn crate::inspection::SpeculativeActivationObserver<_, _>
            }),
            #[cfg(not(feature = "tensor-capture"))]
            None,
        )
    }
    fn verification_policy(&self) -> eredu_core::speculative::SpeculativeVerificationPolicy {
        eredu_core::speculative::SpeculativeVerificationPolicy::TargetMatching
    }
    type Input = SharedBackboneInput<M::Input, SharedBackboneDraftOptions, M::ProposalRandomness>;
    type Cache = M::State;
    type TargetState = SharedBackboneSeed<
        M::Checkpoint,
        SharedBackboneDraftOptions,
        M::ProposalRandomness,
        SharedRefinementState<M>,
    >;
    type DraftState = SharedBackboneProposal<SharedBackboneDraftOptions, M::ProposalRandomness>;
    type CacheCheckpoint = M::Checkpoint;
    type Verification = CausalVerification<M::Output>;
    type Logits = M::Logits;
    type Context<'a>
        = M::Context<'a>
    where
        Self: 'a;
    type Completion = M::Completion;
    type Telemetry = M::Telemetry;
    type Error = M::Error;

    fn max_proposals(&self) -> usize {
        self.capacity.get()
    }

    fn prefill<'context>(
        &mut self,
        input: Self::Input,
        cache: &mut Self::Cache,
        context: Self::Context<'context>,
    ) -> Result<SpeculativePrefill<Self::TargetState, Self::Logits>, Self::Error>
    where
        Self: 'context,
    {
        #[cfg(feature = "tensor-capture")]
        {
            self.execution_started = true;
        }
        let (output, count) = execution::prefill_observed::<M>(
            self.model,
            &input.input,
            cache,
            context,
            #[cfg(feature = "tensor-capture")]
            self.observer.as_mut().map(|observer| {
                &mut **observer as &mut dyn crate::inspection::SpeculativeActivationObserver<_, _>
            }),
            #[cfg(not(feature = "tensor-capture"))]
            None,
        )?;
        if count == 0 {
            return Err(M::invalid(
                "shared-backbone drafting requires a nonempty prompt",
            ));
        }
        Ok(SpeculativePrefill::new(
            M::logits(&output, count - 1, context)?,
            SharedBackboneSeed {
                prefix: M::checkpoint(cache)?,
                randomness: input.randomness,
                refinement_randomness: input.refinement_randomness,
                options: input.options,
                #[cfg(feature = "tensor-capture")]
                activations: None,
                refinement: None,
            },
            count,
        ))
    }

    fn begin_proposal<'context>(
        &mut self,
        seed: &Self::TargetState,
        last_token: u32,
        capacity: usize,
        context: Self::Context<'context>,
    ) -> Result<Self::DraftState, Self::Error>
    where
        Self: 'context,
    {
        if capacity > self.capacity.get() {
            return Err(M::invalid("invalid shared-backbone proposal capacity"));
        }
        #[cfg(feature = "tensor-capture")]
        {
            self.execution_started = true;
        }
        let mut randomness = seed.randomness.clone();
        let tokens = if capacity == 0 {
            Vec::new()
        } else {
            execution::draft::<M>(
                self.model,
                &seed.prefix,
                last_token,
                capacity,
                &seed.options,
                &mut randomness,
                context,
                #[cfg(feature = "tensor-capture")]
                self.observer.as_mut().map(|observer| {
                    &mut **observer
                        as &mut dyn crate::inspection::SpeculativeActivationObserver<_, _>
                }),
                #[cfg(not(feature = "tensor-capture"))]
                None,
            )?
        };
        if tokens.len() != capacity {
            return Err(M::invalid(
                "masked draft returned a different proposal extent",
            ));
        }
        Ok(SharedBackboneProposal {
            tokens,
            cursor: 0,
            preceding_token: last_token,
            refinement_randomness: seed.refinement_randomness.clone(),
            randomness,
            options: seed.options.clone(),
        })
    }

    fn proposal_step<'context>(
        &mut self,
        proposal: &mut Self::DraftState,
        last_token: u32,
        _: Self::Context<'context>,
    ) -> Result<eredu_core::SpeculativeProposalStep<Self::Logits>, Self::Error>
    where
        Self: 'context,
    {
        if proposal.preceding_token != last_token {
            return Err(M::invalid(
                "masked proposal prefix differs from the selected canvas",
            ));
        }
        let token = *proposal
            .tokens
            .get(proposal.cursor)
            .ok_or_else(|| M::invalid("masked proposal capacity is exhausted"))?;
        proposal.cursor += 1;
        proposal.preceding_token = token;
        Ok(eredu_core::SpeculativeProposalStep::TargetMatchingToken(
            token,
        ))
    }

    fn checkpoint(&self, cache: &Self::Cache) -> Result<Self::CacheCheckpoint, Self::Error> {
        M::checkpoint(cache)
    }

    fn restore_checkpoint<'context>(
        &mut self,
        cache: &mut Self::Cache,
        saved: &Self::CacheCheckpoint,
        context: Self::Context<'context>,
    ) -> Result<(), Self::Error>
    where
        Self: 'context,
    {
        *cache = M::restore(saved, context)?;
        Ok(())
    }

    fn submit_verification<'context>(
        &mut self,
        tokens: &[u32],
        cache: &mut Self::Cache,
        context: Self::Context<'context>,
    ) -> Result<Submission<Self::Verification, Self::Completion>, Self::Error>
    where
        Self: 'context,
    {
        if tokens.is_empty() {
            return Err(M::invalid("invalid shared-backbone verification width"));
        }
        #[cfg(feature = "tensor-capture")]
        {
            self.execution_started = true;
        }
        let output = execution::decode_observed::<M>(
            self.model,
            tokens,
            cache,
            AutoregressivePass::Verification,
            context,
            #[cfg(feature = "tensor-capture")]
            self.observer.as_mut().map(|observer| {
                &mut **observer as &mut dyn crate::inspection::SpeculativeActivationObserver<_, _>
            }),
            #[cfg(not(feature = "tensor-capture"))]
            None,
        )?;
        let completion = M::completion(&output, context)?;
        Ok(Submission {
            output: CausalVerification::new(output, tokens.to_vec()),
            completion,
        })
    }

    fn verification_logits<'context>(
        &self,
        output: &Self::Verification,
        index: usize,
        context: Self::Context<'context>,
    ) -> Result<Self::Logits, Self::Error>
    where
        Self: 'context,
    {
        causal_verification::logits::<M>(output, index, context)
    }

    fn commit_verification<'context>(
        &mut self,
        output: Self::Verification,
        draft: Self::DraftState,
        cache: &mut Self::Cache,
        saved: &Self::CacheCheckpoint,
        verified: usize,
        context: Self::Context<'context>,
    ) -> Result<SpeculativeCommit<Self::TargetState>, Self::Error>
    where
        Self: 'context,
    {
        if verified == 0 || verified > output.inputs.len() {
            return Err(M::invalid(
                "invalid shared-backbone verification commitment",
            ));
        }
        let replayed = if verified < output.inputs.len()
            && !M::retain_verified_prefix(cache, saved, verified, context)?
        {
            // Build the accepted prefix in isolation before replacing installed state.
            let mut accepted = M::restore(saved, context)?;
            execution::decode_observed::<M>(
                self.model,
                &output.inputs[..verified],
                &mut accepted,
                AutoregressivePass::TargetCommit,
                context,
                #[cfg(feature = "tensor-capture")]
                self.observer.as_mut().map(|observer| {
                    &mut **observer
                        as &mut dyn crate::inspection::SpeculativeActivationObserver<_, _>
                }),
                #[cfg(not(feature = "tensor-capture"))]
                None,
            )?;
            *cache = accepted;
            verified
        } else {
            0
        };
        Ok(SpeculativeCommit::new(
            SharedBackboneSeed {
                prefix: M::checkpoint(cache)?,
                randomness: draft.randomness,
                refinement_randomness: draft.refinement_randomness,
                options: draft.options,
                #[cfg(feature = "tensor-capture")]
                activations: None,
                refinement: None,
            },
            replayed,
        ))
    }

    fn trim_parameter_conversions(
        &mut self,
    ) -> Result<
        eredu_core::residency::ExecutionConversionRetentionTrimReport,
        eredu_core::residency::ParameterConversionTrimError,
    > {
        Ok(
            eredu_core::residency::ExecutionConversionRetentionTrimReport {
                target: M::trim_parameter_conversions(self.model)?,
                external_drafter: None,
            },
        )
    }
    fn parameter_conversion_retention(
        &self,
    ) -> Result<
        Option<eredu_core::residency::ExecutionConversionRetentionReport>,
        eredu_core::BackendFailure,
    > {
        Ok(Some(
            eredu_core::residency::ExecutionConversionRetentionReport {
                target: M::parameter_conversion_retention(self.model)?,
                external_drafter: None,
            },
        ))
    }
    fn continuation_memory_observation(
        &self,
        cache: &Self::Cache,
        seed: &Self::TargetState,
        additional_positions: u64,
    ) -> Result<
        Option<eredu_core::speculative::SpeculativeContinuationObservation>,
        eredu_core::BackendFailure,
    > {
        Ok(
            M::memory_observation(self.model, cache, additional_positions)?.map(|target| {
                eredu_core::speculative::SpeculativeContinuationObservation {
                    target,
                    draft: None,
                    embedded: None,
                    parameter_conversions: None,
                    seed_bytes: M::checkpoint_retained_bytes(&seed.prefix).and_then(|bytes| {
                        bytes
                            .checked_add(M::proposal_randomness_bytes(&seed.randomness)?)?
                            .checked_add(M::proposal_randomness_bytes(
                                &seed.refinement_randomness,
                            )?)?
                            .checked_add(std::mem::size_of_val(&seed.options) as u64)?
                            .checked_add(seed.refinement.as_ref().map_or(Some(0), |state| {
                                M::refinement_state_estimate(state)
                                    .map(|estimate| estimate.retained_bytes)
                            })?)
                    }),
                }
            }),
        )
    }
    fn control_snapshot_estimate(
        &self,
        cache: &Self::Cache,
        seed: &Self::TargetState,
    ) -> Option<eredu_core::execution_control::SnapshotEstimate> {
        #[cfg(feature = "tensor-capture")]
        let authority = self
            .observer
            .as_ref()
            .map_or(Some(0), |observer| observer.activation_checkpoint_bytes())?;
        #[cfg(not(feature = "tensor-capture"))]
        let authority = 0;
        let cache = M::estimate_state(cache)?;
        let randomness = M::proposal_randomness_bytes(&seed.randomness)?
            .checked_add(M::proposal_randomness_bytes(&seed.refinement_randomness)?)?;
        let options = std::mem::size_of_val(&seed.options) as u64;
        let refinement = seed.refinement.as_ref().map_or(
            Some(eredu_core::execution_control::SnapshotEstimate {
                retained_bytes: 0,
                copy_bytes: 0,
            }),
            M::refinement_state_estimate,
        )?;
        let seed = M::estimate(&seed.prefix)?;
        Some(eredu_core::execution_control::SnapshotEstimate {
            retained_bytes: cache
                .retained_bytes
                .checked_add(seed.retained_bytes)?
                .checked_add(randomness)?
                .checked_add(options)?
                .checked_add(authority)?
                .checked_add(refinement.retained_bytes)?,
            copy_bytes: cache
                .copy_bytes
                .checked_add(seed.copy_bytes)?
                .checked_add(randomness)?
                .checked_add(options)?
                .checked_add(authority)?
                .checked_add(refinement.copy_bytes)?,
        })
    }
    fn control_snapshot<'context>(
        &self,
        cache: &Self::Cache,
        seed: &Self::TargetState,
        _: Self::Context<'context>,
    ) -> Result<
        Option<(Self::CacheCheckpoint, Self::TargetState)>,
        eredu_core::speculative::SpeculativeControlError,
    >
    where
        Self: 'context,
    {
        let mut seed = seed.clone();
        #[cfg(feature = "tensor-capture")]
        {
            seed.activations = self
                .observer
                .as_ref()
                .map(|observer| observer.activation_checkpoint())
                .transpose()?;
        }
        Ok(Some((
            M::checkpoint(cache)
                .map_err(eredu_core::speculative::SpeculativeControlError::backend)?,
            seed,
        )))
    }
    fn restore_control_snapshot<'context>(
        &mut self,
        cache: &mut Self::Cache,
        saved: &Self::CacheCheckpoint,
        seed: &Self::TargetState,
        context: Self::Context<'context>,
    ) -> Result<Option<Self::TargetState>, eredu_core::speculative::SpeculativeControlError>
    where
        Self: 'context,
    {
        #[cfg(feature = "tensor-capture")]
        let prepared = match (self.observer.as_mut(), seed.activations.as_ref()) {
            (Some(observer), Some(saved)) => Some(observer.prepare_activation_restore(saved)?),
            (None, None) => None,
            _ => {
                return Err(eredu_core::speculative::SpeculativeControlError::Invalid(
                    "shared activation snapshot authority changed",
                ));
            }
        };
        let replacement = M::restore(saved, context)
            .map_err(eredu_core::speculative::SpeculativeControlError::backend)?;
        *cache = replacement;
        #[cfg(feature = "tensor-capture")]
        if let Some(prepared) = prepared {
            prepared.commit();
        }
        let mut seed = seed.clone();
        #[cfg(feature = "tensor-capture")]
        {
            seed.activations = None;
        }
        Ok(Some(seed))
    }
}
