//! MLX execution primitives for speculative model sessions.

/// MLX executor for checkpoint-embedded prediction heads.
pub mod embedded;
/// External assistant adapters over neutral family equations and state.
pub mod external;
/// MLX semantic-generation resource adapter over the portable runtime driver.
pub mod scheduler;

pub use scheduler::MtpComponentTimingGuard;

use eredu_core::{
    Completion, ProposalDecision, SamplingPlacement, SpeculativeExecutionTopology,
    SpeculativeRandomness, SpeculativeSampling, TokenizerCompatibilityProof,
};
use safemlx::{
    error::Exception,
    ops::{indexing::TryIndexOp, maximum, softmax_axis},
    random::{self, RandomState},
    transforms::{async_eval_with_event, eval},
    Array, Event, Stream,
};

use crate::{
    backend::error::Error,
    backend::runtime::generation::sampler::SpeculativeSampler,
    backend::ModelLoadOptions,
    composition::gemma4::{load_assistant_gguf, load_assistant_safetensors, Gemma4AssistantModel},
    composition::mlx::ModelCache,
    composition::muse_glimmer::{
        load_dflash_gguf, load_dflash_safetensors, MuseGlimmerDFlashModel,
    },
};

/// Architecture-dispatched MLX draft model with its fixed execution placement.
pub struct MlxDrafter {
    model: MlxDrafterModel,
    tokenizer_compatibility: TokenizerCompatibilityProof,
    stream: Stream,
}

enum MlxDrafterModel {
    Gemma4(Box<Gemma4AssistantModel>),
    MuseGlimmerDFlash(Box<MuseGlimmerDFlashModel>),
}

/// Stable architecture identity for an independently loaded MLX draft model.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum MlxDrafterKind {
    /// Gemma 4 external assistant.
    Gemma4Assistant,
    /// Muse-Glimmer anchor-plus-15-mask DFlash assistant.
    MuseGlimmerDFlash,
}

/// Adapter-owned target caches for independently progressing prepared-chat lanes.
pub struct MlxMtpCache {
    pub lanes: Vec<ModelCache>,
}

impl MlxMtpCache {
    pub fn new(lanes: Vec<ModelCache>) -> Self {
        Self { lanes }
    }

    pub fn len(&self) -> usize {
        self.lanes.len()
    }
}

impl MlxDrafter {
    /// Materializes an architecture-inspected drafter with proven tokenizer compatibility.
    pub(crate) fn materialize_with_compatibility(
        preparation: eredu_architectures::ExternalAssistantPreparationPlan,
        tokenizer_compatibility: TokenizerCompatibilityProof,
        options: ModelLoadOptions,
        stream: &Stream,
        weights_stream: &Stream,
    ) -> Result<Self, Error> {
        use eredu_architectures::{ExternalAssistantCheckpoint, ExternalAssistantPreparationPlan};
        let model = match preparation {
            ExternalAssistantPreparationPlan::Gemma4(plan) => {
                let (checkpoint, config) = plan.into_parts();
                let model = match checkpoint {
                    ExternalAssistantCheckpoint::SafeTensors { source } => {
                        load_assistant_safetensors(
                            &source,
                            config,
                            options,
                            stream,
                            weights_stream,
                        )?
                    }
                    ExternalAssistantCheckpoint::Gguf {
                        checkpoint,
                        resolution,
                    } => load_assistant_gguf(
                        checkpoint,
                        resolution,
                        config,
                        options,
                        stream,
                        weights_stream,
                    )?,
                };
                MlxDrafterModel::Gemma4(Box::new(model))
            }
            ExternalAssistantPreparationPlan::MuseGlimmer(plan) => {
                let (checkpoint, config) = plan.into_parts();
                let model = match checkpoint {
                    ExternalAssistantCheckpoint::SafeTensors { source } => {
                        load_dflash_safetensors(&source, config, options, stream, weights_stream)?
                    }
                    ExternalAssistantCheckpoint::Gguf {
                        checkpoint,
                        resolution,
                    } => load_dflash_gguf(
                        checkpoint,
                        resolution,
                        config,
                        options,
                        stream,
                        weights_stream,
                    )?,
                };
                MlxDrafterModel::MuseGlimmerDFlash(Box::new(model))
            }
        };
        Ok(Self {
            model,
            tokenizer_compatibility,
            stream: stream.clone(),
        })
    }

    /// Architecture fixed by the inspected preparation plan.
    pub(crate) const fn kind(&self) -> MlxDrafterKind {
        match self.model {
            MlxDrafterModel::Gemma4(_) => MlxDrafterKind::Gemma4Assistant,
            MlxDrafterModel::MuseGlimmerDFlash(_) => MlxDrafterKind::MuseGlimmerDFlash,
        }
    }

    pub(crate) fn gemma4(&self) -> &Gemma4AssistantModel {
        match &self.model {
            MlxDrafterModel::Gemma4(model) => model,
            MlxDrafterModel::MuseGlimmerDFlash(_) => {
                panic!("requested Gemma 4 assistant from Muse-Glimmer DFlash drafter")
            }
        }
    }

    pub(crate) fn gemma4_mut(&mut self) -> &mut Gemma4AssistantModel {
        match &mut self.model {
            MlxDrafterModel::Gemma4(model) => model,
            MlxDrafterModel::MuseGlimmerDFlash(_) => {
                panic!("requested Gemma 4 assistant from Muse-Glimmer DFlash drafter")
            }
        }
    }

    pub(crate) fn muse_glimmer(&self) -> &MuseGlimmerDFlashModel {
        match &self.model {
            MlxDrafterModel::MuseGlimmerDFlash(model) => model,
            MlxDrafterModel::Gemma4(_) => {
                panic!("requested Muse-Glimmer DFlash from Gemma 4 assistant")
            }
        }
    }

    pub(crate) fn muse_glimmer_mut(&mut self) -> &mut MuseGlimmerDFlashModel {
        match &mut self.model {
            MlxDrafterModel::MuseGlimmerDFlash(model) => model,
            MlxDrafterModel::Gemma4(_) => {
                panic!("requested Muse-Glimmer DFlash from Gemma 4 assistant")
            }
        }
    }

    /// Returns the portable proof established before this assistant was materialized.
    pub const fn tokenizer_compatibility(&self) -> TokenizerCompatibilityProof {
        self.tokenizer_compatibility
    }

    /// Execution stream selected when this drafter was loaded.
    pub const fn stream(&self) -> &Stream {
        &self.stream
    }
}

/// Target and assistant streams assigned to one speculative session.
#[derive(Debug, Clone, Copy)]
pub struct MtpExecutionStreams<'a> {
    target: &'a Stream,
    draft: &'a Stream,
    topology: SpeculativeExecutionTopology,
}

impl<'a> MtpExecutionStreams<'a> {
    /// Creates an execution assignment and classifies its device topology.
    pub fn new(target: &'a Stream, draft: &'a Stream) -> Result<Self, Exception> {
        let topology = if target == draft {
            SpeculativeExecutionTopology::Single
        } else if target.get_device()? == draft.get_device()? {
            SpeculativeExecutionTopology::SameDeviceSplit
        } else {
            SpeculativeExecutionTopology::CrossDeviceSplit
        };
        Ok(Self {
            target,
            draft,
            topology,
        })
    }

    /// Creates an assignment in which all speculative work uses one stream.
    pub const fn single(stream: &'a Stream) -> Self {
        Self {
            target: stream,
            draft: stream,
            topology: SpeculativeExecutionTopology::Single,
        }
    }

    /// Stream used for target prefill and verification.
    pub const fn target(self) -> &'a Stream {
        self.target
    }

    /// Stream used for proposal generation.
    pub const fn draft(self) -> &'a Stream {
        self.draft
    }

    /// Relationship between the target and assistant streams.
    pub const fn topology(self) -> SpeculativeExecutionTopology {
        self.topology
    }

    /// Whether target and assistant work use different streams.
    pub const fn is_split(self) -> bool {
        !matches!(self.topology, SpeculativeExecutionTopology::Single)
    }

    /// Whether values must be physically transferred between devices.
    pub const fn crosses_devices(self) -> bool {
        matches!(
            self.topology,
            SpeculativeExecutionTopology::CrossDeviceSplit
        )
    }

    /// Submits target outputs and orders subsequent assistant work after them.
    pub fn wait_for_target_outputs<'b>(
        self,
        outputs: impl IntoIterator<Item = &'b Array>,
    ) -> Result<Event, Exception> {
        self.wait_for_same_device_outputs(outputs, self.draft, "target-to-draft")
    }

    /// Submits assistant outputs and orders subsequent target work after them.
    pub fn wait_for_draft_outputs<'b>(
        self,
        outputs: impl IntoIterator<Item = &'b Array>,
    ) -> Result<Event, Exception> {
        self.wait_for_same_device_outputs(outputs, self.target, "draft-to-target")
    }

    fn wait_for_same_device_outputs<'b>(
        self,
        outputs: impl IntoIterator<Item = &'b Array>,
        consumer: &Stream,
        direction: &str,
    ) -> Result<Event, Exception> {
        if self.topology != SpeculativeExecutionTopology::SameDeviceSplit {
            return Err(Exception::custom(format!(
                "MTP {direction} event handoff requires distinct streams on one device, got {}",
                self.topology
            )));
        }
        let completion = async_eval_with_event(outputs)?;
        completion.wait_on(consumer)?;
        Ok(completion)
    }
}

/// Exact completion for one retained MLX speculative verification.
pub struct MlxSpeculativeCompletion {
    event: Event,
    _retained: Vec<Array>,
}

impl MlxSpeculativeCompletion {
    /// Submits all retained verification outputs as one exact completion.
    pub fn submit<'a>(outputs: impl IntoIterator<Item = &'a Array>) -> Result<Self, Exception> {
        let retained = outputs.into_iter().cloned().collect::<Vec<_>>();
        let event = async_eval_with_event(retained.iter())?;
        Ok(Self {
            event,
            _retained: retained,
        })
    }
}

impl Completion for MlxSpeculativeCompletion {
    type Error = Exception;

    fn is_complete(&self) -> Result<bool, Self::Error> {
        self.event.is_complete()
    }

    fn wait(&self) -> Result<(), Self::Error> {
        self.event.synchronize()
    }
}

impl Drop for MlxSpeculativeCompletion {
    fn drop(&mut self) {
        match self.event.is_complete() {
            Ok(true) => {}
            Ok(false) | Err(_) => {
                let _ = self.event.synchronize();
            }
        }
    }
}

/// MLX implementation of opaque speculative sampling operations.
#[derive(Clone)]
pub struct MlxSpeculativeSampling<S> {
    inner: S,
}

impl<S> MlxSpeculativeSampling<S> {
    /// Wraps one facade sampling policy for backend-owned execution.
    pub const fn new(inner: S) -> Self {
        Self { inner }
    }

    /// Returns the public sampling policy after generation completes.
    pub fn into_inner(self) -> S {
        self.inner
    }

    #[cfg(test)]
    pub const fn inner(&self) -> &S {
        &self.inner
    }
}

impl<S> SpeculativeSampling for MlxSpeculativeSampling<S>
where
    S: SpeculativeSampler + Clone,
{
    type Logits = Array;
    type Distribution = Array;
    type Seed = Array;
    type RandomState = RandomState;
    type DraftRandomness = Array;
    type Context<'a>
        = MtpExecutionStreams<'a>
    where
        Self: 'a;
    type Error = Exception;

    fn supports_exact_optimistic_promotion(&self) -> bool {
        self.inner.supports_exact_optimistic_promotion()
    }

    fn grammar_is_complete(&mut self) -> Result<bool, Self::Error> {
        self.inner.grammar_is_complete()
    }

    fn prefix_is_complete(&self, history: &[u32]) -> Result<bool, Self::Error> {
        self.inner.prefix_is_complete(history)
    }

    fn initialize_randomness<'a>(
        seed: Option<Self::Seed>,
        temperature: f32,
        context: Self::Context<'a>,
    ) -> Result<SpeculativeRandomness<Self::RandomState, Self::DraftRandomness>, Self::Error>
    where
        Self: 'a,
    {
        if temperature == 0.0 {
            return Ok(SpeculativeRandomness {
                target: None,
                draft: None,
            });
        }
        let mut root =
            RandomState::from_key(seed.ok_or_else(|| {
                Exception::custom("random operations require an explicit PRNG key")
            })?);
        let target_key = root.next_key(context.target())?;
        let draft_key = root.next_key(context.target())?;
        let draft_key = if context.is_split() {
            if context.crosses_devices() {
                async_eval_with_event([&draft_key])?.synchronize()?;
                let copied = draft_key.copy(context.draft())?;
                async_eval_with_event([&copied])?.synchronize()?;
                copied
            } else {
                let _completion = context.wait_for_target_outputs([&draft_key])?;
                draft_key
            }
        } else {
            draft_key
        };
        Ok(SpeculativeRandomness {
            target: Some(RandomState::from_key(target_key)),
            draft: Some(draft_key),
        })
    }

    fn draft_randomness_at<'a>(
        root: &Self::DraftRandomness,
        position: usize,
        context: Self::Context<'a>,
    ) -> Result<Self::RandomState, Self::Error>
    where
        Self: 'a,
    {
        Ok(RandomState::from_key(random::split_key_at(
            root,
            position,
            context.draft(),
        )?))
    }

    fn process_logits<'a>(
        &mut self,
        logits: &Self::Logits,
        temperature: f32,
        history: &[u32],
        placement: SamplingPlacement,
        context: Self::Context<'a>,
    ) -> Result<Self::Distribution, Self::Error>
    where
        Self: 'a,
    {
        self.inner.process_logits(
            logits,
            temperature,
            history,
            sampling_stream(placement, context),
        )
    }

    fn sample<'a>(
        &self,
        distribution: &Self::Distribution,
        temperature: f32,
        randomness: Option<&mut Self::RandomState>,
        placement: SamplingPlacement,
        context: Self::Context<'a>,
    ) -> Result<u32, Self::Error>
    where
        Self: 'a,
    {
        let stream = sampling_stream(placement, context);
        let token = self
            .inner
            .sample_processed(distribution, temperature, randomness, stream)?;
        eval([&token])?;
        Ok(token.item::<u32>(stream))
    }

    fn decide_proposal<'a>(
        &self,
        target: &Self::Distribution,
        draft: &Self::Distribution,
        proposed: u32,
        temperature: f32,
        randomness: Option<&mut Self::RandomState>,
        context: Self::Context<'a>,
    ) -> Result<ProposalDecision, Self::Error>
    where
        Self: 'a,
    {
        let stream = context.target();
        if temperature == 0.0 {
            let chosen = self
                .inner
                .sample_processed(target, 0.0, None, stream)?
                .item::<u32>(stream);
            return Ok(if chosen == proposed {
                ProposalDecision::Accept
            } else {
                ProposalDecision::Reject(chosen)
            });
        }

        let target_probabilities = probabilities(target, stream)?;
        let draft_probabilities = probabilities(draft, stream)?;
        let target_probability = probability_at(&target_probabilities, proposed, stream)?;
        let draft_probability = probability_at(&draft_probabilities, proposed, stream)?;
        let acceptance = if draft_probability <= 0.0 {
            1.0
        } else {
            (target_probability / draft_probability).min(1.0)
        };
        let mut randomness = randomness;
        if uniform(randomness.as_deref_mut(), stream)? <= acceptance {
            return Ok(ProposalDecision::Accept);
        }
        let residual = maximum(
            target_probabilities.subtract(&draft_probabilities, stream)?,
            Array::from_f32(0.0),
            stream,
        )?;
        let mass = residual.sum(None, stream)?.item::<f32>(stream);
        let logits = if mass <= f32::EPSILON {
            target.clone()
        } else {
            residual.log(stream)?
        };
        let replacement = self
            .inner
            .sample_processed(&logits, temperature, randomness, stream)?
            .item::<u32>(stream);
        Ok(ProposalDecision::Reject(replacement))
    }

    fn commit_token<'a>(
        &mut self,
        distribution: &Self::Distribution,
        token: u32,
        placement: SamplingPlacement,
        context: Self::Context<'a>,
    ) -> Result<(), Self::Error>
    where
        Self: 'a,
    {
        self.inner
            .commit_token(distribution, token, sampling_stream(placement, context))
    }

    fn prepare_verification<'a>(
        &self,
        distributions: &mut [&mut Self::Distribution],
        temperature: f32,
        context: Self::Context<'a>,
    ) -> Result<(), Self::Error>
    where
        Self: 'a,
    {
        if temperature == 0.0 || !context.is_split() {
            return Ok(());
        }
        if context.crosses_devices() {
            async_eval_with_event(distributions.iter().map(|distribution| &**distribution))?
                .synchronize()?;
            for distribution in distributions {
                **distribution = distribution.copy(context.target())?;
            }
        } else {
            let _completion = context
                .wait_for_draft_outputs(distributions.iter().map(|distribution| &**distribution))?;
        }
        Ok(())
    }
}

fn sampling_stream<'a>(
    placement: SamplingPlacement,
    context: MtpExecutionStreams<'a>,
) -> &'a Stream {
    match placement {
        SamplingPlacement::Target => context.target(),
        SamplingPlacement::Draft => context.draft(),
    }
}

fn probabilities(logits: &Array, stream: &Stream) -> Result<Array, Exception> {
    softmax_axis(&logits.as_type::<f32>(stream)?, -1, true, stream)
}

fn probability_at(probabilities: &Array, token: u32, stream: &Stream) -> Result<f32, Exception> {
    if token as i32 >= probabilities.dim(-1) {
        return Err(Exception::custom(format!(
            "sampled token {token} exceeds vocabulary size {}",
            probabilities.dim(-1)
        )));
    }
    let value = match probabilities.ndim() {
        2 => probabilities.try_index_device((0, token as i32), stream)?,
        3 => probabilities.try_index_device((0, 0, token as i32), stream)?,
        ndim => {
            return Err(Exception::custom(format!(
                "speculative distribution must be rank 2 or 3, got rank {ndim}"
            )))
        }
    };
    Ok(value.item::<f32>(stream))
}

fn uniform(state: Option<&mut RandomState>, stream: &Stream) -> Result<f32, Exception> {
    let state = state.ok_or_else(|| Exception::custom("stochastic MTP requires a PRNG key"))?;
    let key = state.next_key(stream)?;
    Ok(random::uniform::<_, f32>(0.0, 1.0, &[1], &key, stream)?.item::<f32>(stream))
}
