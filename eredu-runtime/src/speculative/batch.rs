//! Portable assembly and delivery of admitted speculative lanes.

use eredu_core::{
    execution_control::GenerationControlHandle,
    generation::{
        GenerationCancellationToken, GenerationError, GenerationRequestConfig, SemanticEvent,
    },
    run_preparation::{finish_preparation, TextPreparationStage},
    speculative::{GenerationCache, SpeculativeCacheFactory, SpeculativeDriverError},
    GenerationSequence, PreparedSpeculativeLane, SpeculativeCallbackPublisher,
    SpeculativeExecutionTopology, SpeculativeGenerationBatchOutput, SpeculativeGenerationLane,
    SpeculativeGenerationVisitor, SpeculativeOutputRuntime, SpeculativeSampling,
    SpeculativeSemanticConstraint, SpeculativeSemanticState, SpeculativeTokenFilterController,
    TextGenerationConfig,
};

/// One admitted prompt and its separately owned sampling and delivery state.
pub struct SpeculativeBatchLane<'a, I, S, Seed> {
    /// Prepared model input.
    pub input: I,
    /// Admitted generation and scheduler policy.
    pub config: GenerationRequestConfig,
    /// Native root for stochastic sampling.
    pub prng_key: Option<Seed>,
    /// Seed for model-owned refinement randomness.
    pub refinement_seed: u64,
    /// Native sampling operations.
    pub sampler: S,
    /// Per-lane semantic state.
    pub semantic: Box<dyn SpeculativeSemanticState>,
    /// Cancellation authority for this lane.
    pub cancellation: GenerationCancellationToken,
    /// Delivers committed events.
    pub on_event: Box<dyn FnMut(SemanticEvent) + 'a>,
    /// Prospective commands retained for execution.
    pub control_handle: Option<GenerationControlHandle>,
}

impl<'a, I, S, Seed> SpeculativeBatchLane<'a, I, S, Seed> {
    /// Replaces the model input while preserving every other lane owner.
    pub fn map_input<J, E>(
        self,
        prepare: impl FnOnce(I, u64) -> Result<J, E>,
    ) -> Result<SpeculativeBatchLane<'a, J, S, Seed>, E> {
        Ok(SpeculativeBatchLane {
            input: prepare(self.input, self.refinement_seed)?,
            config: self.config,
            prng_key: self.prng_key,
            refinement_seed: self.refinement_seed,
            sampler: self.sampler,
            semantic: self.semantic,
            cancellation: self.cancellation,
            on_event: self.on_event,
            control_handle: self.control_handle,
        })
    }
}

/// Validates proposal capacity against the installed route, including target-only execution.
pub fn validate_lane_proposal_capacity(
    config: &GenerationRequestConfig,
    capacity: usize,
) -> Result<(), GenerationError> {
    let requested = config.proposal_capacity();
    if capacity == 0 && requested > 0 {
        return Err(GenerationError::NoBackendDraftCapacity);
    }
    if requested > capacity {
        return Err(GenerationError::DraftCapacityExceeded {
            requested,
            capacity,
        });
    }
    Ok(())
}

/// Checks every lane before preparing any native sampler for the batch.
pub fn prepare_speculative_batch_lanes<'a, P, C, S, Seed, E>(
    lanes: Vec<SpeculativeGenerationLane<'a, P, C>>,
    capacity: usize,
    prepare_sampling: impl Fn(TextGenerationConfig, C) -> Result<(Option<Seed>, S), E>,
) -> Result<Vec<SpeculativeBatchLane<'a, P, S, Seed>>, E>
where
    C: SpeculativeTokenFilterController,
    E: From<GenerationError>,
{
    for lane in &lanes {
        validate_lane_proposal_capacity(lane.config(), capacity)?;
    }
    lanes
        .into_iter()
        .map(|lane| prepare_batch_lane(lane, |input, _, _| Ok(input), &prepare_sampling))
        .collect()
}

pub(super) fn prepare_batch_lane<'a, P, C, I, S, Seed, E>(
    mut lane: SpeculativeGenerationLane<'a, P, C>,
    prepare_input: impl FnOnce(P, &GenerationRequestConfig, &TextGenerationConfig) -> Result<I, E>,
    prepare_sampling: impl FnOnce(TextGenerationConfig, C) -> Result<(Option<Seed>, S), E>,
) -> Result<SpeculativeBatchLane<'a, I, S, Seed>, E>
where
    C: SpeculativeTokenFilterController,
{
    let generation = lane.take_generation();
    let config = lane.take_config();
    let input = prepare_input(lane.take_prompt(), &config, &generation)?;
    let refinement_seed = generation.seed();
    let (prng_key, sampler) = prepare_sampling(generation, lane.take_constraint())?;
    Ok(SpeculativeBatchLane {
        input,
        config,
        prng_key,
        refinement_seed,
        sampler,
        semantic: lane.take_semantic(),
        cancellation: lane.take_cancellation(),
        on_event: lane.take_on_event(),
        control_handle: lane.take_control_handle(),
    })
}

/// Assembles output and randomness owners, then starts the scheduler after peer agreement.
pub fn run_speculative_batch<'run, 'lane, E, S, Binding>(
    executor: &'run mut E,
    lanes: Vec<SpeculativeBatchLane<'lane, E::Input, S, S::Seed>>,
    binding: &Binding,
    context: E::Context<'run>,
    topology: SpeculativeExecutionTopology,
    collect_component_timings: bool,
    visitor: impl SpeculativeGenerationVisitor,
) -> Result<SpeculativeGenerationBatchOutput, SpeculativeDriverError<E::Error>>
where
    'lane: 'run,
    E: SpeculativeCacheFactory<Binding> + 'run,
    S: SpeculativeSampling<Logits = E::Logits, Error = E::Error, Context<'run> = E::Context<'run>>
        + 'run,
{
    let caches = (0..lanes.len())
        .map(|_| executor.create_cache(binding, context))
        .collect::<Result<Vec<_>, _>>()
        .map_err(SpeculativeDriverError::Backend);
    let stage = TextPreparationStage::Sampling;
    let caches = finish_preparation(
        stage,
        caches,
        |status| executor.agree_text_preparation(stage, status, context),
        SpeculativeDriverError::Preparation,
    )?;
    let prepared = (|| {
        let verification = executor.verification_policy();
        lanes
            .into_iter()
            .zip(caches)
            .map(|(lane, cache)| {
                let randomness =
                    S::initialize_randomness(lane.prng_key, lane.config.temperature, context)?
                        .with_verification_policy(verification);
                let sequence = GenerationSequence::new(
                    lane.config.max_tokens,
                    lane.config.eos_token_ids.iter().copied(),
                );
                Ok(PreparedSpeculativeLane::new(
                    GenerationCache::Owned(cache),
                    lane.input,
                    lane.config,
                    SpeculativeOutputRuntime::new(
                        lane.sampler,
                        sequence,
                        SpeculativeSemanticConstraint::semantic(lane.semantic),
                        SpeculativeCallbackPublisher::semantic(lane.on_event),
                        lane.cancellation,
                    )
                    .with_control_handle(lane.control_handle),
                    randomness,
                ))
            })
            .collect::<Result<Vec<_>, _>>()
    })();
    let stage = TextPreparationStage::Delivery;
    let prepared = finish_preparation(
        stage,
        prepared,
        |status| executor.agree_text_preparation(stage, status, context),
        SpeculativeDriverError::Preparation,
    )?;
    let component_timings_collected = collect_component_timings && executor.supports_telemetry();
    visitor.run(
        executor,
        prepared,
        topology,
        !matches!(topology, SpeculativeExecutionTopology::Single),
        component_timings_collected,
        context,
    )
}
