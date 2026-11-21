//! Portable assembly of ordinary, independent-draft, and shared-backbone batches.

use crate::capture::SpeculativeCaptureMechanisms;
use std::num::NonZeroUsize;

use eredu_core::{
    decoding::DecodingModeError,
    generation::GenerationError,
    run_preparation::{finish_preparation, TextPreparationStage},
    speculative::GenerationExecutorTypes,
    SpeculativeDriverError, SpeculativeGenerationLane, SpeculativeTokenFilterController,
    TextGenerationBackend, TextGenerationConfig,
};

use super::{
    AutoregressiveExecutor, AutoregressiveInput, AutoregressiveMechanisms, CausalExecutor,
    SharedTargetRefinement, SharedTargetRefinementState,
};
use crate::executable_binding::{ErasedTextExecutable, InstalledTextExecutable};
use crate::speculative::{
    batch::prepare_batch_lane,
    shared_backbone::{
        SharedBackboneDraftOptions, SharedBackboneExecutor, SharedBackboneInput,
        SharedBackboneMechanisms, SharedRefinementState,
    },
    SpeculativeBatchLane,
};
use eredu_core::backend::TextExecutableTypes;

type RefinedInput<M> = AutoregressiveInput<
    <M as AutoregressiveMechanisms>::Input,
    SharedTargetRefinementState<
        <M as SharedBackboneMechanisms>::ProposalRandomness,
        SharedRefinementState<M>,
    >,
>;
type RefinedLanes<'a, M, S, Seed> = Vec<SpeculativeBatchLane<'a, RefinedInput<M>, S, Seed>>;

/// Composes the selected ordinary target and initializes its refinement randomness.
fn prepare_target_batch<'model, 'lane, M, S, Seed>(
    target: &'model mut M::Model,
    lanes: Vec<SpeculativeBatchLane<'lane, M::Input, S, Seed>>,
    context: M::Context<'_>,
) -> Result<
    (
        CausalExecutor<'model, M, SharedTargetRefinement>,
        RefinedLanes<'lane, M, S, Seed>,
    ),
    SpeculativeDriverError<M::Error>,
>
where
    M: SharedBackboneMechanisms,
{
    let substream =
        M::ordinary_refinement_policy(target).and_then(|policy| policy.randomness_substream());
    let lanes = prepare_refinement_lanes::<M, S, Seed>(lanes, substream, context)?;
    Ok((CausalExecutor::new(target, SharedTargetRefinement), lanes))
}

/// Composes the selected ordinary target and independent decoder.
pub(super) fn prepare_independent_batch<'model, 'lane, M, S, Seed>(
    target: &'model mut M::Model,
    draft: &'model mut M::Model,
    capacity: NonZeroUsize,
    lanes: Vec<SpeculativeBatchLane<'lane, M::Input, S, Seed>>,
    context: M::Context<'_>,
) -> Result<
    (
        AutoregressiveExecutor<'model, M, SharedTargetRefinement>,
        RefinedLanes<'lane, M, S, Seed>,
    ),
    SpeculativeDriverError<M::Error>,
>
where
    M: SharedBackboneMechanisms,
{
    // Independent drafting retains separate target, proposal, and refinement substreams.
    let lanes = prepare_refinement_lanes::<M, S, Seed>(lanes, Some(2), context)?;
    Ok((
        AutoregressiveExecutor::new(target, draft, capacity, SharedTargetRefinement),
        lanes,
    ))
}

fn prepare_refinement_lanes<'lane, M, S, Seed>(
    lanes: Vec<SpeculativeBatchLane<'lane, M::Input, S, Seed>>,
    substream: Option<usize>,
    context: M::Context<'_>,
) -> Result<RefinedLanes<'lane, M, S, Seed>, SpeculativeDriverError<M::Error>>
where
    M: SharedBackboneMechanisms,
{
    let local = lanes
        .into_iter()
        .map(|lane| {
            lane.map_input(|input, seed| {
                let root = M::model_randomness_root(seed)?;
                let randomness = match substream {
                    Some(index) => M::split_model_randomness(&root, index, context)?,
                    None => root,
                };
                M::settle_model_randomness(&[&randomness])?;
                Ok(AutoregressiveInput::new(
                    input,
                    SharedTargetRefinementState::new(randomness),
                ))
            })
        })
        .collect::<Result<Vec<_>, M::Error>>()
        .map_err(SpeculativeDriverError::Backend);
    let stage = TextPreparationStage::Sampling;
    finish_preparation(
        stage,
        local,
        |status| M::agree_text_preparation(stage, status, context),
        SpeculativeDriverError::Preparation,
    )
}

type SharedInput<M> = SharedBackboneInput<
    <M as AutoregressiveMechanisms>::Input,
    SharedBackboneDraftOptions,
    <M as SharedBackboneMechanisms>::ProposalRandomness,
>;

/// Composes a shared-backbone executor and its independent proposal and refinement streams.
fn prepare_shared_batch<'model, 'lane, M, B, C, S, Seed, E>(
    target: &'model mut M::Model,
    lanes: Vec<SpeculativeGenerationLane<'lane, crate::text_session_driver::TextPromptFor<B>, C>>,
    context: M::Context<'_>,
    prepare_sampling: impl Fn(TextGenerationConfig, C) -> Result<(Option<Seed>, S), E>,
) -> Result<
    (
        SharedBackboneExecutor<'model, M>,
        Vec<SpeculativeBatchLane<'lane, SharedInput<M>, S, Seed>>,
    ),
    E,
>
where
    B: TextGenerationBackend,
    C: SpeculativeTokenFilterController,
    M: SharedBackboneMechanisms<Input = crate::text_session_driver::TextPromptFor<B>>,
    E: From<GenerationError> + From<M::Error> + From<SpeculativeDriverError<M::Error>>,
{
    let local: Result<_, E> = (|| {
        let policy = M::shared_backbone_policy(target).ok_or_else(|| {
            GenerationError::DecodingMode(DecodingModeError::Unsupported(
                "selected execution has no shared-backbone draft declaration",
            ))
        })?;
        let prepared = lanes
            .into_iter()
            .map(|lane| {
                prepare_batch_lane(
                    lane,
                    |input, config, generation| {
                        let options = policy.prepare_lane(config, generation)?;
                        let root = M::model_randomness_root(generation.seed()).map_err(E::from)?;
                        let proposal =
                            M::split_model_randomness(&root, 1, context).map_err(E::from)?;
                        let refinement =
                            M::split_model_randomness(&root, 2, context).map_err(E::from)?;
                        M::settle_model_randomness(&[&proposal, &refinement]).map_err(E::from)?;
                        Ok(SharedBackboneInput::new(
                            input, options, proposal, refinement,
                        ))
                    },
                    &prepare_sampling,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok((policy.proposal_capacity(), prepared))
    })();
    let stage = TextPreparationStage::Sampling;
    let (capacity, prepared) = finish_preparation(
        stage,
        local,
        |status| M::agree_text_preparation(stage, status, context),
        |error| SpeculativeDriverError::<M::Error>::Preparation(error).into(),
    )?;
    Ok((SharedBackboneExecutor::new(target, capacity), prepared))
}

/// Portable driver installed for the batch's native resource contract.
pub type BatchSession<B, M> = crate::text_session_driver::TextSessionDriver<
    <<M as ResidentBatchMechanisms<B>>::SessionMechanisms as crate::text_session_driver::TextSessionMechanisms<B>>::Resources,
    <M as ResidentBatchMechanisms<B>>::SessionMechanisms,
>;

/// Native resources needed to execute a resident speculative batch.
pub trait ResidentBatchMechanisms<B>:
    for<'a> SharedBackboneMechanisms<
        Context<'a> = crate::speculative::SamplingContext<'a, Self::Sampling>,
        Input = crate::text_session_driver::TextPromptFor<B>,
        Model = dyn ErasedTextExecutable<Self::ExecutableTypes>,
        Capture: SpeculativeCaptureMechanisms<
            Transport = <Self::SessionMechanisms as eredu_core::backend::TextSessionNative<B>>::ControlTransport,
        >,
    > + 'static
where
    B: TextGenerationBackend,
{
    /// Native values carried by the installed executable during speculative execution.
    type ExecutableTypes: TextExecutableTypes<
        InputTensor = B::InputTensor,
        EmbeddedExecutor: for<'a> GenerationExecutorTypes<
            Logits = Self::Logits,
            Context<'a> = Self::Context<'a>,
            Error = Self::Error,
        >,
    >;
    /// Native authority and completion operations.
    type SessionMechanisms: crate::text_session_driver::TextSessionMechanisms<B, Model = InstalledTextExecutable<Self::ExecutableTypes>>
        + eredu_core::backend::TextTransactionMechanisms<Error = B::Error>;
    /// Native numerical primitives instantiated by the portable sampler.
    type Sampling: eredu_core::speculative::sampling::SpeculativeSamplingPrimitives<
        Value = Self::Logits,
        Error = Self::Error,
    >;

    /// Retains the selected target queue without acquiring a submission lease.
    fn target_queue(backend: &B) -> BatchQueue<Self>;

    /// Reports whether native component timing is enabled.
    fn collect_component_timings() -> bool;

    /// Materializes the key for an explicitly stochastic request.
    fn sampling_seed(
        seed: u64,
    ) -> Result<<Self::Sampling as eredu_core::speculative::sampling::SpeculativeSamplingPrimitives>::Seed, B::Error>;

    /// Builds portable sampling and constraints, allocating a key for stochastic requests.
    fn prepare_sampling<C: SpeculativeTokenFilterController + 'static>(
        config: TextGenerationConfig,
        constraint: C,
    ) -> Result<
        (
            Option<<Self::Sampling as eredu_core::speculative::sampling::SpeculativeSamplingPrimitives>::Seed>,
            BatchSampler<B, Self, C>,
        ),
        B::Error,
    >
    where
        B::Error: From<eredu_core::BackendFailure>,
    {
        let sampler = crate::generation::TextSampler::from_config(&config)
            .map_err(eredu_core::BackendFailure::from_error)?;
        let key = (config.sampling().temperature != 0.0)
            .then(|| Self::sampling_seed(config.seed()))
            .transpose()?;
        Ok((
            key,
            crate::speculative::InstalledSpeculativeSampling::new(crate::ConstrainedSampler::new(
                sampler, constraint,
            )),
        ))
    }
}

impl<B, M> ResidentBatchMechanisms<B> for M
where
    B: TextGenerationBackend,
    M: super::InstalledSpeculationPrimitives
        + eredu_core::speculative::installed::ResidentSpeculation<B>,
    M: for<'a> SharedBackboneMechanisms<
        Context<'a> = crate::speculative::SamplingContext<'a, M::Sampling>,
        Input = crate::text_session_driver::TextPromptFor<B>,
        Model = dyn ErasedTextExecutable<crate::executable_binding::PortableTextTypes<M::Text>>,
        Capture: SpeculativeCaptureMechanisms<
            Queue = <M::Text as eredu_core::backend::NativeTextValues>::Context,
            Transport = <M::Session as eredu_core::backend::TextSessionNative<B>>::ControlTransport,
        >,
    >,
    M::Text: eredu_core::backend::NativeTextValues<InputTensor = B::InputTensor>,
    <crate::executable_binding::PortableTextTypes<M::Text> as TextExecutableTypes>::EmbeddedExecutor:
        for<'a> GenerationExecutorTypes<
            Logits = M::Logits, Context<'a> = M::Context<'a>, Error = <M as AutoregressiveMechanisms>::Error,
        >,
    M::Session: crate::text_session_driver::TextSessionMechanisms<
        B, Model = InstalledTextExecutable<crate::executable_binding::PortableTextTypes<M::Text>>,
    >,
    M::Sampling: eredu_core::speculative::sampling::SpeculativeSamplingPrimitives<
        Value = M::Logits,
        Error = <M as AutoregressiveMechanisms>::Error,
    >,
    B::Error: From<<M::Sampling as eredu_core::speculative::sampling::SpeculativeSamplingPrimitives>::Error>,
{
    type ExecutableTypes = crate::executable_binding::PortableTextTypes<M::Text>;
    type SessionMechanisms = M::Session;
    type Sampling = M::Sampling;

    fn target_queue(backend: &B) -> BatchQueue<Self> {
        M::target_queue(backend)
    }

    fn collect_component_timings() -> bool {
        M::collect_component_timings()
    }

    fn sampling_seed(seed: u64) -> Result<<Self::Sampling as eredu_core::speculative::sampling::SpeculativeSamplingPrimitives>::Seed, B::Error> {
        <M::Sampling as eredu_core::speculative::sampling::SpeculativeSamplingPrimitives>::seed(seed)
            .map_err(Into::into)
    }
}

pub(super) type BatchSampler<B, M, C> = crate::speculative::InstalledSpeculativeSampling<
    crate::ConstrainedSampler<crate::generation::TextSampler, C>,
    <M as ResidentBatchMechanisms<B>>::Sampling,
>;

/// Capture and preparation transport admitted for the installed target.
pub type BatchCapture<B, M> = crate::capture::SpeculativePartitionBinding<
    <<M as ResidentBatchMechanisms<B>>::SessionMechanisms as eredu_core::backend::TextSessionNative<B>>::ControlTransport,
>;

/// Exact native queue associated with this portable batch.
pub type BatchQueue<M> =
    <<M as SharedBackboneMechanisms>::Capture as SpeculativeCaptureMechanisms>::Queue;

pub(super) struct BatchBinding<B: TextGenerationBackend, M: ResidentBatchMechanisms<B>> {
    pub(super) target: BatchQueue<M>,
    pub(super) capture: Option<BatchCapture<B, M>>,
}

impl<B: TextGenerationBackend, M: ResidentBatchMechanisms<B>> BatchBinding<B, M> {
    pub(super) fn new(runtime: &eredu_core::ModelRuntime<B, BatchSession<B, M>>) -> Self {
        Self {
            target: M::target_queue(runtime.backend()),
            capture: crate::text_session_driver::speculative_partition_binding::<B, _>(
                runtime.session(),
            ),
        }
    }

    pub(super) fn context(&self) -> M::Context<'_> {
        M::Capture::attach_partition(
            M::Capture::single_context(&self.target),
            self.capture.as_ref(),
        )
    }
}

/// Resident executor selected by portable route admission.
#[derive(Clone, Copy)]
pub enum ResidentBatchRoute {
    /// Ordinary target with no speculative proposals.
    Target,
    /// Proposals from the target's shared backbone.
    SharedBackbone,
}

/// Acquires, prepares, runs, and settles a resident batch under the portable text driver.
pub fn execute_resident_batch<B, M, C, V>(
    runtime: &mut eredu_core::ModelRuntime<B, BatchSession<B, M>>,
    route: ResidentBatchRoute,
    lanes: Vec<SpeculativeGenerationLane<'_, crate::text_session_driver::TextPromptFor<B>, C>>,
    visitor: V,
) -> Result<eredu_core::SpeculativeGenerationBatchOutput, B::Error>
where
    B: TextGenerationBackend,
    B::Error: From<GenerationError>
        + From<M::Error>
        + From<eredu_core::BackendFailure>
        + From<SpeculativeDriverError<M::Error>>,
    C: SpeculativeTokenFilterController + 'static,
    V: eredu_core::SpeculativeGenerationVisitor,
    M: ResidentBatchMechanisms<B>,
{
    use crate::speculative::{prepare_speculative_batch_lanes, run_speculative_batch};
    use eredu_core::SpeculativeExecutionTopology;

    let binding = BatchBinding::<B, M>::new(runtime);
    let agree = |stage, status| M::agree_text_preparation(stage, status, binding.context());
    let agreement_error =
        |failure| B::Error::from(SpeculativeDriverError::<M::Error>::Preparation(failure));
    let agree_request = |local| {
        finish_preparation(
            TextPreparationStage::Request,
            local,
            |status| agree(TextPreparationStage::Request, status),
            agreement_error,
        )
    };
    let timings = M::collect_component_timings();
    let session = runtime.session_mut();

    match route {
        ResidentBatchRoute::Target => session.with_prepared_model_transaction(
            |local| {
                agree_request(local)?;
                finish_preparation(
                    TextPreparationStage::Sampling,
                    prepare_speculative_batch_lanes(lanes, 0, M::prepare_sampling),
                    |status| agree(TextPreparationStage::Sampling, status),
                    agreement_error,
                )
            },
            |model, prepared| {
                let (mut executor, prepared) =
                    prepare_target_batch::<M, _, _>(model.as_mut(), prepared, binding.context())?;
                Ok(run_speculative_batch(
                    &mut executor,
                    prepared,
                    &(),
                    binding.context(),
                    SpeculativeExecutionTopology::Single,
                    timings,
                    visitor,
                )?)
            },
        ),
        ResidentBatchRoute::SharedBackbone => {
            session.with_prepared_model_transaction(agree_request, |model, ()| {
                let (mut executor, prepared) = prepare_shared_batch::<M, B, _, _, _, _>(
                    model.as_mut(),
                    lanes,
                    binding.context(),
                    M::prepare_sampling,
                )?;
                Ok(run_speculative_batch(
                    &mut executor,
                    prepared,
                    &(),
                    binding.context(),
                    SpeculativeExecutionTopology::Single,
                    timings,
                    visitor,
                )?)
            })
        }
    }
}
