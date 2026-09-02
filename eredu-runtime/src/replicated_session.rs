//! Backend-neutral replicated-text execution and session ownership.

#![allow(clippy::type_complexity)]

use std::{
    collections::{BTreeMap, BTreeSet},
    marker::PhantomData,
    path::Path,
};

use eredu_core::cache::{
    validate_prompt_cache_model_identity, PromptCacheDescriptor, PromptCacheError,
    PromptCacheManifest, PromptCacheModelIdentity, PromptCacheOptions,
};
use eredu_nn::{NeuralBackend, Tensor};

use crate::{
    observe_model_logits, replicated_text_materialization_tasks, ActivationObserver,
    ExecutionResidency, LayerWeightResidency, LayerwisePolicy, LayerwiseRuntime,
    LayerwiseRuntimeError, ParameterGroupOwner, PartitionState, ReplicatedTextArchitecture,
    ReplicatedTextMaterializationTask, ReplicatedTextOutputCompanion,
    ReplicatedTextOutputSelection, ReplicatedTextParameterOwner, ReplicatedTextParameterPresence,
    RuntimeState, SelectedReplicatedTextRealization, SelectedStateRealization, StateError,
    SubmissionBackend, WeightLoweringKind,
};

/// Backend mechanisms used by the generic replicated-text constructor.
///
/// Implementations allocate native state, prepare exact materialization tasks,
/// supply a bounded residency policy, persist opaque state bytes, apply a
/// requested native tensor index, and retain resources through final
/// completion. The trait receives selected values but no family identity or
/// caller selection request.
pub trait ReplicatedTextSessionMechanisms<A, B>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    A: ReplicatedTextArchitecture<B, Self::State>,
    Self::State: RuntimeState<B>,
    Self::ResidentPolicy: LayerwisePolicy<B, A::Unit, Error = Self::PolicyError>,
    Self::BoundedPolicy: LayerwisePolicy<B, A::Unit, Error = Self::PolicyError>,
{
    /// Concrete mutable-state realization paired with the architecture.
    type State: RuntimeState<B>;
    /// Shared failure type for resident and bounded runtime policies.
    type PolicyError;
    /// Concrete policy that owns fully resident bound units.
    type ResidentPolicy: LayerwisePolicy<B, A::Unit, Error = Self::PolicyError>;
    /// Concrete policy used for host-windowed or disk-streamed traversal.
    type BoundedPolicy: LayerwisePolicy<B, A::Unit, Error = Self::PolicyError>;
    /// Opaque state checkpoint owned by the backend mechanism.
    type StateCheckpoint;
    /// Backend-native mutable-state residency report.
    type StateReport;
    /// Backend-native parameter/runtime residency report.
    type ExecutionReport;
    /// Mechanism failure.
    type Error;

    /// Consumes the exact selected parameter tasks before runtime construction.
    #[allow(clippy::too_many_arguments)]
    fn prepare_materialization(
        &mut self,
        architecture: &mut A,
        layout: &crate::ExecutionUnitLayout,
        units: &mut [A::Unit],
        source_architecture: Option<&mut A>,
        source_units: Option<&mut [A::Unit]>,
        tasks: &[ReplicatedTextMaterializationTask],
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
    ) -> Result<(), Self::Error>;

    /// Realizes exactly the selected mutable-state components and placements.
    fn realize_state(
        &mut self,
        selected: &SelectedStateRealization,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
    ) -> Result<Self::State, Self::Error>;

    /// Creates the concrete policy owning fully resident bound units.
    fn resident_policy(
        &mut self,
        architecture: &mut A,
        units: Vec<A::Unit>,
        selected: &SelectedReplicatedTextRealization,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
    ) -> Result<Self::ResidentPolicy, Self::Error>;

    /// Creates the concrete bounded-unit policy selected for this session.
    fn bounded_policy(
        &mut self,
        architecture: &mut A,
        selected: &SelectedReplicatedTextRealization,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
    ) -> Result<Self::BoundedPolicy, Self::Error>;

    /// Applies one neutral sequence-axis index to a complete architecture output.
    fn index_text_output(
        &mut self,
        output: B::Tensor,
        sequence_index: i32,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error>;

    /// Captures an opaque checkpoint of every live state component.
    fn checkpoint_state(
        &mut self,
        state: &Self::State,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
    ) -> Result<Self::StateCheckpoint, Self::Error>;

    /// Restores every component from an opaque checkpoint.
    fn restore_state(
        &mut self,
        state: &mut Self::State,
        checkpoint: Self::StateCheckpoint,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
    ) -> Result<(), Self::Error>;

    /// Restores native state bytes from a validated prompt-cache artifact.
    fn load_prompt_cache(
        &mut self,
        directory: &Path,
        expected: &PromptCacheDescriptor,
        identity: &PromptCacheModelIdentity,
        prefix_token_ids: &[u32],
        selected: &SelectedStateRealization,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
    ) -> Result<(Self::State, PromptCacheManifest), Self::Error>;

    /// Serializes native state bytes for a neutrally validated cache identity.
    fn save_prompt_cache(
        &mut self,
        state: &mut Self::State,
        destination: &Path,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &PromptCacheOptions,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
    ) -> Result<PromptCacheManifest, Self::Error>;

    /// Reports the realized mutable-state storage.
    fn state_report(&self, state: &Self::State) -> Result<Self::StateReport, Self::Error>;

    /// Reports the selected resident or bounded runtime realization.
    fn execution_report(
        &self,
        residency: LayerWeightResidency,
        bounded: Option<&Self::BoundedPolicy>,
    ) -> Result<Self::ExecutionReport, Self::Error>;

    /// Retains final output and mutable state through exact completion.
    fn complete(
        &mut self,
        output: &B::Tensor,
        state: &Self::State,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
    ) -> Result<(), Self::Error>;
}

/// Resident or bounded execution selected before construction.
enum ReplicatedTextExecution<A, B, S, R, P>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    S: RuntimeState<B>,
    A: ReplicatedTextArchitecture<B, S>,
    R: LayerwisePolicy<B, A::Unit>,
    P: LayerwisePolicy<B, A::Unit, Error = R::Error>,
{
    /// Every architecture unit remains resident.
    Resident(LayerwiseRuntime<A, B, S, R>),
    /// Units are acquired through a bounded backend policy.
    Bounded(LayerwiseRuntime<A, B, S, P>),
}

impl<A, B, S, R, P> ReplicatedTextExecution<A, B, S, R, P>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    S: RuntimeState<B>,
    A: ReplicatedTextArchitecture<B, S>,
    R: LayerwisePolicy<B, A::Unit>,
    P: LayerwisePolicy<B, A::Unit, Error = R::Error>,
    A::Error: std::fmt::Display,
    P::Error: std::fmt::Display,
{
    fn forward_with_observer<'a, O>(
        &mut self,
        input: A::Input<'a>,
        state: &mut S,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, ReplicatedTextSessionError<A::Error, R::Error, std::convert::Infallible>>
    where
        O: ActivationObserver<B::Tensor, A::Error> + ?Sized,
    {
        match self {
            Self::Resident(runtime) => runtime
                .forward_with_observer(input, state, context, observer)
                .map_err(map_layerwise_error),
            Self::Bounded(runtime) => runtime
                .forward_with_observer(input, state, context, observer)
                .map_err(map_layerwise_error),
        }
    }

    fn bounded_policy(&self) -> Option<&P> {
        match self {
            Self::Resident(_) => None,
            Self::Bounded(runtime) => Some(runtime.policy()),
        }
    }
}

/// Complete backend-neutral replicated-text session.
pub struct ReplicatedTextSession<A, B, M>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    M: ReplicatedTextSessionMechanisms<A, B>,
    A: ReplicatedTextArchitecture<B, M::State>,
{
    selected: SelectedReplicatedTextRealization,
    execution: ReplicatedTextExecution<A, B, M::State, M::ResidentPolicy, M::BoundedPolicy>,
    state: M::State,
    mechanisms: M,
    prompt_cache_identity: PromptCacheModelIdentity,
    output_selection: ReplicatedTextOutputSelection,
    backend: PhantomData<fn() -> B>,
}

/// Cold-path failure from replicated-text construction or session control.
#[derive(Debug, thiserror::Error)]
pub enum ReplicatedTextSessionError<A, P, M>
where
    A: std::fmt::Display,
    P: std::fmt::Display,
    M: std::fmt::Display,
{
    /// The prepared architecture disagreed with its selected realization.
    #[error("replicated text contract mismatch: {0}")]
    Contract(String),
    /// Architecture construction or execution failed.
    #[error("replicated text architecture failed: {0}")]
    Architecture(A),
    /// Bounded residency failed.
    #[error("replicated text residency failed: {0}")]
    Policy(P),
    /// A native mechanism failed.
    #[error("replicated text mechanism failed: {0}")]
    Mechanism(M),
    /// Mutable-state access failed.
    #[error(transparent)]
    State(#[from] StateError),
    /// Prompt-cache identity or manifest validation failed.
    #[error(transparent)]
    PromptCache(#[from] PromptCacheError),
}

/// Immutable session and residency report produced by neutral orchestration.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ReplicatedTextSessionReport<E, S> {
    execution: ExecutionResidency,
    execution_report: E,
    state_report: S,
}

/// Opaque proof that one concrete architecture agrees with an authoritative
/// replicated-text selection and its exact materialization tasks.
///
/// The value can only be created by [`prepare_replicated_text_contract`].
pub struct PreparedReplicatedTextContract {
    selected: SelectedReplicatedTextRealization,
    tasks: Vec<ReplicatedTextMaterializationTask>,
    prompt_cache_identity: PromptCacheModelIdentity,
    output_selection: ReplicatedTextOutputSelection,
}

impl PreparedReplicatedTextContract {
    /// Returns the authoritative selected realization.
    pub const fn selected(&self) -> &SelectedReplicatedTextRealization {
        &self.selected
    }

    /// Returns the validated exact materialization tasks.
    pub fn materialization_tasks(&self) -> &[ReplicatedTextMaterializationTask] {
        &self.tasks
    }

    /// Returns the architecture-derived identity coupled to this proof.
    pub const fn prompt_cache_identity(&self) -> &PromptCacheModelIdentity {
        &self.prompt_cache_identity
    }

    /// Returns the architecture-declared causal output projection.
    pub const fn output_selection(&self) -> ReplicatedTextOutputSelection {
        self.output_selection
    }

    fn into_parts(
        self,
    ) -> (
        SelectedReplicatedTextRealization,
        Vec<ReplicatedTextMaterializationTask>,
        PromptCacheModelIdentity,
        ReplicatedTextOutputSelection,
    ) {
        (
            self.selected,
            self.tasks,
            self.prompt_cache_identity,
            self.output_selection,
        )
    }
}

/// Validates a concrete architecture against its authoritative selection and
/// produces the unforgeable contract consumed by the neutral constructor.
pub fn prepare_replicated_text_contract<A, B, S>(
    architecture: &A,
    source_architecture: Option<&A>,
    selected: SelectedReplicatedTextRealization,
    expected_prompt_cache_architecture_identity: &str,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<PreparedReplicatedTextContract, String>
where
    B: NeuralBackend,
    S: RuntimeState<B>,
    A: ReplicatedTextArchitecture<B, S>,
    A::Error: std::fmt::Display,
{
    validate_selected_state(&selected)?;
    validate_architecture_geometry::<A, B, S>(architecture, &selected)?;
    if let Some(source) = source_architecture {
        validate_architecture_geometry::<A, B, S>(source, &selected)?;
    }
    let has_transform = selected.parameters().iter().any(|parameter| {
        matches!(
            parameter.lowering(),
            WeightLoweringKind::Transform | WeightLoweringKind::DerivedTransform
        )
    });
    if has_transform != source_architecture.is_some() {
        return Err(
            "selected transform tasks and source-format architecture ownership disagree".into(),
        );
    }
    if let Some(source) = source_architecture {
        validate_architecture_parameters::<A, B, S>(source, &selected, false, context)?;
    }
    let mut output_companions =
        validate_architecture_parameters::<A, B, S>(architecture, &selected, true, context)?;
    let mut tasks =
        replicated_text_materialization_tasks(&selected).map_err(|error| error.to_string())?;
    let companion_names = output_companions
        .values()
        .flatten()
        .map(|companion| companion.name().to_owned())
        .collect::<BTreeSet<_>>();
    let companion_tasks = tasks
        .iter()
        .filter(|task| companion_names.contains(task.name()))
        .map(|task| (task.name().to_owned(), task.clone()))
        .collect::<BTreeMap<_, _>>();
    for task in &mut tasks {
        let companions = output_companions
            .remove(task.name())
            .unwrap_or_default()
            .into_iter()
            .map(
                |companion| match companion_tasks.get(companion.name()).cloned() {
                    Some(exact) => companion
                        .with_materialization_task(exact)
                        .map_err(|error| error.to_string()),
                    None => Ok(companion),
                },
            )
            .collect::<Result<Vec<_>, String>>()?;
        task.set_output_companions(companions)
            .map_err(|error| error.to_string())?;
    }
    if !output_companions.is_empty() {
        return Err(format!(
            "output companion catalog contains unknown materialization tasks: {:?}",
            output_companions.keys().collect::<Vec<_>>()
        ));
    }
    // Checkpoint-native companions participate in selected-parameter
    // validation, but their primary packed output owns their one causal
    // materialization task. Do not consume them again as standalone tasks.
    tasks.retain(|task| !companion_names.contains(task.name()));
    let state = PartitionState::new(selected.state().layout().clone(), 0)
        .map_err(|error| error.to_string())?;
    let prompt_cache_identity = state
        .prompt_cache_identity::<B, A>(architecture, Default::default())
        .map_err(|error| error.to_string())?;
    if prompt_cache_identity.architecture_fingerprint()
        != expected_prompt_cache_architecture_identity
        || prompt_cache_identity.layer_count() != selected.state().layout().len()
        || prompt_cache_identity.global_layer_start() != 0
        || prompt_cache_identity.global_layer_end() != selected.state().layout().len()
        || prompt_cache_identity.topology() != &Default::default()
    {
        return Err("architecture prompt-cache identity differs from selection".into());
    }
    let output_selection = architecture.text_output_selection();
    Ok(PreparedReplicatedTextContract {
        selected,
        tasks,
        prompt_cache_identity,
        output_selection,
    })
}

impl<E, S> ReplicatedTextSessionReport<E, S> {
    /// Returns the selected resident or bounded execution class.
    pub const fn execution(&self) -> ExecutionResidency {
        self.execution
    }

    /// Returns backend-native parameter/runtime residency details.
    pub const fn execution_report(&self) -> &E {
        &self.execution_report
    }

    /// Returns backend-native mutable-state residency details.
    pub const fn state_report(&self) -> &S {
        &self.state_report
    }
}

/// Constructs one complete replicated-text session from selected policy and
/// mechanism implementations.
pub fn construct_replicated_text_session<A, B, M>(
    mut architecture: A,
    mut source_architecture: Option<A>,
    prepared: PreparedReplicatedTextContract,
    mut mechanisms: M,
    context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
) -> Result<
    ReplicatedTextSession<A, B, M>,
    ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>,
>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    M: ReplicatedTextSessionMechanisms<A, B>,
    A: ReplicatedTextArchitecture<B, M::State>,
    A::Error: std::fmt::Display,
    M::PolicyError: std::fmt::Display,
    M::Error: std::fmt::Display,
{
    let (selected, tasks, prompt_cache_identity, output_selection) = prepared.into_parts();
    let mut units = construct_units::<A, B, M::State>(
        &architecture,
        selected.requirements().execution_units(),
        context,
    )
    .map_err(ReplicatedTextSessionError::Architecture)?;
    let mut source_units = source_architecture
        .as_ref()
        .map(|source| {
            construct_units::<A, B, M::State>(
                source,
                selected.requirements().execution_units(),
                context,
            )
        })
        .transpose()
        .map_err(ReplicatedTextSessionError::Architecture)?;
    mechanisms
        .prepare_materialization(
            &mut architecture,
            selected.requirements().execution_units(),
            &mut units,
            source_architecture.as_mut(),
            source_units.as_deref_mut(),
            &tasks,
            context,
        )
        .map_err(ReplicatedTextSessionError::Mechanism)?;
    let state = mechanisms
        .realize_state(selected.state(), context)
        .map_err(ReplicatedTextSessionError::Mechanism)?;
    validate_realized_state(&state, selected.state())?;
    let execution = match selected.residency() {
        LayerWeightResidency::FullyResident => {
            let policy = mechanisms
                .resident_policy(&mut architecture, units, &selected, context)
                .map_err(ReplicatedTextSessionError::Mechanism)?;
            ReplicatedTextExecution::Resident(LayerwiseRuntime::new(architecture, policy))
        }
        LayerWeightResidency::LayerwiseHost(_) | LayerWeightResidency::DenseDiskStream(_) => {
            let policy = mechanisms
                .bounded_policy(&mut architecture, &selected, context)
                .map_err(ReplicatedTextSessionError::Mechanism)?;
            ReplicatedTextExecution::Bounded(LayerwiseRuntime::new(architecture, policy))
        }
    };
    Ok(ReplicatedTextSession {
        selected,
        execution,
        state,
        mechanisms,
        prompt_cache_identity,
        output_selection,
        backend: PhantomData,
    })
}

fn construct_units<A, B, S>(
    architecture: &A,
    layout: &crate::ExecutionUnitLayout,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<Vec<A::Unit>, A::Error>
where
    B: NeuralBackend,
    S: RuntimeState<B>,
    A: ReplicatedTextArchitecture<B, S>,
{
    (0..layout.len())
        .map(|ordinal| {
            let address = layout
                .address(ordinal)
                .expect("validated replicated layout contains every ordinal");
            architecture.build_unit(address.group(), address.index(), context)
        })
        .collect()
}

impl<A, B, M> ReplicatedTextSession<A, B, M>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    M: ReplicatedTextSessionMechanisms<A, B>,
    A: ReplicatedTextArchitecture<B, M::State>,
    A::Error: std::fmt::Display,
    M::PolicyError: std::fmt::Display,
    M::Error: std::fmt::Display,
{
    /// Runs one direct forward and returns the complete architecture output.
    pub fn forward(
        &mut self,
        tokens: &B::Tensor,
        mask: Option<&B::Tensor>,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>> {
        self.forward_with_observer(tokens, mask, context, &mut crate::NoopObserver)
    }

    /// Runs one direct forward with unit and final-logits observation and intervention.
    pub fn forward_with_observer<O>(
        &mut self,
        tokens: &B::Tensor,
        mask: Option<&B::Tensor>,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>>
    where
        O: ActivationObserver<B::Tensor, A::Error> + ?Sized,
    {
        let (output, checkpoint) = self.execute_with_observer(tokens, mask, context, observer)?;
        self.publish(output, checkpoint, context)
    }

    /// Runs prompt processing and selects the architecture-declared text output.
    pub fn prefill(
        &mut self,
        tokens: &B::Tensor,
        mask: Option<&B::Tensor>,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>> {
        self.prefill_with_observer(tokens, mask, context, &mut crate::NoopObserver)
    }

    /// Runs observed prompt processing and selects the declared text output.
    pub fn prefill_with_observer<O>(
        &mut self,
        tokens: &B::Tensor,
        mask: Option<&B::Tensor>,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>>
    where
        O: ActivationObserver<B::Tensor, A::Error> + ?Sized,
    {
        let (output, checkpoint) = self.execute_with_observer(tokens, mask, context, observer)?;
        let sequence_index = self.output_selection.sequence_index();
        let output = match self
            .mechanisms
            .index_text_output(output, sequence_index, context)
        {
            Ok(output) => output,
            Err(error) => {
                return self.rollback_failure(
                    checkpoint,
                    ReplicatedTextSessionError::Mechanism(error),
                    context,
                )
            }
        };
        self.publish(output, checkpoint, context)
    }

    /// Runs one decode step and selects the architecture-declared text output.
    pub fn decode(
        &mut self,
        tokens: &B::Tensor,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>> {
        self.prefill(tokens, None, context)
    }

    /// Runs one observed decode step and selects the declared text output.
    pub fn decode_with_observer<O>(
        &mut self,
        tokens: &B::Tensor,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>>
    where
        O: ActivationObserver<B::Tensor, A::Error> + ?Sized,
    {
        self.prefill_with_observer(tokens, None, context, observer)
    }

    /// Captures all mutable state for a later transactional rollback.
    pub fn checkpoint(
        &mut self,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
    ) -> Result<M::StateCheckpoint, ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>>
    {
        self.mechanisms
            .checkpoint_state(&self.state, context)
            .map_err(ReplicatedTextSessionError::Mechanism)
    }

    /// Restores every mutable component from a session checkpoint.
    pub fn rollback(
        &mut self,
        checkpoint: M::StateCheckpoint,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
    ) -> Result<(), ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>> {
        self.mechanisms
            .restore_state(&mut self.state, checkpoint, context)
            .map_err(ReplicatedTextSessionError::Mechanism)
    }

    /// Replaces mutable state with a newly realized selected state.
    pub fn reset(
        &mut self,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
    ) -> Result<(), ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>> {
        let state = self
            .mechanisms
            .realize_state(self.selected.state(), context)
            .map_err(ReplicatedTextSessionError::Mechanism)?;
        validate_realized_state(&state, self.selected.state())?;
        self.state = state;
        Ok(())
    }

    /// Validates and replaces state from a reusable prompt cache.
    pub fn load_prompt_cache(
        &mut self,
        directory: &Path,
        expected: &PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
    ) -> Result<PromptCacheManifest, ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>>
    {
        let identity = self.prompt_cache_identity()?.clone();
        validate_prompt_cache_model_identity(expected, &identity)?;
        let (state, manifest) = self
            .mechanisms
            .load_prompt_cache(
                directory,
                expected,
                &identity,
                prefix_token_ids,
                self.selected.state(),
                context,
            )
            .map_err(ReplicatedTextSessionError::Mechanism)?;
        manifest.validate_compatibility(expected, prefix_token_ids)?;
        validate_realized_state(&state, self.selected.state())?;
        self.state = state;
        Ok(manifest)
    }

    /// Validates identity and persists the current state through native bytes.
    pub fn save_prompt_cache(
        &mut self,
        destination: &Path,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &PromptCacheOptions,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
    ) -> Result<PromptCacheManifest, ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>>
    {
        validate_prompt_cache_model_identity(&descriptor, self.prompt_cache_identity()?)?;
        let manifest = self
            .mechanisms
            .save_prompt_cache(
                &mut self.state,
                destination,
                descriptor.clone(),
                prefix_token_ids,
                options,
                context,
            )
            .map_err(ReplicatedTextSessionError::Mechanism)?;
        manifest.validate_compatibility(&descriptor, prefix_token_ids)?;
        Ok(manifest)
    }

    /// Returns one coherent execution and state residency report.
    pub fn report(
        &self,
    ) -> Result<
        ReplicatedTextSessionReport<M::ExecutionReport, M::StateReport>,
        ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>,
    > {
        let execution_report = self
            .mechanisms
            .execution_report(self.selected.residency(), self.execution.bounded_policy())
            .map_err(ReplicatedTextSessionError::Mechanism)?;
        let state_report = self
            .mechanisms
            .state_report(&self.state)
            .map_err(ReplicatedTextSessionError::Mechanism)?;
        Ok(ReplicatedTextSessionReport {
            execution: self.selected.residency().execution_residency(),
            execution_report,
            state_report,
        })
    }

    fn execute_with_observer<O>(
        &mut self,
        tokens: &B::Tensor,
        mask: Option<&B::Tensor>,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<
        (B::Tensor, M::StateCheckpoint),
        ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>,
    >
    where
        O: ActivationObserver<B::Tensor, A::Error> + ?Sized,
    {
        let checkpoint = self
            .mechanisms
            .checkpoint_state(&self.state, context)
            .map_err(ReplicatedTextSessionError::Mechanism)?;
        let input = A::text_input(tokens, mask);
        let output = match self
            .execution
            .forward_with_observer(input, &mut self.state, context, observer)
            .map_err(widen_infallible)
        {
            Ok(output) => output,
            Err(error) => return self.rollback_failure(checkpoint, error, context),
        };
        let output = match observe_model_logits(observer, &output) {
            Ok(output) => output,
            Err(error) => {
                return self.rollback_failure(
                    checkpoint,
                    ReplicatedTextSessionError::Architecture(error),
                    context,
                )
            }
        };
        Ok((output, checkpoint))
    }

    fn publish(
        &mut self,
        output: B::Tensor,
        checkpoint: M::StateCheckpoint,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>> {
        if self.selected.exact_completion() {
            if let Err(error) = self.mechanisms.complete(&output, &self.state, context) {
                return self.rollback_failure(
                    checkpoint,
                    ReplicatedTextSessionError::Mechanism(error),
                    context,
                );
            }
        }
        Ok(output)
    }

    fn rollback_failure<T>(
        &mut self,
        checkpoint: M::StateCheckpoint,
        error: ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
    ) -> Result<T, ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>> {
        self.mechanisms
            .restore_state(&mut self.state, checkpoint, context)
            .map_err(ReplicatedTextSessionError::Mechanism)?;
        Err(error)
    }

    fn prompt_cache_identity(
        &self,
    ) -> Result<
        &PromptCacheModelIdentity,
        ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>,
    > {
        if !self.selected.prompt_cache() {
            return Err(ReplicatedTextSessionError::Contract(
                "prompt-cache persistence was not selected for this session".into(),
            ));
        }
        Ok(&self.prompt_cache_identity)
    }
}

fn validate_architecture_geometry<A, B, S>(
    architecture: &A,
    selected: &SelectedReplicatedTextRealization,
) -> Result<(), String>
where
    B: NeuralBackend,
    S: RuntimeState<B>,
    A: ReplicatedTextArchitecture<B, S>,
    A::Error: std::fmt::Display,
{
    let requirements = selected.requirements();
    let graph = architecture
        .execution_graph()
        .map_err(|error| error.to_string())?;
    if &graph != requirements.execution_graph() {
        return Err("architecture execution graph differs from selection".into());
    }
    if requirements.execution_units().group_count() != graph.groups().len() {
        return Err("selected execution-unit groups differ from architecture graph".into());
    }
    for group in 0..graph.groups().len() {
        let actual = architecture
            .group_unit_count(group)
            .map_err(|error| error.to_string())?;
        let expected = requirements
            .execution_units()
            .group_range(group)
            .expect("validated requirement exposes every graph group")
            .len();
        if actual != expected
            || architecture.group_transport(group) != requirements.group_transports()[group]
        {
            return Err(format!(
                "architecture execution group {group} differs from selection"
            ));
        }
    }
    let layout = architecture
        .state_layout()
        .map_err(|error| error.to_string())?;
    if &layout != selected.state().layout() {
        return Err("architecture state layout differs from selection".into());
    }
    Ok(())
}

fn validate_architecture_parameters<A, B, S>(
    architecture: &A,
    selected: &SelectedReplicatedTextRealization,
    selected_formats: bool,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<BTreeMap<String, Vec<ReplicatedTextOutputCompanion>>, String>
where
    B: NeuralBackend,
    S: RuntimeState<B>,
    A: ReplicatedTextArchitecture<B, S>,
    A::Error: std::fmt::Display,
{
    let requirements = selected.requirements();
    let description = architecture
        .parameter_description(context)
        .map_err(|error| error.to_string())?;
    let mut actual = BTreeMap::<String, (Vec<usize>, ParameterGroupOwner)>::new();
    let mut companions = Vec::new();
    for group in description.groups() {
        for member in group.group().members() {
            match (member.linear_companion(), member.linear_companion_of()) {
                (None, None) => {
                    if actual
                        .insert(
                            member.target().to_owned(),
                            (member.global_shape().to_vec(), group.owner().clone()),
                        )
                        .is_some()
                    {
                        return Err(format!(
                            "constructed architecture repeats primary parameter {:?}",
                            member.target()
                        ));
                    }
                }
                (Some(role), Some(primary)) => companions.push((
                    member.target().to_owned(),
                    member.global_shape().to_vec(),
                    group.owner().clone(),
                    role,
                    primary.to_owned(),
                )),
                _ => {
                    return Err(format!(
                        "constructed parameter {:?} has incomplete linear companion metadata",
                        member.target()
                    ));
                }
            }
        }
    }
    let expected = requirements
        .parameters()
        .iter()
        .filter(|parameter| {
            !matches!(
                parameter.presence(),
                ReplicatedTextParameterPresence::OptionalAbsent
                    | ReplicatedTextParameterPresence::Tied { .. }
            )
        })
        .map(|parameter| parameter.name())
        .collect::<BTreeSet<_>>();
    for (name, shape, owner, _, _) in &companions {
        if expected.contains(name.as_str())
            && actual
                .insert(name.clone(), (shape.clone(), owner.clone()))
                .is_some()
        {
            return Err(format!(
                "constructed architecture repeats selected parameter {name:?}"
            ));
        }
    }
    let actual_names = actual.keys().map(String::as_str).collect::<BTreeSet<_>>();
    if expected != actual_names {
        return Err(format!(
            "selected parameter catalog differs from constructed architecture: missing {:?}, unexpected {:?}",
            expected.difference(&actual_names).collect::<Vec<_>>(),
            actual_names.difference(&expected).collect::<Vec<_>>()
        ));
    }
    for parameter in requirements.parameters().iter().filter(|parameter| {
        !matches!(
            parameter.presence(),
            ReplicatedTextParameterPresence::OptionalAbsent
                | ReplicatedTextParameterPresence::Tied { .. }
        )
    }) {
        let (shape, owner) = actual
            .get(parameter.name())
            .expect("equal parameter-name sets contain every requirement");
        let owner_matches = match (owner, parameter.owner()) {
            (
                ParameterGroupOwner::StaticRole(actual),
                ReplicatedTextParameterOwner::StaticRole(expected),
            ) => actual == expected,
            (
                ParameterGroupOwner::StaticAnyOf(actual),
                ReplicatedTextParameterOwner::StaticRole(expected),
            ) => actual.iter().any(|role| role == expected),
            (
                ParameterGroupOwner::ExecutionUnit {
                    group, global_unit, ..
                },
                ReplicatedTextParameterOwner::ExecutionUnit {
                    group: expected_group,
                    unit: expected_unit,
                },
            ) => group.as_str() == expected_group && global_unit == expected_unit,
            _ => false,
        };
        let mut expected_shape = parameter.logical_shape().to_vec();
        if selected_formats {
            if let Some(last) = expected_shape.last_mut() {
                let executable = selected
                    .parameters()
                    .iter()
                    .find(|realization| realization.name() == parameter.name())
                    .map_or(parameter.native_executable(), |realization| {
                        realization.executable()
                    });
                match executable {
                    eredu_checkpoint::LinearFormat::Affine(config) => {
                        *last = last.saturating_mul(config.bits as usize) / 32;
                    }
                    eredu_checkpoint::LinearFormat::MxFp4 => {
                        *last = last.saturating_mul(4) / 32;
                    }
                    eredu_checkpoint::LinearFormat::GgufIQuant { ggml_type, .. } => {
                        if let Ok((block, bytes)) = ggml_type.block_and_bytes() {
                            if let (Ok(block), Ok(bytes)) =
                                (usize::try_from(block), usize::try_from(bytes))
                            {
                                *last = last.saturating_mul(bytes) / block;
                            }
                        }
                    }
                    eredu_checkpoint::LinearFormat::Dense
                    | eredu_checkpoint::LinearFormat::E4M3BlockFp8(_) => {}
                }
            }
        }
        if shape != &expected_shape || !owner_matches {
            return Err(format!(
                "selected parameter {:?} differs from constructed shape or owner",
                parameter.name()
            ));
        }
    }

    let mut output_companions = BTreeMap::<String, Vec<ReplicatedTextOutputCompanion>>::new();
    for (name, shape, owner, role, primary) in companions {
        if name == primary || !expected.contains(primary.as_str()) {
            return Err(format!(
                "constructed companion {name:?} names unselected primary {primary:?}"
            ));
        }
        if selected_formats {
            let recipe = selected.requirements().derived_recipes().get(&name);
            let output = selected.requirements().derived_recipe_outputs().get(&name);
            let companion = match (recipe, output) {
                (Some(recipe), Some(output)) => {
                    ReplicatedTextOutputCompanion::new(name, role, shape, owner).map(|companion| {
                        companion.with_derived_recipe(recipe.clone(), output.clone())
                    })
                }
                (None, None) => ReplicatedTextOutputCompanion::new(name, role, shape, owner),
                _ => {
                    return Err(format!(
                        "constructed companion {name:?} has incomplete derived metadata"
                    ))
                }
            }
            .map_err(|error| error.to_string())?;
            let companion = match requirements
                .parameters()
                .iter()
                .find(|parameter| parameter.name() == companion.name())
            {
                Some(parameter)
                    if matches!(
                        parameter.source_encoding(),
                        Some(eredu_checkpoint::SourceTensorEncoding::Gguf { .. })
                    ) =>
                {
                    let [source] = parameter.physical_sources() else {
                        return Err(format!(
                            "translated catalog companion {:?} has ambiguous provenance",
                            companion.name()
                        ));
                    };
                    companion.with_catalog_source(source.clone())
                }
                _ => companion,
            };
            output_companions
                .entry(primary)
                .or_default()
                .push(companion);
        }
    }
    Ok(output_companions)
}

fn validate_selected_state(selected: &SelectedReplicatedTextRealization) -> Result<(), String> {
    use eredu_core::cache::StateResidencyClass;

    if !selected.topology().is_replicated() {
        return Err("selected replicated-text topology is not replicated".into());
    }
    if selected.state().layout() != selected.requirements().state_layout()
        || selected.state().access() != selected.requirements().state_access()
    {
        return Err("selected state contract differs from architecture requirements".into());
    }
    let mut cursor = 0;
    for layer in 0..selected.state().layout().len() {
        for expected in selected
            .state()
            .layout()
            .components(layer)
            .expect("validated state layout exposes every layer")
        {
            let component = selected.state().components().get(cursor).ok_or_else(|| {
                format!("selected state omits component {cursor} at layer {layer}")
            })?;
            if component.layer() != layer || component.component() != expected {
                return Err(format!(
                    "selected state component {cursor} differs from layer {layer} requirements"
                ));
            }
            let expected_placement = match selected.state().policy() {
                crate::CacheResidencyPolicy::Device => crate::StateComponentPlacement::Device,
                crate::CacheResidencyPolicy::Paged(_) => match expected.residency() {
                    StateResidencyClass::SealablePaged => crate::StateComponentPlacement::Paged,
                    StateResidencyClass::AlwaysDeviceMutable
                    | StateResidencyClass::LayerScopedOffloadable => {
                        crate::StateComponentPlacement::Device
                    }
                },
            };
            if component.placement() != expected_placement {
                return Err(format!(
                    "selected state component {cursor} has {:?} placement, expected {expected_placement:?}",
                    component.placement()
                ));
            }
            cursor += 1;
        }
    }
    if cursor != selected.state().components().len() {
        return Err("selected state contains components beyond its architecture layout".into());
    }
    if !selected.state().checkpoint() || !selected.state().rollback() || !selected.state().reset() {
        return Err("selected state omits a required transactional lifecycle facility".into());
    }
    if selected.state().prompt_cache() != selected.prompt_cache()
        || selected.state().observation_retention()
            != (selected.session().output_observation()
                || selected.session().activation_inspection())
    {
        return Err("selected state lifecycle differs from selected session facilities".into());
    }
    if selected.grouped_operations() != selected.requirements().grouped_operations() {
        return Err("selected grouped operations differ from architecture requirements".into());
    }
    Ok(())
}

fn validate_realized_state<A, P, M, B, S>(
    state: &S,
    selected: &SelectedStateRealization,
) -> Result<(), ReplicatedTextSessionError<A, P, M>>
where
    A: std::fmt::Display,
    P: std::fmt::Display,
    M: std::fmt::Display,
    B: NeuralBackend,
    S: RuntimeState<B>,
{
    if state.layout() != selected.layout() {
        return Err(ReplicatedTextSessionError::Contract(
            "realized state layout differs from selection".into(),
        ));
    }
    Ok(())
}

fn map_layerwise_error<A, P>(
    error: LayerwiseRuntimeError<A, P>,
) -> ReplicatedTextSessionError<A, P, std::convert::Infallible>
where
    A: std::fmt::Display,
    P: std::fmt::Display,
{
    match error {
        LayerwiseRuntimeError::Architecture(error) => {
            ReplicatedTextSessionError::Architecture(error)
        }
        LayerwiseRuntimeError::State(error) => ReplicatedTextSessionError::State(error),
        LayerwiseRuntimeError::Layout(error) => {
            ReplicatedTextSessionError::Contract(error.to_string())
        }
        LayerwiseRuntimeError::Policy(error) => ReplicatedTextSessionError::Policy(error),
        LayerwiseRuntimeError::Submission(error) => ReplicatedTextSessionError::Contract(error),
    }
}

fn widen_infallible<A, P, M>(
    error: ReplicatedTextSessionError<A, P, std::convert::Infallible>,
) -> ReplicatedTextSessionError<A, P, M>
where
    A: std::fmt::Display,
    P: std::fmt::Display,
    M: std::fmt::Display,
{
    match error {
        ReplicatedTextSessionError::Contract(error) => ReplicatedTextSessionError::Contract(error),
        ReplicatedTextSessionError::Architecture(error) => {
            ReplicatedTextSessionError::Architecture(error)
        }
        ReplicatedTextSessionError::Policy(error) => ReplicatedTextSessionError::Policy(error),
        ReplicatedTextSessionError::Mechanism(error) => match error {},
        ReplicatedTextSessionError::State(error) => ReplicatedTextSessionError::State(error),
        ReplicatedTextSessionError::PromptCache(error) => {
            ReplicatedTextSessionError::PromptCache(error)
        }
    }
}
