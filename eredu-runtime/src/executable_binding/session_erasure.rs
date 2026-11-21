//! Erasure of installed typed text sessions and their parameter and observation operations.

use super::{
    embedded_erasure::TextExecutablePrediction,
    erased_text::{ErasedExternalPredictionExecutable, ErasedTextExecutable},
    text_invocation::{
        prepare_composite_prompt, TextInvocationMechanisms, TextInvocationResources,
    },
    BoundTextSession, CompositeSessionIngress,
};
use crate::{
    composite_execution::{CompositeArchitecture, PreparedCompositeInput},
    dense::DenseResidencyMechanisms,
    expert::{ParameterBanksResidencyReport, SharedAddressableParameterBank},
    parameter_operations::{
        BankParameterReadPlan, BankParameterReplacementFacts, ParameterSlotOperation,
        PreparedParameterLocation, PreparedParameterSlot,
    },
    replicated_session::{ReplicatedTextSession, ReplicatedTextSnapshotMechanisms},
    text_session_driver::{ErasedTextStateAccess, TextExecutable, TextPromptView},
    CacheResidencyReport, DenseDiskStreamReport, LayeredArchitecture,
    ReplicatedTextExecutionStrategy, ReplicatedTextMaterializationTask, ResidencyReport,
    RuntimeState,
};
use eredu_core::backend::TextExecutableTypes;
use eredu_core::speculative::{
    erasure::ErasedGenerationExecutor, SpeculativeActivationCapture, SpeculativeControlError,
};
use eredu_core::{
    communication::CommunicationAuthority,
    parameters::ParameterError,
    residency::{
        ParameterConversionRetentionReport, ParameterConversionRetentionTrimReport,
        ParameterConversionTrimError,
    },
    BackendFailure, Observed, SubmissionBackend,
};
use eredu_nn::{NeuralBackend, Tensor};
use std::{collections::BTreeMap, sync::Arc};

/// Observations of a copied or installed state owner.
pub trait TextStateReport {
    /// Decoder offset in the installed state.
    fn offset(&self) -> i32;
    /// Residency measured by the live cache.
    fn residency(&self) -> Option<CacheResidencyReport>;
}

/// Native tensor measurements used in portable publication and reporting.
pub trait TextErasureTypes: TextInvocationMechanisms {
    /// Translates a portable observation failure to the selected native boundary.
    fn state_error(error: BackendFailure) -> Self::StateError;
    /// Reads the layout and size of an exact replacement tensor.
    fn replacement_facts(value: &Self::Tensor) -> BankParameterReplacementFacts;
    /// Exact current offset and native storage bounds for an independent causal lane.
    fn continuation_state(
        state: &Self::PredictionState,
        additional: u64,
    ) -> Result<(u64, Option<u64>, Option<u64>), BackendFailure>;
    /// Reads decoder token IDs already materialized by the architecture.
    fn decoder_token_ids(value: &Self::Tensor) -> Result<Vec<u32>, Self::Error>;
}

type TypesOf<N> = <N as TextInvocationResources>::Types;
/// Portable executable profile derived from retained local resources.
pub type LocalTextTypes<N> = TypesOf<<N as super::LocalTextResources>::LocalResources>;
type TensorOf<N> = <TypesOf<N> as TextExecutableTypes>::Tensor;
/// Portable weight pool over the native storage and tensors of this text placement.
pub type BankOf<N> =
    SharedAddressableParameterBank<<N as TextInvocationResources>::BankStorage, TensorOf<N>>;

/// Native resources retained by the portable text-session owner.
pub trait TextErasureResources:
    TextInvocationResources<Types: TextErasureTypes<Error: From<ParameterError>>>
{
    /// Retains the portable rank driver's selected sampling channel and owner.
    fn with_rank_sampling(
        self,
        group: Option<<Self::Types as TextExecutableTypes>::Group>,
        authority: Option<CommunicationAuthority>,
        owner: usize,
    ) -> Self;

    /// Exact native sampling channel and its selected owner.
    fn sampling(
        &self,
    ) -> Option<(
        &<Self::Types as TextExecutableTypes>::Group,
        &CommunicationAuthority,
        &<Self::Types as TextExecutableTypes>::Context,
        usize,
    )>;
    /// Borrows the installed embedded prediction observers.
    fn prediction_resources(
        &mut self,
    ) -> (
        &<Self::Types as TextExecutableTypes>::Context,
        &mut <Self::Types as TextExecutableTypes>::EmbeddedObservers,
    );
    /// Takes the accumulated native activation records.
    fn take_prediction_capture(&mut self) -> Option<SpeculativeActivationCapture>;
    /// Takes a retained native collector error.
    fn take_prediction_error(&mut self) -> Option<SpeculativeControlError>;
}

/// Publication resources owned by a portable text executable.
pub struct TextPublication<M: TextExecutableTypes, N> {
    context: M::Context,
    observers: M::EmbeddedObservers,
    rank: Option<(Option<M::Group>, CommunicationAuthority, usize)>,
    bank: std::marker::PhantomData<fn() -> N>,
}

impl<M: TextExecutableTypes, N> TextPublication<M, N>
where
    M::EmbeddedObservers: Default,
{
    /// Creates the publication owner over the borrowed native queue.
    pub fn new(context: M::Context) -> Self {
        Self {
            context,
            observers: Default::default(),
            rank: None,
            bank: std::marker::PhantomData,
        }
    }
}

impl<M, N> TextInvocationResources for TextPublication<M, N>
where
    M: TextExecutableTypes,
    N: crate::dense::DenseResidencyMechanisms
        + eredu_core::residency::NativeBankMechanisms<
            Backend: eredu_nn::GroupedNeuralBackend<Tensor = M::Tensor>,
            BankError = M::Error,
        >,
{
    type Types = M;
    type BankStorage = N;

    fn context(&self) -> &M::Context {
        &self.context
    }

    fn has_partition_control(&self) -> bool {
        self.rank.is_some()
    }
}

impl<M, N, T, L, E> TextErasureResources for TextPublication<M, N>
where
    M: TextErasureTypes<
        Error: From<ParameterError>,
        EmbeddedObservers = crate::speculative::EmbeddedPredictionObservers<T, L, E>,
    >,
    Self: TextInvocationResources<Types = M>,
{
    fn with_rank_sampling(
        mut self,
        group: Option<M::Group>,
        authority: Option<CommunicationAuthority>,
        owner: usize,
    ) -> Self {
        self.rank = authority.map(|authority| (group, authority, owner));
        self
    }

    fn sampling(&self) -> Option<(&M::Group, &CommunicationAuthority, &M::Context, usize)> {
        let (group, authority, owner) = self.rank.as_ref()?;
        Some((group.as_ref()?, authority, &self.context, *owner))
    }

    fn prediction_resources(&mut self) -> (&M::Context, &mut M::EmbeddedObservers) {
        (&self.context, &mut self.observers)
    }

    fn take_prediction_capture(&mut self) -> Option<SpeculativeActivationCapture> {
        self.observers.take_activation_capture()
    }

    fn take_prediction_error(&mut self) -> Option<SpeculativeControlError> {
        self.observers.take_activation_error()
    }
}

impl<N> super::LocalTextResources for N
where
    N: eredu_core::checkpoint_mechanisms::TextPublicationResources,
{
    type LocalResources = TextPublication<super::PortableTextTypes<N::Values>, N::BankStorage>;

    fn local_resources(&self) -> Self::LocalResources {
        TextPublication::new(self.publication_context().clone())
    }
}

/// Independently retained parameter-bank telemetry provided by a typed execution strategy.
pub trait TextBankTelemetry<B, Bank> {
    /// Reports independently retained parameter storage when available.
    fn parameter_bank_report(
        &self,
    ) -> Result<Option<ParameterBanksResidencyReport>, BackendFailure>;
}

impl<B, Bank> TextBankTelemetry<B, Bank> for crate::DirectReplicatedTextExecution {
    fn parameter_bank_report(
        &self,
    ) -> Result<Option<ParameterBanksResidencyReport>, BackendFailure> {
        Ok(None)
    }
}

impl<B, Bank, E, G, R, I, T, U, V> TextBankTelemetry<B, Bank>
    for crate::PartitionedTextExecution<E, G, R, I, T, U, V>
{
    fn parameter_bank_report(
        &self,
    ) -> Result<Option<ParameterBanksResidencyReport>, BackendFailure> {
        Ok(None)
    }
}

impl<B, Bank, P> TextBankTelemetry<B, Bank> for crate::RoutedBankProviders<P>
where
    B: eredu_nn::GroupedNeuralBackend,
    Bank: crate::expert::AddressableGroupedBank<
        B,
        Report = crate::expert::ParameterBankResidencyReport,
    >,
    P: super::routed_session::RoutedBankObservation<B, Bank>,
{
    fn parameter_bank_report(
        &self,
    ) -> Result<Option<ParameterBanksResidencyReport>, BackendFailure> {
        if !P::ADDRESSABLE {
            return Ok(None);
        }
        let mut banks = BTreeMap::new();
        for (id, provider) in self.banks() {
            if let Some(report) = provider
                .addressable_report()
                .map_err(BackendFailure::from_error)?
            {
                banks.insert(*id, report);
            }
        }
        Ok(Some(ParameterBanksResidencyReport::new(banks)))
    }
}

impl<B, Bank, P: TextBankTelemetry<B, Bank>> TextBankTelemetry<B, Bank>
    for crate::RoutedReplicatedTextExecution<P>
{
    fn parameter_bank_report(
        &self,
    ) -> Result<Option<ParameterBanksResidencyReport>, BackendFailure> {
        self.provider().parameter_bank_report()
    }
}

impl<B, Bank, P, N, D, G> TextBankTelemetry<B, Bank>
    for crate::ParameterProviders<P, Option<crate::row_lookup::BoundRowLookups<B, N, D, G>>>
where
    B: eredu_nn::DistributedNeuralBackend
        + eredu_core::checkpoint_mechanisms::CheckpointParameterBackend,
    N: DenseResidencyMechanisms + Clone,
    D: crate::row_lookup::RowDecode<B, N>,
    D::Error: std::error::Error + Send + Sync + 'static,
    G: Clone + std::borrow::Borrow<B::ParallelContext>,
    N::Transfer: eredu_core::residency::ResidencyTransferOrdering<
        <B::Tensor as Tensor>::Context,
        Error = N::Error,
    >,
    P: TextBankTelemetry<B, Bank>,
{
    fn parameter_bank_report(
        &self,
    ) -> Result<Option<ParameterBanksResidencyReport>, BackendFailure> {
        let grouped = self.grouped.parameter_bank_report()?;
        let rows = self
            .rows
            .as_ref()
            .map(|rows| rows.pool_report())
            .transpose()
            .map_err(BackendFailure::from_error)?
            .flatten();
        Ok(match rows {
            Some(rows) => Some(
                grouped
                    .unwrap_or_else(|| ParameterBanksResidencyReport::new(BTreeMap::new()))
                    .with_rows(rows),
            ),
            None => grouped,
        })
    }
}

trait ErasureIngress<A, B: NeuralBackend, S, M: TextErasureTypes, Model> {
    fn token_ids(
        &self,
        input: TextPromptView<'_, M::InputTensor>,
        context: &M::Context,
    ) -> Result<Vec<u32>, M::Error>;
    fn external_prediction(
        _model: &mut Model,
    ) -> Option<&mut dyn ErasedExternalPredictionExecutable<M>> {
        None
    }
}

impl<A, B: NeuralBackend, S, M: TextErasureTypes, Model> ErasureIngress<A, B, S, M, Model> for () {
    fn token_ids(
        &self,
        input: TextPromptView<'_, M::InputTensor>,
        context: &M::Context,
    ) -> Result<Vec<u32>, M::Error> {
        super::erased_text::text_token_ids::<M>(input, context)
    }
}

impl<A, B, S, M, R, D, N, P>
    ErasureIngress<A, B, S, M, BoundTextSession<ReplicatedTextSession<A, B, R, D>, Self, N, P>>
    for CompositeSessionIngress<A::AdmissionConfig>
where
    M: TextErasureTypes<Tensor = B::Tensor, Context = B::Executor>,
    B: SubmissionBackend<Executor = <<B as eredu_nn::NeuralBackend>::Tensor as Tensor>::Context>,
    S: RuntimeState<B>,
    A: CompositeArchitecture<B, S, Error = eredu_nn::Error>,
    R: crate::ReplicatedTextSessionMechanisms<A, B, State = S>,
    R::PolicyError: std::fmt::Display,
    D: ReplicatedTextExecutionStrategy<A, B, S, R::ResidentPolicy, R::BoundedPolicy>,
    R::Error: std::error::Error + Send + Sync + 'static,
    R::PolicyError: std::error::Error + Send + Sync + 'static,
    D: TextBankTelemetry<B, BankOf<N>> + 'static,
    B: 'static,
    S: 'static,
    R: 'static,
    N: TextErasureResources<Types = M> + 'static,
    P: 'static,
    M::PredictionState: super::text_invocation::TextPredictionLane<S, Error = M::Error>,
{
    fn token_ids(
        &self,
        input: TextPromptView<'_, M::InputTensor>,
        context: &M::Context,
    ) -> Result<Vec<u32>, M::Error> {
        let (prepared, admitted, _) =
            prepare_composite_prompt::<A, B, S, M>(&self.admission, &self.processor, input, true)?;
        let paired = PreparedCompositeInput::new(&prepared, &admitted).map_err(|error| {
            BackendFailure::from_error(std::io::Error::other(error.to_string()))
        })?;
        let tokens = A::prepared_prediction_token_ids(paired, context)
            .map_err(BackendFailure::from_error)?;
        M::decoder_token_ids(&tokens)
    }
    fn external_prediction(
        model: &mut BoundTextSession<ReplicatedTextSession<A, B, R, D>, Self, N, P>,
    ) -> Option<&mut dyn ErasedExternalPredictionExecutable<M>> {
        if A::supports_external_assistant(&model.ingress.admission) {
            Some(model)
        } else {
            None
        }
    }
}

impl<A, B, S, R, D, I, N, P, M> ErasedTextExecutable<M>
    for BoundTextSession<ReplicatedTextSession<A, B, R, D>, I, N, P>
where
    B: SubmissionBackend<Executor = <<B as eredu_nn::NeuralBackend>::Tensor as Tensor>::Context>,
    S: RuntimeState<B> + 'static,
    A: LayeredArchitecture<B, S, Error = eredu_nn::Error>,
    R: ReplicatedTextSnapshotMechanisms<
        A,
        B,
        State = S,
        PolicyError = M::Error,
        StateReport: TextStateReport,
    >,
    R::Error: std::error::Error + Send + Sync + 'static,
    D: ReplicatedTextExecutionStrategy<A, B, S, R::ResidentPolicy, R::BoundedPolicy>
        + TextBankTelemetry<B, BankOf<N>>,
    M: TextErasureTypes<Tensor = B::Tensor, Context = B::Executor>,
    N: TextErasureResources<Types = M>,
    I: ErasureIngress<A, B, S, M, Self>,
    P: TextExecutablePrediction<M, Self>,
    Self: TextExecutable<
        M,
        Decode = M::Decode,
        ObservationTensor = M::Tensor,
        ObserverMapping = M::ObserverMapping,
        Scores = M::Scores,
        Context = M::Context,
        Error = M::Error,
        AttentionState = M::PredictionState,
    >,
{
    fn forecast_state_offset(&self) -> Result<Option<i32>, M::StateError> {
        self.session
            .report()
            .map(|report| Some(report.state_report().offset()))
            .map_err(|error| M::state_error(BackendFailure::from_error(error)))
    }

    fn prepared_input_token_ids(
        &self,
        input: TextPromptView<'_, M::InputTensor>,
        context: &M::Context,
    ) -> Result<Vec<u32>, M::Error> {
        self.ingress.token_ids(input, context)
    }

    fn visit_loaded_parameters(
        &mut self,
        visitor: &mut dyn eredu_nn::ParameterSlotVisitor<M::Tensor>,
    ) -> bool {
        if self.has_embedded_prediction()
            || !self.banks.parameter_banks().is_empty()
            || !self.session.visit_loaded_parameters(visitor)
        {
            return false;
        }
        self.prediction.visit_parameter_slots(visitor);
        true
    }

    fn parameter_materialization_tasks(&self) -> &[ReplicatedTextMaterializationTask] {
        self.session.installed_parameter_tasks()
    }

    fn partition_parameter_description(
        &self,
    ) -> Option<&Arc<crate::ArchitectureParameterDescription>> {
        self.session.partition_parameter_description()
    }

    fn partition_observation_hooks(&self) -> Option<crate::inspection::ObservationHookSupport> {
        self.session.partition_observation_hooks()
    }

    fn prepared_parameter_slots(&self) -> &[PreparedParameterSlot] {
        self.session.installed_parameter_slots()
    }

    fn with_parameter_slots(
        &mut self,
        location: &PreparedParameterLocation,
        ids: &std::collections::BTreeSet<String>,
        operation: &mut ParameterSlotOperation<'_, M::Tensor, M::Error>,
        context: &M::Context,
    ) -> Result<bool, M::Error> {
        if let PreparedParameterLocation::Bank { bank, unit } = location {
            let Some(owner) = self
                .banks
                .parameter_banks()
                .values()
                .find(|owner| owner.owns_parameter_bank(*bank))
            else {
                return Ok(false);
            };
            BankParameterReadPlan::new(*bank, *unit, self.session.installed_bank_parameters(), ids)
                .map_err(|error| {
                    BackendFailure::from_error(ParameterError::Invalid(error.to_string()))
                })?
                .with_resident_slots(owner, ids, context, operation)?;
            return Ok(true);
        }
        if let PreparedParameterLocation::Prediction { module } = location {
            return self.prediction.with_parameter_slots(*module, operation);
        }
        self.session
            .with_parameter_slots(location, operation, context)
            .map_err(|error| match error {
                crate::LayerwiseAcquireError::Architecture(error) => {
                    BackendFailure::from_error(error).into()
                }
                crate::LayerwiseAcquireError::Policy(error) => error,
            })
    }

    fn publish_parameter_replacements(
        &mut self,
        values: &BTreeMap<String, M::Tensor>,
        active: bool,
    ) -> Result<bool, M::Error> {
        let banks = self
            .banks
            .parameter_banks()
            .values()
            .cloned()
            .collect::<Vec<_>>();
        let published = crate::expert::publish_bank_parameter_replacements(
            &banks,
            values,
            active,
            M::replacement_facts,
            || self.session.publish_parameter_replacements(values, active),
        )?;
        if published {
            self.prediction
                .publish_parameter_replacements(values, active);
        }
        Ok(published)
    }

    fn control_state(&self) -> Option<&dyn ErasedTextStateAccess<M::Context>> {
        Some(&self.session)
    }
    fn control_state_mut(
        &mut self,
    ) -> Option<(&mut dyn ErasedTextStateAccess<M::Context>, &M::Context)> {
        Some((&mut self.session, self.native.context()))
    }

    fn partition_sampling_context(
        &self,
    ) -> Option<(&M::Group, &CommunicationAuthority, &M::Context, usize)> {
        self.native.sampling()
    }

    fn bind_embedded_prediction(
        &mut self,
    ) -> Option<
        Box<
            dyn ErasedGenerationExecutor<
                    M::EmbeddedExecutor,
                    crate::text_session_driver::PreparedTextPrompt<M::InputTensor>,
                > + '_,
        >,
    > {
        P::bind_prediction(self)
    }

    fn install_embedded_prediction_observers(&mut self, observers: M::EmbeddedObservers) -> bool {
        if self.has_embedded_prediction() {
            *self.native.prediction_resources().1 = observers;
            true
        } else {
            false
        }
    }

    fn take_speculative_activation_capture(&mut self) -> Option<SpeculativeActivationCapture> {
        self.native.take_prediction_capture()
    }
    fn take_speculative_activation_error(&mut self) -> Option<SpeculativeControlError> {
        self.native.take_prediction_error()
    }

    fn external_prediction_mut(
        &mut self,
    ) -> Option<&mut dyn ErasedExternalPredictionExecutable<M>> {
        I::external_prediction(self)
    }

    fn trim_parameter_conversions(
        &mut self,
    ) -> Result<Vec<ParameterConversionRetentionTrimReport>, ParameterConversionTrimError> {
        self.session.trim_parameter_conversions()
    }

    fn parameter_conversion_retention(
        &self,
    ) -> Result<Observed<Vec<ParameterConversionRetentionReport>>, BackendFailure> {
        self.session
            .parameter_conversion_retention()
            .map_err(BackendFailure::from_error)
    }

    fn residency_report(&self) -> Result<Option<ResidencyReport>, M::Error> {
        self.session
            .report()
            .map(|report| report.residency().cloned())
            .map_err(|error| BackendFailure::from_error(error).into())
    }

    fn dense_stream_report(&self) -> Result<Option<DenseDiskStreamReport>, M::Error> {
        self.session
            .report()
            .map(|report| report.dense_stream().cloned())
            .map_err(|error| BackendFailure::from_error(error).into())
    }

    fn materialization_report(
        &self,
    ) -> Option<&eredu_checkpoint::quantization::WeightMaterializationReport> {
        self.session.materialization_report()
    }

    fn parameter_bank_report(&self) -> Result<Option<ParameterBanksResidencyReport>, M::Error> {
        parameter_bank_report(
            &self.banks,
            self.session.execution_strategy().parameter_bank_report()?,
        )
        .map_err(Into::into)
    }

    fn cache_residency_report(&self) -> Result<Option<CacheResidencyReport>, M::StateError> {
        self.session
            .report()
            .map(|report| report.state_report().residency())
            .map_err(|error| M::state_error(BackendFailure::from_error(error)))
    }
}

fn parameter_bank_report<N: TextInvocationResources>(
    banks: &super::bound_text_session::TextWeightPools<N>,
    report: Option<ParameterBanksResidencyReport>,
) -> Result<Option<ParameterBanksResidencyReport>, ParameterError> {
    let report = if report.is_some() || banks.parameter_banks().is_empty() {
        report
    } else {
        Some(ParameterBanksResidencyReport::new(
            banks
                .parameter_banks()
                .iter()
                .map(|(id, bank)| bank.report().map(|report| (*id, report)))
                .collect::<Result<BTreeMap<_, _>, _>>()?,
        ))
    };
    match banks.row_pool() {
        Some((pool, requirements)) => Ok(Some(
            report
                .unwrap_or_else(|| ParameterBanksResidencyReport::new(BTreeMap::new()))
                .with_rows(pool.row_report(requirements)?),
        )),
        None => Ok(report),
    }
}

impl<A, B, S, R, D, N, P, M> ErasedExternalPredictionExecutable<M>
    for BoundTextSession<
        ReplicatedTextSession<A, B, R, D>,
        CompositeSessionIngress<A::AdmissionConfig>,
        N,
        P,
    >
where
    B: SubmissionBackend<Executor = <<B as eredu_nn::NeuralBackend>::Tensor as Tensor>::Context>
        + 'static,
    S: RuntimeState<B> + 'static,
    A: CompositeArchitecture<B, S, Error = eredu_nn::Error>,
    R: crate::ReplicatedTextSessionMechanisms<A, B, State = S> + 'static,
    R::PolicyError: std::error::Error + Send + Sync + 'static,
    R::Error: std::error::Error + Send + Sync + 'static,
    D: ReplicatedTextExecutionStrategy<A, B, S, R::ResidentPolicy, R::BoundedPolicy>
        + TextBankTelemetry<B, BankOf<N>>
        + 'static,
    M: TextErasureTypes<Tensor = B::Tensor, Context = B::Executor>,
    N: TextErasureResources<Types = M> + 'static,
    P: 'static,
    M::PredictionState: super::text_invocation::TextPredictionLane<S, Error = M::Error>,
{
    fn continuation_memory_observation(
        &self,
        state: &M::PredictionState,
        additional: u64,
    ) -> Result<eredu_core::speculative::SpeculativeModelMemoryObservation, BackendFailure> {
        use crate::text_session_driver::capability::BankMemoryFacts;
        let (current_positions, current_state_bytes, peak_state_bytes) =
            M::continuation_state(state, additional)?;
        let banks = parameter_bank_report(
            &self.banks,
            self.session.execution_strategy().parameter_bank_report()?,
        )
        .map_err(BackendFailure::from_error)?;
        let banks = banks
            .iter()
            .flat_map(|report| report.banks().values())
            .map(|bank| BankMemoryFacts {
                owned_bytes: bank.owned_bytes(),
                host_resident_bytes: bank.host_resident_bytes(),
                device_resident_bytes: bank.device_resident_bytes(),
            });
        let memory = self
            .session
            .parameter_memory_observation_with_banks(banks)
            .map_err(BackendFailure::from_error)?
            .ok_or_else(|| {
                BackendFailure::new(
                    eredu_core::BackendFailureKind::Unsupported,
                    std::io::Error::other("installed backend cannot report continuation memory"),
                )
            })?;
        Ok(eredu_core::speculative::SpeculativeModelMemoryObservation {
            current_positions,
            current_state_bytes,
            peak_state_bytes,
            parameters: memory.parameters,
            parameter_conversions: memory.parameter_conversions,
            available: memory.available,
            allocator_cache_limit: memory.allocator_cache_limit,
        })
    }

    fn target_placements(&mut self) -> &mut crate::speculative::ExternalTargetPlacements {
        &mut self.ingress.external_target
    }

    fn trim_parameter_conversions(
        &mut self,
    ) -> Result<Vec<ParameterConversionRetentionTrimReport>, ParameterConversionTrimError> {
        self.session.trim_parameter_conversions()
    }

    fn parameter_conversion_retention(
        &self,
    ) -> Result<Observed<Vec<ParameterConversionRetentionReport>>, BackendFailure> {
        self.session
            .parameter_conversion_retention()
            .map_err(BackendFailure::from_error)
    }

    fn prepare_external_prediction_target_cache(&mut self) -> Result<M::PredictionState, M::Error> {
        use super::text_invocation::TextPredictionLane;
        self.session
            .prepare_prediction_target_state(self.native.context())
            .map(M::PredictionState::from_state)
            .map_err(|error| BackendFailure::from_error(error).into())
    }

    fn capture_external_prediction_target(
        &mut self,
        input: Result<TextPromptView<'_, M::InputTensor>, M::Error>,
        pass: crate::ExpertPass,
        request: &crate::speculative::ExternalPredictionCaptureRequest,
        cache: &mut M::PredictionState,
    ) -> Result<
        (
            M::Tensor,
            crate::speculative::ExternalPredictionTargetCapture<M::Tensor>,
        ),
        M::Error,
    > {
        use super::text_invocation::TextPredictionLane;
        let prepared = input.and_then(|input| {
            prepare_composite_prompt::<A, B, S, M>(
                &self.ingress.admission,
                &self.ingress.processor,
                input,
                true,
            )
        });
        let mut lane = cache.take_state()?;
        let context = self.native.context();
        let placements = &mut self.ingress.external_target;
        let result = self.session.with_prediction_target_state::<_, M::Error>(
            &mut lane,
            context,
            |session| {
                let parts;
                let input = match prepared {
                    Ok(value) => {
                        parts = value;
                        PreparedCompositeInput::new(&parts.0, &parts.1)
                            .map_err(eredu_nn::Error::backend)
                    }
                    Err(error) => Err(eredu_nn::Error::backend_source(error)),
                };
                session
                    .capture_external_target(input, pass, request, placements, context)
                    .map_err(|error| BackendFailure::from_error(error).into())
            },
            |error| BackendFailure::from_error(error).into(),
        );
        cache.restore_state(lane);
        result
    }

    fn apply_external_prediction_target_operation(
        &mut self,
        operation: crate::speculative::ExternalPredictionTargetOperation<'_, M::Tensor>,
    ) -> Result<M::Tensor, M::Error> {
        self.session
            .execute_external_target_operation(
                operation,
                &mut self.ingress.external_target,
                self.native.context(),
            )
            .map_err(|error| BackendFailure::from_error(error).into())
    }
}
