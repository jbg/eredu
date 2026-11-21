//! Portable assembly and parameter operations for an installed prediction extension.

use super::{
    session_erasure::TextErasureResources, text_invocation::TextInvocationMechanisms,
    BoundTextSession, NoSelectedPrediction, SelectedPrediction,
};
use crate::speculative::PredictionModule;
use crate::{
    parameter_operations::ParameterSlotOperation,
    speculative::{
        EmbeddedPredictionExecutor, EmbeddedPredictionObservers, MaterializedPredictionExecutor,
        PredictionExtensionMaterializer, PredictionModuleVisitor,
        ReplicatedMaterializedPredictionStrategy, ReplicatedPredictionInput,
        ReplicatedPredictionNative, SpeculativeTensorMechanisms,
    },
    LayeredArchitecture, ReplicatedTextExecutionStrategy, ReplicatedTextSession,
    ReplicatedTextSessionMechanisms, RuntimeState,
};
use eredu_core::backend::TextExecutableTypes;
use eredu_core::{
    speculative::{erasure::ErasedGenerationExecutor, GenerationExecutorTypes},
    BackendFailure, BackendFailureKind, SubmissionBackend,
};
use eredu_nn::{
    BlockwiseAttentionBackend, DistributedNeuralBackend, HyperNeuralBackend, ParameterSlotVisitor,
    Parameterized, Tensor, TensorParallelGroupedNeuralBackend,
};
use std::{collections::BTreeMap, convert::Infallible};

/// Operations on a bound prediction selection.
///
/// Implemented by the portable ordinary and prediction selection types.
pub trait TextExecutablePrediction<M: TextExecutableTypes, Model>: Sized {
    /// Borrows the selected prediction executor for one invocation.
    fn bind_prediction(
        _model: &mut Model,
    ) -> Option<
        Box<
            dyn ErasedGenerationExecutor<
                    M::EmbeddedExecutor,
                    crate::text_session_driver::PreparedTextPrompt<M::InputTensor>,
                > + '_,
        >,
    > {
        None
    }
    /// Visits loaded prediction parameter slots.
    fn visit_parameter_slots(
        &mut self,
        _visitor: &mut dyn eredu_nn::ParameterSlotVisitor<M::Tensor>,
    ) {
    }
    /// Borrows one selected parameter module under its native lease.
    fn with_parameter_slots(
        &mut self,
        _module: usize,
        _operation: &mut ParameterSlotOperation<'_, M::Tensor, M::Error>,
    ) -> Result<bool, M::Error> {
        Ok(false)
    }
    /// Installs completed parameter handles into the retained prediction modules.
    fn publish_parameter_replacements(
        &mut self,
        _values: &BTreeMap<String, M::Tensor>,
        _active: bool,
    ) {
    }
}

impl<M: TextExecutableTypes, Model> TextExecutablePrediction<M, Model> for NoSelectedPrediction {}

struct Slots<'a, T>(&'a mut dyn ParameterSlotVisitor<T>);

impl<'a, T: 'a> eredu_nn::ParameterVisitorMut<'a, T> for Slots<'_, T> {
    fn visit_mut(&mut self, metadata: eredu_nn::ParameterMetadata, value: &'a mut T) {
        self.0.visit_slot(metadata, value);
    }
}

/// Native tensor and input representations for an installed embedded predictor.
pub trait EmbeddedTextBinding: TextInvocationMechanisms {
    /// Native tensor operations used by the shared speculative executor.
    type Mechanisms: SpeculativeTensorMechanisms<Tensor = Self::Tensor> + 'static;

    /// Materializes the exact token range requested by portable prefill.
    fn slice_tokens(
        tokens: &Self::Tensor,
        range: std::ops::Range<usize>,
        context: &Self::Context,
    ) -> Result<Self::Tensor, <Self::Mechanisms as SpeculativeTensorMechanisms>::Error>;

    /// Completes one retained ingress tensor before the next native lease.
    fn settle_input(
        tensor: &Self::Tensor,
    ) -> Result<(), <Self::Mechanisms as SpeculativeTensorMechanisms>::Error>;

    /// Preserves the typed cause of an input-preparation failure.
    fn input_failure(
        error: BackendFailure,
    ) -> <Self::Mechanisms as SpeculativeTensorMechanisms>::Error;

    /// Borrows the selected observer representation.
    fn embedded_observers(
        observers: &mut Self::EmbeddedObservers,
    ) -> &mut EmbeddedPredictionObservers<
        Self::Tensor,
        <Self::Mechanisms as SpeculativeTensorMechanisms>::Logits,
        <Self::Mechanisms as SpeculativeTensorMechanisms>::Error,
    >;
}

struct VisitSlots<'a, T>(&'a mut dyn ParameterSlotVisitor<T>);
impl<B, X, E> PredictionModuleVisitor<B, X> for VisitSlots<'_, B::Tensor>
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + eredu_nn::GroupedNeuralBackend
        + HyperNeuralBackend,
    X: PredictionExtensionMaterializer<B, Error = E>,
{
    type Error = Infallible;

    fn visit<U: Parameterized<B::Tensor>>(
        &mut self,
        _: usize,
        module: &mut PredictionModule<B, X, U>,
    ) -> Result<(), Infallible> {
        module.inner.visit_parameters_mut(&mut Slots(self.0));
        Ok(())
    }
}

struct VisitSelectedSlots<'a, 'b, T, E> {
    selected: usize,
    found: bool,
    operation: &'a mut ParameterSlotOperation<'b, T, E>,
}

impl<B, X, E> PredictionModuleVisitor<B, X> for VisitSelectedSlots<'_, '_, B::Tensor, E>
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + eredu_nn::GroupedNeuralBackend
        + HyperNeuralBackend,
    X: PredictionExtensionMaterializer<B, Error = E>,
    E: From<BackendFailure>,
{
    type Error = E;
    fn visit<U: Parameterized<B::Tensor>>(
        &mut self,
        ordinal: usize,
        module: &mut PredictionModule<B, X, U>,
    ) -> Result<(), E> {
        if ordinal != self.selected {
            return Ok(());
        }
        if self.found {
            return Err(BackendFailure::new(
                BackendFailureKind::InvalidSession,
                std::io::Error::other("prediction module ordinal is duplicated"),
            )
            .into());
        }
        self.found = true;
        crate::speculative::with_prediction_module::<B, X, _, _, _>(module, None, |inner| {
            let result = (self.operation)(&mut |visitor| {
                inner.visit_parameters_mut(&mut Slots(visitor));
            });
            (result, Vec::new())
        })?
    }
}

struct PublishSlots<'a, T> {
    values: &'a BTreeMap<String, T>,
    active: bool,
}

impl<B, X, E> PredictionModuleVisitor<B, X> for PublishSlots<'_, B::Tensor>
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + eredu_nn::GroupedNeuralBackend
        + HyperNeuralBackend,
    X: PredictionExtensionMaterializer<B, Error = E>,
{
    type Error = Infallible;

    fn visit<U: Parameterized<B::Tensor>>(
        &mut self,
        _: usize,
        module: &mut PredictionModule<B, X, U>,
    ) -> Result<(), Infallible> {
        module.replacements.clear();
        if self.active {
            module
                .replacements
                .extend(module.parameters.iter().filter_map(|slot| {
                    let name = slot.parameter.id.as_str();
                    self.values
                        .get(name)
                        .map(|value| (name.to_owned(), value.clone()))
                }));
        }
        Ok(())
    }
}

impl<A, B, S, R, D, I, N, P, M, X>
    TextExecutablePrediction<M, BoundTextSession<ReplicatedTextSession<A, B, R, D>, I, N, Self>>
    for SelectedPrediction<P>
where
    B: SubmissionBackend<Executor = <<B as eredu_nn::NeuralBackend>::Tensor as Tensor>::Context>
        + BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + TensorParallelGroupedNeuralBackend
        + HyperNeuralBackend,
    S: RuntimeState<B> + 'static,
    A: LayeredArchitecture<B, S, Error = eredu_nn::Error>,
    R: ReplicatedTextSessionMechanisms<A, B, State = S>,
    R::PolicyError: std::error::Error + Send + Sync + 'static,
    R::Error: std::error::Error + Send + Sync + 'static,
    D: ReplicatedTextExecutionStrategy<A, B, S, R::ResidentPolicy, R::BoundedPolicy>,
    N: TextErasureResources<Types = M>,
    M: EmbeddedTextBinding<Tensor = B::Tensor, Context = B::Executor, PredictionMaterializer = X>,
    for<'a> super::prediction_ingress::PredictionIngress<'a, I, M>: ReplicatedPredictionInput<
        A,
        B,
        S,
        <M::Mechanisms as SpeculativeTensorMechanisms>::Error,
        Input = crate::text_session_driver::PreparedTextPrompt<M::InputTensor>,
    >,
    X: PredictionExtensionMaterializer<B, Error = M::Error>
        + 'static
        + PredictionExtensionMaterializer<B, Error = M::Error>
        + ReplicatedPredictionNative<B, S, M::Mechanisms, ExecutorTypes = M::EmbeddedExecutor>,
    P: MaterializedPredictionExecutor<A, B, X>,
    M::EmbeddedExecutor: GenerationExecutorTypes<
        Logits = <M::Mechanisms as SpeculativeTensorMechanisms>::Logits,
        Completion = <M::Mechanisms as SpeculativeTensorMechanisms>::Completion,
        Error = <M::Mechanisms as SpeculativeTensorMechanisms>::Error,
        Telemetry = <X as ReplicatedPredictionNative<B, S, M::Mechanisms>>::Telemetry,
    >,
{
    fn bind_prediction(
        model: &mut BoundTextSession<ReplicatedTextSession<A, B, R, D>, I, N, Self>,
    ) -> Option<
        Box<
            dyn ErasedGenerationExecutor<
                    M::EmbeddedExecutor,
                    crate::text_session_driver::PreparedTextPrompt<M::InputTensor>,
                > + '_,
        >,
    > {
        let (context, observers) = model.native.prediction_resources();
        let strategy =
            ReplicatedMaterializedPredictionStrategy::<A, B, S, R, D, P, _, X, M::Mechanisms>::new(
                &mut model.session,
                &mut model.prediction.extension,
                &model.prediction.selected,
                super::prediction_ingress::PredictionIngress::new(&model.ingress),
                context,
            );
        let observers = M::embedded_observers(observers);
        Some(EmbeddedPredictionExecutor::<_, M::Mechanisms>::bind::<
            M::EmbeddedExecutor,
        >(strategy, observers))
    }

    fn visit_parameter_slots(&mut self, visitor: &mut dyn ParameterSlotVisitor<M::Tensor>) {
        match self.extension.visit_modules(&mut VisitSlots(visitor)) {
            Ok(()) => {}
            Err(never) => match never {},
        }
    }

    fn with_parameter_slots(
        &mut self,
        selected: usize,
        operation: &mut ParameterSlotOperation<'_, M::Tensor, M::Error>,
    ) -> Result<bool, M::Error> {
        let mut visitor = VisitSelectedSlots {
            selected,
            found: false,
            operation,
        };
        self.extension.visit_modules(&mut visitor)?;
        Ok(visitor.found)
    }

    fn publish_parameter_replacements(
        &mut self,
        values: &BTreeMap<String, M::Tensor>,
        active: bool,
    ) {
        match self
            .extension
            .visit_modules(&mut PublishSlots { values, active })
        {
            Ok(()) => {}
            Err(never) => match never {},
        }
    }
}
