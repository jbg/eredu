//! Portable visitor holding one admitted local prediction extension.

use std::marker::PhantomData;

use eredu_checkpoint::store::SharedCheckpointSource;
use eredu_core::SubmissionBackend;
use eredu_nn::{
    BlockwiseAttentionBackend, DistributedNeuralBackend, GroupedNeuralBackend, HyperNeuralBackend,
    NeuralBackend, Tensor, TensorParallelGroupedNeuralBackend,
};

use crate::{
    composite_execution::CompositeArchitecture,
    executable_binding::{
        embedded_erasure::EmbeddedTextBinding,
        routed_session::{
            BoundRetainedRoutedPrediction, BoundRoutedText, CompositePredictionTargetVisitor,
            NativeRoutedStorage, RetainedCompositePrediction, RetainedRoutedPredictionMechanisms,
            RetainedRoutedTextPrediction, RoutedBankBinding, RoutedPredictionProfileDispatcher,
            RoutedPredictionTargetVisitor, RoutedSessionMechanisms, RoutedTextResources,
        },
        session_erasure::{BankOf, LocalTextTypes, TextBankTelemetry},
        text_invocation::TextPredictionLane,
        BoundTextSession, CompositeSessionIngress, InstalledTextExecutable,
        LocalPredictionResources, LocalTextResources, PredictionBinding,
        ReplicatedPredictionProfileDispatcher, ReplicatedPredictionTargetVisitor,
        SelectedPrediction,
    },
    speculative::{
        PredictionExtensionMaterializer, PredictionTarget, PreparedRoutedPrediction,
        ReplicatedPredictionNative, SpeculativeTensorMechanisms,
    },
    LayerRuntimeState, LayeredArchitecture, ParameterProvider, ParameterProviders,
    ReplicatedTextArchitecture, ReplicatedTextSession, RoutedBankProviders,
    RoutedLayeredArchitecture, RuntimeState,
};
use eredu_core::backend::TextExecutableTypes;

use super::local::{Addressable, Failure};

/// The selected prediction facts stay with portable dispatch until installation.
pub struct LocalPrediction<N, G, P> {
    resources: N,
    prediction: PredictionBinding,
    profiles: PhantomData<fn() -> (G, P)>,
}

impl<N, G, P> LocalPrediction<N, G, P> {
    pub(super) fn new(resources: N, prediction: PredictionBinding) -> Self {
        Self {
            resources,
            prediction,
            profiles: PhantomData,
        }
    }
}

impl<N: LocalTextResources, G, P> LocalTextResources for LocalPrediction<N, G, P> {
    type LocalResources = N::LocalResources;

    fn local_resources(&self) -> Self::LocalResources {
        self.resources.local_resources()
    }
}

impl<N: LocalTextResources, G, P> LocalPredictionResources for LocalPrediction<N, G, P> {
    fn into_prediction_binding(self) -> PredictionBinding {
        self.prediction
    }
}

impl<B, S, N, G, P> RoutedSessionMechanisms<B, S> for LocalPrediction<N, G, P>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    S: RuntimeState<B>,
    N: RoutedSessionMechanisms<B, S>,
{
    type WeightStorage = N::WeightStorage;
    type Mechanisms<A>
        = N::Mechanisms<A>
    where
        A: LayeredArchitecture<B, S, Error = eredu_nn::Error> + 'static;

    fn session_context(&self) -> &B::Executor {
        self.resources.session_context()
    }

    fn session_mechanisms<A>(&self, source: SharedCheckpointSource) -> Self::Mechanisms<A>
    where
        A: LayeredArchitecture<B, S, Error = eredu_nn::Error> + 'static,
    {
        self.resources.session_mechanisms(source)
    }
}

impl<B, N, G, P> NativeRoutedStorage<B> for LocalPrediction<N, G, P>
where
    B: GroupedNeuralBackend
        + DistributedNeuralBackend
        + eredu_core::checkpoint_mechanisms::CheckpointParameterBackend,
    N: NativeRoutedStorage<B>,
{
    type Movement = N::Movement;
    type Decoder = N::Decoder;
    type RowBinding = N::RowBinding;
    type StorageError = N::StorageError;
    type BankParameters = N::BankParameters;
    type BankMechanisms = N::BankMechanisms;

    fn row_lookup_support(&self) -> impl eredu_core::preparation::rows::NativeRowLookupSupport {
        self.resources.row_lookup_support()
    }

    fn bank_residency_context(
        &self,
    ) -> <Self::BankMechanisms as eredu_core::residency::ResidencyStorageMechanisms>::Context {
        self.resources.bank_residency_context()
    }

    fn bank_materializer(
        &self,
    ) -> impl eredu_core::checkpoint_mechanisms::CheckpointEncodingMaterializer<
        Parameters = Self::BankParameters,
    > {
        self.resources.bank_materializer()
    }

    fn indexed_movement(&self) -> Self::Movement {
        self.resources.indexed_movement()
    }

    fn row_binding(&self) -> Self::RowBinding {
        self.resources.row_binding()
    }
}

type Embedded<N> = <LocalTextTypes<N> as EmbeddedTextBinding>::Mechanisms;
type Scheduler<N> = <LocalTextTypes<N> as TextExecutableTypes>::EmbeddedExecutor;

impl<B, S, X, N, G, H> ReplicatedPredictionTargetVisitor<B, S, X> for LocalPrediction<N, G, H>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>
        + BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + TensorParallelGroupedNeuralBackend
        + HyperNeuralBackend
        + Clone
        + 'static,
    B::Executor: Sized,
    S: RuntimeState<B> + 'static,
    N: RoutedSessionMechanisms<
        B,
        S,
        WeightStorage = <X as eredu_core::checkpoint_mechanisms::prediction::NativePredictionMaterialization<B>>::ResidencyMechanisms,
        LocalResources: super::super::text_invocation::TextInvocationResources<Types: EmbeddedTextBinding<PredictionMaterializer = X>> + 'static,
    >,
    <LocalTextTypes<N> as TextExecutableTypes>::PredictionState:
        TextPredictionLane<S, Error = Failure<N>>,
    X: PredictionExtensionMaterializer<B, Error = Failure<N>>
        + 'static
        + ReplicatedPredictionNative<
            B,
            S,
            Embedded<N>,
            ExecutorTypes = Scheduler<N>,
        >,
    Scheduler<N>: eredu_core::speculative::GenerationExecutorTypes<
        Logits = <Embedded<N> as SpeculativeTensorMechanisms>::Logits,
        Completion = <Embedded<N> as SpeculativeTensorMechanisms>::Completion,
        Error = <Embedded<N> as SpeculativeTensorMechanisms>::Error,
        Telemetry = (),
    >,
{
    type Output = InstalledTextExecutable<LocalTextTypes<N>>;
    type Error = Failure<N>;
    type Mechanisms<A>
        = N::Mechanisms<A>
    where
        A: ReplicatedTextArchitecture<B, S, Error = eredu_nn::Error>
            + PredictionTarget<B>
            + 'static,
        A::StaticModules: Clone;

    fn context(&self) -> &B::Executor {
        self.resources.session_context()
    }

    fn prediction_mechanisms<A>(
        &self,
        extension: &mut A::Extension<X>,
        store: SharedCheckpointSource,
    ) -> Result<(Self::Mechanisms<A>, crate::ReplicatedWeightOptions<N::WeightStorage>), Self::Error>
    where
        A: ReplicatedTextArchitecture<B, S, Error = eredu_nn::Error>
            + PredictionTarget<B>
            + 'static,
        A::StaticModules: Clone,
    {
        let mechanisms = self.resources.session_mechanisms(store);
        let residency = crate::speculative::prepare_prediction_residency::<A, B, X, _>(extension)?;
        Ok((mechanisms, crate::ReplicatedWeightOptions { supplementary: residency, ..Default::default() }))
    }

    type BoundPrediction<A>
        = BoundTextSession<
        ReplicatedTextSession<A, B, N::Mechanisms<A>>,
        (),
        N::LocalResources,
        SelectedPrediction<A::Extension<X>>,
    >
    where
        A: ReplicatedTextArchitecture<B, S, Error = eredu_nn::Error>
            + PredictionTarget<B>
            + 'static,
        A::StaticModules: Clone;
}

impl<B, S, X, N, G, H, P> RoutedPredictionTargetVisitor<B, S, X, P> for LocalPrediction<N, G, H>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>
        + BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + TensorParallelGroupedNeuralBackend
        + HyperNeuralBackend
        + Clone
        + 'static,
    B::Executor: Sized,
    S: LayerRuntimeState<B> + 'static,
    N: RoutedSessionMechanisms<
        B,
        S,
        WeightStorage = <X as eredu_core::checkpoint_mechanisms::prediction::NativePredictionMaterialization<B>>::ResidencyMechanisms,
        LocalResources: super::super::text_invocation::TextInvocationResources<Types: EmbeddedTextBinding<PredictionMaterializer = X>> + 'static,
    >,
    <LocalTextTypes<N> as TextExecutableTypes>::PredictionState:
        TextPredictionLane<S, Error = Failure<N>>,
    X: PredictionExtensionMaterializer<B, Error = Failure<N>>
        + 'static
        + ReplicatedPredictionNative<
            B,
            S,
            Embedded<N>,
            ExecutorTypes = Scheduler<N>,
        >,
    Scheduler<N>: eredu_core::speculative::GenerationExecutorTypes<
        Logits = <Embedded<N> as SpeculativeTensorMechanisms>::Logits,
        Completion = <Embedded<N> as SpeculativeTensorMechanisms>::Completion,
        Error = <Embedded<N> as SpeculativeTensorMechanisms>::Error,
        Telemetry = (),
    >,
    N: RoutedTextResources<B, RoutedResources = N::LocalResources, StorageError = Failure<N>>,
    Self: RoutedTextResources<B, RoutedResources = N::LocalResources, StorageError = Failure<N>,
        Bank = N::Bank, Movement = N::Movement, Rows = N::Rows>,
    P: RoutedBankBinding<Resident: ParameterProvider<B> + 'static>,
    N::Rows: 'static,
    Addressable<B, P, N>: 'static,
    ParameterProviders<RoutedBankProviders<P::Resident>, Option<N::Rows>>:
        TextBankTelemetry<B, BankOf<N::LocalResources>>,
    ParameterProviders<RoutedBankProviders<Addressable<B, P, N>>, Option<N::Rows>>:
        TextBankTelemetry<B, BankOf<N::LocalResources>>,
{
    type Output = InstalledTextExecutable<LocalTextTypes<N>>;
    type Error = Failure<N>;

    fn prediction_weight_options<A>(
        &self,
        extension: &mut A::Extension<X>,
    ) -> Result<crate::ReplicatedWeightOptions<N::WeightStorage>, Self::Error>
    where
        A: ReplicatedTextArchitecture<B, S, Error = eredu_nn::Error>
            + PredictionTarget<B>
            + 'static,
    {
        let residency = crate::speculative::prepare_prediction_residency::<A, B, X, _>(extension)?;
        Ok(crate::ReplicatedWeightOptions { supplementary: residency, ..Default::default() })
    }

    type BoundRoutedPrediction<A>
        = BoundRoutedText<B, A, P, N::Mechanisms<A>, Self, (), SelectedPrediction<A::Extension<X>>>
    where
        A: ReplicatedTextArchitecture<B, S, Error = eredu_nn::Error>
            + RoutedLayeredArchitecture<B, S>
            + PredictionTarget<B>
            + 'static,
        A::StaticModules: Clone;
}

impl<B, S, X, N, G, H, P, R, Q> CompositePredictionTargetVisitor<B, S, X, P, R, Q>
    for LocalPrediction<N, G, H>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>
        + BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + TensorParallelGroupedNeuralBackend
        + HyperNeuralBackend
        + Clone
        + 'static,
    B::Executor: Sized,
    S: LayerRuntimeState<B> + 'static,
    N: RoutedSessionMechanisms<
        B,
        S,
        WeightStorage = <X as eredu_core::checkpoint_mechanisms::prediction::NativePredictionMaterialization<B>>::ResidencyMechanisms,
        LocalResources: super::super::text_invocation::TextInvocationResources<Types: EmbeddedTextBinding<PredictionMaterializer = X>> + 'static,
    >,
    <LocalTextTypes<N> as TextExecutableTypes>::PredictionState:
        TextPredictionLane<S, Error = Failure<N>>,
    X: PredictionExtensionMaterializer<B, Error = Failure<N>>
        + 'static
        + ReplicatedPredictionNative<
            B,
            S,
            Embedded<N>,
            ExecutorTypes = Scheduler<N>,
        >,
    Scheduler<N>: eredu_core::speculative::GenerationExecutorTypes<
        Logits = <Embedded<N> as SpeculativeTensorMechanisms>::Logits,
        Completion = <Embedded<N> as SpeculativeTensorMechanisms>::Completion,
        Error = <Embedded<N> as SpeculativeTensorMechanisms>::Error,
        Telemetry = (),
    >,
    N: RoutedTextResources<B, RoutedResources = N::LocalResources, StorageError = Failure<N>>,
    Self: RoutedTextResources<B, RoutedResources = N::LocalResources, StorageError = Failure<N>,
        Bank = N::Bank, Movement = N::Movement, Rows = N::Rows>,
    P: RoutedBankBinding<Resident: ParameterProvider<B> + 'static>,
    N::Rows: 'static,
    Addressable<B, P, N>: 'static,
    ParameterProviders<RoutedBankProviders<P::Resident>, Option<N::Rows>>:
        TextBankTelemetry<B, BankOf<N::LocalResources>>,
    ParameterProviders<RoutedBankProviders<Addressable<B, P, N>>, Option<N::Rows>>:
        TextBankTelemetry<B, BankOf<N::LocalResources>>,
{
    type Output = InstalledTextExecutable<LocalTextTypes<N>>;
    type Error = Failure<N>;

    fn prediction_weight_options<A>(
        &self,
        extension: &mut A::Extension<X>,
    ) -> Result<crate::ReplicatedWeightOptions<N::WeightStorage>, Self::Error>
    where
        A: CompositeArchitecture<B, S, Error = eredu_nn::Error> + PredictionTarget<B>,
    {
        let residency = crate::speculative::prepare_prediction_residency::<A, B, X, _>(extension)?;
        Ok(crate::ReplicatedWeightOptions { supplementary: residency, ..Default::default() })
    }

    type BoundCompositePrediction<A>
        = BoundTextSession<
        ReplicatedTextSession<A, B, N::Mechanisms<A>>,
        CompositeSessionIngress<A::AdmissionConfig>,
        N::LocalResources,
        SelectedPrediction<A::Extension<X>>,
    >
    where
        A: CompositeArchitecture<B, S, Error = eredu_nn::Error> + PredictionTarget<B>,
        A::InputPartPlan: 'static;

    type BoundCompositeRoutedPrediction<A>
        = BoundRoutedText<
        B,
        A,
        P,
        N::Mechanisms<A>,
        Self,
        CompositeSessionIngress<A::AdmissionConfig>,
        SelectedPrediction<A::Extension<X>>,
    >
    where
        A: CompositeArchitecture<B, S, Error = eredu_nn::Error>
            + RoutedLayeredArchitecture<B, S>
            + PredictionTarget<B>,
        A::InputPartPlan: 'static,
        A::StaticModules: Clone;
}

impl<B, G, H, X, N, O, E> ReplicatedPredictionProfileDispatcher<B, X> for LocalPrediction<N, G, H>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>
        + BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + TensorParallelGroupedNeuralBackend
        + HyperNeuralBackend,
    G: LayerRuntimeState<B>,
    X: PredictionExtensionMaterializer<B>,
    Self: ReplicatedPredictionTargetVisitor<B, G, X, Output = O, Error = E>,
    E: From<eredu_core::parameters::ParameterError>,
{
    type Output = O;
    type Error = E;
    type State = G;
    type Visitor = Self;

    fn into_visitor(self) -> Self::Visitor {
        self
    }
}

impl<B, G, H, X, N, P, O, E> RoutedPredictionProfileDispatcher<B, X, P> for LocalPrediction<N, G, H>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>
        + BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + GroupedNeuralBackend
        + HyperNeuralBackend,
    G: LayerRuntimeState<B>,
    H: LayerRuntimeState<B>,
    X: PredictionExtensionMaterializer<B>,
    P: RoutedBankBinding<Resident: ParameterProvider<B>>,
    LocalPrediction<N, G, ()>: RoutedPredictionTargetVisitor<B, G, X, P, Output = O, Error = E>,
    LocalPrediction<N, (), H>: RoutedPredictionTargetVisitor<B, H, X, P, Output = O, Error = E>,
    E: std::fmt::Display,
{
    type Output = O;
    type Error = E;
    type GatedState = G;
    type PoolingState = H;
    type GatedVisitor = LocalPrediction<N, G, ()>;
    type PoolingVisitor = LocalPrediction<N, (), H>;

    fn into_gated_visitor(self) -> Self::GatedVisitor {
        LocalPrediction::new(self.resources, self.prediction)
    }

    fn into_pooling_visitor(self) -> Self::PoolingVisitor {
        LocalPrediction::new(self.resources, self.prediction)
    }
}

impl<B, S, X, N, P> RetainedRoutedTextPrediction<B, S, P> for N
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>
        + BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + TensorParallelGroupedNeuralBackend
        + HyperNeuralBackend
        + Clone
        + 'static,
    B::Executor: Sized,
    S: LayerRuntimeState<B>
        + crate::PredictionModelState<B, LayerState: crate::RuntimeAppendStreams<B>>
        + 'static,
    N: RoutedSessionMechanisms<
        B,
        S,
        LocalResources: super::super::text_invocation::TextInvocationResources<
            Types: EmbeddedTextBinding<PredictionMaterializer = X>,
        > + 'static,
    >,
    <LocalTextTypes<N> as TextExecutableTypes>::PredictionState:
        TextPredictionLane<S, Error = Failure<N>>,
    X: PredictionExtensionMaterializer<B, Error = Failure<N>>
        + 'static
        + PredictionExtensionMaterializer<B, ModelState = S>
        + ReplicatedPredictionNative<B, S, Embedded<N>, ExecutorTypes = Scheduler<N>>,
    Scheduler<N>: eredu_core::speculative::GenerationExecutorTypes<
        Logits = <Embedded<N> as SpeculativeTensorMechanisms>::Logits,
        Completion = <Embedded<N> as SpeculativeTensorMechanisms>::Completion,
        Error = <Embedded<N> as SpeculativeTensorMechanisms>::Error,
        Telemetry = (),
    >,
    N: RetainedRoutedPredictionMechanisms<B, S, Materializer = X>
        + RoutedTextResources<B, RoutedResources = N::LocalResources, StorageError = Failure<N>>,
    P: RoutedBankBinding<
        Resident: crate::TensorParallelParameterProvider<B, Error: std::fmt::Display> + 'static,
    >,
    N::Rows: 'static,
    Addressable<B, P, N>:
        crate::TensorParallelParameterProvider<B, Error: std::fmt::Display> + 'static,
    ParameterProviders<RoutedBankProviders<P::Resident>, Option<N::Rows>>:
        TextBankTelemetry<B, BankOf<N::LocalResources>>,
    ParameterProviders<RoutedBankProviders<Addressable<B, P, N>>, Option<N::Rows>>:
        TextBankTelemetry<B, BankOf<N::LocalResources>>,
{
    type BoundRetainedPrediction<A, W>
        = BoundRetainedRoutedPrediction<B, S, A, P, W, N, ()>
    where
        A: ReplicatedTextArchitecture<B, S, Error = eredu_nn::Error>
            + RoutedLayeredArchitecture<B, S>
            + 'static,
        A::StaticModules: Clone,
        W: PreparedRoutedPrediction<B, A>;
}

impl<B, S, X, N, P> RetainedCompositePrediction<B, S, P> for N
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>
        + BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + TensorParallelGroupedNeuralBackend
        + HyperNeuralBackend
        + Clone
        + 'static,
    B::Executor: Sized,
    S: LayerRuntimeState<B>
        + crate::PredictionModelState<B, LayerState: crate::RuntimeAppendStreams<B>>
        + 'static,
    N: RoutedSessionMechanisms<
        B,
        S,
        LocalResources: super::super::text_invocation::TextInvocationResources<
            Types: EmbeddedTextBinding<PredictionMaterializer = X>,
        > + 'static,
    >,
    <LocalTextTypes<N> as TextExecutableTypes>::PredictionState:
        TextPredictionLane<S, Error = Failure<N>>,
    X: PredictionExtensionMaterializer<B, Error = Failure<N>>
        + 'static
        + PredictionExtensionMaterializer<B, ModelState = S>
        + ReplicatedPredictionNative<B, S, Embedded<N>, ExecutorTypes = Scheduler<N>>,
    Scheduler<N>: eredu_core::speculative::GenerationExecutorTypes<
        Logits = <Embedded<N> as SpeculativeTensorMechanisms>::Logits,
        Completion = <Embedded<N> as SpeculativeTensorMechanisms>::Completion,
        Error = <Embedded<N> as SpeculativeTensorMechanisms>::Error,
        Telemetry = (),
    >,
    N: RetainedRoutedPredictionMechanisms<B, S, Materializer = X>
        + RoutedTextResources<B, RoutedResources = N::LocalResources, StorageError = Failure<N>>,
    P: RoutedBankBinding<
        Resident: crate::TensorParallelParameterProvider<B, Error: std::fmt::Display> + 'static,
    >,
    N::Rows: 'static,
    Addressable<B, P, N>:
        crate::TensorParallelParameterProvider<B, Error: std::fmt::Display> + 'static,
    ParameterProviders<RoutedBankProviders<P::Resident>, Option<N::Rows>>:
        TextBankTelemetry<B, BankOf<N::LocalResources>>,
    ParameterProviders<RoutedBankProviders<Addressable<B, P, N>>, Option<N::Rows>>:
        TextBankTelemetry<B, BankOf<N::LocalResources>>,
{
    type Published = InstalledTextExecutable<LocalTextTypes<N>>;

    type BoundCompositeRetainedPrediction<A, W>
        =
        BoundRetainedRoutedPrediction<B, S, A, P, W, N, CompositeSessionIngress<A::AdmissionConfig>>
    where
        A: CompositeArchitecture<B, S, Error = eredu_nn::Error> + RoutedLayeredArchitecture<B, S>,
        A::InputPartPlan: 'static,
        A::StaticModules: Clone,
        W: PreparedRoutedPrediction<B, A>;
}
