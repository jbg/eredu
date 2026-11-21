//! Portable local visitors over native resource factories.

use eredu_checkpoint::store::SharedCheckpointSource;
use eredu_core::{BackendFailure, SubmissionBackend};
use eredu_nn::{DistributedNeuralBackend, GroupedNeuralBackend, NeuralBackend, Tensor};

use crate::executable_binding::{
    routed_session::{
        BoundRoutedText, CompositeTextArchitectureVisitor, RoutedBankBinding,
        RoutedSessionMechanisms, RoutedStorage, RoutedTextArchitectureVisitor, RoutedTextResources,
    },
    session_erasure::{BankOf, LocalTextTypes, TextBankTelemetry},
    text_invocation::TextPredictionLane,
    BoundTextSession, CompositeSessionIngress, InstalledTextExecutable, NoSelectedPrediction,
    ReplicatedTextArchitectureVisitor,
};
use crate::{
    composite_execution::CompositeArchitecture, ParameterProvider, ParameterProviders,
    ReplicatedTextArchitecture, ReplicatedTextSession, RoutedBankProviders,
    RoutedLayeredArchitecture, RuntimeState,
};
use eredu_core::backend::TextExecutableTypes;

impl<B, S, N> ReplicatedTextArchitectureVisitor<B, S> for N
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context> + 'static,
    B::Executor: Sized,
    S: RuntimeState<B> + 'static,
    N: RoutedSessionMechanisms<B, S, LocalResources: 'static>,
    <LocalTextTypes<N> as TextExecutableTypes>::PredictionState:
        TextPredictionLane<S, Error = <LocalTextTypes<N> as TextExecutableTypes>::Error>,
{
    type Output = InstalledTextExecutable<LocalTextTypes<N>>;
    type Error = <LocalTextTypes<N> as TextExecutableTypes>::Error;
    type Mechanisms<A>
        = N::Mechanisms<A>
    where
        A: ReplicatedTextArchitecture<B, S, Error = eredu_nn::Error> + 'static,
        A::StaticModules: Clone;

    fn construction_started(&mut self) {
        self.observe_construction();
    }

    fn context(&self) -> &B::Executor {
        self.session_context()
    }

    fn mechanisms<A>(&self, store: SharedCheckpointSource) -> Self::Mechanisms<A>
    where
        A: ReplicatedTextArchitecture<B, S, Error = eredu_nn::Error> + 'static,
        A::StaticModules: Clone,
    {
        self.session_mechanisms(store)
    }

    type BoundReplicated<A>
        = BoundTextSession<
        ReplicatedTextSession<A, B, N::Mechanisms<A>>,
        (),
        N::LocalResources,
        NoSelectedPrediction,
    >
    where
        A: ReplicatedTextArchitecture<B, S, Error = eredu_nn::Error> + 'static,
        A::StaticModules: Clone;

    fn construction_failure(error: String) -> Self::Error {
        BackendFailure::from_error(std::io::Error::other(error)).into()
    }
}

pub(super) type Failure<N> = <LocalTextTypes<N> as TextExecutableTypes>::Error;

pub(super) type Addressable<B, P, N> = <P as RoutedBankBinding>::Addressable<
    B,
    <N as RoutedStorage<B>>::Bank,
    <N as RoutedStorage<B>>::Movement,
>;

impl<B, S, N, P> RoutedTextArchitectureVisitor<B, S, P> for N
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>
        + GroupedNeuralBackend
        + 'static,
    B::Executor: Sized,
    S: crate::LayerRuntimeState<B> + 'static,
    N: RoutedSessionMechanisms<B, S, LocalResources: 'static>
        + RoutedTextResources<B, RoutedResources = N::LocalResources, StorageError = Failure<N>>,
    <LocalTextTypes<N> as TextExecutableTypes>::PredictionState:
        TextPredictionLane<S, Error = Failure<N>>,
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

    fn construction_started(&mut self) {
        self.observe_construction();
    }

    type BoundRouted<A>
        = BoundRoutedText<B, A, P, N::Mechanisms<A>, N, ()>
    where
        A: ReplicatedTextArchitecture<B, S, Error = eredu_nn::Error>
            + RoutedLayeredArchitecture<B, S>
            + 'static,
        A::StaticModules: Clone;
}

impl<B, S, N, P, R, Q> CompositeTextArchitectureVisitor<B, S, P, R, Q> for N
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>
        + GroupedNeuralBackend
        + DistributedNeuralBackend
        + Clone
        + 'static,
    B::Executor: Sized,
    S: crate::LayerRuntimeState<B> + 'static,
    N: RoutedSessionMechanisms<B, S, LocalResources: 'static>
        + RoutedTextResources<B, RoutedResources = N::LocalResources, StorageError = Failure<N>>,
    <LocalTextTypes<N> as TextExecutableTypes>::PredictionState:
        TextPredictionLane<S, Error = Failure<N>>,
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

    fn construction_started(&mut self) {
        self.observe_construction();
    }

    type BoundComposite<A>
        = BoundTextSession<
        ReplicatedTextSession<A, B, N::Mechanisms<A>>,
        CompositeSessionIngress<A::AdmissionConfig>,
        N::LocalResources,
        NoSelectedPrediction,
    >
    where
        A: CompositeArchitecture<B, S, Error = eredu_nn::Error> + 'static,
        A::InputPartPlan: 'static;

    type BoundCompositeRouted<A>
        = BoundRoutedText<B, A, P, N::Mechanisms<A>, N, CompositeSessionIngress<A::AdmissionConfig>>
    where
        A: CompositeArchitecture<B, S, Error = eredu_nn::Error> + RoutedLayeredArchitecture<B, S>,
        A::InputPartPlan: 'static,
        A::StaticModules: Clone;
}
