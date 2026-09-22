use super::*;

mod composite;
mod partitioned;
mod routed;

pub(crate) use composite::CompositeBindingVisitor;
pub(crate) use partitioned::{
    PartitionedCompositeBindingVisitor, PartitionedCompositePredictionBindingVisitor,
    PartitionedDenseDecoderBindingVisitor, PartitionedPoolingRoutedDecoderBindingVisitor,
    PartitionedPredictionBindingVisitor, PartitionedRoutedDecoderBindingVisitor,
};
use routed::selected_addressable_partition_bank;
pub(crate) use routed::shard_addressable_members;
pub(in crate::composition::mlx) use routed::{
    PoolingRoutedBindingVisitor, Relu2RoutedBindingVisitor, RoutedBindingVisitor,
};

/// A stack-only once-moved source handoff shared by mutually exclusive routes.
/// The architecture dispatcher selects the consumer; cloning a visitor cannot
/// duplicate this manager, its original source account, or its native owners.
pub(crate) type NativeConstructionSlot =
    std::cell::Cell<Option<crate::composition::mlx::loading::PreparedNativeConstructionSources>>;

/// Family-agnostic MLX visitor that binds neutral parameter topology.
pub(crate) struct BindingVisitor<'a> {
    pub stream: &'a Stream,
    pub weights_stream: &'a Stream,
    pub construction_sources: Option<&'a NativeConstructionSlot>,
}

pub(crate) type AddressableManagerSlot = std::cell::Cell<
    Option<crate::backend::runtime::residency::parameter_bank::PreparedAddressableSource>,
>;

pub(crate) struct PredictionBindingVisitor<'a> {
    pub addressable_manager: Option<&'a AddressableManagerSlot>,
    pub stream: &'a Stream,
    pub weights_stream: &'a Stream,
    pub construction_sources: Option<&'a NativeConstructionSlot>,
    pub selected: eredu_runtime::SelectedSpeculativeRealization,
    pub capability: eredu_architectures::capability::CapabilityEstimate,
}

#[derive(Clone, Copy)]
pub(super) struct OrdinaryReplicatedFinalizer;

pub(super) struct PredictionReplicatedFinalizer<P> {
    prediction: SelectedPrediction<P>,
    capability: eredu_architectures::capability::CapabilityEstimate,
}

pub(super) trait ReplicatedExecutableFinalizer<A, S>
where
    S: MlxStateMechanisms + 'static,
    A: ReplicatedTextArchitecture<MlxNeuralBackend, S, Error = eredu_nn::Error> + 'static,
    A::StaticModules: Clone,
    A::Error: std::fmt::Display,
{
    fn prediction_residency(
        &mut self,
    ) -> Result<
        crate::composition::mlx::replicated_text::prediction::parameters::PredictionResidency,
        Error,
    > {
        Ok(Default::default())
    }

    fn finish<D>(
        self,
        completed: CompletedReplicatedText<A, S, D>,
    ) -> Result<Box<dyn ErasedReplicatedTextExecutable>, Error>
    where
        D: eredu_runtime::ReplicatedTextExecutionStrategy<
                A,
                MlxNeuralBackend,
                S,
                MlxArchitectureLayerwisePolicy<A, S>,
                MlxArchitectureLayerwisePolicy<A, S>,
            > + MlxParameterBankTelemetry
            + 'static;
}

impl<A, S> ReplicatedExecutableFinalizer<A, S> for OrdinaryReplicatedFinalizer
where
    S: MlxStateMechanisms + 'static,
    A: ReplicatedTextArchitecture<MlxNeuralBackend, S, Error = eredu_nn::Error> + 'static,
    A::StaticModules: Clone,
    A::Error: std::fmt::Display,
{
    fn finish<D>(
        self,
        completed: CompletedReplicatedText<A, S, D>,
    ) -> Result<Box<dyn ErasedReplicatedTextExecutable>, Error>
    where
        D: eredu_runtime::ReplicatedTextExecutionStrategy<
                A,
                MlxNeuralBackend,
                S,
                MlxArchitectureLayerwisePolicy<A, S>,
                MlxArchitectureLayerwisePolicy<A, S>,
            > + MlxParameterBankTelemetry
            + 'static,
    {
        Ok(Box::new(completed))
    }
}

impl<A, S, P> ReplicatedExecutableFinalizer<A, S> for PredictionReplicatedFinalizer<P>
where
    S: MlxStateMechanisms + 'static,
    A: ReplicatedTextArchitecture<MlxNeuralBackend, S, Error = eredu_nn::Error> + 'static,
    A::StaticModules: Clone,
    A::Error: std::fmt::Display,
    P: eredu_architectures::prediction_extension::MaterializedPredictionExecutor<
            A,
            MlxNeuralBackend,
            MlxEmbeddedPredictionMaterializer,
        > + 'static,
{
    fn prediction_residency(
        &mut self,
    ) -> Result<
        crate::composition::mlx::replicated_text::prediction::parameters::PredictionResidency,
        Error,
    > {
        crate::composition::mlx::replicated_text::prediction::parameters::residency::<A, P>(
            &mut self.prediction.extension,
        )
    }

    fn finish<D>(
        self,
        completed: CompletedReplicatedText<A, S, D>,
    ) -> Result<Box<dyn ErasedReplicatedTextExecutable>, Error>
    where
        D: eredu_runtime::ReplicatedTextExecutionStrategy<
                A,
                MlxNeuralBackend,
                S,
                MlxArchitectureLayerwisePolicy<A, S>,
                MlxArchitectureLayerwisePolicy<A, S>,
            > + MlxParameterBankTelemetry
            + 'static,
    {
        completed
            .with_prediction(self.prediction, self.capability)
            .map(|completed| Box::new(completed) as Box<dyn ErasedReplicatedTextExecutable>)
    }
}

impl<S>
    eredu_architectures::replicated_text::ReplicatedPredictionTargetVisitor<
        MlxNeuralBackend,
        S,
        MlxEmbeddedPredictionMaterializer,
    > for PredictionBindingVisitor<'_>
where
    S: MlxStateMechanisms + 'static,
{
    type Output = Box<dyn ErasedReplicatedTextExecutable>;
    type Error = Error;

    fn visit<A>(
        self,
        prepared: PreparedReplicatedTextArchitecture<A>,
        extension: <A as eredu_architectures::prediction_extension::MaterializedPredictionTarget<
            MlxNeuralBackend,
        >>::Extension<MlxEmbeddedPredictionMaterializer>,
        store: eredu_checkpoint::store::RetainedCheckpointSource,
    ) -> Result<Self::Output, Self::Error>
    where
        A: ReplicatedTextArchitecture<MlxNeuralBackend, S, Error = eredu_nn::Error>
            + eredu_architectures::prediction_extension::MaterializedPredictionTarget<
                MlxNeuralBackend,
            > + 'static,
        A::StaticModules: Clone,
        A::Error: std::fmt::Display,
    {
        let mut prediction = SelectedPrediction {
            extension,
            selected: self.selected,
        };
        let residency =
            super::super::prediction::parameters::residency::<A, _>(&mut prediction.extension)?;
        CompletedReplicatedText::new_with_construction_sources(
            prepared,
            store,
            self.stream,
            self.weights_stream,
            residency,
            self.construction_sources.and_then(std::cell::Cell::take),
        )?
        .with_prediction(prediction, self.capability)
        .map(|model| Box::new(model) as Box<dyn ErasedReplicatedTextExecutable>)
    }
}

impl
    eredu_architectures::replicated_text::ReplicatedPredictionProfileDispatcher<
        MlxNeuralBackend,
        MlxEmbeddedPredictionMaterializer,
    > for PredictionBindingVisitor<'_>
{
    type Output = Box<dyn ErasedReplicatedTextExecutable>;
    type Error = Error;
    type State = MlxHybridState;
    type Visitor = Self;

    fn into_visitor(self) -> Self::Visitor {
        self
    }
}

impl
    eredu_architectures::routed_text::RoutedPredictionProfileDispatcher<
        MlxNeuralBackend,
        MlxEmbeddedPredictionMaterializer,
    > for PredictionBindingVisitor<'_>
{
    type Output = Box<dyn ErasedReplicatedTextExecutable>;
    type Error = Error;
    type GatedState = MlxHybridState;
    type PoolingState = MlxPoolingAttentionState;
    type GatedVisitor = Self;
    type PoolingVisitor = Self;

    fn into_gated_visitor(self) -> Self::GatedVisitor {
        self
    }

    fn into_pooling_visitor(self) -> Self::PoolingVisitor {
        self
    }
}
impl ReplicatedTextArchitectureVisitor<MlxNeuralBackend, MlxKeyValueState> for BindingVisitor<'_> {
    type Output = Box<dyn ErasedReplicatedTextExecutable>;
    type Error = Error;

    fn construction_started(&mut self) {
        #[cfg(test)]
        crate::tests::support::path_instrumentation::architecture_construction();
    }

    fn visit<A>(
        self,
        prepared: PreparedReplicatedTextArchitecture<A>,
        store: eredu_checkpoint::store::RetainedCheckpointSource,
    ) -> Result<Self::Output, Self::Error>
    where
        A: ReplicatedTextArchitecture<MlxNeuralBackend, MlxKeyValueState, Error = eredu_nn::Error>
            + 'static,
        A::StaticModules: Clone,
        A::Error: std::fmt::Display,
    {
        CompletedReplicatedText::new_with_construction_sources(
            prepared,
            store,
            self.stream,
            self.weights_stream,
            Default::default(),
            self.construction_sources.and_then(std::cell::Cell::take),
        )
        .map(|model| Box::new(model) as Box<dyn ErasedReplicatedTextExecutable>)
    }
}

impl ReplicatedTextArchitectureVisitor<MlxNeuralBackend, MlxHybridState> for BindingVisitor<'_> {
    type Output = Box<dyn ErasedReplicatedTextExecutable>;
    type Error = Error;

    fn construction_started(&mut self) {
        #[cfg(test)]
        crate::tests::support::path_instrumentation::architecture_construction();
    }

    fn visit<A>(
        self,
        prepared: PreparedReplicatedTextArchitecture<A>,
        store: eredu_checkpoint::store::RetainedCheckpointSource,
    ) -> Result<Self::Output, Self::Error>
    where
        A: ReplicatedTextArchitecture<MlxNeuralBackend, MlxHybridState, Error = eredu_nn::Error>
            + 'static,
        A::StaticModules: Clone,
        A::Error: std::fmt::Display,
    {
        CompletedReplicatedText::new_with_construction_sources(
            prepared,
            store,
            self.stream,
            self.weights_stream,
            Default::default(),
            self.construction_sources.and_then(std::cell::Cell::take),
        )
        .map(|model| Box::new(model) as Box<dyn ErasedReplicatedTextExecutable>)
    }
}

pub(crate) struct MlxReplicatedStateProfiles;

impl ReplicatedTextStateProfiles<MlxNeuralBackend> for MlxReplicatedStateProfiles {
    type StatelessState = MlxHybridState;
    type AttentionState = MlxKeyValueState;
    type ComponentState = MlxHybridState;
    type AttentionComponentState = MlxHybridState;
    type CompressedState = MlxHybridState;
    type CompressedComponentState = MlxHybridState;
}

#[cfg(all(
    test,
    feature = "metal",
    target_vendor = "apple",
    not(feature = "cuda")
))]
pub(crate) mod prefill_retention_fixture;
