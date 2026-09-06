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
use routed::{selected_addressable_bank, selected_addressable_partition_bank};
pub(in crate::composition::mlx) use routed::{
    PoolingRoutedBindingVisitor, Relu2RoutedBindingVisitor, RoutedBindingVisitor,
};

/// Family-agnostic MLX visitor that binds neutral parameter topology.
#[derive(Clone, Copy)]
pub(crate) struct BindingVisitor<'a> {
    pub stream: &'a Stream,
    pub weights_stream: &'a Stream,
}

pub(crate) struct PredictionBindingVisitor<'a> {
    pub stream: &'a Stream,
    pub weights_stream: &'a Stream,
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
        store: Arc<dyn CheckpointSource>,
    ) -> Result<Self::Output, Self::Error>
    where
        A: ReplicatedTextArchitecture<MlxNeuralBackend, S, Error = eredu_nn::Error>
            + eredu_architectures::prediction_extension::MaterializedPredictionTarget<
                MlxNeuralBackend,
            > + 'static,
        A::StaticModules: Clone,
        A::Error: std::fmt::Display,
    {
        let prediction = SelectedPrediction {
            extension,
            selected: self.selected,
        };
        CompletedReplicatedText::new(prepared, store, self.stream, self.weights_stream)?
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
        store: Arc<dyn CheckpointSource>,
    ) -> Result<Self::Output, Self::Error>
    where
        A: ReplicatedTextArchitecture<MlxNeuralBackend, MlxKeyValueState, Error = eredu_nn::Error>
            + 'static,
        A::StaticModules: Clone,
        A::Error: std::fmt::Display,
    {
        CompletedReplicatedText::new(prepared, store, self.stream, self.weights_stream)
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
        store: Arc<dyn CheckpointSource>,
    ) -> Result<Self::Output, Self::Error>
    where
        A: ReplicatedTextArchitecture<MlxNeuralBackend, MlxHybridState, Error = eredu_nn::Error>
            + 'static,
        A::StaticModules: Clone,
        A::Error: std::fmt::Display,
    {
        CompletedReplicatedText::new(prepared, store, self.stream, self.weights_stream)
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
