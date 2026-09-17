//! Routed target slots use the same cold parameter collector.
use super::*;
use crate::prediction_extension::MaterializedPredictionTarget;
use crate::routed_text::{
    PreparedRoutedTextArchitecture, Relu2RoutedTextArchitectureVisitor,
    RoutedPredictionProfileDispatcher, RoutedPredictionTargetVisitor,
    RoutedTextArchitectureVisitor,
};

impl RoutedTextArchitectureVisitor<WorkspaceBackend, ResidentState> for DestinationVisitor<'_> {
    type Output = ReplicatedTextBindingDestinations;
    type Error = Error;
    fn visit<A>(
        self,
        prepared: PreparedRoutedTextArchitecture<A>,
        _: RetainedCheckpointSource,
    ) -> Result<Self::Output, Error>
    where
        A: ReplicatedTextArchitecture<WorkspaceBackend, ResidentState, Error = Error>
            + eredu_runtime::RoutedLayeredArchitecture<WorkspaceBackend, ResidentState>
            + 'static,
        A::StaticModules: Clone,
        A::Error: std::fmt::Display,
    {
        let (modules, _, _) = prepared.into_shared_parts();
        self.collect_modules(modules)
    }
}
impl Relu2RoutedTextArchitectureVisitor<WorkspaceBackend, ResidentState>
    for DestinationVisitor<'_>
{
    type Output = ReplicatedTextBindingDestinations;
    type Error = Error;
    fn visit<A>(
        self,
        prepared: PreparedRoutedTextArchitecture<A>,
        _: RetainedCheckpointSource,
    ) -> Result<Self::Output, Error>
    where
        A: ReplicatedTextArchitecture<WorkspaceBackend, ResidentState, Error = Error>
            + eredu_runtime::RoutedLayeredArchitecture<WorkspaceBackend, ResidentState>
            + 'static,
        A::StaticModules: Clone,
        A::Error: std::fmt::Display,
    {
        let (modules, _, _) = prepared.into_shared_parts();
        self.collect_modules(modules)
    }
}
impl RoutedPredictionProfileDispatcher<WorkspaceBackend, ProjectionMaterializer>
    for DestinationVisitor<'_>
{
    type Output = ReplicatedTextBindingDestinations;
    type Error = Error;
    type GatedState = ResidentState;
    type PoolingState = ResidentState;
    type GatedVisitor = Self;
    type PoolingVisitor = Self;
    fn into_gated_visitor(self) -> Self {
        self
    }
    fn into_pooling_visitor(self) -> Self {
        self
    }
}
impl RoutedPredictionTargetVisitor<WorkspaceBackend, ResidentState, ProjectionMaterializer>
    for DestinationVisitor<'_>
{
    type Output = ReplicatedTextBindingDestinations;
    type Error = Error;
    fn visit<A>(
        self,
        prepared: PreparedRoutedTextArchitecture<A>,
        extension: <A as MaterializedPredictionTarget<WorkspaceBackend>>::Extension<
            ProjectionMaterializer,
        >,
        _: RetainedCheckpointSource,
    ) -> Result<Self::Output, Error>
    where
        A: ReplicatedTextArchitecture<WorkspaceBackend, ResidentState, Error = Error>
            + eredu_runtime::RoutedLayeredArchitecture<WorkspaceBackend, ResidentState>
            + MaterializedPredictionTarget<WorkspaceBackend>
            + 'static,
        A::StaticModules: Clone,
        A::Error: std::fmt::Display,
    {
        let (modules, _, _) = prepared.into_shared_parts();
        self.collect_prediction(modules, extension)
    }
}
