//! Cold physical prediction destinations through the shared materializer.
use super::*;
use eredu_core::cache::LayerCachePolicy;
use crate::prediction_extension::{
    MaterializedPredictionExecutor, MaterializedPredictionExtension, MaterializedPredictionTarget,
    PredictionExtensionMaterializer, PredictionModuleRole, PredictionModuleVisitor,
    PreparedPredictionExtension, PreparedPredictionUnit,
};
use crate::replicated_text::{
    CompositePredictionTargetVisitor, ReplicatedPredictionProfileDispatcher,
    ReplicatedPredictionTargetVisitor,
};
use eredu_runtime::working_memory::{WorkspacePoolingLayerState, WorkspacePoolingStateFactory};
use eredu_runtime::{
    ArchitectureStateFactory, LocalModelLayout,
    ReplicatedTextMaterializationTask, StateLayout,
};
use std::{num::NonZeroU32, sync::Arc};

/// One physical module, in the same ordinal space as native residency binding.
/// These cold metadata rows confer no payload, native or execution authority.
pub struct PredictionBindingDestination {
    pub ordinal: usize,
    pub parameters: BTreeMap<String, WorkspaceLayout>,
    pub tasks: Arc<Vec<ReplicatedTextMaterializationTask>>,
    pub layout: Option<LocalModelLayout>,
    pub shared: bool,
}

pub(super) struct Module<U> {
    local: U,
    tasks: Arc<Vec<ReplicatedTextMaterializationTask>>,
    layout: Option<LocalModelLayout>,
    shared: bool,
}
impl<U> AsMut<U> for Module<U> {
    fn as_mut(&mut self) -> &mut U {
        &mut self.local
    }
}
pub(super) struct ProjectionMaterializer;
impl PredictionExtensionMaterializer<WorkspaceBackend> for ProjectionMaterializer {
    type Error = Error;
    type Module<U> = Module<U>;
    type PoolingState = WorkspacePoolingLayerState;
    type SequentialState =
        crate::prediction_extension::workspace::WorkspacePredictionSequentialState;
    type ModelState = ResidentState;
    type Context<'a> = &'a WorkspaceContext;
    type SnapshotContext<'a> = &'a WorkspaceContext;

    fn materialize_module<U: Parameterized<WorkspaceTensor>>(
        _: &mut Self::Context<'_>,
        prepared: PreparedPredictionUnit<U>,
        layout: Option<&LocalModelLayout>,
    ) -> Result<Module<U>, Error> {
        let shared = prepared.role() == PredictionModuleRole::Shared;
        let (_, local, tasks) = prepared.into_shared_parts();
        Ok(Module {
            local,
            tasks,
            layout: layout.cloned(),
            shared,
        })
    }
    fn pooling_state(
        context: &mut Self::Context<'_>,
        _: usize,
        policy: LayerCachePolicy,
    ) -> Result<Self::PoolingState, Error> {
        let layers = eredu_core::LayerSchedule::new(1, vec![policy])
            .map_err(|cause| context.metadata_source(cause))?;
        let layout = StateLayout::new(layers).map_err(|cause| context.metadata_source(cause))?;
        let mut factory = WorkspacePoolingStateFactory::new(NonZeroU32::MIN, context)?;
        let state = factory
            .realize(&layout)
            .map_err(|cause| context.metadata_source(cause))?;
        Ok(state.as_ref()[0].clone())
    }
    fn model_state(
        context: &mut Self::Context<'_>,
        layout: StateLayout,
    ) -> Result<ResidentState, Error> {
        WorkspaceResidentStateFactory::new(NonZeroU32::MIN, NonZeroU32::MIN, context)?
            .realize(&layout)
            .map_err(|cause| context.metadata_source(cause))
    }
    fn sequential_state() -> Self::SequentialState {
        Self::SequentialState::unprojected()
    }
    fn complete_prediction_values<'a>(
        _: impl IntoIterator<Item = &'a WorkspaceTensor>,
        _: &WorkspaceContext,
    ) -> Result<(), eredu_core::BackendFailure> {
        Ok(())
    }
}

pub(super) fn materialize(
    prepared: PreparedPredictionExtension<WorkspaceBackend>,
    context: &WorkspaceContext,
) -> Result<MaterializedPredictionExtension<WorkspaceBackend, ProjectionMaterializer>, Error> {
    prepared.materialize::<ProjectionMaterializer>(&mut &*context)
}

impl DestinationVisitor<'_> {
    pub(super) fn collect_prediction<A, P>(
        self,
        modules: PreparedReplicatedTextModules<A>,
        mut extension: P,
    ) -> Result<ReplicatedTextBindingDestinations, Error>
    where
        A: eredu_runtime::LayeredArchitecture<WorkspaceBackend, ResidentState, Error = Error>,
        P: MaterializedPredictionExecutor<A, WorkspaceBackend, ProjectionMaterializer>,
    {
        struct Collect<'a>(&'a mut Vec<PredictionBindingDestination>);
        impl PredictionModuleVisitor<WorkspaceBackend, ProjectionMaterializer> for Collect<'_> {
            type Error = Error;
            fn visit<U: Parameterized<WorkspaceTensor>>(
                &mut self,
                ordinal: usize,
                module: &mut Module<U>,
            ) -> Result<(), Error> {
                self.0.push(PredictionBindingDestination {
                    ordinal,
                    parameters: collect(&module.local)?,
                    tasks: module.tasks.clone(),
                    layout: module.layout.clone(),
                    shared: module.shared,
                });
                Ok(())
            }
        }
        let mut result = self.collect_modules(modules)?;
        extension.visit_modules(&mut Collect(&mut result.prediction))?;
        Ok(result)
    }
}
impl ReplicatedPredictionProfileDispatcher<WorkspaceBackend, ProjectionMaterializer>
    for DestinationVisitor<'_>
{
    type Output = ReplicatedTextBindingDestinations;
    type Error = Error;
    type State = ResidentState;
    type Visitor = Self;
    fn into_visitor(self) -> Self {
        self
    }
}
impl ReplicatedPredictionTargetVisitor<WorkspaceBackend, ResidentState, ProjectionMaterializer>
    for DestinationVisitor<'_>
{
    type Output = ReplicatedTextBindingDestinations;
    type Error = Error;
    fn visit<A>(
        self,
        prepared: PreparedReplicatedTextArchitecture<A>,
        extension: <A as MaterializedPredictionTarget<WorkspaceBackend>>::Extension<
            ProjectionMaterializer,
        >,
        _: RetainedCheckpointSource,
    ) -> Result<Self::Output, Error>
    where
        A: ReplicatedTextArchitecture<WorkspaceBackend, ResidentState, Error = Error>
            + MaterializedPredictionTarget<WorkspaceBackend>
            + 'static,
        A::StaticModules: Clone,
        A::Error: std::fmt::Display,
    {
        self.collect_prediction(prepared.into_modules(), extension)
    }
}
impl CompositePredictionTargetVisitor<WorkspaceBackend, ResidentState, ProjectionMaterializer>
    for DestinationVisitor<'_>
{
    type Output = ReplicatedTextBindingDestinations;
    type Error = Error;
    fn visit<A>(
        self,
        prepared: PreparedCompositeTextArchitecture<A, A::AdmissionConfig>,
        extension: <crate::composite_execution::PreparedCompositeArchitecture<A> as MaterializedPredictionTarget<WorkspaceBackend>>::Extension<ProjectionMaterializer>,
        _: RetainedCheckpointSource,
    ) -> Result<Self::Output, Error>
    where
        A: CompositeArchitecture<WorkspaceBackend, ResidentState, Error = Error>
            + eredu_runtime::RoutedLayeredArchitecture<WorkspaceBackend, ResidentState>
            + 'static,
        crate::composite_execution::PreparedCompositeArchitecture<A>:
            MaterializedPredictionTarget<WorkspaceBackend>,
        A::InputPartPlan: 'static,
        A::StaticModules: Clone,
        A::Error: std::fmt::Display,
    {
        let (modules, _) = prepared.into_modules();
        self.collect_prediction(modules, extension)
    }
    fn visit_routed<A>(
        self,
        prepared: PreparedRoutedCompositeTextArchitecture<A, A::AdmissionConfig>,
        extension: <crate::composite_execution::PreparedCompositeArchitecture<A> as MaterializedPredictionTarget<WorkspaceBackend>>::Extension<ProjectionMaterializer>,
        _: RetainedCheckpointSource,
    ) -> Result<Self::Output, Error>
    where
        A: CompositeArchitecture<WorkspaceBackend, ResidentState, Error = Error>
            + eredu_runtime::RoutedLayeredArchitecture<WorkspaceBackend, ResidentState>
            + 'static,
        crate::composite_execution::PreparedCompositeArchitecture<A>:
            MaterializedPredictionTarget<WorkspaceBackend>,
        A::InputPartPlan: 'static,
        A::StaticModules: Clone,
        A::Error: std::fmt::Display,
    {
        let (routed, _, _) = prepared.into_parts();
        let (modules, _, _) = routed.into_parts();
        self.collect_prediction(modules, extension)
    }
}

#[cfg(test)]
mod tests;
