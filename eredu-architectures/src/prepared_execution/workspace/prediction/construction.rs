//! The retained total construction selects the typed target exactly once.
use super::*;
use crate::{replicated_text::*, routed_text::*};
type WM<P> = WorkspacePredictionMaterializer<P>;

struct Route<'q, 'a, 'observer, P: WorkspacePredictionParameterSource, Q>(
    &'q Quote<'a, 'observer, P, Q>,
);
impl<P: WorkspacePredictionParameterSource, Q> Copy for Route<'_, '_, '_, P, Q> {}
impl<P: WorkspacePredictionParameterSource, Q> Clone for Route<'_, '_, '_, P, Q> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<P: WorkspacePredictionParameterSource, Q> crate::prepared_execution::sealed::Sealed
    for Route<'_, '_, '_, P, Q>
{
}

struct Visitor<'q, 'a, 'observer, P: WorkspacePredictionParameterSource, Q> {
    quote: &'q Quote<'a, 'observer, P, Q>,
    current: WorkspacePredictionState,
}

impl<'a, 'observer, P, Q> Quote<'a, 'observer, P, Q>
where
    P: WorkspacePredictionParameterSource,
    Q: for<'layout> FnOnce(
        PredictionStateSourceLayout<'layout>,
        &WorkspaceContext,
    ) -> Result<WorkspacePredictionState, Error>,
{
    pub(super) fn construct(&self) -> Result<EquationQuote, PreparedExecutionError<Error>> {
        let routes = value(self.context, || {
            let route = Route(self);
            PreparedExecutionRoutes::new()
                .with_replicated(route)
                .with_routed(route)
                .with_composite(route)
        })
        .map_err(PreparedExecutionError::Metadata)?;
        // This existing branch validates the retained prediction/source agreement
        // without preparing a second extension or publishing prediction placement.
        construct_prepared_execution_impl(
            self.blueprint.sources.clone(),
            None::<()>,
            routes,
            QuoteAssembler(
                self.blueprint
                    .selected()
                    .text_realization()
                    .state()
                    .floating_dtype(),
                self.context,
            ),
            ConstructionPurpose::TargetEquations,
        )
    }
    fn materialize(
        &self,
    ) -> Result<
        (
            MaterializedPredictionExtension<WorkspaceBackend, WM<P>>,
            WorkspacePredictionState,
        ),
        PreparedExecutionError<Error>,
    > {
        let parts = [
            size_of::<crate::prediction_extension::PreparedPredictionExtension<WorkspaceBackend>>(),
            size_of::<MaterializedPredictionExtension<WorkspaceBackend, WM<P>>>(),
            size_of::<WorkspacePredictionState>(),
            size_of::<Result<WorkspacePredictionState, Error>>(),
            size_of::<Visitor<'_, '_, '_, P, Q>>(),
            size_of::<P::Context<'a>>(),
            size_of::<Q>(),
            size_of::<std::cell::RefMut<'_, Option<P::Context<'a>>>>(),
            size_of::<std::cell::RefMut<'_, Option<Q>>>(),
            size_of::<
                Result<
                    (
                        MaterializedPredictionExtension<WorkspaceBackend, WM<P>>,
                        WorkspacePredictionState,
                    ),
                    Error,
                >,
            >(),
        ];
        self.context
            .charge_metadata(
                parts
                    .into_iter()
                    .try_fold(size_of_val(&parts), usize::checked_add)
                    .ok_or_else(|| {
                        PreparedExecutionError::Metadata(WorkspaceMetadataError::Overflow.into())
                    })?,
            )
            .map_err(|cause| PreparedExecutionError::Metadata(cause.into()))?;
        let project = self.project.borrow_mut().take().ok_or_else(|| {
            PreparedExecutionError::Metadata(self.context.metadata_source(QuoteError::Consumed))
        })?;
        let (prepared, current) =
            crate::prediction_extension::prepare_retained::<WorkspaceBackend, _, _>(
                &self.blueprint.sources,
                self.context,
                self.context,
                |layout| project(layout, self.context),
            )
            .map_err(PreparedExecutionError::Metadata)?;
        let parameters = self.parameters.borrow_mut().take().ok_or_else(|| {
            PreparedExecutionError::Metadata(self.context.metadata_source(QuoteError::Consumed))
        })?;
        prepared
            .materialize_workspace_owned::<P>(parameters, current, self.context)
            .map_err(PreparedExecutionError::Metadata)
    }
}

impl<P, Q, C>
    PreparedExecutionRoute<
        eredu_runtime::SelectedReplicatedTextRealization,
        C,
        EquationQuote,
        Error,
    > for Route<'_, '_, '_, P, Q>
where
    P: WorkspacePredictionParameterSource,
    Q: for<'layout> FnOnce(
        PredictionStateSourceLayout<'layout>,
        &WorkspaceContext,
    ) -> Result<WorkspacePredictionState, Error>,
{
    fn construct(
        self,
        branch: PreparedConstructionBranch<eredu_runtime::SelectedReplicatedTextRealization, C>,
    ) -> Result<EquationQuote, PreparedExecutionError<Error>> {
        let (extension, current) = self.0.materialize()?;
        dispatch_replicated_prediction_target_architecture::<WorkspaceBackend, WM<P>, _>(
            &branch.inspection.sources,
            branch.selected,
            extension,
            branch.target,
            self.0.context,
            Visitor {
                quote: self.0,
                current,
            },
        )
        .map_err(replicated_error)
    }
}
impl<P, Q, C> PreparedExecutionRoute<crate::SelectedRoutedTextRealization, C, EquationQuote, Error>
    for Route<'_, '_, '_, P, Q>
where
    P: WorkspacePredictionParameterSource,
    Q: for<'layout> FnOnce(
        PredictionStateSourceLayout<'layout>,
        &WorkspaceContext,
    ) -> Result<WorkspacePredictionState, Error>,
{
    fn construct(
        self,
        branch: PreparedConstructionBranch<crate::SelectedRoutedTextRealization, C>,
    ) -> Result<EquationQuote, PreparedExecutionError<Error>> {
        let (extension, current) = self.0.materialize()?;
        dispatch_routed_prediction_target_architecture::<WorkspaceBackend, WM<P>, _>(
            &branch.inspection.sources,
            branch.selected,
            extension,
            branch.target,
            self.0.context,
            Visitor {
                quote: self.0,
                current,
            },
        )
        .map_err(routed_error)
    }
}
impl<P, Q, C> PreparedExecutionRoute<SelectedCompositeTextRealization, C, EquationQuote, Error>
    for Route<'_, '_, '_, P, Q>
where
    P: WorkspacePredictionParameterSource,
    Q: for<'layout> FnOnce(
        PredictionStateSourceLayout<'layout>,
        &WorkspaceContext,
    ) -> Result<WorkspacePredictionState, Error>,
{
    fn construct(
        self,
        branch: PreparedConstructionBranch<SelectedCompositeTextRealization, C>,
    ) -> Result<EquationQuote, PreparedExecutionError<Error>> {
        let (extension, current) = self.0.materialize()?;
        visit_prepared_composite_prediction_target_text_architecture::<
            WorkspaceBackend,
            ResidentState,
            WM<P>,
            _,
        >(
            &branch.inspection.sources,
            branch.selected,
            extension,
            branch.target,
            self.0.context,
            Visitor {
                quote: self.0,
                current,
            },
        )
        .map_err(replicated_error)
    }
}

impl<P, Q> ReplicatedPredictionProfileDispatcher<WorkspaceBackend, WM<P>>
    for Visitor<'_, '_, '_, P, Q>
where
    P: WorkspacePredictionParameterSource,
    Q: for<'layout> FnOnce(
        PredictionStateSourceLayout<'layout>,
        &WorkspaceContext,
    ) -> Result<WorkspacePredictionState, Error>,
{
    type Output = EquationQuote;
    type Error = Error;
    type State = ResidentState;
    type Visitor = Self;
    fn into_visitor(self) -> Self {
        self
    }
}
impl<P, Q> ReplicatedPredictionTargetVisitor<WorkspaceBackend, ResidentState, WM<P>>
    for Visitor<'_, '_, '_, P, Q>
where
    P: WorkspacePredictionParameterSource,
    Q: for<'layout> FnOnce(
        PredictionStateSourceLayout<'layout>,
        &WorkspaceContext,
    ) -> Result<WorkspacePredictionState, Error>,
{
    type Output = EquationQuote;
    type Error = Error;
    fn visit<A>(
        self,
        prepared: PreparedReplicatedTextArchitecture<A>,
        extension: <A as crate::prediction_extension::MaterializedPredictionTarget<
            WorkspaceBackend,
        >>::Extension<WM<P>>,
        _store: RetainedCheckpointSource,
    ) -> Result<EquationQuote, Error>
    where
        A: ReplicatedTextArchitecture<WorkspaceBackend, ResidentState, Error = Error>
            + crate::prediction_extension::MaterializedPredictionTarget<WorkspaceBackend>
            + 'static,
        A::StaticModules: Clone,
        A::Error: std::fmt::Display,
    {
        self.quote
            .run::<A>(prepared.into_modules(), extension, self.current)
    }
}
impl<P, Q> RoutedPredictionProfileDispatcher<WorkspaceBackend, WM<P>> for Visitor<'_, '_, '_, P, Q>
where
    P: WorkspacePredictionParameterSource,
    Q: for<'layout> FnOnce(
        PredictionStateSourceLayout<'layout>,
        &WorkspaceContext,
    ) -> Result<WorkspacePredictionState, Error>,
{
    type Output = EquationQuote;
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
impl<P, Q> RoutedPredictionTargetVisitor<WorkspaceBackend, ResidentState, WM<P>>
    for Visitor<'_, '_, '_, P, Q>
where
    P: WorkspacePredictionParameterSource,
    Q: for<'layout> FnOnce(
        PredictionStateSourceLayout<'layout>,
        &WorkspaceContext,
    ) -> Result<WorkspacePredictionState, Error>,
{
    type Output = EquationQuote;
    type Error = Error;
    fn visit<A>(
        self,
        prepared: PreparedRoutedTextArchitecture<A>,
        extension: <A as crate::prediction_extension::MaterializedPredictionTarget<
            WorkspaceBackend,
        >>::Extension<WM<P>>,
        _store: RetainedCheckpointSource,
    ) -> Result<EquationQuote, Error>
    where
        A: ReplicatedTextArchitecture<WorkspaceBackend, ResidentState, Error = Error>
            + eredu_runtime::RoutedLayeredArchitecture<WorkspaceBackend, ResidentState>
            + crate::prediction_extension::MaterializedPredictionTarget<WorkspaceBackend>
            + 'static,
        A::StaticModules: Clone,
        A::Error: std::fmt::Display,
    {
        let (modules, _, _) = prepared.into_shared_parts();
        self.quote.run::<A>(modules, extension, self.current)
    }
}
impl<P, Q> CompositePredictionTargetVisitor<WorkspaceBackend, ResidentState, WM<P>>
    for Visitor<'_, '_, '_, P, Q>
where
    P: WorkspacePredictionParameterSource,
    Q: for<'layout> FnOnce(
        PredictionStateSourceLayout<'layout>,
        &WorkspaceContext,
    ) -> Result<WorkspacePredictionState, Error>,
{
    type Output = EquationQuote;
    type Error = Error;
    fn visit<A>(
        self,
        prepared: PreparedCompositeTextArchitecture<
            A,
            <A as crate::composite_execution::CompositeArchitecture<
                WorkspaceBackend,
                ResidentState,
            >>::AdmissionConfig,
        >,
        extension:<crate::composite_execution::PreparedCompositeArchitecture<A> as crate::prediction_extension::MaterializedPredictionTarget<WorkspaceBackend>>::Extension<WM<P>>,
        _store: RetainedCheckpointSource,
    ) -> Result<EquationQuote, Error>
    where
        A: crate::composite_execution::CompositeArchitecture<
                WorkspaceBackend,
                ResidentState,
                Error = Error,
            > + eredu_runtime::RoutedLayeredArchitecture<WorkspaceBackend, ResidentState>
            + 'static,
        crate::composite_execution::PreparedCompositeArchitecture<A>:
            crate::prediction_extension::MaterializedPredictionTarget<WorkspaceBackend>,
        A::InputPartPlan: 'static,
        A::StaticModules: Clone,
        A::Error: std::fmt::Display,
    {
        let (mut modules, _) = prepared.into_modules();
        modules.retain_composite_graph(self.quote.context)?;
        self.quote.run(modules, extension, self.current)
    }
    fn visit_routed<A>(
        self,
        prepared: PreparedRoutedCompositeTextArchitecture<
            A,
            <A as crate::composite_execution::CompositeArchitecture<
                WorkspaceBackend,
                ResidentState,
            >>::AdmissionConfig,
        >,
        extension:<crate::composite_execution::PreparedCompositeArchitecture<A> as crate::prediction_extension::MaterializedPredictionTarget<WorkspaceBackend>>::Extension<WM<P>>,
        _store: RetainedCheckpointSource,
    ) -> Result<EquationQuote, Error>
    where
        A: crate::composite_execution::CompositeArchitecture<
                WorkspaceBackend,
                ResidentState,
                Error = Error,
            > + eredu_runtime::RoutedLayeredArchitecture<WorkspaceBackend, ResidentState>
            + 'static,
        crate::composite_execution::PreparedCompositeArchitecture<A>:
            crate::prediction_extension::MaterializedPredictionTarget<WorkspaceBackend>,
        A::InputPartPlan: 'static,
        A::StaticModules: Clone,
        A::Error: std::fmt::Display,
    {
        let (routed, _, _) = prepared.into_parts();
        let (mut modules, _, _) = routed.into_parts();
        modules.retain_composite_graph(self.quote.context)?;
        self.quote.run(modules, extension, self.current)
    }
}
