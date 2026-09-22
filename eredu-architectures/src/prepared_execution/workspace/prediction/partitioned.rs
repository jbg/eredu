//! Exact partition constructors and typed target operations for prediction quotes.
use super::construction::{Route, Visitor};
use super::*;
use crate::composite_partitioned::*;
use crate::partitioned_execution::*;

pub(super) struct Assembler<'a> {
    pub(super) dtype: Option<eredu_runtime::StateStorageDtype>,
    pub(super) context: &'a WorkspaceContext,
}
impl PreparedExecutableAssembler<&eredu_runtime::RetainedCommunicationSource> for Assembler<'_> {
    type Executable = EquationQuote;
    type Output = EquationQuote;
    type Error = Error;
    fn floating_state_dtype(
        &mut self,
        _: &FloatingStateDtypeSource,
    ) -> Result<eredu_runtime::StateStorageDtype, Error> {
        Ok(self.dtype.unwrap_or(eredu_runtime::StateStorageDtype::F32))
    }
    fn validate_communication(
        &mut self,
        expected: &CommunicationManifest,
        actual: &&eredu_runtime::RetainedCommunicationSource,
    ) -> Result<(), Error> {
        self.context.charge_metadata(size_of::<(
            &mut Self,
            &CommunicationManifest,
            &&eredu_runtime::RetainedCommunicationSource,
            Result<(), Error>,
        )>())?;
        if expected != actual.manifest() {
            return Err(self.context.metadata_error(format_args!(
                "prediction communication differs from its retained partition source"
            )));
        }
        Ok(())
    }
    fn finish(
        self,
        parts: PreparedExecutableParts<
            Self::Executable,
            &eredu_runtime::RetainedCommunicationSource,
        >,
    ) -> Result<Self::Output, Error> {
        Ok(parts.into_executable())
    }
}

impl<P, Q, C> PreparedExecutionRoute<SelectedDensePartitionedExecution, C, EquationQuote, Error>
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
        mut branch: PreparedConstructionBranch<SelectedDensePartitionedExecution, C>,
    ) -> Result<EquationQuote, PreparedExecutionError<Error>> {
        self.0
            .context
            .charge_metadata(size_of::<(
                Self,
                PreparedConstructionBranch<SelectedDensePartitionedExecution, C>,
                Result<EquationQuote, PreparedExecutionError<Error>>,
            )>())
            .map_err(|cause| PreparedExecutionError::Metadata(cause.into()))?;
        let _resources = branch.partition_resources()?;
        let (extension, current) = self.0.materialize()?;
        visit_resident_partitioned_prediction_target_architecture::<
            WorkspaceBackend,
            ResidentState,
            WM<P>,
            _,
        >(
            &branch.inspection.retained(),
            branch.selected,
            extension,
            branch.target,
            self.0.context,
            Visitor {
                quote: self.0,
                current,
            },
        )
        .map_err(partitioned_error)
    }
}
impl<P, Q, C> PreparedExecutionRoute<SelectedRoutedPartitionedExecution, C, EquationQuote, Error>
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
        mut branch: PreparedConstructionBranch<SelectedRoutedPartitionedExecution, C>,
    ) -> Result<EquationQuote, PreparedExecutionError<Error>> {
        self.0
            .context
            .charge_metadata(size_of::<(
                Self,
                PreparedConstructionBranch<SelectedRoutedPartitionedExecution, C>,
                Result<EquationQuote, PreparedExecutionError<Error>>,
            )>())
            .map_err(|cause| PreparedExecutionError::Metadata(cause.into()))?;
        let _resources = branch.partition_resources()?;
        let (extension, current) = self.0.materialize()?;
        let visitor = Visitor {
            quote: self.0,
            current,
        };
        dispatch_routed_partitioned_production(
            &branch.inspection.retained(),
            branch.selected,
            (branch.target, extension, visitor),
            |(target, extension, visitor), inspection, selected| {
                visit_routed_partitioned_prediction_target_production::<
                    WorkspaceBackend,
                    ResidentState,
                    WM<P>,
                    _,
                >(
                    inspection,
                    selected,
                    extension,
                    target,
                    visitor.quote.context,
                    visitor,
                )
                .map_err(partitioned_error)
            },
            |(target, extension, visitor), inspection, selected| {
                visit_relu2_routed_partitioned_prediction_target_production::<
                    WorkspaceBackend,
                    ResidentState,
                    WM<P>,
                    _,
                >(
                    inspection,
                    selected,
                    extension,
                    target,
                    visitor.quote.context,
                    visitor,
                )
                .map_err(partitioned_error)
            },
            |(target, extension, visitor), inspection, selected| {
                visit_pooling_routed_partitioned_prediction_target_production::<
                    WorkspaceBackend,
                    ResidentState,
                    WM<P>,
                    _,
                >(
                    inspection,
                    selected,
                    extension,
                    target,
                    visitor.quote.context,
                    visitor,
                )
                .map_err(partitioned_error)
            },
        )
    }
}
impl<P, Q, C> PreparedExecutionRoute<SelectedCompositePartitionedExecution, C, EquationQuote, Error>
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
        mut branch: PreparedConstructionBranch<SelectedCompositePartitionedExecution, C>,
    ) -> Result<EquationQuote, PreparedExecutionError<Error>> {
        self.0
            .context
            .charge_metadata(size_of::<(
                Self,
                PreparedConstructionBranch<SelectedCompositePartitionedExecution, C>,
                Result<EquationQuote, PreparedExecutionError<Error>>,
            )>())
            .map_err(|cause| PreparedExecutionError::Metadata(cause.into()))?;
        let _resources = branch.partition_resources()?;
        let (extension, current) = self.0.materialize()?;
        visit_authoritative_composite_prediction_target_partition::<
            WorkspaceBackend,
            ResidentState,
            WM<P>,
            _,
        >(
            branch.selected,
            extension,
            self.0.context,
            Visitor {
                quote: self.0,
                current,
            },
        )
        .map_err(composite_partitioned_error)
    }
}
type WM<P> = WorkspacePredictionMaterializer<P>;
impl<P, Q> Quote<'_, '_, P, Q>
where
    P: WorkspacePredictionParameterSource,
    Q: for<'layout> FnOnce(
        PredictionStateSourceLayout<'layout>,
        &WorkspaceContext,
    ) -> Result<WorkspacePredictionState, Error>,
{
    fn run_partition<A>(
        &self,
        architecture: A,
        local_state: Option<&eredu_runtime::PartitionState>,
        rank: eredu_core::ParallelRankTopology,
        extension: <A as crate::prediction_extension::MaterializedPredictionTarget<
            WorkspaceBackend,
        >>::Extension<WM<P>>,
        current: WorkspacePredictionState,
    ) -> Result<EquationQuote, Error>
    where
        A: eredu_runtime::LayeredArchitecture<WorkspaceBackend, ResidentState, Error = Error>
            + crate::prediction_extension::MaterializedPredictionTarget<WorkspaceBackend>
            + 'static,
    {
        self.context.charge_metadata(size_of::<(
            A,
            Option<&eredu_runtime::PartitionState>,
            eredu_core::ParallelRankTopology,
            TargetRuntime<'_, A>,
            Result<EquationQuote, Error>,
        )>())?;
        match local_state {
            Some(state) if state.layout() == self.target.layout() => {}
            None if self.target.layout().is_empty() => {}
            _ => return Err(self.invalid()),
        }
        let communication = self.communication.ok_or_else(|| self.invalid())?;
        let selected = self.blueprint.selected();
        if selected.execution().parallel_topology() != Some(rank) {
            return Err(self.invalid());
        }
        let parallel =
            super::super::parallel::parallel_context(selected, rank, communication, self.context)?;
        self.run_equation(
            TargetRuntime::Partitioned {
                architecture,
                parallel,
            },
            extension,
            current,
        )
    }
}
impl<P, Q> PartitionedPredictionTargetVisitor<WorkspaceBackend, ResidentState, WM<P>>
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
    fn visit<A, G>(
        self,
        prepared:PreparedPartitionedArchitecture<WorkspaceBackend,A,G,
        <A as eredu_runtime::PartitionedLayeredArchitecture<WorkspaceBackend,ResidentState>>::Boundary>,
        extension: <A as crate::prediction_extension::MaterializedPredictionTarget<
            WorkspaceBackend,
        >>::Extension<WM<P>>,
        _store: RetainedCheckpointSource,
    ) -> Result<EquationQuote, Error>
    where
        A: TextPartitionArchitecture<WorkspaceBackend, ResidentState>
            + eredu_runtime::ReplicatedTextArchitecture<
                WorkspaceBackend,
                ResidentState,
                Error = Error,
            > + crate::prediction_extension::MaterializedPredictionTarget<WorkspaceBackend>
            + 'static,
        A::StaticModules: Clone,
        G: 'static,
    {
        let (prepared, source, _layout, _tasks) = prepared.into_parts();
        let (architecture, bound) = prepared.into_parts();
        let result = self.quote.run_partition(
            architecture,
            bound.partition().state(),
            bound.topology(),
            extension,
            self.current,
        );
        drop(source);
        result
    }
}
impl<P, Q>
    RoutedPartitionedPredictionTargetProductionVisitor<WorkspaceBackend, ResidentState, WM<P>>
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
    fn visit<A, G>(
        self,
        prepared:PreparedRoutedPartitionedArchitecture<WorkspaceBackend,A,G,
        <A as eredu_runtime::PartitionedLayeredArchitecture<WorkspaceBackend,ResidentState>>::Boundary>,
        extension: <A as crate::prediction_extension::MaterializedPredictionTarget<
            WorkspaceBackend,
        >>::Extension<WM<P>>,
        _store: RetainedCheckpointSource,
    ) -> Result<EquationQuote, Error>
    where
        A: TextPartitionArchitecture<WorkspaceBackend, ResidentState>
            + eredu_runtime::ReplicatedTextArchitecture<
                WorkspaceBackend,
                ResidentState,
                Error = Error,
            > + eredu_runtime::ParallelRoutedLayeredArchitecture<WorkspaceBackend, ResidentState>
            + crate::prediction_extension::MaterializedPredictionTarget<WorkspaceBackend>
            + 'static,
        A::StaticModules: Clone,
        G: 'static,
    {
        let (prepared, source, _layout, _tasks) = prepared.into_parts();
        let (architecture, bound) = prepared.into_parts();
        let result = self.quote.run_partition(
            architecture,
            bound.partition().state(),
            bound.topology(),
            extension,
            self.current,
        );
        drop(source);
        result
    }
}
impl<P, Q>
    AuthoritativeCompositePartitionPredictionTargetVisitor<WorkspaceBackend, ResidentState, WM<P>>
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
    fn visit<A, G, W>(
        self,
        prepared: PreparedCompositePartition<A, G, W>,
        extension:<crate::composite_execution::PreparedCompositeArchitecture<A> as
            crate::prediction_extension::MaterializedPredictionTarget<WorkspaceBackend>>::Extension<WM<P>>,
    ) -> Result<EquationQuote, Error>
    where
        A: crate::composite_execution::CompositeArchitecture<
                WorkspaceBackend,
                ResidentState,
                Error = Error,
            > + eredu_runtime::PartitionedLayeredArchitecture<
                WorkspaceBackend,
                ResidentState,
                Boundary = W,
            > + eredu_runtime::ParallelRoutedLayeredArchitecture<WorkspaceBackend, ResidentState>
            + 'static,
        A::Error: std::fmt::Display,
        crate::composite_execution::PreparedCompositeArchitecture<A>:
            crate::prediction_extension::MaterializedPredictionTarget<WorkspaceBackend>,
        W: eredu_runtime::ArchitectureBoundary,
    {
        let (prepared, source, _layout, _tasks) = prepared.into_parts();
        let (architecture, bound) = prepared.into_parts();
        let architecture =
            crate::composite_execution::PreparedCompositeArchitecture::new(architecture);
        let result = self.quote.run_partition(
            architecture,
            bound.partition().state(),
            bound.topology(),
            extension,
            self.current,
        );
        drop(source);
        result
    }
}
