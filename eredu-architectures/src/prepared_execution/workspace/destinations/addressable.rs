//! Cold bank declarations from the same typed constructors used by native binding.
use super::*;
use crate::prediction_extension::MaterializedPredictionTarget;
use crate::replicated_text::CompositePredictionTargetVisitor;
use crate::routed_text::{
    PreparedRoutedTextArchitecture, Relu2RoutedTextArchitectureVisitor,
    RoutedPredictionProfileDispatcher, RoutedPredictionTargetVisitor,
    RoutedTextArchitectureVisitor, SelectedRoutedBank,
};
use eredu_runtime::{
    AddressableBankMember, LocalModelLayout, ParameterBankLoadOptions, ParameterBankResidency,
};

/// Exact selected bank members before native loading. This is descriptive
/// geometry only; the backend must separately fund and authenticate its source.
pub struct AddressableBindingDestinations {
    members: Vec<AddressableBankMember>,
    options: ParameterBankLoadOptions,
    layout: Option<LocalModelLayout>,
}
impl AddressableBindingDestinations {
    /// Canonical member recipes and logical parameter tasks.
    pub fn members(&self) -> &[AddressableBankMember] {
        &self.members
    }
    /// Selected bank storage and compaction policy.
    pub fn options(&self) -> ParameterBankLoadOptions {
        self.options
    }
    /// Exact physical partition layout, absent for replicated execution.
    pub fn layout(&self) -> Option<&LocalModelLayout> {
        self.layout.as_ref()
    }
}
fn collect_banks(
    residency: ParameterBankResidency,
    banks: &BTreeMap<eredu_runtime::RoutedBankId, SelectedRoutedBank>,
    layout: Option<&LocalModelLayout>,
) -> Option<AddressableBindingDestinations> {
    let ParameterBankResidency::IndependentCache(options) = residency else {
        return None;
    };
    let members = banks
        .values()
        .flat_map(|bank| bank.addressable_members().iter().cloned())
        .collect::<Vec<_>>();
    (!members.is_empty()).then(|| AddressableBindingDestinations {
        members,
        options,
        layout: layout.cloned(),
    })
}

/// Reuses total selected construction, including local expert ownership,
/// physical tasks and prediction pairing. No source is reopened and no backend
/// tensor, communication group, or request authority is created.
pub fn project_addressable_binding_destinations(
    sources: &PreparedModelSources,
    context: &WorkspaceContext,
) -> Result<Option<AddressableBindingDestinations>, PreparedExecutionError<Error>> {
    if !sources
        .selected()
        .execution()
        .has_independent_parameter_banks()
    {
        return Ok(None);
    }
    let materialize = |prepared, _source| prediction::materialize(prepared, context);
    let routes = PreparedExecutionRoutes::new()
        .with_routed(
            RoutedRoute::<WorkspaceBackend, ResidentState, ResidentState, _, _, _>::new(
                context,
                context,
                AddressableVisitor,
                AddressableVisitor,
                AddressableVisitor,
            )
            .with_prediction::<ProjectionMaterializer, _, _>(materialize, |_| AddressableVisitor),
        )
        .with_composite(
            CompositeRoute::<WorkspaceBackend, ResidentState, _>::new(
                context,
                context,
                AddressableVisitor,
            )
            .with_prediction::<ProjectionMaterializer, _, _>(materialize, |_| AddressableVisitor),
        )
        .with_partitioned_routed(
            PartitionedRoutedRoute::<WorkspaceBackend, ResidentState, ResidentState, _, _>::new(
                context,
                context,
                |_: PreparedPartitionResources<()>| AddressableVisitor,
                |_: PreparedPartitionResources<()>| AddressableVisitor,
            )
            .with_prediction::<ProjectionMaterializer, _, _>(
                materialize,
                |_: PreparedPartitionPredictionResources<()>| AddressableVisitor,
            ),
        )
        .with_partitioned_composite(
            PartitionedCompositeRoute::<WorkspaceBackend, ResidentState, _>::new(
                context,
                context,
                |_: PreparedPartitionResources<()>| AddressableVisitor,
            )
            .with_prediction::<ProjectionMaterializer, _, _>(
                materialize,
                |_: PreparedPartitionPredictionResources<()>| AddressableVisitor,
            ),
        );
    let manifest = sources.selected().communication_manifest();
    construct_prepared_execution_impl(
        sources.clone(),
        manifest.map(|_| ()),
        routes,
        AddressableAssembler {
            manifest,
            dtype: sources
                .selected()
                .text_realization()
                .state()
                .floating_dtype(),
        },
        ConstructionPurpose::BindingDestinations,
    )
}
#[derive(Clone, Copy)]
struct AddressableVisitor;
struct AddressableAssembler<'a> {
    manifest: Option<&'a CommunicationManifest>,
    dtype: Option<eredu_runtime::StateStorageDtype>,
}
impl PreparedExecutableAssembler<()> for AddressableAssembler<'_> {
    type Executable = Option<AddressableBindingDestinations>;
    type Output = Option<AddressableBindingDestinations>;
    type Error = Error;
    fn floating_state_dtype(
        &mut self,
        _: &FloatingStateDtypeSource,
    ) -> Result<eredu_runtime::StateStorageDtype, Error> {
        Ok(self.dtype.unwrap_or(eredu_runtime::StateStorageDtype::F32))
    }
    fn validate_communication(
        &mut self,
        manifest: &CommunicationManifest,
        _: &(),
    ) -> Result<(), Error> {
        if self.manifest != Some(manifest) {
            return Err(Error::backend(
                "bank projection communication differs from retained selection",
            ));
        }
        Ok(())
    }
    fn finish(
        self,
        parts: PreparedExecutableParts<Self::Executable, ()>,
    ) -> Result<Self::Output, Error> {
        Ok(parts.into_executable())
    }
}
impl RoutedTextArchitectureVisitor<WorkspaceBackend, ResidentState> for AddressableVisitor {
    type Output = Option<AddressableBindingDestinations>;
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
        Ok(collect_banks(
            prepared.bank_residency(),
            prepared.banks(),
            None,
        ))
    }
}
impl Relu2RoutedTextArchitectureVisitor<WorkspaceBackend, ResidentState> for AddressableVisitor {
    type Output = Option<AddressableBindingDestinations>;
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
        Ok(collect_banks(
            prepared.bank_residency(),
            prepared.banks(),
            None,
        ))
    }
}
impl RoutedPredictionProfileDispatcher<WorkspaceBackend, ProjectionMaterializer>
    for AddressableVisitor
{
    type Output = Option<AddressableBindingDestinations>;
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
    for AddressableVisitor
{
    type Output = Option<AddressableBindingDestinations>;
    type Error = Error;
    fn visit<A>(
        self,
        prepared: PreparedRoutedTextArchitecture<A>,
        _extension: <A as MaterializedPredictionTarget<WorkspaceBackend>>::Extension<
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
        Ok(collect_banks(
            prepared.bank_residency(),
            prepared.banks(),
            None,
        ))
    }
}
impl CompositeTextArchitectureVisitor<WorkspaceBackend, ResidentState> for AddressableVisitor {
    type Output = Option<AddressableBindingDestinations>;
    type Error = Error;

    fn construction_started(&mut self) {}

    fn visit<A>(
        self,
        _prepared: PreparedCompositeTextArchitecture<A, A::AdmissionConfig>,
        _: RetainedCheckpointSource,
    ) -> Result<Self::Output, Error>
    where
        A: CompositeArchitecture<WorkspaceBackend, ResidentState, Error = Error>
            + eredu_runtime::RoutedLayeredArchitecture<WorkspaceBackend, ResidentState>
            + 'static,
        A::InputPartPlan: 'static,
        A::StaticModules: Clone,
    {
        Ok(None)
    }

    fn visit_routed<A>(
        self,
        prepared: PreparedRoutedCompositeTextArchitecture<A, A::AdmissionConfig>,
        _: RetainedCheckpointSource,
    ) -> Result<Self::Output, Error>
    where
        A: CompositeArchitecture<WorkspaceBackend, ResidentState, Error = Error>
            + eredu_runtime::RoutedLayeredArchitecture<WorkspaceBackend, ResidentState>
            + 'static,
        A::InputPartPlan: 'static,
        A::StaticModules: Clone,
    {
        let (routed, _, _) = prepared.into_parts();
        Ok(collect_banks(routed.bank_residency(), routed.banks(), None))
    }
}
impl CompositePredictionTargetVisitor<WorkspaceBackend, ResidentState, ProjectionMaterializer>
    for AddressableVisitor
{
    type Output = Option<AddressableBindingDestinations>;
    type Error = Error;
    fn visit<A>(
        self,
        _prepared: PreparedCompositeTextArchitecture<A, A::AdmissionConfig>,
        _extension: <crate::composite_execution::PreparedCompositeArchitecture<A> as MaterializedPredictionTarget<WorkspaceBackend>>::Extension<ProjectionMaterializer>,
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
        Ok(None)
    }
    fn visit_routed<A>(
        self,
        prepared: PreparedRoutedCompositeTextArchitecture<A, A::AdmissionConfig>,
        _extension: <crate::composite_execution::PreparedCompositeArchitecture<A> as MaterializedPredictionTarget<WorkspaceBackend>>::Extension<ProjectionMaterializer>,
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
        Ok(collect_banks(routed.bank_residency(), routed.banks(), None))
    }
}
use crate::partitioned_execution::{PreparedRoutedPartitionedArchitecture, TextPartitionArchitecture};
impl
    crate::partitioned_execution::RoutedPartitionedProductionVisitor<
        WorkspaceBackend,
        ResidentState,
    > for AddressableVisitor
{
    type Output = Option<AddressableBindingDestinations>;
    type Error = Error;
    fn visit<A, G>(
        self,
        prepared: PreparedRoutedPartitionedArchitecture<
            WorkspaceBackend,
            A,
            G,
            <A as eredu_runtime::PartitionedLayeredArchitecture<WorkspaceBackend, ResidentState>>::Boundary,
        >,
        _store: eredu_checkpoint::store::RetainedCheckpointSource,
    ) -> Result<Option<AddressableBindingDestinations>, Error>
    where
        A: TextPartitionArchitecture<WorkspaceBackend, ResidentState>
            + eredu_runtime::ReplicatedTextArchitecture<
                WorkspaceBackend,
                ResidentState,
                Error = eredu_nn::Error,
            > + eredu_runtime::ParallelRoutedLayeredArchitecture<WorkspaceBackend, ResidentState>
            + 'static,
        A::StaticModules: Clone,
        G: 'static,
    {
        Ok(collect_banks(
            prepared.bank_residency(),
            prepared.banks(),
            Some(prepared.layout()),
        ))
    }
}
impl
    crate::partitioned_execution::RoutedPartitionedPredictionTargetProductionVisitor<
        WorkspaceBackend,
        ResidentState,
        ProjectionMaterializer,
    > for AddressableVisitor
{
    type Output = Option<AddressableBindingDestinations>;
    type Error = Error;
    fn visit<A, G>(
        self,
        prepared: PreparedRoutedPartitionedArchitecture<
            WorkspaceBackend,
            A,
            G,
            <A as eredu_runtime::PartitionedLayeredArchitecture<WorkspaceBackend, ResidentState>>::Boundary,
        >,
        _extension: <A as crate::prediction_extension::MaterializedPredictionTarget<
            WorkspaceBackend,
        >>::Extension<ProjectionMaterializer>,
        _store: eredu_checkpoint::store::RetainedCheckpointSource,
    ) -> Result<Option<AddressableBindingDestinations>, Error>
    where
        A: TextPartitionArchitecture<WorkspaceBackend, ResidentState>
            + eredu_runtime::ReplicatedTextArchitecture<
                WorkspaceBackend,
                ResidentState,
                Error = eredu_nn::Error,
            > + eredu_runtime::ParallelRoutedLayeredArchitecture<WorkspaceBackend, ResidentState>
            + crate::prediction_extension::MaterializedPredictionTarget<WorkspaceBackend>
            + 'static,
        A::StaticModules: Clone,
        G: 'static,
    {
        Ok(collect_banks(
            prepared.bank_residency(),
            prepared.banks(),
            Some(prepared.layout()),
        ))
    }
}
impl
    crate::composite_partitioned::AuthoritativeCompositePartitionVisitor<
        WorkspaceBackend,
        ResidentState,
    > for AddressableVisitor
{
    type Output = Option<AddressableBindingDestinations>;
    type Error = Error;
    fn visit<A, G, W>(
        self,
        prepared: crate::composite_partitioned::PreparedCompositePartition<A, G, W>,
    ) -> Result<Self::Output, Self::Error>
    where
        A: crate::composite_execution::CompositeArchitecture<
                WorkspaceBackend,
                ResidentState,
                Error = eredu_nn::Error,
            > + eredu_runtime::PartitionedLayeredArchitecture<
                WorkspaceBackend,
                ResidentState,
                Boundary = W,
            > + eredu_runtime::ParallelRoutedLayeredArchitecture<WorkspaceBackend, ResidentState>
            + 'static,
        A::Error: std::fmt::Display,
        W: eredu_runtime::ArchitectureBoundary,
    {
        Ok(prepared.partition_banks().and_then(|banks| {
            collect_banks(banks.residency(), banks.banks(), Some(prepared.layout()))
        }))
    }
}
impl
    crate::composite_partitioned::AuthoritativeCompositePartitionPredictionTargetVisitor<
        WorkspaceBackend,
        ResidentState,
        ProjectionMaterializer,
    > for AddressableVisitor
{
    type Output = Option<AddressableBindingDestinations>;
    type Error = Error;
    fn visit<A, G, W>(
        self,
        prepared: crate::composite_partitioned::PreparedCompositePartition<A, G, W>,
        _extension: <crate::composite_execution::PreparedCompositeArchitecture<A> as crate::prediction_extension::MaterializedPredictionTarget<WorkspaceBackend>>::Extension<ProjectionMaterializer>,
    ) -> Result<Self::Output, Self::Error>
    where
        A: crate::composite_execution::CompositeArchitecture<
                WorkspaceBackend,
                ResidentState,
                Error = eredu_nn::Error,
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
        Ok(prepared.partition_banks().and_then(|banks| {
            collect_banks(banks.residency(), banks.banks(), Some(prepared.layout()))
        }))
    }
}

#[cfg(test)]
mod tests;
