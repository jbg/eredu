//! Rank-local destinations from the retained ordinary partition constructor.
use super::*;
use crate::partitioned_execution::{PartitionedArchitectureVisitor, PreparedPartitionedArchitecture, TextPartitionArchitecture};
use eredu_runtime::{ExecutionUnitAddress, ExecutionUnitLayout, LocalModelLayout, PartitionedUnitScope,
    ReplicatedTextMaterializationTask, SelectedReplicatedTextRealization};

/// Physical module slots and exact local payload tasks selected by a partition.
/// Descriptive cold metadata carries no communication or materialization grant.
pub struct PartitionedTextBindingDestinations {
    selected: SelectedReplicatedTextRealization,
    global_layout: ExecutionUnitLayout,
    local_layout: ExecutionUnitLayout,
    addresses: Vec<ExecutionUnitAddress>,
    physical_layout: LocalModelLayout,
    tasks: Vec<ReplicatedTextMaterializationTask>,
    static_parameters: BTreeMap<String, WorkspaceLayout>,
    units: Vec<BTreeMap<String, WorkspaceLayout>>,
}
impl PartitionedTextBindingDestinations {
    /// The exact retained source/residency selection.
    pub fn selected(&self)->&SelectedReplicatedTextRealization { &self.selected }
    /// Global execution addresses used to classify parameter tasks.
    pub fn global_layout(&self)->&ExecutionUnitLayout { &self.global_layout }
    /// Policy-local slots, in the ordinary materializer's address order.
    pub fn local_layout(&self)->&ExecutionUnitLayout { &self.local_layout }
    /// Ordered selected global unit addresses.
    pub fn addresses(&self)->&[ExecutionUnitAddress] { &self.addresses }
    /// Architecture-derived sharding recipes for these physical destinations.
    pub fn physical_layout(&self)->&LocalModelLayout { &self.physical_layout }
    /// Exact selected rank-local payload work.
    pub fn tasks(&self)->&[ReplicatedTextMaterializationTask] { &self.tasks }
    /// Static slots, including lazy off-rank slots excluded by selected tasks.
    pub fn static_parameters(&self)->&BTreeMap<String,WorkspaceLayout> { &self.static_parameters }
    /// Actual unit slots in policy-local order, preserving global parameter IDs.
    pub fn units(&self)->&[BTreeMap<String,WorkspaceLayout>] { &self.units }
}

/// Projects the selected partition before native load ownership is acquired.
/// Uses the same total architecture dispatcher, typed constructor and addressed
/// unit worker as native binding. The backend validates its final declarations
/// against the admitted source manager before adopting it.
pub fn project_partitioned_text_binding_destinations(sources:&PreparedModelSources, context:&WorkspaceContext)
    -> Result<PartitionedTextBindingDestinations, PreparedExecutionError<Error>> {
    let manifest=sources.selected().communication_manifest()
        .ok_or(PreparedExecutionError::MissingCommunication)?;
    let routes=PreparedExecutionRoutes::new().with_partitioned_dense(
        PartitionedDenseRoute::<WorkspaceBackend,ResidentState,_>::new(context,context,
            |_:PreparedPartitionResources<()>| PartitionDestinationVisitor(context)));
    construct_prepared_execution_impl(sources.clone(),Some(()),routes,
        PartitionDestinationAssembler{manifest,dtype:sources.selected().text_realization().state().floating_dtype()},
        ConstructionPurpose::BindingDestinations)
}
struct PartitionDestinationVisitor<'a>(&'a WorkspaceContext);
impl PartitionedArchitectureVisitor<WorkspaceBackend,ResidentState> for PartitionDestinationVisitor<'_> {
    type Output=PartitionedTextBindingDestinations;
    type Error=Error;
    fn visit<A,G>(self, prepared:PreparedPartitionedArchitecture<WorkspaceBackend,A,G,
        <A as eredu_runtime::PartitionedLayeredArchitecture<WorkspaceBackend,ResidentState>>::Boundary>,
        _:RetainedCheckpointSource)->Result<Self::Output,Error>
    where A:TextPartitionArchitecture<WorkspaceBackend,ResidentState>
        +ReplicatedTextArchitecture<WorkspaceBackend,ResidentState,Error=Error>+'static,
        A::StaticModules:Clone,G:'static {
        prepared.dispatch_execution(self,
            |prepared,visitor|visitor.collect_partition(prepared,PartitionedUnitScope::All),
            |prepared,visitor|visitor.collect_partition(prepared,PartitionedUnitScope::Owned))
    }
}
impl PartitionDestinationVisitor<'_> {
    fn collect_partition<A,G>(self,prepared:PreparedPartitionedArchitecture<WorkspaceBackend,A,G,
        <A as eredu_runtime::PartitionedLayeredArchitecture<WorkspaceBackend,ResidentState>>::Boundary>,
        scope:PartitionedUnitScope)->Result<PartitionedTextBindingDestinations,Error>
    where A:TextPartitionArchitecture<WorkspaceBackend,ResidentState>
        +ReplicatedTextArchitecture<WorkspaceBackend,ResidentState,Error=Error>+'static,
        A::StaticModules:Clone,G:'static {
        let (prepared,source,physical_layout,tasks)=prepared.into_parts();
        // A numerical transform is a separate producer; its target projection
        // alone cannot certify encoded reads for the original source manager.
        if source.is_some(){return Err(Error::backend("partition destination source requires a numerical transform"));}
        let (architecture,bound)=prepared.into_parts();
        let (selected,partition,_communication)=bound.into_parts();
        let addresses=eredu_runtime::partitioned_materialization_addresses(&partition,scope)
            .map_err(Error::backend)?;
        let global_layout=partition.unit_layout().clone();
        let graph=architecture.execution_graph()?;
        let local_layout=eredu_runtime::partitioned_materialization_unit_layout(&graph,&addresses)
            .map_err(Error::backend)?;
        let units=eredu_runtime::ordinary_addressed_units::<A,WorkspaceBackend,ResidentState>(
            &architecture,&addresses,self.0)?;
        let static_parameters=collect(architecture.static_modules())?;
        let units=units.iter().map(collect).collect::<Result<Vec<_>,Error>>()?;
        Ok(PartitionedTextBindingDestinations{selected,global_layout,local_layout,addresses,
            physical_layout,tasks,static_parameters,units})
    }
}
struct PartitionDestinationAssembler<'a> {
    manifest:&'a CommunicationManifest,
    dtype:Option<eredu_runtime::StateStorageDtype>,
}
impl PreparedExecutableAssembler<()> for PartitionDestinationAssembler<'_> {
    type Executable=PartitionedTextBindingDestinations;
    type Output=PartitionedTextBindingDestinations;
    type Error=Error;
    fn floating_state_dtype(&mut self,_:&FloatingStateDtypeSource)->Result<eredu_runtime::StateStorageDtype,Error> {
        Ok(self.dtype.unwrap_or(eredu_runtime::StateStorageDtype::F32))
    }
    fn validate_communication(&mut self,manifest:&CommunicationManifest,_:&())->Result<(),Error> {
        if manifest!=self.manifest{return Err(Error::backend("partition destination communication differs from retained selection"));}
        Ok(())
    }
    fn finish(self,parts:PreparedExecutableParts<Self::Executable,()>)->Result<Self::Output,Error> {
        Ok(parts.into_executable())
    }
}
