//! Closed construction policies for the shared resident unit build loop.
use super::*;
use eredu_nn::workspace::{WorkspaceBackend, WorkspaceContext, WorkspaceMetadataError};
use std::mem::{size_of, size_of_val};

// The same resident worker borrows either an ordinary constructed graph or the
// selected graph retained by a validated contract. No graph is copied here.
pub(super) enum ResidentGraph {
    Constructed(ExecutionGraph),
    Prepared(crate::PreparedReplicatedTextExecutionGeometry),
}
impl std::ops::Deref for ResidentGraph {
    type Target = ExecutionGraph;
    fn deref(&self) -> &ExecutionGraph {
        match self {
            Self::Constructed(graph) => graph,
            Self::Prepared(geometry) => geometry.graph(),
        }
    }
}

// The two implementations below own all reserve behavior. The workspace
// strategy prepares empty exact-count destinations, so the shared loop cannot
// grow them. The ordinary strategy keeps its existing post-build Vec growth.
trait UnitStorage<U, E> {
    fn groups(&mut self, count: usize) -> Result<Vec<Vec<U>>, E>;
    fn units(&mut self, count: usize) -> Result<Vec<U>, E>;
}
struct Ordinary;
impl<U, E> UnitStorage<U, E> for Ordinary {
    fn groups(&mut self, count: usize) -> Result<Vec<Vec<U>>, E> {
        Ok(Vec::with_capacity(count))
    }
    fn units(&mut self, _count: usize) -> Result<Vec<U>, E> {
        Ok(Vec::new())
    }
}
struct Metadata<'a>(&'a WorkspaceContext);
impl<U> UnitStorage<U, eredu_nn::Error> for Metadata<'_> {
    fn groups(&mut self, count: usize) -> Result<Vec<Vec<U>>, eredu_nn::Error> {
        self.0.metadata_vec(count)
    }
    fn units(&mut self, count: usize) -> Result<Vec<U>, eredu_nn::Error> {
        self.0.metadata_vec(count)
    }
}

pub(super) fn ordinary<A, B, S>(
    architecture: A,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<ResidentRuntime<A, B, S>, A::Error>
where
    B: NeuralBackend,
    S: RuntimeState<B>,
    A: LayeredArchitecture<B, S>,
{
    construct(architecture, context, Ordinary)
}

fn construct<A, B, S>(
    architecture: A,
    context: &<B::Tensor as Tensor>::Context,
    storage: impl UnitStorage<A::Unit, A::Error>,
) -> Result<ResidentRuntime<A, B, S>, A::Error>
where
    B: NeuralBackend,
    S: RuntimeState<B>,
    A: LayeredArchitecture<B, S>,
{
    let graph = architecture.execution_graph()?;
    construct_with_graph(
        architecture,
        ResidentGraph::Constructed(graph),
        context,
        storage,
    )
}

fn construct_with_graph<A, B, S>(
    architecture: A,
    graph: ResidentGraph,
    context: &<B::Tensor as Tensor>::Context,
    mut storage: impl UnitStorage<A::Unit, A::Error>,
) -> Result<ResidentRuntime<A, B, S>, A::Error>
where
    B: NeuralBackend,
    S: RuntimeState<B>,
    A: LayeredArchitecture<B, S>,
{
    let mut units = storage.groups(graph.groups().len())?;
    for group in 0..graph.groups().len() {
        let count = architecture.group_unit_count(group)?;
        let mut group_units = storage.units(count)?;
        for index in 0..count {
            // Ordinary construction still builds before growing its Vec. The
            // metadata strategy has already reserved the complete group here.
            push_unit_result(&mut group_units, architecture.build_unit(group, index, context))?;
        }
        units.push(group_units);
    }
    // Unit constructors can recursively quote a large architecture. Assemble
    // the owning runtime only after those calls return, so its by-value output
    // temporaries do not share their active constructor frame.
    finish(architecture, graph, units)
}

/// The same global-address traversal used by ordinary partition construction.
/// The caller owns the exact selected sequence and destination population.
fn construct_addressed_units<A,B,S>(architecture:&A, addresses:&[crate::ExecutionUnitAddress],
    context:&<B::Tensor as Tensor>::Context,mut units:Vec<A::Unit>)
    ->Result<Vec<A::Unit>,A::Error>
where B:NeuralBackend,S:RuntimeState<B>,A:LayeredArchitecture<B,S> {
    for address in addresses {
        push_unit_result(&mut units,architecture.build_unit(address.group(),address.index(),context))?;
    }
    Ok(units)
}

/// Builds the exact ordered global addresses selected by a partition constructor.
/// This is an allocating ordinary/cold construction helper, not a metadata or
/// native admission grant. Paid workspace execution uses its existing window.
pub fn ordinary_addressed_units<A,B,S>(architecture:&A,
    addresses:&[crate::ExecutionUnitAddress],context:&<B::Tensor as Tensor>::Context)
    ->Result<Vec<A::Unit>,A::Error>
where B:NeuralBackend,S:RuntimeState<B>,A:LayeredArchitecture<B,S> {
    construct_addressed_units(architecture,addresses,context,Vec::with_capacity(addresses.len()))
}

impl<U> ResidentUnitWindow<U> {
    /// Builds only the retained partition's global addresses, keeping their
    /// order as local policy slots. Uses the ordinary addressed-unit worker;
    /// each destination vector is paid before construction or allocation.
    pub fn from_workspace_addresses<A,S>(architecture:&A,
        addresses:&[crate::ExecutionUnitAddress],context:&WorkspaceContext)
        ->Result<Self,eredu_nn::Error>
    where S:RuntimeState<WorkspaceBackend>,
        A:LayeredArchitecture<WorkspaceBackend,S,Unit=U,Error=eredu_nn::Error> {
        let parts=[size_of::<Self>(),size_of::<Vec<U>>(),size_of::<Vec<Option<U>>>(),
            size_of::<Result<Vec<U>,eredu_nn::Error>>(),size_of::<Result<Self,eredu_nn::Error>>(),
            size_of::<(&A,&[crate::ExecutionUnitAddress],&WorkspaceContext)>(),
            size_of::<std::slice::Iter<'_,crate::ExecutionUnitAddress>>(),
            size_of::<crate::ExecutionUnitAddress>(),
            size_of::<(&mut Vec<U>,Result<U,eredu_nn::Error>)>(),
            size_of::<U>(),size_of::<U>(),size_of::<Result<(),eredu_nn::Error>>()];
        context.charge_metadata(parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)
            .ok_or(WorkspaceMetadataError::Overflow)?)?;
        let destination=context.metadata_vec(addresses.len())?;
        let populated=construct_addressed_units::<A,WorkspaceBackend,S>(architecture,addresses,context,destination)?;
        let mut units=context.metadata_vec(populated.len())?;
        for unit in populated { units.push(Some(unit)); }
        Ok(Self{units})
    }
}

// A unit can be much larger than its source architecture. Its Result must
// return from build_unit before extraction and Vec argument transports exist.
// Keeping those moves here avoids reserving all of them on the recursive cold
// constructor stack. The ordinary Vec still grows only after a successful build.
#[inline(never)]
fn push_unit_result<U, E>(units: &mut Vec<U>, result: Result<U, E>) -> Result<(), E> {
    match result {
        Ok(unit) => {
            units.push(unit);
            Ok(())
        }
        Err(cause) => Err(cause),
    }
}

#[inline(never)]
fn finish<A, B, S>(
    architecture: A,
    graph: ResidentGraph,
    units: Vec<Vec<A::Unit>>,
) -> Result<ResidentRuntime<A, B, S>, A::Error>
where
    B: NeuralBackend,
    S: RuntimeState<B>,
    A: LayeredArchitecture<B, S>,
{
    Ok(ResidentRuntime {
        architecture,
        graph,
        units,
        observation_binding: ObservationBinding::new(),
        backend: std::marker::PhantomData,
    })
}

impl<A, S> ResidentRuntime<A, WorkspaceBackend, S>
where
    S: RuntimeState<WorkspaceBackend>,
    A: LayeredArchitecture<WorkspaceBackend, S, Error = eredu_nn::Error>,
{
    /// Borrows the graph moved from this runtime's validated construction pair.
    /// This is historical metadata, not permission to alter or execute a source.
    pub fn workspace_prepared_execution_graph(&self) -> Result<&ExecutionGraph, eredu_nn::Error> {
        match &self.graph {
            ResidentGraph::Prepared(geometry) => Ok(geometry.graph()),
            ResidentGraph::Constructed(_) => Err(WorkspaceMetadataError::Unqualified.into()),
        }
    }

    /// Fixed owning transports for the checked resident unit-container worker.
    /// Actual outer/group Vec backings are queried by metadata_vec at their
    /// actual counts. Architecture graph/name and module payload constructors
    /// remain their own producers; this is not a complete architecture bound.
    pub fn workspace_unit_storage_control_bytes() -> Option<usize> {
        let parts = [
            size_of::<Self>(),
            size_of::<A>(),
            size_of::<A::Unit>(),
            size_of::<Metadata<'_>>(),
            size_of::<ResidentGraph>(),
            size_of::<ObservationBinding>(),
            size_of::<Result<Self, eredu_nn::Error>>(),
            size_of::<Result<A::Unit, eredu_nn::Error>>(),
            size_of::<Result<ExecutionGraph, eredu_nn::Error>>(),
            size_of::<Result<usize, eredu_nn::Error>>(),
            size_of::<(usize, usize, usize, &WorkspaceContext)>(),
            // The selected final assembly owns these transports after the unit
            // loop. Its enclosing call and output slot remain independently paid.
            size_of::<(A, ResidentGraph, Vec<Vec<A::Unit>>)>(),
            size_of::<Result<Self, eredu_nn::Error>>(),
            // Completed unit-result extraction and push run in their own frame.
            size_of::<(&mut Vec<A::Unit>, Result<A::Unit, eredu_nn::Error>)>(),
            size_of::<A::Unit>(),
            size_of::<A::Unit>(),
            size_of::<Result<(), eredu_nn::Error>>(),
        ];
        parts
            .into_iter()
            .try_fold(std::mem::size_of_val(&parts), usize::checked_add)
    }

    /// Builds the matching prepared architecture using its already validated
    /// graph. The unit build loop and per-group metadata reserves are identical
    /// to new_workspace; no graph or unit-layout constructor is repeated.
    pub fn new_workspace_with_prepared_geometry(
        architecture: A,
        geometry: crate::PreparedReplicatedTextExecutionGeometry,
        context: &WorkspaceContext,
    ) -> Result<Self, eredu_nn::Error> {
        context.charge_metadata(
            Self::workspace_unit_storage_control_bytes()
                .and_then(|bytes| bytes.checked_add(size_of_val(&geometry)))
                .ok_or(WorkspaceMetadataError::Overflow)?,
        )?;
        construct_with_graph(
            architecture,
            ResidentGraph::Prepared(geometry),
            context,
            Metadata(context),
        )
    }

    /// Builds the same resident units with counted metadata destinations. Each
    /// actual outer/group vector is admitted before its reserve; constructed
    /// prefixes drop on failure while the caller retains its host custody.
    /// The ordinary architecture methods still own graph and module semantics.
    pub fn new_workspace(
        architecture: A,
        context: &WorkspaceContext,
    ) -> Result<Self, eredu_nn::Error> {
        context.charge_metadata(
            Self::workspace_unit_storage_control_bytes().ok_or(WorkspaceMetadataError::Overflow)?,
        )?;
        let graph = architecture.execution_graph_with_metadata(context)?
            .into_owned_with_metadata(context)?;
        construct_with_graph(architecture, ResidentGraph::Constructed(graph), context, Metadata(context))
    }

    /// Moves the actual populated units into the existing resident policy for
    /// partition execution. Both destination vectors are charged before reserve;
    /// native values and the architecture are moved, never cloned.
    pub fn into_partition_workspace(self, context: &WorkspaceContext)
        -> Result<(A, ResidentUnitWindow<A::Unit>), eredu_nn::Error> {
        context.charge_metadata(size_of::<(Self, A, ResidentUnitWindow<A::Unit>,
            Vec<Option<A::Unit>>, Result<(A, ResidentUnitWindow<A::Unit>), eredu_nn::Error>)>())?;
        let count=self.units.iter().try_fold(0usize, |n, units|n.checked_add(units.len()))
            .ok_or(WorkspaceMetadataError::Overflow)?;
        let mut units=context.metadata_vec(count)?;
        let Self { architecture, units: groups, .. }=self;
        for group in groups { for unit in group { units.push(Some(unit)); } }
        Ok((architecture,ResidentUnitWindow{units}))
    }

    /// Moves the already populated resident units into the same one-at-a-time
    /// policy used by direct parallel execution. Each destination is paid before
    /// allocation; no unit or tensor is cloned and construction order is kept.
    pub fn into_layerwise_workspace(
        self,
        context: &WorkspaceContext,
    ) -> Result<LayerwiseRuntime<A, WorkspaceBackend, S, ResidentUnitWindow<A::Unit>>, eredu_nn::Error> {
        context.charge_metadata(size_of::<(
            Self, Vec<Option<A::Unit>>, usize, Vec<usize>, crate::PreparedReplicatedTextExecutionGeometry,
            LayerwiseRuntime<A, WorkspaceBackend, S, ResidentUnitWindow<A::Unit>>,
            Result<LayerwiseRuntime<A, WorkspaceBackend, S, ResidentUnitWindow<A::Unit>>, eredu_nn::Error>,
        )>())?;
        let count = self.units.iter().try_fold(0usize, |count, units| count.checked_add(units.len()))
            .ok_or(WorkspaceMetadataError::Overflow)?;
        let mut units = context.metadata_vec(count)?;
        let mut counts=context.metadata_vec(self.units.len())?;
        counts.extend(self.units.iter().map(Vec::len));
        let Self { architecture, graph, units: groups, .. } = self;
        let geometry=match graph {
            ResidentGraph::Prepared(geometry)=>geometry,
            ResidentGraph::Constructed(graph)=>crate::PreparedReplicatedTextExecutionGeometry::from_workspace_units(graph,&counts,context)?,
        };
        for group in groups { for unit in group { units.push(Some(unit)); } }
        Ok(LayerwiseRuntime::new_with_prepared_geometry(architecture, ResidentUnitWindow { units },geometry))
    }
}

impl<A, S, P> LayerwiseRuntime<A, WorkspaceBackend, S, P>
where
    S: RuntimeState<WorkspaceBackend>,
    A: LayeredArchitecture<WorkspaceBackend, S, Error = eredu_nn::Error>,
    P: LayerwisePolicy<WorkspaceBackend, A::Unit>,
    P::Error: std::fmt::Display,
{
    /// Builds the ordinary architecture graph and unit geometry with paid
    /// metadata, then selects a policy against that exact local unit layout.
    /// Unit materialization remains entirely in the existing acquire driver.
    pub fn new_workspace_with_policy<F>(architecture:A, prepare:F, context:&WorkspaceContext)
        -> Result<Self,eredu_nn::Error>
    where F:FnOnce(&crate::ExecutionUnitLayout)->Result<P,eredu_nn::Error> {
        let frames=[size_of::<Self>(),size_of::<A>(),size_of::<P>(),size_of::<F>(),
            size_of::<Result<Self,eredu_nn::Error>>(),size_of::<Result<P,eredu_nn::Error>>(),
            size_of::<crate::ExecutionGraph>(),size_of::<Vec<usize>>(),
            size_of::<crate::PreparedReplicatedTextExecutionGeometry>(),
            size_of::<Result<crate::PreparedReplicatedTextExecutionGeometry,eredu_nn::Error>>(),
            size_of::<(&crate::ExecutionUnitLayout,&WorkspaceContext)>(),size_of::<usize>()];
        context.charge_metadata(frames.into_iter().try_fold(size_of_val(&frames),usize::checked_add)
            .ok_or(WorkspaceMetadataError::Overflow)?)?;
        let graph=architecture.execution_graph_with_metadata(context)?.into_owned_with_metadata(context)?;
        let mut counts=context.metadata_vec(graph.groups().len())?;
        for group in 0..graph.groups().len() { counts.push(architecture.group_unit_count(group)?); }
        let geometry=crate::PreparedReplicatedTextExecutionGeometry::from_workspace_units(graph,&counts,context)?;
        let policy=prepare(geometry.units())?;
        Ok(Self::new_with_prepared_geometry(architecture,policy,geometry))
    }

    /// Borrows only a still-valid graph moved from the matching prepared pair.
    /// Arbitrary architecture mutation never falls back to graph reconstruction.
    pub fn workspace_prepared_execution_graph(&self) -> Result<&ExecutionGraph, eredu_nn::Error> {
        if self.geometry_stale {
            return Err(WorkspaceMetadataError::Unqualified.into());
        }
        self.prepared_geometry
            .as_ref()
            .map(|geometry| geometry.graph())
            .ok_or_else(|| WorkspaceMetadataError::Unqualified.into())
    }
}
