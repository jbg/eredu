//! Canonical local-stage traversal over the shared partition and residency lifecycle.
use crate::{
    ArchitectureBoundary, ArchitecturePartition, ExecutionUnitLayout, ExecutionUnitLayoutError,
    LayeredArchitecture, LayeredForwardState, LayeredPartitionBeginError, LayeredPartitionDriver,
    LayeredPartitionError, LayeredPartitionOutput, LayerwiseAcquireError, LayerwisePolicy,
    LayerwisePolicyForward, PartitionedLayeredArchitecture, RuntimeState, StateError, StateLayout,
};
use eredu_nn::{NeuralBackend, Tensor};
use std::marker::PhantomData;

/// Rejected local stage geometry before any invocation or parameter acquisition.
#[derive(Debug, thiserror::Error)]
pub enum LayerwisePartitionConstructionError {
    /// Component selection differs from its validated declaration.
    #[error(transparent)]
    Definition(#[from] crate::ComponentDefinitionError),
    /// Canonical partition or architecture declarations disagree.
    #[error("invalid layered stage: {0}")]
    Contract(String),
    /// A selected group does not have the declared unit/state ownership.
    #[error(transparent)]
    Partition(#[from] LayeredPartitionError),
    /// Selected storage addresses cannot represent the canonical units.
    #[error(transparent)]
    Layout(#[from] ExecutionUnitLayoutError),
}

/// A failed local invocation leaves candidate state with its caller and aborts its policy lease.
#[derive(Debug, thiserror::Error)]
pub enum LayerwisePartitionRuntimeError<A: std::fmt::Display, P> {
    /// Architecture-owned ingress or partition preparation failed.
    #[error(transparent)]
    Begin(#[from] LayeredPartitionBeginError<A>),
    /// A unit, group transition or final boundary failed.
    #[error("layered stage architecture failed: {0}")]
    Architecture(#[source] A),
    /// Residency acquisition, completion or finish failed.
    #[error("layered stage residency failed: {0}")]
    Policy(#[source] P),
    /// A retained state root does not belong to the supplied state.
    #[error(transparent)]
    State(#[from] StateError),
}

/// One selected local chain of architecture groups with canonical parameter addresses.
///
/// This owns parameter providers and graph traversal only. Mutable model state,
/// pipeline transport, publication, rollback and generation scheduling remain with
/// their existing owners. One borrowed-state invocation prepares ingress once,
/// retains its forward context across local groups and emits one final boundary.
pub struct LayerwisePartitionRuntime<A, B, S, P>
where
    B: NeuralBackend,
    S: RuntimeState<B>,
    A: PartitionedLayeredArchitecture<B, S>,
    P: LayerwisePolicy<B, A::Unit>,
{
    architecture: A,
    policy: P,
    drivers: Vec<LayeredPartitionDriver>,
    complete_layout: ExecutionUnitLayout,
    selected_layout: ExecutionUnitLayout,
    marker: PhantomData<fn() -> (B, S)>,
}

impl<A, B, S, P> LayerwisePartitionRuntime<A, B, S, P>
where
    B: NeuralBackend,
    S: RuntimeState<B>,
    A: PartitionedLayeredArchitecture<B, S>,
    A::Error: std::fmt::Display,
    P: LayerwisePolicy<B, A::Unit>,
{
    /// Adopts one exact selected stage and its already prepared parameter policy.
    /// Architectures identify each group's global state-slot range, including explicit
    /// NoState schedules. Stateless groups such as final normalization return `None`.
    /// Local groups must form a contiguous dependency chain; joins and branches
    /// require the shared graph/pipeline executor instead of this chain adapter.
    pub fn new<G>(
        architecture: A,
        policy: P,
        partition: &ArchitecturePartition<G, A::Boundary>,
        group_state_range: impl Fn(usize) -> Option<std::ops::Range<usize>>,
    ) -> Result<Self, LayerwisePartitionConstructionError> {
        let invalid = |error: String| LayerwisePartitionConstructionError::Contract(error);
        partition
            .validate_architecture::<B, S, A>(&architecture)
            .map_err(|error| invalid(error.to_string()))?;
        let actual = architecture
            .boundary_schema()
            .map_err(|error| invalid(error.to_string()))?
            .wire_schema()
            .map_err(|error| invalid(error.to_string()))?;
        let selected = partition
            .boundary_schema()
            .wire_schema()
            .map_err(|error| invalid(error.to_string()))?;
        if actual != selected {
            return Err(invalid(
                "partition boundary schema differs from architecture".into(),
            ));
        }
        let graph = partition.graph();
        let mut drivers = Vec::new();
        for &group in graph.execution_order() {
            if let Some(owned) = partition
                .groups()
                .iter()
                .find(|owned| owned.group_index() == group)
            {
                let range = owned.global_units();
                let dependencies = graph.dependencies(group).expect("validated graph group");
                if let Some(previous) = drivers.last() {
                    let previous: &LayeredPartitionDriver = previous;
                    let layout = partition.unit_layout();
                    let previous_group = layout
                        .group_range(previous.group_index())
                        .expect("validated group");
                    let current_group = layout.group_range(group).expect("validated group");
                    if dependencies != [previous.group_index()]
                        || previous.range().end - 1
                            != layout.address(previous_group.end - 1).unwrap().index()
                        || range.start != layout.address(current_group.start).unwrap().index()
                    {
                        return Err(invalid(
                            "local stage groups are not a contiguous complete dependency chain"
                                .into(),
                        ));
                    }
                } else if dependencies.len() > 1 {
                    return Err(invalid(
                        "a local chain ingress cannot merge multiple graph dependencies".into(),
                    ));
                }
                drivers.push(LayeredPartitionDriver::new_with_state_range(
                    partition,
                    group,
                    range,
                    group_state_range(group),
                )?);
            }
        }
        if drivers.is_empty() {
            return Err(invalid("local stage has no execution groups".into()));
        }
        let mut addresses = partition.units().collect::<Vec<_>>();
        addresses.sort_by_key(|address| (address.group(), address.index()));
        let selected_layout = partition.unit_layout().select(addresses)?;
        Ok(Self {
            architecture,
            policy,
            drivers,
            complete_layout: partition.unit_layout().clone(),
            selected_layout,
            marker: PhantomData,
        })
    }

    /// Architecture equations and selected ownership; this borrows no mutable state.
    pub fn architecture(&self) -> &A {
        &self.architecture
    }
    /// Mutable architecture parameter access remains local to this provider.
    pub fn architecture_mut(&mut self) -> &mut A {
        &mut self.architecture
    }
    /// Selected provider and its residency reports.
    pub fn policy(&self) -> &P {
        &self.policy
    }
    /// Selected provider for bounded parameter operations.
    pub fn policy_mut(&mut self) -> &mut P {
        &mut self.policy
    }
    /// Dense local storage ordinals with exact canonical architecture addresses.
    pub fn unit_layout(&self) -> &ExecutionUnitLayout {
        &self.selected_layout
    }

    /// Runs the shared lease/completion lifecycle against a caller-owned state branch.
    /// The architecture callback converts supplied ingress or an upstream boundary.
    /// It executes once; subsequent local groups reuse its exact context and state.
    pub fn forward_with_preparation<S2, F>(
        &mut self,
        state: &mut S2,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
        prepare: F,
    ) -> Result<
        LayeredPartitionOutput<B::Tensor, <<A as PartitionedLayeredArchitecture<B, S2>>::Boundary as ArchitectureBoundary>::Boundary<B::Tensor>>,
        LayerwisePartitionRuntimeError<<A as LayeredArchitecture<B, S>>::Error, P::Error>,
    >
    where
        S2: RuntimeState<B>,
        A: PartitionedLayeredArchitecture<
                B,
                S2,
                Boundary = <A as PartitionedLayeredArchitecture<B, S>>::Boundary,
            > + LayeredArchitecture<B, S2, Unit = <A as LayeredArchitecture<B, S>>::Unit, Error = <A as LayeredArchitecture<B, S>>::Error>,
        F: FnOnce(&mut A, &mut S2, Option<&StateLayout>) -> Result<
            LayeredForwardState<B::Tensor, <A as LayeredArchitecture<B, S2>>::ForwardContext>,
            <A as LayeredArchitecture<B, S>>::Error,
        >,
    {
        let first = &self.drivers[0];
        let mut forward = first.begin_with_preparation::<B, S2, A, _>(
            &mut self.architecture,
            state,
            parallel,
            context,
            prepare,
        )?;
        let mut policy = LayerwisePolicyForward::begin(&mut self.policy, &forward.hidden, context)
            .map_err(LayerwisePartitionRuntimeError::Policy)?;
        for (step, driver) in self.drivers.iter().enumerate() {
            let group = driver.group_index();
            if step > 0 {
                forward.hidden =
                    <A as PartitionedLayeredArchitecture<B, S2>>::enter_partition_group(
                        &mut self.architecture,
                        group,
                        &forward.hidden,
                        state,
                        &mut forward.context,
                        parallel,
                        context,
                    )
                    .map_err(LayerwisePartitionRuntimeError::Architecture)?;
            }
            for index in driver.range() {
                let ordinal = self
                    .selected_layout
                    .ordinal(group, index)
                    .expect("selected canonical unit");
                let address = self
                    .selected_layout
                    .address(ordinal)
                    .expect("selected storage ordinal");
                let canonical_ordinal = self
                    .complete_layout
                    .ordinal(group, index)
                    .expect("canonical unit");
                let lease = policy
                    .acquire(ordinal, address, |executor| {
                        <A as LayeredArchitecture<B, S2>>::build_unit(
                            &self.architecture,
                            group,
                            index,
                            executor,
                        )
                    })
                    .map_err(|error| match error {
                        LayerwiseAcquireError::Architecture(error) => {
                            LayerwisePartitionRuntimeError::Architecture(error)
                        }
                        LayerwiseAcquireError::Policy(error) => {
                            LayerwisePartitionRuntimeError::Policy(error)
                        }
                    })?;
                forward.hidden = match parallel {
                    Some(parallel) => {
                        <A as crate::ParallelLayeredArchitecture<B, S2>>::forward_unit_parallel(
                            &mut self.architecture,
                            group,
                            index,
                            lease,
                            &forward.hidden,
                            state,
                            &mut forward.context,
                            parallel,
                            context,
                        )
                    }
                    None => <A as LayeredArchitecture<B, S2>>::forward_unit(
                        &mut self.architecture,
                        group,
                        index,
                        lease,
                        &forward.hidden,
                        state,
                        &mut forward.context,
                        context,
                    ),
                }
                .map_err(LayerwisePartitionRuntimeError::Architecture)?;
                let mut state_values = Vec::new();
                for state_ordinal in <A as LayeredArchitecture<B, S2>>::retained_state_ordinals(
                    &self.architecture,
                    group,
                    index,
                    canonical_ordinal,
                ) {
                    state_values.extend(
                        state.retained_values(state_ordinal, address.with_index(state_ordinal))?,
                    );
                }
                let context_values = <A as LayeredArchitecture<B, S2>>::retained_context_values(
                    &self.architecture,
                    &forward.context,
                    group,
                    index,
                );
                policy
                    .complete(&forward.hidden, state_values.into_iter(), context_values)
                    .map_err(LayerwisePartitionRuntimeError::Policy)?;
            }
            if step + 1 < self.drivers.len() {
                forward.hidden =
                    <A as PartitionedLayeredArchitecture<B, S2>>::leave_partition_group(
                        &mut self.architecture,
                        group,
                        &forward.hidden,
                        state,
                        &mut forward.context,
                        parallel,
                        context,
                    )
                    .map_err(LayerwisePartitionRuntimeError::Architecture)?;
            }
        }
        let output = self
            .drivers
            .last()
            .expect("validated nonempty stage")
            .finish::<B, S2, A>(
                &mut self.architecture,
                &forward.hidden,
                state,
                &mut forward.context,
                parallel,
                context,
            )
            .map_err(LayerwisePartitionRuntimeError::Architecture)?;
        let primary = match &output {
            LayeredPartitionOutput::Final { output, .. } => output,
            LayeredPartitionOutput::Boundary { hidden, .. } => hidden,
        };
        policy
            .finish(primary)
            .map_err(LayerwisePartitionRuntimeError::Policy)?;
        Ok(output)
    }
}
