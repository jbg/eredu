//! Statically dispatched layered-architecture lifecycle and resident execution.

#![allow(clippy::too_many_arguments, clippy::type_complexity)]

use std::collections::BTreeMap;

use eredu_checkpoint::{recipe::DerivedWeightRecipe, store::CheckpointSource};
use eredu_nn::{NeuralBackend, Parameterized};

use crate::{
    observe_and_intervene, ActivationObserver, ExecutionGraph, ExecutionGroupSchedule,
    ExecutionUnitLayout, ExpertPass, NoAuxiliaryBoundary, ObservedExpertProvider,
    RoutedExpertProvider, RoutedObservationPoint, RuntimeState, StateLayout, SubmissionBackend,
};

/// Statically dispatched visitor over one immutable pinned parameter module.
pub trait StaticParameterVisitor<B: NeuralBackend> {
    /// Failure returned by the consumer.
    type Error;

    /// Visits the module bound to one architecture-declared static role.
    fn visit<M>(&mut self, role: &str, module: &M) -> Result<(), Self::Error>
    where
        M: Parameterized<B::Tensor>;
}

/// Statically dispatched visitor over one mutable pinned parameter module.
pub trait StaticParameterVisitorMut<B: NeuralBackend> {
    /// Failure returned by the consumer.
    type Error;

    /// Visits the mutable module bound to one architecture-declared static role.
    fn visit_mut<M>(&mut self, role: &str, module: &mut M) -> Result<(), Self::Error>
    where
        M: Parameterized<B::Tensor>;
}

/// Architecture-owned enumeration and binding of pinned parameter modules.
///
/// Parameter descriptions select the roles owned by a partition. This
/// contract resolves those roles to concrete neutral modules without making a
/// backend know family fields or checkpoint roots.
pub trait ArchitectureParameters<B: NeuralBackend> {
    /// Architecture-owned failure while deriving geometry or topology.
    type DefinitionError;

    /// Returns the authoritative mutable-state geometry for this realization.
    fn state_layout(&self) -> Result<StateLayout, Self::DefinitionError>;

    /// Describes every parameter with its canonical graph owner and placement.
    fn parameter_description(
        &self,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<crate::ArchitectureParameterDescription, Self::DefinitionError>;

    /// Returns architecture-owned checkpoint rewrites for pinned parameters.
    fn static_parameter_recipes(
        &self,
        _source: &dyn CheckpointSource,
    ) -> Result<BTreeMap<String, DerivedWeightRecipe>, String> {
        Ok(BTreeMap::new())
    }

    /// Visits every available pinned parameter module exactly once.
    fn visit_static_parameters<V>(&self, visitor: &mut V) -> Result<(), V::Error>
    where
        V: StaticParameterVisitor<B>;

    /// Mutably visits every available pinned parameter module exactly once.
    fn visit_static_parameters_mut<V>(&mut self, visitor: &mut V) -> Result<(), V::Error>
    where
        V: StaticParameterVisitorMut<B>;
}

/// Backend-native activation and architecture-owned forward context.
pub struct LayeredForwardState<T, C> {
    /// Initial activation supplied to the first execution unit.
    pub hidden: T,
    /// Masks, positions, or other architecture-owned forward values.
    pub context: C,
}

/// Architecture-authored semantic kind for one transport-visible execution group.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum ArchitectureGroupKind {
    /// Primary text decoding.
    Decoder,
    /// Embedded prediction after the primary decoder output.
    Prediction,
    /// Visual encoding.
    VisionEncoder,
    /// Audio encoding.
    AudioEncoder,
    /// Learned modality projection.
    Projector,
    /// Learned or structural modality merge.
    Merger,
    /// Final multimodal assembly.
    ModalityFinalization,
}

/// Pipeline ownership policy for one architecture execution group.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum ArchitectureGroupPlacement {
    /// Balance the group across every pipeline owner.
    Pipeline,
    /// Place the complete group on the architecture output owner.
    OutputOwner,
}

/// Architecture-level merge destination resolved by a concrete pipeline topology.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum ArchitectureMergeDestination {
    /// Use the group's terminal owner.
    LastOwner,
    /// Return the result to the first pipeline owner for dependency assembly.
    FirstPipelineOwner,
    /// Deliver the result to the architecture output owner.
    OutputOwner,
}

/// Cartesian subgroup semantics required while a group executes.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum ArchitectureParallelSubgroup {
    /// Tensor sharding without routed expert exchange.
    TensorSharded,
    /// Decoder tensor and routed-expert parallelism.
    Decoder,
}

/// Backend-neutral transport and placement semantics for one execution group.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ArchitectureGroupTransport {
    /// Physical pipeline ownership policy.
    pub placement: ArchitectureGroupPlacement,
    /// Semantic compute kind.
    pub kind: ArchitectureGroupKind,
    /// Static roles owned by the group's first physical owner.
    pub first_owner_static_roles: Vec<String>,
    /// Static roles owned by the group's terminal physical owner.
    pub last_owner_static_roles: Vec<String>,
    /// Dependency merge destination.
    pub merge_destination: ArchitectureMergeDestination,
    /// Optional active Cartesian subgroup contract.
    pub parallel_subgroup: Option<ArchitectureParallelSubgroup>,
    /// Whether request data may omit the group entirely.
    pub request_optional: bool,
}

/// Stable layered traversal boundary exposed to generic runtime drivers.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum LayeredTraversalPoint {
    /// Output of one execution unit before the next unit starts.
    Unit {
        /// Execution-group index.
        group: usize,
        /// Group-local unit index.
        index: usize,
    },
    /// Output of one completed execution group.
    Group {
        /// Execution-group index.
        group: usize,
    },
}

/// Decision returned immediately before one execution unit is acquired.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum LayeredUnitAction {
    /// Execute this unit normally.
    Execute,
    /// Omit this unit and every remaining unit in the current group.
    SkipRemainingGroup,
}

/// Statically dispatched hook shared by resident and bounded layered traversal.
///
/// The hook can observe unit/group outputs and can omit only a complete group
/// tail. Drivers are responsible for proving that an omission preserves their
/// semantics before returning [`LayeredUnitAction::SkipRemainingGroup`].
pub trait LayeredTraversalHook<B, C, E>
where
    B: NeuralBackend,
{
    /// Chooses whether to execute the next unit.
    fn before_unit(
        &mut self,
        _group: usize,
        _index: usize,
        _remaining_units: usize,
        _value: &B::Tensor,
        _forward: &mut C,
        _context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<LayeredUnitAction, E> {
        Ok(LayeredUnitAction::Execute)
    }

    /// Observes one executed unit output.
    fn after_unit(
        &mut self,
        _group: usize,
        _index: usize,
        _value: &B::Tensor,
        _forward: &mut C,
        _context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<(), E> {
        Ok(())
    }

    /// Observes one completed execution-group output.
    fn after_group(
        &mut self,
        _group: usize,
        _value: &B::Tensor,
        _forward: &mut C,
        _context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<(), E> {
        Ok(())
    }
}

/// Statically combines two traversal hooks over one production forward pass.
///
/// Both hooks observe every reached boundary in left-to-right order. A unit is
/// skipped when either hook proves that the remaining group tail can be
/// omitted; errors stop delegation before any later callback is invoked.
pub struct CompositeLayeredTraversalHook<L, R> {
    left: L,
    right: R,
}

impl<L, R> CompositeLayeredTraversalHook<L, R> {
    /// Creates one ordered pair of traversal hooks.
    pub const fn new(left: L, right: R) -> Self {
        Self { left, right }
    }

    /// Returns both hooks after traversal.
    pub fn into_parts(self) -> (L, R) {
        (self.left, self.right)
    }
}

impl<B, C, E, L, R> LayeredTraversalHook<B, C, E> for CompositeLayeredTraversalHook<L, R>
where
    B: NeuralBackend,
    L: LayeredTraversalHook<B, C, E>,
    R: LayeredTraversalHook<B, C, E>,
{
    fn before_unit(
        &mut self,
        group: usize,
        index: usize,
        remaining_units: usize,
        value: &B::Tensor,
        forward: &mut C,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<LayeredUnitAction, E> {
        let left = self
            .left
            .before_unit(group, index, remaining_units, value, forward, context)?;
        let right =
            self.right
                .before_unit(group, index, remaining_units, value, forward, context)?;
        Ok(
            if left == LayeredUnitAction::SkipRemainingGroup
                || right == LayeredUnitAction::SkipRemainingGroup
            {
                LayeredUnitAction::SkipRemainingGroup
            } else {
                LayeredUnitAction::Execute
            },
        )
    }

    fn after_unit(
        &mut self,
        group: usize,
        index: usize,
        value: &B::Tensor,
        forward: &mut C,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<(), E> {
        self.left
            .after_unit(group, index, value, forward, context)?;
        self.right.after_unit(group, index, value, forward, context)
    }

    fn after_group(
        &mut self,
        group: usize,
        value: &B::Tensor,
        forward: &mut C,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<(), E> {
        self.left.after_group(group, value, forward, context)?;
        self.right.after_group(group, value, forward, context)
    }
}

struct NoopLayeredTraversalHook;

impl<B, C, E> LayeredTraversalHook<B, C, E> for NoopLayeredTraversalHook where B: NeuralBackend {}

struct AfterUnitTraversalHook<F> {
    after_unit: F,
}

struct AfterUnitContextTraversalHook<F> {
    after_unit: F,
}

impl<B, C, E, F> LayeredTraversalHook<B, C, E> for AfterUnitTraversalHook<F>
where
    B: NeuralBackend,
    F: FnMut(usize, usize, &B::Tensor, &mut C) -> Result<(), E>,
{
    fn after_unit(
        &mut self,
        group: usize,
        index: usize,
        value: &B::Tensor,
        forward: &mut C,
        _context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<(), E> {
        (self.after_unit)(group, index, value, forward)
    }
}

impl<B, C, E, F> LayeredTraversalHook<B, C, E> for AfterUnitContextTraversalHook<F>
where
    B: NeuralBackend,
    F: FnMut(usize, usize, &mut C) -> Result<(), E>,
{
    fn after_unit(
        &mut self,
        group: usize,
        index: usize,
        _value: &B::Tensor,
        forward: &mut C,
        _context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<(), E> {
        (self.after_unit)(group, index, forward)
    }
}

/// Backend-neutral lifecycle implemented once by a layered architecture.
///
/// All hot values remain concrete associated types. Resident and bounded
/// runtime policies call these same methods without erasing tensors, units, or
/// mutable layer state.
pub trait LayeredArchitecture<B, S>:
    ArchitectureParameters<B, DefinitionError = Self::Error>
where
    B: NeuralBackend,
    S: RuntimeState<B>,
{
    /// Borrowed prepared model input.
    type Input<'a>
    where
        Self: 'a;
    /// Pinned model modules such as embeddings, final normalization, and head.
    type StaticModules: Parameterized<B::Tensor>;
    /// One ordered execution unit.
    type Unit: Parameterized<B::Tensor>;
    /// Architecture-owned state retained for one complete forward pass.
    type ForwardContext;
    /// Allocation-free iterator over transient tensors retained by a unit submission.
    type RetainedContextValues<'a>: Iterator<Item = &'a B::Tensor>
    where
        Self: 'a,
        B::Tensor: 'a;
    /// Concrete architecture or backend failure.
    type Error;

    /// Declares transport and physical placement semantics for one canonical group slot.
    fn group_transport(&self, group: usize) -> ArchitectureGroupTransport;

    /// Declares how the complete mutable-state layout is divided among realized partitions.
    fn state_partition_plan(&self, layout: &StateLayout) -> crate::ArchitectureStatePartitionPlan;

    /// Stable architecture compatibility identity.
    fn model_identity(&self) -> &str;

    /// Declares the dependency graph between ordered execution groups.
    fn execution_graph(&self) -> Result<ExecutionGraph, Self::Error>;

    /// Returns the number of ordered execution units in one graph group.
    fn group_unit_count(&self, group: usize) -> Result<usize, Self::Error>;

    /// Returns the stable architecture-owned path of one group-local execution unit.
    fn unit_path(&self, group: usize, index: usize) -> Result<String, Self::Error>;

    /// Borrows pinned modules for parameter discovery and binding.
    fn static_modules(&self) -> &Self::StaticModules;

    /// Mutably borrows pinned modules for parameter binding.
    fn static_modules_mut(&mut self) -> &mut Self::StaticModules;

    /// Builds one unloaded execution unit using backend-native operators.
    fn build_unit(
        &self,
        group: usize,
        index: usize,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<Self::Unit, Self::Error>;

    /// Embeds input and prepares architecture-owned forward values.
    fn begin_forward<'a>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut S,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error>;

    /// Selects or merges the activation consumed by one ready execution group.
    fn begin_execution_group(
        &mut self,
        group: usize,
        initial: &B::Tensor,
        dependencies: &[&B::Tensor],
        state: &mut S,
        forward: &mut Self::ForwardContext,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error>;

    /// Returns whether one ready group is needed for this forward pass.
    fn should_execute_group(&self, _group: usize, _forward: &Self::ForwardContext) -> bool {
        true
    }

    /// Maps one execution unit to its architecture-global mutable-state slot.
    ///
    /// The default matches a single flattened decoder schedule. Composite
    /// architectures can keep parameter-only groups outside their state layout
    /// and remap later groups onto their semantic decoder or predictor layers.
    fn state_ordinal(&self, _group: usize, _index: usize, ordinal: usize) -> usize {
        ordinal
    }

    /// Returns every architecture-global state slot retained by one unit.
    ///
    /// The default retains the single slot returned by [`Self::state_ordinal`].
    /// Composite units can return a contiguous range when one residency unit
    /// internally executes several stateful layers.
    fn retained_state_ordinals(
        &self,
        group: usize,
        index: usize,
        ordinal: usize,
    ) -> std::ops::Range<usize> {
        let state = self.state_ordinal(group, index, ordinal);
        state..state + 1
    }

    /// Executes one ordered unit against its concrete mutable layer state.
    fn forward_unit(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error>;

    /// Converts a completed group's output into its dependency-facing value.
    fn complete_execution_group(
        &mut self,
        _group: usize,
        hidden: &B::Tensor,
        _state: &mut S,
        _forward: &mut Self::ForwardContext,
        _context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        Ok(hidden.clone())
    }

    /// Applies final normalization and output projection.
    fn finish_forward(
        &mut self,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &Self::ForwardContext,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error>;

    /// Borrows transient forward tensors required by one unit's submission.
    fn retained_context_values<'a>(
        &'a self,
        forward: &'a Self::ForwardContext,
        group: usize,
        index: usize,
    ) -> Self::RetainedContextValues<'a>;
}

/// Optional statically dispatched parallel lifecycle for a layered architecture.
///
/// The runtime owns traversal and exact unit completion while the architecture
/// owns parallel embedding, block, and output semantics. Backend-native
/// collective contexts cross this boundary unchanged.
pub trait ParallelLayeredArchitecture<B, S>: LayeredArchitecture<B, S>
where
    B: NeuralBackend,
    S: RuntimeState<B>,
{
    /// Embeds input and prepares forward values for rank-local execution.
    fn begin_forward_parallel<'a>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut S,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error>;

    /// Executes one rank-local unit and its required collectives.
    fn forward_unit_parallel(
        &mut self,
        group_index: usize,
        index: usize,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error>;

    /// Selects or merges a ready group's activation under a parallel context.
    fn begin_execution_group_parallel(
        &mut self,
        group_index: usize,
        initial: &B::Tensor,
        dependencies: &[&B::Tensor],
        state: &mut S,
        forward: &mut Self::ForwardContext,
        _parallel: &B::ParallelContext,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        self.begin_execution_group(group_index, initial, dependencies, state, forward, context)
    }

    /// Converts a completed group's output under a parallel context.
    fn complete_execution_group_parallel(
        &mut self,
        group_index: usize,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        _parallel: &B::ParallelContext,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        self.complete_execution_group(group_index, hidden, state, forward, context)
    }

    /// Applies the rank-local output projection and returns complete logits.
    fn finish_forward_parallel(
        &mut self,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &Self::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error>;
}

/// Input accepted at a rank-local layered partition boundary.
///
/// A partition either embeds borrowed token ids or consumes an owned hidden
/// tensor prepared by architecture ingress or a preceding pipeline owner.
#[derive(Debug)]
pub enum LayeredPartitionInput<'a, T, A = NoAuxiliaryBoundary> {
    /// Token ids supplied to the architecture input owner.
    Tokens(&'a T),
    /// Architecture-prepared or upstream hidden state.
    Hidden {
        /// Evolving activation received from the preceding owner.
        hidden: T,
        /// Architecture-typed context carried across the partition boundary.
        auxiliary: A,
    },
}

/// Architecture-owned result of completing one layered partition.
///
/// The runtime and concrete transports only distinguish a final output from a
/// transport boundary. Families retain ownership of auxiliary boundary values
/// and of any hidden activation required by an embedded predictor.
pub enum LayeredPartitionOutput<T, A = NoAuxiliaryBoundary> {
    /// Complete architecture output produced by the output owner.
    Final {
        /// Projected architecture output, normally vocabulary logits.
        output: T,
        /// Optional pre-projection value consumed by an embedded predictor.
        retained: Option<T>,
    },
    /// Values transported to the next pipeline owner.
    Boundary {
        /// Evolving activation.
        hidden: T,
        /// Architecture-typed auxiliary context.
        auxiliary: A,
    },
}

/// Architecture-owned preparation for rank-local partition execution.
///
/// The neutral partition driver owns validation, group sequencing, and output
/// ownership. Architectures own the semantic conversion of partition inputs,
/// entry into and completion of their selected execution group, and typed
/// partition output.
pub trait PartitionedLayeredArchitecture<B, S>: ParallelLayeredArchitecture<B, S>
where
    B: NeuralBackend,
    S: RuntimeState<B>,
{
    /// Architecture-owned schema for primary and auxiliary partition transport.
    type Boundary: crate::ArchitectureBoundary;

    /// Derives the complete transport schema from the normalized architecture.
    fn boundary_schema(&self) -> Result<Self::Boundary, Self::Error>;

    /// Prepares a replicated partition from tokens or upstream hidden state.
    fn begin_partition<'a>(
        &mut self,
        input: LayeredPartitionInput<
            'a,
            B::Tensor,
            <Self::Boundary as crate::ArchitectureBoundary>::Boundary<B::Tensor>,
        >,
        mask: Option<&B::Tensor>,
        state: &mut S,
        expected: &crate::StateLayout,
        first_state_ordinal: usize,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error>;

    /// Prepares the tensor-parallel form of the same partition.
    #[allow(clippy::too_many_arguments)]
    fn begin_partition_parallel<'a>(
        &mut self,
        input: LayeredPartitionInput<
            'a,
            B::Tensor,
            <Self::Boundary as crate::ArchitectureBoundary>::Boundary<B::Tensor>,
        >,
        mask: Option<&B::Tensor>,
        state: &mut S,
        expected: &crate::StateLayout,
        first_state_ordinal: usize,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error>;

    /// Enters the selected execution group after the partition input has been
    /// prepared. The default is the ordinary layered group entry with no graph
    /// dependencies; graph architectures may override this when their
    /// partition input is already assembled.
    #[allow(clippy::too_many_arguments)]
    fn enter_partition_group(
        &mut self,
        group: usize,
        initial: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        match parallel {
            Some(parallel) => self.begin_execution_group_parallel(
                group,
                initial,
                &[],
                state,
                forward,
                parallel,
                context,
            ),
            None => self.begin_execution_group(group, initial, &[], state, forward, context),
        }
    }

    /// Completes the selected execution group before the architecture emits
    /// its final value or typed pipeline boundary.
    #[allow(clippy::too_many_arguments)]
    fn leave_partition_group(
        &mut self,
        group: usize,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        match parallel {
            Some(parallel) => self.complete_execution_group_parallel(
                group, hidden, state, forward, parallel, context,
            ),
            None => self.complete_execution_group(group, hidden, state, forward, context),
        }
    }

    /// Emits an architecture partition after its execution group has closed.
    #[allow(clippy::too_many_arguments)]
    fn finish_partition(
        &mut self,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &Self::ForwardContext,
        owns_output: bool,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<
        LayeredPartitionOutput<
            B::Tensor,
            <Self::Boundary as crate::ArchitectureBoundary>::Boundary<B::Tensor>,
        >,
        Self::Error,
    >;
}

/// Provider-aware unit execution for architectures with routed feed-forward work.
///
/// Partition drivers retain ownership of expert residency while the neutral
/// architecture retains attention, residual, routing, and unit semantics.
pub trait RoutedLayeredArchitecture<B, S>: LayeredArchitecture<B, S>
where
    B: eredu_nn::RoutedNeuralBackend,
    S: RuntimeState<B>,
{
    /// Returns the architecture-owned routing observation point for one unit.
    ///
    /// Architectures without observable routed work in the selected unit return
    /// `None`. Concrete backends must not reconstruct semantic paths or expert
    /// cardinality.
    fn routed_observation_point(
        &self,
        _group: usize,
        _index: usize,
    ) -> Result<Option<RoutedObservationPoint>, Self::Error> {
        Ok(None)
    }

    /// Executes one unit through a runtime-supplied routed-expert provider.
    #[allow(clippy::too_many_arguments)]
    fn forward_unit_with_provider<P>(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        pass: ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error>
    where
        P: RoutedExpertProvider<B>,
        P::Error: std::fmt::Display;
}

/// Tensor-parallel provider-aware unit execution.
pub trait ParallelRoutedLayeredArchitecture<B, S>:
    RoutedLayeredArchitecture<B, S> + ParallelLayeredArchitecture<B, S>
where
    B: eredu_nn::RoutedNeuralBackend,
    S: RuntimeState<B>,
{
    /// Executes one tensor-parallel unit through a runtime-supplied provider.
    #[allow(clippy::too_many_arguments)]
    fn forward_unit_parallel_with_provider<P>(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        pass: ExpertPass,
        provider: &mut P,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error>
    where
        P: RoutedExpertProvider<B>,
        P::Error: std::fmt::Display;
}

/// Fully resident runtime using the same lifecycle as bounded execution.
pub struct ResidentRuntime<A, B, S>
where
    B: NeuralBackend,
    S: RuntimeState<B>,
    A: LayeredArchitecture<B, S>,
{
    architecture: A,
    graph: ExecutionGraph,
    units: Vec<Vec<A::Unit>>,
    backend: std::marker::PhantomData<fn() -> (B, S)>,
}

impl<A, B, S> ResidentRuntime<A, B, S>
where
    B: NeuralBackend,
    S: RuntimeState<B>,
    A: LayeredArchitecture<B, S>,
{
    /// Builds every execution unit once and keeps it resident.
    pub fn new(
        architecture: A,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<Self, A::Error> {
        let graph = architecture.execution_graph()?;
        let mut units = Vec::with_capacity(graph.groups().len());
        for group in 0..graph.groups().len() {
            let count = architecture.group_unit_count(group)?;
            units.push(
                (0..count)
                    .map(|index| architecture.build_unit(group, index, context))
                    .collect::<Result<Vec<_>, _>>()?,
            );
        }
        Ok(Self {
            architecture,
            graph,
            units,
            backend: std::marker::PhantomData,
        })
    }

    /// Runs one complete prefill or decode pass without dynamic dispatch.
    pub fn forward<'a>(
        &mut self,
        input: A::Input<'a>,
        state: &mut S,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<B::Tensor, A::Error> {
        self.forward_with_context(input, state, context)
            .map(|(output, _)| output)
    }

    /// Runs one complete pass and returns its architecture-owned context.
    ///
    /// This is the resident counterpart of the bounded runtime's context
    /// result and lets callers retain target captures without storing
    /// request-local tensors on the model object.
    pub fn forward_with_context<'a>(
        &mut self,
        input: A::Input<'a>,
        state: &mut S,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<(B::Tensor, A::ForwardContext), A::Error> {
        self.forward_with_traversal_hook(input, state, context, &mut NoopLayeredTraversalHook)
    }

    /// Runs one resident pass through a statically dispatched traversal hook.
    pub fn forward_with_traversal_hook<'a, H>(
        &mut self,
        input: A::Input<'a>,
        state: &mut S,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        hook: &mut H,
    ) -> Result<(B::Tensor, A::ForwardContext), A::Error>
    where
        H: LayeredTraversalHook<B, A::ForwardContext, A::Error>,
    {
        let forward = self.architecture.begin_forward(input, state, context)?;
        let initial = forward.hidden;
        let mut forward_context = forward.context;
        let mut schedule = ExecutionGroupSchedule::new(&self.graph);
        let mut outputs: Vec<Option<B::Tensor>> = vec![None; self.graph.groups().len()];
        for &group in self.graph.execution_order() {
            let dependencies = schedule
                .dependencies(group)
                .expect("validated execution order contains a known group")
                .iter()
                .map(|&dependency| {
                    outputs[dependency]
                        .as_ref()
                        .expect("topological dependency has completed")
                        .clone()
                })
                .collect::<Vec<_>>();
            let dependency_refs = dependencies.iter().collect::<Vec<_>>();
            let mut hidden = self.architecture.begin_execution_group(
                group,
                &initial,
                &dependency_refs,
                state,
                &mut forward_context,
                context,
            )?;
            for dependency in schedule
                .started(group)
                .expect("topological execution starts only ready groups")
            {
                outputs[dependency] = None;
            }
            if self
                .architecture
                .should_execute_group(group, &forward_context)
            {
                let unit_count = self.units[group].len();
                for (index, unit) in self.units[group].iter_mut().enumerate() {
                    if hook.before_unit(
                        group,
                        index,
                        unit_count - index,
                        &hidden,
                        &mut forward_context,
                        context,
                    )? == LayeredUnitAction::SkipRemainingGroup
                    {
                        break;
                    }
                    hidden = self.architecture.forward_unit(
                        group,
                        index,
                        unit,
                        &hidden,
                        state,
                        &mut forward_context,
                        context,
                    )?;
                    hook.after_unit(group, index, &hidden, &mut forward_context, context)?;
                }
            }
            hidden = self.architecture.complete_execution_group(
                group,
                &hidden,
                state,
                &mut forward_context,
                context,
            )?;
            hook.after_group(group, &hidden, &mut forward_context, context)?;
            outputs[group] = Some(hidden);
            schedule
                .ordered(group)
                .expect("started group can be ordered exactly once");
        }
        let hidden = outputs[self.graph.output()]
            .take()
            .expect("validated graph output completed");
        let output = self
            .architecture
            .finish_forward(&hidden, state, &forward_context, context)?;
        Ok((output, forward_context))
    }

    /// Borrows the architecture and its pinned parameter topology.
    pub const fn architecture(&self) -> &A {
        &self.architecture
    }

    /// Mutably borrows the architecture.
    pub fn architecture_mut(&mut self) -> &mut A {
        &mut self.architecture
    }

    /// Borrows resident execution units for loading or inspection.
    pub fn units(&self) -> &[Vec<A::Unit>] {
        &self.units
    }

    /// Mutably borrows resident execution units for parameter binding.
    pub fn units_mut(&mut self) -> &mut [Vec<A::Unit>] {
        &mut self.units
    }

    /// Decomposes the runtime without cloning backend-native values.
    pub fn into_parts(self) -> (A, Vec<A::Unit>) {
        (
            self.architecture,
            self.units.into_iter().flatten().collect(),
        )
    }
}

/// Policy controlling acquisition and exact release of one execution unit.
pub trait LayerwisePolicy<B, U>
where
    B: NeuralBackend,
{
    /// Concrete lease owning one populated unit and all residency guards.
    type Lease: std::ops::DerefMut<Target = U>;
    /// Concrete acquisition or completion failure.
    type Error;

    /// Starts one forward after architecture input preparation.
    fn begin(
        &mut self,
        initial: &B::Tensor,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<(), Self::Error>;

    /// Aborts an incomplete forward and releases all policy-owned state.
    ///
    /// `active` contains the unit lease when execution stopped after
    /// acquisition but before exact completion. Implementations with no
    /// forward-scoped state may rely on the default, which simply drops it.
    fn abort(
        &mut self,
        active: Option<(usize, crate::ExecutionUnitAddress, Self::Lease)>,
        _context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) {
        drop(active);
    }

    /// Acquires one populated unit for exclusive execution.
    ///
    /// The flat ordinal addresses storage while `address` preserves the
    /// architecture execution group and group-local unit index for scheduling.
    fn acquire<E, F>(
        &mut self,
        ordinal: usize,
        address: crate::ExecutionUnitAddress,
        build: F,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<Self::Lease, LayerwiseAcquireError<E, Self::Error>>
    where
        F: FnOnce(&<B::Tensor as eredu_nn::Tensor>::Context) -> Result<U, E>;

    /// Retains the unit and dependent native values through exact completion.
    fn complete<'a, StateValues, ContextValues>(
        &mut self,
        ordinal: usize,
        address: crate::ExecutionUnitAddress,
        lease: Self::Lease,
        output: &'a B::Tensor,
        state_values: StateValues,
        context_values: ContextValues,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<(), Self::Error>
    where
        B::Tensor: 'a,
        StateValues: Iterator<Item = &'a B::Tensor>,
        ContextValues: Iterator<Item = &'a B::Tensor>;

    /// Completes the final output and releases any remaining unit guards.
    fn finish(
        &mut self,
        output: &B::Tensor,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<(), Self::Error>;
}

/// Failure-safe ownership of one policy forward and its current unit lease.
struct LayerwisePolicyForward<'a, B, U, P>
where
    B: NeuralBackend,
    P: LayerwisePolicy<B, U>,
{
    policy: &'a mut P,
    context: &'a <B::Tensor as eredu_nn::Tensor>::Context,
    active: Option<(usize, crate::ExecutionUnitAddress, P::Lease)>,
    finished: bool,
    unit: std::marker::PhantomData<fn() -> U>,
}

impl<'a, B, U, P> LayerwisePolicyForward<'a, B, U, P>
where
    B: NeuralBackend,
    P: LayerwisePolicy<B, U>,
{
    fn begin(
        policy: &'a mut P,
        initial: &B::Tensor,
        context: &'a <B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<Self, P::Error> {
        if let Err(error) = policy.begin(initial, context) {
            policy.abort(None, context);
            return Err(error);
        }
        Ok(Self {
            policy,
            context,
            active: None,
            finished: false,
            unit: std::marker::PhantomData,
        })
    }

    fn acquire<E, F>(
        &mut self,
        ordinal: usize,
        address: crate::ExecutionUnitAddress,
        build: F,
    ) -> Result<&mut P::Lease, LayerwiseAcquireError<E, P::Error>>
    where
        F: FnOnce(&<B::Tensor as eredu_nn::Tensor>::Context) -> Result<U, E>,
    {
        debug_assert!(self.active.is_none());
        let lease = self.policy.acquire(ordinal, address, build, self.context)?;
        self.active = Some((ordinal, address, lease));
        Ok(&mut self
            .active
            .as_mut()
            .expect("acquired policy lease is active")
            .2)
    }

    fn complete<'value, StateValues, ContextValues>(
        &mut self,
        output: &'value B::Tensor,
        state_values: StateValues,
        context_values: ContextValues,
    ) -> Result<(), P::Error>
    where
        B::Tensor: 'value,
        StateValues: Iterator<Item = &'value B::Tensor>,
        ContextValues: Iterator<Item = &'value B::Tensor>,
    {
        let (ordinal, address, lease) = self
            .active
            .take()
            .expect("policy completion follows one acquisition");
        self.policy.complete(
            ordinal,
            address,
            lease,
            output,
            state_values,
            context_values,
            self.context,
        )
    }

    fn finish(&mut self, output: &B::Tensor) -> Result<(), P::Error> {
        self.policy.finish(output, self.context)?;
        self.finished = true;
        Ok(())
    }
}

impl<B, U, P> Drop for LayerwisePolicyForward<'_, B, U, P>
where
    B: NeuralBackend,
    P: LayerwisePolicy<B, U>,
{
    fn drop(&mut self) {
        if !self.finished {
            self.policy.abort(self.active.take(), self.context);
        }
    }
}

/// Failure while a layerwise policy acquires or populates one architecture unit.
#[derive(Debug)]
pub enum LayerwiseAcquireError<A, P> {
    /// The neutral architecture could not construct its unloaded unit.
    Architecture(A),
    /// The execution policy could not acquire residency or populate the unit.
    Policy(P),
}

/// Failure from architecture execution or layerwise residency policy.
#[derive(Debug, thiserror::Error)]
pub enum LayerwiseRuntimeError<A, P>
where
    A: std::fmt::Display,
    P: std::fmt::Display,
{
    /// Architecture construction or forward failure.
    #[error("layered architecture failed: {0}")]
    Architecture(A),
    /// Invalid access to architecture-declared mutable state.
    #[error(transparent)]
    State(#[from] crate::StateError),
    /// Architecture execution groups did not map to one stable residency-unit order.
    #[error(transparent)]
    Layout(#[from] crate::ExecutionUnitLayoutError),
    /// Unit acquisition or exact-completion failure.
    #[error("layerwise execution policy failed: {0}")]
    Policy(P),
    /// Backend-native graph submission or dependency ordering failed.
    #[error("layerwise backend submission failed: {0}")]
    Submission(String),
}

/// Bounded-unit runtime invoking the same architecture lifecycle as resident execution.
pub struct LayerwiseRuntime<A, B, S, P>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as eredu_nn::Tensor>::Context>,
    S: RuntimeState<B>,
    A: LayeredArchitecture<B, S>,
    P: LayerwisePolicy<B, A::Unit>,
{
    architecture: A,
    policy: P,
    executors: Option<Vec<B::OwnedExecutor>>,
    backend: std::marker::PhantomData<fn() -> (B, S)>,
}

impl<A, B, S, P> LayerwiseRuntime<A, B, S, P>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as eredu_nn::Tensor>::Context>,
    S: RuntimeState<B>,
    A: LayeredArchitecture<B, S>,
    P: LayerwisePolicy<B, A::Unit>,
    A::Error: std::fmt::Display,
    P::Error: std::fmt::Display,
{
    /// Creates a layerwise runtime from concrete architecture, state, and policy.
    pub const fn new(architecture: A, policy: P) -> Self {
        Self {
            architecture,
            policy,
            executors: None,
            backend: std::marker::PhantomData,
        }
    }

    /// Creates a layerwise runtime while evaluating the policy before moving
    /// the architecture. This is useful when policy realization needs to
    /// borrow the architecture's canonical unit constructor first.
    pub const fn new_policy_first(policy: P, architecture: A) -> Self {
        Self::new(architecture, policy)
    }

    /// Borrows the concrete architecture instance.
    pub const fn architecture(&self) -> &A {
        &self.architecture
    }

    /// Mutably borrows the concrete architecture instance.
    pub fn architecture_mut(&mut self) -> &mut A {
        &mut self.architecture
    }

    /// Borrows the concrete execution policy for cold-path diagnostics.
    pub const fn policy(&self) -> &P {
        &self.policy
    }

    /// Mutably borrows the concrete execution policy.
    pub fn policy_mut(&mut self) -> &mut P {
        &mut self.policy
    }

    /// Runs one complete prefill or decode pass with exact unit release points.
    pub fn forward<'a>(
        &mut self,
        input: A::Input<'a>,
        state: &mut S,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<B::Tensor, LayerwiseRuntimeError<A::Error, P::Error>> {
        self.forward_with_context_hook(input, state, context, |_, _, _| Ok(()))
            .map(|(output, _)| output)
    }

    /// Runs one pass and exposes mutable architecture context after each unit.
    pub fn forward_with_context_hook<'a, H>(
        &mut self,
        input: A::Input<'a>,
        state: &mut S,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        hook: H,
    ) -> Result<(B::Tensor, A::ForwardContext), LayerwiseRuntimeError<A::Error, P::Error>>
    where
        H: FnMut(usize, usize, &mut A::ForwardContext) -> Result<(), A::Error>,
    {
        self.forward_with_unit_executor_and_context_hook(
            input,
            state,
            context,
            |architecture, group, index, unit, hidden, state, forward, context| {
                architecture.forward_unit(group, index, unit, hidden, state, forward, context)
            },
            hook,
        )
    }

    /// Runs one pass with a statically dispatched architecture-unit executor.
    ///
    /// Composition can use this cold API to inject routed expert execution or
    /// observation while the runtime retains graph traversal, residency, and
    /// exact completion ownership.
    pub fn forward_with_unit_executor<'a, E>(
        &mut self,
        input: A::Input<'a>,
        state: &mut S,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        execute: E,
    ) -> Result<B::Tensor, LayerwiseRuntimeError<A::Error, P::Error>>
    where
        E: FnMut(
            &mut A,
            usize,
            usize,
            &mut A::Unit,
            &B::Tensor,
            &mut S,
            &mut A::ForwardContext,
            &<B::Tensor as eredu_nn::Tensor>::Context,
        ) -> Result<B::Tensor, A::Error>,
    {
        self.forward_with_unit_executor_and_context_hook(
            input,
            state,
            context,
            execute,
            |_, _, _| Ok(()),
        )
        .map(|(output, _)| output)
    }

    /// Runs the production sequential traversal with stable unit-boundary observation.
    pub fn forward_with_observer<'a, Observer>(
        &mut self,
        input: A::Input<'a>,
        state: &mut S,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        observer: &mut Observer,
    ) -> Result<B::Tensor, LayerwiseRuntimeError<A::Error, P::Error>>
    where
        Observer: ActivationObserver<B::Tensor, A::Error> + ?Sized,
    {
        self.forward_with_unit_executor_and_observer(
            input,
            state,
            context,
            |architecture, group, index, unit, hidden, state, forward, context| {
                architecture.forward_unit(group, index, unit, hidden, state, forward, context)
            },
            observer,
        )
    }

    /// Runs a custom production unit executor with stable boundary observation.
    pub fn forward_with_unit_executor_and_observer<'a, E, Observer>(
        &mut self,
        input: A::Input<'a>,
        state: &mut S,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        mut execute: E,
        observer: &mut Observer,
    ) -> Result<B::Tensor, LayerwiseRuntimeError<A::Error, P::Error>>
    where
        E: FnMut(
            &mut A,
            usize,
            usize,
            &mut A::Unit,
            &B::Tensor,
            &mut S,
            &mut A::ForwardContext,
            &<B::Tensor as eredu_nn::Tensor>::Context,
        ) -> Result<B::Tensor, A::Error>,
        Observer: ActivationObserver<B::Tensor, A::Error> + ?Sized,
    {
        self.forward_with_unit_executor(
            input,
            state,
            context,
            |architecture, group, index, unit, hidden, state, forward, context| {
                let path = architecture.unit_path(group, index)?;
                let input = observe_and_intervene(observer, &format!("{path}.input"), hidden)?;
                let output = execute(
                    architecture,
                    group,
                    index,
                    unit,
                    &input,
                    state,
                    forward,
                    context,
                )?;
                observe_and_intervene(observer, &format!("{path}.output"), &output)
            },
        )
    }

    /// Runs canonical provider-backed unit execution with unit-boundary and
    /// routed-expert observation.
    ///
    /// Observation wraps [`RoutedLayeredArchitecture::forward_unit_with_provider`]
    /// instead of replacing it. Architecture-owned validation, state lookup,
    /// shape handling, routing, and provider dispatch therefore remain shared
    /// with ordinary execution.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_with_provider_and_observer<'a, Provider, Observer>(
        &mut self,
        input: A::Input<'a>,
        state: &mut S,
        pass: ExpertPass,
        provider: &mut Provider,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        observer: &mut Observer,
    ) -> Result<B::Tensor, LayerwiseRuntimeError<A::Error, P::Error>>
    where
        B: eredu_nn::RoutedNeuralBackend,
        A: RoutedLayeredArchitecture<B, S>,
        A::Error: std::fmt::Display,
        Provider: RoutedExpertProvider<B>,
        Provider::Error: std::fmt::Display,
        Observer: ActivationObserver<B::Tensor, A::Error> + ?Sized,
    {
        self.forward_with_unit_executor(
            input,
            state,
            context,
            |architecture, group, index, unit, hidden, state, forward, context| {
                let path = architecture.unit_path(group, index)?;
                let input = observe_and_intervene(observer, &format!("{path}.input"), hidden)?;
                let output = match architecture.routed_observation_point(group, index)? {
                    Some(point) => {
                        let mut observed = ObservedExpertProvider::new(provider, observer, point);
                        architecture.forward_unit_with_provider(
                            group,
                            index,
                            unit,
                            &input,
                            state,
                            forward,
                            pass,
                            &mut observed,
                            context,
                        )?
                    }
                    None => architecture.forward_unit_with_provider(
                        group, index, unit, &input, state, forward, pass, provider, context,
                    )?,
                };
                observe_and_intervene(observer, &format!("{path}.output"), &output)
            },
        )
    }

    /// Runs one pass with both a custom unit executor and post-unit context hook.
    pub fn forward_with_unit_executor_and_context_hook<'a, E, H>(
        &mut self,
        input: A::Input<'a>,
        state: &mut S,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        execute: E,
        mut hook: H,
    ) -> Result<(B::Tensor, A::ForwardContext), LayerwiseRuntimeError<A::Error, P::Error>>
    where
        E: FnMut(
            &mut A,
            usize,
            usize,
            &mut A::Unit,
            &B::Tensor,
            &mut S,
            &mut A::ForwardContext,
            &<B::Tensor as eredu_nn::Tensor>::Context,
        ) -> Result<B::Tensor, A::Error>,
        H: FnMut(usize, usize, &mut A::ForwardContext) -> Result<(), A::Error>,
    {
        self.forward_with_unit_executor_and_activation_hook(
            input,
            state,
            context,
            execute,
            |group, index, _hidden, forward| hook(group, index, forward),
        )
    }

    /// Runs one pass with a custom unit executor and exposes each post-unit
    /// activation together with the mutable architecture context.
    ///
    /// The activation is the ordinary output of the execution unit. Target
    /// state taps and inspection therefore observe the production forward
    /// without requiring a second family-specific model path.
    pub fn forward_with_unit_executor_and_activation_hook<'a, E, H>(
        &mut self,
        input: A::Input<'a>,
        state: &mut S,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        execute: E,
        hook: H,
    ) -> Result<(B::Tensor, A::ForwardContext), LayerwiseRuntimeError<A::Error, P::Error>>
    where
        E: FnMut(
            &mut A,
            usize,
            usize,
            &mut A::Unit,
            &B::Tensor,
            &mut S,
            &mut A::ForwardContext,
            &<B::Tensor as eredu_nn::Tensor>::Context,
        ) -> Result<B::Tensor, A::Error>,
        H: FnMut(usize, usize, &B::Tensor, &mut A::ForwardContext) -> Result<(), A::Error>,
    {
        self.forward_with_unit_executor_and_traversal_hook(
            input,
            state,
            context,
            execute,
            &mut AfterUnitTraversalHook { after_unit: hook },
        )
    }

    /// Runs one bounded pass through a statically dispatched traversal hook.
    pub fn forward_with_traversal_hook<'a, H>(
        &mut self,
        input: A::Input<'a>,
        state: &mut S,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        hook: &mut H,
    ) -> Result<(B::Tensor, A::ForwardContext), LayerwiseRuntimeError<A::Error, P::Error>>
    where
        H: LayeredTraversalHook<B, A::ForwardContext, A::Error>,
    {
        self.forward_with_unit_executor_and_traversal_hook(
            input,
            state,
            context,
            |architecture, group, index, unit, hidden, state, forward, context| {
                architecture.forward_unit(group, index, unit, hidden, state, forward, context)
            },
            hook,
        )
    }

    /// Runs one bounded pass with custom unit execution and a shared traversal hook.
    pub fn forward_with_unit_executor_and_traversal_hook<'a, E, H>(
        &mut self,
        input: A::Input<'a>,
        state: &mut S,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        mut execute: E,
        hook: &mut H,
    ) -> Result<(B::Tensor, A::ForwardContext), LayerwiseRuntimeError<A::Error, P::Error>>
    where
        E: FnMut(
            &mut A,
            usize,
            usize,
            &mut A::Unit,
            &B::Tensor,
            &mut S,
            &mut A::ForwardContext,
            &<B::Tensor as eredu_nn::Tensor>::Context,
        ) -> Result<B::Tensor, A::Error>,
        H: LayeredTraversalHook<B, A::ForwardContext, A::Error>,
    {
        let graph = self
            .architecture
            .execution_graph()
            .map_err(LayerwiseRuntimeError::Architecture)?;
        let counts = (0..graph.groups().len())
            .map(|group| {
                self.architecture
                    .group_unit_count(group)
                    .map_err(LayerwiseRuntimeError::Architecture)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let layout = ExecutionUnitLayout::new(&graph, counts)?;
        if self.executors.as_ref().map(Vec::len) != Some(graph.groups().len()) {
            self.executors = Some(
                B::fork_executors(context, graph.groups().len())
                    .map_err(|error| LayerwiseRuntimeError::Submission(error.to_string()))?,
            );
        }
        let executors = self
            .executors
            .as_ref()
            .expect("layered runtime initialized its executor cache");
        let forward = self
            .architecture
            .begin_forward(input, state, context)
            .map_err(LayerwiseRuntimeError::Architecture)?;
        let initial_completion = (graph.groups().len() > 1)
            .then(|| B::submit(context, [&forward.hidden]))
            .transpose()
            .map_err(|error| LayerwiseRuntimeError::Submission(error.to_string()))?;
        let mut policy = LayerwisePolicyForward::begin(&mut self.policy, &forward.hidden, context)
            .map_err(LayerwiseRuntimeError::Policy)?;
        let initial = forward.hidden;
        let mut forward_context = forward.context;
        let mut schedule = ExecutionGroupSchedule::new(&graph);
        let mut outputs: Vec<Option<B::Tensor>> = vec![None; graph.groups().len()];
        let mut completions: Vec<Option<B::Completion>> =
            (0..graph.groups().len()).map(|_| None).collect();
        for &group in graph.execution_order() {
            let executor = std::borrow::Borrow::borrow(&executors[group]);
            let group_dependencies = schedule
                .dependencies(group)
                .expect("validated execution order contains a known group");
            if group_dependencies.is_empty() {
                if let Some(completion) = &initial_completion {
                    B::order_after(completion, executor)
                        .map_err(|error| LayerwiseRuntimeError::Submission(error.to_string()))?;
                }
            }
            for &dependency in group_dependencies {
                B::order_after(
                    completions[dependency]
                        .as_ref()
                        .expect("topological dependency has a completion"),
                    executor,
                )
                .map_err(|error| LayerwiseRuntimeError::Submission(error.to_string()))?;
            }
            let dependencies = schedule
                .dependencies(group)
                .expect("validated execution order contains a known group")
                .iter()
                .map(|&dependency| {
                    outputs[dependency]
                        .as_ref()
                        .expect("topological dependency has completed")
                        .clone()
                })
                .collect::<Vec<_>>();
            let dependency_refs = dependencies.iter().collect::<Vec<_>>();
            let mut hidden = self
                .architecture
                .begin_execution_group(
                    group,
                    &initial,
                    &dependency_refs,
                    state,
                    &mut forward_context,
                    executor,
                )
                .map_err(LayerwiseRuntimeError::Architecture)?;
            for dependency in schedule
                .started(group)
                .expect("topological execution starts only ready groups")
            {
                outputs[dependency] = None;
            }
            if self
                .architecture
                .should_execute_group(group, &forward_context)
            {
                let unit_count = layout
                    .group_range(group)
                    .expect("layout covers every graph group")
                    .len();
                for index in 0..unit_count {
                    if hook
                        .before_unit(
                            group,
                            index,
                            unit_count - index,
                            &hidden,
                            &mut forward_context,
                            executor,
                        )
                        .map_err(LayerwiseRuntimeError::Architecture)?
                        == LayeredUnitAction::SkipRemainingGroup
                    {
                        break;
                    }
                    let ordinal = layout
                        .ordinal(group, index)
                        .expect("group-local unit belongs to the layout");
                    let address = layout
                        .address(ordinal)
                        .expect("group-local unit has a stable policy address");
                    let lease = policy
                        .acquire(ordinal, address, |executor| {
                            self.architecture.build_unit(group, index, executor)
                        })
                        .map_err(|error| match error {
                            LayerwiseAcquireError::Architecture(error) => {
                                LayerwiseRuntimeError::Architecture(error)
                            }
                            LayerwiseAcquireError::Policy(error) => {
                                LayerwiseRuntimeError::Policy(error)
                            }
                        })?;
                    hidden = execute(
                        &mut self.architecture,
                        group,
                        index,
                        lease,
                        &hidden,
                        state,
                        &mut forward_context,
                        executor,
                    )
                    .map_err(LayerwiseRuntimeError::Architecture)?;
                    hook.after_unit(group, index, &hidden, &mut forward_context, executor)
                        .map_err(LayerwiseRuntimeError::Architecture)?;
                    let mut state_values = Vec::new();
                    for state_ordinal in self
                        .architecture
                        .retained_state_ordinals(group, index, ordinal)
                    {
                        state_values.extend(
                            state
                                .retained_values(state_ordinal, address.with_index(state_ordinal))
                                .map_err(LayerwiseRuntimeError::State)?,
                        );
                    }
                    let context_values =
                        self.architecture
                            .retained_context_values(&forward_context, group, index);
                    policy
                        .complete(&hidden, state_values.into_iter(), context_values)
                        .map_err(LayerwiseRuntimeError::Policy)?;
                }
            }
            hidden = self
                .architecture
                .complete_execution_group(group, &hidden, state, &mut forward_context, executor)
                .map_err(LayerwiseRuntimeError::Architecture)?;
            hook.after_group(group, &hidden, &mut forward_context, executor)
                .map_err(LayerwiseRuntimeError::Architecture)?;
            outputs[group] = Some(hidden);
            if graph.groups().len() > 1 {
                completions[group] = Some(
                    B::submit(
                        executor,
                        [outputs[group]
                            .as_ref()
                            .expect("group output was stored before submission")],
                    )
                    .map_err(|error| LayerwiseRuntimeError::Submission(error.to_string()))?,
                );
            }
            schedule
                .ordered(group)
                .expect("started group can be ordered exactly once");
        }
        let hidden = outputs[graph.output()]
            .take()
            .expect("validated graph output completed");
        if let Some(completion) = &completions[graph.output()] {
            B::order_after(completion, context)
                .map_err(|error| LayerwiseRuntimeError::Submission(error.to_string()))?;
        }
        let output = self
            .architecture
            .finish_forward(&hidden, state, &forward_context, context)
            .map_err(LayerwiseRuntimeError::Architecture)?;
        policy
            .finish(&output)
            .map_err(LayerwiseRuntimeError::Policy)?;
        Ok((output, forward_context))
    }

    /// Runs one complete rank-local pass through the neutral parallel lifecycle.
    pub fn forward_parallel<'a>(
        &mut self,
        input: A::Input<'a>,
        state: &mut S,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<B::Tensor, LayerwiseRuntimeError<A::Error, P::Error>>
    where
        A: ParallelLayeredArchitecture<B, S>,
    {
        self.forward_parallel_with_context_hook(input, state, parallel, context, |_, _, _| Ok(()))
            .map(|(output, _)| output)
    }

    /// Runs one rank-local pass and exposes mutable context after each unit.
    pub fn forward_parallel_with_context_hook<'a, H>(
        &mut self,
        input: A::Input<'a>,
        state: &mut S,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        hook: H,
    ) -> Result<(B::Tensor, A::ForwardContext), LayerwiseRuntimeError<A::Error, P::Error>>
    where
        A: ParallelLayeredArchitecture<B, S>,
        H: FnMut(usize, usize, &mut A::ForwardContext) -> Result<(), A::Error>,
    {
        self.forward_parallel_with_unit_executor_and_traversal_hook(
            input,
            state,
            parallel,
            context,
            |architecture, group, index, unit, hidden, state, forward, parallel, context| {
                architecture.forward_unit_parallel(
                    group, index, unit, hidden, state, forward, parallel, context,
                )
            },
            &mut AfterUnitContextTraversalHook { after_unit: hook },
        )
    }

    /// Runs one parallel pass with a custom statically dispatched unit executor.
    pub fn forward_parallel_with_unit_executor<'a, E>(
        &mut self,
        input: A::Input<'a>,
        state: &mut S,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        execute: E,
    ) -> Result<B::Tensor, LayerwiseRuntimeError<A::Error, P::Error>>
    where
        A: ParallelLayeredArchitecture<B, S>,
        E: FnMut(
            &mut A,
            usize,
            usize,
            &mut A::Unit,
            &B::Tensor,
            &mut S,
            &mut A::ForwardContext,
            &B::ParallelContext,
            &<B::Tensor as eredu_nn::Tensor>::Context,
        ) -> Result<B::Tensor, A::Error>,
    {
        self.forward_parallel_with_unit_executor_and_context_hook(
            input,
            state,
            parallel,
            context,
            execute,
            |_, _, _| Ok(()),
        )
        .map(|(output, _)| output)
    }

    /// Runs the production parallel traversal with stable unit-boundary observation.
    pub fn forward_parallel_with_observer<'a, Observer>(
        &mut self,
        input: A::Input<'a>,
        state: &mut S,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        observer: &mut Observer,
    ) -> Result<B::Tensor, LayerwiseRuntimeError<A::Error, P::Error>>
    where
        A: ParallelLayeredArchitecture<B, S>,
        Observer: ActivationObserver<B::Tensor, A::Error> + ?Sized,
    {
        self.forward_parallel_with_unit_executor_and_observer(
            input,
            state,
            parallel,
            context,
            |architecture, group, index, unit, hidden, state, forward, parallel, context| {
                architecture.forward_unit_parallel(
                    group, index, unit, hidden, state, forward, parallel, context,
                )
            },
            observer,
        )
    }

    /// Runs a custom parallel unit executor with stable boundary observation.
    pub fn forward_parallel_with_unit_executor_and_observer<'a, E, Observer>(
        &mut self,
        input: A::Input<'a>,
        state: &mut S,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        mut execute: E,
        observer: &mut Observer,
    ) -> Result<B::Tensor, LayerwiseRuntimeError<A::Error, P::Error>>
    where
        A: ParallelLayeredArchitecture<B, S>,
        E: FnMut(
            &mut A,
            usize,
            usize,
            &mut A::Unit,
            &B::Tensor,
            &mut S,
            &mut A::ForwardContext,
            &B::ParallelContext,
            &<B::Tensor as eredu_nn::Tensor>::Context,
        ) -> Result<B::Tensor, A::Error>,
        Observer: ActivationObserver<B::Tensor, A::Error> + ?Sized,
    {
        self.forward_parallel_with_unit_executor(
            input,
            state,
            parallel,
            context,
            |architecture, group, index, unit, hidden, state, forward, parallel, context| {
                let path = architecture.unit_path(group, index)?;
                let input = observe_and_intervene(observer, &format!("{path}.input"), hidden)?;
                let output = execute(
                    architecture,
                    group,
                    index,
                    unit,
                    &input,
                    state,
                    forward,
                    parallel,
                    context,
                )?;
                observe_and_intervene(observer, &format!("{path}.output"), &output)
            },
        )
    }

    /// Runs provider-backed parallel execution with boundary and routing observation.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_parallel_with_provider_and_observer<'a, Provider, Observer>(
        &mut self,
        input: A::Input<'a>,
        state: &mut S,
        pass: ExpertPass,
        provider: &mut Provider,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        observer: &mut Observer,
    ) -> Result<B::Tensor, LayerwiseRuntimeError<A::Error, P::Error>>
    where
        B: eredu_nn::RoutedNeuralBackend,
        A: ParallelRoutedLayeredArchitecture<B, S>,
        Provider: RoutedExpertProvider<B>,
        Provider::Error: std::fmt::Display,
        Observer: ActivationObserver<B::Tensor, A::Error> + ?Sized,
    {
        self.forward_parallel_with_unit_executor(
            input,
            state,
            parallel,
            context,
            |architecture, group, index, unit, hidden, state, forward, parallel, context| {
                let path = architecture.unit_path(group, index)?;
                let input = observe_and_intervene(observer, &format!("{path}.input"), hidden)?;
                let output = match architecture.routed_observation_point(group, index)? {
                    Some(point) => {
                        let mut observed = ObservedExpertProvider::new(provider, observer, point);
                        architecture.forward_unit_parallel_with_provider(
                            group,
                            index,
                            unit,
                            &input,
                            state,
                            forward,
                            pass,
                            &mut observed,
                            parallel,
                            context,
                        )
                    }
                    None => architecture.forward_unit_parallel_with_provider(
                        group, index, unit, &input, state, forward, pass, provider, parallel,
                        context,
                    ),
                }?;
                observe_and_intervene(observer, &format!("{path}.output"), &output)
            },
        )
    }

    /// Runs one parallel pass with custom unit execution and a post-unit hook.
    pub fn forward_parallel_with_unit_executor_and_context_hook<'a, E, H>(
        &mut self,
        input: A::Input<'a>,
        state: &mut S,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        execute: E,
        hook: H,
    ) -> Result<(B::Tensor, A::ForwardContext), LayerwiseRuntimeError<A::Error, P::Error>>
    where
        A: ParallelLayeredArchitecture<B, S>,
        E: FnMut(
            &mut A,
            usize,
            usize,
            &mut A::Unit,
            &B::Tensor,
            &mut S,
            &mut A::ForwardContext,
            &B::ParallelContext,
            &<B::Tensor as eredu_nn::Tensor>::Context,
        ) -> Result<B::Tensor, A::Error>,
        H: FnMut(usize, usize, &mut A::ForwardContext) -> Result<(), A::Error>,
    {
        self.forward_parallel_with_unit_executor_and_traversal_hook(
            input,
            state,
            parallel,
            context,
            execute,
            &mut AfterUnitContextTraversalHook { after_unit: hook },
        )
    }

    /// Runs one parallel pass through a statically dispatched traversal hook.
    pub fn forward_parallel_with_traversal_hook<'a, H>(
        &mut self,
        input: A::Input<'a>,
        state: &mut S,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        hook: &mut H,
    ) -> Result<(B::Tensor, A::ForwardContext), LayerwiseRuntimeError<A::Error, P::Error>>
    where
        A: ParallelLayeredArchitecture<B, S>,
        H: LayeredTraversalHook<B, A::ForwardContext, A::Error>,
    {
        self.forward_parallel_with_unit_executor_and_traversal_hook(
            input,
            state,
            parallel,
            context,
            |architecture, group, index, unit, hidden, state, forward, parallel, context| {
                architecture.forward_unit_parallel(
                    group, index, unit, hidden, state, forward, parallel, context,
                )
            },
            hook,
        )
    }

    /// Runs one parallel pass with custom unit execution and a shared traversal hook.
    pub fn forward_parallel_with_unit_executor_and_traversal_hook<'a, E, H>(
        &mut self,
        input: A::Input<'a>,
        state: &mut S,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        mut execute: E,
        hook: &mut H,
    ) -> Result<(B::Tensor, A::ForwardContext), LayerwiseRuntimeError<A::Error, P::Error>>
    where
        A: ParallelLayeredArchitecture<B, S>,
        E: FnMut(
            &mut A,
            usize,
            usize,
            &mut A::Unit,
            &B::Tensor,
            &mut S,
            &mut A::ForwardContext,
            &B::ParallelContext,
            &<B::Tensor as eredu_nn::Tensor>::Context,
        ) -> Result<B::Tensor, A::Error>,
        H: LayeredTraversalHook<B, A::ForwardContext, A::Error>,
    {
        let graph = self
            .architecture
            .execution_graph()
            .map_err(LayerwiseRuntimeError::Architecture)?;
        let counts = (0..graph.groups().len())
            .map(|group| {
                self.architecture
                    .group_unit_count(group)
                    .map_err(LayerwiseRuntimeError::Architecture)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let layout = ExecutionUnitLayout::new(&graph, counts)?;
        if self.executors.as_ref().map(Vec::len) != Some(graph.groups().len()) {
            self.executors = Some(
                B::fork_executors(context, graph.groups().len())
                    .map_err(|error| LayerwiseRuntimeError::Submission(error.to_string()))?,
            );
        }
        let executors = self
            .executors
            .as_ref()
            .expect("layered runtime initialized its executor cache");
        let forward = self
            .architecture
            .begin_forward_parallel(input, state, parallel, context)
            .map_err(LayerwiseRuntimeError::Architecture)?;
        let initial_completion = (graph.groups().len() > 1)
            .then(|| B::submit(context, [&forward.hidden]))
            .transpose()
            .map_err(|error| LayerwiseRuntimeError::Submission(error.to_string()))?;
        let mut policy = LayerwisePolicyForward::begin(&mut self.policy, &forward.hidden, context)
            .map_err(LayerwiseRuntimeError::Policy)?;
        let initial = forward.hidden;
        let mut forward_context = forward.context;
        let mut schedule = ExecutionGroupSchedule::new(&graph);
        let mut outputs: Vec<Option<B::Tensor>> = vec![None; graph.groups().len()];
        let mut completions: Vec<Option<B::Completion>> =
            (0..graph.groups().len()).map(|_| None).collect();
        for &group in graph.execution_order() {
            let executor = std::borrow::Borrow::borrow(&executors[group]);
            let group_dependencies = schedule
                .dependencies(group)
                .expect("validated execution order contains a known group");
            if group_dependencies.is_empty() {
                if let Some(completion) = &initial_completion {
                    B::order_after(completion, executor)
                        .map_err(|error| LayerwiseRuntimeError::Submission(error.to_string()))?;
                }
            }
            for &dependency in group_dependencies {
                B::order_after(
                    completions[dependency]
                        .as_ref()
                        .expect("topological dependency has a completion"),
                    executor,
                )
                .map_err(|error| LayerwiseRuntimeError::Submission(error.to_string()))?;
            }
            let dependencies = schedule
                .dependencies(group)
                .expect("validated execution order contains a known group")
                .iter()
                .map(|&dependency| {
                    outputs[dependency]
                        .as_ref()
                        .expect("topological dependency has completed")
                        .clone()
                })
                .collect::<Vec<_>>();
            let dependency_refs = dependencies.iter().collect::<Vec<_>>();
            let mut hidden = self
                .architecture
                .begin_execution_group_parallel(
                    group,
                    &initial,
                    &dependency_refs,
                    state,
                    &mut forward_context,
                    parallel,
                    executor,
                )
                .map_err(LayerwiseRuntimeError::Architecture)?;
            for dependency in schedule
                .started(group)
                .expect("topological execution starts only ready groups")
            {
                outputs[dependency] = None;
            }
            if self
                .architecture
                .should_execute_group(group, &forward_context)
            {
                let unit_count = layout
                    .group_range(group)
                    .expect("layout covers every graph group")
                    .len();
                for index in 0..unit_count {
                    if hook
                        .before_unit(
                            group,
                            index,
                            unit_count - index,
                            &hidden,
                            &mut forward_context,
                            executor,
                        )
                        .map_err(LayerwiseRuntimeError::Architecture)?
                        == LayeredUnitAction::SkipRemainingGroup
                    {
                        break;
                    }
                    let ordinal = layout
                        .ordinal(group, index)
                        .expect("group-local unit belongs to the layout");
                    let address = layout
                        .address(ordinal)
                        .expect("group-local unit has a stable policy address");
                    let lease = policy
                        .acquire(ordinal, address, |executor| {
                            self.architecture.build_unit(group, index, executor)
                        })
                        .map_err(|error| match error {
                            LayerwiseAcquireError::Architecture(error) => {
                                LayerwiseRuntimeError::Architecture(error)
                            }
                            LayerwiseAcquireError::Policy(error) => {
                                LayerwiseRuntimeError::Policy(error)
                            }
                        })?;
                    hidden = execute(
                        &mut self.architecture,
                        group,
                        index,
                        lease,
                        &hidden,
                        state,
                        &mut forward_context,
                        parallel,
                        executor,
                    )
                    .map_err(LayerwiseRuntimeError::Architecture)?;
                    hook.after_unit(group, index, &hidden, &mut forward_context, executor)
                        .map_err(LayerwiseRuntimeError::Architecture)?;
                    let mut state_values = Vec::new();
                    for state_ordinal in self
                        .architecture
                        .retained_state_ordinals(group, index, ordinal)
                    {
                        state_values.extend(
                            state
                                .retained_values(state_ordinal, address.with_index(state_ordinal))
                                .map_err(LayerwiseRuntimeError::State)?,
                        );
                    }
                    let context_values =
                        self.architecture
                            .retained_context_values(&forward_context, group, index);
                    policy
                        .complete(&hidden, state_values.into_iter(), context_values)
                        .map_err(LayerwiseRuntimeError::Policy)?;
                }
            }
            hidden = self
                .architecture
                .complete_execution_group_parallel(
                    group,
                    &hidden,
                    state,
                    &mut forward_context,
                    parallel,
                    executor,
                )
                .map_err(LayerwiseRuntimeError::Architecture)?;
            hook.after_group(group, &hidden, &mut forward_context, executor)
                .map_err(LayerwiseRuntimeError::Architecture)?;
            outputs[group] = Some(hidden);
            if graph.groups().len() > 1 {
                completions[group] = Some(
                    B::submit(
                        executor,
                        [outputs[group]
                            .as_ref()
                            .expect("group output was stored before submission")],
                    )
                    .map_err(|error| LayerwiseRuntimeError::Submission(error.to_string()))?,
                );
            }
            schedule
                .ordered(group)
                .expect("started group can be ordered exactly once");
        }
        let hidden = outputs[graph.output()]
            .take()
            .expect("validated graph output completed");
        if let Some(completion) = &completions[graph.output()] {
            B::order_after(completion, context)
                .map_err(|error| LayerwiseRuntimeError::Submission(error.to_string()))?;
        }
        let output = self
            .architecture
            .finish_forward_parallel(&hidden, state, &forward_context, parallel, context)
            .map_err(LayerwiseRuntimeError::Architecture)?;
        policy
            .finish(&output)
            .map_err(LayerwiseRuntimeError::Policy)?;
        Ok((output, forward_context))
    }
}

/// Indexed owned unit used by [`ResidentUnitWindow`].
pub struct ResidentUnitLease<U> {
    index: usize,
    unit: U,
}

impl<U> std::ops::Deref for ResidentUnitLease<U> {
    type Target = U;

    fn deref(&self) -> &Self::Target {
        &self.unit
    }
}

impl<U> std::ops::DerefMut for ResidentUnitLease<U> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.unit
    }
}

/// Minimal one-at-a-time unit window useful for conformance and resident storage.
pub struct ResidentUnitWindow<U> {
    units: Vec<Option<U>>,
}

impl<U> ResidentUnitWindow<U> {
    /// Creates a window over an ordered set of already populated units.
    pub fn new(units: Vec<U>) -> Self {
        Self {
            units: units.into_iter().map(Some).collect(),
        }
    }
}

impl<B, U> LayerwisePolicy<B, U> for ResidentUnitWindow<U>
where
    B: NeuralBackend,
{
    type Lease = ResidentUnitLease<U>;
    type Error = ResidentUnitWindowError;

    fn begin(
        &mut self,
        _initial: &B::Tensor,
        _context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<(), Self::Error> {
        Ok(())
    }

    fn abort(
        &mut self,
        active: Option<(usize, crate::ExecutionUnitAddress, Self::Lease)>,
        _context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) {
        let Some((ordinal, _, lease)) = active else {
            return;
        };
        debug_assert_eq!(lease.index, ordinal);
        if let Some(slot) = self.units.get_mut(lease.index) {
            debug_assert!(slot.is_none());
            if slot.is_none() {
                *slot = Some(lease.unit);
            }
        }
    }

    fn acquire<E, F>(
        &mut self,
        index: usize,
        _address: crate::ExecutionUnitAddress,
        _build: F,
        _context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<Self::Lease, LayerwiseAcquireError<E, Self::Error>>
    where
        F: FnOnce(&<B::Tensor as eredu_nn::Tensor>::Context) -> Result<U, E>,
    {
        let count = self.units.len();
        let unit = self
            .units
            .get_mut(index)
            .ok_or(ResidentUnitWindowError::UnknownUnit { index, count })
            .map_err(LayerwiseAcquireError::Policy)?
            .take()
            .ok_or(ResidentUnitWindowError::AlreadyAcquired { index })
            .map_err(LayerwiseAcquireError::Policy)?;
        Ok(ResidentUnitLease { index, unit })
    }

    fn complete<'a, StateValues, ContextValues>(
        &mut self,
        index: usize,
        _address: crate::ExecutionUnitAddress,
        lease: Self::Lease,
        _output: &'a B::Tensor,
        _state_values: StateValues,
        _context_values: ContextValues,
        _context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<(), Self::Error>
    where
        B::Tensor: 'a,
        StateValues: Iterator<Item = &'a B::Tensor>,
        ContextValues: Iterator<Item = &'a B::Tensor>,
    {
        if lease.index != index {
            return Err(ResidentUnitWindowError::MismatchedUnit {
                expected: index,
                actual: lease.index,
            });
        }
        let slot = self
            .units
            .get_mut(index)
            .expect("acquired unit index remains in the window");
        if slot.replace(lease.unit).is_some() {
            return Err(ResidentUnitWindowError::AlreadyResident { index });
        }
        Ok(())
    }

    fn finish(
        &mut self,
        _output: &B::Tensor,
        _context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<(), Self::Error> {
        Ok(())
    }
}

/// Invalid access to an owned resident-unit window.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
pub enum ResidentUnitWindowError {
    /// The requested unit is outside the ordered window.
    #[error("unit {index} is outside the {count}-unit window")]
    UnknownUnit {
        /// Requested index.
        index: usize,
        /// Window size.
        count: usize,
    },
    /// A unit was acquired twice without an intervening completion.
    #[error("unit {index} is already acquired")]
    AlreadyAcquired {
        /// Requested index.
        index: usize,
    },
    /// Completion returned a lease for the wrong unit.
    #[error("unit completion expected {expected}, received {actual}")]
    MismatchedUnit {
        /// Expected unit.
        expected: usize,
        /// Lease unit.
        actual: usize,
    },
    /// A completion attempted to overwrite a resident unit.
    #[error("unit {index} is already resident")]
    AlreadyResident {
        /// Conflicting index.
        index: usize,
    },
}
