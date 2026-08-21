//! Statically dispatched layered-architecture lifecycle and resident execution.

#![allow(clippy::too_many_arguments, clippy::type_complexity)]

use eredu_nn::{NeuralBackend, Parameterized};

use crate::{
    ExecutionGraph, ExecutionGroupSchedule, ExecutionUnitLayout, RuntimeState, SubmissionBackend,
};

/// Backend-native activation and architecture-owned forward context.
pub struct LayeredForwardState<T, C> {
    /// Initial activation supplied to the first execution unit.
    pub hidden: T,
    /// Masks, positions, or other architecture-owned forward values.
    pub context: C,
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
pub trait LayeredArchitecture<B, S>
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

    /// Acquires one populated unit for exclusive execution.
    ///
    /// The flat ordinal addresses storage while `address` preserves the
    /// architecture execution group and group-local unit index for scheduling.
    fn acquire(
        &mut self,
        ordinal: usize,
        address: crate::ExecutionUnitAddress,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<Self::Lease, Self::Error>;

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
            backend: std::marker::PhantomData,
        }
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
        let executors = B::fork_executors(context, graph.groups().len())
            .map_err(|error| LayerwiseRuntimeError::Submission(error.to_string()))?;
        let forward = self
            .architecture
            .begin_forward(input, state, context)
            .map_err(LayerwiseRuntimeError::Architecture)?;
        let initial_completion = (graph.groups().len() > 1)
            .then(|| B::submit(context, [&forward.hidden]))
            .transpose()
            .map_err(|error| LayerwiseRuntimeError::Submission(error.to_string()))?;
        self.policy
            .begin(&forward.hidden, context)
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
                    let mut lease = self
                        .policy
                        .acquire(ordinal, address, executor)
                        .map_err(LayerwiseRuntimeError::Policy)?;
                    hidden = execute(
                        &mut self.architecture,
                        group,
                        index,
                        &mut lease,
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
                    self.policy
                        .complete(
                            ordinal,
                            address,
                            lease,
                            &hidden,
                            state_values.into_iter(),
                            context_values,
                            executor,
                        )
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
        self.policy
            .finish(&output, context)
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
        let executors = B::fork_executors(context, graph.groups().len())
            .map_err(|error| LayerwiseRuntimeError::Submission(error.to_string()))?;
        let forward = self
            .architecture
            .begin_forward_parallel(input, state, parallel, context)
            .map_err(LayerwiseRuntimeError::Architecture)?;
        let initial_completion = (graph.groups().len() > 1)
            .then(|| B::submit(context, [&forward.hidden]))
            .transpose()
            .map_err(|error| LayerwiseRuntimeError::Submission(error.to_string()))?;
        self.policy
            .begin(&forward.hidden, context)
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
                    let mut lease = self
                        .policy
                        .acquire(ordinal, address, executor)
                        .map_err(LayerwiseRuntimeError::Policy)?;
                    hidden = execute(
                        &mut self.architecture,
                        group,
                        index,
                        &mut lease,
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
                    self.policy
                        .complete(
                            ordinal,
                            address,
                            lease,
                            &hidden,
                            state_values.into_iter(),
                            context_values,
                            executor,
                        )
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
        self.policy
            .finish(&output, context)
            .map_err(LayerwiseRuntimeError::Policy)?;
        Ok((output, forward_context))
    }

    /// Runs one pass with statically dispatched unit-boundary observation and
    /// optional causal intervention.
    pub fn forward_with_unit_hook<'a, F>(
        &mut self,
        input: A::Input<'a>,
        state: &mut S,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        mut hook: F,
    ) -> Result<B::Tensor, LayerwiseRuntimeError<A::Error, P::Error>>
    where
        F: FnMut(&str, &B::Tensor, &B::Tensor) -> Result<Option<B::Tensor>, A::Error>,
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
        let executors = B::fork_executors(context, graph.groups().len())
            .map_err(|error| LayerwiseRuntimeError::Submission(error.to_string()))?;
        let forward = self
            .architecture
            .begin_forward(input, state, context)
            .map_err(LayerwiseRuntimeError::Architecture)?;
        let initial_completion = (graph.groups().len() > 1)
            .then(|| B::submit(context, [&forward.hidden]))
            .transpose()
            .map_err(|error| LayerwiseRuntimeError::Submission(error.to_string()))?;
        self.policy
            .begin(&forward.hidden, context)
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
                for index in 0..layout
                    .group_range(group)
                    .expect("layout covers every graph group")
                    .len()
                {
                    let ordinal = layout
                        .ordinal(group, index)
                        .expect("group-local unit belongs to the layout");
                    let address = layout
                        .address(ordinal)
                        .expect("group-local unit has a stable policy address");
                    let path = self
                        .architecture
                        .unit_path(group, index)
                        .map_err(LayerwiseRuntimeError::Architecture)?;
                    let mut lease = self
                        .policy
                        .acquire(ordinal, address, executor)
                        .map_err(LayerwiseRuntimeError::Policy)?;
                    let unit_input = hidden.clone();
                    let output = self
                        .architecture
                        .forward_unit(
                            group,
                            index,
                            &mut lease,
                            &unit_input,
                            state,
                            &mut forward_context,
                            executor,
                        )
                        .map_err(LayerwiseRuntimeError::Architecture)?;
                    hidden = hook(&path, &unit_input, &output)
                        .map_err(LayerwiseRuntimeError::Architecture)?
                        .unwrap_or(output);
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
                    self.policy
                        .complete(
                            ordinal,
                            address,
                            lease,
                            &hidden,
                            state_values.into_iter(),
                            context_values,
                            executor,
                        )
                        .map_err(LayerwiseRuntimeError::Policy)?;
                }
            }
            hidden = self
                .architecture
                .complete_execution_group(group, &hidden, state, &mut forward_context, executor)
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
        self.policy
            .finish(&output, context)
            .map_err(LayerwiseRuntimeError::Policy)?;
        Ok(output)
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

    fn acquire(
        &mut self,
        index: usize,
        _address: crate::ExecutionUnitAddress,
        _context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<Self::Lease, Self::Error> {
        let count = self.units.len();
        let unit = self
            .units
            .get_mut(index)
            .ok_or(ResidentUnitWindowError::UnknownUnit { index, count })?
            .take()
            .ok_or(ResidentUnitWindowError::AlreadyAcquired { index })?;
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
