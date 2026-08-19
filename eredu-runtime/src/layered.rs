//! Statically dispatched layered-architecture lifecycle and resident execution.

use eredu_nn::{NeuralBackend, Parameterized};

use crate::{ExecutionGraph, RuntimeState};

/// Backend-native activation and architecture-owned forward context.
pub struct LayeredForwardState<T, C> {
    /// Initial activation supplied to the first execution unit.
    pub hidden: T,
    /// Masks, positions, or other architecture-owned forward values.
    pub context: C,
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

    /// Returns the total number of ordered execution units.
    fn unit_count(&self) -> Result<usize, Self::Error>;

    /// Returns the stable architecture-owned path of one execution unit.
    fn unit_path(&self, index: usize) -> Result<String, Self::Error>;

    /// Borrows pinned modules for parameter discovery and binding.
    fn static_modules(&self) -> &Self::StaticModules;

    /// Mutably borrows pinned modules for parameter binding.
    fn static_modules_mut(&mut self) -> &mut Self::StaticModules;

    /// Builds one unloaded execution unit using backend-native operators.
    fn build_unit(
        &self,
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

    /// Executes one ordered unit against its concrete mutable layer state.
    fn forward_unit(
        &mut self,
        index: usize,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error>;

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
        index: usize,
    ) -> Self::RetainedContextValues<'a>;
}

/// Fully resident runtime using the same lifecycle as bounded execution.
pub struct ResidentRuntime<A, B, S>
where
    B: NeuralBackend,
    S: RuntimeState<B>,
    A: LayeredArchitecture<B, S>,
{
    architecture: A,
    units: Vec<A::Unit>,
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
        let count = architecture.unit_count()?;
        let units = (0..count)
            .map(|index| architecture.build_unit(index, context))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            architecture,
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
        let mut forward = self.architecture.begin_forward(input, state, context)?;
        for (index, unit) in self.units.iter_mut().enumerate() {
            forward.hidden = self.architecture.forward_unit(
                index,
                unit,
                &forward.hidden,
                state,
                &mut forward.context,
                context,
            )?;
        }
        self.architecture
            .finish_forward(&forward.hidden, state, &forward.context, context)
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
    pub fn units(&self) -> &[A::Unit] {
        &self.units
    }

    /// Mutably borrows resident execution units for parameter binding.
    pub fn units_mut(&mut self) -> &mut [A::Unit] {
        &mut self.units
    }

    /// Decomposes the runtime without cloning backend-native values.
    pub fn into_parts(self) -> (A, Vec<A::Unit>) {
        (self.architecture, self.units)
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

    /// Acquires one populated unit for exclusive execution.
    fn acquire(
        &mut self,
        index: usize,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<Self::Lease, Self::Error>;

    /// Retains the unit and dependent native values through exact completion.
    fn complete<'a, StateValues, ContextValues>(
        &mut self,
        index: usize,
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
    /// Unit acquisition or exact-completion failure.
    #[error("layerwise execution policy failed: {0}")]
    Policy(P),
}

/// Bounded-unit runtime invoking the same architecture lifecycle as resident execution.
pub struct LayerwiseRuntime<A, B, S, P>
where
    B: NeuralBackend,
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
    B: NeuralBackend,
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
        let count = self
            .architecture
            .unit_count()
            .map_err(LayerwiseRuntimeError::Architecture)?;
        let mut forward = self
            .architecture
            .begin_forward(input, state, context)
            .map_err(LayerwiseRuntimeError::Architecture)?;
        for index in 0..count {
            let mut lease = self
                .policy
                .acquire(index, context)
                .map_err(LayerwiseRuntimeError::Policy)?;
            forward.hidden = self
                .architecture
                .forward_unit(
                    index,
                    &mut lease,
                    &forward.hidden,
                    state,
                    &mut forward.context,
                    context,
                )
                .map_err(LayerwiseRuntimeError::Architecture)?;
            let state_values = state
                .retained_values(index)
                .map_err(LayerwiseRuntimeError::State)?;
            let context_values = self
                .architecture
                .retained_context_values(&forward.context, index);
            self.policy
                .complete(
                    index,
                    lease,
                    &forward.hidden,
                    state_values,
                    context_values,
                    context,
                )
                .map_err(LayerwiseRuntimeError::Policy)?;
        }
        let output = self
            .architecture
            .finish_forward(&forward.hidden, state, &forward.context, context)
            .map_err(LayerwiseRuntimeError::Architecture)?;
        self.policy
            .finish(&output, context)
            .map_err(LayerwiseRuntimeError::Policy)?;
        Ok(output)
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
        let count = self
            .architecture
            .unit_count()
            .map_err(LayerwiseRuntimeError::Architecture)?;
        let mut forward = self
            .architecture
            .begin_forward(input, state, context)
            .map_err(LayerwiseRuntimeError::Architecture)?;
        for index in 0..count {
            let path = self
                .architecture
                .unit_path(index)
                .map_err(LayerwiseRuntimeError::Architecture)?;
            let mut lease = self
                .policy
                .acquire(index, context)
                .map_err(LayerwiseRuntimeError::Policy)?;
            let input = forward.hidden.clone();
            let output = self
                .architecture
                .forward_unit(
                    index,
                    &mut lease,
                    &input,
                    state,
                    &mut forward.context,
                    context,
                )
                .map_err(LayerwiseRuntimeError::Architecture)?;
            forward.hidden = hook(&path, &input, &output)
                .map_err(LayerwiseRuntimeError::Architecture)?
                .unwrap_or(output);
            let state_values = state
                .retained_values(index)
                .map_err(LayerwiseRuntimeError::State)?;
            let context_values = self
                .architecture
                .retained_context_values(&forward.context, index);
            self.policy
                .complete(
                    index,
                    lease,
                    &forward.hidden,
                    state_values,
                    context_values,
                    context,
                )
                .map_err(LayerwiseRuntimeError::Policy)?;
        }
        let output = self
            .architecture
            .finish_forward(&forward.hidden, state, &forward.context, context)
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

    fn acquire(
        &mut self,
        index: usize,
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
