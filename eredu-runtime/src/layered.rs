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
    /// Concrete architecture or backend failure.
    type Error;

    /// Stable architecture compatibility identity.
    fn model_identity(&self) -> &str;

    /// Declares the dependency graph between ordered execution groups.
    fn execution_graph(&self) -> Result<ExecutionGraph, Self::Error>;

    /// Returns the total number of ordered execution units.
    fn unit_count(&self) -> Result<usize, Self::Error>;

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
    state: S,
    backend: std::marker::PhantomData<fn() -> B>,
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
        state: S,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<Self, A::Error> {
        let count = architecture.unit_count()?;
        let units = (0..count)
            .map(|index| architecture.build_unit(index, context))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            architecture,
            units,
            state,
            backend: std::marker::PhantomData,
        })
    }

    /// Runs one complete prefill or decode pass without dynamic dispatch.
    pub fn forward<'a>(
        &mut self,
        input: A::Input<'a>,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<B::Tensor, A::Error> {
        let mut forward = self
            .architecture
            .begin_forward(input, &mut self.state, context)?;
        for (index, unit) in self.units.iter_mut().enumerate() {
            forward.hidden = self.architecture.forward_unit(
                index,
                unit,
                &forward.hidden,
                &mut self.state,
                &mut forward.context,
                context,
            )?;
        }
        self.architecture.finish_forward(
            &forward.hidden,
            &mut self.state,
            &forward.context,
            context,
        )
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

    /// Borrows the concrete mutable-state realization.
    pub const fn state(&self) -> &S {
        &self.state
    }

    /// Mutably borrows the concrete mutable-state realization.
    pub fn state_mut(&mut self) -> &mut S {
        &mut self.state
    }
}
