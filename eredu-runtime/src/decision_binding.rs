//! Portable construction of a decision program from native resources.

use eredu_core::{
    decision::{DecisionBinding, DecisionBindingFactory},
    BackendFailure, DevicePlan,
};
use eredu_nn::{NeuralBackend, Tensor};

use crate::LayerWeightResidency;

/// Portable decision state over the native component cache storage.
pub type DecisionState<B> =
    crate::state::HybridState<<B as eredu_core::backend::LocalComponentStorage>::CacheStorage>;

/// Family-owned construction that consumes ordinary backend mechanisms.
pub trait PreparedDecisionProgram<B: NeuralBackend, S, M, C, I> {
    /// Bound decision model.
    type Model;
    /// Portable preparation or native binding failure.
    type Error: From<BackendFailure>;

    /// Constructs the admitted task with the supplied native mechanisms.
    fn bind(
        self,
        residency: LayerWeightResidency,
        mechanisms: &M,
        context: &<B::Tensor as Tensor>::Context,
        input_lowering: I,
    ) -> Result<Self::Model, Self::Error>;
}

/// Constructs the selected program and establishes completion when requested.
pub fn bind_decision_model<D, F, P>(
    factory: &F,
    program: P,
    device: &DevicePlan,
    residency: LayerWeightResidency,
    synchronize_load: bool,
) -> Result<P::Model, P::Error>
where
    D: DecisionBinding,
    F: DecisionBindingFactory<Binding = D>,
    <D::Backend as eredu_core::backend::LocalComponentStorage>::CacheStorage:
        crate::state::HybridCacheStorage,
    P: PreparedDecisionProgram<
        D::Backend,
        DecisionState<D::Backend>,
        D,
        D::Parallel,
        D::InputLowering,
    >,
{
    let binding = factory.decision_binding(device)?;
    let model = program.bind(
        residency,
        &binding,
        binding.context(),
        binding.input_lowering(),
    )?;
    if synchronize_load {
        binding.synchronize()?;
    }
    Ok(model)
}
