//! Exact result and state dependencies of one physical prediction-module call.

/// An equation outcome together with every output and changed-state tensor
/// needed to settle its module loan. An error does not imply native completion.
pub struct PredictionInvocation<T, O> {
    outcome: Result<O, eredu_nn::Error>,
    retained: Vec<T>,
}

impl<T, O> PredictionInvocation<T, O> {
    /// Retains the exact dependency set on both successful and failed work.
    pub fn new(outcome: Result<O, eredu_nn::Error>, retained: impl IntoIterator<Item = T>) -> Self {
        Self {
            outcome,
            retained: retained.into_iter().collect(),
        }
    }

    /// Borrows evaluation roots while the native module lease remains held.
    pub fn retained_values(&self) -> impl Iterator<Item = &T> {
        self.retained.iter()
    }

    /// Publishes the equation outcome after the caller has established safe
    /// completion, terminal failure, or teardown of the retained native work.
    pub fn into_outcome(self) -> Result<O, eredu_nn::Error> {
        self.outcome
    }
}

pub(crate) fn prediction_invocation<'a, T: Clone + 'a, O, const N: usize>(
    outcome: Result<O, eredu_nn::Error>,
    state: impl IntoIterator<Item = &'a T>,
    outputs: impl FnOnce(&O) -> [&T; N],
) -> PredictionInvocation<T, O> {
    let mut retained = state.into_iter().cloned().collect::<Vec<_>>();
    if let Ok(output) = &outcome {
        retained.extend(outputs(output).into_iter().cloned());
    }
    PredictionInvocation::new(outcome, retained)
}

/// Scoped physical-module access used by family loops without changing equations.
pub(crate) trait PredictionModuleInvoker<B: eredu_nn::NeuralBackend, U, C> {
    type Module;
    fn invoke<O, const N: usize>(
        module: &mut Self::Module,
        state: &mut C,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        operation: impl FnOnce(&mut U, &mut C) -> Result<O, eredu_nn::Error>,
        outputs: impl FnOnce(&O) -> [&B::Tensor; N],
    ) -> Result<O, eredu_nn::Error>;
}

pub(crate) struct ResidentPredictionModules<W>(std::marker::PhantomData<W>);
impl<B: eredu_nn::NeuralBackend, U, C, W: AsMut<U>> PredictionModuleInvoker<B, U, C>
    for ResidentPredictionModules<W>
{
    type Module = W;
    fn invoke<O, const N: usize>(
        module: &mut W,
        state: &mut C,
        _context: &<B::Tensor as eredu_nn::Tensor>::Context,
        operation: impl FnOnce(&mut U, &mut C) -> Result<O, eredu_nn::Error>,
        _outputs: impl FnOnce(&O) -> [&B::Tensor; N],
    ) -> Result<O, eredu_nn::Error> {
        operation(module.as_mut(), state)
    }
}

pub(crate) struct MaterializedPredictionModules<B, M>(std::marker::PhantomData<fn() -> (B, M)>);
impl<B, U, C, M> PredictionModuleInvoker<B, U, C> for MaterializedPredictionModules<B, M>
where
    B: eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::GroupedNeuralBackend
        + eredu_nn::HyperNeuralBackend,
    U: eredu_nn::Parameterized<B::Tensor>,
    C: eredu_runtime::RuntimeLayerState<B>,
    M: super::PredictionExtensionMaterializer<B>,
{
    type Module = M::Module<U>;
    fn invoke<O, const N: usize>(
        module: &mut Self::Module,
        state: &mut C,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        operation: impl FnOnce(&mut U, &mut C) -> Result<O, eredu_nn::Error>,
        outputs: impl FnOnce(&O) -> [&B::Tensor; N],
    ) -> Result<O, eredu_nn::Error> {
        M::invoke_module(module, context, |module| {
            let outcome = operation(module, state);
            prediction_invocation(outcome, state.retained_values(), outputs)
        })
    }
}
