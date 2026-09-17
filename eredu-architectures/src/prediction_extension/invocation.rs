//! Exact result and state dependencies of one physical prediction-module call.

/// Borrowed paid destination belonging to the exact physical module call.
/// Its provider owns native clone slots and keeps every failed-prefix root under
/// the same module/source recovery. It supplies no numerical or completion grant.
pub trait PreparedPredictionInvocationRoots<T> {
    /// Pays concrete shared caller/iterator/error frames before construction.
    fn controls(&self, bytes: usize) -> Result<(), eredu_nn::Error>;
    /// Invokes the borrowed visitor exactly once and moves its already prepared
    /// final destination on success. On failure the provider retains all filled
    /// roots; returning an error is not a completion or retirement signal.
    fn retain(
        &mut self,
        visit: &mut dyn FnMut(&mut dyn FnMut(&T)),
    ) -> Result<Vec<T>, eredu_nn::Error>;
}

pub(super) fn prediction_invocation_prepared<'a, T: 'a, O, I, F, const N: usize>(
    outcome: Result<O, eredu_nn::Error>,
    state: I,
    outputs: F,
    source: &mut dyn PreparedPredictionInvocationRoots<T>,
) -> PredictionInvocation<T, O>
where
    I: IntoIterator<Item = &'a T>,
    F: FnOnce(&O) -> Option<[&T; N]>,
{
    use std::mem::{size_of, size_of_val};
    let controls = [
        size_of::<PredictionInvocation<T, O>>(),
        size_of::<Result<O, eredu_nn::Error>>(),
        size_of::<I>(),
        size_of::<Option<I::IntoIter>>(),
        size_of::<Option<F>>(),
        size_of::<Option<[&T; N]>>(),
        size_of::<Result<Vec<T>, eredu_nn::Error>>(),
        size_of::<(
            &mut Option<I::IntoIter>,
            &mut Option<F>,
            &Result<O, eredu_nn::Error>,
        )>(),
        size_of::<&mut dyn PreparedPredictionInvocationRoots<T>>(),
    ];
    let paid = controls
        .into_iter()
        .try_fold(size_of_val(&controls), usize::checked_add)
        .ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)
        .map_err(eredu_nn::Error::from)
        .and_then(|bytes| source.controls(bytes));
    if let Err(cause) = paid {
        return PredictionInvocation {
            outcome: Err(cause),
            retained: Vec::new(),
        };
    }
    let mut state = Some(state.into_iter());
    let mut outputs = Some(outputs);
    let roots = source.retain(&mut |visit| {
        if let Some(state) = state.take() {
            for value in state {
                visit(value);
            }
            if let Ok(value) = &outcome {
                if let Some(values) = outputs.take().expect("once-only root visitor")(value) {
                    for value in values {
                        visit(value);
                    }
                }
            }
        }
    });
    match roots {
        Ok(retained) => PredictionInvocation { outcome, retained },
        Err(cause) => PredictionInvocation {
            outcome: Err(cause),
            retained: Vec::new(),
        },
    }
}

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

    /// Moves an existing destination without collecting or cloning its roots.
    /// The caller still owns its source, storage admission and completion duty.
    pub fn from_prepared_parts(outcome: Result<O, eredu_nn::Error>, retained: Vec<T>) -> Self {
        Self { outcome, retained }
    }

    /// Borrows evaluation roots while the native module lease remains held.
    pub fn retained_values(&self) -> impl Iterator<Item = &T> {
        self.retained.iter()
    }

    /// Transfers the existing root destination to a backend completion owner.
    /// The outcome remains provisional until that owner establishes completion,
    /// terminal failure or safe teardown. No second root inventory is allocated.
    pub fn into_parts(self) -> (Result<O, eredu_nn::Error>, Vec<T>) {
        (self.outcome, self.retained)
    }

    /// Publishes the equation outcome after the caller has established safe
    /// completion, terminal failure, or teardown of the retained native work.
    pub fn into_outcome(self) -> Result<O, eredu_nn::Error> {
        self.outcome
    }
}

/// Optional readout retains changed state even when no score/capture output exists.
pub(crate) fn prediction_invocation_optional<'a, T: Clone + 'a, O, const N: usize>(
    outcome: Result<O, eredu_nn::Error>,
    state: impl IntoIterator<Item = &'a T>,
    outputs: impl FnOnce(&O) -> Option<[&T; N]>,
) -> PredictionInvocation<T, O> {
    let mut retained = state.into_iter().cloned().collect::<Vec<_>>();
    if let Ok(output) = &outcome {
        retained.extend(outputs(output).into_iter().flatten().cloned());
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
        M::invoke_module_with_roots(module, context, |module, source| {
            let outcome = operation(module, state);
            M::retain_prediction_invocation_from_source(
                outcome,
                state.retained_values(),
                outputs,
                source,
                context,
            )
        })
    }
}
