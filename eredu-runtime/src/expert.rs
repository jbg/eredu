//! Runtime ownership boundary for routed expert acquisition and residency.

use eredu_nn::{
    GatedProductExpertBankOperator, Relu2ExpertBankOperator, RoutedNeuralBackend, RoutingResult,
    Tensor, TensorParallelExpertOutput,
};

use crate::ExpertPass;

/// One architecture route batch submitted to a runtime expert provider.
pub struct RoutedExpertRequest<'a, T> {
    /// Global decoder layer requesting experts.
    pub layer: usize,
    /// Flattened token rows submitted to the selected experts.
    pub input: &'a T,
    /// Backend-native selected expert IDs, scores, and weights.
    pub routes: &'a RoutingResult<T>,
    /// Whether this route batch belongs to prefill or decode.
    pub pass: ExpertPass,
}

/// Provider result that distinguishes complete outputs from rank-local TP work.
pub enum RoutedExpertTensorParallelOutput<T> {
    /// Provider already completed every required collective and bias addition.
    Complete(T),
    /// Caller must all-sum `reducible`, then add `post_reduce` exactly once.
    Partial(TensorParallelExpertOutput<T>),
}

/// Completes one rank-local expert output with one all-sum and one post-bias add.
pub fn reduce_tensor_parallel_expert_output<B>(
    output: TensorParallelExpertOutput<B::Tensor>,
    parallel: &B::ParallelContext,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<B::Tensor, eredu_nn::Error>
where
    B: RoutedNeuralBackend,
{
    let reduced = B::sum_parallel(output.reducible, parallel, context)?;
    match output.post_reduce {
        Some(bias) => reduced.add(&bias, context),
        None => Ok(reduced),
    }
}

/// Combines two rank-local expert partials without introducing another collective.
pub fn combine_tensor_parallel_expert_outputs<B>(
    left: TensorParallelExpertOutput<B::Tensor>,
    right: TensorParallelExpertOutput<B::Tensor>,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<TensorParallelExpertOutput<B::Tensor>, eredu_nn::Error>
where
    B: RoutedNeuralBackend,
{
    let post_reduce = match (left.post_reduce, right.post_reduce) {
        (Some(left), Some(right)) => Some(left.add(&right, context)?),
        (Some(bias), None) | (None, Some(bias)) => Some(bias),
        (None, None) => None,
    };
    Ok(TensorParallelExpertOutput {
        reducible: left.reducible.add(&right.reducible, context)?,
        post_reduce,
    })
}

/// Combines routed/shared provider outputs while requiring one coherent TP mode.
pub fn combine_routed_expert_tensor_parallel<B>(
    left: RoutedExpertTensorParallelOutput<B::Tensor>,
    right: RoutedExpertTensorParallelOutput<B::Tensor>,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<RoutedExpertTensorParallelOutput<B::Tensor>, eredu_nn::Error>
where
    B: RoutedNeuralBackend,
{
    match (left, right) {
        (
            RoutedExpertTensorParallelOutput::Complete(left),
            RoutedExpertTensorParallelOutput::Complete(right),
        ) => Ok(RoutedExpertTensorParallelOutput::Complete(
            left.add(&right, context)?,
        )),
        (
            RoutedExpertTensorParallelOutput::Partial(left),
            RoutedExpertTensorParallelOutput::Partial(right),
        ) => combine_tensor_parallel_expert_outputs::<B>(left, right, context)
            .map(RoutedExpertTensorParallelOutput::Partial),
        _ => Err(eredu_nn::Error::backend(
            "provider mixed complete and rank-local expert outputs in one block",
        )),
    }
}

/// Completes a provider TP result while preserving provider-owned collectives.
pub fn reduce_routed_expert_tensor_parallel<B>(
    output: RoutedExpertTensorParallelOutput<B::Tensor>,
    parallel: &B::ParallelContext,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<B::Tensor, eredu_nn::Error>
where
    B: RoutedNeuralBackend,
{
    match output {
        RoutedExpertTensorParallelOutput::Complete(output) => Ok(output),
        RoutedExpertTensorParallelOutput::Partial(output) => {
            reduce_tensor_parallel_expert_output::<B>(output, parallel, context)
        }
    }
}

/// Runtime boundary for resident or independently cached routed experts.
///
/// Implementations own identity ordering, acquisition, leases, chunking,
/// budgets, and residency reports. They keep every lease alive until the
/// backend-native routed result is safe to return. The backend retains tensor
/// storage, transfers, compact-bank construction, and execution kernels.
pub trait RoutedExpertProvider<B>
where
    B: RoutedNeuralBackend,
{
    /// Provider-specific acquisition or execution failure.
    type Error;

    /// Whether legacy ReLU2 provider output is a rank-local partial.
    fn output_is_tensor_parallel_partial(&self) -> bool {
        false
    }

    /// Executes one typed route batch while retaining its acquired resources.
    fn forward_routed(
        &mut self,
        resident_bank: &mut B::GatedProductExpertBank,
        request: RoutedExpertRequest<'_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error>;

    /// Executes a rank-local tensor-parallel routed contribution. Providers
    /// returning complete outputs may use the ordinary path unchanged.
    fn forward_routed_tensor_parallel(
        &mut self,
        resident_bank: &mut B::GatedProductExpertBank,
        request: RoutedExpertRequest<'_, B::Tensor>,
        partitions: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<RoutedExpertTensorParallelOutput<B::Tensor>, Self::Error> {
        let _ = partitions;
        self.forward_routed(resident_bank, request, context)
            .map(RoutedExpertTensorParallelOutput::Complete)
    }

    /// Executes one ReLU-squared route batch through the same residency boundary.
    fn forward_relu2_routed(
        &mut self,
        resident_bank: &mut B::Relu2ExpertBank,
        request: RoutedExpertRequest<'_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error>;
}

/// Provider for a fully resident expert bank.
#[derive(Debug, Default, Clone, Copy)]
pub struct ResidentExpertProvider;

impl<B> RoutedExpertProvider<B> for ResidentExpertProvider
where
    B: RoutedNeuralBackend,
{
    type Error = eredu_nn::Error;

    fn output_is_tensor_parallel_partial(&self) -> bool {
        true
    }

    fn forward_routed(
        &mut self,
        resident_bank: &mut B::GatedProductExpertBank,
        request: RoutedExpertRequest<'_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        resident_bank.forward_routed(request.input, request.routes, context)
    }

    fn forward_routed_tensor_parallel(
        &mut self,
        resident_bank: &mut B::GatedProductExpertBank,
        request: RoutedExpertRequest<'_, B::Tensor>,
        partitions: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<RoutedExpertTensorParallelOutput<B::Tensor>, Self::Error> {
        resident_bank
            .forward_routed_tensor_parallel(request.input, request.routes, partitions, context)
            .map(RoutedExpertTensorParallelOutput::Partial)
    }

    fn forward_relu2_routed(
        &mut self,
        resident_bank: &mut B::Relu2ExpertBank,
        request: RoutedExpertRequest<'_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        resident_bank.forward_routed(request.input, request.routes, context)
    }
}
