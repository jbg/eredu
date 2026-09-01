//! Runtime ownership boundary for routed expert acquisition and residency.

use eredu_nn::{
    GroupSelection, GroupedGatedProductOperator, GroupedNeuralBackend, GroupedRelu2Operator,
    Tensor, TensorParallelGroupedOutput,
};

use crate::ExpertPass;
use crate::{ActivationObserver, RoutingObservation};

/// One architecture route batch submitted to a runtime expert provider.
pub struct RoutedExpertRequest<'a, T> {
    /// Global decoder layer requesting experts.
    pub layer: usize,
    /// Flattened token rows submitted to the selected experts.
    pub input: &'a T,
    /// Backend-native selected expert IDs, scores, and weights.
    pub routes: &'a GroupSelection<T>,
    /// Whether this route batch belongs to prefill or decode.
    pub pass: ExpertPass,
}

/// Provider result that distinguishes complete outputs from rank-local TP work.
pub enum RoutedExpertTensorParallelOutput<T> {
    /// Provider already completed every required collective and bias addition.
    Complete(T),
    /// Caller must all-sum `reducible`, then add `post_reduce` exactly once.
    Partial(TensorParallelGroupedOutput<T>),
}

/// Completes one rank-local expert output with one all-sum and one post-bias add.
pub fn reduce_tensor_parallel_expert_output<B>(
    output: TensorParallelGroupedOutput<B::Tensor>,
    parallel: &B::ParallelContext,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<B::Tensor, eredu_nn::Error>
where
    B: GroupedNeuralBackend,
{
    let reduced = B::sum_parallel(output.reducible().clone(), parallel, context)?;
    match output.post_reduce().cloned() {
        Some(bias) => reduced.add(&bias, context),
        None => Ok(reduced),
    }
}

/// Combines two rank-local expert partials without introducing another collective.
pub fn combine_tensor_parallel_expert_outputs<B>(
    left: TensorParallelGroupedOutput<B::Tensor>,
    right: TensorParallelGroupedOutput<B::Tensor>,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<TensorParallelGroupedOutput<B::Tensor>, eredu_nn::Error>
where
    B: GroupedNeuralBackend,
{
    let post_reduce = match (left.post_reduce().cloned(), right.post_reduce().cloned()) {
        (Some(left), Some(right)) => Some(left.add(&right, context)?),
        (Some(bias), None) | (None, Some(bias)) => Some(bias),
        (None, None) => None,
    };
    Ok(TensorParallelGroupedOutput::new(
        left.reducible().add(right.reducible(), context)?,
        post_reduce,
    ))
}

/// Combines routed/shared provider outputs while requiring one coherent TP mode.
pub fn combine_routed_expert_tensor_parallel<B>(
    left: RoutedExpertTensorParallelOutput<B::Tensor>,
    right: RoutedExpertTensorParallelOutput<B::Tensor>,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<RoutedExpertTensorParallelOutput<B::Tensor>, eredu_nn::Error>
where
    B: GroupedNeuralBackend,
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
    B: GroupedNeuralBackend,
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
    B: GroupedNeuralBackend,
{
    /// Provider-specific acquisition or execution failure.
    type Error;

    /// Executes one typed route batch while retaining its acquired resources.
    fn forward_grouped(
        &mut self,
        resident_bank: &mut B::GatedProductGroups,
        request: RoutedExpertRequest<'_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error>;

    /// Executes one ReLU-squared route batch through the same residency boundary.
    fn forward_relu2_routed(
        &mut self,
        resident_bank: &mut B::Relu2Groups,
        request: RoutedExpertRequest<'_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error>;
}

/// Additive provider mechanism for tensor-parallel grouped partials.
pub trait TensorParallelRoutedExpertProvider<B>: RoutedExpertProvider<B>
where
    B: GroupedNeuralBackend,
{
    /// Executes a rank-local gated-product contribution.
    fn forward_grouped_tensor_parallel(
        &mut self,
        resident_bank: &mut B::GatedProductGroups,
        request: RoutedExpertRequest<'_, B::Tensor>,
        partitions: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<RoutedExpertTensorParallelOutput<B::Tensor>, Self::Error>;

    /// Executes a rank-local ReLU-squared contribution.
    fn forward_relu2_routed_tensor_parallel(
        &mut self,
        resident_bank: &mut B::Relu2Groups,
        request: RoutedExpertRequest<'_, B::Tensor>,
        partitions: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<RoutedExpertTensorParallelOutput<B::Tensor>, Self::Error>;
}

/// Stable routing metadata supplied by an architecture composition at one
/// canonical unit boundary.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct RoutedObservationPoint {
    path: String,
    expert_count: i32,
}

impl RoutedObservationPoint {
    /// Creates one routed observation point.
    pub fn new(path: impl Into<String>, expert_count: i32) -> Self {
        Self {
            path: path.into(),
            expert_count,
        }
    }

    /// Returns the stable routed-module path.
    pub fn path(&self) -> &str {
        &self.path
    }

    /// Returns the total number of routed experts.
    pub const fn expert_count(&self) -> i32 {
        self.expert_count
    }
}

/// Failure from either canonical expert execution or its observation hook.
#[derive(Debug)]
pub enum ObservedExpertProviderError<P, O> {
    /// The wrapped provider rejected or failed the expert request.
    Provider(P),
    /// The observer rejected the normalized routing event.
    Observer(O),
}

impl<P, O> std::fmt::Display for ObservedExpertProviderError<P, O>
where
    P: std::fmt::Display,
    O: std::fmt::Display,
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Provider(error) => write!(formatter, "routed expert provider failed: {error}"),
            Self::Observer(error) => write!(formatter, "routed expert observer failed: {error}"),
        }
    }
}

impl<P, O> std::error::Error for ObservedExpertProviderError<P, O>
where
    P: std::error::Error + 'static,
    O: std::error::Error + 'static,
{
}

/// Decorates a routed provider with normalized routing observation.
///
/// The decorator sees the exact request and output of canonical provider
/// execution. It therefore adds observation without reimplementing a model
/// family's block, routing, shape, or residency lifecycle. Tensor-parallel
/// requests are delegated without an event because their provider result may
/// still require an architecture-owned reduction before it is observable.
pub struct ObservedExpertProvider<'a, P, O: ?Sized, E> {
    provider: &'a mut P,
    observer: &'a mut O,
    point: RoutedObservationPoint,
    error: std::marker::PhantomData<fn() -> E>,
}

impl<'a, P, O: ?Sized, E> ObservedExpertProvider<'a, P, O, E> {
    /// Wraps `provider` for one canonical routed module invocation.
    pub fn new(provider: &'a mut P, observer: &'a mut O, point: RoutedObservationPoint) -> Self {
        Self {
            provider,
            observer,
            point,
            error: std::marker::PhantomData,
        }
    }

    fn observe<T, ObservationError>(
        &mut self,
        routes: &eredu_nn::GroupSelection<T>,
        output: &T,
    ) -> Result<(), ObservationError>
    where
        O: ActivationObserver<T, ObservationError>,
    {
        self.observer.observe_routing(RoutingObservation {
            path: self.point.path(),
            selected_experts: routes.group_indices(),
            selected_scores: routes.selected_scores(),
            coefficients: routes.coefficients(),
            routed_output: output,
            local_routed_output: None,
            reduced_routed_output: None,
            shared_output: None,
            combined_output: None,
            expert_count: self.point.expert_count(),
        })
    }
}

impl<B, P, O, E> RoutedExpertProvider<B> for ObservedExpertProvider<'_, P, O, E>
where
    B: GroupedNeuralBackend,
    P: RoutedExpertProvider<B>,
    O: ActivationObserver<B::Tensor, E> + ?Sized,
{
    type Error = ObservedExpertProviderError<P::Error, E>;

    fn forward_grouped(
        &mut self,
        resident_bank: &mut B::GatedProductGroups,
        request: RoutedExpertRequest<'_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        let routes = request.routes;
        let output = self
            .provider
            .forward_grouped(resident_bank, request, context)
            .map_err(ObservedExpertProviderError::Provider)?;
        self.observe(routes, &output)
            .map_err(ObservedExpertProviderError::Observer)?;
        Ok(output)
    }

    fn forward_relu2_routed(
        &mut self,
        resident_bank: &mut B::Relu2Groups,
        request: RoutedExpertRequest<'_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        let routes = request.routes;
        let output = self
            .provider
            .forward_relu2_routed(resident_bank, request, context)
            .map_err(ObservedExpertProviderError::Provider)?;
        self.observe(routes, &output)
            .map_err(ObservedExpertProviderError::Observer)?;
        Ok(output)
    }
}

impl<B, P, O, E> TensorParallelRoutedExpertProvider<B> for ObservedExpertProvider<'_, P, O, E>
where
    B: GroupedNeuralBackend,
    P: TensorParallelRoutedExpertProvider<B>,
    O: ActivationObserver<B::Tensor, E> + ?Sized,
{
    fn forward_grouped_tensor_parallel(
        &mut self,
        resident_bank: &mut B::GatedProductGroups,
        request: RoutedExpertRequest<'_, B::Tensor>,
        partitions: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<RoutedExpertTensorParallelOutput<B::Tensor>, Self::Error> {
        self.provider
            .forward_grouped_tensor_parallel(resident_bank, request, partitions, context)
            .map_err(ObservedExpertProviderError::Provider)
    }

    fn forward_relu2_routed_tensor_parallel(
        &mut self,
        resident_bank: &mut B::Relu2Groups,
        request: RoutedExpertRequest<'_, B::Tensor>,
        partitions: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<RoutedExpertTensorParallelOutput<B::Tensor>, Self::Error> {
        self.provider
            .forward_relu2_routed_tensor_parallel(resident_bank, request, partitions, context)
            .map_err(ObservedExpertProviderError::Provider)
    }
}

/// Provider for a fully resident expert bank.
#[derive(Debug, Default, Clone, Copy)]
pub struct ResidentExpertProvider;

impl<B> RoutedExpertProvider<B> for ResidentExpertProvider
where
    B: GroupedNeuralBackend,
{
    type Error = eredu_nn::Error;

    fn forward_grouped(
        &mut self,
        resident_bank: &mut B::GatedProductGroups,
        request: RoutedExpertRequest<'_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        resident_bank.forward_grouped(request.input, request.routes, context)
    }

    fn forward_relu2_routed(
        &mut self,
        resident_bank: &mut B::Relu2Groups,
        request: RoutedExpertRequest<'_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        resident_bank.forward_grouped(request.input, request.routes, context)
    }
}

impl<B> TensorParallelRoutedExpertProvider<B> for ResidentExpertProvider
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend,
{
    fn forward_grouped_tensor_parallel(
        &mut self,
        resident_bank: &mut B::GatedProductGroups,
        request: RoutedExpertRequest<'_, B::Tensor>,
        partitions: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<RoutedExpertTensorParallelOutput<B::Tensor>, Self::Error> {
        B::gated_product_groups_tensor_parallel(
            resident_bank,
            request.input,
            request.routes,
            partitions,
            context,
        )
        .map(RoutedExpertTensorParallelOutput::Partial)
    }

    fn forward_relu2_routed_tensor_parallel(
        &mut self,
        resident_bank: &mut B::Relu2Groups,
        request: RoutedExpertRequest<'_, B::Tensor>,
        partitions: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<RoutedExpertTensorParallelOutput<B::Tensor>, Self::Error> {
        B::relu2_groups_tensor_parallel(
            resident_bank,
            request.input,
            request.routes,
            partitions,
            context,
        )
        .map(RoutedExpertTensorParallelOutput::Partial)
    }
}
