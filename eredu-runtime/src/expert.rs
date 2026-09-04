//! Runtime ownership boundary for routed expert acquisition and residency.

use eredu_nn::{
    DistributedNeuralBackend, GroupSelection, GroupedGatedProductOperator, GroupedNeuralBackend,
    GroupedRelu2Operator, Tensor, TensorParallelGroupedOutput,
};

use crate::ExpertPass;
use crate::{
    observe_and_intervene, ActivationObserver, OffloadUnit, ParameterBankAccess, ParameterBankKey,
    ResidencyDeclarationError, RoutingObservation, WeightBinding,
};

/// Exact generic storage member projected from an architecture-owned bank catalog.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct AddressableBankMember {
    key: ParameterBankKey,
    source: OffloadUnit,
    source_bytes: u64,
    selected_bytes: u64,
}

impl AddressableBankMember {
    /// Validates one atomic source unit and its selected executable byte geometry.
    pub fn new(
        key: ParameterBankKey,
        bindings: impl IntoIterator<Item = WeightBinding>,
        selected_bytes: u64,
    ) -> Result<Self, AddressableBankMemberError> {
        if selected_bytes == 0 {
            return Err(AddressableBankMemberError::ZeroSelectedBytes { key });
        }
        let source = OffloadUnit::new(key.unit_id(), bindings)?;
        let source_bytes = source.bindings().iter().try_fold(0u64, |total, binding| {
            total
                .checked_add(binding.expected_bytes())
                .ok_or(AddressableBankMemberError::SourceByteOverflow { key })
        })?;
        Ok(Self {
            key,
            source,
            source_bytes,
            selected_bytes,
        })
    }

    /// Returns the generic bank key selected by neutral composition.
    pub const fn key(&self) -> ParameterBankKey {
        self.key
    }

    /// Returns exact source bindings without architecture roles or naming policy.
    pub const fn source(&self) -> &OffloadUnit {
        &self.source
    }

    /// Returns admitted source bytes before an optional lowering.
    pub const fn source_bytes(&self) -> u64 {
        self.source_bytes
    }

    /// Returns executable bytes after the selected lowering.
    pub const fn selected_bytes(&self) -> u64 {
        self.selected_bytes
    }
}

/// Invalid generic addressable-bank member projection.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
pub enum AddressableBankMemberError {
    /// Source binding declarations were invalid.
    #[error(transparent)]
    Residency(#[from] ResidencyDeclarationError),
    /// Source binding byte accounting overflowed.
    #[error("addressable bank member {key:?} source byte geometry overflowed")]
    SourceByteOverflow {
        /// Invalid member.
        key: ParameterBankKey,
    },
    /// Selected executable storage was empty.
    #[error("addressable bank member {key:?} selected byte geometry is zero")]
    ZeroSelectedBytes {
        /// Invalid member.
        key: ParameterBankKey,
    },
}

/// Generic indexed tensor movement required by bounded grouped execution.
///
/// Implementations expose integer-index discovery and tensor movement without
/// receiving architecture plans, bank meaning, or text lifecycle policy.
pub trait IndexedMovement<B>
where
    B: GroupedNeuralBackend,
{
    /// Indexed movement failure.
    type Error;

    /// Returns deterministic demand counts for integer indices below `upper_bound`.
    fn index_demands(
        &mut self,
        indices: &B::Tensor,
        upper_bound: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Vec<(usize, u64)>, Self::Error>;

    /// Rewrites source indices through one exact source-to-compact mapping.
    fn remap_indices(
        &mut self,
        indices: &B::Tensor,
        mapping: &[(usize, usize)],
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error>;

    /// Selects a contiguous range along the leading row axis.
    fn select_rows(
        &mut self,
        value: &B::Tensor,
        start: usize,
        end: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error>;

    /// Concatenates row partitions in their original order.
    fn concatenate_rows(
        &mut self,
        values: &[B::Tensor],
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error>;
}

/// Backend-neutral tensor movement needed by an expert-exchange protocol.
///
/// Architecture code supplies already validated row and flattened-route
/// indices. Implementations retain tensor storage and completion ownership;
/// they do not receive expert identities, topology, or model-family policy.
pub trait ExpertRouteTensorMovement<T> {
    /// Tensor movement failure.
    type Error;

    /// Returns the logical tensor shape without materializing its values.
    fn shape(&self, value: &T) -> Vec<usize>;

    /// Duplicates and reorders leading-axis rows in the supplied order.
    fn gather_rows(&mut self, value: &T, rows: &[usize]) -> Result<T, Self::Error>;

    /// Selects flattened route scalars and returns them as `[routes, 1]`.
    fn gather_route_values(
        &mut self,
        value: &T,
        flattened_routes: &[usize],
    ) -> Result<T, Self::Error>;

    /// Additively combines route rows into their architecture source rows.
    ///
    /// Every input row must be consumed exactly once. Repeated destination
    /// rows are intentional and implement weighted routed-expert summation.
    fn scatter_add_rows(
        &mut self,
        value: T,
        destination_rows: &[usize],
        output_rows: usize,
    ) -> Result<T, Self::Error>;
}

/// Opaque variable-count transport used by architecture-owned expert routing.
///
/// Implementations must preserve peer-block and within-block order, validate
/// every tensor against the selected communication requirement, and retain all
/// native resources until the exact completion has finished.
pub trait ExpertRouteExchange<T> {
    /// Communication or metadata transport failure.
    type Error;

    /// Exchanges one tensor whose leading rows match the supplied peer counts.
    fn exchange_tensor(
        &mut self,
        counts: &crate::CommunicationPeerCounts,
        value: T,
    ) -> Result<T, Self::Error>;

    /// Exchanges one unsigned metadata value per leading tensor row.
    fn exchange_indices(
        &mut self,
        counts: &crate::CommunicationPeerCounts,
        values: Vec<usize>,
    ) -> Result<Vec<usize>, Self::Error>;
}

/// Architecture-selected combination for one expert-exchange batch.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
#[non_exhaustive]
pub enum ExpertRouteCombination {
    /// Apply each route coefficient once, then add routes targeting one token.
    CoefficientWeightedSum,
}

/// One owner-local grouped batch submitted after expert exchange.
pub struct AddressableExpertRouteRequest<'a, T> {
    /// Global execution unit containing the addressable expert bank.
    pub unit: usize,
    /// Rows received from every source peer.
    pub input: &'a T,
    /// Checkpoint-global expert identity for every received row.
    ///
    /// Addressable storage keys must be derived from this identity. It is
    /// deliberately kept separate from `owner_local_experts`, whose values
    /// are valid only as indices into the rank-local grouped operator.
    pub global_experts: &'a [usize],
    /// Dense owner-local expert identity for every received row.
    pub owner_local_experts: &'a [usize],
    /// Selected router scores aligned one-for-one with received rows.
    pub selected_scores: &'a T,
    /// Final route coefficients aligned one-for-one with received rows.
    pub coefficients: &'a T,
    /// Prefill or decode execution classification.
    pub pass: ExpertPass,
    /// Storage access classification derived from `pass`.
    pub access: ParameterBankAccess,
    /// Architecture-declared route combination.
    pub combination: ExpertRouteCombination,
}

impl<T> AddressableExpertRouteRequest<'_, T> {
    /// Returns the only valid addressable-bank key for one routed row.
    ///
    /// The owner-local ID is intentionally not accepted here: it addresses the
    /// compact grouped operator, not checkpoint-global storage.
    pub fn addressable_bank_key(&self, row: usize) -> Option<ParameterBankKey> {
        self.global_experts
            .get(row)
            .copied()
            .map(|global| ParameterBankKey::new(self.unit, global))
    }

    /// Returns the rank-local grouped-operator ID for one routed row.
    pub fn owner_local_execution_id(&self, row: usize) -> Option<usize> {
        self.owner_local_experts.get(row).copied()
    }
}

/// Local addressable grouped execution used by expert exchange.
///
/// The provider must consume every submitted row exactly once, select its
/// corresponding owner-local expert, and apply its route coefficient exactly
/// once. Acquired bank resources remain provider-owned until the returned
/// tensor is natively complete.
pub trait AddressableExpertRouteProvider<T> {
    /// Acquisition or grouped execution failure.
    type Error;

    /// Executes one owner-local grouped batch.
    fn execute_addressable_routes(
        &mut self,
        request: AddressableExpertRouteRequest<'_, T>,
    ) -> Result<T, Self::Error>;

    /// Executes one owner-local grouped batch while retaining tensor-parallel
    /// reduction structure.
    ///
    /// Providers without rank-local TP work inherit complete-output behavior.
    /// A TP provider overrides this method and returns its reducible activation
    /// contribution plus the optional selection-weighted post-reduction bias.
    /// The exchange protocol returns both values to their source-token order;
    /// it must not add the bias before the caller's tensor all-sum.
    fn execute_addressable_routes_tensor_parallel(
        &mut self,
        request: AddressableExpertRouteRequest<'_, T>,
    ) -> Result<RoutedExpertTensorParallelOutput<T>, Self::Error> {
        self.execute_addressable_routes(request)
            .map(RoutedExpertTensorParallelOutput::Complete)
    }
}

/// Exact generic request for an independently addressable bank acquisition.
#[derive(Debug, Clone, Copy)]
pub struct ParameterBankAcquisition<'a> {
    entries: &'a [(ParameterBankKey, u64)],
    access: ParameterBankAccess,
}

impl<'a> ParameterBankAcquisition<'a> {
    /// Creates one deterministic acquisition request in compact-bank order.
    pub const fn new(entries: &'a [(ParameterBankKey, u64)], access: ParameterBankAccess) -> Self {
        Self { entries, access }
    }

    /// Returns generic bank keys and duplicate-preserving demand counts.
    pub const fn entries(&self) -> &'a [(ParameterBankKey, u64)] {
        self.entries
    }

    /// Returns the selected generic storage access class.
    pub const fn access(&self) -> ParameterBankAccess {
        self.access
    }
}

/// Generic addressable storage and grouped-operator construction mechanisms.
///
/// The mechanism receives already translated bank keys, compact specifications,
/// and access classes. Architecture identity, routing policy, global identity
/// mapping, chunking, and text-session behavior remain outside this contract.
pub trait AddressableGroupedBank<B>
where
    B: GroupedNeuralBackend,
{
    /// Live native storage retained across grouped execution.
    type Acquisition;
    /// Generic bank telemetry snapshot.
    type Report;
    /// Storage, transfer, lowering, or construction failure.
    type Error;

    /// Returns the selected byte geometry for one admitted bank member.
    fn member_bytes(&self, key: ParameterBankKey) -> Option<u64>;

    /// Acquires exact generic keys in caller-supplied compact order.
    fn acquire(
        &mut self,
        request: ParameterBankAcquisition<'_>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::Acquisition, Self::Error>;

    /// Constructs one compact gated-product operator from acquired bindings.
    fn gated_product_groups(
        &mut self,
        acquisition: &Self::Acquisition,
        spec: &eredu_nn::GroupedGatedProductSpec,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::GatedProductGroups, Self::Error>;

    /// Constructs one compact ReLU-squared operator from acquired bindings.
    fn relu2_groups(
        &mut self,
        acquisition: &Self::Acquisition,
        spec: &eredu_nn::GroupedRelu2Spec,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Relu2Groups, Self::Error>;

    /// Retains acquired storage until the grouped output is natively complete.
    fn complete(
        &mut self,
        acquisition: Self::Acquisition,
        output: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(), Self::Error>;

    /// Returns generic key, byte, tier, acquisition, and eviction telemetry.
    fn report(&self) -> Result<Self::Report, Self::Error>;
}

/// Mechanism-only lookup of one grouped operator in an addressable parameter bank.
pub trait AddressableGatedProductBank<B>
where
    B: GroupedNeuralBackend,
{
    /// Bank lookup or construction failure.
    type Error;

    /// Resolves one generic bank key and exact grouped construction specification.
    fn acquire(
        &mut self,
        key: ParameterBankKey,
        spec: &eredu_nn::GroupedGatedProductSpec,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<&mut B::GatedProductGroups, Self::Error>;
}

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

impl<T> RoutedExpertRequest<'_, T> {
    /// Projects architecture execution semantics into the storage workload
    /// class exposed to backend parameter-bank mechanisms.
    pub const fn parameter_bank_access(&self) -> ParameterBankAccess {
        self.pass.parameter_bank_access()
    }
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
    B: GroupedNeuralBackend + DistributedNeuralBackend,
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
    B: GroupedNeuralBackend + DistributedNeuralBackend,
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

    /// Executes destination-local rows that were already expanded to one
    /// owner-local expert per row by the neutral expert exchange.
    ///
    /// The compact request deliberately has route cardinality one; providers
    /// must not compare it with the architecture's original top-k cardinality.
    fn forward_compact_grouped(
        &mut self,
        resident_bank: &mut B::GatedProductGroups,
        request: RoutedExpertRequest<'_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        self.forward_grouped(resident_bank, request, context)
    }

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

    /// Executes destination-local, one-expert-per-row contributions while
    /// preserving the backend's TP reduction and post-bias structure.
    fn forward_compact_grouped_tensor_parallel(
        &mut self,
        resident_bank: &mut B::GatedProductGroups,
        request: RoutedExpertRequest<'_, B::Tensor>,
        partitions: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<RoutedExpertTensorParallelOutput<B::Tensor>, Self::Error> {
        self.forward_grouped_tensor_parallel(resident_bank, request, partitions, context)
    }

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
    ) -> Result<T, ObservationError>
    where
        T: Clone,
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
        })?;
        observe_and_intervene(
            self.observer,
            &format!("{}.output", self.point.path()),
            output,
        )
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
            .map_err(ObservedExpertProviderError::Observer)
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
            .map_err(ObservedExpertProviderError::Observer)
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
