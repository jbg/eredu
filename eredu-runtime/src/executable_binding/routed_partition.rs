//! Construction of rank-local banks under a single retained residency budget.

use std::collections::BTreeMap;

use eredu_nn::{GroupedNeuralBackend, Tensor};

use crate::{
    AddressableGroupedBank, ParameterBankKey, ParameterBankLoadOptions, ParameterBankResidency,
    ParameterProvider, RoutedExpertRequest,
};
use eredu_nn::tensor_movement::IndexedMovement;

use super::PreparedExecutionError;

mod partition_units;
pub use partition_units::PartitionUnitProvider;

/// Exact rank-local storage inputs for a selected routed partition.
pub struct PartitionBankBinding<'a> {
    /// Physical members admitted to the shared acquisition pool.
    pub members: Vec<crate::AddressableBankMember>,
    /// Logical scopes with independently materialized members.
    pub scopes: Vec<crate::RoutedBankId>,
    /// Physical placement covering the selected provider, including auxiliary owners.
    pub layout: &'a crate::LocalModelLayout,
    /// Row acquisition policy, present only on the selected tensor owner.
    pub acquiring_rows: Option<&'a crate::SelectedRowLookups>,
}

impl<'a> PartitionBankBinding<'a> {
    /// Projects only banks with work for the native cache.
    fn from_banks<P: PartitionBankPlan>(
        banks: &BTreeMap<crate::RoutedBankId, P>,
        layout: &'a crate::LocalModelLayout,
        acquiring_rows: Option<&'a crate::SelectedRowLookups>,
    ) -> Self {
        let (scopes, members) = banks
            .iter()
            .filter(|(_, bank)| !bank.addressable_members().is_empty())
            .fold(
                (Vec::new(), Vec::new()),
                |(mut scopes, mut members), (id, bank)| {
                    scopes.push(*id);
                    members.extend_from_slice(bank.addressable_members());
                    (scopes, members)
                },
            );
        Self {
            members,
            scopes,
            layout,
            acquiring_rows,
        }
    }
}

/// Invalid routed plan or failure from a generic execution mechanism.
#[derive(Debug, Clone, thiserror::Error)]
#[non_exhaustive]
pub enum RoutedTextExecutionError {
    /// Architecture plan, catalog, unit address, or grouped geometry disagreed.
    #[error("invalid routed text execution contract: {0}")]
    Contract(String),
    /// Indexed movement, bank acquisition, grouped construction, or completion failed.
    #[error("routed text execution mechanism failed: {0}")]
    Mechanism(String),
    /// Original compute, storage, movement or completion failure.
    #[error("routed text execution mechanism failed: {0}")]
    Source(#[source] eredu_nn::Error),
}

impl From<String> for RoutedTextExecutionError {
    fn from(message: String) -> Self {
        Self::Contract(message)
    }
}
impl From<&str> for RoutedTextExecutionError {
    fn from(message: &str) -> Self {
        Self::Contract(message.into())
    }
}

impl RoutedTextExecutionError {
    /// Retains a mechanism failure without erasing its typed cause.
    pub fn from_error(error: impl std::error::Error + Send + Sync + 'static) -> Self {
        Self::Source(eredu_nn::Error::backend_source(error))
    }
}

/// Provider installed on a pipeline rank whose exact local routed catalog is empty.
#[derive(Debug, Clone, Copy, Default)]
pub struct EmptyPartitionParameterProvider;

impl<B> ParameterProvider<B> for EmptyPartitionParameterProvider
where
    B: GroupedNeuralBackend,
{
    type Error = RoutedTextExecutionError;

    fn forward_grouped(
        &mut self,
        _: &mut B::GatedProductGroups,
        _: RoutedExpertRequest<'_, '_, B::Tensor>,
        _: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        Err(RoutedTextExecutionError::Contract(
            "pipeline rank with no routed units received gated-product work".into(),
        ))
    }

    /// Executes an activated selected-linear bank with owned output rows.
    fn forward_linear_routed(
        &mut self,
        _: &mut B::LinearGroups,
        _: RoutedExpertRequest<'_, '_, B::Tensor>,
        _: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        Err(RoutedTextExecutionError::Contract(
            "selected provider equation is not a linear bank".into(),
        ))
    }

    fn forward_relu2_routed(
        &mut self,
        _: &mut B::Relu2Groups,
        _: RoutedExpertRequest<'_, '_, B::Tensor>,
        _: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        Err(RoutedTextExecutionError::Contract(
            "pipeline rank with no routed units received ReLU-squared work".into(),
        ))
    }
}

impl<B> crate::TensorParallelParameterProvider<B> for EmptyPartitionParameterProvider
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend,
{
    fn forward_grouped_tensor_parallel(
        &mut self,
        _: &mut B::GatedProductGroups,
        _: RoutedExpertRequest<'_, '_, B::Tensor>,
        _: usize,
        _: &<B::Tensor as Tensor>::Context,
    ) -> Result<crate::RoutedExpertTensorParallelOutput<B::Tensor>, Self::Error> {
        Err(RoutedTextExecutionError::Contract(
            "pipeline rank with no routed units received tensor-parallel gated-product work".into(),
        ))
    }

    fn forward_relu2_routed_tensor_parallel(
        &mut self,
        _: &mut B::Relu2Groups,
        _: RoutedExpertRequest<'_, '_, B::Tensor>,
        _: usize,
        _: &<B::Tensor as Tensor>::Context,
    ) -> Result<crate::RoutedExpertTensorParallelOutput<B::Tensor>, Self::Error> {
        Err(RoutedTextExecutionError::Contract(
            "pipeline rank with no routed units received tensor-parallel ReLU-squared work".into(),
        ))
    }
}

/// Native bank mechanisms bound to exact rank-local member byte geometry.
pub struct PartitionBankMechanisms<Bank, Movement, Retained> {
    member_bytes: BTreeMap<ParameterBankKey, u64>,
    bank: Bank,
    movement: Movement,
    retained: Retained,
}

impl<Bank, Movement, Retained> PartitionBankMechanisms<Bank, Movement, Retained> {
    /// Pairs native bank mechanisms with the adapter's optional telemetry handle.
    ///
    /// The neutral provider validates these physical byte facts against its
    /// retained local expert plan before the execution binder receives them.
    pub fn new(
        member_bytes: BTreeMap<ParameterBankKey, u64>,
        bank: Bank,
        movement: Movement,
        retained: Retained,
    ) -> Self {
        Self {
            member_bytes,
            bank,
            movement,
            retained,
        }
    }
}

/// Providers whose equations and placement were selected before backend binding.
pub type PartitionBankProviders<B> = crate::RoutedBankProviders<
    Box<dyn crate::TensorParallelParameterProvider<B, Error = RoutedTextExecutionError>>,
>;

/// Projects an exact prepared execution onto its independently bound banks.
pub trait PartitionBankSelection {
    /// The checked bank plan retained for each local provider.
    type Bank: PartitionBankPlan;

    /// Returns the complete admitted rank-local bank collection.
    fn partition_bank_selection(&self) -> &PreparedPartitionBanks<Self::Bank>;
}

/// Portable bank plans capable of binding resident or addressable providers.
pub trait PartitionBankPlan {
    /// The exact physical member work assigned to this bank.
    fn addressable_members(&self) -> &[crate::AddressableBankMember];

    /// Names owned by independently materialized bank members and companions.
    fn addressable_logical_targets(&self) -> impl Iterator<Item = &str>;

    /// Whether the retained placement assigns any members to this rank.
    fn has_local_members(&self) -> bool;

    /// Constructs one resident provider for this checked bank.
    fn resident<B: eredu_nn::TensorParallelGroupedNeuralBackend + 'static>(
        &self,
        expert_exchange: bool,
    ) -> Result<
        Box<dyn crate::TensorParallelParameterProvider<B, Error = RoutedTextExecutionError>>,
        RoutedTextExecutionError,
    >;

    /// Constructs one addressable provider from checked geometry and native mechanisms.
    fn addressable<B, Bank, Movement>(
        &self,
        expert_exchange: bool,
        member_bytes: BTreeMap<ParameterBankKey, u64>,
        bank: Bank,
        movement: Movement,
        options: ParameterBankLoadOptions,
    ) -> Result<
        Box<dyn crate::TensorParallelParameterProvider<B, Error = RoutedTextExecutionError>>,
        RoutedTextExecutionError,
    >
    where
        B: eredu_nn::TensorParallelGroupedNeuralBackend + 'static,
        Bank: AddressableGroupedBank<B> + 'static,
        Bank::Error: std::fmt::Display,
        Movement: IndexedMovement<B> + 'static,
        Movement::Error: std::fmt::Display;
}

/// Exact bank residency, sources and ownership retained by a partition.
/// Construction is architecture-owned; native binders supply only cache mechanisms.
#[derive(Clone)]
pub struct PreparedPartitionBanks<P> {
    residency: ParameterBankResidency,
    banks: BTreeMap<crate::RoutedBankId, P>,
    expert_exchange: bool,
}
impl<P: PartitionBankPlan> PreparedPartitionBanks<P> {
    /// Couples the selected residency and expert exchange policy to checked banks.
    pub fn new(
        residency: ParameterBankResidency,
        banks: BTreeMap<crate::RoutedBankId, P>,
        expert_exchange: bool,
    ) -> Self {
        Self {
            residency,
            banks,
            expert_exchange,
        }
    }

    /// Adds independent prediction banks after checking ownership is disjoint.
    pub fn extend_banks(&mut self, banks: BTreeMap<crate::RoutedBankId, P>) -> Result<(), String> {
        if banks.keys().any(|id| self.banks.contains_key(id)) {
            return Err("prediction and target parameter banks overlap".into());
        }
        self.banks.extend(banks);
        Ok(())
    }

    /// The selected immutable-weight cache policy.
    pub const fn residency(&self) -> ParameterBankResidency {
        self.residency
    }
    /// Every local bank, including idle owners with no local members.
    pub fn banks(&self) -> &BTreeMap<crate::RoutedBankId, P> {
        &self.banks
    }
    /// Exact physical member work admitted for these bank owners.
    pub fn addressable_members(&self) -> Vec<crate::AddressableBankMember> {
        self.banks
            .values()
            .flat_map(|bank| bank.addressable_members().iter().cloned())
            .collect()
    }
    /// Projects the storage and row-acquisition inputs needed by native mechanisms.
    pub fn binding<'a>(
        &self,
        layout: &'a crate::LocalModelLayout,
        acquiring_rows: Option<&'a crate::SelectedRowLookups>,
    ) -> PartitionBankBinding<'a> {
        PartitionBankBinding::from_banks(&self.banks, layout, acquiring_rows)
    }

    /// Parameters excluded from ordinary unit materialization when acquired independently.
    pub fn addressable_logical_targets(&self) -> std::collections::BTreeSet<String> {
        if !matches!(self.residency, ParameterBankResidency::IndependentCache(_)) {
            return Default::default();
        }
        self.banks
            .values()
            .flat_map(|bank| bank.addressable_logical_targets().map(str::to_owned))
            .collect()
    }
}

/// Realizes retained partition banks through one shared resident/addressable driver.
/// All addressable banks share the single cache constructed by `make_banks`.
pub fn construct_partition_bank_providers<B, P, Bank, Movement, Retained, MakeBanks, E>(
    selection: &PreparedPartitionBanks<P>,
    make_banks: MakeBanks,
) -> Result<
    (
        PartitionBankProviders<B>,
        BTreeMap<crate::RoutedBankId, Retained>,
    ),
    PreparedExecutionError<E>,
>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend + 'static,
    Bank: AddressableGroupedBank<B> + 'static,
    Bank::Error: std::fmt::Display,
    Movement: IndexedMovement<B> + 'static,
    Movement::Error: std::fmt::Display,
    P: PartitionBankPlan,
    MakeBanks: FnOnce(
        &PreparedPartitionBanks<P>,
        ParameterBankLoadOptions,
    ) -> Result<
        BTreeMap<crate::RoutedBankId, PartitionBankMechanisms<Bank, Movement, Retained>>,
        E,
    >,
{
    let options = match selection.residency() {
        ParameterBankResidency::WithLayer => None,
        ParameterBankResidency::IndependentCache(options) => Some(options),
    };
    let mut mechanisms =
        if let Some(options) = options.filter(|_| !selection.addressable_members().is_empty()) {
            make_banks(selection, options).map_err(PreparedExecutionError::Backend)?
        } else {
            BTreeMap::new()
        };
    let mut retained = BTreeMap::new();
    let mut providers = BTreeMap::new();
    for (id, bank) in selection.banks() {
        let provider: Box<
            dyn crate::TensorParallelParameterProvider<B, Error = RoutedTextExecutionError>,
        > = if !bank.has_local_members()
            || (options.is_some() && bank.addressable_members().is_empty())
        {
            Box::new(EmptyPartitionParameterProvider)
        } else if let Some(options) = options {
            let mechanism = mechanisms.remove(id).ok_or_else(|| {
                PreparedExecutionError::Architecture(format!(
                    "missing mechanism for routed bank {id:?}"
                ))
            })?;
            retained.insert(*id, mechanism.retained);
            bank.addressable(
                selection.expert_exchange,
                mechanism.member_bytes,
                mechanism.bank,
                mechanism.movement,
                options,
            )
            .map_err(|error| PreparedExecutionError::Architecture(error.to_string()))?
        } else {
            bank.resident(selection.expert_exchange)
                .map_err(|error| PreparedExecutionError::Architecture(error.to_string()))?
        };
        providers.insert(*id, provider);
    }
    if !mechanisms.is_empty() {
        return Err(PreparedExecutionError::Architecture(
            "unexpected partition bank mechanisms".into(),
        ));
    }
    let providers = crate::RoutedBankProviders::new(providers)
        .map_err(|e| PreparedExecutionError::Architecture(e.to_string()))?;
    Ok((providers, retained))
}

/// Binds the local physical member tasks over one shared native pool.
pub fn bind_partition_addressable_banks<
    Bank,
    Movement,
    Retained,
    Pool,
    E,
    M: eredu_core::checkpoint_mechanisms::CheckpointEncodingMaterializer,
>(
    binding: PartitionBankBinding<'_>,
    source: eredu_checkpoint::store::SharedCheckpointSource,
    options: ParameterBankLoadOptions,
    support: &impl crate::RowLookupMechanismSupport,
    materializer: &M,
    open: impl FnOnce(
        crate::expert::PreparedAddressableBankStorage<M::Parameters>,
    ) -> Result<Pool, PreparedExecutionError<E>>,
    scope: impl Fn(&Pool, crate::RoutedBankId) -> Result<(Bank, Movement, Retained), E>,
) -> Result<
    BTreeMap<crate::RoutedBankId, PartitionBankMechanisms<Bank, Movement, Retained>>,
    PreparedExecutionError<E>,
> {
    use eredu_core::BackendFailure;
    let members = crate::place_addressable_bank_members(&binding.members, &*source, binding.layout)
        .map_err(BackendFailure::from_error)
        .map_err(PreparedExecutionError::Binding)?;
    let rows = match binding.acquiring_rows {
        Some(rows) if rows.options() != options => {
            return Err(PreparedExecutionError::Binding(BackendFailure::from_error(
                crate::RowLookupSelectionError::PoolMismatch,
            )));
        }
        Some(rows) => std::borrow::Cow::Borrowed(rows),
        None => std::borrow::Cow::Owned(
            crate::SelectedRowLookups::select(Default::default(), options, 0, support)
                .map_err(BackendFailure::from_error)
                .map_err(PreparedExecutionError::Binding)?,
        ),
    };
    let prepared = crate::expert::prepare_addressable_bank_members(
        &members,
        source,
        &rows,
        support,
        materializer,
    )
    .map_err(PreparedExecutionError::Binding)?;
    let mut bytes_by_bank = BTreeMap::<usize, BTreeMap<ParameterBankKey, u64>>::new();
    for member in members {
        let key = member.key();
        bytes_by_bank
            .entry(key.bank())
            .or_default()
            .insert(key, member.selected_bytes());
    }
    let pool = open(prepared)?;
    binding
        .scopes
        .into_iter()
        .map(|id| {
            let (bank, movement, retained) =
                scope(&pool, id).map_err(PreparedExecutionError::Backend)?;
            let bytes = bytes_by_bank
                .remove(&(id.value() as usize))
                .unwrap_or_default();
            Ok((
                id,
                PartitionBankMechanisms::new(bytes, bank, movement, retained),
            ))
        })
        .collect()
}

/// Uses the grouped pool or prepares a row-only pool on the admitted owner.
pub fn bind_partition_row_pool<
    Pool: Clone,
    E,
    M: eredu_core::checkpoint_mechanisms::CheckpointEncodingMaterializer,
>(
    rank: usize,
    owner: usize,
    grouped_pool: Option<&Pool>,
    rows: &crate::SelectedRowLookups,
    source: eredu_checkpoint::store::SharedCheckpointSource,
    support: &impl crate::RowLookupMechanismSupport,
    materializer: &M,
    open: impl FnOnce(
        crate::expert::PreparedAddressableBankStorage<M::Parameters>,
    ) -> Result<Pool, PreparedExecutionError<E>>,
) -> Result<Option<Pool>, PreparedExecutionError<E>> {
    if rank != owner {
        return Ok(None);
    }
    if let Some(pool) = grouped_pool {
        return Ok(Some(pool.clone()));
    }
    let prepared =
        crate::expert::prepare_addressable_bank_members(&[], source, rows, support, materializer)
            .map_err(PreparedExecutionError::Binding)?;
    open(prepared).map(Some)
}
