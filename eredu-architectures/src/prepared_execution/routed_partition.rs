//! Rank-local provider selection from retained architecture plans.

use std::collections::BTreeMap;

use eredu_runtime::{
    AddressableGroupedBank, IndexedMovement, ParameterBankKey, ParameterBankLoadOptions,
    ParameterBankResidency,
};

use super::PreparedExecutionError;
use crate::{
    partitioned_execution::PreparedRoutedPartitionedArchitecture,
    routed_text::EmptyPartitionRoutedExpertProvider,
};

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
pub type PartitionBankProviders<B> = eredu_runtime::RoutedBankProviders<
    Box<
        dyn eredu_runtime::TensorParallelRoutedExpertProvider<
            B,
            Error = crate::routed_text::RoutedTextExecutionError,
        >,
    >,
>;

/// Constructs every bank under one selected residency budget, including banks
/// with no local work. Backends bind the entire member collection once.
pub fn construct_selected_partition_providers<
    B,
    A,
    G,
    W,
    Bank,
    Movement,
    Retained,
    C,
    MakeBanks,
    Finish,
    O,
    E,
>(
    prepared: PreparedRoutedPartitionedArchitecture<B, A, G, W>,
    make_banks: MakeBanks,
    native: C,
    finish: Finish,
) -> Result<O, PreparedExecutionError<E>>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend + 'static,
    Bank: AddressableGroupedBank<B> + 'static,
    Bank::Error: std::fmt::Display,
    Movement: IndexedMovement<B> + 'static,
    Movement::Error: std::fmt::Display,
    MakeBanks: FnOnce(
        &PreparedRoutedPartitionedArchitecture<B, A, G, W>,
        ParameterBankLoadOptions,
    ) -> Result<
        BTreeMap<eredu_runtime::RoutedBankId, PartitionBankMechanisms<Bank, Movement, Retained>>,
        E,
    >,
    Finish: FnOnce(
        C,
        PreparedRoutedPartitionedArchitecture<B, A, G, W>,
        PartitionBankProviders<B>,
        BTreeMap<eredu_runtime::RoutedBankId, Retained>,
    ) -> Result<O, E>,
{
    let resident_source = match prepared.bank_residency() {
        ParameterBankResidency::WithLayer => Some(prepared.resident_partition_source()
            .map_err(|cause| PreparedExecutionError::Architecture(cause.to_string()))?.clone()),
        _ => None,
    };
    let selection = PreparedPartitionBanks::new(
        prepared.bank_residency(),
        prepared.banks().clone(),
        prepared.prepared().selected().expert_group().is_some(),
        resident_source,
    );
    let (providers, retained) = construct_partition_bank_providers(&selection, |_, options| {
        make_banks(&prepared, options)
    })?;
    finish(native, prepared, providers, retained).map_err(PreparedExecutionError::Backend)
}

/// Exact bank residency, sources and ownership retained by a partition.
/// Construction is architecture-owned; native binders supply only cache mechanisms.
#[derive(Clone)]
pub struct PreparedPartitionBanks {
    residency: ParameterBankResidency,
    banks: BTreeMap<eredu_runtime::RoutedBankId, crate::routed_text::SelectedRoutedBank>,
    expert_exchange: bool,
    resident_source: Option<crate::routed_text::RetainedPartitionResidentSource>,
}
impl PreparedPartitionBanks {
    pub(crate) fn new(
        residency: ParameterBankResidency,
        banks: BTreeMap<eredu_runtime::RoutedBankId, crate::routed_text::SelectedRoutedBank>,
        expert_exchange: bool,
        resident_source: Option<crate::routed_text::RetainedPartitionResidentSource>,
    ) -> Self {
        Self {
            residency,
            banks,
            expert_exchange,
            resident_source,
        }
    }

    pub(crate) fn prepare(
        residency: ParameterBankResidency,
        banks: BTreeMap<eredu_runtime::RoutedBankId, crate::routed_text::SelectedRoutedBank>,
        expert_exchange: bool,
    ) -> Result<Self, crate::RoutedTextExecutionError> {
        let source = if matches!(residency, ParameterBankResidency::WithLayer) {
            Some(crate::routed_text::RetainedPartitionResidentSource::prepare(&banks, expert_exchange)?)
        } else { None };
        Ok(Self::new(residency, banks, expert_exchange, source))
    }

    pub(crate) fn resident_source(&self) -> Result<&crate::routed_text::RetainedPartitionResidentSource, crate::RoutedTextExecutionError> {
        self.resident_source.as_ref().ok_or(crate::RoutedTextExecutionError::ResidentSourceUnavailable)
    }

    pub(crate) fn adopt_execution_sources(&mut self,
        plans: &BTreeMap<eredu_runtime::RoutedBankId, crate::routed_text::RoutedGroupedPlan>) -> Result<(), String> {
        if self.banks.keys().ne(plans.keys()) || self.banks.iter().any(|(id, bank)| bank.plan() != &plans[id]) {
            return Err("retained composite bank execution differs".into());
        }
        for (id, bank) in &mut self.banks { bank.adopt_execution_source(&plans[id])?; }
        Ok(())
    }

    /// The selected immutable-weight cache policy.
    pub const fn residency(&self) -> ParameterBankResidency {
        self.residency
    }
    /// Every local bank, including idle owners with no local members.
    pub fn banks(
        &self,
    ) -> &BTreeMap<eredu_runtime::RoutedBankId, crate::routed_text::SelectedRoutedBank> {
        &self.banks
    }
    /// Exact physical member work admitted for these bank owners.
    pub fn addressable_members(&self) -> Vec<eredu_runtime::AddressableBankMember> {
        self.banks
            .values()
            .flat_map(|bank| bank.addressable_members().iter().cloned())
            .collect()
    }
    /// Parameters excluded from ordinary unit materialization when acquired independently.
    pub fn addressable_logical_targets(&self) -> std::collections::BTreeSet<String> {
        if !matches!(self.residency, ParameterBankResidency::IndependentCache(_)) {
            return Default::default();
        }
        self.banks
            .values()
            .flat_map(|bank| {
                bank.catalog()
                    .logical_targets()
                    .into_iter()
                    .chain(
                        bank.addressable_members()
                            .iter()
                            .flat_map(|member| member.parameters())
                            .flat_map(|parameter| parameter.task().output_companions())
                            .map(|companion| companion.name()),
                    )
                    .map(str::to_owned)
            })
            .collect()
    }
}

/// Realizes retained partition banks through one shared resident/addressable driver.
/// All addressable banks share the single cache constructed by `make_banks`.
pub fn construct_partition_bank_providers<B, Bank, Movement, Retained, MakeBanks, E>(
    selection: &PreparedPartitionBanks,
    make_banks: MakeBanks,
) -> Result<
    (
        PartitionBankProviders<B>,
        BTreeMap<eredu_runtime::RoutedBankId, Retained>,
    ),
    PreparedExecutionError<E>,
>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend + 'static,
    Bank: AddressableGroupedBank<B> + 'static,
    Bank::Error: std::fmt::Display,
    Movement: IndexedMovement<B> + 'static,
    Movement::Error: std::fmt::Display,
    MakeBanks: FnOnce(
        &PreparedPartitionBanks,
        ParameterBankLoadOptions,
    ) -> Result<
        BTreeMap<eredu_runtime::RoutedBankId, PartitionBankMechanisms<Bank, Movement, Retained>>,
        E,
    >,
{
    use crate::routed_text::{PartitionUnitProvider, PlannedAddressableBank};
    let exchange = selection.expert_exchange;
    let options = match selection.residency() {
        ParameterBankResidency::WithLayer => {
            let providers = selection.resident_source()
                .and_then(|source| source.ordinary::<B>())
                .map_err(|cause| PreparedExecutionError::Architecture(cause.to_string()))?;
            return Ok((providers, BTreeMap::new()));
        }
        ParameterBankResidency::IndependentCache(options) => options,
        _ => {
            return Err(PreparedExecutionError::Architecture(
                "selected partition bank residency has no construction mechanism".into(),
            ))
        }
    };
    let mut mechanisms =
        if !selection.addressable_members().is_empty() {
            make_banks(selection, options).map_err(PreparedExecutionError::Backend)?
        } else {
            BTreeMap::new()
        };
    let mut retained = BTreeMap::new();
    let mut providers = BTreeMap::new();
    for (id, bank) in selection.banks() {
        let provider: Box<
            dyn eredu_runtime::TensorParallelRoutedExpertProvider<
                B,
                Error = crate::routed_text::RoutedTextExecutionError,
            >,
        > = if bank.plan().local_global_group_indices().is_empty()
            || bank.addressable_members().is_empty()
        {
            if !bank.addressable_members().is_empty() {
                return Err(PreparedExecutionError::Architecture(
                    "empty partition provider retains nonempty physical members".into()));
            }
            bank.complete_partition_source(BTreeMap::new(), options)
                .map_err(|cause| PreparedExecutionError::Architecture(cause.to_string()))?;
            Box::new(EmptyPartitionRoutedExpertProvider)
        } else {
            let mechanism = mechanisms.remove(id).ok_or_else(|| {
                PreparedExecutionError::Architecture(format!(
                    "missing mechanism for routed bank {id:?}"
                ))
            })?;
            retained.insert(*id, mechanism.retained);
            Box::new(PartitionUnitProvider::new(
                bank,
                PlannedAddressableBank::from_partitioned(
                    bank,
                    exchange,
                    mechanism.member_bytes,
                    mechanism.bank,
                    mechanism.movement,
                    options,
                )
                .map_err(|e| PreparedExecutionError::Architecture(e.to_string()))?,
            ))
        };
        providers.insert(*id, provider);
    }
    if !mechanisms.is_empty() {
        return Err(PreparedExecutionError::Architecture(
            "unexpected partition bank mechanisms".into(),
        ));
    }
    let providers = eredu_runtime::RoutedBankProviders::new(providers)
        .map_err(|e| PreparedExecutionError::Architecture(e.to_string()))?;
    Ok((providers, retained))
}
