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
    use crate::routed_text::PlannedAddressableBank;
    let exchange = prepared.prepared().selected().expert_group().is_some();
    let options = match prepared.bank_residency() {
        ParameterBankResidency::WithLayer => None,
        ParameterBankResidency::IndependentCache(options) => Some(options),
        _ => {
            return Err(PreparedExecutionError::Architecture(
                "selected partition bank residency has no construction mechanism".into(),
            ))
        }
    };
    if options.is_none() {
        let providers = prepared
            .resident_partition_providers()
            .map_err(|e| PreparedExecutionError::Architecture(e.to_string()))?;
        return finish(native, prepared, providers, BTreeMap::new())
            .map_err(PreparedExecutionError::Backend);
    }
    let mut mechanisms =
        if let Some(options) = options.filter(|_| !prepared.addressable_members().is_empty()) {
            make_banks(&prepared, options).map_err(PreparedExecutionError::Backend)?
        } else {
            BTreeMap::new()
        };
    let mut retained = BTreeMap::new();
    let mut providers = BTreeMap::new();
    for (id, bank) in prepared.banks() {
        let provider: Box<
            dyn eredu_runtime::TensorParallelRoutedExpertProvider<
                B,
                Error = crate::routed_text::RoutedTextExecutionError,
            >,
        > = if bank.plan().local_global_group_indices().is_empty()
            || (options.is_some() && bank.addressable_members().is_empty())
        {
            Box::new(EmptyPartitionRoutedExpertProvider)
        } else if let Some(options) = options {
            let mechanism = mechanisms.remove(id).ok_or_else(|| {
                PreparedExecutionError::Architecture(format!(
                    "missing mechanism for routed bank {id:?}"
                ))
            })?;
            retained.insert(*id, mechanism.retained);
            Box::new(
                PlannedAddressableBank::from_partitioned(
                    bank,
                    exchange,
                    mechanism.member_bytes,
                    mechanism.bank,
                    mechanism.movement,
                    options,
                )
                .map_err(|e| PreparedExecutionError::Architecture(e.to_string()))?,
            )
        } else {
            unreachable!("resident collection was constructed above")
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
    finish(native, prepared, providers, retained).map_err(PreparedExecutionError::Backend)
}
