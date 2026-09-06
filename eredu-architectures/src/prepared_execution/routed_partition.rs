//! Rank-local provider selection from retained architecture plans.

use std::collections::BTreeMap;

use eredu_nn::GroupedNeuralBackend;
use eredu_runtime::{
    AddressableGroupedBank, IndexedMovement, ParameterBankKey, ParameterBankLoadOptions,
    ParameterBankResidency,
};

use super::PreparedExecutionError;
use crate::{
    partitioned_execution::PreparedRoutedPartitionedArchitecture,
    routed_text::{
        EmptyPartitionRoutedExpertProvider, PlannedAddressableGatedProduct,
        PlannedAddressableRelu2, PlannedResidentGatedProduct, PlannedResidentRelu2,
    },
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

/// Constructs the exact selected gated provider, including empty local expert ownership.
///
/// No bank mechanism is called for a resident selection or an empty addressable
/// partition. Native callbacks receive an already checked typed provider and
/// cannot choose a different residency branch.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn construct_selected_gated_partition_provider<
    B,
    A,
    G,
    W,
    Bank,
    Movement,
    Retained,
    C,
    MakeBank,
    R,
    I,
    Z,
    O,
    E,
>(
    prepared: PreparedRoutedPartitionedArchitecture<B, A, G, W>,
    make_bank: MakeBank,
    native: C,
    finish_resident: R,
    finish_addressable: I,
    finish_empty: Z,
) -> Result<O, PreparedExecutionError<E>>
where
    B: GroupedNeuralBackend,
    Bank: AddressableGroupedBank<B>,
    Bank::Error: std::fmt::Display,
    Movement: IndexedMovement<B>,
    Movement::Error: std::fmt::Display,
    MakeBank: FnOnce(
        &PreparedRoutedPartitionedArchitecture<B, A, G, W>,
        ParameterBankLoadOptions,
    ) -> Result<PartitionBankMechanisms<Bank, Movement, Retained>, E>,
    R: FnOnce(
        C,
        PreparedRoutedPartitionedArchitecture<B, A, G, W>,
        PlannedResidentGatedProduct,
    ) -> Result<O, E>,
    I: FnOnce(
        C,
        PreparedRoutedPartitionedArchitecture<B, A, G, W>,
        PlannedAddressableGatedProduct<B, Bank, Movement>,
        Retained,
    ) -> Result<O, E>,
    Z: FnOnce(
        C,
        PreparedRoutedPartitionedArchitecture<B, A, G, W>,
        EmptyPartitionRoutedExpertProvider,
    ) -> Result<O, E>,
{
    match prepared.bank_residency() {
        ParameterBankResidency::WithLayer => {
            let provider = prepared
                .resident_gated_product_provider()
                .map_err(|error| PreparedExecutionError::Architecture(error.to_string()))?;
            finish_resident(native, prepared, provider).map_err(PreparedExecutionError::Backend)
        }
        ParameterBankResidency::IndependentCache(options) => {
            if prepared.addressable_members().is_empty() {
                return finish_empty(native, prepared, EmptyPartitionRoutedExpertProvider)
                    .map_err(PreparedExecutionError::Backend);
            }
            let mechanisms =
                make_bank(&prepared, options).map_err(PreparedExecutionError::Backend)?;
            let provider = prepared
                .addressable_gated_product_provider(
                    mechanisms.member_bytes,
                    mechanisms.bank,
                    mechanisms.movement,
                    options,
                )
                .map_err(|error| PreparedExecutionError::Architecture(error.to_string()))?;
            finish_addressable(native, prepared, provider, mechanisms.retained)
                .map_err(PreparedExecutionError::Backend)
        }
        _ => Err(PreparedExecutionError::Architecture(
            "selected gated partition bank residency has no construction mechanism".into(),
        )),
    }
}

/// Constructs the exact selected ReLU-squared provider, including empty local ownership.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn construct_selected_relu2_partition_provider<
    B,
    A,
    G,
    W,
    Bank,
    Movement,
    Retained,
    C,
    MakeBank,
    R,
    I,
    Z,
    O,
    E,
>(
    prepared: PreparedRoutedPartitionedArchitecture<B, A, G, W, eredu_nn::GroupedRelu2Spec>,
    make_bank: MakeBank,
    native: C,
    finish_resident: R,
    finish_addressable: I,
    finish_empty: Z,
) -> Result<O, PreparedExecutionError<E>>
where
    B: GroupedNeuralBackend,
    Bank: AddressableGroupedBank<B>,
    Bank::Error: std::fmt::Display,
    Movement: IndexedMovement<B>,
    Movement::Error: std::fmt::Display,
    MakeBank: FnOnce(
        &PreparedRoutedPartitionedArchitecture<B, A, G, W, eredu_nn::GroupedRelu2Spec>,
        ParameterBankLoadOptions,
    ) -> Result<PartitionBankMechanisms<Bank, Movement, Retained>, E>,
    R: FnOnce(
        C,
        PreparedRoutedPartitionedArchitecture<B, A, G, W, eredu_nn::GroupedRelu2Spec>,
        PlannedResidentRelu2,
    ) -> Result<O, E>,
    I: FnOnce(
        C,
        PreparedRoutedPartitionedArchitecture<B, A, G, W, eredu_nn::GroupedRelu2Spec>,
        PlannedAddressableRelu2<B, Bank, Movement>,
        Retained,
    ) -> Result<O, E>,
    Z: FnOnce(
        C,
        PreparedRoutedPartitionedArchitecture<B, A, G, W, eredu_nn::GroupedRelu2Spec>,
        EmptyPartitionRoutedExpertProvider,
    ) -> Result<O, E>,
{
    match prepared.bank_residency() {
        ParameterBankResidency::WithLayer => {
            let provider = prepared
                .resident_relu2_provider()
                .map_err(|error| PreparedExecutionError::Architecture(error.to_string()))?;
            finish_resident(native, prepared, provider).map_err(PreparedExecutionError::Backend)
        }
        ParameterBankResidency::IndependentCache(options) => {
            if prepared.addressable_members().is_empty() {
                return finish_empty(native, prepared, EmptyPartitionRoutedExpertProvider)
                    .map_err(PreparedExecutionError::Backend);
            }
            let mechanisms =
                make_bank(&prepared, options).map_err(PreparedExecutionError::Backend)?;
            let provider = prepared
                .addressable_relu2_provider(
                    mechanisms.member_bytes,
                    mechanisms.bank,
                    mechanisms.movement,
                    options,
                )
                .map_err(|error| PreparedExecutionError::Architecture(error.to_string()))?;
            finish_addressable(native, prepared, provider, mechanisms.retained)
                .map_err(PreparedExecutionError::Backend)
        }
        _ => Err(PreparedExecutionError::Architecture(
            "selected ReLU-squared partition bank residency has no construction mechanism".into(),
        )),
    }
}
