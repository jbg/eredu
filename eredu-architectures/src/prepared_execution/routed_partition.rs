//! Projects retained routed architecture selections onto neutral provider plans.

use std::collections::BTreeMap;

use eredu_nn::tensor_movement::IndexedMovement;
use eredu_runtime::{
    executable_binding::routed_partition::{PartitionBankPlan, PartitionUnitProvider},
    AddressableBankMember, AddressableGroupedBank, ParameterBankKey, ParameterBankLoadOptions,
    TensorParallelParameterProvider,
};

use crate::routed_text::{
    PlannedAddressableBank, PlannedResidentBank, RoutedTextExecutionError, SelectedRoutedBank,
};

pub use eredu_runtime::executable_binding::routed_partition::{
    construct_partition_bank_providers, PartitionBankMechanisms, PartitionBankProviders,
};

/// Exact bank residency, sources and ownership retained by this partition.
pub type PreparedPartitionBanks =
    eredu_runtime::executable_binding::routed_partition::PreparedPartitionBanks<SelectedRoutedBank>;

impl PartitionBankPlan for SelectedRoutedBank {
    fn addressable_members(&self) -> &[AddressableBankMember] {
        self.addressable_members()
    }

    fn addressable_logical_targets(&self) -> impl Iterator<Item = &str> {
        self.catalog().logical_targets().into_iter().chain(
            self.addressable_members()
                .iter()
                .flat_map(|member| member.parameters())
                .flat_map(|parameter| parameter.task().output_companions())
                .map(|companion| companion.name()),
        )
    }

    fn has_local_members(&self) -> bool {
        !self.plan().local_global_group_indices().is_empty()
    }

    fn resident<B: eredu_nn::TensorParallelGroupedNeuralBackend + 'static>(
        &self,
        expert_exchange: bool,
    ) -> Result<
        Box<dyn TensorParallelParameterProvider<B, Error = RoutedTextExecutionError>>,
        RoutedTextExecutionError,
    > {
        let exchange = expert_exchange && self.plan().expert_parallel_size() > 1;
        Ok(Box::new(PartitionUnitProvider::new(
            self.partition_unit_coordinates().clone(),
            PlannedResidentBank::from_partitioned(self, exchange)?,
        )))
    }

    fn addressable<B, Bank, Movement>(
        &self,
        expert_exchange: bool,
        member_bytes: BTreeMap<ParameterBankKey, u64>,
        bank: Bank,
        movement: Movement,
        options: ParameterBankLoadOptions,
    ) -> Result<
        Box<dyn TensorParallelParameterProvider<B, Error = RoutedTextExecutionError>>,
        RoutedTextExecutionError,
    >
    where
        B: eredu_nn::TensorParallelGroupedNeuralBackend + 'static,
        Bank: AddressableGroupedBank<B> + 'static,
        Bank::Error: std::fmt::Display,
        Movement: IndexedMovement<B> + 'static,
        Movement::Error: std::fmt::Display,
    {
        let exchange = expert_exchange && self.plan().expert_parallel_size() > 1;
        Ok(Box::new(PartitionUnitProvider::new(
            self.partition_unit_coordinates().clone(),
            PlannedAddressableBank::from_partitioned(
                self,
                exchange,
                member_bytes,
                bank,
                movement,
                options,
            )?,
        )))
    }
}
