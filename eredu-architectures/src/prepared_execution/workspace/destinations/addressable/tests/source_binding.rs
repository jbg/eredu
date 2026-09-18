//! Descriptive bank mechanisms exercise the real checked provider constructor.
//! Payload methods must never run while completing or quoting a physical source.
use super::*;
use eredu_nn::GroupedNeuralBackend;
use eredu_runtime::{AddressableGroupedBank, IndexedMovement, ParameterBankKey};

pub(super) type Bytes = BTreeMap<ParameterBankKey, u64>;
pub(super) struct Bank(Bytes);
pub(super) struct Movement;

impl AddressableGroupedBank<WorkspaceBackend> for Bank {
    type Acquisition = ();
    type Report = ();
    type Error = Error;
    fn member_bytes(&self, key: ParameterBankKey) -> Option<u64> {
        self.0.get(&key).copied()
    }
    fn acquire(
        &mut self,
        _: eredu_runtime::ParameterBankAcquisition<'_>,
        _: &WorkspaceContext,
    ) -> Result<(), Error> {
        panic!("source completion must not acquire payloads")
    }
    fn gated_product_groups(
        &mut self,
        _: &(),
        _: &eredu_nn::GroupedGatedProductSpec,
        _: &WorkspaceContext,
    ) -> Result<<WorkspaceBackend as GroupedNeuralBackend>::GatedProductGroups, Error> {
        panic!("source completion must not construct numerical groups")
    }
    fn linear_groups(
        &mut self,
        _: &(),
        _: &eredu_nn::GroupedLinearSpec,
        _: &WorkspaceContext,
    ) -> Result<<WorkspaceBackend as GroupedNeuralBackend>::LinearGroups, Error> {
        panic!("source completion must not construct numerical groups")
    }
    fn relu2_groups(
        &mut self,
        _: &(),
        _: &eredu_nn::GroupedRelu2Spec,
        _: &WorkspaceContext,
    ) -> Result<<WorkspaceBackend as GroupedNeuralBackend>::Relu2Groups, Error> {
        panic!("source completion must not construct numerical groups")
    }
    fn complete(&mut self, _: (), _: &WorkspaceTensor, _: &WorkspaceContext) -> Result<(), Error> {
        panic!("source completion must not complete numerical work")
    }
    fn report(&self) -> Result<(), Error> {
        Ok(())
    }
}
impl IndexedMovement<WorkspaceBackend> for Movement {
    type Error = Error;
    fn index_demands(
        &mut self,
        _: &WorkspaceTensor,
        _: usize,
        _: &WorkspaceContext,
    ) -> Result<Vec<(usize, u64)>, Error> {
        panic!("source completion must not inspect route values")
    }
    fn remap_indices(
        &mut self,
        _: &WorkspaceTensor,
        _: &[(usize, usize)],
        _: &WorkspaceContext,
    ) -> Result<WorkspaceTensor, Error> {
        panic!("source completion must not remap routes")
    }
    fn select_rows(
        &mut self,
        _: &WorkspaceTensor,
        _: usize,
        _: usize,
        _: &WorkspaceContext,
    ) -> Result<WorkspaceTensor, Error> {
        panic!("source completion must not select rows")
    }
    fn concatenate_rows(
        &mut self,
        _: &[WorkspaceTensor],
        _: &WorkspaceContext,
    ) -> Result<WorkspaceTensor, Error> {
        panic!("source completion must not concatenate rows")
    }
}

/// The mock mechanism reports a distinct local physical size, as the real TP
/// adapter does after recipe sharding. The public checked builder authenticates
/// these reports against the exact selected local identities.
pub(super) fn physical_bytes(
    banks: &BTreeMap<eredu_runtime::RoutedBankId, SelectedRoutedBank>,
) -> Bytes {
    banks
        .values()
        .flat_map(|bank| bank.addressable_members())
        .map(|member| {
            let bytes = member.selected_bytes() / 2;
            assert!(bytes > 0 && bytes != member.selected_bytes());
            (member.key(), bytes)
        })
        .collect()
}

pub(super) fn bind(
    banks: &BTreeMap<eredu_runtime::RoutedBankId, SelectedRoutedBank>,
    options: eredu_runtime::ParameterBankLoadOptions,
    declared: &Bytes,
    actual: &Bytes,
) -> Result<(), crate::prepared_execution::PreparedExecutionError<Error>> {
    let selection = crate::prepared_execution::PreparedPartitionBanks::prepare(
        eredu_runtime::ParameterBankResidency::IndependentCache(options),
        banks.clone(),
        true,
    )
    .unwrap();
    let (providers, retained) = crate::prepared_execution::construct_partition_bank_providers::<
        WorkspaceBackend,
        Bank,
        Movement,
        (),
        _,
        Error,
    >(&selection, |selection, _| {
        Ok(selection
            .banks()
            .iter()
            .filter(|(_, bank)| !bank.addressable_members().is_empty())
            .map(|(id, _)| {
                let declared = declared
                    .iter()
                    .filter(|(key, _)| key.bank() == id.value() as usize)
                    .map(|(key, bytes)| (*key, *bytes))
                    .collect();
                let actual = actual
                    .iter()
                    .filter(|(key, _)| key.bank() == id.value() as usize)
                    .map(|(key, bytes)| (*key, *bytes))
                    .collect();
                (
                    *id,
                    crate::prepared_execution::PartitionBankMechanisms::new(
                        declared,
                        Bank(actual),
                        Movement,
                        (),
                    ),
                )
            })
            .collect())
    })?;
    assert_eq!(
        retained.len(),
        banks
            .values()
            .filter(|bank| !bank.addressable_members().is_empty())
            .count()
    );
    drop((providers, retained));
    Ok(())
}

pub(super) fn refused(
    result: Result<(), crate::prepared_execution::PreparedExecutionError<Error>>,
) {
    assert!(
        matches!(
            result,
            Err(crate::prepared_execution::PreparedExecutionError::Architecture(_))
        ),
        "physical source mismatch must remain a checked construction refusal: {result:?}"
    );
}
