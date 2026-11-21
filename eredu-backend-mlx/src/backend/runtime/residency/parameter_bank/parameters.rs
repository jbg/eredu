//! Native binding access and completed copies for admitted parameter-bank transfers.
use super::*;
use crate::backend::runtime::residency::manager::MlxResidencyResources;
use eredu_core::residency::{CompactGroupBindings, NativeBankMechanisms};

impl NativeBankMechanisms for MlxResidencyResources {
    type Backend = MlxNeuralBackend;
    type BankError = Error;
    type Loan = super::movement::MlxParameterLoan;

    fn host_capacity_exhausted(error: &ResidencyError) -> bool {
        matches!(
            error,
            ResidencyError::Ledger(ResidencyLedgerError::BudgetExhausted {
                tier: MemoryTier::Host,
                ..
            })
        )
    }

    fn member(transfer: &ResidentTransfer, index: usize, name: &str) -> Result<MlxTensor, Error> {
        Ok(MlxTensor::from_array(
            transfer.leases()[index].device_value(name)?.clone(),
        ))
    }

    fn replacement(value: &MlxTensor, row: usize, stream: &Stream) -> Result<MlxTensor, Error> {
        let end = row
            .checked_add(1)
            .and_then(|end| i32::try_from(end).ok())
            .ok_or_else(|| {
                Error::ArchitectureModel("replacement row exceeds MLX i32 indexing".into())
            })?;
        Ok(MlxTensor::from_array(
            value
                .as_array()
                .try_index_device(((end - 1)..end, ..), stream)?,
        ))
    }

    fn concatenate(values: &[MlxTensor], stream: &Stream) -> Result<MlxTensor, Error> {
        MlxIndexedMovement.concatenate_rows(values, stream)
    }

    fn begin_loan(transfer: ResidentTransfer) -> Result<Self::Loan, Error> {
        Self::Loan::begin(transfer)
    }

    fn assemble_members(copies: Vec<MlxTensor>, stream: &Stream) -> Result<MlxTensor, Error> {
        let value = concatenate_axis(
            &copies.iter().map(MlxTensor::as_array).collect::<Vec<_>>(),
            0,
            stream,
        )?;
        value.evaluated()?;
        Ok(MlxTensor::from_array(value))
    }
}

macro_rules! compact_bindings {
    ($($group:ident),+) => {$(
        impl CompactGroupBindings<MlxTensor> for <MlxNeuralBackend as GroupedNeuralBackend>::$group {
            fn binding_names(&self) -> Vec<String> {
                self.local_parameter_names()
            }

            fn bind_compact(&mut self, values: BTreeMap<String, MlxTensor>) -> Result<(), eredu_nn::Error> {
                self.bind_local_parameters(values.into_iter().map(|(name, tensor)| (name, tensor.as_array().clone())).collect())
            }
        }
    )+};
}

compact_bindings!(GatedProductGroups, LinearGroups, Relu2Groups);
