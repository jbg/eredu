//! Parameter work over retained members, independent of routing and cache eviction.
use super::*;
use eredu_runtime::parameter_operations::{
    ParameterSlotOperation, PreparedBankParameter, PreparedParameterLocation,
};
use std::collections::BTreeSet;
mod prepared;
mod publication;
mod compact_rows;
pub(crate) use compact_rows::{compact_rows_control_bytes, fill_compact_rows, CompactRowsError};
pub(crate) use publication::with_bank_parameter_publication;
#[cfg(all(
    test,
    target_vendor = "apple",
    feature = "metal",
    not(feature = "cuda")
))]
mod prepared_tests;

impl SharedAddressableParameterBank {
    pub(crate) fn owns_parameter_bank(&self, bank: usize) -> bool {
        self.scope.is_none_or(|scope| scope == bank)
    }

    /// The exact borrowed preparation admits selected source loans and numerical
    /// copies before construction; no stream-only fallback grants this access.
    pub(crate) fn with_parameter_slots(
        &self,
        bank: usize,
        unit: usize,
        parameters: &[PreparedBankParameter],
        _selected: &BTreeSet<String>,
        operation: &mut ParameterSlotOperation<'_, MlxTensor, Error>,
        stream: &Stream,
        preparation: Option<
            &crate::backend::runtime::execution::generic::MlxParameterPreparation<'_>,
        >,
    ) -> Result<bool, Error> {
        if !self.owns_parameter_bank(bank) {
            return Ok(false);
        }
        let preparation = preparation.ok_or(Error::PrefillControl(
            eredu_runtime::working_memory::WorkingMemoryError::UnknownBound,
        ))?;
        self.with_prepared_parameter_slots(bank, unit, parameters, operation, stream, preparation)
    }
}

impl AddressableParameterBank {
    pub(super) fn compact_parameter_binding(
        &self,
        acquisition: &AcquiredParameterGroups,
        binding: &str,
        stream: &Stream,
    ) -> Result<Array, AddressableParameterBankError> {
        let mut values = Vec::with_capacity(acquisition.identities.len());
        for (key, lease) in acquisition
            .identities
            .iter()
            .zip(acquisition.transfer.leases())
        {
            let replacement = self
                .parameter_members
                .iter()
                .find(|member| member.key == *key && member.binding == binding)
                .and_then(|member| {
                    self.parameter_replacements
                        .get(&member.parameter)
                        .map(|value| (member, value))
                });
            let value = if let Some((member, value)) = replacement {
                let row = self
                    .parameter_members
                    .iter()
                    .filter(|candidate| {
                        candidate.parameter == member.parameter && candidate.key < *key
                    })
                    .count() as i32;
                value
                    .as_array()
                    .try_index_device((row..row + 1, ..), stream)?
            } else {
                lease.device_value(binding)?.clone()
            };
            values.push(value);
        }
        if values.is_empty() {
            return Err(AddressableParameterBankError::EmptyCompactBinding {
                name: binding.into(),
            });
        }
        Ok(concatenate_axis(&values, 0, stream)?)
    }
}
