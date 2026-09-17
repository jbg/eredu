//! Parameter work over retained members, independent of routing and cache eviction.
use super::*;
use eredu_runtime::parameter_operations::{
    ParameterSlotOperation, PreparedBankParameter, PreparedParameterLocation,
};
use std::collections::BTreeSet;

impl SharedAddressableParameterBank {
    pub(crate) fn owns_parameter_bank(&self, bank: usize) -> bool {
        self.scope.is_none_or(|scope| scope == bank)
    }

    /// The caller has admitted and charged selected output copies before entry.
    /// Member sources use the existing residency manager and completion-safe loans.
    pub(crate) fn with_parameter_slots(
        &self,
        bank: usize,
        unit: usize,
        parameters: &[PreparedBankParameter],
        selected: &BTreeSet<String>,
        operation: &mut ParameterSlotOperation<'_, MlxTensor, Error>,
        stream: &Stream,
    ) -> Result<bool, Error> {
        if !self.owns_parameter_bank(bank) {
            return Ok(false);
        }
        let location = PreparedParameterLocation::Bank { bank, unit };
        let parameters = selected
            .iter()
            .map(|id| {
                parameters
                    .iter()
                    .find(|parameter| {
                        parameter.slot.parameter.id.as_str() == id
                            && parameter.slot.location == location
                    })
                    .ok_or_else(|| {
                        Error::ArchitectureModel("selected bank parameter owner differs".into())
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let (manager, replacements) = {
            let bank = self.inner.lock().map_err(|_| {
                Error::ArchitectureModel("addressable parameter bank lock was poisoned".into())
            })?;
            let replacements = selected
                .iter()
                .filter_map(|id| {
                    bank.parameter_replacements
                        .get(id)
                        .map(|value| (id.clone(), value.clone()))
                })
                .collect::<BTreeMap<_, _>>();
            (bank.manager.clone(), replacements)
        };
        let mut values = Vec::with_capacity(parameters.len());
        for parameter in parameters {
            if let Some(value) = replacements.get(parameter.slot.parameter.id.as_str()) {
                values.push((parameter.slot.parameter.clone(), value.clone()));
                continue;
            }
            let mut copies = Vec::with_capacity(parameter.members.len());
            for member in &parameter.members {
                let transfer = manager
                    .acquire_many_with_transfer(&[(member.key.unit_id(), 1)], MemoryTier::Device)?;
                let value = crate::backend::runtime::execution::generic::with_parameter_transfer(
                    transfer,
                    stream,
                    |lease| {
                        let value = lease.device_value(&member.binding)?.copy(stream)?;
                        value.evaluated()?;
                        Ok(value)
                    },
                )?;
                copies.push(value);
            }
            let value = concatenate_axis(&copies, 0, stream)?;
            value.evaluated()?;
            values.push((
                parameter.slot.parameter.clone(),
                MlxTensor::from_array(value),
            ));
        }
        operation(&mut |visitor| {
            for (metadata, value) in &mut values {
                visitor.visit_slot(metadata.clone(), value);
            }
        })?;
        Ok(true)
    }
}

/// Publishes ordinary and independent-bank handles only after every pool validates.
/// No native work occurs here. Locks cover the final infallible bank handle swaps;
/// the caller owns model quiescence and the outer all-rank transaction.
pub(crate) fn publish_bank_parameter_replacements(
    banks: &[SharedAddressableParameterBank],
    values: &BTreeMap<String, MlxTensor>,
    active: bool,
    ordinary: impl FnOnce() -> Result<bool, Error>,
) -> Result<bool, Error> {
    let mut pools = Vec::new();
    for bank in banks {
        if !pools.iter().any(|pool| Arc::ptr_eq(pool, &bank.inner)) {
            pools.push(Arc::clone(&bank.inner));
        }
    }
    pools.sort_unstable_by_key(|pool| Arc::as_ptr(pool) as usize);
    let mut guards = pools
        .iter()
        .map(|pool| {
            pool.lock().map_err(|_| {
                Error::ArchitectureModel("addressable parameter bank lock was poisoned".into())
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut prepared = Vec::with_capacity(guards.len());
    for bank in &guards {
        let mut replacements = BTreeMap::new();
        let mut bytes = bank.catalog.clone();
        if active {
            for member in &bank.parameter_members {
                let Some(value) = values.get(&member.parameter) else {
                    continue;
                };
                let count = bank
                    .parameter_members
                    .iter()
                    .filter(|candidate| candidate.parameter == member.parameter)
                    .count();
                if value.as_array().shape().first().copied() != i32::try_from(count).ok()
                    || !matches!(
                        value.as_array().dtype(),
                        Dtype::Float32 | Dtype::Float16 | Dtype::Bfloat16
                    )
                    || value.as_array().nbytes() % count != 0
                {
                    return Err(Error::ArchitectureModel(
                        "bank replacement has incompatible member geometry or dtype".into(),
                    ));
                }
                let entry_bytes = bytes.get_mut(&member.key).expect("retained member");
                *entry_bytes = entry_bytes
                    .checked_sub(member.materialized.byte_len)
                    .and_then(|n| n.checked_add((value.as_array().nbytes() / count) as u64))
                    .ok_or_else(|| {
                        Error::ArchitectureModel("bank replacement byte count overflow".into())
                    })?;
                replacements.insert(member.parameter.clone(), value.clone());
            }
        }
        let revision = bank.parameter_revision.checked_add(1).ok_or_else(||
            Error::ArchitectureModel("bank parameter revision overflow".into()))?;
        prepared.push((replacements, bytes, revision));
    }
    if !ordinary()? {
        return Ok(false);
    }
    for (bank, (replacements, bytes, revision)) in guards.iter_mut().zip(prepared) {
        bank.parameter_replacements = replacements;
        bank.effective_member_bytes = bytes;
        bank.parameter_revision = revision;
    }
    Ok(true)
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
