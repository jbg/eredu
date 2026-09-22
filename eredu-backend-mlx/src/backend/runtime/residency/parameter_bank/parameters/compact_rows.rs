//! One physical row worker for compact bindings from actual acquired members.
use super::*;
use crate::backend::nn::shared::PreparedCompactBindings;
use eredu_nn::{
    workspace::{HostMetadataFunding, HostMetadataFundingError},
    Tensor,
};
use std::mem::{size_of, size_of_val};

#[derive(Debug, thiserror::Error)]
pub(crate) enum CompactRowsError {
    #[error("compact member source or binding identity differs")]
    Identity,
    #[error("compact member source geometry differs")]
    Geometry,
    #[error("compact member host quotation overflowed")]
    Overflow,
    #[error("compact member funding: {0}")]
    Funding(#[source] HostMetadataFundingError),
    #[error("compact member native handle: {0}")]
    Clone(#[source] safemlx::PreparedArrayCloneCause),
    #[error("compact member resident source: {0}")]
    Residency(#[source] ResidencyError),
    #[error("compact member view or binding: {0}")]
    Neural(#[source] eredu_nn::Error),
    #[error("compact member concatenate: {0}")]
    Native(#[source] safemlx::error::Exception),
}

pub(crate) fn compact_rows_control_bytes() -> Option<usize> {
    let frames = [
        size_of::<CompactRowsError>(),
        size_of::<Result<(), CompactRowsError>>(),
        size_of::<(
            &AddressableBankSourceLoan<'_>,
            &AcquiredParameterGroups,
            &[&str],
            &Stream,
            &HostMetadataFunding,
            &mut PreparedCompactBindings<'_>,
            &mut Vec<Array>,
        )>(),
        size_of::<(usize, &str, &ParameterBankKey)>(),
        size_of::<[i32; 2]>(),
        size_of::<Option<(&MlxTensor, usize)>>(),
        size_of::<safemlx::PreparedArrayClone>(),
    ];
    frames
        .into_iter()
        .try_fold(size_of_val(&frames), usize::checked_add)
}

/// Caller-owned rows have their finite payload and clone capacities paid before
/// entry. This source loan fixes replacement publication through all row reads.
pub(crate) fn fill_compact_rows<'names>(
    source: &AddressableBankSourceLoan<'_>,
    acquisition: &AcquiredParameterGroups,
    names: &'names [&'names str],
    stream: &Stream,
    funding: &HostMetadataFunding,
    bindings: &mut PreparedCompactBindings<'names>,
    rows: &mut Vec<Array>,
) -> Result<(), CompactRowsError> {
    use CompactRowsError as E;
    funding
        .reserve_metadata(compact_rows_control_bytes().ok_or(E::Overflow)?)
        .map_err(E::Funding)?;
    if names.is_empty()
        || names.iter().any(|name| name.is_empty())
        || names.windows(2).any(|pair| pair[0] >= pair[1])
        || acquisition.identities.len() != acquisition.transfer.leases().len()
        || acquisition.identities.is_empty()
        || rows.capacity() < acquisition.identities.len()
    {
        return Err(E::Identity);
    }
    for key in &acquisition.identities {
        let mut found = 0usize;
        for member in source.members().filter(|member| member.key == *key) {
            if names.binary_search(&member.binding.as_str()).is_err() {
                return Err(E::Identity);
            }
            found = found.checked_add(1).ok_or(E::Overflow)?;
        }
        if found != names.len() {
            return Err(E::Identity);
        }
    }
    for &name in names {
        rows.clear();
        for (key, lease) in acquisition
            .identities
            .iter()
            .zip(acquisition.transfer.leases())
        {
            let mut matches = source
                .members()
                .filter(|member| member.key == *key && member.binding == name);
            let member = matches.next().ok_or(E::Identity)?;
            if matches.next().is_some() {
                return Err(E::Identity);
            }
            let value = if let Some((replacement, row)) = source.replacement(member) {
                let start = i32::try_from(row).map_err(|_| E::Geometry)?;
                let end = start.checked_add(1).ok_or(E::Overflow)?;
                let controls = crate::tensor::narrow::control_bytes(replacement.shape().len())
                    .ok_or(E::Overflow)?;
                funding.reserve_metadata(controls).map_err(E::Funding)?;
                replacement
                    .narrow_axis(0, start, end, stream)
                    .map_err(E::Neural)?
                    .into_array()
            } else {
                if lease.tier() != MemoryTier::Device
                    || !lease.binding_names().any(|actual| actual == name)
                {
                    return Err(E::Identity);
                }
                let value = lease.device_value(name).map_err(E::Residency)?;
                let mut slot =
                    safemlx::PreparedArrayClone::try_prepare_for_inspection().map_err(E::Clone)?;
                slot.fill_for_inspection(value).map_err(E::Clone)?
            };
            if value.shape().first() != Some(&1) {
                return Err(E::Geometry);
            }
            rows.push(value);
        }
        let value = concatenate_axis(&*rows, 0, stream).map_err(E::Native)?;
        bindings.push(name, value).map_err(E::Neural)?;
    }
    Ok(())
}
