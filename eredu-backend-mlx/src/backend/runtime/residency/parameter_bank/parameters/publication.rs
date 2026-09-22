//! Borrowed all-bank publication under one retained, globally ordered lock set.
use super::*;
use eredu_nn::workspace::{WorkspaceContext, WorkspaceMetadataError};
use eredu_runtime::parameter_operations::{ParameterPublication, ParameterReplacementValues};
use std::{
    mem::{size_of, size_of_val},
    sync::MutexGuard,
};

type Visit<'a> = dyn FnMut(&mut dyn ParameterPublication<MlxTensor>) -> Result<bool, Error> + 'a;

/// All bank locks remain held while the operation repeatedly visits, validates
/// and exchanges fields. Iterator clones must remain borrowed and allocation-free.
pub(crate) fn with_bank_parameter_publication<'a, I, R>(
    banks: I,
    context: &WorkspaceContext,
    operation: impl FnOnce(&mut Visit<'_>) -> Result<R, Error>,
) -> Result<R, Error>
where
    I: Iterator<Item = &'a SharedAddressableParameterBank> + Clone,
{
    let mut count = 0usize;
    let mut after = None;
    while let Some(bank) = next_bank(&banks, after) {
        count = count.checked_add(1).ok_or_else(overflow)?;
        after = Some(address(bank));
    }
    // Every recursive frame borrows one actual guard and the preceding visitor;
    // neither lock acquisition nor participant traversal builds a collection.
    let frame = [
        size_of::<I>(),
        size_of::<&I>(),
        size_of::<Option<usize>>(),
        size_of::<Option<&SharedAddressableParameterBank>>(),
        size_of::<MutexGuard<'_, AddressableParameterBank>>(),
        size_of::<&mut Visit<'_>>() * 2,
        size_of::<(
            &mut Visit<'_>,
            &mut MutexGuard<'_, AddressableParameterBank>,
        )>(),
        size_of::<Result<R, Error>>(),
        size_of::<std::collections::btree_map::IterMut<'_, ParameterBankKey, u64>>(),
        size_of::<(&ParameterBankKey, &mut u64)>(),
        size_of::<(
            ParameterBankKey,
            u64,
            &[eredu_runtime::parameter_operations::PreparedBankParameterMember],
            &ParameterReplacementValues<MlxTensor>,
            bool,
        )>(),
        size_of::<Result<u64, eredu_nn::Error>>(),
        size_of::<&Array>(),
        size_of::<usize>() * 3,
        size_of::<u64>() * 3,
    ];
    let bytes = frame
        .into_iter()
        .try_fold(size_of_val(&frame), usize::checked_add)
        .and_then(|n| n.checked_mul(count.checked_add(1)?))
        .and_then(|n| n.checked_add(size_of_val(&operation)))
        .ok_or_else(overflow)?;
    context
        .charge_metadata(bytes)
        .map_err(|cause| Error::Neural(cause.into()))?;
    lock_next(&banks, None, &mut |_| Ok(true), &mut Some(operation))
}

fn overflow() -> Error {
    Error::Neural(WorkspaceMetadataError::Overflow.into())
}
fn address(bank: &SharedAddressableParameterBank) -> usize {
    Arc::as_ptr(&bank.inner).addr()
}
fn next_bank<'a, I>(banks: &I, after: Option<usize>) -> Option<&'a SharedAddressableParameterBank>
where
    I: Iterator<Item = &'a SharedAddressableParameterBank> + Clone,
{
    banks
        .clone()
        .filter(|bank| after.is_none_or(|last| address(bank) > last))
        .min_by_key(|bank| address(bank))
}
fn lock_next<'a, I, R, F>(
    banks: &I,
    after: Option<usize>,
    previous: &mut Visit<'_>,
    operation: &mut Option<F>,
) -> Result<R, Error>
where
    I: Iterator<Item = &'a SharedAddressableParameterBank> + Clone,
    F: FnOnce(&mut Visit<'_>) -> Result<R, Error>,
{
    let Some(bank) = next_bank(banks, after) else {
        return operation.take().expect("one bank transaction")(previous);
    };
    let mut guard = bank
        .inner
        .lock()
        .map_err(|_| Error::Neural(WorkspaceMetadataError::Unqualified.into()))?;
    let mut visit = |visitor: &mut dyn ParameterPublication<MlxTensor>| {
        if !previous(visitor)? {
            return Ok(false);
        }
        visit_bank(&mut guard, visitor)?;
        Ok(true)
    };
    lock_next(banks, Some(address(bank)), &mut visit, operation)
}

fn visit_bank(
    bank: &mut AddressableParameterBank,
    visitor: &mut dyn ParameterPublication<MlxTensor>,
) -> Result<(), Error> {
    let AddressableParameterBank {
        parameter_members,
        parameter_replacements,
        parameter_revision,
        effective_member_bytes,
        catalog,
        ..
    } = bank;
    let revision = *parameter_revision;
    visitor.counter(parameter_revision, &mut |_, _| {
        revision
            .checked_add(1)
            .ok_or_else(|| WorkspaceMetadataError::Overflow.into())
    });
    for (key, value) in effective_member_bytes {
        let baseline = *catalog
            .get(key)
            .ok_or_else(|| Error::Neural(WorkspaceMetadataError::Unqualified.into()))?;
        visitor.counter(value, &mut |values, active| {
            member_bytes(*key, baseline, parameter_members, values, active)
        });
    }
    visitor.replacement_source(parameter_replacements);
    Ok(())
}

fn member_bytes(
    key: ParameterBankKey,
    baseline: u64,
    members: &[eredu_runtime::parameter_operations::PreparedBankParameterMember],
    values: &ParameterReplacementValues<MlxTensor>,
    active: bool,
) -> Result<u64, eredu_nn::Error> {
    if !active {
        return Ok(baseline);
    }
    let invalid = || eredu_nn::Error::from(WorkspaceMetadataError::Unqualified);
    let overflow = || eredu_nn::Error::from(WorkspaceMetadataError::Overflow);
    let mut bytes = baseline;
    for member in members.iter().filter(|member| member.key == key) {
        let Some(value) = values.get(&member.parameter) else {
            continue;
        };
        let count = members
            .iter()
            .filter(|other| other.parameter == member.parameter)
            .count();
        let array = value.as_array();
        if array.shape().first().copied() != i32::try_from(count).ok()
            || !matches!(
                array.dtype(),
                Dtype::Float32 | Dtype::Float16 | Dtype::Bfloat16
            )
            || array.nbytes() % count != 0
        {
            return Err(invalid());
        }
        let capacity = u64::try_from(array.nbytes() / count).map_err(|_| overflow())?;
        bytes = bytes
            .checked_sub(member.materialized.byte_len)
            .and_then(|n| n.checked_add(capacity))
            .ok_or_else(overflow)?;
    }
    Ok(bytes)
}
