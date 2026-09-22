//! Per-transfer publication rows and ID destinations, prepared under the same Q.
//! Actual tensor ownership stays in the existing named-array publication worker.
use super::super::named_arrays::NamedPreparationSource;
use super::*;
use eredu_runtime::working_memory::OriginalOperationMetadataCustody;
use std::{
    alloc::Layout,
    mem::size_of,
    ops::{Deref, DerefMut},
    sync::TryLockError,
};

type Row = (OffloadUnitId, PreparedResidentArrays);
type IdDestinations = (Option<OffloadUnitId>, Option<OffloadUnitId>);

pub(super) struct TransferPublication {
    values: Vec<Row>,
    ids: Vec<IdDestinations>,
    // Local rows can outlive the recovery variable on failure. No bare Vec
    // extraction exposes them without the actual prepaid storage custody.
    custody: Option<ResidencyControlCustody>,
}
impl TransferPublication {
    pub(super) fn ordinary() -> Self {
        Self {
            values: Vec::new(),
            ids: Vec::new(),
            custody: None,
        }
    }
    pub(super) fn original(controls: &OriginalOperationMetadataCustody) -> Self {
        Self::with_custody(controls.clone().into())
    }
    pub(super) fn with_custody(custody: ResidencyControlCustody) -> Self {
        Self {
            values: Vec::new(),
            ids: Vec::new(),
            custody: Some(custody),
        }
    }
    pub(super) fn prepare(
        &mut self,
        manager: &ResidencyManager,
        roots: &[OffloadUnitId],
        scratch: &mut [eredu_runtime::residency::ResidencyClosureSlot],
        shape: TransferPayloadShape,
    ) -> Result<(), NamedPreparationSource> {
        if self.custody.is_none() || !self.ids.is_empty() || !self.values.is_empty() {
            return Err(NamedArrayError::InvalidSource.into());
        }
        let state = manager
            .inner
            .state
            .try_lock()
            .map_err(|cause| match cause {
                TryLockError::WouldBlock => NamedArrayError::ManagerBusy,
                TryLockError::Poisoned(_) => NamedArrayError::ManagerPoisoned,
            })?;
        let closure = state
            .control
            .operation_closure(roots, scratch)
            .map_err(|_| NamedArrayError::InvalidSource)?;
        let bytes = closure
            .units()
            .try_fold(0usize, |n, unit| n.checked_add(unit.id().as_str().len()))
            .ok_or(NamedArrayError::Overflow)?;
        if closure.len() != shape.prepared_units
            || closure.len() != shape.unit_ids
            || bytes != shape.unit_id_bytes
        {
            return Err(NamedArrayError::InvalidSource.into());
        }
        self.values.try_reserve_exact(shape.prepared_units)?;
        self.ids.try_reserve_exact(shape.prepared_units)?;
        for unit in closure.units() {
            // Retain each successful first ID if the second allocation fails.
            self.ids.push((None, None));
            let row = self.ids.last_mut().expect("new final ID destination");
            row.0 = Some(copy_id(unit.id())?);
            row.1 = Some(copy_id(unit.id())?);
        }
        Ok(())
    }
    pub(super) fn select_application(
        &mut self,
        ids: &[OffloadUnitId],
        missing: &[bool],
        output: &mut Vec<OffloadUnitId>,
    ) -> Result<(), NamedArrayError> {
        if self.custody.is_none() || ids.len() != missing.len() || !output.is_empty() {
            return Err(NamedArrayError::InvalidSource);
        }
        let count = missing.iter().filter(|value| **value).count();
        if count > output.capacity() || count > self.values.capacity() {
            return Err(NamedArrayError::InvalidSource);
        }
        // Refuse the complete selection before consuming any final destination.
        for (index, (id, selected)) in ids.iter().zip(missing).enumerate() {
            if !selected {
                continue;
            }
            if ids[..index]
                .iter()
                .zip(&missing[..index])
                .any(|(old, selected)| *selected && old == id)
                || !self
                    .ids
                    .iter()
                    .any(|row| row.0.as_ref() == Some(id) && row.1.as_ref() == Some(id))
            {
                return Err(NamedArrayError::InvalidSource);
            }
        }
        for (id, selected) in ids.iter().zip(missing) {
            if *selected {
                let row = self
                    .ids
                    .iter_mut()
                    .find(|row| row.0.as_ref() == Some(id))
                    .expect("validated final application ID");
                output.push(row.0.take().expect("unconsumed application ID"));
            }
        }
        Ok(())
    }
    pub(super) fn push(
        &mut self,
        id: &OffloadUnitId,
        value: PreparedResidentArrays,
    ) -> Result<(), ResidencyError> {
        let id = if self.custody.is_some() {
            if self.values.len() == self.values.capacity() {
                return Err(NamedArrayError::InvalidSource.into());
            }
            let row = self
                .ids
                .iter_mut()
                .find(|row| row.0.is_none() && row.1.as_ref() == Some(id))
                .ok_or(NamedArrayError::InvalidSource)?;
            row.1.take().expect("unconsumed final publication ID")
        } else {
            id.clone()
        };
        self.values.push((id, value));
        Ok(())
    }
    pub(super) fn drain(&mut self) -> std::vec::Drain<'_, Row> {
        self.values.drain(..)
    }
    pub(super) fn control_bytes(shape: TransferPayloadShape) -> Option<u64> {
        let bytes = Layout::array::<Row>(shape.prepared_units)
            .ok()?
            .size()
            .checked_add(
                Layout::array::<IdDestinations>(shape.prepared_units)
                    .ok()?
                    .size(),
            )?
            .checked_add(shape.unit_id_bytes.checked_mul(2)?)?;
        let controls = [
            size_of::<Self>(),
            size_of::<Option<Self>>(),
            size_of::<Result<(), NamedPreparationSource>>(),
            size_of::<Result<(), NamedArrayError>>(),
            size_of::<Result<(), ResidencyError>>(),
            size_of::<
                Result<(ResidentRecovery<Rc<ResidentTransferResources>>, Self), ResidencyError>,
            >(),
            size_of::<std::vec::Drain<'static, Row>>(),
            size_of::<std::borrow::Cow<'static, [WeightBinding]>>(),
            size_of::<std::sync::MutexGuard<'static, ManagerState>>(),
            size_of::<std::sync::TryLockResult<std::sync::MutexGuard<'static, ManagerState>>>(),
            size_of::<eredu_runtime::residency::ResidencyClosure<'static>>(),
            size_of::<
                Result<
                    eredu_runtime::residency::ResidencyClosure<'static>,
                    eredu_runtime::residency::ResidencyClosureError,
                >,
            >(),
            size_of::<String>(),
            size_of::<Result<OffloadUnitId, std::collections::TryReserveError>>(),
        ]
        .into_iter()
        .try_fold(bytes, usize::checked_add)?;
        u64::try_from(controls).ok()
    }
}
fn copy_id(id: &OffloadUnitId) -> Result<OffloadUnitId, std::collections::TryReserveError> {
    let mut value = String::new();
    value.try_reserve_exact(id.as_str().len())?;
    value.push_str(id.as_str());
    Ok(OffloadUnitId::new(value).expect("borrowed canonical ID is nonempty"))
}
impl Deref for TransferPublication {
    type Target = [Row];
    fn deref(&self) -> &Self::Target {
        &self.values
    }
}
impl DerefMut for TransferPublication {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.values
    }
}
