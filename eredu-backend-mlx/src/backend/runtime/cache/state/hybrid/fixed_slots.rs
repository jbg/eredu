//! Closed per-layer role slots. Value mutation cannot change their roles.

use super::{MlxTensor, StateTensorRole};
use eredu_core::cache::{CachePolicyError, LayerCachePolicy};
use eredu_runtime::{HostSlotMetadata, HostSlotTable};
use std::{iter::Map, slice};

mod prepared_copy;
mod empty_copy;
pub(in crate::backend::runtime::cache::state::hybrid) use prepared_copy::copy_slot_retained;
use prepared_copy::PreparedFixedStateCopy;

pub(in crate::backend::runtime::cache::state::hybrid) type Slot =
    (StateTensorRole, Option<MlxTensor>);
type Iter<'a> =
    Map<slice::Iter<'a, Slot>, fn(&'a Slot) -> (&'a StateTensorRole, &'a Option<MlxTensor>)>;
type IterMut<'a> = Map<
    slice::IterMut<'a, Slot>,
    fn(&'a mut Slot) -> (&'a StateTensorRole, &'a mut Option<MlxTensor>),
>;

/// An exact-size host payload. Numerical array storage is owned by its tensors.
///
/// The box retains no spare slots. Cloning copies the slot payload and shares
/// tensor handles, matching a transaction checkpoint. Each independent box has
/// fresh metadata custody; same-size clone_from preserves the existing owner.
/// Isolated numerical copies use `try_map_values`. Value mutation cannot insert
/// or remove a role; replacing
/// the whole table (including `clone_from`) installs the replacement role set.
#[derive(Debug)]
pub(super) struct FixedStateSlots {
    slots: HostSlotTable<Slot>,
}

impl FixedStateSlots {
    /// Only empty role tables have a zero-payload copy path. The caller already
    /// owns the enclosing admitted layer construction; this allocates custody
    /// metadata only, never a slot or nested numerical payload.
    pub(in crate::backend::runtime::cache::state::hybrid) fn copy_empty(&self) -> Option<Self> {
        (self.is_empty() && self.payload_bytes() == Some(0)).then(|| Self {
            slots: HostSlotTable::new(Box::new([])),
        })
    }

    pub(super) fn from_policy(policy: &LayerCachePolicy) -> Result<Self, CachePolicyError> {
        let tensors = policy.fixed_state();
        // Public layouts already validate uniqueness. Keep malformed private
        // construction fallible instead of silently retaining one duplicate.
        // This check needs no temporary set or retained spare capacity.
        for (index, tensor) in tensors.iter().enumerate() {
            if tensors[..index]
                .iter()
                .any(|other| other.role == tensor.role)
            {
                return Err(CachePolicyError::Invalid(format!(
                    "duplicate fixed-state tensor role {:?}",
                    tensor.role
                )));
            }
        }
        let mut slots = tensors
            .iter()
            .map(|tensor| (tensor.role, None))
            .collect::<Box<[_]>>();
        slots.sort_unstable_by_key(|(role, _)| *role);
        Ok(Self {
            slots: HostSlotTable::new(slots),
        })
    }

    pub(in crate::backend::runtime::cache::state::hybrid) fn slot(
        &self,
        index: usize,
    ) -> Option<&Slot> {
        self.slots.slots().get(index)
    }

    pub(in crate::backend::runtime::cache::state::hybrid) fn prepare_slots(
        &self,
    ) -> Result<
        eredu_runtime::HostSlotInitialization<'_, Slot>,
        eredu_runtime::HostSlotInitializationError,
    > {
        self.slots.prepare_copy_slots()
    }

    /// Only a closed admitted publication supplies this actual table in the
    /// grouped copy path. This move preserves its attached custody and pointer.
    pub(in crate::backend::runtime::cache::state::hybrid) fn from_published_slots(
        slots: HostSlotTable<Slot>,
    ) -> Self {
        Self { slots }
    }

    pub(super) fn prepare_copy(&self) -> PreparedFixedStateCopy<'_> {
        PreparedFixedStateCopy::new(self)
    }

    pub(super) fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    pub(in crate::backend::runtime::cache::state::hybrid) fn table(&self) -> &HostSlotTable<Slot> {
        &self.slots
    }

    /// Exact original table identity and attachment custody, without its payload.
    pub(super) fn metadata(&self) -> &HostSlotMetadata {
        self.slots.metadata()
    }

    /// Logical allocator payload only: all retained entries, even absent ones.
    /// Excludes native backing, identity/custody metadata, bookkeeping and `Self`.
    pub(super) fn payload_bytes(&self) -> Option<u64> {
        self.metadata().capacity_bytes()
    }

    pub(super) fn get_mut(&mut self, role: &StateTensorRole) -> Option<&mut Option<MlxTensor>> {
        let index = self
            .slots
            .slots()
            .binary_search_by_key(role, |(role, _)| *role)
            .ok()?;
        Some(&mut self.slots.slots_mut()[index].1)
    }

    pub(super) fn iter(&self) -> Iter<'_> {
        self.slots.slots().iter().map(slot_ref)
    }

    fn iter_mut(&mut self) -> IterMut<'_> {
        self.slots.slots_mut().iter_mut().map(slot_mut)
    }

    pub(super) fn values(&self) -> impl ExactSizeIterator<Item = &Option<MlxTensor>> {
        self.slots.slots().iter().map(|(_, value)| value)
    }

    pub(super) fn values_mut(&mut self) -> impl ExactSizeIterator<Item = &mut Option<MlxTensor>> {
        self.slots.slots_mut().iter_mut().map(|(_, value)| value)
    }

    /// Allocates precisely the source's slot count before copying any value.
    /// Preserves absent roles and sorted order; a failure drops partial copies.
    pub(super) fn try_map_values<E>(
        &self,
        mut copy: impl FnMut(&MlxTensor) -> Result<MlxTensor, E>,
    ) -> Result<Self, E> {
        let mut result = Self {
            slots: HostSlotTable::new(
                self.slots
                    .slots()
                    .iter()
                    .map(|(role, _)| (*role, None))
                    .collect(),
            ),
        };
        for ((_, source), (_, destination)) in self
            .slots
            .slots()
            .iter()
            .zip(result.slots.slots_mut().iter_mut())
        {
            *destination = source.as_ref().map(&mut copy).transpose()?;
        }
        Ok(result)
    }
}

impl Clone for FixedStateSlots {
    fn clone(&self) -> Self {
        Self {
            slots: HostSlotTable::new(Box::<[Slot]>::from(self.slots.slots())),
        }
    }

    fn clone_from(&mut self, source: &Self) {
        if self.slots.len() == source.slots.len() {
            self.slots
                .slots_mut()
                .clone_from_slice(source.slots.slots());
        } else {
            *self = source.clone();
        }
    }
}

fn slot_ref(slot: &Slot) -> (&StateTensorRole, &Option<MlxTensor>) {
    (&slot.0, &slot.1)
}

fn slot_mut(slot: &mut Slot) -> (&StateTensorRole, &mut Option<MlxTensor>) {
    (&slot.0, &mut slot.1)
}

impl<'a> IntoIterator for &'a FixedStateSlots {
    type Item = (&'a StateTensorRole, &'a Option<MlxTensor>);
    type IntoIter = Iter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a> IntoIterator for &'a mut FixedStateSlots {
    type Item = (&'a StateTensorRole, &'a mut Option<MlxTensor>);
    type IntoIter = IterMut<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

#[cfg(test)]
mod tests;
