//! Finite source closure over the controller's retained canonical alias rows.
//! This does not acquire, pin, materialize, or publish a unit.
use super::{OffloadUnit, OffloadUnitId, ResidencyController, WeightBinding};
use std::alloc::Layout;

/// One caller-owned destination per registered unit. Its representation is not
/// authority; only the controller worker fills a valid borrowed closure.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ResidencyClosureSlot {
    selected: bool,
}
impl ResidencyClosureSlot {
    /// Exact backing layout for the fixed source-inspection destination.
    pub fn layout(units: usize) -> Option<Layout> {
        Layout::array::<Self>(units).ok()
    }
}

/// Fixed source/destination validation failure; never owns formatted names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResidencyClosureError {
    /// The destination must have exactly one slot per retained unit.
    DestinationLength,
    /// A requested identity is absent from this controller.
    UnknownRoot,
    /// A retained canonical location was absent from its validated unit table.
    InvalidOwner,
}

/// Genuine borrowed source units reached through canonical binding ownership.
/// Repeated roots and owner edges do not duplicate units. Cyclic *unit* edges
/// are valid even though individual tensor alias cycles are invalid.
pub struct ResidencyClosure<'a> {
    units: &'a [OffloadUnit],
    slots: &'a [ResidencyClosureSlot],
    count: usize,
}
impl ResidencyClosure<'_> {
    /// Stable controller order, borrowing the original declarations and names.
    pub fn units(&self) -> impl Iterator<Item = &OffloadUnit> {
        self.units
            .iter()
            .zip(self.slots)
            .filter_map(|(unit, slot)| slot.selected.then_some(unit))
    }
    /// Number of distinct reached units, including requested roots.
    pub const fn len(&self) -> usize {
        self.count
    }
    /// Whether the request had no roots.
    pub const fn is_empty(&self) -> bool {
        self.count == 0
    }
}
impl ResidencyController {
    /// Borrows the same allocation-free canonical ordinal lookup as ordinary callers.
    pub fn binding_owner_borrowed(
        &self,
        unit: &OffloadUnitId,
        binding: &WeightBinding,
    ) -> Option<(&OffloadUnitId, &WeightBinding)> {
        self.binding_owner(unit, binding)
    }

    /// Computes the finite visited closure without recursion or allocation.
    /// Destination and every root are checked before any destination write.
    /// Each successful pass adds at least one of N source units, so at most N
    /// changing passes occur; no call stack follows induced unit cycles.
    pub fn operation_closure<'a>(
        &'a self,
        roots: &[OffloadUnitId],
        scratch: &'a mut [ResidencyClosureSlot],
    ) -> Result<ResidencyClosure<'a>, ResidencyClosureError> {
        closure_with(&self.units, roots.iter(), scratch, |from, visit| {
            // Constructor order is canonical (unit, binding) ordinal order.
            let start = self
                .alias_owners
                .partition_point(|row| row.alias.unit < from);
            let end = self
                .alias_owners
                .partition_point(|row| row.alias.unit <= from);
            for row in &self.alias_owners[start..end] {
                visit(row.owner.unit)?;
            }
            Ok(())
        })
    }
}

/// The same finite fixed-point worker serves the retained ordinal table and
/// the pre-construction borrowed declarations. The owner adapter alone differs.
pub(super) fn closure_with<'a, 'r>(
    units: &'a [OffloadUnit],
    roots: impl Iterator<Item = &'r OffloadUnitId> + Clone,
    scratch: &'a mut [ResidencyClosureSlot],
    mut owners: impl FnMut(
        usize,
        &mut dyn FnMut(usize) -> Result<(), ResidencyClosureError>,
    ) -> Result<(), ResidencyClosureError>,
) -> Result<ResidencyClosure<'a>, ResidencyClosureError> {
    if scratch.len() != units.len() {
        return Err(ResidencyClosureError::DestinationLength);
    }
    for root in roots.clone() {
        if !units.iter().any(|unit| unit.id() == root) {
            return Err(ResidencyClosureError::UnknownRoot);
        }
    }
    scratch.fill(ResidencyClosureSlot::default());
    for (unit, slot) in units.iter().zip(scratch.iter_mut()) {
        slot.selected = roots.clone().any(|root| root == unit.id());
    }
    loop {
        let mut changed = false;
        for from in 0..units.len() {
            if !scratch[from].selected {
                continue;
            }
            owners(from, &mut |to| {
                let destination = scratch
                    .get_mut(to)
                    .ok_or(ResidencyClosureError::InvalidOwner)?;
                if !destination.selected {
                    destination.selected = true;
                    changed = true;
                }
                Ok(())
            })?;
        }
        if !changed {
            break;
        }
    }
    let count = scratch.iter().filter(|slot| slot.selected).count();
    Ok(ResidencyClosure {
        units,
        slots: scratch,
        count,
    })
}
