//! Fixed native state-owner visits, excluding all manager-owned catalogs.

/// Scalar diagnostics for the actual immutable state representation. Aliases
/// count separately; these are handle slots, never numerical byte bounds.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct NativeStateSlotCounts {
    pub(crate) arrays: usize,
    pub(crate) layouts: usize,
    pub(crate) slot_tables: usize,
    pub(crate) manager_roles: usize,
}

impl NativeStateSlotCounts {
    pub(super) fn arrays(arrays: usize, manager_roles: usize) -> Self {
        Self {
            arrays,
            manager_roles,
            ..Self::default()
        }
    }

    pub(super) fn checked_add(self, other: Self) -> Option<Self> {
        Some(Self {
            arrays: self.arrays.checked_add(other.arrays)?,
            layouts: self.layouts.checked_add(other.layouts)?,
            slot_tables: self.slot_tables.checked_add(other.slot_tables)?,
            manager_roles: self.manager_roles.checked_add(other.manager_roles)?,
        })
    }
}

#[cfg(test)]
mod tests;
