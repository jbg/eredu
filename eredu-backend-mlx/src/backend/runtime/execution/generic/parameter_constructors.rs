//! Exact unloaded-slot construction facts, distinct from leased parameter rows.
use eredu_nn::workspace::{WorkspaceDtype, WorkspaceLayout};
use std::collections::BTreeMap;

/// Every freshly constructed module slot has its own scalar seed, including
/// slots subsequently populated by an independent parameter owner.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct ParameterConstructors {
    pub(crate) slots: usize,
    pub(crate) scalar_bytes: u64,
    pub(crate) rank: usize,
    pub(crate) dtypes: [usize; 5],
}

impl ParameterConstructors {
    pub(crate) fn from_layouts(layouts: &BTreeMap<String, WorkspaceLayout>) -> Option<Self> {
        let mut result = Self::default();
        for layout in layouts.values() {
            result.include(layout.dtype(), layout.shape().len())?;
        }
        Some(result)
    }

    pub(crate) fn include(&mut self, dtype: WorkspaceDtype, rank: usize) -> Option<()> {
        let index = Self::dtype_index(dtype);
        let next = Self {
            slots: self.slots.checked_add(1)?,
            scalar_bytes: self.scalar_bytes.checked_add(dtype.bytes())?,
            rank: self.rank.max(rank),
            dtypes: {
                let mut counts = self.dtypes;
                counts[index] = counts[index].checked_add(1)?;
                counts
            },
        };
        *self = next;
        Some(())
    }

    pub(crate) fn checked_merge(self, other: Self) -> Option<Self> {
        let mut dtypes = self.dtypes;
        for (count, additional) in dtypes.iter_mut().zip(other.dtypes) {
            *count = count.checked_add(additional)?;
        }
        Some(Self {
            slots: self.slots.checked_add(other.slots)?,
            scalar_bytes: self.scalar_bytes.checked_add(other.scalar_bytes)?,
            rank: self.rank.max(other.rank),
            dtypes,
        })
    }

    pub(crate) fn dtype_index(dtype: WorkspaceDtype) -> usize {
        match dtype {
            WorkspaceDtype::Float32 => 0,
            WorkspaceDtype::Int32 => 1,
            WorkspaceDtype::Bool => 2,
            WorkspaceDtype::Uint8 => 3,
            WorkspaceDtype::Uint32 => 4,
        }
    }
}
