//! Retains an actual canonical cell when the source offers one.
use super::super::manager::CanonicalArrayOwner;
use safemlx::Array;

pub(crate) enum RetainedArray {
    Plain(Array),
    Canonical {
        // A prior plain alias must not be destroyed under the manager loan.
        // It is moved here without cloning and retires with this inventory.
        displaced: Option<Array>,
        cell: CanonicalArrayOwner,
    },
}
impl RetainedArray {
    pub(crate) fn control_bytes() -> Option<usize> {
        let frames = [
            size_of::<&Self>(),
            size_of::<&mut Self>(),
            size_of::<Self>(),
            size_of::<Self>(),
            size_of::<&CanonicalArrayOwner>(),
            size_of::<CanonicalArrayOwner>(),
            size_of::<Option<&CanonicalArrayOwner>>(),
            size_of::<Option<Array>>(),
            size_of::<Option<&mut (u64, Self)>>(),
            size_of::<bool>(),
            CanonicalArrayOwner::publication_control_bytes()?,
        ];
        frames
            .into_iter()
            .try_fold(std::mem::size_of_val(&frames), usize::checked_add)
    }

    pub(crate) fn canonical(&self) -> Option<&CanonicalArrayOwner> {
        match self {
            Self::Canonical { cell, .. } => Some(cell),
            Self::Plain(_) => None,
        }
    }
    pub(crate) fn from_canonical(cell: &CanonicalArrayOwner) -> Self {
        Self::Canonical {
            cell: cell.clone(),
            displaced: None,
        }
    }
    pub(crate) fn upgrade(&mut self, owner: &CanonicalArrayOwner) -> bool {
        if let Self::Canonical { cell, .. } = self {
            return cell.same_cell(owner);
        }
        let previous = std::mem::replace(self, Self::from_canonical(owner));
        if let (Self::Canonical { displaced, .. }, Self::Plain(array)) = (self, previous) {
            *displaced = Some(array);
        }
        true
    }
}
impl From<Array> for RetainedArray {
    fn from(value: Array) -> Self {
        Self::Plain(value)
    }
}
impl std::ops::Deref for RetainedArray {
    type Target = Array;
    fn deref(&self) -> &Array {
        match self {
            Self::Plain(array) => array,
            Self::Canonical { cell, .. } => cell.array(),
        }
    }
}
