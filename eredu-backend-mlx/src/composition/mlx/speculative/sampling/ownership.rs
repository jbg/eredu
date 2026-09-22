//! Opaque speculative key ownership, including detached native failure paths.

use super::*;

/// Native storage and completed original keys cannot be interchanged. The
/// original variant never exposes an Array or acquires an unquoted owner.
#[derive(Debug, Clone)]
pub(super) enum KeyValue<T> {
    Ordinary(T),
    Original(numerical::OriginalNumericalKey),
}
impl<T> KeyValue<T> {
    pub(super) fn ordinary(&self) -> Option<&T> {
        match self {
            Self::Ordinary(value) => Some(value),
            Self::Original(_) => None,
        }
    }
    pub(super) fn ordinary_mut(&mut self) -> Option<&mut T> {
        match self {
            Self::Ordinary(value) => Some(value),
            Self::Original(_) => None,
        }
    }
    pub(super) fn into_ordinary(self) -> Result<T, Exception> {
        match self {
            Self::Ordinary(value) => Ok(value),
            Self::Original(_) => Err(Exception::from_source(Error::PrefillControl(
                eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
            ))),
        }
    }
    pub(super) fn original(&self) -> Option<&numerical::OriginalNumericalKey> {
        match self {
            Self::Original(value) => Some(value),
            Self::Ordinary(_) => None,
        }
    }
    pub(super) fn original_mut(&mut self) -> Option<&mut numerical::OriginalNumericalKey> {
        match self {
            Self::Original(value) => Some(value),
            Self::Ordinary(_) => None,
        }
    }
}

/// Caller seed or position-addressable assistant key with retained authority.
/// Native values stay private so clones cannot escape their allocation owner.
#[derive(Debug, Clone)]
pub struct MlxSpeculativeSeed {
    pub(super) value: KeyValue<Array>,
    pub(super) memory_retention: NativeMemoryRetention,
}

impl MlxSpeculativeSeed {
    /// Takes existing storage without evaluating or allocating another key.
    pub(crate) fn from_array(value: Array) -> Self {
        Self {
            value: KeyValue::Ordinary(value),
            memory_retention: NativeMemoryRetention::default(),
        }
    }

    /// Couples a prepared key to the authority held before its allocation.
    pub(crate) fn with_memory_owner(mut self, owner: &NativeMemoryOwner) -> Self {
        if self.value.ordinary().is_some() {
            self.memory_retention.retain(owner);
        }
        self
    }

    #[cfg(test)]
    pub(crate) fn as_array(&self) -> &Array {
        self.value.ordinary().expect("ordinary test seed")
    }
}

/// Sequential key splitting retains the same authority through every clone.
/// Updating the native key replaces a value; it never releases that authority.
#[derive(Debug, Clone)]
pub struct MlxSpeculativeRandomState {
    pub(super) value: KeyValue<RandomState>,
    pub(super) memory_retention: NativeMemoryRetention,
}

/// Cover the selected ledger before mutation, retaining every source authority.
/// An existing lease from that ledger suffices; another ledger requires its own
/// authority before native key splitting or sampling can start.
pub(super) fn derivation_memory(
    retained: &NativeMemoryRetention,
    context: SpeculativeExecutionStreams<'_>,
) -> Result<NativeMemoryRetention, Exception> {
    let mut memory = retained.clone();
    if let Some(owner) = context.memory_owner() {
        memory.retain(owner);
    } else {
        let pool = context.memory_ledger();
        if !memory.covers_pool(&pool) {
            let owner = NativeMemoryOwner::acquire(&pool).map_err(Exception::from_source)?;
            memory.retain(&owner);
        }
    }
    Ok(memory)
}

/// Native scopes independently retain authority after a caller drops its state.
/// Failure/unwinding never treats an unresolved submission as completed work.
pub(super) fn with_memory_recovery<T>(
    memory: NativeMemoryRetention,
    operation: impl FnOnce() -> Result<T, Exception>,
) -> Result<T, Exception> {
    crate::backend::submission_recovery::detached_retained(memory, || {
        operation().map_err(Error::from)
    })
    .map_err(Exception::from_source)
}
