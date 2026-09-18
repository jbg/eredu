//! Processed logits retain the actual source representation and authority.
use super::*;
#[derive(Debug, Clone)]
enum Storage {
    Ordinary {
        value: Array,
        memory: NativeMemoryRetention,
    },
    Original(numerical::OriginalNumericalValue),
}
/// Opaque processed distribution; original values expose no raw array handle.
#[derive(Debug, Clone)]
pub struct MlxSpeculativeDistribution {
    storage: Storage,
}
impl MlxSpeculativeDistribution {
    pub(super) fn new(value: Array, memory: NativeMemoryRetention) -> Self {
        Self {
            storage: Storage::Ordinary { value, memory },
        }
    }
    pub(super) fn original(
        value: numerical::OriginalNumericalValue,
        sources: &crate::composition::mlx::speculative::OriginalSpeculativeNumericalSources,
    ) -> Result<Self, Error> {
        use eredu_nn::workspace::HostMetadataFundingError;
        use std::mem::{size_of, size_of_val};
        let funding = value.validate_consumer(sources)?;
        // The inner value/Rc/native owners were paid by their producer. Only
        // this new enclosing representation and its constructor transports are
        // born here; the moved value retains that same account until drop.
        let parts = [
            size_of::<Storage>(),
            size_of::<Self>(),
            size_of::<Result<Self, Error>>(),
            size_of::<&crate::composition::mlx::speculative::OriginalSpeculativeNumericalSources>(),
            size_of::<Result<&eredu_nn::workspace::HostMetadataFunding, Error>>(),
        ];
        let bytes = parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
            .ok_or(Error::WorkspacePlanning(
                HostMetadataFundingError::Overflow,
            ))?;
        funding
            .reserve_metadata(bytes)
            .map_err(Error::WorkspacePlanning)?;
        Ok(Self {
            storage: Storage::Original(value),
        })
    }
    pub(super) fn original_value(&self) -> Option<&numerical::OriginalNumericalValue> {
        match &self.storage {
            Storage::Original(value) => Some(value),
            _ => None,
        }
    }
    pub(super) fn value(&self) -> &Array {
        match &self.storage {
            Storage::Ordinary { value, .. } => value,
            _ => unreachable!("ordinary distribution loan"),
        }
    }
    pub(super) fn value_mut(&mut self) -> &mut Array {
        match &mut self.storage {
            Storage::Ordinary { value, .. } => value,
            _ => unreachable!("ordinary distribution loan"),
        }
    }
    pub(super) fn memory(&self) -> &NativeMemoryRetention {
        match &self.storage {
            Storage::Ordinary { memory, .. } => memory,
            _ => unreachable!("ordinary distribution loan"),
        }
    }
    pub(super) fn memory_mut(&mut self) -> &mut NativeMemoryRetention {
        match &mut self.storage {
            Storage::Ordinary { memory, .. } => memory,
            _ => unreachable!("ordinary distribution loan"),
        }
    }
    #[cfg(test)]
    pub(crate) fn as_array(&self) -> &Array {
        self.value()
    }
}
