//! Disjoint existing coverage. These private owners cannot be minted from a key.
use super::*;
use funding::native_partition::NativePartition;
use crate::working_memory::OriginalHostMetadataCustody;

#[derive(Clone, Debug)]
pub(in crate::working_memory) struct PrepaidHostOrigin {
    bytes: u64,
    raw: OriginalHostMetadataCustody,
}
impl PrepaidHostOrigin {
    // Only the successful selected source worker calls this after its C+B debit.
    pub(in crate::working_memory) fn from_source_attempt(
        bytes: u64,
        raw: OriginalHostMetadataCustody,
    ) -> Self {
        Self { bytes, raw }
    }
    pub(in crate::working_memory) fn raw(&self) -> &OriginalHostMetadataCustody {
        &self.raw
    }
    fn same(&self, other: &Self) -> bool {
        self.bytes == other.bytes && self.raw.same(&other.raw)
    }
}
impl Drop for PrepaidHostOrigin {
    fn drop(&mut self) {
        // All active aliases retire outside Usage, including retained failure
        // prefixes. A surviving alias must immediately see origin quarantine.
        if std::thread::panicking() {
            self.raw.quarantine();
        }
    }
}

#[derive(Clone, Debug)]
pub(in crate::working_memory) enum PrepaidStorageOrigin {
    Native(NativePartition),
    Immutable(PrepaidHostOrigin),
    Source(crate::working_memory::gguf_source::SourceInventoryOrigin),
}
impl PrepaidStorageOrigin {
    pub(in crate::working_memory) fn buffer_kind(&self, immutable: bool) -> bool {
        matches!(
            (self, immutable),
            (Self::Native(_), false) | (Self::Immutable(_), true)
        )
    }
    pub(in crate::working_memory) fn residual_source_charge(&self) -> Option<u64> {
        match self {
            Self::Source(origin) => Some(origin.residual()),
            _ => None,
        }
    }

    pub(in crate::working_memory) fn immutable(&self) -> bool {
        matches!(self, Self::Immutable(_))
    }
    pub(in crate::working_memory) fn same_origin(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Native(a), Self::Native(b)) => a.same_origin(b),
            (Self::Immutable(a), Self::Immutable(b)) => a.same(b),
            (Self::Source(a), Self::Source(b)) => a.same(b),
            _ => false,
        }
    }
    pub(in crate::working_memory) fn validate_origin(
        &self,
        usage: &crate::working_memory::Usage,
    ) -> Result<(), WorkingMemoryError> {
        match self {
            Self::Native(origin) => origin.validate_origin(usage),
            Self::Source(origin) => origin.validate(),
            Self::Immutable(origin) => origin.raw.validate_origin_locked(origin.raw.pool(), usage),
        }
    }
    pub(in crate::working_memory) fn validate_pool(
        &self,
        pool: &WorkingMemoryPool,
        usage: &crate::working_memory::Usage,
    ) -> Result<(), WorkingMemoryError> {
        match self {
            Self::Native(origin) => origin.validate_pool(pool, usage),
            Self::Source(origin) => origin.validate_pool(pool),
            Self::Immutable(origin) => origin.raw.validate_origin_locked(pool, usage),
        }
    }
    pub(in crate::working_memory) fn validate_capacity(
        &self,
        bytes: u64,
    ) -> Result<(), WorkingMemoryError> {
        let valid = match self {
            Self::Native(origin) => bytes <= origin.capacity(),
            Self::Source(origin) => bytes == origin.total(),
            Self::Immutable(origin) => bytes == origin.bytes,
        };
        if valid {
            Ok(())
        } else {
            Err(WorkingMemoryError::IdentityMismatch)
        }
    }
    pub(in crate::working_memory) fn validate_publisher(
        &self,
        scope: &WorkingMemoryFundingScope,
        usage: &crate::working_memory::Usage,
    ) -> Result<(), WorkingMemoryError> {
        match self {
            Self::Native(origin) => origin.validate_publisher(scope, usage),
            Self::Source(_) => Err(WorkingMemoryError::IdentityMismatch),
            Self::Immutable(origin) => {
                if origin.raw.account() != scope.id || !origin.raw.pool().same_domain(scope.pool())
                {
                    return Err(WorkingMemoryError::IdentityMismatch);
                }
                origin.raw.validate_origin_locked(scope.pool(), usage)
            }
        }
    }
}
