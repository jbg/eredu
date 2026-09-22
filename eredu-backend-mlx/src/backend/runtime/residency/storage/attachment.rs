//! Scalar evidence from an actual successful native attachment.
use eredu_runtime::working_memory::{MemoryLedger, OriginalOperationMetadataCustody};
use safemlx::AllocationInfo;

/// Private construction stays in the existing checked publication producers.
/// The proof owns no native payload, registry pin, or accounting authority.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PublishedAllocation(AllocationInfo);
impl PublishedAllocation {
    pub(super) fn attached(allocation: AllocationInfo) -> Self {
        Self(allocation)
    }
    pub(crate) fn allocation(self) -> AllocationInfo {
        self.0
    }
    pub(crate) fn borrow(
        self,
        custody: &OriginalOperationMetadataCustody,
    ) -> RetainedAllocationReceipt<'_> {
        RetainedAllocationReceipt {
            allocation: self.0,
            custody,
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct RetainedAllocationReceipt<'a> {
    allocation: AllocationInfo,
    custody: &'a OriginalOperationMetadataCustody,
}

/// The completed producer's immutable proof and accounting-only source custody.
/// Native aliases retain the actual backing independently of this receipt.
#[derive(Clone, Debug)]
pub(crate) struct RetainedAllocationSource {
    allocation: AllocationInfo,
    custody: OriginalOperationMetadataCustody,
}
impl RetainedAllocationSource {
    pub(crate) fn allocation(&self) -> AllocationInfo {
        self.allocation
    }
    pub(crate) fn borrow(&self) -> RetainedAllocationReceipt<'_> {
        RetainedAllocationReceipt {
            allocation: self.allocation,
            custody: &self.custody,
        }
    }
    pub(crate) fn same_source(&self, other: &Self) -> bool {
        self.allocation == other.allocation && self.custody.same_account(&other.custody)
    }
}
impl RetainedAllocationReceipt<'_> {
    pub(crate) fn retain(self) -> RetainedAllocationSource {
        RetainedAllocationSource {
            allocation: self.allocation,
            custody: self.custody.clone(),
        }
    }
    pub(crate) fn matches(&self, allocation: AllocationInfo, pool: &MemoryLedger) -> bool {
        self.allocation == allocation && self.custody.validate_retained_origin(pool).is_ok()
    }
    pub(crate) fn control_bytes() -> Option<usize> {
        let frames = [
            size_of::<(&Self, AllocationInfo, &MemoryLedger)>(),
            size_of::<Option<Self>>(),
            size_of::<bool>(),
            size_of::<(PublishedAllocation, &OriginalOperationMetadataCustody)>(),
            size_of::<Self>(),
            OriginalOperationMetadataCustody::retained_origin_control_bytes()?,
        ];
        frames
            .into_iter()
            .try_fold(std::mem::size_of_val(&frames), usize::checked_add)
    }
}
