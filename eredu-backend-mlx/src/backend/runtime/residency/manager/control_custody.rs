//! Host control storage retains its actual preparation account independently
//! of the native submission mode selected by the acquisition worker.
use eredu_nn::workspace::HostMetadataFunding;
use eredu_runtime::working_memory::OriginalOperationMetadataCustody;

#[derive(Clone, Debug)]
pub(super) enum ResidencyControlCustody {
    Original(OriginalOperationMetadataCustody),
    Ordinary(HostMetadataFunding),
}

impl ResidencyControlCustody {
    pub(super) fn original(&self) -> Option<&OriginalOperationMetadataCustody> {
        match self {
            Self::Original(value) => Some(value),
            Self::Ordinary(_) => None,
        }
    }
    pub(super) fn same_account(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Original(a), Self::Original(b)) => a.same_account(b),
            (Self::Ordinary(a), Self::Ordinary(b)) => a.same_account(b),
            _ => false,
        }
    }
}

impl From<OriginalOperationMetadataCustody> for ResidencyControlCustody {
    fn from(value: OriginalOperationMetadataCustody) -> Self {
        Self::Original(value)
    }
}

impl From<HostMetadataFunding> for ResidencyControlCustody {
    fn from(value: HostMetadataFunding) -> Self {
        Self::Ordinary(value)
    }
}
