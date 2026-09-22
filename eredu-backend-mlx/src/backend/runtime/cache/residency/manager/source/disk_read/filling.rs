//! One payload-write worker over actual prepared Original or ordinary buffers.
use super::*;
use safemlx::{AllocationPlacement, PreparedAllocationOwner, PreparedHostTransferWriter};

pub(super) enum Filling {
    Original(filled_host::Pending),
    Ordinary(PreparedHostTransferWriter),
}
impl Filling {
    pub(super) fn bytes_mut(&mut self) -> &mut [u8] {
        match self {
            Self::Original(pending) => pending.completed_mut().bytes_mut(),
            Self::Ordinary(writer) => writer.as_bytes_mut(),
        }
    }
}
/// The creator source prepared these payload-free accounting nodes before
/// any file task or physical buffer. Queued writer teardown retains them.
pub(super) struct OrdinaryReadOwner {
    pub(super) occupancy: DiskReadOccupancy,
    pub(super) funding: HostMetadataFunding,
}
pub(super) type OrdinaryAttachment = PreparedAllocationOwner<OrdinaryReadOwner>;
pub(super) struct OrdinaryPreparation {
    pub(super) attachments: [Option<OrdinaryAttachment>; 2],
    pub(super) host: Option<eredu_runtime::cache::PreparedCachePoolReservation>,
    pub(super) transfer: Option<eredu_runtime::cache::PreparedCachePoolReservation>,
    pub(super) placements: [AllocationPlacement; 2],
    pub(super) identity: Arc<()>,
}
