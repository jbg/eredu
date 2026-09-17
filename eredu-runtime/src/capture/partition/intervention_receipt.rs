//! Shared fixed outcome transcript for ordinary and original partition work.
use super::{PartitionCaptureExchangeError as Error, PartitionCaptureExchangeStage};
use eredu_core::DistributedCommitEpoch;
use std::mem::{size_of, size_of_val};
const WORDS: usize = 18;

/// Borrowed wire source only. It carries no edit, allocation or outcome authority;
/// the enclosing operation retains its original plan, epoch and actual members.
pub(crate) struct PartitionInterventionReceipt<'a> {
    rank: usize,
    world: usize,
    epoch: DistributedCommitEpoch,
    operation: usize,
    members: &'a [usize],
    descriptor: &'a [u8; 32],
    routed: bool,
}
impl<'a> PartitionInterventionReceipt<'a> {
    pub const WORDS: usize = WORDS;
    const MAGIC: u32 = 0x4552_4953;
    pub fn new(rank: usize, world: usize, epoch: DistributedCommitEpoch, operation: usize,
        members: &'a [usize], descriptor: &'a [u8; 32], routed: bool) -> Result<Self, Error> {
        if world == 0 || rank >= world || u32::try_from(world).is_err()
            || u32::try_from(operation).is_err() || members.is_empty()
            || members.iter().any(|rank| *rank >= world)
            || members.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(Error::Protocol("intervention outcome source membership"));
        }
        Ok(Self { rank, world, epoch, operation, members, descriptor, routed })
    }
    pub fn frame(&self, accepted: bool, affected: u64) -> Result<[u32; WORDS], Error> {
        let member = self.members.contains(&self.rank);
        if affected != 0 && (!self.routed || !member || !accepted) {
            return Err(Error::Protocol("intervention outcome count has no accepted source"));
        }
        let mut frame = [0; Self::WORDS];
        frame[..8].copy_from_slice(&[Self::MAGIC, 1, self.rank as u32, self.world as u32,
            self.epoch.value() as u32, (self.epoch.value() >> 32) as u32,
            self.operation as u32, if member { u32::from(accepted) } else { 2 }]);
        for (word, bytes) in frame[8..16].iter_mut().zip(self.descriptor.chunks_exact(4)) {
            *word = u32::from_le_bytes(bytes.try_into().expect("digest word"));
        }
        frame[16] = affected as u32;
        frame[17] = (affected >> 32) as u32;
        Ok(frame)
    }
    /// Validate every identity and membership before accepting an outcome.
    /// A failed member stays a peer rejection; absence is never success.
    pub fn validate(&self, gathered: &[u32]) -> Result<u64, Error> {
        if self.world.checked_mul(Self::WORDS) != Some(gathered.len()) {
            return Err(Error::Protocol("intervention outcome frame extent"));
        }
        let mut digest = [0; 8];
        for (word, bytes) in digest.iter_mut().zip(self.descriptor.chunks_exact(4)) {
            *word = u32::from_le_bytes(bytes.try_into().expect("digest word"));
        }
        for (rank, peer) in gathered.chunks_exact(Self::WORDS).enumerate() {
            if peer[..7] != [Self::MAGIC, 1, rank as u32, self.world as u32,
                self.epoch.value() as u32, (self.epoch.value() >> 32) as u32, self.operation as u32]
                || peer[8..16] != digest
                || (!self.routed && peer[16..] != [0, 0])
                || (peer[7] != 1 && peer[16..] != [0, 0])
                || peer[7] > 2 || (peer[7] == 2) == self.members.contains(&rank) {
                return Err(Error::Protocol("intervention outcome identity or membership"));
            }
        }
        if let Some(rank) = gathered.chunks_exact(Self::WORDS).position(|peer| peer[7] == 0) {
            return Err(Error::PeerRejected { rank, stage: PartitionCaptureExchangeStage::Delivery });
        }
        gathered.chunks_exact(Self::WORDS).try_fold(0u64, |sum, peer|
            sum.checked_add(u64::from(peer[16]) | (u64::from(peer[17]) << 32))
                .ok_or_else(|| eredu_core::capture::CaptureError::Overflow.into()))
    }
    pub fn control_bytes() -> Option<usize> {
        let frames = [size_of::<Self>(), size_of::<[u32; WORDS]>(), size_of::<[u32; 8]>(),
            size_of::<Result<Self, Error>>(), size_of::<Result<[u32; WORDS], Error>>(),
            size_of::<Result<u64, Error>>(), size_of::<(&Self, &[u32])>(),
            size_of::<(usize, usize, DistributedCommitEpoch, usize, &[usize], &[u8; 32], bool)>(),
            size_of::<std::iter::Enumerate<std::slice::ChunksExact<'_, u32>>>(),
            size_of::<std::ops::Range<usize>>()];
        frames.into_iter().try_fold(size_of_val(&frames), usize::checked_add)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn outcome_transcript_requires_actual_members_and_preserves_rejection() {
        let digest = [0x31; 32];
        let members = [1usize, 3];
        let source = PartitionInterventionReceipt::new(0, 4, DistributedCommitEpoch::FIRST,
            7, &members, &digest, false).unwrap();
        let mut frames = Vec::new();
        for rank in 0..4 {
            let row = PartitionInterventionReceipt::new(rank, 4, DistributedCommitEpoch::FIRST,
                7, &members, &digest, false).unwrap();
            frames.extend_from_slice(&row.frame(true, 0).unwrap());
        }
        assert_eq!(source.validate(&frames).unwrap(), 0);
        for word in [2, 4, 6, 8, 16] {
            let mut foreign = frames.clone();
            foreign[word] ^= 1;
            assert!(source.validate(&foreign).is_err());
        }
        let mut failed = frames.clone();
        failed[PartitionInterventionReceipt::WORDS + 7] = 0;
        assert!(matches!(source.validate(&failed), Err(Error::PeerRejected { rank: 1, .. })));
        let mut absent = frames.clone();
        absent[PartitionInterventionReceipt::WORDS + 7] = 2;
        assert!(matches!(source.validate(&absent), Err(Error::Protocol(_))));
        assert!(source.validate(&frames[..frames.len()-1]).is_err());
        assert!(PartitionInterventionReceipt::new(0, 4, DistributedCommitEpoch::FIRST,
            7, &[1, 1], &digest, false).is_err());
        let sparse = PartitionInterventionReceipt::new(1, 4, DistributedCommitEpoch::FIRST,
            7, &members, &digest, true).unwrap();
        assert!(sparse.frame(false, 1).is_err());
        frames[PartitionInterventionReceipt::WORDS + 16] = 5;
        let sparse_receiver = PartitionInterventionReceipt::new(0, 4, DistributedCommitEpoch::FIRST,
            7, &members, &digest, true).unwrap();
        assert_eq!(sparse_receiver.validate(&frames).unwrap(), 5);
    }
}
