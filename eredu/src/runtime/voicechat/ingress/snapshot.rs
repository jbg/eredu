//! Exact host PCM queue copies; external admissions cannot be rewound.

use super::*;
use eredu_core::{execution_control::SnapshotEstimate, BackendFailure};
use eredu_runtime::execution_control::{
    snapshot_value_estimate, SnapshotReservation, SnapshotValue,
};

impl Ingress {
    pub(in crate::runtime::voicechat) fn retain_fork(
        &mut self,
        retention: Option<SnapshotReservation>,
    ) {
        self.fork_retention = retention;
    }
    pub(in crate::runtime::voicechat) fn fork_owner(
        mut self,
        retention: &SnapshotReservation,
    ) -> Rc<RefCell<Self>> {
        self.external = Rc::new(());
        self.fork_retention = Some(retention.clone());
        Rc::new(RefCell::new(self))
    }

    pub(in crate::runtime::voicechat) fn external_identity(&self) -> Rc<()> {
        self.external.clone()
    }
}

impl SnapshotValue for Ingress {
    fn estimate_snapshot_value(&self) -> Result<SnapshotEstimate, BackendFailure> {
        let allocated = self
            .queue
            .snapshot_storage_bytes()
            .ok_or_else(|| BackendFailure::from_error(AudioStreamError::SnapshotOverflow))?;
        // Forks continue accepting PCM through the same bounded sender. Reserve
        // its full logical capacity now, plus conservative growth slack, rather
        // than admitting only the carry present at the copy boundary.
        let maximum = u64::try_from(self.config.capacity_samples)
            .ok()
            .and_then(|samples| samples.checked_mul(2 * std::mem::size_of::<f32>() as u64))
            .and_then(|bytes| bytes.checked_add(std::mem::size_of::<PcmFrameQueue>() as u64))
            .ok_or_else(|| BackendFailure::from_error(AudioStreamError::SnapshotOverflow))?;
        let bytes = allocated
            .max(maximum)
            .checked_add(64)
            .ok_or_else(|| BackendFailure::from_error(AudioStreamError::SnapshotOverflow))?;
        snapshot_value_estimate::<Self>([Ok(SnapshotEstimate {
            retained_bytes: bytes,
            copy_bytes: bytes,
        })])
    }
    fn copy_snapshot_value(&self, _: &SnapshotReservation) -> Result<Self, BackendFailure> {
        Ok(Self {
            external: self.external.clone(),
            fork_retention: None,
            config: self.config,
            queue: self
                .queue
                .snapshot_copy()
                .map_err(BackendFailure::from_error)?,
            accepted: self.accepted,
            committed: self.committed,
            frames: self.frames,
            continuation_frames: self.continuation_frames,
            finish_declared: self.finish_declared,
            pending: self.pending,
            reserved: self.reserved,
            finish_committed: self.finish_committed,
            closed: self.closed,
        })
    }
}
