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

#[cfg(test)]
mod tests {
    use super::*;
    use eredu_core::execution_control::{SnapshotLimits, SnapshotResourceKind};
    use eredu_runtime::execution_control::SnapshotBudget;

    #[test]
    fn empty_fork_admits_full_future_pcm_capacity_and_sender_retains_its_charge() {
        let config = VoiceChatInputConfig {
            capacity_samples: 3 * INPUT_FRAME_SAMPLES,
            pcm_precision: PcmPrecision::F32,
        };
        let (source, _) = Ingress::new(config).unwrap();
        let estimate = source.borrow().estimate_snapshot_value().unwrap();
        let budget = SnapshotBudget::new(SnapshotLimits {
            max_snapshots: 0,
            max_branches: 1,
            retained_bytes: estimate.retained_bytes,
            cumulative_copy_bytes: estimate.copy_bytes,
        });
        let retention = budget
            .reserve(SnapshotResourceKind::Branch, Some(estimate))
            .unwrap();
        let fork = source
            .borrow()
            .copy_snapshot_value(&retention)
            .unwrap()
            .fork_owner(&retention);
        let mut sender = VoiceChatInputSender::from_owner(fork.clone());
        drop(retention);
        let charged = budget.usage();
        sender
            .submit(&vec![0.25; config.capacity_samples], INPUT_SAMPLE_RATE)
            .unwrap();
        assert!(fork.borrow().queue.snapshot_storage_bytes().unwrap() <= charged.retained_bytes);
        assert_eq!(budget.usage(), charged);
        assert!(matches!(
            sender.submit(&[0.5], INPUT_SAMPLE_RATE),
            Err(VoiceChatInputError::Audio(
                AudioStreamError::Backpressure { .. }
            ))
        ));
        assert_eq!(
            sender.status().accepted_samples,
            config.capacity_samples as u64
        );
        drop(fork);
        assert_eq!(
            budget.usage(),
            charged,
            "live sender retains the child allocation"
        );
        drop(sender);
        assert_eq!(budget.usage().retained_bytes, 0);
        assert_eq!(
            budget.usage().cumulative_copy_bytes,
            charged.cumulative_copy_bytes
        );
    }
}
