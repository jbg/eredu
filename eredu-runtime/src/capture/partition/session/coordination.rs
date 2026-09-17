use super::*;
use eredu_core::{BoundedCompletion, BoundedCompletionWait};

const MAGIC: u32 = 0x4552_4353;
const WORDS: usize = 16;

/// Prepaid, move-only coordination for all scheduled selections in one forward.
/// It freezes local preparation before the shared all-rank preparation vote.
/// Even a forward with no scheduled selections participates in one fixed frame.
pub struct SessionPartitionCoordination<'a, T: PartitionCaptureTransport> {
    epoch: DistributedCommitEpoch,
    transport: &'a T,
    wait: BoundedCompletionWait,
    participants: usize,
    rank: usize,
    digest: [u8; 32],
    reserved: CaptureUsage,
    // Last: all retained plan/work payloads retire before preparation custody.
    owner: Arc<crate::capture::CaptureHostOwner>,
}

impl<T: PartitionCaptureTransport> SessionPartitionCoordination<'_, T> {
    /// Global nonrefundable reservation made before any coordination work.
    pub const fn global_reserved(&self) -> CaptureUsage {
        self.reserved
    }
}

impl CaptureSession {
    /// Freezes the exact prepared selection set and reserves a fixed bounded
    /// identity/quota exchange on every rank. This submits no native work; call it
    /// during local observer preparation, before its all-rank success vote.
    pub fn prepare_partition_coordination<'a, T: PartitionCaptureTransport>(
        &mut self,
        transport: &'a T,
    ) -> Result<SessionPartitionCoordination<'a, T>, PartitionCaptureExchangeError>
    where
        T::Error: Send + Sync + 'static,
        <T::Completion as Completion>::Error: Send + Sync + 'static,
    {
        let reservation = self.reserve_partition_coordination(transport)?;
        self.seal_partition_coordination(reservation)
    }

    /// Reserve the mandatory common vote before optional producer allowances.
    pub(in crate::capture::partition) fn reserve_partition_coordination<
        'a,
        T: PartitionCaptureTransport,
    >(
        &mut self,
        transport: &'a T,
    ) -> Result<SessionPartitionCoordination<'a, T>, PartitionCaptureExchangeError>
    where
        T::Error: Send + Sync + 'static,
        <T::Completion as Completion>::Error: Send + Sync + 'static,
    {
        let epoch = self.partition_epoch()?;
        let run = self
            .partition
            .as_ref()
            .ok_or_else(|| invalid("partition capture identity is not bound"))?;
        let participants = run.identity.world_size;
        let rank = transport.capture_rank();
        if run.coordination_reserved
            || transport.participant_count() != participants
            || rank >= participants
        {
            return Err(invalid(
                "partition coordination is already prepared or has another topology",
            )
            .into());
        }
        transport.ensure_capture_active()?;
        let wait = transport.capture_wait()?;
        if !T::Completion::supports_cancellation(wait.cancellation()) {
            return Err(CaptureError::Unsupported(
                "partition coordination cancellation policy".into(),
            )
            .into());
        }
        let local = super::super::exchange::gather_usage(transport, participants, WORDS)?
            .checked_add(CaptureUsage {
                host_bytes: 8192,
                ..Default::default()
            })?;
        let reserved = local.checked_mul(participants as u64)?;
        // These credits belong to this one fixed exchange. They cannot be spent
        // by a producer, receipt decoder, another epoch, or a second coordination.
        self.ledger.reserve_quota(reserved)?;
        self.partition
            .as_mut()
            .expect("checked binding")
            .coordination_reserved = true;
        Ok(SessionPartitionCoordination {
            owner: Arc::clone(&self.owner),
            epoch,
            transport,
            wait,
            participants,
            rank,
            digest: [0; 32],
            reserved,
        })
    }

    /// Freeze successful claims, skipped selections and all nonrefundable costs.
    pub(in crate::capture::partition) fn seal_partition_coordination<
        'a,
        T: PartitionCaptureTransport,
    >(
        &mut self,
        mut coordination: SessionPartitionCoordination<'a, T>,
    ) -> Result<SessionPartitionCoordination<'a, T>, PartitionCaptureExchangeError> {
        let epoch = self.partition_epoch()?;
        let run = self
            .partition
            .as_ref()
            .ok_or_else(|| invalid("unbound partition"))?;
        if !Arc::ptr_eq(&self.owner, &coordination.owner)
            || epoch != coordination.epoch
            || !run.coordination_reserved
            || run.coordination_prepared
        {
            return Err(invalid(
                "partition coordination reservation belongs to another owner or forward",
            )
            .into());
        }
        let context = run.identity.context(
            &self.plan,
            0,
            self.phase,
            self.prediction,
            epoch,
            self.invocation,
        );
        let mut digest = Sha256::new();
        digest.update(b"eredu-partition-capture-step-v3\0");
        digest.update(
            serde_json::to_vec(&context)
                .map_err(|_| invalid("partition coordination context encoding"))?,
        );
        digest.update((run.claimed.len() as u64).to_le_bytes());
        for (key, descriptor) in &run.claimed {
            key.hash(&mut digest);
            digest.update(descriptor);
        }
        digest.update((run.skipped.len() as u64).to_le_bytes());
        for (index, (_, reason, charged)) in &run.skipped {
            digest.update((*index as u64).to_le_bytes());
            digest.update(
                serde_json::to_vec(reason)
                    .map_err(|_| invalid("partition skip reason encoding"))?,
            );
            hash_usage(&mut digest, *charged);
        }
        // Compare global intervention intent, never process-local session IDs.
        // Owner/admission authority remains checked separately on every rank.
        if let Some(run) = &self.interventions {
            digest.update([1]);
            digest.update(run.plan.intent_identity().as_bytes());
        } else {
            digest.update([0]);
        }
        digest.update((run.intervention_claimed.len() as u64).to_le_bytes());
        for (index, descriptor) in &run.intervention_claimed {
            digest.update((*index as u64).to_le_bytes());
            digest.update(descriptor);
        }
        hash_usage(&mut digest, self.ledger.step());
        hash_usage(&mut digest, self.ledger.total());
        coordination.digest = digest.finalize().into();
        self.partition
            .as_mut()
            .expect("checked binding")
            .coordination_prepared = true;
        Ok(coordination)
    }

    /// Common pre-forward coordination after every rank successfully prepared.
    /// A completed identity/quota disagreement rejects this step on every rank
    /// without poisoning a healthy communication owner. Native/protocol failures
    /// retain their ordinary failure and completion ownership.
    pub fn coordinate_partition_capture<T: PartitionCaptureTransport>(
        &mut self,
        coordination: SessionPartitionCoordination<'_, T>,
    ) -> Result<(), PartitionCaptureExchangeError>
    where
        T::Error: Send + Sync + 'static,
        <T::Completion as Completion>::Error: Send + Sync + 'static,
    {
        if !Arc::ptr_eq(&self.owner, &coordination.owner)
            || self.partition_epoch()? != coordination.epoch
            || !self
                .partition
                .as_ref()
                .is_some_and(|run| run.coordination_prepared && !run.coordinated)
        {
            return Err(
                invalid("partition coordination belongs to another owner or forward").into(),
            );
        }
        let result = coordination.exchange();
        match result {
            Ok(()) => {
                self.partition
                    .as_mut()
                    .expect("checked binding")
                    .coordinated = true;
                Ok(())
            }
            Err(error) => {
                if !matches!(
                    error,
                    PartitionCaptureExchangeError::PeerRejected {
                        stage: PartitionCaptureExchangeStage::Coordination,
                        ..
                    }
                ) {
                    coordination.transport.fail_capture_exchange(&error);
                }
                Err(error)
            }
        }
    }
}

impl<T: PartitionCaptureTransport> SessionPartitionCoordination<'_, T>
where
    T::Error: Send + Sync + 'static,
    <T::Completion as Completion>::Error: Send + Sync + 'static,
{
    fn exchange(&self) -> Result<(), PartitionCaptureExchangeError> {
        exchange_coordination(self.transport, self.wait, self.rank, self.participants,
            self.epoch, &self.digest)
    }
}

// Both receipt owners use the identical reached frame and comparison worker.
pub(in crate::capture::partition) fn exchange_coordination<T: PartitionCaptureTransport>(
    transport: &T, wait: BoundedCompletionWait, rank: usize, participants: usize,
    epoch: DistributedCommitEpoch, digest: &[u8; 32],
) -> Result<(), PartitionCaptureExchangeError>
where T::Error: Send + Sync + 'static,
    <T::Completion as Completion>::Error: Send + Sync + 'static,
{
        transport.ensure_capture_active()?;
        let mut frame = [0u32; WORDS];
        frame[..6].copy_from_slice(&[
            MAGIC,
            2,
            rank as u32,
            participants as u32,
            epoch.value() as u32,
            (epoch.value() >> 32) as u32,
        ]);
        for (word, bytes) in frame[6..14].iter_mut().zip(digest.chunks_exact(4)) {
            *word = u32::from_le_bytes(bytes.try_into().expect("digest word"));
        }
        let source = PartitionCaptureFrame::new(PartitionCaptureFrameKind::Coordination,
            rank, participants, &frame, WORDS)?;
        let gathered = super::super::exchange::gather_capture_words(transport, wait, &source)?;
        for (rank, peer) in gathered.chunks_exact(WORDS).enumerate() {
            if peer[..6]
                != [
                    MAGIC,
                    2,
                    rank as u32,
                    participants as u32,
                    epoch.value() as u32,
                    (epoch.value() >> 32) as u32,
                ]
                || peer[14..] != [0, 0]
            {
                return Err(PartitionCaptureExchangeError::Protocol(
                    "capture coordination version, world or epoch",
                ));
            }
        }
        // Compare the same rank-major transcript on every rank. A mismatch is a
        // completed common rejection, not merely this rank's private opinion.
        if let Some(rank) = gathered
            .chunks_exact(WORDS)
            .position(|peer| peer[6..14] != gathered[6..14])
        {
            return Err(PartitionCaptureExchangeError::PeerRejected {
                rank,
                stage: PartitionCaptureExchangeStage::Coordination,
            });
        }
        Ok(())
}
