//! A prepaid status vote on the ranks that execute one observation boundary.
use super::*;
use eredu_core::{
    BackendFailure, BoundedCompletion, BoundedCompletionWait, BoundedSubmissionOutcome, Submission,
};

/// Native status agreement on an exact architecture-declared invocation group.
/// This is separate from world receipt transport: inactive pipeline ranks must
/// never enter a hook collective. Implementations validate retained membership
/// and the selected failure-agreement requirement before returning a cold bound.
pub trait PartitionCaptureHookTransport: PartitionCaptureTransport {
    /// Status whose host resolution is permitted only after exact completion.
    type HookOutput;
    /// Cold per-member logical bound, including completion and host resolution.
    /// Called on every world rank, including nonmembers. Must submit no work.
    fn estimate_capture_hook(&self, members: &[usize]) -> Result<CaptureUsage, CaptureError>;
    /// Submits one payload-free success vote on the declared invocation group.
    /// Runtime settles singleton groups locally without calling this method.
    fn submit_capture_hook(
        &self,
        members: &[usize],
        success: bool,
    ) -> Result<Submission<Self::HookOutput, Self::Completion>, Self::Error>;
    /// True only when every invocation-group member reported success.
    fn resolve_capture_hook(&self, output: Self::HookOutput) -> Result<bool, Self::Error>;
}

/// Move-only error-vote credits, isolated from producer and receipt allowances.
/// Membership is fixed before common pre-forward coordination. Replicated
/// nonproducers participate; inactive pipeline ranks do not.
pub struct SessionPartitionHook<'a, T: PartitionCaptureHookTransport> {
    owner: Arc<()>,
    epoch: DistributedCommitEpoch,
    key: PartitionCaptureKey,
    transport: &'a T,
    members: Vec<usize>,
    wait: BoundedCompletionWait,
    reserved: CaptureUsage,
    source_preflight: bool,
}

impl<T: PartitionCaptureHookTransport> SessionPartitionHook<'_, T> {
    /// All invocation members' nonrefundable vote and metadata reservation.
    pub const fn global_reserved(&self) -> CaptureUsage {
        self.reserved
    }
    /// Whether this world rank must encounter the selected observation hook.
    pub fn participates(&self) -> bool {
        self.members.contains(&self.transport.capture_rank())
    }
}

impl CaptureSession {
    /// Adds an exact invocation-group vote to a locally prepared selection.
    /// This must precede `prepare_partition_coordination`; its membership and
    /// costs become part of the common selection digest. No collective is run.
    pub fn prepare_partition_hook<'a, T: PartitionCaptureHookTransport>(
        &mut self,
        work: &mut SessionPartitionCapture<'a, T>,
        members: Vec<usize>,
    ) -> Result<SessionPartitionHook<'a, T>, PartitionCaptureExchangeError>
    where
        T::Error: Send + Sync + 'static,
        <T::Completion as Completion>::Error: Send + Sync + 'static,
    {
        self.prepare_partition_vote(work, members, false)
    }

    pub(super) fn prepare_partition_vote<'a, T: PartitionCaptureHookTransport>(
        &mut self,
        work: &mut SessionPartitionCapture<'a, T>,
        members: Vec<usize>,
        source_preflight: bool,
    ) -> Result<SessionPartitionHook<'a, T>, PartitionCaptureExchangeError>
    where
        T::Error: Send + Sync + 'static,
        <T::Completion as Completion>::Error: Send + Sync + 'static,
    {
        let epoch = self.partition_epoch()?;
        let run = self
            .partition
            .as_ref()
            .ok_or_else(|| invalid("partition capture identity is not bound"))?;
        if !Arc::ptr_eq(&self.owner, &work.owner)
            || epoch != work.epoch
            || run.coordination_prepared
            || if source_preflight {
                work.source_accepted.is_some()
            } else {
                work.hook_accepted.is_some()
            }
            || !run.claimed.contains_key(&work.key)
        {
            return Err(invalid("partition hook requires fresh local selection authority").into());
        }
        if members.is_empty()
            || members.windows(2).any(|pair| pair[0] >= pair[1])
            || members.iter().any(|rank| *rank >= run.identity.world_size)
            || work
                .exchange
                .receipt_plan()
                .producers()
                .any(|(rank, _)| !members.contains(&rank))
        {
            return Err(invalid(
                "partition hook membership does not cover its producers exactly once",
            )
            .into());
        }
        let transport = work.exchange.transport();
        let wait = transport.capture_wait()?;
        if !T::Completion::supports_cancellation(wait.cancellation()) {
            return Err(
                CaptureError::Unsupported("partition hook cancellation policy".into()).into(),
            );
        }
        let native = transport.estimate_capture_hook(&members)?;
        let metadata = CaptureUsage {
            host_bytes: add(1024, mul(members.len() as u64, 8)?)?,
            ..Default::default()
        };
        let reserved = native
            .checked_mul(members.len() as u64)?
            .checked_add(metadata.checked_mul(run.identity.world_size as u64)?)?;
        // Deliberately discard the granted quota: the token itself authorizes
        // one fixed vote, and no source factory can consume its allowance.
        self.ledger.reserve_quota(reserved)?;
        let descriptor = self
            .partition
            .as_mut()
            .expect("checked binding")
            .claimed
            .get_mut(&work.key)
            .expect("checked selection");
        let mut digest = Sha256::new();
        digest.update(b"eredu-partition-capture-hook-v1\0");
        digest.update([u8::from(source_preflight)]);
        digest.update(*descriptor);
        digest.update((members.len() as u64).to_le_bytes());
        for rank in &members {
            digest.update((*rank as u64).to_le_bytes());
        }
        hash_usage(&mut digest, native);
        *descriptor = digest.finalize().into();
        if source_preflight {
            work.source_accepted = Some(!members.contains(&transport.capture_rank()));
        } else {
            work.hook_accepted = Some(!members.contains(&transport.capture_rank()));
        }
        Ok(SessionPartitionHook {
            owner: Arc::clone(&self.owner),
            epoch,
            key: work.key,
            transport,
            members,
            wait,
            reserved,
            source_preflight,
        })
    }

    /// Settles local capture success before an active tensor group can proceed
    /// to its next model collective. A completed false decision is recoverable
    /// by the enclosing model transaction; transport failure fences its owner.
    /// The caller retains and propagates its original local capture error.
    pub fn agree_partition_hook<T: PartitionCaptureHookTransport>(
        &mut self,
        work: &mut SessionPartitionCapture<'_, T>,
        hook: SessionPartitionHook<'_, T>,
        local_success: bool,
    ) -> Result<bool, PartitionCaptureExchangeError>
    where
        T::Error: Send + Sync + 'static,
        <T::Completion as Completion>::Error: Send + Sync + 'static,
    {
        if hook.source_preflight {
            return Err(invalid("source preflight cannot authorize completed capture").into());
        }
        self.agree_partition_vote(work, hook, local_success)
    }

    pub(super) fn agree_partition_vote<T: PartitionCaptureHookTransport>(
        &mut self,
        work: &mut SessionPartitionCapture<'_, T>,
        hook: SessionPartitionHook<'_, T>,
        local_success: bool,
    ) -> Result<bool, PartitionCaptureExchangeError>
    where
        T::Error: Send + Sync + 'static,
        <T::Completion as Completion>::Error: Send + Sync + 'static,
    {
        self.validate_partition_work(work)?;
        if !Arc::ptr_eq(&self.owner, &hook.owner)
            || hook.epoch != work.epoch
            || hook.key != work.key
            || if hook.source_preflight {
                work.source_accepted != Some(false)
            } else {
                work.hook_accepted != Some(false) || !work.observed
            }
            || !hook.participates()
        {
            return Err(invalid("partition hook belongs to another source or invocation").into());
        }
        let result = agree(
            work.exchange.transport(),
            &hook.members,
            hook.wait,
            local_success,
        );
        match result {
            Ok(success) => {
                // Source completion has a separate owner; a successful cold
                // geometry vote must not certify native source preparation.
                if !hook.source_preflight {
                    work.hook_accepted = Some(success);
                }
                Ok(success)
            }
            Err(error) => Err(error),
        }
    }
}

/// Native vote mechanism shared by capture and intervention authorities.
pub(super) fn agree<T: PartitionCaptureHookTransport>(
    transport: &T,
    members: &[usize],
    wait: BoundedCompletionWait,
    local_success: bool,
) -> Result<bool, PartitionCaptureExchangeError>
where
    T::Error: Send + Sync + 'static,
    <T::Completion as Completion>::Error: Send + Sync + 'static,
{
    let result = (|| {
        transport.ensure_capture_active()?;
        if members.len() == 1 {
            return Ok(local_success);
        }
        let submission = transport
            .submit_capture_hook(members, local_success)
            .map_err(BackendFailure::from_error)?;
        let output = match submission
            .wait_bounded(wait)
            .map_err(BackendFailure::from_error)?
        {
            BoundedSubmissionOutcome::Completed(output) => output,
            BoundedSubmissionOutcome::DeadlineExceeded { cancellation } => {
                return Err(PartitionCaptureExchangeError::Deadline { cancellation });
            }
        };
        let success = transport
            .resolve_capture_hook(output)
            .map_err(BackendFailure::from_error)?;
        if success && !local_success {
            return Err(PartitionCaptureExchangeError::Protocol(
                "hook agreement omitted local failure",
            ));
        }
        Ok(success)
    })();
    if let Err(error) = &result {
        transport.fail_capture_exchange(error);
    }
    result
}
