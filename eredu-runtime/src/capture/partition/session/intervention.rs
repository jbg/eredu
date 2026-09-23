//! Global activation operations bound to the same live owner and forward as capture.
use super::*;
use crate::intervention::{PartitionActivationLayout, ReservedPartitionActivation};
use eredu_core::{
    intervention::{InterventionBackend, InterventionOutcome},
    BoundedCompletion, BoundedCompletionOutcome, BoundedCompletionWait,
};

const RECEIPT_WORDS: usize = 18;
const RECEIPT_MAGIC: u32 = 0x4552_4953;
mod routed;

/// Move-only authority for one globally admitted activation operation. Every
/// actual invocation member, including replicas and empty shards, acknowledges
/// it. Pipeline nonmembers join only the shared completion boundary.
pub struct SessionPartitionIntervention<'a, T: PartitionCaptureHookTransport> {
    owner: Arc<()>,
    epoch: DistributedCommitEpoch,
    pub(in crate::capture::partition) index: usize,
    plan_identity: String,
    transport: &'a T,
    members: Vec<usize>,
    wait: BoundedCompletionWait,
    local: Option<ReservedPartitionActivation>,
    routed: Option<routed::SparseWork>,
    attempted: bool,
    accepted: bool,
    descriptor: [u8; 32],
    charged: CaptureUsage,
    evidence: Vec<SessionPartitionCapture<'a, T>>,
}

// Projection preparation runs on every rank. Native retention is charged once
// per actual member; replicated host preparation is charged on every world rank.
struct GlobalProjectionReservation<'a> {
    ledger: &'a mut CaptureLedger,
    world: u64,
}
impl CaptureReservation for GlobalProjectionReservation<'_> {
    fn reserve(
        &mut self,
        mut usage: CaptureUsage,
    ) -> Result<Option<CaptureSkipReason>, CaptureError> {
        usage.host_bytes = mul(usage.host_bytes, self.world)?;
        self.ledger.reserve(usage)
    }
}

impl CaptureSession {
    pub(crate) fn validate_ordinary_intervention(&self) -> Result<(), CaptureError> {
        if self.partition.is_some() {
            return Err(invalid(
                "partition interventions require live coordinated operation authority",
            ));
        }
        Ok(())
    }
    pub(crate) fn partition_intervention_completed(&self, index: usize) -> bool {
        matches!(self.transaction, Some((epoch, CaptureTransactionStatus::Pending))
            if self.partition.as_ref().is_some_and(|run| run.intervention_completed(index, epoch)))
    }
    /// Prepare the actual member projections, dependency execution, two group
    /// votes and a bounded world outcome frame before model work. Costs and
    /// global intent join the ordinary pre-forward coordination digest.
    pub fn prepare_partition_intervention<'a, B, T, L>(
        &mut self,
        transport: &'a T,
        layout: &L,
        backend: &B,
        index: usize,
        limits: PartitionCaptureReceiptLimits,
    ) -> Result<SessionPartitionIntervention<'a, T>, PartitionCaptureExchangeError>
    where
        B: InterventionBackend,
        T: PartitionCaptureHookTransport,
        L: PartitionActivationLayout + PartitionCaptureLayout,
        T::Error: Send + Sync + 'static,
        <T::Completion as Completion>::Error: Send + Sync + 'static,
    {
        let epoch = self.partition_epoch()?;
        let partition = self
            .partition
            .as_ref()
            .ok_or_else(|| invalid("unbound partition"))?;
        let world = partition.identity.world_size;
        let rank = transport.capture_rank();
        if transport.participant_count() != world
            || rank >= world
            || partition.coordination_prepared
            || partition.intervention_claimed.contains_key(&index)
        {
            return Err(invalid("partition intervention requires fresh local authority").into());
        }
        let run = self
            .interventions
            .as_ref()
            .ok_or_else(|| invalid("no intervention admission"))?;
        if run.plan.artifact_identity() != partition.identity.artifact {
            return Err(
                invalid("partition intervention belongs to another prepared artifact").into(),
            );
        }
        let record = run
            .records
            .as_ref()
            .and_then(|records| records.get(index))
            .ok_or_else(|| invalid("no scheduled intervention record"))?;
        if record.outcome != InterventionOutcome::Missing {
            return Err(invalid("partition intervention is not scheduled").into());
        }
        if run.plan.points()[index].routed_units.is_some() {
            return self
                .prepare_partition_routed_intervention(transport, layout, backend, index, limits);
        }
        let wait = transport.capture_wait()?;
        if !T::Completion::supports_cancellation(wait.cancellation()) {
            return Err(CaptureError::Unsupported(
                "partition intervention cancellation policy".into(),
            )
            .into());
        }
        transport.ensure_capture_active()?;
        // Bound all metadata before architecture projection allocates its region
        // lists. Host payload copies are separately prepaid by each projection.
        let operation = &run.plan.plan().operations[index];
        let point = &run.plan.points()[index];
        let shape = run
            .plan
            .geometry_at(self.phase, self.prediction, self.invocation)?
            .resolve(&point.observation_geometry())?
            .ok_or_else(|| invalid("partition intervention source geometry is unknown"))?;
        run.plan.validate_at(
            index,
            self.phase,
            self.prediction,
            self.invocation,
            &shape,
            operation.action.dtype(),
        )?;
        let regions = layout.activation_region_bound(
            &run.plan,
            index,
            limits.max_producers,
            limits.max_fragments,
        )?;
        if regions == 0 || regions > limits.max_fragments {
            return Err(invalid("invalid partition intervention preparation bound").into());
        }
        let metadata = CaptureUsage {
            host_bytes: mul(
                add(
                    4096,
                    add(
                        mul(limits.max_producers as u64, 512)?,
                        mul(regions as u64, add(256, mul(shape.len() as u64, 128)?)?)?,
                    )?,
                )?,
                world as u64,
            )?,
            ..Default::default()
        };
        self.ledger.reserve_quota(metadata)?;
        let projections = layout.activation_members_at(
            &run.plan,
            index,
            self.phase,
            self.prediction,
            self.invocation,
            limits.max_producers,
            regions,
        )?;
        let members: Vec<_> = projections.iter().map(|member| member.rank).collect();
        if members.is_empty()
            || members.windows(2).any(|pair| pair[0] >= pair[1])
            || members.iter().any(|member| *member >= world)
        {
            return Err(invalid("invalid partition intervention invocation membership").into());
        }
        let mut digest = Sha256::new();
        digest.update(b"eredu-partition-intervention-v1\0");
        digest.update(run.plan.intent_identity().as_bytes());
        digest.update((index as u64).to_le_bytes());
        digest.update((members.len() as u64).to_le_bytes());
        let mut local = None;
        let mut charged = metadata;
        for member in projections {
            digest.update(member.projection.geometry_identity());
            let shape = member.projection.local_shape();
            digest.update((member.rank as u64).to_le_bytes());
            digest.update((shape.len() as u64).to_le_bytes());
            for dimension in shape {
                digest.update(dimension.to_le_bytes());
            }
            let dependencies = backend
                .estimate_partition_source(shape, wait)?
                .checked_mul(2)?;
            self.ledger.reserve_quota(dependencies)?;
            hash_usage(&mut digest, dependencies);
            charged = charged.checked_add(dependencies)?;
            let work = member.projection.reserve(
                &mut GlobalProjectionReservation {
                    ledger: &mut self.ledger,
                    world: world as u64,
                },
                run.estimator.as_ref(),
            )?;
            let mut usage = work.charged();
            usage.host_bytes = mul(usage.host_bytes, world as u64)?;
            hash_usage(&mut digest, usage);
            charged = charged.checked_add(usage)?;
            if member.rank == rank {
                local = Some(work);
            }
        }
        let vote = transport
            .estimate_capture_hook(&members)?
            .checked_mul(mul(members.len() as u64, 2)?)?;
        let receipt = super::super::exchange::gather_usage(transport, world, RECEIPT_WORDS)?
            .checked_mul(world as u64)?;
        let coordination = vote.checked_add(receipt)?;
        self.ledger.reserve_quota(coordination)?;
        charged = charged.checked_add(coordination)?;
        hash_usage(&mut digest, charged);
        let descriptor = digest.finalize().into();
        let plan_identity = run.plan.identity().to_owned();
        charged = charged.checked_add(record.charged)?;
        self.partition
            .as_mut()
            .expect("validated binding")
            .intervention_claimed
            .insert(index, descriptor);
        let evidence = self.prepare_intervention_evidence(transport, layout, index, limits)?;
        Ok(SessionPartitionIntervention {
            owner: Arc::clone(&self.owner),
            epoch,
            index,
            plan_identity,
            transport,
            members,
            wait,
            local,
            routed: None,
            attempted: false,
            accepted: false,
            descriptor,
            charged,
            evidence,
        })
    }

    fn validate_partition_intervention<T: PartitionCaptureHookTransport>(
        &self,
        work: &SessionPartitionIntervention<'_, T>,
    ) -> Result<(), CaptureError> {
        if !Arc::ptr_eq(&self.owner, &work.owner)
            || self.partition_epoch()? != work.epoch
            || !self.partition.as_ref().is_some_and(|run| {
                run.coordinated
                    && run.intervention_claimed.get(&work.index) == Some(&work.descriptor)
                    && !run.intervention_completed.contains_key(&work.index)
            })
            || !self
                .interventions
                .as_ref()
                .is_some_and(|run| run.plan.identity() == work.plan_identity)
        {
            return Err(invalid(
                "partition intervention belongs to another owner, admission or forward",
            ));
        }
        Ok(())
    }

    /// Preflight local geometry on all invocation members before evaluating a
    /// dependency, apply the prepaid projection, and settle the updated value
    /// before the final group vote permits later model collectives.
    pub fn apply_partition_intervention<B: InterventionBackend, T: PartitionCaptureHookTransport>(
        &mut self,
        work: &mut SessionPartitionIntervention<'_, T>,
        backend: &mut B,
        input: &B::Tensor,
    ) -> Result<Option<B::Tensor>, PartitionCaptureObserverError<B::Error>>
    where
        T::Error: Send + Sync + 'static,
        <T::Completion as Completion>::Error: Send + Sync + 'static,
    {
        self.validate_partition_intervention(work)
            .map_err(CaptureExecutionError::Admission)?;
        if work.routed.is_some()
            || work.attempted
            || !work.members.contains(&work.transport.capture_rank())
        {
            return Err(CaptureExecutionError::Admission(invalid(
                "partition intervention reached an absent or repeated invocation",
            ))
            .into());
        }
        work.attempted = true;
        let local = work.local.take().ok_or_else(|| {
            CaptureExecutionError::Admission(invalid("missing local intervention projection"))
        })?;
        let valid = local.validate_source(backend, input);
        let agreed = hook::agree(work.transport, &work.members, work.wait, valid.is_ok());
        let result = (|| {
            valid?;
            if !agreed? {
                return Err(PartitionCaptureObserverError::HookRejected);
            }
            prepare_value(work, backend, input)?;
            if let Some(evidence) = work.evidence.first_mut() {
                self.observe_partition(evidence, backend, input)?;
            }
            let output = local.apply(backend, input)?;
            if let Some(output) = &output {
                prepare_value(work, backend, output)?;
            }
            if let Some(evidence) = work.evidence.get_mut(1) {
                self.observe_partition(evidence, backend, output.as_ref().unwrap_or(input))?;
            }
            Ok(output)
        })();
        let accepted = hook::agree(work.transport, &work.members, work.wait, result.is_ok());
        let result = result.and_then(|output| {
            if !accepted? {
                return Err(PartitionCaptureObserverError::HookRejected);
            }
            work.accepted = true;
            Ok(output)
        });
        if let Err(error) = &result {
            if let Some(record) = self
                .interventions
                .as_mut()
                .and_then(|run| run.records.as_mut())
                .and_then(|records| records.get_mut(work.index))
            {
                record.outcome = InterventionOutcome::Failed {
                    message: bounded_diagnostic(error),
                };
            }
        }
        result
    }

    /// World receipt delivery at the shared completed-forward boundary. A missing
    /// active invocation is a common rejection; inactive pipeline ranks cannot
    /// certify it by merely finishing their own stage. Publication waits for commit.
    pub fn complete_partition_intervention<T: PartitionCaptureHookTransport>(
        &mut self,
        mut work: SessionPartitionIntervention<'_, T>,
    ) -> Result<(), PartitionCaptureExchangeError>
    where
        T::Error: Send + Sync + 'static,
        <T::Completion as Completion>::Error: Send + Sync + 'static,
    {
        self.validate_partition_intervention(&work)?;
        let affected = match work.exchange() {
            Ok(affected) => affected,
            Err(error) => {
                if !matches!(error, PartitionCaptureExchangeError::PeerRejected { .. }) {
                    work.transport.fail_capture_exchange(&error);
                }
                return Err(error);
            }
        };
        if let Some(routed) = &work.routed {
            if affected > routed.maximum_affected()? {
                let error = PartitionCaptureExchangeError::Protocol(
                    "sparse outcome count exceeds admitted logical values",
                );
                work.transport.fail_capture_exchange(&error);
                return Err(error);
            }
            let record = &mut self
                .interventions
                .as_mut()
                .expect("retained plan")
                .records
                .as_mut()
                .expect("active records")[work.index];
            record.routed_units = Some(eredu_core::intervention::RoutedUnitInterventionReceipt {
                source_tokens: routed.source_tokens,
                completed_tokens: routed.source_tokens,
                affected_values: affected,
            });
        }
        for evidence in work.evidence.drain(..) {
            self.complete_partition_capture(evidence)?;
        }
        self.partition
            .as_mut()
            .expect("validated binding")
            .intervention_completed
            .insert(work.index, (work.epoch, work.charged));
        Ok(())
    }
}

fn prepare_value<B: InterventionBackend, T: PartitionCaptureHookTransport>(
    work: &SessionPartitionIntervention<'_, T>,
    backend: &mut B,
    value: &B::Tensor,
) -> Result<(), PartitionCaptureObserverError<B::Error>> {
    match backend.prepare_partition_source(value, work.wait) {
        Ok(BoundedCompletionOutcome::Completed) => Ok(()),
        Ok(BoundedCompletionOutcome::DeadlineExceeded { cancellation }) => {
            let error = PartitionCaptureExchangeError::Deadline { cancellation };
            work.transport.fail_capture_exchange(&error);
            Err(error.into())
        }
        Err(error) => {
            work.transport
                .fail_capture_exchange(&PartitionCaptureExchangeError::Protocol(
                    "intervention dependency failed",
                ));
            Err(CaptureExecutionError::Backend(error).into())
        }
    }
}

impl<T: PartitionCaptureHookTransport> SessionPartitionIntervention<'_, T>
where
    T::Error: Send + Sync + 'static,
    <T::Completion as Completion>::Error: Send + Sync + 'static,
{
    fn exchange(&self) -> Result<u64, PartitionCaptureExchangeError> {
        self.transport.ensure_capture_active()?;
        let rank = self.transport.capture_rank();
        let world = self.transport.participant_count();
        let status = if self.members.contains(&rank) {
            u32::from(self.accepted)
        } else {
            2
        };
        let mut frame = [0u32; RECEIPT_WORDS];
        frame[..8].copy_from_slice(&[
            RECEIPT_MAGIC,
            1,
            rank as u32,
            world as u32,
            self.epoch.value() as u32,
            (self.epoch.value() >> 32) as u32,
            self.index as u32,
            status,
        ]);
        for (word, bytes) in frame[8..16]
            .iter_mut()
            .zip(self.descriptor.as_chunks::<4>().0.iter())
        {
            *word = u32::from_le_bytes(*bytes);
        }
        let affected = self.routed.as_ref().map_or(0, |work| work.affected);
        frame[16] = affected as u32;
        frame[17] = (affected >> 32) as u32;
        let gathered =
            super::super::exchange::gather_capture_words(self.transport, self.wait, world, &frame)?;
        for (rank, peer) in gathered.as_chunks::<RECEIPT_WORDS>().0.iter().enumerate() {
            if peer[..7]
                != [
                    RECEIPT_MAGIC,
                    1,
                    rank as u32,
                    world as u32,
                    frame[4],
                    frame[5],
                    frame[6],
                ]
                || peer[8..16] != frame[8..16]
                || (self.routed.is_none() && peer[16..] != [0, 0])
                || (peer[7] != 1 && peer[16..] != [0, 0])
                || peer[7] > 2
                || (peer[7] == 2) == self.members.contains(&rank)
            {
                return Err(PartitionCaptureExchangeError::Protocol(
                    "intervention outcome identity or membership",
                ));
            }
        }
        if let Some(rank) = gathered
            .as_chunks::<RECEIPT_WORDS>()
            .0
            .iter()
            .position(|peer| peer[7] == 0)
        {
            return Err(PartitionCaptureExchangeError::PeerRejected {
                rank,
                stage: PartitionCaptureExchangeStage::Delivery,
            });
        }
        gathered
            .as_chunks::<RECEIPT_WORDS>()
            .0
            .iter()
            .try_fold(0u64, |sum, peer| {
                add(sum, u64::from(peer[16]) | (u64::from(peer[17]) << 32)).map_err(Into::into)
            })
    }
}
