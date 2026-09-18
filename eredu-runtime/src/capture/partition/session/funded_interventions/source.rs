//! Finite original member/window transcript; no native result is synthesized.
use super::*;
use super::super::funded_coordination::HashWriter;
/// One actual architecture member and its retained post-window native program.
/// All usages remain descriptive until the original ledger prepays this table.
pub struct PartitionInterventionMemberSource<'a> {
    pub rank: usize,
    pub projection: &'a PreparedPartitionInterventionProjection,
    pub shape: &'a [u64],
    /// Ordered update geometry/payload identity after component/window projection.
    pub execution_identity: [u8; 32],
    pub usage: CaptureUsage,
    pub projection_usage: [CaptureUsage; 2],
    pub source_usage: CaptureUsage,
}
/// A terminal/decode callback, or one exact canonical prefill window.
pub struct PartitionInterventionInvocationSource<'a> {
    pub window: Option<InterventionPrefillWindow>,
    pub members: &'a [PartitionInterventionMemberSource<'a>],
}
/// Original declaration plus paid finite member descriptors. The native source
/// keeps its concrete programs separately and must match them at the local hook.
#[derive(Debug)]
pub struct PreparedPartitionInterventionSource {
    pub(super) source: OriginalInterventionSource,
    pub(super) operation: usize,
    pub(super) phase: CapturePhase,
    pub(super) prediction: u64,
    pub(super) world: usize,
    pub(super) invocations: Vec<Invocation>,
    pub(super) members: Vec<usize>,
    pub(super) descriptor: [u8; 32],
    pub(super) identity: Arc<Identity>,
    pub(super) metadata: HostMetadataFunding,
    metadata_bytes: usize,
}
impl PreparedPartitionInterventionSource {
    pub fn new(
        source: &OriginalInterventionSource,
        operation: usize,
        phase: CapturePhase,
        prediction: u64,
        world: usize,
        invocations: &[PartitionInterventionInvocationSource<'_>],
        metadata: &HostMetadataFunding,
    ) -> Result<Self, PartitionInterventionSourceError> {
        let result = (|| -> Result<Self, Cause> {
            let mut bytes = Self::control_bytes().ok_or(CaptureError::Overflow)?;
            metadata.reserve_metadata(bytes)?;
            let plan = source.plan().admission();
            let operation_source = plan
                .plan()
                .operations
                .get(operation)
                .ok_or(Cause::Source("operation is absent"))?;
            let point = plan
                .points()
                .get(operation)
                .ok_or(Cause::Source("operation point is absent"))?;
            if world == 0
                || u32::try_from(world).is_err()
                || u32::try_from(operation).is_err()
                || !operation_source.schedule.includes(phase, prediction)
                || operation_source.action.dtype().is_none()
                || point.routing.is_some()
                || point.routed_units.is_some()
                || invocations.is_empty()
            {
                return Err(Cause::Source("dense source schedule or world differs"));
            }
            let windowed =
                phase == CapturePhase::Prefill && InterventionPrefillWindow::row_axis(point);
            if !windowed && (invocations.len() != 1 || invocations[0].window.is_some()) {
                return Err(Cause::Source(
                    "terminal operation requires one original callback",
                ));
            }
            let first = invocations[0].window;
            let mut next = 0;
            for row in invocations {
                if row.members.is_empty()
                    || row.members.len() > world
                    || row.members.iter().any(|member| member.rank >= world)
                    || row
                        .members
                        .windows(2)
                        .any(|pair| pair[0].rank >= pair[1].rank)
                {
                    return Err(Cause::Source(
                        "members are not the ordered nonempty original group",
                    ));
                }
                if windowed {
                    let window = row
                        .window
                        .ok_or(Cause::Source("prefill callback is missing its window"))?;
                    window.validate(plan)?;
                    if window.range()[0] != next
                        || first.is_none_or(|first| first.inference() != window.inference())
                    {
                        return Err(Cause::Source(
                            "prefill source omits or reorders a canonical window",
                        ));
                    }
                    next = window.range()[1];
                }
                for member in row.members {
                    if !member.projection.source().same_source(source)
                        || member.projection.coordinate() != (operation, phase, prediction, None)
                        || member.shape.len() != member.projection.local_shape().len()
                    {
                        return Err(Cause::Source(
                            "member projection belongs to another original source",
                        ));
                    }
                    validate_usage(member.usage, member.projection_usage, member.source_usage)?;
                }
            }
            if windowed
                && !invocations
                    .last()
                    .and_then(|row| row.window)
                    .is_some_and(|window| window.is_final())
            {
                return Err(Cause::Source(
                    "prefill source does not reach its original final window",
                ));
            }
            let count = (0..world)
                .filter(|rank| {
                    invocations
                        .iter()
                        .any(|row| row.members.iter().any(|member| member.rank == *rank))
                })
                .count();
            let mut members = vector(metadata, count, &mut bytes)?;
            members.extend((0..world).filter(|rank| {
                invocations
                    .iter()
                    .any(|row| row.members.iter().any(|member| member.rank == *rank))
            }));
            let mut rows = vector(metadata, invocations.len(), &mut bytes)?;
            let mut digest = Sha256::new();
            digest.update(b"eredu-original-partition-intervention-v1\0");
            digest.update(plan.intent_identity().as_bytes());
            digest.update((operation as u64).to_le_bytes());
            digest.update((world as u64).to_le_bytes());
            serde_json::to_writer(&mut HashWriter(&mut digest), &(phase, prediction))
                .expect("closed source coordinate and infallible digest writer");
            digest.update((invocations.len() as u64).to_le_bytes());
            for row in invocations {
                digest.update([u8::from(row.window.is_some())]);
                if let Some(window) = row.window {
                    serde_json::to_writer(
                        &mut HashWriter(&mut digest),
                        &(window.inference(), window.range()),
                    )
                    .expect("closed source window and infallible digest writer");
                }
                digest.update((row.members.len() as u64).to_le_bytes());
                let mut ranks = vector(metadata, row.members.len(), &mut bytes)?;
                let mut values = vector(metadata, row.members.len(), &mut bytes)?;
                for member in row.members {
                    let geometry = member.projection.geometry_identity();
                    digest.update((member.rank as u64).to_le_bytes());
                    digest.update(geometry);
                    digest.update(member.execution_identity);
                    digest.update((member.shape.len() as u64).to_le_bytes());
                    for dimension in member.shape {
                        digest.update(dimension.to_le_bytes());
                    }
                    for usage in [
                        member.usage,
                        member.projection_usage[0],
                        member.projection_usage[1],
                        member.source_usage,
                    ] {
                        hash_usage(&mut digest, usage);
                    }
                    let mut shape = vector(metadata, member.shape.len(), &mut bytes)?;
                    shape.extend_from_slice(member.shape);
                    ranks.push(member.rank);
                    values.push(Some(Member {
                        rank: member.rank,
                        geometry,
                        execution: member.execution_identity,
                        shape,
                        usage: member.usage,
                        projection: member.projection_usage,
                        source_usage: member.source_usage,
                    }));
                }
                rows.push(Invocation {
                    window: row.window,
                    ranks,
                    members: values,
                    local: None,
                    quota: None,
                    vote: None,
                    vote_usage: CaptureUsage::default(),
                    state: State::Unprepared,
                });
            }
            let alias =
                WorkspaceContext::metadata_arc_bytes::<Identity>().ok_or(CaptureError::Overflow)?;
            metadata.reserve_metadata(alias)?;
            bytes = bytes.checked_add(alias).ok_or(CaptureError::Overflow)?;
            Ok(Self {
                source: source.clone(),
                operation,
                phase,
                prediction,
                world,
                invocations: rows,
                members,
                descriptor: digest.finalize().into(),
                identity: Arc::new(Identity),
                metadata: metadata.clone(),
                metadata_bytes: bytes,
            })
        })();
        result.map_err(|cause| PartitionInterventionSourceError {
            cause,
            _source: source.clone(),
            _metadata: metadata.clone(),
        })
    }
    pub fn original(&self) -> &OriginalInterventionSource {
        &self.source
    }
    pub fn operation(&self) -> usize {
        self.operation
    }
    pub fn coordinate(&self) -> (CapturePhase, u64) {
        (self.phase, self.prediction)
    }
    pub fn invocation_count(&self) -> usize {
        self.invocations.len()
    }
    pub fn descriptor(&self) -> &[u8; 32] {
        &self.descriptor
    }
    pub(super) fn error(&self, cause: Cause) -> PartitionInterventionSourceError {
        PartitionInterventionSourceError {
            cause,
            _source: self.source.clone(),
            _metadata: self.metadata.clone(),
        }
    }
    /// Reserve the complete operation once before common pre-forward agreement.
    /// No parent-ledger reservation occurs in a local native callback.
    pub(crate) fn prepare<'t, T: PartitionCaptureHookTransport>(
        mut self,
        transport: &'t T,
        epoch: DistributedCommitEpoch,
        ledger: &mut CaptureLedger,
    ) -> Result<PreparedPartitionIntervention<'t, T>, PartitionInterventionSourceError>
    where
        T::Error: Send + Sync + 'static,
        <T::Completion as Completion>::Error: Send + Sync + 'static,
    {
        let source = self.source.clone();
        let metadata = self.metadata.clone();
        let result = (|| -> Result<_, Cause> {
            let controls = runtime_controls::<T>().ok_or(CaptureError::Overflow)?;
            metadata.reserve_metadata(controls)?;
            let bytes = self
                .metadata_bytes
                .checked_add(controls)
                .ok_or(CaptureError::Overflow)?;
            let rank = transport.capture_rank();
            if transport.participant_count() != self.world || rank >= self.world {
                return Err(Cause::Source("prepared world differs"));
            }
            let wait = transport.capture_wait()?;
            if !T::Completion::supports_cancellation(wait.cancellation()) {
                return Err(Cause::Source("unsupported hook cancellation policy"));
            }
            transport
                .ensure_capture_active()
                .map_err(PartitionCaptureExchangeError::from)?;
            let mut global = CaptureUsage {
                host_bytes: u64::try_from(bytes)
                    .map_err(|_| CaptureError::Overflow)?
                    .checked_mul(self.world as u64)
                    .ok_or(CaptureError::Overflow)?,
                ..Default::default()
            };
            let mut local = CaptureUsage::default();
            for row in &mut self.invocations {
                row.local = row.ranks.binary_search(&rank).ok();
                row.vote_usage = transport.estimate_capture_hook(&row.ranks)?;
                if row.vote_usage.captures != 0 || row.vote_usage.encoded_bytes != 0 {
                    return Err(Cause::Source(
                        "member vote contains capture payload credits",
                    ));
                }
                global = global.checked_add(
                    row.vote_usage.checked_mul(
                        (row.ranks.len() as u64)
                            .checked_mul(2)
                            .ok_or(CaptureError::Overflow)?,
                    )?,
                )?;
                for member in row.members.iter().flatten() {
                    global = global.checked_add(member.global(self.world as u64)?)?;
                }
                if let Some(index) = row.local {
                    local = local.checked_add(
                        row.members[index]
                            .as_ref()
                            .expect("original member")
                            .total()?,
                    )?;
                }
            }
            let receipt_usage = super::super::super::exchange::gather_usage(
                transport,
                self.world,
                PartitionInterventionReceipt::WORDS,
            )?;
            global = global.checked_add(receipt_usage.checked_mul(self.world as u64)?)?;
            let mut remaining = ledger.reserve_quota(global)?;
            for row in &mut self.invocations {
                if let Some(index) = row.local {
                    row.quota = Some(
                        remaining.reserve_quota(
                            row.members[index]
                                .as_ref()
                                .expect("original member")
                                .total()?,
                        )?,
                    );
                    row.vote = Some(remaining.reserve_quota(row.vote_usage.checked_mul(2)?)?);
                    row.state = State::Ready;
                } else {
                    row.state = State::Absent;
                }
            }
            let receipt = remaining.reserve_quota(receipt_usage)?;
            let mut digest = Sha256::new();
            digest.update(self.descriptor);
            digest.update(epoch.value().to_le_bytes());
            hash_usage(&mut digest, global);
            self.descriptor = digest.finalize().into();
            Ok(PreparedPartitionIntervention {
                transport,
                source: self,
                epoch,
                rank,
                wait,
                agree: super::super::hook::agree::<T>,
                global,
                local,
                receipt,
                receipt_usage,
                remaining,
                delivered: false,
            })
        })();
        result.map_err(|cause| PartitionInterventionSourceError {
            cause,
            _source: source,
            _metadata: metadata,
        })
    }
    pub fn control_bytes() -> Option<usize> {
        let frames = [
            size_of::<Self>() * 2,
            size_of::<Member>(),
            size_of::<Invocation>(),
            size_of::<Identity>(),
            size_of::<Cause>(),
            size_of::<PartitionInterventionSourceError>(),
            size_of::<Result<Self, Cause>>(),
            size_of::<Result<Self, PartitionInterventionSourceError>>(),
            size_of::<OriginalInterventionSource>(),
            size_of::<HostMetadataFunding>(),
            size_of::<Arc<Identity>>(),
            size_of::<Sha256>(),
            size_of::<HashWriter<'_>>(),
            crate::capture::RECORD_ENCODING_CONTROL_BYTES,
            size_of::<serde_json::Serializer<&mut HashWriter<'_>>>(),
            size_of::<(
                &OriginalInterventionSource,
                usize,
                CapturePhase,
                u64,
                usize,
                &[PartitionInterventionInvocationSource<'_>],
                &HostMetadataFunding,
            )>(),
            size_of::<PartitionInterventionInvocationSource<'_>>(),
            size_of::<PartitionInterventionMemberSource<'_>>(),
            size_of::<[CaptureUsage; 4]>(),
            size_of::<Option<InterventionPrefillWindow>>(),
            size_of::<std::ops::Range<usize>>(),
            size_of::<std::slice::Iter<'_, PartitionInterventionInvocationSource<'_>>>(),
            size_of::<std::slice::Iter<'_, PartitionInterventionMemberSource<'_>>>(),
            size_of::<[u8; 32]>() * 3,
        ];
        frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)
    }
}
fn vector<T>(
    metadata: &HostMetadataFunding,
    count: usize,
    bytes: &mut usize,
) -> Result<Vec<T>, Cause> {
    *bytes = bytes
        .checked_add(
            WorkspaceContext::metadata_vec_bytes::<T>(count).ok_or(CaptureError::Overflow)?,
        )
        .ok_or(CaptureError::Overflow)?;
    Ok(metadata.metadata_vec(count)?)
}
fn validate_usage(
    usage: CaptureUsage,
    projection: [CaptureUsage; 2],
    source: CaptureUsage,
) -> Result<(), Cause> {
    if usage.captures != 0
        || usage.encoded_bytes != 0
        || source.captures != 0
        || source.encoded_bytes != 0
        || projection
            .iter()
            .any(|usage| usage.captures != 0 || usage.retained_bytes != 0)
    {
        return Err(Cause::Source(
            "edit, projection and dependency credit roles differ",
        ));
    }
    usage
        .checked_add(projection[0])?
        .checked_add(projection[1])?
        .checked_add(source)?;
    Ok(())
}
fn runtime_controls<T: PartitionCaptureTransport>() -> Option<usize> {
    let frames = [
        size_of::<PreparedPartitionIntervention<'_, T>>() * 2,
        size_of::<PartitionInterventionLocalAllowance>() * 2,
        size_of::<
            Result<Option<PartitionInterventionLocalAllowance>, PartitionInterventionSourceError>,
        >(),
        size_of::<Result<(), PartitionInterventionSourceError>>(),
        size_of::<Result<bool, PartitionCaptureExchangeError>>(),
        size_of::<Result<PreparedPartitionIntervention<'_, T>, Cause>>(),
        size_of::<CaptureQuota>() * 4,
        size_of::<CaptureUsage>() * 8,
        size_of::<(&T, DistributedCommitEpoch, &mut CaptureLedger)>(),
        size_of::<(&T, Option<InterventionPrefillWindow>, bool)>(),
        size_of::<(&T, bool)>(),
        size_of::<PartitionCaptureFrame<'_>>(),
        size_of::<PartitionCaptureBuffer<u32>>(),
        PartitionInterventionReceipt::control_bytes()?,
        PartitionInterventionLocalAllowance::control_bytes()?,
    ];
    frames
        .into_iter()
        .try_fold(size_of_val(&frames), usize::checked_add)
}

use eredu_nn::workspace::WorkspaceMetadataAllocation;
