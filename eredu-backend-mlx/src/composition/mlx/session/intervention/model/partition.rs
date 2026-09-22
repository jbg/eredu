//! Per-operation local/peer source slots within the existing model row driver.
use super::super::PreparedPartitionModelIntervention;
use super::super::partition::PreparedPartitionEvidenceSource;
use super::*;
use eredu_architectures::component_partition::ComponentPartitionLayouts;
use eredu_runtime::{
    RetainedCommunicationSource, capture::partition::PartitionInterventionMemberSource,
    intervention::PreparedPartitionInterventionProjection,
};
struct Member {
    rank: usize,
    source: PreparedPartitionModelIntervention,
}
struct Operation {
    members: Vec<Member>,
    local: Option<usize>,
    remote: bool,
    attempted: bool,
}
enum Shard {
    Inactive,
    Active(Operation),
}
pub(super) struct Binding {
    rank: usize,
    world: usize,
    shards: Vec<Shard>,
    communication: RetainedCommunicationSource,
}
impl PreparedModelInterventions {
    /// Bind actual architecture membership before the shared row begins. Peer
    /// descriptors carry geometry/cost only; only a real local hook binds native work.
    pub(crate) fn bind_partition(
        &mut self,
        layouts: &ComponentPartitionLayouts,
        rank: usize,
        communication: &RetainedCommunicationSource,
        context: &WorkspaceContext,
    ) -> Result<(), eredu_nn::Error> {
        context.charge_metadata(control_bytes().ok_or(WorkspaceMetadataError::Overflow)?)?;
        if self.begun
            || self.sealed
            || self.partition.is_some()
            || rank != communication.manifest().rank()
            || layouts.topology().world_size() != communication.manifest().world_size()
        {
            return Err(context.metadata_source(CaptureProtocolError::Invocation));
        }
        let world = layouts.topology().world_size();
        let wait = communication
            .manifest()
            .completion_policy()
            .ok_or_else(|| context.metadata_source(Failure::SourceChanged))?
            .bounded_wait();
        let plan = self.source.plan().admission();
        let mut shards = context.metadata_vec(self.rows.len())?;
        for index in 0..self.rows.len() {
            if matches!(self.rows[index], Row::Inactive) {
                shards.push(Shard::Inactive);
                continue;
            }
            if !matches!(self.rows[index], Row::Pending) {
                return Err(context.metadata_source(CaptureProtocolError::Transaction));
            }
            let operation = &plan.plan().operations[index];
            let point = &plan.points()[index];
            if point.routed_units.is_some() || point.routing.is_some() {
                return Err(context.metadata_source(CaptureProtocolError::Geometry));
            }
            layouts
                .observation_combination(&point.path)
                .map_err(|e| context.metadata_source(e))?;
            let invocation = self
                .invocation
                .map(|physical| match self.window {
                    Some(window) => window.validate(physical),
                    None => {
                        physical.validate()?;
                        Ok(physical)
                    }
                })
                .transpose()
                .map_err(|e| context.metadata_source(e))?;
            let logical = plan
                .geometry_at(self.phase, self.prediction, invocation)
                .map_err(|e| context.metadata_source(e))?;
            let mut global = context.metadata_vec(point.axes.len())?;
            global.resize(point.axes.len(), 0);
            if !logical
                .resolve_axes_into(&point.axes, &mut global)
                .map_err(|e| context.metadata_source(e))?
            {
                return Err(context.metadata_source(Failure::ShapeMismatch));
            }
            let mut members = context.metadata_vec(world)?;
            let mut local = None;
            for peer in 0..world {
                let layout = layouts
                    .rank(peer)
                    .ok_or_else(|| context.metadata_source(CaptureProtocolError::Geometry))?;
                let member = layout
                    .intervention_source(plan, index, self.phase, self.prediction)
                    .map_err(|e| context.metadata_source(e))?;
                let Some(member) = member else {
                    continue;
                };
                let funding = context
                    .metadata_funding()
                    .ok_or(WorkspaceMetadataError::Unqualified)?;
                // Each retained component can introduce at most one ordinary run.
                let projection = PreparedPartitionInterventionProjection::prepare(
                    &self.source,
                    index,
                    self.phase,
                    self.prediction,
                    invocation,
                    &global,
                    member.axis(),
                    member.coordinates(),
                    member.sum_offset_owner(),
                    member.coordinates().local_count().max(1),
                    funding,
                )
                .map_err(|e| context.metadata_source(e))?;
                let mut physical = context.metadata_vec(global.len())?;
                physical.extend_from_slice(projection.local_shape());
                if let Some(invocation) = self.invocation {
                    let actual = plan
                        .geometry_at(self.phase, self.prediction, Some(invocation))
                        .map_err(|e| context.metadata_source(e))?;
                    if !actual
                        .resolve_axes_into(&point.axes, &mut physical)
                        .map_err(|e| context.metadata_source(e))?
                    {
                        return Err(context.metadata_source(Failure::ShapeMismatch));
                    }
                    physical[member.axis()] = projection.local_shape()[member.axis()];
                }
                if let Some(span) = self.scheduled_span {
                    span.validate(plan)
                        .map_err(|e| context.metadata_source(e))?;
                    if !span
                        .physical()
                        .resolve_axes_into(&point.axes, &mut physical)
                        .map_err(|e| context.metadata_source(e))?
                    {
                        return Err(context.metadata_source(Failure::ShapeMismatch));
                    }
                    physical[member.axis()] = projection.local_shape()[member.axis()];
                }
                let evidence = if operation.evidence != InterventionEvidence::None {
                    let selected = layouts
                        .component_capture_source(&point.path)
                        .map_err(|cause| context.metadata_source(cause))?;
                    Some(PreparedPartitionEvidenceSource::prepare_model(
                        &projection,
                        &selected,
                        peer,
                        &physical,
                        self.scheduled_span,
                        self.invocation,
                        self.window,
                        context,
                    )?)
                } else {
                    None
                };
                let source =
                    PreparedPartitionModelIntervention::prepare_model_projection_with_evidence(
                        projection,
                        self.scheduled_span,
                        self.invocation,
                        self.window,
                        &physical,
                        wait,
                        evidence,
                        context,
                    )?;
                if peer == rank {
                    local = Some(members.len());
                }
                members.push(Member { rank: peer, source });
            }
            if members.is_empty() {
                return Err(context.metadata_source(CaptureProtocolError::Geometry));
            }
            let remote = local.is_none()
                && layouts
                    .rank(rank)
                    .and_then(|layout| layout.observation(&point.path))
                    .is_some_and(|point| {
                        point.site() == eredu_runtime::inspection::ObservationHookSite::Publication
                    });
            shards.push(Shard::Active(Operation {
                members,
                local,
                remote,
                attempted: false,
            }));
        }
        for (row, shard) in self.rows.iter_mut().zip(&shards) {
            if matches!(
                shard,
                Shard::Active(Operation {
                    local: None,
                    remote: false,
                    ..
                })
            ) {
                *row = Row::PartitionAbsent;
            }
        }
        self.partition = Some(Binding {
            rank,
            world,
            shards,
            communication: communication.clone(),
        });
        Ok(())
    }
    pub(crate) fn partition_source(
        &self,
        index: usize,
    ) -> Result<Option<&PreparedPartitionModelIntervention>, Failure> {
        if !self.sealed {
            return Err(Failure::ClaimMismatch);
        }
        let binding = self.partition.as_ref().ok_or(Failure::ClaimMismatch)?;
        match binding.shards.get(index) {
            Some(Shard::Active(operation)) => match operation.local {
                Some(local) if operation.attempted => Ok(Some(&operation.members[local].source)),
                None => Ok(None),
                _ => Err(Failure::SourceChanged),
            },
            Some(Shard::Inactive) => Ok(None),
            None => Err(Failure::ClaimMismatch),
        }
    }
    pub(crate) fn partition_members(
        &self,
        index: usize,
    ) -> Result<impl ExactSizeIterator<Item = PartitionInterventionMemberSource<'_>>, Failure> {
        if !self.sealed {
            return Err(Failure::ClaimMismatch);
        }
        let binding = self.partition.as_ref().ok_or(Failure::ClaimMismatch)?;
        let members: &[Member] = match binding.shards.get(index) {
            Some(Shard::Active(operation)) => &operation.members,
            Some(Shard::Inactive) => &[],
            None => return Err(Failure::ClaimMismatch),
        };
        Ok(members
            .iter()
            .map(|member| PartitionInterventionMemberSource {
                rank: member.rank,
                projection: member.source.projection(),
                shape: member.source.shape(),
                execution_identity: member.source.execution_identity(),
                usage: member.source.usage(),
                projection_usage: member.source.projection_usage(),
                source_usage: member.source.source_usage(),
            }))
    }
    pub(crate) fn partition_binding(&self) -> Option<(usize, usize, &RetainedCommunicationSource)> {
        self.partition
            .as_ref()
            .map(|binding| (binding.rank, binding.world, &binding.communication))
    }
}
pub(super) fn trace(
    binding: &mut Binding,
    index: usize,
    input: &WorkspaceTensor,
    context: &WorkspaceContext,
    roots: &mut Vec<WorkspaceTensor>,
) -> Result<(Option<WorkspaceTensor>, CaptureNativePopulation), eredu_nn::Error> {
    context.charge_metadata(control_bytes().ok_or(WorkspaceMetadataError::Overflow)?)?;
    let Some(Shard::Active(operation)) = binding.shards.get_mut(index) else {
        return Err(context.metadata_source(Failure::ClaimMismatch));
    };
    let local = operation
        .local
        .ok_or_else(|| context.metadata_source(Failure::ClaimMismatch))?;
    if operation.attempted {
        return Err(context.metadata_source(CaptureProtocolError::Transaction));
    }
    operation.attempted = true;
    operation.members[local]
        .source
        .trace_bound(input, context, roots)
}
pub(super) fn remote(binding: &Binding, index: usize) -> bool {
    matches!(
        binding.shards.get(index),
        Some(Shard::Active(Operation { remote: true, .. }))
    )
}
pub(super) fn control_bytes() -> Option<usize> {
    let frames = [
        size_of::<Binding>(),
        size_of::<Option<Binding>>(),
        size_of::<Shard>(),
        size_of::<Vec<Shard>>(),
        size_of::<Member>(),
        size_of::<(
            &PreparedModelInterventions,
            &ComponentPartitionLayouts,
            usize,
            &RetainedCommunicationSource,
            &WorkspaceContext,
        )>(),
        size_of::<(
            &mut Binding,
            usize,
            &WorkspaceTensor,
            &WorkspaceContext,
            &mut Vec<WorkspaceTensor>,
        )>(),
        size_of::<Result<(Option<WorkspaceTensor>, CaptureNativePopulation), eredu_nn::Error>>(),
        size_of::<Result<Option<&PreparedPartitionModelIntervention>, Failure>>(),
        size_of::<Vec<PartitionInterventionMemberSource<'_>>>(),
        PreparedPartitionModelIntervention::control_bytes()?,
    ];
    frames
        .into_iter()
        .try_fold(size_of_val(&frames), usize::checked_add)
}

impl PreparedModelInterventions {
    pub(crate) fn source(&self) -> &eredu_runtime::working_memory::OriginalInterventionSource {
        &self.source
    }
    pub(crate) fn invocation_coordinate(
        &self,
    ) -> Option<(
        CapturePhase,
        u64,
        CaptureInvocationShape,
        Option<CaptureInvocationWindow>,
    )> {
        Some((self.phase, self.prediction, self.invocation?, self.window))
    }
    /// Copy the exact sealed member sources into the consumed model frame's
    /// Host account. Physical rows/window remain attached to every operation.
    pub(crate) fn partition_sources(
        &self,
        metadata: &HostMetadataFunding,
    ) -> Result<
        Vec<Option<eredu_runtime::capture::partition::PreparedPartitionInterventionSource>>,
        eredu_nn::Error,
    > {
        use eredu_runtime::capture::partition::PreparedPartitionInterventionSource;
        metadata
            .reserve_metadata(model_source_control_bytes().ok_or(WorkspaceMetadataError::Overflow)?)
            .map_err(|cause| metadata.metadata_source(cause))?;
        let binding = self
            .partition
            .as_ref()
            .filter(|_| self.sealed)
            .ok_or_else(|| metadata.metadata_source(Failure::SourceChanged))?;
        let physical = self
            .invocation
            .ok_or_else(|| metadata.metadata_source(Failure::SourceChanged))?;
        if self.scheduled_span.is_some() {
            return Err(metadata.metadata_source(Failure::SourceChanged));
        }
        let mut output = metadata.metadata_vec(binding.shards.len())?;
        for (index, shard) in binding.shards.iter().enumerate() {
            if matches!(shard, Shard::Inactive) {
                output.push(None);
                continue;
            }
            let members = self
                .partition_members(index)
                .map_err(|cause| metadata.metadata_source(cause))?;
            let mut copied = metadata.metadata_vec(members.len())?;
            copied.extend(members);
            output.push(Some(
                PreparedPartitionInterventionSource::new_invocation(
                    &self.source,
                    index,
                    self.phase,
                    self.prediction,
                    binding.world,
                    physical,
                    self.window,
                    &copied,
                    metadata,
                )
                .map_err(|cause| metadata.metadata_source(cause))?,
            ));
        }
        Ok(output)
    }
    /// The same source-copy and runtime transcript populations. The temporary
    /// borrowed member table is paid by cold preparation, never a new grant.
    pub(crate) fn partition_sources_metadata_bytes(
        &self,
        context: &WorkspaceContext,
    ) -> Result<usize, eredu_nn::Error> {
        use eredu_runtime::capture::partition::{
            PartitionInterventionInvocationSource, PreparedPartitionInterventionSource,
        };
        context.charge_metadata(
            model_source_control_bytes().ok_or(WorkspaceMetadataError::Overflow)?,
        )?;
        let binding = self
            .partition
            .as_ref()
            .filter(|_| self.sealed)
            .ok_or_else(|| context.metadata_source(Failure::SourceChanged))?;
        if self.invocation.is_none() || self.scheduled_span.is_some() {
            return Err(context.metadata_source(Failure::SourceChanged));
        }
        let mut bytes = model_source_control_bytes()
            .and_then(|n| {
                n.checked_add(WorkspaceContext::metadata_vec_bytes::<
                    Option<PreparedPartitionInterventionSource>,
                >(binding.shards.len())?)
            })
            .ok_or(WorkspaceMetadataError::Overflow)?;
        for (index, shard) in binding.shards.iter().enumerate() {
            if matches!(shard, Shard::Inactive) {
                continue;
            }
            let members = self
                .partition_members(index)
                .map_err(|cause| context.metadata_source(cause))?;
            let mut copied = context.metadata_vec(members.len())?;
            copied.extend(members);
            bytes = bytes
                .checked_add(
                    WorkspaceContext::metadata_vec_bytes::<PartitionInterventionMemberSource<'_>>(
                        copied.len(),
                    )
                    .ok_or(WorkspaceMetadataError::Overflow)?,
                )
                .and_then(|n| {
                    n.checked_add(
                        PreparedPartitionInterventionSource::preparation_metadata_bytes(
                            binding.world,
                            &[PartitionInterventionInvocationSource {
                                window: None,
                                members: &copied,
                            }],
                        )?,
                    )
                })
                .ok_or(WorkspaceMetadataError::Overflow)?;
        }
        Ok(bytes)
    }
    pub(crate) fn partition_operation_selected(&self, operation: usize) -> Result<bool, Failure> {
        let binding = self
            .partition
            .as_ref()
            .filter(|_| self.sealed)
            .ok_or(Failure::SourceChanged)?;
        match binding.shards.get(operation) {
            Some(Shard::Inactive) => Ok(false),
            Some(Shard::Active(_)) => Ok(true),
            None => Err(Failure::SourceChanged),
        }
    }
    pub(crate) fn partition_evidence_skip_reasons(
        &self,
        operation: usize,
    ) -> Option<&[Option<CaptureSkipReason>; 2]> {
        self.evidence_skips.as_ref()?.get(operation)
    }
    pub(crate) fn partition_evidence_scalar_control_bytes() -> Option<usize> {
        size_of::<(
            &Self,
            CapturePhase,
            u64,
            usize,
            eredu_core::capture::InterventionEvidenceSide,
            &HostMetadataFunding,
        )>()
        .checked_add(size_of::<
            Result<Option<eredu_nn::workspace::WorkspaceFloatingType>, eredu_nn::Error>,
        >())
    }
    /// Exact local provider representation for one retained evidence side.
    pub(crate) fn partition_evidence_scalar(
        &self,
        phase: CapturePhase,
        prediction: u64,
        operation: usize,
        side: eredu_core::capture::InterventionEvidenceSide,
        metadata: &HostMetadataFunding,
    ) -> Result<Option<eredu_nn::workspace::WorkspaceFloatingType>, eredu_nn::Error> {
        metadata
            .reserve_metadata(
                Self::partition_evidence_scalar_control_bytes()
                    .ok_or(WorkspaceMetadataError::Overflow)?,
            )
            .map_err(|cause| metadata.metadata_source(cause))?;
        let result = (|| {
            if (phase, prediction) != (self.phase, self.prediction) {
                return Err(Failure::SourceChanged);
            }
            let binding = self.partition.as_ref().ok_or(Failure::SourceChanged)?;
            let Some(local) = self.partition_source(operation)? else {
                return Ok(None);
            };
            let companion = self
                .source
                .plan()
                .evidence(operation)
                .ok_or(Failure::SourceChanged)?;
            let evidence = local.evidence().ok_or(Failure::SourceChanged)?;
            if evidence.rank() != binding.rank
                || !evidence
                    .source()
                    .same_storage(companion.shared_geometry_source())
            {
                return Err(Failure::SourceChanged);
            }
            evidence
                .scalar(side)
                .map(Some)
                .ok_or(Failure::SourceChanged)
        })();
        result.map_err(|cause| metadata.metadata_source(cause))
    }
}
fn model_source_control_bytes() -> Option<usize> {
    use eredu_runtime::capture::partition::{
        PartitionInterventionInvocationSource, PreparedPartitionInterventionSource,
    };
    let parts = [
        size_of::<(&PreparedModelInterventions, &HostMetadataFunding)>(),
        size_of::<(&PreparedModelInterventions, &WorkspaceContext)>(),
        size_of::<PartitionInterventionMemberSource<'_>>(),
        size_of::<Vec<PartitionInterventionMemberSource<'_>>>(),
        size_of::<[PartitionInterventionInvocationSource<'_>; 1]>(),
        size_of::<Vec<Option<PreparedPartitionInterventionSource>>>(),
        size_of::<Result<Vec<Option<PreparedPartitionInterventionSource>>, eredu_nn::Error>>(),
        size_of::<(
            usize,
            usize,
            CaptureInvocationShape,
            Option<CaptureInvocationWindow>,
        )>(),
    ];
    parts
        .into_iter()
        .try_fold(size_of_val(&parts), usize::checked_add)
}
