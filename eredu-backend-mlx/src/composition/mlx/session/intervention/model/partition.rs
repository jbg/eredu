//! Per-operation local/peer source slots within the existing model row driver.
use super::super::PreparedPartitionModelIntervention;
use super::super::partition::PreparedPartitionEvidenceSource;
use super::*;
use eredu_architectures::component_partition::ComponentPartitionLayouts;
use eredu_runtime::{
    capture::partition::PartitionInterventionMemberSource,
    intervention::PreparedPartitionInterventionProjection, RetainedCommunicationSource,
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
    pub(in super::super) fn bind_partition(
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
            || self.invocation.is_some()
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
            if point.routed_units.is_some()
                || point.routing.is_some()
            {
                return Err(context.metadata_source(CaptureProtocolError::Geometry));
            }
            layouts
                .observation_combination(&point.path)
                .map_err(|e| context.metadata_source(e))?;
            let logical = plan
                .geometry_at(self.phase, self.prediction, None)
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
                    None,
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
                let evidence=if operation.evidence!=InterventionEvidence::None {
                    let selected=layouts.component_capture_source(&point.path)
                        .map_err(|cause|context.metadata_source(cause))?;
                    Some(PreparedPartitionEvidenceSource::prepare(&projection,&selected,peer,
                        &physical,self.scheduled_span,context)?)
                }else {None};
                let source = PreparedPartitionModelIntervention::prepare_projection_with_evidence(
                    projection,self.scheduled_span,&physical,wait,evidence,context,
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
