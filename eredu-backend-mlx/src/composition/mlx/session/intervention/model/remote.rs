//! A received publication is descriptive source evidence, never a local edit.
use super::*;
use eredu_architectures::component_partition::{
    CompletePartitionCaptureSource, ComponentPartitionLayouts,
};

/// Original source geometry for one operation executed by the publication owner.
/// Outcome and before/after payloads still require the partition receipt worker.
/// A completed broadcast alone cannot certify that an intervention was applied.
pub(super) struct Source {
    pub declaration: CompletePartitionCaptureSource,
    pub receiver: usize,
    pub world: usize,
    pub shape: Vec<u64>,
    pub dtype: InterventionDtype,
    pub selected: ResolvedCaptureSlice,
}

impl PreparedModelInterventions {
    pub(in crate::composition::mlx::session) fn trace_remote_output(
        &mut self,
        layouts: &ComponentPartitionLayouts,
        rank: usize,
        value: &WorkspaceTensor,
        context: &WorkspaceContext,
    ) -> Result<(), eredu_nn::Error> {
        context.charge_metadata(control_bytes().ok_or(WorkspaceMetadataError::Overflow)?)?;
        if !self.begun || self.sealed {
            return Err(context.metadata_source(CaptureProtocolError::Transaction));
        }
        context.validate_values([value])?;
        let path = eredu_core::MODEL_LOGITS_OBSERVATION_PATH;
        let declaration = layouts
            .complete_capture_source(path)
            .filter(|source| {
                source.site() == eredu_runtime::inspection::ObservationHookSite::Publication
                    && source.producer() != rank
                    && layouts.rank(rank).is_some()
            })
            .ok_or_else(|| context.metadata_source(CaptureProtocolError::Geometry))?;
        let plan = self.source.plan().admission();
        for (index, (operation, point)) in
            plan.plan().operations.iter().zip(plan.points()).enumerate()
        {
            let outcome = match &self.rows[index] {
                Row::Inactive => InterventionOutcome::Inactive,
                Row::Pending => InterventionOutcome::Missing,
                Row::Ready(_) | Row::Remote(_) | Row::PartitionReady => {
                    InterventionOutcome::Applied
                }
                Row::PartitionAbsent => InterventionOutcome::Inactive,
                Row::Routed(_) => InterventionOutcome::Missing,
                Row::Routing(edit) => {
                    if edit.complete {
                        InterventionOutcome::Applied
                    } else {
                        InterventionOutcome::Missing
                    }
                }
            };
            match activation_hook(operation, point, &outcome, path) {
                ActivationHook::Unrelated => continue,
                ActivationHook::Repeated => {
                    return Err(context.metadata_source(CaptureProtocolError::Transaction));
                }
                ActivationHook::Active => (),
            }
            if value.shape().len() != point.axes.len() || value.shape().len() > 32 {
                return Err(context.metadata_source(Failure::ShapeMismatch));
            }
            let dtype = operation
                .action
                .dtype()
                .ok_or_else(|| context.metadata_source(Failure::ShapeMismatch))?;
            PreparedStaticActivation::validate_workspace_source(value, dtype)
                .map_err(|cause| context.metadata_source(cause))?;
            let mut shape = context.metadata_vec(value.shape().len())?;
            for extent in value.shape() {
                shape.push(u64::try_from(*extent).map_err(|cause| context.metadata_source(cause))?);
            }
            let mut selected = window::slice(shape.len(), context)?;
            let span = self.scheduled_span;
            let physical = span.map(|source| source.physical()).or(self.invocation);
            let window = span.map(|source| source.window()).or(self.window);
            if let Some(window) = window {
                let physical = physical
                    .ok_or_else(|| context.metadata_source(CaptureProtocolError::Invocation))?;
                let mut global = context.metadata_vec(shape.len())?;
                global.resize(shape.len(), 0);
                let (logical, _) = window
                    .source_axes_into(physical, Some(&point.axes), &shape, &mut global)
                    .map_err(|cause| context.metadata_source(cause))?;
                match plan.invocation_bounds() {
                    Some(_) => plan.resolve_prepared_invocation_at(
                        index,
                        self.phase,
                        self.prediction,
                        logical,
                        &global,
                        dtype,
                        &mut selected,
                    ),
                    None => plan.resolve_prepared_at(
                        index,
                        self.phase,
                        self.prediction,
                        &global,
                        dtype,
                        &mut selected,
                    ),
                }
                .map_err(|cause| context.metadata_source(cause))?;
            } else {
                match physical {
                    Some(physical) => plan.resolve_prepared_invocation_at(
                        index,
                        self.phase,
                        self.prediction,
                        physical,
                        &shape,
                        dtype,
                        &mut selected,
                    ),
                    None => plan.resolve_prepared_at(
                        index,
                        self.phase,
                        self.prediction,
                        &shape,
                        dtype,
                        &mut selected,
                    ),
                }
                .map_err(|cause| context.metadata_source(cause))?;
            }
            self.rows[index] = Row::Remote(Source {
                declaration,
                receiver: rank,
                world: layouts.topology().world_size(),
                shape,
                dtype,
                selected,
            });
        }
        Ok(())
    }
}

fn control_bytes() -> Option<usize> {
    let frames = [
        size_of::<Source>(),
        size_of::<[Vec<u64>; 2]>(),
        size_of::<ResolvedCaptureSlice>(),
        size_of::<(
            &mut PreparedModelInterventions,
            &ComponentPartitionLayouts,
            usize,
            &WorkspaceTensor,
            &WorkspaceContext,
        )>(),
        size_of::<(
            Option<CaptureInvocationShape>,
            Option<CaptureInvocationWindow>,
        )>(),
        size_of::<(InterventionOutcome, ActivationHook)>(),
        size_of::<Result<(), eredu_nn::Error>>(),
        ComponentPartitionLayouts::complete_capture_source_control_bytes()?,
        PreparedStaticActivation::inspection_control_bytes()?,
        window::control_bytes()?,
    ];
    frames
        .into_iter()
        .try_fold(size_of_val(&frames), usize::checked_add)
}
