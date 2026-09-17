//! Borrowed invocation placement shared by ordinary and paid intervention projection.
use super::*;
use eredu_core::{
    capture::{CaptureError, CapturePhase},
    intervention::AdmittedInterventionPlan,
};

/// Actual architecture member; publication filtering is already applied.
/// No shape, projected payload, native value or execution authority is constructed.
#[derive(Clone, Copy)]
pub struct PartitionInterventionMemberLayout<'a> {
    axis: usize,
    coordinates: &'a ComponentCoordinateMap,
    sum_offset_owner: Option<bool>,
}
impl<'a> PartitionInterventionMemberLayout<'a> {
    /// Original declared component axis, before any token-window projection.
    pub fn axis(self) -> usize {
        self.axis
    }
    /// Exact retained local-to-global component map, including empty members.
    pub fn coordinates(self) -> &'a ComponentCoordinateMap {
        self.coordinates
    }
    /// Existing additive owner policy; replicas remain independent members.
    pub fn sum_offset_owner(self) -> Option<bool> {
        self.sum_offset_owner
    }
}
impl ComponentPartitionLayout {
    /// Describe one scheduled operation without constructing its projection.
    /// Ordinary and original paths consume this same publication/member decision.
    pub fn intervention_source<'a>(
        &'a self,
        plan: &AdmittedInterventionPlan,
        operation: usize,
        phase: CapturePhase,
        prediction: u64,
    ) -> Result<Option<PartitionInterventionMemberLayout<'a>>, ComponentPartitionError> {
        let operation_plan = plan.plan().operations.get(operation).ok_or_else(|| {
            CaptureError::Invalid("unknown partition intervention operation".into())
        })?;
        if prediction >= plan.request().max_predictions
            || !operation_plan.schedule.includes(phase, prediction)
        {
            return Err(CaptureError::Invalid(
                "partition intervention is outside its admitted schedule".into(),
            )
            .into());
        }
        let point = &plan.points()[operation];
        let placement = self
            .observation(&point.path)
            .ok_or_else(|| CaptureError::MissingPath(point.path.clone()))?;
        let Some(coordinates) = placement.intervention_coordinates() else {
            return Ok(None);
        };
        let mut matching = point
            .axes
            .iter()
            .enumerate()
            .filter(|(_, axis)| axis.name == placement.axis());
        let Some((axis, _)) = matching.next() else {
            return Err(CaptureError::Invalid(
                "intervention has no retained observation axis".into(),
            )
            .into());
        };
        if matching.next().is_some() {
            return Err(CaptureError::Invalid(
                "intervention repeats its retained observation axis".into(),
            )
            .into());
        }
        let sum_offset_owner = match placement.combination() {
            PartitionCaptureCombination::Disjoint => None,
            PartitionCaptureCombination::SumF64ToF32 => {
                Some(self.topology.tensor_parallel_rank() == 0)
            }
        };
        Ok(Some(PartitionInterventionMemberLayout {
            axis,
            coordinates,
            sum_offset_owner,
        }))
    }
}
