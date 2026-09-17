//! Shared logical projection charge; separate from physical metadata funding.
use super::*;
/// The ordinary projection policy, accumulated over actual local regions.
/// It grants no work, allocation or source permission; reserve its returned usage.
#[derive(Debug)]
pub struct PartitionInterventionProjectionCost {
    rank: usize,
    host_bytes: u64,
}
impl PartitionInterventionProjectionCost {
    /// The same base envelope, immutable admission names and physical shape.
    pub fn new(plan: &AdmittedInterventionPlan, rank: usize) -> Result<Self, CaptureError> {
        Ok(Self {
            rank,
            host_bytes: add(
                256,
                add(
                    plan.identity().len() as u64,
                    add(plan.intent_identity().len() as u64, mul(rank as u64, 8)?)?,
                )?,
            )?,
        })
    }
    /// One actual local region, including its selected payload or ID inventory.
    pub fn include(
        &mut self,
        action: &InterventionAction,
        selected: &[u64],
    ) -> Result<(), CaptureError> {
        let payload = match action {
            InterventionAction::Replace { tensor } | InterventionAction::Add { tensor } => mul(
                elements(selected)?,
                if tensor.values.dtype() == InterventionDtype::Float32 {
                    4
                } else {
                    2
                },
            )?,
            InterventionAction::Mask { .. } => elements(selected)?,
            InterventionAction::MaskComponents { indices, .. } => mul(indices.len() as u64, 96)?,
            InterventionAction::MaskLogits { token_ids, .. } => mul(token_ids.len() as u64, 96)?,
            _ => 0,
        };
        self.host_bytes = add(
            self.host_bytes,
            add(256, add(mul(self.rank as u64, 48)?, payload)?)?,
        )?;
        Ok(())
    }
    /// The same host-only charge used by ordinary partition reservation.
    pub fn usage(&self) -> CaptureUsage {
        CaptureUsage {
            host_bytes: self.host_bytes,
            ..Default::default()
        }
    }
    /// Fixed accumulation controls; actual region containers are separate.
    pub fn control_bytes() -> Option<usize> {
        use std::mem::{size_of, size_of_val};
        let frames = [
            size_of::<Self>(),
            size_of::<Result<Self, CaptureError>>(),
            size_of::<(&AdmittedInterventionPlan, usize)>(),
            size_of::<(&mut Self, &InterventionAction, &[u64])>(),
            size_of::<Result<(), CaptureError>>(),
            size_of::<CaptureUsage>(),
            size_of::<[u64; 4]>(),
            size_of::<(&InterventionOperation, &InterventionPoint)>(),
            size_of::<(&str, &str, &str, Option<usize>)>(),
            size_of::<Result<CaptureUsage, CaptureError>>(),
            size_of::<(&InterventionAction, &[u64], &ResolvedCaptureSlice)>(),
            size_of::<PartitionInterventionColumnError>(),
            size_of::<Result<bool, PartitionInterventionColumnError>>(),
        ];
        frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)
    }
}
/// Existing per-window metadata/encoding reservation, before shape projection.
pub fn intervention_window_metadata(
    operation: &InterventionOperation,
    point: &InterventionPoint,
) -> Result<CaptureUsage, CaptureError> {
    crate::capture::metadata_reservation_fields(
        &operation.id,
        &operation.target,
        &point.node_id,
        Some(point.axes.len()),
    )
}
/// Fixed refusal from the shared complete-column policy.
#[derive(Debug, Clone, Copy, thiserror::Error)]
#[error("column intervention requires the complete global final axis")]
pub struct PartitionInterventionColumnError;
/// Same ordinary compact-mask policy on an already normalized global slice.
pub fn validate_partition_column_region(
    action: &InterventionAction,
    global: &[u64],
    slice: &ResolvedCaptureSlice,
) -> Result<bool, PartitionInterventionColumnError> {
    let column = matches!(
        action,
        InterventionAction::MaskComponents { .. } | InterventionAction::MaskLogits { .. }
    );
    if column {
        let last = global
            .len()
            .checked_sub(1)
            .ok_or(PartitionInterventionColumnError)?;
        if slice.starts.get(last) != Some(&0)
            || slice.ends.get(last) != global.get(last)
            || slice.strides.get(last) != Some(&1)
        {
            return Err(PartitionInterventionColumnError);
        }
    }
    Ok(column)
}
