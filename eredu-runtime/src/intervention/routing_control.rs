//! Shared source projection for ordinary and prepared routing interventions.
use super::*;
use eredu_nn::routing_intervention::{
    GroupScoreStage, GroupSelectionAction, GroupSelectionControl,
};
use std::mem::{size_of, size_of_val};

/// Fixed preparation failures; no formatted diagnostic or payload allocation.
#[derive(Debug, thiserror::Error)]
pub enum PreparedRoutingControlError {
    /// The source action, policy or selected axes disagree.
    #[error("routing control differs from its admitted source")]
    Source,
    /// A checked extent cannot be represented.
    #[error("routing control extent overflow")]
    Overflow,
    /// The actual constructor grant was refused.
    #[error(transparent)]
    Funding(#[from] eredu_core::HostMetadataFundingError),
    /// The destination allocator refused the paid request.
    #[error(transparent)]
    Allocation(#[from] std::collections::TryReserveError),
    /// The destination capacity differs from the quoted request.
    #[error("routing control allocation capacity differs from its quote")]
    Capacity,
}

/// Borrowed control projection from one admitted operation and its actual rows.
/// This describes the existing selector and grants no execution authority.
pub struct PreparedRoutingControl<'a> {
    operation: &'a InterventionOperation,
    policy: &'a InterventionRoutingPolicy,
    expected: eredu_nn::TopKGroupSelectionSpec,
    first_row: u64,
    end_row: u64,
    row_stride: u64,
    payload: std::ops::Range<usize>,
    elements: usize,
}
impl<'a> PreparedRoutingControl<'a> {
    /// Resolve physical overlap and the corresponding immutable payload range.
    /// The caller first resolves `slice` through the admitted plan's common
    /// geometry worker and authenticates its source, phase and invocation.
    pub fn inspect(
        operation: &'a InterventionOperation,
        policy: &'a InterventionRoutingPolicy,
        token_rows: u64,
        slice: &ResolvedCaptureSlice,
        window: Option<CaptureInvocationWindow>,
    ) -> Result<Option<Self>, PreparedRoutingControlError> {
        use PreparedRoutingControlError as E;
        if slice.starts.len() != 2
            || slice.ends.len() != 2
            || slice.strides.len() != 2
            || slice.shape.len() != 2
            || slice.strides[0] == 0
            || slice.starts[1] != 0
            || slice.ends[1] != u64::from(policy.top_k)
            || slice.strides[1] != 1
            || slice.shape[1] != u64::from(policy.top_k)
        {
            return Err(E::Source);
        }
        let (first_row, end_row, payload_start, payload_end) = if let Some(window) = window {
            let end = window.start.checked_add(token_rows).ok_or(E::Overflow)?;
            let ordinal = window
                .start
                .saturating_sub(slice.starts[0])
                .div_ceil(slice.strides[0]);
            let first = slice.starts[0]
                .checked_add(ordinal.checked_mul(slice.strides[0]).ok_or(E::Overflow)?)
                .ok_or(E::Overflow)?;
            let limit = end.min(slice.ends[0]);
            if first >= limit {
                return Ok(None);
            }
            let rows = (limit - first).div_ceil(slice.strides[0]);
            (
                first - window.start,
                limit - window.start,
                ordinal
                    .checked_mul(u64::from(policy.top_k))
                    .ok_or(E::Overflow)?,
                ordinal
                    .checked_add(rows)
                    .and_then(|n| n.checked_mul(u64::from(policy.top_k)))
                    .ok_or(E::Overflow)?,
            )
        } else {
            if slice.ends[0] > token_rows {
                return Err(E::Source);
            }
            (
                slice.starts[0],
                slice.ends[0],
                0,
                slice.shape[0]
                    .checked_mul(u64::from(policy.top_k))
                    .ok_or(E::Overflow)?,
            )
        };
        let payload = usize::try_from(payload_start).map_err(|_| E::Overflow)?
            ..usize::try_from(payload_end).map_err(|_| E::Overflow)?;
        let elements = match &operation.action {
            InterventionAction::ExcludeExperts { expert_ids }
            | InterventionAction::ZeroExpertContribution { expert_ids } => expert_ids.len(),
            InterventionAction::ForceExperts { expert_ids, .. } => {
                expert_ids.get(payload.clone()).ok_or(E::Source)?.len()
            }
            InterventionAction::BiasRoutingScores {
                expert_ids, biases, ..
            } => {
                if expert_ids.len() != biases.len() {
                    return Err(E::Source);
                }
                expert_ids
                    .len()
                    .checked_add(biases.len())
                    .ok_or(E::Overflow)?
            }
            _ => return Err(E::Source),
        };
        let groups = i32::try_from(policy.groups).map_err(|_| E::Overflow)?;
        let selected_groups = i32::try_from(policy.selected_groups).map_err(|_| E::Overflow)?;
        let expected = eredu_nn::TopKGroupSelectionSpec::new(
            i32::try_from(policy.expert_count).map_err(|_| E::Overflow)?,
            i32::try_from(policy.top_k).map_err(|_| E::Overflow)?,
            match policy.scoring {
                RoutingScoring::Softmax => eredu_nn::GroupScoring::Softmax,
                RoutingScoring::SelectedSoftmax => eredu_nn::GroupScoring::SelectedSoftmax,
                RoutingScoring::Sigmoid => eredu_nn::GroupScoring::Sigmoid,
                RoutingScoring::SqrtSoftplus => eredu_nn::GroupScoring::SqrtSoftplus,
            },
            policy.normalize_selected,
        )
        .and_then(|spec| spec.with_groups(groups, selected_groups))
        .and_then(|spec| {
            spec.with_weight_policy(policy.normalization_epsilon, policy.coefficient_scale)
        })
        .map_err(|_| E::Source)?;
        Ok(Some(Self {
            operation,
            policy,
            expected,
            first_row,
            end_row,
            row_stride: slice.strides[0],
            payload,
            elements,
        }))
    }

    /// Existing logical host charge; native selector work remains independently
    /// attributed by its actual operation and evidence mechanisms.
    pub fn usage(&self) -> Result<CaptureUsage, PreparedRoutingControlError> {
        let host_bytes = size_of::<GroupSelectionControl>()
            .checked_add(
                self.elements
                    .checked_mul(size_of::<u32>())
                    .ok_or(PreparedRoutingControlError::Overflow)?,
            )
            .and_then(|n| u64::try_from(n).ok())
            .ok_or(PreparedRoutingControlError::Overflow)?;
        Ok(CaptureUsage {
            host_bytes,
            ..Default::default()
        })
    }
    /// Requested payload capacities and fixed construction/retirement controls.
    pub fn required_bytes(&self) -> Option<usize> {
        let frames = [
            size_of::<Self>(),
            size_of::<GroupSelectionControl>(),
            size_of::<GroupSelectionAction>(),
            size_of::<[Vec<u32>; 2]>(),
            size_of::<Result<GroupSelectionControl, PreparedRoutingControlError>>(),
            size_of::<Result<(), std::collections::TryReserveError>>(),
            size_of::<(&Self, &eredu_core::HostMetadataFunding)>(),
            size_of::<[usize; 2]>(),
        ];
        frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)?
            .checked_add(self.elements.checked_mul(size_of::<u32>())?)
    }
    /// Construct only after the original host account accepts every payload and
    /// control byte. The enclosing source retains that account through its use.
    pub fn construct(
        &self,
        funding: &eredu_core::HostMetadataFunding,
    ) -> Result<GroupSelectionControl, PreparedRoutingControlError> {
        funding.reserve_metadata(
            self.required_bytes()
                .ok_or(PreparedRoutingControlError::Overflow)?,
        )?;
        self.copy_control()
    }
    // Ordinary CaptureSession has its existing independent host owner and has
    // already spent usage(); both paths share this exact payload copy worker.
    pub(super) fn copy_control(
        &self,
    ) -> Result<GroupSelectionControl, PreparedRoutingControlError> {
        fn copy<T: Copy>(source: &[T]) -> Result<Vec<T>, PreparedRoutingControlError> {
            let mut output = Vec::new();
            output.try_reserve_exact(source.len())?;
            if output.capacity() != source.len() {
                return Err(PreparedRoutingControlError::Capacity);
            }
            output.extend_from_slice(source);
            Ok(output)
        }
        let action = match &self.operation.action {
            InterventionAction::ExcludeExperts { expert_ids } => {
                GroupSelectionAction::Exclude(copy(expert_ids)?)
            }
            InterventionAction::ZeroExpertContribution { expert_ids } => {
                GroupSelectionAction::ZeroContribution(copy(expert_ids)?)
            }
            InterventionAction::ForceExperts { expert_ids, .. } => {
                GroupSelectionAction::Force(copy(&expert_ids[self.payload.clone()])?)
            }
            InterventionAction::BiasRoutingScores {
                stage,
                expert_ids,
                biases,
            } => GroupSelectionAction::Bias {
                stage: match stage {
                    RoutingScoreStage::RawLogits => GroupScoreStage::RawLogits,
                    RoutingScoreStage::TransformedScores => GroupScoreStage::TransformedScores,
                    RoutingScoreStage::RankingScores => GroupScoreStage::RankingScores,
                },
                ids: copy(expert_ids)?,
                values: copy(biases)?,
            },
            _ => return Err(PreparedRoutingControlError::Source),
        };
        Ok(GroupSelectionControl {
            expected: self.expected.clone(),
            learned_coefficient_scale: self.policy.learned_coefficient_scale,
            first_row: self.first_row,
            end_row: self.end_row,
            row_stride: self.row_stride,
            action,
            capture_original: self.operation.evidence != InterventionEvidence::None,
        })
    }
}
