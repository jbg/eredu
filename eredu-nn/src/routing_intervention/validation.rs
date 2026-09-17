//! Shared control validation with ordinary or borrowed exclusion workspace.
use super::{GroupSelectionAction, GroupSelectionControl};
use crate::TopKGroupSelectionSpec;

/// A fixed semantic refusal before native routing work. No error string is owned.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum GroupSelectionValidationError {
    /// Actual router policy differs from the admitted source.
    #[error("runtime router policy differs from intervention admission")]
    Policy,
    /// Token-row range or stride is invalid.
    #[error("invalid intervention token-row selection")]
    Rows,
    /// IDs are empty, duplicated or outside the expert namespace.
    #[error("duplicate or out-of-range intervention expert IDs")]
    Ids,
    /// Exclusion leaves a possible group selection with too few experts.
    #[error("exclusion makes grouped top-k infeasible")]
    Exclusion,
    /// IDs and finite bias values differ in count or validity.
    #[error("routing bias requires one finite value per ID")]
    Bias,
    /// Forced IDs do not match selected rows and top-k width.
    #[error("forced IDs differ from selected-token/top-k shape")]
    ForcedShape,
    /// Forced IDs occupy more groups than the selected policy permits.
    #[error("forced IDs exceed selected-group count")]
    ForcedGroups,
}

pub(super) enum ExclusionWorkspace {
    Ordinary,
    Borrowed,
}
impl ExclusionWorkspace {
    fn available_sum(&self, width: i32, partitions: i32, selected: i32, ids: &[u32]) -> i32 {
        match self {
            Self::Ordinary => {
                // Preserve the existing ordinary Vec and sort. Its host payload
                // remains part of the selected native operation's host facts.
                let mut available = vec![width; partitions as usize];
                for id in ids {
                    available[*id as usize / width as usize] -= 1;
                }
                available.sort_unstable();
                available.iter().take(selected as usize).sum()
            }
            Self::Borrowed => {
                // Exact order statistics through repeated borrowed passes. A
                // partition ordinal breaks equal-count ties without storage;
                // ties contribute equal values to the ordinary sorted sum.
                let available = |partition: i32| {
                    width
                        - ids
                            .iter()
                            .filter(|id| **id / width as u32 == partition as u32)
                            .count() as i32
                };
                let mut sum = 0;
                for partition in 0..partitions {
                    let count = available(partition);
                    let rank = (0..partitions)
                        .filter(|other| (available(*other), *other) < (count, partition))
                        .count();
                    if rank < selected as usize {
                        sum += count;
                    }
                }
                sum
            }
        }
    }
}
impl GroupSelectionControl {
    /// Validates the exact policy and control source with no temporary sort
    /// vector. Repeated passes trade inspection time for zero workspace; they
    /// preserve the ordinary grouping feasibility and first-error ordering.
    pub fn validate_fixed(
        &self,
        actual: TopKGroupSelectionSpec,
        learned_scale: bool,
        token_rows: u64,
    ) -> Result<(), GroupSelectionValidationError> {
        self.validate_with_workspace(
            actual,
            learned_scale,
            token_rows,
            ExclusionWorkspace::Borrowed,
        )
    }
    pub(super) fn validate_with_workspace(
        &self,
        actual: TopKGroupSelectionSpec,
        learned_scale: bool,
        token_rows: u64,
        workspace: ExclusionWorkspace,
    ) -> Result<(), GroupSelectionValidationError> {
        let ensure = |condition, message| {
            if condition {
                Ok(())
            } else {
                Err(message)
            }
        };
        ensure(
            self.expected == actual && self.learned_coefficient_scale == learned_scale,
            GroupSelectionValidationError::Policy,
        )?;
        ensure(
            self.row_stride > 0 && self.first_row < self.end_row && self.end_row <= token_rows,
            GroupSelectionValidationError::Rows,
        )?;
        let rows = (self.end_row - self.first_row).div_ceil(self.row_stride);
        let check_ids = |ids: &[u32]| {
            ensure(
                !ids.is_empty()
                    && ids
                        .iter()
                        .enumerate()
                        .all(|(i, id)| *id < actual.group_count() as u32 && !ids[..i].contains(id)),
                GroupSelectionValidationError::Ids,
            )
        };
        match &self.action {
            GroupSelectionAction::Exclude(ids) => {
                check_ids(ids)?;
                let width = actual.group_count() / actual.selection_partitions();
                ensure(
                    workspace.available_sum(
                        width,
                        actual.selection_partitions(),
                        actual.selected_groups(),
                        ids,
                    ) >= actual.top_k(),
                    GroupSelectionValidationError::Exclusion,
                )?;
            }
            GroupSelectionAction::ZeroContribution(ids) => check_ids(ids)?,
            GroupSelectionAction::Bias { ids, values, .. } => {
                check_ids(ids)?;
                ensure(
                    ids.len() == values.len() && values.iter().all(|v| v.is_finite()),
                    GroupSelectionValidationError::Bias,
                )?;
            }
            GroupSelectionAction::Force(ids) => {
                ensure(
                    rows.checked_mul(actual.top_k() as u64) == Some(ids.len() as u64),
                    GroupSelectionValidationError::ForcedShape,
                )?;
                for row in ids.chunks(actual.top_k() as usize) {
                    check_ids(row)?;
                    let width = actual.group_count() / actual.selection_partitions();
                    // Borrow the admitted IDs rather than building another
                    // allocation of per-row IDs and partition membership.
                    let groups = row
                        .iter()
                        .enumerate()
                        .filter(|(i, id)| {
                            !row[..*i]
                                .iter()
                                .any(|prior| *prior / width as u32 == **id / width as u32)
                        })
                        .count();
                    ensure(
                        groups <= actual.selected_groups() as usize,
                        GroupSelectionValidationError::ForcedGroups,
                    )?;
                }
            }
        }
        Ok(())
    }
}
