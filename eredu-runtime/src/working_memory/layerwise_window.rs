//! Capacity of the same group-bounded windows used by sequential execution.

use crate::ExecutionUnitLayout;
use eredu_core::{CapabilityError, WorkspaceBound};
use std::num::NonZeroUsize;

/// Fixed refusal from the allocation-free window calculation.
#[derive(Debug, Clone, Copy, thiserror::Error)]
pub enum LayerwiseWindowError {
    /// Every unit of a nonempty selected layout needs one physical fact.
    #[error("materialization facts differ from the selected execution layout")]
    Units,
    /// Summing one simultaneously live window overflowed.
    #[error("completed layerwise materialization window overflow")]
    Overflow,
}

/// Calculates the same completed window envelope from borrowed physical facts.
/// The callback supplies one immutable cost per ordinal and must not allocate.
/// Unknown costs remain unknown; known overlapping costs still detect overflow.
/// This supplies neither completion evidence nor source/admission authority.
pub fn completed_layerwise_window_bytes(
    layout: &ExecutionUnitLayout,
    depth: NonZeroUsize,
    unit_count: usize,
    mut unit_bytes: impl FnMut(usize) -> Option<u64>,
) -> Result<Option<u64>, LayerwiseWindowError> {
    if unit_count != layout.len() || layout.is_empty() {
        return Err(LayerwiseWindowError::Units);
    }
    let mut maximum = 0;
    let mut complete = true;
    for ordinal in 0..layout.len() {
        let range = layout
            .window_range(ordinal, depth)
            .expect("validated unit ordinal");
        let mut bytes = 0_u64;
        for index in range {
            match unit_bytes(index) {
                Some(amount) => {
                    bytes = bytes
                        .checked_add(amount)
                        .ok_or(LayerwiseWindowError::Overflow)?
                }
                None => complete = false,
            }
        }
        maximum = maximum.max(bytes);
    }
    Ok(complete.then_some(maximum))
}

/// Prices future materialization in a sequential, completed device window.
///
/// Each supplied bound covers all possible new destination allocations and
/// transfer staging for that physical invocation, including alias-named copies.
/// The backend must finish the preceding consumer/transfer and retire its native
/// references before replacing the window. Existing source/static/window backing
/// is separately registered, never subtracted here. Unit construction and equation
/// helpers belong in the enclosing equation span, not these transfer bounds.
///
/// This is an estimate, not completion evidence, a storage pin or admission.
/// Concurrent windows and retained old native references require a different
/// schedule or an additional explicit bound. Unknown required unit costs remain
/// unknown even if other windows have a known maximum.
pub fn quote_completed_layerwise_window(
    layout: &ExecutionUnitLayout,
    depth: NonZeroUsize,
    units: &[WorkspaceBound],
) -> Result<WorkspaceBound, CapabilityError> {
    if units.len() != layout.len() || layout.is_empty() {
        return Err(CapabilityError::InvalidConfiguration {
            field: "layerwise_workspace_units",
            detail: "materialization facts must name every unit of a nonempty selected layout"
                .into(),
        });
    }
    let mut reasons = std::collections::BTreeSet::new();
    let mut assumptions = std::collections::BTreeSet::new();
    for bound in units {
        match bound {
            WorkspaceBound::Unknown { reason } => {
                reasons.insert(reason.as_str());
            }
            WorkspaceBound::Bounded {
                assumptions: value, ..
            }
            | WorkspaceBound::PerDomain { assumptions: value } => {
                assumptions.insert(value.as_str());
            }
        }
    }
    let maximum =
        completed_layerwise_window_bytes(layout, depth, units.len(), |i| units[i].bytes())
            .map_err(|_| CapabilityError::ArithmeticOverflow {
                operation: "completed layerwise materialization window",
            })?;
    if !reasons.is_empty() {
        return Ok(WorkspaceBound::Unknown {
            reason: format!(
                "selected layerwise transfer window: {}",
                reasons.into_iter().collect::<Vec<_>>().join("; ")
            ),
        });
    }
    if maximum.is_none() {
        return Ok(WorkspaceBound::PerDomain {
            assumptions: format!(
                "selected completed layerwise windows retain per-domain requirements; {}",
                assumptions.into_iter().collect::<Vec<_>>().join("; ")
            ),
        });
    }
    Ok(WorkspaceBound::bounded(
        maximum.expect("all unit scalar facts are known"),
        format!(
            "maximum over sequential group-bounded lookahead windows of depth {}; preceding consumer and transfer settle and release native references before replacement; registered existing host/static/window storage is separate; constructor/equation costs are separate; {}",
            depth.get(),
            assumptions.into_iter().collect::<Vec<_>>().join("; "),
        ),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ExecutionGraph, ExecutionGroupSpec};

    fn layout() -> ExecutionUnitLayout {
        let graph = ExecutionGraph::new(
            vec![
                ExecutionGroupSpec::root("encoder"),
                ExecutionGroupSpec::with_dependencies("decoder", ["encoder"]),
            ],
            "decoder",
        )
        .unwrap();
        ExecutionUnitLayout::new(&graph, [3, 2]).unwrap()
    }

    #[test]
    fn windows_preserve_group_boundaries_and_partial_tails() {
        let layout = layout();
        let depth = NonZeroUsize::new(2).unwrap();
        let ranges: Vec<_> = (0..5)
            .map(|i| layout.window_range(i, depth).unwrap())
            .collect();
        assert_eq!(ranges, [0..2, 1..3, 2..3, 3..5, 4..5]);
        let costs = [3, 5, 101, 103, 7].map(|bytes| WorkspaceBound::bounded(bytes, "native copy"));
        assert_eq!(
            quote_completed_layerwise_window(&layout, depth, &costs)
                .unwrap()
                .bytes(),
            Some(110)
        );
        assert_eq!(
            quote_completed_layerwise_window(&layout, NonZeroUsize::new(1).unwrap(), &costs)
                .unwrap()
                .bytes(),
            Some(103)
        );
        assert_eq!(
            quote_completed_layerwise_window(
                &layout,
                NonZeroUsize::new(usize::MAX).unwrap(),
                &costs
            )
            .unwrap()
            .bytes(),
            Some(110)
        );
        assert_eq!(layout.window_range(5, depth), None);
    }

    #[test]
    fn unknown_and_mismatched_physical_facts_cannot_become_complete() {
        let layout = layout();
        let depth = NonZeroUsize::new(2).unwrap();
        let mut costs = vec![WorkspaceBound::bounded(3, "native"); 5];
        costs[4] = WorkspaceBound::Unknown {
            reason: "host copy unavailable".into(),
        };
        assert!(matches!(
            quote_completed_layerwise_window(&layout, depth, &costs).unwrap(),
            WorkspaceBound::Unknown { .. }
        ));
        assert!(quote_completed_layerwise_window(&layout, depth, &costs[..4]).is_err());
        costs[0] = WorkspaceBound::bounded(u64::MAX, "native");
        assert!(quote_completed_layerwise_window(&layout, depth, &costs).is_err());
    }
}
