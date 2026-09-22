//! Shared logical snapshot policy; these allowances never certify native fit.
use eredu_runtime::StateLayout;

pub(crate) fn array_bytes(bytes: usize, rank: usize) -> Option<u64> {
    u64::try_from(bytes)
        .ok()?
        .checked_mul(2)?
        .checked_add(4096)?
        .checked_add(u64::try_from(rank).ok()?.checked_mul(16)?)
}

pub(crate) fn state_metadata(
    layout: &StateLayout,
    state_bytes: usize,
    inference_metadata: u64,
) -> Option<u64> {
    let mut retained = layout
        .logical_metadata_bytes()?
        .checked_add(inference_metadata)?
        .checked_add(u64::try_from(state_bytes).ok()?)?
        .checked_add(u64::try_from(layout.len()).ok()?.checked_mul(1024)?)?;
    for layer in 0..layout.len() {
        // Logical role slots remain present when their tensor is absent.
        retained = retained.checked_add(
            u64::try_from(layout.components(layer)?.len())
                .ok()?
                .checked_mul(256)?,
        )?;
    }
    Some(retained)
}

/// Shared logical copy-policy allowance from an immutable state description.
/// This neither reserves physical domains nor grants execution authority.
pub(crate) fn continuation_growth(
    layout: &StateLayout,
    frontier: u64,
    additional: u64,
    capacity: u64,
    auxiliary: u64,
) -> Option<u64> {
    // Ordinary text is a single sequence. Interpret only the declared
    // component geometry, including absent fixed tensors. MLX floating
    // storage is at most eight bytes; explicit integer components use four.
    // Charge the full future payload as an additional conservative allowance
    // (rather than subtracting currently retained data), including the same
    // materialization/descriptor allowance as an immutable copy.
    use eredu_core::cache::{StateTensorDimension as Dim, StateTensorDtype};
    let absolute = frontier.checked_add(additional)?;
    i32::try_from(absolute).ok()?;
    let prefix = absolute.max(capacity);
    i32::try_from(prefix).ok()?;
    let mut bytes = 0u64;
    for layer in 0..layout.len() {
        for component in layout.components(layer)? {
            let mut elements = 1u64;
            for dimension in component.shape() {
                let extent = match dimension {
                    Dim::Batch | Dim::Scalar => 1,
                    Dim::Fixed(n) => u64::from(n.get()),
                    Dim::PrefixTokens => prefix,
                    Dim::PrefixTokensDiv(n) => prefix / u64::from(n.get()),
                    // The final remainder is not the maximum over a run.
                    Dim::PrefixTokensRem(n) => prefix.min(u64::from(n.get()) - 1),
                };
                elements = elements.checked_mul(extent)?;
            }
            let width = match component.dtype() {
                StateTensorDtype::Floating => 8,
                StateTensorDtype::Float32 | StateTensorDtype::Int32 | StateTensorDtype::Uint32 => 4,
            };
            bytes = bytes
                .checked_add(elements.checked_mul(width)?.checked_mul(2)?)?
                .checked_add(4096)?
                .checked_add(
                    u64::try_from(component.shape().len())
                        .ok()?
                        .checked_mul(16)?,
                )?;
        }
    }
    bytes.checked_add(auxiliary)
}

#[cfg(test)]
mod tests {
    use super::*;
    use eredu_core::{cache::LayerCachePolicy, AttentionPolicy, LayerSchedule};

    fn layout() -> StateLayout {
        StateLayout::new(
            LayerSchedule::new(
                1,
                vec![LayerCachePolicy::key_value(AttentionPolicy::Full, 1, 2).unwrap()],
            )
            .unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn continuation_uses_retained_capacity_and_separate_auxiliary_growth() {
        let layout = layout();
        // Two rank-four KV tensors, each [1,1,8,2], plus the native catalog.
        assert_eq!(continuation_growth(&layout, 2, 2, 8, 128), Some(8960));
        assert!(continuation_growth(&layout, 2, 2, 4, 128).unwrap() < 8960);
        assert!(continuation_growth(&layout, 2, 7, 8, 128).unwrap() > 8960);
    }

    #[test]
    fn continuation_rejects_frontier_and_allowance_overflow() {
        let layout = layout();
        assert_eq!(continuation_growth(&layout, u64::MAX, 1, 0, 0), None);
        assert_eq!(
            continuation_growth(&layout, 0, 1, i32::MAX as u64 + 1, 0),
            None
        );
        assert_eq!(continuation_growth(&layout, 0, 1, 1, u64::MAX), None);
    }
}
