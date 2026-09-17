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
