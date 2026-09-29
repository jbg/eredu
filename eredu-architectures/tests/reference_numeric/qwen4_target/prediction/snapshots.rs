//! Snapshot estimates for the scalar fixture's complete combined state profile.
use super::*;
use eredu_core::execution_control::SnapshotEstimate;

fn estimate(bytes: u64) -> SnapshotEstimate {
    SnapshotEstimate {
        retained_bytes: bytes,
        copy_bytes: bytes,
    }
}
pub(super) fn tensor(value: &NumericTensor) -> Option<SnapshotEstimate> {
    let bytes = (std::mem::size_of::<NumericTensor>() as u64)
        .checked_add(std::mem::size_of::<NumericTensorScalars>() as u64)?
        .checked_add((value.shape.capacity() as u64).checked_mul(4)?)?
        .checked_add((value.data.capacity() as u64).checked_mul(4)?)?
        .checked_add(value.exact_i32.as_ref().map_or(Some(0), |ids| {
            (ids.capacity() as u64).checked_mul(4)?.checked_add(64)
        })?)?;
    Some(estimate(bytes))
}
pub(super) fn state(state: &State) -> Option<SnapshotEstimate> {
    let mut bytes = (std::mem::size_of::<State>() as u64)
        .checked_add(state.layout().logical_metadata_bytes()?.checked_mul(2)?)?;
    for layer in state.as_ref() {
        // These fixtures exercise ordinary K/V, fixed components and append streams.
        // Do not advertise estimates for an unaccounted compressed/pooling container.
        if layer.compressed.is_some() || layer.pooling.is_some() {
            return None;
        }
        // Charge an entire conservative B-tree node allowance per fixed entry.
        // Snapshot Vec clones have length-sized backing; source capacities below
        // also conservatively cover retained stream/tensor catalogs.
        bytes = bytes
            .checked_add(std::mem::size_of::<NumericHybridLayerState>() as u64)?
            .checked_add((layer.fixed.len() as u64).checked_mul(4096)?)?
            .checked_add(
                (layer.streams.capacity() as u64).checked_mul(std::mem::size_of::<(
                    u32,
                    u32,
                    eredu_runtime::ResidentAppendStream<NumericTensor>,
                )>() as u64)?,
            )?;
        for (_, _, stream) in &layer.streams {
            bytes = bytes.checked_add(stream.catalog_bytes()?)?;
        }
        for value in RuntimeLayerState::retained_values(layer) {
            bytes = bytes.checked_add(tensor(value)?.retained_bytes)?;
        }
    }
    Some(estimate(bytes))
}
thread_local! {
    pub(super) static COPIES: Cell<usize> = const { Cell::new(0) };
    pub(super) static FAIL_PREDICTION: Cell<bool> = const { Cell::new(false) };
}
pub(super) fn copied() {
    COPIES.with(|c| c.set(c.get() + 1));
}
