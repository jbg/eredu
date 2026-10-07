//! Completed independent host tensor copies for portable value snapshots.

use eredu_core::{
    execution_control::{ExecutionControlError, SnapshotEstimate},
    BackendFailure,
};
use eredu_runtime::execution_control::{SnapshotReservation, SnapshotValue};
use safemlx::{Array, Stream};

use super::{
    ordinary_retirement::{self, OrdinaryRetirement},
    submission_recovery::{self, Recovery, Retention, Status},
};

struct CopyResources {
    source: Array,
    copied: Option<Array>,
    contiguous: Option<Array>,
    stream: Stream,
    _reservation: SnapshotReservation,
}
struct CopyRetention(OrdinaryRetirement<CopyResources>);
impl Retention for CopyRetention {
    fn observe(&self, _: Status) {}
}

pub(crate) fn estimate_array(source: &Array) -> Result<SnapshotEstimate, BackendFailure> {
    // Source payload, compaction temporary and independent output, including
    // conservative native descriptors. This is logical snapshot accounting,
    // not a census of an aliased source's physical backing or allocator workspace.
    let bytes = (source.nbytes() as u64)
        .checked_add(4096)
        .and_then(|bytes| bytes.checked_add((source.shape().len() as u64).checked_mul(16)?))
        .and_then(|bytes| bytes.checked_mul(3))
        .and_then(|bytes| bytes.checked_add(std::mem::size_of::<CopyResources>() as u64))
        .ok_or_else(|| BackendFailure::from_error(ExecutionControlError::Overflow))?;
    Ok(SnapshotEstimate {
        retained_bytes: bytes,
        copy_bytes: bytes,
    })
}

pub(crate) fn copy_array(
    source: &Array,
    retention: &SnapshotReservation,
) -> Result<Array, BackendFailure> {
    submission_recovery::reap();
    ordinary_retirement::reclaim();
    let mut recovery = Recovery::begin(CopyRetention(OrdinaryRetirement::new(CopyResources {
        source: source.clone(),
        copied: None,
        contiguous: None,
        stream: Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0)),
        _reservation: retention.clone(),
    })))
    .map_err(BackendFailure::from_error)?;
    // Snapshot payloads use independent host storage. Explicit CPU compaction
    // preserves arbitrary logical strides before the host deep copy; consuming
    // model operations retain their own selected execution streams.
    let resources = &recovery.retention().0;
    let contiguous = resources
        .source
        .contiguous(false, &resources.stream)
        .map_err(BackendFailure::from_error)?;
    recovery.retention_mut().0.contiguous = Some(contiguous);
    let copied = recovery
        .retention()
        .0
        .contiguous
        .as_ref()
        .expect("retained compact source")
        .clone()
        .deep_clone()
        .map_err(BackendFailure::from_error)?;
    recovery.retention_mut().0.copied = Some(copied);
    recovery.seal();
    let status = recovery.wait();
    if status.failed || status.blocked {
        return Err(BackendFailure::from_error(
            safemlx::error::Exception::custom(
                "tensor snapshot copy failed; unresolved resources remain retained",
            ),
        ));
    }
    let copied = recovery
        .retention_mut()
        .0
        .copied
        .take()
        .expect("completed independent copy");
    recovery.finish();
    ordinary_retirement::reclaim();
    Ok(copied)
}

impl SnapshotValue for crate::MlxTensor {
    fn estimate_snapshot_value(&self) -> Result<SnapshotEstimate, BackendFailure> {
        estimate_array(self.as_array())
    }
    fn copy_snapshot_value(&self, retention: &SnapshotReservation) -> Result<Self, BackendFailure> {
        copy_array(self.as_array(), retention).map(Self::from_array)
    }
}
