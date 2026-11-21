//! Independent value copies and opaque native retention authority.

use std::{any::Any, rc::Rc};

use super::SnapshotEstimate;
use crate::{BackendFailure, RealtimeDecisionDiagnostics};

/// Keeps a caller-owned retention charge alive through native completion.
/// Native code can share this authority with recovery resources but cannot
/// observe the budget, change the charge, or admit a new copy.
#[derive(Clone)]
pub struct SnapshotRetention {
    _owner: Rc<dyn Any>,
}

impl SnapshotRetention {
    /// Erases a shared retention owner without changing its lifetime.
    pub fn from_owner(owner: Rc<impl Any>) -> Self {
        Self { _owner: owner }
    }
}

/// A native or component copy cost could not be represented.

#[derive(Debug, thiserror::Error)]
#[error("snapshot storage cost overflow")]
pub struct SnapshotCostOverflow;

/// A self-contained value copied alongside an isolated model-state snapshot.
///
/// Unlike `Clone`, this contract requires independent mutable contents and a
/// complete conservative storage estimate. Immutable resources may be shared only
/// when their lifetime and storage remain accounted for by the enclosing owner.
/// Implementations perform no random draws or model transitions.
pub trait SnapshotValue: Sized {
    /// Complete retained and copy cost, including dynamic storage and native data.
    fn estimate_snapshot_value(&self) -> Result<SnapshotEstimate, BackendFailure>;

    /// Makes an exact independent copy and establishes any native copy completion.
    /// Native recovery must retain a clone of `retention` when failed submitted
    /// work can outlive this call. Success leaves no pending copy operation.
    fn copy_snapshot_value(&self, retention: &SnapshotRetention) -> Result<Self, BackendFailure>;
}

/// Adds complete component estimates to a container's inline metadata, using
/// checked arithmetic. Double-counted inline fields are conservative; callers
/// must still include every independently copied dynamic allocation.
pub fn snapshot_value_estimate<T>(
    components: impl IntoIterator<Item = Result<SnapshotEstimate, BackendFailure>>,
) -> Result<SnapshotEstimate, BackendFailure> {
    let mut total = SnapshotEstimate {
        retained_bytes: std::mem::size_of::<T>() as u64,
        copy_bytes: std::mem::size_of::<T>() as u64,
    };
    for component in components {
        let component = component?;
        total.retained_bytes = total
            .retained_bytes
            .checked_add(component.retained_bytes)
            .ok_or_else(|| BackendFailure::from_error(SnapshotCostOverflow))?;
        total.copy_bytes = total
            .copy_bytes
            .checked_add(component.copy_bytes)
            .ok_or_else(|| BackendFailure::from_error(SnapshotCostOverflow))?;
    }
    Ok(total)
}

impl<T: SnapshotValue> SnapshotValue for Option<T> {
    fn estimate_snapshot_value(&self) -> Result<SnapshotEstimate, BackendFailure> {
        snapshot_value_estimate::<Self>(self.iter().map(SnapshotValue::estimate_snapshot_value))
    }
    fn copy_snapshot_value(&self, retention: &SnapshotRetention) -> Result<Self, BackendFailure> {
        self.as_ref()
            .map(|value| value.copy_snapshot_value(retention))
            .transpose()
    }
}
impl<T: SnapshotValue> SnapshotValue for Vec<T> {
    fn estimate_snapshot_value(&self) -> Result<SnapshotEstimate, BackendFailure> {
        // Estimates also admit live fork successors, whose allocations have not
        // been compacted by copy_snapshot_value. Price spare source slots too.
        let slots = self
            .capacity()
            .checked_mul(std::mem::size_of::<T>())
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or_else(|| BackendFailure::from_error(SnapshotCostOverflow))?;
        snapshot_value_estimate::<Self>(
            std::iter::once(Ok(SnapshotEstimate {
                retained_bytes: slots,
                copy_bytes: 0,
            }))
            .chain(self.iter().map(SnapshotValue::estimate_snapshot_value)),
        )
    }
    fn copy_snapshot_value(&self, retention: &SnapshotRetention) -> Result<Self, BackendFailure> {
        let mut copied = Vec::new();
        copied
            .try_reserve_exact(self.len())
            .map_err(BackendFailure::from_error)?;
        for value in self {
            copied.push(value.copy_snapshot_value(retention)?);
        }
        Ok(copied)
    }
}
macro_rules! scalar_snapshot {
    ($($scalar:ty),*) => { $(
        impl SnapshotValue for $scalar {
            fn estimate_snapshot_value(&self) -> Result<SnapshotEstimate, BackendFailure> {
                snapshot_value_estimate::<Self>([])
            }
            fn copy_snapshot_value(&self, _: &SnapshotRetention) -> Result<Self, BackendFailure> { Ok(*self) }
        }
    )* };
}
scalar_snapshot!(i32, u32, usize, f32, ());

impl SnapshotValue for RealtimeDecisionDiagnostics {
    fn estimate_snapshot_value(&self) -> Result<SnapshotEstimate, BackendFailure> {
        let bytes = self
            .tensor()
            .retained_storage_bytes()
            .ok_or_else(|| BackendFailure::from_error(SnapshotCostOverflow))?;
        snapshot_value_estimate::<Self>([Ok(SnapshotEstimate {
            retained_bytes: bytes,
            copy_bytes: bytes,
        })])
    }

    fn copy_snapshot_value(&self, _: &SnapshotRetention) -> Result<Self, BackendFailure> {
        let mut shape = Vec::new();
        shape
            .try_reserve_exact(self.shape().len())
            .map_err(BackendFailure::from_error)?;
        shape.extend_from_slice(self.shape());
        let mut logits = Vec::new();
        logits
            .try_reserve_exact(self.logits().len())
            .map_err(BackendFailure::from_error)?;
        logits.extend_from_slice(self.logits());
        Self::new(self.prediction(), shape, logits).map_err(BackendFailure::from_error)
    }
}
