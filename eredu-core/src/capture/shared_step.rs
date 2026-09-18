//! Read-only aliases of an already constructed capture frame and its custody.
use super::*;
use std::fmt;
#[cfg(test)]
use std::sync::Arc;

use super::retained_payload::RetainedCaptureOwner;

type StepOwner = RetainedCaptureOwner<CapturedStep>;

/// Shared immutable ownership of one completed or failed capture frame.
///
/// Clone shares the frame and its retained custody. Serialization is identical
/// to `CapturedStep`, including the capture tensor's nonfinite wire policy.
/// There is no mutable or owning raw export or detachable custody. Deserialization
/// creates independent diagnostic ownership; it does not recover source identity,
/// funding, transaction authority or native completion from the wire.
///
/// This type alone certifies no allocation bound, provenance, quota, transaction
/// success or native completion. Those remain the constructing runtime's job.
#[derive(Clone)]
pub struct SharedCapturedStep(StepOwner);

/// Unique, preallocated frame ownership before its final transaction outcome.
///
/// This runtime bridge exposes no payload, alias, mutable export or admission
/// authority. Construction has the same accounting obligations as `retain`;
/// finalization only records the enclosing driver's outcome.
#[doc(hidden)]
pub struct UnpublishedCapturedStep(StepOwner);

impl UnpublishedCapturedStep {
    /// Fixed retained allocation layouts for this concrete custody representation.
    /// Arc's two atomic counters follow the pinned standard-library ArcInner
    /// representation; allocator overhead and a compiler stack are not included.
    #[doc(hidden)]
    pub fn retained_control_bytes<T: Send + Sync + 'static>() -> Option<u64> {
        StepOwner::retained_control_bytes::<T>()
    }

    /// Packages already accounted payload/custody, allocating the final shared
    /// owner now. The caller must finish all fallible checks before publication.
    /// Custody must not retain this owner, directly or indirectly.
    pub fn retain(step: CapturedStep, custody: impl Send + Sync + 'static) -> Self {
        Self(StepOwner::retain(step, custody))
    }

    /// Records one final outcome and publishes the existing owner. No allocation,
    /// callback, refund, account validation or native certification occurs here.
    pub fn finish(mut self, outcome: CaptureStepOutcome) -> SharedCapturedStep {
        // This closed type has no Clone, weak/strong alias, or payload export.
        // Therefore the allocation remains uniquely owned until this move.
        self.0.get_mut().outcome = outcome;
        SharedCapturedStep(self.0)
    }
}
impl fmt::Debug for UnpublishedCapturedStep {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("UnpublishedCapturedStep")
            .finish_non_exhaustive()
    }
}

impl SharedCapturedStep {
    /// Runtime ownership bridge, not a construction or funding grant.
    ///
    /// The caller must already own and account for the complete frame and any
    /// required construction overlap. Nested independently shared tensors retain
    /// their own custody. `custody` must not retain this new shared owner itself.
    /// No byte allowance or native completion can be established by this call.
    #[doc(hidden)]
    pub fn retain(step: CapturedStep, custody: impl Send + Sync + 'static) -> Self {
        let outcome = step.outcome;
        UnpublishedCapturedStep::retain(step, custody).finish(outcome)
    }

    /// Borrows the frame without transferring ownership or allocation authority.
    /// An explicit caller clone of the raw DTO is a separate caller-owned copy.
    pub fn as_step(&self) -> &CapturedStep {
        self.0.get()
    }

    /// Whether both aliases retain the same actual frame owner.
    pub fn same_storage(&self, other: &Self) -> bool {
        self.0.same_storage(&other.0)
    }

    /// Outcome of the original enclosing model transaction.
    pub fn outcome(&self) -> CaptureStepOutcome {
        self.0.get().outcome
    }

    /// Original forward phase.
    pub fn phase(&self) -> CapturePhase {
        self.0.get().phase
    }

    /// Explicit physical invocation geometry, when present.
    pub fn invocation(&self) -> Option<CaptureInvocationShape> {
        self.0.get().invocation
    }

    /// Original run-relative prediction index.
    pub fn prediction_index(&self) -> u64 {
        self.0.get().prediction_index
    }

    /// Borrows every record, including missing, skipped and failed outcomes.
    pub fn records(&self) -> &[CaptureRecord] {
        &self.0.get().records
    }

    /// Borrows verified partition provenance without granting producer authority.
    pub fn partitions(&self) -> &[PartitionCaptureEvidence] {
        &self.0.get().partitions
    }

    /// Borrows attributed intervention results; no execution authority is supplied.
    pub fn interventions(&self) -> &[crate::intervention::InterventionRecord] {
        &self.0.get().interventions
    }

    /// Existing nonrefunding logical usage for this step.
    pub fn step_usage(&self) -> CaptureUsage {
        self.0.get().step_usage
    }

    /// Existing nonrefunding logical usage for the original run.
    pub fn cumulative_usage(&self) -> CaptureUsage {
        self.0.get().cumulative_usage
    }

    /// Recorded capture wall time.
    pub fn capture_seconds(&self) -> f64 {
        self.0.get().capture_seconds
    }
}
impl AsRef<CapturedStep> for SharedCapturedStep {
    fn as_ref(&self) -> &CapturedStep {
        self.as_step()
    }
}
impl PartialEq for SharedCapturedStep {
    fn eq(&self, other: &Self) -> bool {
        self.as_step() == other.as_step()
    }
}
impl fmt::Debug for SharedCapturedStep {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SharedCapturedStep")
            .field("phase", &self.phase())
            .field("prediction_index", &self.prediction_index())
            .field("outcome", &self.outcome())
            .field("records", &self.records().len())
            .finish_non_exhaustive()
    }
}
impl Serialize for SharedCapturedStep {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.as_step().serialize(serializer)
    }
}

impl std::ops::Deref for SharedCapturedStep {
    type Target = CapturedStep;
    fn deref(&self) -> &Self::Target { self.as_step() }
}
impl<'de> serde::Deserialize<'de> for SharedCapturedStep {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        // Wire diagnostics have independent ownership and create no source,
        // funding, transaction, or completion authority.
        let step = <CapturedStep as serde::Deserialize>::deserialize(deserializer)?;
        Ok(Self::retain(step, ()))
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod precommit_tests;

mod prepared;
pub use prepared::PreparedCapturedStep;
