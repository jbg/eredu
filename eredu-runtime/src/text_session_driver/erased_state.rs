//! Type erasure for isolated text-executable state.

use crate::execution_control::{SessionState, SessionStateSnapshot};
use eredu_core::{
    execution_control::{ControlSupport, SnapshotEstimate},
    BackendFailure,
};
use std::any::Any;

/// One saved executable state plus its native retirement lifetime.
pub struct ErasedTextState {
    value: Box<dyn Any>,
    // Drop the state before allowing the backend to reclaim retired resources.
    retirement: Option<Box<dyn Any>>,
}

impl ErasedTextState {
    pub(crate) fn new<S: 'static>(state: S) -> Self {
        Self {
            value: Box::new(state),
            retirement: None,
        }
    }

    pub(super) fn retain(mut self, retirement: Option<Box<dyn Any>>) -> Self {
        self.retirement = retirement;
        self
    }

    pub(crate) fn include_slot_cost(estimate: SnapshotEstimate) -> Option<SnapshotEstimate> {
        let bytes = std::mem::size_of::<Self>() as u64;
        Some(SnapshotEstimate {
            retained_bytes: estimate.retained_bytes.checked_add(bytes)?,
            copy_bytes: estimate.copy_bytes.checked_add(bytes)?,
        })
    }

    pub(crate) fn typed<S: 'static>(&self) -> Result<&S, BackendFailure> {
        self.value.downcast_ref().ok_or_else(foreign_state)
    }

    pub(crate) fn typed_mut<S: 'static>(&mut self) -> Result<&mut S, BackendFailure> {
        self.value.downcast_mut().ok_or_else(foreign_state)
    }
}

fn foreign_state() -> BackendFailure {
    state_error("saved text state has a different executable type")
}

fn state_error(message: impl Into<Box<dyn std::error::Error + Send + Sync>>) -> BackendFailure {
    BackendFailure::from_error(std::io::Error::other(message))
}

/// Snapshot operations for one portable state owner, erased over its state type.
pub trait ErasedTextStateAccess<C: ?Sized> {
    /// Reports support for this exact installed realization.
    fn support(&self, embedded_prediction: bool) -> ControlSupport;
    /// Estimates copying installed or saved state.
    fn estimate_copy(
        &self,
        saved: Option<&ErasedTextState>,
    ) -> Result<Option<SnapshotEstimate>, BackendFailure>;
    /// Estimates growth of saved state without allocating.
    fn estimate_growth(
        &self,
        saved: &ErasedTextState,
        additional: u64,
    ) -> Result<Option<u64>, BackendFailure>;
    /// Estimates growth of the installed state without allocating.
    fn estimate_installed_growth(&self, additional: u64) -> Option<u64>;
    /// Makes an independent copy of installed state.
    fn save(&mut self, context: &C) -> Result<ErasedTextState, BackendFailure>;
    /// Makes an independent copy of saved state.
    fn copy(
        &mut self,
        saved: &ErasedTextState,
        context: &C,
    ) -> Result<ErasedTextState, BackendFailure>;
    /// Validates original ownership and selected geometry.
    fn validate(&self, saved: &ErasedTextState) -> Result<(), BackendFailure>;
    /// Exchanges the saved and installed states.
    fn exchange(&mut self, saved: &mut ErasedTextState, context: &C) -> Result<(), BackendFailure>;
    /// Estimates the empty state required by a parameter overlay.
    fn estimate_parameter_reset(&self) -> Option<SnapshotEstimate> {
        None
    }
    /// Prepares a local reset slot under the enclosing parameter transaction.
    fn prepare_parameter_reset(&mut self, _: &C) -> Result<ErasedTextState, BackendFailure> {
        Err(state_error("parameter state reset is unavailable"))
    }
    /// Exchanges the reset slot after the enclosing parameter coordinator agrees.
    fn exchange_parameter_reset(&mut self, _: &mut ErasedTextState) -> Result<(), BackendFailure> {
        Err(state_error("parameter state exchange is unavailable"))
    }
    /// Invalidates snapshots that predate parameter publication.
    fn invalidate_parameter_snapshots(&mut self) {}
}

/// Binds an installed executable's portable state owner to its native context.
pub trait InstalledTextState {
    /// Native context used by independent state copies or rank agreement.
    type Context: ?Sized;

    /// State owner, if the executable exposes one.
    fn control_state(&self) -> Option<&dyn ErasedTextStateAccess<Self::Context>>;
    /// Mutable state owner and its exact native context.
    fn control_state_mut(
        &mut self,
    ) -> Option<(
        &mut dyn ErasedTextStateAccess<Self::Context>,
        &Self::Context,
    )>;
    /// Whether the executable also has embedded prediction state.
    fn has_embedded_prediction(&self) -> bool;
    /// Optional backend resource reclamation guard, released after the saved state.
    /// Any heap storage owned by the guard is part of the backend copy estimate.
    fn snapshot_retirement(&self) -> Option<Box<dyn Any>> {
        None
    }
}

pub(super) fn state_support<M: InstalledTextState>(model: &M) -> ControlSupport {
    match model.control_state() {
        Some(state) => state.support(model.has_embedded_prediction()),
        None => ControlSupport::Unsupported {
            reason: "complete native state copying is unavailable for this executable".into(),
        },
    }
}

pub(super) fn supported_state<M: InstalledTextState>(
    model: &M,
) -> Result<&dyn ErasedTextStateAccess<M::Context>, BackendFailure> {
    if let ControlSupport::Unsupported { reason } = state_support(model) {
        return Err(state_error(reason));
    }
    model.control_state().ok_or_else(unavailable_state)
}

pub(super) fn unavailable_state() -> BackendFailure {
    state_error("text state ownership is unavailable")
}

/// Native copying and storage estimates for a state retained by a portable session.
pub trait TextSnapshotStorage<C: ?Sized>: Sized {
    /// Complete cost of an isolated copy, when known.
    fn snapshot_estimate(&self) -> Option<SnapshotEstimate>;
    /// Additional storage needed for a continuation, when known.
    fn snapshot_growth(&self, additional: u64) -> Option<u64>;
    /// Completes an independently writable copy.
    fn snapshot_copy(&self, context: &C) -> Result<Self, BackendFailure>;
}

impl<S: TextSnapshotStorage<C> + 'static, C: ?Sized> ErasedTextStateAccess<C> for SessionState<S> {
    fn support(&self, _: bool) -> ControlSupport {
        if self.get().snapshot_estimate().is_some() {
            ControlSupport::Supported
        } else {
            ControlSupport::Unsupported {
                reason: "decoder state storage is unavailable for an isolated snapshot".into(),
            }
        }
    }

    fn estimate_copy(
        &self,
        saved: Option<&ErasedTextState>,
    ) -> Result<Option<SnapshotEstimate>, BackendFailure> {
        let state = match saved {
            Some(saved) => self
                .saved_state(saved.typed()?)
                .map_err(BackendFailure::from_error)?,
            None => self.get(),
        };
        Ok(state.snapshot_estimate())
    }

    fn estimate_growth(
        &self,
        saved: &ErasedTextState,
        additional: u64,
    ) -> Result<Option<u64>, BackendFailure> {
        let state = self
            .saved_state(saved.typed()?)
            .map_err(BackendFailure::from_error)?;
        Ok(state.snapshot_growth(additional))
    }

    fn estimate_installed_growth(&self, additional: u64) -> Option<u64> {
        self.get().snapshot_growth(additional)
    }

    fn save(&mut self, context: &C) -> Result<ErasedTextState, BackendFailure> {
        self.capture_state(|state| state.snapshot_copy(context))
            .map(ErasedTextState::new)
    }

    fn copy(
        &mut self,
        saved: &ErasedTextState,
        context: &C,
    ) -> Result<ErasedTextState, BackendFailure> {
        self.copy_state(
            saved.typed::<SessionStateSnapshot<S>>()?,
            |state| state.snapshot_copy(context),
            BackendFailure::from_error,
        )
        .map(ErasedTextState::new)
    }

    fn validate(&self, saved: &ErasedTextState) -> Result<(), BackendFailure> {
        self.saved_state(saved.typed()?)
            .map(|_| ())
            .map_err(BackendFailure::from_error)
    }

    fn exchange(&mut self, saved: &mut ErasedTextState, _: &C) -> Result<(), BackendFailure> {
        self.exchange_state(saved.typed_mut()?)
            .map_err(BackendFailure::from_error)
    }
}
