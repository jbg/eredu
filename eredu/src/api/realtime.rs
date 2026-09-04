//! Backend-neutral realtime preparation and scheduling facade.
//!
//! These types remain available when no concrete execution backend feature is
//! enabled. Architecture and runtime crates own the underlying selection,
//! transaction, and scheduling semantics; this module gives applications one
//! facade namespace for composing those semantics with reference, mock, or
//! selected native mechanisms.

pub use eredu_architectures::moshi::{
    inspect_moshi_realtime, prepare_realtime_model_from_catalog, select_inspected_moshi_realtime,
    InspectedMoshiRealtime, MoshiRealtimeRequest, PreparedMoshiRealtime, RealtimePreparationPlan,
};
pub use eredu_core::{
    scheduler::{
        RequestId, RequestStatus, SchedulerCapabilities, SchedulerLimits, SchedulerReport,
    },
    RealtimeInputFrame, RealtimeOutputFrame, RealtimeSampling, RealtimeSpeechConfig,
    SessionCapabilities,
};
pub use eredu_runtime::{
    RealtimeGenerationState, RealtimeModelSessionIdentity, RealtimeSessionScheduler,
    ReleasedRealtimeSession,
};

/// A backend-neutral prepared realtime model bound to one authoritative
/// selected realization.
///
/// `M` may be an architecture-owned constructed model, a reference mechanism,
/// or an opaque selected-native execution object. Session identity and
/// capabilities stay portable regardless of that mechanism type.
pub struct PreparedRealtimeModel<M> {
    mechanism: M,
    identity: RealtimeModelSessionIdentity,
    capabilities: SessionCapabilities,
}

impl<M> PreparedRealtimeModel<M> {
    /// Binds a mechanism to the exact identity selected before construction.
    pub fn new(
        mechanism: M,
        selected: &eredu_runtime::SelectedRealtimeRealization,
        capabilities: SessionCapabilities,
    ) -> Self {
        Self {
            mechanism,
            identity: RealtimeModelSessionIdentity::from_selected(selected),
            capabilities,
        }
    }

    /// Returns the exact portable identity required by every session.
    pub const fn session_identity(&self) -> &RealtimeModelSessionIdentity {
        &self.identity
    }

    /// Returns capabilities of the exact selected realization.
    pub const fn session_capabilities(&self) -> SessionCapabilities {
        self.capabilities
    }

    /// Borrows the architecture or execution mechanism retained by this model.
    pub const fn mechanism(&self) -> &M {
        &self.mechanism
    }

    /// Mutably borrows the retained mechanism for frame execution.
    pub const fn mechanism_mut(&mut self) -> &mut M {
        &mut self.mechanism
    }

    /// Consumes the facade model into its retained mechanism.
    pub fn into_mechanism(self) -> M {
        self.mechanism
    }
}
