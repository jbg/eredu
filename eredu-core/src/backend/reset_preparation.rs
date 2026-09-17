//! Explicit ordinary settlement followed by one original, nonblocking reset.

use super::{
    BackendFailure, BackendFailureKind, BackendSession, ModelRuntime, SessionCapabilities,
    TextGenerationBackend,
};
use crate::{SessionAdmission, SessionResetClaim, SessionResetLimits, SessionResetRejection};

/// Read-only admission evidence from an actual exclusively borrowed session.
///
/// This reports capabilities only; it grants no readiness, bytes or access to
/// the session. The backend's associated owner establishes readiness and retains
/// the exclusive borrow. Querying this report must not allocate, execute,
/// synchronize, reap or invoke application code. It must describe the current
/// session rather than a saved report from before ordinary preparation.
pub trait SessionResetReadiness {
    /// Returns the actual current session capabilities for exact core validation.
    fn capabilities(&self) -> SessionCapabilities;
}

/// Optional preparation for a consuming originally admitted reset.
///
/// The associated owner must hold the actual mutable session borrow, establish
/// successful completion of loading, transfers and submissions, and exclude all
/// aliases capable of starting session work for that borrow. An idle submission
/// counter alone is insufficient. Unknown producer coverage rejects.
///
/// Concrete readiness constructors and fields must remain private to the
/// implementation. Do not expose a session reference, Deref, Clone or another
/// route to re-enter execution through the owner. Dropping it only ends the
/// borrow: it must not synchronize, progress native work or publish state.
/// Existing backends need not implement this extension.
pub trait SessionResetPreparationBackend: TextGenerationBackend {
    /// Concrete readiness retaining the actual exclusive session borrow.
    type Readiness<'s>: SessionResetReadiness
    where
        Self: 's;

    /// Explicit ordinary preparation, outside the later original reset account.
    ///
    /// Waiting and its controls obey the ordinary construction/accounting
    /// contract. This must not obtain an unquoted owner when existing original
    /// accounts exclude it, clear prior accounts, or attribute preparation to a
    /// future reset grant. Failure returns no readiness. Live producers may
    /// return a Busy failure; unknown/unsupported completion coverage rejects.
    fn prepare_session_reset_ordinary<'s>(
        _backend: &'s Self,
        _session: &'s mut Self::Session,
    ) -> Result<Self::Readiness<'s>, BackendFailure>
    where
        Self: 's,
    {
        Err(unsupported())
    }

    /// Consumes real readiness and the claim for that same actual session.
    ///
    /// Validate claim/session, current health and the exact source, selection,
    /// revision and domain before one original comparison and construction.
    /// No unreserved wait, native progress, global reap or replacement grant is
    /// permitted. Failure preserves installed state and all unresolved owners.
    /// Host publication and displaced-state retirement need their own complete
    /// originally priced controls. Returning success does not authorize reuse
    /// of readiness after queued retirement or another operation.
    fn reset_ready_session_admitted<'s>(
        _backend: &'s Self,
        _ready: Self::Readiness<'s>,
        _claim: SessionResetClaim<'_>,
    ) -> Result<(), BackendFailure>
    where
        Self: 's,
    {
        Err(unsupported())
    }
}

fn unsupported() -> BackendFailure {
    BackendFailure::new(
        BackendFailureKind::Unsupported,
        SessionResetRejection::Unsupported,
    )
}

/// Exclusive preparation for one originally admitted reset.
///
/// Created only by [`ModelRuntime::prepare_reset_ordinary`]. It provides no
/// runtime/session/backend accessor, generic callback or claim extraction.
/// Dropping this owner only abandons the prepared operation and ends its borrow.
/// Preparation is ordinary work; this owner does not fund that earlier work.
///
/// The runtime cannot be re-entered while its readiness is retained:
///
/// ```compile_fail
/// use eredu_core::{ModelRuntime, SessionResetPreparationBackend, SessionResetLimits};
/// fn overlap<B: SessionResetPreparationBackend>(runtime: &mut ModelRuntime<B>) {
///     let ready = runtime.prepare_reset_ordinary().unwrap();
///     let _ = runtime.session_mut();
///     ready.reset_admitted(SessionResetLimits::new(1024)).unwrap();
/// }
/// ```
///
/// Readiness is consumed even when the reset rejects:
///
/// ```compile_fail
/// use eredu_core::{ModelRuntime, SessionResetPreparationBackend, SessionResetLimits};
/// fn twice<B: SessionResetPreparationBackend>(runtime: &mut ModelRuntime<B>) {
///     let ready = runtime.prepare_reset_ordinary().unwrap();
///     let _ = ready.reset_admitted(SessionResetLimits::new(1024));
///     let _ = ready.reset_admitted(SessionResetLimits::new(1024));
/// }
/// ```
#[must_use = "preparation is consumed by one reset or dropped without resetting"]
pub struct PreparedSessionReset<'s, B: SessionResetPreparationBackend + 's> {
    backend: &'s B,
    admission: &'s SessionAdmission,
    session_address: usize,
    ready: B::Readiness<'s>,
}

impl<B: SessionResetPreparationBackend> ModelRuntime<B> {
    /// Explicitly prepares this session for one consuming original reset.
    ///
    /// The backend may wait under its ordinary accounting contract. Those costs
    /// are not paid by the later reset. Unsupported preparation rejects; a live
    /// detached producer may return Busy without establishing completion. No
    /// original claim or reset account exists during this preparation.
    pub fn prepare_reset_ordinary(
        &mut self,
    ) -> Result<PreparedSessionReset<'_, B>, BackendFailure> {
        self.admission.validate(self.session.capabilities())?;
        let session_address = std::ptr::from_ref(&self.session).cast::<()>() as usize;
        let ready = B::prepare_session_reset_ordinary(&self.backend, &mut self.session)?;
        self.admission.validate(ready.capabilities())?;
        Ok(PreparedSessionReset {
            backend: &self.backend,
            admission: &self.admission,
            session_address,
            ready,
        })
    }
}

impl<B: SessionResetPreparationBackend> PreparedSessionReset<'_, B> {
    /// Consumes this readiness for one original reset without implicitly waiting.
    ///
    /// Core revalidates actual capabilities, then creates the genuine claim for
    /// the session captured before the exclusive borrow. The backend must bind
    /// the actual source and obtain original domain acceptance before allocation.
    /// This neither refills prior accounts nor authorizes native waiting.
    pub fn reset_admitted(self, limits: SessionResetLimits) -> Result<(), BackendFailure> {
        self.admission.validate(self.ready.capabilities())?;
        let claim =
            SessionResetClaim::from_prepared_address(self.session_address, self.admission, limits);
        B::reset_ready_session_admitted(self.backend, self.ready, claim)
    }
}
