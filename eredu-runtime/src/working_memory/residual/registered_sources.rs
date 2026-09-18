//! Mandatory registered inputs carried alongside, without changing, a quote.

use super::{IncrementalInferenceQuote, RegisteredStoragePin};
use crate::working_memory::{
    Usage, WorkingMemoryError, WorkingMemoryPool, WorkingMemoryStorage,
    saved_source::SavedSourceValidation,
};
use std::sync::Arc;

#[derive(Clone)]
pub(super) struct RegisteredInferenceSources {
    ordinary: Option<Arc<[Arc<dyn SavedSourceValidation + Send + Sync>]>>,
    publication: Option<Arc<dyn SavedSourceValidation + Send + Sync>>,
    inherited_capture:
        Option<crate::working_memory::storage::capture_publication::CaptureSourceOwner>,
}

impl RegisteredInferenceSources {
    // Moves only the quote's already retained immutable accounting bundle. No source
    // can be appended here; there is no new allocation, identity or credit.
    pub(super) fn into_witness(self, pool: WorkingMemoryPool) -> RegisteredInferenceSourceWitness {
        RegisteredInferenceSourceWitness {
            sources: Some(self),
            capture: None,
            pool,
        }
    }
}

impl std::fmt::Debug for RegisteredInferenceSources {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RegisteredInferenceSources")
            .field(
                "registrations",
                &self.ordinary.as_ref().map_or(0, |s| s.len()),
            )
            .field("publication", &self.publication.is_some())
            .finish()
    }
}

/// Accounting-only custody for the exact additional sources attached to a quote.
///
/// Extraction consumes the quote and keeps only its existing source bundle and
/// pool. This carries no numerical payload, diagnostic workspace, source credit,
/// reservation, execution permission or completion proof. It cannot add sources
/// or retrofit pins into an already-created funding run or its work scopes.
/// Keep the actual immutable source owners independently alive.
#[derive(Debug, Clone)]
#[must_use = "retain and revalidate until the accepted source installation boundary"]
pub struct RegisteredInferenceSourceWitness {
    sources: Option<RegisteredInferenceSources>,
    capture: Option<crate::working_memory::storage::capture_publication::CaptureSourceOwner>,
    pool: WorkingMemoryPool,
}

impl RegisteredInferenceSourceWitness {
    /// Fixed borrowed validator frames; no source payload or registration birth.
    pub fn capture_validation_control_bytes() -> Option<usize> {
        use std::mem::size_of;
        [size_of::<&Self>(), size_of::<&eredu_core::capture::SharedCapturePlan>(),
            size_of::<&WorkingMemoryPool>(),
            size_of::<Option<&crate::working_memory::storage::capture_publication::CaptureSourceOwner>>(),
            size_of::<std::sync::MutexGuard<'_, Usage>>(),
            size_of::<WorkingMemoryError>(), size_of::<Result<(), WorkingMemoryError>>()]
            .into_iter().try_fold(0usize,usize::checked_add)
    }
    /// Authenticate this exact previously published capture C source. The
    /// payload remains independently owned; this neither imports an ordinary
    /// plan nor publishes, credits or creates a capture/numerical allowance.
    pub fn validate_capture_source(
        &self,
        source: &eredu_core::capture::SharedCapturePlan,
        pool: &WorkingMemoryPool,
    ) -> Result<(), WorkingMemoryError> {
        let capture = self.capture.as_ref().or_else(||
            self.sources.as_ref().and_then(|sources| sources.inherited_capture.as_ref()))
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        if !capture.same_source(source.storage_identity()) {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        self.validate(pool)
    }
    /// Rechecks the original domain and every retained registered origin now.
    ///
    /// This borrows existing accounting data under the same Usage lock as the
    /// quote's source validator. It allocates no storage and changes no ledger,
    /// capacity, hold or scope. Success is a point-in-time health check, not a
    /// promise that another source account cannot become quarantined later.
    pub fn validate(&self, pool: &WorkingMemoryPool) -> Result<(), WorkingMemoryError> {
        if !self.pool.same_domain(pool) {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        let usage = self
            .pool
            .0
            .usage
            .lock()
            .map_err(|_| WorkingMemoryError::Poisoned)?;
        self.validate_locked(pool, &usage)
    }
    pub(in crate::working_memory) fn validate_locked(
        &self,
        pool: &WorkingMemoryPool,
        usage: &Usage,
    ) -> Result<(), WorkingMemoryError> {
        if !self.pool.same_domain(pool) {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        if let Some(sources) = &self.sources {
            sources.validate(pool, usage)?;
        }
        if let Some(capture) = &self.capture {
            capture.validate(pool, usage)?;
        }
        Ok(())
    }
    pub(in crate::working_memory) fn with_capture(
        previous: Option<Self>,
        pool: WorkingMemoryPool,
        capture: crate::working_memory::storage::capture_publication::CaptureSourceOwner,
    ) -> Self {
        Self {
            sources: previous.and_then(|p| p.sources),
            capture: Some(capture),
            pool,
        }
    }
}

struct Source<K: Ord + Send + 'static>(WorkingMemoryStorage<K>);

impl<K: Clone + Ord + Send + Sync + 'static> SavedSourceValidation for Source<K> {
    fn validate(&self, pool: &WorkingMemoryPool, usage: &Usage) -> Result<(), WorkingMemoryError> {
        self.0.validate_copy_source(pool, usage)
    }

    fn pin(&self) -> RegisteredStoragePin {
        RegisteredStoragePin::new(self.0.clone())
    }
}

impl SavedSourceValidation for RegisteredInferenceSources {
    fn validate(&self, pool: &WorkingMemoryPool, usage: &Usage) -> Result<(), WorkingMemoryError> {
        for source in self
            .ordinary
            .iter()
            .flat_map(|sources| sources.iter())
            .chain(self.publication.iter())
        {
            source.validate(pool, usage)?;
        }
        if let Some(capture) = &self.inherited_capture {
            capture.validate(pool, usage)?;
        }
        Ok(())
    }

    fn pin(&self) -> RegisteredStoragePin {
        RegisteredStoragePin::Sources(self.clone())
    }
}

impl IncrementalInferenceQuote {
    /// Consumes full diagnostics and extracts only already-attached source pins.
    ///
    /// The existing immutable bundle and pool move without allocation or a new
    /// identity. Diagnostic, controller and residual-credit custody retire here,
    /// outside any accounting lock. A quote with no attached sources returns
    /// `None`; an explicitly attached empty inventory still preserves its domain.
    /// Extraction itself does not validate current source health or change an
    /// existing reservation. Call the witness's `validate` at the later boundary.
    pub fn into_registered_source_witness(self) -> Option<RegisteredInferenceSourceWitness> {
        let Self {
            state,
            geometry: _,
            incremental_bytes: _,
            equation_incremental_bytes: _,
            pool,
            pin,
            controller,
            sources,
            span_workspace,
            span_seal,
        } = self;
        drop((state, pin, controller, span_workspace, span_seal));
        sources.map(|sources| sources.into_witness(pool))
    }

    /// Reuse the complete immutable source bundle from an actual installed
    /// capture. No registration, Vec, Arc body, source credit or grant is born.
    /// The fresh quote must still seal/publish its own capture destination.
    pub(in crate::working_memory) fn with_saved_capture_sources(
        mut self,
        witness: &RegisteredInferenceSourceWitness,
        source: &eredu_core::capture::SharedCapturePlan,
    ) -> Result<Self, WorkingMemoryError> {
        if self.sources.is_some() || !self.pool.same_domain(&witness.pool) {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        let capture = witness
            .capture
            .as_ref()
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        // A changed declaration is accepted only through the closed compiler's
        // exact retained parent. This transfers the parent's existing custody;
        // the new source still needs its own separately funded publication.
        let inherited = source.limit_revision_source().unwrap_or(source);
        if !capture.same_source(inherited.storage_identity()) {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        witness.validate(&self.pool)?;
        let mut sources = witness
            .sources
            .clone()
            .unwrap_or(RegisteredInferenceSources {
                ordinary: None,
                publication: None,
                inherited_capture: None,
            });
        // Repeated snapshots replace only the prior pin to this identical C
        // source. The latest completed publication retains its real funding.
        sources.inherited_capture = Some(capture.clone());
        self.sources = Some(sources);
        Ok(self)
    }

    /// Retains additional already charged inputs required by this exact quote.
    /// Full diagnostics, incremental demand and controller policy are unchanged;
    /// this supplies no source discount, registration or execution authority.
    ///
    /// Checks the pool and every attached registration's current funding origin
    /// now and again atomically when reserving, before any capacity handoff or
    /// accounting mutation. Successful reservations carry these accounting pins
    /// through the original funding run, work scopes and possible quarantine.
    /// Native providers still bind the complete actual inputs to this quote and
    /// establish safe access; the pins alone do not retain numerical payloads
    /// or prove completion. Cold quote clones retain the same registrations.
    pub fn with_registered_sources<K: Clone + Ord + Send + Sync + 'static>(
        mut self,
        storage: WorkingMemoryStorage<K>,
    ) -> Result<Self, WorkingMemoryError> {
        let source = Source(storage);
        {
            let usage = self
                .pool
                .0
                .usage
                .lock()
                .map_err(|_| WorkingMemoryError::Poisoned)?;
            source.validate(&self.pool, &usage)?;
            if let Some(previous) = &self.sources {
                previous.validate(&self.pool, &usage)?;
            }
        }
        let count = self
            .sources
            .as_ref()
            .and_then(|s| s.ordinary.as_ref())
            .map_or(0, |s| s.len())
            .checked_add(1)
            .ok_or(WorkingMemoryError::Overflow)?;
        let mut joined: Vec<Arc<dyn SavedSourceValidation + Send + Sync>> =
            Vec::with_capacity(count);
        if let Some(previous) = self.sources.as_ref().and_then(|s| s.ordinary.as_ref()) {
            joined.extend(previous.iter().cloned());
        }
        joined.push(Arc::new(source));
        let publication = self.sources.as_ref().and_then(|s| s.publication.clone());
        self.sources = Some(RegisteredInferenceSources {
            ordinary: Some(joined.into()),
            publication,
            inherited_capture: self
                .sources
                .as_ref()
                .and_then(|sources| sources.inherited_capture.clone()),
        });
        Ok(self)
    }
    // Fixed extra slot for the prepared existing C. No source Vec/slice is
    // rebuilt, and the original source bundle remains shared with every alias.
    pub(in crate::working_memory) fn with_source_validation(
        mut self,
        publication: Arc<dyn SavedSourceValidation + Send + Sync>,
    ) -> Result<Self, WorkingMemoryError> {
        if self
            .sources
            .as_ref()
            .is_some_and(|s| s.publication.is_some())
        {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        {
            let usage = self
                .pool
                .0
                .usage
                .lock()
                .map_err(|_| WorkingMemoryError::Poisoned)?;
            publication.validate(&self.pool, &usage)?;
            if let Some(prior) = &self.sources {
                prior.validate(&self.pool, &usage)?;
            }
        }
        self.sources
            .get_or_insert_with(|| RegisteredInferenceSources {
                ordinary: None,
                publication: None,
                inherited_capture: None,
            })
            .publication = Some(publication);
        Ok(self)
    }
}
