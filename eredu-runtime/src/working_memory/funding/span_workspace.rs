//! Closed original-account host custody for the actual accepted span plan.
use super::*;
mod owners;
use crate::working_memory::{
    residual::TextControlBinding, InferenceRequest, RegisteredInferenceSourceWitness,
    ReservedInferenceSpanWorkspace,
};
pub(in crate::working_memory) use owners::retirement_control_bytes;
pub(in crate::working_memory) use owners::{RawSpanHostOwner, SpanHostOwner};

#[derive(Debug)]
pub(in crate::working_memory) struct SpanHostCustody {
    controls: Option<TextControlBinding>,
    sources: Option<RegisteredInferenceSourceWitness>,
    // Required only for the sealed single-C transition. No ordinary guard may
    // escape until this immutable, complete S+C association has been installed.
    publication: std::sync::OnceLock<RegisteredInferenceSourceWitness>,
    // Last, with no back-edge: source sidecars retain only this raw hold.
    raw: RawSpanHostOwner,
}

#[derive(Debug)]
pub(in crate::working_memory) struct RawSpanHostCustody {
    execution: InferenceExecutionIdentity,
    reservation: u64,
    host: WorkingMemoryDecoderHostScope,
}
impl SpanHostCustody {
    pub(in crate::working_memory) fn validate_issuance_locked(
        &self,
        pool: &WorkingMemoryPool,
        usage: &Usage,
        reservation: &WorkingMemoryReservation,
    ) -> Result<(), WorkingMemoryError> {
        let binding = self
            .controls
            .as_ref()
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        self.validate_control_locked(pool, usage, reservation, binding)
    }

    pub(in crate::working_memory) fn validate_native_buffer_role(
        &self,
    ) -> Result<(), WorkingMemoryError> {
        let pool = self.raw.host.pool();
        let usage = pool
            .0
            .usage
            .lock()
            .map_err(|_| WorkingMemoryError::Poisoned)?;
        // A previously consumed native role can outlive the run owner. Keep
        // all origin/source/publication health, without reopening issuance.
        self.validate_pin_origin_locked(pool, &usage)
    }
    pub(in crate::working_memory) fn validate_native_publication(
        &self,
        native: &WorkingMemoryFundingScope,
    ) -> Result<(), WorkingMemoryError> {
        let usage = native
            .pool()
            .0
            .usage
            .lock()
            .map_err(|_| WorkingMemoryError::Poisoned)?;
        self.validate_native_publication_locked(native, &usage)
    }
    pub(in crate::working_memory) fn validate_native_publication_locked(
        &self,
        native: &WorkingMemoryFundingScope,
        usage: &Usage,
    ) -> Result<(), WorkingMemoryError> {
        let original = self
            .raw
            .host
            .scope
            .as_ref()
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        if self.controls.is_none()
            || original.id != native.id
            || !self.raw.host.pool().same_domain(native.pool())
        {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        // Publication may complete after the request handle and quote metadata
        // retire. Preserve every source/publication witness check, without
        // incorrectly requiring the allocation issuer to remain open.
        self.validate_pin_origin_locked(native.pool(), usage)?;
        FundingSource::NativeScope(native).validate(usage, &self.raw.execution)?;
        usage
            .funding
            .get(&native.id)
            .ok_or(WorkingMemoryError::IdentityMismatch)?
            .validate_native_publication(native)
    }

    fn validate_account_locked(
        &self,
        pool: &WorkingMemoryPool,
        usage: &Usage,
    ) -> Result<(), WorkingMemoryError> {
        self.validate_prepublication_locked(pool, usage)?;
        if self.requires_publication() {
            self.publication
                .get()
                .ok_or(WorkingMemoryError::PreparationAlreadyStarted)?
                .validate_locked(pool, usage)?;
        }
        let scope = self
            .raw
            .host
            .scope
            .as_ref()
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        let state = usage
            .funding
            .get(&scope.id)
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        if !state.run_open || !state.metadata_live {
            return Err(WorkingMemoryError::ExecutionFenced);
        }
        Ok(())
    }
    fn matches_reservation(
        &self,
        reservation: &WorkingMemoryReservation,
    ) -> Result<(), WorkingMemoryError> {
        let scope = self
            .raw
            .host
            .scope
            .as_ref()
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        if self.raw.reservation != reservation.0.account_id
            || reservation.0.funding != Some(scope.id)
            || !self.raw.host.pool().same_domain(&reservation.0.pool)
        {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        Ok(())
    }
    pub(in crate::working_memory) fn validate(
        &self,
        receipt: &ReservedInferenceSpanWorkspace<'_>,
    ) -> Result<(), WorkingMemoryError> {
        self.matches_reservation(receipt.reservation())?;
        let pool = self.raw.host.pool();
        let usage = pool
            .0
            .usage
            .lock()
            .map_err(|_| WorkingMemoryError::Poisoned)?;
        self.validate_account_locked(pool, &usage)?;
        receipt.validate_sources(pool, &usage)
    }
    pub(in crate::working_memory) fn validate_control_locked(
        &self,
        pool: &WorkingMemoryPool,
        usage: &Usage,
        reservation: &WorkingMemoryReservation,
        binding: &TextControlBinding,
    ) -> Result<(), WorkingMemoryError> {
        if !self
            .controls
            .as_ref()
            .is_some_and(|actual| actual.same(binding))
        {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        self.matches_reservation(reservation)?;
        self.validate_account_locked(pool, usage)
    }
    pub(in crate::working_memory) fn validate_control_reservation(
        &self,
        reservation: &WorkingMemoryReservation,
    ) -> Result<(), WorkingMemoryError> {
        let binding = self
            .controls
            .as_ref()
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        self.matches_reservation(reservation)?;
        let pool = self.raw.host.pool();
        let usage = pool
            .0
            .usage
            .lock()
            .map_err(|_| WorkingMemoryError::Poisoned)?;
        self.validate_control_locked(pool, &usage, reservation, binding)
    }
    pub(in crate::working_memory) fn validate_control_native(
        &self,
        native: &WorkingMemoryFundingScope,
    ) -> Result<(), WorkingMemoryError> {
        if self.controls.is_none() {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        let original = self
            .raw
            .host
            .scope
            .as_ref()
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        if native.id != original.id || !native.pool.same_domain(self.raw.host.pool()) {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        let pool = self.raw.host.pool();
        let usage = pool
            .0
            .usage
            .lock()
            .map_err(|_| WorkingMemoryError::Poisoned)?;
        self.validate_account_locked(pool, &usage)?;
        FundingSource::NativeScope(native).validate(&usage, &self.raw.execution)
    }
}
impl WorkingMemoryFundingRun {
    pub(in crate::working_memory) fn hold_inference_span_workspace(
        &self,
        receipt: &ReservedInferenceSpanWorkspace<'_>,
    ) -> Result<SpanHostOwner, WorkingMemoryError> {
        let reservation = receipt.reservation();
        if reservation.0.funding != Some(self.id) || !self.pool.same_domain(&reservation.0.pool) {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        let bytes = receipt.workspace().protected_peak_bytes()?;
        // Accounting-only clones are staged outside Usage, including every
        // original explicit source validator required by escaped control guards.
        let execution = reservation.0.execution.clone();
        let reservation_identity = reservation.0.account_id;
        let controls = receipt.workspace().control_binding().cloned();
        let sources = receipt.source_witness();
        let mut usage = self
            .pool
            .0
            .usage
            .lock()
            .map_err(|_| WorkingMemoryError::Poisoned)?;
        FundingSource::CopyRun(self).validate(&usage, &execution)?;
        receipt.validate_sources(&self.pool, &usage)?;
        let state = usage
            .funding
            .get_mut(&self.id)
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        if !state.metadata_live {
            return Err(WorkingMemoryError::ExecutionFenced);
        }
        state.validate_span_spend(None)?;
        let available = state.spendable_remaining()?;
        if bytes > available {
            return Err(WorkingMemoryError::BudgetExceeded {
                required_bytes: bytes,
                available_bytes: available,
            });
        }
        let held = state
            .host_held
            .checked_add(bytes)
            .ok_or(WorkingMemoryError::Overflow)?;
        let scopes = state
            .scopes
            .checked_add(1)
            .ok_or(WorkingMemoryError::Overflow)?;
        state.host_held = held;
        state.scopes = scopes;
        drop(usage);
        Ok(SpanHostOwner::new(SpanHostCustody {
            controls,
            sources,
            publication: std::sync::OnceLock::new(),
            raw: RawSpanHostOwner::new(RawSpanHostCustody {
                execution,
                reservation: reservation_identity,
                host: WorkingMemoryDecoderHostScope {
                    scope: Some(WorkingMemoryFundingScope {
                        purpose: ScopePurpose::Host,
                        pool: self.pool.clone(),
                        id: self.id,
                        active: true,
                        borrowed_storage: None,
                        capture_source: None,
                        native_publication_identity: None,
                    }),
                    held: bytes,
                },
            }),
        }))
    }
}

#[cfg(test)]
thread_local! {
    static SOURCE_TO_QUARANTINE: std::cell::RefCell<Option<WorkingMemoryFundingScope>> = const { std::cell::RefCell::new(None) };
}
#[cfg(test)]
pub(in crate::working_memory) fn quarantine_source_after_next_attachment(
    source: WorkingMemoryFundingScope,
) {
    SOURCE_TO_QUARANTINE.with(|slot| assert!(slot.borrow_mut().replace(source).is_none()));
}
#[cfg(test)]
pub(in crate::working_memory) fn after_span_attachment() {
    // A real independent original source account becomes unhealthy here. This
    // callback runs outside Usage and removes only an actual native scope.
    let source = SOURCE_TO_QUARANTINE.with(|slot| slot.borrow_mut().take());
    drop(source);
}

impl SpanHostCustody {
    pub(in crate::working_memory) fn pin_source(
        &self,
    ) -> Result<&eredu_core::SharedStorageIdentity, WorkingMemoryError> {
        self.controls
            .as_ref()
            .and_then(|controls| controls.source_identity())
            .ok_or(WorkingMemoryError::IdentityMismatch)
    }
    pub(in crate::working_memory) fn validate_pin_locked(
        &self,
        pool: &WorkingMemoryPool,
        usage: &Usage,
        native: &WorkingMemoryFundingScope,
        layout: &Arc<crate::working_memory::storage::bounded_pin::PinLayout>,
        request: &InferenceRequest,
    ) -> Result<(), WorkingMemoryError> {
        let binding = self
            .controls
            .as_ref()
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        if !binding
            .pin_layout()
            .is_some_and(|actual| Arc::ptr_eq(actual, layout))
        {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        let reservation = request
            .memory_reservation()
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        self.validate_control_locked(pool, usage, reservation, binding)?;
        let scope = self
            .raw
            .host
            .scope
            .as_ref()
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        if native.id != scope.id || !native.pool.same_domain(pool) {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        FundingSource::NativeScope(native).validate(usage, &self.raw.execution)
    }
    // Healthy registered sources outlive run closure. Preserve original account
    // quarantine and explicit-origin checks, without requiring run_open.
    pub(in crate::working_memory) fn validate_pin_origin_locked(
        &self,
        pool: &WorkingMemoryPool,
        usage: &Usage,
    ) -> Result<(), WorkingMemoryError> {
        self.validate_prepublication_locked(pool, usage)?;
        if self.requires_publication() {
            self.publication
                .get()
                .ok_or(WorkingMemoryError::PreparationAlreadyStarted)?
                .validate_locked(pool, usage)?;
        }
        Ok(())
    }
}

impl RawSpanHostCustody {
    pub(in crate::working_memory) fn execution(&self) -> &InferenceExecutionIdentity {
        &self.execution
    }
    pub(in crate::working_memory) fn pool(&self) -> &WorkingMemoryPool {
        self.host.pool()
    }
    pub(in crate::working_memory) fn account(&self) -> u64 {
        self.host.scope.as_ref().expect("live raw host custody").id
    }
    pub(in crate::working_memory) fn quarantine(&self) {
        let mut usage = lock_for_retirement(self.pool());
        usage
            .funding
            .get_mut(&self.account())
            .expect("live raw host custody")
            .quarantined = true;
    }
    pub(in crate::working_memory) fn validate_origin_locked(
        &self,
        pool: &WorkingMemoryPool,
        usage: &Usage,
    ) -> Result<(), WorkingMemoryError> {
        self.host.validate(pool, usage, &self.execution)
    }
    pub(in crate::working_memory) fn validate_publication_locked(
        &self,
        native: &WorkingMemoryFundingScope,
        usage: &Usage,
    ) -> Result<(), WorkingMemoryError> {
        let original = self
            .host
            .scope
            .as_ref()
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        if original.id != native.id || !self.host.pool().same_domain(&native.pool) {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        self.validate_origin_locked(&native.pool, usage)?;
        FundingSource::NativeScope(native).validate(usage, &self.execution)?;
        let state = usage
            .funding
            .get(&native.id)
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        if !state.run_open || !state.metadata_live {
            return Err(WorkingMemoryError::ExecutionFenced);
        }
        state.validate_span_spend(Some(native))
    }
}
impl SpanHostCustody {
    pub(in crate::working_memory) fn source_scope(
        &self,
    ) -> Result<&WorkingMemoryFundingScope, WorkingMemoryError> {
        self.raw
            .host
            .scope
            .as_ref()
            .ok_or(WorkingMemoryError::IdentityMismatch)
    }
    pub(in crate::working_memory) fn validate_source_publication_locked(
        &self,
        usage: &Usage,
        reservation: &WorkingMemoryReservation,
    ) -> Result<(), WorkingMemoryError> {
        let binding = self
            .controls
            .as_ref()
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        self.validate_control_locked(self.raw.pool(), usage, reservation, binding)
    }
    pub(in crate::working_memory) fn raw(&self) -> &RawSpanHostOwner {
        &self.raw
    }
    pub(in crate::working_memory) fn requires_publication(&self) -> bool {
        self.controls
            .as_ref()
            .is_some_and(|c| c.publication_layout().is_some())
    }
    pub(in crate::working_memory) fn validate_prepublication_locked(
        &self,
        pool: &WorkingMemoryPool,
        usage: &Usage,
    ) -> Result<(), WorkingMemoryError> {
        self.raw.validate_origin_locked(pool, usage)?;
        if let Some(sources) = &self.sources {
            sources.validate_locked(pool, usage)?;
        }
        Ok(())
    }
    pub(in crate::working_memory) fn validate_pending(
        &self,
        receipt: &ReservedInferenceSpanWorkspace<'_>,
    ) -> Result<(), WorkingMemoryError> {
        self.matches_reservation(receipt.reservation())?;
        if !self.requires_publication() || self.publication.get().is_some() {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        let pool = self.raw.host.pool();
        let usage = pool
            .0
            .usage
            .lock()
            .map_err(|_| WorkingMemoryError::Poisoned)?;
        self.validate_prepublication_locked(pool, &usage)?;
        receipt.validate_sources(pool, &usage)
    }
    pub(in crate::working_memory) fn complete_publication(
        &self,
        witness: RegisteredInferenceSourceWitness,
    ) -> Result<(), WorkingMemoryError> {
        if !self.requires_publication() {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        self.publication
            .set(witness)
            .map_err(|_| WorkingMemoryError::PreparationAlreadyStarted)
    }
}

impl SpanHostCustody {
    pub(in crate::working_memory) fn validate_batch_publication_locked<
        K: crate::working_memory::storage::CapturePlanStorageKey,
    >(
        &self,
        pool: &WorkingMemoryPool,
        usage: &Usage,
        native: &WorkingMemoryFundingScope,
        layout: &Arc<crate::working_memory::storage::bounded_publication::BatchPublicationLayout>,
        request: &InferenceRequest,
    ) -> Result<(), WorkingMemoryError> {
        let binding = self
            .controls
            .as_ref()
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        if !binding
            .batch_layout()
            .is_some_and(|actual| Arc::ptr_eq(actual, layout))
            || !binding
                .publication_layout()
                .is_some_and(|actual| Arc::ptr_eq(actual, &layout.original))
            || !layout.matches::<K>()
        {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        let reservation = request
            .memory_reservation()
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        self.validate_control_locked(pool, usage, reservation, binding)?;
        self.raw.validate_publication_locked(native, usage)?;
        // Full witness health above includes the actual completed original C
        // attachment. Require that same typed key in the same existing directory;
        // a bounded commit never creates the TypeId namespace.
        crate::working_memory::storage::bounded_publication::validate_original_namespace::<K>(
            pool,
            usage,
            &layout.original,
        )
    }
}
