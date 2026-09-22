//! Closed account exclusion for one accepted, canonically stamped equation span.
use super::*;
use crate::{
    inspection::PrefillChunkRetentionContext,
    working_memory::{InferenceWorkspaceSpan, ReservedInferenceSpanWorkspace},
};

/// Inline account metadata, sharing the already allocated stamp identity weakly.
/// Never upgrade this to custody: the stamp itself owns the original reservation
/// and host bank. Expiration remains exclusion, not implicit reopening.
#[derive(Debug)]
pub(in crate::working_memory::funding) struct ActiveCaptureSpan(Weak<CaptureSourceIdentity>);
impl ActiveCaptureSpan {
    pub(in crate::working_memory::funding) fn matches(
        &self,
        scope: Option<&WorkingMemoryFundingScope>,
    ) -> bool {
        scope
            .and_then(|scope| scope.capture_source.as_ref())
            .is_some_and(|slot| self.0.as_ptr() == Arc::as_ptr(&slot.identity))
    }
}

impl CaptureSourceSegment {
    /// Exclude other spenders only after the exact accepted original span fits.
    ///
    /// Working-memory private until the canonical opening-state/native bridge
    /// also proves complete current opening publication and actual native
    /// quiescence. This mechanism grants neither of those facts. No scalar
    /// allowance, new hold, scope, reservation, certification or credit is made.
    pub(in crate::working_memory) fn activate_reserved_span(
        &self,
        native: &mut WorkingMemoryFundingScope,
        receipt: &ReservedInferenceSpanWorkspace<'_>,
        context: &PrefillChunkRetentionContext<'_>,
    ) -> Result<(), WorkingMemoryError> {
        if receipt.workspace().text_controls().is_some() {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        self.activate_span_impl(native, receipt, None, context)
    }
    /// P+Q variant requiring the matching existing-only opening group. Neither
    /// complete opening inventory nor native settlement follows from this
    /// typed association.
    pub(in crate::working_memory) fn activate_reserved_text_span(
        &self,
        native: &mut WorkingMemoryFundingScope,
        receipt: &crate::working_memory::ReservedTextSpanWorkspace<'_>,
        context: &PrefillChunkRetentionContext<'_>,
    ) -> Result<(), WorkingMemoryError> {
        self.activate_span_impl(native, &receipt.span, Some(receipt), context)
    }
    fn activate_span_impl(
        &self,
        native: &mut WorkingMemoryFundingScope,
        receipt: &ReservedInferenceSpanWorkspace<'_>,
        text: Option<&crate::working_memory::ReservedTextSpanWorkspace<'_>>,
        context: &PrefillChunkRetentionContext<'_>,
    ) -> Result<(), WorkingMemoryError> {
        let pool = native.pool.clone();
        let mut usage = pool
            .0
            .usage
            .lock()
            .map_err(|_| WorkingMemoryError::Poisoned)?;
        // Immutable identity checks and mutable account/source checks share the
        // same lock. No observed headroom escapes this transaction.
        if let Some(text) = text {
            text.validate_locked(&pool, &usage)?;
        } else {
            receipt.validate_sources(&pool, &usage)?;
        }
        receipt.validate_request(context.request())?;
        let stamp = self
            .identity
            .prefill
            .as_ref()
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        stamp.validate_context_coordinates(context)?;
        self.custody
            .validate_scheduled_native_locked(native, &usage)?;
        let reservation = receipt.reservation();
        if reservation.0.funding != Some(native.id) || !pool.same_ledger(&reservation.0.pool) {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        FundingSource::NativeScope(native).validate(&usage, &reservation.0.execution)?;
        let slot = self.slot(native)?;
        if let Some(text) = text {
            // This exact text custody was validated above. Inspect each opening
            // origin directly once, with no nested Usage loan or slot recursion.
            slot.opening
                .as_ref()
                .ok_or(WorkingMemoryError::IdentityMismatch)?
                .validate_activation(native, &usage, self, context, text.controls)?;
            slot.validate_sources(&pool, &usage)?;
        } else {
            slot.validate(&pool, &usage)?;
        }
        let span = InferenceWorkspaceSpan::Prefill(context.chunk().clone());
        let index = receipt
            .workspace()
            .plan()
            .records()
            .iter()
            .position(|record| record.span() == &span)
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        let state = usage
            .funding
            .get_mut(&native.id)
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        // Even the matching scope cannot replace an already active identity.
        if state.active_span.is_some() {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        // Every slot is checked before publishing the single active marker.
        // The selected native partition covers only its declared domain charges;
        // host metadata and unrelated domains retain their protected balances.
        for (slot, (domain, _)) in pool.topology().domains().enumerate() {
            let balance = &state.domains[slot];
            let required = receipt
                .workspace()
                .span_domain_bytes(index, domain)?
                .ok_or(WorkingMemoryError::UnknownBound)?;
            let required = match receipt.workspace().control_binding() {
                Some(binding) => binding
                    .native_span_domain_remainder(
                        receipt.workspace(),
                        index,
                        domain,
                        balance.native_held,
                        balance.native_registered,
                    )?
                    .unwrap_or(required),
                None => required,
            };
            let host = if domain == pool.topology().host_domain() {
                state.host_held
            } else {
                0
            };
            let protected = host
                .checked_add(balance.native_held.unwrap_or(0))
                .ok_or(WorkingMemoryError::Poisoned)?;
            let available = balance
                .remaining
                .checked_sub(protected)
                .ok_or(WorkingMemoryError::Poisoned)?;
            if required > available {
                return Err(WorkingMemoryError::DomainAllowanceExceeded {
                    domain: domain,
                    required_bytes: required,
                    available_bytes: available,
                });
            }
        }
        state.active_span = Some(ActiveCaptureSpan(Arc::downgrade(&self.identity)));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expired_marker_does_not_reopen_account_and_has_only_one_weak_pointer() {
        let identity = Arc::new(CaptureSourceIdentity { prefill: None });
        let marker = ActiveCaptureSpan(Arc::downgrade(&identity));
        assert_eq!(Arc::strong_count(&identity), 1);
        drop(identity);
        assert!(marker.0.upgrade().is_none());
        let mut state = FundingState::host(
            &crate::working_memory::memory_fixture::host_topology(),
            31,
            Some(
                crate::working_memory::memory_fixture::host_limits(31)
                    .resolve(&crate::working_memory::memory_fixture::host_topology())
                    .unwrap(),
            ),
            &InferenceExecutionIdentity::default(),
            true,
            0,
            0,
        )
        .unwrap();
        state.active_span = Some(marker);
        state.run_open = false;
        assert!(
            state.retains_workspace(),
            "expired identity still excludes terminal trimming"
        );
        assert!(matches!(
            state.validate_span_spend(None),
            Err(WorkingMemoryError::ExecutionFenced)
        ));
        assert_eq!(state.remaining, 31);
        assert_eq!(
            std::mem::size_of::<Option<ActiveCaptureSpan>>(),
            std::mem::size_of::<Weak<()>>()
        );
    }
}
