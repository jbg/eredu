//! Consume the exact accepted quote into unready source custody, then join S+C.
use super::*;
use crate::working_memory::{
    CapturePlanPublicationCause,
    storage::capture_publication::{CaptureSourceOwner, PreparedCaptureStorage},
};

/// One move-only originally accepted publication attempt. No guard, span view,
/// mutable native authority or unguarded storage handle can be extracted here.
/// Dropping it cannot retry the original plan attachment or refund escaped aliases.
#[derive(Debug)]
#[must_use]
pub struct PendingCapturePlanPublication<K: CapturePlanStorageKey> {
    quote: Option<IncrementalInferenceQuote>,
    source: SharedCapturePlan,
    storage: PreparedCaptureStorage<K>,
    reservation: WorkingMemoryReservation,
    // Last: partial buffers, source aliases and quote fields retire first.
    custody: SpanHostOwner,
}
/// Failed publication keeps all partially constructed owners under the original
/// hold. It exposes the typed cause only, never a retry or a normal control guard.
#[derive(Debug)]
#[must_use]
pub struct FailedCapturePlanPublication<K: CapturePlanStorageKey> {
    cause: CapturePlanPublicationCause,
    _pending: PendingCapturePlanPublication<K>,
}
impl<K: CapturePlanStorageKey> FailedCapturePlanPublication<K> {
    /// Original typed failure; pending owners cannot be extracted or retried.
    pub fn cause(&self) -> &CapturePlanPublicationCause {
        &self.cause
    }
}
impl<K: CapturePlanStorageKey> std::fmt::Display for FailedCapturePlanPublication<K> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("original capture-plan publication failed")
    }
}
impl<K: CapturePlanStorageKey + std::fmt::Debug> std::error::Error
    for FailedCapturePlanPublication<K>
{
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.cause)
    }
}
impl IncrementalInferenceQuote {
    /// Hold the sealed P+Q+slot contribution once, attach it to the actual plan
    /// (including earlier aliases), then construct the one bounded publication.
    /// The source must be the exact physically prepared C owner. C capacity is
    /// already in the original accepted reservation and remains outside the hold.
    pub fn begin_capture_plan_publication<K: CapturePlanStorageKey>(
        self,
        run: &WorkingMemoryFundingRun,
        reservation: &WorkingMemoryReservation,
        source: &SharedCapturePlan,
    ) -> Result<PendingCapturePlanPublication<K>, SpanWorkspaceOwnerError> {
        let prepare = || -> Result<SpanHostOwner, WorkingMemoryError> {
            let receipt = self.reserved_span_workspace(reservation)?;
            let layout = self
                .span_workspace
                .control_binding()
                .and_then(|b| b.publication_layout())
                .ok_or(WorkingMemoryError::IdentityMismatch)?;
            layout.key::<K>()?;
            if &layout.source != source.storage_identity()
                || source.capacity_bytes() != Some(layout.capacity)
            {
                return Err(WorkingMemoryError::IdentityMismatch);
            }
            self.span_workspace
                .plan
                .attach_pending_capture_host(run, &receipt)?;
            Ok(self.span_workspace.plan.original_host()?.clone())
        };
        let custody = match prepare() {
            Ok(custody) => custody,
            Err(cause) => return Err(SpanWorkspaceOwnerError { quote: self, cause }),
        };
        let layout = self
            .span_workspace
            .control_binding()
            .and_then(|b| b.publication_layout())
            .expect("validated publication layout");
        // All registration/key/Arc allocations occur AFTER the same plan's
        // original hold. Provider clone panic leaves that attachment on aliases.
        let storage = match PreparedCaptureStorage::<K>::new(layout, custody.raw().clone()) {
            Ok(storage) => storage,
            Err(cause) => return Err(SpanWorkspaceOwnerError { quote: self, cause }),
        };
        Ok(PendingCapturePlanPublication {
            quote: Some(self),
            source: source.clone(),
            storage,
            reservation: reservation.clone(),
            custody,
        })
    }
}
impl<K: CapturePlanStorageKey> PendingCapturePlanPublication<K> {
    /// Publish only the sealed C key through this exact original native scope.
    /// An existing raced source is retained without reducing original C. A
    /// successful S+C health join is required before any ordinary guard escapes.
    pub fn publish_and_finish(
        mut self,
        native: &WorkingMemoryFundingScope,
    ) -> Result<
        (OwnedTextSpanWorkspace, RegisteredInferenceSourceWitness),
        FailedCapturePlanPublication<K>,
    > {
        let finish =
            (|| -> Result<RegisteredInferenceSourceWitness, CapturePlanPublicationCause> {
                self.storage.publish(&self.source, native, &self.custody)?;
                #[cfg(test)]
                after_capture_publication(self.source.storage_identity());
                let quote = self.quote.as_ref().expect("single pending quote");
                let receipt = quote.reserved_span_workspace(&self.reservation)?;
                let witness = RegisteredInferenceSourceWitness::with_capture(
                    receipt.source_witness(),
                    quote.pool.clone(),
                    CaptureSourceOwner::new(self.storage.published.clone()),
                );
                {
                    let usage = quote.pool.0.usage.try_lock().map_err(|error| match error {
                        std::sync::TryLockError::WouldBlock => CapturePlanPublicationCause::Busy,
                        std::sync::TryLockError::Poisoned(_) => WorkingMemoryError::Poisoned.into(),
                    })?;
                    self.custody
                        .raw()
                        .validate_publication_locked(native, &usage)?;
                    self.custody
                        .validate_prepublication_locked(&quote.pool, &usage)?;
                    witness.validate_locked(&quote.pool, &usage)?;
                    // OnceLock contains only accounting Arc clones, not a buffer
                    // allocation or provider callback. The sole pending owner sets it.
                    self.custody.complete_publication(witness.clone())?;
                }
                Ok(witness)
            })();
        let witness = match finish {
            Ok(witness) => witness,
            Err(cause) => {
                return Err(FailedCapturePlanPublication {
                    cause,
                    _pending: self,
                });
            }
        };
        let (span, prior) = self
            .quote
            .take()
            .expect("completed pending quote")
            .finish_span_workspace(&self.reservation);
        drop(prior);
        Ok((OwnedTextSpanWorkspace::from_promoted(span), witness))
    }
}

#[cfg(test)]
thread_local! {
    static AFTER_CAPTURE_PUBLICATION: std::cell::RefCell<Option<(eredu_core::SharedStorageIdentity, WorkingMemoryFundingScope)>> = const { std::cell::RefCell::new(None) };
}
#[cfg(test)]
#[derive(Debug)]
pub(in crate::working_memory) struct PostPublicationFault(eredu_core::SharedStorageIdentity);
#[cfg(test)]
fn take_post_publication_scope(
    source: &eredu_core::SharedStorageIdentity,
) -> Option<WorkingMemoryFundingScope> {
    AFTER_CAPTURE_PUBLICATION.with(|slot| {
        let mut slot = slot.borrow_mut();
        if slot
            .as_ref()
            .is_some_and(|(identity, _)| identity == source)
        {
            slot.take().map(|(_, scope)| scope)
        } else {
            None
        }
    })
}
#[cfg(test)]
impl Drop for PostPublicationFault {
    fn drop(&mut self) {
        drop(take_post_publication_scope(&self.0));
    }
}
#[cfg(test)]
fn after_capture_publication(source: &eredu_core::SharedStorageIdentity) {
    // Actual source/Usage guards and the TLS loan are gone before scope Drop.
    drop(take_post_publication_scope(source));
}
#[cfg(test)]
impl<K: CapturePlanStorageKey> PendingCapturePlanPublication<K> {
    pub(in crate::working_memory) fn quarantine_source_after_publication(
        &self,
        scope: WorkingMemoryFundingScope,
    ) -> PostPublicationFault {
        let source = self.source.storage_identity().clone();
        AFTER_CAPTURE_PUBLICATION.with(|slot| {
            let mut slot = slot.borrow_mut();
            assert!(slot.is_none(), "one scoped publication fault");
            *slot = Some((source.clone(), scope));
        });
        PostPublicationFault(source)
    }
}
