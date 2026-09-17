//! Original parent admission retained beside a descriptive evidence receipt.
use super::*;
use crate::working_memory::OriginalInterventionSource;
use eredu_core::capture::CaptureLimits;
use std::mem::{size_of, size_of_val};

/// Immutable source aliases only. The enclosing original ledger reserves all
/// work; this private loan cannot issue quota or replace a capture admission.
#[derive(Debug)]
pub(in crate::capture::partition) struct EvidenceBudgetSource {
    parent: SharedCapturePlan,
    original: OriginalInterventionSource,
    operation: usize,
}
impl EvidenceBudgetSource {
    pub(in crate::capture::partition) fn controls() -> Option<usize> {
        let parts = [size_of::<Self>() * 3, size_of::<Option<Self>>(),
            size_of::<Result<Self, &'static str>>(), size_of::<sha2::Sha256>(),
            size_of::<(&SharedCapturePlan, &OriginalInterventionSource, usize, &SharedCapturePlan)>(),
            size_of::<(&mut PartitionCaptureReceiptPlan, Self)>(), size_of::<String>(),
            size_of::<Result<String, CaptureError>>()];
        parts.into_iter().try_fold(size_of_val(&parts), usize::checked_add)
    }
    pub(in crate::capture::partition) fn new(parent: &SharedCapturePlan,
        original: &OriginalInterventionSource, operation: usize, companion: &SharedCapturePlan)
        -> Result<Self, &'static str> {
        let plan = original.plan().admission();
        if plan.request() != parent.admission().request()
            || plan.text_origin() != parent.admission().text_origin()
            || plan.invocation_bounds() != parent.admission().invocation_bounds()
            || original.plan().evidence(operation).is_none_or(|source|
                !source.shared_geometry_source().same_storage(companion)) {
            return Err("evidence budget does not belong to the original parent and companion");
        }
        Ok(Self { parent: parent.clone(), original: original.clone(), operation })
    }
    fn matches(&self, companion: &SharedCapturePlan) -> bool {
        self.original.plan().evidence(self.operation).is_some_and(|source|
            source.shared_geometry_source().same_storage(companion))
    }
    pub(super) fn limits(&self) -> &CaptureLimits { &self.parent.admission().plan().limits }
    pub(super) fn parent_identity(&self) -> &str { self.parent.admission().identity() }
    pub(super) fn intervention_intent(&self) -> &str { self.original.plan().admission().intent_identity() }
    pub(super) fn operation(&self) -> usize { self.operation }
}
impl PartitionCaptureReceiptPlan {
    pub(in crate::capture::partition) fn bind_evidence_budget(&mut self, source: EvidenceBudgetSource)
        -> Result<(), CaptureError> {
        if self.evidence_budget.is_some() || self.shared_plan_source().is_none_or(|plan| !source.matches(plan)) {
            return Err(invalid("receipt evidence budget source differs or is already bound"));
        }
        self.evidence_budget = Some(source);
        // Same fixed digest destination, before allowance/vote/transport. The
        // original loaded authority remains local; peers compare global intent.
        let mut identity = std::mem::take(&mut self.identity);
        identity.clear();
        self.identity = receipt_identity(&self.context, &self.producers, &self.funded,
            &self.routed, Some(identity), self.combination, self.world_size, self.limits,
            self.evidence_budget.as_ref())?;
        Ok(())
    }
}
