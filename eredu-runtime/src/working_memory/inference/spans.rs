//! Immutable diagnostics from the actual completed equation schedule.
use super::*;
use crate::working_memory::{
    funding::SpanHostOwner, ReservedInferenceSpanWorkspace, WorkingMemoryError,
    WorkingMemoryFundingRun,
};
use std::{
    mem::size_of,
    sync::{Arc, Mutex, OnceLock},
};

/// One actual scheduled span and its complete newly allocated workspace.
/// Opening roots receive no scalar credit: they were never new allocations.
#[derive(Debug, Clone)]
pub struct InferenceSpanWorkspaceRecord {
    span: InferenceWorkspaceSpan,
    new_allocation_bytes: Option<u64>,
    new_tensor_allocation_bytes: Option<u64>,
    host_workspace_bytes: Option<u64>,
}
impl InferenceSpanWorkspaceRecord {
    /// Actual ordinary prefill/decode invocation supplied to the quote visitor.
    pub fn span(&self) -> &InferenceWorkspaceSpan {
        &self.span
    }
    /// New tensor backing, scratch and managed host work, including new closing
    /// state. Unknown if state or either operation domain is incomplete.
    pub fn new_allocation_bytes(&self) -> Option<u64> {
        self.new_allocation_bytes
    }
    /// This span's new tensor buffers and tensor scratch, including new closing
    /// state but excluding opening backing. This independently known domain may
    /// survive a missing host or state bound. It does not certify native births,
    /// retained generations, or complete inference coverage.
    pub fn new_tensor_allocation_bytes(&self) -> Option<u64> {
        self.new_tensor_allocation_bytes
    }
    /// This same span's disjoint operation-owned host workspace. Missing tensor
    /// or state coverage does not erase a known host diagnostic. Source and
    /// preparation owners outside the equation trace remain separate terms.
    pub fn host_workspace_bytes(&self) -> Option<u64> {
        self.host_workspace_bytes
    }
    pub(super) fn new(span: InferenceWorkspaceSpan, trace: &WorkspaceTraceReport) -> Self {
        Self {
            span,
            new_allocation_bytes: trace.inference_transient_bytes().and(trace.total_bytes),
            new_tensor_allocation_bytes: trace.tensor_buffers.total_bytes,
            host_workspace_bytes: trace.host_workspace_bytes,
        }
    }
}
#[derive(Debug)]
struct SpanPlan {
    geometry: InferenceGeometry,
    records: Vec<InferenceSpanWorkspaceRecord>,
    // Construction is serialized only here. Usage-locked readers never acquire
    // this mutex: the immutable attachment is inspected through OnceLock.
    promotion: Mutex<()>,
    // Records and construction controls retire before full P.
    custody: OnceLock<SpanHostOwner>,
    // The closed shared shell and all report/native custody retire before the
    // independent host planning account. No execution permission is carried.
    metadata_funding: Option<eredu_nn::workspace::WorkspaceMetadataFunding>,
}
/// Immutable, identity-preserving diagnostics from one actual quote traversal.
/// This is neither registered opening storage nor an execution/capacity grant.
/// Clones alias the same records; no caller can construct records from bytes.
/// Once attached, original custody survives both the plan allocation's
/// retirement and destruction of its records. No Arc or Weak is exported.
#[derive(Debug)]
pub struct InferenceSpanWorkspacePlan(Option<Arc<SpanPlan>>);
impl Clone for InferenceSpanWorkspacePlan {
    fn clone(&self) -> Self {
        Self(Some(Arc::clone(self.inner())))
    }
}
impl Drop for InferenceSpanWorkspacePlan {
    fn drop(&mut self) {
        if let Some(owner) = self.0.take() {
            // Every strong plan owner reaches this operation; no Arc or Weak
            // escapes. The winner owns the extracted payload after its Arc
            // allocation retires. Records and promotion controls precede the
            // original full custody field when that payload is destroyed.
            drop(Arc::into_inner(owner));
        }
    }
}
impl InferenceSpanWorkspacePlan {
    // Private borrowed access preserves identity without exporting an Arc.
    fn inner(&self) -> &Arc<SpanPlan> {
        self.0.as_ref().expect("live span plan owner")
    }

    pub(in crate::working_memory) fn metadata_funding(
        &self,
    ) -> Option<eredu_nn::workspace::WorkspaceMetadataFunding> {
        self.inner().metadata_funding.clone()
    }

    // The extracted plan remains live while its full/raw custody retires.
    // Their nested extraction terms are added separately by original P.
    pub(in crate::working_memory) fn retirement_control_bytes() -> Option<usize> {
        size_of::<Option<SpanPlan>>().checked_add(size_of::<Arc<SpanPlan>>())
    }

    pub(super) fn new(
        geometry: InferenceGeometry,
        records: Vec<InferenceSpanWorkspaceRecord>,
    ) -> Self {
        Self::new_with_funding(geometry, records, None)
    }
    fn new_with_funding(
        geometry: InferenceGeometry,
        records: Vec<InferenceSpanWorkspaceRecord>,
        metadata_funding: Option<eredu_nn::workspace::WorkspaceMetadataFunding>,
    ) -> Self {
        Self(Some(Arc::new(SpanPlan {
            geometry,
            records,
            promotion: Mutex::new(()),
            custody: OnceLock::new(),
            metadata_funding,
        })))
    }
    pub(super) fn new_metadata(
        geometry: InferenceGeometry,
        records: Vec<InferenceSpanWorkspaceRecord>,
        context: Option<&eredu_nn::workspace::WorkspaceContext>,
    ) -> Result<Self, eredu_nn::Error> {
        if let Some(context) = context {
            let allocation = std::alloc::Layout::new::<[std::sync::atomic::AtomicUsize; 2]>()
                .extend(std::alloc::Layout::new::<SpanPlan>())
                .ok()
                .map(|value| value.0.pad_to_align().size())
                .ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?;
            let controls = [
                allocation,
                size_of::<SpanPlan>(),
                size_of::<Self>(),
                size_of::<Result<Self, eredu_nn::Error>>(),
                size_of::<Option<SpanPlan>>(),
            ];
            let bytes = controls
                .into_iter()
                .try_fold(std::mem::size_of_val(&controls), usize::checked_add)
                .ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?;
            context.charge_metadata(bytes)?;
        }
        Ok(Self::new_with_funding(
            geometry,
            records,
            context.and_then(eredu_nn::workspace::WorkspaceContext::metadata_funding),
        ))
    }

    // Only the consuming quote conversion supplies an authenticated receipt.
    // Never accepts caller bytes, a custody provider, or a native work callback.
    pub(in crate::working_memory) fn attach_original_host(
        &self,
        run: &WorkingMemoryFundingRun,
        receipt: &ReservedInferenceSpanWorkspace<'_>,
    ) -> Result<(), WorkingMemoryError> {
        self.attach_host(run, receipt, false)
    }
    pub(in crate::working_memory) fn attach_pending_capture_host(
        &self,
        run: &WorkingMemoryFundingRun,
        receipt: &ReservedInferenceSpanWorkspace<'_>,
    ) -> Result<(), WorkingMemoryError> {
        self.attach_host(run, receipt, true)
    }
    fn attach_host(
        &self,
        run: &WorkingMemoryFundingRun,
        receipt: &ReservedInferenceSpanWorkspace<'_>,
        pending: bool,
    ) -> Result<(), WorkingMemoryError> {
        let expects_publication = receipt
            .workspace()
            .control_binding()
            .is_some_and(|b| b.publication_layout().is_some());
        if pending != expects_publication || !self.same_plan(receipt.workspace().plan()) {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        let _promotion = self
            .inner()
            .promotion
            .lock()
            .map_err(|_| WorkingMemoryError::Poisoned)?;
        if self.inner().custody.get().is_some() {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        let custody = run.hold_inference_span_workspace(receipt)?;
        // This sole insertion cannot race under the construction mutex. In an
        // unexpected duplicate case the returned custody drops outside Usage.
        self.inner()
            .custody
            .set(custody)
            .map_err(|_| WorkingMemoryError::IdentityMismatch)?;
        #[cfg(test)]
        crate::working_memory::funding::after_span_attachment();
        // Failure preserves the attached hold through the quote/error and every
        // earlier plan alias. It never refunds a successfully attached owner.
        let custody = self
            .inner()
            .custody
            .get()
            .expect("just attached span custody");
        if pending {
            custody.validate_pending(receipt)
        } else {
            custody.validate(receipt)
        }
    }
    pub(in crate::working_memory) fn original_host(
        &self,
    ) -> Result<&SpanHostOwner, WorkingMemoryError> {
        self.inner()
            .custody
            .get()
            .ok_or(WorkingMemoryError::IdentityMismatch)
    }
    #[cfg(test)]
    pub(in crate::working_memory) fn strong_owner_count(&self) -> usize {
        Arc::strong_count(self.inner())
    }
    /// Original complete request geometry, including physical output demand.
    pub fn geometry(&self) -> InferenceGeometry {
        self.inner().geometry
    }
    /// Ordered actual equation spans, including conservative final decode spans.
    pub fn records(&self) -> &[InferenceSpanWorkspaceRecord] {
        &self.inner().records
    }
    /// Invocations for generation that samples its first token from prefill.
    /// The diagnostic records also include one conservative final decode; it
    /// stays in the workspace bound but has no generation submission. Scoring
    /// and verification consumers continue to use the complete record list.
    /// State-only prefill cannot supply that first token's scores.
    pub fn generation_forward_count(&self) -> Option<usize> {
        self.generation_records().map(Iterator::count)
    }
    /// The exact generation subset of this same retained traversal. The final
    /// conservative decode remains in `records()` for scoring and verification;
    /// it is not submitted after generation's last sampled token.
    pub fn generation_records(
        &self,
    ) -> Option<impl Iterator<Item = &InferenceSpanWorkspaceRecord> + '_> {
        let geometry = self.geometry();
        if geometry.max_output_tokens != 0 && geometry.output == OutputDemand::StateOnly {
            return None;
        }
        let decodes = geometry.max_output_tokens.saturating_sub(1);
        Some(self.records().iter().filter(move |record| match record.span() {
            InferenceWorkspaceSpan::Prefill(_) => true,
            InferenceWorkspaceSpan::Decode { index, .. } => *index < decodes,
        }))
    }
    /// Whether these diagnostics alias the very same original traversal.
    pub fn same_plan(&self, other: &Self) -> bool {
        Arc::ptr_eq(self.inner(), other.inner())
    }
    /// Actual owned record capacity and fixed Arc payload/control. This is host
    /// storage, not native workspace; construction moves are priced by sealing.
    pub fn capacity_bytes(&self) -> Option<u64> {
        self.inner()
            .records
            .capacity()
            .checked_mul(size_of::<InferenceSpanWorkspaceRecord>())?
            .checked_add(size_of::<SpanPlan>())?
            .checked_add(2 * size_of::<usize>())?
            .try_into()
            .ok()
    }
    /// Storage still requiring admission when attaching this exact shared plan.
    /// A retained planning account already covers the original Vec and Arc;
    /// attaching custody does not allocate either destination again.
    pub(in crate::working_memory) fn unreserved_capacity_bytes(&self) -> Option<u64> {
        if self.inner().metadata_funding.is_some() {
            Some(0)
        } else {
            self.capacity_bytes()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn capacity_counts_actual_spare_record_payload_and_fixed_owner() {
        let records = Vec::with_capacity(19);
        let capacity = records.capacity();
        let geometry = InferenceGeometry {
            batch_size: 1,
            cached_positions: 0,
            input_positions: 1,
            max_output_tokens: 1,
            prefill_chunk_positions: 1,
            output: OutputDemand::LastPosition,
        };
        let plan = InferenceSpanWorkspacePlan::new(geometry, records);
        assert!(plan.records().is_empty());
        assert_eq!(
            plan.capacity_bytes(),
            Some(
                (capacity * size_of::<InferenceSpanWorkspaceRecord>()
                    + size_of::<SpanPlan>()
                    + 2 * size_of::<usize>()) as u64
            )
        );
    }
}
