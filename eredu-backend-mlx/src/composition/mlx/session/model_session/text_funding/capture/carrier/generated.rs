//! Immediate generated-root retention without lending a carrier borrow to code.
use super::*;
impl FundedWork {
    pub(in crate::composition::mlx::session::model_session::text_funding) fn prepare_generated_capture(
        &self,
        claim: &CapturePrefillFragmentClaim<'_, '_, '_, '_>,
        construct: bool,
    ) -> Result<(), Error> {
        let capture = self.capture.try_borrow().map_err(error)?;
        let capture = capture
            .as_ref()
            .ok_or_else(|| error(CaptureCarrierError::Identity))?;
        if capture.progress.get() != Progress::Ready {
            return Err(error(CaptureCarrierError::Incomplete));
        }
        let scope = self.scope.try_borrow().map_err(error)?;
        let scope = scope
            .as_ref()
            .ok_or_else(|| error(WorkingMemoryError::ExecutionFenced))?;
        claim.validate_native_scope(scope).map_err(error)?;
        let segment = capture
            .segment
            .as_ref()
            .ok_or_else(|| error(CaptureCarrierError::Identity))?;
        segment
            .validate_prefill_fragment(claim.fragment())
            .map_err(error)?;
        segment.validate_native_scope(scope).map_err(error)?;
        let logical = claim.fragment().assembly().logical_geometry();
        let selected = &logical.admission().plan().selections[logical.selection_index()];
        capture.require_capacity(
            selection_descriptors(&selected.transform) + if construct { 9 } else { 0 },
        )?;
        // All scope/carrier loans end before actual source visitation/generation.
        Ok(())
    }
    pub(in crate::composition::mlx::session::model_session::text_funding) fn retain_generated_capture_root(
        &self,
        value: &Array,
        claim: &CapturePrefillFragmentClaim<'_, '_, '_, '_>,
    ) -> Result<(), Error> {
        let retained = value.clone();
        {
            let capture = self.capture.try_borrow().map_err(error)?;
            let capture = capture
                .as_ref()
                .ok_or_else(|| error(CaptureCarrierError::Identity))?;
            let mut roots = capture.roots.try_borrow_mut().map_err(error)?;
            if roots.len() == roots.capacity() {
                return Err(error(WorkingMemoryError::UnknownBound));
            }
            roots.push(retained);
            self.published.set(false);
        }
        // Retain first, then recheck: even a concurrent fence keeps every prior
        // source/intermediate in the original recovery. No producer borrows leak.
        self.prepare_generated_capture(claim, false)
    }
}
