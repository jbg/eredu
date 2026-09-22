//! Canonical stamped entry over the private source-bound transfer mechanism.
use super::*;
use crate::working_memory::CaptureSourceSegment;

impl<'t, 'f, 'p, 'a> CapturePrefillFragmentClaim<'t, 'f, 'p, 'a> {
    /// Consumes this fragment claim to construct its one prepaid source pin.
    /// The canonical chunk, capture account and original host custody must all
    /// match before the bounded source-control constructor can allocate.
    pub fn prepare_with_original_source<'s, K: Clone + Ord + Send + Sync + 'static>(
        self,
        native: &'s mut WorkingMemoryFundingScope,
        segment: &'s mut CaptureSourceSegment,
        custody: &crate::working_memory::OriginalTextMetadataCustody,
        source: [Option<(K, u64)>; 2],
    ) -> Result<CapturePrefillFragmentTransfer<'t, 'f, 'p, 'a, 's, K>, CaptureRunHostError> {
        self.validate_native_scope(native)?;
        segment.validate_prefill_fragment(self.fragment())?;
        segment.validate_native_scope(native)?;
        let pin = native
            .pool()
            .pin_original_capture_source(native, custody, source)?;
        self.prepare_with_segment_source(native, segment, pin)
    }

    /// Bind this actual fragment to its canonical announced chunk, then borrow
    /// the existing target under the original schedule and exact native scope.
    /// An unstamped foundation segment cannot construct a transfer here.
    ///
    /// This checks placement and source custody only. The caller still supplies
    /// the actual matching native source and retains all native recovery roots.
    /// Host finish proves no chunk settlement, source release or causal row
    /// equivalence, and acquires no second host hold or native allowance.
    pub fn prepare_with_segment_source<'s, K: Clone + Ord + Send + Sync + 'static>(
        self,
        native: &'s mut WorkingMemoryFundingScope,
        segment: &'s mut CaptureSourceSegment,
        complete_source: WorkingMemoryStorage<K>,
    ) -> Result<CapturePrefillFragmentTransfer<'t, 'f, 'p, 'a, 's, K>, CaptureRunHostError> {
        segment.validate_prefill_fragment(self.fragment())?;
        self.prepare_segment_transfer(native, segment, complete_source)
    }

    // Keep the mechanism private: only the stamped entry above may expose it
    // outside working_memory. Internal foundation tests exercise its separate
    // accounting contract without manufacturing a native chunk stamp.
    pub(in crate::working_memory) fn prepare_segment_transfer<
        's,
        K: Clone + Ord + Send + Sync + 'static,
    >(
        self,
        native: &'s mut WorkingMemoryFundingScope,
        segment: &'s mut CaptureSourceSegment,
        complete_source: WorkingMemoryStorage<K>,
    ) -> Result<CapturePrefillFragmentTransfer<'t, 'f, 'p, 'a, 's, K>, CaptureRunHostError> {
        // All prior origins plus this source and the original schedule/scope
        // are revalidated atomically. Key-owned staging/drop occurs outside Usage.
        let rollback = if self.partition {
            self.custody.bind_projected_segment_source(
                native,
                segment,
                &complete_source,
                self.fragment().assembly().logical_geometry().admission(),
            )?
        } else {
            self.custody
                .bind_segment_source(native, segment, &complete_source)?
        };
        #[cfg(test)]
        crate::working_memory::capture_run::tests::before_transfer_allocation();
        // Reuse the only full logical destination. This is not the whole-value
        // allocator: later fragments keep the same initialized shape/data Vecs.
        let writer = self.prepare()?;
        let native = rollback.commit();
        Ok(CapturePrefillFragmentTransfer {
            writer,
            source: complete_source,
            _segment: segment,
            native,
        })
    }
}

/// Opaque short transfer into one existing prefill target. Public construction
/// requires the canonical native-chunk stamp on the actual source segment.
///
/// Its actual target/source payloads retain original custody; exclusive loans
/// keep the exact segment and native scope fixed. There is no raw writer/scope
/// export, retry claim, source release, native allowance or completion proof.
#[must_use = "finish this host fragment or retain its partial target and native recovery"]
pub struct CapturePrefillFragmentTransfer<'t, 'f, 'p, 'a, 's, K: Ord + Send + 'static> {
    writer: CapturePrefillFragmentWriter<'t, 'f, 'p, 'a>,
    source: WorkingMemoryStorage<K>,
    _segment: &'s mut CaptureSourceSegment,
    native: &'s mut WorkingMemoryFundingScope,
}
impl<'t, 'f, 'p, 'a, K: Ord + Send + 'static>
    CapturePrefillFragmentTransfer<'t, 'f, 'p, 'a, '_, K>
{
    /// Exact read-only mapping carried by the originating claim. Native code
    /// must also authenticate actual source geometry, work and canonical chunk.
    pub fn fragment(&self) -> &'f CapturePrefillFragment<'p, 'a> {
        self.writer.fragment
    }
    /// Rechecks the original host parent, exact active native scope, this source
    /// and every earlier segment origin together, then rejects a terminally
    /// failed target. Call before native work and again after exact settlement;
    /// success itself proves no settlement.
    pub fn validate(&self) -> Result<(), CaptureRunHostError> {
        self.writer
            .slot
            .tensor
            .as_ref()
            .expect("prepared target")
            .custody
            .validate_transfer(self.native, &self.source)?;
        if self.writer.slot.state == TargetState::Failed {
            return Err(CapturePrefillHostError::Incomplete {
                index: self.writer.index,
            }
            .into());
        }
        Ok(())
    }
    /// Write one settled scalar using only the checked fragment mapping. Any
    /// rejection is terminal for this target; partial storage stays in the frame.
    pub fn push_f32(&mut self, value: f32) -> Result<(), CaptureRunHostError> {
        if let Err(error) = self.validate() {
            self.writer.slot.state = TargetState::Failed;
            return Err(error);
        }
        self.writer.push_f32(value)
    }
    /// Copies one unsigned scalar into the same fixed, source-bound destination.
    pub fn push_u64(&mut self, value: u64) -> Result<(), CaptureRunHostError> {
        if let Err(error) = self.validate() {
            self.writer.slot.state = TargetState::Failed;
            return Err(error);
        }
        self.writer.push_u64(value)
    }
    /// Complete one host fragment only. Typed failure drops the short loan and
    /// poisons the target, whose data remains in the frame/aborted sidecar. All
    /// source pins remain on the original scope; actual native recovery remains
    /// the caller's obligation. No chunk, scope or source is retired/certified.
    pub fn finish(self) -> Result<(), CaptureRunHostError> {
        self.validate()?;
        self.writer.finish()
    }
}
impl<K: Ord + Send + 'static> fmt::Debug for CapturePrefillFragmentTransfer<'_, '_, '_, '_, '_, K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CapturePrefillFragmentTransfer")
            .field("writer", &self.writer)
            .finish_non_exhaustive()
    }
}
