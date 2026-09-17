//! Sparse partition destinations reuse the ordinary fixed row/scalar producer.
use super::*;
use crate::capture::partition::PartitionCaptureReceiptPlan;
use std::mem::size_of_val;

/// Exact global sparse fragment and receive-order range capacity from its receipt.
/// This borrows original ownership; it grants no native work or destination.
#[derive(Debug)]
pub struct CapturePartitionRoutedHostPlan<'a> {
    host: CaptureRoutedHostPlan<'a>,
    receipt: &'a PartitionCaptureReceiptPlan,
    ownership: &'a RoutedUnitCaptureOwnership,
    producer: usize,
    fragment: usize,
    peak: u64,
}
impl<'a> CapturePartitionRoutedHostPlan<'a> {
    /// Inspect the same original receipt without allocating ownership or row data.
    pub fn prepare(receipt: &'a PartitionCaptureReceiptPlan, producer: usize, fragment: usize)
        -> Result<Self, CaptureRunHostError> {
        let source = receipt.shared_plan_source().ok_or(CaptureRunHostError::ReceiptMismatch)?;
        let context = receipt.context();
        if receipt.combination() != PartitionCaptureCombination::Disjoint
            || context.capture_plan_identity != source.admission().identity() {
            return Err(CaptureRunHostError::ReceiptMismatch);
        }
        let ownership = receipt.routed_producer(producer).ok_or(CaptureRunHostError::ReceiptMismatch)?;
        let projection = receipt.producer(producer).ok_or(CaptureRunHostError::ReceiptMismatch)?;
        let geometry = CaptureRoutedUnitsGeometry::prepare_partition(source.admission(),
            context.selection_index, context.phase, context.prediction, context.invocation, projection, fragment)
            .map_err(CaptureStepError::from)?;
        ownership.validate_geometry(geometry.bank()).map_err(CaptureRoutedHostError::from)?;
        let native_rows = ownership.maximum_source_rows_checked(geometry.source_shape()[0] as u64,
            geometry.bank().routes_per_token).map_err(CaptureRoutedHostError::from)?;
        let host = CaptureRoutedHostPlan::prepare_source_rows(geometry,
            usize::try_from(native_rows).map_err(|_| WorkingMemoryError::Overflow)?)?;
        let peak = host.initialization_peak_bytes().checked_add(
            u64::try_from(Self::control_bytes().ok_or(WorkingMemoryError::Overflow)?)
                .map_err(|_| WorkingMemoryError::Overflow)?)
            .ok_or(WorkingMemoryError::Overflow)?;
        Ok(Self { host, receipt, ownership, producer, fragment, peak })
    }
    /// Original global source and selected token/slot/unit coordinates.
    pub fn geometry(&self) -> &CaptureRoutedUnitsGeometry<'a> { self.host.geometry() }
    /// Exact retained expert/unit/source-peer placement.
    pub fn ownership(&self) -> &'a RoutedUnitCaptureOwnership { self.ownership }
    /// Paid sparse buffers, native chunk evidence and fixed construction controls.
    pub fn initialization_peak_bytes(&self) -> u64 { self.peak }
    /// Fixed source, result, short writer and retained-failure transport frames.
    pub fn control_bytes() -> Option<usize> {
        let parts = [CaptureRoutedUnitsGeometry::partition_control_bytes()?,
            size_of::<Self>() * 2, size_of::<Result<Self, CaptureRunHostError>>(),
            size_of::<CapturePartitionRoutedClaim<'_, '_>>() * 2,
            size_of::<PreparedPartitionRoutedCapture>() * 2,
            size_of::<Result<PreparedPartitionRoutedCapture, CaptureRunHostError>>(),
            size_of::<CapturePartitionRoutedWriter<'_, '_>>() * 2,
            size_of::<Result<CapturePartitionRoutedWriter<'_, '_>, CaptureRoutedHostError>>(),
            size_of::<CapturePartitionRoutedTransfer<'_, '_, '_, u8>>() * 2,
            size_of::<PartitionRoutedCaptureFailure>(), size_of::<ClaimedPartitionRoutedUnits>(),
            size_of::<Result<ClaimedPartitionRoutedUnits, PartitionRoutedCaptureFailure>>(),
            size_of::<PartitionRoutedUnitCaptureRequest<'_>>(),size_of::<PartitionRoutedUnitCaptureLayout<'_>>(), size_of::<SharedCapturePlan>(),
            size_of::<[u8; 64]>() * 2, size_of::<(usize, usize, u64, u64)>(),
            size_of::<CaptureRoutedTokenWindow>()*2,size_of::<Option<CaptureRoutedTokenWindow>>(),
            size_of::<NativeExtent>()*2,size_of::<[u64;2]>(),size_of::<Result<[u64;2],CaptureRoutedHostError>>(),
            size_of::<(&mut PreparedPartitionRoutedCapture,u64)>(),size_of::<Option<u64>>()*2,
            size_of::<Result<(), CaptureRoutedHostError>>(), size_of::<Result<(), RoutedUnitValidationError>>(),
            size_of::<(&PartitionCaptureReceiptPlan, usize, usize)>(),
            size_of::<(&RoutedUnitCapture, &ResolvedCaptureSlice, &RoutedUnitCaptureOwnership, u64, &mut [RoutedUnitRowIdentity])>(),
            size_of::<std::slice::Iter<'_, RoutedUnitCaptureRow>>(), size_of::<std::slice::Iter<'_, [u64; 2]>>()];
        parts.into_iter().try_fold(size_of_val(&parts), usize::checked_add)
    }
}
/// One spent original fragment Host destination, independent of its native loan.
#[derive(Debug)]
pub struct CapturePartitionRoutedClaim<'a, 'c> {
    plan: CapturePartitionRoutedHostPlan<'a>,
    identity: claims::ReceiptIdentity,
    exclusive: PhantomData<&'c mut ()>,
}
impl<'a, 'c> CapturePartitionRoutedClaim<'a, 'c> {
    pub(in crate::working_memory) fn from_plan(plan: CapturePartitionRoutedHostPlan<'a>,
        identity: claims::ReceiptIdentity) -> Self { Self { plan, identity, exclusive: PhantomData } }
    /// Exact source-bound planning contract.
    pub fn plan(&self) -> &CapturePartitionRoutedHostPlan<'a> { &self.plan }
    /// Allocate once within the existing original H. Actual native rows may be
    /// zero for an exchanged idle producer, but never exceed retained placement.
    pub fn prepare(self, actual_native_rows: u64) -> Result<PreparedPartitionRoutedCapture, CaptureRunHostError> {
        self.prepare_extent(actual_native_rows,false)
    }
    pub(in crate::working_memory) fn prepare_accumulating(self) -> Result<PreparedPartitionRoutedCapture, CaptureRunHostError> {
        self.prepare_extent(0,true)
    }
    pub(in crate::working_memory) fn prepare_received(self)->Result<PreparedPartitionRoutedCapture,CaptureRunHostError> {
        let mut target=self.prepare_accumulating()?;target.extent=NativeExtent::Received;Ok(target)
    }
    fn prepare_extent(self, actual_native_rows:u64, accumulating:bool) -> Result<PreparedPartitionRoutedCapture,CaptureRunHostError> {
        if actual_native_rows > self.plan.host.source_rows as u64
            || (!accumulating && self.plan.ownership.source_peer.is_none()
                && actual_native_rows != self.plan.geometry().source_shape()[0] as u64) {
            return Err(CaptureRoutedHostError::Geometry(RoutedUnitValidationError::Chunks).into());
        }
        let mut receipt = [0; 64];
        if self.plan.receipt.identity().len() != receipt.len() { return Err(CaptureRunHostError::ReceiptMismatch); }
        receipt.copy_from_slice(self.plan.receipt.identity().as_bytes());
        let source = self.plan.receipt.shared_plan_source().ok_or(CaptureRunHostError::ReceiptMismatch)?.clone();
        let source_tokens = self.plan.geometry().source_shape()[0] as u64;
        let owner = OwnedCaptureRoutedUnits::allocate(&self.plan.host, self.identity)?;
        Ok(PreparedPartitionRoutedCapture { owner, source, receipt,
            producer: self.plan.producer, fragment: self.plan.fragment, source_tokens,
            native_rows: actual_native_rows, token_window:None, extent:if accumulating { NativeExtent::Accumulating(None) } else { NativeExtent::Fixed } })
    }
}
#[derive(Debug)]
enum NativeExtent { Fixed, Accumulating(Option<[u64;2]>), Received }

/// Partial sparse data outlives short provider loans under its original custody.
/// No receipt or callback can replace its source, rank, fragment or native extent.
#[derive(Debug)]
pub struct PreparedPartitionRoutedCapture {
    owner: OwnedCaptureRoutedUnits,
    source: SharedCapturePlan,
    receipt: [u8; 64],
    producer: usize,
    fragment: usize,
    source_tokens: u64,
    native_rows: u64,
    extent: NativeExtent,
    token_window:Option<CaptureRoutedTokenWindow>,
}
impl PreparedPartitionRoutedCapture {
    fn active_native_range(&self)->Result<[u64;2],CaptureRoutedHostError> {
        match self.extent {
            NativeExtent::Fixed=>Ok([0,self.native_rows]),
            NativeExtent::Accumulating(Some(range))=>Ok(range),
            NativeExtent::Accumulating(None)|NativeExtent::Received=>Err(RoutedUnitValidationError::Chunks.into()),
        }
    }
    pub(in crate::working_memory) fn begin_invocation(&mut self,actual_rows:u64,token_window:CaptureRoutedTokenWindow)->Result<(),CaptureRoutedHostError> {
        let result=(||{
            self.owner.check()?;
            if !matches!(self.extent,NativeExtent::Accumulating(None)) || self.owner.data.active.is_some()
                || self.owner.data.value.source_token_ranges.last().map_or(0,|r|r[1])!=self.native_rows {
                return Err(RoutedUnitValidationError::Chunks.into());
            }
            let end=self.native_rows.checked_add(actual_rows).filter(|n|*n<=self.owner.source_tokens)
                .ok_or(RoutedUnitValidationError::Chunks)?;
            self.extent=NativeExtent::Accumulating(Some([self.native_rows,end]));self.native_rows=end;self.token_window=Some(token_window);
            Ok(())
        })();self.owner.remember(result)
    }
    pub(in crate::working_memory) fn finish_invocation(&mut self)->Result<(),CaptureRoutedHostError> {
        let result=(||{
            self.owner.check()?;
            if !matches!(self.extent,NativeExtent::Accumulating(Some(_))) || self.owner.data.active.is_some()
                || self.owner.data.value.source_token_ranges.last().map_or(0,|r|r[1])!=self.native_rows {
                return Err(RoutedUnitValidationError::Chunks.into());
            }
            self.extent=NativeExtent::Accumulating(None);self.token_window=None;Ok(())
        })();self.owner.remember(result)
    }
    fn received(&self)->Result<(),CaptureRoutedHostError> {
        self.owner.check()?;
        if matches!(self.extent,NativeExtent::Received){Ok(())}else{Err(RoutedUnitValidationError::Ownership.into())}
    }
    pub(in crate::working_memory) fn received_begin_row(&mut self,receipt:&PartitionCaptureReceiptPlan,
        peer:Option<u64>,token:u64,slot:u64,expert:u64,coefficient:f32)->Result<(),CaptureRoutedHostError> {
        self.received()?;let ownership=self.ownership(receipt)?;
        if peer!=ownership.source_peer||usize::try_from(expert).ok()
            .and_then(|e|ownership.coordinates.experts().global_to_local(e)).is_none(){
            return self.owner.remember(Err(RoutedUnitValidationError::Ownership.into()));
        }
        self.owner.begin_row(peer,token,slot,expert,coefficient)
    }
    pub(in crate::working_memory) fn received_push(&mut self,value:f32)->Result<(),CaptureRoutedHostError> {
        self.received()?;self.owner.push_f32(value)
    }
    pub(in crate::working_memory) fn received_finish_row(&mut self)->Result<(),CaptureRoutedHostError> {
        self.received()?;self.owner.finish_row()
    }
    pub(in crate::working_memory) fn received_chunk(&mut self,start:u64,end:u64)->Result<(),CaptureRoutedHostError> {
        self.received()?;let previous=self.owner.data.value.source_token_ranges.last().map_or(0,|r|r[1]);
        if start!=previous{return self.owner.remember(Err(RoutedUnitValidationError::Chunks.into()));}
        self.owner.source_chunk(start,end)
    }
    fn ownership<'a>(&self, receipt: &'a PartitionCaptureReceiptPlan)
        -> Result<&'a RoutedUnitCaptureOwnership, CaptureRoutedHostError> {
        self.owner.check()?;
        if receipt.identity().as_bytes() != self.receipt
            || !receipt.shared_plan_source().is_some_and(|source| self.source.same_storage(source))
            || receipt.producer(self.producer).and_then(|p| p.fragments().get(self.fragment)).is_none() {
            return Err(RoutedUnitValidationError::Ownership.into());
        }
        receipt.routed_producer(self.producer).ok_or_else(|| RoutedUnitValidationError::Ownership.into())
    }
    /// Borrow the original request and fixed buffers for one provider callback.
    /// Dropping an unfinished writer poisons this target; no destination is refunded.
    pub fn writer<'t, 'r>(&'t mut self, receipt: &'r PartitionCaptureReceiptPlan)
        -> Result<CapturePartitionRoutedWriter<'t, 'r>, CaptureRoutedHostError> {
        let ownership = self.ownership(receipt)?;
        self.active_native_range()?;
        Ok(CapturePartitionRoutedWriter { target: self, ownership, finished: false })
    }
    /// Seal only after all actual native rows completed. Zero-row EP producers
    /// retain a real empty payload/acknowledgment and their original ownership.
    pub fn finish(mut self, receipt: &PartitionCaptureReceiptPlan)
        -> Result<ClaimedPartitionRoutedUnits, PartitionRoutedCaptureFailure> {
        let result = (|| {
            let ownership = self.ownership(receipt)?;
            if self.owner.data.active.is_some() || matches!(self.extent,NativeExtent::Accumulating(Some(_))) { return Err(RoutedUnitValidationError::Incomplete.into()); }
            self.owner.data.value.validate_partition_with_scratch(&self.owner.data.slice,
                ownership, self.source_tokens, &mut self.owner.data.scratch)?;
            let end = self.owner.data.value.source_token_ranges.last().map_or(0, |range| range[1]);
            if matches!(self.extent,NativeExtent::Received){self.native_rows=end;}
            else if end != self.native_rows { return Err(RoutedUnitValidationError::Chunks.into()); }
            Ok(())
        })();
        if let Err(cause) = result { return Err(PartitionRoutedCaptureFailure { cause, target: self }); }
        self.owner.data.value.rows.sort_unstable_by_key(|r| (r.source_peer, r.token, r.slot));
        Ok(ClaimedPartitionRoutedUnits {
            value: ClaimedCaptureRoutedUnits { value: self.owner.data.value,
                scratch: self.owner.data.scratch, slice: self.owner.data.slice, identity: self.owner.identity },
            source: self.source, native_rows: self.native_rows,
        })
    }
}
/// Exact sparse Host result. The fragment bank must still authenticate its own
/// custody identity before transport or global assembly can consume it.
#[derive(Debug)]
pub struct ClaimedPartitionRoutedUnits {
    value: ClaimedCaptureRoutedUnits,
    source: SharedCapturePlan,
    native_rows: u64,
}
impl ClaimedPartitionRoutedUnits {
    /// Original-coordinate sparse rows and receive-order chunk evidence.
    pub fn observation(&self) -> &RoutedUnitCapture { self.value.observation() }
    /// Actual completed native invocation extent, including an idle zero.
    pub const fn native_rows(&self) -> u64 { self.native_rows }
    pub(in crate::working_memory) fn identity(&self) -> &claims::ReceiptIdentity { &self.value.identity }
}
/// First source/row/completion failure retains all initialized sparse buffers
/// and original source/custody; it cannot return another writer or native loan.
#[derive(Debug, thiserror::Error)]
#[error("partition routed capture: {cause}")]
pub struct PartitionRoutedCaptureFailure {
    #[source] cause: CaptureRoutedHostError,
    target: PreparedPartitionRoutedCapture,
}
/// Short exclusive writer over the shared fixed sparse destination.
#[derive(Debug)]
pub struct CapturePartitionRoutedWriter<'t, 'r> {
    target: &'t mut PreparedPartitionRoutedCapture,
    ownership: &'r RoutedUnitCaptureOwnership,
    finished: bool,
}
impl<'t, 'r> CapturePartitionRoutedWriter<'t, 'r> {
    /// Borrow original global selection and exact producer/source placement.
    pub fn request(&self) -> PartitionRoutedUnitCaptureRequest<'_> {
        PartitionRoutedUnitCaptureRequest { geometry: self.target.owner.data.value.geometry,
            source_tokens: self.target.source_tokens, ownership: self.ownership, slice: &self.target.owner.data.slice }
    }
    /// Borrow actual invocation layout from the retained receipt independently
    /// of this target's slice loan, so native source binding can precede transfer.
    pub fn source_layout(&self)->PartitionRoutedUnitCaptureLayout<'r> {
        PartitionRoutedUnitCaptureLayout{geometry:self.target.owner.data.value.geometry,
            source_tokens:self.invocation_source_tokens(),ownership:self.ownership}
    }
    /// Actual active invocation rows, separate from original logical tokens.
    pub fn native_rows(&self) -> u64 { let range=self.target.active_native_range().expect("writer has an active native extent");range[1]-range[0] }
    /// Pre-exchange provider tokens for this invocation, preserving all batches.
    pub fn invocation_source_tokens(&self)->u64 {self.target.token_window.map_or(self.target.source_tokens,|window|window.source_tokens())}
    /// Translate a physical/origin token to the retained whole-prompt coordinates.
    /// Native received-row offsets and source-peer tags remain independent.
    pub fn logical_token(&self,physical:u64)->Option<u64> {
        self.target.token_window.map_or_else(||(physical<self.target.source_tokens).then_some(physical),|window|window.logical_token(physical))
    }
    /// Authenticate the same original account before native source settlement.
    pub fn validate_native_scope(&self, native: &WorkingMemoryFundingScope) -> Result<(), WorkingMemoryError> {
        self.target.owner.identity.custody.validate_scheduled_native(native)
    }
    /// Begin one route with its real source peer and original token/slot tags.
    pub fn begin_row(&mut self, peer: Option<u64>, token: u64, slot: u64, expert: u64,
        coefficient: f32) -> Result<(), CaptureRoutedHostError> {
        if peer != self.ownership.source_peer || usize::try_from(expert).ok()
            .and_then(|e| self.ownership.coordinates.experts().global_to_local(e)).is_none() {
            return self.target.owner.remember(Err(RoutedUnitValidationError::Ownership.into()));
        }
        self.target.owner.begin_row(peer, token, slot, expert, coefficient)
    }
    /// Append one scalar to the existing selected-unit row.
    pub fn push_f32(&mut self, value: f32) -> Result<(), CaptureRoutedHostError> { self.target.owner.push_f32(value) }
    /// Commit one completely filled row without reallocating storage.
    pub fn finish_row(&mut self) -> Result<(), CaptureRoutedHostError> { self.target.owner.finish_row() }
    /// Record invocation-relative native coverage, retaining checked cumulative
    /// receive-order coordinates across prefill invocations, including chunks
    /// with no selected exported routes. Empty native invocations add no range.
    pub fn source_chunk(&mut self, start: u64, end: u64) -> Result<(), CaptureRoutedHostError> {
        let range=self.target.active_native_range()?;
        let previous = self.target.owner.data.value.source_token_ranges.last().map_or(0, |r| r[1]);
        let start=range[0].checked_add(start);
        let end=range[0].checked_add(end);
        let (Some(start),Some(end))=(start,end) else {
            return self.target.owner.remember(Err(RoutedUnitValidationError::Chunks.into()));
        };
        if start != previous || end > range[1] {
            return self.target.owner.remember(Err(RoutedUnitValidationError::Chunks.into()));
        }
        self.target.owner.source_chunk(start, end)
    }
    /// Complete this callback; whole-invocation coverage is sealed separately.
    pub fn finish(mut self) -> Result<(), CaptureRoutedHostError> {
        self.target.owner.check()?;
        if self.target.owner.data.active.is_some() {
            return self.target.owner.remember(Err(RoutedUnitValidationError::Incomplete.into()));
        }
        self.finished = true; Ok(())
    }
    /// Retain the exact source backing while the original native scope is lent
    /// to this same row writer. Failure leaves its partial target poisoned.
    pub fn prepare_with_source<'s, K: Clone + Ord + Send + Sync + 'static>(self,
        native: &'s mut WorkingMemoryFundingScope, source: WorkingMemoryStorage<K>)
        -> Result<CapturePartitionRoutedTransfer<'t, 'r, 's, K>, CaptureRunHostError> {
        let rollback = self.target.owner.identity.custody.bind_scheduled_source(native, &source)?;
        Ok(CapturePartitionRoutedTransfer { writer: self, source, native: rollback.commit() })
    }
}
impl Drop for CapturePartitionRoutedWriter<'_, '_> {
    fn drop(&mut self) {
        if !self.finished { let _ = self.target.owner.remember(Err(RoutedUnitValidationError::Incomplete.into())); }
    }
}
/// A source-bound native transfer and the exact sparse Host writer it funds.
pub struct CapturePartitionRoutedTransfer<'t, 'r, 's, K: Ord + Send + 'static> {
    writer: CapturePartitionRoutedWriter<'t, 'r>,
    source: WorkingMemoryStorage<K>,
    native: &'s mut WorkingMemoryFundingScope,
}
impl<'t,'r,'s,K: Ord + Send + 'static> CapturePartitionRoutedTransfer<'t, 'r, 's, K> {
    /// Original global selection and ownership used by the native row decoder.
    pub fn request(&self) -> PartitionRoutedUnitCaptureRequest<'_> { self.writer.request() }
    /// Receipt-owned invocation layout, independent of the mutable target slice.
    pub fn source_layout(&self)->PartitionRoutedUnitCaptureLayout<'r> {self.writer.source_layout()}
    /// Actual full native source extent.
    pub fn native_rows(&self) -> u64 { self.writer.native_rows() }
    /// Actual pre-exchange token count for this canonical invocation.
    pub fn invocation_source_tokens(&self)->u64 {self.writer.invocation_source_tokens()}
    /// Map a physical/origin token through the original batch-aware prompt window.
    pub fn logical_token(&self,physical:u64)->Option<u64> {self.writer.logical_token(physical)}
    /// Revalidate the actual source/account before and after native settlement.
    pub fn validate(&self) -> Result<(), WorkingMemoryError> {
        self.writer.target.owner.identity.custody.validate_transfer(self.native, &self.source)
    }
    /// Begin one original route after source authentication.
    pub fn begin_row(&mut self, peer: Option<u64>, token: u64, slot: u64, expert: u64,
        coefficient: f32) -> Result<(), CaptureRoutedHostError> {
        self.validate()?; self.writer.begin_row(peer, token, slot, expert, coefficient)
    }
    /// Write one scalar into the fixed selected row.
    pub fn push_f32(&mut self, value: f32) -> Result<(), CaptureRoutedHostError> { self.validate()?; self.writer.push_f32(value) }
    /// Complete one original route.
    pub fn finish_row(&mut self) -> Result<(), CaptureRoutedHostError> { self.validate()?; self.writer.finish_row() }
    /// Record one actual native chunk.
    pub fn source_chunk(&mut self, start: u64, end: u64) -> Result<(), CaptureRoutedHostError> {
        self.validate()?; self.writer.source_chunk(start, end)
    }
    /// Finish this source loan; the original target retains its sparse payload.
    pub fn finish(self) -> Result<(), CaptureRoutedHostError> { self.validate()?; self.writer.finish() }
}
impl<K: Ord + Send + 'static> fmt::Debug for CapturePartitionRoutedTransfer<'_, '_, '_, K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { f.debug_struct("CapturePartitionRoutedTransfer").finish_non_exhaustive() }
}
