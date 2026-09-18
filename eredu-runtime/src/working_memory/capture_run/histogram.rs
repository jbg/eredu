//! Fixed-edge bins under the existing once-only capture source and host account.
use super::*;
/// Exact immutable Histogram layout and its live accumulator/fragment payloads.
#[derive(Debug)]
pub struct CaptureHistogramHostPlan<'a> {
    geometry: CaptureHistogramGeometry<'a>,
    peak: u64,
}
impl<'a> CaptureHistogramHostPlan<'a> {
    /// Price both retained accumulator and current physical fragment before birth.
    pub fn prepare(geometry: CaptureHistogramGeometry<'a>) -> Result<Self, WorkingMemoryError> {
        let edges = geometry.edges().len();
        let payload = edges
            .checked_mul(size_of::<f32>())
            .and_then(|n| n.checked_add(edges.checked_sub(1)?.checked_mul(size_of::<u64>())?))
            .filter(|n| *n <= isize::MAX as usize)
            .ok_or(WorkingMemoryError::Overflow)?;
        let frames = [
            size_of::<Self>(),
            size_of::<CaptureHistogramClaim<'a, 'a>>(),
            size_of::<ScheduledCaptureHistogram<'a, 'a>>(),
            size_of::<ClaimedCaptureHistogram>(),
            size_of::<CaptureHistogramFailure>(),
            size_of::<ScheduledCaptureHistogramTransfer<'a, 'a, 'a, u8>>(),
            size_of::<CaptureHistogram>(),
            size_of::<Result<ClaimedCaptureHistogram, CaptureHistogramFailure>>(),
        ];
        // Cold plan, moved claim and transfer/result frames are simultaneously nested.
        // Exactly two heap payloads can coexist: accumulated earlier chunks and the
        // current fragment (which moves into the accumulator for its first chunk).
        let peak = frames
            .into_iter()
            .try_fold(std::mem::size_of_val(&frames), usize::checked_add)
            .and_then(|n| n.checked_mul(3))
            .and_then(|n| n.checked_add(CaptureHistogramGeometry::preparation_control_bytes()?))
            .and_then(|n| n.checked_add(payload.checked_mul(2)?))
            .and_then(|n| u64::try_from(n).ok())
            .ok_or(WorkingMemoryError::Overflow)?;
        Ok(Self { geometry, peak })
    }
    /// Original immutable shape/edge facts.
    pub fn geometry(&self) -> &CaptureHistogramGeometry<'a> {
        &self.geometry
    }
    /// Actual host payload and named constructor/control envelope.
    pub fn initialization_peak_bytes(&self) -> u64 {
        self.peak
    }
}
/// Exclusive once-only claim issued by its scheduled frame.
#[derive(Debug)]
pub struct CaptureHistogramClaim<'a, 'c> {
    plan: CaptureHistogramHostPlan<'a>,
    identity: claims::ReceiptIdentity,
    chunk: Option<u64>,
    partition: bool,
    exclusive: PhantomData<&'c mut ()>,
}
impl<'a, 'c> CaptureHistogramClaim<'a, 'c> {
    /// Decode into this claim's fixed paid edges/bins using the canonical
    /// partition reader. Failed input retains its partial payload and account.
    pub fn decode_partition_receipt(self, bytes: &[u8], expected: PartitionCaptureTensorReceipt<'_>,
        funding: &eredu_nn::workspace::HostMetadataFunding,
    ) -> Result<ClaimedCaptureHistogram, PartitionCaptureTensorDecodeError> {
        let custody=self.identity.custody.share_scheduled();
        claims::prepare_histogram_decoder(&custody,funding)?;
        let mut destination=self.prepare().map_err(|cause|
            claims::histogram_preparation_failure(cause,&custody,funding))?;
        if let Err(error)=claims::decode_histogram_receipt(&mut destination.value,bytes,expected,
            destination.claim.geometry(),&custody,funding) {
            return Err(error.retaining_histogram_payload(destination.value));
        }
        let (below,above,non_finite)=(destination.value.below,destination.value.above,destination.value.non_finite);
        destination.finish(below,above,non_finite).map_err(|cause|
            PartitionCaptureTensorDecodeError::retaining_histogram_failure(cause,custody,funding))
    }

    /// Exact geometry; edges borrow the immutable original admission.
    pub fn geometry(&self) -> &CaptureHistogramGeometry<'a> {
        self.plan.geometry()
    }
    /// Authenticate the exact model occurrence that paid this host claim.
    /// This supplies no native source, scope or completion evidence.
    pub fn validate_model_custody(
        &self,
        expected: &crate::working_memory::OriginalSpeculativeBudgetCustody,
    ) -> Result<(), WorkingMemoryError> {
        self.identity.custody.validate_model(expected)
    }
    /// Authenticate the exact numerical phase before constructing its bin storage.
    pub fn validate_numerical_custody(
        &self,
        expected: &crate::working_memory::OriginalSpeculativeNumericalBudgetCustody,
    ) -> Result<(), WorkingMemoryError> {
        match &self.identity.custody {
            CaptureTensorCustody::Speculative(actual) if actual.same_account(expected) => {
                self.identity.custody.validate()
            }
            _ => Err(WorkingMemoryError::IdentityMismatch),
        }
    }
    /// Authenticate the existing native owner without granting an allocation.
    pub fn validate_native_scope(
        &self,
        native: &WorkingMemoryFundingScope,
    ) -> Result<(), WorkingMemoryError> {
        self.identity.custody.validate_scheduled_native(native)
    }
    /// Construct fixed bin destinations only after the accepted host account validates.
    pub fn prepare(self) -> Result<ScheduledCaptureHistogram<'a, 'c>, CaptureRunHostError> {
        self.identity.custody.validate()?;
        let edges = self.geometry().edges();
        let value = CaptureHistogram {
            edges: edges.to_vec(),
            counts: vec![0; edges.len() - 1],
            below: 0,
            above: 0,
            non_finite: 0,
        };
        Ok(ScheduledCaptureHistogram {
            value,
            failure: None,
            claim: self,
        })
    }
    /// Pin actual original backing before constructing bin storage.
    pub fn prepare_with_source<'s, K: Clone + Ord + Send + Sync + 'static>(
        self,
        native: &'s mut WorkingMemoryFundingScope,
        source: WorkingMemoryStorage<K>,
    ) -> Result<ScheduledCaptureHistogramTransfer<'a, 'c, 's, K>, CaptureRunHostError> {
        let rollback = self
            .identity
            .custody
            .bind_scheduled_source(native, &source)?;
        let builder = self.prepare()?;
        let native = rollback.commit();
        Ok(ScheduledCaptureHistogramTransfer {
            builder,
            source,
            _segment: None,
            native,
        })
    }
    /// Use the same canonical per-chunk source/pin transaction.
    pub fn prepare_with_segment_source<'s, K: Clone + Ord + Send + Sync + 'static>(
        self,
        native: &'s mut WorkingMemoryFundingScope,
        segment: &'s mut crate::working_memory::CaptureSourceSegment,
        source: WorkingMemoryStorage<K>,
    ) -> Result<ScheduledCaptureHistogramTransfer<'a, 'c, 's, K>, CaptureRunHostError> {
        segment.validate_histogram_geometry(self.geometry())?;
        let rollback = if self.partition {
            self.identity.custody.bind_projected_segment_source(native, segment, &source,
                self.geometry().admission())?
        } else {
            self.identity.custody.bind_segment_source(native, segment, &source)?
        };
        let builder = self.prepare()?;
        let native = rollback.commit();
        Ok(ScheduledCaptureHistogramTransfer {
            builder,
            source,
            _segment: Some(segment),
            native,
        })
    }
}
/// Partial bins retire before their original paying claim; no raw allocation escape.
#[derive(Debug)]
pub struct ScheduledCaptureHistogram<'a, 'c> {
    value: CaptureHistogram,
    failure: Option<WorkingMemoryError>,
    claim: CaptureHistogramClaim<'a, 'c>,
}
impl ScheduledCaptureHistogram<'_, '_> {
    /// Add one completed scalar bin count without growth; any failure poisons the builder.
    pub fn add_bin(&mut self, index: usize, count: u64) -> Result<(), WorkingMemoryError> {
        if let Some(error) = &self.failure {
            return Err(error.clone());
        }
        let result: Result<(), WorkingMemoryError> = (|| {
            self.claim.identity.custody.validate()?;
            let bin = self
                .value
                .counts
                .get_mut(index)
                .ok_or(WorkingMemoryError::IdentityMismatch)?;
            *bin = bin.checked_add(count).ok_or(WorkingMemoryError::Overflow)?;
            Ok(())
        })();
        if let Err(error) = &result {
            self.failure = Some(error.clone());
        }
        result
    }
    fn fail(self, error: CaptureRunHostError) -> CaptureHistogramFailure {
        CaptureHistogramFailure {
            value: self.value,
            error,
            custody: self.claim.identity.custody,
        }
    }
    /// Publish only under the same numerical phase that constructed these bins.
    /// A mismatch retains the partial payload and its actual paying account.
    pub fn finish_numerical(
        self,
        expected: &crate::working_memory::OriginalSpeculativeNumericalBudgetCustody,
        below: u64,
        above: u64,
        non_finite: u64,
    ) -> Result<ClaimedCaptureHistogram, CaptureHistogramFailure> {
        if let Err(error) = self.claim.validate_numerical_custody(expected) {
            return Err(self.fail(error.into()));
        }
        self.finish(below, above, non_finite)
    }
    /// Publish only under the same model occurrence that constructed these bins.
    /// A mismatch retains the partial payload and its actual paying account.
    pub fn finish_model(
        self,
        expected: &crate::working_memory::OriginalSpeculativeBudgetCustody,
        below: u64,
        above: u64,
        non_finite: u64,
    ) -> Result<ClaimedCaptureHistogram, CaptureHistogramFailure> {
        if let Err(error) = self.claim.validate_model_custody(expected) {
            return Err(self.fail(error.into()));
        }
        self.finish(below, above, non_finite)
    }
    /// Publish only a complete exact partition after the native worker settles its reads.
    pub fn finish(
        mut self,
        below: u64,
        above: u64,
        non_finite: u64,
    ) -> Result<ClaimedCaptureHistogram, CaptureHistogramFailure> {
        self.value.below = below;
        self.value.above = above;
        self.value.non_finite = non_finite;
        let result = (|| {
            if let Some(error) = &self.failure {
                return Err(error.clone().into());
            }
            self.claim.identity.custody.validate()?;
            crate::capture::reduction::validate_histogram_fixed(
                &self.value,
                self.claim.geometry().edges(),
                self.claim.geometry().elements() as u64,
            )
            .map_err(CaptureStepError::from)?;
            Ok::<(), CaptureRunHostError>(())
        })();
        if let Err(error) = result {
            return Err(self.fail(error));
        }
        Ok(ClaimedCaptureHistogram {
            value: self.value,
            chunk: self.claim.chunk,
            identity: self.claim.identity,
        })
    }
}
/// Closed complete payload with its original receipt identity/account last.
#[derive(Debug)]
pub struct ClaimedCaptureHistogram {
    value: CaptureHistogram,
    chunk: Option<u64>,
    identity: claims::ReceiptIdentity,
}
impl ClaimedCaptureHistogram {
    /// Immutable completed bins; mutation and raw payload extraction remain private.
    pub fn observation(&self) -> &CaptureHistogram {
        &self.value
    }
}
/// Rejected partial/complete payload retains original H through vector destruction.
#[derive(Debug)]
pub struct CaptureHistogramFailure {
    value: CaptureHistogram,
    error: CaptureRunHostError,
    custody: CaptureTensorCustody,
}
impl CaptureHistogramFailure {
    /// Original typed cause.
    pub fn error(&self) -> &CaptureRunHostError {
        &self.error
    }
}
impl fmt::Display for CaptureHistogramFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.error, f)
    }
}
impl std::error::Error for CaptureHistogramFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}
/// Source-bound fixed bins, retaining actual source pin through all native reads.
pub struct ScheduledCaptureHistogramTransfer<'a, 'c, 's, K: Ord + Send + 'static> {
    builder: ScheduledCaptureHistogram<'a, 'c>,
    source: WorkingMemoryStorage<K>,
    _segment: Option<&'s mut crate::working_memory::CaptureSourceSegment>,
    native: &'s mut WorkingMemoryFundingScope,
}
impl<K: Ord + Send + 'static> ScheduledCaptureHistogramTransfer<'_, '_, '_, K> {
    /// Check the original schedule and source pin.
    pub fn validate(&self) -> Result<(), WorkingMemoryError> {
        self.builder
            .claim
            .identity
            .custody
            .validate_transfer(self.native, &self.source)
    }
    /// Add a completed bin scalar; no backing, vector or account is exposed.
    pub fn add_bin(&mut self, index: usize, count: u64) -> Result<(), WorkingMemoryError> {
        self.builder.add_bin(index, count)
    }
    /// Move a completed exact partition into its once-only receipt.
    pub fn finish(
        self,
        below: u64,
        above: u64,
        non_finite: u64,
    ) -> Result<ClaimedCaptureHistogram, CaptureHistogramFailure> {
        if let Err(error) = self.validate() {
            return Err(self.builder.fail(error.into()));
        }
        self.builder.finish(below, above, non_finite)
    }
}
impl<K: Ord + Send + 'static> fmt::Debug for ScheduledCaptureHistogramTransfer<'_, '_, '_, K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ScheduledCaptureHistogramTransfer")
            .field("builder", &self.builder)
            .finish_non_exhaustive()
    }
}
impl<'a> ScheduledCaptureStep<'a> {
    /// Spend one whole-source histogram claim under this actual step's original H.
    pub fn take_histogram(
        &mut self,
        index: usize,
    ) -> Result<CaptureHistogramClaim<'a, '_>, CaptureRunHostError> {
        self.claim.custody.validate()?;
        if self.frame.prefill.is_some() {
            return Err(CapturePrefillHostError::Identity.into());
        }
        if self
            .claim
            .row
            .get(index.checked_add(1).ok_or(WorkingMemoryError::Overflow)?)
            != Some(&ClaimState::Available)
            || !self
                .frame
                .records()
                .get(index)
                .is_some_and(|r| matches!(r.outcome, CaptureOutcome::Missing))
        {
            return Err(CaptureRunHostError::ClaimUnavailable { index });
        }
        let geometry = self
            .claim
            .policy()?
            .histogram_geometry(index)
            .map_err(CaptureStepError::from)?;
        if self.claim.window.is_some() && geometry.elements() == 0 {
            return Err(CaptureRunHostError::ClaimUnavailable { index });
        }
        let plan = CaptureHistogramHostPlan::prepare(geometry)?;
        self.claim.row[index + 1] = ClaimState::Spent;
        Ok(CaptureHistogramClaim {
            plan,
            chunk: None,
            identity: claims::ReceiptIdentity {
                phase: self.claim.phase,
                prediction: self.claim.prediction,
                index,
                custody: self.claim.custody.share_scheduled(),
            },
            partition: false, exclusive: PhantomData,
        })
    }
    /// Move a result only to the original bank and scheduling coordinate.
    pub fn record_histogram(
        &mut self,
        receipt: ClaimedCaptureHistogram,
        dtype: TensorDtype,
        usage: CaptureUsage,
    ) -> Result<(), CaptureRunHostError> {
        self.claim.custody.validate()?;
        if !self.claim.custody.same_schedule(&receipt.identity.custody)
            || self.claim.phase != receipt.identity.phase
            || self.claim.prediction != receipt.identity.prediction
            || receipt.chunk.is_some()
        {
            return Err(CaptureRunHostError::ReceiptMismatch);
        }
        self.frame
            .record_histogram(receipt.identity.index, dtype, receipt.value, usage)?;
        Ok(())
    }
}

impl<'a> ScheduledCaptureStep<'a> {
    /// Claim one exact physical contribution under the already charged logical histogram.
    pub fn take_prefill_histogram(
        &mut self,
        index: usize,
        fragment: &CapturePrefillTransformFragment<'_, '_>,
    ) -> Result<CaptureHistogramClaim<'a, '_>, CaptureRunHostError> {
        use crate::working_memory::capture_tensor::prefill::TargetState;
        self.claim.custody.validate()?;
        let targets = self
            .frame
            .prefill
            .as_mut()
            .ok_or(CapturePrefillHostError::Identity)?;
        if !std::ptr::eq(fragment.plan().admission(), self.claim.source.admission())
            || fragment.plan().selection_index() != index
            || fragment.plan().inference_geometry() != targets.inference
            || fragment.chunk_index() != targets.next
            || self
                .claim
                .row
                .get(index.checked_add(1).ok_or(WorkingMemoryError::Overflow)?)
                != Some(&ClaimState::Spent)
        {
            return Err(CapturePrefillHostError::Identity.into());
        }
        let geometry = CaptureHistogramGeometry::prepare(
            self.claim.source.admission(),
            index,
            CapturePhase::Prefill,
            0,
            None,
        )
        .and_then(|geometry| geometry.fragment(fragment))
        .map_err(CaptureStepError::from)?;
        let plan = CaptureHistogramHostPlan::prepare(geometry)?;
        let slot = targets
            .slots
            .get_mut(index)
            .ok_or(CapturePrefillHostError::Target { index })?;
        if slot.done || !matches!(slot.state, TargetState::Claimed | TargetState::Active) {
            return Err(CapturePrefillHostError::Target { index }.into());
        }
        // Issuance is nonrefundable; a dropped/failed claim cannot be retried.
        slot.done = true;
        Ok(CaptureHistogramClaim {
            plan,
            chunk: Some(targets.next),
            identity: claims::ReceiptIdentity {
                phase: CapturePhase::Prefill,
                prediction: 0,
                index,
                custody: self.claim.custody.share_scheduled(),
            },
            partition: false, exclusive: PhantomData,
        })
    }
    /// Validate the entire scalar contribution before mutating the existing accumulator.
    pub fn record_prefill_histogram(
        &mut self,
        receipt: ClaimedCaptureHistogram,
        fragment: &CapturePrefillTransformFragment<'_, '_>,
    ) -> Result<(), CaptureRunHostError> {
        use crate::working_memory::capture_tensor::prefill::TargetState;
        self.claim.custody.validate()?;
        let index = receipt.identity.index;
        let targets = self
            .frame
            .prefill
            .as_mut()
            .ok_or(CapturePrefillHostError::Identity)?;
        if !self.claim.custody.same_schedule(&receipt.identity.custody)
            || receipt.identity.phase != CapturePhase::Prefill
            || receipt.identity.prediction != 0
            || receipt.chunk != Some(targets.next)
            || fragment.chunk_index() != targets.next
            || !std::ptr::eq(fragment.plan().admission(), self.claim.source.admission())
            || fragment.plan().selection_index() != index
            || fragment.plan().inference_geometry() != targets.inference
        {
            return Err(CaptureRunHostError::ReceiptMismatch);
        }
        let slot = targets
            .slots
            .get_mut(index)
            .ok_or(CapturePrefillHostError::Target { index })?;
        if !slot.done || !matches!(slot.state, TargetState::Claimed | TargetState::Active) {
            return Err(CapturePrefillHostError::Target { index }.into());
        }
        crate::capture::reduction::validate_histogram_fixed(
            &receipt.value,
            match &fragment.plan().selection().transform {
                CaptureTransform::Histogram { edges } => edges,
                _ => return Err(CaptureRunHostError::ReceiptMismatch),
            },
            fragment.selected_elements(),
        )
        .map_err(CaptureStepError::from)?;
        if let Some(out) = &mut slot.histogram {
            crate::capture::reduction::add_histogram_fixed(out, &receipt.value)
                .map_err(CaptureStepError::from)?;
        } else {
            slot.histogram = Some(receipt.value);
        }
        slot.state = TargetState::Active;
        Ok(())
    }
    pub(crate) fn finish_histogram_prefill_hook(
        &mut self,
        index: usize,
        fragment: &CapturePrefillTransformFragment<'_, '_>,
    ) -> Result<(), CaptureRunHostError> {
        self.claim.custody.validate()?;
        let targets = self
            .frame
            .prefill
            .as_mut()
            .ok_or(CapturePrefillHostError::Identity)?;
        let policy = crate::capture::CapturePrefillObservationPolicy::new(
            self.claim.source,
            targets.inference,
        )
        .map_err(CapturePrefillHostError::from)?;
        let row = policy.row(index).map_err(CapturePrefillHostError::from)?;
        let progress = targets
            .slots
            .get_mut(index)
            .and_then(|slot| slot.progression.as_mut())
            .ok_or(CapturePrefillHostError::Identity)?;
        row.finish_transform_hook(progress, fragment)
            .map_err(CapturePrefillHostError::from)?;
        Ok(())
    }
}

impl<'a> ScheduledCaptureStep<'a> {
    pub(crate) fn take_remote_prefill_histogram<'c>(&'c mut self,index:usize)
        -> Result<CaptureHistogramClaim<'a,'c>,CaptureRunHostError> {
        self.validate_remote_prefill_claim(index)?;
        let geometry=self.claim.policy()?.histogram_geometry(index).map_err(CaptureStepError::from)?;
        let plan=CaptureHistogramHostPlan::prepare(geometry)?;
        self.begin_remote_prefill_histogram_claim(index)?;
        Ok(CaptureHistogramClaim {plan,chunk:None,partition:false,exclusive:PhantomData,
            identity:claims::ReceiptIdentity {phase:self.claim.phase,prediction:self.claim.prediction,
                index,custody:self.claim.custody.share_scheduled()}})
    }
    pub(crate) fn record_remote_prefill_histogram(&mut self,receipt:ClaimedCaptureHistogram,
        source_dtype:TensorDtype)->Result<(),CaptureRunHostError> {
        let index=receipt.identity.index;
        self.validate_remote_prefill_record(index,&source_dtype)?;
        self.record_histogram(receipt,source_dtype,CaptureUsage::default())?;
        self.mark_remote_prefill_recorded(index)
    }
}

impl<'a,'c> CaptureHistogramClaim<'a,'c> {
    pub(in crate::working_memory::capture_run) fn from_partition_plan(plan:CaptureHistogramHostPlan<'a>,identity:claims::ReceiptIdentity)->Self {
        Self {plan,identity,chunk:None,partition:false,exclusive:PhantomData}
    }
}
impl ClaimedCaptureHistogram {
    pub(in crate::working_memory::capture_run) fn partition_identity(&self)->&claims::ReceiptIdentity {&self.identity}
}

impl CaptureHistogramClaim<'_, '_> {
    pub(in crate::working_memory::capture_run) fn partition_identity(&self)->&claims::ReceiptIdentity {&self.identity}
}
impl ScheduledCaptureHistogram<'_, '_> {
    pub(in crate::working_memory::capture_run) fn merge_partition(&mut self,value:&CaptureHistogram)->Result<(),CaptureRunHostError> {
        self.claim.identity.custody.validate()?;
        crate::capture::reduction::add_histogram_fixed(&mut self.value,value).map_err(CaptureStepError::from)?;
        Ok(())
    }
    pub(in crate::working_memory::capture_run) fn fill_partition_sum(&mut self,values:&[f32])->Result<(),CaptureRunHostError> {
        self.claim.identity.custody.validate()?;
        if values.len()!=self.claim.geometry().elements(){return Err(CaptureRunHostError::ReceiptMismatch);}
        crate::capture::partition::fill_histogram_f32(values,&mut self.value).map_err(CaptureStepError::from)?;
        Ok(())
    }
    pub(in crate::working_memory::capture_run) fn finish_partition(self)->Result<ClaimedCaptureHistogram,CaptureHistogramFailure> {
        let (below,above,non_finite)=(self.value.below,self.value.above,self.value.non_finite);
        self.finish(below,above,non_finite)
    }
}

impl<'a,'c> CaptureHistogramClaim<'a,'c> {
    pub(in crate::working_memory::capture_run) fn from_partition_prefill_plan(
        plan:CaptureHistogramHostPlan<'a>,identity:claims::ReceiptIdentity,chunk:u64,
    )->Self {Self{plan,identity,chunk:Some(chunk),partition:true,exclusive:PhantomData}}
    // The exact earlier-chunk buffer becomes the final receipt without another
    // bin allocation; use the same completed histogram validator as all workers.
    pub(in crate::working_memory::capture_run) fn finish_partition_accumulator(
        self,value:CaptureHistogram,
    )->Result<ClaimedCaptureHistogram,CaptureHistogramFailure>{
        ScheduledCaptureHistogram{value,failure:None,claim:self}.finish_partition()
    }
}
impl ClaimedCaptureHistogram {
    pub(in crate::working_memory::capture_run) fn partition_chunk(&self)->Option<u64>{self.chunk}
    pub(in crate::working_memory::capture_run) fn into_partition_histogram(self)->CaptureHistogram{self.value}
}
