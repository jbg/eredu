//! Fixed scalar summary receipts from the existing original capture schedule.
use super::*;

/// Pure layout of one immutable summary selection and its fixed receipt/error.
#[derive(Debug)]
pub struct CaptureSummaryHostPlan<'a> {
    geometry: CaptureSummaryGeometry<'a>,
    peak: u64,
}
impl<'a> CaptureSummaryHostPlan<'a> {
    /// Price the actual fixed constructor/control owners before their birth.
    pub fn prepare(geometry: CaptureSummaryGeometry<'a>) -> Result<Self, WorkingMemoryError> {
        let frames = [
            size_of::<Self>(),
            size_of::<CaptureSummaryClaim<'a, 'a>>(),
            size_of::<ClaimedCaptureSummary>(),
            size_of::<CaptureSummaryFailure>(),
            size_of::<ScheduledCaptureSummaryTransfer<'a, 'a, 'a, u8>>(),
            size_of::<CaptureSummary>(),
            size_of::<crate::capture::reduction::Summary>(),
            size_of::<Result<ClaimedCaptureSummary, CaptureSummaryFailure>>(),
        ];
        let peak = frames
            .into_iter()
            .try_fold(std::mem::size_of_val(&frames), usize::checked_add)
            // Three nested representations: cold plan/query, the moved claim
            // argument, and the transfer/receipt-or-failure return. No extra
            // payload buffers are constructed for these fixed control frames.
            .and_then(|n| n.checked_mul(3))
            .and_then(|n| n.checked_add(CaptureSummaryGeometry::preparation_control_bytes()?))
            .and_then(|n| u64::try_from(n).ok())
            .ok_or(WorkingMemoryError::Overflow)?;
        Ok(Self { geometry, peak })
    }
    /// Exact immutable geometry, not source backing or native admission.
    pub fn geometry(&self) -> &CaptureSummaryGeometry<'a> {
        &self.geometry
    }
    /// Fixed host constructor/receipt/error envelope; no numerical payload Vec.
    pub fn initialization_peak_bytes(&self) -> u64 {
        self.peak
    }
}
/// Once-only scalar claim; the issuing frame remains exclusively borrowed.
#[derive(Debug)]
pub struct CaptureSummaryClaim<'a, 'c> {
    plan: CaptureSummaryHostPlan<'a>,
    identity: claims::ReceiptIdentity,
    chunk: Option<u64>,
    partition: bool,
    exclusive: PhantomData<&'c mut ()>,
}
impl<'a, 'c> CaptureSummaryClaim<'a, 'c> {
    /// Decode a complete remote producer directly into this spent fixed scalar
    /// destination. The shared parser validates the canonical receipt envelope;
    /// the same local summary validator checks counts and finite aggregates.
    /// Native completion and all-rank delivery agreement are separate obligations.
    pub fn decode_partition_receipt(
        self, bytes: &[u8], expected: PartitionCaptureTensorReceipt<'_>,
        funding: &eredu_nn::workspace::WorkspaceMetadataFunding,
    ) -> Result<ClaimedCaptureSummary, PartitionCaptureTensorDecodeError> {
        let custody = self.identity.custody.share_scheduled();
        let value = claims::decode_summary_receipt(bytes, expected, self.geometry(), &custody, funding)?;
        self.finish(value).map_err(|cause| PartitionCaptureTensorDecodeError::retaining_summary_failure(
            cause, custody, funding))
    }

    /// Actual source shape and original selected offsets.
    pub fn geometry(&self) -> &CaptureSummaryGeometry<'a> {
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
    /// Authenticate the exact numerical occurrence that paid this scalar claim.
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
    /// Accept an already completed scalar summary from its exact numerical phase.
    /// Validation failures retain the payload and its original paid custody.
    pub fn finish_numerical(
        self,
        expected: &crate::working_memory::OriginalSpeculativeNumericalBudgetCustody,
        value: CaptureSummary,
    ) -> Result<ClaimedCaptureSummary, CaptureSummaryFailure> {
        if let Err(error) = self.validate_numerical_custody(expected) {
            return Err(CaptureSummaryFailure {
                value,
                error: error.into(),
                custody: self.identity.custody,
            });
        }
        self.finish(value)
    }
    /// Accept the completed summary under the same exact model role.
    /// A validation failure retains the scalar payload and original custody.
    pub fn finish_model(
        self,
        expected: &crate::working_memory::OriginalSpeculativeBudgetCustody,
        value: CaptureSummary,
    ) -> Result<ClaimedCaptureSummary, CaptureSummaryFailure> {
        if let Err(error) = self.validate_model_custody(expected) {
            return Err(CaptureSummaryFailure {
                value,
                error: error.into(),
                custody: self.identity.custody,
            });
        }
        self.finish(value)
    }
    /// Authenticate the existing scheduled native owner without issuing funding.
    pub fn validate_native_scope(
        &self,
        native: &WorkingMemoryFundingScope,
    ) -> Result<(), WorkingMemoryError> {
        self.identity.custody.validate_scheduled_native(native)
    }
    /// Bind exact existing source backing before returning a destination loan.
    pub fn prepare_with_source<'s, K: Clone + Ord + Send + Sync + 'static>(
        self,
        native: &'s mut WorkingMemoryFundingScope,
        source: WorkingMemoryStorage<K>,
    ) -> Result<ScheduledCaptureSummaryTransfer<'a, 'c, 's, K>, CaptureRunHostError> {
        let rollback = self
            .identity
            .custody
            .bind_scheduled_source(native, &source)?;
        self.identity.custody.validate()?;
        let native = rollback.commit();
        Ok(ScheduledCaptureSummaryTransfer {
            claim: self,
            source,
            _segment: None,
            native,
        })
    }
    /// Bind the same scalar claim to its actual canonical prefill source channel.
    pub fn prepare_with_segment_source<'s, K: Clone + Ord + Send + Sync + 'static>(
        self,
        native: &'s mut WorkingMemoryFundingScope,
        segment: &'s mut crate::working_memory::CaptureSourceSegment,
        source: WorkingMemoryStorage<K>,
    ) -> Result<ScheduledCaptureSummaryTransfer<'a, 'c, 's, K>, CaptureRunHostError> {
        segment.validate_summary_geometry(self.geometry())?;
        let rollback = if self.partition {
            self.identity.custody.bind_projected_segment_source(native, segment, &source,
                self.geometry().admission())?
        } else {
            self.identity.custody.bind_segment_source(native, segment, &source)?
        };
        self.identity.custody.validate()?;
        let native = rollback.commit();
        Ok(ScheduledCaptureSummaryTransfer {
            claim: self,
            source,
            _segment: Some(segment),
            native,
        })
    }
    fn finish(self, value: CaptureSummary) -> Result<ClaimedCaptureSummary, CaptureSummaryFailure> {
        let result = (|| {
            self.identity.custody.validate()?;
            crate::capture::reduction::Summary::default()
                .appended_fixed(&value, self.geometry().elements() as u64)
                .map_err(CaptureStepError::from)?;
            Ok::<(), CaptureRunHostError>(())
        })();
        if let Err(error) = result {
            return Err(CaptureSummaryFailure {
                value,
                error,
                custody: self.identity.custody,
            });
        }
        Ok(ClaimedCaptureSummary {
            value,
            chunk: self.chunk,
            identity: self.identity,
        })
    }
    pub(crate) fn finish_empty(self) -> Result<ClaimedCaptureSummary, CaptureSummaryFailure> {
        let value = crate::capture::reduction::Summary::default().value();
        if self.geometry().elements() != 0 {
            return Err(CaptureSummaryFailure {
                value,
                error: CaptureRunHostError::ReceiptMismatch,
                custody: self.identity.custody,
            });
        }
        self.finish(value)
    }
}
/// Completed scalar payload retains its original H until recorded or destroyed.
#[derive(Debug)]
pub struct ClaimedCaptureSummary {
    value: CaptureSummary,
    chunk: Option<u64>,
    identity: claims::ReceiptIdentity,
}
impl ClaimedCaptureSummary {
    /// Read-only finite statistics; no mutable payload/claim escape.
    pub fn observation(&self) -> &CaptureSummary {
        &self.value
    }
}
/// Rejected complete/partial scalar payload; original H is the final member.
#[derive(Debug)]
pub struct CaptureSummaryFailure {
    value: CaptureSummary,
    error: CaptureRunHostError,
    custody: CaptureTensorCustody,
}
impl CaptureSummaryFailure {
    /// Original typed validation failure.
    pub fn error(&self) -> &CaptureRunHostError {
        &self.error
    }
}
impl fmt::Display for CaptureSummaryFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.error, f)
    }
}
impl std::error::Error for CaptureSummaryFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}
/// Source-bound fixed result destination; no replacement account or raw buffer.
pub struct ScheduledCaptureSummaryTransfer<'a, 'c, 's, K: Ord + Send + 'static> {
    claim: CaptureSummaryClaim<'a, 'c>,
    source: WorkingMemoryStorage<K>,
    _segment: Option<&'s mut crate::working_memory::CaptureSourceSegment>,
    native: &'s mut WorkingMemoryFundingScope,
}
impl<K: Ord + Send + 'static> ScheduledCaptureSummaryTransfer<'_, '_, '_, K> {
    /// Validate source pins and exact scheduled owner around native work.
    pub fn validate(&self) -> Result<(), WorkingMemoryError> {
        self.claim
            .identity
            .custody
            .validate_transfer(self.native, &self.source)
    }
    /// Accept only a completed numerical result with exact declared counts.
    pub fn finish(
        self,
        value: CaptureSummary,
    ) -> Result<ClaimedCaptureSummary, CaptureSummaryFailure> {
        if let Err(error) = self.validate() {
            return Err(CaptureSummaryFailure {
                value,
                error: error.into(),
                custody: self.claim.identity.custody,
            });
        }
        self.claim.finish(value)
    }
}
impl<'a> ScheduledCaptureStep<'a> {
    /// Spend one whole-source summary claim under this actual step's original H.
    pub fn take_summary(
        &mut self,
        index: usize,
    ) -> Result<CaptureSummaryClaim<'a, '_>, CaptureRunHostError> {
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
            .summary_geometry(index)
            .map_err(CaptureStepError::from)?;
        if self.claim.window.is_some() && geometry.elements() == 0 {
            return Err(CaptureRunHostError::ClaimUnavailable { index });
        }
        let plan = CaptureSummaryHostPlan::prepare(geometry)?;
        self.claim.row[index + 1] = ClaimState::Spent;
        Ok(CaptureSummaryClaim {
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
    pub fn record_summary(
        &mut self,
        receipt: ClaimedCaptureSummary,
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
            .record_summary(receipt.identity.index, dtype, receipt.value, usage)?;
        Ok(())
    }
}

impl<K: Ord + Send + 'static> fmt::Debug for ScheduledCaptureSummaryTransfer<'_, '_, '_, K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ScheduledCaptureSummaryTransfer")
            .field("claim", &self.claim)
            .finish_non_exhaustive()
    }
}

impl<'a> ScheduledCaptureStep<'a> {
    /// Claim one exact physical contribution under the already charged logical summary.
    pub fn take_prefill_summary(
        &mut self,
        index: usize,
        fragment: &CapturePrefillTransformFragment<'_, '_>,
    ) -> Result<CaptureSummaryClaim<'a, '_>, CaptureRunHostError> {
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
        let geometry = CaptureSummaryGeometry::prepare(
            self.claim.source.admission(),
            index,
            CapturePhase::Prefill,
            0,
            None,
        )
        .and_then(|geometry| geometry.fragment(fragment))
        .map_err(CaptureStepError::from)?;
        let plan = CaptureSummaryHostPlan::prepare(geometry)?;
        let slot = targets
            .slots
            .get_mut(index)
            .ok_or(CapturePrefillHostError::Target { index })?;
        if slot.done || !matches!(slot.state, TargetState::Claimed | TargetState::Active) {
            return Err(CapturePrefillHostError::Target { index }.into());
        }
        // Issuance is nonrefundable; a dropped/failed claim cannot be retried.
        slot.done = true;
        Ok(CaptureSummaryClaim {
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
    pub fn record_prefill_summary(
        &mut self,
        receipt: ClaimedCaptureSummary,
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
        let empty = crate::capture::reduction::Summary::default();
        let summary = slot
            .summary
            .as_ref()
            .unwrap_or(&empty)
            .appended_fixed(&receipt.value, fragment.selected_elements())
            .map_err(CaptureStepError::from)?;
        slot.summary = Some(summary);
        slot.state = TargetState::Active;
        Ok(())
    }
    pub(crate) fn finish_summary_prefill_hook(
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

impl<'a, 'c> CaptureSummaryClaim<'a, 'c> {
    pub(super) fn for_evidence(
        geometry: CaptureSummaryGeometry<'a>,
        identity: claims::ReceiptIdentity,
    ) -> Result<Self, CaptureRunHostError> {
        Ok(Self {
            plan: CaptureSummaryHostPlan::prepare(geometry)?,
            identity,
            chunk: None,
            partition: false, exclusive: PhantomData,
        })
    }
}
impl ClaimedCaptureSummary {
    pub(super) fn evidence_identity(&self) -> &claims::ReceiptIdentity {
        &self.identity
    }
    pub(super) fn into_evidence(
        self,
    ) -> Result<(CaptureSummary, claims::ReceiptIdentity), CaptureRunHostError> {
        if self.chunk.is_some() {
            return Err(CaptureRunHostError::ReceiptMismatch);
        }
        Ok((self.value, self.identity))
    }
}

impl<'a> ScheduledCaptureStep<'a> {
    /// Spend the complete scalar destination only after all real remote prompt
    /// hooks have completed. The same original H owns both progress and result.
    pub(crate) fn take_remote_prefill_summary<'c>(&'c mut self, index: usize)
        -> Result<CaptureSummaryClaim<'a, 'c>, CaptureRunHostError> {
        self.validate_remote_prefill_claim(index)?;
        let geometry = self.claim.policy()?.summary_geometry(index).map_err(CaptureStepError::from)?;
        let plan = CaptureSummaryHostPlan::prepare(geometry)?;
        self.begin_remote_prefill_summary_claim(index)?;
        Ok(CaptureSummaryClaim { plan, chunk: None, partition: false, exclusive: PhantomData,
            identity: claims::ReceiptIdentity { phase: self.claim.phase, prediction: self.claim.prediction,
                index, custody: self.claim.custody.share_scheduled() } })
    }
    pub(crate) fn record_remote_prefill_summary(&mut self, receipt: ClaimedCaptureSummary,
        source_dtype: TensorDtype) -> Result<(), CaptureRunHostError> {
        let index = receipt.identity.index;
        self.validate_remote_prefill_record(index, &source_dtype)?;
        self.record_summary(receipt, source_dtype, CaptureUsage::default())?;
        self.mark_remote_prefill_recorded(index)
    }
}

impl<'a,'c> CaptureSummaryClaim<'a,'c> {
    pub(in crate::working_memory::capture_run) fn from_partition_plan(plan:CaptureSummaryHostPlan<'a>,identity:claims::ReceiptIdentity)->Self {
        Self {plan,identity,chunk:None,partition:false,exclusive:PhantomData}
    }
}

impl CaptureSummaryClaim<'_, '_> {
    pub(in crate::working_memory::capture_run) fn partition_identity(&self)->&claims::ReceiptIdentity {&self.identity}
    pub(in crate::working_memory::capture_run) fn finish_partition(self,value:CaptureSummary)->Result<ClaimedCaptureSummary,CaptureSummaryFailure> {self.finish(value)}
}

impl<'a,'c> CaptureSummaryClaim<'a,'c> {
    pub(in crate::working_memory::capture_run) fn from_partition_prefill_plan(
        plan:CaptureSummaryHostPlan<'a>,identity:claims::ReceiptIdentity,chunk:u64,
    )->Self {Self{plan,identity,chunk:Some(chunk),partition:true,exclusive:PhantomData}}
}
impl ClaimedCaptureSummary {
    pub(in crate::working_memory::capture_run) fn partition_chunk(&self)->Option<u64>{self.chunk}
}
