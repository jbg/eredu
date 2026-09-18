//! One candidate destination from an existing original whole-run H claim.
use super::*;

/// Allocation-free fixed candidate destination layout; native workspace is separate.
#[derive(Debug)]
pub struct CaptureCandidateHostPlan<'a> {
    geometry: CaptureCandidateGeometry<'a>,
    peak: u64,
}
impl<'a> CaptureCandidateHostPlan<'a> {
    /// Derive actual candidate slots and constructor/receipt/error controls.
    pub fn prepare(geometry: CaptureCandidateGeometry<'a>) -> Result<Self, WorkingMemoryError> {
        let slots = geometry
            .count()
            .checked_mul(size_of::<CaptureCandidate>())
            .filter(|n| *n <= isize::MAX as usize)
            .ok_or(WorkingMemoryError::Overflow)?;
        let controls = size_of::<CaptureCandidates>()
            .checked_add(size_of::<ScheduledCaptureCandidates<'_, '_>>())
            .and_then(|n| n.checked_add(size_of::<ClaimedCaptureCandidates>()))
            .and_then(|n| n.checked_add(size_of::<CaptureCandidateFailure>()))
            .and_then(|n| n.checked_add(size_of::<CaptureCandidateClaim<'_, '_>>()))
            .and_then(|n| {
                n.checked_add(size_of::<ScheduledCaptureCandidatesTransfer<'_, '_, '_, u8>>())
            })
            .and_then(|n| n.checked_mul(3))
            .ok_or(WorkingMemoryError::Overflow)?;
        let peak = slots
            .checked_add(controls)
            .and_then(|n| u64::try_from(n).ok())
            .ok_or(WorkingMemoryError::Overflow)?;
        Ok(Self { geometry, peak })
    }
    /// Fixed peak of requested payload/control capacity; no allocator grant.
    pub fn initialization_peak_bytes(&self) -> u64 {
        self.peak
    }
    /// Borrow the exact semantic/physical source description.
    pub fn geometry(&self) -> &CaptureCandidateGeometry<'a> {
        &self.geometry
    }
}

/// Exclusive once-only candidate claim, issued only by its original frame.
#[derive(Debug)]
pub struct CaptureCandidateClaim<'a, 'c> {
    plan: CaptureCandidateHostPlan<'a>,
    identity: claims::ReceiptIdentity,
    exclusive: PhantomData<&'c mut ()>,
}
impl CaptureCandidateClaim<'_, '_> {
    /// Exact original selection, with the actual terminal readout rows.
    pub fn geometry(&self) -> &CaptureCandidateGeometry<'_> {
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
    /// Authenticate the exact numerical phase that paid this already-spent
    /// claim. No source, native scope or additional authority is supplied.
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
    /// Authenticate the same original native work; no replacement hold is issued.
    pub fn validate_native_scope(
        &self,
        native: &WorkingMemoryFundingScope,
    ) -> Result<(), WorkingMemoryError> {
        self.identity.custody.validate_scheduled_native(native)
    }
}
impl<'a, 'c> CaptureCandidateClaim<'a, 'c> {
    /// Bind complete source origins to this exact active scope before allocating
    /// candidate slots. Failure restores prior pins; success keeps them in native
    /// recovery through settlement/quarantine, without another hold or claim.
    pub fn prepare_with_source<'s, K: Clone + Ord + Send + Sync + 'static>(
        self,
        native: &'s mut WorkingMemoryFundingScope,
        source: WorkingMemoryStorage<K>,
        domain: Option<CandidateDomain>,
    ) -> Result<ScheduledCaptureCandidatesTransfer<'a, 'c, 's, K>, CaptureRunHostError> {
        let rollback = self
            .identity
            .custody
            .bind_scheduled_source(native, &source)?;
        let builder = self.prepare(domain)?;
        let native = rollback.commit();
        Ok(ScheduledCaptureCandidatesTransfer {
            builder,
            source,
            _segment: None,
            native,
        })
    }
    /// Same closed transfer on the actual canonical terminal chunk channel.
    pub fn prepare_with_segment_source<'s, K: Clone + Ord + Send + Sync + 'static>(
        self,
        native: &'s mut WorkingMemoryFundingScope,
        segment: &'s mut crate::working_memory::CaptureSourceSegment,
        source: WorkingMemoryStorage<K>,
        domain: Option<CandidateDomain>,
    ) -> Result<ScheduledCaptureCandidatesTransfer<'a, 'c, 's, K>, CaptureRunHostError> {
        segment.validate_candidate_geometry(self.geometry())?;
        let rollback = self
            .identity
            .custody
            .bind_segment_source(native, segment, &source)?;
        let builder = self.prepare(domain)?;
        let native = rollback.commit();
        Ok(ScheduledCaptureCandidatesTransfer {
            builder,
            source,
            _segment: Some(segment),
            native,
        })
    }
    /// Allocate the fixed host slots after original account authentication.
    /// Domain is evidence about the actual decision, never allocation authority.
    pub fn prepare(
        self,
        domain: Option<CandidateDomain>,
    ) -> Result<ScheduledCaptureCandidates<'a, 'c>, CaptureRunHostError> {
        self.identity.custody.validate()?;
        let values = Vec::with_capacity(self.plan.geometry.count());
        Ok(ScheduledCaptureCandidates {
            values,
            failure: None,
            plan: self.plan,
            domain,
            identity: self.identity,
            exclusive: self.exclusive,
        })
    }
}

/// Partial candidate storage remains under the original H on every failure prefix.
#[derive(Debug)]
pub struct ScheduledCaptureCandidates<'a, 'c> {
    values: Vec<CaptureCandidate>,
    failure: Option<WorkingMemoryError>,
    plan: CaptureCandidateHostPlan<'a>,
    domain: Option<CandidateDomain>,
    identity: claims::ReceiptIdentity,
    exclusive: PhantomData<&'c mut ()>,
}
impl ScheduledCaptureCandidates<'_, '_> {
    /// Fill one existing slot; growth, nonfinite values and ascending scores reject.
    pub fn push(
        &mut self,
        token_id: u32,
        score: f32,
        allowed: bool,
    ) -> Result<(), WorkingMemoryError> {
        if let Some(error) = &self.failure {
            return Err(error.clone());
        }
        if let Err(error) = self.identity.custody.validate() {
            self.failure = Some(error.clone());
            return Err(error);
        }
        if self.values.len() == self.plan.geometry.count()
            || token_id as usize >= self.plan.geometry.vocabulary()
            || !score.is_finite()
            || self
                .values
                .last()
                .is_some_and(|previous| previous.score < score)
        {
            self.failure = Some(WorkingMemoryError::IdentityMismatch);
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        self.values.push(CaptureCandidate {
            token_id,
            score,
            allowed,
        });
        Ok(())
    }
    /// Number of actual initialized entries, distinct from fixed capacity.
    pub fn initialized_count(&self) -> usize {
        self.values.len()
    }
    /// Finish by move only; an error keeps partial values and their original custody.
    pub fn finish(self) -> Result<ClaimedCaptureCandidates, CaptureCandidateFailure> {
        let error = self
            .failure
            .or_else(|| self.identity.custody.validate().err())
            .or_else(|| {
                (self.values.len() != self.plan.geometry.count())
                    .then_some(WorkingMemoryError::IdentityMismatch)
            });
        let candidates = CaptureCandidates {
            stage: CandidateScoreStage::RawLogitsBeforeSampling,
            source: candidate_source(&self.plan.geometry),
            candidates: self.values,
            domain: self.domain,
        };
        if let Some(error) = error {
            return Err(CaptureCandidateFailure {
                error,
                candidates,
                custody: self.identity.custody,
            });
        }
        Ok(ClaimedCaptureCandidates {
            candidates,
            shape: *self.plan.geometry.source_shape(),
            identity: self.identity,
        })
    }
}
fn candidate_source(geometry: &CaptureCandidateGeometry<'_>) -> CandidateLogitsSource {
    if geometry.admission().points()[geometry.selection_index()].position
        == eredu_core::ObservationPosition::AfterIntervention
    {
        CandidateLogitsSource::Effective
    } else {
        CandidateLogitsSource::Original
    }
}
/// Exact typed error and partial candidate payload retire before their final H.
#[derive(Debug)]
pub struct CaptureCandidateFailure {
    error: WorkingMemoryError,
    candidates: CaptureCandidates,
    custody: CaptureTensorCustody,
}
impl fmt::Display for CaptureCandidateFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.error, f)
    }
}
impl std::error::Error for CaptureCandidateFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}
/// No raw owning export. Only the originating frame can consume this receipt.
#[derive(Debug)]
pub struct ClaimedCaptureCandidates {
    candidates: CaptureCandidates,
    shape: [usize; 3],
    identity: claims::ReceiptIdentity,
}
impl ClaimedCaptureCandidates {
    /// Borrow the same initialized buffer; copying is separately caller-owned.
    pub fn observation(&self) -> &CaptureCandidates {
        &self.candidates
    }
}
impl<'a> ScheduledCaptureStep<'a> {
    // Shared once-only terminal destination coordinate validation. Specific
    // candidate/score geometry and payload validation remain in their typed plan.
    pub(super) fn terminal_claim_rows(
        &self,
        index: usize,
    ) -> Result<Option<usize>, CaptureRunHostError> {
        self.claim.custody.validate()?;
        if self.claim.row.get(index + 1) != Some(&ClaimState::Available)
            || !self
                .frame
                .records()
                .get(index)
                .is_some_and(|r| matches!(r.outcome, CaptureOutcome::Missing))
        {
            return Err(CaptureRunHostError::ClaimUnavailable { index });
        }
        if let Some(targets) = self.frame.prefill.as_ref() {
            let end = (targets.next + 1)
                .checked_mul(targets.inference.prefill_chunk_positions)
                .ok_or(WorkingMemoryError::Overflow)?
                .min(targets.inference.input_positions);
            if end != targets.inference.input_positions {
                return Err(CaptureRunHostError::Coordinate);
            }
            let rows = if targets.inference.output == eredu_core::OutputDemand::Sequence {
                end - targets.next * targets.inference.prefill_chunk_positions
            } else {
                1
            };
            return Ok(Some(
                usize::try_from(rows).map_err(|_| WorkingMemoryError::Overflow)?,
            ));
        }
        Ok(None)
    }

    /// Spend exactly one original candidate claim after terminal span validation.
    pub fn take_candidates(
        &mut self,
        index: usize,
    ) -> Result<CaptureCandidateClaim<'a, '_>, CaptureRunHostError> {
        let rows = self.terminal_claim_rows(index)?;
        let mut geometry = CaptureCandidateGeometry::prepare(
            self.claim.source.admission(),
            index,
            self.claim.phase,
            self.claim.prediction,
            self.claim.invocation,
        )
        .map_err(CaptureStepError::from)?;
        if let Some(rows) = rows {
            geometry = geometry
                .terminal_readout(rows)
                .map_err(CaptureStepError::from)?;
        }
        let plan = CaptureCandidateHostPlan::prepare(geometry)?;
        self.claim.row[index + 1] = ClaimState::Spent;
        Ok(CaptureCandidateClaim {
            plan,
            identity: claims::ReceiptIdentity {
                phase: self.claim.phase,
                prediction: self.claim.prediction,
                index,
                custody: self.claim.custody.share_scheduled(),
            },
            exclusive: PhantomData,
        })
    }
    /// Validate origin before any mutation, then move the actual Vec into its frame.
    pub fn record_candidates(
        &mut self,
        receipt: ClaimedCaptureCandidates,
        dtype: TensorDtype,
        usage: CaptureUsage,
    ) -> Result<(), CaptureRunHostError> {
        if !self.claim.custody.same_schedule(&receipt.identity.custody)
            || self.claim.phase != receipt.identity.phase
            || self.claim.prediction != receipt.identity.prediction
        {
            return Err(CaptureRunHostError::ReceiptMismatch);
        }
        let index = receipt.identity.index;
        self.frame
            .record_candidates(index, dtype, receipt.shape, receipt.candidates, usage)?;
        if let Some(slot) = self
            .frame
            .prefill
            .as_mut()
            .and_then(|p| p.slots.get_mut(index))
        {
            slot.state = capture_tensor::prefill::TargetState::Recorded;
        }
        Ok(())
    }
}

/// Exclusive source/scope coupling for the fixed candidate host transfer.
pub struct ScheduledCaptureCandidatesTransfer<'a, 'c, 's, K: Ord + Send + 'static> {
    builder: ScheduledCaptureCandidates<'a, 'c>,
    source: WorkingMemoryStorage<K>,
    _segment: Option<&'s mut crate::working_memory::CaptureSourceSegment>,
    native: &'s mut WorkingMemoryFundingScope,
}
impl<K: Ord + Send + 'static> ScheduledCaptureCandidatesTransfer<'_, '_, '_, K> {
    /// Validate both original H and complete registered sources on the same scope.
    pub fn validate(&self) -> Result<(), WorkingMemoryError> {
        self.builder
            .identity
            .custody
            .validate_transfer(self.native, &self.source)
    }
    /// Fill one settled scalar pair; no native completion is inferred here.
    pub fn push(&mut self, id: u32, score: f32, allowed: bool) -> Result<(), WorkingMemoryError> {
        if let Err(error) = self.validate() {
            self.builder.failure = Some(error.clone());
            return Err(error);
        }
        self.builder.push(id, score, allowed)
    }
    /// Preserve partial payload and original H if final source/account validation fails.
    pub fn finish(self) -> Result<ClaimedCaptureCandidates, CaptureCandidateFailure> {
        if let Err(error) = self.validate() {
            return Err(CaptureCandidateFailure {
                error,
                candidates: CaptureCandidates {
                    stage: CandidateScoreStage::RawLogitsBeforeSampling,
                    source: candidate_source(&self.builder.plan.geometry),
                    candidates: self.builder.values,
                    domain: self.builder.domain,
                },
                custody: self.builder.identity.custody,
            });
        }
        self.builder.finish()
    }
}

impl<'a> ScheduledCaptureStep<'a> {
    pub(crate) fn take_remote_prefill_candidates<'c>(&'c mut self,index:usize)->Result<CaptureCandidateClaim<'a,'c>,CaptureRunHostError> {
        self.validate_remote_prefill_claim(index)?;
        let rows=self.partition_terminal_rows(index)?.ok_or(CapturePrefillHostError::Identity)?;
        let geometry=CaptureCandidateGeometry::prepare(self.claim.source.admission(),index,self.claim.phase,self.claim.prediction,None)
            .and_then(|value|value.terminal_readout(rows)).map_err(CaptureStepError::from)?;
        let plan=CaptureCandidateHostPlan::prepare(geometry)?;
        self.begin_remote_terminal_claim(index)?;
        Ok(CaptureCandidateClaim {plan,identity:claims::ReceiptIdentity {phase:self.claim.phase,prediction:self.claim.prediction,
            index,custody:self.claim.custody.share_scheduled()},exclusive:PhantomData})
    }
    pub(crate) fn record_remote_prefill_candidates(&mut self,value:ClaimedCaptureCandidates,dtype:TensorDtype)->Result<(),CaptureRunHostError> {
        let index=value.identity.index;self.validate_remote_prefill_record(index,&dtype)?;
        self.record_candidates(value,dtype,CaptureUsage::default())?;
        self.finish_remote_terminal_record(index)
    }
}

impl<'a,'c> ScheduledCaptureCandidates<'a,'c> {
    pub(super) fn partition_source(&self)->(&'a AdmittedCapturePlan,usize,CapturePhase,u64,[usize;3]) {
        let geometry=&self.plan.geometry;
        (geometry.admission(),geometry.selection_index(),geometry.phase(),geometry.prediction(),*geometry.source_shape())
    }
    pub(super) fn partition_count(&self)->usize {self.plan.geometry.count()}
    pub(super) fn partition_failure(self)->CaptureCandidateFailure {CaptureCandidateFailure {error:WorkingMemoryError::IdentityMismatch,candidates:CaptureCandidates {
            stage:CandidateScoreStage::RawLogitsBeforeSampling,source:candidate_source(&self.plan.geometry),candidates:self.values,domain:self.domain},custody:self.identity.custody}}
    fn finish_partition(mut self,domain:Option<CandidateDomain>,_log_partition:f64)->Result<ClaimedCaptureCandidates,CaptureCandidateFailure> {
        self.domain=domain;self.finish()
    }
}
impl ClaimedCaptureCandidates {
    fn partition_failure(self)->CaptureCandidateFailure {CaptureCandidateFailure {error:WorkingMemoryError::IdentityMismatch,candidates:self.candidates,custody:self.identity.custody}}
}
impl<'a,'c> CaptureCandidateClaim<'a,'c> {
    /// Fill the original fixed slots from one authenticated complete-producer receipt.
    /// Invalid receipts retain their initialized prefix and the spent original account.
    pub fn decode_partition_receipt(self,bytes:&[u8],expected:PartitionCaptureTensorReceipt<'_>,
        funding:&eredu_nn::workspace::HostMetadataFunding)->Result<ClaimedCaptureCandidates,PartitionCaptureTensorDecodeError> {
        let custody=self.identity.custody.share_scheduled();claims::prepare_vocabulary_decoder(&custody,funding)?;
        let mut destination=self.prepare(None).map_err(|cause|claims::histogram_preparation_failure(cause,&custody,funding))?;
        let (source,index,_,_,shape)=destination.partition_source();
        let metadata=claims::decode_vocabulary_receipt(claims::VocabularyDestination::Candidates(&mut destination),bytes,expected,&custody,funding);
        let (domain,log_partition)=match metadata {Ok(value)=>value,Err(error)=>return Err(error.retaining_candidates(destination.partition_failure()))};
        let value=destination.finish_partition(domain,log_partition)
            .map_err(|cause|PartitionCaptureTensorDecodeError::candidate_failure(cause,custody.share_scheduled(),funding))?;
        let shape=shape.map(|n|n as u64);
        if !crate::capture::partition::vocabulary_payload_valid(&source.plan().selections[index],source.points()[index].position,
            &shape,crate::capture::partition::VocabularyPayload::Candidates(value.observation())) {
            return Err(PartitionCaptureTensorDecodeError::candidate_failure(value.partition_failure(),custody,funding));
        }
        Ok(value)
    }
}
