//! Ordered token-score destination under the existing original capture claim.
use super::*;

/// Fixed result slots and exact typed constructor/receipt/failure controls.
#[derive(Debug)]
pub struct CaptureTokenScoreHostPlan<'a> {
    geometry: CaptureTokenScoreGeometry<'a>,
    peak: u64,
}
impl<'a> CaptureTokenScoreHostPlan<'a> {
    /// Derive the destination from the same immutable selection, without allocation.
    pub fn prepare(geometry: CaptureTokenScoreGeometry<'a>) -> Result<Self, WorkingMemoryError> {
        let slots = geometry
            .count()
            .checked_mul(size_of::<CaptureTokenScore>())
            .filter(|n| *n <= isize::MAX as usize)
            .ok_or(WorkingMemoryError::Overflow)?;
        let controls = [
            size_of::<Self>(),
            size_of::<CaptureTokenScores>(),
            size_of::<CaptureTokenScoreClaim<'_, '_>>(),
            size_of::<ScheduledCaptureTokenScores<'_, '_>>(),
            size_of::<ClaimedCaptureTokenScores>(),
            size_of::<CaptureTokenScoreFailure>(),
            size_of::<Box<CaptureTokenScoreFailure>>(),
            size_of::<Box<CaptureRunHostError>>(),
            size_of::<ScheduledCaptureTokenScoresTransfer<'_, '_, '_, u8>>(),
            size_of::<Result<ClaimedCaptureTokenScores, CaptureTokenScoreFailure>>(),
            size_of::<Result<(), WorkingMemoryError>>(),
            size_of::<Option<CandidateDomain>>(),
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)
        .and_then(|n| n.checked_mul(3))
        .ok_or(WorkingMemoryError::Overflow)?;
        let peak = slots
            .checked_add(controls)
            .and_then(|n| u64::try_from(n).ok())
            .ok_or(WorkingMemoryError::Overflow)?;
        Ok(Self { geometry, peak })
    }
    /// Complete requested result/control construction envelope; no allocation grant.
    pub fn initialization_peak_bytes(&self) -> u64 {
        self.peak
    }
    /// Original ordered selection and actual physical source geometry.
    pub fn geometry(&self) -> &CaptureTokenScoreGeometry<'a> {
        &self.geometry
    }
}

/// Exclusive one-use host claim, issued by its originating scheduled frame.
#[derive(Debug)]
pub struct CaptureTokenScoreClaim<'a, 'c> {
    plan: CaptureTokenScoreHostPlan<'a>,
    identity: claims::ReceiptIdentity,
    exclusive: PhantomData<&'c mut ()>,
}
impl<'a> CaptureTokenScoreClaim<'a, '_> {
    /// Borrow exact semantic and physical source facts.
    pub fn geometry(&self) -> &CaptureTokenScoreGeometry<'a> {
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
    /// Authenticate the original native scope without another reservation.
    pub fn validate_native_scope(
        &self,
        native: &WorkingMemoryFundingScope,
    ) -> Result<(), WorkingMemoryError> {
        self.identity.custody.validate_scheduled_native(native)
    }
}
impl<'a, 'c> CaptureTokenScoreClaim<'a, 'c> {
    /// Pin complete registered source origins before allocating the result buffer.
    pub fn prepare_with_source<'s, K: Clone + Ord + Send + Sync + 'static>(
        self,
        native: &'s mut WorkingMemoryFundingScope,
        source: WorkingMemoryStorage<K>,
        domain: Option<CandidateDomain>,
    ) -> Result<ScheduledCaptureTokenScoresTransfer<'a, 'c, 's, K>, CaptureRunHostError> {
        let rollback = self
            .identity
            .custody
            .bind_scheduled_source(native, &source)?;
        let builder = self.prepare(domain)?;
        let native = rollback.commit();
        Ok(ScheduledCaptureTokenScoresTransfer {
            builder,
            source,
            _segment: None,
            native,
        })
    }
    /// Use the same canonical terminal-prefill source channel and pin worker.
    pub fn prepare_with_segment_source<'s, K: Clone + Ord + Send + Sync + 'static>(
        self,
        native: &'s mut WorkingMemoryFundingScope,
        segment: &'s mut crate::working_memory::CaptureSourceSegment,
        source: WorkingMemoryStorage<K>,
        domain: Option<CandidateDomain>,
    ) -> Result<ScheduledCaptureTokenScoresTransfer<'a, 'c, 's, K>, CaptureRunHostError> {
        segment.validate_token_score_geometry(self.geometry())?;
        let rollback = self
            .identity
            .custody
            .bind_segment_source(native, segment, &source)?;
        let builder = self.prepare(domain)?;
        let native = rollback.commit();
        Ok(ScheduledCaptureTokenScoresTransfer {
            builder,
            source,
            _segment: Some(segment),
            native,
        })
    }
    /// Allocate the exact fixed slots only after the original account is validated.
    pub fn prepare(
        self,
        domain: Option<CandidateDomain>,
    ) -> Result<ScheduledCaptureTokenScores<'a, 'c>, CaptureRunHostError> {
        self.identity.custody.validate()?;
        let values = Vec::with_capacity(self.plan.geometry.count());
        Ok(ScheduledCaptureTokenScores {
            values,
            failure: None,
            plan: self.plan,
            domain,
            identity: self.identity,
            exclusive: self.exclusive,
        })
    }
}

/// Partial results and their allocation always precede the original host custody.
#[derive(Debug)]
pub struct ScheduledCaptureTokenScores<'a, 'c> {
    values: Vec<CaptureTokenScore>,
    failure: Option<WorkingMemoryError>,
    plan: CaptureTokenScoreHostPlan<'a>,
    domain: Option<CandidateDomain>,
    identity: claims::ReceiptIdentity,
    exclusive: PhantomData<&'c mut ()>,
}
impl ScheduledCaptureTokenScores<'_, '_> {
    /// Fill exactly the next requested ID. This neither infers native completion
    /// nor grows storage; invalid payload poisons the one-use destination.
    pub fn push(&mut self, value: CaptureTokenScore) -> Result<(), WorkingMemoryError> {
        if let Some(error) = &self.failure {
            return Err(error.clone());
        }
        if let Err(error) = self.identity.custody.validate() {
            self.failure = Some(error.clone());
            return Err(error);
        }
        let geometry = &self.plan.geometry;
        let vocabulary = geometry.vocabulary() as u64;
        let valid = geometry.token_ids().get(self.values.len()) == Some(&value.target.token_id)
            && value.target.score.is_finite()
            && value.log_probability.is_finite()
            && value.log_probability <= 0.0
            && value.rank > 0
            && value.rank <= vocabulary
            && match &value.strongest_alternative {
                None => vocabulary == 1 && value.rank == 1 && value.log_probability == 0.0,
                Some(other) => {
                    vocabulary > 1
                        && other.token_id != value.target.token_id
                        && u64::from(other.token_id) < vocabulary
                        && other.score.is_finite()
                        && ((other.score > value.target.score) == (value.rank > 1))
                }
            };
        if !valid {
            self.failure = Some(WorkingMemoryError::IdentityMismatch);
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        self.values.push(value);
        Ok(())
    }
    /// Number of initialized ordered slots.
    pub fn initialized_count(&self) -> usize {
        self.values.len()
    }
    /// Move the complete result after checking its full-distribution partition.
    /// Every refusal retains partial scores and their original account together.
    pub fn finish(
        self,
        log_partition: f64,
    ) -> Result<ClaimedCaptureTokenScores, CaptureTokenScoreFailure> {
        let error = self
            .failure
            .or_else(|| self.identity.custody.validate().err())
            .or_else(|| {
                (self.values.len() != self.plan.geometry.count()
                    || !log_partition.is_finite()
                    || self.values.iter().any(|score| {
                        let tolerance = 32.0
                            * f64::EPSILON
                            * log_partition
                                .abs()
                                .max(f64::from(score.target.score).abs())
                                .max(1.0);
                        ((f64::from(score.target.score) - log_partition) - score.log_probability)
                            .abs()
                            > tolerance
                            || score.strongest_alternative.as_ref().is_some_and(|other| {
                                log_partition + tolerance < f64::from(other.score)
                            })
                    }))
                .then_some(WorkingMemoryError::IdentityMismatch)
            });
        if let Some(error) = error {
            return Err(CaptureTokenScoreFailure {
                error,
                values: self.values,
                custody: self.identity.custody,
            });
        }
        let scores = CaptureTokenScores {
            stage: CandidateScoreStage::RawLogitsBeforeSampling,
            source: if self.plan.geometry.admission().points()[self.plan.geometry.selection_index()]
                .position
                == eredu_core::ObservationPosition::AfterIntervention
            {
                CandidateLogitsSource::Effective
            } else {
                CandidateLogitsSource::Original
            },
            vocabulary: self.plan.geometry.vocabulary() as u64,
            log_partition,
            scores: self.values,
            domain: self.domain,
        };
        Ok(ClaimedCaptureTokenScores {
            scores,
            shape: *self.plan.geometry.source_shape(),
            identity: self.identity,
        })
    }
}

/// Actual typed failure and initialized slots retain the paying host account last.
#[derive(Debug)]
pub struct CaptureTokenScoreFailure {
    error: WorkingMemoryError,
    values: Vec<CaptureTokenScore>,
    custody: CaptureTensorCustody,
}
impl fmt::Display for CaptureTokenScoreFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.error, f)
    }
}
impl std::error::Error for CaptureTokenScoreFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}
/// Only the originating frame can consume this result; no raw owning export.
#[derive(Debug)]
pub struct ClaimedCaptureTokenScores {
    scores: CaptureTokenScores,
    shape: [usize; 3],
    identity: claims::ReceiptIdentity,
}
impl ClaimedCaptureTokenScores {
    /// Borrow scores while retaining their complete original host owner.
    pub fn observation(&self) -> &CaptureTokenScores {
        &self.scores
    }
}
impl<'a> ScheduledCaptureStep<'a> {
    /// Spend one exact ordered-score claim on the same terminal-row schedule.
    pub fn take_token_scores(
        &mut self,
        index: usize,
    ) -> Result<CaptureTokenScoreClaim<'a, '_>, CaptureRunHostError> {
        let rows = self.terminal_claim_rows(index)?;
        let mut geometry = CaptureTokenScoreGeometry::prepare(
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
        let plan = CaptureTokenScoreHostPlan::prepare(geometry)?;
        self.claim.row[index + 1] = ClaimState::Spent;
        Ok(CaptureTokenScoreClaim {
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
    /// Authenticate origin before moving the initialized result into its record.
    pub fn record_token_scores(
        &mut self,
        receipt: ClaimedCaptureTokenScores,
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
            .record_token_scores(index, dtype, receipt.shape, receipt.scores, usage)?;
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

/// Same exclusive native source/account pin loan through result construction.
pub struct ScheduledCaptureTokenScoresTransfer<'a, 'c, 's, K: Ord + Send + 'static> {
    builder: ScheduledCaptureTokenScores<'a, 'c>,
    source: WorkingMemoryStorage<K>,
    _segment: Option<&'s mut crate::working_memory::CaptureSourceSegment>,
    native: &'s mut WorkingMemoryFundingScope,
}
impl<K: Ord + Send + 'static> ScheduledCaptureTokenScoresTransfer<'_, '_, '_, K> {
    /// Validate original host custody and complete pinned source coverage together.
    pub fn validate(&self) -> Result<(), WorkingMemoryError> {
        self.builder
            .identity
            .custody
            .validate_transfer(self.native, &self.source)
    }
    /// Fill one completed ordered score without any implicit native wait.
    pub fn push(&mut self, value: CaptureTokenScore) -> Result<(), WorkingMemoryError> {
        if let Err(error) = self.validate() {
            self.builder.failure = Some(error.clone());
            return Err(error);
        }
        self.builder.push(value)
    }
    /// Final source/account validation precedes moving the actual result buffer.
    pub fn finish(
        mut self,
        log_partition: f64,
    ) -> Result<ClaimedCaptureTokenScores, CaptureTokenScoreFailure> {
        if let Err(error) = self.validate() {
            self.builder.failure = Some(error);
        }
        self.builder.finish(log_partition)
    }
}

impl<'a> ScheduledCaptureStep<'a> {
    pub(crate) fn take_remote_prefill_token_scores<'c>(&'c mut self,index:usize)->Result<CaptureTokenScoreClaim<'a,'c>,CaptureRunHostError> {
        self.validate_remote_prefill_claim(index)?;
        let rows=self.partition_terminal_rows(index)?.ok_or(CapturePrefillHostError::Identity)?;
        let geometry=CaptureTokenScoreGeometry::prepare(self.claim.source.admission(),index,self.claim.phase,self.claim.prediction,None)
            .and_then(|value|value.terminal_readout(rows)).map_err(CaptureStepError::from)?;
        let plan=CaptureTokenScoreHostPlan::prepare(geometry)?;
        self.begin_remote_terminal_claim(index)?;
        Ok(CaptureTokenScoreClaim {plan,identity:claims::ReceiptIdentity {phase:self.claim.phase,prediction:self.claim.prediction,
            index,custody:self.claim.custody.share_scheduled()},exclusive:PhantomData})
    }
    pub(crate) fn record_remote_prefill_token_scores(&mut self,value:ClaimedCaptureTokenScores,dtype:TensorDtype)->Result<(),CaptureRunHostError> {
        let index=value.identity.index;self.validate_remote_prefill_record(index,&dtype)?;
        self.record_token_scores(value,dtype,CaptureUsage::default())?;
        self.finish_remote_terminal_record(index)
    }
}

impl<'a,'c> ScheduledCaptureTokenScores<'a,'c> {
    pub(super) fn partition_source(&self)->(&'a AdmittedCapturePlan,usize,CapturePhase,u64,[usize;3]) {
        let geometry=&self.plan.geometry;
        (geometry.admission(),geometry.selection_index(),geometry.phase(),geometry.prediction(),*geometry.source_shape())
    }
    pub(super) fn partition_count(&self)->usize {self.plan.geometry.count()}
    pub(super) fn partition_failure(self)->CaptureTokenScoreFailure {CaptureTokenScoreFailure {error:WorkingMemoryError::IdentityMismatch,values:self.values,custody:self.identity.custody}}
    fn finish_partition(mut self,domain:Option<CandidateDomain>,log_partition:f64)->Result<ClaimedCaptureTokenScores,CaptureTokenScoreFailure> {
        self.domain=domain;self.finish(log_partition)
    }
}
impl ClaimedCaptureTokenScores {
    fn partition_failure(self)->CaptureTokenScoreFailure {CaptureTokenScoreFailure {error:WorkingMemoryError::IdentityMismatch,values:self.scores.scores,custody:self.identity.custody}}
}
impl<'a,'c> CaptureTokenScoreClaim<'a,'c> {
    /// Fill the original fixed slots from one authenticated complete-producer receipt.
    /// Invalid receipts retain their initialized prefix and the spent original account.
    pub fn decode_partition_receipt(self,bytes:&[u8],expected:PartitionCaptureTensorReceipt<'_>,
        funding:&eredu_nn::workspace::WorkspaceMetadataFunding)->Result<ClaimedCaptureTokenScores,PartitionCaptureTensorDecodeError> {
        let custody=self.identity.custody.share_scheduled();claims::prepare_vocabulary_decoder(&custody,funding)?;
        let mut destination=self.prepare(None).map_err(|cause|claims::histogram_preparation_failure(cause,&custody,funding))?;
        let (source,index,_,_,shape)=destination.partition_source();
        let metadata=claims::decode_vocabulary_receipt(claims::VocabularyDestination::Scores(&mut destination),bytes,expected,&custody,funding);
        let (domain,log_partition)=match metadata {Ok(value)=>value,Err(error)=>return Err(error.retaining_scores(destination.partition_failure()))};
        let value=destination.finish_partition(domain,log_partition)
            .map_err(|cause|PartitionCaptureTensorDecodeError::score_failure(cause,custody.share_scheduled(),funding))?;
        let shape=shape.map(|n|n as u64);
        if !crate::capture::partition::vocabulary_payload_valid(&source.plan().selections[index],source.points()[index].position,
            &shape,crate::capture::partition::VocabularyPayload::Scores(value.observation())) {
            return Err(PartitionCaptureTensorDecodeError::score_failure(value.partition_failure(),custody,funding));
        }
        Ok(value)
    }
}
