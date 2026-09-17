//! Fixed sampler-input observation through the existing funded transaction.
use super::*;

/// One actual numerical phase owns one capture bank. The physical phase ordinal
/// is distinct from logical prediction/target/draft attribution, which remains
/// the shared speculative driver's responsibility. Repeated coordinates use new
/// numerical phases while retaining the request's shared cumulative ledger.
#[derive(Debug)]
pub struct FundedSpeculativeCaptureInvocation {
    session: FundedCaptureSession,
    prediction: u64,
    ordinal: usize,
    epoch: DistributedCommitEpoch,
}

pub(crate) fn construct(
    run: PreparedCaptureRun,
    lineage: crate::working_memory::CaptureRunLedger,
    ordinal: usize,
) -> Result<FundedSpeculativeCaptureInvocation, CaptureRunHostError> {
    let epoch = u64::try_from(ordinal)
        .ok()
        .and_then(|n| n.checked_add(1))
        .and_then(DistributedCommitEpoch::new)
        .ok_or(WorkingMemoryError::Overflow)?;
    let prediction = run.first_prediction();
    let mut session = FundedCaptureSession::from_run_with_lineage(run, lineage);
    session.untracked = true;
    Ok(FundedSpeculativeCaptureInvocation {
        session,
        prediction,
        ordinal,
        epoch,
    })
}
impl FundedSpeculativeCaptureInvocation {
    /// Retained declaration source; this borrow issues no C/native authority.
    pub fn source(&self) -> &SharedCapturePlan {
        self.session.source()
    }
    /// Logical capture position. Rollback/replay may repeat this value.
    pub const fn prediction(&self) -> u64 {
        self.prediction
    }
    /// Already claimed numerical occurrence. No independent counter is created.
    pub const fn ordinal(&self) -> usize {
        self.ordinal
    }
    /// Run the same observation/claim/terminal worker as ordinary funded capture.
    /// The backend must authenticate the source and this actual phase before
    /// reading it, and settle all native work before allowing delivery to escape.
    /// Errors leave any aborted frame in this owner; they never refund spending.
    /// The source has the admitted one-row model.logits shape, including any
    /// reshape already priced and performed by the numerical producer.
    pub fn observe<T, E>(
        &mut self,
        backend: &mut dyn ScheduledCaptureBackend<Tensor = T, Error = E>,
        value: &T,
    ) -> Result<(), FundedCaptureError<E>>
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        self.run(backend,value,false).map(|_|())
    }
    /// Observe the original source, then apply the same ordered intervention
    /// hook before sealing this frame. The replacement remains provisional until
    /// the enclosing numerical owner completes its full root union.
    pub fn observe_and_intervene<T,E>(
        &mut self,backend:&mut dyn ScheduledCaptureBackend<Tensor=T,Error=E>,value:&T,
    )->Result<Option<T>,FundedCaptureError<E>>
    where E:std::error::Error+Send+Sync+'static {
        self.run(backend,value,true)
    }
    fn run<T,E>(
        &mut self,backend:&mut dyn ScheduledCaptureBackend<Tensor=T,Error=E>,value:&T,apply:bool,
    )->Result<Option<T>,FundedCaptureError<E>>
    where E:std::error::Error+Send+Sync+'static {
        let epoch = self.epoch;
        let pass = if self.prediction == 0 {
            crate::ExpertPass::Prefill
        } else {
            crate::ExpertPass::Decode
        };
        self.session
            .with_observer(backend, self.prediction, &|cause| cause, |observer| {
                let guard = crate::inspection::ObservationTransactionGuard::new(observer, epoch);
                guard.observer.prepare_transaction(epoch, pass)?;
                guard
                    .observer
                    .observe(eredu_core::MODEL_LOGITS_OBSERVATION_PATH, value)?;
                let effective=if apply {
                    guard.observer.intervene(eredu_core::MODEL_LOGITS_OBSERVATION_PATH,value)?
                } else {None};
                guard.observer.complete_transaction(epoch)?;
                guard.finish(true);
                Ok(effective)
            })
            .map_err(FundedCaptureError::Protocol)?
    }
    /// Move the paid shared frame after exact native completion/recovery. Its
    /// Untracked outcome is the ordinary speculative logical disposition, never
    /// an unowned payload or a conversion to the legacy Vec carrier.
    pub fn take_shared_step(
        &mut self,
    ) -> Result<Option<SharedCapturedStep>, FundedCaptureDrainError> {
        self.session.take_shared_step()
    }
    /// Move already sealed host evidence after a failed numerical call. This
    /// neither certifies native completion nor changes the record disposition.
    /// The native caller must retain all unresolved work in its existing recovery
    /// owner. A successful capture before a later policy error stays Untracked;
    /// an observation-transaction failure stays Aborted. No native value escapes.
    pub fn take_failed_evidence(&mut self)->Result<Option<SharedCapturedStep>,FundedCaptureDrainError> {
        self.session.take_failed_numerical_evidence()
    }
    /// Logical usage after the last attempted observation, including failures.
    pub fn usage(&self) -> CaptureUsage {
        self.session.usage()
    }
}

pub(crate) fn control_bytes() -> Result<u64, WorkingMemoryError> {
    [
        size_of::<FundedSpeculativeCaptureInvocation>(),
        size_of::<DistributedCommitEpoch>(),
        size_of::<usize>(),
        size_of::<u64>(),
        size_of::<crate::ExpertPass>(),
        size_of::<Result<FundedSpeculativeCaptureInvocation, CaptureRunHostError>>(),
        size_of::<Result<Option<SharedCapturedStep>, FundedCaptureDrainError>>(),
        size_of::<
            crate::inspection::ObservationTransactionGuard<
                '_,
                (),
                (),
                dyn crate::ActivationObserver<(), ()>,
            >,
        >(),
    ]
    .into_iter()
    .try_fold(0usize, usize::checked_add)
    .and_then(|bytes| u64::try_from(bytes).ok())
    .ok_or(WorkingMemoryError::Overflow)
}
