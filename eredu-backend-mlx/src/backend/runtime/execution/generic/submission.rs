//! Per-unit ownership spanning eager execution and event publication.

use super::*;
use crate::backend::submission_recovery::observed::{
    operation::OperationRecovery, CompletedObservedRetention, FinishRetainingError, Observation,
    ObservedRecovery,
};
use crate::backend::{
    ordinary_retirement::OrdinaryRetirement,
    submission_recovery::{Probe, Recovery, Retention, Status},
};
use eredu_runtime::working_memory::OriginalOperationMetadataCustody;
use std::cell::Cell;
type UnitRecovery<U> = OperationRecovery<UnitRetention<U>, OriginalOperationMetadataCustody>;

struct UnitResources<U> {
    unit: MlxModule<U>,
    _transfer: MlxUnitTransfer,
}

// The independently prepared payload node may outlive the native recovery node.
// Its original custody therefore follows its actual module/manager resources.
struct UnitResourcesSlot<U> {
    value: Option<UnitResources<U>>,
    // The same native recovery node retires only after actual resource Drop.
    cleanup: Option<crate::backend::submission_recovery::observed::OriginalRetirementCleanup>,
    _custody: Option<eredu_runtime::working_memory::OriginalOperationMetadataCustody>,
}

struct UnitRetention<U: 'static> {
    resources: Option<OrdinaryRetirement<UnitResourcesSlot<U>>>,
    event: Option<OperationEvent>,
    failed: Cell<bool>,
}

impl<U: 'static> UnitRetention<U> {
    fn resources(&self) -> &UnitResources<U> {
        self.resources
            .as_ref()
            .expect("unit payload node")
            .value
            .as_ref()
            .expect("populated unit payload")
    }
    fn resources_mut(&mut self) -> &mut UnitResources<U> {
        self.resources
            .as_mut()
            .expect("unit payload node")
            .value
            .as_mut()
            .expect("populated unit payload")
    }
}

mod prepared;
pub(crate) use prepared::PreparedUnit;

impl<U: 'static> Retention for UnitRetention<U> {
    fn observe(&self, status: Status) {
        if status.failed || status.blocked {
            self.failed.set(true);
        }
    }

    fn retire_original(
        mut self,
        cleanup: crate::backend::submission_recovery::observed::OriginalRetirementCleanup,
    ) {
        if let Some(mut resources) = self.resources.take() {
            // Terminal native observation permits staging this exact payload.
            // Its existing retirement Box still postpones manager/module Drop
            // until an unlocked host boundary. Keep the SAME cleanup node in
            // that payload through destruction and any native wrapper deferral.
            debug_assert!(resources.cleanup.is_none());
            resources.cleanup = Some(cleanup);
            drop(resources);
        } else {
            drop(self);
            drop(cleanup);
        }
    }
}

/// One populated MLX unit retained through its exact consumer submission.
///
/// Dropping an active or pending unit never waits. Its native scope retains
/// the unit and residency ownership until terminal evidence; final module and
/// manager-lease destruction runs only at an ordinary unlocked host entry.
pub struct MlxUnitLease<U: 'static> {
    recovery: UnitRecovery<U>,
}

/// A completed source loan whose module and residency pins remain live while
/// a separately admitted consumer reads them outside the source scope.
pub(super) struct CompletedUnitLease<U: 'static> {
    retained: CompletedObservedRetention<UnitRetention<U>, OriginalOperationMetadataCustody>,
}

impl<U: 'static> CompletedUnitLease<U> {
    pub(super) fn retained_leases(&self) -> &[ResidentUnitLease] {
        match &self.retained.retention().resources()._transfer {
            MlxUnitTransfer::Ordinary { _transfer } => _transfer.leases(),
            MlxUnitTransfer::Dense { _transfer } => std::slice::from_ref(_transfer.lease()),
        }
    }
    pub(super) fn unit_mut(&mut self) -> &mut U {
        &mut self.retained.retention_mut().resources_mut().unit.inner
    }

    pub(super) fn retire(self) -> Result<(), Error> {
        self.retained
            .release_with(release_unit_payload::<U> as fn(&mut UnitRetention<U>))
            .map(|_| ())
            .map_err(|error| {
                let cause = finish_error(error.cause);
                drop(error.callback);
                drop(error.pending);
                cause
            })
    }
}

impl<U: 'static> MlxUnitLease<U> {
    pub(super) fn new(unit: MlxModule<U>, transfer: MlxUnitTransfer) -> Result<Self, Error> {
        Ok(Self {
            recovery: OperationRecovery::ordinary(Recovery::begin(UnitRetention {
                resources: Some(OrdinaryRetirement::new(UnitResourcesSlot {
                    value: Some(UnitResources {
                        unit,
                        _transfer: transfer,
                    }),
                    cleanup: None,
                    _custody: None,
                })),
                event: None,
                failed: Cell::new(false),
            })?),
        })
    }

    /// The selected request bank supplies both final nodes and an observer
    /// authenticated before unit construction. This constructor only moves.
    pub(super) fn from_prepared(
        prepared: PreparedUnit<U>,
        unit: MlxModule<U>,
        transfer: MlxUnitTransfer,
        observer: safemlx::OriginalScopeObserver,
    ) -> Self {
        Self {
            recovery: OperationRecovery::Original(prepared.activate(unit, transfer, observer)),
        }
    }

    /// Borrows the retained original role for an explicit native producer.
    /// An escaped lease cannot select an ordinary producer from current TLS.
    pub(super) fn original_observer(&self) -> Option<&safemlx::OriginalScopeObserver> {
        self.recovery.original_observer()
    }

    pub(super) fn submitted(&mut self, event: OperationEvent) -> Result<(), Error> {
        self.recovery.retention_mut().event = Some(event);
        self.recovery.seal();
        self.check_status().map(|_| ())
    }

    pub(super) fn population_parts(&mut self) -> (&mut MlxModule<U>, &ResidentUnitLease) {
        let resources = self.recovery.retention_mut().resources_mut();
        let lease = match &resources._transfer {
            MlxUnitTransfer::Ordinary { _transfer } => &_transfer.leases()[0],
            MlxUnitTransfer::Dense { _transfer } => _transfer.lease(),
        };
        (&mut resources.unit, lease)
    }

    fn check_status(&self) -> Result<bool, Error> {
        match &self.recovery {
            OperationRecovery::Ordinary(value) => unit_status(value),
            OperationRecovery::Original(value) => original_unit_status(value, value.progress()?),
        }
    }

    pub(super) fn is_complete(&self) -> Result<bool, Error> {
        match &self.recovery {
            OperationRecovery::Ordinary(value) => unit_is_complete(value),
            OperationRecovery::Original(value) => safemlx::try_with_submission_retirement(|| {
                let observed = value.progress()?;
                if observed.outcome == safemlx::ScopedSubmissionProgress::Busy {
                    return Ok(false);
                }
                if !original_unit_status(value, observed)? {
                    return Ok(false);
                }
                value
                    .retention()
                    .event
                    .as_ref()
                    .expect("submitted original unit retains its event")
                    .is_complete()
                    .map_err(Into::into)
            })
            .unwrap_or(Ok(false)),
        }
    }

    pub(super) fn wait(&self) -> Result<(), Error> {
        match &self.recovery {
            OperationRecovery::Ordinary(value) => wait_for_unit(value),
            OperationRecovery::Original(value) => {
                // Only observed pending native work is retried by wait. Fixed
                // Busy/funded/unknown outcomes return with original custody held.
                if !original_unit_status(value, value.wait()?)? {
                    return Err(Error::PrefillScopeUnavailable);
                }
                value
                    .retention()
                    .event
                    .as_ref()
                    .expect("submitted original unit retains its event")
                    .synchronize()
                    .map_err(Into::into)
            }
        }
    }

    /// Complete the actual borrowed unit under its retained original observer.
    /// Failure leaves its existing recovery/payload node armed. A successful
    /// callback still passes the common terminal-observation and unlocked
    /// retirement path; no polling failure is interpreted as completion.
    pub(super) fn complete_original(
        self,
        complete: impl FnOnce(&U, &safemlx::OriginalScopeObserver) -> Result<(), Error>,
    ) -> Result<(), Error> {
        self.complete_original_retaining(complete)?.retire()
    }

    /// Certify this source's actual transfer and module completion, retaining
    /// their existing payload and cleanup nodes through a later consumer.
    pub(super) fn complete_original_retaining(
        mut self,
        complete: impl FnOnce(&U, &safemlx::OriginalScopeObserver) -> Result<(), Error>,
    ) -> Result<CompletedUnitLease<U>, Error> {
        let observer = self
            .recovery
            .original_observer()
            .ok_or(Error::PrefillScopeUnavailable)?;
        complete(&self.recovery.retention().resources().unit.inner, observer)?;
        self.recovery.seal();
        self.synchronize_transfer()?;
        match self.recovery {
            OperationRecovery::Original(value) => Ok(CompletedUnitLease {
                retained: value.finish_retaining().map_err(finish_error)?,
            }),
            OperationRecovery::Ordinary(_) => Err(Error::PrefillScopeUnavailable),
        }
    }

    pub(super) fn finish(self) -> Result<(), Error> {
        self.wait()?;
        self.finish_retired()
    }

    fn synchronize_transfer(&mut self) -> Result<(), Error> {
        match &mut self.recovery.retention_mut().resources_mut()._transfer {
            MlxUnitTransfer::Ordinary { _transfer } => _transfer.synchronize()?,
            MlxUnitTransfer::Dense { _transfer } => _transfer.synchronize()?,
        }
        Ok(())
    }

    fn finish_retired(mut self) -> Result<(), Error> {
        self.synchronize_transfer()?;
        match self.recovery {
            OperationRecovery::Ordinary(value) => {
                let status = value.finish()?;
                if status.failed || status.blocked {
                    return Err(Error::ArchitectureModel(
                        "native execution-unit retirement failed".into(),
                    ));
                }
                Ok(())
            }
            OperationRecovery::Original(value) => CompletedUnitLease {
                retained: value.finish_retaining().map_err(finish_error)?,
            }
            .retire(),
        }
    }
}

fn release_unit_payload<U: 'static>(retention: &mut UnitRetention<U>) {
    // release_with established an unlocked host boundary. Consume this exact
    // final payload node now; no global queue pass or replacement allocation.
    if let Some(resources) = retention.resources.take() {
        let mut slot = resources.into_inner();
        if let Some(UnitResources { unit, _transfer }) = slot.value.take() {
            // Native completion was established before release_with. Drop the
            // module first, then retire this transfer's exact application pins;
            // a global ordinary queue sweep would also touch unrelated owners.
            drop(unit);
            match _transfer {
                MlxUnitTransfer::Ordinary { _transfer } => {
                    _transfer.retire_completed_original();
                }
                MlxUnitTransfer::Dense { _transfer } => {
                    _transfer.retire_completed_original();
                }
            }
        }
        drop(slot);
    }
}

fn finish_error(cause: FinishRetainingError<safemlx::error::Exception>) -> Error {
    match cause {
        FinishRetainingError::Native(cause) => cause.into(),
        FinishRetainingError::Retirement(cause) => cause.into_error(),
        // Native observers translate fixed/retained causes before consumption.
        // This fallback fences an unavailable observation without new strings.
        FinishRetainingError::Observation(_) => Error::PrefillScopeUnavailable,
    }
}

fn original_unit_status<U: 'static>(
    recovery: &ObservedRecovery<UnitRetention<U>, OriginalOperationMetadataCustody>,
    observed: Observation,
) -> Result<bool, Error> {
    if let Some(cause) = recovery.observer().observation_error(observed.outcome) {
        return Err(cause.into());
    }
    if recovery.retention().failed.get() || observed.status.failed || observed.status.blocked {
        if let Some(cause) = recovery.observer().retained_failure().or_else(|| {
            recovery
                .observer()
                .observation_error(safemlx::ScopedSubmissionProgress::Unobservable)
        }) {
            return Err(cause.into());
        }
        return Err(Error::PrefillScopeUnavailable);
    }
    Ok(observed.status.settled)
}

fn unit_status<U: 'static, P: Probe>(
    recovery: &Recovery<UnitRetention<U>, P>,
) -> Result<bool, Error> {
    let status = recovery.progress();
    if recovery.retention().failed.get() || status.failed || status.blocked {
        return Err(Error::ArchitectureModel(
            "native execution-unit submission failed; unresolved ownership is retained".into(),
        ));
    }
    Ok(status.settled)
}

fn unit_is_complete<U: 'static, P: Probe>(
    recovery: &Recovery<UnitRetention<U>, P>,
) -> Result<bool, Error> {
    safemlx::try_with_submission_retirement(|| {
        if !unit_status(recovery)? {
            return Ok(false);
        }
        recovery
            .retention()
            .event
            .as_ref()
            .expect("submitted execution unit retains its event")
            .is_complete()
            .map_err(Into::into)
    })
    .unwrap_or(Ok(false))
}

fn wait_for_unit<U: 'static, P: Probe>(
    recovery: &Recovery<UnitRetention<U>, P>,
) -> Result<(), Error> {
    // An event marker can precede release of a worker's callback captures.
    // Observe the whole scope before waiting for the output event: a failed
    // side submission must not leave us blocked on unrelated output work.
    while !unit_is_complete(recovery)? {
        std::thread::yield_now();
    }
    Ok(())
}

impl<U: 'static> std::ops::Deref for MlxUnitLease<U> {
    type Target = U;

    fn deref(&self) -> &Self::Target {
        &self.recovery.retention().resources().unit.inner
    }
}

impl<U: 'static> std::ops::DerefMut for MlxUnitLease<U> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.recovery.retention_mut().resources_mut().unit.inner
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{ordinary_retirement, submission_recovery};
    use std::{
        sync::{
            atomic::{AtomicBool, AtomicUsize, Ordering},
            mpsc, Arc,
        },
        time::{Duration, Instant},
    };

    struct DropWitness(Arc<AtomicUsize>);
    impl Drop for DropWitness {
        fn drop(&mut self) {
            assert!(safemlx::can_reclaim_submission_resources());
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    fn resources(drops: &Arc<AtomicUsize>) -> UnitResources<DropWitness> {
        UnitResources {
            unit: MlxModule::new(DropWitness(Arc::clone(drops))),
            _transfer: MlxUnitTransfer::Ordinary {
                _transfer: ResidentTransfer::immediate(Vec::new(), MemoryTier::Device),
            },
        }
    }

    struct FailedUntil(Arc<AtomicBool>);
    impl submission_recovery::Probe for FailedUntil {
        fn seal(&mut self) {}
        fn progress(&self) -> Status {
            Status {
                settled: self.0.load(Ordering::SeqCst),
                failed: true,
                blocked: true,
            }
        }
    }

    #[test]
    fn failed_unit_retains_ownership_until_terminal_then_defers_its_destructor() {
        let drops = Arc::new(AtomicUsize::new(0));
        let terminal = Arc::new(AtomicBool::new(false));
        let recovery = Recovery::with_probe(
            UnitRetention {
                resources: Some(OrdinaryRetirement::new(UnitResourcesSlot {
                    value: Some(resources(&drops)),
                    cleanup: None,
                    _custody: None,
                })),
                event: None,
                failed: Cell::new(false),
            },
            FailedUntil(Arc::clone(&terminal)),
        );
        let status = recovery.progress();
        assert!(!status.settled);
        assert!(status.failed);
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            if safemlx::try_with_submission_retirement(|| {
                let start = Instant::now();
                // No event was published. The failed side submission must be
                // reported before touching the absent/pending output event.
                assert!(wait_for_unit(&recovery).is_err());
                assert!(start.elapsed() < Duration::from_secs(1));
            })
            .is_some()
            {
                break;
            }
            assert!(Instant::now() < deadline);
            std::thread::yield_now();
        }
        drop(recovery);
        submission_recovery::reap();
        ordinary_retirement::reclaim();
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        terminal.store(true, Ordering::SeqCst);
        submission_recovery::reap();
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        reclaim_until_dropped(&drops);
    }

    #[test]
    fn unit_poll_and_drop_do_not_wait_for_foreign_runtime_ownership() {
        let drops = Arc::new(AtomicUsize::new(0));
        let UnitResources { unit, _transfer } = resources(&drops);
        let mut lease = MlxUnitLease::new(unit, _transfer).unwrap();
        let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
        let array = safemlx::Array::from_slice(&[1.0f32], &[1])
            .square(&stream)
            .unwrap();
        let event = async_eval_with_operation_event([&array]).unwrap();
        lease.submitted(event).unwrap();
        let (ready_tx, ready_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let holder = std::thread::spawn(move || loop {
            if safemlx::try_with_submission_retirement(|| {
                ready_tx.send(()).unwrap();
                let _ = release_rx.recv_timeout(Duration::from_secs(5));
            })
            .is_some()
            {
                break;
            }
            std::thread::yield_now();
        });
        ready_rx.recv_timeout(Duration::from_secs(10)).unwrap();
        let start = Instant::now();
        assert!(!lease.is_complete().unwrap());
        drop(lease);
        submission_recovery::reap();
        assert!(start.elapsed() < Duration::from_secs(1));
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        release_tx.send(()).unwrap();
        holder.join().unwrap();
        submission_recovery::reap();
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        reclaim_until_dropped(&drops);
    }

    fn reclaim_until_dropped(drops: &AtomicUsize) {
        let deadline = Instant::now() + Duration::from_secs(10);
        while drops.load(Ordering::SeqCst) == 0 {
            submission_recovery::reap();
            ordinary_retirement::reclaim();
            assert!(Instant::now() < deadline, "terminal unit was not reclaimed");
            std::thread::yield_now();
        }
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }
}
