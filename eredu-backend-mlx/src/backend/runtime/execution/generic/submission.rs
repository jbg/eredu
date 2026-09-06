//! Per-unit ownership spanning eager execution and event publication.

use super::*;
use crate::backend::{
    ordinary_retirement::OrdinaryRetirement,
    submission_recovery::{Probe, Recovery, Retention, Status},
};
use std::cell::Cell;

struct UnitResources<U> {
    unit: MlxModule<U>,
    _transfer: MlxUnitTransfer,
}

struct UnitRetention<U: 'static> {
    resources: OrdinaryRetirement<UnitResources<U>>,
    event: Option<Event>,
    failed: Cell<bool>,
}

impl<U: 'static> Retention for UnitRetention<U> {
    fn observe(&self, status: Status) {
        if status.failed || status.blocked {
            self.failed.set(true);
        }
    }
}

/// One populated MLX unit retained through its exact consumer submission.
///
/// Dropping an active or pending unit never waits. Its native scope retains
/// the unit and residency ownership until terminal evidence; final module and
/// manager-lease destruction runs only at an ordinary unlocked host entry.
pub struct MlxUnitLease<U: 'static> {
    recovery: Recovery<UnitRetention<U>>,
}

impl<U: 'static> MlxUnitLease<U> {
    pub(super) fn new(unit: MlxModule<U>, transfer: MlxUnitTransfer) -> Result<Self, Error> {
        Ok(Self {
            recovery: Recovery::begin(UnitRetention {
                resources: OrdinaryRetirement::new(UnitResources {
                    unit,
                    _transfer: transfer,
                }),
                event: None,
                failed: Cell::new(false),
            })?,
        })
    }

    pub(super) fn submitted(&mut self, event: Event) -> Result<(), Error> {
        self.recovery.retention_mut().event = Some(event);
        self.recovery.seal();
        self.check_status().map(|_| ())
    }

    pub(super) fn population_parts(&mut self) -> (&mut MlxModule<U>, &ResidentUnitLease) {
        let resources = &mut *self.recovery.retention_mut().resources;
        let lease = match &resources._transfer {
            MlxUnitTransfer::Ordinary { _transfer } => &_transfer.leases()[0],
            MlxUnitTransfer::Dense { _transfer } => _transfer.lease(),
        };
        (&mut resources.unit, lease)
    }

    fn check_status(&self) -> Result<bool, Error> {
        unit_status(&self.recovery)
    }

    pub(super) fn is_complete(&self) -> Result<bool, Error> {
        unit_is_complete(&self.recovery)
    }

    pub(super) fn wait(&self) -> Result<(), Error> {
        wait_for_unit(&self.recovery)
    }
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
        &self.recovery.retention().resources.unit.inner
    }
}

impl<U: 'static> std::ops::DerefMut for MlxUnitLease<U> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.recovery.retention_mut().resources.unit.inner
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
                resources: OrdinaryRetirement::new(resources(&drops)),
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
        let event = async_eval_with_event([&array]).unwrap();
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
