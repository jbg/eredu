//! Per-unit ownership spanning eager execution and event publication.

use super::*;
use crate::backend::{
    ordinary_retirement::OrdinaryRetirement,
    submission_recovery::{Recovery, Retention, Status},
};
use std::cell::Cell;

struct UnitResources<U> {
    unit: MlxModule<U>,
    transfer: ResidentTransfer,
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
    pub(super) fn new(unit: MlxModule<U>, transfer: ResidentTransfer) -> Result<Self, Error> {
        Ok(Self {
            recovery: Recovery::begin(UnitRetention {
                resources: OrdinaryRetirement::new(UnitResources { unit, transfer }),
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
        (&mut resources.unit, &resources.transfer.leases()[0])
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

    pub(super) fn finish(mut self) -> Result<(), Error> {
        self.wait()?;
        self.recovery
            .retention_mut()
            .resources
            .transfer
            .synchronize()?;
        let status = self.recovery.finish();
        if status.failed || status.blocked {
            return Err(Error::ArchitectureModel(
                "native execution-unit retirement failed".into(),
            ));
        }
        Ok(())
    }
}

fn unit_status<U: 'static>(recovery: &Recovery<UnitRetention<U>>) -> Result<bool, Error> {
    let status = recovery.progress();
    if recovery.retention().failed.get() || status.failed || status.blocked {
        return Err(Error::ArchitectureModel(
            "native execution-unit submission failed; unresolved ownership is retained".into(),
        ));
    }
    Ok(status.settled)
}

fn unit_is_complete<U: 'static>(recovery: &Recovery<UnitRetention<U>>) -> Result<bool, Error> {
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

fn wait_for_unit<U: 'static>(recovery: &Recovery<UnitRetention<U>>) -> Result<(), Error> {
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
