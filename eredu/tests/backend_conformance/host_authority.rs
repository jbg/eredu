use super::MockError;
use eredu_core::{BackendFailure, HostPreparationAuthority};
use eredu_runtime::working_memory::WorkingMemoryPool;
use std::cell::RefCell;

#[derive(Default)]
pub(crate) struct Probe {
    pub pool: Option<WorkingMemoryPool>,
    pub attempts: usize,
    pub reject_at: Option<usize>,
    pub copies: Vec<&'static str>,
    pub fail_copy: Option<&'static str>,
    pub steps: Vec<eredu_core::backend::TextStepContext>,
}

thread_local! {
    static PROBE: RefCell<Option<Probe>> = const { RefCell::new(None) };
}

pub(crate) fn record_step(context: &eredu_core::backend::TextStepContext) {
    PROBE.with(|slot| {
        if let Some(probe) = slot.borrow_mut().as_mut() {
            probe.steps.push(context.clone());
        }
    });
}

pub(crate) fn acquire() -> Result<HostPreparationAuthority, BackendFailure> {
    let pool = PROBE.with(|slot| {
        let mut slot = slot.borrow_mut();
        let Some(probe) = slot.as_mut() else {
            return Ok(None);
        };
        probe.attempts += 1;
        if probe.reject_at == Some(probe.attempts) {
            return Err(BackendFailure::from_error(MockError::Capture(
                "host snapshot preparation rejected".into(),
            )));
        }
        Ok(probe.pool.clone())
    })?;
    // No test state borrow is held while acquiring or retiring an erased owner.
    pool.map(|pool| {
        pool.acquire_unquoted()
            .map(HostPreparationAuthority::retain)
            .map_err(BackendFailure::from_error)
    })
    .unwrap_or_else(|| Ok(HostPreparationAuthority::unmanaged()))
}

pub(crate) fn copy(operation: &'static str) -> Result<(), MockError> {
    PROBE.with(|slot| {
        let mut slot = slot.borrow_mut();
        let Some(probe) = slot.as_mut() else {
            return Ok(());
        };
        if let Some(pool) = &probe.pool {
            assert!(pool.unquoted_owner_count().unwrap() > 0);
        }
        probe.copies.push(operation);
        if probe.fail_copy == Some(operation) {
            return Err(MockError::Capture("host snapshot copy failed".into()));
        }
        Ok(())
    })
}

pub(crate) struct Guard;

impl Guard {
    pub fn new(pool: &WorkingMemoryPool) -> Self {
        PROBE.with(|slot| {
            let mut slot = slot.borrow_mut();
            assert!(slot.is_none());
            *slot = Some(Probe {
                pool: Some(pool.clone()),
                ..Default::default()
            });
        });
        Self
    }

    pub fn update<T>(&self, f: impl FnOnce(&mut Probe) -> T) -> T {
        PROBE.with(|slot| f(slot.borrow_mut().as_mut().unwrap()))
    }
}

impl Drop for Guard {
    fn drop(&mut self) {
        PROBE.with(|slot| drop(slot.borrow_mut().take()));
    }
}

pub(super) fn source_pool() -> Option<WorkingMemoryPool> {
    PROBE.with(|slot| slot.borrow().as_ref().and_then(|p| p.pool.clone()))
}
