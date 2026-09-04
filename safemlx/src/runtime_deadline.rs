//! Deadline-aware admission to the process-wide MLX runtime.

use std::time::{Duration, Instant};

use crate::{error::Exception, utils::runtime_lock};

/// One absolute deadline shared by synchronous MLX graph-construction calls.
#[derive(Debug, Clone, Copy)]
pub struct RuntimeCallDeadline {
    deadline: Instant,
}

impl RuntimeCallDeadline {
    /// Creates an absolute deadline from a positive selected timeout.
    pub fn new(timeout: Duration) -> Result<Self, Exception> {
        if timeout.is_zero() {
            return Err(Exception::custom(
                "MLX runtime call deadline requires a positive timeout",
            ));
        }
        let deadline = Instant::now().checked_add(timeout).ok_or_else(|| {
            Exception::custom("MLX runtime call deadline exceeds the monotonic clock range")
        })?;
        Ok(Self { deadline })
    }

    /// Fails if the shared deadline has elapsed before native mutation.
    pub fn check(self) -> Result<(), Exception> {
        if Instant::now() >= self.deadline {
            Err(Exception::custom(
                "MLX synchronous communication setup exceeded its selected deadline before native mutation",
            ))
        } else {
            Ok(())
        }
    }

    /// Acquires the global MLX runtime lock without allowing lock contention to
    /// defeat the selected deadline.
    pub fn enter(self) -> Result<RuntimeCallGuard, Exception> {
        loop {
            if let Some(guard) = runtime_lock::try_enter() {
                self.check()?;
                return Ok(RuntimeCallGuard {
                    _guard: guard,
                    deadline: self,
                });
            }
            self.check()?;
            std::thread::yield_now();
        }
    }
}

/// Retained ownership of the process-wide MLX runtime lock.
pub struct RuntimeCallGuard {
    _guard: runtime_lock::RuntimeLockGuard,
    deadline: RuntimeCallDeadline,
}

impl RuntimeCallGuard {
    /// Rechecks the retained setup deadline immediately before native mutation.
    pub fn check(&self) -> Result<(), Exception> {
        self.deadline.check()
    }
}

impl std::fmt::Debug for RuntimeCallGuard {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("RuntimeCallGuard").finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Barrier};

    #[test]
    fn runtime_lock_contention_cannot_outlive_selected_deadline() {
        let entered = Arc::new(Barrier::new(2));
        let release = Arc::new(Barrier::new(2));
        std::thread::scope(|scope| {
            let worker_entered = Arc::clone(&entered);
            let worker_release = Arc::clone(&release);
            scope.spawn(move || {
                let _guard = runtime_lock::enter();
                worker_entered.wait();
                worker_release.wait();
            });
            entered.wait();
            let started = Instant::now();
            let error = RuntimeCallDeadline::new(Duration::from_millis(5))
                .unwrap()
                .enter()
                .expect_err("busy runtime lock must respect the selected deadline");
            assert!(error.what().contains("selected deadline"));
            assert!(started.elapsed() < Duration::from_secs(1));
            release.wait();
        });
    }
}
