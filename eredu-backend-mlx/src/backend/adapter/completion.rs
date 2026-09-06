use super::*;
use crate::backend::submission_recovery::{Recovery, Retention, Status};

struct CompletionResources {
    event: Option<Event>,
    arrays: Vec<Array>,
}

impl Retention for CompletionResources {
    fn observe(&self, _: Status) {}
}

/// Exact MLX event plus retained output arrays.
pub struct MlxCompletion {
    retained: Recovery<CompletionResources>,
}

impl Completion for MlxCompletion {
    type Error = Error;
    fn is_complete(&self) -> Result<bool, Self::Error> {
        // Retain one try-lock across both the recovery query and event access;
        // otherwise another host can acquire the runtime between those steps.
        safemlx::try_with_submission_retirement(|| {
            if !self.check_native_status()? {
                return Ok(false);
            }
            self.retained
                .retention()
                .event
                .as_ref()
                .expect("submitted event")
                .is_complete()
                .map_err(Into::into)
        })
        .unwrap_or(Ok(false))
    }
    fn wait(&self) -> Result<(), Self::Error> {
        while !self.is_complete()? {
            std::thread::yield_now();
        }
        Ok(())
    }
}

impl MlxCompletion {
    fn check_native_status(&self) -> Result<bool, Error> {
        let status = self.retained.progress();
        if status.failed || status.blocked {
            Err(Error::ArchitectureModel(
                "native submission failed; unresolved resources remain retained".into(),
            ))
        } else {
            Ok(status.settled)
        }
    }

    pub(crate) fn submission(output: Array) -> Result<Submission<Array, Self>, Error> {
        Self::submission_retaining(output, std::iter::empty())
    }

    pub(crate) fn submission_retaining(
        output: Array,
        additional: impl IntoIterator<Item = Array>,
    ) -> Result<Submission<Array, Self>, Error> {
        let retained = std::iter::once(output.clone())
            .chain(additional)
            .collect::<Vec<_>>();
        let mut retained = Recovery::begin(CompletionResources {
            event: None,
            arrays: retained,
        })?;
        let event = async_eval_with_event(retained.retention().arrays.iter());
        retained.seal();
        retained.retention_mut().event = Some(event?);
        let completion = Self { retained };
        completion.check_native_status()?;
        Ok(Submission { output, completion })
    }

    /// Number of arrays held until exact completion.
    pub fn retained_resources(&self) -> usize {
        self.retained.retention().arrays.len()
    }
}

#[cfg(test)]
mod retirement_tests {
    use super::*;
    use std::{
        sync::mpsc,
        time::{Duration, Instant},
    };

    thread_local! {
        static POLL_HOUSEKEEPING_CALLS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    }

    fn record_poll_housekeeping() {
        POLL_HOUSEKEEPING_CALLS.with(|calls| calls.set(calls.get() + 1));
    }

    #[test]
    fn completion_poll_returns_pending_when_another_thread_owns_the_runtime() {
        let stream =
            safemlx::Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
        let array = Array::ones::<f32>(&[4], &stream).unwrap();
        let submission = MlxCompletion::submission(array).unwrap();
        submission.completion.wait().unwrap();
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
        let started = Instant::now();
        let polled = submission.completion.is_complete();
        let elapsed = started.elapsed();
        let _ = release_tx.send(());
        holder.join().unwrap();
        assert!(!polled.unwrap(), "contention must report pending, not wait");
        assert!(elapsed < Duration::from_secs(1));
        // A recovery-only try-lock would end before Event::is_complete, causing
        // its ordinary runtime entry to run this hook. Keep one suppressed
        // guard through the entire successful poll, not only the busy path.
        POLL_HOUSEKEEPING_CALLS.with(|calls| calls.set(0));
        safemlx::register_thread_runtime_housekeeping(record_poll_housekeeping);
        let completed = submission.completion.wait();
        safemlx::unregister_thread_runtime_housekeeping(record_poll_housekeeping);
        completed.unwrap();
        assert_eq!(POLL_HOUSEKEEPING_CALLS.with(std::cell::Cell::get), 0);
    }

    #[test]
    fn completion_drop_retains_its_event_when_another_thread_owns_the_runtime() {
        let stream =
            safemlx::Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
        let array = Array::ones::<f32>(&[4], &stream).unwrap();
        let submission = MlxCompletion::submission(array).unwrap();
        submission.completion.wait().unwrap();
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
        let started = Instant::now();
        drop(submission.completion);
        assert!(started.elapsed() < Duration::from_secs(1));
        release_tx.send(()).unwrap();
        holder.join().unwrap();
        crate::backend::submission_recovery::reap();
    }
}
