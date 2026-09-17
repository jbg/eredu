use super::*;
use crate::backend::submission_recovery::{Recovery, Retention, Status};

mod original;
pub(crate) use original::SamplingEventSource;

struct CompletionResources {
    event: Option<Event>,
    arrays: Vec<Array>,
    host_roots: [Option<Array>; 2],
    _original_recovery: Option<eredu_runtime::working_memory::OriginalPredictionRecoveryCustody>,
}

type CompletionArrays<'a> = std::iter::Chain<
    std::slice::Iter<'a, Array>,
    std::iter::Flatten<std::slice::Iter<'a, Option<Array>>>,
>;
impl CompletionResources {
    fn arrays(&self) -> CompletionArrays<'_> {
        self.arrays.iter().chain(self.host_roots.iter().flatten())
    }
}

impl crate::backend::submission_recovery::prediction::PredictionRetention for CompletionResources {
    fn install_prediction_custody(
        &mut self,
        custody: eredu_runtime::working_memory::OriginalPredictionRecoveryCustody,
    ) {
        self._original_recovery = Some(custody);
    }
}

impl Retention for CompletionResources {
    fn observe(&self, _: Status) {}
}

/// Exact MLX event plus retained output arrays.
pub struct MlxCompletion {
    inner: CompletionKind,
}
enum CompletionKind {
    Ordinary(OrdinaryCompletion),
    Original(original::OriginalSamplingCompletion),
}
struct OrdinaryCompletion {
    retained: Recovery<CompletionResources>,
}
impl Completion for MlxCompletion {
    type Error = Error;
    fn resources_releasable(&self) -> bool {
        match &self.inner {
            CompletionKind::Ordinary(value) => value.resources_releasable(),
            CompletionKind::Original(value) => value.resources_releasable(),
        }
    }
    fn is_complete(&self) -> Result<bool, Error> {
        match &self.inner {
            CompletionKind::Ordinary(value) => value.is_complete(),
            CompletionKind::Original(value) => value.is_complete(),
        }
    }
    fn wait(&self) -> Result<(), Error> {
        match &self.inner {
            CompletionKind::Ordinary(value) => value.wait(),
            CompletionKind::Original(value) => value.wait(),
        }
    }
}

impl Completion for OrdinaryCompletion {
    type Error = Error;
    fn resources_releasable(&self) -> bool {
        self.retained.progress().settled
    }

    fn is_complete(&self) -> Result<bool, Self::Error> {
        // Retain one try-lock across both the recovery query and event access;
        // otherwise another host can acquire the runtime between those steps.
        safemlx::try_with_submission_retirement(|| {
            if !self.check_native_status()? {
                return Ok(false);
            }
            if !self
                .retained
                .retention()
                .event
                .as_ref()
                .expect("submitted event")
                .is_complete()?
            {
                return Ok(false);
            }
            // These exact roots were encoded before the event was returned.
            // After it signals, eval only settles their descriptor events; it
            // cannot start an unscheduled graph or allocate another payload.
            // Cold allocation inventory deliberately does not perform this step.
            for array in self.retained.retention().arrays() {
                array.evaluated()?;
            }
            Ok(true)
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
    pub(crate) fn original_sampling_source(&self) -> Option<SamplingEventSource> {
        match &self.inner {
            CompletionKind::Original(value) => Some(value.source()),
            CompletionKind::Ordinary(_) => None,
        }
    }
    #[cfg(test)]
    pub(crate) fn sampling_event_native_cause(
        error: &Error,
    ) -> Option<safemlx::error::ScopedEvaluationCause> {
        let mut current: &(dyn std::error::Error + 'static) = error;
        loop {
            if let Some(error) = current.downcast_ref::<original::SamplingEventFailure>() {
                return match error.cause() {
                    original::SamplingEventCause::Native(error) => error.scoped_evaluation_cause(),
                    _ => None,
                };
            }
            current = current.source()?;
        }
    }

    pub(crate) fn sampling_borrowed_with_scope(
        output: &Array,
        random: Option<Array>,
        stream: &Stream,
        role: crate::backend::submission_recovery::prediction::PredictionRole,
    ) -> Result<Self, Error> {
        #[cfg(test)]
        crate::backend::submission_recovery::prediction::test_counts::record(2, true);
        Ok(Self {
            inner: CompletionKind::Original(original::OriginalSamplingCompletion::submit(
                output, random, stream, role,
            )?),
        })
    }
    pub(crate) fn prediction_scope_control_bytes() -> Option<u64> {
        Some(original::control_bytes()?.max(Self::host_sequence_control_bytes()?))
    }

    fn host_sequence_control_bytes() -> Option<u64> {
        use std::mem::size_of;
        let fixed = [
            size_of::<CompletionResources>(),
            size_of::<OrdinaryCompletion>(),
            size_of::<Self>(),
            size_of::<Submission<Array, Self>>(),
            size_of::<Result<Submission<Array, Self>, Error>>(),
            size_of::<Option<Array>>(),
            size_of::<Result<Event, safemlx::error::Exception>>(),
            size_of::<CompletionArrays<'static>>(),
            size_of::<safemlx::PreparedArrayClone>(),
            safemlx::PreparedArrayClone::control_bytes()?,
            Array::inspection_clone_handle_bytes(),
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)?;
        crate::backend::submission_recovery::prediction::control_bytes::<CompletionResources>()?
            .checked_add(u64::try_from(fixed).ok()?)
    }

    fn submission_host_sequence(
        output: Array,
        random: Option<Array>,
        role: crate::backend::submission_recovery::prediction::PredictionRole,
    ) -> Result<Submission<Array, Self>, Error> {
        // Fixed root slots and the prepared C clone precede the final recovery
        // custody. Native Event construction remains the existing legacy worker.
        #[cfg(test)]
        crate::backend::submission_recovery::prediction::test_counts::record(2, true);
        let mut retained = crate::backend::submission_recovery::prediction::begin(
            Some(role),
            CompletionResources {
                event: None,
                arrays: Vec::new(),
                host_roots: [None, None],
                _original_recovery: None,
            },
        )?;
        let clone_error = |cause| {
            Error::from(
                crate::backend::runtime::residency::manager::ResidencyError::OriginalClone(cause),
            )
        };
        let mut slot =
            safemlx::PreparedArrayClone::try_prepare_for_inspection().map_err(clone_error)?;
        retained.retention_mut().host_roots[0] =
            Some(slot.fill_for_inspection(&output).map_err(clone_error)?);
        retained.retention_mut().host_roots[1] = random;
        let event = async_eval_with_event(retained.retention().arrays());
        retained.seal();
        retained.retention_mut().event = Some(event?);
        let completion = OrdinaryCompletion { retained };
        completion.check_native_status()?;
        Ok(Submission {
            output,
            completion: Self {
                inner: CompletionKind::Ordinary(completion),
            },
        })
    }

    #[cfg(test)]
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
        #[cfg(test)]
        crate::backend::submission_recovery::prediction::test_counts::record(2, false);
        let mut retained = crate::backend::submission_recovery::prediction::begin(
            None,
            CompletionResources {
                event: None,
                arrays: retained,
                host_roots: [None, None],
                _original_recovery: None,
            },
        )?;
        let event = async_eval_with_event(retained.retention().arrays());
        retained.seal();
        retained.retention_mut().event = Some(event?);
        let completion = OrdinaryCompletion { retained };
        completion.check_native_status()?;
        Ok(Submission {
            output,
            completion: Self {
                inner: CompletionKind::Ordinary(completion),
            },
        })
    }

    /// The selected sampling producer has exactly one token and at most one
    /// next-RNG root. The general ordinary iterator API remains independent.
    pub(crate) fn submission_retaining_with_scope(
        output: Array,
        random: Option<Array>,
        stream: &Stream,
        role: Option<crate::backend::submission_recovery::prediction::PredictionRole>,
    ) -> Result<Submission<Array, Self>, Error> {
        let Some(role) = role else {
            return Self::submission_retaining(output, random);
        };
        if role.is_host_sequence() {
            return Self::submission_host_sequence(output, random, role);
        }
        #[cfg(test)]
        crate::backend::submission_recovery::prediction::test_counts::record(2, true);
        let completion =
            original::OriginalSamplingCompletion::submit(&output, random, stream, role)?;
        Ok(Submission {
            output,
            completion: Self {
                inner: CompletionKind::Original(completion),
            },
        })
    }

    /// Number of arrays held until exact completion.
    pub fn retained_resources(&self) -> usize {
        match &self.inner {
            CompletionKind::Ordinary(value) => value.retained.retention().arrays().count(),
            CompletionKind::Original(value) => value.retained_resources(),
        }
    }
}

impl OrdinaryCompletion {
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
    fn completed_sampling_siblings_have_cold_storage_without_token_observation() {
        use safemlx::ops::indexing::TryIndexOp;

        let mut devices = vec![safemlx::DeviceType::Cpu];
        #[cfg(all(target_vendor = "apple", feature = "metal", not(feature = "cuda")))]
        devices.push(safemlx::DeviceType::Gpu);
        for device in devices {
            let stream = safemlx::Stream::new_with_device(&safemlx::Device::new(device, 0));
            let key = safemlx::random::key(19).unwrap();
            let split = safemlx::random::split_n(&key, 2, &stream).unwrap();
            let next_key = split.try_index_device(0, &stream).unwrap();
            let token = split.try_index_device((1, 0), &stream).unwrap();
            let submission =
                MlxCompletion::submission_retaining(token, [next_key.clone()]).unwrap();
            submission.completion.wait().unwrap();
            assert!(next_key.allocation_info().unwrap().is_some());
            assert!(submission.output.allocation_info().unwrap().is_some());
        }
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
        assert!(!submission.completion.resources_releasable());
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
        crate::backend::submission_recovery::wait_for_retirement(|| {
            submission.completion.resources_releasable()
        });
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
