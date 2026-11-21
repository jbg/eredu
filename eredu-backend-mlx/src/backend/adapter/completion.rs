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
