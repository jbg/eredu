use super::*;

/// Exact MLX event plus retained output arrays.
pub struct MlxCompletion {
    event: Event,
    retained: Vec<Array>,
}

impl Completion for MlxCompletion {
    type Error = Error;
    fn is_complete(&self) -> Result<bool, Self::Error> {
        self.event.is_complete().map_err(Into::into)
    }
    fn wait(&self) -> Result<(), Self::Error> {
        self.event.synchronize().map_err(Into::into)
    }
}

impl Drop for MlxCompletion {
    fn drop(&mut self) {
        match self.event.is_complete() {
            Ok(true) => {}
            Ok(false) | Err(_) => {
                let _ = self.event.synchronize();
            }
        }
    }
}

impl MlxCompletion {
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
        let event = async_eval_with_event(retained.iter())?;
        Ok(Submission {
            output,
            completion: Self { event, retained },
        })
    }

    /// Number of arrays held until exact completion.
    pub fn retained_resources(&self) -> usize {
        self.retained.len()
    }
}
