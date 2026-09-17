//! Saved numerical sampling/input data, distinct from runnable sampling state.

use super::*;
use crate::composition::mlx::session::model_session::saved_array_copy::decoder::CopiedTextComponentsOwner;
use crate::composition::mlx::session::model_session::saved_array_copy::{
    CopiedTextSampling, PreparedTextArrayCopy,
};
use eredu_runtime::execution_control::SamplingCopyPolicy;

type Pending = Option<PendingTextInput<MlxModelInput, MlxTextToken>>;

/// Immutable sampling/input checkpoint. Its funded representation exposes no
/// source receipt, live quote, mutable sampler or continuation grant.
pub struct MlxSavedSamplingState {
    inner: SavedSampling,
}

enum SavedSampling {
    Unquoted {
        sampling: MlxTextSamplingState,
        pending: Pending,
    },
    Funded(CopiedTextSampling),
    // A stable sampling view of one sealed decoder/sampler pair. Retaining this
    // view retains the same aggregate custody, not a second reservation.
    Paired(CopiedTextComponentsOwner),
}

fn unknown() -> Error {
    Error::Other(Box::new(
        eredu_runtime::working_memory::WorkingMemoryError::UnknownBound,
    ))
}

impl MlxSavedSamplingState {
    pub(super) fn paired(pair: CopiedTextComponentsOwner) -> Self {
        Self {
            inner: SavedSampling::Paired(pair),
        }
    }

    pub(super) fn paired_control_bytes() -> Option<usize> {
        [
            std::mem::size_of::<Self>(),
            std::mem::size_of::<SavedSampling>(),
            std::mem::size_of::<CopiedTextComponentsOwner>(),
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)
    }

    fn funded_ref(&self) -> Option<&CopiedTextSampling> {
        match &self.inner {
            SavedSampling::Unquoted { .. } => None,
            SavedSampling::Funded(saved) => Some(saved),
            SavedSampling::Paired(pair) => Some(pair.sampling()),
        }
    }

    pub(super) fn capture(
        runtime: &mut ModelRuntime<MlxBackend<'_>>,
        sampling: &MlxTextSamplingState,
        pending: Option<PendingTextInput<&MlxModelInput, &MlxTextToken>>,
        policy: SamplingCopyPolicy,
    ) -> Result<Self, Error> {
        let inner = match policy {
            SamplingCopyPolicy::Unquoted => {
                // Both legacy workers acquire real unquoted ownership before
                // copying. A funded live source cannot enter this variant.
                if sampling.quote.is_some() || sampling.sampler.is_funded() {
                    return Err(unknown());
                }
                let pending = MlxBackend::copy_pending_input(runtime, pending)?;
                let sampling = MlxBackend::copy_sampling_state(runtime, sampling)?;
                SavedSampling::Unquoted { sampling, pending }
            }
            SamplingCopyPolicy::Bounded(limits) => {
                let pending = match pending {
                    None => None,
                    Some(PendingTextInput::Decode(token)) => Some(token),
                    // Prefill input/cache identity needs its own closed host
                    // plan before it can join this physical aggregate.
                    Some(PendingTextInput::Prefill(_)) => return Err(unknown()),
                };
                let prepared = PreparedTextArrayCopy::prepare(runtime, sampling, pending)?;
                SavedSampling::Funded(prepared.copy_sampling(runtime, limits)?)
            }
        };
        Ok(Self { inner })
    }

    pub(super) fn copy(
        &self,
        runtime: &mut ModelRuntime<MlxBackend<'_>>,
        policy: SamplingCopyPolicy,
    ) -> Result<Self, Error> {
        match (&self.inner, policy) {
            (SavedSampling::Unquoted { sampling, pending }, SamplingCopyPolicy::Unquoted) => {
                Self::capture(
                    runtime,
                    sampling,
                    pending.as_ref().map(PendingTextInput::as_ref),
                    SamplingCopyPolicy::Unquoted,
                )
            }
            (SavedSampling::Funded(_), SamplingCopyPolicy::Bounded(limits))
            | (SavedSampling::Paired(_), SamplingCopyPolicy::Bounded(limits)) => {
                let saved = self.funded_ref().expect("matched funded sampling source");
                let prepared = PreparedTextArrayCopy::prepare_saved(runtime, saved)?;
                // Copying the sampling view produces an independent sampling
                // component; it does not silently copy or extract a decoder.
                Ok(Self {
                    inner: SavedSampling::Funded(prepared.copy_sampling(runtime, limits)?),
                })
            }
            // Copy policy neither promotes unquoted sources nor strips custody
            // from funded saved state.
            _ => Err(unknown()),
        }
    }

    pub(super) fn prediction(&self) -> u64 {
        match &self.inner {
            SavedSampling::Unquoted { sampling, .. } => sampling.next_prediction,
            SavedSampling::Funded(saved) => saved.next_prediction(),
            SavedSampling::Paired(pair) => pair.sampling().next_prediction(),
        }
    }

    pub(super) fn estimate(
        &self,
        runtime: &ModelRuntime<MlxBackend<'_>>,
    ) -> Result<Option<SnapshotEstimate>, Error> {
        match &self.inner {
            SavedSampling::Unquoted { sampling, pending } => {
                let sampling = MlxBackend::estimate_sampling_state(runtime, sampling)?;
                let pending = MlxBackend::estimate_pending_input(
                    runtime,
                    pending.as_ref().map(PendingTextInput::as_ref),
                )?;
                Ok(sampling.zip(pending).and_then(|(sampling, pending)| {
                    Some(SnapshotEstimate {
                        retained_bytes: sampling
                            .retained_bytes
                            .checked_add(pending.retained_bytes)?,
                        copy_bytes: sampling.copy_bytes.checked_add(pending.copy_bytes)?,
                    })
                }))
            }
            SavedSampling::Funded(_) | SavedSampling::Paired(_) => {
                let saved = self.funded_ref().expect("matched funded sampling source");
                let prepared = PreparedTextArrayCopy::prepare_saved(runtime, saved)?;
                Ok(Some(SnapshotEstimate {
                    retained_bytes: saved.bytes(),
                    copy_bytes: prepared.required_sampling_bytes()?,
                }))
            }
        }
    }

    pub(super) fn input_tokens(&self, predictions: u64) -> Option<u64> {
        match &self.inner {
            SavedSampling::Unquoted { pending, .. } => MlxBackend::continuation_input_tokens(
                pending.as_ref().map(PendingTextInput::as_ref),
                predictions,
            ),
            SavedSampling::Funded(_) | SavedSampling::Paired(_) => {
                let saved = self.funded_ref().expect("matched funded sampling source");
                saved.input_tokens(predictions)
            }
        }
    }

    pub(super) fn prepare_resume(
        &self,
        runtime: &mut ModelRuntime<MlxBackend<'_>>,
    ) -> Result<(MlxTextSamplingState, Pending), Error> {
        // A saved physical copy is not a fresh inference admission. A resumed
        // initial decode needs its own run/context and revision binding before
        // it can produce a live sampler or executable pending token.
        if self.funded_ref().is_some() {
            return Err(unknown());
        }
        let copied = self.copy(runtime, SamplingCopyPolicy::Unquoted)?;
        match copied.inner {
            SavedSampling::Unquoted { sampling, pending } => Ok((sampling, pending)),
            SavedSampling::Funded(_) | SavedSampling::Paired(_) => {
                unreachable!("unquoted copy preserves its representation")
            }
        }
    }
}
