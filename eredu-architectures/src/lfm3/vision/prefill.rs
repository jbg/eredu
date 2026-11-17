//! Request-local media preparation and incremental decoder prefill.

use std::{num::NonZeroUsize, ops::Range};

use eredu_nn::{Error, Tensor};
use eredu_runtime::prefill::PrefillStep;

use super::{ProjectedVisionPrompt, VisionInput, VisionPrompt, VisionPromptPreparation, VlInput};

/// Progress from one media invocation or one decoder prefix.
pub enum VlPrefillProgress<T> {
    /// One media projection completed.
    Media,
    /// Scores for the next contiguous decoder prefix.
    Scores {
        /// Absolute range of expanded prompt positions.
        positions: Range<usize>,
        /// Full vocabulary scores for this range.
        scores: T,
    },
}

/// Admits a rank-local cursor after all ranks agree on its prompt and chunk size.
pub fn admit_partitioned_vl_prefill<T: Tensor, S>(
    prepared: Result<((VisionPromptPreparation<T>, NonZeroUsize), S), eredu_core::BackendFailure>,
    maximum_prefix: Option<NonZeroUsize>,
    agree: impl FnOnce([u64; 8], bool) -> Result<(), eredu_core::BackendFailure>,
) -> Result<(VlPrefill<T>, S), eredu_core::BackendFailure> {
    let prepared = prepared.map(|((prompt, chunk), state)| {
        (
            prompt,
            maximum_prefix.map_or(chunk, |limit| chunk.min(limit)),
            state,
        )
    });
    let descriptor = prepared.as_ref().map_or([0; 8], |(prompt, chunk, _)| {
        prompt.prefill_descriptor(*chunk)
    });
    let agreed = agree(descriptor, prepared.is_ok());
    let (prompt, chunk, state) = prepared?;
    agreed?;
    Ok((VlPrefill::new(prompt, chunk), state))
}

/// Advances the selected rank's media or decoder phase and agrees its completion.
#[allow(clippy::too_many_arguments)]
pub(super) fn advance_partitioned_vl_prefill<B, P, M>(
    cursor: &mut VlPrefill<B::Tensor>,
    context: &<B::Tensor as Tensor>::Context,
    vision: &mut super::PolicyVision<B, P>,
    mechanisms: &mut M,
    distributed: bool,
    mut agree: impl FnMut(&mut M, [u64; 8], bool) -> Result<(), eredu_core::BackendFailure>,
    decode: impl FnOnce(
        &mut M,
        &VisionPrompt<'_, B::Tensor>,
    ) -> Result<B::Tensor, eredu_core::BackendFailure>,
    settle: impl FnOnce(&mut VlPrefill<B::Tensor>) -> Result<(), eredu_core::BackendFailure>,
) -> Result<Option<VlPrefillProgress<B::Tensor>>, eredu_core::BackendFailure>
where
    B: eredu_nn::NeuralBackend,
    P: eredu_runtime::LayerwisePolicy<B, super::VisionModule<B>>,
    P::Error: std::error::Error + Send + Sync + 'static,
{
    use eredu_core::BackendFailure;

    let step = cursor.next_step().map_err(BackendFailure::from_error);
    let (phase, start, end) = match &step {
        Ok(PrefillStep::PrepareInput) => (1, 0, 0),
        Ok(PrefillStep::Decoder(range)) => (2, range.start as u64, range.end as u64),
        Ok(PrefillStep::Complete) => (3, 0, 0),
        Err(_) => (0, 0, 0),
    };
    let agreed = agree(
        mechanisms,
        [
            0x564c505246535445,
            phase,
            start,
            end,
            cursor.token_count() as u64,
            0,
            0,
            0,
        ],
        step.is_ok(),
    );
    if let Err(error) = step.and(agreed) {
        if distributed {
            cursor.invalidate();
        }
        return Err(error);
    }
    let progress = cursor
        .advance_input_checked(context, |input| {
            let ready = agree(
                mechanisms,
                [0x564c505246494e50, phase, start, end, 0, 0, 0, 0],
                input.is_ok(),
            );
            let input = input?;
            ready.map_err(Error::backend_source)?;
            match input {
                VlInput::Media(media) => vision.forward(media, context).map(|(_, rows)| rows),
                VlInput::Projected(prompt) => {
                    decode(mechanisms, prompt).map_err(Error::backend_source)
                }
                VlInput::Text(_) => unreachable!("VL prefill uses a projected prompt"),
            }
        })
        .map_err(BackendFailure::from_error)
        .and_then(|progress| {
            settle(cursor)?;
            Ok(progress)
        });
    let agreed = agree(
        mechanisms,
        [0x564c505246444f4e, phase, start, end, 0, 0, 0, 0],
        progress.is_ok(),
    );
    if progress.is_err() || agreed.is_err() {
        cursor.invalidate();
    }
    let progress = progress?;
    agreed?;
    Ok(progress)
}

/// Replaces the installed state after rank agreement on allocation.
pub fn reset_partitioned_vl_state<S>(
    installed: &mut S,
    replacement: Result<S, eredu_core::BackendFailure>,
    agree: impl FnOnce([u64; 8], bool) -> Result<(), eredu_core::BackendFailure>,
) -> Result<(), eredu_core::BackendFailure> {
    let agreed = agree(
        [0x564c524553455400, 0, 0, 0, 0, 0, 0, 0],
        replacement.is_ok(),
    );
    let replacement = replacement?;
    agreed?;
    *installed = replacement;
    Ok(())
}

/// Drives one rank-local decoder invocation and installs only a committed candidate.
#[allow(clippy::too_many_arguments)]
pub fn run_partitioned_vl_decoder<T: Tensor, S, M>(
    input: Result<crate::lfm3::Lfm3Input<'_, T>, eredu_core::BackendFailure>,
    state_offset: i32,
    state: &mut S,
    epoch: &mut eredu_core::DistributedCommitEpoch,
    maximum_prefix: NonZeroUsize,
    context: &T::Context,
    mechanisms: &mut M,
    fork_state: impl FnOnce(&S) -> Result<S, eredu_core::BackendFailure>,
    agree: impl FnOnce(&mut M, [u64; 8], bool) -> Result<(), eredu_core::BackendFailure>,
    execute: impl FnOnce(
        &mut M,
        crate::lfm3::Lfm3Input<'_, T>,
        &mut S,
        [u64; 8],
        eredu_core::DistributedCommitEpoch,
    ) -> Result<
        (T, eredu_core::DistributedCommitOutcome),
        eredu_core::BackendFailure,
    >,
) -> Result<T, eredu_core::BackendFailure> {
    use eredu_core::{BackendFailure, DistributedCommitOutcome};

    let contract = |message| BackendFailure::from_error(Error::backend(message));
    let prepared = input.and_then(|input| {
        if input
            .vl_pipeline_sequence_len()
            .map_err(BackendFailure::from_error)?
            > maximum_prefix.get()
        {
            return Err(contract(
                "VL input exceeds the retained pipeline prefix limit",
            ));
        }
        let descriptor = input
            .vl_pipeline_descriptor(context)
            .map_err(BackendFailure::from_error)?;
        let candidate = fork_state(state)?;
        let next = epoch
            .next()
            .ok_or_else(|| contract("VL pipeline epoch exhausted"))?;
        Ok((input, descriptor, candidate, next))
    });
    let agreed = agree(
        mechanisms,
        [
            0x564c505245504152,
            epoch.value(),
            state_offset as u64,
            0,
            0,
            0,
            0,
            0,
        ],
        prepared.is_ok(),
    );
    let (input, descriptor, mut candidate, next) = prepared?;
    agreed?;
    let current = *epoch;
    *epoch = next;
    let (scores, outcome) = execute(mechanisms, input, &mut candidate, descriptor, current)?;
    if outcome != DistributedCommitOutcome::Committed(current) {
        return Err(contract("VL pipeline did not commit its decoder pass"));
    }
    *state = candidate;
    Ok(scores)
}

/// Owns the admitted media, projected rows, and decoder prefix position.
#[derive(Clone)]
pub struct VlPrefill<T> {
    phase: Phase<T>,
    chunk: NonZeroUsize,
    tokens: usize,
    position: usize,
    identity: Option<eredu_runtime::PreparedInputCacheIdentity>,
}

#[derive(Clone)]
enum Phase<T> {
    Media(VisionPromptPreparation<T>),
    Decoder(ProjectedVisionPrompt<T>),
    Complete,
    Failed,
}

impl<T: Tensor> eredu_runtime::prefill::RetainedPrefillCursor for VlPrefill<T> {
    type Tensor = T;
    type Error = Error;

    fn decoder_position(&self) -> usize {
        self.decoder_position()
    }

    fn token_count(&self) -> usize {
        self.token_count()
    }

    fn next_step(&self) -> Result<PrefillStep, Self::Error> {
        self.next_step()
    }

    fn retained_metadata_bytes(&self) -> Option<usize> {
        self.retained_metadata_bytes()
    }

    fn visit_retained_values<'a>(&'a self, visit: impl FnMut(&'a T)) {
        self.retained_values().for_each(visit);
    }
}

impl<T: Tensor> VlPrefill<T> {
    /// Starts an admitted request with a positive decoder prefix limit.
    pub fn new(prompt: VisionPromptPreparation<T>, chunk: NonZeroUsize) -> Self {
        Self {
            identity: prompt.cache_identity().cloned(),
            tokens: prompt.token_ids().len(),
            position: 0,
            phase: Phase::Media(prompt),
            chunk,
        }
    }

    /// Semantic identity retained across media and decoder advancement.
    pub fn cache_identity(&self) -> Option<&eredu_runtime::PreparedInputCacheIdentity> {
        self.identity.as_ref()
    }

    /// Number of admitted decoder positions.
    pub fn token_count(&self) -> usize {
        self.tokens
    }

    /// Number of decoder positions committed by completed prefix steps.
    pub fn decoder_position(&self) -> usize {
        self.position
    }

    /// Immutable tensor roots retained by this cursor at a settled boundary.
    pub fn retained_values(&self) -> impl Iterator<Item = &T> {
        let (media, projected) = match &self.phase {
            Phase::Media(media) => (Some(media), None),
            Phase::Decoder(prompt) => (None, Some(prompt)),
            Phase::Complete | Phase::Failed => (None, None),
        };
        media
            .into_iter()
            .flat_map(VisionPromptPreparation::retained_values)
            .chain(
                projected
                    .into_iter()
                    .flat_map(ProjectedVisionPrompt::retained_values),
            )
    }

    /// Host bytes retained by this cursor, excluding native tensor storage.
    pub fn retained_metadata_bytes(&self) -> Option<usize> {
        let phase = match &self.phase {
            Phase::Media(media) => media.retained_metadata_bytes()?,
            Phase::Decoder(prompt) => prompt.retained_metadata_bytes()?,
            Phase::Complete | Phase::Failed => 0,
        };
        let identity = match &self.identity {
            Some(identity) => usize::try_from(identity.logical_metadata_bytes()?).ok()?,
            None => 0,
        };
        std::mem::size_of::<Self>()
            .checked_add(phase)?
            .checked_add(identity)
    }

    /// Whether all expanded positions have been committed to the decoder.
    pub fn is_complete(&self) -> bool {
        matches!(self.phase, Phase::Complete)
    }

    /// Discards a continuation whose backend completion failed.
    pub fn invalidate(&mut self) {
        self.phase = Phase::Failed;
    }

    /// Classifies the next step before execution or mutation.
    pub fn next_step(&self) -> Result<PrefillStep, Error> {
        match &self.phase {
            Phase::Media(prompt) if prompt.next_media().is_some() => Ok(PrefillStep::PrepareInput),
            Phase::Complete => Ok(PrefillStep::Complete),
            Phase::Failed => Err(Error::backend("LFM3 VL prefill has failed")),
            Phase::Media(_) | Phase::Decoder(_) => {
                let end = self
                    .position
                    .saturating_add(self.chunk.get())
                    .min(self.tokens);
                Ok(PrefillStep::Decoder(self.position..end))
            }
        }
    }

    /// Runs at most one media invocation or one decoder prefix.
    /// An execution error terminates the request without advancing the cursor.
    pub fn advance(
        &mut self,
        context: &T::Context,
        project: impl FnOnce(VisionInput<'_, T>) -> Result<T, Error>,
        decode: impl FnOnce(&VisionPrompt<'_, T>) -> Result<T, Error>,
    ) -> Result<Option<VlPrefillProgress<T>>, Error> {
        self.advance_input(context, |input| match input {
            VlInput::Media(input) => project(input),
            VlInput::Projected(input) => decode(input),
            VlInput::Text(_) => unreachable!("VL prefill uses a projected prompt"),
        })
    }

    /// Runs the next scheduled operation through the combined component graph.
    pub fn advance_input(
        &mut self,
        context: &T::Context,
        execute: impl for<'a> FnOnce(VlInput<'a, T>) -> Result<T, Error>,
    ) -> Result<Option<VlPrefillProgress<T>>, Error> {
        self.advance_input_checked(context, |input| execute(input?))
    }

    /// Offers local input preparation failures to the same execution callback
    /// as prepared inputs, so ranks can agree before entering a collective.
    pub fn advance_input_checked(
        &mut self,
        context: &T::Context,
        execute: impl for<'a> FnOnce(Result<VlInput<'a, T>, Error>) -> Result<T, Error>,
    ) -> Result<Option<VlPrefillProgress<T>>, Error> {
        let step = self.next_step()?;
        let prompt = match std::mem::replace(&mut self.phase, Phase::Failed) {
            Phase::Media(mut preparation) => {
                if let Some(media) = preparation.next_media() {
                    preparation.accept_projection(execute(Ok(VlInput::Media(media)))?)?;
                    self.phase = Phase::Media(preparation);
                    return Ok(Some(VlPrefillProgress::Media));
                }
                match preparation.finish(context) {
                    Ok(prompt) => prompt,
                    Err(error) => {
                        execute(Err(error))?;
                        return Err(Error::backend(
                            "VL prefill execution accepted an invalid input",
                        ));
                    }
                }
            }
            Phase::Decoder(prompt) => prompt,
            Phase::Complete => {
                self.phase = Phase::Complete;
                return Ok(None);
            }
            Phase::Failed => return Err(Error::backend("LFM3 VL prefill has failed")),
        };
        let PrefillStep::Decoder(positions) = step else {
            unreachable!("media and terminal steps return before decoder execution");
        };
        let (position, end) = (positions.start, positions.end);
        let full = prompt.prompt();
        let prefix = (|| {
            full.slice(
                i32::try_from(position).map_err(Error::backend_source)?
                    ..i32::try_from(end).map_err(Error::backend_source)?,
                context,
            )
        })();
        let scores = match prefix {
            Ok(prefix) => execute(Ok(VlInput::Projected(&prefix)))?,
            Err(error) => {
                execute(Err(error))?;
                return Err(Error::backend(
                    "VL prefill execution accepted an invalid input",
                ));
            }
        };
        self.position = end;
        self.phase = if end < self.tokens {
            Phase::Decoder(prompt)
        } else {
            Phase::Complete
        };
        Ok(Some(VlPrefillProgress::Scores { positions, scores }))
    }
}
