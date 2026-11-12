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
