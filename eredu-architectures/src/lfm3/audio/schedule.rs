//! Checkpoint-owned response modes and commit boundaries for speech generation.

use super::{AudioTokens, DepthDeclaration, DepthFrame, PreparedAudio};
use eredu_runtime::RealtimeScheduleState;

/// How a prepared prompt enters generation.
#[derive(Debug, Clone, Copy, Default, Eq, PartialEq)]
pub enum AudioMode {
    /// Infer the initial mode from the final prompt token.
    #[default]
    Auto,
    /// Text only. Speech and interleaving markers are suppressed.
    Text,
    /// Start a speech segment, then return to text on speech EOS.
    Speech,
    /// Alternate text and speech according to the checkpoint cadence.
    Interleaved,
}

/// A categorical phase selected by the speech protocol.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum AudioPhase {
    /// Sample one language token.
    Text,
    /// Advance one depth decoder pass.
    Speech,
    /// End of response or exhausted output budget.
    Finished,
}

/// Invalid speech protocol configuration or out-of-order advancement.
#[derive(Debug, Clone, Copy, Eq, PartialEq, thiserror::Error)]
pub enum AudioScheduleError {
    /// Geometry, token IDs or cadence disagree with the checkpoint contract.
    #[error("invalid LFM3 Audio generation schedule")]
    Invalid,
    /// The proposed decision does not belong to the current phase.
    #[error("LFM3 Audio decision is out of phase")]
    OutOfPhase,
    /// A proposed successor differs from the admitted request.
    #[error("LFM3 Audio transition does not match the admitted schedule")]
    Incompatible,
}

/// Immutable generation protocol selected before a session starts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AudioSchedulePolicy {
    tokens: AudioTokens,
    eos: Vec<i32>,
    vocabulary: i32,
    text_slots: usize,
    speech_slots: usize,
    depth_schedule: Vec<usize>,
    audio_eos: i32,
    mode: AudioMode,
    budget: usize,
}

impl AudioSchedulePolicy {
    /// Derives the schedule from retained checkpoint assets and a finite slot budget.
    pub fn from_artifact(
        artifact: &PreparedAudio,
        mode: AudioMode,
        budget: usize,
    ) -> Result<Self, AudioScheduleError> {
        let vocabulary = artifact.language.args().vocab_size;
        let generation = &artifact.text_assets.generation;
        let scalar_id = |v: &serde_json::Value| {
            v.as_i64()
                .and_then(|n| i32::try_from(n).ok())
                .filter(|n| (0..vocabulary).contains(n))
                .ok_or(AudioScheduleError::Invalid)
        };
        let ids = generation
            .get("eos_token_id")
            .ok_or(AudioScheduleError::Invalid)?;
        let mut eos = match ids.as_array() {
            Some(ids) => ids.iter().map(scalar_id).collect::<Result<Vec<_>, _>>()?,
            None => vec![scalar_id(ids)?],
        };
        eos.sort_unstable();
        eos.dedup();
        if eos.is_empty()
            || budget == 0
            || budget
                .checked_mul(artifact.depth.schedule().len())
                .is_none()
        {
            return Err(AudioScheduleError::Invalid);
        }
        Ok(Self {
            tokens: artifact.tokens,
            eos,
            vocabulary,
            text_slots: artifact.interleaved_text_slots(),
            speech_slots: artifact.interleaved_speech_slots(),
            depth_schedule: artifact.depth.schedule().to_vec(),
            audio_eos: artifact.depth.eos_code(),
            mode,
            budget,
        })
    }

    /// Returns a marker to append before prefill for an explicitly selected mode.
    pub fn prompt_marker(&self, last: i32) -> Option<i32> {
        match self.mode {
            AudioMode::Speech if last != self.tokens.audio_start && last != self.tokens.output => {
                Some(self.tokens.audio_start)
            }
            AudioMode::Interleaved if last != self.tokens.mixed_start => {
                Some(self.tokens.mixed_start)
            }
            _ => None,
        }
    }

    /// Admits the final prompt token after marker insertion. Replayed terminal
    /// speech resumes text when automatic mode is selected.
    pub fn start(self, last: i32, replayed_eos: bool) -> Result<AudioSchedule, AudioScheduleError> {
        if !(0..self.vocabulary).contains(&last)
            || self.prompt_marker(last).is_some()
            || (replayed_eos && last != self.tokens.output)
        {
            return Err(AudioScheduleError::Invalid);
        }
        let mode = match self.mode {
            AudioMode::Auto if last == self.tokens.mixed_start => Mode::Interleaved {
                speaking: false,
                left: self.text_slots,
                text_done: false,
            },
            AudioMode::Auto
                if last == self.tokens.audio_start
                    || (last == self.tokens.output && !replayed_eos) =>
            {
                Mode::Speech
            }
            AudioMode::Auto | AudioMode::Text => Mode::Text,
            AudioMode::Speech => Mode::Speech,
            AudioMode::Interleaved => Mode::Interleaved {
                speaking: false,
                left: self.text_slots,
                text_done: false,
            },
        };
        Ok(AudioSchedule {
            policy: self,
            mode,
            committed: 0,
            advances: 0,
            pending: None,
        })
    }

    /// IDs suppressed during language sampling, including embedding placeholders.
    pub fn suppressed_tokens(&self) -> impl Iterator<Item = i32> {
        [self.tokens.input, self.tokens.output].into_iter().chain(
            (self.mode == AudioMode::Text)
                .then_some([self.tokens.audio_start, self.tokens.mixed_start])
                .into_iter()
                .flatten(),
        )
    }

    /// Number of codebooks participating in each speech frame.
    pub fn codebooks(&self) -> usize {
        self.depth_schedule.iter().sum()
    }

    /// Language slot budget, including committed speech frames.
    pub fn budget(&self) -> usize {
        self.budget
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum Mode {
    Text,
    Speech,
    Interleaved {
        speaking: bool,
        left: usize,
        text_done: bool,
    },
    Finished,
}

/// One branchable sequence of language slots and incomplete depth passes.
#[derive(Debug, Clone)]
pub struct AudioSchedule {
    policy: AudioSchedulePolicy,
    mode: Mode,
    committed: usize,
    advances: usize,
    pending: Option<DepthFrame>,
}

impl AudioSchedule {
    /// Selects the next model operation.
    pub fn phase(&self) -> AudioPhase {
        match self.mode {
            Mode::Finished => AudioPhase::Finished,
            Mode::Speech | Mode::Interleaved { speaking: true, .. } => AudioPhase::Speech,
            _ => AudioPhase::Text,
        }
    }

    /// Number of completed language slots. Intermediate depth passes do not consume slots.
    pub fn committed_slots(&self) -> usize {
        self.committed
    }

    /// Number of successful controlled advancements.
    pub fn advances(&self) -> usize {
        self.advances
    }

    /// Exact retained policy.
    pub fn policy(&self) -> &AudioSchedulePolicy {
        &self.policy
    }

    /// Current depth pass, if a frame has already begun.
    pub fn depth_frame(&self) -> Option<&DepthFrame> {
        self.pending.as_ref()
    }

    /// Installs the permutation for a new speech frame on an unpublished branch.
    pub fn begin_depth(
        &mut self,
        declaration: &DepthDeclaration,
        order: &[usize],
    ) -> Result<(), AudioScheduleError> {
        if self.phase() != AudioPhase::Speech || self.pending.is_some() {
            return Err(AudioScheduleError::OutOfPhase);
        }
        if declaration.schedule() != self.policy.depth_schedule
            || declaration.eos_code() != self.policy.audio_eos
        {
            return Err(AudioScheduleError::Incompatible);
        }
        self.pending = Some(
            declaration
                .begin_frame(order)
                .map_err(|_| AudioScheduleError::Invalid)?,
        );
        Ok(())
    }

    /// Commits one text decision, including marker switches and turn termination.
    pub fn commit_text(&mut self, token: i32) -> Result<(), AudioScheduleError> {
        if self.phase() != AudioPhase::Text || self.pending.is_some() {
            return Err(AudioScheduleError::OutOfPhase);
        }
        if !(0..self.policy.vocabulary).contains(&token)
            || self.policy.suppressed_tokens().any(|id| id == token)
        {
            return Err(AudioScheduleError::Invalid);
        }
        let was_interleaved = matches!(self.mode, Mode::Interleaved { .. });
        let tokens = self.policy.tokens;
        if token == tokens.text_end {
            self.mode = if was_interleaved {
                Mode::Interleaved {
                    speaking: true,
                    left: self.policy.speech_slots,
                    text_done: true,
                }
            } else {
                Mode::Finished
            };
        } else if self.policy.eos.contains(&token) {
            self.mode = Mode::Finished;
        } else if !was_interleaved {
            self.mode = if token == tokens.audio_start {
                Mode::Speech
            } else if token == tokens.mixed_start {
                Mode::Interleaved {
                    speaking: false,
                    left: self.policy.text_slots,
                    text_done: false,
                }
            } else {
                Mode::Text
            };
        }
        if was_interleaved {
            self.tick_interleaving();
        }
        self.consume_slot()
    }

    /// Commits one depth pass. A complete frame returns codec codes or speech EOS.
    /// No partial codes leave the schedule.
    pub fn reveal(
        &mut self,
        pass: usize,
        sampled: &[i32],
    ) -> Result<DepthProgress, AudioScheduleError> {
        if self.phase() != AudioPhase::Speech {
            return Err(AudioScheduleError::OutOfPhase);
        }
        let frame = self
            .pending
            .as_mut()
            .ok_or(AudioScheduleError::OutOfPhase)?;
        frame
            .reveal(pass, sampled)
            .map_err(|_| AudioScheduleError::Invalid)?;
        let Some(output) = frame.completed() else {
            self.advance_count()?;
            return Ok(DepthProgress::Pending);
        };
        let result = match output {
            super::CompletedSpeechFrame::Audio(codes) => DepthProgress::Audio(codes.to_vec()),
            super::CompletedSpeechFrame::End => DepthProgress::End,
        };
        self.pending = None;
        if result == DepthProgress::End {
            self.mode = Mode::Text;
        } else {
            self.tick_interleaving();
        }
        self.consume_slot()?;
        Ok(result)
    }

    fn tick_interleaving(&mut self) {
        if let Mode::Interleaved {
            speaking,
            left,
            text_done: false,
        } = &mut self.mode
        {
            *left -= 1;
            if *left == 0 {
                *speaking = !*speaking;
                *left = if *speaking {
                    self.policy.speech_slots
                } else {
                    self.policy.text_slots
                };
            }
        }
    }

    fn advance_count(&mut self) -> Result<(), AudioScheduleError> {
        self.advances = self
            .advances
            .checked_add(1)
            .ok_or(AudioScheduleError::Invalid)?;
        Ok(())
    }

    fn consume_slot(&mut self) -> Result<(), AudioScheduleError> {
        self.advance_count()?;
        self.committed += 1;
        if self.committed == self.policy.budget {
            self.mode = Mode::Finished;
        }
        Ok(())
    }
}

/// Result of advancing one depth pass.
#[derive(Debug, PartialEq, Eq)]
pub enum DepthProgress {
    /// Another pass remains.
    Pending,
    /// A complete frame in codec codebook order.
    Audio(Vec<i32>),
    /// Speech EOS. Language feedback uses the EOS code in every codebook.
    End,
}

impl RealtimeScheduleState for AudioSchedule {
    type Identity = AudioSchedulePolicy;
    type Error = AudioScheduleError;

    fn identity(&self) -> &Self::Identity {
        &self.policy
    }

    fn validate_successor(&self, next: &Self) -> Result<(), Self::Error> {
        if next.policy != self.policy
            || self.phase() == AudioPhase::Finished
            || self.advances.checked_add(1) != Some(next.advances)
            || next.committed < self.committed
            || next.committed - self.committed > 1
            || next.committed > self.policy.budget
        {
            return Err(AudioScheduleError::Incompatible);
        }
        Ok(())
    }
}
