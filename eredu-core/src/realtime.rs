//! Portable realtime token frames, schedules, and facade errors.

use crate::{
    observation::{ObservationError, TensorObservation, TensorObservationData},
    scheduler::{SchedulerError, SemanticStateTransaction, WorkDescriptor, WorkId},
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Largest admitted text or audio delay in one portable realtime schedule.
///
/// The bound keeps every delay representable by backends whose token and cache
/// coordinates use signed 32-bit integers. Runtime frontier arithmetic remains
/// checked independently.
pub const MAX_REALTIME_FRAME_DELAY: usize = i32::MAX as usize;

/// Coordinate convention used by a realtime text-plus-audio frame schedule.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum RealtimeFrameConvention {
    /// Inputs are retained as undelayed history while generated values are
    /// written back to the history frame selected by their delay.
    FeedbackAlignedHistory,
    /// Every value is placed directly at its absolute `frontier + delay`
    /// position and model inputs read the preceding absolute position.
    AbsoluteDelayedSlots,
}

/// Static codec-token geometry shared by every session of one realtime model.
#[derive(Debug, Clone, Eq, PartialEq, Serialize)]
pub struct RealtimeSpeechConfig {
    total_audio_codebooks: usize,
    input_audio_codebooks: usize,
    generated_audio_codebooks: usize,
    depth_audio_codebooks: usize,
    text_padding_token: i32,
    audio_padding_token: i32,
    frame_convention: RealtimeFrameConvention,
    delays: Vec<usize>,
}

impl<'de> Deserialize<'de> for RealtimeSpeechConfig {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Raw {
            total_audio_codebooks: usize,
            input_audio_codebooks: usize,
            generated_audio_codebooks: usize,
            depth_audio_codebooks: usize,
            text_padding_token: i32,
            audio_padding_token: i32,
            frame_convention: RealtimeFrameConvention,
            delays: Vec<usize>,
        }
        let raw = Raw::deserialize(deserializer)?;
        Self::new(
            raw.total_audio_codebooks,
            raw.input_audio_codebooks,
            raw.generated_audio_codebooks,
            raw.depth_audio_codebooks,
            raw.text_padding_token,
            raw.audio_padding_token,
            raw.frame_convention,
            raw.delays,
        )
        .map_err(serde::de::Error::custom)
    }
}

impl RealtimeSpeechConfig {
    /// Creates and validates portable realtime codec geometry.
    #[allow(clippy::too_many_arguments)] // Public codec geometry is intentionally explicit.
    pub fn new(
        total_audio_codebooks: usize,
        input_audio_codebooks: usize,
        generated_audio_codebooks: usize,
        depth_audio_codebooks: usize,
        text_padding_token: i32,
        audio_padding_token: i32,
        frame_convention: RealtimeFrameConvention,
        delays: Vec<usize>,
    ) -> Result<Self, RealtimeConfigError> {
        if total_audio_codebooks == 0
            || generated_audio_codebooks == 0
            || depth_audio_codebooks == 0
        {
            return Err(RealtimeConfigError::EmptyCodebookGeometry);
        }
        if input_audio_codebooks.checked_add(generated_audio_codebooks)
            != Some(total_audio_codebooks)
        {
            return Err(RealtimeConfigError::CodebookPartition {
                total: total_audio_codebooks,
                input: input_audio_codebooks,
                generated: generated_audio_codebooks,
            });
        }
        if generated_audio_codebooks > depth_audio_codebooks
            || depth_audio_codebooks > total_audio_codebooks
        {
            return Err(RealtimeConfigError::DepthCodebookGeometry {
                generated: generated_audio_codebooks,
                depth: depth_audio_codebooks,
                total: total_audio_codebooks,
            });
        }
        if text_padding_token < 0 || audio_padding_token < 0 {
            return Err(RealtimeConfigError::NegativePaddingToken {
                text: text_padding_token,
                audio: audio_padding_token,
            });
        }
        let expected_delays = total_audio_codebooks
            .checked_add(1)
            .ok_or(RealtimeConfigError::CodebookCountOverflow)?;
        if delays.len() != expected_delays {
            return Err(RealtimeConfigError::DelayCount {
                expected: expected_delays,
                actual: delays.len(),
            });
        }
        if let Some((slot, delay)) = delays
            .iter()
            .copied()
            .enumerate()
            .find(|(_, delay)| *delay > MAX_REALTIME_FRAME_DELAY)
        {
            return Err(RealtimeConfigError::DelayOutOfRange {
                slot,
                delay,
                maximum: MAX_REALTIME_FRAME_DELAY,
            });
        }
        Ok(Self {
            total_audio_codebooks,
            input_audio_codebooks,
            generated_audio_codebooks,
            depth_audio_codebooks,
            text_padding_token,
            audio_padding_token,
            frame_convention,
            delays,
        })
    }

    /// Total number of temporal-model audio codebooks.
    pub const fn total_audio_codebooks(&self) -> usize {
        self.total_audio_codebooks
    }
    /// Number of live input-side codebooks per frame.
    pub const fn input_audio_codebooks(&self) -> usize {
        self.input_audio_codebooks
    }
    /// Number of generated-side codebooks per frame.
    pub const fn generated_audio_codebooks(&self) -> usize {
        self.generated_audio_codebooks
    }
    /// Number of depth-transformer codebooks per frame.
    pub const fn depth_audio_codebooks(&self) -> usize {
        self.depth_audio_codebooks
    }
    /// Text token used before sampled text is available.
    pub const fn text_padding_token(&self) -> i32 {
        self.text_padding_token
    }
    /// Audio token used while delayed streams warm up.
    pub const fn audio_padding_token(&self) -> i32 {
        self.audio_padding_token
    }
    /// Explicit delayed-frame coordinate convention.
    pub const fn frame_convention(&self) -> RealtimeFrameConvention {
        self.frame_convention
    }
    /// Complete text-plus-audio delay schedule in canonical slot order.
    pub fn delays(&self) -> &[usize] {
        &self.delays
    }
    /// Leading text-stream delay.
    pub fn text_delay(&self) -> usize {
        self.delays[0]
    }
    /// Per-codebook delays following the leading text delay.
    pub fn audio_delays(&self) -> &[usize] {
        &self.delays[1..]
    }
    /// Largest delay across the complete text-plus-audio schedule.
    pub fn max_delay(&self) -> usize {
        self.delays.iter().copied().max().unwrap_or(0)
    }
    /// Largest audio delay in frames.
    pub fn max_audio_delay(&self) -> usize {
        self.audio_delays().iter().copied().max().unwrap_or(0)
    }
}

/// Invalid portable realtime configuration.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum RealtimeConfigError {
    /// Every realtime codebook dimension must be nonzero.
    #[error("realtime codebook geometry must be nonzero")]
    EmptyCodebookGeometry,
    /// Input and generated codebooks must partition the temporal codebooks.
    #[error(
        "realtime input ({input}) and generated ({generated}) codebooks do not partition total {total}"
    )]
    CodebookPartition {
        /// Total temporal codebooks.
        total: usize,
        /// Input codebooks.
        input: usize,
        /// Generated codebooks.
        generated: usize,
    },
    /// The depth predictor must contain every generated codebook and no more
    /// than the complete temporal audio geometry.
    #[error(
        "realtime generated ({generated}), depth ({depth}), and total ({total}) codebooks must satisfy 0 < generated <= depth <= total"
    )]
    DepthCodebookGeometry {
        /// Generated audio codebooks.
        generated: usize,
        /// Depth-predictor codebooks.
        depth: usize,
        /// Total temporal audio codebooks.
        total: usize,
    },
    /// The text-plus-audio slot count overflowed portable geometry.
    #[error("realtime text-plus-audio slot count overflowed")]
    CodebookCountOverflow,
    /// Padding tokens used by realtime model inputs must be non-negative.
    #[error("realtime padding tokens must be non-negative, got text={text} audio={audio}")]
    NegativePaddingToken {
        /// Invalid text padding token.
        text: i32,
        /// Invalid audio padding token.
        audio: i32,
    },
    /// The delay schedule must describe text and every temporal audio codebook.
    #[error("realtime delay schedule has {actual} entries, expected {expected}")]
    DelayCount {
        /// Expected delay count.
        expected: usize,
        /// Actual delay count.
        actual: usize,
    },
    /// One delay exceeds the portable coordinate bound.
    #[error("realtime delay {delay} at slot {slot} exceeds maximum {maximum}")]
    DelayOutOfRange {
        /// Canonical text-plus-audio slot.
        slot: usize,
        /// Invalid delay.
        delay: usize,
        /// Largest admitted delay.
        maximum: usize,
    },
    /// Sampling temperatures must be finite and nonnegative.
    #[error(
        "realtime sampling temperatures must be finite and non-negative, got text={text} audio={audio}"
    )]
    SamplingTemperature {
        /// Invalid text temperature.
        text: f32,
        /// Invalid audio temperature.
        audio: f32,
    },
    /// Top-k truncation, when selected, must admit at least one token.
    #[error(
        "realtime sampling top-k must be positive when set, got text={text:?} audio={audio:?}"
    )]
    SamplingTopK {
        /// Invalid text top-k value.
        text: Option<usize>,
        /// Invalid audio top-k value.
        audio: Option<usize>,
    },
}

/// Canonical stream slot in a text-plus-audio realtime schedule.
#[derive(Debug, Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[non_exhaustive]
pub enum RealtimeFrameSlot {
    /// Text stream.
    Text,
    /// Zero-based temporal audio codebook.
    Audio(usize),
}

impl RealtimeFrameSlot {
    fn index(self) -> usize {
        match self {
            Self::Text => 0,
            Self::Audio(codebook) => codebook + 1,
        }
    }
}

/// One absolute position in a canonical text-plus-audio slot timeline.
#[derive(Debug, Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct RealtimeSlotCoordinate {
    position: usize,
    slot: RealtimeFrameSlot,
}

impl RealtimeSlotCoordinate {
    /// Creates one portable delayed-slot coordinate.
    pub const fn new(position: usize, slot: RealtimeFrameSlot) -> Self {
        Self { position, slot }
    }

    /// Absolute history or delayed-timeline position.
    pub const fn position(self) -> usize {
        self.position
    }

    /// Text or audio stream slot.
    pub const fn slot(self) -> RealtimeFrameSlot {
        self.slot
    }
}

/// Provenance of an occupied portable schedule slot.
///
/// This records only scheduling metadata. Token payloads remain backend-owned.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
#[non_exhaustive]
pub enum RealtimeSlotOccupancy {
    /// Live input-side audio supplied for this transition.
    Input,
    /// Caller-forced text or generated audio target.
    Forced,
    /// Convention-defined warm-up padding.
    Padding,
    /// A model-selected text or audio target.
    Sampled,
}

/// Source selected for one temporal model input slot.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
#[non_exhaustive]
pub enum RealtimeTemporalSource {
    /// The model consumes the configured text or audio padding token.
    Padding(RealtimeFrameSlot),
    /// The model consumes the backend payload stored at this occupied slot.
    Occupied {
        /// Exact portable coordinate.
        coordinate: RealtimeSlotCoordinate,
        /// Scheduling provenance of the payload.
        occupancy: RealtimeSlotOccupancy,
    },
}

/// How one text or depth-codebook decision is resolved.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
#[non_exhaustive]
pub enum RealtimeForcedSource {
    /// The forcing payload belongs to the currently submitted portable frame.
    CurrentInput,
    /// The forcing payload was retained at the target coordinate by an earlier frame.
    Retained,
}

/// How one text or depth-codebook decision is resolved.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
#[non_exhaustive]
pub enum RealtimeTargetSource {
    /// Caller forcing resolves this decision.
    Forced(RealtimeForcedSource),
    /// The model sampler resolves this decision.
    Sampled,
    /// An already placed input, forced value, or padding resolves the decision.
    Existing(RealtimeSlotOccupancy),
}

/// Portable placement for one text or depth-codebook decision.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct RealtimeTargetDecision {
    slot: RealtimeFrameSlot,
    coordinate: Option<RealtimeSlotCoordinate>,
    source: RealtimeTargetSource,
}

impl RealtimeTargetDecision {
    /// Text or audio decision slot.
    pub const fn slot(self) -> RealtimeFrameSlot {
        self.slot
    }

    /// Destination coordinate, or `None` during feedback-history warm-up.
    pub const fn coordinate(self) -> Option<RealtimeSlotCoordinate> {
        self.coordinate
    }

    /// Forced, sampled, or already occupied resolution.
    pub const fn source(self) -> RealtimeTargetSource {
        self.source
    }
}

/// Per-decision forcing mask supplied for one realtime transition.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct RealtimeFrameForcing {
    text: bool,
    generated_audio: Vec<bool>,
}

impl RealtimeFrameForcing {
    /// Creates a forcing mask in generated-codebook order.
    pub fn new(text: bool, generated_audio: Vec<bool>) -> Self {
        Self {
            text,
            generated_audio,
        }
    }

    /// No forced text or generated-audio decisions.
    pub fn none(config: &RealtimeSpeechConfig) -> Self {
        Self::new(false, vec![false; config.generated_audio_codebooks])
    }

    /// Whether the text target is forced.
    pub const fn text(&self) -> bool {
        self.text
    }

    /// Per-generated-codebook forcing mask.
    pub fn generated_audio(&self) -> &[bool] {
        &self.generated_audio
    }
}

/// Complete portable scheduling decision for one accepted input-side frame.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct RealtimeFrameTransition {
    frontier: usize,
    input_placements: Vec<RealtimeSlotCoordinate>,
    forced_placements: Vec<RealtimeSlotCoordinate>,
    warmup_padding: Vec<RealtimeSlotCoordinate>,
    temporal_inputs: Vec<RealtimeTemporalSource>,
    targets: Vec<RealtimeTargetDecision>,
    output: Option<Vec<RealtimeSlotCoordinate>>,
    model_call_required: bool,
    next_frontier: usize,
}

impl RealtimeFrameTransition {
    /// Committed frontier before this transition.
    pub const fn frontier(&self) -> usize {
        self.frontier
    }

    /// Coordinates receiving live input-side audio payloads.
    pub fn input_placements(&self) -> &[RealtimeSlotCoordinate] {
        &self.input_placements
    }

    /// Coordinates receiving caller-forced text or generated audio payloads.
    pub fn forced_placements(&self) -> &[RealtimeSlotCoordinate] {
        &self.forced_placements
    }

    /// Coordinates initialized to the configured warm-up padding token.
    pub fn warmup_padding(&self) -> &[RealtimeSlotCoordinate] {
        &self.warmup_padding
    }

    /// Temporal text-plus-audio inputs in canonical slot order.
    pub fn temporal_inputs(&self) -> &[RealtimeTemporalSource] {
        &self.temporal_inputs
    }

    /// Ordered text followed by depth-codebook decisions.
    pub fn targets(&self) -> &[RealtimeTargetDecision] {
        &self.targets
    }

    /// Delay-aligned generated-audio frame, in generated-codebook order.
    pub fn output(&self) -> Option<&[RealtimeSlotCoordinate]> {
        self.output.as_deref()
    }

    /// Whether this transition requires temporal/depth model execution.
    pub const fn model_call_required(&self) -> bool {
        self.model_call_required
    }

    /// Frontier published when the transition branch commits.
    pub const fn next_frontier(&self) -> usize {
        self.next_frontier
    }
}

/// Portable delayed-frame schedule state with no token or backend payloads.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct RealtimeFrameScheduleState {
    schedule: RealtimeSpeechConfig,
    frontier: usize,
    occupied: BTreeMap<RealtimeSlotCoordinate, RealtimeSlotOccupancy>,
}

impl RealtimeFrameScheduleState {
    /// Creates an empty state bound to one exact normalized schedule.
    pub fn new(schedule: RealtimeSpeechConfig) -> Self {
        Self {
            schedule,
            frontier: 0,
            occupied: BTreeMap::new(),
        }
    }

    /// Exact normalized schedule identity carried by this state.
    pub const fn schedule(&self) -> &RealtimeSpeechConfig {
        &self.schedule
    }

    /// Next input-side frame coordinate to accept.
    pub const fn frontier(&self) -> usize {
        self.frontier
    }

    /// Returns scheduling provenance for one retained coordinate.
    pub fn occupancy(&self, coordinate: RealtimeSlotCoordinate) -> Option<RealtimeSlotOccupancy> {
        self.occupied.get(&coordinate).copied()
    }

    /// Rejects state handoff to any materially different normalized schedule.
    pub fn validate_schedule(
        &self,
        schedule: &RealtimeSpeechConfig,
    ) -> Result<(), RealtimeScheduleError> {
        if &self.schedule == schedule {
            Ok(())
        } else {
            Err(RealtimeScheduleError::ScheduleMismatch)
        }
    }

    /// Accepts one input frame, resolves all portable coordinates, records
    /// target occupancy, and advances this transaction-local branch.
    ///
    /// The update is atomic: malformed masks, missing history, and arithmetic
    /// overflow leave `self` unchanged.
    pub fn advance(
        &mut self,
        schedule: &RealtimeSpeechConfig,
        forcing: &RealtimeFrameForcing,
    ) -> Result<RealtimeFrameTransition, RealtimeScheduleError> {
        self.validate_schedule(schedule)?;
        if forcing.generated_audio.len() != schedule.generated_audio_codebooks {
            return Err(RealtimeScheduleError::ForcingCount {
                expected: schedule.generated_audio_codebooks,
                actual: forcing.generated_audio.len(),
            });
        }
        let mut branch = self.clone();
        let transition = match schedule.frame_convention {
            RealtimeFrameConvention::FeedbackAlignedHistory => branch.advance_feedback(forcing)?,
            RealtimeFrameConvention::AbsoluteDelayedSlots => branch.advance_absolute(forcing)?,
        };
        *self = branch;
        Ok(transition)
    }

    fn advance_feedback(
        &mut self,
        forcing: &RealtimeFrameForcing,
    ) -> Result<RealtimeFrameTransition, RealtimeScheduleError> {
        let schedule = &self.schedule;
        let frontier = self.frontier;
        let next_frontier = checked_add(frontier, 1)?;
        let generated = schedule.generated_audio_codebooks;
        let mut input_placements = Vec::with_capacity(schedule.input_audio_codebooks);
        for codebook in generated..schedule.total_audio_codebooks {
            let coordinate = coordinate(frontier, RealtimeFrameSlot::Audio(codebook));
            self.occupied
                .insert(coordinate, RealtimeSlotOccupancy::Input);
            input_placements.push(coordinate);
        }

        let mut forced_placements = Vec::new();
        for (codebook, forced) in forcing.generated_audio.iter().copied().enumerate() {
            if forced {
                let coordinate = coordinate(frontier, RealtimeFrameSlot::Audio(codebook));
                self.occupied
                    .insert(coordinate, RealtimeSlotOccupancy::Forced);
                forced_placements.push(coordinate);
            }
        }

        let mut temporal_inputs = Vec::with_capacity(schedule.delays.len());
        for slot in slots(schedule.total_audio_codebooks) {
            let delay = schedule.delays[slot.index()];
            let source = frontier
                .checked_sub(1)
                .and_then(|position| position.checked_sub(delay));
            match source {
                None => temporal_inputs.push(RealtimeTemporalSource::Padding(slot)),
                Some(position) => {
                    let coordinate = coordinate(position, slot);
                    let occupancy = self.required_occupancy(coordinate)?;
                    temporal_inputs.push(RealtimeTemporalSource::Occupied {
                        coordinate,
                        occupancy,
                    });
                }
            }
        }

        let mut targets = Vec::with_capacity(1 + schedule.depth_audio_codebooks);
        let text_coordinate = frontier
            .checked_sub(schedule.text_delay())
            .map(|position| coordinate(position, RealtimeFrameSlot::Text));
        let text_source = if forcing.text {
            RealtimeTargetSource::Forced(RealtimeForcedSource::CurrentInput)
        } else {
            RealtimeTargetSource::Sampled
        };
        if let Some(coordinate) = text_coordinate {
            self.occupied.insert(
                coordinate,
                if forcing.text {
                    RealtimeSlotOccupancy::Forced
                } else {
                    RealtimeSlotOccupancy::Sampled
                },
            );
            if forcing.text {
                forced_placements.push(coordinate);
            }
        }
        targets.push(RealtimeTargetDecision {
            slot: RealtimeFrameSlot::Text,
            coordinate: text_coordinate,
            source: text_source,
        });
        for codebook in 0..schedule.depth_audio_codebooks {
            let slot = RealtimeFrameSlot::Audio(codebook);
            if codebook < generated {
                let target_coordinate = frontier
                    .checked_sub(schedule.audio_delays()[codebook])
                    .map(|position| coordinate(position, slot));
                let forced = forcing.generated_audio[codebook];
                if let Some(coordinate) = target_coordinate {
                    self.occupied.insert(
                        coordinate,
                        if forced {
                            RealtimeSlotOccupancy::Forced
                        } else {
                            RealtimeSlotOccupancy::Sampled
                        },
                    );
                }
                targets.push(RealtimeTargetDecision {
                    slot,
                    coordinate: target_coordinate,
                    source: if forced {
                        RealtimeTargetSource::Forced(RealtimeForcedSource::CurrentInput)
                    } else {
                        RealtimeTargetSource::Sampled
                    },
                });
            } else {
                let input_coordinate = coordinate(frontier, slot);
                let occupancy = self.required_occupancy(input_coordinate)?;
                targets.push(RealtimeTargetDecision {
                    slot,
                    coordinate: Some(input_coordinate),
                    source: RealtimeTargetSource::Existing(occupancy),
                });
            }
        }

        let output = frontier
            .checked_sub(schedule.max_delay())
            .map(|position| self.output_at_same_position(position))
            .transpose()?;
        self.frontier = next_frontier;
        self.prune_before(frontier.saturating_sub(schedule.max_delay()));
        Ok(RealtimeFrameTransition {
            frontier,
            input_placements,
            forced_placements,
            warmup_padding: Vec::new(),
            temporal_inputs,
            targets,
            output,
            model_call_required: true,
            next_frontier,
        })
    }

    fn advance_absolute(
        &mut self,
        forcing: &RealtimeFrameForcing,
    ) -> Result<RealtimeFrameTransition, RealtimeScheduleError> {
        let schedule = &self.schedule;
        let frontier = self.frontier;
        let next_frontier = checked_add(frontier, 1)?;
        let generated = schedule.generated_audio_codebooks;
        let mut input_placements = Vec::with_capacity(schedule.input_audio_codebooks);
        for codebook in generated..schedule.total_audio_codebooks {
            let position = checked_add(frontier, schedule.audio_delays()[codebook])?;
            let coordinate = coordinate(position, RealtimeFrameSlot::Audio(codebook));
            self.occupied
                .insert(coordinate, RealtimeSlotOccupancy::Input);
            input_placements.push(coordinate);
        }

        let mut forced_placements = Vec::new();
        if forcing.text {
            let position = checked_add(frontier, schedule.text_delay())?;
            let coordinate = coordinate(position, RealtimeFrameSlot::Text);
            self.occupied
                .insert(coordinate, RealtimeSlotOccupancy::Forced);
            forced_placements.push(coordinate);
        }
        for (codebook, forced) in forcing.generated_audio.iter().copied().enumerate() {
            if forced {
                let position = checked_add(frontier, schedule.audio_delays()[codebook])?;
                let coordinate = coordinate(position, RealtimeFrameSlot::Audio(codebook));
                self.occupied
                    .insert(coordinate, RealtimeSlotOccupancy::Forced);
                forced_placements.push(coordinate);
            }
        }

        let mut warmup_padding = Vec::new();
        for slot in slots(schedule.total_audio_codebooks) {
            if frontier <= schedule.delays[slot.index()] {
                let coordinate = coordinate(frontier, slot);
                self.occupied
                    .insert(coordinate, RealtimeSlotOccupancy::Padding);
                warmup_padding.push(coordinate);
            }
        }

        let mut temporal_inputs = Vec::new();
        let mut targets = Vec::new();
        let output = if frontier == 0 {
            None
        } else {
            let input_position = frontier - 1;
            temporal_inputs.reserve(schedule.delays.len());
            for slot in slots(schedule.total_audio_codebooks) {
                let coordinate = coordinate(input_position, slot);
                let occupancy = self.required_occupancy(coordinate)?;
                temporal_inputs.push(RealtimeTemporalSource::Occupied {
                    coordinate,
                    occupancy,
                });
            }
            targets.reserve(1 + schedule.depth_audio_codebooks);
            for slot in std::iter::once(RealtimeFrameSlot::Text)
                .chain((0..schedule.depth_audio_codebooks).map(RealtimeFrameSlot::Audio))
            {
                let coordinate = coordinate(frontier, slot);
                let (source, occupancy) = match self.occupied.get(&coordinate).copied() {
                    Some(RealtimeSlotOccupancy::Forced) => (
                        RealtimeTargetSource::Forced(RealtimeForcedSource::Retained),
                        RealtimeSlotOccupancy::Forced,
                    ),
                    Some(occupancy) => (RealtimeTargetSource::Existing(occupancy), occupancy),
                    None => (
                        RealtimeTargetSource::Sampled,
                        RealtimeSlotOccupancy::Sampled,
                    ),
                };
                self.occupied.insert(coordinate, occupancy);
                targets.push(RealtimeTargetDecision {
                    slot,
                    coordinate: Some(coordinate),
                    source,
                });
            }
            if frontier <= schedule.max_delay() {
                None
            } else {
                let base = frontier - schedule.max_delay();
                let coordinates = (0..generated)
                    .map(|codebook| {
                        let position = checked_add(base, schedule.audio_delays()[codebook])?;
                        let coordinate = coordinate(position, RealtimeFrameSlot::Audio(codebook));
                        self.required_occupancy(coordinate)?;
                        Ok(coordinate)
                    })
                    .collect::<Result<Vec<_>, RealtimeScheduleError>>()?;
                Some(coordinates)
            }
        };

        self.frontier = next_frontier;
        self.prune_before(next_frontier.saturating_sub(schedule.max_delay().saturating_add(1)));
        Ok(RealtimeFrameTransition {
            frontier,
            input_placements,
            forced_placements,
            warmup_padding,
            temporal_inputs,
            targets,
            output,
            model_call_required: frontier != 0,
            next_frontier,
        })
    }

    fn required_occupancy(
        &self,
        coordinate: RealtimeSlotCoordinate,
    ) -> Result<RealtimeSlotOccupancy, RealtimeScheduleError> {
        self.occupied
            .get(&coordinate)
            .copied()
            .ok_or(RealtimeScheduleError::MissingSlot { coordinate })
    }

    fn output_at_same_position(
        &self,
        position: usize,
    ) -> Result<Vec<RealtimeSlotCoordinate>, RealtimeScheduleError> {
        (0..self.schedule.generated_audio_codebooks)
            .map(|codebook| {
                let coordinate = coordinate(position, RealtimeFrameSlot::Audio(codebook));
                self.required_occupancy(coordinate)?;
                Ok(coordinate)
            })
            .collect()
    }

    fn prune_before(&mut self, minimum: usize) {
        self.occupied
            .retain(|coordinate, _| coordinate.position >= minimum);
    }
}

impl SemanticStateTransaction for RealtimeFrameScheduleState {
    type Branch = Self;
    type Error = RealtimeScheduleError;

    fn branch(&self) -> Result<Self::Branch, Self::Error> {
        Ok(self.clone())
    }

    fn commit_branch(&mut self, branch: Self::Branch) -> Result<(), Self::Error> {
        self.validate_schedule(&branch.schedule)?;
        *self = branch;
        Ok(())
    }
}

/// Invalid portable frame-schedule transition or state handoff.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum RealtimeScheduleError {
    /// State belongs to a different normalized schedule.
    #[error("realtime frame schedule state does not match the normalized schedule")]
    ScheduleMismatch,
    /// Per-generated-codebook forcing cardinality is invalid.
    #[error("realtime forcing mask has {actual} audio entries, expected {expected}")]
    ForcingCount {
        /// Expected generated-codebook count.
        expected: usize,
        /// Actual mask length.
        actual: usize,
    },
    /// A required delayed slot has not been initialized.
    #[error("realtime delayed slot {coordinate:?} is not occupied")]
    MissingSlot {
        /// Missing coordinate.
        coordinate: RealtimeSlotCoordinate,
    },
    /// Frontier plus delay exceeded portable integer coordinates.
    #[error("realtime frame coordinate overflowed")]
    CoordinateOverflow,
}

fn checked_add(left: usize, right: usize) -> Result<usize, RealtimeScheduleError> {
    left.checked_add(right)
        .ok_or(RealtimeScheduleError::CoordinateOverflow)
}

fn coordinate(position: usize, slot: RealtimeFrameSlot) -> RealtimeSlotCoordinate {
    RealtimeSlotCoordinate::new(position, slot)
}

fn slots(total_audio_codebooks: usize) -> impl Iterator<Item = RealtimeFrameSlot> {
    std::iter::once(RealtimeFrameSlot::Text)
        .chain((0..total_audio_codebooks).map(RealtimeFrameSlot::Audio))
}

/// Portable sampling controls for one realtime request.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct RealtimeSampling {
    text_temperature: f32,
    audio_temperature: f32,
    text_top_k: Option<usize>,
    audio_top_k: Option<usize>,
    seed: u64,
}

impl<'de> Deserialize<'de> for RealtimeSampling {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Raw {
            text_temperature: f32,
            audio_temperature: f32,
            text_top_k: Option<usize>,
            audio_top_k: Option<usize>,
            seed: u64,
        }
        let raw = Raw::deserialize(deserializer)?;
        Self::new(raw.text_temperature, raw.audio_temperature, raw.seed)
            .and_then(|sampling| sampling.with_top_k(raw.text_top_k, raw.audio_top_k))
            .map_err(serde::de::Error::custom)
    }
}

impl RealtimeSampling {
    /// Creates validated request-local controls.
    pub fn new(
        text_temperature: f32,
        audio_temperature: f32,
        seed: u64,
    ) -> Result<Self, RealtimeConfigError> {
        if !text_temperature.is_finite()
            || text_temperature < 0.0
            || !audio_temperature.is_finite()
            || audio_temperature < 0.0
        {
            return Err(RealtimeConfigError::SamplingTemperature {
                text: text_temperature,
                audio: audio_temperature,
            });
        }
        Ok(Self {
            text_temperature,
            audio_temperature,
            text_top_k: None,
            audio_top_k: None,
            seed,
        })
    }

    /// Applies optional top-k truncation independently to text and audio decisions.
    pub fn with_top_k(
        mut self,
        text_top_k: Option<usize>,
        audio_top_k: Option<usize>,
    ) -> Result<Self, RealtimeConfigError> {
        if text_top_k == Some(0) || audio_top_k == Some(0) {
            return Err(RealtimeConfigError::SamplingTopK {
                text: text_top_k,
                audio: audio_top_k,
            });
        }
        self.text_top_k = text_top_k;
        self.audio_top_k = audio_top_k;
        Ok(self)
    }

    /// Deterministic greedy sampling.
    pub const fn greedy() -> Self {
        Self {
            text_temperature: 0.0,
            audio_temperature: 0.0,
            text_top_k: None,
            audio_top_k: None,
            seed: 0,
        }
    }
    /// Text sampling temperature.
    pub const fn text_temperature(self) -> f32 {
        self.text_temperature
    }
    /// Audio sampling temperature.
    pub const fn audio_temperature(self) -> f32 {
        self.audio_temperature
    }
    /// Optional number of highest-scoring text tokens admitted for sampling.
    pub const fn text_top_k(self) -> Option<usize> {
        self.text_top_k
    }
    /// Optional number of highest-scoring audio tokens admitted for sampling.
    pub const fn audio_top_k(self) -> Option<usize> {
        self.audio_top_k
    }
    /// Deterministic root seed interpreted by the selected backend.
    pub const fn seed(self) -> u64 {
        self.seed
    }
    /// Whether either stream requires stochastic sampling.
    pub const fn is_stochastic(self) -> bool {
        self.text_temperature != 0.0 || self.audio_temperature != 0.0
    }
}

impl Default for RealtimeSampling {
    fn default() -> Self {
        Self::greedy()
    }
}

/// Portable host representation of one realtime input frame.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct RealtimeInputFrame {
    batch: usize,
    input_audio_tokens: Vec<i32>,
    forced_generated_audio_tokens: Option<Vec<i32>>,
    forced_generated_audio_codebooks: Option<Vec<bool>>,
    forced_text_tokens: Option<Vec<i32>>,
    retain_diagnostics: bool,
}

impl RealtimeInputFrame {
    /// Creates one batch-major encoded input-audio frame.
    pub fn new(batch: usize, input_audio_tokens: Vec<i32>) -> Self {
        Self {
            batch,
            input_audio_tokens,
            forced_generated_audio_tokens: None,
            forced_generated_audio_codebooks: None,
            forced_text_tokens: None,
            retain_diagnostics: false,
        }
    }

    /// Forces every generated-audio decision from batch-major token values.
    pub fn with_forced_generated_audio(mut self, tokens: Vec<i32>) -> Self {
        self.forced_generated_audio_tokens = Some(tokens);
        self.forced_generated_audio_codebooks = None;
        self
    }

    /// Forces selected generated-audio codebooks from batch-major token values.
    pub fn with_partially_forced_generated_audio(
        mut self,
        tokens: Vec<i32>,
        codebooks: Vec<bool>,
    ) -> Self {
        self.forced_generated_audio_tokens = Some(tokens);
        self.forced_generated_audio_codebooks = Some(codebooks);
        self
    }

    /// Forces one text decision per batch row.
    pub fn with_forced_text(mut self, tokens: Vec<i32>) -> Self {
        self.forced_text_tokens = Some(tokens);
        self
    }

    /// Requests complete decision logits in the observed step output.
    pub fn with_diagnostics(mut self) -> Self {
        self.retain_diagnostics = true;
        self
    }

    /// Batch dimension.
    pub const fn batch(&self) -> usize {
        self.batch
    }
    /// Batch-major input-audio tokens.
    pub fn input_audio_tokens(&self) -> &[i32] {
        &self.input_audio_tokens
    }
    /// Optional batch-major generated-audio forcing tokens.
    pub fn forced_generated_audio_tokens(&self) -> Option<&[i32]> {
        self.forced_generated_audio_tokens.as_deref()
    }
    /// Optional generated-codebook forcing mask.
    pub fn forced_generated_audio_codebooks(&self) -> Option<&[bool]> {
        self.forced_generated_audio_codebooks.as_deref()
    }
    /// Optional forced text tokens, one per batch row.
    pub fn forced_text_tokens(&self) -> Option<&[i32]> {
        self.forced_text_tokens.as_deref()
    }
    /// Whether complete decision diagnostics were requested.
    pub const fn retains_diagnostics(&self) -> bool {
        self.retain_diagnostics
    }
}

impl WorkDescriptor for RealtimeInputFrame {
    type Error = RealtimeInputDescriptorError;

    fn encode_descriptor(&self, output: &mut Vec<u32>) -> Result<(), Self::Error> {
        output.push(descriptor_len(self.batch)?);
        encode_i32_descriptor(&self.input_audio_tokens, output)?;
        encode_optional_i32_descriptor(self.forced_generated_audio_tokens.as_deref(), output)?;
        match self.forced_generated_audio_codebooks.as_deref() {
            Some(mask) => {
                output.push(1);
                output.push(descriptor_len(mask.len())?);
                output.extend(mask.iter().copied().map(u32::from));
            }
            None => output.push(0),
        }
        encode_optional_i32_descriptor(self.forced_text_tokens.as_deref(), output)?;
        output.push(u32::from(self.retain_diagnostics));
        Ok(())
    }
}

fn encode_i32_descriptor(
    values: &[i32],
    output: &mut Vec<u32>,
) -> Result<(), RealtimeInputDescriptorError> {
    output.push(descriptor_len(values.len())?);
    output.extend(
        values
            .iter()
            .map(|value| u32::from_ne_bytes(value.to_ne_bytes())),
    );
    Ok(())
}

fn encode_optional_i32_descriptor(
    values: Option<&[i32]>,
    output: &mut Vec<u32>,
) -> Result<(), RealtimeInputDescriptorError> {
    match values {
        Some(values) => {
            output.push(1);
            encode_i32_descriptor(values, output)
        }
        None => {
            output.push(0);
            Ok(())
        }
    }
}

fn descriptor_len(value: usize) -> Result<u32, RealtimeInputDescriptorError> {
    u32::try_from(value).map_err(|_| RealtimeInputDescriptorError { value })
}

/// A portable realtime work descriptor exceeded its stable wire representation.
#[derive(Debug, Clone, Copy, Eq, PartialEq, thiserror::Error)]
#[error("realtime input descriptor length {value} exceeds the u32 wire range")]
pub struct RealtimeInputDescriptorError {
    value: usize,
}

/// Materialized logits for one ordered realtime decision.
#[derive(Debug, Clone, PartialEq)]
pub struct RealtimeDecisionDiagnostics {
    prediction: usize,
    tensor: TensorObservation,
}

impl RealtimeDecisionDiagnostics {
    /// Creates one portable diagnostic observation.
    pub fn new(
        prediction: usize,
        shape: Vec<usize>,
        logits: Vec<f32>,
    ) -> Result<Self, ObservationError> {
        Ok(Self {
            prediction,
            tensor: TensorObservation::new(shape, TensorObservationData::F32(logits))?,
        })
    }
    /// Decision ordinal in text-then-depth order.
    pub const fn prediction(&self) -> usize {
        self.prediction
    }
    /// Complete materialized logits shape.
    pub fn shape(&self) -> &[usize] {
        self.tensor.shape()
    }
    /// Complete row-major logits values.
    pub fn logits(&self) -> &[f32] {
        let TensorObservationData::F32(values) = self.tensor.data() else {
            unreachable!("realtime diagnostics are constructed from F32 values")
        };
        values
    }

    /// General portable tensor observation for this decision.
    pub const fn tensor(&self) -> &TensorObservation {
        &self.tensor
    }
}

/// Portable host observation of one completed realtime output frame.
#[derive(Debug, Clone, PartialEq)]
pub struct RealtimeOutputFrame {
    batch: usize,
    text_tokens: Vec<i32>,
    decision_audio_tokens: Vec<i32>,
    sampled_audio_tokens: Vec<i32>,
    output_audio_tokens: Option<Vec<i32>>,
    diagnostics: Vec<RealtimeDecisionDiagnostics>,
}

impl RealtimeOutputFrame {
    /// Creates a completed host observation in batch-major order.
    pub fn new(
        batch: usize,
        text_tokens: Vec<i32>,
        decision_audio_tokens: Vec<i32>,
        sampled_audio_tokens: Vec<i32>,
        output_audio_tokens: Option<Vec<i32>>,
        diagnostics: Vec<RealtimeDecisionDiagnostics>,
    ) -> Self {
        Self {
            batch,
            text_tokens,
            decision_audio_tokens,
            sampled_audio_tokens,
            output_audio_tokens,
            diagnostics,
        }
    }
    /// Batch dimension.
    pub const fn batch(&self) -> usize {
        self.batch
    }
    /// One sampled text token per batch row.
    pub fn text_tokens(&self) -> &[i32] {
        &self.text_tokens
    }
    /// Batch-major audio tokens resolved at every ordered depth decision.
    pub fn decision_audio_tokens(&self) -> &[i32] {
        &self.decision_audio_tokens
    }
    /// Batch-major generated-audio tokens resolved for this frame.
    pub fn sampled_audio_tokens(&self) -> &[i32] {
        &self.sampled_audio_tokens
    }
    /// Optional batch-major delay-aligned output-audio tokens.
    pub fn output_audio_tokens(&self) -> Option<&[i32]> {
        self.output_audio_tokens.as_deref()
    }
    /// Ordered decision diagnostics, empty unless explicitly requested.
    pub fn diagnostics(&self) -> &[RealtimeDecisionDiagnostics] {
        &self.diagnostics
    }
}

impl WorkDescriptor for RealtimeOutputFrame {
    type Error = RealtimeInputDescriptorError;

    fn encode_descriptor(&self, output: &mut Vec<u32>) -> Result<(), Self::Error> {
        output.push(descriptor_len(self.batch)?);
        encode_i32_descriptor(&self.text_tokens, output)?;
        encode_i32_descriptor(&self.decision_audio_tokens, output)?;
        encode_i32_descriptor(&self.sampled_audio_tokens, output)?;
        encode_optional_i32_descriptor(self.output_audio_tokens.as_deref(), output)?;
        output.push(descriptor_len(self.diagnostics.len())?);
        for diagnostic in &self.diagnostics {
            output.push(descriptor_len(diagnostic.prediction())?);
            output.push(descriptor_len(diagnostic.shape().len())?);
            for &dimension in diagnostic.shape() {
                output.push(descriptor_len(dimension)?);
            }
            output.push(descriptor_len(diagnostic.logits().len())?);
            output.extend(diagnostic.logits().iter().map(|value| value.to_bits()));
        }
        Ok(())
    }
}

/// Realtime coordination failure with structured execution context.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum RealtimeError<E: std::error::Error + 'static> {
    /// Selected mechanisms rejected model, session, input, or execution.
    #[error("realtime execution failed: {0}")]
    Execution(#[source] E),
    /// Generic scheduler lifecycle or capacity failure.
    #[error(transparent)]
    Scheduler(#[from] SchedulerError),
    /// A runtime or released session belongs to a different selected realization.
    #[error("realtime model {component} does not match the scheduler model")]
    ModelMismatch {
        /// Backend-defined identity component that differs.
        component: String,
    },
    /// A bounded drain must permit at least one frame.
    #[error("realtime scheduler frame bound must be positive")]
    EmptyRunBound,
    /// Sampling cannot change while accepted frames still use the prior state.
    #[error("realtime request {request} has {queued} queued frames; drain or cancel them before changing sampling")]
    SamplingWhileQueued {
        /// Request identity.
        request: u64,
        /// Accepted queued frames.
        queued: usize,
    },
    /// At least one submitted transition failed asynchronously.
    #[error("realtime work {work:?} failed asynchronously: {message}")]
    Asynchronous {
        /// Failed work identity.
        work: WorkId,
        /// Scheduler-provided failure context.
        message: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sampling_and_speech_config_validate_portably() {
        assert!(RealtimeSampling::new(f32::NAN, 0.0, 0).is_err());
        let config = RealtimeSpeechConfig::new(
            4,
            2,
            2,
            3,
            11,
            12,
            RealtimeFrameConvention::AbsoluteDelayedSlots,
            vec![2, 0, 1, 2, 3],
        )
        .unwrap();
        assert_eq!(config.max_audio_delay(), 3);
        assert_eq!(config.max_delay(), 3);
        assert_eq!(config.text_delay(), 2);
        assert_eq!(config.generated_audio_codebooks(), 2);
        assert_eq!(
            serde_json::from_str::<RealtimeSpeechConfig>(&serde_json::to_string(&config).unwrap())
                .unwrap(),
            config
        );
        let sampling = RealtimeSampling::new(0.7, 0.9, 42)
            .unwrap()
            .with_top_k(Some(25), Some(250))
            .unwrap();
        assert_eq!(sampling.text_top_k(), Some(25));
        assert_eq!(sampling.audio_top_k(), Some(250));
        assert!(RealtimeSampling::greedy()
            .with_top_k(Some(0), None)
            .is_err());
        assert_eq!(
            serde_json::from_str::<RealtimeSampling>(&serde_json::to_string(&sampling).unwrap())
                .unwrap(),
            sampling
        );
        let frame = RealtimeInputFrame::new(1, vec![1, 2])
            .with_forced_generated_audio(vec![3, 4])
            .with_forced_text(vec![5])
            .with_diagnostics();
        assert_eq!(frame.input_audio_tokens(), [1, 2]);
        assert_eq!(frame.forced_generated_audio_tokens(), Some(&[3, 4][..]));
        assert_eq!(frame.forced_text_tokens(), Some(&[5][..]));
        assert!(frame.retains_diagnostics());
        assert!(RealtimeSpeechConfig::new(
            4,
            1,
            1,
            1,
            0,
            0,
            RealtimeFrameConvention::FeedbackAlignedHistory,
            vec![0; 5],
        )
        .is_err());
        assert!(RealtimeSpeechConfig::new(
            1,
            0,
            1,
            1,
            -1,
            0,
            RealtimeFrameConvention::FeedbackAlignedHistory,
            vec![0; 2],
        )
        .is_err());
        assert!(RealtimeSpeechConfig::new(
            1,
            0,
            1,
            1,
            0,
            0,
            RealtimeFrameConvention::FeedbackAlignedHistory,
            vec![0, MAX_REALTIME_FRAME_DELAY + 1],
        )
        .is_err());
    }

    fn released_schedule(
        convention: RealtimeFrameConvention,
        depth_audio_codebooks: usize,
    ) -> RealtimeSpeechConfig {
        RealtimeSpeechConfig::new(
            16,
            8,
            8,
            depth_audio_codebooks,
            32_000,
            2_048,
            convention,
            vec![0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
        )
        .unwrap()
    }

    #[test]
    fn released_feedback_schedule_covers_warmup_forcing_and_output_alignment() {
        let config = released_schedule(RealtimeFrameConvention::FeedbackAlignedHistory, 8);
        let mut state = RealtimeFrameScheduleState::new(config.clone());
        let first = state
            .advance(&config, &RealtimeFrameForcing::none(&config))
            .unwrap();
        assert!(first.model_call_required());
        assert_eq!(first.input_placements().len(), 8);
        assert_eq!(first.temporal_inputs().len(), 17);
        assert!(first
            .temporal_inputs()
            .iter()
            .all(|source| matches!(source, RealtimeTemporalSource::Padding(_))));
        assert_eq!(first.targets().len(), 9);
        assert!(first.output().is_none());

        let forcing = RealtimeFrameForcing::new(
            true,
            vec![true, false, false, false, false, false, false, false],
        );
        let second = state.advance(&config, &forcing).unwrap();
        assert_eq!(second.frontier(), 1);
        assert_eq!(second.next_frontier(), 2);
        assert_eq!(second.output().unwrap().len(), 8);
        assert_eq!(
            second.targets()[0].source(),
            RealtimeTargetSource::Forced(RealtimeForcedSource::CurrentInput)
        );
        assert_eq!(
            second.targets()[1].source(),
            RealtimeTargetSource::Forced(RealtimeForcedSource::CurrentInput)
        );
        assert_eq!(second.targets()[2].source(), RealtimeTargetSource::Sampled);
        assert!(matches!(
            second.temporal_inputs()[2],
            RealtimeTemporalSource::Padding(RealtimeFrameSlot::Audio(1))
        ));
        assert_eq!(
            second.output().unwrap()[0],
            RealtimeSlotCoordinate::new(0, RealtimeFrameSlot::Audio(0))
        );
    }

    #[test]
    fn released_absolute_schedule_has_initialization_step_and_absolute_targets() {
        let config = released_schedule(RealtimeFrameConvention::AbsoluteDelayedSlots, 16);
        let mut state = RealtimeFrameScheduleState::new(config.clone());
        let initialization = state
            .advance(&config, &RealtimeFrameForcing::none(&config))
            .unwrap();
        assert!(!initialization.model_call_required());
        assert!(initialization.temporal_inputs().is_empty());
        assert!(initialization.targets().is_empty());
        assert_eq!(initialization.warmup_padding().len(), 17);
        assert!(initialization.output().is_none());

        let forcing = RealtimeFrameForcing::new(
            true,
            vec![true, true, false, false, false, false, false, false],
        );
        let first_model = state.advance(&config, &forcing).unwrap();
        assert!(first_model.model_call_required());
        assert_eq!(first_model.temporal_inputs().len(), 17);
        assert!(first_model.temporal_inputs().iter().all(|source| matches!(
            source,
            RealtimeTemporalSource::Occupied {
                occupancy: RealtimeSlotOccupancy::Padding,
                ..
            }
        )));
        assert_eq!(first_model.targets().len(), 17);
        assert_eq!(
            first_model.targets()[0].source(),
            RealtimeTargetSource::Forced(RealtimeForcedSource::Retained)
        );
        assert_eq!(
            first_model.targets()[1].source(),
            RealtimeTargetSource::Forced(RealtimeForcedSource::Retained)
        );
        assert_eq!(
            first_model.targets()[2].source(),
            RealtimeTargetSource::Existing(RealtimeSlotOccupancy::Padding)
        );
        assert!(first_model.output().is_none());

        let second_model = state
            .advance(&config, &RealtimeFrameForcing::none(&config))
            .unwrap();
        assert_eq!(second_model.output().unwrap().len(), 8);
        assert_eq!(
            second_model.output().unwrap()[0],
            RealtimeSlotCoordinate::new(1, RealtimeFrameSlot::Audio(0))
        );
        assert_eq!(
            second_model.output().unwrap()[1],
            RealtimeSlotCoordinate::new(2, RealtimeFrameSlot::Audio(1))
        );
    }

    #[test]
    fn frame_schedule_branch_commit_and_rollback_are_atomic() {
        let config = released_schedule(RealtimeFrameConvention::FeedbackAlignedHistory, 8);
        let mut state = RealtimeFrameScheduleState::new(config.clone());
        let mut discarded = state.branch().unwrap();
        discarded
            .advance(&config, &RealtimeFrameForcing::none(&config))
            .unwrap();
        assert_eq!(state.frontier(), 0);
        RealtimeFrameScheduleState::discard_branch(discarded).unwrap();

        let mut committed = state.branch().unwrap();
        committed
            .advance(&config, &RealtimeFrameForcing::none(&config))
            .unwrap();
        state.commit_branch(committed).unwrap();
        assert_eq!(state.frontier(), 1);

        let other = released_schedule(RealtimeFrameConvention::AbsoluteDelayedSlots, 16);
        assert_eq!(
            state.validate_schedule(&other),
            Err(RealtimeScheduleError::ScheduleMismatch)
        );
    }

    #[test]
    fn frame_schedule_rejects_masks_missing_history_and_coordinate_overflow_atomically() {
        let config = released_schedule(RealtimeFrameConvention::AbsoluteDelayedSlots, 16);
        let mut state = RealtimeFrameScheduleState::new(config.clone());
        assert!(matches!(
            state.advance(&config, &RealtimeFrameForcing::new(false, vec![false; 7])),
            Err(RealtimeScheduleError::ForcingCount { .. })
        ));
        assert_eq!(state.frontier(), 0);

        let before = state.clone();
        state.frontier = usize::MAX;
        let overflow_before = state.clone();
        assert_eq!(
            state.advance(&config, &RealtimeFrameForcing::none(&config)),
            Err(RealtimeScheduleError::CoordinateOverflow)
        );
        assert_eq!(state, overflow_before);

        let mut missing = before;
        missing.frontier = 1;
        let missing_before = missing.clone();
        assert!(matches!(
            missing.advance(&config, &RealtimeFrameForcing::none(&config)),
            Err(RealtimeScheduleError::MissingSlot { .. })
        ));
        assert_eq!(missing, missing_before);
    }

    #[test]
    fn portable_input_frame_has_one_exact_scheduler_descriptor() {
        let frame = RealtimeInputFrame::new(1, vec![-1])
            .with_partially_forced_generated_audio(vec![5], vec![true])
            .with_forced_text(vec![3])
            .with_diagnostics();
        let mut descriptor = Vec::new();
        frame.encode_descriptor(&mut descriptor).unwrap();
        assert_eq!(
            descriptor,
            vec![1, 1, u32::MAX, 1, 1, 5, 1, 1, 1, 1, 1, 3, 1,]
        );

        let mut changed = Vec::new();
        RealtimeInputFrame::new(1, vec![0])
            .encode_descriptor(&mut changed)
            .unwrap();
        assert_ne!(descriptor, changed);
    }
}
