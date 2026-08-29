//! Backend-generic realtime token-session execution and scheduling.

use crate::{
    backend::{Completion, Submission},
    observation::{ObservationError, TensorObservation, TensorObservationData},
    scheduler::{
        RequestId, RequestStatus, Scheduler, SchedulerCapabilities, SchedulerError,
        SchedulerLimits, SchedulerReport, SemanticStateTransaction, TransitionOutput,
        WorkDescriptor, WorkId,
    },
};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, fmt::Debug, time::Instant};

/// Largest admitted text or audio delay in one portable realtime schedule.
///
/// The bound keeps every delay representable by backends whose token and cache
/// coordinates use signed 32-bit integers. Runtime frontier arithmetic remains
/// checked independently.
pub const MAX_REALTIME_FRAME_DELAY: usize = i32::MAX as usize;

/// Coordinate convention used by a realtime text-plus-audio frame schedule.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
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
pub enum RealtimeTargetSource {
    /// Caller forcing resolves this decision.
    Forced,
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
            RealtimeTargetSource::Forced
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
                        RealtimeTargetSource::Forced
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
                    Some(RealtimeSlotOccupancy::Forced) => {
                        (RealtimeTargetSource::Forced, RealtimeSlotOccupancy::Forced)
                    }
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

/// High-level contract implemented once per realtime execution backend.
///
/// Codec frames, generated outputs, cache/session state, model values, and
/// completions are opaque associated types. Core schedules complete realtime
/// steps and never models tensor operations or exposes native streams.
pub trait RealtimeBackend {
    /// Backend-owned loaded realtime model.
    type Model;
    /// Stable identity used to reject cross-model session handoff.
    type ModelIdentity: Clone + Debug + Eq;
    /// Backend-owned encoded frame or prompt transition.
    type Input: WorkDescriptor;
    /// Backend-owned generated text/audio frame.
    type Output;
    /// Request-local cache, delayed streams, sampler, and random state.
    type Session: SemanticStateTransaction<Error = Self::Error>;
    /// Exact completion retaining submitted input/output resources.
    type Completion: Completion<Error = Self::Error>;
    /// Structured backend failure.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Stable backend name used for scheduler capability telemetry.
    fn name(&self) -> &str;
    /// Returns the complete model identity.
    fn model_identity(&self, model: &Self::Model) -> Self::ModelIdentity;
    /// Returns fail-closed capabilities of the exact loaded realtime session.
    fn session_capabilities(&self, model: &Self::Model) -> crate::SessionCapabilities;
    /// Describes the first material difference between two model identities.
    fn model_identity_mismatch(
        &self,
        expected: &Self::ModelIdentity,
        actual: &Self::ModelIdentity,
    ) -> Option<String> {
        (expected != actual).then(|| "model identity".into())
    }
    /// Returns portable codec geometry.
    fn speech_config(&self, model: &Self::Model) -> RealtimeSpeechConfig;
    /// Materializes a portable encoded or forced frame for this backend.
    fn materialize_input(
        &self,
        model: &Self::Model,
        frame: &RealtimeInputFrame,
    ) -> Result<Self::Input, Self::Error>;
    /// Materializes tokens and requested diagnostics from one completed output.
    fn observe_output(&self, output: &Self::Output) -> Result<RealtimeOutputFrame, Self::Error>;
    /// Creates one request-local session.
    fn create_session(
        &self,
        model: &Self::Model,
        sampling: RealtimeSampling,
    ) -> Result<Self::Session, Self::Error>;
    /// Validates a released session before attaching it to this model.
    fn validate_session(
        &self,
        model: &Self::Model,
        session: &Self::Session,
    ) -> Result<(), Self::Error>;
    /// Validates one backend-owned input frame against model geometry.
    fn validate_input(&self, model: &Self::Model, input: &Self::Input) -> Result<(), Self::Error>;
    /// Returns the stable batch dimension of a validated input.
    fn input_batch_size(&self, input: &Self::Input) -> usize;
    /// Replaces request-local sampling and randomness state.
    fn set_sampling(
        &self,
        session: &mut Self::Session,
        sampling: RealtimeSampling,
    ) -> Result<(), Self::Error>;
    /// Submits one complete temporal/depth transition.
    fn submit_step(
        &self,
        model: &mut Self::Model,
        session: &mut <Self::Session as SemanticStateTransaction>::Branch,
        input: &Self::Input,
    ) -> Result<Submission<Self::Output, Self::Completion>, Self::Error>;
    /// Number of backend resources explicitly retained by a completion.
    fn retained_resources(&self, _completion: &Self::Completion) -> usize {
        0
    }
}

/// Materialization contract for an architecture-prepared realtime model.
///
/// Architecture code inspects the artifact and produces [`Self::Preparation`]
/// before this boundary. The selected backend consumes that neutral plan and
/// owns only device-specific materialization.
pub trait RealtimeModelLoadingBackend: RealtimeBackend + Sized {
    /// Architecture-owned, backend-neutral artifact preparation.
    type Preparation;
    /// Backend-specific materialization policy.
    type LoadOptions;

    /// Materializes one architecture-prepared realtime model.
    fn materialize_realtime_model(
        &self,
        preparation: Self::Preparation,
        options: Self::LoadOptions,
    ) -> Result<Self::Model, Self::Error>;
}

/// Materializes an architecture-prepared realtime model using default policy.
pub fn load_realtime_model<B>(
    backend: B,
    preparation: B::Preparation,
) -> Result<RealtimeModel<B>, B::Error>
where
    B: RealtimeModelLoadingBackend,
    B::LoadOptions: Default,
{
    load_realtime_model_with_options(backend, preparation, B::LoadOptions::default())
}

/// Materializes an architecture-prepared realtime model using explicit policy.
pub fn load_realtime_model_with_options<B: RealtimeModelLoadingBackend>(
    backend: B,
    preparation: B::Preparation,
    options: B::LoadOptions,
) -> Result<RealtimeModel<B>, B::Error> {
    let model = backend.materialize_realtime_model(preparation, options)?;
    Ok(RealtimeModel::new(backend, model))
}

/// Selected realtime backend and its loaded model.
pub struct RealtimeModel<B: RealtimeBackend> {
    backend: B,
    model: B::Model,
}

impl<B: RealtimeBackend> RealtimeModel<B> {
    /// Binds one loaded model to its execution backend.
    pub const fn new(backend: B, model: B::Model) -> Self {
        Self { backend, model }
    }
    /// Borrows the selected backend.
    pub const fn backend(&self) -> &B {
        &self.backend
    }
    /// Borrows the backend-owned model.
    pub const fn model(&self) -> &B::Model {
        &self.model
    }
    /// Mutably borrows the backend-owned model.
    pub fn model_mut(&mut self) -> &mut B::Model {
        &mut self.model
    }
    /// Portable codec-token geometry.
    pub fn speech_config(&self) -> RealtimeSpeechConfig {
        self.backend.speech_config(&self.model)
    }
    /// Fail-closed capabilities of the exact loaded realtime session.
    pub fn session_capabilities(&self) -> crate::SessionCapabilities {
        self.backend.session_capabilities(&self.model)
    }
    /// Consumes the runtime into backend and model values.
    pub fn into_parts(self) -> (B, B::Model) {
        (self.backend, self.model)
    }
}

/// Request-local realtime state released from a scheduler.
pub struct RealtimeSession<B: RealtimeBackend> {
    model_identity: B::ModelIdentity,
    state: B::Session,
    batch_size: Option<usize>,
}

/// Unpublished request-local branch passed to one backend submission.
pub struct RealtimeSessionBranch<B: RealtimeBackend> {
    state: <B::Session as SemanticStateTransaction>::Branch,
    batch_size: Option<usize>,
}

impl<B: RealtimeBackend> RealtimeSession<B> {
    /// Borrows backend-owned request state.
    pub const fn state(&self) -> &B::Session {
        &self.state
    }
    /// Mutably borrows backend-owned request state.
    pub fn state_mut(&mut self) -> &mut B::Session {
        &mut self.state
    }
    /// Committed batch dimension, when at least one frame was accepted.
    pub const fn batch_size(&self) -> Option<usize> {
        self.batch_size
    }
}

impl<B: RealtimeBackend> SemanticStateTransaction for RealtimeSession<B> {
    type Branch = RealtimeSessionBranch<B>;
    type Error = B::Error;

    fn branch(&self) -> Result<Self::Branch, Self::Error> {
        Ok(RealtimeSessionBranch {
            state: self.state.branch()?,
            batch_size: self.batch_size,
        })
    }

    fn commit_branch(&mut self, branch: Self::Branch) -> Result<(), Self::Error> {
        self.state.commit_branch(branch.state)?;
        self.batch_size = branch.batch_size;
        Ok(())
    }

    fn discard_branch(branch: Self::Branch) -> Result<(), Self::Error> {
        B::Session::discard_branch(branch.state)
    }
}

struct RealtimeTransition<B: RealtimeBackend> {
    backend_name: String,
    retained_resources: usize,
    output: B::Output,
    completion: B::Completion,
}

impl<B: RealtimeBackend> TransitionOutput for RealtimeTransition<B> {
    type Error = B::Error;

    fn is_complete(&self) -> Result<bool, Self::Error> {
        self.completion.is_complete()
    }
    fn backend_name(&self) -> Option<String> {
        Some(self.backend_name.clone())
    }
    fn retained_resources(&self) -> usize {
        self.retained_resources
    }
}

/// One committed realtime transition and its scheduler identity.
pub struct RealtimeCompletedStep<O> {
    work: WorkId,
    output: O,
}

impl<O> RealtimeCompletedStep<O> {
    /// Scheduler-assigned work identity.
    pub const fn work(&self) -> WorkId {
        self.work
    }
    /// Borrows the backend-owned generated frame.
    pub const fn output(&self) -> &O {
        &self.output
    }
    /// Consumes this completion.
    pub fn into_parts(self) -> (WorkId, O) {
        (self.work, self.output)
    }
}

/// Realtime coordination failure with structured backend context.
#[derive(Debug, thiserror::Error)]
pub enum RealtimeError<E: std::error::Error + 'static> {
    /// Selected backend rejected model, session, input, or execution.
    #[error("realtime backend failed: {0}")]
    Backend(#[source] E),
    /// Generic scheduler lifecycle or capacity failure.
    #[error(transparent)]
    Scheduler(#[from] SchedulerError),
    /// A runtime or released session belongs to a different model.
    #[error("realtime model {component} does not match the scheduler model")]
    ModelMismatch {
        /// Backend-defined identity component that differs.
        component: String,
    },
    /// A request changed its batch size after its first accepted frame.
    #[error("realtime request {request} changed batch size from {expected} to {actual}")]
    BatchSize {
        /// Request identity.
        request: u64,
        /// Committed batch size.
        expected: usize,
        /// New input batch size.
        actual: usize,
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

/// Fair bounded realtime scheduler generic over the selected backend.
pub struct RealtimeScheduler<B: RealtimeBackend> {
    model_identity: B::ModelIdentity,
    scheduler: Scheduler<B::Input, RealtimeSession<B>, RealtimeTransition<B>>,
}

impl<B: RealtimeBackend> RealtimeScheduler<B> {
    /// Binds an empty scheduler to one selected model.
    pub fn new(
        model: &RealtimeModel<B>,
        limits: SchedulerLimits,
    ) -> Result<Self, RealtimeError<B::Error>> {
        Ok(Self {
            model_identity: model.backend.model_identity(&model.model),
            scheduler: Scheduler::new(limits)?,
        })
    }

    fn validate_model(&self, model: &RealtimeModel<B>) -> Result<(), RealtimeError<B::Error>> {
        let actual = model.backend.model_identity(&model.model);
        if let Some(component) = model
            .backend
            .model_identity_mismatch(&self.model_identity, &actual)
        {
            return Err(RealtimeError::ModelMismatch { component });
        }
        Ok(())
    }

    /// Registers a request with fresh backend-owned state.
    pub fn register_request(
        &mut self,
        model: &RealtimeModel<B>,
        request: RequestId,
        sampling: RealtimeSampling,
    ) -> Result<(), RealtimeError<B::Error>> {
        self.validate_model(model)?;
        self.scheduler.validate_registration(request)?;
        let state = model
            .backend
            .create_session(&model.model, sampling)
            .map_err(RealtimeError::Backend)?;
        self.scheduler.register(
            request,
            RealtimeSession {
                model_identity: self.model_identity.clone(),
                state,
                batch_size: None,
            },
        )?;
        Ok(())
    }

    /// Registers a previously released request session.
    pub fn register_request_with_session(
        &mut self,
        model: &RealtimeModel<B>,
        request: RequestId,
        session: RealtimeSession<B>,
    ) -> Result<(), RealtimeError<B::Error>> {
        self.validate_model(model)?;
        self.scheduler.validate_registration(request)?;
        if let Some(component) = model
            .backend
            .model_identity_mismatch(&self.model_identity, &session.model_identity)
        {
            return Err(RealtimeError::ModelMismatch { component });
        }
        model
            .backend
            .validate_session(&model.model, &session.state)
            .map_err(RealtimeError::Backend)?;
        self.scheduler.register(request, session)?;
        Ok(())
    }

    /// Enqueues one encoded or forced frame.
    pub fn enqueue(
        &mut self,
        model: &RealtimeModel<B>,
        request: RequestId,
        input: B::Input,
    ) -> Result<WorkId, RealtimeError<B::Error>> {
        self.enqueue_with_deadline(model, request, input, None)
    }

    /// Enqueues one frame with an optional absolute deadline.
    pub fn enqueue_with_deadline(
        &mut self,
        model: &RealtimeModel<B>,
        request: RequestId,
        input: B::Input,
        deadline: Option<Instant>,
    ) -> Result<WorkId, RealtimeError<B::Error>> {
        self.validate_model(model)?;
        model
            .backend
            .validate_input(&model.model, &input)
            .map_err(RealtimeError::Backend)?;
        let batch = model.backend.input_batch_size(&input);
        self.validate_batch(request, batch)?;
        let work = self
            .scheduler
            .enqueue_with_deadline(request, input, deadline)?;
        self.scheduler
            .request_state_mut(request)?
            .batch_size
            .get_or_insert(batch);
        Ok(work)
    }

    /// Atomically enqueues ordered frames.
    pub fn enqueue_batch(
        &mut self,
        model: &RealtimeModel<B>,
        request: RequestId,
        inputs: Vec<B::Input>,
    ) -> Result<Vec<WorkId>, RealtimeError<B::Error>> {
        self.validate_model(model)?;
        let mut expected = self
            .scheduler
            .request_state(request)
            .ok_or(SchedulerError::UnknownRequest(request))?
            .batch_size;
        for input in &inputs {
            model
                .backend
                .validate_input(&model.model, input)
                .map_err(RealtimeError::Backend)?;
            let actual = model.backend.input_batch_size(input);
            if let Some(expected) = expected {
                if actual != expected {
                    return Err(RealtimeError::BatchSize {
                        request: request.value(),
                        expected,
                        actual,
                    });
                }
            } else {
                expected = Some(actual);
            }
        }
        let work = self.scheduler.enqueue_batch(request, inputs)?;
        if let Some(batch) = expected {
            self.scheduler
                .request_state_mut(request)?
                .batch_size
                .get_or_insert(batch);
        }
        Ok(work)
    }

    fn validate_batch(
        &self,
        request: RequestId,
        actual: usize,
    ) -> Result<(), RealtimeError<B::Error>> {
        let state = self
            .scheduler
            .request_state(request)
            .ok_or(SchedulerError::UnknownRequest(request))?;
        if let Some(expected) = state.batch_size {
            if expected != actual {
                return Err(RealtimeError::BatchSize {
                    request: request.value(),
                    expected,
                    actual,
                });
            }
        }
        Ok(())
    }

    /// Advances one unbounded fair scheduling turn.
    pub fn run_queued(
        &mut self,
        model: &mut RealtimeModel<B>,
    ) -> Result<Vec<RealtimeCompletedStep<B::Output>>, RealtimeError<B::Error>> {
        self.run_bounded(model, usize::MAX)
    }

    /// Advances at most `max_frames` fair-ordered transitions.
    pub fn run_bounded(
        &mut self,
        model: &mut RealtimeModel<B>,
        max_frames: usize,
    ) -> Result<Vec<RealtimeCompletedStep<B::Output>>, RealtimeError<B::Error>> {
        self.validate_model(model)?;
        if max_frames == 0 {
            return Err(RealtimeError::EmptyRunBound);
        }
        let now = Instant::now();
        let mut progress = self.scheduler.poll_completions(now);
        self.scheduler.prepare_bounded(max_frames, now)?;
        let backend_name = model.backend.name().to_owned();
        let backend = &model.backend;
        let backend_model = &mut model.model;
        progress.newly_submitted = self.scheduler.submit_prepared(
            now,
            |_, input, session| -> Result<RealtimeTransition<B>, B::Error> {
                let submission = backend.submit_step(backend_model, &mut session.state, input)?;
                let retained_resources = backend.retained_resources(&submission.completion);
                Ok(RealtimeTransition {
                    backend_name: backend_name.clone(),
                    retained_resources,
                    output: submission.output,
                    completion: submission.completion,
                })
            },
        )?;
        let completed = self.scheduler.poll_completions(now);
        progress.committed.extend(completed.committed);
        progress.failed.extend(completed.failed);
        if let Some((work, failure)) = progress.failed.first() {
            return Err(RealtimeError::Asynchronous {
                work: *work,
                message: failure.to_string(),
            });
        }
        Ok(progress
            .committed
            .into_iter()
            .map(|(work, _, transition)| RealtimeCompletedStep {
                work,
                output: transition.output,
            })
            .collect())
    }

    /// Completes one request and drops its backend session.
    pub fn finish_request(&mut self, request: RequestId) -> Result<(), RealtimeError<B::Error>> {
        self.scheduler.finish(request)?;
        Ok(())
    }
    /// Cancels one request and discards queued frames.
    pub fn cancel_request(&mut self, request: RequestId) -> Result<(), RealtimeError<B::Error>> {
        self.scheduler.cancel(request)?;
        Ok(())
    }
    /// Releases an idle request for persistence or resumption.
    pub fn release_request(
        &mut self,
        request: RequestId,
    ) -> Result<RealtimeSession<B>, RealtimeError<B::Error>> {
        Ok(self.scheduler.release(request)?)
    }
    /// Removes a terminal identity for explicit reuse.
    pub fn forget_terminal_request(
        &mut self,
        request: RequestId,
    ) -> Result<RequestStatus, RealtimeError<B::Error>> {
        Ok(self.scheduler.forget_terminal(request)?)
    }
    /// Lifecycle state for a known request.
    pub fn request_status(&self, request: RequestId) -> Option<RequestStatus> {
        self.scheduler.request_status(request)
    }
    /// Queued frame count for one request.
    pub fn queued_for_request(&self, request: RequestId) -> usize {
        self.scheduler.queued_for_request(request)
    }
    /// Replaces sampling controls for an idle active request.
    pub fn set_request_sampling(
        &mut self,
        model: &RealtimeModel<B>,
        request: RequestId,
        sampling: RealtimeSampling,
    ) -> Result<(), RealtimeError<B::Error>> {
        self.validate_model(model)?;
        let queued = self.scheduler.queued_for_request(request);
        if queued != 0 {
            return Err(RealtimeError::SamplingWhileQueued {
                request: request.value(),
                queued,
            });
        }
        let state = self.scheduler.request_state_mut(request)?;
        model
            .backend
            .set_sampling(&mut state.state, sampling)
            .map_err(RealtimeError::Backend)
    }
    /// Generic occupancy and lifecycle telemetry.
    pub fn report(&self) -> SchedulerReport {
        self.scheduler.report()
    }
    /// Configured bounds and observed backend capabilities.
    pub fn capabilities(&self) -> SchedulerCapabilities {
        self.scheduler.capabilities()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::convert::Infallible;

    #[derive(Clone)]
    struct MockSession {
        step: u32,
        sampling: RealtimeSampling,
    }
    impl SemanticStateTransaction for MockSession {
        type Branch = Self;
        type Error = Infallible;
        fn branch(&self) -> Result<Self, Self::Error> {
            Ok(self.clone())
        }
        fn commit_branch(&mut self, branch: Self) -> Result<(), Self::Error> {
            *self = branch;
            Ok(())
        }
    }

    #[derive(Clone)]
    struct Frame(Vec<u32>);
    impl WorkDescriptor for Frame {
        type Error = Infallible;
        fn encode_descriptor(&self, output: &mut Vec<u32>) -> Result<(), Self::Error> {
            output.extend_from_slice(&self.0);
            Ok(())
        }
    }

    struct Done;
    impl Completion for Done {
        type Error = Infallible;
        fn is_complete(&self) -> Result<bool, Self::Error> {
            Ok(true)
        }
        fn wait(&self) -> Result<(), Self::Error> {
            Ok(())
        }
    }

    struct MockBackend;
    impl RealtimeBackend for MockBackend {
        type Model = u64;
        type ModelIdentity = u64;
        type Input = Frame;
        type Output = u32;
        type Session = MockSession;
        type Completion = Done;
        type Error = Infallible;

        fn name(&self) -> &str {
            "mock-realtime"
        }
        fn model_identity(&self, model: &u64) -> u64 {
            *model
        }
        fn session_capabilities(&self, _: &u64) -> crate::SessionCapabilities {
            crate::SessionCapabilities {
                persistent_cache: true,
                output_observation: true,
                activation_inspection: false,
            }
        }
        fn speech_config(&self, _: &u64) -> RealtimeSpeechConfig {
            RealtimeSpeechConfig::new(
                2,
                1,
                1,
                1,
                0,
                0,
                RealtimeFrameConvention::FeedbackAlignedHistory,
                vec![0, 0, 1],
            )
            .unwrap()
        }
        fn materialize_input(
            &self,
            _: &u64,
            frame: &RealtimeInputFrame,
        ) -> Result<Frame, Infallible> {
            Ok(Frame(
                frame
                    .input_audio_tokens()
                    .iter()
                    .map(|token| *token as u32)
                    .collect(),
            ))
        }
        fn observe_output(&self, output: &u32) -> Result<RealtimeOutputFrame, Infallible> {
            Ok(RealtimeOutputFrame::new(
                1,
                vec![*output as i32],
                Vec::new(),
                Vec::new(),
                None,
                Vec::new(),
            ))
        }
        fn create_session(
            &self,
            _: &u64,
            sampling: RealtimeSampling,
        ) -> Result<MockSession, Infallible> {
            Ok(MockSession { step: 0, sampling })
        }
        fn validate_session(&self, _: &u64, _: &MockSession) -> Result<(), Infallible> {
            Ok(())
        }
        fn validate_input(&self, _: &u64, _: &Frame) -> Result<(), Infallible> {
            Ok(())
        }
        fn input_batch_size(&self, input: &Frame) -> usize {
            input.0.len()
        }
        fn set_sampling(
            &self,
            session: &mut MockSession,
            sampling: RealtimeSampling,
        ) -> Result<(), Infallible> {
            session.sampling = sampling;
            Ok(())
        }
        fn submit_step(
            &self,
            model: &mut u64,
            session: &mut MockSession,
            input: &Frame,
        ) -> Result<Submission<u32, Done>, Infallible> {
            session.step += 1;
            Ok(Submission {
                output: *model as u32 + session.step + input.0.iter().sum::<u32>(),
                completion: Done,
            })
        }
    }

    impl RealtimeModelLoadingBackend for MockBackend {
        type Preparation = u64;
        type LoadOptions = u64;

        fn materialize_realtime_model(
            &self,
            preparation: Self::Preparation,
            _: Self::LoadOptions,
        ) -> Result<Self::Model, Self::Error> {
            Ok(preparation)
        }
    }

    #[test]
    fn selected_backend_materializes_architecture_preparation() {
        let model = load_realtime_model_with_options(MockBackend, 37, 0).unwrap();
        assert_eq!(*model.model(), 37);
        assert_eq!(model.backend().name(), "mock-realtime");
        assert_eq!(
            model.session_capabilities(),
            crate::SessionCapabilities {
                persistent_cache: true,
                output_observation: true,
                activation_inspection: false,
            }
        );
    }

    #[test]
    fn mock_backend_runs_fair_realtime_sessions_without_accelerator_types() {
        let mut model = RealtimeModel::new(MockBackend, 10);
        let limits = SchedulerLimits::with_execution_bounds(2, 4, 2, 2, 1, usize::MAX).unwrap();
        let mut scheduler = RealtimeScheduler::new(&model, limits).unwrap();
        let first = RequestId::new(1);
        let second = RequestId::new(2);
        scheduler
            .register_request(&model, first, RealtimeSampling::greedy())
            .unwrap();
        scheduler
            .register_request(&model, second, RealtimeSampling::greedy())
            .unwrap();
        scheduler.enqueue(&model, first, Frame(vec![1])).unwrap();
        scheduler.enqueue(&model, second, Frame(vec![2])).unwrap();
        assert!(matches!(
            scheduler.set_request_sampling(
                &model,
                first,
                RealtimeSampling::new(0.5, 0.5, 7).unwrap()
            ),
            Err(RealtimeError::SamplingWhileQueued { .. })
        ));
        assert_eq!(
            scheduler
                .run_queued(&mut model)
                .unwrap()
                .into_iter()
                .map(|step| step.into_parts().1)
                .collect::<Vec<_>>(),
            vec![12, 13]
        );
        let updated = RealtimeSampling::new(0.5, 0.5, 7).unwrap();
        scheduler
            .set_request_sampling(&model, first, updated)
            .unwrap();
        assert_eq!(
            scheduler.release_request(first).unwrap().state().sampling,
            updated
        );
    }

    #[test]
    fn portable_scheduler_lifecycle_rejects_mismatch_and_preserves_resumed_state() {
        let mut model = RealtimeModel::new(MockBackend, 10);
        let other_model = RealtimeModel::new(MockBackend, 11);
        let limits = SchedulerLimits::with_execution_bounds(2, 4, 2, 2, 1, usize::MAX).unwrap();
        let mut scheduler = RealtimeScheduler::new(&model, limits).unwrap();

        let cancelled = RequestId::new(10);
        scheduler
            .register_request(&model, cancelled, RealtimeSampling::greedy())
            .unwrap();
        scheduler
            .enqueue(&model, cancelled, Frame(vec![1]))
            .unwrap();
        assert!(matches!(
            scheduler.enqueue(&model, cancelled, Frame(vec![1, 2])),
            Err(RealtimeError::BatchSize {
                request: 10,
                expected: 1,
                actual: 2,
            })
        ));
        assert_eq!(scheduler.queued_for_request(cancelled), 1);
        scheduler.cancel_request(cancelled).unwrap();
        assert_eq!(
            scheduler.request_status(cancelled),
            Some(RequestStatus::Cancelled)
        );
        assert_eq!(scheduler.queued_for_request(cancelled), 0);

        assert!(matches!(
            scheduler.register_request(
                &other_model,
                RequestId::new(11),
                RealtimeSampling::greedy()
            ),
            Err(RealtimeError::ModelMismatch { .. })
        ));

        let original = RequestId::new(20);
        scheduler
            .register_request(&model, original, RealtimeSampling::greedy())
            .unwrap();
        scheduler.enqueue(&model, original, Frame(vec![2])).unwrap();
        assert_eq!(scheduler.run_queued(&mut model).unwrap()[0].output(), &13);
        let released = scheduler.release_request(original).unwrap();
        assert_eq!(released.state().step, 1);
        assert_eq!(released.batch_size(), Some(1));

        let resumed = RequestId::new(21);
        scheduler
            .register_request_with_session(&model, resumed, released)
            .unwrap();
        scheduler.enqueue(&model, resumed, Frame(vec![3])).unwrap();
        assert_eq!(scheduler.run_queued(&mut model).unwrap()[0].output(), &15);
        let released = scheduler.release_request(resumed).unwrap();
        assert_eq!(released.state().step, 2);

        let mut other_scheduler = RealtimeScheduler::new(&other_model, limits).unwrap();
        assert!(matches!(
            other_scheduler.register_request_with_session(
                &other_model,
                RequestId::new(22),
                released
            ),
            Err(RealtimeError::ModelMismatch { .. })
        ));
    }

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
        assert_eq!(second.targets()[0].source(), RealtimeTargetSource::Forced);
        assert_eq!(second.targets()[1].source(), RealtimeTargetSource::Forced);
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
            RealtimeTargetSource::Forced
        );
        assert_eq!(
            first_model.targets()[1].source(),
            RealtimeTargetSource::Forced
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
}
