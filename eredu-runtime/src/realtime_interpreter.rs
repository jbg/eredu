//! Family-blind interpretation of portable realtime schedule transitions.

use crate::host_metadata::funded_vec;
use eredu_core::realtime::RealtimeSlotTable;

use eredu_core::{
    RealtimeForcedSource, RealtimeFrameScheduleState, RealtimeFrameSlot, RealtimeFrameTransition,
    RealtimeScheduleError, RealtimeSpeechConfig, RealtimeTargetSource, RealtimeTemporalSource,
};

use crate::{
    MaterializedRealtimeInput, PredictionDirective, RealtimePayloadHistory,
    RealtimePayloadHistoryError,
};

/// Narrow tensor mechanisms needed to populate and resolve scheduled slots.
pub trait RealtimeFrameTensorMechanisms {
    /// Opaque/native token payload.
    type Tensor: Clone;
    /// Mechanism failure.
    type Error;

    /// Copies an actual descriptor through its original host source.
    fn clone_with_host_source(&mut self,_value:&Self::Tensor,_funding:&eredu_core::HostMetadataFunding)
        ->Result<Self::Tensor,eredu_core::BackendFailure> {
        Err(eredu_core::HostMetadataFundingError::Unavailable.into())
    }

    /// Selects one token column while retaining the batch and singleton axes.
    fn column(&mut self, matrix: &Self::Tensor, column: usize)
        -> Result<Self::Tensor, Self::Error>;

    /// Creates one batch-by-one column containing an architecture-selected token.
    fn filled_column(&mut self, token: i32, batch: usize) -> Result<Self::Tensor, Self::Error>;

    /// Stacks batch-by-one columns into one batch-by-column matrix.
    fn stack_columns(
        &mut self,
        columns: &[Self::Tensor],
        batch: usize,
    ) -> Result<Self::Tensor, Self::Error>;
}

mod tensor;
pub use tensor::NeuralRealtimeFrameTensorMechanisms;

/// Fully resolved temporal inputs and ordered target directives for one step.
pub struct PreparedRealtimeFrame<T> {
    schedule: RealtimeSpeechConfig,
    transition: RealtimeFrameTransition,
    temporal: Vec<T>,
    directives: Vec<PredictionDirective<T>>,
    batch: usize,
    retain_diagnostics: bool,
}

impl<T> PreparedRealtimeFrame<T> {
    /// Returns the authoritative portable schedule transition.
    pub const fn transition(&self) -> &RealtimeFrameTransition {
        &self.transition
    }

    /// Returns temporal text-plus-audio payloads in canonical slot order.
    pub fn temporal(&self) -> &[T] {
        &self.temporal
    }

    /// Returns ordered text-then-depth forced or sampled directives.
    pub fn directives(&self) -> &[PredictionDirective<T>] {
        &self.directives
    }

    /// Returns the validated positive batch size.
    pub const fn batch(&self) -> usize {
        self.batch
    }

    /// Returns whether complete ordered logits diagnostics are requested.
    pub const fn retains_diagnostics(&self) -> bool {
        self.retain_diagnostics
    }

    /// Consumes the step into schedule, inputs, directives, and output policy.
    pub fn into_parts(
        self,
    ) -> (
        RealtimeSpeechConfig,
        RealtimeFrameTransition,
        Vec<T>,
        Vec<PredictionDirective<T>>,
        usize,
        bool,
    ) {
        (
            self.schedule,
            self.transition,
            self.temporal,
            self.directives,
            self.batch,
            self.retain_diagnostics,
        )
    }
}

/// Opaque output tensors produced by one completely interpreted frame.
pub struct CompletedRealtimeFrame<T, D> {
    text: T,
    decision_audio: T,
    sampled_audio: T,
    aligned_audio: Option<T>,
    diagnostics: Vec<D>,
}

impl<T, D> CompletedRealtimeFrame<T, D> {
    /// Returns one text-token column.
    pub const fn text(&self) -> &T {
        &self.text
    }

    /// Returns every depth decision in codebook order.
    pub const fn decision_audio(&self) -> &T {
        &self.decision_audio
    }

    /// Returns generated-audio decisions in generated-codebook order.
    pub const fn sampled_audio(&self) -> &T {
        &self.sampled_audio
    }

    /// Returns delay-aligned generated audio when the schedule exposes it.
    pub const fn aligned_audio(&self) -> Option<&T> {
        self.aligned_audio.as_ref()
    }

    /// Returns ordered per-decision diagnostics.
    pub fn diagnostics(&self) -> &[D] {
        &self.diagnostics
    }

    /// Consumes every opaque output and diagnostic.
    pub fn into_parts(self) -> (T, T, T, Option<T>, Vec<D>) {
        (
            self.text,
            self.decision_audio,
            self.sampled_audio,
            self.aligned_audio,
            self.diagnostics,
        )
    }
}

/// Records sampled decisions, resolves aligned output, and prunes history.
fn host_error<E>(cause:eredu_core::HostMetadataFundingError)->RealtimeFrameInterpretationError<E> {
    RealtimeFrameInterpretationError::History(RealtimePayloadHistoryError::HostMetadata(cause))
}

pub fn complete_realtime_frame<M, D>(
    schedule: &RealtimeSpeechConfig,
    history: &mut RealtimePayloadHistory<M::Tensor>,
    prepared: PreparedRealtimeFrame<M::Tensor>,
    decisions: Vec<M::Tensor>,
    diagnostics: Vec<D>,
    mechanisms: &mut M,
) -> Result<CompletedRealtimeFrame<M::Tensor, D>, RealtimeFrameInterpretationError<M::Error>>
where
    M: RealtimeFrameTensorMechanisms,
{
    history
        .validate_schedule(schedule)
        .map_err(RealtimeFrameInterpretationError::History)?;
    let funding=history.host_funding().cloned();
    if let Some(funding)=&funding {
        funding.reserve_metadata(completed_host_header_bytes::<M::Tensor,D,M::Error>())
            .map_err(host_error)?;
    }
    let (prepared_schedule, transition, _, directives, batch, retain_diagnostics) =
        prepared.into_parts();
    if &prepared_schedule != schedule {
        return Err(RealtimeFrameInterpretationError::PreparedScheduleMismatch);
    }
    if !transition.model_call_required() {
        if !decisions.is_empty() {
            return Err(RealtimeFrameInterpretationError::DecisionCount {
                expected: 0,
                actual: decisions.len(),
            });
        }
        if !diagnostics.is_empty() {
            return Err(RealtimeFrameInterpretationError::DiagnosticCount {
                expected: 0,
                actual: diagnostics.len(),
            });
        }
        let text = mechanisms
            .filled_column(schedule.text_padding_token(), batch)
            .map_err(RealtimeFrameInterpretationError::Mechanism)?;
        let padding = mechanisms
            .filled_column(schedule.audio_padding_token(), batch)
            .map_err(RealtimeFrameInterpretationError::Mechanism)?;
        let mut sampled_columns=funded_vec(schedule.generated_audio_codebooks(),funding.as_ref()).map_err(host_error)?;
        for _ in 0..schedule.generated_audio_codebooks() {sampled_columns.push(clone_frame_tensor(&padding,funding.as_ref(),mechanisms)?);}
        let sampled_audio = mechanisms
            .stack_columns(&sampled_columns, batch)
            .map_err(RealtimeFrameInterpretationError::Mechanism)?;
        let decision_audio = mechanisms
            .stack_columns(&[], batch)
            .map_err(RealtimeFrameInterpretationError::Mechanism)?;
        return Ok(CompletedRealtimeFrame {
            text,
            decision_audio,
            sampled_audio,
            aligned_audio: None,
            diagnostics,
        });
    }
    if decisions.len() != directives.len() {
        return Err(RealtimeFrameInterpretationError::DecisionCount {
            expected: directives.len(),
            actual: decisions.len(),
        });
    }
    let expected_diagnostics = usize::from(retain_diagnostics) * decisions.len();
    if diagnostics.len() != expected_diagnostics {
        return Err(RealtimeFrameInterpretationError::DiagnosticCount {
            expected: expected_diagnostics,
            actual: diagnostics.len(),
        });
    }
    let text = clone_frame_tensor(decisions.first()
        .ok_or(RealtimeFrameInterpretationError::MissingTextDecision)?,funding.as_ref(),mechanisms)?;
    let generated = schedule.generated_audio_codebooks();
    let generated_end = 1usize
        .checked_add(generated)
        .ok_or(RealtimeFrameInterpretationError::DecisionCountOverflow { generated })?;
    if generated_end > decisions.len() {
        return Err(RealtimeFrameInterpretationError::DecisionCount {
            expected: generated_end,
            actual: decisions.len(),
        });
    }

    let mut history_branch = clone_frame_history(history,mechanisms)?;
    let mut resolved_targets=funded_vec(transition.targets().iter()
        .filter(|target|target.coordinate().is_some()).count(),funding.as_ref()).map_err(host_error)?;
    for (target,payload) in transition.targets().iter().zip(&decisions) {
        if let Some(coordinate)=target.coordinate() {resolved_targets.push((coordinate,clone_frame_tensor(payload,funding.as_ref(),mechanisms)?));}
    }
    history_branch
        .overwrite_many(schedule, resolved_targets)
        .map_err(RealtimeFrameInterpretationError::History)?;

    let decision_audio = mechanisms
        .stack_columns(&decisions[1..], batch)
        .map_err(RealtimeFrameInterpretationError::Mechanism)?;
    let sampled_audio = mechanisms
        .stack_columns(&decisions[1..generated_end], batch)
        .map_err(RealtimeFrameInterpretationError::Mechanism)?;
    let aligned_audio = transition
        .output()
        .map(|coordinates| {
            let required=history_branch.resolve_required(schedule,coordinates.iter().copied())
                .map_err(RealtimeFrameInterpretationError::History)?;
            let mut columns=funded_vec(required.len(),funding.as_ref()).map_err(host_error)?;
            for value in required {columns.push(clone_frame_tensor(value,funding.as_ref(),mechanisms)?);}
            mechanisms
                .stack_columns(&columns, batch)
                .map_err(RealtimeFrameInterpretationError::Mechanism)
        })
        .transpose()?;
    history_branch
        .prune_for_next_frontier(schedule, transition.next_frontier())
        .map_err(RealtimeFrameInterpretationError::History)?;
    *history = history_branch;
    Ok(CompletedRealtimeFrame {
        text,
        decision_audio,
        sampled_audio,
        aligned_audio,
        diagnostics,
    })
}

/// Advances one neutral schedule and resolves all pre-model opaque payloads.
///
/// Schedule and history publication is atomic. Sampled target values are
/// recorded later, after the ordered model decision driver completes.
pub fn prepare_realtime_frame<M>(
    schedule: &RealtimeSpeechConfig,
    schedule_state: &mut RealtimeFrameScheduleState,
    history: &mut RealtimePayloadHistory<M::Tensor>,
    input: &MaterializedRealtimeInput<M::Tensor>,
    mechanisms: &mut M,
) -> Result<PreparedRealtimeFrame<M::Tensor>, RealtimeFrameInterpretationError<M::Error>>
where
    M: RealtimeFrameTensorMechanisms,
{
    if input.schedule() != schedule {
        return Err(RealtimeFrameInterpretationError::InputScheduleMismatch);
    }
    schedule_state
        .validate_schedule(schedule)
        .map_err(RealtimeFrameInterpretationError::Schedule)?;
    history
        .validate_schedule(schedule)
        .map_err(RealtimeFrameInterpretationError::History)?;
    let funding=history.host_funding().cloned();
    let mut schedule_branch = match &funding {
        Some(funding)=>schedule_state.try_clone_with_host_source(funding)
            .map_err(RealtimeScheduleError::HostMetadata)
            .map_err(RealtimeFrameInterpretationError::Schedule)?,
        None=>schedule_state.clone(),
    };
    let mut history_branch = clone_frame_history(history,mechanisms)?;
    let transition = match &funding {
        Some(funding)=>schedule_branch.advance_with_host_source(schedule,input.forcing(),funding),
        None=>schedule_branch.advance(schedule,input.forcing()),
    }.map_err(RealtimeFrameInterpretationError::Schedule)?;

    if let Some(funding)=&funding {
        funding.reserve_metadata(prepared_host_header_bytes::<M::Tensor,M::Error>())
            .map_err(host_error)?;
    }
    let mut insertions = RealtimeSlotTable::new();
    if let Some(funding)=&funding {
        let count=transition.input_placements().len().checked_add(transition.forced_placements().len())
            .and_then(|n|n.checked_add(transition.warmup_padding().len()))
            .ok_or_else(||host_error(eredu_core::HostMetadataFundingError::Overflow))?;
        insertions.reserve_prepared(count,funding).map_err(host_error)?;
    }
    for (column, coordinate) in transition.input_placements().iter().copied().enumerate() {
        let payload = mechanisms
            .column(input.input_audio(), column)
            .map_err(RealtimeFrameInterpretationError::Mechanism)?;
        insertions.insert_checked(coordinate, payload).map_err(|(cause,_)|host_error(cause))?;
    }
    for coordinate in transition.forced_placements().iter().copied() {
        insertions.insert_checked(
            coordinate,
            forced_payload(input, coordinate.slot(), funding.as_ref(), mechanisms)?,
        ).map_err(|(cause,_)|host_error(cause))?;
    }
    for coordinate in transition.warmup_padding().iter().copied() {
        let payload = mechanisms
            .filled_column(padding_token(schedule, coordinate.slot())?, input.batch())
            .map_err(RealtimeFrameInterpretationError::Mechanism)?;
        insertions.insert_checked(coordinate, payload).map_err(|(cause,_)|host_error(cause))?;
    }
    history_branch
        .overwrite_many(schedule, insertions)
        .map_err(RealtimeFrameInterpretationError::History)?;

    let mut temporal=funded_vec(transition.temporal_inputs().len(),funding.as_ref()).map_err(host_error)?;
    for source in transition.temporal_inputs() {
        let value=match source {
            RealtimeTemporalSource::Padding(slot) => mechanisms
                .filled_column(padding_token(schedule, *slot)?, input.batch())
                .map_err(RealtimeFrameInterpretationError::Mechanism),
            RealtimeTemporalSource::Occupied { coordinate, .. } => clone_frame_tensor(
                history_branch.required(schedule,*coordinate).map_err(RealtimeFrameInterpretationError::History)?,
                funding.as_ref(),mechanisms),
            _ => Err(RealtimeFrameInterpretationError::UnsupportedTemporalSource),
        }?;
        temporal.push(value);
    }
    let mut directives=funded_vec(transition.targets().len(),funding.as_ref()).map_err(host_error)?;
    for target in transition.targets() {
        let value=match target.source() {
            RealtimeTargetSource::Sampled => Ok(PredictionDirective::Sample),
            RealtimeTargetSource::Forced(RealtimeForcedSource::CurrentInput) => {
                forced_payload(input, target.slot(), funding.as_ref(), mechanisms).map(PredictionDirective::Force)
            }
            RealtimeTargetSource::Forced(RealtimeForcedSource::Retained) => target
                .coordinate()
                .ok_or(RealtimeFrameInterpretationError::MissingTargetCoordinate)
                .and_then(|coordinate| {
                    clone_frame_tensor(history_branch.required(schedule,coordinate)
                        .map_err(RealtimeFrameInterpretationError::History)?,funding.as_ref(),mechanisms)
                })
                .map(PredictionDirective::Force),
            RealtimeTargetSource::Existing(_) => target
                .coordinate()
                .ok_or(RealtimeFrameInterpretationError::MissingTargetCoordinate)
                .and_then(|coordinate| {
                    clone_frame_tensor(history_branch.required(schedule,coordinate)
                        .map_err(RealtimeFrameInterpretationError::History)?,funding.as_ref(),mechanisms)
                })
                .map(PredictionDirective::Force),
            _ => Err(RealtimeFrameInterpretationError::UnsupportedTargetSource),
        }?;
        directives.push(value);
    }

    let prepared_schedule=match &funding {
        Some(funding)=>schedule.try_clone_with_host_source(funding).map_err(host_error)?,
        None=>schedule.clone(),
    };
    *schedule_state = schedule_branch;
    *history = history_branch;
    Ok(PreparedRealtimeFrame {
        schedule: prepared_schedule,
        transition,
        temporal,
        directives,
        batch: input.batch(),
        retain_diagnostics: input.retains_diagnostics(),
    })
}

fn forced_payload<M>(
    input: &MaterializedRealtimeInput<M::Tensor>,
    slot: RealtimeFrameSlot,
    funding:Option<&eredu_core::HostMetadataFunding>,
    mechanisms: &mut M,
) -> Result<M::Tensor, RealtimeFrameInterpretationError<M::Error>>
where
    M: RealtimeFrameTensorMechanisms,
{
    match slot {
        RealtimeFrameSlot::Text => clone_frame_tensor(input.forced_text()
            .ok_or(RealtimeFrameInterpretationError::MissingForcedPayload { slot })?,funding,mechanisms),
        RealtimeFrameSlot::Audio(codebook) => mechanisms
            .column(
                input
                    .forced_audio()
                    .ok_or(RealtimeFrameInterpretationError::MissingForcedPayload { slot })?,
                codebook,
            )
            .map_err(RealtimeFrameInterpretationError::Mechanism),
        _ => Err(RealtimeFrameInterpretationError::UnsupportedSlot { slot }),
    }
}

fn padding_token<E>(
    schedule: &RealtimeSpeechConfig,
    slot: RealtimeFrameSlot,
) -> Result<i32, RealtimeFrameInterpretationError<E>> {
    match slot {
        RealtimeFrameSlot::Text => Ok(schedule.text_padding_token()),
        RealtimeFrameSlot::Audio(codebook) if codebook < schedule.total_audio_codebooks() => {
            Ok(schedule.audio_padding_token())
        }
        _ => Err(RealtimeFrameInterpretationError::UnsupportedSlot { slot }),
    }
}

/// Stable failure while interpreting one portable schedule transition.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum RealtimeFrameInterpretationError<E> {
    /// A retained native descriptor copy failed under its original account.
    #[error(transparent)]
    HostClone(eredu_core::BackendFailure),
    /// Opaque input was validated under a different normalized schedule.
    #[error("materialized realtime input does not match the normalized schedule")]
    InputScheduleMismatch,
    /// Prepared frame was resolved under a different normalized schedule.
    #[error("prepared realtime frame does not match the normalized schedule")]
    PreparedScheduleMismatch,
    /// Portable schedule state rejected the transition.
    #[error(transparent)]
    Schedule(RealtimeScheduleError),
    /// Delayed-coordinate payload history rejected an operation.
    #[error(transparent)]
    History(RealtimePayloadHistoryError),
    /// A narrow opaque tensor mechanism failed.
    #[error("realtime frame tensor mechanism failed")]
    Mechanism(#[source] E),
    /// A forced schedule target had no validated opaque payload.
    #[error("realtime forced slot {slot:?} has no payload")]
    MissingForcedPayload {
        /// Missing forced slot.
        slot: RealtimeFrameSlot,
    },
    /// An existing target did not retain its required coordinate.
    #[error("realtime existing target has no coordinate")]
    MissingTargetCoordinate,
    /// Model execution returned the wrong number of ordered decisions.
    #[error("realtime model returned {actual} decisions, expected {expected}")]
    DecisionCount {
        /// Required decision count.
        expected: usize,
        /// Actual decision count.
        actual: usize,
    },
    /// Generated decision cardinality overflowed its text prefix.
    #[error("realtime generated decision count {generated} overflowed")]
    DecisionCountOverflow {
        /// Generated-audio decision count.
        generated: usize,
    },
    /// A model-required transition did not return its leading text decision.
    #[error("realtime model returned no text decision")]
    MissingTextDecision,
    /// Diagnostic retention did not match the exact ordered decision count.
    #[error("realtime model returned {actual} diagnostics, expected {expected}")]
    DiagnosticCount {
        /// Required diagnostic count.
        expected: usize,
        /// Actual diagnostic count.
        actual: usize,
    },
    /// A schedule exposed a temporal source unknown to this runtime version.
    #[error("unsupported realtime temporal source")]
    UnsupportedTemporalSource,
    /// A schedule exposed a target source unknown to this runtime version.
    #[error("unsupported realtime target source")]
    UnsupportedTargetSource,
    /// A schedule exposed a frame slot unknown to this runtime version.
    #[error("unsupported realtime frame slot {slot:?}")]
    UnsupportedSlot {
        /// Unsupported slot.
        slot: RealtimeFrameSlot,
    },
}

#[cfg(test)]
mod tests {
    use std::convert::Infallible;

    use eredu_core::{
        RealtimeFrameConvention, RealtimeInputFrame, RealtimeSlotCoordinate, RealtimeSlotOccupancy,
    };

    use crate::{
        RealtimeHostTokenMaterializer, RealtimeIngressContract, RealtimePayloadContract,
        RealtimePayloadGeneration, RealtimePayloadOwnerIdentity, TokenDomain,
    };

    use super::*;

    #[derive(Debug, Clone, Eq, PartialEq)]
    struct Matrix {
        values: Vec<i32>,
        shape: [usize; 2],
    }

    #[derive(Default)]
    struct Mechanisms {
        calls: usize,
    }

    impl RealtimeHostTokenMaterializer for Mechanisms {
        type Tensor = Matrix;
        type Error = Infallible;

        fn materialize_i32(
            &mut self,
            values: &[i32],
            shape: [usize; 2],
        ) -> Result<Self::Tensor, Self::Error> {
            self.calls += 1;
            Ok(Matrix {
                values: values.to_vec(),
                shape,
            })
        }
    }

    impl RealtimeFrameTensorMechanisms for Mechanisms {
        type Tensor = Matrix;
        type Error = Infallible;

        fn column(
            &mut self,
            matrix: &Self::Tensor,
            column: usize,
        ) -> Result<Self::Tensor, Self::Error> {
            self.calls += 1;
            Ok(Matrix {
                values: matrix
                    .values
                    .chunks_exact(matrix.shape[1])
                    .map(|row| row[column])
                    .collect(),
                shape: [matrix.shape[0], 1],
            })
        }

        fn filled_column(&mut self, token: i32, batch: usize) -> Result<Self::Tensor, Self::Error> {
            self.calls += 1;
            Ok(Matrix {
                values: vec![token; batch],
                shape: [batch, 1],
            })
        }

        fn stack_columns(
            &mut self,
            columns: &[Self::Tensor],
            batch: usize,
        ) -> Result<Self::Tensor, Self::Error> {
            self.calls += 1;
            let mut values = Vec::with_capacity(batch * columns.len());
            for row in 0..batch {
                values.extend(columns.iter().map(|column| column.values[row]));
            }
            Ok(Matrix {
                values,
                shape: [batch, columns.len()],
            })
        }
    }

    fn schedule(convention: RealtimeFrameConvention) -> RealtimeSpeechConfig {
        RealtimeSpeechConfig::new(2, 1, 1, 1, 9, 8, convention, vec![0, 0, 1]).unwrap()
    }

    fn payload_history(schedule: &RealtimeSpeechConfig) -> RealtimePayloadHistory<Matrix> {
        RealtimePayloadHistory::with_contract(
            RealtimePayloadContract::new(
                schedule.clone(),
                1,
                TokenDomain::new(10),
                TokenDomain::new(9),
                RealtimePayloadGeneration::new(1).unwrap(),
                RealtimePayloadOwnerIdentity::new(1).unwrap(),
            )
            .unwrap(),
        )
    }

    fn input(
        schedule: &RealtimeSpeechConfig,
        mechanisms: &mut Mechanisms,
    ) -> MaterializedRealtimeInput<Matrix> {
        materialize_input(
            schedule,
            RealtimeInputFrame::new(1, vec![4])
                .with_forced_text(vec![3])
                .with_partially_forced_generated_audio(vec![5], vec![false]),
            mechanisms,
        )
    }

    fn materialize_input(
        schedule: &RealtimeSpeechConfig,
        input: RealtimeInputFrame,
        mechanisms: &mut Mechanisms,
    ) -> MaterializedRealtimeInput<Matrix> {
        RealtimeIngressContract::new(schedule.clone(), TokenDomain::new(10), TokenDomain::new(9))
            .unwrap()
            .validate(&input)
            .unwrap()
            .materialize(mechanisms)
            .unwrap()
    }

    fn column(value: i32) -> Matrix {
        Matrix {
            values: vec![value],
            shape: [1, 1],
        }
    }

    #[test]
    fn feedback_transition_resolves_temporal_and_ordered_targets_once() {
        let schedule = schedule(RealtimeFrameConvention::FeedbackAlignedHistory);
        let mut schedule_state = RealtimeFrameScheduleState::new(schedule.clone());
        let mut history = payload_history(&schedule);
        let mut mechanisms = Mechanisms::default();
        let input = input(&schedule, &mut mechanisms);

        let first = prepare_realtime_frame(
            &schedule,
            &mut schedule_state,
            &mut history,
            &input,
            &mut mechanisms,
        )
        .unwrap();
        assert!(first.transition().model_call_required());
        assert_eq!(first.temporal().len(), 3);
        assert_eq!(first.directives().len(), 2);
        assert!(matches!(
            first.directives()[0],
            PredictionDirective::Force(_)
        ));
        assert!(matches!(first.directives()[1], PredictionDirective::Sample));
        assert_eq!(schedule_state.frontier(), 1);
        assert_eq!(
            history
                .required(
                    &schedule,
                    RealtimeSlotCoordinate::new(0, RealtimeFrameSlot::Audio(1))
                )
                .unwrap()
                .values,
            vec![4]
        );
        let completed = complete_realtime_frame(
            &schedule,
            &mut history,
            first,
            vec![column(3), column(6)],
            Vec::<Matrix>::new(),
            &mut mechanisms,
        )
        .unwrap();
        assert_eq!(completed.text().values, vec![3]);
        assert_eq!(completed.decision_audio().values, vec![6]);
        assert_eq!(completed.sampled_audio().values, vec![6]);
        assert!(completed.aligned_audio().is_none());
        assert_eq!(
            history
                .required(
                    &schedule,
                    RealtimeSlotCoordinate::new(0, RealtimeFrameSlot::Audio(0))
                )
                .unwrap()
                .values,
            vec![6]
        );
    }

    #[test]
    fn absolute_initialization_populates_padding_without_model_work() {
        let schedule = schedule(RealtimeFrameConvention::AbsoluteDelayedSlots);
        let mut schedule_state = RealtimeFrameScheduleState::new(schedule.clone());
        let mut history = payload_history(&schedule);
        let mut mechanisms = Mechanisms::default();
        let input = input(&schedule, &mut mechanisms);
        let prepared = prepare_realtime_frame(
            &schedule,
            &mut schedule_state,
            &mut history,
            &input,
            &mut mechanisms,
        )
        .unwrap();
        assert!(!prepared.transition().model_call_required());
        assert!(prepared.temporal().is_empty());
        assert!(prepared.directives().is_empty());
        assert_eq!(
            schedule_state.occupancy(RealtimeSlotCoordinate::new(0, RealtimeFrameSlot::Text)),
            Some(RealtimeSlotOccupancy::Padding)
        );
        assert_eq!(history.len(), 4);
        let completed = complete_realtime_frame(
            &schedule,
            &mut history,
            prepared,
            vec![],
            Vec::<Matrix>::new(),
            &mut mechanisms,
        )
        .unwrap();
        assert_eq!(completed.text().values, vec![9]);
        assert!(completed.decision_audio().values.is_empty());
        assert_eq!(completed.sampled_audio().values, vec![8]);
    }

    #[test]
    fn absolute_delayed_frames_retain_forcing_overwrite_targets_and_align_output() {
        let schedule = RealtimeSpeechConfig::new(
            2,
            1,
            1,
            1,
            9,
            8,
            RealtimeFrameConvention::AbsoluteDelayedSlots,
            vec![1, 1, 0],
        )
        .unwrap();
        let mut schedule_state = RealtimeFrameScheduleState::new(schedule.clone());
        let mut history = payload_history(&schedule);
        let mut mechanisms = Mechanisms::default();

        let initialization_input = materialize_input(
            &schedule,
            RealtimeInputFrame::new(1, vec![4]),
            &mut mechanisms,
        );
        let initialization = prepare_realtime_frame(
            &schedule,
            &mut schedule_state,
            &mut history,
            &initialization_input,
            &mut mechanisms,
        )
        .unwrap();
        complete_realtime_frame(
            &schedule,
            &mut history,
            initialization,
            vec![],
            Vec::<Matrix>::new(),
            &mut mechanisms,
        )
        .unwrap();

        let second_input = materialize_input(
            &schedule,
            RealtimeInputFrame::new(1, vec![6])
                .with_forced_text(vec![7])
                .with_forced_generated_audio(vec![6]),
            &mut mechanisms,
        );
        let forced = prepare_realtime_frame(
            &schedule,
            &mut schedule_state,
            &mut history,
            &second_input,
            &mut mechanisms,
        )
        .unwrap();
        assert!(matches!(
            &forced.directives()[0],
            PredictionDirective::Force(matrix) if matrix.values == vec![9]
        ));
        assert!(matches!(
            &forced.directives()[1],
            PredictionDirective::Force(matrix) if matrix.values == vec![8]
        ));
        let forced_completed = complete_realtime_frame(
            &schedule,
            &mut history,
            forced,
            vec![column(9), column(8)],
            Vec::<Matrix>::new(),
            &mut mechanisms,
        )
        .unwrap();
        assert!(forced_completed.aligned_audio().is_none());

        let next_input = materialize_input(
            &schedule,
            RealtimeInputFrame::new(1, vec![7]),
            &mut mechanisms,
        );
        let next = prepare_realtime_frame(
            &schedule,
            &mut schedule_state,
            &mut history,
            &next_input,
            &mut mechanisms,
        )
        .unwrap();
        assert_eq!(
            next.temporal()
                .iter()
                .map(|matrix| matrix.values[0])
                .collect::<Vec<_>>(),
            vec![9, 8, 6]
        );
        assert!(matches!(
            &next.directives()[0],
            PredictionDirective::Force(matrix) if matrix.values == vec![7]
        ));
        assert!(matches!(
            &next.directives()[1],
            PredictionDirective::Force(matrix) if matrix.values == vec![6]
        ));
        let next_completed = complete_realtime_frame(
            &schedule,
            &mut history,
            next,
            vec![column(7), column(6)],
            Vec::<Matrix>::new(),
            &mut mechanisms,
        )
        .unwrap();
        assert_eq!(next_completed.aligned_audio().unwrap().values, vec![6]);
        assert_eq!(next_completed.sampled_audio().values, vec![6]);
    }

    #[test]
    fn feedback_frames_overwrite_current_targets_and_align_prior_output() {
        let schedule = schedule(RealtimeFrameConvention::FeedbackAlignedHistory);
        let mut schedule_state = RealtimeFrameScheduleState::new(schedule.clone());
        let mut history = payload_history(&schedule);
        let mut mechanisms = Mechanisms::default();

        let first_input = materialize_input(
            &schedule,
            RealtimeInputFrame::new(1, vec![4])
                .with_forced_text(vec![3])
                .with_forced_generated_audio(vec![5]),
            &mut mechanisms,
        );
        let first = prepare_realtime_frame(
            &schedule,
            &mut schedule_state,
            &mut history,
            &first_input,
            &mut mechanisms,
        )
        .unwrap();
        complete_realtime_frame(
            &schedule,
            &mut history,
            first,
            vec![column(3), column(5)],
            Vec::<Matrix>::new(),
            &mut mechanisms,
        )
        .unwrap();

        let second_input = materialize_input(
            &schedule,
            RealtimeInputFrame::new(1, vec![7]).with_forced_text(vec![6]),
            &mut mechanisms,
        );
        let second = prepare_realtime_frame(
            &schedule,
            &mut schedule_state,
            &mut history,
            &second_input,
            &mut mechanisms,
        )
        .unwrap();
        assert_eq!(
            second
                .temporal()
                .iter()
                .map(|matrix| matrix.values[0])
                .collect::<Vec<_>>(),
            vec![3, 5, 8]
        );
        let second_completed = complete_realtime_frame(
            &schedule,
            &mut history,
            second,
            vec![column(6), column(2)],
            Vec::<Matrix>::new(),
            &mut mechanisms,
        )
        .unwrap();
        assert_eq!(second_completed.aligned_audio().unwrap().values, vec![5]);
        assert_eq!(
            history
                .required(
                    &schedule,
                    RealtimeSlotCoordinate::new(1, RealtimeFrameSlot::Text),
                )
                .unwrap()
                .values,
            vec![6]
        );
        assert_eq!(
            history
                .required(
                    &schedule,
                    RealtimeSlotCoordinate::new(1, RealtimeFrameSlot::Audio(0)),
                )
                .unwrap()
                .values,
            vec![2]
        );
    }

    #[test]
    fn input_schedule_mismatch_fails_before_tensor_mechanisms() {
        let input_schedule = schedule(RealtimeFrameConvention::FeedbackAlignedHistory);
        let attempted_schedule = schedule(RealtimeFrameConvention::AbsoluteDelayedSlots);
        let mut mechanisms = Mechanisms::default();
        let input = materialize_input(
            &input_schedule,
            RealtimeInputFrame::new(1, vec![4]),
            &mut mechanisms,
        );
        let calls_before_prepare = mechanisms.calls;
        let mut schedule_state = RealtimeFrameScheduleState::new(attempted_schedule.clone());
        let mut history = payload_history(&attempted_schedule);

        let result = prepare_realtime_frame(
            &attempted_schedule,
            &mut schedule_state,
            &mut history,
            &input,
            &mut mechanisms,
        );

        assert!(matches!(
            result,
            Err(RealtimeFrameInterpretationError::InputScheduleMismatch)
        ));
        assert_eq!(mechanisms.calls, calls_before_prepare);
        assert_eq!(schedule_state.frontier(), 0);
        assert!(history.is_empty());
    }

    #[test]
    fn depth_targets_beyond_generated_codebooks_resolve_existing_input() {
        let schedule = RealtimeSpeechConfig::new(
            3,
            2,
            1,
            3,
            9,
            8,
            RealtimeFrameConvention::FeedbackAlignedHistory,
            vec![0, 0, 0, 0],
        )
        .unwrap();
        let mut schedule_state = RealtimeFrameScheduleState::new(schedule.clone());
        let mut history = payload_history(&schedule);
        let mut mechanisms = Mechanisms::default();
        let input = materialize_input(
            &schedule,
            RealtimeInputFrame::new(1, vec![4, 7]),
            &mut mechanisms,
        );

        let prepared = prepare_realtime_frame(
            &schedule,
            &mut schedule_state,
            &mut history,
            &input,
            &mut mechanisms,
        )
        .unwrap();

        assert_eq!(prepared.directives().len(), 4);
        assert!(matches!(
            &prepared.directives()[0],
            PredictionDirective::Sample
        ));
        assert!(matches!(
            &prepared.directives()[1],
            PredictionDirective::Sample
        ));
        assert!(matches!(
            &prepared.directives()[2],
            PredictionDirective::Force(matrix) if matrix.values == vec![4]
        ));
        assert!(matches!(
            &prepared.directives()[3],
            PredictionDirective::Force(matrix) if matrix.values == vec![7]
        ));
    }
}

pub(crate) fn prepared_host_header_bytes<T,E>()->usize {
    std::mem::size_of::<PreparedRealtimeFrame<T>>()*2
        +std::mem::size_of::<Result<PreparedRealtimeFrame<T>,RealtimeFrameInterpretationError<E>>>()
}
pub(crate) fn completed_host_header_bytes<T,D,E>()->usize {
    std::mem::size_of::<CompletedRealtimeFrame<T,D>>()*2
        +std::mem::size_of::<Result<CompletedRealtimeFrame<T,D>,RealtimeFrameInterpretationError<E>>>()
}

pub(crate) fn clone_frame_tensor<M:RealtimeFrameTensorMechanisms>(value:&M::Tensor,
    funding:Option<&eredu_core::HostMetadataFunding>,mechanisms:&mut M)
    ->Result<M::Tensor,RealtimeFrameInterpretationError<M::Error>> {
    match funding {Some(funding)=>mechanisms.clone_with_host_source(value,funding)
        .map_err(RealtimeFrameInterpretationError::HostClone),None=>Ok(value.clone())}
}
pub(crate) fn clone_frame_history<M:RealtimeFrameTensorMechanisms>(history:&RealtimePayloadHistory<M::Tensor>,mechanisms:&mut M)
    ->Result<RealtimePayloadHistory<M::Tensor>,RealtimeFrameInterpretationError<M::Error>> {
    history.clone_with_payload_source(&mut |value,funding|mechanisms.clone_with_host_source(value,funding))
        .map_err(RealtimeFrameInterpretationError::HostClone)
}
