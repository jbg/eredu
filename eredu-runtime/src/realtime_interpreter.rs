//! Family-blind interpretation of portable realtime schedule transitions.

use std::collections::BTreeMap;

use eredu_core::{
    RealtimeChannelId, RealtimeForcedSource, RealtimeFrameForcing, RealtimeFrameScheduleState,
    RealtimeFrameTransition, RealtimeScheduleError, RealtimeSpeechConfig, RealtimeTargetSource,
    RealtimeTemporalSource,
};

use crate::{
    CompletedRealtimeFrame, MaterializedRealtimeInput, PredictionDirective,
    RealtimeCompletedFrameError, RealtimeFrameChannel, RealtimeIngressContract,
    RealtimePayloadHistory, RealtimePayloadHistoryError,
};

/// Narrow tensor mechanisms needed to populate and resolve scheduled slots.
pub trait RealtimeFrameTensorMechanisms {
    /// Opaque/native token payload.
    type Tensor: Clone;
    /// Mechanism failure.
    type Error;

    /// Creates one batch-by-one column containing an architecture-selected token.
    fn filled_column(&mut self, token: i32, batch: usize) -> Result<Self::Tensor, Self::Error>;
}

/// Fully resolved temporal inputs and ordered target directives for one step.
pub struct PreparedRealtimeFrame<T> {
    schedule: RealtimeSpeechConfig,
    channels: eredu_core::RealtimeChannelLayout,
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
        eredu_core::RealtimeChannelLayout,
        RealtimeFrameTransition,
        Vec<T>,
        Vec<PredictionDirective<T>>,
        usize,
        bool,
    ) {
        (
            self.schedule,
            self.channels,
            self.transition,
            self.temporal,
            self.directives,
            self.batch,
            self.retain_diagnostics,
        )
    }
}

/// Records sampled decisions, resolves aligned output, and prunes history.
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
    let (prepared_schedule, channels, transition, _, directives, batch, retain_diagnostics) =
        prepared.into_parts();
    history
        .validate_schedule(&channels)
        .map_err(RealtimeFrameInterpretationError::History)?;
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
        let mut values = vec![RealtimeFrameChannel::new(
            RealtimeChannelId::new(0),
            None,
            text,
        )];
        values.extend((1..=schedule.generated_audio_codebooks()).map(|index| {
            RealtimeFrameChannel::new(RealtimeChannelId::new(index), None, padding.clone())
        }));
        return CompletedRealtimeFrame::new(
            channels,
            batch,
            values,
            Vec::new(),
            Vec::new(),
            diagnostics,
        )
        .map_err(RealtimeFrameInterpretationError::Completed);
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

    let mut history_branch = history.clone();
    let resolved_targets = transition
        .targets()
        .iter()
        .zip(&decisions)
        .filter_map(|(target, payload)| {
            target
                .coordinate()
                .map(|coordinate| (coordinate, payload.clone()))
        })
        .collect::<Vec<_>>();
    history_branch
        .overwrite_many(&channels, resolved_targets)
        .map_err(RealtimeFrameInterpretationError::History)?;

    let values = transition
        .targets()
        .iter()
        .zip(decisions)
        .map(|(target, payload)| {
            RealtimeFrameChannel::new(target.slot(), target.coordinate(), payload)
        })
        .collect();
    let decision_channels = transition
        .targets()
        .iter()
        .map(|target| target.slot())
        .collect();
    let aligned = transition
        .output()
        .unwrap_or_default()
        .iter()
        .map(|&coordinate| {
            history_branch
                .required(&channels, coordinate)
                .cloned()
                .map(|payload| {
                    RealtimeFrameChannel::new(coordinate.slot(), Some(coordinate), payload)
                })
                .map_err(RealtimeFrameInterpretationError::History)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let completed = CompletedRealtimeFrame::new(
        channels.clone(),
        batch,
        values,
        aligned,
        decision_channels,
        diagnostics,
    )
    .map_err(RealtimeFrameInterpretationError::Completed)?;
    history_branch
        .prune_for_next_frontier(&channels, transition.next_frontier())
        .map_err(RealtimeFrameInterpretationError::History)?;
    *history = history_branch;
    Ok(completed)
}

/// Advances one neutral schedule and resolves all pre-model opaque payloads.
///
/// Schedule and history publication is atomic. Sampled target values are
/// recorded later, after the ordered model decision driver completes.
pub fn prepare_realtime_frame<M>(
    contract: &RealtimeIngressContract,
    schedule_state: &mut RealtimeFrameScheduleState,
    history: &mut RealtimePayloadHistory<M::Tensor>,
    input: &MaterializedRealtimeInput<M::Tensor>,
    mechanisms: &mut M,
) -> Result<PreparedRealtimeFrame<M::Tensor>, RealtimeFrameInterpretationError<M::Error>>
where
    M: RealtimeFrameTensorMechanisms,
{
    if input.channels() != contract.channels() {
        return Err(RealtimeFrameInterpretationError::InputScheduleMismatch);
    }
    let schedule = contract.schedule();
    for &(slot, _) in input.inputs() {
        if slot.index() <= schedule.generated_audio_codebooks()
            || slot.index() > schedule.total_audio_codebooks()
        {
            return Err(RealtimeFrameInterpretationError::InputRole {
                slot,
                role: "ordinary input",
            });
        }
    }
    for &(slot, _) in input.forced_values() {
        if slot.index() > schedule.generated_audio_codebooks() {
            return Err(RealtimeFrameInterpretationError::InputRole {
                slot,
                role: "forced output",
            });
        }
    }
    for index in schedule.generated_audio_codebooks() + 1..=schedule.total_audio_codebooks() {
        let slot = RealtimeChannelId::new(index);
        if input.input(slot).is_none() {
            return Err(RealtimeFrameInterpretationError::MissingInputPayload { slot });
        }
    }
    let channels = input.channels();
    schedule_state
        .validate_schedule(schedule)
        .map_err(RealtimeFrameInterpretationError::Schedule)?;
    history
        .validate_schedule(channels)
        .map_err(RealtimeFrameInterpretationError::History)?;
    let mut schedule_branch = schedule_state.clone();
    let mut history_branch = history.clone();
    let transition = schedule_branch
        .advance(
            schedule,
            &RealtimeFrameForcing::new(
                input.forced(RealtimeChannelId::new(0)).is_some(),
                (1..=schedule.generated_audio_codebooks())
                    .map(|index| input.forced(RealtimeChannelId::new(index)).is_some())
                    .collect(),
            ),
        )
        .map_err(RealtimeFrameInterpretationError::Schedule)?;

    let mut insertions = BTreeMap::new();
    for coordinate in transition.input_placements().iter().copied() {
        let payload = input.input(coordinate.slot()).cloned().ok_or(
            RealtimeFrameInterpretationError::MissingInputPayload {
                slot: coordinate.slot(),
            },
        )?;
        insertions.insert(coordinate, payload);
    }
    for coordinate in transition.forced_placements().iter().copied() {
        insertions.insert(
            coordinate,
            forced_payload::<_, M::Error>(input, coordinate.slot())?,
        );
    }
    for coordinate in transition.warmup_padding().iter().copied() {
        let payload = mechanisms
            .filled_column(padding_token(schedule, coordinate.slot())?, input.batch())
            .map_err(RealtimeFrameInterpretationError::Mechanism)?;
        insertions.insert(coordinate, payload);
    }
    history_branch
        .overwrite_many(channels, insertions)
        .map_err(RealtimeFrameInterpretationError::History)?;

    let temporal = transition
        .temporal_inputs()
        .iter()
        .map(|source| match source {
            RealtimeTemporalSource::Padding(slot) => mechanisms
                .filled_column(padding_token(schedule, *slot)?, input.batch())
                .map_err(RealtimeFrameInterpretationError::Mechanism),
            RealtimeTemporalSource::Occupied { coordinate, .. } => history_branch
                .required(channels, *coordinate)
                .cloned()
                .map_err(RealtimeFrameInterpretationError::History),
            _ => Err(RealtimeFrameInterpretationError::UnsupportedTemporalSource),
        })
        .collect::<Result<Vec<_>, _>>()?;
    let directives = transition
        .targets()
        .iter()
        .map(|target| match target.source() {
            RealtimeTargetSource::Sampled => Ok(PredictionDirective::Sample),
            RealtimeTargetSource::Forced(RealtimeForcedSource::CurrentInput) => {
                forced_payload::<_, M::Error>(input, target.slot()).map(PredictionDirective::Force)
            }
            RealtimeTargetSource::Forced(RealtimeForcedSource::Retained) => target
                .coordinate()
                .ok_or(RealtimeFrameInterpretationError::MissingTargetCoordinate)
                .and_then(|coordinate| {
                    history_branch
                        .required(channels, coordinate)
                        .cloned()
                        .map_err(RealtimeFrameInterpretationError::History)
                })
                .map(PredictionDirective::Force),
            RealtimeTargetSource::Existing(_) => target
                .coordinate()
                .ok_or(RealtimeFrameInterpretationError::MissingTargetCoordinate)
                .and_then(|coordinate| {
                    history_branch
                        .required(channels, coordinate)
                        .cloned()
                        .map_err(RealtimeFrameInterpretationError::History)
                })
                .map(PredictionDirective::Force),
            _ => Err(RealtimeFrameInterpretationError::UnsupportedTargetSource),
        })
        .collect::<Result<Vec<_>, _>>()?;

    *schedule_state = schedule_branch;
    *history = history_branch;
    Ok(PreparedRealtimeFrame {
        schedule: schedule.clone(),
        channels: channels.clone(),
        transition,
        temporal,
        directives,
        batch: input.batch(),
        retain_diagnostics: input.retains_diagnostics(),
    })
}

fn forced_payload<T: Clone, E>(
    input: &MaterializedRealtimeInput<T>,
    slot: RealtimeChannelId,
) -> Result<T, RealtimeFrameInterpretationError<E>> {
    input
        .forced(slot)
        .cloned()
        .ok_or(RealtimeFrameInterpretationError::MissingForcedPayload { slot })
}

fn padding_token<E>(
    schedule: &RealtimeSpeechConfig,
    slot: RealtimeChannelId,
) -> Result<i32, RealtimeFrameInterpretationError<E>> {
    match slot.index() {
        0 => Ok(schedule.text_padding_token()),
        index if index <= schedule.total_audio_codebooks() => Ok(schedule.audio_padding_token()),
        _ => Err(RealtimeFrameInterpretationError::UnsupportedSlot { slot }),
    }
}

/// Stable failure while interpreting one portable schedule transition.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum RealtimeFrameInterpretationError<E> {
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
        slot: RealtimeChannelId,
    },
    /// A required live channel has no materialized input payload.
    #[error("realtime input slot {slot:?} has no payload")]
    MissingInputPayload {
        /// Missing ordinary input channel.
        slot: RealtimeChannelId,
    },
    /// A supplied input role is not part of this categorical schedule.
    #[error("realtime channel {slot:?} cannot be supplied as {role}")]
    InputRole {
        /// Channel with the invalid role.
        slot: RealtimeChannelId,
        /// Required scheduling role.
        role: &'static str,
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
    /// Completed output attribution disagrees with the declared channel contract.
    #[error(transparent)]
    Completed(RealtimeCompletedFrameError),
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
        slot: RealtimeChannelId,
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

        fn filled_column(&mut self, token: i32, batch: usize) -> Result<Self::Tensor, Self::Error> {
            self.calls += 1;
            Ok(Matrix {
                values: vec![token; batch],
                shape: [batch, 1],
            })
        }
    }

    fn schedule(convention: RealtimeFrameConvention) -> RealtimeSpeechConfig {
        RealtimeSpeechConfig::new(2, 1, 1, 1, 9, 8, convention, vec![0, 0, 1]).unwrap()
    }

    fn ingress(schedule: &RealtimeSpeechConfig) -> RealtimeIngressContract {
        RealtimeIngressContract::new(schedule.clone(), TokenDomain::new(10), TokenDomain::new(9))
            .unwrap()
    }

    fn payload_history(schedule: &RealtimeSpeechConfig) -> RealtimePayloadHistory<Matrix> {
        RealtimePayloadHistory::with_contract(
            RealtimePayloadContract::new(
                RealtimeIngressContract::new(
                    schedule.clone(),
                    TokenDomain::new(10),
                    TokenDomain::new(9),
                )
                .unwrap()
                .channels()
                .clone(),
                1,
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
            &ingress(&schedule),
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
                    history.schedule(),
                    RealtimeSlotCoordinate::new(0, RealtimeChannelId::new(2))
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
        assert_eq!(
            completed.value(RealtimeChannelId::new(0)).unwrap().values,
            vec![3]
        );
        assert_eq!(
            completed.decision_channels(),
            &[RealtimeChannelId::new(0), RealtimeChannelId::new(1)]
        );
        assert_eq!(
            completed.value(RealtimeChannelId::new(1)).unwrap().values,
            vec![6]
        );
        assert!(completed.aligned().is_empty());
        assert_eq!(
            history
                .required(
                    history.schedule(),
                    RealtimeSlotCoordinate::new(0, RealtimeChannelId::new(1))
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
            &ingress(&schedule),
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
            schedule_state.occupancy(RealtimeSlotCoordinate::new(0, RealtimeChannelId::new(0))),
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
        assert_eq!(
            completed.value(RealtimeChannelId::new(0)).unwrap().values,
            vec![9]
        );
        assert!(completed.decision_channels().is_empty());
        assert_eq!(
            completed.value(RealtimeChannelId::new(1)).unwrap().values,
            vec![8]
        );
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
            &ingress(&schedule),
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
            &ingress(&schedule),
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
        assert!(forced_completed.aligned().is_empty());

        let next_input = materialize_input(
            &schedule,
            RealtimeInputFrame::new(1, vec![7]),
            &mut mechanisms,
        );
        let next = prepare_realtime_frame(
            &ingress(&schedule),
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
        assert_eq!(
            next_completed.aligned()[0].channel(),
            RealtimeChannelId::new(1)
        );
        assert_eq!(next_completed.aligned()[0].payload().values, vec![6]);
        assert_eq!(
            next_completed
                .value(RealtimeChannelId::new(1))
                .unwrap()
                .values,
            vec![6]
        );
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
            &ingress(&schedule),
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
            &ingress(&schedule),
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
        assert_eq!(
            second_completed.aligned()[0].channel(),
            RealtimeChannelId::new(1)
        );
        assert_eq!(second_completed.aligned()[0].payload().values, vec![5]);
        assert_eq!(
            history
                .required(
                    history.schedule(),
                    RealtimeSlotCoordinate::new(1, RealtimeChannelId::new(0)),
                )
                .unwrap()
                .values,
            vec![6]
        );
        assert_eq!(
            history
                .required(
                    history.schedule(),
                    RealtimeSlotCoordinate::new(1, RealtimeChannelId::new(1)),
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
            &ingress(&attempted_schedule),
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
    fn channel_roles_are_checked_before_warmup_and_payloads_resolve_by_identity() {
        let schedule = RealtimeSpeechConfig::new(
            3,
            2,
            1,
            3,
            9,
            8,
            RealtimeFrameConvention::AbsoluteDelayedSlots,
            vec![0, 0, 0, 0],
        )
        .unwrap();
        let contract = ingress(&schedule);
        let mut state = RealtimeFrameScheduleState::new(schedule.clone());
        let mut history = payload_history(&schedule);
        let mut mechanisms = Mechanisms::default();
        let entry = |id, value| (RealtimeChannelId::new(id), column(value));
        for (ordinary, forced) in [
            (vec![entry(2, 4)], vec![]),
            (vec![entry(2, 4), entry(3, 7), entry(1, 6)], vec![]),
            (vec![entry(2, 4), entry(3, 7)], vec![entry(2, 6)]),
        ] {
            let input = MaterializedRealtimeInput::new(
                contract.channels().clone(),
                1,
                ordinary,
                forced,
                false,
            )
            .unwrap();
            assert!(prepare_realtime_frame(
                &contract,
                &mut state,
                &mut history,
                &input,
                &mut mechanisms
            )
            .is_err());
            assert_eq!(mechanisms.calls, 0);
            assert_eq!(state.frontier(), 0);
            assert!(history.is_empty());
        }
        let input = MaterializedRealtimeInput::new(
            contract.channels().clone(),
            1,
            vec![entry(3, 7), entry(2, 4)],
            vec![entry(1, 6), entry(0, 3)],
            false,
        )
        .unwrap();
        let warmup =
            prepare_realtime_frame(&contract, &mut state, &mut history, &input, &mut mechanisms)
                .unwrap();
        assert!(!warmup.transition().model_call_required());
        let prepared =
            prepare_realtime_frame(&contract, &mut state, &mut history, &input, &mut mechanisms)
                .unwrap();
        let forced = prepared
            .directives()
            .iter()
            .map(|directive| match directive {
                PredictionDirective::Force(value) => value.values[0],
                PredictionDirective::Sample => {
                    panic!("all four target channels have supplied values")
                }
            })
            .collect::<Vec<_>>();
        assert_eq!(forced, [3, 6, 4, 7]);
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
            &ingress(&schedule),
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
