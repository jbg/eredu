//! Coordinated greedy RNNT decoding with independently transported recurrent phases.
//!
//! The caller retains the predictor invocation cursor outside conversational state.
//! A frame reserves its worst-case recurrent work before model reads. Reservations
//! are never refunded by failure, replay, restore, or an early blank decision.

use super::{
    rnnt::{
        RnntDimensions, RnntError, RnntExecution, RnntFrame, RnntLimits, RnntPredictionState,
        RnntProjectionExecution, RnntState,
    },
    rnnt_coordination,
};
use eredu_core::{DistributedCommitEpoch, DistributedCommitOutcome};
use eredu_nn::{NeuralBackend, Tensor};
use eredu_runtime::replicated_execution::ReplicatedCoordinator;
use std::{marker::PhantomData, ops::Range};

const PROTOCOL: u64 = 0x524e4e5450500001;

/// One recurrent phase, using the existing component pipeline and caller-owned identity.
/// Each rank owns only its declared recurrent interval plus the published readout.
pub trait RnntPredictorExecution<B: NeuralBackend> {
    /// Exact shared geometry.
    fn dimensions(&self) -> RnntDimensions;
    /// Exact global recurrent layers retained on this rank.
    fn predictor_ownership(&self) -> Range<usize>;
    /// Validates restored stage state before any projection or predictor reads.
    /// The released pipeline transports and retains exact F32 recurrent rows.
    fn validate_parent(
        &self,
        parent: Option<&RnntPredictionState<B::Tensor>>,
    ) -> Result<(), RnntError> {
        if let Some(parent) = parent {
            parent.validate(self.dimensions(), &self.predictor_ownership())?;
            let f32_row = |tensor: &B::Tensor| {
                tensor.element_type() == Some(eredu_nn::TensorElementType::F32)
            };
            if !f32_row(parent.output())
                || parent
                    .local_state()
                    .layers
                    .iter()
                    .any(|layer| !f32_row(&layer.hidden) || !f32_row(&layer.cell))
            {
                return Err(RnntError::InvalidState(
                    "RNNT pipeline requires F32 recurrent carry and published readout",
                ));
            }
        }
        Ok(())
    }
    /// Completes and publishes one predictor readout under an explicit epoch.
    /// The caller retains the parent until the enclosing audio frame commits.
    #[allow(clippy::too_many_arguments)]
    fn predictor_step(
        &mut self,
        input: &B::Tensor,
        parent: Option<&RnntPredictionState<B::Tensor>>,
        step: u64,
        descriptor: [u64; 7],
        epoch: DistributedCommitEpoch,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(RnntPredictionState<B::Tensor>, DistributedCommitOutcome), RnntError>;
}

/// Explicit non-refundable recurrent invocation authority.
///
/// This cursor is not part of `RnntState` and is not snapshotted or cloned. A replay
/// uses fresh steps and epochs from this same cursor. The caller supplies a disjoint
/// epoch range when several components share distributed commit authority.
#[derive(Debug)]
pub struct RnntPipelineCursor {
    next_step: u64,
    next_epoch: DistributedCommitEpoch,
    remaining: u64,
}
impl RnntPipelineCursor {
    /// Selects the next identities and a finite maximum number of predictor phases.
    pub fn new(next_step: u64, next_epoch: DistributedCommitEpoch, capacity: u64) -> Self {
        Self {
            next_step,
            next_epoch,
            remaining: capacity,
        }
    }
    /// First recurrent identity not yet reserved.
    pub fn next_step(&self) -> u64 {
        self.next_step
    }
    /// First distributed epoch not yet reserved.
    pub fn next_epoch(&self) -> DistributedCommitEpoch {
        self.next_epoch
    }
    /// Remaining unreserved recurrent phases.
    pub fn remaining(&self) -> u64 {
        self.remaining
    }
    fn reserve(&mut self, count: u64) -> Result<Reservation, RnntError> {
        if count > self.remaining {
            return Err(RnntError::PipelineCapacity {
                required: count,
                remaining: self.remaining,
            });
        }
        let next_step = self
            .next_step
            .checked_add(count)
            .ok_or(RnntError::InvalidState(
                "RNNT pipeline step identities exhausted",
            ))?;
        let next_epoch = self
            .next_epoch
            .value()
            .checked_add(count)
            .and_then(DistributedCommitEpoch::new)
            .ok_or(RnntError::InvalidState(
                "RNNT pipeline commit epochs exhausted",
            ))?;
        let reservation = Reservation {
            step: self.next_step,
            epoch: self.next_epoch,
            count,
            used: 0,
        };
        self.next_step = next_step;
        self.next_epoch = next_epoch;
        self.remaining -= count;
        Ok(reservation)
    }
}
struct Reservation {
    step: u64,
    epoch: DistributedCommitEpoch,
    count: u64,
    used: u64,
}
impl Reservation {
    fn take(&mut self) -> Result<(u64, DistributedCommitEpoch, u64), RnntError> {
        if self.used == self.count {
            return Err(RnntError::InvalidState(
                "RNNT predictor exceeded its reserved frame bound",
            ));
        }
        let ordinal = self.used;
        self.used += 1;
        // reserve checked both exclusive range ends before any neural operation.
        Ok((
            self.step + ordinal,
            DistributedCommitEpoch::new(self.epoch.value() + ordinal).unwrap(),
            ordinal,
        ))
    }
}

/// Executes the shared greedy frame policy over recurrent pipeline substeps.
///
/// Every rank supplies the same explicit frame identity, input, limits and cursor.
/// Geometry, parent ownership and reservation capacity are agreed before parameter
/// acquisition. An accepted reservation is consumed even when this call fails;
/// insufficient capacity or an unrepresentable range submits no predictor phase.
/// A committed predictor phase remains an unpublished candidate of this frame.
#[allow(clippy::too_many_arguments)]
pub fn decode_frame<B, P, R, C>(
    projections: &mut P,
    predictor: &mut R,
    coordinator: &mut C,
    identity: [u64; 3],
    cursor: &mut RnntPipelineCursor,
    encoder: &B::Tensor,
    state: &RnntState<B::Tensor>,
    limits: RnntLimits,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<RnntFrame<B::Tensor>, RnntError>
where
    B: NeuralBackend,
    P: RnntProjectionExecution<B>,
    R: RnntPredictorExecution<B>,
    C: ReplicatedCoordinator<B::Tensor, <B::Tensor as Tensor>::Context>,
{
    let model = prepare_frame(
        projections,
        predictor,
        coordinator,
        identity,
        cursor,
        state,
        limits,
        context,
    )?;
    decode_reserved(model, coordinator, encoder, state, limits, context)
}

pub(crate) fn prepare_frame<'a, B, P, R, C>(
    projections: &'a mut P,
    predictor: &'a mut R,
    coordinator: &mut C,
    identity: [u64; 3],
    cursor: &mut RnntPipelineCursor,
    state: &RnntState<B::Tensor>,
    limits: RnntLimits,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<Pipeline<'a, B, P, R>, RnntError>
where
    B: NeuralBackend,
    P: RnntProjectionExecution<B>,
    R: RnntPredictorExecution<B>,
    C: ReplicatedCoordinator<B::Tensor, <B::Tensor as Tensor>::Context>,
{
    let count = (limits.max_symbols_per_frame as u64).checked_add(if state.predictor().is_none() {
        2
    } else {
        0
    });
    let descriptor = [
        PROTOCOL,
        identity[0],
        identity[1],
        identity[2],
        cursor.next_step,
        cursor.next_epoch.value(),
        cursor.remaining,
        count.unwrap_or(u64::MAX),
    ];
    let local = count
        .ok_or(RnntError::InvalidState(
            "RNNT predictor reservation overflow",
        ))
        .and_then(|count| cursor.reserve(count))
        .and_then(|reservation| {
            if projections.dimensions() != predictor.dimensions() {
                return Err(RnntError::InvalidState(
                    "RNNT projection and predictor geometries differ",
                ));
            }
            projections.dimensions().validate()?;
            predictor.validate_parent(state.predictor())?;
            let model = Pipeline {
                projections,
                predictor,
                reservation,
                identity,
                frame: state.frames(),
                backend: PhantomData::<B>,
            };
            model.validate_decode_state(state, limits)?;
            Ok(model)
        });
    if !coordinator.agree(descriptor, local.is_ok(), context)? {
        return Err(local.err().unwrap_or(RnntError::InvalidState(
            "RNNT pipeline reservation or frame preflight differs",
        )));
    }
    let model = local?;
    rnnt_coordination::prepare_state(&model, coordinator, identity, state, limits, context)?;
    Ok(model)
}

pub(crate) fn decode_reserved<B, P, R, C>(
    mut model: Pipeline<'_, B, P, R>,
    coordinator: &mut C,
    encoder: &B::Tensor,
    state: &RnntState<B::Tensor>,
    limits: RnntLimits,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<RnntFrame<B::Tensor>, RnntError>
where
    B: NeuralBackend,
    P: RnntProjectionExecution<B>,
    R: RnntPredictorExecution<B>,
    C: ReplicatedCoordinator<B::Tensor, <B::Tensor as Tensor>::Context>,
{
    let identity = model.identity;
    rnnt_coordination::decode_prepared_frame(
        &mut model,
        coordinator,
        identity,
        encoder,
        state,
        limits,
        context,
    )
}

pub(crate) struct Pipeline<'a, B: NeuralBackend, P, R> {
    projections: &'a mut P,
    predictor: &'a mut R,
    reservation: Reservation,
    identity: [u64; 3],
    frame: u64,
    backend: PhantomData<B>,
}
impl<B, P, R> RnntExecution<B> for Pipeline<'_, B, P, R>
where
    B: NeuralBackend,
    P: RnntProjectionExecution<B>,
    R: RnntPredictorExecution<B>,
{
    fn dimensions(&self) -> RnntDimensions {
        self.projections.dimensions()
    }
    fn predictor_ownership(&self) -> Range<usize> {
        self.predictor.predictor_ownership()
    }
    fn embed_token(
        &mut self,
        token: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, RnntError> {
        self.projections.embed_token(token, context)
    }
    fn predictor_step(
        &mut self,
        input: &B::Tensor,
        parent: Option<&RnntPredictionState<B::Tensor>>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<RnntPredictionState<B::Tensor>, RnntError> {
        let (step, epoch, ordinal) = self.reservation.take()?;
        let descriptor = [
            PROTOCOL + 1,
            self.identity[0],
            self.identity[1],
            self.identity[2],
            self.frame,
            ordinal,
            epoch.value(),
        ];
        let (state, outcome) = self
            .predictor
            .predictor_step(input, parent, step, descriptor, epoch, context)?;
        if outcome != DistributedCommitOutcome::Committed(epoch) {
            return Err(RnntError::PredictorCommit {
                expected: epoch,
                observed: outcome,
            });
        }
        self.predictor.validate_parent(Some(&state))?;
        Ok(state)
    }
    fn encode_frame(
        &mut self,
        input: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, RnntError> {
        self.projections.encode_frame(input, context)
    }
    fn project_prediction(
        &mut self,
        input: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, RnntError> {
        self.projections.project_prediction(input, context)
    }
    fn project_output(
        &mut self,
        input: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, RnntError> {
        self.projections.project_output(input, context)
    }
    fn select_symbol(
        &mut self,
        logits: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<usize, RnntError> {
        self.projections.select_symbol(logits, context)
    }
}
