//! Coordinator-owned RNNT decisions over the ordinary greedy transition.
//!
//! Every participant retains its recurrent replica. This driver agrees explicit
//! invocation identity and host control state before parameter acquisition, then
//! delegates all equations and label-loop policy to `RnntExecution`. Returned
//! state remains unpublished until the enclosing session commits.

use super::rnnt::{
    RnntDimensions, RnntError, RnntExecution, RnntFrame, RnntLimits, RnntPredictionState, RnntState,
};
use eredu_nn::{NeuralBackend, Tensor};
use eredu_runtime::replicated_execution::ReplicatedCoordinator;

const PROTOCOL: u64 = 0x524e4e5400010000;

fn agreed<
    T,
    B: NeuralBackend,
    C: ReplicatedCoordinator<B::Tensor, <B::Tensor as Tensor>::Context>,
>(
    coordinator: &mut C,
    descriptor: [u64; 8],
    local: Result<T, RnntError>,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<T, RnntError> {
    if !coordinator.agree(descriptor, local.is_ok(), context)? {
        return Err(local.err().unwrap_or(RnntError::InvalidState(
            "RNNT participant failed or invocation descriptors differ",
        )));
    }
    local
}

/// Executes one frame with an explicit `[request, turn, invocation] identity.
///
/// All participants call this operation in the same order with the same recorded
/// input and parent branch. Descriptor agreement checks the complete hypothesis,
/// dimensions, frame frontier and limits; tensor contents remain the enclosing
/// session's state ownership responsibility. The coordinator alone evaluates
/// greedy selection, and broadcasts each result before any feedback prediction.
/// This operation neither publishes host events nor commits a session branch.
pub fn decode_frame<B, R, C>(
    model: &mut R,
    coordinator: &mut C,
    identity: [u64; 3],
    encoder: &B::Tensor,
    state: &RnntState<B::Tensor>,
    limits: RnntLimits,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<RnntFrame<B::Tensor>, RnntError>
where
    B: NeuralBackend,
    R: RnntExecution<B>,
    C: ReplicatedCoordinator<B::Tensor, <B::Tensor as Tensor>::Context>,
{
    prepare_state(model, coordinator, identity, state, limits, context)?;
    decode_prepared_frame(
        model,
        coordinator,
        identity,
        encoder,
        state,
        limits,
        context,
    )
}

pub(crate) fn prepare_state<B, R, C>(
    model: &R,
    coordinator: &mut C,
    identity: [u64; 3],
    state: &RnntState<B::Tensor>,
    limits: RnntLimits,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<(), RnntError>
where
    B: NeuralBackend,
    R: RnntExecution<B>,
    C: ReplicatedCoordinator<B::Tensor, <B::Tensor as Tensor>::Context>,
{
    let [request, turn, invocation] = identity;
    agreed::<_, B, C>(
        coordinator,
        [
            PROTOCOL,
            request,
            turn,
            invocation,
            state.frames(),
            state.hypothesis().len() as u64,
            limits.max_symbols_per_frame as u64,
            limits.hypothesis_capacity as u64,
        ],
        model.validate_decode_state(state, limits),
        context,
    )?;
    let d = model.dimensions();
    agreed::<_, B, C>(
        coordinator,
        [
            PROTOCOL + 1,
            d.vocab_size as u64,
            d.predictor_width as u64,
            d.predictor_layers as u64,
            d.encoder_width as u64,
            d.joint_width as u64,
            u64::from(state.predictor().is_some()),
            0,
        ],
        d.validate(),
        context,
    )?;
    for (index, chunk) in state.hypothesis().chunks(6).enumerate() {
        let mut descriptor = [u64::MAX; 8];
        descriptor[0] = PROTOCOL + 2;
        descriptor[1] = index as u64;
        for (slot, token) in descriptor[2..].iter_mut().zip(chunk) {
            *slot = *token as u64;
        }
        agreed::<_, B, C>(coordinator, descriptor, Ok(()), context)?;
    }
    Ok(())
}

pub(crate) fn decode_prepared_frame<B, R, C>(
    model: &mut R,
    coordinator: &mut C,
    identity: [u64; 3],
    encoder: &B::Tensor,
    state: &RnntState<B::Tensor>,
    limits: RnntLimits,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<RnntFrame<B::Tensor>, RnntError>
where
    B: NeuralBackend,
    R: RnntExecution<B>,
    C: ReplicatedCoordinator<B::Tensor, <B::Tensor as Tensor>::Context>,
{
    agreed::<_, B, C>(
        coordinator,
        [
            PROTOCOL + 4,
            identity[0],
            identity[1],
            identity[2],
            state.frames(),
            0,
            0,
            0,
        ],
        model.validate_frame(encoder),
        context,
    )?;
    let mut execution = Coordinated {
        model,
        coordinator,
        identity,
        phase: 0,
        fenced: false,
        backend: std::marker::PhantomData::<B>,
    };
    let candidate = execution.decode_frame(encoder, state, limits, context);
    execution.complete(7, 0, candidate, context)
}

struct Coordinated<'a, B, R, C> {
    model: &'a mut R,
    coordinator: &'a mut C,
    identity: [u64; 3],
    phase: u64,
    fenced: bool,
    backend: std::marker::PhantomData<B>,
}
impl<B, R, C> Coordinated<'_, B, R, C>
where
    B: NeuralBackend,
    C: ReplicatedCoordinator<B::Tensor, <B::Tensor as Tensor>::Context>,
{
    fn prepare(
        &mut self,
        operation: u64,
        argument: u64,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(), RnntError> {
        self.complete(operation + 16, argument, Ok(()), context)
    }
    fn complete<T>(
        &mut self,
        operation: u64,
        argument: u64,
        local: Result<T, RnntError>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<T, RnntError> {
        if self.fenced {
            return Err(local
                .err()
                .unwrap_or(RnntError::InvalidState("RNNT coordination is fenced")));
        }
        let [request, turn, invocation] = self.identity;
        let descriptor = [
            PROTOCOL + 3,
            request,
            turn,
            invocation,
            self.phase,
            operation,
            argument,
            0,
        ];
        self.phase = self
            .phase
            .checked_add(1)
            .ok_or(RnntError::InvalidState("RNNT coordination phase exhausted"))?;
        let result = agreed::<_, B, C>(self.coordinator, descriptor, local, context);
        self.fenced = result.is_err();
        result
    }
}
impl<B, R, C> RnntExecution<B> for Coordinated<'_, B, R, C>
where
    B: NeuralBackend,
    R: RnntExecution<B>,
    C: ReplicatedCoordinator<B::Tensor, <B::Tensor as Tensor>::Context>,
{
    fn dimensions(&self) -> RnntDimensions {
        self.model.dimensions()
    }
    fn predictor_ownership(&self) -> std::ops::Range<usize> {
        self.model.predictor_ownership()
    }
    fn embed_token(
        &mut self,
        token: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, RnntError> {
        self.prepare(0, token as u64, context)?;
        let result = self.model.embed_token(token, context);
        self.complete(0, token as u64, result, context)
    }
    fn predictor_step(
        &mut self,
        input: &B::Tensor,
        previous: Option<&RnntPredictionState<B::Tensor>>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<RnntPredictionState<B::Tensor>, RnntError> {
        self.prepare(1, u64::from(previous.is_some()), context)?;
        let result = self.model.predictor_step(input, previous, context);
        self.complete(1, u64::from(previous.is_some()), result, context)
    }
    fn encode_frame(
        &mut self,
        input: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, RnntError> {
        self.prepare(2, 0, context)?;
        let result = self.model.encode_frame(input, context);
        self.complete(2, 0, result, context)
    }
    fn project_prediction(
        &mut self,
        input: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, RnntError> {
        self.prepare(3, 0, context)?;
        let result = self.model.project_prediction(input, context);
        self.complete(3, 0, result, context)
    }
    fn project_output(
        &mut self,
        input: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, RnntError> {
        self.prepare(4, 0, context)?;
        let result = self.model.project_output(input, context);
        self.complete(4, 0, result, context)
    }
    fn select_symbol(
        &mut self,
        logits: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<usize, RnntError> {
        self.prepare(5, 0, context)?;
        let selected = if self.coordinator.is_owner() {
            self.model.select_symbol(logits, context)
        } else {
            Ok(0)
        };
        let value = selected.and_then(|token| {
            if token > self.dimensions().vocab_size as usize {
                return Err(RnntError::InvalidState("RNNT decision exceeds vocabulary"));
            }
            Ok(B::Tensor::from_i32_slice(&[token as i32], &[1], context)?)
        });
        let value = self.complete(5, 0, value, context)?;
        let value = self.coordinator.publish(value, context).map_err(|error| {
            self.fenced = true;
            RnntError::from(error)
        })?;
        let selected = (|| {
            let tokens = value.to_i32_vec(context)?;
            if value.shape() != [1]
                || tokens.len() != 1
                || tokens[0] < 0
                || tokens[0] > self.dimensions().vocab_size
            {
                return Err(RnntError::InvalidState("invalid coordinated RNNT decision"));
            }
            Ok(tokens[0] as usize)
        })();
        self.complete(6, 0, selected, context)
    }
}
