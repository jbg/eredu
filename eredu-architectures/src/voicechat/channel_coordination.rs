//! Selected owner authority for one unpublished agent/function decision.

use eredu_core::{BackendFailure, BackendFailureKind};
use eredu_nn::{Tensor, TensorElementType};
use eredu_runtime::{SamplingBackend, replicated_execution::ReplicatedCoordinator};

use super::{ChannelDecision, FeedbackTokens};

const PROTOCOL: u64 = 0x4348414e4e454c01;
const PREPARE_PUBLICATION: u64 = 100;
const DECODE_PUBLICATION: u64 = 101;

/// Exact agreement boundaries in the shared channel-selection transition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u64)]
pub enum ChannelDecisionPhase {
    /// Invocation admission before selection or override resolution.
    Admission = 0,
    /// Owner selection and semantic override resolution.
    Selection = 1,
    /// Validation of the published owner decision.
    Publication = 2,
    /// Prospective history validation before branch adoption.
    Candidate = 3,
}

/// Optional selected coordination for the ordinary channel-selection driver.
///
/// This authority neither submits work nor establishes native completion. The
/// caller retains the unpublished branch and its completion resources.
pub trait ChannelDecisionCoordination<B: SamplingBackend> {
    /// Whether this participant alone selects tokens and resolves host policy.
    fn is_owner(&self) -> bool;

    /// Consumes one invocation and agrees admission, including local failures.
    fn admit(
        &mut self,
        words: &[u64],
        success: bool,
        context: &B::Context,
    ) -> Result<(), BackendFailure>;

    /// Agrees exact phase metadata and whether local work succeeded.
    fn agree(
        &mut self,
        phase: ChannelDecisionPhase,
        words: &[u64],
        success: bool,
        context: &B::Context,
    ) -> Result<(), BackendFailure>;

    /// Publishes the owner's decisions; followers supply no local decision.
    fn publish(
        &mut self,
        owner: Option<&ChannelDecision<B::Token>>,
        context: &B::Context,
    ) -> Result<ChannelDecision<B::Token>, BackendFailure>;
}

/// One channel invocation over retained bounded agreement and broadcast resources.
///
/// Request, turn and invocation identity are supplied explicitly by the caller.
/// Admission consumes authority even on failure; retries use a fresh authority.
/// This value owns neither a scheduler nor mutable model state and is not part
/// of a session snapshot.
pub struct ReplicatedChannelDecisions<'a, T: Tensor> {
    coordinator: &'a mut dyn ReplicatedCoordinator<T, T::Context>,
    identity: [u64; 3],
    consumed: bool,
    fenced: bool,
}

impl<'a, T: Tensor> ReplicatedChannelDecisions<'a, T> {
    /// Borrows selected communication with a recorded invocation identity.
    pub fn new(
        coordinator: &'a mut dyn ReplicatedCoordinator<T, T::Context>,
        identity: [u64; 3],
    ) -> Self {
        Self {
            coordinator,
            identity,
            consumed: false,
            fenced: false,
        }
    }

    /// Whether the single permitted invocation has entered admission.
    pub fn is_consumed(&self) -> bool {
        self.consumed
    }

    fn exchange(
        &mut self,
        phase: u64,
        words: &[u64],
        success: bool,
        context: &T::Context,
    ) -> Result<(), BackendFailure> {
        if self.fenced {
            return Err(invalid("channel-decision authority is fenced"));
        }
        let result = (|| {
            // Agree the payload length before issuing variable exchanges.
            // Different policy/history lengths cannot diverge collective counts.
            if !self.coordinator.agree(
                [
                    PROTOCOL,
                    self.identity[0],
                    self.identity[1],
                    self.identity[2],
                    phase,
                    words.len() as u64,
                    0,
                    0,
                ],
                success,
                context,
            )? {
                return Err(invalid(
                    "channel-decision participants disagree or a participant failed",
                ));
            }
            for (index, chunk) in words.chunks(6).enumerate() {
                let mut descriptor = [u64::MAX; 8];
                descriptor[0] = PROTOCOL;
                descriptor[1] = index as u64;
                descriptor[2..2 + chunk.len()].copy_from_slice(chunk);
                if !self.coordinator.agree(descriptor, true, context)? {
                    return Err(invalid(
                        "channel-decision metadata differs between participants",
                    ));
                }
            }
            Ok(())
        })();
        if result.is_err() {
            self.fenced = true;
        }
        result
    }

    fn gate<V>(
        &mut self,
        phase: u64,
        local: Result<V, BackendFailure>,
        context: &T::Context,
    ) -> Result<V, BackendFailure> {
        let agreement = self.exchange(phase, &[], local.is_ok(), context);
        // A local native error retains its source even when agreement also fails.
        let value = local?;
        agreement?;
        Ok(value)
    }
}

impl<B, T> ChannelDecisionCoordination<B> for ReplicatedChannelDecisions<'_, T>
where
    T: Tensor,
    B: SamplingBackend<Token = T, Context = T::Context>,
    B::Error: std::error::Error + Send + Sync + 'static,
{
    fn is_owner(&self) -> bool {
        self.coordinator.is_owner()
    }

    fn admit(
        &mut self,
        words: &[u64],
        success: bool,
        context: &B::Context,
    ) -> Result<(), BackendFailure> {
        let fresh = !self.consumed;
        self.consumed = true;
        self.exchange(
            ChannelDecisionPhase::Admission as u64,
            words,
            fresh && success,
            context,
        )
    }

    fn agree(
        &mut self,
        phase: ChannelDecisionPhase,
        words: &[u64],
        success: bool,
        context: &B::Context,
    ) -> Result<(), BackendFailure> {
        self.exchange(phase as u64, words, self.consumed && success, context)
    }

    fn publish(
        &mut self,
        owner: Option<&ChannelDecision<T>>,
        context: &B::Context,
    ) -> Result<ChannelDecision<T>, BackendFailure> {
        if self.fenced {
            return Err(invalid("channel-decision authority is fenced"));
        }
        let prepared = (|| {
            if !self.consumed || owner.is_some() != self.coordinator.is_owner() {
                return Err(invalid(
                    "channel-decision publication has invalid ownership",
                ));
            }
            let mut values = [0; 5];
            if let Some(owner) = owner {
                for (destination, id) in values.iter_mut().zip([
                    owner.raw_agent_id,
                    owner.raw_function_id,
                    owner.feedback.agent(),
                    owner.feedback.function(),
                    owner.spoken_text,
                ]) {
                    *destination = i32::try_from(id)
                        .map_err(|_| invalid("channel-decision token exceeds I32"))?;
                }
            }
            let prepared =
                T::from_i32_slice(&values, &[5], context).map_err(BackendFailure::from_error)?;
            if prepared.shape() != [5] || prepared.element_type() != Some(TensorElementType::I32) {
                return Err(invalid("channel-decision destination requires I32 [5]"));
            }
            Ok(prepared)
        })();
        let prepared = self.gate(PREPARE_PUBLICATION, prepared, context)?;
        let published = match self.coordinator.publish(prepared, context) {
            Ok(value) => value,
            Err(error) => {
                self.fenced = true;
                return Err(error);
            }
        };
        let decoded = (|| {
            if published.shape() != [5] || published.element_type() != Some(TensorElementType::I32)
            {
                return Err(invalid("channel-decision publication requires I32 [5]"));
            }
            let values = published
                .to_i32_vec(context)
                .map_err(BackendFailure::from_error)?;
            if values.len() != 5 || values.iter().any(|&value| value < 0) {
                return Err(invalid(
                    "channel-decision publication has invalid token IDs",
                ));
            }
            let feedback = FeedbackTokens::new(values[2] as u32, values[3] as u32)
                .map_err(BackendFailure::from_error)?;
            Ok(ChannelDecision {
                raw_agent: T::from_i32_slice(&values[..1], &[1], context)
                    .map_err(BackendFailure::from_error)?,
                raw_function: T::from_i32_slice(&values[1..2], &[1], context)
                    .map_err(BackendFailure::from_error)?,
                raw_agent_id: values[0] as u32,
                raw_function_id: values[1] as u32,
                feedback,
                spoken_text: values[4] as u32,
            })
        })();
        self.gate(DECODE_PUBLICATION, decoded, context)
    }
}

#[derive(Debug, thiserror::Error)]
#[error("invalid VoiceChat channel coordination: {0}")]
struct ChannelCoordinationError(&'static str);

fn invalid(reason: &'static str) -> BackendFailure {
    BackendFailure::new(
        BackendFailureKind::InvalidInput,
        ChannelCoordinationError(reason),
    )
}
