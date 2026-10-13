//! Agent/function decisions and effective feedback in an unpublished shared branch.
//!
//! Raw agent selection always precedes overrides. Tool serialization keeps PAD
//! agent feedback independent of text spoken by TTS. This operation does not
//! advance audio time, establish completion, or publish a conversational event.

#[path = "channel_coordination.rs"]
mod coordination;
pub use coordination::{
    ChannelDecisionCoordination, ChannelDecisionPhase, ReplicatedChannelDecisions,
};

use eredu_core::BackendFailure;
use eredu_nn::RetainedTensorRoots;

use eredu_nn::{Tensor, TensorElementType};
use eredu_runtime::{
    PredictionDirective, RealtimeGenerationBranch, RealtimeScheduleState, SamplingBackend,
    SequentialDecisionError, SequentialDecisionPlan, SequentialDecisionPlanError, TokenDomain,
};

use super::{
    agent_sampling::{AgentSampler, AgentSamplingError},
    channels::{DualChannelLogits, TEXT_PAD_ID, TEXT_VOCABULARY_SIZE as VOCABULARY},
};

/// Validated previous-channel IDs for one subsequent backbone position.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FeedbackTokens {
    agent: u32,
    function: u32,
}

impl FeedbackTokens {
    /// Validates both IDs against the released vocabulary before retaining them.
    pub fn new(agent: u32, function: u32) -> Result<Self, ChannelDecisionPreflightError> {
        validate_id(agent, VOCABULARY)?;
        validate_id(function, VOCABULARY)?;
        Ok(Self { agent, function })
    }

    /// Effective agent feedback, including any turn or tool override.
    pub const fn agent(self) -> u32 {
        self.agent
    }

    /// Effective function feedback, independent of spoken text.
    pub const fn function(self) -> u32 {
        self.function
    }

    /// Creates the signed `[1, 1]` inputs accepted by channel fusion.
    pub fn materialize<T: Tensor>(
        self,
        context: &T::Context,
    ) -> Result<FeedbackTensors<T>, eredu_nn::Error> {
        Ok(FeedbackTensors {
            agent: T::from_i32_slice(&[self.agent as i32], &[1, 1], context)?,
            function: T::from_i32_slice(&[self.function as i32], &[1, 1], context)?,
        })
    }
}

/// Native feedback inputs, constructed only from validated IDs.
#[derive(Debug, Clone)]
pub struct FeedbackTensors<T> {
    /// Previous effective agent token, signed `[1, 1]`.
    pub agent: T,
    /// Previous effective function token, signed `[1, 1]`.
    pub function: T,
}

/// Explicit semantic overrides for a backbone invocation.
/// Reminder speech has no backbone decision and does not use this operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelDecisionPolicy {
    /// Ordinary live audio, including explicitly supplied response tokens.
    Main {
        /// Replaces feedback after raw agent selection and its random draw.
        agent_feedback: Option<u32>,
        /// Replaces the greedy function decision in subsequent feedback.
        function_feedback: Option<u32>,
        /// Spoken token; absent means the effective agent token.
        spoken_text: Option<u32>,
    },
    /// Tool-channel substep. Agent feedback is always PAD.
    Tool {
        /// Replaces the greedy function decision in subsequent feedback.
        function_feedback: Option<u32>,
        /// Independently selected speech; absent means PAD.
        spoken_text: Option<u32>,
    },
}

impl ChannelDecisionPolicy {
    /// Checks explicit IDs and the admitted head domain without native work.
    pub fn preflight(self, domain: TokenDomain) -> Result<(), ChannelDecisionPreflightError> {
        let vocabulary = domain.cardinality();
        if !(TEXT_PAD_ID as usize + 1..=VOCABULARY).contains(&vocabulary) {
            return Err(ChannelDecisionPreflightError::Vocabulary { vocabulary });
        }
        let (agent, function, spoken) = match self {
            Self::Main {
                agent_feedback,
                function_feedback,
                spoken_text,
            } => (agent_feedback, function_feedback, spoken_text),
            Self::Tool {
                function_feedback,
                spoken_text,
            } => (None, function_feedback, spoken_text),
        };
        for token in [agent, function, spoken].into_iter().flatten() {
            validate_id(token, vocabulary)?;
        }
        Ok(())
    }

    fn apply(self, agent: u32, function: u32) -> (FeedbackTokens, u32) {
        match self {
            Self::Main {
                agent_feedback,
                function_feedback,
                spoken_text,
            } => {
                let agent = agent_feedback.unwrap_or(agent);
                (
                    FeedbackTokens {
                        agent,
                        function: function_feedback.unwrap_or(function),
                    },
                    spoken_text.unwrap_or(agent),
                )
            }
            Self::Tool {
                function_feedback,
                spoken_text,
            } => (
                FeedbackTokens {
                    agent: TEXT_PAD_ID as u32,
                    function: function_feedback.unwrap_or(function),
                },
                spoken_text.unwrap_or(TEXT_PAD_ID as u32),
            ),
        }
    }
}

/// Fresh perception evidence from this invocation, before feedback and speech.
/// Cached-audio invocations supply neither field.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DecisionEvidence {
    /// Whether this frame's first RNNT joint decision selected blank.
    pub first_blank: Option<bool>,
    /// Zero-based source-audio frame, independent of tool substeps.
    pub audio_frame_index: Option<u64>,
}

/// Validated raw decisions and same-invocation perception evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecisionInput {
    /// Agent selection before semantic overrides.
    pub raw_agent: u32,
    /// Greedy function selection before semantic overrides.
    pub raw_function: u32,
    /// Fresh RNNT/source attribution, absent for retained or silence input.
    pub evidence: DecisionEvidence,
}

/// Pure host policy evaluated between raw selection and effective feedback.
///
/// Implementations read an immutable candidate and must not mutate canonical
/// session state, publish events, execute tools or read a clock. The enclosing
/// driver adopts any corresponding host successor only when this branch commits.
/// Errors retain their original source and leave sampling state unchanged.
pub trait DecisionResolver {
    /// Validates retained policy before any native work.
    fn preflight(&self, domain: TokenDomain) -> Result<(), BackendFailure>;

    /// Resolves overrides using this invocation's raw decisions and perception.
    fn resolve(&self, input: DecisionInput) -> Result<ChannelDecisionPolicy, BackendFailure>;
}

impl DecisionResolver for ChannelDecisionPolicy {
    fn preflight(&self, domain: TokenDomain) -> Result<(), BackendFailure> {
        (*self)
            .preflight(domain)
            .map_err(BackendFailure::from_error)
    }

    fn resolve(&self, _input: DecisionInput) -> Result<ChannelDecisionPolicy, BackendFailure> {
        Ok(*self)
    }
}

/// A fixed override or a borrowed immutable conversational-policy candidate.
#[derive(Clone, Copy)]
pub enum DecisionSource<'a> {
    /// Explicit overrides known before perception and raw sampling.
    Fixed(ChannelDecisionPolicy),
    /// Overrides requiring this invocation's perception and raw selection.
    Resolved(&'a dyn DecisionResolver),
}

impl From<ChannelDecisionPolicy> for DecisionSource<'_> {
    fn from(policy: ChannelDecisionPolicy) -> Self {
        Self::Fixed(policy)
    }
}

impl DecisionResolver for DecisionSource<'_> {
    fn preflight(&self, domain: TokenDomain) -> Result<(), BackendFailure> {
        // Domain validity is an architecture invariant even when host policy
        // has no static overrides to inspect.
        ChannelDecisionPolicy::Main {
            agent_feedback: None,
            function_feedback: None,
            spoken_text: None,
        }
        .preflight(domain)
        .map_err(BackendFailure::from_error)?;
        match self {
            Self::Fixed(policy) => DecisionResolver::preflight(policy, domain),
            Self::Resolved(resolver) => resolver.preflight(domain),
        }
    }

    fn resolve(&self, input: DecisionInput) -> Result<ChannelDecisionPolicy, BackendFailure> {
        match self {
            Self::Fixed(policy) => Ok(*policy),
            Self::Resolved(resolver) => resolver.resolve(input),
        }
    }
}

/// Tentative decisions, retained with the caller's unpublished transition.
#[derive(Debug, Clone)]
pub struct ChannelDecision<T> {
    /// Raw agent selection before any override, retained for native completion.
    pub raw_agent: T,
    /// Greedy function selection, retained for native completion.
    pub raw_function: T,
    /// Validated host identity of the raw agent token.
    pub raw_agent_id: u32,
    /// Validated host identity of the raw function token.
    pub raw_function_id: u32,
    /// Effective previous tokens for the next backbone invocation.
    pub feedback: FeedbackTokens,
    /// Independently selected text token passed to the speech transition.
    pub spoken_text: u32,
}

impl<T> RetainedTensorRoots<T> for ChannelDecision<T> {
    fn visit_retained_tensors<'a>(&'a self, visitor: &mut dyn FnMut(&'a T)) {
        visitor(&self.raw_agent);
        visitor(&self.raw_function);
    }
}

/// Selects from one pair of admitted single-row heads using shared RNG ownership.
///
/// `domain` describes both admitted heads (131072 for the released checkpoint),
/// rather than admitting or reconfiguring a checkpoint. The shape is `[vocab]`,
/// `[1, vocab]`, or `[1, 1, vocab]`. Static admission precedes native work;
/// resolved overrides are checked after raw selection and before RNG adoption.
/// Any failure leaves branch sampling state unchanged. Successful state remains
/// unpublished, so subsequent TTS draws and completion use this same branch.
/// The enclosing submission owner must retain native resources and establish
/// exact completion even when selection fails after a native operation.
/// Optional coordination selects one owner for sampling and policy resolution.
/// Followers receive validated decisions and consume a forced decision through
/// the same driver without RNG. Agreement precedes adoption of RNG and history;
/// each retry requires fresh recorded invocation authority.
pub fn select_channels<B, M, Q, P, C>(
    branch: &mut RealtimeGenerationBranch<M, Q, P, AgentSampler, B::RandomState, C>,
    logits: &DualChannelLogits<B::Logits>,
    policy: DecisionSource<'_>,
    evidence: DecisionEvidence,
    domain: TokenDomain,
    mut coordination: Option<&mut dyn ChannelDecisionCoordination<B>>,
    context: &B::Context,
) -> Result<ChannelDecision<B::Token>, ChannelDecisionError<B::Error>>
where
    B: SamplingBackend,
    B::Logits: Tensor,
    B::RandomState: Clone,
    Q: RealtimeScheduleState,
{
    let owner = coordination
        .as_ref()
        .is_none_or(|authority| authority.is_owner());
    let admitted: Result<(), ChannelDecisionError<B::Error>> = (|| {
        // Conversational policy is authoritative only on the selected owner.
        // Followers validate the common model/sampling contract independently.
        if owner {
            policy
                .preflight(domain)
                .map_err(ChannelDecisionError::Resolution)?;
        }
        ChannelDecisionPolicy::Main {
            agent_feedback: None,
            function_feedback: None,
            spoken_text: None,
        }
        .preflight(domain)?;
        if branch.has_submission_completion() {
            return Err(ChannelDecisionPreflightError::Submitted.into());
        }
        if branch.samplers().len() != 1 {
            return Err(ChannelDecisionPreflightError::SamplerCount {
                actual: branch.samplers().len(),
            }
            .into());
        }
        let sampler = &branch.samplers()[0];
        sampler
            .preflight()
            .map_err(ChannelDecisionPreflightError::Sampling)?;
        for &token in sampler.history() {
            validate_id(token, domain.cardinality())?;
        }
        for head in [&logits.agent, &logits.function] {
            let shape = head.shape();
            if !(1..=3).contains(&shape.len())
                || shape.last().copied() != Some(domain.cardinality() as i32)
                || shape[..shape.len() - 1].iter().any(|&extent| extent != 1)
            {
                return Err(ChannelDecisionPreflightError::Shape.into());
            }
            if !matches!(
                head.element_type(),
                Some(
                    TensorElementType::F16
                        | TensorElementType::Bf16
                        | TensorElementType::F32
                        | TensorElementType::F64
                )
            ) {
                return Err(ChannelDecisionPreflightError::Precision.into());
            }
        }
        Ok(())
    })();
    if let Some(authority) = coordination.as_deref_mut() {
        let mut words = vec![
            domain.cardinality() as u64,
            evidence
                .first_blank
                .map_or(0, |blank| if blank { 2 } else { 1 }),
            u64::from(evidence.audio_frame_index.is_some()),
            evidence.audio_frame_index.unwrap_or(0),
            branch.samplers().len() as u64,
        ];
        if let Some(sampler) = branch.samplers().first() {
            let policy = sampler.policy();
            words.extend([
                policy.temperature().to_bits() as u64,
                policy.top_p().to_bits() as u64,
                policy.repetition_penalty().to_bits() as u64,
                policy.max_history() as u64,
                sampler.history().len() as u64,
            ]);
            words.extend(sampler.history().iter().map(|&id| id as u64));
        }
        for head in [&logits.agent, &logits.function] {
            words.push(head.shape().len() as u64);
            words.extend(head.shape().iter().map(|&extent| extent as u64));
            words.push(match head.element_type() {
                Some(TensorElementType::F16) => 0,
                Some(TensorElementType::Bf16) => 1,
                Some(TensorElementType::F32) => 2,
                Some(TensorElementType::F64) => 3,
                _ => u64::MAX,
            });
        }
        let agreement = authority.admit(&words, admitted.is_ok(), context);
        admitted?;
        agreement.map_err(ChannelDecisionError::Coordination)?;
    } else {
        admitted?;
    }
    let sampler = branch.samplers()[0].clone();
    let mut driver = None;
    let selected: Result<_, ChannelDecisionError<B::Error>> = (|| {
        if !owner {
            return Ok(None);
        }
        let plan = SequentialDecisionPlan::new([PredictionDirective::Sample], false, false)?;
        let mut tentative =
            branch.decision_driver::<B>(plan, vec![sampler.policy().temperature()])?;
        let raw_agent = tentative
            .resolve(0, &logits.agent, domain, context)
            .map_err(ChannelDecisionError::decision)?;
        let raw_function = B::sample_raw(&logits.function, 0.0, None, context)
            .map_err(ChannelDecisionError::Backend)?;
        let raw_function = B::validate_token(&raw_function, domain, context)
            .map_err(ChannelDecisionError::Backend)?;
        let [raw_agent_id, raw_function_id] = B::token_ids([&raw_agent, &raw_function], context)
            .map_err(ChannelDecisionError::Backend)?;
        validate_id(raw_agent_id, domain.cardinality())?;
        validate_id(raw_function_id, domain.cardinality())?;
        let policy = policy
            .resolve(DecisionInput {
                raw_agent: raw_agent_id,
                raw_function: raw_function_id,
                evidence,
            })
            .map_err(ChannelDecisionError::Resolution)?;
        policy.preflight(domain)?;
        let (feedback, spoken_text) = policy.apply(raw_agent_id, raw_function_id);
        driver = Some(tentative);
        Ok(Some(ChannelDecision {
            raw_agent,
            raw_function,
            raw_agent_id,
            raw_function_id,
            feedback,
            spoken_text,
        }))
    })();
    let agreement = match coordination.as_deref_mut() {
        Some(authority) => authority.agree(
            ChannelDecisionPhase::Selection,
            &[],
            selected.is_ok(),
            context,
        ),
        None => Ok(()),
    };
    let selected = selected?;
    agreement.map_err(ChannelDecisionError::Coordination)?;
    let publication = match coordination.as_deref_mut() {
        Some(authority) => authority.publish(selected.as_ref(), context),
        None => Ok(selected.expect("local selection produces a decision")),
    };
    let agreement = match coordination.as_deref_mut() {
        Some(authority) => authority.agree(
            ChannelDecisionPhase::Publication,
            &[],
            publication.is_ok(),
            context,
        ),
        None => Ok(()),
    };
    let mut decision = publication.map_err(ChannelDecisionError::Coordination)?;
    agreement.map_err(ChannelDecisionError::Coordination)?;
    let candidate: Result<_, ChannelDecisionError<B::Error>> = (|| {
        for token in [
            decision.raw_agent_id,
            decision.raw_function_id,
            decision.feedback.agent(),
            decision.feedback.function(),
            decision.spoken_text,
        ] {
            validate_id(token, domain.cardinality())?;
        }
        decision.raw_agent = B::validate_token(&decision.raw_agent, domain, context)
            .map_err(ChannelDecisionError::Backend)?;
        decision.raw_function = B::validate_token(&decision.raw_function, domain, context)
            .map_err(ChannelDecisionError::Backend)?;
        let ids = B::token_ids([&decision.raw_agent, &decision.raw_function], context)
            .map_err(ChannelDecisionError::Backend)?;
        if ids != [decision.raw_agent_id, decision.raw_function_id] {
            return Err(ChannelDecisionPreflightError::PublishedDecision.into());
        }
        if !owner {
            let plan = SequentialDecisionPlan::new(
                [PredictionDirective::Force(decision.raw_agent.clone())],
                false,
                false,
            )?;
            let mut follower =
                branch.decision_driver::<B>(plan, vec![sampler.policy().temperature()])?;
            follower
                .resolve(0, &logits.agent, domain, context)
                .map_err(ChannelDecisionError::decision)?;
            driver = Some(follower);
        }
        driver
            .as_ref()
            .expect("each participant has a decision driver")
            .finish()
            .map_err(ChannelDecisionError::decision)?;
        let mut effective = sampler;
        effective
            .accept_effective(decision.feedback.agent())
            .map_err(ChannelDecisionPreflightError::Sampling)?;
        Ok(effective)
    })();
    let agreement = match coordination {
        Some(authority) => authority.agree(
            ChannelDecisionPhase::Candidate,
            &[
                decision.raw_agent_id as u64,
                decision.raw_function_id as u64,
                decision.feedback.agent() as u64,
                decision.feedback.function() as u64,
                decision.spoken_text as u64,
            ],
            candidate.is_ok(),
            context,
        ),
        None => Ok(()),
    };
    let effective_sampler = candidate?;
    agreement.map_err(ChannelDecisionError::Coordination)?;
    // Finish was validated before agreement. Adoption has no native work or
    // callbacks; RNG and effective history become visible together on this branch.
    branch
        .adopt_decision_driver(driver.expect("validated completed decision driver"))
        .map_err(ChannelDecisionError::decision)?;
    branch.samplers_mut()[0] = effective_sampler;
    Ok(decision)
}

fn validate_id(token: u32, vocabulary: usize) -> Result<(), ChannelDecisionPreflightError> {
    if token as usize >= vocabulary {
        return Err(ChannelDecisionPreflightError::Token { token, vocabulary });
    }
    Ok(())
}

/// Portable failures detected without consuming sampling state.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ChannelDecisionPreflightError {
    /// Published host identities disagree with their backend token values.
    #[error("VoiceChat published decisions have inconsistent token identities")]
    PublishedDecision,
    /// The admitted head domain cannot represent released controls.
    #[error("VoiceChat decision vocabulary {vocabulary} is outside released bounds")]
    Vocabulary {
        /// Supplied head width.
        vocabulary: usize,
    },
    /// An override, retained token, or backend result is outside its head domain.
    #[error("VoiceChat decision token {token} is outside vocabulary {vocabulary}")]
    Token {
        /// Invalid token.
        token: u32,
        /// Admitted head width.
        vocabulary: usize,
    },
    /// An invocation requires exactly one agent sampler.
    #[error("VoiceChat channel selection requires one sampler, received {actual}")]
    SamplerCount {
        /// Supplied sampler count.
        actual: usize,
    },
    /// Submitted work cannot accept further decisions.
    #[error("VoiceChat channel selection requires an unpublished branch without completion")]
    Submitted,
    /// Both heads must contain exactly one row of the admitted vocabulary.
    #[error("VoiceChat channel heads require one row of the admitted vocabulary")]
    Shape,
    /// The native logits must declare a supported floating representation.
    #[error("VoiceChat channel heads require known floating precision")]
    Precision,
    /// Agent sampling capacity or history is invalid.
    #[error(transparent)]
    Sampling(#[from] AgentSamplingError),
}

/// Decision failure retaining the native source without changing branch state.
#[derive(Debug, thiserror::Error)]
pub enum ChannelDecisionError<E> {
    /// A replicated phase or owner publication failed without adopting state.
    #[error("VoiceChat channel coordination failed: {0}")]
    Coordination(#[source] BackendFailure),
    /// Host conversational policy failed without adopting RNG or history.
    #[error("VoiceChat channel policy failed: {0}")]
    Resolution(#[source] BackendFailure),
    /// Portable admission failed.
    #[error(transparent)]
    Preflight(#[from] ChannelDecisionPreflightError),
    /// The shared driver rejected its decision plan.
    #[error(transparent)]
    Plan(#[from] SequentialDecisionPlanError),
    /// A shared decision sequencing invariant failed.
    #[error("VoiceChat sequential decision failed: {0}")]
    Decision(#[source] SequentialDecisionError<E>),
    /// A native mechanism failed; the original error remains the source.
    #[error("VoiceChat channel selection failed: {0}")]
    Backend(#[source] E),
}

impl<E> ChannelDecisionError<E> {
    fn decision(error: SequentialDecisionError<E>) -> Self {
        match error {
            SequentialDecisionError::Backend(error) => Self::Backend(error),
            error => Self::Decision(error),
        }
    }
}
