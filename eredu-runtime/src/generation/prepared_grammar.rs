//! Owning dynamic grammar decisions over the existing constrained sampler.
use super::*;
use crate::execution_control::{
    PreparedGrammarBranch, PreparedGrammarBranchError, PreparedGrammarChoice,
    PreparedGrammarChoiceError, TokenChoiceError,
};
use eredu_core::{
    speculative::{PreparedGrammarController, PreparedGrammarInstallError, PreparedGrammarSource},
    HostMetadataFunding, HostMetadataFundingError, HostPreparationAuthority,
};
use std::{
    convert::Infallible,
    mem::{size_of, size_of_val},
};

/// First real refusal before publishing a provisional sampler.
#[derive(Debug, thiserror::Error)]
pub enum PreparedGrammarSamplerCause<G: PreparedGrammarController> {
    /// This source does not expose the required closed workers.
    #[error("sampler has no complete prepared grammar and policy source")]
    Unknown,
    /// An ordinary/unprepared identity cannot bind a processed value.
    #[error(transparent)] Source(#[from] eredu_core::speculative::PlainControllerError),
    /// Exact policy/temperature branch selection refused before native work.
    #[error(transparent)] Choice(#[from] PreparedGreedyError),
    /// The actual account refused before the next constructor.
    #[error(transparent)]
    Funding(#[from] HostMetadataFundingError),
    /// The actual provisional parser and any failed prefix remain owned.
    #[error(transparent)]
    Decision(#[from] PreparedGrammarChoiceError<G>),
    /// Ordinary pending-choice validation rejected the committed ID.
    #[error(transparent)]
    Token(#[from] TokenChoiceError<Infallible>),
    /// The exact source-checked copy retains its failed destination.
    #[error(transparent)]
    Branch(#[from] PreparedGrammarBranchError<G>),
    /// The actual grammar operation retains its failed prefix.
    #[error("prepared grammar operation failed: {0}")]
    Operation(#[source] G::Error),
    /// Failed publication retains the supplied controller.
    #[error(transparent)]
    Install(#[from] PreparedGrammarInstallError<G>),
    /// The existing adaptive policy rejected its probability or commitment.
    #[error(transparent)]
    Adaptive(#[from] PreparedAdaptiveCommitError),
}
/// Failed parser/policy work is retired before its request funding.
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
pub struct PreparedGrammarSamplerError<G: PreparedGrammarController> {
    #[source]
    cause: PreparedGrammarSamplerCause<G>,
    funding: HostMetadataFunding,
}
impl<G: PreparedGrammarController> PreparedGrammarSamplerError<G> {
    /// Exact first refusal, including owning grammar failures.
    pub fn cause(&self) -> &PreparedGrammarSamplerCause<G> {
        &self.cause
    }
}
/// An actual independent grammar decision paired with its unchanged logit policy.
/// Consumers must authenticate the source and apply the mask before the policy.
#[derive(Debug)]
pub struct PreparedGrammarLogits<'a, G: PreparedGrammarController> {
    decision: PreparedGrammarChoice<G>,
    policy: PreparedLogitPolicy<'a>,
}
impl<'a, G: PreparedGrammarController> PreparedGrammarLogits<'a, G> {
    /// Transfers both parts; the decision retains the real completed mask owner.
    pub fn into_parts(self) -> (PreparedGrammarChoice<G>, PreparedLogitPolicy<'a>) {
        (self.decision, self.policy)
    }
}
/// Typed loan of the actual shared constrained sampler. It does not grant native
/// execution authority or convert a grammar into a fixed-controller source.
#[derive(Debug)]
pub struct PreparedGrammarSampler<'a, S, G: PreparedGrammarController> {
    source: &'a S,
    needs_probability: bool,
    validate_commit: fn(&S, u32) -> Result<(), TokenChoiceError<Infallible>>,
    source_view: for<'b> fn(&'b S) -> Option<PreparedGrammarSource<'b>>,
    choice: fn(&S, f32, &HostMetadataFunding) -> Result<PreparedControllerChoice, PreparedGrammarSamplerError<G>>,
    matches_choice: fn(&S, &PreparedControllerChoice) -> bool,

    logits: for<'b> fn(
        &'b S,
        &[u32],
        &HostMetadataFunding,
    ) -> Result<PreparedGrammarLogits<'b, G>, PreparedGrammarSamplerError<G>>,
    copy_bytes: fn(&S) -> Option<usize>,
    copy: fn(&S, &HostMetadataFunding) -> Result<S, PreparedGrammarSamplerError<G>>,
    commit:
        fn(&S, u32, Option<f32>, &HostMetadataFunding) -> Result<S, PreparedGrammarSamplerError<G>>,
    force: fn(
        &S,
        u32,
        TokenDomain,
        usize,
        &HostMetadataFunding,
    ) -> Result<S, PreparedGrammarSamplerError<G>>,
    clear: fn(&S, &HostMetadataFunding) -> Result<S, PreparedGrammarSamplerError<G>>,
    prefix: fn(&S, &[u32], &HostMetadataFunding) -> Result<bool, PreparedGrammarSamplerError<G>>,
    greedy: fn(&S, f32) -> Result<SpeculativeGreedyProgram, PreparedGreedyError>,
    categorical: fn(&S, f32) -> Result<SpeculativeCategoricalProgram, PreparedGreedyError>,
}
impl<'a, S, G: PreparedGrammarController> PreparedGrammarSampler<'a, S, G> {
    /// Whether this actual policy consumes a native accepted-token probability.
    pub fn requires_commit_probability(&self) -> bool { self.needs_probability }
    /// Same pending-choice validation before native probability observation.
    pub fn validate_commit(&self, token: u32) -> Result<(), TokenChoiceError<Infallible>> {
        (self.validate_commit)(self.source, token)
    }
    /// Actual recipe/declaration/tokenizer/history; no execution authority.
    pub fn controller_source(&self) -> Option<PreparedGrammarSource<'a>> {
        (self.source_view)(self.source)
    }
    /// Retains the actual canonical source and selected policy branch. Only a
    /// successful native mask/policy producer may attach it to a numerical value.
    pub fn choice(&self, temperature: f32, funding: &HostMetadataFunding) -> Result<PreparedControllerChoice, PreparedGrammarSamplerError<G>> {
        (self.choice)(self.source, temperature, funding)
    }
    /// Rejects values from a different history allocation, source or forced state.
    pub fn matches_choice(&self, choice: &PreparedControllerChoice) -> bool {
        (self.matches_choice)(self.source, choice)
    }
    /// Pays and computes the same provisional history before the logit policy.
    pub fn logits(
        &self,
        history: &[u32],
        funding: &HostMetadataFunding,
    ) -> Result<PreparedGrammarLogits<'a, G>, PreparedGrammarSamplerError<G>> {
        (self.logits)(self.source, history, funding)
    }
    /// Complete exact policy/grammar/installation copy payment from this
    /// committed source, without executing a callback or constructing a parser.
    pub fn copy_metadata_bytes(&self) -> Option<usize> { (self.copy_bytes)(self.source) }
    /// Copies the policy and every mutable grammar field under new funding.
    pub fn copy(self, funding: &HostMetadataFunding) -> Result<S, PreparedGrammarSamplerError<G>> {
        (self.copy)(self.source, funding)
    }
    /// Builds a provisional successor using the ordinary policy-before-controller
    /// sequence. Adaptive policies require the actual observed probability.
    pub fn commit(
        self,
        token: u32,
        probability: Option<f32>,
        funding: &HostMetadataFunding,
    ) -> Result<S, PreparedGrammarSamplerError<G>> {
        (self.commit)(self.source, token, probability, funding)
    }
    /// Copies and stages a future choice with the ordinary validation order.
    pub fn force(
        self,
        token: u32,
        domain: TokenDomain,
        position: usize,
        funding: &HostMetadataFunding,
    ) -> Result<S, PreparedGrammarSamplerError<G>> {
        (self.force)(self.source, token, domain, position, funding)
    }
    /// Copies and clears a future choice without committing or drawing randomness.
    pub fn clear(self, funding: &HostMetadataFunding) -> Result<S, PreparedGrammarSamplerError<G>> {
        (self.clear)(self.source, funding)
    }
    /// Checks prospective history before the same provisional terminal policy.
    pub fn prefix_is_complete(
        &self,
        history: &[u32],
        funding: &HostMetadataFunding,
    ) -> Result<bool, PreparedGrammarSamplerError<G>> {
        (self.prefix)(self.source, history, funding)
    }
    /// Exact existing greedy worker; processed-value authentication is separate.
    pub fn greedy(
        &self,
        temperature: f32,
    ) -> Result<SpeculativeGreedyProgram, PreparedGreedyError> {
        (self.greedy)(self.source, temperature)
    }
    /// Exact existing categorical worker; explicit-key admission remains separate.
    pub fn categorical(
        &self,
        temperature: f32,
    ) -> Result<SpeculativeCategoricalProgram, PreparedGreedyError> {
        (self.categorical)(self.source, temperature)
    }
}
fn failure<G: PreparedGrammarController>(
    cause: PreparedGrammarSamplerCause<G>,
    funding: &HostMetadataFunding,
) -> PreparedGrammarSamplerError<G> {
    PreparedGrammarSamplerError {
        cause,
        funding: funding.clone(),
    }
}
fn controls<P, C: SpeculativeTokenFilterController>() -> Option<usize> {
    type S<P, C> = ConstrainedSampler<P, C>;
    let parts = [
        size_of::<S<P, C>>(),
        PreparedControllerChoice::metadata_bytes(),
        size_of::<Result<PreparedControllerChoice, PreparedGrammarSamplerError<C::PreparedGrammar>>>(),
        size_of::<Result<PreparedControllerChoice, PreparedGrammarSamplerCause<C::PreparedGrammar>>>(),
        size_of::<(&S<P,C>, f32, &HostMetadataFunding)>(),

        size_of::<P>(),
        size_of::<TokenChoiceController<C>>(),
        size_of::<C::PreparedGrammar>(),
        size_of::<PreparedGrammarBranch<C::PreparedGrammar>>(),
        size_of::<
            Result<
                PreparedGrammarBranch<C::PreparedGrammar>,
                PreparedGrammarBranchError<C::PreparedGrammar>,
            >,
        >(),
        size_of::<Option<P>>(),
        size_of::<Option<C::PreparedGrammar>>(),
        size_of::<PreparedGrammarSampler<'_, S<P, C>, C::PreparedGrammar>>(),
        size_of::<PreparedGrammarLogits<'_, C::PreparedGrammar>>(),
        size_of::<PreparedGrammarSamplerCause<C::PreparedGrammar>>(),
        size_of::<PreparedGrammarSamplerError<C::PreparedGrammar>>(),
        size_of::<Result<S<P, C>, PreparedGrammarSamplerCause<C::PreparedGrammar>>>(),
        size_of::<Result<S<P, C>, PreparedGrammarSamplerError<C::PreparedGrammar>>>(),
        size_of::<
            Result<
                PreparedGrammarLogits<'_, C::PreparedGrammar>,
                PreparedGrammarSamplerError<C::PreparedGrammar>,
            >,
        >(),
        size_of::<Result<bool, PreparedGrammarSamplerCause<C::PreparedGrammar>>>(),
        size_of::<Result<bool, PreparedGrammarSamplerError<C::PreparedGrammar>>>(),
        size_of::<Result<(), PreparedGrammarSamplerCause<C::PreparedGrammar>>>(),
        size_of::<
            Result<
                C::PreparedGrammar,
                <C::PreparedGrammar as PreparedGrammarController>::CopyError,
            >,
        >(),
        size_of::<
            Result<C::PreparedGrammar, <C::PreparedGrammar as PreparedGrammarController>::Error>,
        >(),
        size_of::<PreparedSpeculativeSamplerCopy<'_, P>>(),
        size_of::<PreparedAdaptiveCommit<'_, P>>(),
        size_of::<Option<PreparedAdaptiveCommit<'_, P>>>(),
        size_of::<PreparedGreedyPolicy<'_>>(),
        size_of::<Option<PreparedLogitPolicy<'_>>>(),
        size_of::<HostPreparationAuthority>(),
        size_of::<(&S<P, C>, &[u32], &HostMetadataFunding)>(),
        size_of::<(&S<P, C>, u32, Option<f32>, &HostMetadataFunding)>(),
        size_of::<(&S<P, C>, u32, TokenDomain, usize, &HostMetadataFunding)>(),
        size_of::<(&mut Option<P>, &mut Option<C::PreparedGrammar>, u32)>(),
        size_of::<(&S<P, C>, &HostMetadataFunding, Option<f32>)>(),
        size_of::<(&C::PreparedGrammar, &HostMetadataFunding, usize)>(),
        size_of::<(usize, u32, bool)>(),
        PreparedGrammarSource::control_bytes()?,
        HostMetadataFunding::reservation_control_bytes(),
    ];
    parts
        .into_iter()
        .try_fold(size_of_val(&parts), usize::checked_add)
}
fn reserve<P, C: SpeculativeTokenFilterController>(
    funding: &HostMetadataFunding,
) -> Result<(), PreparedGrammarSamplerCause<C::PreparedGrammar>> {
    funding.reserve_metadata(controls::<P, C>().ok_or(HostMetadataFundingError::Overflow)?)?;
    Ok(())
}
pub(super) fn constrained<B, P, C>(
    source: &ConstrainedSampler<P, C>,
) -> Option<PreparedGrammarSampler<'_, ConstrainedSampler<P, C>, C::PreparedGrammar>>
where
    B: SamplingBackend,
    P: SpeculativeSampler<B> + Clone,
    C: SpeculativeTokenFilterController,
{
    source.controller.grammar_source()?;
    source.policy.prepared_host_copy()?;
    source.policy.prepared_logit_policy()?;
    if source.policy.prepared_greedy_policy().is_none()
        && source.policy.prepared_adaptive_commit().is_none()
    {
        return None;
    }
    Some(PreparedGrammarSampler {
        source,
        needs_probability: source.policy.prepared_adaptive_commit().is_some(),
        validate_commit: |source, token| source.controller.validate_grammar_commit(token),
        source_view: |source| {
            Some(
                source
                    .controller
                    .grammar_source()?
                    .prepared_grammar_source(),
            )
        },
        choice: choice::<B,P,C>,
        matches_choice: |source, choice| match choice.identity() {
            crate::execution_control::ControllerChoiceIdentity::Grammar(identity) => source.controller.matches_grammar_choice(identity),
            _ => false,
        },
        logits: logits::<B, P, C>,
        copy_bytes: copy_bytes::<B, P, C>,
        copy: copy::<B, P, C>,
        commit: commit::<B, P, C>,
        force: force::<B, P, C>,
        clear: clear::<B, P, C>,
        prefix: prefix::<B, P, C>,
        greedy: |source, temperature| {
            source
                .policy
                .prepared_greedy_policy()
                .ok_or(PreparedGreedyError::Unknown)?
                .greedy(temperature)
        },
        categorical: |source, temperature| {
            source
                .policy
                .prepared_categorical_policy()
                .ok_or(PreparedGreedyError::Unknown)?
                .bind(temperature)
        },
    })
}
fn logits<'a, B, P, C>(
    source: &'a ConstrainedSampler<P, C>,
    history: &[u32],
    funding: &HostMetadataFunding,
) -> Result<
    PreparedGrammarLogits<'a, C::PreparedGrammar>,
    PreparedGrammarSamplerError<C::PreparedGrammar>,
>
where
    B: SamplingBackend,
    P: SpeculativeSampler<B> + Clone,
    C: SpeculativeTokenFilterController,
{
    let result = (|| {
        reserve::<P, C>(funding)?;
        let decision =
            source
                .controller
                .prepared_grammar_decision(history, history.len(), funding)?;
        let policy = source
            .policy
            .prepared_logit_policy()
            .ok_or(PreparedGrammarSamplerCause::Unknown)?;
        Ok(PreparedGrammarLogits { decision, policy })
    })();
    result.map_err(|cause| failure(cause, funding))
}
fn copy_bytes<B, P, C>(source: &ConstrainedSampler<P, C>) -> Option<usize>
where B: SamplingBackend, P: SpeculativeSampler<B> + Clone,
    C: SpeculativeTokenFilterController,
{
    let grammar = source.controller.grammar_source()?;
    let capacity = grammar.prepared_grammar_source().history().len();
    controls::<P, C>()?
        .checked_add(source.policy.prepared_host_copy()?.metadata_bytes())?
        .checked_add(PreparedGrammarBranch::committed_copy_bytes(grammar, capacity)?)?
        .checked_add(source.controller.prepared_grammar_replacement_bytes()?)
}
fn copy<B, P, C>(
    source: &ConstrainedSampler<P, C>,
    funding: &HostMetadataFunding,
) -> Result<ConstrainedSampler<P, C>, PreparedGrammarSamplerError<C::PreparedGrammar>>
where
    B: SamplingBackend,
    P: SpeculativeSampler<B> + Clone,
    C: SpeculativeTokenFilterController,
{
    let result = (|| {
        reserve::<P, C>(funding)?;
        let plan = source
            .policy
            .prepared_host_copy()
            .ok_or(PreparedGrammarSamplerCause::Unknown)?;
        funding.reserve_metadata(plan.metadata_bytes())?;
        let policy = plan.copy();
        let grammar = source
            .controller
            .grammar_source()
            .ok_or(PreparedGrammarSamplerCause::Unknown)?;
        let history = grammar.prepared_grammar_source().history();
        let copied = PreparedGrammarBranch::fork_at(grammar, history, history.len(), funding)?
            .into_controller();
        let controller = source
            .controller
            .replace_prepared_grammar(copied, funding)?;
        Ok(ConstrainedSampler { policy, controller })
    })();
    result.map_err(|cause| failure(cause, funding))
}
fn commit<B, P, C>(
    source: &ConstrainedSampler<P, C>,
    token: u32,
    probability: Option<f32>,
    funding: &HostMetadataFunding,
) -> Result<ConstrainedSampler<P, C>, PreparedGrammarSamplerError<C::PreparedGrammar>>
where
    B: SamplingBackend,
    P: SpeculativeSampler<B> + Clone,
    C: SpeculativeTokenFilterController,
{
    let result = (|| {
        reserve::<P, C>(funding)?;
        source.controller.validate_grammar_commit(token)?;
        let original = source
            .controller
            .grammar_source()
            .ok_or(PreparedGrammarSamplerCause::Unknown)?;
        let capacity = original
            .prepared_grammar_source()
            .history()
            .len()
            .checked_add(1)
            .ok_or(HostMetadataFundingError::Overflow)?;
        let mut policy = None;
        let mut grammar = None;
        commit_components(
            &mut policy,
            &mut grammar,
            token,
            |destination, token| -> Result<(), PreparedGrammarSamplerCause<C::PreparedGrammar>> {
                *destination = Some(
                    if let Some(plan) = source.policy.prepared_adaptive_commit() {
                        let probability =
                            probability.ok_or(PreparedAdaptiveCommitError::Probability)?;
                        let bytes = plan
                            .metadata_bytes()
                            .checked_add(
                                HostPreparationAuthority::retention_bytes::<HostMetadataFunding>()
                                    .ok_or(HostMetadataFundingError::Overflow)?,
                            )
                            .ok_or(HostMetadataFundingError::Overflow)?;
                        funding.reserve_metadata(bytes)?;
                        plan.commit(
                            token,
                            probability,
                            HostPreparationAuthority::retain(funding.clone()),
                        )?
                    } else {
                        let commit = source
                            .policy
                            .prepared_greedy_policy()
                            .ok_or(PreparedGrammarSamplerCause::Unknown)?;
                        let copy = source
                            .policy
                            .prepared_host_copy()
                            .ok_or(PreparedGrammarSamplerCause::Unknown)?;
                        funding.reserve_metadata(copy.metadata_bytes())?;
                        let copied = copy.copy();
                        commit.commit_without_mutation();
                        copied
                    },
                );
                Ok(())
            },
            |destination, token| -> Result<(), PreparedGrammarSamplerCause<C::PreparedGrammar>> {
                let copied = PreparedGrammarBranch::fork_at(
                    original,
                    original.prepared_grammar_source().history(),
                    capacity,
                    funding,
                )?
                .into_controller();
                *destination = Some(
                    copied
                        .commit_prepared_grammar(token)
                        .map_err(PreparedGrammarSamplerCause::Operation)?,
                );
                Ok(())
            },
        )?;
        let mut controller = source
            .controller
            .replace_prepared_grammar(grammar.take().expect("committed grammar"), funding)?;
        controller.finish_grammar_commit();
        Ok(ConstrainedSampler {
            policy: policy.take().expect("committed policy"),
            controller,
        })
    })();
    result.map_err(|cause| failure(cause, funding))
}
fn force<B, P, C>(
    source: &ConstrainedSampler<P, C>,
    token: u32,
    domain: TokenDomain,
    position: usize,
    funding: &HostMetadataFunding,
) -> Result<ConstrainedSampler<P, C>, PreparedGrammarSamplerError<C::PreparedGrammar>>
where
    B: SamplingBackend,
    P: SpeculativeSampler<B> + Clone,
    C: SpeculativeTokenFilterController,
{
    source
        .controller
        .validate_grammar_force(token, domain, funding)
        .map_err(|cause| failure(cause.into(), funding))?;
    let mut provisional = copy::<B, P, C>(source, funding)?;
    provisional
        .controller
        .install_grammar_force(token, domain, position);
    Ok(provisional)
}
fn clear<B, P, C>(
    source: &ConstrainedSampler<P, C>,
    funding: &HostMetadataFunding,
) -> Result<ConstrainedSampler<P, C>, PreparedGrammarSamplerError<C::PreparedGrammar>>
where
    B: SamplingBackend,
    P: SpeculativeSampler<B> + Clone,
    C: SpeculativeTokenFilterController,
{
    let mut provisional = copy::<B, P, C>(source, funding)?;
    provisional.controller.clear_forced();
    Ok(provisional)
}
fn prefix<B, P, C>(
    source: &ConstrainedSampler<P, C>,
    history: &[u32],
    funding: &HostMetadataFunding,
) -> Result<bool, PreparedGrammarSamplerError<C::PreparedGrammar>>
where
    B: SamplingBackend,
    P: SpeculativeSampler<B> + Clone,
    C: SpeculativeTokenFilterController,
{
    let result = (|| {
        reserve::<P, C>(funding)?;
        Ok(source
            .controller
            .prepared_grammar_prefix_complete(history, history.len(), funding)?)
    })();
    result.map_err(|cause| failure(cause, funding))
}

fn choice<B, P, C>(source: &ConstrainedSampler<P,C>, temperature: f32, funding: &HostMetadataFunding)
    -> Result<PreparedControllerChoice, PreparedGrammarSamplerError<C::PreparedGrammar>>
where B: SamplingBackend, P: SpeculativeSampler<B> + Clone, C: SpeculativeTokenFilterController {
    let result = (|| {
        reserve::<P,C>(funding)?;
        let identity = crate::execution_control::ControllerChoiceIdentity::Grammar(source.controller.retain_grammar_choice_identity()?);
        Ok(PreparedControllerChoice::from_identity::<B,P>(&source.policy, temperature, identity)?)
    })();
    result.map_err(|cause| failure(cause, funding))
}
