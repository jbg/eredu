//! Paid provisional fixed-controller copies over the existing commit sequence.
use super::*;
use crate::execution_control::{
    PreparedControllerCause, PreparedControllerDecision, PreparedControllerSource, TokenChoiceError,
};
use eredu_core::{HostPreparationAuthority, speculative::PlainControllerError};
use std::mem::{size_of, size_of_val};

/// Fixed decision/copy refusal, preserving allocation-error custody inline.
pub type PreparedControllerError = TokenChoiceError<PreparedControllerCause>;
/// Both parts of an actual constrained logit program. A consumer must apply the
/// decision's authentic tokenizer filter/forced choice before this policy; the
/// policy alone is never proof for the constrained sampler.
#[derive(Debug)]
pub struct PreparedControlledLogits<'a> {
    decision: PreparedControllerDecision<'a>,
    policy: PreparedLogitPolicy<'a>,
}
impl<'a> PreparedControlledLogits<'a> {
    /// Transfers the paired decision and ordinary policy. The actual source
    /// storage still needs native authentication and its own admitted producer.
    pub fn into_parts(self) -> (PreparedControllerDecision<'a>, PreparedLogitPolicy<'a>) {
        (self.decision, self.policy)
    }
}
/// Borrows one complete known fixed-controller sampler. Copy/commit destinations
/// are source-bound and do not invoke opaque filter or commit callbacks. No host
/// funding, numerical permission or final state publication is supplied here.
#[derive(Debug)]
pub struct PreparedSpeculativeController<'a, S> {
    source: &'a S,
    choice: fn(&S, f32) -> Result<PreparedControllerChoice, PreparedControlledChoiceError>,
    matches_choice: fn(&S, &PreparedControllerChoice) -> bool,
    controller_source:
        for<'b> fn(&'b S) -> Result<PreparedControllerSource<'b>, PreparedControllerError>,
    copy_bytes: usize,
    commit_bytes: usize,
    copy: fn(&S, HostPreparationAuthority) -> Result<S, PreparedControllerError>,
    commit: fn(&S, u32, HostPreparationAuthority) -> Result<S, PreparedControllerError>,
    validate: fn(&S, u32) -> Result<(), PreparedControllerError>,
    validate_force: fn(&S, u32, crate::TokenDomain) -> Result<(), PreparedControllerError>,
    force: fn(
        &S,
        u32,
        crate::TokenDomain,
        usize,
        HostPreparationAuthority,
    ) -> Result<S, PreparedControllerError>,
    clear: fn(&S, HostPreparationAuthority) -> Result<S, PreparedControllerError>,
    prefix: fn(&S, &[u32]) -> Result<bool, PreparedControllerError>,
    logits:
        for<'b> fn(&'b S, &[u32]) -> Result<PreparedControlledLogits<'b>, PreparedControllerError>,
}
impl<'a, S> PreparedSpeculativeController<'a, S> {
    /// Retains only exact source identity and closed choice programs. The native
    /// consumer may attach this to a value only after its filter/policy succeeds.
    pub fn choice(
        &self,
        temperature: f32,
    ) -> Result<PreparedControllerChoice, PreparedControlledChoiceError> {
        (self.choice)(self.source, temperature)
    }
    /// Rejects a value produced before a controller replacement or forced change.
    pub fn matches_choice(&self, choice: &PreparedControllerChoice) -> bool {
        (self.matches_choice)(self.source, choice)
    }
    /// Exact immutable controller source; numerical use still needs source
    /// authentication and the paired fixed decision from logits(history).
    pub fn controller_source(
        &self,
    ) -> Result<PreparedControllerSource<'a>, PreparedControllerError> {
        (self.controller_source)(self.source)
    }
    /// Neither fixed controller has a grammar completion condition.
    pub fn grammar_is_complete(&self) -> bool {
        false
    }
    /// Preserves the actual forced-prefix then durable-prefix validation order.
    pub fn prefix_is_complete(&self, history: &[u32]) -> Result<bool, PreparedControllerError> {
        (self.prefix)(self.source, history)
    }
    /// Complete copy destination/control request before a snapshot copy.
    pub fn copy_metadata_bytes(&self) -> usize {
        self.copy_bytes
    }
    /// Complete provisional destination/control request including one new token.
    pub fn commit_metadata_bytes(&self) -> usize {
        self.commit_bytes
    }
    /// Paired fixed source decision and policy; unknown histories refuse first.
    pub fn logits(
        &self,
        history: &[u32],
    ) -> Result<PreparedControlledLogits<'a>, PreparedControllerError> {
        (self.logits)(self.source, history)
    }
    /// Copies the actual source after the caller pays copy_metadata_bytes.
    pub fn copy(self, host: HostPreparationAuthority) -> Result<S, PreparedControllerError> {
        (self.copy)(self.source, host)
    }
    /// Validates prospective forcing with the same pending/domain/filter order
    /// as the ordinary controller, without creating a filter or destination.
    pub fn validate_force(
        &self,
        token: u32,
        domain: crate::TokenDomain,
    ) -> Result<(), PreparedControllerError> {
        (self.validate_force)(self.source, token, domain)
    }
    /// Copies and stages one future choice after copy_metadata_bytes is paid.
    /// The source remains immutable even when validation or allocation fails.
    pub fn force(
        self,
        token: u32,
        domain: crate::TokenDomain,
        position: usize,
        host: HostPreparationAuthority,
    ) -> Result<S, PreparedControllerError> {
        (self.force)(self.source, token, domain, position, host)
    }
    /// Copies and clears the prospective choice after copy_metadata_bytes is paid.
    /// This does not change the committed history or any random state.
    pub fn clear(self, host: HostPreparationAuthority) -> Result<S, PreparedControllerError> {
        (self.clear)(self.source, host)
    }
    /// Checks the exact pending choice and tokenizer domain before admission.
    pub fn validate_commit(&self, token: u32) -> Result<(), PreparedControllerError> {
        (self.validate)(self.source, token)
    }
    /// Builds and commits a provisional copy for an unchanged policy commit.
    /// Adaptive policies use their separate PreparedAdaptiveCommit witness and
    /// actual observed probability. Only success may replace the old sampler.
    pub fn commit(
        self,
        token: u32,
        host: HostPreparationAuthority,
    ) -> Result<S, PreparedControllerError> {
        (self.commit)(self.source, token, host)
    }
}
fn unknown() -> PreparedControllerError {
    TokenChoiceError::Constraint(PreparedControllerCause::Plain(
        PlainControllerError::Unknown,
    ))
}
fn controller_controls<S>(policy: usize, controller: usize) -> Option<usize> {
    let parts = [
        policy,
        controller,
        size_of::<S>(),
        size_of::<PreparedSpeculativeController<'_, S>>(),
        size_of::<Result<S, PreparedControllerError>>(),
        size_of::<PreparedControllerError>(),
        size_of::<Result<(), PreparedControllerError>>(),
        size_of::<crate::TokenDomain>(),
        size_of::<usize>(),
        size_of::<u32>(),
        size_of::<bool>(),
        size_of::<Option<u32>>(),
        size_of::<Result<bool, PlainControllerError>>(),
        size_of::<PreparedControlledLogits<'_>>(),
        size_of::<Result<PreparedControlledLogits<'_>, PreparedControllerError>>(),
        size_of::<HostPreparationAuthority>(),
    ];
    parts
        .into_iter()
        .try_fold(size_of_val(&parts), usize::checked_add)
}
pub(super) fn constrained<B, P, C>(
    source: &ConstrainedSampler<P, C>,
) -> Option<PreparedSpeculativeController<'_, ConstrainedSampler<P, C>>>
where
    B: SamplingBackend,
    P: SpeculativeSampler<B> + Clone,
    C: SpeculativeTokenFilterController,
{
    let policy = source.policy.prepared_host_copy()?;
    if source.policy.prepared_greedy_policy().is_none()
        && source.policy.prepared_adaptive_commit().is_none()
    {
        return None;
    }
    let length = source.controller.fixed_history_len()?;
    let copy_bytes = controller_controls::<ConstrainedSampler<P, C>>(
        policy.metadata_bytes(),
        source.controller.prepared_copy_bytes(length)?,
    )?;
    let commit_bytes = controller_controls::<ConstrainedSampler<P, C>>(
        policy.metadata_bytes(),
        source
            .controller
            .prepared_copy_bytes(length.checked_add(1)?)?,
    )?;
    Some(PreparedSpeculativeController {
        source,
        choice: choice::<B, P, C>,
        matches_choice: |source, choice| source.controller.matches_fixed_choice(&choice.identity),
        controller_source: |source| source.controller.prepared_source().ok_or_else(unknown),
        copy_bytes,
        commit_bytes,
        copy: copy::<B, P, C>,
        commit: commit::<B, P, C>,
        validate: |source, token| source.controller.validate_fixed_commit(token),
        validate_force: |source, token, domain| {
            source.controller.validate_fixed_force(token, domain)
        },
        force: force::<B, P, C>,
        clear: clear::<B, P, C>,
        prefix: |source, history| source.controller.prepared_fixed_prefix_complete(history),
        logits: logits::<B, P, C>,
    })
}
pub(super) fn copied<B, P, C>(
    source: &ConstrainedSampler<P, C>,
    extra: usize,
    host: HostPreparationAuthority,
) -> Result<ConstrainedSampler<P, C>, PreparedControllerError>
where
    B: SamplingBackend,
    P: SpeculativeSampler<B> + Clone,
    C: SpeculativeTokenFilterController,
{
    if host.is_unmanaged() {
        return Err(unknown());
    }
    let policy = source
        .policy
        .prepared_host_copy()
        .ok_or_else(unknown)?
        .copy();
    let capacity = source
        .controller
        .fixed_history_len()
        .and_then(|n| n.checked_add(extra))
        .ok_or(TokenChoiceError::Constraint(PlainControllerError::Overflow))?;
    let controller = source
        .controller
        .copy_prepared_fixed(capacity, host.clone())
        .map_err(TokenChoiceError::Constraint)?;
    Ok(ConstrainedSampler { policy, controller })
}
fn copy<B, P, C>(
    source: &ConstrainedSampler<P, C>,
    host: HostPreparationAuthority,
) -> Result<ConstrainedSampler<P, C>, PreparedControllerError>
where
    B: SamplingBackend,
    P: SpeculativeSampler<B> + Clone,
    C: SpeculativeTokenFilterController,
{
    copied::<B, P, C>(source, 0, host)
}
fn force<B, P, C>(
    source: &ConstrainedSampler<P, C>,
    token: u32,
    domain: crate::TokenDomain,
    position: usize,
    host: HostPreparationAuthority,
) -> Result<ConstrainedSampler<P, C>, PreparedControllerError>
where
    B: SamplingBackend,
    P: SpeculativeSampler<B> + Clone,
    C: SpeculativeTokenFilterController,
{
    source.controller.validate_fixed_force(token, domain)?;
    let mut provisional = copied::<B, P, C>(source, 0, host)?;
    provisional
        .controller
        .force_prepared_fixed(token, domain, position)?;
    Ok(provisional)
}
fn clear<B, P, C>(
    source: &ConstrainedSampler<P, C>,
    host: HostPreparationAuthority,
) -> Result<ConstrainedSampler<P, C>, PreparedControllerError>
where
    B: SamplingBackend,
    P: SpeculativeSampler<B> + Clone,
    C: SpeculativeTokenFilterController,
{
    let mut provisional = copied::<B, P, C>(source, 0, host)?;
    provisional.controller.clear_forced();
    Ok(provisional)
}
fn commit<B, P, C>(
    source: &ConstrainedSampler<P, C>,
    token: u32,
    host: HostPreparationAuthority,
) -> Result<ConstrainedSampler<P, C>, PreparedControllerError>
where
    B: SamplingBackend,
    P: SpeculativeSampler<B> + Clone,
    C: SpeculativeTokenFilterController,
{
    source.controller.validate_fixed_commit(token)?;
    source.policy.prepared_greedy_policy().ok_or_else(unknown)?;
    let mut provisional = copied::<B, P, C>(source, 1, host)?;
    commit_components(
        &mut provisional.policy,
        &mut provisional.controller,
        token,
        |policy, _| {
            policy
                .prepared_greedy_policy()
                .ok_or_else(unknown)?
                .commit_without_mutation();
            Ok(())
        },
        |controller, token| controller.commit_prepared_fixed(token),
    )?;
    Ok(provisional)
}
fn logits<'a, B, P, C>(
    source: &'a ConstrainedSampler<P, C>,
    history: &[u32],
) -> Result<PreparedControlledLogits<'a>, PreparedControllerError>
where
    B: SamplingBackend,
    P: SpeculativeSampler<B> + Clone,
    C: SpeculativeTokenFilterController,
{
    let decision = source.controller.prepared_fixed_decision(history)?;
    let policy = source.policy.prepared_logit_policy().ok_or_else(unknown)?;
    Ok(PreparedControlledLogits { decision, policy })
}

/// Fixed controller/choice refusal before native work or provisional publication.
#[derive(Debug, thiserror::Error)]
pub enum PreparedControlledChoiceError {
    /// Actual retained source could not produce an immutable identity.
    #[error(transparent)]
    Source(#[from] PreparedControllerCause),
    /// The actual prepared policy does not support this numerical choice.
    #[error(transparent)]
    Choice(#[from] PreparedGreedyError),
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Choice {
    Greedy(SpeculativeGreedyProgram),
    Categorical(SpeculativeCategoricalProgram, u32),
}
/// Source-bound immutable choice selection. This proves neither completion nor
/// that a filter was applied; only the native producer can bind it to a value.
#[derive(Debug, Clone)]
pub struct PreparedControllerChoice {
    identity: crate::execution_control::ControllerChoiceIdentity,
    choice: Choice,
}
impl PreparedControllerChoice {
    pub(super) fn from_identity<B, P>(policy: &P, temperature: f32, identity: crate::execution_control::ControllerChoiceIdentity)
        -> Result<Self, PreparedGreedyError>
    where B: SamplingBackend, P: SpeculativeSampler<B> {
        Ok(Self { identity, choice: selected_choice::<B,P>(policy, temperature)? })
    }
    pub(super) fn identity(&self) -> &crate::execution_control::ControllerChoiceIdentity { &self.identity }
    /// Exact source and selected temperature/program, used for derived values.
    pub fn same_source(&self, other: &Self) -> bool {
        self.choice == other.choice && self.identity.same_source(&other.identity)
    }
    /// Closed zero-temperature branch selected for this processed value.
    pub fn greedy(
        &self,
        temperature: f32,
    ) -> Result<SpeculativeGreedyProgram, PreparedGreedyError> {
        match self.choice {
            Choice::Greedy(program) if temperature == 0.0 => Ok(program),
            _ => Err(PreparedGreedyError::Unknown),
        }
    }
    /// Closed explicit-key branch selected for this processed value.
    pub fn categorical(
        &self,
        temperature: f32,
    ) -> Result<SpeculativeCategoricalProgram, PreparedGreedyError> {
        match self.choice {
            Choice::Categorical(program, bits) if temperature.to_bits() == bits => Ok(program),
            _ => Err(PreparedGreedyError::Unknown),
        }
    }
    /// Fixed source-alias/control copies; existing shared allocations are reused.
    pub fn metadata_bytes() -> usize {
        size_of::<Self>()
            + size_of::<Option<Self>>()
            + size_of::<Result<Self, PreparedControlledChoiceError>>()
            + eredu_core::speculative::PreparedPlainControllerIdentity::metadata_bytes()
            + eredu_core::speculative::PreparedGrammarIdentity::metadata_bytes()
            + size_of::<eredu_core::speculative::PreparedForbiddenControllerIdentity>()
            + size_of::<eredu_core::speculative::ForbiddenControllerInputs>()
            + size_of::<eredu_core::speculative::PlainControllerHistory>()
            + size_of::<eredu_core::SharedTokenFilter>()
    }
}
fn choice<B, P, C>(
    source: &ConstrainedSampler<P, C>,
    temperature: f32,
) -> Result<PreparedControllerChoice, PreparedControlledChoiceError>
where
    B: SamplingBackend,
    P: SpeculativeSampler<B> + Clone,
    C: SpeculativeTokenFilterController,
{
    let choice = selected_choice::<B,P>(&source.policy, temperature)?;
    Ok(PreparedControllerChoice {
        identity: source.controller.retain_fixed_choice_identity()?,
        choice,
    })
}

fn selected_choice<B, P>(policy: &P, temperature: f32) -> Result<Choice, PreparedGreedyError>
where B: SamplingBackend, P: SpeculativeSampler<B> {
    Ok(if temperature == 0.0 {
        Choice::Greedy(
            policy
                .prepared_greedy_policy()
                .ok_or(PreparedGreedyError::Unknown)?
                .greedy(temperature)?,
        )
    } else {
        Choice::Categorical(
            policy
                .prepared_categorical_policy()
                .ok_or(PreparedGreedyError::Unknown)?
                .bind(temperature)?,
            temperature.to_bits(),
        )
    })
}
