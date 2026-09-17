//! Dynamic grammar decisions over the same source history and prospective choice.
use super::*;
use crate::execution_control::{PreparedGrammarBranch, PreparedGrammarBranchError};
use eredu_core::{
    speculative::{PreparedGrammarController, PreparedGrammarSource},
    HostMetadataFunding, HostMetadataFundingError, PackedTokenFilter, PackedTokenFilterError,
};
use std::{
    convert::Infallible,
    mem::{size_of, size_of_val},
};

/// First source, parser, or prospective-choice refusal.
#[derive(Debug, thiserror::Error)]
pub enum PreparedGrammarChoiceCause<G: PreparedGrammarController> {
    /// No unambiguous actual grammar producer is attached to this controller.
    #[error("controller has no unique prepared grammar source")]
    Unknown,
    /// The actual account refused before the next choice operation.
    #[error(transparent)]
    Funding(#[from] HostMetadataFundingError),
    /// The actual copied parser/history prefix is retained by its failure.
    #[error(transparent)]
    Branch(#[from] PreparedGrammarBranchError<G>),
    /// Ordinary pending-choice/domain semantics rejected this token or history.
    #[error(transparent)]
    Choice(#[from] TokenChoiceError<Infallible>),
    /// The actual completed mask is unavailable.
    #[error(transparent)]
    Mask(#[from] PackedTokenFilterError),
}
/// A rejected choice retains its completed branch before returning its account.
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
pub struct PreparedGrammarChoiceError<G: PreparedGrammarController> {
    #[source]
    cause: PreparedGrammarChoiceCause<G>,
    branch: Option<PreparedGrammarBranch<G>>,
    funding: HostMetadataFunding,
}
impl<G: PreparedGrammarController> PreparedGrammarChoiceError<G> {
    /// Exact first refusal, including owning parser failures.
    pub fn cause(&self) -> &PreparedGrammarChoiceCause<G> {
        &self.cause
    }
}
/// Actual provisional grammar and the same one-position forced restriction.
#[derive(Debug)]
pub struct PreparedGrammarChoice<G: PreparedGrammarController> {
    branch: PreparedGrammarBranch<G>,
    forced: Option<u32>,
}
impl<G: PreparedGrammarController> PreparedGrammarChoice<G> {
    /// Complete actual semantic source. It remains unauthenticated metadata.
    pub fn source(&self) -> PreparedGrammarSource<'_> {
        self.branch.controller().prepared_grammar_source()
    }
    /// Exact mask before prospective forcing.
    pub fn before_forcing(&self) -> Result<PackedTokenFilter<'_>, PackedTokenFilterError> {
        self.branch.mask()
    }
    /// Sole forced candidate at this exact position, if any.
    pub fn forced_token(&self) -> Option<u32> {
        self.forced
    }
    /// Paired pre-forcing domain for the shared capture worker.
    pub fn capture_domain(
        &self,
    ) -> Result<eredu_core::capture::CaptureTokenDomain<'_>, PackedTokenFilterError> {
        Ok(eredu_core::capture::CaptureTokenDomain {
            filter: eredu_core::capture::CaptureTokenFilter::Packed(self.before_forcing()?),
            tokenizer_validity: self.source().validity(),
        })
    }
    /// Uses the existing ordinary/original Boolean mask expansion worker.
    pub fn mask_plan<'a>(
        &'a self,
        shape: &'a [i32],
    ) -> Result<crate::generation::TokenMaskPlan<'a>, crate::generation::TokenMaskError> {
        let packed = self.before_forcing().map_err(|cause| match cause {
            PackedTokenFilterError::Geometry => crate::generation::TokenMaskError::Shape,
            PackedTokenFilterError::Filter(cause) => {
                crate::generation::TokenMaskError::Filter(cause)
            }
        })?;
        crate::generation::TokenMaskPlan::packed(packed, shape, self.forced)
    }
}
fn controls<C: eredu_core::SpeculativeTokenFilterController>() -> Option<usize> {
    let parts = [
        size_of::<TokenChoiceController<C>>(),
        size_of::<PreparedGrammarChoice<C::PreparedGrammar>>(),
        size_of::<PreparedGrammarChoiceError<C::PreparedGrammar>>(),
        size_of::<PreparedGrammarChoiceCause<C::PreparedGrammar>>(),
        size_of::<Option<PreparedGrammarBranch<C::PreparedGrammar>>>(),
        size_of::<
            Result<
                PreparedGrammarBranch<C::PreparedGrammar>,
                PreparedGrammarBranchError<C::PreparedGrammar>,
            >,
        >(),
        size_of::<
            Result<
                (PreparedGrammarBranch<C::PreparedGrammar>, bool),
                PreparedGrammarBranchError<C::PreparedGrammar>,
            >,
        >(),
        size_of::<Result<bool, PreparedGrammarChoiceCause<C::PreparedGrammar>>>(),
        size_of::<Result<(), PreparedGrammarChoiceError<C::PreparedGrammar>>>(),
        size_of::<Result<(), PreparedGrammarChoiceCause<C::PreparedGrammar>>>(),
        size_of::<
            Result<
                PreparedGrammarChoice<C::PreparedGrammar>,
                PreparedGrammarChoiceError<C::PreparedGrammar>,
            >,
        >(),
        size_of::<TokenChoiceError<PreparedGrammarChoiceError<C::PreparedGrammar>>>(),
        size_of::<(&TokenChoiceController<C>, &HostMetadataFunding)>(),
        size_of::<Result<bool, PreparedGrammarChoiceError<C::PreparedGrammar>>>(),
        size_of::<Option<&C::PreparedGrammar>>(),
        size_of::<Option<u32>>(),
        size_of::<TokenChoiceError<Infallible>>(),
        size_of::<(
            &TokenChoiceController<C>,
            &[u32],
            usize,
            &HostMetadataFunding,
        )>(),
        size_of::<(u32, usize, bool)>(),
        PreparedGrammarSource::control_bytes()?,
        HostMetadataFunding::reservation_control_bytes(),
        PackedTokenFilter::control_bytes(),
    ];
    parts
        .into_iter()
        .try_fold(size_of_val(&parts), usize::checked_add)
}
impl<C: eredu_core::SpeculativeTokenFilterController> TokenChoiceController<C> {
    pub(crate) fn grammar_source(&self) -> Option<&C::PreparedGrammar> {
        if self.inner.prepared_plain_source().is_some()
            || self.inner.prepared_forbidden_source().is_some()
        {
            return None;
        }
        self.inner.prepared_grammar()
    }
    /// Evaluates the same logical history before applying the pending forced ID.
    pub fn prepared_grammar_decision(
        &self,
        history: &[u32],
        capacity: usize,
        funding: &HostMetadataFunding,
    ) -> Result<
        PreparedGrammarChoice<C::PreparedGrammar>,
        PreparedGrammarChoiceError<C::PreparedGrammar>,
    > {
        let mut branch = None;
        let mut forced = None;
        let result = (|| -> Result<(), PreparedGrammarChoiceCause<C::PreparedGrammar>> {
            funding.reserve_metadata(controls::<C>().ok_or(HostMetadataFundingError::Overflow)?)?;
            let source = self
                .grammar_source()
                .ok_or(PreparedGrammarChoiceCause::Unknown)?;
            branch = Some(PreparedGrammarBranch::at(
                source, history, capacity, funding,
            )?);
            if let (Some(token), Some(position)) = (self.pending, self.pending_position) {
                if history.len() == position {
                    if !branch
                        .as_ref()
                        .expect("computed grammar branch")
                        .mask()?
                        .allows(token)
                    {
                        return Err(TokenChoiceError::<Infallible>::Forbidden(token).into());
                    }
                    forced = Some(token);
                } else if let Some(&actual) = history.get(position) {
                    if actual != token {
                        return Err(TokenChoiceError::<Infallible>::UnexpectedCommit {
                            expected: token,
                            actual,
                        }
                        .into());
                    }
                }
            }
            Ok(())
        })();
        match result {
            Ok(()) => Ok(PreparedGrammarChoice {
                branch: branch.take().expect("complete grammar choice"),
                forced,
            }),
            Err(cause) => Err(PreparedGrammarChoiceError {
                cause,
                branch,
                funding: funding.clone(),
            }),
        }
    }
    /// Checks forced history before the same provisional terminal query; no mask
    /// computation is introduced ahead of the ordinary terminal policy.
    pub fn prepared_grammar_prefix_complete(
        &self,
        history: &[u32],
        capacity: usize,
        funding: &HostMetadataFunding,
    ) -> Result<bool, PreparedGrammarChoiceError<C::PreparedGrammar>> {
        let result = (|| -> Result<bool, PreparedGrammarChoiceCause<C::PreparedGrammar>> {
            funding.reserve_metadata(controls::<C>().ok_or(HostMetadataFundingError::Overflow)?)?;
            if let (Some(expected), Some(position)) = (self.pending, self.pending_position) {
                if let Some(&actual) = history.get(position) {
                    if actual != expected {
                        return Err(TokenChoiceError::<Infallible>::UnexpectedCommit {
                            expected,
                            actual,
                        }
                        .into());
                    }
                }
            }
            let source = self
                .grammar_source()
                .ok_or(PreparedGrammarChoiceCause::Unknown)?;
            let (_, terminal) =
                PreparedGrammarBranch::fork_at(source, history, capacity, funding)?.terminal()?;
            Ok(terminal)
        })();
        result.map_err(|cause| PreparedGrammarChoiceError {
            cause,
            branch: None,
            funding: funding.clone(),
        })
    }
    pub(crate) fn validate_grammar_force(
        &self,
        token: u32,
        domain: crate::TokenDomain,
        funding: &HostMetadataFunding,
    ) -> Result<(), PreparedGrammarChoiceError<C::PreparedGrammar>> {
        let reservation = controls::<C>()
            .ok_or(HostMetadataFundingError::Overflow)
            .and_then(|bytes| funding.reserve_metadata(bytes));
        if let Err(cause) = reservation {
            return Err(PreparedGrammarChoiceError {
                cause: cause.into(),
                branch: None,
                funding: funding.clone(),
            });
        }
        let result = Self::validate_forced_choice(self.pending, domain, token, || {
            let source = self
                .grammar_source()
                .ok_or_else(|| PreparedGrammarChoiceError {
                    cause: PreparedGrammarChoiceCause::Unknown,
                    branch: None,
                    funding: funding.clone(),
                })?;
            let history = source.prepared_grammar_source().history();
            let decision = self.prepared_grammar_decision(history, history.len(), funding)?;
            let allowed = decision.before_forcing().map(|mask| mask.allows(token));
            match allowed {
                Ok(allowed) => Ok(allowed),
                Err(cause) => Err(PreparedGrammarChoiceError {
                    cause: cause.into(),
                    branch: Some(decision.branch),
                    funding: funding.clone(),
                }),
            }
        });
        let cause = match result {
            Ok(()) => return Ok(()),
            Err(TokenChoiceError::Constraint(cause)) => return Err(cause),
            Err(TokenChoiceError::InvalidToken(token)) => TokenChoiceError::InvalidToken(token),
            Err(TokenChoiceError::Forbidden(token)) => TokenChoiceError::Forbidden(token),
            Err(TokenChoiceError::AlreadyPending) => TokenChoiceError::AlreadyPending,
            Err(TokenChoiceError::UnexpectedCommit { expected, actual }) => {
                TokenChoiceError::UnexpectedCommit { expected, actual }
            }
        };
        Err(PreparedGrammarChoiceError {
            cause: PreparedGrammarChoiceCause::Choice(cause),
            branch: None,
            funding: funding.clone(),
        })
    }
    pub(crate) fn install_grammar_force(
        &mut self,
        token: u32,
        domain: crate::TokenDomain,
        position: usize,
    ) {
        self.domain = domain;
        self.pending = Some(token);
        self.pending_position = Some(position);
    }
    pub(crate) fn validate_grammar_commit(
        &self,
        token: u32,
    ) -> Result<(), TokenChoiceError<Infallible>> {
        self.check_commit_token(token)
    }
    pub(crate) fn finish_grammar_commit(&mut self) {
        self.last_forced = self.pending.take().is_some();
    }
}

#[derive(Debug, Clone)]
pub(crate) struct GrammarChoiceIdentity {
    source: eredu_core::speculative::PreparedGrammarIdentity,
    domain: crate::TokenDomain,
    pending: Option<u32>,
    last_forced: bool,
    pending_position: Option<usize>,
}
impl GrammarChoiceIdentity {
    pub(crate) fn same_source(&self, other: &Self) -> bool {
        self.source.same_source(&other.source) && self.domain == other.domain
            && self.pending == other.pending && self.last_forced == other.last_forced
            && self.pending_position == other.pending_position
    }
}
impl<C: eredu_core::SpeculativeTokenFilterController> TokenChoiceController<C> {
    pub(crate) fn retain_grammar_choice_identity(&self) -> Result<GrammarChoiceIdentity, eredu_core::speculative::PlainControllerError> {
        let source = self.grammar_source().ok_or(eredu_core::speculative::PlainControllerError::Unknown)?
            .prepared_grammar_source().retain_prepared_identity()?;
        Ok(GrammarChoiceIdentity { source, domain: self.domain, pending: self.pending,
            last_forced: self.last_forced, pending_position: self.pending_position })
    }
    pub(crate) fn matches_grammar_choice(&self, identity: &GrammarChoiceIdentity) -> bool {
        self.grammar_source().is_some_and(|source| source.prepared_grammar_source().matches_prepared_identity(&identity.source))
            && self.domain == identity.domain && self.pending == identity.pending
            && self.last_forced == identity.last_forced && self.pending_position == identity.pending_position
    }
}
