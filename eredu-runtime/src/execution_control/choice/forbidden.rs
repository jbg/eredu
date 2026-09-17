//! Forbidden-trigger decisions use the same prospective-choice transitions as
//! ordinary execution; only the finite underlying candidate worker differs.
use super::*;
use eredu_core::{
    HostPreparationAuthority, SpeculativeTokenFilterController,
    speculative::{
        ForbiddenControllerDecision, ForbiddenControllerError, ForbiddenControllerSource,
        PreparedForbiddenControllerIdentity,
    },
};
use std::mem::{size_of, size_of_val};

/// Borrowed forbidden decision with the actual prospective forced token. The
/// dynamic pre-forcing domain is the byte predicate, never tokenizer validity
/// alone. Native capture/sampling must materialize it under its own source role.
#[derive(Debug, Clone, Copy)]
pub struct PreparedForbiddenDecision<'a> {
    decision: ForbiddenControllerDecision<'a>,
    forced: Option<u32>,
}
impl<'a> PreparedForbiddenDecision<'a> {
    /// Exact source/prefix candidate predicate before prospective forcing.
    pub fn before_forcing(self) -> ForbiddenControllerDecision<'a> {
        self.decision
    }
    /// Exact semantic domain before forcing for the existing capture worker.
    pub fn capture_domain(self) -> eredu_core::capture::CaptureTokenDomain<'a> {
        eredu_core::capture::CaptureTokenDomain {
            filter: eredu_core::capture::CaptureTokenFilter::Forbidden(self.decision),
            tokenizer_validity: self.decision.source().validity(),
        }
    }
    /// Actual independently retained controller inputs and durable history.
    pub fn source(self) -> ForbiddenControllerSource<'a> {
        self.decision.source()
    }
    /// Sole allowed candidate at this absolute position, if a choice is pending.
    pub fn forced_token(self) -> Option<u32> {
        self.forced
    }
    /// Candidate predicate after applying the same one-token forced restriction.
    pub fn allows(self, token: u32) -> bool {
        self.forced.is_none_or(|forced| forced == token) && self.decision.allows(token)
    }
}
impl<C: SpeculativeTokenFilterController> TokenChoiceController<C> {
    /// Validates the provisional suffix first, then resolves forcing in the same
    /// order as the ordinary filter-at worker, without constructing a mask.
    pub fn prepared_forbidden_decision(
        &self,
        history: &[u32],
    ) -> Result<PreparedForbiddenDecision<'_>, TokenChoiceError<ForbiddenControllerError>> {
        let source = self
            .inner
            .prepared_forbidden_source()
            .ok_or(TokenChoiceError::Constraint(
                ForbiddenControllerError::Unknown,
            ))?;
        let decision = source
            .decision_at(history)
            .map_err(TokenChoiceError::Constraint)?;
        let mut forced = None;
        if let (Some(token), Some(position)) = (self.pending, self.pending_position) {
            if history.len() == position {
                if !decision.allows(token) {
                    return Err(TokenChoiceError::Forbidden(token));
                }
                forced = Some(token);
            } else if let Some(&actual) = history.get(position) {
                if actual != token {
                    return Err(TokenChoiceError::UnexpectedCommit {
                        expected: token,
                        actual,
                    });
                }
            }
        }
        Ok(PreparedForbiddenDecision { decision, forced })
    }
    pub(crate) fn prepared_forbidden_prefix_complete(
        &self,
        history: &[u32],
    ) -> Result<bool, TokenChoiceError<ForbiddenControllerError>> {
        if let (Some(expected), Some(position)) = (self.pending, self.pending_position) {
            if let Some(&actual) = history.get(position) {
                if actual != expected {
                    return Err(TokenChoiceError::UnexpectedCommit { expected, actual });
                }
            }
        }
        self.inner
            .prepared_forbidden_source()
            .ok_or(TokenChoiceError::Constraint(
                ForbiddenControllerError::Unknown,
            ))?
            .decision_at(history)
            .map_err(TokenChoiceError::Constraint)?;
        Ok(false)
    }
    pub(crate) fn prepared_forbidden_copy_bytes(&self, capacity: usize) -> Option<usize> {
        self.inner.prepared_forbidden_source()?;
        let parts = [
            self.inner.prepared_forbidden_copy_bytes(capacity)?,
            size_of::<Self>(),
            size_of::<PreparedForbiddenDecision<'_>>(),
            size_of::<ForbiddenChoiceIdentity>(),
            size_of::<Result<Self, ForbiddenControllerError>>(),
            size_of::<Result<(), TokenChoiceError<ForbiddenControllerError>>>(),
            size_of::<
                Result<PreparedForbiddenDecision<'_>, TokenChoiceError<ForbiddenControllerError>>,
            >(),
            size_of::<Result<ForbiddenChoiceIdentity, ForbiddenControllerError>>(),
            size_of::<Result<bool, TokenChoiceError<ForbiddenControllerError>>>(),
            size_of::<(&Self, u32, crate::TokenDomain, usize)>(),
            size_of::<Option<u32>>(),
            size_of::<Option<usize>>(),
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
    pub(crate) fn copy_prepared_forbidden(
        &self,
        capacity: usize,
        host: HostPreparationAuthority,
    ) -> Result<Self, ForbiddenControllerError> {
        let source = self
            .inner
            .prepared_forbidden_source()
            .ok_or(ForbiddenControllerError::Unknown)?;
        let inner = self.inner.copy_prepared_forbidden(capacity, host)?;
        let copied = inner
            .prepared_forbidden_source()
            .ok_or(ForbiddenControllerError::Source)?;
        if !source.matches_copy(copied, capacity) {
            return Err(ForbiddenControllerError::Source);
        }
        Ok(Self {
            inner,
            domain: self.domain,
            pending: self.pending,
            last_forced: self.last_forced,
            pending_position: self.pending_position,
        })
    }
    pub(crate) fn validate_forbidden_commit(
        &self,
        token: u32,
    ) -> Result<(), TokenChoiceError<ForbiddenControllerError>> {
        self.check_commit_token(token)?;
        self.inner
            .prepared_forbidden_source()
            .ok_or(TokenChoiceError::Constraint(
                ForbiddenControllerError::Unknown,
            ))?
            .validate_token(token)
            .map_err(TokenChoiceError::Constraint)
    }
    pub(crate) fn commit_prepared_forbidden(
        &mut self,
        token: u32,
    ) -> Result<(), TokenChoiceError<ForbiddenControllerError>> {
        self.commit_with(token, |inner, token| {
            inner.prepared_forbidden_mutation()?.commit(token)
        })
    }
    pub(crate) fn validate_forbidden_force(
        &self,
        token: u32,
        domain: TokenDomain,
    ) -> Result<(), TokenChoiceError<ForbiddenControllerError>> {
        Self::validate_forced_choice(self.pending, domain, token, || {
            let source = self
                .inner
                .prepared_forbidden_source()
                .ok_or(ForbiddenControllerError::Unknown)?;
            Ok(source.decision_at(source.history())?.allows(token))
        })
    }
    pub(crate) fn force_prepared_forbidden(
        &mut self,
        token: u32,
        domain: TokenDomain,
        position: usize,
    ) -> Result<(), TokenChoiceError<ForbiddenControllerError>> {
        self.validate_forbidden_force(token, domain)?;
        self.domain = domain;
        self.pending = Some(token);
        self.pending_position = Some(position);
        Ok(())
    }
    pub(crate) fn forbidden_history_len(&self) -> Option<usize> {
        Some(self.inner.prepared_forbidden_source()?.history().len())
    }
}
#[derive(Debug, Clone)]
pub(crate) struct ForbiddenChoiceIdentity {
    source: PreparedForbiddenControllerIdentity,
    domain: TokenDomain,
    pending: Option<u32>,
    last_forced: bool,
    pending_position: Option<usize>,
}
impl ForbiddenChoiceIdentity {
    pub(crate) fn same_source(&self, other: &Self) -> bool {
        self.source.same_source(&other.source)
            && self.domain == other.domain
            && self.pending == other.pending
            && self.last_forced == other.last_forced
            && self.pending_position == other.pending_position
    }
}
impl<C: SpeculativeTokenFilterController> TokenChoiceController<C> {
    pub(crate) fn retain_forbidden_choice_identity(
        &self,
    ) -> Result<ForbiddenChoiceIdentity, ForbiddenControllerError> {
        let source = self
            .inner
            .prepared_forbidden_source()
            .ok_or(ForbiddenControllerError::Unknown)?
            .retain_prepared_identity()?;
        Ok(ForbiddenChoiceIdentity {
            source,
            domain: self.domain,
            pending: self.pending,
            last_forced: self.last_forced,
            pending_position: self.pending_position,
        })
    }
    pub(crate) fn matches_forbidden_choice(&self, identity: &ForbiddenChoiceIdentity) -> bool {
        self.inner
            .prepared_forbidden_source()
            .is_some_and(|source| source.matches_prepared_identity(&identity.source))
            && self.domain == identity.domain
            && self.pending == identity.pending
            && self.last_forced == identity.last_forced
            && self.pending_position == identity.pending_position
    }
}
