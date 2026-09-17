//! Fixed plain decisions and provisional destinations over the ordinary choice state.
use super::*;
use eredu_core::speculative::{PlainControllerError, PlainControllerSource};
use eredu_core::{HostPreparationAuthority, SpeculativeTokenFilterController};
use std::mem::{size_of, size_of_val};

/// Borrowed exact plain decision. A native consumer must authenticate the
/// retained filter's source storage and construct the same optional forced mask.
#[derive(Debug, Clone, Copy)]
pub struct PreparedPlainDecision<'a> {
    source: PlainControllerSource<'a>,
    forced: Option<u32>,
}
impl<'a> PreparedPlainDecision<'a> {
    /// Actual prefix and tokenizer source, with its storage evidence.
    pub fn source(self) -> PlainControllerSource<'a> {
        self.source
    }
    /// Exact plain tokenizer domain before forcing, matching the ordinary
    /// decision's observation semantics. This borrows the retained source only.
    pub fn capture_domain(self) -> eredu_core::capture::CaptureTokenDomain<'a> {
        eredu_core::capture::CaptureTokenDomain {
            filter: self.source.validity().into(),
            tokenizer_validity: self.source.validity(),
        }
    }
    /// Sole allowed ID at this position, if a prospective choice applies.
    pub fn forced_token(self) -> Option<u32> {
        self.forced
    }
}
impl<C: SpeculativeTokenFilterController> TokenChoiceController<C> {
    /// Borrows the actual source and preserves the ordinary filter-at ordering.
    pub fn prepared_plain_decision(
        &self,
        history: &[u32],
    ) -> Result<PreparedPlainDecision<'_>, TokenChoiceError<PlainControllerError>> {
        let source = self
            .inner
            .prepared_plain_source()
            .ok_or(TokenChoiceError::Constraint(PlainControllerError::Unknown))?;
        source
            .validate_history(history)
            .map_err(TokenChoiceError::Constraint)?;
        let mut forced = None;
        if let (Some(token), Some(position)) = (self.pending, self.pending_position) {
            if history.len() == position {
                if !source.validity().allows(token) {
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
        Ok(PreparedPlainDecision { source, forced })
    }
    pub(crate) fn prepared_plain_prefix_complete(
        &self,
        history: &[u32],
    ) -> Result<bool, TokenChoiceError<PlainControllerError>> {
        if let (Some(expected), Some(position)) = (self.pending, self.pending_position) {
            if let Some(&actual) = history.get(position) {
                if actual != expected {
                    return Err(TokenChoiceError::UnexpectedCommit { expected, actual });
                }
            }
        }
        self.inner
            .prepared_plain_source()
            .ok_or(TokenChoiceError::Constraint(PlainControllerError::Unknown))?
            .validate_history(history)
            .map_err(TokenChoiceError::Constraint)?;
        Ok(false)
    }
    /// Actual controller copy and this wrapper's scalar/control destinations.
    pub(crate) fn prepared_plain_copy_bytes(&self, capacity: usize) -> Option<usize> {
        self.inner.prepared_plain_source()?;
        let parts = [
            self.inner.prepared_plain_copy_bytes(capacity)?,
            size_of::<Self>(),
            size_of::<PreparedPlainDecision<'_>>(),
            size_of::<Result<Self, PlainControllerError>>(),
            size_of::<Result<(), TokenChoiceError<PlainControllerError>>>(),
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
    pub(crate) fn copy_prepared_plain(
        &self,
        capacity: usize,
        host: HostPreparationAuthority,
    ) -> Result<Self, PlainControllerError> {
        let source = self
            .inner
            .prepared_plain_source()
            .ok_or(PlainControllerError::Unknown)?;
        let inner = self.inner.copy_prepared_plain(capacity, host)?;
        let copied = inner
            .prepared_plain_source()
            .ok_or(PlainControllerError::Source)?;
        if !source.matches_copy(copied, capacity) {
            return Err(PlainControllerError::Source);
        }
        Ok(Self {
            inner,
            domain: self.domain,
            pending: self.pending,
            last_forced: self.last_forced,
            pending_position: self.pending_position,
        })
    }
    pub(crate) fn validate_prepared_commit(
        &self,
        token: u32,
    ) -> Result<(), TokenChoiceError<PlainControllerError>> {
        self.check_commit_token(token)?;
        self.inner
            .prepared_plain_source()
            .ok_or(TokenChoiceError::Constraint(PlainControllerError::Unknown))?
            .validate_token(token)
            .map_err(TokenChoiceError::Constraint)
    }
    pub(crate) fn commit_prepared_plain(
        &mut self,
        token: u32,
    ) -> Result<(), TokenChoiceError<PlainControllerError>> {
        self.commit_with(token, |inner, token| {
            let source = inner
                .prepared_plain_source()
                .ok_or(PlainControllerError::Unknown)?;
            source.validate_token(token)?;
            let length = source.history().len();
            let history = inner
                .prepared_plain_history_mut()
                .ok_or(PlainControllerError::Unknown)?;
            if history.len() != length {
                return Err(PlainControllerError::Source);
            }
            history.try_push_prepared(token)
        })
    }
    pub(crate) fn validate_prepared_force(
        &self,
        token: u32,
        domain: TokenDomain,
    ) -> Result<(), TokenChoiceError<PlainControllerError>> {
        Self::validate_forced_choice(self.pending, domain, token, || {
            self.inner
                .prepared_plain_source()
                .map(|source| source.validity().allows(token))
                .ok_or(PlainControllerError::Unknown)
        })
    }
    pub(crate) fn force_prepared_plain(
        &mut self,
        token: u32,
        domain: TokenDomain,
        position: usize,
    ) -> Result<(), TokenChoiceError<PlainControllerError>> {
        self.validate_prepared_force(token, domain)?;
        self.domain = domain;
        self.pending = Some(token);
        self.pending_position = Some(position);
        Ok(())
    }
    pub(crate) fn plain_history_len(&self) -> Option<usize> {
        Some(self.inner.prepared_plain_source()?.history().len())
    }
}

#[derive(Debug, Clone)]
pub(crate) struct PlainChoiceIdentity {
    source: eredu_core::speculative::PreparedPlainControllerIdentity,
    domain: crate::TokenDomain,
    pending: Option<u32>,
    last_forced: bool,
    pending_position: Option<usize>,
}
impl PlainChoiceIdentity {
    pub(crate) fn same_source(&self, other: &Self) -> bool {
        self.source.same_source(&other.source)
            && self.domain == other.domain
            && self.pending == other.pending
            && self.last_forced == other.last_forced
            && self.pending_position == other.pending_position
    }
}
impl<C: SpeculativeTokenFilterController> TokenChoiceController<C> {
    pub(crate) fn retain_plain_choice_identity(
        &self,
    ) -> Result<PlainChoiceIdentity, PlainControllerError> {
        let source = self
            .inner
            .prepared_plain_source()
            .ok_or(PlainControllerError::Unknown)?
            .retain_prepared_identity()?;
        Ok(PlainChoiceIdentity {
            source,
            domain: self.domain,
            pending: self.pending,
            last_forced: self.last_forced,
            pending_position: self.pending_position,
        })
    }
    pub(crate) fn matches_plain_choice(&self, identity: &PlainChoiceIdentity) -> bool {
        self.inner
            .prepared_plain_source()
            .is_some_and(|source| source.matches_prepared_identity(&identity.source))
            && self.domain == identity.domain
            && self.pending == identity.pending
            && self.last_forced == identity.last_forced
            && self.pending_position == identity.pending_position
    }
}
