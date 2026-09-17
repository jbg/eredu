//! Closed semantic dispatch over the existing token-choice worker.
use super::*;
use eredu_core::{
    HostPreparationAuthority, SpeculativeTokenFilterController, TextControllerStorage,
    speculative::{
        ForbiddenControllerError, ForbiddenControllerSource, PlainControllerError,
        PlainControllerSource,
    },
};

/// Distinct fixed source errors preserve their actual cause and failed-copy custody.
#[derive(Debug, thiserror::Error)]
pub enum PreparedControllerCause {
    /// The plain controller refused its prefix or destination.
    #[error(transparent)]
    Plain(#[from] PlainControllerError),
    /// The forbidden controller refused a prefix, candidate or destination.
    #[error(transparent)]
    Forbidden(#[from] ForbiddenControllerError),
}
fn map_error<E: std::error::Error + 'static>(
    error: TokenChoiceError<E>,
    map: impl FnOnce(E) -> PreparedControllerCause,
) -> TokenChoiceError<PreparedControllerCause> {
    match error {
        TokenChoiceError::Constraint(cause) => TokenChoiceError::Constraint(map(cause)),
        TokenChoiceError::InvalidToken(token) => TokenChoiceError::InvalidToken(token),
        TokenChoiceError::Forbidden(token) => TokenChoiceError::Forbidden(token),
        TokenChoiceError::AlreadyPending => TokenChoiceError::AlreadyPending,
        TokenChoiceError::UnexpectedCommit { expected, actual } => {
            TokenChoiceError::UnexpectedCommit { expected, actual }
        }
    }
}
impl From<TokenChoiceError<PlainControllerError>> for TokenChoiceError<PreparedControllerCause> {
    fn from(error: TokenChoiceError<PlainControllerError>) -> Self {
        map_error(error, PreparedControllerCause::Plain)
    }
}
impl From<TokenChoiceError<ForbiddenControllerError>>
    for TokenChoiceError<PreparedControllerCause>
{
    fn from(error: TokenChoiceError<ForbiddenControllerError>) -> Self {
        map_error(error, PreparedControllerCause::Forbidden)
    }
}
/// The exact finite controller source selected by the semantic owner. Native
/// code must authenticate each variant's actual storage independently.
#[derive(Debug, Clone, Copy)]
pub enum PreparedControllerSource<'a> {
    /// Tokenizer-domain-only semantics.
    Plain(PlainControllerSource<'a>),
    /// Tokenizer validity plus a source-derived forbidden byte sequence.
    Forbidden(ForbiddenControllerSource<'a>),
}
impl<'a> PreparedControllerSource<'a> {
    /// Actual durable canonical history.
    pub fn history(self) -> &'a [u32] {
        match self {
            Self::Plain(s) => s.history(),
            Self::Forbidden(s) => s.history(),
        }
    }
    /// The independently retained tokenizer-validity source.
    pub fn validity(self) -> &'a eredu_core::SharedTokenFilter {
        match self {
            Self::Plain(s) => s.validity(),
            Self::Forbidden(s) => s.validity(),
        }
    }
    /// Legacy filter-only storage is complete only for the plain variant.
    /// Forbidden inputs need their separate original-source authentication.
    pub fn storage(self) -> TextControllerStorage<'a> {
        match self {
            Self::Plain(s) => s.storage(),
            Self::Forbidden(_) => TextControllerStorage::Unknown,
        }
    }
}
/// Paired actual semantic predicate and optional prospective forcing.
#[derive(Debug, Clone, Copy)]
pub enum PreparedControllerDecision<'a> {
    /// Concrete tokenizer-domain decision.
    Plain(PreparedPlainDecision<'a>),
    /// Concrete source-derived forbidden decision.
    Forbidden(PreparedForbiddenDecision<'a>),
}
impl<'a> PreparedControllerDecision<'a> {
    /// Complete actual source; matching content never substitutes owner identity.
    pub fn source(self) -> PreparedControllerSource<'a> {
        match self {
            Self::Plain(d) => PreparedControllerSource::Plain(d.source()),
            Self::Forbidden(d) => PreparedControllerSource::Forbidden(d.source()),
        }
    }
    /// Sole allowed forced candidate at this absolute position, when present.
    pub fn forced_token(self) -> Option<u32> {
        match self {
            Self::Plain(d) => d.forced_token(),
            Self::Forbidden(d) => d.forced_token(),
        }
    }
    /// Exact semantic domain before forcing for the shared observer.
    pub fn capture_domain(self) -> eredu_core::capture::CaptureTokenDomain<'a> {
        match self {
            Self::Plain(d) => d.capture_domain(),
            Self::Forbidden(d) => d.capture_domain(),
        }
    }
    /// Same mask expansion producer, preserving the selected semantic predicate.
    pub fn mask_plan(
        self,
        shape: &'a [i32],
    ) -> Result<crate::generation::TokenMaskPlan<'a>, crate::generation::TokenMaskError> {
        match self {
            Self::Plain(d) => crate::generation::TokenMaskPlan::new(
                d.source().validity(),
                shape,
                d.forced_token(),
            ),
            Self::Forbidden(d) => crate::generation::TokenMaskPlan::forbidden(
                d.before_forcing(),
                shape,
                d.forced_token(),
            ),
        }
    }
}
#[derive(Debug, Clone)]
pub(crate) enum ControllerChoiceIdentity {
    Plain(PlainChoiceIdentity),
    Forbidden(ForbiddenChoiceIdentity),
    Grammar(super::GrammarChoiceIdentity),
}
impl ControllerChoiceIdentity {
    pub(crate) fn same_source(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Plain(a), Self::Plain(b)) => a.same_source(b),
            (Self::Forbidden(a), Self::Forbidden(b)) => a.same_source(b),
            (Self::Grammar(a), Self::Grammar(b)) => a.same_source(b),
            _ => false,
        }
    }
}
impl<C: SpeculativeTokenFilterController> TokenChoiceController<C> {
    pub(crate) fn prepared_source(&self) -> Option<PreparedControllerSource<'_>> {
        match (
            self.inner.prepared_plain_source(),
            self.inner.prepared_forbidden_source(),
        ) {
            (Some(source), None) => Some(PreparedControllerSource::Plain(source)),
            (None, Some(source)) => Some(PreparedControllerSource::Forbidden(source)),
            _ => None,
        }
    }
    pub(crate) fn fixed_history_len(&self) -> Option<usize> {
        Some(self.prepared_source()?.history().len())
    }
    pub(crate) fn prepared_copy_bytes(&self, capacity: usize) -> Option<usize> {
        let selected = match self.prepared_source()? {
            PreparedControllerSource::Plain(_) => self.prepared_plain_copy_bytes(capacity)?,
            PreparedControllerSource::Forbidden(_) => {
                self.prepared_forbidden_copy_bytes(capacity)?
            }
        };
        let parts = [
            selected,
            std::mem::size_of::<PreparedControllerSource<'_>>(),
            std::mem::size_of::<PreparedControllerDecision<'_>>(),
            std::mem::size_of::<ControllerChoiceIdentity>(),
            std::mem::size_of::<PreparedControllerCause>(),
            std::mem::size_of::<TokenChoiceError<PreparedControllerCause>>(),
            std::mem::size_of::<Result<Self, PreparedControllerCause>>(),
            std::mem::size_of::<Result<ControllerChoiceIdentity, PreparedControllerCause>>(),
        ];
        parts
            .into_iter()
            .try_fold(std::mem::size_of_val(&parts), usize::checked_add)
    }
    pub(crate) fn copy_prepared_fixed(
        &self,
        capacity: usize,
        host: HostPreparationAuthority,
    ) -> Result<Self, PreparedControllerCause> {
        match self
            .prepared_source()
            .ok_or(PlainControllerError::Unknown)?
        {
            PreparedControllerSource::Plain(_) => {
                self.copy_prepared_plain(capacity, host).map_err(Into::into)
            }
            PreparedControllerSource::Forbidden(_) => self
                .copy_prepared_forbidden(capacity, host)
                .map_err(Into::into),
        }
    }
    pub(crate) fn prepared_fixed_decision(
        &self,
        history: &[u32],
    ) -> Result<PreparedControllerDecision<'_>, TokenChoiceError<PreparedControllerCause>> {
        match self
            .prepared_source()
            .ok_or(TokenChoiceError::Constraint(PlainControllerError::Unknown))?
        {
            PreparedControllerSource::Plain(_) => self
                .prepared_plain_decision(history)
                .map(PreparedControllerDecision::Plain)
                .map_err(Into::into),
            PreparedControllerSource::Forbidden(_) => self
                .prepared_forbidden_decision(history)
                .map(PreparedControllerDecision::Forbidden)
                .map_err(Into::into),
        }
    }
    pub(crate) fn prepared_fixed_prefix_complete(
        &self,
        history: &[u32],
    ) -> Result<bool, TokenChoiceError<PreparedControllerCause>> {
        match self
            .prepared_source()
            .ok_or(TokenChoiceError::Constraint(PlainControllerError::Unknown))?
        {
            PreparedControllerSource::Plain(_) => self
                .prepared_plain_prefix_complete(history)
                .map_err(Into::into),
            PreparedControllerSource::Forbidden(_) => self
                .prepared_forbidden_prefix_complete(history)
                .map_err(Into::into),
        }
    }
    pub(crate) fn validate_fixed_commit(
        &self,
        token: u32,
    ) -> Result<(), TokenChoiceError<PreparedControllerCause>> {
        match self
            .prepared_source()
            .ok_or(TokenChoiceError::Constraint(PlainControllerError::Unknown))?
        {
            PreparedControllerSource::Plain(_) => {
                self.validate_prepared_commit(token).map_err(Into::into)
            }
            PreparedControllerSource::Forbidden(_) => {
                self.validate_forbidden_commit(token).map_err(Into::into)
            }
        }
    }
    pub(crate) fn commit_prepared_fixed(
        &mut self,
        token: u32,
    ) -> Result<(), TokenChoiceError<PreparedControllerCause>> {
        match self
            .prepared_source()
            .ok_or(TokenChoiceError::Constraint(PlainControllerError::Unknown))?
        {
            PreparedControllerSource::Plain(_) => {
                self.commit_prepared_plain(token).map_err(Into::into)
            }
            PreparedControllerSource::Forbidden(_) => {
                self.commit_prepared_forbidden(token).map_err(Into::into)
            }
        }
    }
    pub(crate) fn validate_fixed_force(
        &self,
        token: u32,
        domain: TokenDomain,
    ) -> Result<(), TokenChoiceError<PreparedControllerCause>> {
        match self
            .prepared_source()
            .ok_or(TokenChoiceError::Constraint(PlainControllerError::Unknown))?
        {
            PreparedControllerSource::Plain(_) => self
                .validate_prepared_force(token, domain)
                .map_err(Into::into),
            PreparedControllerSource::Forbidden(_) => self
                .validate_forbidden_force(token, domain)
                .map_err(Into::into),
        }
    }
    pub(crate) fn force_prepared_fixed(
        &mut self,
        token: u32,
        domain: TokenDomain,
        position: usize,
    ) -> Result<(), TokenChoiceError<PreparedControllerCause>> {
        match self
            .prepared_source()
            .ok_or(TokenChoiceError::Constraint(PlainControllerError::Unknown))?
        {
            PreparedControllerSource::Plain(_) => self
                .force_prepared_plain(token, domain, position)
                .map_err(Into::into),
            PreparedControllerSource::Forbidden(_) => self
                .force_prepared_forbidden(token, domain, position)
                .map_err(Into::into),
        }
    }
    pub(crate) fn retain_fixed_choice_identity(
        &self,
    ) -> Result<ControllerChoiceIdentity, PreparedControllerCause> {
        match self
            .prepared_source()
            .ok_or(PlainControllerError::Unknown)?
        {
            PreparedControllerSource::Plain(_) => self
                .retain_plain_choice_identity()
                .map(ControllerChoiceIdentity::Plain)
                .map_err(Into::into),
            PreparedControllerSource::Forbidden(_) => self
                .retain_forbidden_choice_identity()
                .map(ControllerChoiceIdentity::Forbidden)
                .map_err(Into::into),
        }
    }
    pub(crate) fn matches_fixed_choice(&self, identity: &ControllerChoiceIdentity) -> bool {
        match (self.prepared_source(), identity) {
            (Some(PreparedControllerSource::Plain(_)), ControllerChoiceIdentity::Plain(id)) => {
                self.matches_plain_choice(id)
            }
            (
                Some(PreparedControllerSource::Forbidden(_)),
                ControllerChoiceIdentity::Forbidden(id),
            ) => self.matches_forbidden_choice(id),
            _ => false,
        }
    }
}
