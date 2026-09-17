//! Actual mutable grammar controllers remain distinct from fixed token filters.
use super::PlainControllerHistory;
use crate::{
    HostMetadataFunding, OriginalTokenDomainWitness, PackedTokenFilter, PackedTokenFilterError,
    SharedControllerBytes, SharedControllerDeclaration, SharedTokenFilter,
};
use std::{
    error::Error,
    fmt::Debug,
    mem::{size_of, size_of_val},
};

/// Borrowed complete source identity of a paid mutable grammar controller.
/// These facts grant no construction, sampling, registration or native authority.
#[derive(Debug, Clone, Copy)]
pub struct PreparedGrammarSource<'a> {
    history: &'a PlainControllerHistory,
    validity: &'a SharedTokenFilter,
    recipe: &'a SharedControllerBytes,
    declaration: &'a SharedControllerDeclaration,
    tokenizer: OriginalTokenDomainWitness<'a>,
    funding: &'a HostMetadataFunding,
}
impl<'a> PreparedGrammarSource<'a> {
    /// Describes the actual owners retained by the controller. Consumers must
    /// separately authenticate the exact original tokenizer and declaration.
    pub fn new(
        history: &'a PlainControllerHistory,
        validity: &'a SharedTokenFilter,
        recipe: &'a SharedControllerBytes,
        declaration: &'a SharedControllerDeclaration,
        tokenizer: OriginalTokenDomainWitness<'a>,
        funding: &'a HostMetadataFunding,
    ) -> Self {
        Self {
            history,
            validity,
            recipe,
            declaration,
            tokenizer,
            funding,
        }
    }
    /// Complete canonical committed history, including terminal aliases.
    pub fn history(self) -> &'a [u32] {
        self.history
    }
    /// Actual paid history destination capacity.
    pub fn capacity(self) -> usize {
        self.history.capacity()
    }
    /// Exact retained tokenizer-validity owner.
    pub fn validity(self) -> &'a SharedTokenFilter {
        self.validity
    }
    /// Historical registered recipe; byte equality is insufficient.
    pub fn recipe(self) -> &'a SharedControllerBytes {
        self.recipe
    }
    /// Historical immutable declaration owner; no parser execution authority.
    pub fn declaration(self) -> &'a SharedControllerDeclaration {
        self.declaration
    }
    /// Borrowed actual original tokenizer/trie source, still unauthenticated.
    pub fn tokenizer(self) -> OriginalTokenDomainWitness<'a> {
        self.tokenizer
    }
    /// Actual account paying mutable state and each subsequent operation.
    pub fn funding(self) -> &'a HostMetadataFunding {
        self.funding
    }
    /// Exact immutable owners shared by a successor, independent of its history.
    pub fn same_inputs(self, other: Self) -> bool {
        self.validity.same_storage(other.validity)
            && self.recipe.same_storage(other.recipe)
            && self.declaration.same_storage(other.declaration)
            && self.tokenizer.same_borrowed_source(other.tokenizer)
    }
    /// Checks a newly copied independent history and all immutable sources.
    /// A matching copy still needs the original consumer's source qualification.
    pub fn matches_copy(
        self,
        copied: Self,
        capacity: usize,
        funding: &HostMetadataFunding,
    ) -> bool {
        self.same_inputs(copied)
            && self.history() == copied.history()
            && copied.history.is_unique_prepared()
            && copied.capacity() == capacity
            && copied.funding.same_account(funding)
    }
    /// Fixed source-view, identity-comparison and borrowing call transports.
    pub fn control_bytes() -> Option<usize> {
        let parts = [
            size_of::<Self>(),
            size_of::<(Self, Self, usize, &HostMetadataFunding)>(),
            size_of::<bool>(),
            size_of::<&[u32]>(),
            size_of::<Option<usize>>(),
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
}

/// Paid semantic state over the ordinary grammar workers. Unlike fixed filters,
/// a decision can advance a lexer/parser and must own its entire mutable prefix.
/// All consuming failures must retain their actual changed state and funding;
/// copy failures must retain any constructed destination and immutable sources.
/// Implementing this contract never authenticates an original execution source.
pub trait PreparedGrammarController: Debug + Send + Sync + Sized + 'static {
    /// Owning failure from a real semantic operation.
    type Error: Error + Send + Sync + 'static;
    /// Owning failure from the independent state/history copy worker.
    type CopyError: Error + Send + Sync + 'static;
    /// Borrows exact source identities and complete canonical history.
    fn prepared_grammar_source(&self) -> PreparedGrammarSource<'_>;
    /// Complete prospective independent-copy payment for the requested history
    /// capacity, from actual retained sources. Inspection must not allocate or
    /// run a grammar/funding operation. Unknown/failed sources return None.
    fn prepared_grammar_copy_bytes(&self, capacity: usize) -> Option<usize>;
    /// Copies all mutable state and canonical history under the supplied account.
    /// Immutable sources remain shared; no old execution account is substituted.
    fn copy_prepared_grammar(
        &self,
        capacity: usize,
        funding: &HostMetadataFunding,
    ) -> Result<Self, Self::CopyError>;
    /// Validates and commits exactly one canonical token with the ordinary policy.
    fn commit_prepared_grammar(self, token: u32) -> Result<Self, Self::Error>;
    /// Computes the actual semantic mask, retaining any changed state on failure.
    fn compute_prepared_grammar_mask(self) -> Result<Self, Self::Error>;
    /// Uses the ordinary terminal policy, including committed secondary EOS aliases.
    fn prepared_grammar_terminal(self) -> Result<(Self, bool), Self::Error>;
    /// Borrows the actual completed packed mask and tokenizer validity together.
    fn prepared_grammar_mask(&self) -> Result<PackedTokenFilter<'_>, PackedTokenFilterError>;
}

/// Explicit absence of a prepared grammar producer on a fixed controller.
/// It is uninhabited: no fixed source can impersonate a mutable grammar owner.
#[derive(Debug)]
pub enum NoPreparedGrammar {}
impl PreparedGrammarController for NoPreparedGrammar {
    type Error = std::convert::Infallible;
    type CopyError = std::convert::Infallible;
    fn prepared_grammar_source(&self) -> PreparedGrammarSource<'_> { match *self {} }
    fn prepared_grammar_copy_bytes(&self, _: usize) -> Option<usize> { match *self {} }
    fn copy_prepared_grammar(&self, _: usize, _: &HostMetadataFunding) -> Result<Self, Self::CopyError> { match *self {} }
    fn commit_prepared_grammar(self, _: u32) -> Result<Self, Self::Error> { match self {} }
    fn compute_prepared_grammar_mask(self) -> Result<Self, Self::Error> { match self {} }
    fn prepared_grammar_terminal(self) -> Result<(Self, bool), Self::Error> { match self {} }
    fn prepared_grammar_mask(&self) -> Result<PackedTokenFilter<'_>, PackedTokenFilterError> { match *self {} }
}
/// Fixed refusal while publishing an independently prepared grammar successor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum PreparedGrammarInstallCause {
    /// This controller has no prepared grammar publication producer.
    #[error("controller has no prepared grammar publication producer")]
    Unknown,
    /// The successor names different immutable inputs or a different account.
    #[error("prepared grammar successor source or account differs")]
    Source,
    /// Publication storage was refused before allocating its shared owner.
    #[error(transparent)]
    Funding(#[from] crate::HostMetadataFundingError),
}
/// The actual uninstalled successor retires before publication funding.
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
pub struct PreparedGrammarInstallError<G: PreparedGrammarController> {
    #[source] cause: PreparedGrammarInstallCause,
    grammar: G,
    funding: HostMetadataFunding,
}
impl<G: PreparedGrammarController> PreparedGrammarInstallError<G> {
    /// Fixed controls for forwarding an owning successor through one wrapper.
    /// This prices no grammar copy, shared publication or execution authority.
    pub fn wrapper_control_bytes<C>() -> Option<usize> {
        let parts = [std::mem::size_of::<C>(), std::mem::size_of::<G>(),
            std::mem::size_of::<Result<C, Self>>(),
            std::mem::size_of::<(&C, &HostMetadataFunding)>(),
            HostMetadataFunding::reservation_control_bytes()];
        parts.into_iter().try_fold(std::mem::size_of_val(&parts), usize::checked_add)
    }

    /// Retains the real successor without formatting or allocating an error.
    pub fn new(cause: PreparedGrammarInstallCause, grammar: G, funding: &HostMetadataFunding) -> Self {
        Self { cause, grammar, funding: funding.clone() }
    }
    /// Exact publication refusal.
    pub fn cause(&self) -> PreparedGrammarInstallCause { self.cause }
}

/// Retained identity of an already prepared canonical history and its immutable
/// inputs. This holds no parser or tokenizer execution authority. Native users
/// revalidate the actual trie loan before interpreting a processed value.
#[derive(Debug, Clone)]
pub struct PreparedGrammarIdentity {
    history: PlainControllerHistory,
    validity: SharedTokenFilter,
    recipe: SharedControllerBytes,
    declaration: SharedControllerDeclaration,
    funding: HostMetadataFunding,
}
impl PreparedGrammarIdentity {
    /// Exact source allocation identity, never a history-content comparison.
    pub fn same_source(&self, other: &Self) -> bool {
        self.history.same_prepared_source(&other.history)
            && self.validity.same_storage(&other.validity)
            && self.recipe.same_storage(&other.recipe)
            && self.declaration.same_storage(&other.declaration)
            && self.funding.same_account(&other.funding)
    }
    /// Fixed clone, retained-source and identity-query transports.
    pub fn metadata_bytes() -> usize {
        size_of::<Self>() + size_of::<Option<Self>>()
            + size_of::<Result<Self, super::PlainControllerError>>()
            + size_of::<PlainControllerHistory>() + size_of::<SharedTokenFilter>()
            + size_of::<SharedControllerBytes>() + size_of::<SharedControllerDeclaration>()
            + size_of::<HostMetadataFunding>()
    }
}
impl PreparedGrammarSource<'_> {
    /// Loans only already prepared immutable aliases. Ordinary history cannot
    /// acquire a numerical source witness through this operation.
    pub fn retain_prepared_identity(self) -> Result<PreparedGrammarIdentity, super::PlainControllerError> {
        if !self.history.is_prepared() { return Err(super::PlainControllerError::Source); }
        Ok(PreparedGrammarIdentity { history: self.history.clone(), validity: self.validity.clone(),
            recipe: self.recipe.clone(), declaration: self.declaration.clone(), funding: self.funding.clone() })
    }
    /// Checks this actual source against a successfully bound value. Every fresh
    /// mutable copy has an independent history allocation even in one account.
    pub fn matches_prepared_identity(self, identity: &PreparedGrammarIdentity) -> bool {
        self.history.same_prepared_source(&identity.history)
            && self.validity.same_storage(&identity.validity)
            && self.recipe.same_storage(&identity.recipe)
            && self.declaration.same_storage(&identity.declaration)
            && self.funding.same_account(&identity.funding)
    }
}
