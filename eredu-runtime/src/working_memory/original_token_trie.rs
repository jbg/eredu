//! Originally constructed tokenizer trie with exact source and account custody.
use super::{
    OriginalTokenizer, WorkingMemoryError, WorkingMemoryPool, original_declaration_source::Account,
};
use eredu_text::token_trie_storage::{
    PreparedTokenTrie, TokRxInfo, TokTrie, TokenTrieConstructionFailure, TokenTriePlan,
    TokenTrieSourceError,
};
use std::{
    alloc::Layout,
    mem::{size_of, size_of_val},
    sync::{Arc, atomic::AtomicUsize},
};

#[derive(Debug)]
struct Payload {
    trie: PreparedTokenTrie,
    tokenizer: OriginalTokenizer,
    bytes: u64,
    // Array/vector contents, tokenizer alias and Arc allocation retire first.
    account: Account,
}
/// Closed actual tokenizer-derived trie. No existing trie, raw array, caller
/// capacity or account is accepted. This source alone does not admit a grammar,
/// mutable recognizer, lexer, mask calculation or tokenization operation.
#[derive(Debug)]
pub struct OriginalTokenTrieSource(Option<Arc<Payload>>);
impl Clone for OriginalTokenTrieSource {
    fn clone(&self) -> Self {
        Self(Some(self.0.as_ref().expect("live trie source").clone()))
    }
}
impl Drop for OriginalTokenTrieSource {
    fn drop(&mut self) {
        if let Some(owner) = self.0.take() {
            drop(Arc::into_inner(owner));
        }
    }
}
impl OriginalTokenTrieSource {
    /// Authenticate the original trie and immutable compilation outputs.
    /// Mutable state funding is checked separately against its retained payer.
    pub fn validate_grammar_source(&self, source: eredu_core::speculative::PreparedGrammarSource<'_>, pool: &WorkingMemoryPool)
        -> Result<(), WorkingMemoryError> {
        let supplied = source.tokenizer().downcast_ref::<Self>().ok_or(WorkingMemoryError::IdentityMismatch)?;
        let compilation = source.compilation().downcast_ref::<super::OriginalControllerCompilation>()
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        if !self.same_source(supplied) { return Err(WorkingMemoryError::IdentityMismatch); }
        self.validate_grammar_inputs(source.validity(), source.recipe(), source.declaration(), compilation, pool)
    }
    pub(super) fn validate_grammar_inputs(
        &self, validity: &eredu_core::SharedTokenFilter, recipe: &eredu_core::SharedControllerBytes,
        declaration: &eredu_core::SharedControllerDeclaration,
        compilation: &super::OriginalControllerCompilation, pool: &WorkingMemoryPool,
    ) -> Result<(), WorkingMemoryError> {
        self.payload().account.validate(pool)?;
        self.payload().tokenizer.validate_pool(pool)?;
        pool.validate_shared_controller_source(eredu_core::SharedControllerSource::Filter(validity))?;
        compilation.validate_grammar_sources(recipe, declaration, self)
    }
    /// Exact read-only original grammar/source authentication frames.
    pub fn grammar_validation_control_bytes() -> Option<usize> {
        let parts = [Self::validation_control_bytes()?,
            WorkingMemoryPool::shared_controller_source_validation_control_bytes()?,
            super::OriginalControllerCompilation::validation_control_bytes()?,
            eredu_core::speculative::PreparedGrammarSource::control_bytes()?,
            size_of::<(&Self, eredu_core::speculative::PreparedGrammarSource<'_>, &WorkingMemoryPool)>(),
            size_of::<(&Self, &eredu_core::SharedTokenFilter, &eredu_core::SharedControllerBytes,
                &eredu_core::SharedControllerDeclaration, &super::OriginalControllerCompilation, &WorkingMemoryPool)>(),
            size_of::<Option<&Self>>(), size_of::<Option<&super::OriginalControllerCompilation>>(),
            size_of::<eredu_core::SharedControllerSource<'_>>(),
            size_of::<Result<(), WorkingMemoryError>>()];
        parts.into_iter().try_fold(size_of_val(&parts), usize::checked_add)
    }
    /// Encodes in the exact tokenizer/source pool retained by this trie. The
    /// existing original E producer admits each operation before construction;
    /// no tokenizer, source account or raw destination is substituted.
    pub fn encode_tokenizer_ids(
        &self,
        input: &str,
    ) -> Result<super::OriginalEncodedTokenIds, super::OriginalTokenizerEncodeError> {
        let tokenizer = &self.payload().tokenizer;
        tokenizer
            .pool()
            .encode_tokenizer_ids(tokenizer, input, false)
    }
    /// Exact original E source comparison, independent of equal token IDs.
    pub fn matches_encoded_source(&self, ids: &super::OriginalEncodedTokenIds) -> bool {
        ids.matches_source(&self.payload().tokenizer)
    }
    /// Accepts the exact original tokenizer or its directly constructed checked
    /// input-prefix derivative. Equal vocabularies or independently constructed
    /// sources grant no relationship, and encoding still uses the exact source.
    pub fn matches_semantic_tokenizer(&self, tokenizer: &OriginalTokenizer) -> bool {
        self.payload().tokenizer.matches_semantic_root(tokenizer)
    }
    /// Reads the deterministic tokenization fact from the retained actual source.
    pub fn tokenization_is_canonical(&self) -> bool {
        self.payload().tokenizer.tokenization_is_canonical()
    }
    fn payload(&self) -> &Payload {
        self.0.as_deref().expect("live trie source")
    }
    /// Borrows the actual immutable trie; no shared owner or mutation escapes.
    pub fn trie(&self) -> &TokTrie {
        self.payload().trie.trie()
    }
    /// Complete originally accepted trie allowance, retained through all aliases.
    pub fn original_bytes(&self) -> u64 {
        self.payload().bytes
    }
    /// Exact shared construction identity, independent of equal vocabulary bytes.
    pub fn same_source(&self, other: &Self) -> bool {
        Arc::ptr_eq(
            self.0.as_ref().expect("live source"),
            other.0.as_ref().expect("live source"),
        )
    }
    /// Compares the exact source pool/tokenizer and all selected metadata. No
    /// deep clone, reconstruction or funding operation occurs during validation.
    pub fn validate(
        &self,
        pool: &WorkingMemoryPool,
        tokenizer: &OriginalTokenizer,
        info: &TokRxInfo,
        eos: &[u32],
    ) -> Result<(), WorkingMemoryError> {
        self.payload().account.validate(pool)?;
        self.payload().tokenizer.validate_pool(pool)?;
        if !self.payload().tokenizer.same_source(tokenizer)
            || self.trie().info() != info
            || self.trie().eos_tokens() != eos
        {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        Ok(())
    }
    /// Named read-only source validation controls; no source capacity is granted.
    pub fn validation_control_bytes() -> Option<usize> {
        let parts = [
            Account::control_bytes()?,
            size_of::<Self>(),
            size_of::<&Payload>(),
            size_of::<(&Self, &OriginalTokenizer)>(),
            size_of::<[&OriginalTokenizer; 2]>(),
            size_of::<Option<&OriginalTokenizer>>(),
            size_of::<bool>(),
            size_of::<(
                &Self,
                &WorkingMemoryPool,
                &OriginalTokenizer,
                &TokRxInfo,
                &[u32],
            )>(),
            size_of::<Result<(), WorkingMemoryError>>(),
            size_of::<TokRxInfo>(),
            size_of::<std::slice::Iter<'_, u32>>(),
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
}
#[derive(Debug, thiserror::Error)]
enum Cause {
    #[error(transparent)]
    Memory(#[from] WorkingMemoryError),
    #[error(transparent)]
    Source(#[from] TokenTrieSourceError),
    #[error(transparent)]
    Construction(#[from] TokenTrieConstructionFailure),
}
/// Typed refusal or completed/partial trie construction. All actual destination
/// prefixes and the source alias are destroyed before the original account.
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
pub struct OriginalTokenTrieSourceError {
    #[source]
    cause: Cause,
    settlement: Option<WorkingMemoryError>,
    completed: Option<OriginalTokenTrieSource>,
    tokenizer: Option<OriginalTokenizer>,
    bytes: u64,
    account: Option<Account>,
}
impl OriginalTokenTrieSourceError {
    fn rejected(cause: impl Into<Cause>) -> Self {
        Self {
            cause: cause.into(),
            settlement: None,
            completed: None,
            tokenizer: None,
            bytes: 0,
            account: None,
        }
    }
    /// Actual retained trie allowance; zero means rejection preceded construction.
    pub fn retained_bytes(&self) -> u64 {
        self.bytes
    }
    /// Actual compiler prefix, when destination allocation or trie encoding failed.
    pub fn construction_failure(&self) -> Option<&TokenTrieConstructionFailure> {
        match &self.cause {
            Cause::Construction(cause) => Some(cause),
            _ => None,
        }
    }
    /// Exact admission or settlement refusal, preserving its typed cause.
    pub fn accounting_failure(&self) -> Option<&WorkingMemoryError> {
        self.settlement.as_ref().or_else(|| match &self.cause {
            Cause::Memory(cause) => Some(cause),
            _ => None,
        })
    }
}
fn required(plan: &TokenTriePlan<'_>) -> Result<u64, WorkingMemoryError> {
    let arc = Layout::new::<[AtomicUsize; 2]>()
        .extend(Layout::new::<Payload>())
        .map_err(|_| WorkingMemoryError::Overflow)?
        .0
        .pad_to_align()
        .size();
    let parts = [
        plan.requirements().required_bytes(),
        Account::control_bytes().ok_or(WorkingMemoryError::Overflow)?,
        OriginalTokenTrieSource::validation_control_bytes().ok_or(WorkingMemoryError::Overflow)?,
        arc,
        size_of::<Payload>(),
        size_of::<Option<Payload>>(),
        size_of::<OriginalTokenTrieSourceError>(),
        size_of::<Result<OriginalTokenTrieSource, OriginalTokenTrieSourceError>>(),
        size_of::<Result<PreparedTokenTrie, TokenTrieConstructionFailure>>(),
        size_of::<Result<TokenTriePlan<'_>, TokenTrieSourceError>>(),
        size_of::<Result<OriginalTokenTrieSource, Cause>>(),
        size_of::<Result<Account, WorkingMemoryError>>(),
        size_of::<(&WorkingMemoryPool, &OriginalTokenizer, &TokRxInfo, &[u32])>(),
        size_of::<Result<(), WorkingMemoryError>>(),
        size_of::<Option<OriginalTokenTrieSource>>(),
        size_of::<Arc<Payload>>(),
        size_of::<Option<OriginalTokenizer>>(),
        size_of::<Cause>(),
        size_of::<Result<u64, WorkingMemoryError>>(),
        size_of::<(&TokenTriePlan<'_>, u64)>(),
        size_of::<Result<(Layout, usize), std::alloc::LayoutError>>(),
    ];
    parts
        .into_iter()
        .try_fold(size_of_val(&parts), usize::checked_add)
        .and_then(|n| u64::try_from(n).ok())
        .ok_or(WorkingMemoryError::Overflow)
}
impl WorkingMemoryPool {
    /// Exact original tokenizer source geometry; no destination allocation or
    /// account admission occurs while deriving this quote.
    pub fn token_trie_source_required_bytes(
        tokenizer: &OriginalTokenizer,
        info: &TokRxInfo,
        eos: &[u32],
    ) -> Result<u64, OriginalTokenTrieSourceError> {
        let plan = tokenizer
            .token_trie_plan(info, eos)
            .map_err(OriginalTokenTrieSourceError::rejected)?;
        required(&plan).map_err(OriginalTokenTrieSourceError::rejected)
    }
    /// Compares and admits the entire real constructor before creating packed
    /// bytes, borrowed slice storage or trie nodes. Source facts cannot adopt an
    /// existing trie or promote unregistered storage into this source account.
    pub fn compile_token_trie_source(
        &self,
        tokenizer: &OriginalTokenizer,
        info: &TokRxInfo,
        eos: &[u32],
    ) -> Result<OriginalTokenTrieSource, OriginalTokenTrieSourceError> {
        tokenizer
            .validate_pool(self)
            .map_err(OriginalTokenTrieSourceError::rejected)?;
        let plan = tokenizer
            .token_trie_plan(info, eos)
            .map_err(OriginalTokenTrieSourceError::rejected)?;
        let bytes = required(&plan).map_err(OriginalTokenTrieSourceError::rejected)?;
        let account =
            Account::admit(self, bytes).map_err(OriginalTokenTrieSourceError::rejected)?;
        let result = plan.compile().map(|trie| {
            OriginalTokenTrieSource(Some(Arc::new(Payload {
                trie,
                tokenizer: tokenizer.clone(),
                bytes,
                account: account.clone(),
            })))
        });
        let settlement = account.finish();
        match result {
            Ok(source) => match settlement {
                Ok(()) => Ok(source),
                Err(cause) => Err(OriginalTokenTrieSourceError {
                    cause: cause.into(),
                    settlement: None,
                    completed: Some(source),
                    tokenizer: None,
                    bytes,
                    account: Some(account),
                }),
            },
            Err(cause) => Err(OriginalTokenTrieSourceError {
                cause: cause.into(),
                settlement: settlement.err(),
                completed: None,
                tokenizer: Some(tokenizer.clone()),
                bytes,
                account: Some(account),
            }),
        }
    }
}
impl OriginalTokenizer {
    /// Runs the same compiler in this tokenizer's retained original source domain.
    pub fn compile_token_trie_source(
        &self,
        info: &TokRxInfo,
        eos: &[u32],
    ) -> Result<OriginalTokenTrieSource, OriginalTokenTrieSourceError> {
        self.pool().compile_token_trie_source(self, info, eos)
    }
}
