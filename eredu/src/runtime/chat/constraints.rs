//! Private constrained-decoding implementation for native tool plans.

mod declaration;
#[cfg(test)]
pub(crate) mod fixtures;
pub(crate) mod forbidden;
#[cfg(test)]
mod frozen_tests;
mod grammar_source;
#[cfg(test)]
mod released_funding_probe;
mod source;
mod stock_parser;
pub(crate) use grammar_source::OriginalPreparedGrammarController;
mod grammar_policy;
mod original;
mod preparation_error;
pub(crate) mod prepared;
pub(crate) use original::OriginalControllerSourceError;
pub(crate) use preparation_error::PreparationFailure;
pub(crate) mod recipe;
mod selection;
mod trigger;
mod vocabulary;
use recipe::ConstraintRecipe;
use trigger::TriggerPrefix;

#[cfg(test)]
use std::sync::atomic::{AtomicUsize, Ordering};
use std::{num::NonZeroUsize, sync::Arc};

use eredu_core::{
    HostPreparationAuthority, SharedTokenFilter, SpeculativeTokenFilterController,
    TextControllerStorage, TokenFilter, TokenFilterController,
};
use eredu_text::tokenizer::Tokenizer as ChatTokenizer;
#[cfg(test)]
use llguidance::Matcher;
use llguidance::{ParserFactory, toktrie::TokEnv};
use serde_json::{Value, json};

use super::tool_schema::{ToolDeclarations, ToolDefinition};

use crate::{
    api::ConstraintError,
    runtime::chat::dialect::{DeclarativeCallId, DialectParameters, FormatDialect},
    runtime::chat::{
        GenerationConstraint, GenerationRuntimePlan, GenerationRuntimePlanParts,
        ParallelToolCallPolicy, ToolChoice,
    },
};

/// Canonical backend-independent grammar and activation state.
pub(crate) struct ConstraintController {
    runtime: ConstraintRuntime,
    committed_tokens: eredu_core::speculative::PlainControllerHistory,
    validity: SharedTokenFilter,
    // Retires only after all controller payload, including copied history.
    authority: HostPreparationAuthority,
    preparation: Option<eredu_runtime::working_memory::PreparedControllerBinding>,
}

#[derive(Clone)]
enum ConstraintRuntime {
    Text,
    PreparedForbidden {
        inputs: eredu_core::speculative::ForbiddenControllerInputs,
        pending: TriggerPrefix,
        original: eredu_runtime::working_memory::OriginalForbiddenSource,
    },
    PreparedGrammar(
        eredu_core::SharedStorageOwner<grammar_source::OriginalPreparedGrammarController>,
    ),
}

impl Clone for ConstraintController {
    fn clone(&self) -> Self {
        Self {
            runtime: self.runtime.clone(),
            committed_tokens: self.committed_tokens.clone(),
            validity: self.validity.clone(),
            authority: self.authority.clone(),
            preparation: self.preparation.clone(),
        }
    }
}

impl ConstraintController {
    /// Grammar-free generation still retains the exact tokenizer-valid domain.
    pub(crate) fn text(validity: SharedTokenFilter) -> Self {
        Self::text_prepared(validity, HostPreparationAuthority::unmanaged())
    }

    /// Fixed empty controller destination. Shared validity keeps its original
    /// source account; the empty history allocates nothing. Native provisional
    /// copies use the existing paid history worker before any committed push.
    pub(crate) fn text_metadata_bytes() -> Option<usize> {
        let parts = [
            std::mem::size_of::<Self>(),
            std::mem::size_of::<ConstraintRuntime>(),
            std::mem::size_of::<eredu_core::speculative::PlainControllerHistory>(),
            std::mem::size_of::<SharedTokenFilter>(),
            std::mem::size_of::<HostPreparationAuthority>(),
        ];
        parts
            .into_iter()
            .try_fold(std::mem::size_of_val(&parts), usize::checked_add)
    }
    pub(crate) fn text_prepared(
        validity: SharedTokenFilter,
        authority: HostPreparationAuthority,
    ) -> Self {
        Self {
            runtime: ConstraintRuntime::Text,
            committed_tokens: eredu_core::speculative::PlainControllerHistory::default(),
            validity,
            authority,
            preparation: None,
        }
    }

    fn validate_token(&self, token: u32) -> Result<(), ConstraintError> {
        if self.validity.allows(token) {
            Ok(())
        } else {
            Err(ConstraintError::new(format!(
                "token {token} has no consistent tokenizer mapping"
            )))
        }
    }

    fn restrict(&self, filter: TokenFilter) -> Result<TokenFilter, ConstraintError> {
        self.validity
            .intersection(&filter)
            .map_err(|error| constraint_error(error.to_string()))
    }

    pub(crate) fn continuation_storage_bytes(&self, predictions: u64) -> Option<u64> {
        match &self.runtime {
            ConstraintRuntime::Text | ConstraintRuntime::PreparedForbidden { .. } => {
                predictions.checked_mul(4)
            }
            ConstraintRuntime::PreparedGrammar(_) => None,
        }
    }
    #[cfg(test)]
    pub(crate) fn constraint_is_active(&self) -> bool {
        self.prepared_grammar()
            .is_some_and(|grammar| grammar.is_active())
    }

    pub(crate) fn grammar_is_complete(&mut self) -> Result<bool, ConstraintError> {
        match &self.runtime {
            ConstraintRuntime::PreparedGrammar(_) => self.prepared_terminal(),
            ConstraintRuntime::Text | ConstraintRuntime::PreparedForbidden { .. } => Ok(false),
        }
    }

    pub(crate) fn prefix_is_complete(&self, history: &[u32]) -> Result<bool, ConstraintError> {
        match &self.runtime {
            ConstraintRuntime::PreparedGrammar(_) => self.prepared_terminal_at(history),
            ConstraintRuntime::Text => {
                self.prepared_plain_source()
                    .expect("text source")
                    .validate_history(history)
                    .map_err(ConstraintError::plain)?;
                Ok(false)
            }
            ConstraintRuntime::PreparedForbidden { .. } => {
                self.forbidden_source()
                    .expect("forbidden source")
                    .decision_at(history)
                    .map_err(ConstraintError::forbidden)?;
                Ok(false)
            }
        }
    }

    pub(crate) fn filter_at(&self, history: &[u32]) -> Result<TokenFilter, ConstraintError> {
        match &self.runtime {
            ConstraintRuntime::PreparedGrammar(_) => self.prepared_filter_at(history),
            ConstraintRuntime::Text => {
                self.prepared_plain_source()
                    .expect("text source")
                    .validate_history(history)
                    .map_err(ConstraintError::plain)?;
                self.restrict(TokenFilter::All)
            }
            ConstraintRuntime::PreparedForbidden { .. } => {
                let source = self.forbidden_source().expect("forbidden source");
                let decision = source
                    .decision_at(history)
                    .map_err(ConstraintError::forbidden)?;
                TokenFilter::allowed(
                    (0..source.inputs().vocabulary_len())
                        .map(|token| decision.allows(token as u32))
                        .collect(),
                )
                .map_err(ConstraintError::filter)
            }
        }
    }

    pub(crate) fn commit(&mut self, token: u32) -> Result<(), ConstraintError> {
        if self.prepared_grammar().is_some() {
            return self.commit_prepared(token);
        }
        self.prepare_history_mutation()?;
        if matches!(self.runtime, ConstraintRuntime::PreparedForbidden { .. }) {
            return self
                .forbidden_mutation()
                .and_then(|mut mutation| mutation.commit(token))
                .map_err(ConstraintError::forbidden);
        }
        self.validate_token(token)?;
        self.committed_tokens.try_push(token).map_err(|_| {
            ConstraintError::fixed(
                "plain controller mutation requires a unique prepared history destination",
            )
        })
    }

    #[cfg(test)]
    pub(crate) fn valid_token_ids(&mut self) -> Result<Option<Vec<u32>>, ConstraintError> {
        if self.prepared_grammar().is_none() {
            return Ok(None);
        }
        let filter = self.prepared_filter()?;
        Ok(match filter {
            TokenFilter::All => None,
            TokenFilter::Allowed(mask) => Some(
                mask.iter()
                    .enumerate()
                    .filter_map(|(token, &allowed)| allowed.then_some(token as u32))
                    .collect(),
            ),
        })
    }
}

impl TokenFilterController for ConstraintController {
    type Error = ConstraintError;

    fn inference_storage(&self) -> TextControllerStorage<'_> {
        if let Some(preparation) = &self.preparation {
            return preparation.storage(self);
        }
        if matches!(self.runtime, ConstraintRuntime::Text) {
            TextControllerStorage::RunOwnedWithSharedFilters(std::slice::from_ref(&self.validity))
        } else {
            TextControllerStorage::Unknown
        }
    }

    fn inference_workspace(
        &self,
        max_output_tokens: u64,
    ) -> Option<eredu_core::TextControllerWorkspace<'_>> {
        if let Some(preparation) = &self.preparation {
            return preparation.workspace();
        }
        if !matches!(self.runtime, ConstraintRuntime::Text) {
            return None;
        }
        let maximum = (self.committed_tokens.len() as u64).checked_add(max_output_tokens)?;
        // Vec growth can retain the old buffer while allocating a doubled
        // successor, including the minimum four-token allocation.
        let history = (self.committed_tokens.capacity() as u64)
            .checked_add(maximum.checked_mul(2)?.max(4))?
            .checked_mul(4)?;
        let source_filter = match self.validity.as_ref() {
            TokenFilter::All => 0,
            TokenFilter::Allowed(mask) => mask.capacity() as u64,
        };
        Some(eredu_core::TextControllerWorkspace {
            filter: self.validity.as_ref().into(),
            additional_host_bytes: history.checked_add(source_filter)?,
        })
    }

    fn current_filter(&mut self) -> Result<TokenFilter, Self::Error> {
        if self.prepared_grammar().is_some() {
            return self.prepared_filter();
        }
        self.filter_at(&self.committed_tokens)
    }

    fn current_decision(&mut self) -> Result<eredu_core::TokenSamplingDecision<'_>, Self::Error> {
        let filter = self.current_filter()?;
        if let Some(preparation) = &self.preparation {
            let source = preparation.tokenizer();
            return Ok(eredu_core::TokenSamplingDecision::new(filter)
                .with_original_tokenizer_validity(
                    source.generation_domain().expect("bound generation domain"),
                    eredu_core::OriginalSourceWitness::new(source),
                )
                .with_controller_storage(preparation.storage(self)));
        }
        let decision = eredu_core::TokenSamplingDecision::new(filter)
            .with_shared_tokenizer_validity(&self.validity);
        Ok(match self.inference_storage() {
            TextControllerStorage::Unknown => decision,
            storage => decision.with_controller_storage(storage),
        })
    }

    fn commit_token(&mut self, token_id: u32) -> Result<(), Self::Error> {
        self.commit(token_id)
    }

    fn is_complete(&mut self) -> Result<bool, Self::Error> {
        self.grammar_is_complete()
    }
}

impl eredu_runtime::execution_control::SnapshotTokenController for ConstraintController {
    fn original_snapshot_storage_bytes(&self) -> Option<u64> {
        let preparation = self.preparation.as_ref()?;
        if matches!(preparation.storage(self), TextControllerStorage::Unknown) {
            return None;
        }
        // Grammar owners are immutable: every operation produces a paid
        // successor. Plain/forbidden histories become immutable aliases;
        // their next mutation pays an independent history before appending.
        match self.runtime {
            ConstraintRuntime::Text
            | ConstraintRuntime::PreparedForbidden { .. }
            | ConstraintRuntime::PreparedGrammar(_) => {
                u64::try_from(std::mem::size_of::<Self>()).ok()
            }
        }
    }
    fn fork_original_snapshot(&self) -> Option<Self> {
        self.original_snapshot_storage_bytes()?;
        Some(self.clone())
    }
    fn snapshot_storage_bytes(&self) -> Option<u64> {
        let Self {
            runtime,
            committed_tokens,
            validity: _,    // Immutable shared storage is not copied by snapshots.
            authority: _,   // Shared exclusion custody is not a copy-byte allowance.
            preparation: _, // Retained source/account identity is not copied payload.
        } = self;
        let bytes = (std::mem::size_of::<Self>() as u64)
            .checked_add((committed_tokens.len() as u64).checked_mul(4)?)?;
        match runtime {
            ConstraintRuntime::Text => Some(bytes),
            // Prepared forbidden snapshots use copy_prepared_forbidden, which
            // owns an independent paid history. Ordinary Clone is identity-only.
            ConstraintRuntime::PreparedForbidden { .. } => None,
            ConstraintRuntime::PreparedGrammar(_) => None,
        }
    }
    fn fork_snapshot(&self) -> Result<Self, String> {
        if self.snapshot_storage_bytes().is_none() {
            return Err("complete grammar storage estimate is unavailable".into());
        }
        Ok(self.clone())
    }
}

impl SpeculativeTokenFilterController for ConstraintController {
    type PreparedGrammar = grammar_source::OriginalPreparedGrammarController;
    fn prepared_grammar(&self) -> Option<&Self::PreparedGrammar> {
        match &self.runtime {
            ConstraintRuntime::PreparedGrammar(owner) => Some(owner),
            _ => None,
        }
    }
    fn prepared_grammar_replacement_bytes(&self) -> Option<usize> {
        self.original_grammar_replacement_bytes()
    }
    fn replace_prepared_grammar(
        &self,
        grammar: Self::PreparedGrammar,
        funding: &eredu_core::HostMetadataFunding,
    ) -> Result<Self, eredu_core::speculative::PreparedGrammarInstallError<Self::PreparedGrammar>>
    {
        self.replace_original_grammar(grammar, funding)
    }

    fn prepared_plain_source(&self) -> Option<eredu_core::speculative::PlainControllerSource<'_>> {
        matches!(self.runtime, ConstraintRuntime::Text).then(|| {
            eredu_core::speculative::PlainControllerSource::new(
                &self.committed_tokens,
                &self.validity,
                TextControllerStorage::RunOwnedWithSharedFilters(std::slice::from_ref(
                    &self.validity,
                )),
            )
        })
    }
    fn prepared_plain_copy_bytes(&self, capacity: usize) -> Option<usize> {
        use eredu_core::speculative::{PlainControllerError, PlainControllerHistory};
        use std::mem::{size_of, size_of_val};
        let source = self.prepared_plain_source()?;
        if capacity < source.history().len() {
            return None;
        }
        let parts = [
            PlainControllerHistory::copy_metadata_bytes(capacity)?,
            size_of::<Self>(),
            size_of::<ConstraintRuntime>(),
            size_of::<SharedTokenFilter>(),
            size_of::<HostPreparationAuthority>(),
            size_of::<Result<Self, PlainControllerError>>(),
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
    fn copy_prepared_plain(
        &self,
        capacity: usize,
        host: HostPreparationAuthority,
    ) -> Result<Self, eredu_core::speculative::PlainControllerError> {
        use eredu_core::speculative::PlainControllerError;
        self.prepared_plain_source()
            .ok_or(PlainControllerError::Unknown)?;
        let committed_tokens = self
            .committed_tokens
            .copy_prepared(capacity, host.clone())?;
        Ok(Self {
            runtime: ConstraintRuntime::Text,
            committed_tokens,
            validity: self.validity.clone(),
            authority: host,
            preparation: self.preparation.clone(),
        })
    }
    fn prepared_plain_history_mut(
        &mut self,
    ) -> Option<&mut eredu_core::speculative::PlainControllerHistory> {
        matches!(self.runtime, ConstraintRuntime::Text).then_some(&mut self.committed_tokens)
    }
    fn prepared_forbidden_source(
        &self,
    ) -> Option<eredu_core::speculative::ForbiddenControllerSource<'_>> {
        self.forbidden_source()
    }
    fn prepared_forbidden_copy_bytes(&self, capacity: usize) -> Option<usize> {
        self.forbidden_copy_bytes(capacity)
    }
    fn copy_prepared_forbidden(
        &self,
        capacity: usize,
        host: HostPreparationAuthority,
    ) -> Result<Self, eredu_core::speculative::ForbiddenControllerError> {
        self.copy_forbidden(capacity, host)
    }
    fn prepared_forbidden_mutation(
        &mut self,
    ) -> Result<
        eredu_core::speculative::ForbiddenControllerMutation<'_>,
        eredu_core::speculative::ForbiddenControllerError,
    > {
        self.forbidden_mutation()
    }
    fn filter_at(&self, history: &[u32]) -> Result<TokenFilter, Self::Error> {
        ConstraintController::filter_at(self, history)
    }

    fn decision_at(
        &self,
        history: &[u32],
    ) -> Result<eredu_core::TokenSamplingDecision<'_>, Self::Error> {
        let decision = eredu_core::TokenSamplingDecision::new(self.filter_at(history)?)
            .with_shared_tokenizer_validity(&self.validity);
        Ok(match self.inference_storage() {
            TextControllerStorage::Unknown => decision,
            storage => decision.with_controller_storage(storage),
        })
    }

    fn prefix_is_complete(&self, history: &[u32]) -> Result<bool, Self::Error> {
        ConstraintController::prefix_is_complete(self, history)
    }
}

fn constraint_error(error: String) -> ConstraintError {
    ConstraintError::new(error)
}

/// Temporary tokenizer-wide compilation state. Runtime preparation freezes the
/// validated input recipe and retires these opaque products before publication.
pub(crate) struct ConstraintCompiler {
    factory: Option<Arc<ParserFactory>>,
    original_trie: Option<eredu_runtime::working_memory::OriginalTokenTrieSource>,
    allocation_funding: crate::runtime::chat::preparation_memory::PreparationFunding,
    environment: Arc<stock_parser::Environment>,
    eos_token_ids: Vec<u32>,
    tokenizer_json: Option<Vec<u8>>,
    grammar_tokenizer: Option<super::tokenizer_env::recipe::FrozenGrammarTokenizer>,
    #[cfg(test)]
    tokenizer_analysis_runs: usize,
    #[cfg(test)]
    schema_compilation_runs: AtomicUsize,
    _authority: HostPreparationAuthority,
}

pub(crate) struct ConstraintBlueprint {
    pub(crate) recipe: ConstraintRecipe,
    declaration: Option<declaration::HistoricalGrammarDeclaration>,
    // Arbitrary synthetic TokEnv fixtures are deliberately not serialized.
    #[cfg(test)]
    fixture_matcher: Option<Matcher>,
}

struct CompiledDeclaration {
    declaration: declaration::PendingGrammarDeclaration,
    #[cfg(test)]
    fixture_matcher: Option<Matcher>,
}

#[derive(Debug)]
enum DeclarationConstructionError<E> {
    Grammar(stock_parser::Error),
    Source(E),
    #[cfg(test)]
    Fixture(String),
}
impl<E: std::fmt::Display> std::fmt::Display for DeclarationConstructionError<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Grammar(error) => error.fmt(f),
            Self::Source(error) => error.fmt(f),
            #[cfg(test)]
            Self::Fixture(error) => f.write_str(error),
        }
    }
}
impl<E: std::error::Error + 'static> std::error::Error for DeclarationConstructionError<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Grammar(error) => Some(error),
            Self::Source(error) => Some(error),
            #[cfg(test)]
            Self::Fixture(_) => None,
        }
    }
}

#[derive(Debug, thiserror::Error)]
enum CompilerSourceCause {
    #[error("tokenizer input-prefix validation failed: {0}")]
    Tokenizer(#[source] eredu_runtime::working_memory::OriginalTokenizerPrefixError),
    #[error("original tokenizer trie construction failed: {0}")]
    Trie(#[source] eredu_runtime::working_memory::OriginalTokenTrieSourceError),
    #[error("tokenizer has an empty or unrepresentable canonical token domain")]
    Vocabulary,
    #[error("EOS token ID {0} has no consistent tokenizer mapping")]
    Eos(u32),
    #[error(transparent)]
    Storage(crate::runtime::chat::preparation_memory::StorageFailure),
    #[error("{0}")]
    Metadata(#[source] eredu_core::HostMetadataFundingError),
    #[error("grammar environment construction failed: {0}")]
    Parser(#[source] stock_parser::Error),
    #[error("tokenizer analysis failed: {0}")]
    Analysis(#[source] anyhow::Error),
}

/// Source construction failure retains its actual trie and funding after any
/// partial syntax, recognition and destination state has retired.
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
pub(crate) struct ConstraintCompilerSourceError {
    #[source]
    cause: CompilerSourceCause,
    source: CompilerSource,
    funding: eredu_core::HostMetadataFunding,
}

#[derive(Debug)]
enum CompilerSource {
    Tokenizer(eredu_runtime::working_memory::OriginalTokenizer),
    Trie(eredu_runtime::working_memory::OriginalTokenTrieSource),
}

impl ConstraintCompiler {
    /// Retains the same producer account for facade metadata construction.
    pub(crate) fn allocation_funding(
        &self,
    ) -> &crate::runtime::chat::preparation_memory::PreparationFunding {
        &self.allocation_funding
    }

    /// Analyzes the accepted trie with stock public APIs. Compiler headroom and
    /// facade-owned destinations retain the original metadata account.
    pub(crate) fn from_original_source(
        trie: eredu_runtime::working_memory::OriginalTokenTrieSource,
        funding: &eredu_core::HostMetadataFunding,
    ) -> Result<Self, ConstraintCompilerSourceError> {
        Self::from_original_source_with_memory_policy(
            trie,
            funding,
            eredu_runtime::working_memory::DependencyMemoryPolicy::default(),
        )
    }
    pub(crate) fn from_original_source_with_memory_policy(
        trie: eredu_runtime::working_memory::OriginalTokenTrieSource,
        funding: &eredu_core::HostMetadataFunding,
        memory: eredu_runtime::working_memory::DependencyMemoryPolicy,
    ) -> Result<Self, ConstraintCompilerSourceError> {
        let retain = |cause| ConstraintCompilerSourceError {
            cause,
            source: CompilerSource::Trie(trie.clone()),
            funding: funding.clone(),
        };
        let allocation_funding =
            crate::runtime::chat::preparation_memory::PreparationFunding::from_metadata(funding)
                .with_memory_policy(memory);
        let environment = stock_parser::Environment::new(
            stock_parser::TokenizerSource::Original(trie.clone()),
            &allocation_funding,
        )
        .map_err(|error| retain(CompilerSourceCause::Parser(error)))?;
        let input_bytes = (0..trie.trie().vocab_size())
            .try_fold(0usize, |total, id| {
                total
                    .checked_add(trie.trie().token_len(id as u32))?
                    .checked_add(std::mem::size_of::<u32>())
            })
            .ok_or_else(|| {
                retain(CompilerSourceCause::Metadata(
                    eredu_core::HostMetadataFundingError::Overflow,
                ))
            })?;
        let headroom = allocation_funding
            .memory_policy()
            .estimate(input_bytes)
            .and_then(|n| {
                n.checked_add(eredu_nn::workspace::WorkspaceContext::metadata_arc_bytes::<
                    ParserFactory,
                >()?)
            })
            .and_then(|n| {
                n.checked_add(HostPreparationAuthority::retention_bytes::<
                    eredu_core::HostMetadataFunding,
                >()?)
            })
            .ok_or_else(|| {
                retain(CompilerSourceCause::Metadata(
                    eredu_core::HostMetadataFundingError::Overflow,
                ))
            })?;
        funding
            .reserve_metadata(headroom)
            .map_err(|error| retain(CompilerSourceCause::Metadata(error)))?;
        let token_env: TokEnv = environment.clone();
        let mut factory = ParserFactory::new_simple(&token_env)
            .map_err(|error| retain(CompilerSourceCause::Analysis(error)))?;
        environment
            .take_failure()
            .map_err(|error| retain(CompilerSourceCause::Parser(error)))?;
        factory.quiet();
        let mut eos_token_ids = Vec::new();
        for &id in trie.trie().eos_tokens() {
            // Stock toktrie retains INVALID_TOKEN for an absent primary EOS.
            // Public policy metadata contains only actual vocabulary IDs.
            if (id as usize) < trie.trie().vocab_size() {
                allocation_funding
                    .try_push(&mut eos_token_ids, id)
                    .map_err(|error| retain(CompilerSourceCause::Storage(error)))?;
            }
        }
        Ok(Self {
            factory: Some(Arc::new(factory)),
            original_trie: Some(trie),
            allocation_funding,
            environment,
            eos_token_ids,
            tokenizer_json: None,
            grammar_tokenizer: None,
            #[cfg(test)]
            tokenizer_analysis_runs: 1,
            #[cfg(test)]
            schema_compilation_runs: AtomicUsize::new(0),
            _authority: HostPreparationAuthority::retain(funding.clone()),
        })
    }

    fn trie(&self) -> &llguidance::toktrie::TokTrie {
        match &self.original_trie {
            Some(source) => source.trie(),
            None => self
                .factory
                .as_ref()
                .expect("compiler tokenizer source")
                .tok_env()
                .tok_trie(),
        }
    }

    pub(crate) fn from_tokenizer(
        tokenizer: &ChatTokenizer,
        eos_token_ids: &[u32],
    ) -> Result<Self, String> {
        Self::from_tokenizer_with_authority(
            tokenizer,
            eos_token_ids,
            &HostPreparationAuthority::unmanaged(),
        )
    }

    pub(crate) fn from_tokenizer_with_authority(
        tokenizer: &ChatTokenizer,
        eos_token_ids: &[u32],
        authority: &HostPreparationAuthority,
    ) -> Result<Self, String> {
        let frozen = super::tokenizer_env::recipe::freeze(tokenizer, eos_token_ids, authority)?;
        let mut compiler = Self::from_prepared_environment(
            frozen.environment,
            eos_token_ids.to_vec(),
            Some(frozen.bytes),
            authority,
        )?;
        compiler.grammar_tokenizer = Some(frozen.grammar);
        Ok(compiler)
    }

    #[cfg(test)]
    fn from_tok_env(token_env: TokEnv, eos_token_ids: Vec<u32>) -> Result<Self, String> {
        Self::from_prepared_environment(
            token_env,
            eos_token_ids,
            None,
            &HostPreparationAuthority::unmanaged(),
        )
    }

    fn from_prepared_environment(
        token_env: TokEnv,
        eos_token_ids: Vec<u32>,
        tokenizer_json: Option<Vec<u8>>,
        authority: &HostPreparationAuthority,
    ) -> Result<Self, String> {
        let allocation_funding =
            crate::runtime::chat::preparation_memory::PreparationFunding::unmanaged();
        let environment = stock_parser::Environment::new(
            stock_parser::TokenizerSource::Ordinary(token_env),
            &allocation_funding,
        )
        .map_err(|error| error.to_string())?;
        let token_env: TokEnv = environment.clone();
        let mut factory = ParserFactory::new_simple(&token_env)
            .map_err(|error| format!("failed to analyze tokenizer trie: {error}"))?;
        factory.quiet();
        Ok(Self {
            factory: Some(Arc::new(factory)),
            original_trie: None,
            allocation_funding:
                crate::runtime::chat::preparation_memory::PreparationFunding::unmanaged(),
            environment,
            eos_token_ids,
            tokenizer_json,
            grammar_tokenizer: None,
            #[cfg(test)]
            tokenizer_analysis_runs: 1,
            #[cfg(test)]
            schema_compilation_runs: AtomicUsize::new(0),
            _authority: authority.clone(),
        })
    }

    #[cfg(test)]
    pub(crate) fn synthetic_for_tests() -> Self {
        Self::from_tok_env(
            llguidance::toktrie::ApproximateTokEnv::single_byte_env(),
            vec![255],
        )
        .expect("single-byte tokenizer must support llguidance")
    }

    #[cfg(test)]
    pub(crate) fn synthetic_with_eos_aliases_for_tests(eos_token_ids: &[u32]) -> Self {
        use llguidance::toktrie::{ApproximateTokEnv, TokRxInfo, TokTrie};

        let words = (0..=255).map(|byte| vec![byte]).collect::<Vec<_>>();
        let info = TokRxInfo {
            vocab_size: words.len() as u32,
            tok_eos: eos_token_ids[0],
            tok_bos: None,
            tok_pad: None,
            tok_unk: None,
            tok_end_of_turn: None,
        };
        let trie = TokTrie::from(&info, &words).with_eos_tokens(eos_token_ids);
        let environment = Arc::new(ApproximateTokEnv::new(trie));
        Self::from_tok_env(environment, eos_token_ids.to_vec())
            .expect("single-byte tokenizer with EOS aliases must support llguidance")
    }

    #[cfg(test)]
    pub(crate) fn synthetic_with_tokens_for_tests(extra_tokens: &[&[u8]]) -> Self {
        use llguidance::toktrie::{ApproximateTokEnv, TokRxInfo, TokTrie};

        let mut words = (0..=255).map(|byte| vec![byte]).collect::<Vec<_>>();
        words.extend(
            [
                b"\xFF<|tool|>".as_slice(),
                b"\xFF<|/tool|>",
                b"\xFF<|user|>",
                b"\xFF<|system|>",
                b"\xFF<|assistant|>",
                b"\xFF<|end|>",
            ]
            .into_iter()
            .map(<[u8]>::to_vec),
        );
        let eos = words.len() as u32 - 1;
        words.extend(extra_tokens.iter().map(|token| token.to_vec()));
        let info = TokRxInfo {
            vocab_size: words.len() as u32,
            tok_eos: eos,
            tok_bos: None,
            tok_pad: None,
            tok_unk: None,
            tok_end_of_turn: None,
        };
        let environment = Arc::new(ApproximateTokEnv::new(TokTrie::from(&info, &words)));
        Self::from_tok_env(environment, vec![eos])
            .expect("synthetic tokenizer must support llguidance")
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn compile_generation_plan(
        &self,
        dialect: &'static dyn FormatDialect,
        parameters: DialectParameters,
        tools: &[Value],
        tool_choice: ToolChoice,
        parallel_tool_calls: ParallelToolCallPolicy,
        runtime_structural_token_spellings: Vec<String>,
        resolved_structural_token_ids: Vec<u32>,
        runtime_stop_sequences: Vec<String>,
        tool_surface: bool,
    ) -> Result<GenerationRuntimePlan, PreparationFailure> {
        use preparation_error::Cause as PreparationCause;
        let retained =
            |cause| PreparationFailure::new(cause, &self._authority, &self.allocation_funding);
        let controls = [
            std::mem::size_of::<PreparationFailure>(),
            std::mem::size_of::<PreparationCause>(),
            std::mem::size_of::<Result<GenerationRuntimePlan, PreparationFailure>>(),
            std::mem::size_of::<Result<GenerationRuntimePlan, PreparationCause>>(),
            std::mem::size_of::<crate::api::ConstraintError>(),
            std::mem::size_of::<crate::api::TextModelError>(),
        ];
        let bytes = controls
            .into_iter()
            .try_fold(std::mem::size_of_val(&controls), usize::checked_add)
            .ok_or_else(|| retained(PreparationCause::Overflow))?;
        self.allocation_funding
            .reserve(bytes)
            .map_err(|cause| retained(PreparationCause::Allocation(cause)))?;
        let result = (|| -> Result<GenerationRuntimePlan, PreparationCause> {
            #[cfg(test)]
            self.schema_compilation_runs.fetch_add(1, Ordering::Relaxed);

            let grammar_structural_token_spellings =
                dialect.required_structural_tokens(parameters)?;
            if runtime_structural_token_spellings.len() != resolved_structural_token_ids.len()
                || runtime_structural_token_spellings.len()
                    < grammar_structural_token_spellings.len()
                || !runtime_structural_token_spellings
                    .iter()
                    .zip(grammar_structural_token_spellings)
                    .all(|(runtime, grammar)| runtime == grammar)
            {
                return Err(self.allocation_funding.try_format(format_args!(
                "format dialect declares {} leading structural tokens but {} runtime spellings and {} tokenizer IDs were resolved",
                grammar_structural_token_spellings.len(),
                runtime_structural_token_spellings.len(),
                resolved_structural_token_ids.len()
            ))?.into());
            }
            let grammar_structural_token_ids =
                &resolved_structural_token_ids[..grammar_structural_token_spellings.len()];
            let declarations = ToolDeclarations::prepare(tools, &self.allocation_funding)?;
            let declared_tools = declarations.as_slice();
            // Schema admission is independent of the grammar engine's supported subset.
            // Every completed call is checked against the original schema by the sink.
            let schemas = crate::runtime::chat::tool_schema::registered::PendingSchemas::compile(
            declared_tools, &self.allocation_funding, &self._authority,
            dialect.original_channel_program(parameters).is_ok_and(|spec|
                matches!(spec.payload_shape, crate::runtime::chat::dialect::DeclarativePayloadShape::TaggedParameters(_))),
        )?;
            let trigger = if matches!(tool_choice, ToolChoice::None | ToolChoice::Auto) {
                dialect.auto_activation_trigger(parameters)?
            } else {
                None
            };
            // Serialize the borrowed candidate before handing its owned grammar to
            // the compiler. This avoids a deep source clone. An unsuccessful strict
            // candidate retires its paid recipe before the syntax candidate starts.
            let compile_configuration = |configuration: crate::runtime::chat::dialect::ConstraintConfiguration|
            -> Result<_, PreparationCause> {
            let mut recipe = ConstraintRecipe::new_with_trie_info(
            self.tokenizer_json.as_deref(),
            &configuration.grammar,
            tools,
            &self.eos_token_ids,
            &runtime_structural_token_spellings,
            &resolved_structural_token_ids,
            &runtime_stop_sequences,
            trigger,
            *self.trie().info(),
            self.grammar_tokenizer.as_ref(),
            &self.allocation_funding,
            &self._authority,
        )?;
            recipe.bind_original_trie(self.original_trie.as_ref());
            let compiled = self.compile_declaration(configuration.grammar)?;
            Ok((recipe, compiled))
        };
            let (recipe, compiled) = if tool_surface {
                let configuration = dialect.constraint_configuration(
                    parameters,
                    declared_tools,
                    tool_choice,
                    parallel_tool_calls,
                    grammar_structural_token_ids,
                    &self.allocation_funding,
                )?;
                match compile_configuration(configuration) {
                    Ok(compiled) => compiled,
                    Err(error)
                        if match &error {
                            PreparationCause::Declaration(
                                DeclarationConstructionError::Grammar(cause),
                            ) => cause.allows_schema_fallback(),
                            _ => false,
                        } =>
                    {
                        // Keep protocol, function names and call limits constrained.
                        // Only argument-schema enforcement moves to completion.
                        let syntax_schema = Value::Bool(true);
                        let mut syntax_tools = Vec::new();
                        self.allocation_funding
                            .try_grow_vec(&mut syntax_tools, declared_tools.len())?;
                        syntax_tools.extend(declared_tools.iter().map(|tool| ToolDefinition {
                            name: tool.name,
                            parameters: &syntax_schema,
                        }));
                        let configuration = dialect.constraint_configuration(
                            parameters,
                            &syntax_tools,
                            tool_choice,
                            parallel_tool_calls,
                            grammar_structural_token_ids,
                            &self.allocation_funding,
                        )?;
                        compile_configuration(configuration)?
                    }
                    Err(error) => return Err(error),
                }
            } else {
                let configuration = dialect.semantic_constraint_configuration(
                    parameters,
                    grammar_structural_token_ids,
                    &self.eos_token_ids,
                    &self.allocation_funding,
                )?;
                compile_configuration(configuration)?
            };
            let fingerprint = recipe.fingerprint();
            let declaration = compiled.declaration.bind(&recipe);
            // Production plans retain independent immutable inputs only. Synthetic fixtures explicitly
            // keep their nonserializable environment in test builds.
            #[cfg(test)]
            let fixture_matcher = compiled.fixture_matcher;
            let mut plan = GenerationRuntimePlan::new(GenerationRuntimePlanParts {
                tool_choice,
                tool_surface,
                generation_constraint: GenerationConstraint::new(
                    fingerprint,
                    ConstraintBlueprint {
                        recipe,
                        declaration: Some(declaration),
                        #[cfg(test)]
                        fixture_matcher,
                    },
                ),
                dialect,
                dialect_parameters: parameters,
            });
            if tool_choice != ToolChoice::None {
                plan.bind_tool_schema_sources(schemas);
            }
            Ok(plan)
        })();
        result.map_err(retained)
    }

    fn compile_grammar(
        &self,
        grammar: llguidance::api::TopLevelGrammar,
    ) -> Result<stock_parser::Template, stock_parser::Error> {
        stock_parser::Template::compile(
            self.factory.as_ref().expect("compiler factory"),
            self.environment.clone(),
            grammar,
            &self.allocation_funding,
        )
    }

    #[cfg(test)]
    fn compile_matcher(
        &self,
        grammar: llguidance::api::TopLevelGrammar,
    ) -> Result<Matcher, String> {
        let compiled = self
            .compile_grammar(grammar)
            .map_err(|error| error.to_string())?;
        let matcher = Matcher::new(compiled.fixture_parser().map_err(anyhow::Error::new));
        match matcher.get_error() {
            Some(error) => Err(error),
            None => Ok(matcher),
        }
    }

    fn compile_declaration(
        &self,
        grammar: llguidance::api::TopLevelGrammar,
    ) -> Result<
        CompiledDeclaration,
        DeclarationConstructionError<declaration::PendingGrammarDeclarationError>,
    > {
        let compiled = self
            .compile_grammar(grammar)
            .map_err(DeclarationConstructionError::Grammar)?;
        #[cfg(test)]
        let fixture_matcher = if self.original_trie.is_none() {
            Some(Matcher::new(
                compiled.fixture_parser().map_err(anyhow::Error::new),
            ))
        } else {
            None
        };
        let declaration = declaration::PendingGrammarDeclaration::from_compiled(
            compiled,
            &self._authority,
            &self.allocation_funding,
        )
        .map_err(DeclarationConstructionError::Source)?;
        Ok(CompiledDeclaration {
            declaration,
            #[cfg(test)]
            fixture_matcher,
        })
    }

    #[cfg(test)]
    pub(crate) fn compile_tool_plan(
        &self,
        dialect: &'static dyn FormatDialect,
        parameters: DialectParameters,
        tools: &[Value],
        tool_choice: ToolChoice,
        parallel_tool_calls: ParallelToolCallPolicy,
        resolved_structural_token_ids: Vec<u32>,
    ) -> Result<GenerationRuntimePlan, String> {
        let structural_token_spellings = dialect
            .required_structural_tokens(parameters)?
            .iter()
            .map(|spelling| (*spelling).to_owned())
            .collect();
        let stop_sequences = dialect
            .stop_sequences(parameters)?
            .iter()
            .map(|sequence| (*sequence).to_owned())
            .collect();
        self.compile_generation_plan(
            dialect,
            parameters,
            tools,
            tool_choice,
            parallel_tool_calls,
            structural_token_spellings,
            resolved_structural_token_ids,
            stop_sequences,
            true,
        )
        .map_err(|error| error.to_string())
    }

    #[cfg(test)]
    pub(crate) fn cache_analysis_counts(&self) -> (usize, usize) {
        (
            self.tokenizer_analysis_runs,
            self.schema_compilation_runs.load(Ordering::Relaxed),
        )
    }
}

impl ConstraintBlueprint {
    pub(crate) fn compiled_grammar_source(
        &self,
    ) -> Option<&eredu_core::SharedControllerDeclaration> {
        self.declaration.as_ref().map(|source| source.source())
    }
}

impl GenerationConstraint {
    pub(crate) fn new(fingerprint: [u8; 32], inner: ConstraintBlueprint) -> Self {
        Self {
            fingerprint,
            inner: Arc::new(inner),
        }
    }

    /// Direct dependency matcher for independent declaration semantics in tests.
    #[cfg(test)]
    pub(crate) fn grammar_matcher(&self) -> Matcher {
        self.inner
            .fixture_matcher
            .as_ref()
            .expect("independent fixture parser")
            .deep_clone()
    }
}

pub(crate) fn tool_call_bounds(
    tool_choice: ToolChoice,
    parallel_tool_calls: ParallelToolCallPolicy,
    tools: &[ToolDefinition<'_>],
) -> Result<(usize, Option<usize>), &'static str> {
    if tool_choice == ToolChoice::Required && tools.is_empty() {
        return Err("tool_choice is required but no tools were supplied".into());
    }

    let (min_calls, max_calls) = match tool_choice {
        ToolChoice::None => (0, Some(0)),
        ToolChoice::Auto => (
            0,
            match parallel_tool_calls {
                ParallelToolCallPolicy::Disabled => Some(1),
                ParallelToolCallPolicy::Enabled { max_calls } => max_calls.map(NonZeroUsize::get),
            },
        ),
        ToolChoice::Required => (
            1,
            match parallel_tool_calls {
                ParallelToolCallPolicy::Disabled => Some(1),
                ParallelToolCallPolicy::Enabled { max_calls } => max_calls.map(NonZeroUsize::get),
            },
        ),
    };
    if max_calls.is_some_and(|maximum| maximum < min_calls) {
        return Err("parallel tool-call limit cannot satisfy tool_choice".into());
    }
    Ok((min_calls, max_calls))
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use llguidance::toktrie::TokenId;
    use serde_json::json;

    use super::{ConstraintCompiler, ParallelToolCallPolicy, ToolChoice};
    use crate::runtime::chat::dialect::{
        DECLARATIVE_DIALECT, DeclarativeDialectSpec, DeclarativePayloadShape, DialectParameters,
        ExactEnvelope, GenerationPromptBehavior, JsonFunctionEnvelope, ParallelCallLayout,
    };

    #[test]
    fn prepared_plain_controller_commit_preserves_source_forcing_and_host_custody() {
        use eredu_core::{
            HostPreparationAuthority, SharedTokenFilter, SpeculativeTokenFilterController,
            TokenFilter, TokenFilterController,
        };
        use eredu_runtime::{
            TokenDomain,
            execution_control::TokenChoiceController,
            generation::{ConstrainedSampler, DefaultSampler, SpeculativeSampler},
            working_memory::WorkspaceSamplingBackend,
        };
        use std::sync::{
            Arc,
            atomic::{AtomicBool, Ordering},
        };
        struct Retires(Arc<AtomicBool>);
        impl Drop for Retires {
            fn drop(&mut self) {
                self.0.store(true, Ordering::SeqCst);
            }
        }
        fn host(retired: &Arc<AtomicBool>) -> HostPreparationAuthority {
            HostPreparationAuthority::retain(Retires(retired.clone()))
        }
        type Plain = ConstrainedSampler<DefaultSampler, super::ConstraintController>;
        fn plan(
            source: &Plain,
        ) -> eredu_runtime::generation::PreparedSpeculativeController<'_, Plain> {
            SpeculativeSampler::<WorkspaceSamplingBackend>::prepared_controller(source).unwrap()
        }
        let validity =
            SharedTokenFilter::new(TokenFilter::allowed(vec![false, true, true]).unwrap());
        let mut controller = super::ConstraintController::text(validity);
        controller.commit_token(1).unwrap();
        let mut ordinary = TokenChoiceController::new(controller.clone(), TokenDomain::new(3));
        ordinary.force_at(2, TokenDomain::new(3), 1).unwrap();
        let mut source = Plain::new(DefaultSampler, controller);
        SpeculativeSampler::<WorkspaceSamplingBackend>::control_force_next(
            &mut source,
            2,
            TokenDomain::new(3),
            1,
        )
        .unwrap();
        assert!(
            SpeculativeSampler::<WorkspaceSamplingBackend>::prepared_host_copy(&source).is_none()
        );
        assert!(
            SpeculativeSampler::<WorkspaceSamplingBackend>::prepared_logit_policy(&source)
                .is_none()
        );
        let initial = Arc::new(AtomicBool::new(false));
        assert!(plan(&source).copy_metadata_bytes() > 0);
        let copied = plan(&source).copy(host(&initial)).unwrap();
        let alias = copied.clone();
        let (decision, _) = plan(&copied).logits(&[1]).unwrap().into_parts();
        assert_eq!(decision.forced_token(), Some(2));
        assert!(
            decision.source().validity().same_storage(
                source
                    .controller()
                    .prepared_plain_source()
                    .unwrap()
                    .validity()
            )
        );
        assert!(plan(&copied).logits(&[0]).is_err());
        assert!(plan(&copied).logits(&[1, 1]).is_err());
        let refused = Arc::new(AtomicBool::new(false));
        assert!(plan(&copied).commit(1, host(&refused)).is_err());
        assert!(refused.load(Ordering::SeqCst));
        assert_eq!(
            copied
                .controller()
                .prepared_plain_source()
                .unwrap()
                .history(),
            &[1]
        );
        assert_eq!(
            SpeculativeSampler::<WorkspaceSamplingBackend>::control_pending_forced(&copied),
            Some(2)
        );
        let committed_host = Arc::new(AtomicBool::new(false));
        let committed = plan(&copied).commit(2, host(&committed_host)).unwrap();
        ordinary.commit_token(2).unwrap();
        assert_eq!(
            committed
                .controller()
                .prepared_plain_source()
                .unwrap()
                .history(),
            ordinary.inner().prepared_plain_source().unwrap().history()
        );
        assert_eq!(
            SpeculativeSampler::<WorkspaceSamplingBackend>::control_pending_forced(&committed),
            ordinary.pending_forced()
        );
        assert_eq!(
            copied
                .controller()
                .prepared_plain_source()
                .unwrap()
                .history(),
            &[1]
        );
        drop(copied);
        assert!(!initial.load(Ordering::SeqCst));
        drop(alias);
        assert!(initial.load(Ordering::SeqCst));
        assert!(!committed_host.load(Ordering::SeqCst));
        drop(committed);
        assert!(committed_host.load(Ordering::SeqCst));
    }

    #[test]
    fn prepared_plain_choice_tracks_controller_replacement_and_mask_expansion() {
        use eredu_core::{
            HostPreparationAuthority, SharedTokenFilter, SpeculativeTokenFilterController,
            TokenFilter, TokenFilterController,
        };
        use eredu_runtime::{
            TokenDomain,
            generation::{ConstrainedSampler, DefaultSampler, SpeculativeSampler, TokenMaskPlan},
            working_memory::WorkspaceSamplingBackend,
        };
        use std::sync::{
            Arc,
            atomic::{AtomicBool, Ordering},
        };
        struct Retires(Arc<AtomicBool>);
        impl Drop for Retires {
            fn drop(&mut self) {
                self.0.store(true, Ordering::SeqCst);
            }
        }
        type Plain = ConstrainedSampler<DefaultSampler, super::ConstraintController>;
        fn plan(
            source: &Plain,
        ) -> eredu_runtime::generation::PreparedSpeculativeController<'_, Plain> {
            SpeculativeSampler::<WorkspaceSamplingBackend>::prepared_controller(source).unwrap()
        }
        let mut controller = super::ConstraintController::text(SharedTokenFilter::new(
            TokenFilter::allowed(vec![true, false, true]).unwrap(),
        ));
        controller.commit_token(2).unwrap();
        let mut source = Plain::new(DefaultSampler, controller);
        SpeculativeSampler::<WorkspaceSamplingBackend>::control_force_next(
            &mut source,
            2,
            TokenDomain::new(3),
            1,
        )
        .unwrap();
        // An unpriced ordinary history cannot become a completed-value witness.
        assert!(plan(&source).choice(0.0).is_err());
        let retired = Arc::new(AtomicBool::new(false));
        let source = plan(&source)
            .copy(HostPreparationAuthority::retain(Retires(retired.clone())))
            .unwrap();
        let choice = plan(&source).choice(0.0).unwrap();
        assert!(plan(&source).matches_choice(&choice));
        assert!(choice.greedy(0.0).is_ok());
        assert!(choice.categorical(0.7).is_err());
        let stochastic = plan(&source).choice(0.7).unwrap();
        assert!(stochastic.categorical(0.7).is_ok());
        assert!(stochastic.categorical(0.8).is_err());
        let (decision, _) = plan(&source).logits(&[2]).unwrap().into_parts();
        let mask = TokenMaskPlan::new(
            decision.source().validity(),
            &[1, 2, 5],
            decision.forced_token(),
        )
        .unwrap();
        let mut invalid = Vec::with_capacity(mask.elements());
        mask.fill(&mut invalid).unwrap();
        assert_eq!(
            invalid,
            [true, true, false, true, true, true, true, false, true, true]
        );
        let unforced = TokenMaskPlan::new(decision.source().validity(), &[1, 5], None).unwrap();
        let mut invalid = Vec::with_capacity(unforced.elements());
        unforced.fill(&mut invalid).unwrap();
        assert_eq!(invalid, [false, true, false, true, true]);
        let next = plan(&source)
            .commit(2, HostPreparationAuthority::retain(()))
            .unwrap();
        assert!(!plan(&next).matches_choice(&choice));
        assert_eq!(
            next.controller().prepared_plain_source().unwrap().history(),
            &[2, 2]
        );
        assert!(choice.same_source(&choice.clone()));
        drop(source);
        assert!(!retired.load(Ordering::SeqCst));
        drop(choice);
        assert!(!retired.load(Ordering::SeqCst));
        drop(stochastic);
        assert!(retired.load(Ordering::SeqCst));
        assert!(
            TokenMaskPlan::new(&TokenFilter::All, &[1, 5], None)
                .unwrap()
                .is_identity()
        );
    }

    const SYNTHETIC_JSON_FUNCTION: JsonFunctionEnvelope = JsonFunctionEnvelope {
        envelope: ExactEnvelope {
            prefix: "",
            suffix: "",
        },
        name_field: "name",
        arguments_field: "arguments",
        call_id: None,
    };

    const SYNTHETIC_SPEC: DeclarativeDialectSpec = DeclarativeDialectSpec {
        generation_prompt_behavior: GenerationPromptBehavior::HonorRequest,
        reasoning_template_kwarg: "enable_thinking",
        supports_tool_reasoning: true,
        output: ExactEnvelope {
            prefix: r#"{"calls":"#,
            suffix: "}",
        },
        call: ExactEnvelope {
            prefix: "",
            suffix: "",
        },
        payload_shape: DeclarativePayloadShape::JsonList,
        json_function: Some(&SYNTHETIC_JSON_FUNCTION),
        reasoning_channel: None,
        text_channel: None,
        raw_text_before_calls: false,
        call_separator: ",",
        parallel_layout: ParallelCallLayout::SingleEnvelope,
        protocol_max_tools: None,
        protocol_max_calls: None,
        auto_activation_trigger: Some(r#"{"calls":"#),
        required_structural_tokens: &[],
        stop_sequences: &[],
    };

    const SYNTHETIC_PARAMETERS: DialectParameters = DialectParameters::Declarative(&SYNTHETIC_SPEC);

    fn compiler() -> ConstraintCompiler {
        ConstraintCompiler::synthetic_for_tests()
    }

    fn activation_tokenizer(markers: &[&str], special: bool) -> super::ChatTokenizer {
        use tokenizers::{AddedToken, decoders::byte_level::ByteLevel, models::bpe::BPE};

        let vocabulary = (b'!'..=b'~')
            .map(|byte| (byte as char).to_string())
            .chain(["Ġ".to_owned()])
            .enumerate()
            .map(|(id, spelling)| (spelling, id as u32))
            .collect::<tokenizers::models::bpe::Vocab>();
        let model = BPE::builder()
            .vocab_and_merges(vocabulary, Vec::new())
            .build()
            .unwrap();
        let mut tokenizer = tokenizers::Tokenizer::new(model);
        tokenizer.with_pre_tokenizer(Some(ByteLevel::new(false, false, false)));
        tokenizer.with_decoder(Some(ByteLevel::default()));
        tokenizer
            .add_tokens(
                markers
                    .iter()
                    .map(|&marker| AddedToken::from(marker, special).normalized(false)),
            )
            .unwrap();
        super::ChatTokenizer::from_tokenizer(tokenizer)
    }

    #[test]
    fn lfm_auto_activation_preserves_added_token_identity() {
        use crate::runtime::chat::lfm2::{LFM2_DIALECT, LFM2_PARAMETERS};

        let markers = ["<|tool_call_start|>", "<|tool_call_end|>", "<|im_end|>"];
        // LFM's tool markers are added tokens with special=false. Exercise
        // special=true too: explicit grammar token IDs must work for both.
        for special in [false, true] {
            let tokenizer = activation_tokenizer(&markers, special);
            let ids = markers
                .iter()
                .map(|marker| tokenizer.token_to_id(marker).unwrap())
                .collect::<Vec<_>>();
            let compiler = super::fixtures::Compiler::new(&tokenizer, &[ids[2]]);
            let tools = [tool(
                "ping",
                json!({"type": "object", "additionalProperties": false}),
            )];
            let plan = compiler
                .compile_tool_plan(
                    &LFM2_DIALECT,
                    DialectParameters::Custom(&LFM2_PARAMETERS),
                    &tools,
                    ToolChoice::Auto,
                    ParallelToolCallPolicy::Disabled,
                    ids.clone(),
                )
                .unwrap();
            for preamble in ["", "hello"] {
                let mut controller = plan.controller();
                for &token in tokenizer.encode(preamble, false).unwrap().get_ids() {
                    controller.commit(token).unwrap();
                }
                assert!(!controller.constraint_is_active());
                let marker = tokenizer.encode(markers[0], false).unwrap();
                assert_eq!(marker.get_ids(), &[ids[0]]);
                // Filtering examines the marker even before it is sampled.
                let history = tokenizer
                    .encode(preamble, false)
                    .unwrap()
                    .get_ids()
                    .to_vec();
                assert!(controller.filter_at(&history).unwrap().allows(ids[0]));
                controller.commit(ids[0]).unwrap();
                assert!(controller.constraint_is_active());
                assert!(
                    !controller
                        .valid_token_ids()
                        .unwrap()
                        .unwrap()
                        .contains(&ids[1])
                );
                for &token in tokenizer.encode("[ping()]", false).unwrap().get_ids() {
                    controller.commit(token).unwrap();
                }
                controller.commit(ids[1]).unwrap();
                assert!(controller.grammar_is_complete().unwrap());
            }

            let forbidden = compiler
                .compile_tool_plan(
                    &LFM2_DIALECT,
                    DialectParameters::Custom(&LFM2_PARAMETERS),
                    &tools,
                    ToolChoice::None,
                    ParallelToolCallPolicy::Disabled,
                    ids.clone(),
                )
                .unwrap();
            let mut controller = forbidden.controller();
            assert!(!controller.filter_at(&[]).unwrap().allows(ids[0]));
            assert!(controller.commit(ids[0]).is_err());
        }
    }

    #[test]
    fn auto_activation_spanning_special_token_and_text_preserves_exact_bytes() {
        const SPEC: DeclarativeDialectSpec = DeclarativeDialectSpec {
            output: ExactEnvelope {
                prefix: "<|tool|> ",
                suffix: "",
            },
            auto_activation_trigger: Some("<|tool|> "),
            required_structural_tokens: &["<|tool|>"],
            ..SYNTHETIC_SPEC
        };
        let tokenizer = activation_tokenizer(&["<|tool|>", "<|end|>"], true);
        let marker = tokenizer.token_to_id("<|tool|>").unwrap();
        let eos = tokenizer.token_to_id("<|end|>").unwrap();
        let compiler = super::fixtures::Compiler::new(&tokenizer, &[eos]);
        let plan = compiler
            .compile_tool_plan(
                &DECLARATIVE_DIALECT,
                DialectParameters::Declarative(&SPEC),
                &[tool(
                    "ping",
                    json!({"type": "object", "additionalProperties": false}),
                )],
                ToolChoice::Auto,
                ParallelToolCallPolicy::Disabled,
                vec![marker],
            )
            .unwrap();
        let mut controller = plan.controller();
        controller.commit(marker).unwrap();
        assert!(!controller.constraint_is_active());
        let space = tokenizer.encode(" ", false).unwrap().get_ids()[0];
        assert!(controller.filter_at(&[marker]).unwrap().allows(space));
        controller.commit(space).unwrap();
        assert!(controller.constraint_is_active());
        for &token in tokenizer
            .encode(r#"[{"name":"ping","arguments":{}}]"#, false)
            .unwrap()
            .get_ids()
        {
            controller.commit(token).unwrap();
        }
        assert!(controller.grammar_is_complete().unwrap());
    }

    fn tool(name: &str, parameters: serde_json::Value) -> serde_json::Value {
        json!({
            "type": "function",
            "function": {"name": name, "parameters": parameters}
        })
    }

    #[test]
    fn nanbeige_atomic_opening_marker_activates_before_whitespace_and_forbids_early_eos() {
        let tokenizer = super::ChatTokenizer::from_tokenizer(
            tokenizers::Tokenizer::from_bytes(include_bytes!(
                "../../../tests/fixtures/chat_templates/nanbeige4.2-0e137298-tokenizer.json"
            ))
            .unwrap(),
        );
        let eos = tokenizer.token_to_id("<|im_end|>").unwrap();
        let marker = tokenizer.token_to_id("<tool_call>").unwrap();
        let compiler = super::fixtures::Compiler::new(&tokenizer, &[eos]);
        let tools = [tool(
            "ping",
            json!({"type":"object", "additionalProperties":false}),
        )];
        for choice in [ToolChoice::Auto, ToolChoice::Required, ToolChoice::None] {
            let plan = compiler
                .compile_tool_plan(
                    &DECLARATIVE_DIALECT,
                    DialectParameters::Declarative(
                        &crate::runtime::chat::QWEN_TAGGED_TOOL_SPEC_NO_REASONING,
                    ),
                    &tools,
                    choice,
                    ParallelToolCallPolicy::Disabled,
                    vec![eos],
                )
                .unwrap();
            let mut controller = plan.controller();
            if choice == ToolChoice::None {
                assert!(!controller.filter_at(&[]).unwrap().allows(marker));
                continue;
            }
            assert!(controller.filter_at(&[]).unwrap().allows(marker));
            controller.commit(marker).unwrap();
            assert!(controller.constraint_is_active());
            let filter = controller.filter_at(&[marker]).unwrap();
            assert!(!filter.allows(eos));
            for whitespace in [" ", "\n", "\r", "\t"] {
                let id = tokenizer
                    .token_to_id(if whitespace == " " { "▁" } else { whitespace })
                    .unwrap_or_else(|| {
                        tokenizer
                            .token_to_id(&format!("<0x{:02X}>", whitespace.as_bytes()[0]))
                            .unwrap()
                    });
                assert!(filter.allows(id), "{whitespace:?}");
            }
        }
    }

    #[test]
    fn qwen_parallel_calls_remain_open_until_eos_or_call_limit() {
        use crate::runtime::chat::{
            QWEN_TAGGED_TOOL_SPEC_NO_REASONING, QWEN_XML_TOOL_SPEC, QWEN3_XML_TOOL_SPEC,
        };

        let json_call = "<tool_call>\n{\"name\":\"ping\",\"arguments\":{}}\n</tool_call>";
        for (spec, call) in [
            (&QWEN_XML_TOOL_SPEC, json_call),
            (&QWEN3_XML_TOOL_SPEC, json_call),
            (
                &QWEN_TAGGED_TOOL_SPEC_NO_REASONING,
                "<tool_call>\n<function=ping>\n</function>\n</tool_call>",
            ),
        ] {
            let first: Vec<u32> = call.bytes().map(u32::from).collect();
            let second: Vec<u32> = format!("\n{call}").bytes().map(u32::from).collect();
            for choice in [ToolChoice::Auto, ToolChoice::Required] {
                for max_calls in [None, NonZeroUsize::new(2)] {
                    let plan = super::fixtures::Compiler::byte_tokens(&[255, 254])
                        .compile_tool_plan(
                            &DECLARATIVE_DIALECT,
                            DialectParameters::Declarative(spec),
                            &[tool(
                                "ping",
                                json!({"type": "object", "additionalProperties": false}),
                            )],
                            choice,
                            ParallelToolCallPolicy::Enabled { max_calls },
                            vec![255],
                        )
                        .unwrap();
                    let mut controller = plan.controller();

                    // Speculative prefixes must make the same termination decision
                    // as the committed controller, without advancing its state.
                    assert!(!controller.prefix_is_complete(&first).unwrap());
                    for &token in &first {
                        controller.commit(token).unwrap();
                    }
                    assert!(!controller.grammar_is_complete().unwrap());
                    let filter = controller.filter_at(&first).unwrap();
                    assert!(filter.allows(255));
                    assert!(filter.allows(u32::from(b'\n')));

                    // EOS may end a parallel request before its call limit.
                    for eos in [255, 254] {
                        let mut ended = controller.clone();
                        ended.commit(eos).unwrap();
                        assert!(ended.grammar_is_complete().unwrap());
                    }

                    let history = [first.as_slice(), second.as_slice()].concat();
                    assert_eq!(
                        controller.prefix_is_complete(&history).unwrap(),
                        max_calls.is_some()
                    );
                    for &token in &second {
                        controller.commit(token).unwrap();
                    }
                    assert_eq!(
                        controller.grammar_is_complete().unwrap(),
                        max_calls.is_some()
                    );
                    if max_calls.is_some() {
                        assert!(
                            !controller
                                .filter_at(&history)
                                .unwrap()
                                .allows(u32::from(b'\n'))
                        );
                    } else {
                        controller.commit(255).unwrap();
                        assert!(controller.grammar_is_complete().unwrap());
                    }
                }
            }
        }
    }

    fn accepts(
        plan: &crate::runtime::chat::GenerationRuntimePlan,
        value: serde_json::Value,
    ) -> bool {
        let bytes = serde_json::to_vec(&value).unwrap();
        let mut state = plan.generation_constraint().grammar_matcher();
        for byte in bytes {
            if state.consume_token(byte as TokenId).is_err() {
                return false;
            }
        }
        state.is_accepting().unwrap() && {
            let mut parser = plan.create_parser().unwrap();
            parser.push(&value.to_string()).is_ok()
                && parser
                    .finish(eredu_core::generation::FinishReason::GrammarComplete)
                    .is_ok()
        }
    }

    #[test]
    fn restricts_function_names_and_supports_required_optional_and_nested_values() {
        let compiler = compiler();
        let plan = compiler
            .compile_tool_plan(
                &DECLARATIVE_DIALECT,
                SYNTHETIC_PARAMETERS,
                &[
                    tool(
                        "lookup",
                        json!({
                            "type": "object",
                            "properties": {
                                "query": {"type": "string"},
                                "options": {
                                    "type": "object",
                                    "properties": {
                                        "limit": {"type": "integer"},
                                        "exact": {"type": "boolean"}
                                    },
                                    "required": ["limit"],
                                    "additionalProperties": false
                                }
                            },
                            "required": ["query"],
                            "additionalProperties": false
                        }),
                    ),
                    tool(
                        "ping",
                        json!({
                            "type": "object",
                            "properties": {},
                            "additionalProperties": false
                        }),
                    ),
                ],
                ToolChoice::Required,
                ParallelToolCallPolicy::Disabled,
                Vec::new(),
            )
            .unwrap();

        assert!(accepts(
            &plan,
            json!({"calls": [{
                "name": "lookup",
                "arguments": {
                    "query": "snowman ☃ and \"quotes\"",
                    "options": {"limit": 3, "exact": true}
                }
            }]})
        ));
        assert!(accepts(
            &plan,
            json!({"calls": [{"name": "lookup", "arguments": {"query": "optional omitted"}}]})
        ));
        assert!(!accepts(
            &plan,
            json!({"calls": [{"name": "unknown", "arguments": {}}]})
        ));
    }

    #[test]
    fn resolves_local_references_and_supports_arrays_enums_and_scalars() {
        let compiler = compiler();
        let plan = compiler
            .compile_tool_plan(
                &DECLARATIVE_DIALECT,
                SYNTHETIC_PARAMETERS,
                &[tool(
                    "batch",
                    json!({
                        "type": "object",
                        "properties": {
                            "items": {
                                "type": "array",
                                "items": {"$ref": "#/$defs/item~1type~0v1"},
                                "minItems": 1,
                                "maxItems": 2
                            },
                            "mode": {"type": "string", "enum": ["fast", "安全"]},
                            "ratio": {"type": "number"},
                            "enabled": {"type": "boolean"},
                            "nothing": {"type": "null"}
                        },
                        "required": ["items", "mode", "ratio", "enabled", "nothing"],
                        "additionalProperties": false,
                        "$defs": {
                            "item/type~v1": {
                                "type": "object",
                                "properties": {"value": {"type": "string"}},
                                "required": ["value"],
                                "additionalProperties": false
                            }
                        }
                    }),
                )],
                ToolChoice::Required,
                ParallelToolCallPolicy::Disabled,
                Vec::new(),
            )
            .unwrap();

        assert!(accepts(
            &plan,
            json!({"calls": [{
                "name": "batch",
                "arguments": {
                    "items": [{"value": "α"}, {"value": "β"}],
                    "mode": "安全",
                    "ratio": 1.5,
                    "enabled": false,
                    "nothing": null
                }
            }]})
        ));
        assert!(!accepts(
            &plan,
            json!({"calls": [{"name": "batch", "arguments": {
                "items": [], "mode": "slow", "ratio": 1, "enabled": true, "nothing": null
            }}]})
        ));
    }

    #[test]
    fn rejects_invalid_tool_envelopes_functions_and_names() {
        let valid_parameters =
            || json!({"type": "object", "properties": {}, "additionalProperties": false});
        let invalid = [
            json!(null),
            json!({}),
            json!({"type": "command", "function": {}}),
            json!({"type": "function", "function": "lookup"}),
            json!({"type": "function", "function": {
                "name": "lookup", "parameters": valid_parameters(), "unknown": true
            }}),
            json!({"type": "function", "function": {
                "name": "", "parameters": valid_parameters()
            }}),
            json!({"type": "function", "function": {
                "name": "contains space", "parameters": valid_parameters()
            }}),
            json!({"type": "function", "function": {
                "name": "slash/name", "parameters": valid_parameters()
            }}),
            json!({"type": "function", "function": {
                "name": "x".repeat(65), "parameters": valid_parameters()
            }}),
            json!({"type": "function", "function": {
                "name": "lookup", "description": 7, "parameters": valid_parameters()
            }}),
            json!({"type": "function", "function": {"name": "lookup"}}),
        ];

        for tool in invalid {
            let error = compiler()
                .compile_tool_plan(
                    &DECLARATIVE_DIALECT,
                    SYNTHETIC_PARAMETERS,
                    std::slice::from_ref(&tool),
                    ToolChoice::Required,
                    ParallelToolCallPolicy::Disabled,
                    Vec::new(),
                )
                .unwrap_err();
            assert!(
                error.contains("tools[0]"),
                "invalid tool {tool} produced an unscoped diagnostic: {error}"
            );
        }

        let duplicate = tool("lookup", valid_parameters());
        let error = compiler()
            .compile_tool_plan(
                &DECLARATIVE_DIALECT,
                SYNTHETIC_PARAMETERS,
                &[duplicate.clone(), duplicate],
                ToolChoice::Required,
                ParallelToolCallPolicy::Disabled,
                Vec::new(),
            )
            .unwrap_err();
        assert!(error.contains("duplicate tool function name"));
    }

    #[test]
    fn rejects_invalid_references_and_malformed_schemas() {
        let compiler = compiler();
        let invalid = [
            tool(
                "missing",
                json!({"type": "object", "properties": {"x": {"$ref": "#/$defs/nope"}}}),
            ),
            tool(
                "external",
                json!({"type": "object", "properties": {"x": {"$ref": "https://example.test/schema"}}}),
            ),
            tool("malformed", json!({"type": "object", "required": "x"})),
            tool(
                "malformed_union",
                json!({"type": "object", "properties": {"x": {"type": ["string", 7]}}}),
            ),
            tool(
                "malformed_minimum",
                json!({"type": "object", "properties": {"x": {"minimum": "zero"}}}),
            ),
        ];
        for tool in invalid {
            let error = compiler
                .compile_tool_plan(
                    &DECLARATIVE_DIALECT,
                    SYNTHETIC_PARAMETERS,
                    &[tool],
                    ToolChoice::Required,
                    ParallelToolCallPolicy::Disabled,
                    Vec::new(),
                )
                .unwrap_err();
            assert!(error.contains("tools[0].function.parameters"), "{error}");
        }
    }

    #[test]
    fn common_constraints_remain_enforced_during_decoding() {
        let plan = compiler()
            .compile_tool_plan(
                &DECLARATIVE_DIALECT,
                SYNTHETIC_PARAMETERS,
                &[tool(
                    "check",
                    json!({"type": "object", "properties": {
                "count": {"type": "integer", "minimum": 2},
                "value": {"oneOf": [{"type": "string"}, {"type": "null"}]}
            }, "required": ["count", "value"], "additionalProperties": false}),
                )],
                ToolChoice::Required,
                ParallelToolCallPolicy::Disabled,
                Vec::new(),
            )
            .unwrap();
        for arguments in [
            json!({"count": 1, "value": null}),
            json!({"count": 2, "value": false}),
        ] {
            let output = json!({"calls": [{"name": "check", "arguments": arguments}]}).to_string();
            let mut grammar = plan.generation_constraint().grammar_matcher();
            assert!(
                output
                    .bytes()
                    .any(|byte| grammar.consume_token(u32::from(byte)).is_err())
            );
        }
    }

    #[test]
    fn each_tool_keeps_its_own_reference_root() {
        let schema = |kind| {
            json!({
                "$defs": {"value": {"type": kind}},
                "properties": {"x": {"$ref": "#/$defs/value"}}, "required": ["x"]
            })
        };
        let plan = compiler()
            .compile_tool_plan(
                &DECLARATIVE_DIALECT,
                SYNTHETIC_PARAMETERS,
                &[
                    tool("text", schema("string")),
                    tool("number", schema("integer")),
                ],
                ToolChoice::Required,
                ParallelToolCallPolicy::Disabled,
                Vec::new(),
            )
            .unwrap();
        for (name, value, valid) in [
            ("text", json!("hi"), true),
            ("number", json!(2), true),
            ("text", json!(2), false),
            ("number", json!("hi"), false),
        ] {
            assert_eq!(
                accepts(
                    &plan,
                    json!({"calls": [{"name": name, "arguments": {"x": value}}]})
                ),
                valid
            );
        }
    }

    #[test]
    fn accepts_application_schemas_and_checks_completed_arguments() {
        let cases = [
            (
                json!({"type": "integer", "minimum": 2, "maximum": 8, "multipleOf": 2}),
                json!(4),
                json!(3),
            ),
            (
                json!({"anyOf": [{"type": "string"}, {"type": "null"}]}),
                json!(null),
                json!(9),
            ),
            (
                json!({"oneOf": [{"type": "string"}, {"type": "number"}]}),
                json!("ok"),
                json!(true),
            ),
            // Overlapping oneOf needs exact completion validation, not anyOf coercion.
            (
                json!({"oneOf": [{"type": "integer"}, {"type": "number", "minimum": 0}]}),
                json!(-1),
                json!(1),
            ),
            (
                json!({"type": ["string", "null"]}),
                json!(null),
                json!(false),
            ),
            (
                json!({"allOf": [{"minimum": 2}, {"maximum": 4}]}),
                json!(3),
                json!(1),
            ),
            (
                json!({"type": "string", "minLength": 2, "maxLength": 4, "pattern": "^[a-z]+$"}),
                json!("abc"),
                json!("ABC"),
            ),
            (
                json!({"const": {"literal": {"$ref": "this is data"}}}),
                json!({"literal": {"$ref": "this is data"}}),
                json!({}),
            ),
            (
                json!({"type": "array", "uniqueItems": true}),
                json!([1, 2]),
                json!([1, 1]),
            ),
            (
                json!({"type": "array", "contains": {"const": 1}}),
                json!([0, 1]),
                json!([0]),
            ),
            (json!({"not": {"type": "null"}}), json!("ok"), json!(null)),
            (
                json!({"if": {"type": "string"}, "then": {"minLength": 2}, "else": {"const": 0}}),
                json!("ok"),
                json!(1),
            ),
            (
                json!({"type": "object", "patternProperties": {"^x": {"type": "integer"}}, "additionalProperties": false}),
                json!({"x1": 1}),
                json!({"x1": "bad"}),
            ),
            (
                json!({"type": "object", "dependentRequired": {"x": ["y"]}}),
                json!({"x": 1, "y": 2}),
                json!({"x": 1}),
            ),
            (
                json!({"type": "object", "properties": {"x": true, "y": false}, "unevaluatedProperties": false}),
                json!({"x": null}),
                json!({"y": 1}),
            ),
            (
                json!({"type": "string", "examples": ["ok"], "default": {"$ref": "not a schema"}, "x-app": {"arbitrary": true}}),
                json!("ok"),
                json!(0),
            ),
        ];
        for (schema, valid, invalid) in cases {
            let plan = compiler().compile_tool_plan(
                &DECLARATIVE_DIALECT, SYNTHETIC_PARAMETERS,
                &[tool("check", json!({"type": "object", "properties": {"value": schema}, "required": ["value"], "additionalProperties": false}))],
                ToolChoice::Required, ParallelToolCallPolicy::Disabled, Vec::new(),
            ).unwrap_or_else(|error| panic!("{schema}: {error}"));
            let call = |value| json!({"calls": [{"name": "check", "arguments": {"value": value}}]});
            assert!(
                accepts(&plan, call(valid)),
                "valid value rejected for {schema}"
            );
            assert!(
                !accepts(&plan, call(invalid)),
                "invalid value accepted for {schema}"
            );
        }
    }

    #[test]
    fn accepts_boolean_typeless_recursive_and_scoped_root_schemas() {
        let cases = [
            (json!(true), json!({"anything": [null, 1]})),
            (json!({}), json!({})),
            (
                json!({"required": ["undeclared"]}),
                json!({"undeclared": true}),
            ),
            (
                json!({"anyOf": [{"required": ["x"]}, {"required": ["y"]}]}),
                json!({"y": 1}),
            ),
            (
                json!({"$defs": {"value": {"type": "integer"}}, "type": "object", "properties": {"x": {"$ref": "#/$defs/value", "minimum": 2}}}),
                json!({"x": 3}),
            ),
            (
                json!({"type": "object", "properties": {"child": {"anyOf": [{"type": "null"}, {"$ref": "#"}]}}}),
                json!({"child": {"child": null}}),
            ),
            (
                json!({"$id": "https://example.test/root", "$defs": {"value": {"$anchor": "value", "type": "integer"}}, "properties": {"x": {"$ref": "#value"}}}),
                json!({"x": 3}),
            ),
        ];
        for (schema, arguments) in cases {
            let plan = compiler()
                .compile_tool_plan(
                    &DECLARATIVE_DIALECT,
                    SYNTHETIC_PARAMETERS,
                    &[tool("check", schema.clone())],
                    ToolChoice::Required,
                    ParallelToolCallPolicy::Disabled,
                    Vec::new(),
                )
                .unwrap_or_else(|error| panic!("{schema}: {error}"));
            assert!(
                accepts(
                    &plan,
                    json!({"calls": [{"name": "check", "arguments": arguments}]})
                ),
                "{schema}"
            );
        }
        let plan = compiler()
            .compile_tool_plan(
                &DECLARATIVE_DIALECT,
                SYNTHETIC_PARAMETERS,
                &[tool("never", json!(false))],
                ToolChoice::Required,
                ParallelToolCallPolicy::Disabled,
                Vec::new(),
            )
            .unwrap();
        assert!(!accepts(
            &plan,
            json!({"calls": [{"name": "never", "arguments": {}}]})
        ));
    }

    #[test]
    fn completion_validation_survives_parser_forks_and_never_ends_invalid_calls() {
        use eredu_core::generation::{FinishReason, SemanticEvent};
        let plan = compiler().compile_tool_plan(
            &DECLARATIVE_DIALECT, SYNTHETIC_PARAMETERS,
            &[tool("check", json!({"properties": {"values": {"type": "array", "uniqueItems": true}}, "required": ["values"]}))],
            ToolChoice::Required, ParallelToolCallPolicy::Disabled, Vec::new(),
        ).unwrap();
        let mut parser = plan.create_parser().unwrap();
        parser
            .push(r#"{"calls":[{"name":"check","arguments":{"values":[1,"#)
            .unwrap();
        parser.take_events();
        let mut fork = parser.fork().unwrap();
        assert!(fork.push("1]}}]}").unwrap_err().contains("do not match"));
        assert!(!fork.events().contains(&SemanticEvent::ToolCallEnd));
        parser.push("2]}}]}").unwrap();
        parser.finish(FinishReason::GrammarComplete).unwrap();
        assert!(parser.events().contains(&SemanticEvent::ToolCallEnd));
    }

    #[test]
    fn enforces_single_and_parallel_call_limits() {
        let compiler = compiler();
        let tools = [tool(
            "ping",
            json!({"type": "object", "properties": {}, "additionalProperties": false}),
        )];
        let single = compiler
            .compile_tool_plan(
                &DECLARATIVE_DIALECT,
                SYNTHETIC_PARAMETERS,
                &tools,
                ToolChoice::Required,
                ParallelToolCallPolicy::Disabled,
                Vec::new(),
            )
            .unwrap();
        let parallel = compiler
            .compile_tool_plan(
                &DECLARATIVE_DIALECT,
                SYNTHETIC_PARAMETERS,
                &tools,
                ToolChoice::Required,
                ParallelToolCallPolicy::Enabled {
                    max_calls: NonZeroUsize::new(2),
                },
                Vec::new(),
            )
            .unwrap();
        let two = json!({"calls": [
            {"name": "ping", "arguments": {}},
            {"name": "ping", "arguments": {}}
        ]});
        let three = json!({"calls": [
            {"name": "ping", "arguments": {}},
            {"name": "ping", "arguments": {}},
            {"name": "ping", "arguments": {}}
        ]});
        assert!(!accepts(&single, two.clone()));
        assert!(accepts(&parallel, two));
        assert!(!accepts(&parallel, three));
    }

    #[test]
    fn grammar_state_forks_commits_and_completes() {
        let compiler = compiler();
        let plan = compiler
            .compile_tool_plan(
                &DECLARATIVE_DIALECT,
                SYNTHETIC_PARAMETERS,
                &[tool(
                    "ping",
                    json!({"type": "object", "properties": {}, "additionalProperties": false}),
                )],
                ToolChoice::Required,
                ParallelToolCallPolicy::Disabled,
                Vec::new(),
            )
            .unwrap();
        let bytes =
            serde_json::to_vec(&json!({"calls": [{"name": "ping", "arguments": {}}]})).unwrap();
        let split = bytes.len() / 2;
        let mut state = plan.generation_constraint().grammar_matcher();
        for byte in &bytes[..split] {
            state.consume_token(*byte as TokenId).unwrap();
        }
        let mut fork = state.deep_clone();
        assert!(!state.compute_mask_or_eos().unwrap().is_empty());
        for byte in &bytes[split..] {
            state.consume_token(*byte as TokenId).unwrap();
        }
        assert!(state.is_accepting().unwrap());
        for byte in &bytes[split..] {
            fork.consume_token(*byte as TokenId).unwrap();
        }
        assert!(fork.is_accepting().unwrap());
    }
}
