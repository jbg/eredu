//! Stock parser templates and independent sessions with retained host headroom.
use crate::runtime::chat::{
    preparation_memory::{PreparationFailure, PreparationFunding, StorageFailure},
    tokenizer_env::bytes::{self, TokenizationMode},
};
use eredu_core::HostMetadataFunding;
use eredu_runtime::working_memory::{
    DependencyMemoryPolicy, OriginalTokenTrieSource, OriginalTokenizerEncodeError,
};
use llguidance::{
    ParserFactory, TokenParser,
    api::{StopReason, TopLevelGrammar},
    toktrie::{SimpleVob, TokEnv, TokTrie, TokenizerEnv},
};
use std::{
    fmt,
    mem::{size_of, size_of_val},
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{Arc, Mutex},
};

#[derive(Clone)]
pub(super) enum TokenizerSource {
    Original(OriginalTokenTrieSource),
    Ordinary(TokEnv),
}
impl TokenizerSource {
    fn trie(&self) -> &TokTrie {
        match self {
            Self::Original(source) => source.trie(),
            Self::Ordinary(source) => source.tok_trie(),
        }
    }
}
#[derive(Debug, thiserror::Error)]
enum Cause {
    #[error(transparent)]
    Funding(#[from] PreparationFailure),
    #[error(transparent)]
    Storage(#[from] StorageFailure),
    #[error(transparent)]
    Encoding(#[from] OriginalTokenizerEncodeError),
    #[error(transparent)]
    Parser(#[from] anyhow::Error),
    #[error("grammar produced unsupported warnings: {0:?}")]
    Warnings(Vec<String>),
    #[error("grammar admission estimate overflow")]
    Overflow,
    #[error("grammar tokenizer error state is poisoned")]
    Poisoned,
    #[error("upstream parser operation panicked")]
    Panicked,
    #[error("grammar requested unsupported token backtracking")]
    Backtracking,
}
/// Diagnostics retain their operation account after an upstream temporary retires.
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
pub(super) struct Error {
    #[source]
    cause: Cause,
    funding: PreparationFunding,
}
impl Error {
    /// Only grammar compilation/diagnostic failures can use the syntax-only
    /// tool fallback. Funding, tokenizer and session failures must propagate.
    pub(super) fn allows_schema_fallback(&self) -> bool {
        matches!(self.cause, Cause::Parser(_) | Cause::Warnings(_))
    }
    fn new(cause: impl Into<Cause>, funding: &PreparationFunding) -> Self {
        Self {
            cause: cause.into(),
            funding: funding.clone(),
        }
    }
}

/// Its error slot belongs to exactly one compiler or independently mutable
/// session. Parser copies receive a fresh environment and never share this slot.
pub(super) struct Environment {
    source: TokenizerSource,
    compilation: Mutex<()>,
    failure: Mutex<Option<Error>>,
    funding: PreparationFunding,
}
impl fmt::Debug for Environment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Environment")
            .field("vocabulary", &self.source.trie().vocab_size())
            .finish_non_exhaustive()
    }
}
impl Environment {
    pub(super) fn new(
        source: TokenizerSource,
        funding: &PreparationFunding,
    ) -> Result<Arc<Self>, Error> {
        funding
            .reserve(
                eredu_nn::workspace::WorkspaceContext::metadata_arc_bytes::<Self>()
                    .ok_or_else(|| Error::new(Cause::Overflow, funding))?,
            )
            .map_err(|e| Error::new(e, funding))?;
        Ok(Arc::new(Self {
            source,
            compilation: Mutex::new(()),
            failure: Mutex::new(None),
            funding: funding.clone(),
        }))
    }
    fn failed(&self) -> bool {
        self.failure.lock().map_or(true, |slot| slot.is_some())
    }
    fn record(&self, error: Error) {
        if let Ok(mut slot) = self.failure.lock() {
            if slot.is_none() {
                *slot = Some(error);
            }
        }
    }
    pub(super) fn take_failure(&self) -> Result<(), Error> {
        let mut slot = self
            .failure
            .lock()
            .map_err(|_| Error::new(Cause::Poisoned, &self.funding))?;
        match slot.take() {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }
    fn original_bytes(
        &self,
        source: &OriginalTokenTrieSource,
        input: &[u8],
        mode: TokenizationMode,
    ) -> Vec<u32> {
        if self.failed() {
            return Vec::new();
        }
        let result = (|| -> Result<Vec<u32>, Error> {
            self.funding
                .reserve_dependency(input.len())
                .map_err(|e| Error::new(e, &self.funding))?;
            let (ids, _) = bytes::tokenize(source.trie(), input, mode, |text| {
                let encoded = source
                    .encode_tokenizer_ids(text)
                    .map_err(|e| Error::new(e, &self.funding))?;
                let mut ids = Vec::new();
                self.funding
                    .try_extend_copy(&mut ids, encoded.ids())
                    .map_err(|e| Error::new(e, &self.funding))?;
                Ok(ids)
            })
            .map_err(|error| error.cause)?;
            Ok(ids)
        })();
        match result {
            Ok(ids) => ids,
            Err(error) => {
                self.record(error);
                Vec::new()
            }
        }
    }
}
impl TokenizerEnv for Environment {
    fn tok_trie(&self) -> &TokTrie {
        self.source.trie()
    }
    fn tokenize_bytes(&self, input: &[u8]) -> Vec<u32> {
        match &self.source {
            TokenizerSource::Original(source) => {
                self.original_bytes(source, input, TokenizationMode::Plain)
            }
            TokenizerSource::Ordinary(source) => source.tokenize_bytes(input),
        }
    }
    fn tokenize_bytes_special(&self, input: &[u8]) -> Vec<u32> {
        match &self.source {
            TokenizerSource::Original(source) => {
                self.original_bytes(source, input, TokenizationMode::Special)
            }
            TokenizerSource::Ordinary(source) => source.tokenize_bytes_special(input),
        }
    }
    fn tokenize_is_canonical(&self) -> bool {
        match &self.source {
            TokenizerSource::Original(source) => source.tokenization_is_canonical(),
            TokenizerSource::Ordinary(source) => source.tokenize_is_canonical(),
        }
    }
}

/// A closed, unstarted parser used only as a template for deep copies. Neither
/// its mutable parser nor a shallow clone can escape this module.
pub(super) struct Template {
    parser: TokenParser,
    environment: Arc<Environment>,
    admission_bytes: usize,
    memory: DependencyMemoryPolicy,
    funding: PreparationFunding,
}
impl fmt::Debug for Template {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Template")
            .field("admission_bytes", &self.admission_bytes)
            .finish_non_exhaustive()
    }
}
impl Template {
    pub(super) fn compile(
        factory: &ParserFactory,
        environment: Arc<Environment>,
        grammar: TopLevelGrammar,
        funding: &PreparationFunding,
    ) -> Result<Self, Error> {
        // A factory can be shared by cold request preparation. Serialize its
        // scoped diagnostic slot; independent live sessions use fresh slots.
        let compilation = environment
            .compilation
            .lock()
            .map_err(|_| Error::new(Cause::Poisoned, funding))?;
        // Serialization counts public input without retaining a second string.
        // Stock compiler, initial lexer and parser construction use this allowance.
        funding
            .reserve_dependency(0)
            .map_err(|e| Error::new(e, funding))?;
        struct Count(usize);
        impl std::io::Write for Count {
            fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
                self.0 = self
                    .0
                    .checked_add(bytes.len())
                    .ok_or(std::io::ErrorKind::FileTooLarge)?;
                Ok(bytes.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let mut count = Count(0);
        serde_json::to_writer(&mut count, &grammar)
            .map_err(|e| Error::new(Cause::Parser(e.into()), funding))?;
        // Sessions share vocabulary/slicer data with their source. Their
        // vocabulary-sized mutable input is a packed mask, not an ID array.
        // Apply the configurable overhead to that logical width and grammar
        // text; this remains an estimate of opaque compiler/session storage.
        let mask_bytes = environment
            .tok_trie()
            .vocab_size()
            .div_ceil(u32::BITS as usize)
            .checked_add(1)
            .and_then(|words| words.checked_mul(size_of::<u32>()))
            .ok_or_else(|| Error::new(Cause::Overflow, funding))?;
        let input_bytes = count
            .0
            .checked_add(mask_bytes)
            .ok_or_else(|| Error::new(Cause::Overflow, funding))?;
        let admission_bytes = funding
            .reserve_dependency(input_bytes)
            .map_err(|e| Error::new(e, funding))?;
        let result = factory.create_parser(grammar);
        environment.take_failure()?;
        let mut parser = result.map_err(|e| Error::new(e, funding))?;
        let warnings = parser.grammar_warnings();
        if !warnings.is_empty() {
            return Err(Error::new(Cause::Warnings(warnings), funding));
        }
        drop(compilation);
        Ok(Self {
            parser,
            environment,
            admission_bytes,
            memory: funding.memory_policy(),
            funding: funding.clone(),
        })
    }
    pub(super) fn memory_policy(&self) -> DependencyMemoryPolicy {
        self.memory
    }
    pub(super) fn admission_bytes(&self) -> usize {
        self.admission_bytes
    }
    pub(super) fn grammar(&self) -> &llguidance::earley::CGrammar {
        self.parser.parser.grammar()
    }
    pub(super) fn matches_source(&self, source: &OriginalTokenTrieSource) -> bool {
        matches!(&self.environment.source, TokenizerSource::Original(actual) if actual.same_source(source))
    }
    pub(super) fn create_session(&self, funding: &HostMetadataFunding) -> Result<Session, Error> {
        let funding = PreparationFunding::from_metadata(funding).with_memory_policy(self.memory);
        funding
            .reserve(self.admission_bytes)
            .map_err(|e| Error::new(e, &funding))?;
        let environment = Environment::new(self.environment.source.clone(), &funding)?;
        let mut parser = catch_unwind(AssertUnwindSafe(|| self.parser.deep_clone()))
            .map_err(|_| Error::new(Cause::Panicked, &funding))?;
        // Token-level forcing uses this public field. Earley and slicer retain
        // the same immutable trie in their environments and do not encode text.
        parser.token_env = environment.clone();
        parser.start_without_prompt();
        Ok(Session {
            parser,
            environment,
            mask: None,
            base_bytes: self.admission_bytes,
            funding,
        })
    }
    #[cfg(test)]
    pub(super) fn fixture_parser(&self) -> Result<TokenParser, Error> {
        self.funding
            .reserve(self.admission_bytes)
            .map_err(|e| Error::new(e, &self.funding))?;
        let environment = Environment::new(self.environment.source.clone(), &self.funding)?;
        let mut parser = self.parser.deep_clone();
        parser.token_env = environment;
        Ok(parser)
    }
}

/// The native-free stock session owns its mutable state, current output mask,
/// tokenizer failure slot, and cumulative host account independently of peers.
pub(super) struct Session {
    parser: TokenParser,
    environment: Arc<Environment>,
    mask: Option<SimpleVob>,
    base_bytes: usize,
    funding: PreparationFunding,
}
impl fmt::Debug for Session {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Session")
            .field("tokens", &self.parser.num_tokens())
            .field("stop", &self.parser.stop_reason())
            .finish_non_exhaustive()
    }
}
impl Session {
    pub(super) fn copy_required_bytes(&self) -> Option<usize> {
        self.base_bytes
            .checked_add(
                self.funding
                    .memory_policy()
                    .estimate(self.parser.final_bytes().len())?,
            )?
            .checked_add(eredu_nn::workspace::WorkspaceContext::metadata_arc_bytes::<
                Environment,
            >()?)
    }
    pub(super) fn try_copy(&self, account: &HostMetadataFunding) -> Result<Self, Error> {
        let funding = PreparationFunding::from_metadata(account)
            .with_memory_policy(self.funding.memory_policy());
        let bytes = self
            .base_bytes
            .checked_add(
                self.funding
                    .memory_policy()
                    .estimate(self.parser.final_bytes().len())
                    .ok_or_else(|| Error::new(Cause::Overflow, &funding))?,
            )
            .ok_or_else(|| Error::new(Cause::Overflow, &funding))?;
        funding
            .reserve(bytes)
            .map_err(|e| Error::new(e, &funding))?;
        let environment = Environment::new(self.environment.source.clone(), &funding)?;
        let mut parser = catch_unwind(AssertUnwindSafe(|| self.parser.deep_clone()))
            .map_err(|_| Error::new(Cause::Panicked, &funding))?;
        parser.token_env = environment.clone();
        Ok(Self {
            parser,
            environment,
            mask: self.mask.clone(),
            base_bytes: self.base_bytes,
            funding,
        })
    }
    fn operation<T>(
        &mut self,
        input_bytes: usize,
        run: impl FnOnce(&mut TokenParser) -> Result<T, Cause>,
    ) -> Result<T, Error> {
        self.funding
            .reserve_dependency(input_bytes)
            .map_err(|e| Error::new(e, &self.funding))?;
        let result = catch_unwind(AssertUnwindSafe(|| run(&mut self.parser)))
            .map_err(|_| Cause::Panicked)
            .and_then(|result| result);
        self.environment.take_failure()?;
        result.map_err(|e| Error::new(e, &self.funding))
    }
    pub(super) fn compute_mask(&mut self) -> Result<(), Error> {
        let mask = self.operation(0, |parser| {
            if parser.stop_reason() != StopReason::NotStopped && parser.stop_reason().is_ok() {
                Ok(parser.token_env.tok_trie().eos_token_set())
            } else {
                Ok(parser.compute_mask()?)
            }
        })?;
        self.mask = Some(mask);
        Ok(())
    }
    pub(super) fn token_mask(&self) -> Option<&SimpleVob> {
        self.mask.as_ref()
    }
    pub(super) fn num_tokens(&self) -> usize {
        self.parser.num_tokens()
    }
    #[cfg(test)]
    pub(super) fn final_bytes(&self) -> &[u8] {
        self.parser.final_bytes()
    }
    pub(super) fn stop_reason(&self) -> StopReason {
        self.parser.stop_reason()
    }
    pub(super) fn is_accepting(&mut self) -> Result<bool, Error> {
        self.operation(0, |parser| Ok(parser.is_accepting()))
    }
    pub(super) fn try_consume_tokens(&mut self, tokens: &[u32]) -> Result<usize, Error> {
        let bytes = size_of_val(tokens);
        let count = self.operation(bytes, |parser| {
            let mut consumed = 0;
            for &token in tokens {
                if !parser.validate_token(token)? {
                    break;
                }
                if parser.consume_token(token)? != 0 {
                    return Err(Cause::Backtracking);
                }
                consumed += 1;
                parser.check_stop()?;
            }
            Ok(consumed)
        })?;
        self.mask = None;
        Ok(count)
    }
    pub(super) fn rollback(&mut self, tokens: usize) -> Result<(), Error> {
        self.operation(0, |parser| Ok(parser.rollback(tokens)?))?;
        self.mask = None;
        Ok(())
    }
    pub(super) fn reset(&mut self) -> Result<(), Error> {
        self.operation(0, |parser| Ok(parser.reset()?))?;
        self.mask = None;
        Ok(())
    }
    #[cfg(test)]
    pub(super) fn force_tokens(&mut self) -> Result<Vec<u32>, Error> {
        self.operation(0, |parser| Ok(parser.compute_ff_tokens()))
    }
}

#[cfg(test)]
#[path = "stock_parser/tests.rs"]
mod tests;
