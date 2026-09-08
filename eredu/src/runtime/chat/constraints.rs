//! Private constrained-decoding implementation for native tool plans.

#[cfg(test)]
use std::sync::atomic::{AtomicUsize, Ordering};
use std::{num::NonZeroUsize, sync::Arc};

use eredu_core::{SpeculativeTokenFilterController, TokenFilter, TokenFilterController};
use eredu_text::tokenizer::Tokenizer as ChatTokenizer;
use llguidance::{
    toktrie::{SimpleVob, TokEnv, TokenId},
    Matcher, ParserFactory,
};
use serde_json::{json, Map, Value};

pub(crate) use super::tool_schema::parse_tools;
use sha2::{Digest, Sha256};

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
    committed_tokens: Vec<u32>,
    validity: Arc<TokenFilter>,
}

enum ConstraintRuntime {
    Text,
    Forbidden {
        vocabulary: Arc<Vec<Vec<u8>>>,
        trigger: Vec<u8>,
        pending: Vec<u8>,
    },
    Auto {
        grammar: GrammarState,
        vocabulary: Arc<Vec<Vec<u8>>>,
        trigger: Vec<u8>,
        pending: Vec<u8>,
    },
    Active(GrammarState),
}

impl Clone for ConstraintRuntime {
    fn clone(&self) -> Self {
        match self {
            Self::Text => Self::Text,
            Self::Forbidden {
                vocabulary,
                trigger,
                pending,
            } => Self::Forbidden {
                vocabulary: Arc::clone(vocabulary),
                trigger: trigger.clone(),
                pending: pending.clone(),
            },
            Self::Auto {
                grammar,
                vocabulary,
                trigger,
                pending,
            } => Self::Auto {
                grammar: grammar.fork(),
                vocabulary: Arc::clone(vocabulary),
                trigger: trigger.clone(),
                pending: pending.clone(),
            },
            Self::Active(grammar) => Self::Active(grammar.fork()),
        }
    }
}

impl Clone for ConstraintController {
    fn clone(&self) -> Self {
        Self {
            runtime: self.runtime.clone(),
            committed_tokens: self.committed_tokens.clone(),
            validity: Arc::clone(&self.validity),
        }
    }
}

impl ConstraintController {
    /// Grammar-free generation still retains the exact tokenizer-valid domain.
    pub(crate) fn text(validity: Arc<TokenFilter>) -> Self {
        Self {
            runtime: ConstraintRuntime::Text,
            committed_tokens: Vec::new(),
            validity,
        }
    }

    /// Retains the facade's immutable tokenizer domain across grammar forks,
    /// speculative histories and execution-control snapshots.
    pub(crate) fn with_validity(mut self, validity: Arc<TokenFilter>) -> Self {
        self.validity = validity;
        self
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
            ConstraintRuntime::Text => predictions.checked_mul(4),
            ConstraintRuntime::Forbidden { trigger, .. } => predictions
                .checked_mul(4)?
                .checked_add(trigger.len() as u64),
            ConstraintRuntime::Auto { .. } | ConstraintRuntime::Active(_) => None,
        }
    }
    /// Creates canonical constraint state from one prepared-chat plan.
    pub(crate) fn from_generation_plan(
        plan: &GenerationRuntimePlan,
    ) -> Result<Self, ConstraintError> {
        let constraint = plan.generation_constraint().clone();
        let runtime = if !plan.has_tool_surface() {
            ConstraintRuntime::Active(constraint.grammar_state())
        } else {
            match plan.tool_choice() {
                ToolChoice::None => {
                    let trigger =
                        required_tool_call_trigger(plan.tool_call_trigger(), "forbidden")?;
                    let vocabulary = constraint
                        .grammar_state()
                        .token_vocabulary()
                        .map_err(constraint_error)?;
                    ConstraintRuntime::Forbidden {
                        vocabulary: Arc::new(vocabulary),
                        trigger,
                        pending: Vec::new(),
                    }
                }
                ToolChoice::Auto => {
                    let trigger =
                        required_tool_call_trigger(plan.auto_activation_trigger(), "automatic")?;
                    let grammar = constraint.grammar_state();
                    let vocabulary = grammar.token_vocabulary().map_err(constraint_error)?;
                    ConstraintRuntime::Auto {
                        grammar,
                        vocabulary: Arc::new(vocabulary),
                        trigger,
                        pending: Vec::new(),
                    }
                }
                ToolChoice::Required => ConstraintRuntime::Active(constraint.grammar_state()),
            }
        };
        Ok(Self {
            runtime,
            committed_tokens: Vec::new(),
            validity: Arc::new(TokenFilter::All),
        })
    }

    #[cfg(test)]
    pub(crate) fn constraint_is_active(&self) -> bool {
        matches!(self.runtime, ConstraintRuntime::Active(_))
    }

    pub(crate) fn grammar_is_complete(&mut self) -> Result<bool, ConstraintError> {
        match &mut self.runtime {
            ConstraintRuntime::Active(grammar) => grammar.is_complete().map_err(constraint_error),
            ConstraintRuntime::Text
            | ConstraintRuntime::Forbidden { .. }
            | ConstraintRuntime::Auto { .. } => Ok(false),
        }
    }

    pub(crate) fn prefix_is_complete(&self, history: &[u32]) -> Result<bool, ConstraintError> {
        match &mut self.runtime_at(history)? {
            ConstraintRuntime::Active(grammar) => grammar.is_complete().map_err(constraint_error),
            ConstraintRuntime::Text
            | ConstraintRuntime::Forbidden { .. }
            | ConstraintRuntime::Auto { .. } => Ok(false),
        }
    }

    fn runtime_at(&self, history: &[u32]) -> Result<ConstraintRuntime, ConstraintError> {
        if !history.starts_with(&self.committed_tokens) {
            return Err(ConstraintError::new(
                "constrained sampler history diverges from its committed logical prefix",
            ));
        }
        let mut runtime = self.runtime.clone();
        for &token in &history[self.committed_tokens.len()..] {
            self.validate_token(token)?;
            commit_runtime_token(&mut runtime, token)?;
        }
        Ok(runtime)
    }

    pub(crate) fn filter_at(&self, history: &[u32]) -> Result<TokenFilter, ConstraintError> {
        self.restrict(token_filter_at_runtime(&mut self.runtime_at(history)?)?)
    }

    pub(crate) fn commit(&mut self, token: u32) -> Result<(), ConstraintError> {
        self.validate_token(token)?;
        commit_runtime_token(&mut self.runtime, token)?;
        self.committed_tokens.push(token);
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn valid_token_ids(&mut self) -> Result<Option<Vec<u32>>, ConstraintError> {
        match &mut self.runtime {
            ConstraintRuntime::Active(grammar) => grammar
                .allowed_tokens()
                .map(|mask| Some(mask.iter().collect()))
                .map_err(constraint_error),
            ConstraintRuntime::Text
            | ConstraintRuntime::Forbidden { .. }
            | ConstraintRuntime::Auto { .. } => Ok(None),
        }
    }
}

impl TokenFilterController for ConstraintController {
    type Error = ConstraintError;

    fn current_filter(&mut self) -> Result<TokenFilter, Self::Error> {
        let filter = token_filter_at_runtime(&mut self.runtime)?;
        self.restrict(filter)
    }

    fn commit_token(&mut self, token_id: u32) -> Result<(), Self::Error> {
        self.commit(token_id)
    }

    fn is_complete(&mut self) -> Result<bool, Self::Error> {
        self.grammar_is_complete()
    }
}

impl eredu_runtime::execution_control::SnapshotTokenController for ConstraintController {
    fn snapshot_storage_bytes(&self) -> Option<u64> {
        let Self {
            runtime,
            committed_tokens,
            validity: _, // Immutable Arc data is shared, not copied by snapshots.
        } = self;
        let bytes = (std::mem::size_of::<Self>() as u64)
            .checked_add((committed_tokens.len() as u64).checked_mul(4)?)?;
        match runtime {
            ConstraintRuntime::Text => Some(bytes),
            // Immutable token bytes are already retained by Arc. Only the trigger
            // matcher and committed canonical history are copied for this mode.
            ConstraintRuntime::Forbidden {
                vocabulary: _,
                trigger,
                pending,
            } => bytes
                .checked_add(trigger.len() as u64)?
                .checked_add(pending.len() as u64),
            // llguidance 1.8.0 supports independent deep cloning, but exposes no
            // complete live Matcher storage estimate. Regex-table byte estimates
            // and per-step counters omit parser buffers and caches.
            ConstraintRuntime::Auto { .. } | ConstraintRuntime::Active(_) => None,
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
    fn filter_at(&self, history: &[u32]) -> Result<TokenFilter, Self::Error> {
        ConstraintController::filter_at(self, history)
    }

    fn prefix_is_complete(&self, history: &[u32]) -> Result<bool, Self::Error> {
        ConstraintController::prefix_is_complete(self, history)
    }
}

fn token_filter_at_runtime(
    runtime: &mut ConstraintRuntime,
) -> Result<TokenFilter, ConstraintError> {
    let allowed = match runtime {
        // This grammar filter is intersected with baseline validity by the controller.
        ConstraintRuntime::Text => return Ok(TokenFilter::All),
        ConstraintRuntime::Active(grammar) => {
            let allowed = grammar.allowed_tokens().map_err(constraint_error)?;
            (0..allowed.len())
                .map(|token| allowed.is_allowed(token as u32))
                .collect::<Vec<_>>()
        }
        ConstraintRuntime::Forbidden {
            vocabulary,
            trigger,
            pending,
        } => vocabulary
            .iter()
            .map(|bytes| !completes_trigger(pending, bytes, trigger))
            .collect(),
        ConstraintRuntime::Auto {
            grammar,
            vocabulary,
            trigger,
            pending,
        } => vocabulary
            .iter()
            .enumerate()
            .map(|(token, bytes)| {
                let Some(activation) = trigger_activation_bytes(pending, bytes, trigger) else {
                    return Ok(true);
                };
                let mut candidate = grammar.fork();
                if activation.starts_at_token_boundary {
                    candidate.try_commit(token as u32)
                } else {
                    candidate.try_commit_bytes(&activation.bytes)
                }
                .map_err(constraint_error)
            })
            .collect::<Result<Vec<_>, ConstraintError>>()?,
    };
    TokenFilter::allowed(allowed).map_err(|error| ConstraintError::new(error.to_string()))
}

fn commit_runtime_token(
    runtime: &mut ConstraintRuntime,
    token: u32,
) -> Result<(), ConstraintError> {
    match runtime {
        ConstraintRuntime::Text => Ok(()),
        ConstraintRuntime::Forbidden {
            vocabulary,
            trigger,
            pending,
        } => {
            let bytes = vocabulary.get(token as usize).ok_or_else(|| {
                ConstraintError::new(format!(
                    "token {token} is outside constraint vocabulary {}",
                    vocabulary.len()
                ))
            })?;
            if completes_trigger(pending, bytes, trigger) {
                return Err(ConstraintError::new(
                    "token would emit a tool-call activation trigger while tool_choice is None",
                ));
            }
            advance_trigger_prefix(pending, bytes, trigger);
            Ok(())
        }
        ConstraintRuntime::Active(grammar) => grammar.commit(token).map_err(constraint_error),
        ConstraintRuntime::Auto {
            grammar,
            vocabulary,
            trigger,
            pending,
        } => {
            let bytes = vocabulary.get(token as usize).ok_or_else(|| {
                ConstraintError::new(format!(
                    "token {token} is outside constraint vocabulary {}",
                    vocabulary.len()
                ))
            })?;
            if let Some(activation) = trigger_activation_bytes(pending, bytes, trigger) {
                let mut active = grammar.fork();
                let valid = if activation.starts_at_token_boundary {
                    active.try_commit(token)
                } else {
                    active.try_commit_bytes(&activation.bytes)
                };
                if !valid.map_err(constraint_error)? {
                    return Err(ConstraintError::new(
                        "token crosses the tool-call activation boundary with bytes that are not allowed by the tool grammar",
                    ));
                }
                *runtime = ConstraintRuntime::Active(active);
            } else {
                advance_trigger_prefix(pending, bytes, trigger);
            }
            Ok(())
        }
    }
}

fn required_tool_call_trigger(
    trigger: Option<&str>,
    mode: &str,
) -> Result<Vec<u8>, ConstraintError> {
    let trigger = trigger.ok_or_else(|| {
        ConstraintError::new(format!(
            "{mode} tool-call sampling requires an exact activation trigger"
        ))
    })?;
    if trigger.is_empty() {
        return Err(ConstraintError::new(format!(
            "{mode} tool-call sampling requires a non-empty activation trigger"
        )));
    }
    Ok(trigger.as_bytes().to_vec())
}

pub(crate) fn completes_trigger(pending: &[u8], bytes: &[u8], trigger: &[u8]) -> bool {
    trigger_activation_bytes(pending, bytes, trigger).is_some()
}

struct TriggerActivation {
    bytes: Vec<u8>,
    starts_at_token_boundary: bool,
}

fn trigger_activation_bytes(
    pending: &[u8],
    bytes: &[u8],
    trigger: &[u8],
) -> Option<TriggerActivation> {
    for start in 0..pending.len() {
        let pending_suffix = &pending[start..];
        if pending_suffix.len() + bytes.len() < trigger.len()
            || !pending_suffix
                .iter()
                .chain(bytes)
                .take(trigger.len())
                .eq(trigger)
        {
            continue;
        }
        let trigger_bytes_in_token = trigger.len() - pending_suffix.len();
        let mut activation = Vec::with_capacity(trigger.len() + bytes.len());
        activation.extend_from_slice(trigger);
        activation.extend_from_slice(&bytes[trigger_bytes_in_token..]);
        return Some(TriggerActivation {
            bytes: activation,
            starts_at_token_boundary: false,
        });
    }
    let start = bytes
        .windows(trigger.len())
        .position(|window| window == trigger)?;
    Some(TriggerActivation {
        bytes: bytes[start..].to_vec(),
        starts_at_token_boundary: start == 0,
    })
}

pub(crate) fn advance_trigger_prefix(pending: &mut Vec<u8>, bytes: &[u8], trigger: &[u8]) {
    pending.extend_from_slice(bytes);
    let keep = (0..=pending.len().min(trigger.len().saturating_sub(1)))
        .rev()
        .find(|&length| pending.ends_with(&trigger[..length]))
        .unwrap_or(0);
    let discard = pending.len() - keep;
    pending.drain(..discard);
}

fn constraint_error(error: String) -> ConstraintError {
    ConstraintError::new(error)
}

/// Tokenizer-wide llguidance data. A `LoadedModel` constructs exactly one and
/// every request grammar shares it.
pub(crate) struct ConstraintCompiler {
    factory: Arc<ParserFactory>,
    eos_token_ids: Vec<u32>,
    #[cfg(test)]
    tokenizer_analysis_runs: usize,
    #[cfg(test)]
    schema_compilation_runs: AtomicUsize,
}

pub(crate) struct ConstraintBlueprint {
    matcher: Matcher,
}

pub(crate) struct GrammarState {
    matcher: Matcher,
    terminal_eos_alias_committed: bool,
}

impl ConstraintCompiler {
    pub(crate) fn from_tokenizer(
        tokenizer: &ChatTokenizer,
        eos_token_ids: &[u32],
    ) -> Result<Self, String> {
        let token_env = super::tokenizer_env::from_tokenizer(tokenizer, eos_token_ids)?;
        Self::from_tok_env(token_env, eos_token_ids.to_vec())
    }

    fn from_tok_env(token_env: TokEnv, eos_token_ids: Vec<u32>) -> Result<Self, String> {
        let mut factory = ParserFactory::new_simple(&token_env)
            .map_err(|error| format!("failed to analyze tokenizer trie: {error}"))?;
        factory.quiet();
        Ok(Self {
            factory: Arc::new(factory),
            eos_token_ids,
            #[cfg(test)]
            tokenizer_analysis_runs: 1,
            #[cfg(test)]
            schema_compilation_runs: AtomicUsize::new(0),
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
    ) -> Result<GenerationRuntimePlan, String> {
        #[cfg(test)]
        self.schema_compilation_runs.fetch_add(1, Ordering::Relaxed);

        let grammar_structural_token_spellings = dialect.required_structural_tokens(parameters)?;
        if runtime_structural_token_spellings.len() != resolved_structural_token_ids.len()
            || runtime_structural_token_spellings.len() < grammar_structural_token_spellings.len()
            || !runtime_structural_token_spellings
                .iter()
                .zip(grammar_structural_token_spellings)
                .all(|(runtime, grammar)| runtime == grammar)
        {
            return Err(format!(
                "format dialect declares {} leading structural tokens but {} runtime spellings and {} tokenizer IDs were resolved",
                grammar_structural_token_spellings.len(),
                runtime_structural_token_spellings.len(),
                resolved_structural_token_ids.len()
            ));
        }
        let grammar_structural_token_ids =
            &resolved_structural_token_ids[..grammar_structural_token_spellings.len()];
        dialect.incremental_parser_state_with_tools(parameters, tools)?;
        // Schema admission is independent of the grammar engine's supported subset.
        // Every completed call is checked against the original schema by the sink.
        parse_tools(tools)?;
        let configuration = if tool_surface {
            let configuration = dialect.constraint_configuration(
                parameters,
                tools,
                tool_choice,
                parallel_tool_calls,
                grammar_structural_token_ids,
            )?;
            match self.compile_matcher(configuration.grammar.clone()) {
                Ok(matcher) => (configuration, matcher),
                Err(_) => {
                    // Keep protocol, function names and call limits constrained.
                    // Only argument-schema enforcement moves to completion.
                    let syntax_tools = tools
                        .iter()
                        .map(|tool| {
                            let mut tool = tool.clone();
                            tool["function"]["parameters"] = json!({"type": "object"});
                            tool
                        })
                        .collect::<Vec<_>>();
                    let configuration = dialect.constraint_configuration(
                        parameters,
                        &syntax_tools,
                        tool_choice,
                        parallel_tool_calls,
                        grammar_structural_token_ids,
                    )?;
                    let matcher = self.compile_matcher(configuration.grammar.clone())?;
                    (configuration, matcher)
                }
            }
        } else {
            let configuration = dialect.semantic_constraint_configuration(
                parameters,
                grammar_structural_token_ids,
                &self.eos_token_ids,
            )?;
            let matcher = self.compile_matcher(configuration.grammar.clone())?;
            (configuration, matcher)
        };
        let (configuration, matcher) = configuration;
        let fingerprint: [u8; 32] = Sha256::digest(
            serde_json::to_vec(&(&configuration.grammar, tools))
                .expect("grammar configuration serializes"),
        )
        .into();
        Ok(GenerationRuntimePlan::new(GenerationRuntimePlanParts {
            tool_choice,
            tool_surface,
            generation_constraint: GenerationConstraint::new(
                fingerprint,
                ConstraintBlueprint { matcher },
            ),
            tool_call_trigger: if matches!(tool_choice, ToolChoice::None | ToolChoice::Auto) {
                dialect
                    .auto_activation_trigger(parameters)?
                    .map(str::to_owned)
            } else {
                None
            },
            dialect,
            dialect_parameters: parameters,
            tools: tools.to_vec(),
            structural_token_spellings: runtime_structural_token_spellings,
            resolved_structural_token_ids,
            profile_stop_sequences: runtime_stop_sequences,
        }))
    }

    fn compile_matcher(
        &self,
        grammar: llguidance::api::TopLevelGrammar,
    ) -> Result<Matcher, String> {
        let parser = self
            .factory
            .create_parser(grammar)
            .map_err(|error| format!("failed to compile tool grammar: {error}"))?;
        let mut matcher = Matcher::new(Ok(parser));
        if let Some(error) = matcher.get_error() {
            return Err(format!("failed to compile tool grammar: {error}"));
        }
        let warnings = matcher.grammar_warnings();
        if !warnings.is_empty() {
            return Err(format!(
                "tool grammar produced unsupported warnings: {}",
                warnings.join("; ")
            ));
        }
        Ok(matcher)
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
    fn state(&self) -> GrammarState {
        GrammarState {
            matcher: self.matcher.deep_clone(),
            terminal_eos_alias_committed: false,
        }
    }
}

impl GrammarState {
    pub(crate) fn fork(&self) -> Self {
        Self {
            matcher: self.matcher.deep_clone(),
            terminal_eos_alias_committed: self.terminal_eos_alias_committed,
        }
    }

    pub(crate) fn allowed_tokens(&mut self) -> Result<SimpleVob, String> {
        self.matcher
            .compute_mask_or_eos()
            .map_err(|error| format!("failed to compute grammar token mask: {error}"))
    }

    pub(crate) fn commit(&mut self, token: TokenId) -> Result<(), String> {
        if self.terminal_eos_alias_committed {
            return Err("generation grammar already committed its terminal EOS token".into());
        }
        if self.try_commit(token)? {
            return Ok(());
        }

        // llguidance treats the first configured EOS token as canonical, but
        // secondary EOS aliases cannot be consumed as explicit structural
        // literals. Checkpoint protocols can legitimately use such an alias
        // as their required turn terminator (Muse ATEM's <|eot|> is one).
        // Preserve mask/commit consistency by accepting an EOS alias only
        // when the grammar mask allowed that exact token at this prefix.
        if self.is_eos_token(token)? && self.allowed_tokens()?.is_allowed(token) {
            self.terminal_eos_alias_committed = true;
            return Ok(());
        }

        Err(format!(
            "token {token} is not allowed by the generation grammar"
        ))
    }

    pub(crate) fn try_commit(&mut self, token: TokenId) -> Result<bool, String> {
        let consumed = self
            .matcher
            .try_consume_tokens(&[token])
            .map_err(|error| format!("failed to commit grammar token: {error}"))?;
        Ok(consumed == 1)
    }

    pub(crate) fn try_commit_bytes(&mut self, bytes: &[u8]) -> Result<bool, String> {
        let token_env = self
            .matcher
            .tok_env()
            .map_err(|error| format!("failed to inspect grammar tokenizer: {error}"))?;
        let tokens = token_env.tokenize_bytes(bytes);
        let trie = token_env.tok_trie();
        let round_trip = tokens
            .iter()
            .flat_map(|&token| trie.token(token).iter().copied())
            .collect::<Vec<_>>();
        if round_trip != bytes {
            return Err("grammar tokenizer could not represent activation bytes exactly".into());
        }
        let consumed = self
            .matcher
            .try_consume_tokens(&tokens)
            .map_err(|error| format!("failed to commit grammar bytes: {error}"))?;
        Ok(consumed == tokens.len())
    }

    pub(crate) fn is_complete(&mut self) -> Result<bool, String> {
        if self.terminal_eos_alias_committed {
            return Ok(true);
        }
        self.matcher
            .is_accepting()
            .map_err(|error| format!("failed to inspect grammar completion: {error}"))
    }

    fn is_eos_token(&self, token: TokenId) -> Result<bool, String> {
        let token_env = self
            .matcher
            .tok_env()
            .map_err(|error| format!("failed to inspect grammar tokenizer: {error}"))?;
        Ok(token_env.tok_trie().eos_tokens().contains(&token))
    }

    pub(crate) fn token_vocabulary(&self) -> Result<Vec<Vec<u8>>, String> {
        let token_env = self
            .matcher
            .tok_env()
            .map_err(|error| format!("failed to inspect grammar tokenizer: {error}"))?;
        let trie = token_env.tok_trie();
        Ok((0..trie.vocab_size() as TokenId)
            .map(|token| trie.token(token).to_vec())
            .collect())
    }
}

impl GenerationConstraint {
    pub(crate) fn new(fingerprint: [u8; 32], inner: ConstraintBlueprint) -> Self {
        Self {
            fingerprint,
            inner: Arc::new(inner),
        }
    }

    pub(crate) fn grammar_state(&self) -> GrammarState {
        self.inner.state()
    }
}

pub(crate) fn tool_call_bounds(
    tool_choice: ToolChoice,
    parallel_tool_calls: ParallelToolCallPolicy,
    tools: &[Value],
) -> Result<(usize, Option<usize>), String> {
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

pub(crate) fn tool_call_schema(
    tools: &[Value],
    name_field: &str,
    arguments_field: &str,
    call_id: Option<DeclarativeCallId>,
) -> Result<Value, String> {
    let tools = parse_tools(tools)?;
    let tool_count = tools.len();
    let item_schema = if tools.is_empty() {
        json!({"type": "null"})
    } else {
        let alternatives = tools
            .into_iter()
            .enumerate()
            .map(|(index, tool)| {
                let mut properties = Map::from_iter([
                    (
                        name_field.to_owned(),
                        json!({"type": "string", "enum": [tool.name]}),
                    ),
                    (
                        arguments_field.to_owned(),
                        crate::runtime::chat::tool_schema::arguments_schema(
                            &tool.parameters,
                            &format!(
                                "{}/properties/{}",
                                if tool_count == 1 {
                                    String::new()
                                } else {
                                    format!("/oneOf/{index}")
                                },
                                arguments_field.replace('~', "~0").replace('/', "~1")
                            ),
                        )?,
                    ),
                ]);
                let mut required = vec![
                    Value::String(name_field.to_owned()),
                    Value::String(arguments_field.to_owned()),
                ];
                if let Some(call_id) = call_id {
                    let mut id_schema =
                        Map::from_iter([("type".to_owned(), Value::String("string".to_owned()))]);
                    if let Some(length) = call_id.length {
                        id_schema.insert("minLength".to_owned(), json!(length));
                        id_schema.insert("maxLength".to_owned(), json!(length));
                    }
                    properties.insert(call_id.field.to_owned(), Value::Object(id_schema));
                    required.push(Value::String(call_id.field.to_owned()));
                }
                Ok(json!({
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": false,
                }))
            })
            .collect::<Result<Vec<_>, String>>()?;
        if alternatives.len() == 1 {
            alternatives.into_iter().next().expect("one alternative")
        } else {
            json!({"oneOf": alternatives})
        }
    };

    Ok(item_schema)
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use llguidance::toktrie::TokenId;
    use serde_json::json;

    use super::{ConstraintCompiler, ParallelToolCallPolicy, ToolChoice};
    use crate::runtime::chat::dialect::{
        DeclarativeDialectSpec, DeclarativePayloadShape, DialectParameters, ExactEnvelope,
        GenerationPromptBehavior, JsonFunctionEnvelope, ParallelCallLayout, DECLARATIVE_DIALECT,
    };

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

    fn tool(name: &str, parameters: serde_json::Value) -> serde_json::Value {
        json!({
            "type": "function",
            "function": {"name": name, "parameters": parameters}
        })
    }

    fn accepts(
        plan: &crate::runtime::chat::GenerationRuntimePlan,
        value: serde_json::Value,
    ) -> bool {
        let bytes = serde_json::to_vec(&value).unwrap();
        let mut state = plan.generation_constraint().grammar_state();
        for byte in bytes {
            if state.commit(byte as TokenId).is_err() {
                return false;
            }
        }
        state.is_complete().unwrap() && {
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
            let mut grammar = plan.generation_constraint().grammar_state();
            assert!(output
                .bytes()
                .any(|byte| grammar.commit(u32::from(byte)).is_err()));
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
        let mut state = plan.generation_constraint().grammar_state();
        for byte in &bytes[..split] {
            state.commit(*byte as TokenId).unwrap();
        }
        let mut fork = state.fork();
        assert!(!state.allowed_tokens().unwrap().is_empty());
        for byte in &bytes[split..] {
            state.commit(*byte as TokenId).unwrap();
        }
        assert!(state.is_complete().unwrap());
        for byte in &bytes[split..] {
            fork.commit(*byte as TokenId).unwrap();
        }
        assert!(fork.is_complete().unwrap());
    }
}
