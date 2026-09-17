pub(crate) mod rollback;
pub mod forcing;
pub(crate) mod progress;
use std::{fmt::Display, hint::black_box, panic::AssertUnwindSafe, sync::Arc, time::Duration};

use crate::{
    api::{GrammarInit, ParserLimits, StopReason},
    earley::{BiasComputer, Parser, ParserError, ParserStats},
    infoln, panic_utils, warn, Instant, Logger, ParserFactory,
};
use anyhow::{ensure, Result};
use toktrie::{InferenceCapabilities, SimpleVob, TokEnv, TokenId, INVALID_TOKEN};

/// Token-level parser that drives a single constrained-generation session.
///
/// Created by [`ParserFactory::create_parser()`] and typically wrapped in a
/// [`crate::Constraint`] for the sampling loop.  Maintains the grammar state,
/// computes token masks, and processes sampled tokens.
pub struct TokenParser {
    pub token_env: TokEnv,
    pub parser: Parser,
    pub compute_mask_start_time: Instant,
    pub last_bias_time: Duration,
    pub inference_caps: InferenceCapabilities,
    pub logger: Logger,
    pub limits: ParserLimits,
    pub bias_computer: Arc<dyn BiasComputer>,
    pub dbg_grammar: String,
    last_step_stats: ParserStats,
    max_step_stats: ParserStats,
    eos_tokens: Vec<TokenId>,

    had_rollback: bool,
    had_backtrack: bool,

    is_accepting_cache: Option<bool>,
    ff_tokens_cache: Option<(Vec<TokenId>, Vec<u8>)>,
    stop_reason: StopReason,
    error_message: Option<String>,
    max_tokens_total: usize,

    // tokens currently in KV cache
    llm_tokens: Vec<TokenId>,
    llm_bytes: Vec<u8>,

    grm_prefix: Vec<u8>,
    is_fresh: bool,
}

impl Clone for TokenParser {
    fn clone(&self) -> Self {
        self.clone_with_parser(self.parser.clone())
    }
}

impl TokenParser {
    // Construct around the chosen parser copy once. Ordinary clones share its
    // lexer; deep clones supply an independently copied lexer and parser state.
    fn clone_with_parser(&self, parser: Parser) -> Self {
        Self {
            token_env: self.token_env.clone(),
            parser,
            compute_mask_start_time: self.compute_mask_start_time,
            last_bias_time: self.last_bias_time,
            inference_caps: self.inference_caps.clone(),
            logger: self.logger.clone(),
            limits: self.limits.clone(),
            bias_computer: self.bias_computer.clone(),
            dbg_grammar: self.dbg_grammar.clone(),
            last_step_stats: self.last_step_stats.clone(),
            max_step_stats: self.max_step_stats.clone(),
            eos_tokens: self.eos_tokens.clone(),
            had_rollback: self.had_rollback,
            had_backtrack: self.had_backtrack,
            is_accepting_cache: self.is_accepting_cache,
            ff_tokens_cache: self.ff_tokens_cache.clone(),
            stop_reason: self.stop_reason,
            error_message: self.error_message.clone(),
            max_tokens_total: self.max_tokens_total,
            llm_tokens: self.llm_tokens.clone(),
            llm_bytes: self.llm_bytes.clone(),
            grm_prefix: self.grm_prefix.clone(),
            is_fresh: self.is_fresh,
        }
    }

    // use ParserFactory externally
    pub(crate) fn from_init(
        factory: &ParserFactory,
        grammar_init: GrammarInit,
        logger: Logger,
        inference_caps: InferenceCapabilities,
        limits: ParserLimits,
    ) -> Result<Self> {
        panic_utils::catch_unwind(AssertUnwindSafe(|| {
            Self::init_inner(factory, grammar_init, logger, inference_caps, limits)
        }))
    }

    fn init_inner(
        factory: &ParserFactory,
        grammar_init: GrammarInit,
        mut logger: Logger,
        inference_caps: InferenceCapabilities,
        limits: ParserLimits,
    ) -> Result<Self> {
        let token_env = factory.tok_env().clone();
        ensure!(
            token_env.tokenize_is_canonical() || !inference_caps.ff_tokens,
            "ff_tokens requires canonical tokenization"
        );
        ensure!(
            !inference_caps.backtrack || inference_caps.ff_tokens,
            "backtrack requires ff_tokens"
        );

        let compute_mask_start_time = Instant::now();
        let mut max_tokens = usize::MAX;
        if let GrammarInit::Serialized(input) = &grammar_init {
            if let Some(m) = input.max_tokens {
                max_tokens = m;
            }
        }
        let compiled_grammar = grammar_init.to_cgrammar(
            Some(token_env.clone()),
            &mut logger,
            limits.clone(),
            factory.extra_lexemes(),
        )?;
        let parser = Parser::new(
            token_env.clone(),
            compiled_grammar,
            limits.clone(),
            factory.perf_counters(),
        )?;
        let eos_tokens = token_env.tok_trie().eos_tokens().to_vec();

        Ok(TokenParser {
            bias_computer: factory.slicer().clone(),
            logger,
            token_env,
            inference_caps,
            limits,
            max_step_stats: ParserStats::default(),
            last_step_stats: ParserStats::default(),
            compute_mask_start_time,
            is_accepting_cache: None,
            ff_tokens_cache: None,
            stop_reason: StopReason::NotStopped,
            error_message: None,
            parser,
            dbg_grammar: String::new(),
            eos_tokens,
            llm_tokens: Vec::new(),
            llm_bytes: Vec::new(),
            grm_prefix: Vec::new(),
            max_tokens_total: max_tokens,
            last_bias_time: Duration::from_secs(0),
            is_fresh: true,
            had_backtrack: false,
            had_rollback: false,
        })
    }

    pub fn grammar_warnings(&mut self) -> Vec<String> {
        self.parser.grammar_warnings()
    }

    pub fn get_capture(&self, name: &str) -> Option<&[u8]> {
        self.parser.get_capture(name)
    }

    pub fn captures(&self) -> &[(String, Vec<u8>)] {
        self.parser.captures()
    }

    // regular .clone() uses a shared lexer state
    pub fn deep_clone(&self) -> Self {
        self.clone_with_parser(self.parser.deep_clone())
    }

    pub fn stop_reason(&self) -> StopReason {
        self.stop_reason
    }

    pub fn stopped(&self) -> bool {
        self.stop_reason != StopReason::NotStopped
    }

    pub fn is_fresh(&self) -> bool {
        self.is_fresh
    }

    pub fn parser_stats(&self) -> &ParserStats {
        self.parser.stats()
    }

    pub fn last_step_stats(&self) -> &ParserStats {
        &self.last_step_stats
    }

    pub fn max_step_stats(&self) -> &ParserStats {
        &self.max_step_stats
    }

    pub fn num_tokens(&self) -> usize {
        self.llm_tokens.len()
    }

    pub fn final_bytes(&self) -> &[u8] {
        &self.llm_bytes[self.grm_prefix.len()..]
    }

    pub fn is_accepting(&mut self) -> bool {
        if let Some(acc) = self.is_accepting_cache {
            acc
        } else {
            let r = !self.has_ff_bytes() && self.parser.is_accepting();
            self.is_accepting_cache = Some(r);
            r
        }
    }

    pub fn bytes_since(&self, mut idx: usize) -> &[u8] {
        idx += self.grm_prefix.len();
        let endp = std::cmp::min(self.llm_bytes.len(), self.parser.hidden_start());
        if idx >= self.llm_bytes.len() || idx >= endp {
            return &[];
        }
        &self.llm_bytes[idx..endp]
    }

    pub fn start_without_prompt(&mut self) {
        infoln!(
            self,
            "initial lexer cost: {} (no prompt)",
            self.parser.lexer_stats()
        );

        assert!(self.is_fresh);
        self.is_fresh = false;
    }

    fn tokenize_and_chop(
        &mut self,
        mut tokens: Vec<TokenId>,
        num_fixed: usize,
    ) -> (Vec<TokenId>, usize) {
        let trie = self.token_env.tok_trie();
        let (chop_tokens, chop_bytes) = self
            .parser
            .with_recognizer(|r| trie.chop_tokens(r, &tokens[num_fixed..]));
        infoln!(
            self,
            "tokenize -> {}; chop: {} tokens, {} bytes",
            trie.tokens_dbg(&tokens),
            chop_tokens,
            chop_bytes
        );
        // here we remove a suffix from tokens that could be possibly tokenized differently
        tokens.truncate(tokens.len() - chop_tokens);
        (tokens, chop_bytes)
    }

    pub fn process_prompt(&mut self, prompt: Vec<TokenId>) -> Vec<TokenId> {
        infoln!(self, "initial lexer cost: {}", self.parser.lexer_stats());

        assert!(self.token_env.tokenize_is_canonical());
        assert!(self.is_fresh);
        self.is_fresh = false;

        assert!(self.llm_tokens.is_empty());

        let trie = self.token_env.tok_trie();
        infoln!(self, "prompt: {}", trie.tokens_dbg(&prompt));
        let mut prompt_bytes = trie.decode_raw(&prompt);
        if self.can_force_bytes() {
            self.parser.force_bytes();
        }
        let grm_bytes = self.parser.get_bytes().to_vec();
        prompt_bytes.extend_from_slice(&grm_bytes);

        let (tokens, num_fixed) = self.token_env.tokenize_bytes_marker(&prompt_bytes);
        let (res_prompt, chop_bytes) = self.tokenize_and_chop(tokens, num_fixed);

        let trie = self.token_env.tok_trie();
        infoln!(
            self,
            "prompt+grm: {} {}",
            trie.tokens_dbg(&res_prompt),
            self.parser.grammar().lexer_spec().no_forcing
        );

        // if we moved a bunch of grammar to the prompt, update llm_tokens to reflect that
        if chop_bytes <= grm_bytes.len() {
            self.llm_bytes = grm_bytes[0..grm_bytes.len() - chop_bytes].to_vec();
            self.llm_tokens = self.token_env.tokenize_bytes_marker(&self.llm_bytes).0;
            self.parser.apply_forced(self.llm_bytes.len());
            let decoded = self.tok_trie().decode_raw(&self.llm_tokens);
            if !self.llm_bytes.is_empty()
                && !decoded.is_empty()
                && decoded[1..] == self.llm_bytes
                && decoded[0] == b' '
            {
                infoln!(self, "applying <s>space hack");
                self.grm_prefix = decoded[0..1].to_vec();
                self.llm_bytes = decoded;
            }
            infoln!(self, "ini_tokens: {}", trie.tokens_dbg(&self.llm_tokens));
        } else {
            // pretend the final bit of prompt was the prefix of the grammar
            self.grm_prefix = prompt_bytes
                [prompt_bytes.len() - chop_bytes..prompt_bytes.len() - grm_bytes.len()]
                .to_vec();
            infoln!(
                self,
                "force_prefix: {:?}",
                String::from_utf8_lossy(&self.grm_prefix)
            );
        }

        infoln!(self, "res_prompt: {}", trie.tokens_dbg(&res_prompt));
        res_prompt
    }

    pub fn augment_err(&self, e: impl Display) -> String {
        if self.limits.verbose_errors {
            format!(
                "{e}\n<state>\n{}\n</state><grammar>\n{}\n</grammar>",
                self.dump_state(),
                self.dbg_grammar
            )
        } else {
            format!("{e}\n<non-verbose/>")
        }
    }

    pub fn dump_state(&self) -> String {
        // make sure not take self.parser.shared lock
        // for example, self.parser.lexer_stats() takes it
        // if we take it after panic, it will be poisoned
        format!(
            "Tokens: {}\n{} tokens, {} bytes; grm_prefix: {:?}\nFlags:{}{}\nParser: {}\nStop: {}\nError: {}",
            self.tok_trie().tokens_dbg(&self.llm_tokens),
            self.llm_tokens.len(),
            self.llm_bytes.len(),
            String::from_utf8_lossy(&self.grm_prefix),
            if self.had_backtrack {
                " had_backtrack"
            } else {
                ""
            },
            if self.had_rollback {
                " had_rollback"
            } else {
                ""
            },
            self.parser.stats(),
            self.stop_reason,
            self.error_message.as_deref().unwrap_or("None"),
        )
    }

    fn clear_caches(&mut self) {
        self.is_accepting_cache = None;
        self.ff_tokens_cache = None;
    }

    fn stop(&mut self, warn: &str, reason: StopReason) -> anyhow::Error {
        if !warn.is_empty() {
            self.error_message = Some(warn.to_string());
            warn!(self, "{}; stopping", warn);
        }
        self.stop_reason = reason;
        self.anyhow_error()
    }

    fn tok_trie(&self) -> &toktrie::TokTrie {
        self.token_env.tok_trie()
    }

    pub fn error_message(&self) -> Option<String> {
        self.error_message.clone()
    }

    fn check_initialized(&self, lbl: &str) -> Result<()> {
        ensure!(!self.is_fresh, "process_prompt() not called in {}", lbl);
        ensure!(
            !self.stopped(),
            "parser stopped in {}; {}",
            lbl,
            self.error_message()
                .unwrap_or("no error message".to_string())
        );
        Ok(())
    }

    pub fn validate_token(&mut self, token: TokenId) -> Result<bool> {
        if self.stopped() {
            return Ok(false);
        }
        self.check_initialized("validate_token")?;
        self.validate_tokens_raw(&[token]).map(|n| n > 0)
    }

    pub fn reset(&mut self) -> Result<()> {
        self.rollback(self.llm_tokens.len())
    }

    pub fn rollback(&mut self, n_tokens: usize) -> Result<()> {
        rollback::run(self, n_tokens)
    }

    /// Returns how many of the passed tokens can be accepted by the parser.
    /// It does not tokenize forced bytes, so will accept non-canonical tokenizations.
    /// If called with more than one token, it may ignore max_tokens constraints.
    pub fn validate_tokens_raw(&mut self, tokens: &[TokenId]) -> Result<usize> {
        if self.stopped() {
            return Ok(0);
        }
        self.check_initialized("validate_tokens_raw")?;

        if tokens.is_empty() {
            return Ok(0);
        }

        let n_vocab = self.tok_trie().vocab_size();
        for &t in tokens {
            if t as usize >= n_vocab {
                return Err(self.stop(
                    &format!("token id {t} out of range"),
                    StopReason::InternalError,
                ));
            }
        }

        let n_valid = self.parser.validate_tokens(tokens);
        Ok(n_valid)
    }

    fn anyhow_error(&self) -> anyhow::Error {
        anyhow::anyhow!(self
            .error_message
            .clone()
            .unwrap_or(self.stop_reason.to_string()))
    }

    // compute_mask() is a top-level method in this file.
    // compute_mask() is called by Constraint::compute_mask().
    pub fn compute_mask(&mut self) -> Result<SimpleVob> {
        self.compute_mask_start_time = Instant::now();
        let r = self.compute_mask_inner();
        self.parser
            .perf_counters()
            .compute_mask
            .record(self.compute_mask_start_time.elapsed());
        r
    }

    fn compute_mask_inner(&mut self) -> Result<SimpleVob> {
        self.check_initialized("compute_mask")?;

        infoln!(self, "compute_mask");

        let prefix = if self.can_force_bytes() {
            let (ff_tokens, token_prefix) = self
                .ff_tokens_cache
                .take()
                .unwrap_or_else(|| self.ff_tokens());
            if !ff_tokens.is_empty() {
                let t = ff_tokens[0];
                infoln!(self, "forcing ff_token by mask: {}", t);
                let mask = self.tok_trie().singleton_token_set(t);
                self.last_step_stats = ParserStats::default();
                return Ok(mask);
            } else {
                // no tokens, so we got all our bytes back
                token_prefix
            }
        } else {
            let mut trg = Vec::new();
            self.compute_ff_bytes_to(&mut trg);
            trg
        };

        let mut allowed_tokens = self.compute_bias(&prefix);

        if let Some(s) = self.parser.get_error() {
            return Err(self.stop_for_parser_error("", s));
        }

        let accepting = self.is_accepting();
        progress::finish_mask(&mut allowed_tokens, &self.eos_tokens, accepting);

        self.log_final(&prefix, &allowed_tokens);

        if allowed_tokens.is_zero() {
            infoln!(self, "no tokens allowed, stopping");
            return Err(self.stop("", StopReason::NoExtensionBias));
        }

        Ok(allowed_tokens)
    }

    fn stop_for_parser_error(&mut self, pref: &str, err: ParserError) -> anyhow::Error {
        self.stop(&format!("{}{}", pref, err.message()), err.stop_reason())
    }

    fn apply_token(&mut self, tok_id: TokenId) -> Result<usize> {
        self.clear_caches();

        let trie = self.token_env.tok_trie();

        if (tok_id as usize) >= trie.vocab_size() {
            return Err(self.stop(
                &format!("token id {tok_id} out of range"),
                StopReason::InternalError,
            ));
        }

        self.llm_tokens.push(tok_id);

        let tok_bytes = trie.decode_raw(&[tok_id]);

        // first, check we're still in grm_prefix
        let prefix_len = self.grm_prefix.len().saturating_sub(self.llm_bytes.len());

        infoln!(
            self,
            "consume_token: {} {} prefix={}",
            tok_id,
            trie.token_dbg(tok_id),
            prefix_len
        );

        let tok_bytes = if prefix_len > 0 {
            let to_apply = &tok_bytes[0..std::cmp::min(tok_bytes.len(), prefix_len)];
            self.llm_bytes.extend_from_slice(to_apply);

            if self.grm_prefix[0..self.llm_bytes.len()] != self.llm_bytes {
                return Err(self.stop(
                    &format!(
                        "prefix mismatch: applying {:?}; {:?} vs {:?}",
                        String::from_utf8_lossy(to_apply),
                        String::from_utf8_lossy(&self.grm_prefix),
                        String::from_utf8_lossy(&self.llm_bytes)
                    ),
                    StopReason::InternalError,
                ));
            }

            if prefix_len < tok_bytes.len() {
                &tok_bytes[prefix_len..]
            } else {
                // still completely in prefix, nothing more to apply
                return Ok(0);
            }
        } else {
            &tok_bytes
        };

        if let Some(err) = self.parser.get_error() {
            return Err(self.stop_for_parser_error("", err));
        }

        // now apply normally
        match self.parser.apply_token(tok_bytes, tok_id) {
            Err(e) => {
                return Err(self.stop(
                    &format!("Parser Error: {e}"),
                    StopReason::ParserTooComplex, // TODO - there are other reasons
                ));
            }
            Ok(backtrack_bytes0) => {
                self.llm_bytes.extend_from_slice(tok_bytes);

                if backtrack_bytes0 != 0 {
                    self.had_backtrack = true;
                    let backtrack = progress::backtrack(trie, &self.llm_tokens, backtrack_bytes0)
                        .expect("complete native token backtrack span");
                    let mut backtrack_tokens = backtrack.tokens;
                    let additional_backtrack_bytes = backtrack.additional_bytes;
                    let full_backtrack_bytes = backtrack.full_bytes;

                    let byte_ptr = self.llm_bytes.len() - full_backtrack_bytes;
                    infoln!(
                        self,
                        "backtrack: {} tokens / {}+{} bytes (deletes: {:?})",
                        backtrack_tokens,
                        backtrack_bytes0,
                        additional_backtrack_bytes,
                        String::from_utf8_lossy(&self.llm_bytes[byte_ptr..])
                    );
                    self.llm_bytes.truncate(byte_ptr);

                    let token_ptr = self.llm_tokens.len() - backtrack_tokens;
                    if !self.inference_caps.backtrack {
                        warn!(
                            self,
                            "can't backtrack over {}; this may confuse the model",
                            trie.tokens_dbg(&self.llm_tokens[token_ptr..])
                        );
                        // pretend there's no backtrack
                        backtrack_tokens = 0;
                    } else {
                        // make sure the parser know we actually don't have
                        // the non-backtracked bytes of backtracked token
                        self.parser.additional_backtrack(additional_backtrack_bytes);
                    }
                    self.llm_tokens.truncate(token_ptr);
                    return Ok(backtrack_tokens);
                }
            }
        }

        Ok(0)
    }

    fn pending_grm_prefix(&self) -> &[u8] {
        &self.grm_prefix[std::cmp::min(self.grm_prefix.len(), self.llm_bytes.len())..]
    }

    fn has_ff_bytes(&self) -> bool {
        !self.pending_grm_prefix().is_empty() || !self.parser.currently_forced_bytes().is_empty()
    }

    fn can_force_bytes(&self) -> bool {
        !self.parser.grammar().lexer_spec().no_forcing && self.token_env.tokenize_is_canonical()
    }

    pub fn force_bytes(&mut self) -> Vec<u8> {
        self.parser.force_bytes();
        let mut trg = Vec::new();
        self.compute_ff_bytes_inner(&mut trg);
        trg
    }

    fn compute_ff_bytes_to(&mut self, trg: &mut Vec<u8>) {
        // PERF: in some cases, this may be long
        if self.can_force_bytes() {
            self.parser.force_bytes();
        }
        self.compute_ff_bytes_inner(trg);
    }

    fn compute_ff_bytes_inner(&mut self, trg: &mut Vec<u8>) {
        // handle grm_prefix we might have injected
        if self.llm_bytes.len() < self.grm_prefix.len() {
            let inject = &self.grm_prefix[self.llm_bytes.len()..];
            trg.extend_from_slice(inject);
            infoln!(
                self,
                "injecting prefix: {:?}",
                String::from_utf8_lossy(inject)
            );
        }

        trg.extend_from_slice(self.parser.currently_forced_bytes());
    }

    /// Converts forced bytes into tokens.
    /// Also returns any bytes that need to be prefix of the
    /// next sampled token (token healing).
    fn ff_tokens(&mut self) -> (Vec<TokenId>, Vec<u8>) {
        forcing::ordinary(self)
    }

    fn compute_bias(&mut self, token_prefix: &[u8]) -> SimpleVob {
        let pre_stats = self.parser.stats().clone();
        let set = self.parser.compute_bias(&*self.bias_computer, token_prefix);
        let p_stats = self.parser.stats().delta(&pre_stats);
        self.last_bias_time = Duration::from_micros(p_stats.compute_time_us);
        self.last_step_stats = p_stats.clone();
        self.max_step_stats = self.max_step_stats.max(&p_stats);
        set
    }

    fn log_final(&mut self, token_prefix: &[u8], allowed_tokens: &SimpleVob) {
        infoln!(
            self,
            "step-stats: {}us; {} lex fuel; {} items; {}",
            self.compute_mask_start_time.elapsed().as_micros(),
            self.last_step_stats.lexer_cost,
            self.last_step_stats.all_items,
            self.parser.lexer_stats(),
        );

        infoln!(
            self,
            "bias: (pref: {:?}; accpt: {}; temp: {:.3}) {}",
            String::from_utf8_lossy(token_prefix),
            self.is_accepting_cache.unwrap(),
            self.parser.temperature().unwrap_or(0.0),
            self.token_env.tok_trie().token_set_dbg(allowed_tokens)
        );
    }

    pub fn temperature(&self) -> Option<f32> {
        self.parser.temperature()
    }

    /// Extend the current state of the parser with given token.
    /// Returns number of tokens to backtrack if any.
    pub fn consume_token(&mut self, token: TokenId) -> Result<usize> {
        progress::consume(self, token)
    }

    /// Check whether the current parser state forces the sequence to stop.
    /// If so, puts the parser in stop state and returns true.
    /// Otherwise, returns false.
    /// This generally should be called after consume_token().
    pub fn check_stop(&mut self) -> Result<bool> {
        progress::check_stop(self)
    }

    /// Check if there are any tokens to fast-forward, forced by the current
    /// parser state.
    pub fn compute_ff_tokens(&mut self) -> Vec<TokenId> {
        let r = self.ff_tokens();
        if self.can_force_bytes() {
            self.ff_tokens_cache = Some(r.clone());
        }
        r.0
    }

    /// Compute and then consume fast-forward tokens.
    pub fn consume_ff_tokens(&mut self) -> Result<Vec<TokenId>> {
        let ff_tokens = self.compute_ff_tokens();
        for &t in &ff_tokens {
            let num_backtrack = self.consume_token(t)?;
            if num_backtrack > 0 {
                return Err(self.stop(
                    &format!("backtrack required after ff_token: {t}"),
                    StopReason::InternalError,
                ));
            }
        }
        Ok(ff_tokens)
    }

    /// This function documents typical use of this interface.
    /// The `tokens` array simulates tokens being sampled.
    #[allow(dead_code)]
    fn typical_use(&mut self, prompt: Vec<TokenId>) -> Result<()> {
        // First, check if we need to token-heal the prompt,
        // and if there are some tokens forced by the beginning
        // of the grammar.
        let new_prompt = self.process_prompt(prompt);

        // pass new prompt to inference engine
        black_box(new_prompt);

        let mut tokens = vec![];

        loop {
            let temp = self.temperature();
            let mask = self.compute_mask()?;

            // model forward pass in parallel with compute_mask() goes here

            // simulate sampling a token with given mask
            black_box((temp, mask));
            let sampled_token = 42;

            let num_backtrack = self.consume_token(sampled_token)?;

            if num_backtrack == 0 {
                // normal situation - the token was accepted
                tokens.push(sampled_token);
            } else {
                // this will only happen if you enable backtrack
                assert!(self.inference_caps.backtrack);
                if num_backtrack == 1 {
                    // don't add the token to the list
                } else if num_backtrack > 1 {
                    // backtrack
                    tokens.truncate(tokens.len() - num_backtrack - 1);
                }
            }

            // This is optional; if you don't check, compute_mask() will
            // return an error when it cannot continue anymore.
            // If you check here, you can distinguish between normal stop
            // and an error.
            if self.check_stop()? {
                break;
            }

            // This is optional - call if you have the ability to append
            // several tokens at once.
            let forced = self.consume_ff_tokens()?;
            tokens.extend_from_slice(&forced);
        }

        Ok(())
    }

    pub fn invalidate_bias_cache(&mut self) {
        self.parser.invalidate_bias_cache();
    }
}
