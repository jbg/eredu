mod policy;
mod precompute;
pub mod prepared;
mod schedule;
use anyhow::{Context, Result};
use std::fmt::Debug;
use toktrie::{SimpleVob, TokTrie};

use crate::api::ParserLimits;

use super::{
    lexerspec::{Lexeme, LexemeIdx, LexerSpec},
    parser::ParserError,
    regexvec::{LexemeSet, MatchingLexemes, NextByte, RegexVec, StateDesc},
};

const DEBUG: bool = true;

macro_rules! debug {
    ($($arg:tt)*) => {
        if cfg!(feature = "logging") && DEBUG {
            eprintln!($($arg)*);
        }
    }
}

#[derive(Clone)]
pub struct Lexer {
    pub(crate) dfa: RegexVec,
    // set of bytes that are allowed in any of the lexemes
    // this is used to fail states quickly
    allowed_first_byte: SimpleVob,
    spec: LexerSpec,
}

pub type StateID = derivre::StateID;

/// PreLexeme contains index of the lexeme but not the bytes.
#[derive(Debug, Clone, Copy)]
pub struct PreLexeme {
    pub idx: MatchingLexemesIdx,
    pub byte: Option<u8>,
    /// Does the 'byte' above belong to the next lexeme?
    pub byte_next_row: bool,
    // Length in bytes of the hidden part of the lexeme.
    // pub hidden_len: u32,
}

impl PreLexeme {
    pub fn just_idx(idx: MatchingLexemesIdx) -> Self {
        PreLexeme {
            idx,
            byte: None,
            byte_next_row: false,
        }
    }
}

#[derive(Debug)]
pub enum LexerResult {
    Lexeme(PreLexeme),
    SpecialToken(StateID),
    State(StateID, u8),
    Error,
}

impl Lexer {
    pub fn from(spec: &LexerSpec, limits: &mut ParserLimits, dbg: bool) -> Result<Self> {
        let mut dfa = spec.to_regex_vec(limits)?;

        if dbg {
            debug!("lexer: {:?}\n  ==> dfa: {:?}", spec, dfa);
        }

        let s0 = dfa.initial_state(&spec.all_lexemes());
        check_dfa_error(&dfa).context("initial lexer state")?;
        let mut allowed_first_byte = SimpleVob::alloc(256);
        policy::warm_first_bytes(&mut allowed_first_byte, |byte| {
            let next = dfa.transition(s0, byte);
            check_dfa_error(&dfa).context("warming lexer first bytes")?;
            Ok::<_, anyhow::Error>(next)
        })?;

        let lex = Lexer {
            dfa,
            allowed_first_byte,
            spec: spec.clone(), // TODO check perf of Rc<> ?
        };

        Ok(lex)
    }

    pub fn lexer_spec(&self) -> &LexerSpec {
        &self.spec
    }

    pub fn start_state(&mut self, allowed_lexemes: &LexemeSet) -> StateID {
        self.dfa.initial_state(allowed_lexemes)
    }

    pub(super) fn check_error(&self) -> Result<()> {
        check_dfa_error(&self.dfa)
    }

    pub fn precompute_for(&mut self, trie: &TokTrie, allowed_lexemes: &LexemeSet) -> Result<()> {
        let state = self.start_state(allowed_lexemes);
        self.check_error()?;
        let plan = precompute::LexerPrecomputePlan::prepare(trie)
            .map_err(precompute::LexerPrecomputeFailure::into_ordinary)?;
        let storage = plan
            .compile()
            .map_err(precompute::LexerPrecomputeFailure::into_ordinary)?;
        storage
            .run_from_state(self, state)
            .map(drop)
            .map_err(precompute::LexerPrecomputeFailure::into_ordinary)
    }

    /// Runs the same large-lexeme startup schedule used by ParserState, with
    /// its remaining initial fuel and actual declaration order.
    pub fn prepare_large_lexemes(&mut self, trie: &TokTrie, limits: &ParserLimits) -> Result<()> {
        struct Ordinary<'a> {
            lexer: &'a mut Lexer,
            trie: &'a TokTrie,
        }
        impl schedule::Context for Ordinary<'_> {
            type Error = anyhow::Error;
            fn len(&self) -> usize {
                self.lexer.spec.lexemes.len()
            }
            fn lexeme(&self, index: usize) -> LexemeIdx {
                self.lexer.spec.lexemes[index].idx
            }
            fn set_fuel(&mut self, fuel: u64) {
                self.lexer.dfa.set_fuel(fuel);
            }
            fn weight(&mut self, lexeme: LexemeIdx) -> Result<u32> {
                self.lexer.dfa.lexeme_weight(lexeme)
            }
            fn precompute(&mut self, lexeme: LexemeIdx) -> Result<()> {
                let mut allowed = self.lexer.spec.alloc_lexeme_set();
                allowed.add(lexeme);
                self.lexer
                    .precompute_for(self.trie, &allowed)
                    .context("precomputing large lexeme")
            }
        }
        schedule::run(&mut Ordinary { lexer: self, trie }, limits)
    }

    pub fn transition_start_state(&mut self, s: StateID, first_byte: Option<u8>) -> StateID {
        first_byte.map(|b| self.dfa.transition(s, b)).unwrap_or(s)
    }

    pub fn a_dead_state(&self) -> StateID {
        StateID::DEAD
    }

    pub fn possible_hidden_len(&mut self, state: StateID) -> usize {
        self.dfa.possible_lookahead_len(state)
    }

    fn state_info(&self, state: StateID) -> &StateDesc {
        self.dfa.state_desc(state)
    }

    pub fn allows_eos(&mut self, state: StateID) -> bool {
        policy::allows_eos(&self.spec, self.state_info(state))
    }

    pub fn limit_state_to(&mut self, state: StateID, allowed_lexemes: &LexemeSet) -> StateID {
        self.dfa.limit_state_to(state, allowed_lexemes)
    }

    pub fn possible_lexemes(&self, state: StateID) -> &LexemeSet {
        &self.state_info(state).possible
    }

    pub fn force_lexeme_end(&self, prev: StateID) -> LexerResult {
        policy::force_end(self.state_info(prev))
    }

    pub fn try_lexeme_end(&mut self, prev: StateID) -> LexerResult {
        policy::try_end(prev, self.state_info(prev))
    }

    pub fn check_for_single_byte_lexeme(&mut self, state: StateID, b: u8) -> Option<PreLexeme> {
        policy::single_byte(state, b, self.dfa.next_byte(state))
    }

    pub fn subsume_possible(&mut self, state: StateID) -> bool {
        self.dfa.subsume_possible(state)
    }

    pub fn check_subsume(&mut self, state: StateID, extra_idx: usize, budget: u64) -> Result<bool> {
        self.dfa
            .check_subsume(state, self.spec.extra_lexeme(extra_idx), budget)
    }

    pub fn next_byte(&mut self, state: StateID) -> NextByte {
        // there should be no transition from a state with a lazy match
        // - it should have generated a lexeme
        assert!(!state.has_lowest_match());

        let forced = self.dfa.next_byte(state);
        policy::next_byte(forced, self.dfa.state_desc(state))
    }

    #[inline(always)]
    pub fn advance(&mut self, prev: StateID, byte: u8, enable_logging: bool) -> LexerResult {
        let state = self.dfa.transition(prev, byte);

        if enable_logging {
            let info = self.state_info(state);
            debug!(
                "lex: {:?} -{:?}-> {:?}, acpt={:?}/{:?}",
                prev, byte as char, state, info.greedy_accepting, info.lazy_accepting
            );
        }

        policy::advance(
            prev,
            byte,
            state,
            &self.allowed_first_byte,
            self.dfa.state_desc(prev),
            self.dfa.state_desc(state),
        )
    }

    pub fn lexemes_from_idx(&self, idx: MatchingLexemesIdx) -> &MatchingLexemes {
        matching(&self.spec.lexemes, idx, |state| Some(self.dfa.state_desc(state)))
            .expect("matching lexical source")
    }

    #[inline(always)]
    pub fn lexeme_props(&self, idx: MatchingLexemesIdx) -> (u32, bool) {
        properties(&self.spec, idx, |state| Some(self.dfa.state_desc(state)))
            .expect("lexeme property source")
    }

    pub fn dbg_lexeme(&self, lex: &Lexeme) -> String {
        let set = self.lexemes_from_idx(lex.idx);

        // let info = &self.lexemes[lex.idx.as_usize()];
        // if matches!(info.rx, RegexAst::Literal(_)) && lex.hidden_len == 0 {
        //     format!("[{}]", info.name)
        // }

        format!("{lex:?} {set:?}")
    }
}

fn check_dfa_error(dfa: &RegexVec) -> Result<()> {
    match dfa.get_error() {
        Some(message) => Err(ParserError::LexerError(message).into()),
        None => Ok(()),
    }
}

impl LexerResult {
    #[inline(always)]
    pub fn is_error(&self) -> bool {
        matches!(self, LexerResult::Error)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatchingLexemesIdx {
    Single(LexemeIdx),
    GreedyAccepting(StateID),
    LazyAccepting(StateID),
}

#[cfg(test)]
mod construction_tests;

pub use precompute::{
    LexerPrecompute, LexerPrecomputeFailure, LexerPrecomputePlan, LexerPrecomputeRequirements,
    LexerPrecomputeRunFailure,
};

pub(crate) use policy::{allows_eos, matching, properties};
