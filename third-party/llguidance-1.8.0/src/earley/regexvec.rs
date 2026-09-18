/// This file implements regex vectors.  To match tokens to lexemes, llguidance uses
/// a DFA whose nodes are regex vectors.  For more on this see
/// S. Owens, J. Reppy, and A. Turon.
/// Regular Expression Derivatives Reexamined".
/// Journal of Functional Programming 19(2):173-190, March 2009.
/// <https://www.khoury.northeastern.edu/home/turon/re-deriv.pdf> (retrieved 15 Nov 2024)
mod construction;
mod descriptor;
pub mod prepared;
mod subsumption;
mod transition;
use anyhow::{bail, Result};
use derivre::raw::{DerivCache, ExprSet, NextByteCache, RelevanceCache, VecHashCons};
use serde::{Deserialize, Serialize};
use std::fmt::{Debug, Display};
use toktrie::SimpleVob;

pub use derivre::{AlphabetInfo, ExprRef, NextByte, StateID};

use crate::api::ParserLimits;

use super::{lexerspec::LexemeIdx, parser::ParserError};

#[derive(Clone, Serialize, Deserialize, Default)]
pub struct LexerStats {
    pub num_regexps: usize,
    pub num_ast_nodes: usize,
    pub num_derived: usize,
    pub num_derivatives: usize,
    pub total_fuel_spent: usize,
    pub num_states: usize,
    pub num_transitions: usize,
    pub num_bytes: usize,
    pub alphabet_size: usize,
    pub error: bool,
}

#[derive(Clone)]
pub enum MatchingLexemes {
    None,
    One(LexemeIdx),
    Two([LexemeIdx; 2]),
    Many(Vec<LexemeIdx>),
}

impl MatchingLexemes {
    pub fn is_some(&self) -> bool {
        !matches!(self, MatchingLexemes::None)
    }

    pub fn is_none(&self) -> bool {
        !self.is_some()
    }

    pub fn first(&self) -> Option<LexemeIdx> {
        match self {
            MatchingLexemes::None => None,
            MatchingLexemes::One(idx) => Some(*idx),
            MatchingLexemes::Two([idx, _]) => Some(*idx),
            MatchingLexemes::Many(v) => v.first().copied(),
        }
    }

    pub fn contains(&self, idx: LexemeIdx) -> bool {
        match self {
            MatchingLexemes::None => false,
            MatchingLexemes::One(idx2) => *idx2 == idx,
            MatchingLexemes::Two([idx1, idx2]) => *idx1 == idx || *idx2 == idx,
            MatchingLexemes::Many(v) => v.contains(&idx),
        }
    }

    pub fn add(&mut self, idx: LexemeIdx) {
        self.add_with::<std::convert::Infallible>(idx, |values, total| {
            if values.capacity() == 0 {
                values.reserve_exact(total);
            } else {
                values.reserve(total - values.len());
            }
            Ok(())
        })
        .expect("ordinary matching-lexeme append");
    }

    fn add_with<E>(
        &mut self,
        idx: LexemeIdx,
        mut prepare: impl FnMut(&mut Vec<LexemeIdx>, usize) -> std::result::Result<(), E>,
    ) -> std::result::Result<(), E> {
        match self {
            Self::None => *self = Self::One(idx),
            Self::One(previous) => *self = Self::Two([*previous, idx]),
            Self::Two(previous) => {
                let previous = *previous;
                *self = Self::Many(Vec::new());
                let Self::Many(values) = self else {
                    unreachable!()
                };
                prepare(values, 3)?;
                values.extend_from_slice(&previous);
                values.push(idx);
            }
            Self::Many(values) => {
                prepare(values, values.len() + 1)?;
                values.push(idx);
            }
        }
        Ok(())
    }

    pub fn len(&self) -> usize {
        match self {
            MatchingLexemes::None => 0,
            MatchingLexemes::One(_) => 1,
            MatchingLexemes::Two(_) => 2,
            MatchingLexemes::Many(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn as_slice(&self) -> &[LexemeIdx] {
        match self {
            MatchingLexemes::None => &[],
            MatchingLexemes::One(idx) => std::slice::from_ref(idx),
            MatchingLexemes::Two(v) => v,
            MatchingLexemes::Many(v) => v.as_slice(),
        }
    }
}

impl Debug for MatchingLexemes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MatchingLexemes::None => write!(f, "Lex:[]"),
            MatchingLexemes::One(idx) => write!(f, "Lex:[{}]", idx.as_usize()),
            MatchingLexemes::Two([idx1, idx2]) => {
                write!(f, "Lex:[{},{}]", idx1.as_usize(), idx2.as_usize())
            }
            MatchingLexemes::Many(v) => write!(
                f,
                "Lex:[{}]",
                v.iter()
                    .map(|idx| idx.as_usize().to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            ),
        }
    }
}

impl Display for LexerStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "regexps: {} with {} nodes (+ {} derived via {} derivatives with total fuel {}), states: {}; transitions: {}; bytes: {}; alphabet size: {} {}",
            self.num_regexps,
            self.num_ast_nodes,
            self.num_derived,
            self.num_derivatives,
            self.total_fuel_spent,
            self.num_states,
            self.num_transitions,
            self.num_bytes,
            self.alphabet_size,
            if self.error { "ERROR" } else { "" }
        )
    }
}

impl Debug for LexerStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Display::fmt(self, f)
    }
}

#[derive(Clone, Debug)]
pub struct LexemeSet {
    vob: SimpleVob,
}

impl LexemeSet {
    pub fn new(size: usize) -> Self {
        LexemeSet {
            vob: SimpleVob::alloc(size),
        }
    }

    pub fn len(&self) -> usize {
        self.vob.len()
    }

    pub fn from_vob(vob: &SimpleVob) -> Self {
        Self::from_owned_vob(vob.clone())
    }

    pub(crate) fn copy_plan(
        &self,
    ) -> Result<toktrie::TokenMaskConstructionPlan<'_>, toktrie::TokenMaskSourceError> {
        toktrie::TokenMaskConstructionPlan::copy(&self.vob)
    }

    pub(crate) fn from_owned_vob(vob: SimpleVob) -> Self {
        Self { vob }
    }

    pub fn is_empty(&self) -> bool {
        self.vob.is_zero()
    }

    #[inline(always)]
    pub fn iter(&self) -> impl Iterator<Item = LexemeIdx> + '_ {
        self.vob.iter().map(|e| LexemeIdx::new(e as usize))
    }

    pub fn add(&mut self, idx: LexemeIdx) {
        self.vob.set(idx.as_usize(), true);
    }

    pub fn remove(&mut self, idx: LexemeIdx) {
        self.vob.set(idx.as_usize(), false);
    }

    pub fn first(&self) -> Option<LexemeIdx> {
        self.vob.first_bit_set().map(LexemeIdx::new)
    }

    pub fn contains(&self, idx: LexemeIdx) -> bool {
        self.vob.get(idx.as_usize())
    }

    pub fn clear(&mut self) {
        self.vob.set_all(false);
    }
}

#[derive(Clone)]
pub struct RegexVec {
    exprs: ExprSet,
    deriv: DerivCache,
    next_byte: NextByteCache,
    relevance: RelevanceCache,
    alpha: AlphabetInfo,
    #[allow(dead_code)]
    rx_lexemes: Vec<RxLexeme>,
    lazy: LexemeSet,
    subsumable: LexemeSet,
    rx_list: Vec<ExprRef>,
    special_token_rx: Option<ExprRef>,
    rx_sets: VecHashCons,
    state_table: Vec<StateID>,
    state_descs: Vec<StateDesc>,
    num_transitions: usize,
    num_ast_nodes: usize,
    max_states: usize,
    state_growth_error: Option<StateGrowthError>,
    fuel: u64,
}

#[derive(Clone, Debug)]
pub struct StateDesc {
    pub state: StateID,
    pub greedy_accepting: MatchingLexemes,
    pub possible: LexemeSet,

    /// Index of lowest matching regex if any.
    /// Lazy regexes match as soon as they accept, while greedy only
    /// if they accept and force EOI.
    pub lazy_accepting: MatchingLexemes,
    pub lazy_hidden_len: u32,

    pub has_special_token: bool,

    possible_lookahead_len: Option<usize>,
    lookahead_len: Option<Option<usize>>,
    next_byte: Option<NextByte>,
}

impl StateDesc {
    fn empty(state: StateID, possible: LexemeSet) -> Self {
        Self {
            state,
            greedy_accepting: MatchingLexemes::None,
            possible,
            possible_lookahead_len: None,
            lookahead_len: None,
            next_byte: None,
            lazy_accepting: MatchingLexemes::None,
            lazy_hidden_len: 0,
            has_special_token: false,
        }
    }
}

// These checks bound state publication only. Derivative construction and the
// candidate vector needed to recognize an existing state have separate costs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StateGrowthError {
    Limit { states: usize, limit: usize },
    StateId,
    TableCapacity,
    RegexSetCapacity,
}

impl Display for StateGrowthError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Limit { states, limit } => write!(f, "too many states: {states} >= {limit}"),
            Self::StateId => f.write_str("lexer state identifier capacity exceeded"),
            Self::TableCapacity => f.write_str("lexer state transition table capacity exceeded"),
            Self::RegexSetCapacity => f.write_str("lexer regex-set storage capacity exceeded"),
        }
    }
}

impl std::error::Error for StateGrowthError {}

fn checked_state_table_growth(
    states: usize,
    alphabet: usize,
    limit: usize,
) -> std::result::Result<usize, StateGrowthError> {
    if states >= limit {
        return Err(StateGrowthError::Limit { states, limit });
    }
    // StateID reserves its low bit for lowest-match metadata. StateID::new
    // shifts the supplied index, so a full-width u32 check is insufficient.
    if states > (u32::MAX >> 1) as usize {
        return Err(StateGrowthError::StateId);
    }
    let count = states
        .checked_add(1)
        .ok_or(StateGrowthError::TableCapacity)?;
    if count > isize::MAX as usize / std::mem::size_of::<StateDesc>() {
        return Err(StateGrowthError::TableCapacity);
    }
    let entries = count
        .checked_mul(alphabet)
        .filter(|_| alphabet != 0)
        .ok_or(StateGrowthError::TableCapacity)?;
    if entries > isize::MAX as usize / std::mem::size_of::<StateID>() {
        return Err(StateGrowthError::TableCapacity);
    }
    Ok(entries)
}

// public implementation
impl RegexVec {
    pub fn alpha(&self) -> &AlphabetInfo {
        &self.alpha
    }

    pub fn lazy_regexes(&self) -> &LexemeSet {
        &self.lazy
    }

    /// Create and return the initial state of a DFA for this
    /// regex vector
    pub fn initial_state(&mut self, selected: &LexemeSet) -> StateID {
        if self.has_error() {
            return StateID::DEAD;
        }
        let mut vec_desc = vec![];
        construction::selected::<std::convert::Infallible>(&self.rx_list, selected, |idx, root| {
            Self::push_rx(&mut vec_desc, idx, root);
            Ok(())
        })
        .expect("ordinary initial candidate");
        self.insert_state(vec_desc)
    }

    #[inline(always)]
    pub fn state_desc(&self, state: StateID) -> &StateDesc {
        &self.state_descs[state.as_usize()]
    }

    pub fn possible_lookahead_len(&mut self, state: StateID) -> usize {
        let desc = &mut self.state_descs[state.as_usize()];
        if let Some(len) = desc.possible_lookahead_len {
            return len;
        }
        let max_len = descriptor::possible_hidden(&self.exprs, self.rx_sets.get(state.as_u32()));
        desc.possible_lookahead_len = Some(max_len);
        max_len
    }

    pub fn lookahead_len_for_state(&mut self, state: StateID) -> Option<usize> {
        let desc = &mut self.state_descs[state.as_usize()];
        if desc.greedy_accepting.is_none() {
            return None;
        }
        if let Some(len) = desc.lookahead_len {
            return len;
        }
        let res = descriptor::lookahead(
            &self.exprs,
            self.rx_sets.get(state.as_u32()),
            &desc.greedy_accepting,
        );
        desc.lookahead_len = Some(res);
        res
    }

    /// Given a transition (a from-state and a byte) of the DFA
    /// for this regex vector, return the to-state.  It is taken
    /// from the cache, if it is cached, and created otherwise.
    #[inline(always)]
    pub fn transition(&mut self, state: StateID, b: u8) -> StateID {
        if self.has_error() {
            return StateID::DEAD;
        }
        let idx = self.alpha.map_state(state, b);
        let new_state = self.state_table[idx];
        if new_state != StateID::MISSING {
            new_state
        } else {
            self.transition_inner(state, b, idx)
        }
    }

    /// "Subsumption" is a feature implementing regex containment.
    /// subsume_possible() returns true if it's possible for this
    /// state, false otherwise.
    pub fn subsume_possible(&mut self, state: StateID) -> bool {
        if state.is_dead() || self.has_error() {
            return false;
        }
        for (idx, _) in iter_state(&self.rx_sets, state) {
            if self.lazy.contains(idx) {
                return false;
            }
        }
        true
    }

    /// Part of the interface for "subsumption", a feature implementing
    /// regex containment.
    pub fn check_subsume(
        &mut self,
        state: StateID,
        lexeme_idx: LexemeIdx,
        budget: u64,
    ) -> Result<bool> {
        assert!(self.subsume_possible(state));
        let small = self.get_rx(lexeme_idx);
        Ok(subsumption::run(
            &mut subsumption::Ordinary {
                source: &mut self.exprs,
                derivative: &mut self.deriv,
                relevance: &mut self.relevance,
            },
            self.rx_sets.get(state.as_u32()),
            &self.subsumable,
            small,
            budget,
        )?)
    }

    /// Estimate the size of the regex tables in bytes.
    pub fn num_bytes(&self) -> usize {
        self.exprs.num_bytes()
            + self.deriv.num_bytes()
            + self.next_byte.num_bytes()
            + self.relevance.num_bytes()
            + self.state_descs.len() * 100
            + self.state_table.len() * std::mem::size_of::<StateID>()
            + self.rx_sets.num_bytes()
    }

    // Find the lowest, or best, match in 'state'.  It is the first lazy regex.
    // If there is no lazy regex, and all greedy lexemes have reached the end of
    // the lexeme, then it is the first greedy lexeme.  If neither of these
    // criteria produce a choice for "best", 'None' is returned.
    fn lowest_match_inner(&mut self, desc: &mut StateDesc) {
        let mut expressions = descriptor::Ordinary {
            source: &self.exprs,
            next: &mut self.next_byte,
        };
        descriptor::lowest(
            &mut expressions,
            self.rx_sets.get(desc.state.as_u32()),
            &self.rx_list,
            self.special_token_rx,
            &self.lazy,
            desc,
            &mut descriptor::Selection::new(),
        )
        .expect("ordinary lowest matching lexeme");
    }

    /// Check if the there is only one transition out of state.
    /// This is an approximation - see docs for NextByte.
    pub fn next_byte(&mut self, state: StateID) -> NextByte {
        let desc = &mut self.state_descs[state.as_usize()];
        if let Some(next_byte) = desc.next_byte {
            return next_byte;
        }

        let next_byte = descriptor::next_bytes(
            &mut descriptor::Ordinary {
                source: &self.exprs,
                next: &mut self.next_byte,
            },
            self.rx_sets.get(state.as_u32()),
        )
        .expect("ordinary next-byte query");
        desc.next_byte = Some(next_byte);
        next_byte
    }

    pub fn limit_state_to(&mut self, state: StateID, allowed_lexemes: &LexemeSet) -> StateID {
        if self.has_error() {
            return StateID::DEAD;
        }
        let mut vec_desc = vec![];
        descriptor::restricted::<std::convert::Infallible>(
            self.rx_sets.get(state.as_u32()),
            allowed_lexemes,
            |index, root| {
                Self::push_rx(&mut vec_desc, index, root);
                Ok(())
            },
        )
        .expect("ordinary restricted candidate");
        self.insert_state(vec_desc)
    }

    pub fn total_fuel_spent(&self) -> u64 {
        self.exprs.cost()
    }

    pub fn lexeme_weight(&mut self, lexeme_idx: LexemeIdx) -> anyhow::Result<u32> {
        let e = self.rx_list[lexeme_idx.as_usize()];
        Ok(self.exprs.get_weight(e)?)
    }

    pub fn set_max_states(&mut self, max_states: usize) {
        if !self.has_error() {
            self.max_states = max_states;
        }
    }

    // Each fuel point is on the order 100ns (though it varies).
    // So, for ~10ms limit, do a .set_fuel(100_000).
    pub fn set_fuel(&mut self, fuel: u64) {
        if !self.has_error() {
            self.fuel = fuel;
        }
    }

    pub fn get_fuel(&self) -> u64 {
        self.fuel
    }

    pub fn has_error(&self) -> bool {
        self.alpha.has_error()
    }

    pub fn get_error(&self) -> Option<String> {
        if self.has_error() {
            if self.fuel == 0 {
                Some("too many expressions constructed".to_string())
            } else if let Some(error) = self.state_growth_error {
                Some(error.to_string())
            } else {
                Some("unknown error".to_string())
            }
        } else {
            None
        }
    }

    pub fn stats(&self) -> LexerStats {
        LexerStats {
            num_regexps: self.rx_list.len(),
            num_ast_nodes: self.num_ast_nodes,
            num_derived: self.exprs.len() - self.num_ast_nodes,
            num_derivatives: self.deriv.num_deriv,
            total_fuel_spent: self.total_fuel_spent() as usize,
            num_states: self.state_descs.len(),
            num_transitions: self.num_transitions,
            num_bytes: self.num_bytes(),
            alphabet_size: self.alpha.len(),
            error: self.has_error(),
        }
    }

    pub fn print_state_table(&self) {
        for (state, row) in self.state_table.chunks(self.alpha.len()).enumerate() {
            println!("state: {state}");
            for (b, &new_state) in row.iter().enumerate() {
                println!("  s{b:?} -> {new_state:?}");
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct RxLexeme {
    pub rx: ExprRef,
    pub lazy: bool,
    #[allow(dead_code)]
    pub priority: i32,
}

// private implementation
impl RegexVec {
    pub(crate) fn new_with_exprset(
        exprset: ExprSet,
        rx_lexemes: Vec<RxLexeme>,
        special_token_rx: Option<ExprRef>,
        limits: &mut ParserLimits,
    ) -> Result<Self> {
        // Both sentinels are real table entries. Reject an impossible limit
        // before alphabet compression, relevance work, or table allocation.
        limits.validate_lexer_state_limit()?;
        let roots = super::lexerspec::LexerRootSource::from_rows(rx_lexemes, special_token_rx)?;
        Self::new_with_input(roots.ordinary(exprset)?, limits)
    }

    pub(crate) fn new_with_input(
        input: super::lexerspec::RegexVectorInput,
        limits: &mut ParserLimits,
    ) -> Result<Self> {
        limits.validate_lexer_state_limit()?;
        let super::lexerspec::RegexVectorInput {
            alpha,
            expressions: mut exprset,
            rows: rx_lexemes,
            roots: mut rx_list,
            special_token_rx,
        } = input;
        let num_ast_nodes = exprset.len();

        let fuel0 = limits.initial_lexer_fuel;
        let mut relevance = RelevanceCache::new();
        let mut expressions = construction::Ordinary {
            source: &mut exprset,
            relevance: &mut relevance,
        };
        if construction::roots(
            &mut expressions,
            &mut rx_list,
            &mut limits.initial_lexer_fuel,
        )
        .is_err()
        {
            bail!(ParserError::LexerError(format!(
                "fuel exhausted when checking relevance of lexemes ({fuel0})"
            )));
        }
        let mut lazy = LexemeSet::new(rx_lexemes.len());
        let mut subsumable = LexemeSet::new(rx_lexemes.len());
        construction::classify(&mut expressions, &rx_lexemes, &mut lazy, &mut subsumable)?;

        let rx_sets = StateID::new_hash_cons();
        let mut r = RegexVec {
            deriv: DerivCache::new(),
            next_byte: NextByteCache::new(),
            special_token_rx,
            relevance,
            lazy,
            subsumable,
            rx_lexemes,
            exprs: exprset,
            alpha,
            rx_list,
            rx_sets,
            state_table: vec![],
            state_descs: vec![],
            num_transitions: 0,
            num_ast_nodes,
            fuel: u64::MAX,
            max_states: limits.max_lexer_states,
            state_growth_error: None,
        };

        assert!(r.lazy.len() == r.rx_list.len());

        // Hashcons already contains DEAD and MISSING. Bootstrap their two
        // descriptors explicitly; every later hashcons hit has a descriptor.
        let dead = r.compute_state_desc(StateID::DEAD);
        let table_len = r.prepare_state_append().expect("initial dead state fits");
        r.append_state(dead, table_len);
        let missing = r.state_descs[0].clone();
        let table_len = r
            .prepare_state_append()
            .expect("initial missing state fits");
        r.append_state(missing, table_len);
        // in fact, transition from MISSING and DEAD should both lead to DEAD
        r.state_table.fill(StateID::DEAD);
        assert!(!r.alpha.is_empty());
        Ok(r)
    }

    fn get_rx(&self, idx: LexemeIdx) -> ExprRef {
        self.rx_list[idx.as_usize()]
    }

    fn prepare_state_append(&mut self) -> Option<usize> {
        if self.has_error() {
            return None;
        }
        match checked_state_table_growth(self.state_descs.len(), self.alpha.len(), self.max_states)
        {
            Ok(table_len) => Some(table_len),
            Err(error) => {
                self.state_growth_error = Some(error);
                self.alpha.enter_error_state();
                None
            }
        }
    }

    fn append_state(&mut self, state_desc: StateDesc, table_len: usize) {
        // The checked target replaces the temporary row Vec. No new state or
        // descriptor computation occurs before the state-count/geometry guard.
        self.state_table.resize(table_len, StateID::MISSING);
        self.state_descs.push(state_desc);
    }

    fn insert_state(&mut self, lst: Vec<u32>) -> StateID {
        assert!(lst.len().is_multiple_of(2));
        if self.has_error() {
            return StateID::DEAD;
        }
        let id = if let Some(id) = self.rx_sets.lookup(&lst) {
            // Existing hits remain valid at the limit, including after a limit
            // was lowered below the number of already constructed states.
            assert!((id as usize) < self.state_descs.len());
            StateID::new(id)
        } else {
            let Some(table_len) = self.prepare_state_append() else {
                return StateID::DEAD;
            };
            if self.rx_sets.validate_insert_geometry(lst.len()).is_err() {
                self.state_growth_error = Some(StateGrowthError::RegexSetCapacity);
                self.alpha.enter_error_state();
                return StateID::DEAD;
            }
            // This remains the ordinary growing hashcons insertion. The guard
            // is not a transactional allocator or a complete parser byte bound.
            let id = self.rx_sets.insert(&lst);
            assert_eq!(id as usize, self.state_descs.len());
            let id = StateID::new(id);
            let state_desc = self.compute_state_desc(id);
            self.append_state(state_desc, table_len);
            id
        };
        if self.state_desc(id).lazy_accepting.is_some() {
            id._set_lowest_match()
        } else {
            id
        }
    }

    fn compute_state_desc(&mut self, state: StateID) -> StateDesc {
        let mut res = StateDesc::empty(state, LexemeSet::new(self.rx_list.len()));
        descriptor::possible(
            &mut descriptor::Ordinary {
                source: &self.exprs,
                next: &mut self.next_byte,
            },
            self.rx_sets.get(state.as_u32()),
            &mut res,
        )
        .expect("ordinary possible lexemes");
        self.lowest_match_inner(&mut res);

        // println!("state {:?} desc: {:?}", state, res);

        res
    }

    fn push_rx(vec_desc: &mut Vec<u32>, idx: LexemeIdx, e: ExprRef) {
        vec_desc.push(idx.as_usize() as u32);
        vec_desc.push(e.as_u32());
    }

    /// Given a transition (from-state and byte), create the to-state.
    /// It is assumed the to-state does not exist.
    fn transition_inner(&mut self, state: StateID, b: u8, idx: usize) -> StateID {
        assert!(state.is_valid());

        let mut vec_desc = vec![];

        let _ = transition::build(
            &mut transition::Ordinary {
                source: &mut self.exprs,
                derivative: &mut self.deriv,
                relevance: &mut self.relevance,
                candidate: &mut vec_desc,
            },
            self.rx_sets.get(state.as_u32()),
            b,
            &mut self.fuel,
        );
        if self.fuel == 0 {
            self.alpha.enter_error_state();
        }
        let new_state = self.insert_state(vec_desc);
        self.num_transitions += 1;
        self.state_table[idx] = new_state;
        new_state
    }
}

impl Debug for RegexVec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "RegexVec({})", self.stats())
    }
}

fn iter_state(
    rx_sets: &VecHashCons,
    state: StateID,
) -> impl Iterator<Item = (LexemeIdx, ExprRef)> + '_ {
    let lst = rx_sets.get(state.as_u32());
    (0..lst.len()).step_by(2).map(move |idx| {
        (
            LexemeIdx::new(lst[idx] as usize),
            ExprRef::new(lst[idx + 1]),
        )
    })
}

// #[test]
// fn test_fuel() {
//     let mut rx = RegexVec::new_single("a(bc+|b[eh])g|.h").unwrap();
//     println!("{:?}", rx);
//     rx.set_fuel(200);
//     match_(&mut rx, "abcg");
//     assert!(!rx.has_error());
//     let mut rx = RegexVec::new_single("a(bc+|b[eh])g|.h").unwrap();
//     println!("{:?}", rx);
//     rx.set_fuel(20);
//     no_match(&mut rx, "abcg");
//     assert!(rx.has_error());
// }

#[cfg(test)]
mod tests;
