mod rollback;
mod seed;
pub use seed::{
    PreparedEarleySeed, PreparedEarleySeedError, PreparedTokenParser, PreparedTokenParserError,
};
// In this file, "Kallmeyer 2018" refers to the
// slides for "Parsing: Earley parsing", Winter 2017/2018,
// Laura Kallmeyer, Heinrich Heine Universitaet, Dusseldorf,
// https://user.phil-fak.uni-duesseldorf.de/~kallmeyer/Parsing/earley.pdf
// (Retrieved 18 Sep 2024).

use std::{
    fmt::{Debug, Display},
    hash::Hash,
    ops::Range,
    sync::{Arc, Mutex},
};

use crate::{
    earley::{grammar::ParamValue, lexer::MatchingLexemesIdx, ParamCond},
    HashMap, HashSet, Instant,
};
use anyhow::{bail, ensure, Context, Result};
use derivre::{NextByte, RegexAst, StateID};
use serde::{Deserialize, Serialize};
use toktrie::{
    parse_numeric_token, Recognizer, SimpleVob, TokEnv, TokTrie, TokenId, INVALID_TOKEN,
};

use crate::{
    api::{ParserLimits, SkipRepetition, StopReason},
    earley::{lexer::Lexer, lexerspec::LexemeClass},
    id32_type,
};

use super::{
    grammar::{CGrammar, CSymIdx, CSymbol, RhsPtr, SharedGrammar},
    lexer::{LexerResult, PreLexeme},
    lexerspec::{Lexeme, LexemeIdx, LexemeSpec, LexerSpec},
    perf::ParserPerfCounters,
    regexvec::{LexemeSet, LexerStats},
};

const TRACE: bool = false;
const DEBUG: bool = true;
pub(crate) const ITEM_TRACE: bool = false;

macro_rules! trace {
    ($($arg:tt)*) => {
        if cfg!(feature = "logging") && TRACE {
            eprintln!($($arg)*);
        }
    }
}

macro_rules! debug {
    ($($arg:tt)*) => {
        if cfg!(feature = "logging") && DEBUG {
            eprintln!($($arg)*);
        }
    }
}

macro_rules! debug_def {
    ($s:expr, $($arg:tt)*) => {
        if cfg!(feature = "logging") && DEBUG && $s.scratch.log_enabled() {
            eprintln!($($arg)*);
        }
    }
}

macro_rules! item_trace {
    ($($arg:tt)*) => {
        if ITEM_TRACE {
            eprint!("    ");
            eprintln!($($arg)*);
        }
    }
}

mod advance;
mod agenda;
mod bias;
mod capture;
mod force;
mod row;
mod scan;
mod speculation;
mod token;
mod validate;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct Item {
    data: u64,
}

#[derive(Clone)]
struct SavedParserState {
    lexer_stack_length: usize,
}

#[derive(Debug, Default, Serialize, Deserialize, Clone)]
pub struct ParserStats {
    pub compute_time_us: u64,
    pub rows: usize,
    pub cached_rows: usize,
    pub all_items: usize,
    pub lexer_cost: u64,
    pub slices_applied: usize,
    pub trie_nodes_walked: usize,

    pub definitive_bytes: usize,
    pub lexer_ops: usize,
    pub num_lex_errors: usize,
    pub num_lexemes: usize,
}

#[derive(Debug, Default, Clone)]
pub struct ParserMetrics {
    pub message: String,
    pub slicer_leftover_us: usize,
}

impl ParserStats {
    pub fn delta(&self, previous: &ParserStats) -> ParserStats {
        ParserStats {
            rows: self.rows.saturating_sub(previous.rows),
            cached_rows: self.cached_rows.saturating_sub(previous.cached_rows),
            definitive_bytes: self
                .definitive_bytes
                .saturating_sub(previous.definitive_bytes),
            lexer_ops: self.lexer_ops.saturating_sub(previous.lexer_ops),
            num_lexemes: self.num_lexemes.saturating_sub(previous.num_lexemes),
            num_lex_errors: self.num_lex_errors.saturating_sub(previous.num_lex_errors),
            all_items: self.all_items.saturating_sub(previous.all_items),
            lexer_cost: self.lexer_cost.saturating_sub(previous.lexer_cost),
            compute_time_us: self
                .compute_time_us
                .saturating_sub(previous.compute_time_us),
            slices_applied: self.slices_applied.saturating_sub(previous.slices_applied),
            trie_nodes_walked: self
                .trie_nodes_walked
                .saturating_sub(previous.trie_nodes_walked),
        }
    }

    pub fn max(&self, other: &ParserStats) -> ParserStats {
        ParserStats {
            rows: self.rows.max(other.rows),
            cached_rows: self.cached_rows.max(other.cached_rows),
            definitive_bytes: self.definitive_bytes.max(other.definitive_bytes),
            lexer_ops: self.lexer_ops.max(other.lexer_ops),
            num_lexemes: self.num_lexemes.max(other.num_lexemes),
            num_lex_errors: self.num_lex_errors.max(other.num_lex_errors),
            all_items: self.all_items.max(other.all_items),
            lexer_cost: self.lexer_cost.max(other.lexer_cost),
            compute_time_us: self.compute_time_us.max(other.compute_time_us),
            slices_applied: self.slices_applied.max(other.slices_applied),
            trie_nodes_walked: self.trie_nodes_walked.max(other.trie_nodes_walked),
        }
    }
}

impl Display for ParserStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", serde_json::to_string_pretty(self).unwrap())
    }
}

id32_type!(GrammarStackPtr);

#[derive(Clone, Copy, Debug)]
struct GrammarStackNode {
    back_ptr: GrammarStackPtr,
    token_horizon: u32,
    grammar_id: LexemeClass,
    start_item: Item,
    start_item_idx: usize,
}

impl GrammarStackNode {
    fn root() -> Self {
        Self {
            back_ptr: GrammarStackPtr::new(0),
            token_horizon: u32::MAX,
            grammar_id: LexemeClass::ROOT,
            start_item: Item::new(RhsPtr::from_index(0), 0),
            start_item_idx: 0,
        }
    }
}
fn seed_predictions<E>(
    grammar: &CGrammar,
    mut emit: impl FnMut(Item, ParamValue) -> Result<(), E>,
) -> Result<(), E> {
    for rule in grammar.rules_of(grammar.start()) {
        emit(Item::new(*rule, 0), ParamValue::default())?;
    }
    Ok(())
}

// In this, code a "Row" is what is usually called an Earley set in the literature.
// The term "row" comes from Kallmeyer 2018, which uses a chart parsing algorithm
// in which the rows are Earley sets.
#[derive(Clone, Copy)]
struct Row {
    first_item: u32,
    last_item: u32,

    grammar_stack_ptr: GrammarStackPtr,

    // The lexer state below only allows certain lexemes.
    // The allowed lexemes (aka acceptable
    // lexemes, aka relevant lexemes) are those which the recognizer
    // will accept in the next row.  They are all and only those lexemes
    // which can lead to a successful parse.
    lexer_start_state: StateID,

    lexeme_idx: MatchingLexemesIdx,
}

impl Row {
    fn item_indices(&self) -> Range<usize> {
        self.first_item as usize..self.last_item as usize
    }
}

// In this code, an "Item" is what is called in the literature, an
// "Earley item".
impl Item {
    #[allow(dead_code)]
    const NULL: Self = Item { data: 0 };

    fn new(rule: RhsPtr, start: usize) -> Self {
        Item {
            data: rule.as_index() as u64 | ((start as u64) << 32),
        }
    }

    // this is dot position
    fn rhs_ptr(&self) -> RhsPtr {
        RhsPtr::from_index(self.data as u32)
    }

    fn start_pos(&self) -> usize {
        (self.data >> 32) as usize
    }

    fn advance_dot(&self) -> Self {
        Item {
            data: self.data + 1,
        }
    }

    fn rewind_dot(&self) -> Self {
        Item {
            data: self.data - 1,
        }
    }
}

impl Debug for Item {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let rule = self.rhs_ptr();
        write!(f, "Item(rhs={} @{})", rule.as_index(), self.start_pos())
    }
}

// This structure implements the Earley table, and in particular working data
// used when processing a row.
#[derive(Clone)]
struct Scratch {
    grammar: SharedGrammar,

    // The current "working row"
    row_start: usize,
    row_end: usize,

    // these two are not really "scratch" - they are just here for convenience
    // grammar_stack only grows, until the trie is finished
    items: Vec<Item>,
    item_args: Vec<ParamValue>,
    grammar_stack: Vec<GrammarStackNode>,

    push_allowed_grammar_ids: SimpleVob,
    push_allowed_lexemes: LexemeSet,
    push_grm_top: GrammarStackPtr,
    push_lexeme_idx: MatchingLexemesIdx,

    // Is this Earley table in "definitive" mode?
    // 'definitive' is set when the new lexeme is being 'defined',
    // as indicated by the creation of a 'struct Rowinfo' to track
    // the lexeme.  The opposite of definitive mode is "speculative"
    // mode, which is used for computing the token mask on the
    // pre-lexemes.
    definitive: bool,

    log_override: bool,

    // whether to use item_args
    parametric: bool,
}

#[derive(Clone)]
struct RowInfo {
    // TODO: possibly use u32 not usize here
    start_byte_idx: usize,
    lexeme: Lexeme,
    token_idx_start: usize,
    token_idx_stop: usize,
}

impl RowInfo {
    fn apply_token_idx(&mut self, idx: usize) {
        self.token_idx_start = self.token_idx_start.min(idx);
        self.token_idx_stop = self.token_idx_stop.max(idx);
    }

    fn set_token_idx(&mut self, idx: usize) {
        self.token_idx_start = idx;
        self.token_idx_stop = idx;
    }

    fn dbg(&self, lexer: &Lexer) -> String {
        format!(
            "token_idx: {}-{}; b:{}; {}",
            self.token_idx_start,
            self.token_idx_stop,
            self.start_byte_idx,
            lexer.dbg_lexeme(&self.lexeme),
        )
    }
}

// State transition is:
// if (next_lexeme, next_lexer_state) := lexer(top.lexer_state, next_byte) {
//     row_idx = scan(top.row_idx, next_lexeme)
//     push(LexerState { row_idx, next_byte, next_lexer_state })
// }
//
// The LLM thinks in tokens, while the parser only deals with lexemes.
// There is no easy translation between these, and the parser cannot work
// with tokens. On the other hand, forcing the LLM to deal with lexemes will increase
// token perplexity and degrade the quality of the LLM's output.
//
// The data structure used to resolve this "impedance mismatch" is a stack of 'LexerState' items.
// Tokens are broken down into single bytes when they go into this stack,
// and the bytes are assembled into lexemes by the lexer.
// The 'LexerState' items are left on the stack (unless backtracking).
//
// The stack of lexer states also manages a virtual stack of Earley sets, via the
// 'row_idx' field.  The current Earley table/stack is rows 0 through 'row_idx'.
#[derive(Clone, Copy, Debug)]
struct LexerState {
    row_idx: u32,         // Index of corresponding row (Earley set)
    lexer_state: StateID, // state after consuming 'byte'
    byte: Option<u8>,
}

#[derive(Clone)]
struct Captures {
    capture_list: Vec<(String, Vec<u8>)>,
    capture_map: HashMap<String, Vec<u8>>,
}

impl Captures {
    fn new() -> Self {
        Captures {
            capture_list: vec![],
            capture_map: HashMap::default(),
        }
    }

    fn push(&mut self, cap: (String, Vec<u8>)) {
        let (name, bytes) = cap;
        if !agenda::capture_changed(
            &name,
            &bytes,
            self.capture_map.get(&name).map(Vec::as_slice),
        ) {
            return;
        }
        self.capture_list.push((name.clone(), bytes.clone()));
        self.capture_map.insert(name, bytes);
    }
}

#[derive(Clone)]
struct ParserState {
    grammar: SharedGrammar,
    tok_env: TokEnv,
    scratch: Scratch,
    trie_lexer_stack: usize,
    trie_grammar_stack: usize,
    captures: Captures,
    special_token_marker_token: TokenId,

    // These are updated also in speculative mode.
    // Both are stacks only in the sense that items can be popped on backtracking
    // (when walking the token trie). Otherwise, they represent the full parsing
    // history - items are not popped in definitive mode.
    lexer_stack: Vec<LexerState>,
    lexer_stack_top_eos: bool,
    lexer_stack_flush_position: usize,
    rows: Vec<Row>,
    rows_valid_end: usize,

    trace_byte_stack: Vec<u8>,
    trace_stats0: ParserStats,
    trace_start: Instant,

    // These are only updated in definitive mode.
    row_infos: Vec<RowInfo>,
    token_idx: usize,
    bytes: Vec<u8>,
    // use u32 to save space
    byte_to_token_idx: Vec<u32>,

    last_force_bytes_len: usize,

    stats: ParserStats,
    perf_counters: Arc<ParserPerfCounters>,
    limits: ParserLimits,
    metrics: ParserMetrics,
    max_all_items: usize,
    parser_error: Option<String>,
    backtrack_byte_count: usize,

    // Cache for compute_bias - avoids recomputing identical masks when lexer state hasn't changed
    // (common in long lexemes, e.g. the interior of JSON strings)
    bias_cache: Option<BiasCache>,

    shared_box: Box<SharedState>,

    // Last so the test witness is cloned after payload fields and retired
    // after them. Production layout and clone behavior are unchanged.
    #[cfg(test)]
    copy_witness: copy_tests::CloneWitness,
}

#[derive(Clone)]
struct BiasCache {
    lexer_state: StateID,
    row_idx: u32,
    has_pending_lexeme_bytes: bool,
    mask: SimpleVob,
}

#[derive(Clone, Default)]
struct SharedState {
    lexer_opt: Option<Lexer>,
}

impl SharedState {
    #[inline(always)]
    fn lexer_mut(&mut self) -> &mut Lexer {
        self.lexer_opt.as_mut().unwrap()
    }

    #[inline(always)]
    fn lexer(&self) -> &Lexer {
        self.lexer_opt.as_ref().unwrap()
    }
}

#[derive(Clone)]
pub struct Parser {
    shared: Arc<Mutex<Box<SharedState>>>,
    state: ParserState,
}

impl Scratch {
    fn new(grammar: SharedGrammar) -> Self {
        let lexemes = grammar.lexer_spec().alloc_lexeme_set();
        let grammars = grammar.lexer_spec().alloc_grammar_set();
        Self::from_masks(grammar, lexemes, grammars)
    }
    fn from_masks(grammar: SharedGrammar, lexemes: LexemeSet, grammars: SimpleVob) -> Self {
        Scratch {
            push_allowed_lexemes: lexemes,
            push_allowed_grammar_ids: grammars,
            push_grm_top: GrammarStackPtr::new(0),
            push_lexeme_idx: MatchingLexemesIdx::Single(LexemeIdx::new(0)),
            parametric: grammar.parametric(),
            grammar,
            row_start: 0,
            row_end: 0,
            items: vec![],
            item_args: vec![],
            grammar_stack: vec![],
            definitive: true,
            log_override: false,
        }
    }

    fn log_enabled(&self) -> bool {
        self.definitive || self.log_override
    }

    // Set current working Earley to empty set
    // The set backing data is at `pos`
    fn new_row(&mut self, pos: usize) {
        self.row_start = pos;
        self.row_end = pos;
    }

    // Number of items in the current working Earley set
    fn row_len(&self) -> usize {
        self.row_end - self.row_start
    }

    // Add a new row to the Earley table.  It will be the
    // current, working, row.
    fn work_row(&self, lexer_start_state: StateID) -> Row {
        Row {
            first_item: self.row_start as u32,
            last_item: self.row_end as u32,
            grammar_stack_ptr: self.push_grm_top,
            lexer_start_state,
            lexeme_idx: self.push_lexeme_idx,
        }
    }

    // Make sure there is enough space in the Earley table,
    // usually in preparation for adding Earley items.
    #[inline(always)]
    fn ensure_items(&mut self, n: usize) {
        let k = n.saturating_sub(self.items.len());
        self.items.reserve(k);
        if self.parametric {
            self.item_args.reserve(k);
        }
    }

    fn push_grammar_stack(&mut self, node: GrammarStackNode) {
        if self.log_enabled() {
            debug!("push_grammar_stack: {:?}", node);
        }
        self.put_grammar_stack(node);
    }

    fn put_grammar_stack(&mut self, node: GrammarStackNode) {
        let ptr = GrammarStackPtr::new(self.grammar_stack.len());
        self.grammar_stack.push(node);
        self.push_grm_top = ptr;
    }

    fn just_add_idx(&mut self, item: Item, src_item_idx: usize, info: &str) {
        let arg = if self.parametric {
            self.item_args[src_item_idx]
        } else {
            ParamValue::default()
        };
        self.just_add(item, arg, info)
    }

    // Add a new Earley item with default values to the Earley table.  It is
    // "just" added in the sense that no checks are performed, except the one
    // that ensures there is enough space in the table.  The other checks are
    // assumed to be unnecessary or to have been performed.  For example, it
    // is assumed the caller knows that this Earley item will be unique.
    #[inline(always)]
    fn just_add(&mut self, item: Item, param: ParamValue, info: &str) {
        self.put_item(item, param);
        if self.log_enabled() {
            debug!(
                "      addu: {} ({}) ::{}",
                self.item_to_string(self.row_end),
                info,
                param
            );
        }
        self.row_end += 1;
    }

    fn put_item(&mut self, item: Item, param: ParamValue) {
        if self.items.len() == self.row_end {
            self.items.push(item);
        } else {
            self.items[self.row_end] = item;
        }
        if self.parametric {
            if self.item_args.len() == self.row_end {
                self.item_args.push(param);
            } else {
                self.item_args[self.row_end] = param;
            }
        }
    }
    fn contains_item_arg(&self, item: Item, param: ParamValue) -> bool {
        if self.parametric {
            (self.row_start..self.row_end)
                .any(|idx| self.items[idx] == item && self.item_args[idx] == param)
        } else {
            self.find_item(item).is_some()
        }
    }

    // Find 'item' in the current, working, row.
    #[inline(always)]
    fn find_item(&self, item: Item) -> Option<usize> {
        self.items[self.row_start..self.row_end]
            .iter()
            .position(|&x| x == item)
            .map(|x| x + self.row_start)
    }

    #[inline(always)]
    fn add_unique(&mut self, item: Item, origin_item_idx: usize, info: &str) {
        if self.parametric {
            self.add_unique_arg(item, info, self.item_args[origin_item_idx])
        } else {
            self.add_unique_arg(item, info, ParamValue::default())
        }
    }

    // Ensure that Earley table 'self' contains
    // Earley item 'item'.  That is, look for 'item' in 'self',
    // and add 'item' to 'self' if it is not there already.
    #[inline(always)]
    fn add_unique_arg(&mut self, item: Item, info: &str, param: ParamValue) {
        if !self.contains_item_arg(item, param) {
            self.just_add(item, param, info);
        }
    }

    // Write item at index 'idx' as a string.
    fn item_to_string(&self, idx: usize) -> String {
        item_to_string(
            &self.grammar,
            &self.items[idx],
            self.item_args.get(idx).copied().unwrap_or_default(),
        )
    }
}

macro_rules! ensure_internal {
    ($cond:expr, $msg:expr) => {
        ensure!($cond, "Internal error: {}", $msg)
    };
}

impl ParserState {
    // Create a new state for an empty parser.
    // The parser starts in definitive mode.
    fn new(
        tok_env: TokEnv,
        grammar: SharedGrammar,
        mut limits: ParserLimits,
        perf_counters: Arc<ParserPerfCounters>,
    ) -> Result<(Self, Lexer)> {
        let mut lexer = Lexer::from(grammar.lexer_spec(), &mut limits, true)?;
        if limits.precompute_large_lexemes {
            let t0 = crate::Instant::now();
            lexer.prepare_large_lexemes(tok_env.tok_trie(), &limits)?;
            perf_counters.precompute.record(t0.elapsed());
        }
        let scratch = Scratch::new(grammar.clone());
        let lexer_state = lexer.a_dead_state(); // placeholder
        let special_marker_token = bias::marker_token(tok_env.tok_trie());
        let mut r = ParserState {
            grammar,
            tok_env,
            special_token_marker_token: special_marker_token,
            trie_lexer_stack: usize::MAX,
            rows: vec![],
            rows_valid_end: 0,
            row_infos: vec![],
            captures: Captures::new(),
            scratch,
            stats: ParserStats::default(),
            metrics: ParserMetrics::default(),
            trace_stats0: ParserStats::default(),
            trace_byte_stack: vec![],
            trace_start: Instant::now(),
            token_idx: 0,
            byte_to_token_idx: vec![],
            bytes: vec![],
            last_force_bytes_len: usize::MAX,
            max_all_items: usize::MAX,
            limits,
            backtrack_byte_count: 0,
            lexer_stack_top_eos: false,
            lexer_stack_flush_position: 0,
            lexer_stack: vec![LexerState {
                row_idx: 0,
                lexer_state,
                byte: None,
            }],
            trie_grammar_stack: 0,
            parser_error: None,
            bias_cache: None,
            shared_box: Box::new(SharedState {
                lexer_opt: Some(lexer),
            }),
            perf_counters,
            #[cfg(test)]
            copy_witness: copy_tests::CloneWitness::default(),
        };

        r.scratch.grammar_stack.push(GrammarStackNode::root());
        seed_predictions(&r.grammar, |item, param| {
            r.scratch.add_unique_arg(item, "init", param);
            Ok::<(), std::convert::Infallible>(())
        })
        .unwrap();
        debug!("initial push");
        let _ = r.push_row(0, &Lexeme::bogus());
        r.lexer().check_error().context("initial parser row")?;
        ensure_internal!(
            r.num_rows() == 1 && r.rows.len() == 1,
            "initial push failed"
        );
        assert!(r.lexer_stack.len() == 1);

        // Set the correct initial lexer state

        if !r.lexer_spec().allow_initial_skip {
            // Disallow initial SKIP if asked to.
            // This is done, for example, we are trying to force
            // the generation of JSON to start.
            let skip_id = r.lexer_spec().skip_id(LexemeClass::ROOT);
            let mut possible = r
                .lexer()
                .possible_lexemes(r.rows[0].lexer_start_state)
                .clone();
            possible.remove(skip_id);
            let new_state = r.lexer_mut().start_state(&possible);
            r.lexer().check_error().context("initial skip selection")?;
            r.rows[0].lexer_start_state = new_state;
            debug!(
                "disallowing initial SKIP; {}",
                r.allowed_lexemes_dbg(new_state)
            );
        }

        let state = r.rows[0].lexer_start_state;
        r.lexer_stack[0].lexer_state = state;
        r.assert_definitive();

        let lexer = std::mem::take(&mut r.shared_box).lexer_opt.unwrap();

        r.stats.lexer_cost = lexer.dfa.total_fuel_spent();

        Ok((r, lexer))
    }

    #[inline(always)]
    fn lexer(&self) -> &Lexer {
        self.shared_box.lexer()
    }

    #[inline(always)]
    fn lexer_mut(&mut self) -> &mut Lexer {
        self.shared_box.lexer_mut()
    }

    fn with_items_limit<T>(
        &mut self,
        limit: usize,
        lbl: &str,
        f: impl FnOnce(&mut Self) -> T,
    ) -> T {
        self.max_all_items = self.stats.all_items + limit;

        let r = f(self);

        if self.stats.all_items > self.max_all_items && self.parser_error.is_none() {
            self.parser_error = Some(format!(
                "Too many items (limit {limit}; {lbl}); try avoiding single-byte/short lexemes"
            ));
        }

        self.max_all_items = usize::MAX;

        r
    }

    fn compute_bias(&mut self, computer: &dyn BiasComputer, start: &[u8]) -> SimpleVob {
        let t0 = Instant::now();

        // Check cache - only valid when start is empty (common case)
        if start.is_empty() {
            let curr_state = self.lexer_state();
            let has_pending = self.has_pending_lexeme_bytes();
            if let Some(ref cache) = self.bias_cache {
                if bias::Key::new(curr_state, has_pending).matches(
                    cache.lexer_state,
                    cache.row_idx,
                    cache.has_pending_lexeme_bytes,
                ) {
                    // Cache hit - return cloned mask
                    let d = t0.elapsed();
                    self.stats.compute_time_us += d.as_micros() as u64;
                    self.perf_counters.compute_bias.record(d);
                    return cache.mask.clone();
                }
            }
        }

        let limits = self.limits.clone();
        let dfa = &mut self.lexer_mut().dfa;
        dfa.set_fuel(limits.step_lexer_fuel);
        dfa.set_max_states(limits.max_lexer_states);

        let mut set = self.with_items_limit(limits.step_max_items, "mask", |state| {
            let mut r = ParserRecognizer { state };
            computer.compute_bias(&mut r, start)
        });

        self.stats.lexer_cost = self.lexer().dfa.total_fuel_spent();

        bias::finish(self, &mut set, start).expect("ordinary mask finalization");

        // Update cache when start is empty
        if start.is_empty() {
            let curr_state = self.lexer_state();
            self.bias_cache = Some(BiasCache {
                lexer_state: curr_state.lexer_state,
                row_idx: curr_state.row_idx,
                has_pending_lexeme_bytes: self.has_pending_lexeme_bytes(),
                mask: set.clone(),
            });
        }

        let d = t0.elapsed();
        self.stats.compute_time_us += d.as_micros() as u64;
        self.perf_counters.compute_bias.record(d);

        set
    }

    fn after_dots(&self) -> impl Iterator<Item = RhsPtr> + '_ {
        self.curr_row()
            .item_indices()
            .map(|i| self.scratch.items[i].rhs_ptr())
    }

    fn after_dots_symdata(&self) -> impl Iterator<Item = &CSymbol> + '_ {
        self.after_dots().map(|pos| self.grammar.sym_data_dot(pos))
    }

    fn can_advance_inner(&self) -> bool {
        speculation::can_advance(&self.grammar, &self.scratch, self.curr_row())
    }

    pub fn can_advance(&self) -> bool {
        self.has_pending_lexeme_bytes() || self.can_advance_inner()
    }

    pub fn has_pending_lexeme_bytes(&self) -> bool {
        speculation::pending(&self.lexer_stack)
    }

    // Does the parse succeed in this Earley set?
    // That is, does this Earley set contain a completed
    // start rule?
    fn row_is_accepting(&self) -> bool {
        speculation::accepting(&self.grammar, &self.scratch, self.curr_row())
    }

    pub fn lexer_allows_eos(&mut self) -> bool {
        if self.has_pending_lexeme_bytes() {
            let lexer_state = self.lexer_state().lexer_state;
            self.lexer_mut().allows_eos(lexer_state)
        } else {
            // empty lexemes are not allowed
            false
        }
    }

    fn item_to_string(&self, idx: usize) -> String {
        self.scratch.item_to_string(idx)
    }

    fn print_row(&self, row_idx: usize) {
        let row = &self.rows[row_idx];
        println!(
            "row {}; lexer_stack={} top_state={:?}",
            row_idx,
            self.lexer_stack.len(),
            self.lexer_stack.last().unwrap().lexer_state
        );

        println!(
            "  allowed: {}",
            self.allowed_lexemes_dbg(row.lexer_start_state)
        );

        if row_idx < self.row_infos.len() {
            let info = &self.row_infos[row_idx];
            if info.lexeme.is_bogus() {
                println!("  lexeme: placeholder");
            } else {
                println!("  lexeme: {}", self.lexer().dbg_lexeme(&info.lexeme));
            }
        } else {
            println!("  speculative");
        }
        for i in row.item_indices() {
            println!("  {}", self.item_to_string(i));
        }
    }

    #[inline(always)]
    fn lexer_state(&self) -> LexerState {
        self.lexer_stack[self.lexer_stack.len() - 1]
    }

    /// Current size of the Earley table -- that is,
    /// the number of Earley sets.
    #[inline(always)]
    pub fn num_rows(&self) -> usize {
        // The number of rows is taken, not from the physical Earley table,
        // but from the virtual Earley stack kept in the lexer state.
        self.lexer_state().row_idx as usize + 1
    }

    #[inline(always)]
    fn pop_lexer_states(&mut self, n: usize) {
        self.lexer_stack
            .truncate(self.lexer_stack.len().saturating_sub(n));
    }

    #[allow(dead_code)]
    pub fn print_stats(&mut self) {
        println!("stats: {:?}", self.stats);
        self.stats = ParserStats::default();
    }

    fn assert_definitive_inner(&self) {
        assert!(self.scratch.definitive);
        assert!(self.backtrack_byte_count == 0);
        if self.num_rows() != self.row_infos.len() {
            panic!(
                "num_rows={} row_infos={}",
                self.num_rows(),
                self.row_infos.len()
            );
        }
    }

    fn assert_definitive(&self) {
        self.assert_definitive_inner();

        if self.lexer_spec().can_rollback() {
            self.check_lexer_bytes_invariant();
        }
    }

    pub fn get_bytes(&self) -> &[u8] {
        &self.bytes
    }

    fn item_lhs(&self, item: &Item) -> CSymIdx {
        self.grammar.sym_idx_lhs(item.rhs_ptr())
    }

    #[allow(dead_code)]
    fn item_sym_data(&self, item: &Item) -> &CSymbol {
        self.grammar.sym_data(self.item_lhs(item))
    }

    fn hidden_start(&self, lexer: &mut Lexer) -> usize {
        let lexer_state = self.lexer_state().lexer_state;
        let hidden_len = lexer.possible_hidden_len(lexer_state);
        if hidden_len == 0 {
            return usize::MAX;
        }
        let last_lexeme_visible_len = self.curr_row_bytes().len() - hidden_len;
        let prefix_len = self.row_infos[self.num_rows() - 1].start_byte_idx;
        prefix_len + last_lexeme_visible_len
    }

    pub fn temperature(&self) -> Option<f32> {
        let mut temp = -1000.0f32;
        for data in self.after_dots_symdata() {
            if data.is_terminal {
                temp = temp.max(data.props.temperature);
            }
        }
        if temp < 0.00000001 {
            None
        } else {
            Some(temp)
        }
    }

    pub fn rollback(&mut self, n_bytes: usize) -> Result<()> {
        debug!("rollback: {} bytes", n_bytes);
        ensure!(self.parser_error.is_none(), "rollback: parser error");
        self.assert_definitive();
        rollback::run(rollback::Fields {
            byte_to_token_idx: &mut self.byte_to_token_idx,
            bytes: &mut self.bytes,
            lexer_stack: &mut self.lexer_stack,
            row_infos: &mut self.row_infos,
            token_idx: &mut self.token_idx,
            last_force_bytes_len: &mut self.last_force_bytes_len,
            lexer_stack_top_eos: &mut self.lexer_stack_top_eos,
            rows_valid_end: &mut self.rows_valid_end,
        }, n_bytes)?;

        self.assert_definitive();

        Ok(())
    }

    pub fn validate_tokens(&mut self, tokens: &[TokenId]) -> usize {
        self.assert_definitive();
        self.run_speculative("validate_tokens", |state| {
            state.scratch.log_override = true;
            validate::run(state, tokens).unwrap()
        })
    }

    fn add_numeric_token(&mut self, idx: LexemeIdx, tok_bytes: &[u8]) -> Result<()> {
        token::add_numeric(self, idx, tok_bytes)
    }

    fn flush_and_check_numeric(&mut self, tok_id: TokenId) -> Option<LexemeIdx> {
        token::numeric(self, tok_id).unwrap()
    }

    // apply_tokens() "pushes" the bytes in 'tokens' into the lexer and parser.  It is a top-level
    // method in this file.  It is well below llguidance's top-level methods, but in the llguidance
    // LLInterpreter interface, it is called indirectly via the commit_token() method.
    pub fn apply_token(&mut self, tok_bytes: &[u8], tok_id: TokenId) -> Result<usize> {
        token::apply(self, tok_bytes, tok_id)
    }

    fn token_range_lexemes(&self) -> impl Iterator<Item = &LexemeSpec> {
        let state = self.lexer_state().lexer_state;
        let possible = self.lexer().possible_lexemes(state);
        self.lexer_spec().iter_token_range_lexemes(possible)
    }

    pub fn needs_force_bytes(&self) -> bool {
        self.bytes.len() != self.last_force_bytes_len
    }

    /// force_bytes() forces bytes into the parser, definitively.
    /// They must be, at each point, the only bytes allowed by
    /// the parser.  force_bytes() returns a 'Vec' of the bytes pushed.
    pub fn force_bytes(&mut self) {
        self.assert_definitive();
        if !self.needs_force_bytes() {
            return;
        }
        self.with_items_limit(self.limits.step_max_items, "ff_tokens", |state| {
            force::run(state).expect("ordinary forced-byte worker");
        });
        self.assert_definitive();
        self.last_force_bytes_len = self.bytes.len();
    }

    fn special_pre_lexeme(&mut self, state: StateID) -> bool {
        let bytes = self.curr_row_bytes();
        let Some(index) = scan::special(
            self.lexer_spec(),
            self.lexer().possible_lexemes(state),
            &bytes,
        ) else {
            return false;
        };
        self.advance_parser(PreLexeme {
            idx: MatchingLexemesIdx::Single(index),
            byte: Some(b']'),
            byte_next_row: false,
        })
    }

    // Advance the parser or the lexer, depending on whether 'lex_result'
    // is a pre-lexeme or not.
    #[inline(always)]
    fn advance_lexer_or_parser(&mut self, lex_result: LexerResult, curr: LexerState) -> bool {
        advance::route(self, lex_result, curr).unwrap()
    }

    fn check_lexer_bytes_invariant(&self) {
        let off = if self.lexer_stack_top_eos { 2 } else { 1 };
        if self.lexer_stack.len() != self.bytes.len() + off {
            panic!(
                "lexer_stack={:?} bytes={:?} {}!={}+{off}",
                self.lexer_stack,
                String::from_utf8_lossy(&self.bytes),
                self.lexer_stack.len(),
                self.bytes.len()
            );
        }
    }

    fn trie_started_inner(&mut self, lbl: &str) {
        // debug!("trie_started: rows={} lexer={}", self.num_rows(), self.lexer_stack.len());
        self.assert_definitive();

        let saved = speculation::begin(&mut self.scratch, &self.lexer_stack);
        self.trie_lexer_stack = saved.lexer;
        self.trie_grammar_stack = saved.grammar;
        if ITEM_TRACE {
            self.trace_stats0 = self.stats.clone();
            self.trace_start = Instant::now();
            self.trace_byte_stack.clear();
            item_trace!("trie started; {}", lbl);
        }
        self.rows_valid_end = self.num_rows();
    }

    fn trie_finished_inner(&mut self) {
        // debug!("trie_finished: rows={} lexer={}", self.num_rows(), self.lexer_stack.len());
        assert!(!self.scratch.definitive);
        assert!(self.row_infos.len() <= self.num_rows());

        if ITEM_TRACE {
            let mut st = self.stats.clone();
            st.lexer_cost = self.lexer().dfa.total_fuel_spent();
            st = st.delta(&self.trace_stats0);
            st.compute_time_us = self.trace_start.elapsed().as_micros() as u64;
            item_trace!("trie finished: {}", serde_json::to_string(&st).unwrap());
            self.trace_byte_stack.clear();
        }

        speculation::finish(
            &mut self.scratch,
            &mut self.lexer_stack,
            speculation::Snapshot {
                lexer: self.trie_lexer_stack,
                grammar: self.trie_grammar_stack,
            },
        );
        self.assert_definitive();
        self.rows_valid_end = self.num_rows();
        self.scratch.log_override = false; // reset
        self.lexer_stack_flush_position = 0;
    }

    fn run_speculative<T>(&mut self, lbl: &str, f: impl FnOnce(&mut Self) -> T) -> T {
        self.trie_started_inner(lbl);
        let r = f(self);
        self.trie_finished_inner();
        r
    }

    fn is_accepting_inner(&mut self) -> bool {
        self.flush_lexer() && self.row_is_accepting()
    }

    pub fn is_accepting(&mut self) -> bool {
        self.run_speculative("is_accepting", |s| s.is_accepting_inner())
    }

    // try_push_byte_definitive() attempts to 'push' a byte (that is advance
    // the parse with 'byte') into the parse in definitive mode.
    // Returns 'false' if this is not possible.
    fn try_push_byte_definitive(&mut self, byte: Option<u8>) -> (bool, usize) {
        advance::definitive(self, byte).unwrap()
    }

    /// The current Earley set (row) as kept track of
    /// in the lexer stack.
    fn curr_row(&self) -> &Row {
        &self.rows[self.lexer_state().row_idx as usize]
    }

    fn save_state(&self) -> SavedParserState {
        SavedParserState {
            lexer_stack_length: self.lexer_stack.len(),
        }
    }

    fn restore_state(&mut self, state: SavedParserState) {
        self.lexer_stack.truncate(state.lexer_stack_length);
    }

    /// Advance the parser as if the current lexeme (if any)
    /// finished right here.
    /// Returns true if the parser was able to advance (or there were no pending bytes for a lexeme).
    fn flush_lexer(&mut self) -> bool {
        token::flush(self).unwrap()
    }

    pub fn scan_eos(&mut self) -> bool {
        token::eos(self).unwrap()
    }

    // this just copies current row
    fn scan_skip_lexeme(&mut self, lexeme: &Lexeme, skip_repetition: SkipRepetition) -> bool {
        let src = self.curr_row().item_indices();
        let n = src.len();
        if n == 0 {
            return false;
        }
        self.scratch.ensure_items(src.end + n + 10);
        self.scratch.new_row(src.end);
        self.scratch.push_lexeme_idx = lexeme.idx;

        // we'll not re-run process_agenda() for the newly added row, so save its allowed lexemes
        // (this is unless we hit max_tokens case)
        let mut lex_start = Some(self.rows[self.num_rows() - 1].lexer_start_state);

        scan::items(&mut self.scratch, src, None, |scratch, item, arg| {
            scratch.just_add(item, arg, "skip_lexeme");
            Ok::<(), std::convert::Infallible>(())
        })
        .unwrap();

        let (mut grammar_id, max_token_ptr) = self.maybe_pop_grammar_stack(lexeme.idx);
        let hit_max_tokens = max_token_ptr.is_some();

        // no process_agenda() in the normal case

        if let Some(ptr) = max_token_ptr {
            // but we have to do it if we hit the max tokens case
            self.process_max_tokens(ptr, lexeme);
            // process_agenda() will recompute push_allowed_lexemes etc
            lex_start = None;
            grammar_id =
                self.scratch.grammar_stack[self.scratch.push_grm_top.as_usize()].grammar_id;
        } else if skip_repetition == SkipRepetition::Once {
            let skip_id = self.lexer_spec().skip_id(grammar_id);
            let mut possible = self
                .shared_box
                .lexer()
                .possible_lexemes(lex_start.unwrap())
                .clone();
            possible.remove(skip_id);
            lex_start = Some(self.shared_box.lexer_mut().start_state(&possible));
        }

        // A max-token pop moves to the parent grammar, whose skip has not been consumed.
        let allow_skip = hit_max_tokens || skip_repetition == SkipRepetition::Unbounded;
        let push_res = self.just_push_row(grammar_id, lex_start, allow_skip);
        assert!(push_res);

        true
    }

    // scan() implements the version of Earley described in Kallmeyer 2018.
    // An important difference between the algorithm implemented here
    // and Kallmeyer's is that in scan(), the token scan is performed
    // first, while in Kallmeyer it is performed last.

    // Returns false if the parse is exhausted, true otherwise.

    // lexeme body only used for captures (in definitive mode)
    // and debugging (lexeme.idx used always)
    fn scan(&mut self, lexeme: &Lexeme) -> bool {
        let set = self.shared_box.lexer().lexemes_from_idx(lexeme.idx);

        let lex_spec = self.lexer_spec();
        if let Some(skip_repetition) = set.as_slice().iter().find_map(|lx| {
            let spec = lex_spec.lexeme_spec(*lx);
            spec.is_skip.then_some(spec.skip_repetition)
        }) {
            return self.scan_skip_lexeme(lexeme, skip_repetition);
        }

        let row_idx = self.num_rows() - 1;
        let items = self.rows[row_idx].item_indices();
        self.scratch.ensure_items(items.end + items.len() + 10);
        self.scratch.new_row(items.end);
        self.scratch.push_lexeme_idx = lexeme.idx;

        debug_def!(
            self,
            "  scan: {} at row={} token={}",
            self.lexer().dbg_lexeme(lexeme),
            row_idx,
            self.token_idx,
        );

        scan::items(&mut self.scratch, items, Some(set), |scratch, item, arg| {
            scratch.just_add(item, arg, "scan");
            Ok::<(), std::convert::Infallible>(())
        })
        .unwrap();

        // Perform the other inference rules on this Earley set.
        self.push_row(self.num_rows(), lexeme)
    }

    fn mk_capture(&self, var_name: &str, bytes: &[u8]) -> (String, Vec<u8>) {
        debug!(
            "      capture: {} {:?}",
            var_name,
            String::from_utf8_lossy(bytes)
        );

        let bytes = self.tok_env.tok_trie().decode_raw_to_decode(bytes);
        (var_name.to_string(), bytes)
    }

    fn process_one_capture(
        &mut self,
        lhs: CSymIdx,
        curr_idx: usize,
        lexeme: &Lexeme,
        is_lexeme: bool,
        capture_start: usize,
    ) {
        let sym_data = self.grammar.sym_data(lhs);

        debug!(
            "      process_one_capture: {} {}-{} {}",
            self.grammar.sym_name(lhs),
            capture_start,
            curr_idx,
            if is_lexeme { "lexeme" } else { "full" }
        );

        if let Some(var_name) = sym_data.props.stop_capture_name.as_ref() {
            let bytes = lexeme.hidden_bytes();
            self.captures.push(self.mk_capture(var_name, bytes));
        }

        if let Some(var_name) = sym_data.props.capture_name.as_ref() {
            let mut bytes = Vec::new();
            capture::visit(
                &self.row_infos,
                capture_start,
                curr_idx,
                lexeme,
                is_lexeme,
                false,
                |part| {
                    bytes.extend_from_slice(part);
                    Ok::<(), std::convert::Infallible>(())
                },
            )
            .unwrap();
            self.captures.push(self.mk_capture(var_name, &bytes));
        }
    }

    fn process_captures(&mut self, item: Item, curr_idx: usize, lexeme: &Lexeme, for_lexeme: bool) {
        let grammar = self.grammar.clone();
        agenda::capture_targets(
            &grammar,
            item,
            curr_idx,
            for_lexeme,
            |symbol, is_lexeme, start| {
                self.process_one_capture(symbol, curr_idx, lexeme, is_lexeme, start);
                Ok::<(), std::convert::Infallible>(())
            },
        )
        .unwrap();
    }

    #[inline(always)]
    fn process_agenda(&mut self, curr_idx: usize, lexeme: &Lexeme) {
        agenda::run(self, curr_idx, lexeme).unwrap();
    }

    #[inline(always)]
    fn just_push_row(
        &mut self,
        grammar_id: LexemeClass,
        lex_start: Option<StateID>,
        allow_skip: bool,
    ) -> bool {
        let row_len = self.scratch.row_len();

        self.stats.rows += 1;

        if row_len == 0 {
            false
        } else {
            self.stats.all_items += row_len;

            let lex_start = if let Some(l) = lex_start {
                l
            } else {
                row::add_skip(&mut self.scratch, grammar_id, allow_skip);

                self.shared_box
                    .lexer_mut()
                    .start_state(&self.scratch.push_allowed_lexemes)
            };

            debug_def!(
                self,
                "  push row: {} {:?}",
                self.allowed_lexemes_dbg(lex_start),
                grammar_id
            );

            // Add the working row to the parser state
            let idx = self.num_rows();

            let row = self.scratch.work_row(lex_start);
            row::store(&mut self.rows, idx, row);
            self.rows_valid_end = idx + 1;

            if self.scratch.definitive {
                row::store_info(&mut self.row_infos, idx, self.token_idx, self.bytes.len());
            }

            true
        }
    }

    fn process_max_tokens(&mut self, ptr: GrammarStackPtr, lexeme: &Lexeme) {
        debug_def!(self, "  process_max_tokens");
        let curr_idx = self.num_rows();
        let top = &self.scratch.grammar_stack[ptr.as_usize()];
        self.scratch.push_grm_top = top.back_ptr;
        let item = top.start_item.advance_dot();
        // remove everything from the current row
        self.scratch.row_end = self.scratch.row_start;
        self.scratch
            .just_add_idx(item, top.start_item_idx, "max_tokens");
        self.process_agenda(curr_idx, lexeme);
    }

    // push_row() does the agenda processing.  There is an agenda for
    // each Earley set (aka row).

    // Returns false if an empty Earley set is added (and therefore
    // the parse is exhausted); and true otherwise.

    // lexeme value only used for captures (in definitive mode)
    #[inline(always)]
    fn push_row(&mut self, curr_idx: usize, lexeme: &Lexeme) -> bool {
        let (grammar_id, max_token_ptr) = self.maybe_pop_grammar_stack(lexeme.idx);

        self.process_agenda(curr_idx, lexeme);

        if let Some(ptr) = max_token_ptr {
            assert!(curr_idx == self.num_rows(), "max_tokens on first row");
            self.process_max_tokens(ptr, lexeme);
        }

        self.just_push_row(grammar_id, None, true)
    }

    fn mk_grammar_stack_node(&self, sym_data: &CSymbol, curr_idx: usize) -> GrammarStackNode {
        agenda::grammar_node(
            &self.grammar,
            &self.scratch,
            sym_data,
            curr_idx,
            self.token_idx,
        )
    }

    // when this is called, the current row has only rules with lx at the dot
    #[inline(always)]
    fn maybe_pop_grammar_stack(
        &mut self,
        lx: MatchingLexemesIdx,
    ) -> (LexemeClass, Option<GrammarStackPtr>) {
        let set = self.shared_box.lexer().lexemes_from_idx(lx);
        let top = if self.rows.is_empty() {
            GrammarStackPtr::new(0)
        } else {
            self.rows[self.num_rows() - 1].grammar_stack_ptr
        };
        scan::pop(&mut self.scratch, set, top, self.token_idx)
    }

    // curr_row_bytes() looks in the lexer stack, and returns
    // the bytes for the current row as a 'Vec'.
    #[inline(always)]
    fn curr_row_bytes(&self) -> Vec<u8> {
        let mut bytes: Vec<u8> =
            scan::row_bytes(&self.lexer_stack, (self.num_rows() - 1) as u32).collect();
        bytes.reverse();
        bytes
    }

    fn lexer_spec(&self) -> &LexerSpec {
        self.grammar.lexer_spec()
    }

    fn allowed_lexemes_dbg(&self, lex_state: StateID) -> String {
        self.lexer_spec()
            .dbg_lexeme_set(self.lexer().possible_lexemes(lex_state))
    }

    // mk_lexeme() converts a pre-lexeme for the current row into
    // a lexeme (ie., it determines the bytes that go into the lexeme), and returns it.
    #[inline(always)]
    fn mk_lexeme(&self, byte: Option<u8>, pre_lexeme: PreLexeme) -> Lexeme {
        let mut bytes = self.curr_row_bytes();
        if let Some(byte) = byte {
            bytes.push(byte);
        }

        let (hidden, is_suffix) = self.lexer().lexeme_props(pre_lexeme.idx);
        Lexeme::new(pre_lexeme.idx, bytes, hidden, is_suffix)
    }

    fn has_forced_bytes(&self, allowed_lexemes: &LexemeSet, bytes: &[u8]) -> bool {
        scan::forced(self.lexer_spec(), allowed_lexemes, bytes)
    }

    #[inline(always)]
    fn lexer_state_for_added_row(
        &mut self,
        lexeme: Lexeme,
        transition_byte: Option<u8>,
    ) -> LexerState {
        // note, that while self.rows[] is updated, the lexer stack is not
        // so the last added row is at self.num_rows(), and not self.num_rows() - 1
        let added_row = self.num_rows();
        let added_row_start_state = self.rows[added_row].lexer_start_state;

        let no_hidden = LexerState {
            row_idx: added_row as u32,
            lexer_state: self
                .shared_box
                .lexer_mut()
                .transition_start_state(added_row_start_state, transition_byte),
            byte: transition_byte,
        };

        if self.scratch.definitive {
            // save lexeme at the last row, before we mess with the stack
            self.row_infos[added_row - 1].lexeme = lexeme;
            // if there is a transition byte it means it goes to the next lexeme,
            // and thus we were overeager assigning start_byte_idx,
            // so we need to correct it
            if transition_byte.is_some() {
                let new_start = self.row_infos[added_row - 1]
                    .start_byte_idx
                    .saturating_sub(1);
                self.row_infos[added_row].start_byte_idx -= new_start;
            }
        }
        debug_def!(
            self,
            "lex: re-start {:?} (via {:?}); allowed: {}",
            no_hidden.lexer_state,
            transition_byte.map(|b| b as char),
            self.allowed_lexemes_dbg(added_row_start_state)
        );

        no_hidden
    }

    #[inline(always)]
    fn handle_hidden_bytes(
        &mut self,
        no_hidden: LexerState,
        lexeme_byte: Option<u8>,
        pre_lexeme: PreLexeme,
    ) -> bool {
        advance::hidden(self, no_hidden, lexeme_byte, pre_lexeme).unwrap()
    }

    fn lexer_stack_top(&self) -> String {
        String::from_utf8_lossy(&self.trace_byte_stack).to_string()
    }

    /// Advance the parser with given 'pre_lexeme'.
    /// On return, the lexer_state will be the state *after* consuming
    /// 'pre_lexeme'.  As a special case, a following single byte lexeme
    /// is also consumed.
    ///
    // The new lexer state will be an initial lexer states when the lexing
    // is lazy.  If the lexing was greedy, it will be an initial lexer state
    // advanced to the byte which produced the greedy lexeme.
    // This is never inlined anyways, so better make it formal
    #[inline(never)]
    fn advance_parser(&mut self, pre_lexeme: PreLexeme) -> bool {
        advance::run(self, pre_lexeme).unwrap()
    }
}

pub struct ParserRecognizer<'a> {
    state: &'a mut ParserState,
}

impl ParserRecognizer<'_> {
    pub fn lexer_mut(&mut self) -> &mut Lexer {
        self.state.lexer_mut()
    }
    pub fn lexer(&self) -> &Lexer {
        self.state.lexer()
    }
    pub fn lexer_state(&self) -> StateID {
        self.state.lexer_state().lexer_state
    }
    pub fn stats_mut(&mut self) -> &mut ParserStats {
        &mut self.state.stats
    }
    pub fn metrics_mut(&mut self) -> &mut ParserMetrics {
        &mut self.state.metrics
    }
}

pub trait BiasComputer: Send + Sync {
    fn compute_bias(&self, rec: &mut ParserRecognizer<'_>, start: &[u8]) -> SimpleVob;
    fn trie(&self) -> &TokTrie;
}

// Processing of the parser and the lexer is heavily interlocked.
// The 'Recognizer' trait is used as the interface for this.
// See the documentation for TokTrie in README.md and toktrie.md:
// https://github.com/microsoft/llguidance/blob/main/toktrie/README.md
// and
// https://github.com/microsoft/llguidance/blob/main/docs/toktrie.md .
impl Recognizer for ParserRecognizer<'_> {
    #[inline(always)]
    fn pop_bytes(&mut self, num: usize) {
        if ITEM_TRACE {
            self.state
                .trace_byte_stack
                .truncate(self.state.trace_byte_stack.len() - num);
        }
        self.state.pop_lexer_states(num);
    }

    // For this Earley parser, collapse does nothing -- it is a no-op
    fn collapse(&mut self) {
        // This actually means "commit" - can no longer backtrack past this point.
        // However, this parser ignores it.
    }

    fn trie_started(&mut self, lbl: &str) {
        self.state.trie_started_inner(lbl);
    }

    fn trie_finished(&mut self) {
        self.state.trie_finished_inner();
    }

    // try_push_byte() is the "speculative" version of try_push_byte_definitive().
    // It attempts to advance the lexer and parser one byte.  It returns true
    // if it succeeds in doing this, true otherwise.  It is often invoked indirectly by the
    // add_bias_inner() method of TokTrie.  In this file, that can happen via the add_bias()
    // and the various compute_bias() methods.
    #[inline(always)]
    fn try_push_byte(&mut self, byte: u8) -> bool {
        let stats = false;

        let lexer_logging = false;
        let curr = self.state.lexer_state();
        let res = self
            .state
            .lexer_mut()
            .advance(curr.lexer_state, byte, lexer_logging);

        if ITEM_TRACE {
            self.state.trace_byte_stack.push(byte);
        }

        if stats {
            // this is always true (not only with stats) but checking it has significant cost
            assert!(!self.state.scratch.definitive);

            self.state.stats.lexer_ops += 1;
            match res {
                LexerResult::State(_, _) => {}
                LexerResult::Error => self.state.stats.num_lex_errors += 1,
                LexerResult::Lexeme(_) | LexerResult::SpecialToken(_) => {
                    self.state.stats.num_lexemes += 1
                }
            }
        }

        let r = self.state.advance_lexer_or_parser(res, curr);

        if ITEM_TRACE && !r {
            self.state.trace_byte_stack.pop();
        }

        r
    }

    fn save_stats(&mut self, nodes_walked: usize) {
        self.state.stats.trie_nodes_walked += nodes_walked;
    }
}

fn item_to_string(g: &CGrammar, item: &Item, param: ParamValue) -> String {
    let mut r = format!(
        "{} @{}",
        g.rule_to_string(item.rhs_ptr(), &ParamCond::True),
        item.start_pos()
    );
    if !param.is_default() {
        r.push_str(&format!(" ::{param}"));
    }
    r
}

#[derive(Clone, Debug)]
pub enum ParserError {
    LexerError(String),
    ParserError(String),
}

impl Display for ParserError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LexerError(message) => write!(f, "lexer error: {message}"),
            Self::ParserError(message) => write!(f, "parser error: {message}"),
        }
    }
}

impl std::error::Error for ParserError {}

impl ParserError {
    pub fn stop_reason(&self) -> StopReason {
        match self {
            ParserError::LexerError(_) => StopReason::LexerTooComplex,
            ParserError::ParserError(_) => StopReason::ParserTooComplex,
        }
    }

    pub fn message(&self) -> String {
        match self {
            ParserError::LexerError(s) => format!("lexer error: {s}"),
            ParserError::ParserError(s) => format!("parser error: {s}"),
        }
    }
}

impl Parser {
    pub fn new(
        tok_env: TokEnv,
        grammar: SharedGrammar,
        limits: ParserLimits,
        perf_counters: Arc<ParserPerfCounters>,
    ) -> Result<Self> {
        let (state, lexer) = ParserState::new(tok_env, grammar, limits, perf_counters)?;
        let shared = Arc::new(Mutex::new(Box::new(SharedState {
            lexer_opt: Some(lexer),
        })));
        Ok(Parser { shared, state })
    }

    /// This is a top-level method in this file.  It is called by compute_mask_inner()
    /// in TokenParser in tokenparser.rs.  It is used by the compute_mask() method of
    /// the LLInterpreter interface.
    pub fn compute_bias(&mut self, computer: &dyn BiasComputer, start: &[u8]) -> SimpleVob {
        self.with_shared(|state| state.compute_bias(computer, start))
    }

    pub fn captures(&self) -> &[(String, Vec<u8>)] {
        &self.state.captures.capture_list
    }

    pub fn get_capture(&self, name: &str) -> Option<&[u8]> {
        self.state.captures.capture_map.get(name).map(|v| &v[..])
    }

    pub fn stats(&self) -> &ParserStats {
        &self.state.stats
    }

    #[inline(always)]
    pub fn perf_counters(&self) -> &ParserPerfCounters {
        &self.state.perf_counters
    }

    pub fn metrics_mut(&mut self) -> &mut ParserMetrics {
        &mut self.state.metrics
    }

    // The "hidden" feature must be supported for historical reasons.
    // It is used for 'gen(stop="foo')'.  The result of this 'gen'
    // must not include 'foo', even though the LLM generated 'foo'.
    // The bytes in 'foo' are therefore said to be "hidden".
    pub fn hidden_start(&self) -> usize {
        let mut shared = self.shared.lock().unwrap();
        self.state.hidden_start(shared.lexer_mut())
    }

    pub fn lexer_stats(&self) -> LexerStats {
        self.shared.lock().unwrap().lexer().dfa.stats()
    }

    pub fn get_error(&self) -> Option<ParserError> {
        let shared = self.shared.lock().unwrap();
        self.error_from_shared(&shared)
    }

    // Diagnostic only: a busy or poisoned lexer has no safely readable cause.
    // In particular, do not panic again while reporting a caught callback panic.
    pub(crate) fn try_get_error(&self) -> Option<ParserError> {
        let shared = self.shared.try_lock().ok()?;
        self.error_from_shared(&shared)
    }

    fn error_from_shared(&self, shared: &SharedState) -> Option<ParserError> {
        if let Some(e) = shared.lexer().dfa.get_error() {
            return Some(ParserError::LexerError(e));
        }
        if let Some(e) = &self.state.parser_error {
            return Some(ParserError::ParserError(e.clone()));
        }
        None
    }

    pub fn with_recognizer<T>(&mut self, f: impl FnOnce(&mut ParserRecognizer) -> T) -> T {
        self.with_shared(|state| {
            let mut rec = ParserRecognizer { state };
            f(&mut rec)
        })
    }

    pub fn get_bytes(&self) -> &[u8] {
        self.state.get_bytes()
    }

    pub fn force_bytes(&mut self) -> &[u8] {
        if !self.state.needs_force_bytes() {
            self.currently_forced_bytes()
        } else {
            let t0 = Instant::now();
            let prev_len = self.currently_forced_bytes().len();
            self.with_shared(|state| state.force_bytes());
            let r = self.currently_forced_bytes();
            if r.len() > prev_len {
                self.state.perf_counters.force_bytes.record(t0.elapsed());
            } else {
                self.state
                    .perf_counters
                    .force_bytes_empty
                    .record(t0.elapsed());
            }
            r
        }
    }

    pub fn scan_eos(&mut self) -> bool {
        self.with_shared(|state| state.scan_eos())
    }

    pub fn grammar_warnings(&mut self) -> Vec<String> {
        self.with_shared(|state| state.lexer_spec().render_warnings())
    }

    pub(crate) fn apply_forced(&mut self, byte_idx: usize) {
        self.state.byte_to_token_idx.resize(byte_idx, 0);
    }

    pub(crate) fn additional_backtrack(&mut self, n_bytes: usize) {
        // we can be sometimes asked to backtrack more than we have
        // in case the prompt was token-healed; see https://github.com/guidance-ai/guidance/issues/1131
        let new_len = self.state.byte_to_token_idx.len().saturating_sub(n_bytes);
        self.state.byte_to_token_idx.truncate(new_len);
    }

    pub fn apply_token(&mut self, tok_bytes: &[u8], tok_id: TokenId) -> Result<usize> {
        let r = self.with_shared(|state| state.apply_token(tok_bytes, tok_id));
        self.state.token_idx += 1;
        r
    }

    fn with_shared<T>(&mut self, f: impl FnOnce(&mut ParserState) -> T) -> T {
        let mut shared = self.shared.lock().unwrap();
        self.state.shared_box = std::mem::take(&mut *shared);
        let r = f(&mut self.state);
        *shared = std::mem::take(&mut self.state.shared_box);
        assert!(shared.lexer_opt.is_some());
        r
    }

    pub fn rollback(&mut self, n_bytes: usize) -> Result<()> {
        self.state.lexer_spec().check_rollback()?;
        self.with_shared(|state| state.rollback(n_bytes))
    }

    /// Returns how many tokens can be applied.
    pub fn validate_tokens(&mut self, tokens: &[TokenId]) -> usize {
        self.with_shared(|state| {
            let r = state.validate_tokens(tokens);
            debug!(
                "validate_tokens: {} -> {}/{}",
                state.tok_env.tok_trie().tokens_dbg(tokens),
                r,
                tokens.len()
            );
            r
        })
    }

    pub fn log_row_infos(&mut self, label: &str) {
        if cfg!(feature = "logging") && DEBUG {
            self.with_shared(|state| {
                debug!(
                    "row infos {}: token_idx: {}; applied bytes: {}/{}",
                    label,
                    state.token_idx,
                    state.byte_to_token_idx.len(),
                    state.bytes.len()
                );
                for infos in state.row_infos.iter() {
                    debug!("  {}", infos.dbg(state.lexer()));
                }
            })
        }
    }

    pub fn is_accepting(&mut self) -> bool {
        self.with_shared(|state| state.is_accepting())
    }

    pub fn currently_forced_bytes(&self) -> &[u8] {
        &self.state.bytes[self.state.byte_to_token_idx.len()..]
    }

    pub fn has_pending_lexeme_bytes(&self) -> bool {
        self.state.has_pending_lexeme_bytes()
    }

    pub fn grammar(&self) -> &CGrammar {
        &self.state.grammar
    }

    pub fn can_advance(&self) -> bool {
        self.state.can_advance()
    }

    pub fn temperature(&self) -> Option<f32> {
        self.state.temperature()
    }

    pub fn deep_clone(&self) -> Self {
        let mut copy = self.clone();
        let shared = self.shared.lock().unwrap();
        copy.shared = Arc::new(Mutex::new(shared.clone()));
        copy
    }

    pub fn test_trigger_lexer_error(&mut self) -> Result<()> {
        self.with_shared(|_state| {
            panic!("synthetic error");
        })
    }

    pub fn invalidate_bias_cache(&mut self) {
        self.state.bias_cache = None;
    }
}

#[cfg(test)]
mod construction_tests;

#[cfg(test)]
mod copy_tests;
