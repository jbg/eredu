use crate::{
    api::{LLGuidanceOptions, ParserLimits, SkipSpec},
    earley::{
        lexerspec::{LexemeClass, LexemeIdx, LexerSpec, TokenRangesDisplay},
        Grammar, ParamCond, ParamExpr, SymIdx, SymbolProps,
    },
    hashcons::{HashCons, HashId},
};
use derivre::{ParserResult as Result, ParserError, parser_error as anyhow, parser_bail as bail, parser_ensure as ensure};
use derivre::SourceHashMap as HashMap;
use derivre::{ExprRef, RegexAst};
use std::ops::RangeInclusive;
use toktrie::{TokTrie, INVALID_TOKEN};

use crate::api::{GenGrammarOptions, GenOptions, NodeProps};

const DEBUG: bool = false;
macro_rules! debug {
    ($($arg:tt)*) => {
        if cfg!(feature = "logging") && DEBUG {
            eprint!("GRM>      ");
            eprintln!($($arg)*);
        }
    };
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct NodeRef {
    idx: SymIdx,
    param_id: Option<HashId<ParamExpr>>,
    grammar_id: usize,
}

impl NodeRef {
    pub const BOGUS: NodeRef = NodeRef {
        idx: SymIdx::BOGUS,
        grammar_id: usize::MAX,
        param_id: None,
    };

    pub fn is_parametric(&self) -> bool {
        self.param_id.is_some()
    }
}

const K: usize = 4;

pub struct GrammarBuilder<'t> {
    pub(crate) grammar: Grammar,
    // this is only used for validation of NodeRef's
    // we could drop it if needed
    curr_grammar_idx: usize,
    curr_lexeme_class: LexemeClass,
    curr_start_idx: NodeRef,
    pub regex: RegexBuilder,
    tok_trie: Option<&'t TokTrie>,
    limits: ParserLimits,
    warnings: derivre::SourceHashMap<String, usize>,

    strings: HashMap<String, NodeRef>,
    at_most_cache: HashMap<(NodeRef, usize), NodeRef>,
    repeat_exact_cache: HashMap<(NodeRef, usize), NodeRef>,

    params: HashCons<ParamExpr>,
    pub(crate) funding: derivre::ParserAllocationFunding,
    self_ref: HashId<ParamExpr>,
}

pub struct GrammarResult<'t> {
    pub builder: GrammarBuilder<'t>,
    pub start_node: SymIdx,
}

pub struct RegexBuilder {
    pub(crate) spec: LexerSpec,
}

pub type RegexId = derivre::ExprRef;

impl Default for RegexBuilder {
    fn default() -> Self {
        Self::new(derivre::ParserAllocationFunding::unenforced()).expect("unenforced regex builder")
    }
}

impl RegexBuilder {
    pub fn new(funding: derivre::ParserAllocationFunding) -> Result<Self> {
        Ok(Self {
            spec: LexerSpec::new(funding)?,
        })
    }

    fn map_ids(&self, nodes: &[RegexId]) -> Result<Vec<RegexAst>> {
        let mut result = Vec::new();
        self.spec.funding.try_grow_vec(&mut result, nodes.len())?;
        for node in nodes {
            result.push(RegexAst::ExprRef(*node));
        }
        Ok(result)
    }

    pub fn add_ast(&mut self, ast: RegexAst) -> Result<RegexId> {
        self.spec.regex_builder.mk(&ast)
    }

    pub fn regex(&mut self, rx: &str) -> Result<RegexId> {
        self.spec.regex_builder.mk_regex(rx)
    }

    pub fn literal(&mut self, value: &str) -> Result<RegexId> {
        let value = self.spec.funding.try_copy_str(value)?;
        self.add_ast(RegexAst::Literal(value))
    }

    pub fn concat(&mut self, nodes: &[RegexId]) -> Result<RegexId> {
        if nodes.len() == 1 {
            return Ok(nodes[0]);
        }
        if nodes.is_empty() {
            return Ok(ExprRef::EMPTY_STRING);
        }
        let args = self.map_ids(nodes)?;
        self.add_ast(RegexAst::Concat(args))
    }

    pub fn select(&mut self, nodes: &[RegexId]) -> Result<RegexId> {
        if nodes.len() == 1 {
            return Ok(nodes[0]);
        }
        if nodes.is_empty() {
            return Ok(ExprRef::NO_MATCH);
        }
        let args = self.map_ids(nodes)?;
        self.add_ast(RegexAst::Or(args))
    }

    pub fn zero_or_more(&mut self, node: RegexId) -> Result<RegexId> {
        self.repeat(node, 0, None)
    }
    pub fn one_or_more(&mut self, node: RegexId) -> Result<RegexId> {
        self.repeat(node, 1, None)
    }
    pub fn optional(&mut self, node: RegexId) -> Result<RegexId> {
        self.repeat(node, 0, Some(1))
    }

    pub fn repeat(&mut self, node: RegexId, min: u32, max: Option<u32>) -> Result<RegexId> {
        let node = self.spec.funding.try_box(RegexAst::ExprRef(node))?;
        self.add_ast(RegexAst::Repeat(node, min, max.unwrap_or(u32::MAX)))
    }

    pub fn not(&mut self, node: RegexId) -> Result<RegexId> {
        let node = self.spec.funding.try_box(RegexAst::ExprRef(node))?;
        self.add_ast(RegexAst::Not(node))
    }

    pub fn and(&mut self, nodes: &[RegexId]) -> Result<RegexId> {
        if nodes.len() == 1 {
            return Ok(nodes[0]);
        }
        let args = self.map_ids(nodes)?;
        self.add_ast(RegexAst::And(args))
    }

    pub fn or(&mut self, nodes: &[RegexId]) -> Result<RegexId> {
        self.select(nodes)
    }
}

impl<'t> GrammarBuilder<'t> {
    pub fn new(
        tok_trie: Option<&'t TokTrie>,
        limits: ParserLimits,
        funding: derivre::ParserAllocationFunding,
    ) -> Result<Self> {
        let mut params = HashCons::new(funding.clone());
        let self_ref = params.insert(ParamExpr::SelfRef)?;
        Ok(Self {
            grammar: Grammar::new(None, funding.clone()),
            curr_grammar_idx: 0,
            curr_lexeme_class: LexemeClass::ROOT,
            curr_start_idx: NodeRef::BOGUS,
            strings: HashMap::default(),
            regex: RegexBuilder::new(funding.clone())?,
            at_most_cache: HashMap::default(),
            repeat_exact_cache: HashMap::default(),
            warnings: derivre::SourceHashMap::default(),
            limits,
            tok_trie,
            self_ref,
            params,
            funding,
        })
    }

    pub fn check_limits(&self) -> Result<()> {
        ensure!(&self.funding,
            self.regex.spec.cost() <= self.limits.initial_lexer_fuel,
            "initial lexer configuration (grammar) too big (limit for this grammar: {})",
            self.limits.initial_lexer_fuel
        );

        let size = self.grammar.num_symbols();
        ensure!(&self.funding,
            size <= self.limits.max_grammar_size,
            "grammar size (number of symbols) too big (limit for this grammar: {})",
            self.limits.max_grammar_size,
        );

        Ok(())
    }

    pub fn add_warning(&mut self, msg: &str) -> Result<()> {
        if let Some(count) = self.warnings.get_mut(msg) {
            *count += 1;
        } else {
            let message = self.funding.try_copy_str(msg)?;
            self.funding.try_insert(&mut self.warnings, message, 1)?;
        }
        Ok(())
    }

    pub fn get_warnings(&self) -> Result<Vec<(String, usize)>> {
        let mut warnings = Vec::new();
        self.funding
            .try_grow_vec(&mut warnings, self.warnings.len())?;
        for (message, count) in &self.warnings {
            let message = self.funding.try_copy_str(message)?;
            warnings.push((message, *count));
        }
        warnings.sort_by(|a, b| a.0.cmp(&b.0));
        Ok(warnings)
    }

    pub fn add_grammar(&mut self, options: LLGuidanceOptions, skip: RegexAst) -> Result<SymIdx> {
        self.add_grammar_with_skip(options, SkipSpec::unbounded(skip))
    }

    pub fn add_grammar_with_skip(
        &mut self,
        options: LLGuidanceOptions,
        skip: SkipSpec,
    ) -> Result<SymIdx> {
        self.check_limits()?;

        self.curr_lexeme_class = self.regex.spec.setup_lexeme_class_with_skip(skip)?;

        self.strings.clear();
        self.at_most_cache.clear();
        self.repeat_exact_cache.clear();

        self.curr_grammar_idx += 1;

        // We'll swap these as we add more grammars,
        // so this setting is local to the grammar
        let utf8 = !options.allow_invalid_utf8;
        self.regex.spec.regex_builder.unicode(utf8);
        self.regex.spec.regex_builder.utf8(utf8);

        // if any grammar sets it, it is inherited by the lexer
        if options.no_forcing {
            self.regex.spec.no_forcing = true;
        }
        if options.allow_initial_skip {
            self.regex.spec.allow_initial_skip = true;
        }

        // add root node
        self.curr_start_idx = self.new_node("start")?;
        self.grammar.sym_props_mut(self.curr_start_idx.idx).is_start = true;
        Ok(self.curr_start_idx.idx)
    }

    fn lexeme_to_node(&mut self, lx_id: LexemeIdx) -> Result<NodeRef> {
        Ok({
            let lname = self
                .funding
                .try_copy_str(&self.regex.spec.lexeme_spec(lx_id).name)?;
            let r = self.new_node(&lname)?;
            self.grammar.make_terminal(r.idx, lx_id, &self.regex.spec)?;
            r
        })
    }

    pub fn size(&self) -> usize {
        self.grammar.num_symbols()
    }

    fn literal_name(&self, value: &str) -> Result<String> {
        if value.len() <= 20 {
            return Ok(self.funding.try_copy_str(value)?);
        }
        let prefix = &value.as_bytes()[..20];
        Ok(match std::str::from_utf8(prefix) {
            Ok(prefix) => self.funding.try_format(format_args!("{prefix}..."))?,
            Err(error) => self
                .funding
                .try_format(format_args!("{}�...", &value[..error.valid_up_to()]))?,
        })
    }

    pub fn string(&mut self, s: &str) -> Result<NodeRef> {
        Ok({
            if let Some(r) = self.strings.get(s) {
                return Ok(*r);
            }
            let r = if s.is_empty() {
                let r = self.new_node("empty")?;
                self.grammar.add_rule(r.idx, vec![])?;
                r
            } else {
                let lx_id = self.regex.spec.add_greedy_lexeme(
                    self.literal_name(s)?,
                    RegexAst::Literal(self.funding.try_copy_str(s)?),
                    false,
                    None,
                    usize::MAX,
                )?;
                self.lexeme_to_node(lx_id)?
            };
            let key = self.funding.try_copy_str(s)?;
            self.funding.try_insert(&mut self.strings, key, r)?;
            r
        })
    }

    pub fn token_ranges(&mut self, token_ranges: Vec<RangeInclusive<u32>>) -> Result<NodeRef> {
        self.check_limits()?;

        let trie = self.tok_trie;
        for r in &token_ranges {
            ensure!(&self.funding, r.start() <= r.end(), "Invalid token range: {:?}", r);
            if let Some(trie) = &trie {
                ensure!(&self.funding,
                    *r.end() < trie.vocab_size() as u32,
                    "Token range end too large: {:?}",
                    r.end()
                );
            }
        }

        if trie.is_none() {
            self.add_warning("no tokenizer - can't validate <[...]>")?;
        }

        let name = self
            .funding
            .try_format(format_args!("{}", TokenRangesDisplay(&token_ranges)))?;
        let id = self.regex.spec.add_special_token(name, token_ranges)?;
        Ok(self.lexeme_to_node(id)?)
    }

    pub fn negated_token_ranges(
        &mut self,
        token_ranges: Vec<RangeInclusive<u32>>,
    ) -> Result<NodeRef> {
        let negated_ranges = if let Some(trie) = self.tok_trie {
            let (min, max) = (0u32, trie.vocab_size() as u32 - 1);
            ensure!(&self.funding,
                !token_ranges.is_empty(),
                "negation of empty token ranges is not supported"
            );

            let mut sorted = token_ranges;
            sorted.sort_by_key(|r| *r.start());

            let mut negated = vec![];
            let mut current = min;
            for range in sorted {
                ensure!(&self.funding,
                    *range.end() < trie.vocab_size() as u32,
                    "Token range end too large: {:?}",
                    range.end()
                );
                ensure!(&self.funding,
                    range.start() <= range.end(),
                    "Invalid token range: {:?}",
                    range
                );

                let (&start, &end) = (range.start(), range.end());
                ensure!(&self.funding, start <= end, "Invalid token range: {:?}", range);
                if end < current {
                    // skip this range, it is already covered by the previous one
                    continue;
                }
                if start > current {
                    // add a range from the current to the start of this one
                    self.funding.try_push(&mut negated, current..=start - 1)?;
                }
                // update the current to the end of this range
                current = current.max(end + 1);
            }
            if current <= max {
                // add the last range from the current to the max
                self.funding.try_push(&mut negated, current..=max)?;
            }
            negated
        } else {
            self.add_warning("no tokenizer - can't validate <[^...]>")?;
            let mut ranges = Vec::new();
            self.funding
                .try_push(&mut ranges, INVALID_TOKEN..=INVALID_TOKEN)?;
            ranges
        };

        let name = self
            .funding
            .try_format(format_args!("{}", TokenRangesDisplay(&negated_ranges)))?;
        let id = self.regex.spec.add_special_token(name, negated_ranges)?;
        Ok(self.lexeme_to_node(id)?)
    }

    pub fn special_token(&mut self, token: &str) -> Result<NodeRef> {
        self.check_limits()?;

        let tok_id = if let Some(trie) = self.tok_trie {
            if let Some(tok_id) = trie.get_special_token(token) {
                tok_id
            } else {
                let spec = trie.get_special_tokens();
                bail!(&self.funding,
                    "unknown special token: {:?}; following special tokens are available: {}",
                    token,
                    trie.tokens_dbg(&spec)
                );
            }
        } else {
            self.add_warning("no tokenizer - can't validate <special_token>")?;
            INVALID_TOKEN
        };

        let mut ranges = Vec::new();
        self.funding.try_push(&mut ranges, tok_id..=tok_id)?;
        let name = self.funding.try_copy_str(token)?;
        let idx = self.regex.spec.add_special_token(name, ranges)?;
        Ok(self.lexeme_to_node(idx)?)
    }

    pub fn any_token(&mut self) -> Result<NodeRef> {
        self.check_limits()?;
        let range = if let Some(trie) = self.tok_trie {
            0..=trie.vocab_size() as u32 - 1
        } else {
            self.add_warning("no tokenizer - can't validate <any_token>")?;
            INVALID_TOKEN..=INVALID_TOKEN
        };
        let mut ranges = Vec::new();
        self.funding.try_push(&mut ranges, range)?;
        let name = self.funding.try_copy_str("<[*]>")?;
        let idx = self.regex.spec.add_special_token(name, ranges)?;
        Ok(self.lexeme_to_node(idx)?)
    }

    pub fn gen_grammar(&mut self, data: GenGrammarOptions, props: NodeProps) -> Result<NodeRef> {
        Ok({
            if props.max_tokens.is_some() {
                self.regex.spec.has_max_tokens = true;
            }
            if data.temperature.is_some() {
                self.regex.spec.has_temperature = true;
            }
            let r = self.new_node("gg")?;
            self.grammar.apply_node_props(r.idx, props);
            self.grammar.make_gen_grammar(r.idx, data)?;
            r
        })
    }

    pub fn gen(&mut self, data: GenOptions, props: NodeProps) -> Result<NodeRef> {
        self.check_limits()?;

        let empty_stop = matches!(data.stop_rx, RegexAst::EmptyString);
        let lazy = data.lazy.unwrap_or(!empty_stop);
        let name = self
            .funding
            .try_copy_str(props.capture_name.as_deref().unwrap_or("gen"))?;
        let lhs = self.new_node(&name)?;
        let lx_id = self.regex.spec.add_rx_and_stop(
            self.funding.try_copy_str(self.grammar.sym_name(lhs.idx))?,
            data.body_rx,
            data.stop_rx,
            lazy,
            props.max_tokens.unwrap_or(usize::MAX),
            data.is_suffix.unwrap_or(false),
        )?;
        self.grammar.apply_node_props(lhs.idx, props);
        let symprops = self.grammar.sym_props_mut(lhs.idx);
        if let Some(t) = data.temperature {
            symprops.temperature = t;
        }
        symprops.stop_capture_name = data.stop_capture_name;
        self.grammar
            .make_terminal(lhs.idx, lx_id, &self.regex.spec)?;
        Ok(lhs)
    }

    pub fn lexeme(&mut self, rx: ExprRef) -> Result<NodeRef> {
        Ok({ self.lexeme_ext(rx, None, NodeProps::default())? })
    }

    pub fn lexeme_ext(
        &mut self,
        rx: ExprRef,
        temperature: Option<f32>,
        props: NodeProps,
    ) -> Result<NodeRef> {
        Ok({
            let idx = self.regex.spec.add_greedy_lexeme(
                self.funding
                    .try_copy_str(props.capture_name.as_deref().unwrap_or_default())?,
                RegexAst::ExprRef(rx),
                false,
                None,
                props.max_tokens.unwrap_or(usize::MAX),
            )?;
            let r = self.lexeme_to_node(idx)?;
            self.grammar.apply_node_props(r.idx, props);
            if let Some(t) = temperature {
                self.grammar.set_temperature(r.idx, t);
            }
            r
        })
    }

    pub fn needs_param(&self, node: NodeRef) -> bool {
        if let Some(param_id) = node.param_id {
            self.params.get(param_id).needs_param()
        } else {
            false
        }
    }

    pub fn node_to_string(&self, node: NodeRef) -> String {
        if node.is_parametric() {
            let param = self.params.get(node.param_id.unwrap());
            format!("{}::{}", self.grammar.sym_name(node.idx), param)
        } else {
            self.grammar.sym_name(node.idx).to_string()
        }
    }

    fn child_nodes(&mut self, nodes: &[NodeRef]) -> Result<(Vec<(SymIdx, ParamExpr)>, bool)> {
        Ok({
            let mut opts = Vec::new();
            self.funding.try_grow_vec(&mut opts, nodes.len())?;
            for node in nodes {
                assert_eq!(node.grammar_id, self.curr_grammar_idx);
                let param = node
                    .param_id
                    .map(|id| self.params.get(id).clone())
                    .unwrap_or(ParamExpr::Null);
                opts.push((node.idx, param));
            }
            let needs_param = opts.iter().any(|(_, p)| p.needs_param());
            debug!("needs_param: {}", needs_param);
            (opts, needs_param)
        })
    }

    pub fn rename(&mut self, node: NodeRef, name: &str) -> Result<()> {
        Ok({
            self.grammar.rename_symbol(node.idx, name)?;
        })
    }

    pub fn select(&mut self, options: &[NodeRef]) -> Result<NodeRef> {
        Ok({ self.select_with_cond(options, Vec::new())? })
    }

    pub fn select_with_cond(
        &mut self,
        options: &[NodeRef],
        mut conds: Vec<ParamCond>,
    ) -> Result<NodeRef> {
        Ok({
            let (ch, needs_param) = self.child_nodes(options)?;
            if options.len() == 1 && conds.is_empty() {
                return Ok(options[0]);
            }
            let r = self.new_param_node("", needs_param || !conds.is_empty())?;
            let empty = self.empty()?.idx;
            assert!(conds.is_empty() || ch.len() == conds.len());
            conds.reverse();
            for n in &ch {
                let cond = conds.pop().unwrap_or(ParamCond::True);
                let mut v = Vec::new();
                if n.0 != empty {
                    self.funding.try_push(&mut v, n.clone())?;
                }
                self.grammar.add_rule_ext(r.idx, cond, v)?;
            }
            r
        })
    }

    pub fn join(&mut self, values: &[NodeRef]) -> Result<NodeRef> {
        Ok({ self.join_props(values, NodeProps::default())? })
    }

    pub fn join_props(&mut self, values: &[NodeRef], props: NodeProps) -> Result<NodeRef> {
        Ok({
            let (mut ch, needs_param) = self.child_nodes(values)?;
            let empty = self.empty()?.idx;
            ch.retain(|n| n.0 != empty);
            if ch.is_empty() {
                return Ok(self.empty()?);
            }
            if ch.len() == 1 && props == NodeProps::default() {
                let param_id = values.iter().find(|v| v.idx == ch[0].0).unwrap().param_id;
                // if param_id is Some, but needs_param is false,
                // it means this node doesn't have to be parametric if we wrap it (so we do it)
                if param_id.is_none() || needs_param {
                    return Ok(NodeRef {
                        idx: ch[0].0,
                        grammar_id: self.curr_grammar_idx,
                        param_id,
                    });
                }
            }
            let r = self.new_param_node("", needs_param)?;
            self.grammar.apply_node_props(r.idx, props);
            self.grammar.add_rule_ext(r.idx, ParamCond::True, ch)?;
            r
        })
    }

    pub fn empty(&mut self) -> Result<NodeRef> {
        Ok({ self.string("")? })
    }

    pub fn num_nodes(&self) -> usize {
        self.grammar.num_symbols()
    }

    fn add_rule(&mut self, node: NodeRef, children: &[NodeRef]) -> Result<()> {
        Ok({
            let (ch, needs_param) = self.child_nodes(children)?;
            assert!(!needs_param || node.is_parametric());
            self.grammar.add_rule_ext(node.idx, ParamCond::True, ch)?;
        })
    }

    pub fn optional(&mut self, value: NodeRef) -> Result<NodeRef> {
        Ok({
            let p = self.new_wrapper_node("", value)?;
            self.add_rule(p, &[])?;
            self.add_rule(p, &[value])?;
            p
        })
    }

    pub fn one_or_more(&mut self, elt: NodeRef) -> Result<NodeRef> {
        Ok({
            let p = self.new_wrapper_node("plus", elt)?;
            self.add_rule(p, &[elt])?;
            self.add_rule(p, &[p, elt])?;
            p
        })
    }

    pub fn zero_or_more(&mut self, elt: NodeRef) -> Result<NodeRef> {
        Ok({
            let p = self.new_wrapper_node("star", elt)?;
            self.add_rule(p, &[])?;
            self.add_rule(p, &[p, elt])?;
            p
        })
    }

    // at_most() creates a rule which accepts at most 'n' copies
    // of element 'elt'.

    // The first-time reader of at_most() might want to consult
    // the comments for repeat_exact(), where similar logic is
    // used in a simpler form.
    //
    // at_most() recursively factors the sequence into K-size pieces,
    // in an attempt to keep grammar size O(log(n)).
    fn at_most(&mut self, elt: NodeRef, n: usize) -> Result<NodeRef> {
        Ok({
            if let Some(r) = self.at_most_cache.get(&(elt, n)) {
                return Ok(*r);
            }
            let r = if n == 0 {
                // If the max ('n') is 0, an empty rule
                self.empty()?
            } else if n == 1 {
                // If 'n' is 1, an optional rule of length 1
                self.optional(elt)?
            } else if n < 3 * K {
                // If 'n' is below a fixed number (currently 12),
                // the rule is a choice of all the rules of fixed length
                // from 0 to 'n'.
                let mut options = Vec::new();
                self.funding.try_grow_vec(&mut options, n + 1)?;
                for k in 0..=n {
                    options.push(self.simple_repeat(elt, k)?);
                }
                self.select(&options)?
            } else {
                // Above a fixed number (again, currently 12),
                // we "factor" the sequence into K-sized pieces.
                // Let 'elt_k' be a k-element --- the repetition
                // of 'k' copies of the element ('elt').
                let elt_k = self.simple_repeat(elt, K)?;

                // First we deal with the sequences of length less than
                // (n/K)*K.
                // 'elt_max_nk' is all the sequences of k-elements
                // of length less than n/K.
                let elt_max_nk = self.at_most(elt_k, (n / K) - 1)?;
                // The may be up to K-1 elements not accounted by the sequences
                // of k-elements in 'elt_max_k'.  The choices in 'elt_max_k'
                // account for these "remainders".
                let elt_max_k = self.at_most(elt, K - 1)?;
                let elt_max_nk = self.join(&[elt_max_nk, elt_max_k])?;

                // Next we deal with the sequences of length between
                // (n/K)*K and 'n', inclusive. It is integer arithmetic, so there
                // will be n%K of these.
                // Here we call n/K the quotient and n%K the remainder.
                // 'elt_nk' repeats the k-element exactly the quotient
                // number of times, to ensure all our sequences are of
                // length at least (n/K)*K.
                let elt_nk = self.repeat_exact(elt_k, n / K)?;
                // 'left' repeats 'elt' at most the remainder number
                // of times.  The remainder is always less than K.
                let left = self.at_most(elt, n % K)?;
                // Join 'elt_nk' and 'left' into 'elt_n'.
                // 'elt_nk' is a constant-sized piece,
                // which ensures all the sequences of 'elt' in 'elt_n',
                // will be of length at least (n/K)*K.
                // 'left' will be a choice of rules which
                // produce at most K-1 copies of 'elt'.
                let elt_n = self.join(&[elt_nk, left])?;

                // We have accounted for all the sequences of less than
                // (n/K)*K elements in 'elt_max_nk'.  We have accounted
                // for all the sequences of length between (n/K)*K elements and n elements
                // (inclusive) in 'elt_n'.  Clearly, the sequences of length at most 'n'
                // are the alternation of 'elt_max_nk' and 'elt_n'.
                self.select(&[elt_n, elt_max_nk])?
            };
            self.funding
                .try_insert(&mut self.at_most_cache, (elt, n), r)?;
            r
        })
    }

    // simple_repeat() "simply" repeats the element ('elt') 'n' times.
    // Here "simple" means we do not factor into K-size pieces, so that
    // time will be O(n).  The intent is that simple_repeat() only be
    // called for small 'n'.
    fn simple_repeat(&mut self, elt: NodeRef, n: usize) -> Result<NodeRef> {
        Ok({
            let mut elt_n = Vec::new();
            self.funding.try_grow_vec(&mut elt_n, n)?;
            elt_n.resize(n, elt);
            self.join(&elt_n)?
        })
    }

    // Repeat element 'elt' exactly 'n' times, using factoring
    // in an attempt to keep grammar size O(log(n)).
    fn repeat_exact(&mut self, elt: NodeRef, n: usize) -> Result<NodeRef> {
        Ok({
            if let Some(r) = self.repeat_exact_cache.get(&(elt, n)) {
                return Ok(*r);
            }
            let r = if n > 2 * K {
                // For large 'n', try to keep the number of rules O(log(n))
                // by "factoring" the sequence into K-sized pieces

                // Create a K-element -- 'elt' repeated 'K' times.
                let elt_k = self.simple_repeat(elt, K)?;

                // Repeat the K-element n/K times.  The repetition
                // is itself factored, so that the process is
                // recursive.
                let inner = self.repeat_exact(elt_k, n / K)?;

                // 'inner' will contain ((n/K)K) be an 'elt'-sequence
                // of length ((n/K)K), which is n-((n/K)K), or n%K,
                // short of what we want.  We create 'elt_left' to contain
                // the n%K additional items we need, and concatenate it
                // with 'inner' to form our result.
                let left = n % K;
                let mut elt_left = Vec::new();
                self.funding.try_grow_vec(&mut elt_left, left + 1)?;
                elt_left.resize(left, elt);
                elt_left.push(inner);
                self.join(&elt_left)?
            } else {
                // For small 'n' (currently, 8 or less), simply
                // repeat 'elt' 'n' times.
                self.simple_repeat(elt, n)?
            };
            self.funding
                .try_insert(&mut self.repeat_exact_cache, (elt, n), r)?;
            r
        })
    }

    // at_least() accepts a sequence of at least 'n' copies of
    // element 'elt'.
    fn at_least(&mut self, elt: NodeRef, n: usize) -> Result<NodeRef> {
        Ok({
            let z = self.zero_or_more(elt)?;
            if n == 0 {
                // If n==0, atleast() is equivalent to zero_or_more().
                z
            } else {
                // If n>0, first sequence is a factored repetition of
                // exactly 'n' copies of 'elt', ...
                let r = self.repeat_exact(elt, n)?;
                // ... followed by zero or more copies of 'elt'
                self.join(&[r, z])?
            }
        })
    }

    // Create a rule which accepts from 'min' to 'max' copies of element
    // 'elt', inclusive.
    pub fn repeat(&mut self, elt: NodeRef, min: usize, max: Option<usize>) -> Result<NodeRef> {
        Ok({
            if max.is_none() {
                // If no 'max', what we want is equivalent to a rule accepting at least
                // 'min' elements.
                return Ok(self.at_least(elt, min)?);
            }
            let max = max.unwrap();
            assert!(min <= max);
            if min == max {
                // Where 'min' is equal to 'max', what we want is equivalent to a rule
                // repeating element 'elt' exactly 'min' times.
                self.repeat_exact(elt, min)?
            } else if min == 0 {
                // If 'min' is zero, what we want is equivalent to a rule accepting at least
                // 'min' elements.
                self.at_most(elt, max)?
            } else {
                // In the general case, what we want is equivalent to
                // a rule accepting a fixed-size block of length 'min',
                // followed by a rule accepting at most 'd' elements,
                // where 'd' is the difference between 'min' and 'max'
                let d = max - min;
                let common = self.repeat_exact(elt, min)?;
                let extra = self.at_most(elt, d)?;
                self.join(&[common, extra])?
            }
        })
    }

    pub fn new_node(&mut self, name: &str) -> Result<NodeRef> {
        Ok({
            let id = self.grammar.fresh_symbol_ext(
                name,
                SymbolProps {
                    grammar_id: self.curr_lexeme_class,
                    ..Default::default()
                },
            )?;
            NodeRef {
                idx: id,
                grammar_id: self.curr_grammar_idx,
                param_id: None,
            }
        })
    }

    pub fn new_param_node(&mut self, name: &str, needs_param: bool) -> Result<NodeRef> {
        Ok({
            let mut r = self.new_node(name)?;
            if needs_param {
                self.grammar.make_parametric(r.idx)?;
                r.param_id = Some(self.self_ref);
            }
            r
        })
    }

    fn new_wrapper_node(&mut self, name: &str, wrapper_for: NodeRef) -> Result<NodeRef> {
        Ok({ self.new_param_node(name, self.needs_param(wrapper_for))? })
    }

    pub fn set_placeholder(&mut self, placeholder: NodeRef, node: NodeRef) -> Result<()> {
        Ok({
            let (mut ch, _) = self.child_nodes(&[node, placeholder])?;
            ch.pop();
            self.grammar
                .check_empty_symbol_parametric_ok(placeholder.idx)
                .expect("placeholder already set");
            self.grammar
                .add_rule_ext(placeholder.idx, ParamCond::True, ch)?;
        })
    }

    pub fn set_start_node(&mut self, node: NodeRef) -> Result<()> {
        Ok({
            self.set_placeholder(self.curr_start_idx, node)?;
        })
    }

    pub fn link_gen_grammar(&mut self, gg: NodeRef, grm_start: SymIdx) -> Result<()> {
        self.grammar.link_gen_grammar(gg.idx, grm_start)
    }

    pub fn finalize(self, symidx: SymIdx) -> GrammarResult<'t> {
        GrammarResult {
            start_node: symidx,
            builder: self,
        }
    }

    pub fn apply(&mut self, id: NodeRef, param: Option<ParamExpr>) -> Result<NodeRef> {
        if param.is_none() {
            ensure!(&self.funding,
                !id.is_parametric(),
                "rule '{}' is parametric, but no parameter provided",
                self.grammar.sym_name(id.idx)
            );
            return Ok(id);
        }
        ensure!(&self.funding,
            id.is_parametric(),
            "rule '{}' is not parametric, but parameter provided",
            self.grammar.sym_name(id.idx)
        );
        let param_id = self.params.insert(param.unwrap())?;
        Ok(NodeRef {
            idx: id.idx,
            param_id: Some(param_id),
            grammar_id: id.grammar_id,
        })
    }
}
