mod frames;
use frames::FundingScope;
use crate::{
    earley::{ParamCond, ParamExpr},
    grammar_builder::{GrammarResult, RegexId},
    substring::substring,
};
use derivre::{ParserResult as Result, ParserError, parser_error as anyhow, parser_bail as bail, parser_ensure as ensure};
use derivre::{RegexAst, ParserAllocationFunding, SourceHashMap as HashMap};
use serde::Deserialize;

use crate::{
    api::{
        GenGrammarOptions, GenOptions, GrammarId, LLGuidanceOptions, NodeProps, RegexExt, SkipSpec,
    },
    substring::{chunk_into_chars, chunk_into_words},
    GrammarBuilder, JsonCompileOptions, NodeRef,
};

use super::{
    ast::*,
    common::lookup_common_regex,
    lexer::Location,
    parser::{parse_lark, ParsedLark},
};

const DEBUG: bool = false;

/// Options accepted by Lark's `%llguidance` directive. `ignore_once` is
/// Lark-specific because it controls how `%ignore` expressions are compiled.
#[derive(Debug, Default)]
struct LarkLLGuidanceOptions {
    general: LLGuidanceOptions,
    ignore_once: bool,
}

impl LarkLLGuidanceOptions {
    // The directive has only scalar boolean fields. Consume its already parsed
    // object directly instead of serializing, cloning, or collecting flatten
    // intermediates. Omitted fields keep the prior directive's value.
    fn apply_value(&mut self, value: serde_json::Value, funding: &ParserAllocationFunding, frames: &FundingScope<'_>) -> Result<()> {
        let _frame = frames::enter::<frames::ApplyOptions<'_>>(frames, funding)?;
        let serde_json::Value::Object(object) = value else {
            bail!(funding, "failed to parse %llguidance declaration: expected object");
        };
        for (name, value) in object {
            let target = match name.as_str() {
                "no_forcing" => &mut self.general.no_forcing,
                "allow_invalid_utf8" => &mut self.general.allow_invalid_utf8,
                "allow_initial_skip" => &mut self.general.allow_initial_skip,
                "ignore_once" => &mut self.ignore_once,
                _ => continue,
            };
            *target = match value { serde_json::Value::Bool(value) => value, _ => bail!(funding, "expected a boolean in %llguidance declaration") };
        }
        Ok(())
    }
}

macro_rules! debug {
    ($($arg:tt)*) => {
        if cfg!(feature = "logging") && DEBUG {
            eprint!("LRK> ");
            eprintln!($($arg)*);
        }
    };
}

#[derive(Debug)]
struct Grammar {
    rules: HashMap<String, Rule>,
    tokens: HashMap<String, TokenDef>,
    ignore: Vec<Expansions>,
    llguidance_options: LarkLLGuidanceOptions,
}

impl Default for Grammar {
    fn default() -> Self {
        Self {
            rules: HashMap::default(),
            tokens: HashMap::default(),
            ignore: vec![],
            llguidance_options: LarkLLGuidanceOptions::default(),
        }
    }
}

enum PendingGrammar {
    Json(serde_json::Value),
    Lark(Vec<Item>),
}

struct Compiler<'t, 'f> {
    frames: &'f FundingScope<'f>,
    builder: GrammarBuilder<'t>,
    parsed: ParsedLark,
    grammar: Grammar,
    node_ids: HashMap<String, NodeRef>,
    regex_ids: HashMap<String, RegexId>,
    in_progress: HashMap<String, bool>,
    pending_grammars: Vec<(NodeRef, Location, PendingGrammar)>,
}

fn compile_lark<'t, 'f>(builder: GrammarBuilder<'t>, parsed: ParsedLark, frames: &'f FundingScope<'f>) -> Result<GrammarResult<'t>> {
    let _frame = frames::enter::<frames::Compile<'_, 't, 'f>>(frames, &builder.funding)?;
    let c = Compiler {
        frames,
        builder,
        parsed,
        grammar: Grammar::default(),
        node_ids: HashMap::default(),
        regex_ids: HashMap::default(),
        in_progress: HashMap::default(),
        pending_grammars: vec![],
    };
    c.execute()
}

pub fn lark_to_llguidance<'t>(
    mut builder: GrammarBuilder<'t>,
    lark: &str,
) -> Result<GrammarResult<'t>> {
    let funding = builder.funding.clone();
    let scope = derivre::prepared_funding::Scope::new(&funding)
        .map_err(|error| frames::failure(error, &funding))?;
    let _frame = frames::enter::<frames::Public<'_, 't>>(&scope, &funding)?;
    let parsed = parse_lark(lark, builder.funding.clone())?;

    compile_lark(builder, parsed, &scope)
}

impl<'t, 'f> Compiler<'t, 'f> {
    fn enter<T>(&self) -> Result<derivre::prepared_funding::Frame<'f>> {
        frames::enter::<T>(self.frames, &self.builder.funding)
    }
    fn do_token(&mut self, name: &str) -> Result<RegexId> {
        let _frame = self.enter::<frames::Token<'_, 't, 'f>>()?;
        if let Some(id) = self.regex_ids.get(name) {
            return Ok(*id);
        }
        if self.in_progress.contains_key(name) {
            bail!(&self.builder.funding, "circular reference in token {:?} definition", name);
        }
        self.builder.funding.try_insert(&mut self.in_progress, self.builder.funding.try_copy_str(name)?, false)?;
        let token = self
            .grammar
            .tokens
            .remove(name)
            .ok_or_else(|| anyhow!(&self.builder.funding, "unknown name: {:?}", name))?;
        let id = self.do_token_expansions(token.expansions)?;
        self.builder.funding.try_insert(&mut self.regex_ids, self.builder.funding.try_copy_str(name)?, id)?;
        self.in_progress.remove(name);
        Ok(id)
    }

    fn mk_regex(&mut self, info: &str, rx: String) -> Result<RegexId> {
        let _frame = self.enter::<frames::Regex<'_, 't, 'f>>()?;
        self.builder.regex.regex(&rx).map_err(|error| {
            if crate::earley::is_grammar_storage_failure(&error) { error }
            else { error.context(format_args!("invalid regex {rx:?} (in {info})"), &self.builder.funding) }
        })
    }

    fn do_token_atom(&mut self, atom: Atom) -> Result<RegexId> {
        let _frame = self.enter::<frames::TokenAtom<'_, 't, 'f>>()?;
        self.builder.check_limits()?;
        match atom {
            Atom::Group(expansions) => self.do_token_expansions(expansions),
            Atom::Maybe(expansions) => {
                let id = self.do_token_expansions(expansions)?;
                Ok(self.builder.regex.optional(id)?)
            }
            Atom::Not(inner) => {
                let id = self.do_token_atom(*inner)?;
                Ok(self.builder.regex.not(id)?)
            }
            Atom::Value(value) => match value {
                Value::LiteralRange(a, b) => {
                    ensure!(&self.builder.funding,
                        a.chars().count() == 1,
                        "range start must be a single character"
                    );
                    ensure!(&self.builder.funding,
                        b.chars().count() == 1,
                        "range end must be a single character"
                    );
                    let a = a.chars().next().unwrap();
                    let b = b.chars().next().unwrap();
                    if a <= b {
                        self.mk_regex(
                            "range",
                            self.builder.funding.try_format(format_args!(
                                "[{}-{}]", EscapedChar(a), EscapedChar(b)
                            ))?,
                        )
                    } else {
                        bail!(&self.builder.funding, "invalid range order: {:?}..{:?}", a, b);
                    }
                }
                Value::Name(n) => self.do_token(&n),
                Value::LiteralString(val, flags) => {
                    if flags.contains("i") {
                        self.mk_regex(
                            "string with i-flag",
                            self.builder.funding.try_format(format_args!("(?i){}", Escaped(&val)))?,
                        )
                    } else {
                        Ok(self.builder.regex.literal(&val)?)
                    }
                }
                Value::LiteralRegex(val, flags) => {
                    ensure!(&self.builder.funding, !flags.contains("l"), "l-flag is not supported in regexes");
                    let rx = if flags.is_empty() {
                        val
                    } else {
                        self.builder.funding.try_format(format_args!("(?{flags}){val}"))?
                    };
                    self.mk_regex("regex", rx)
                }
                Value::RegexExt(s) => compile_lark_regex(&mut self.builder, s, self.frames),
                Value::SpecialToken(s) => {
                    bail!(&self.builder.funding, "special tokens (like {:?}) cannot be used in terminals", s);
                }
                Value::Json(_) => {
                    bail!(&self.builder.funding, "%json literals cannot be used in terminals");
                }
                Value::GrammarRef(g) => {
                    bail!(&self.builder.funding,
                        "grammar references (like {:?}) cannot be used in terminals",
                        g
                    );
                }
                Value::NestedLark(_) => {
                    bail!(&self.builder.funding, "nested %lark {{ ... }} cannot be used in terminals");
                }
                Value::NameParam(_, _) => {
                    bail!(&self.builder.funding, "name::param cannot be used in terminals");
                }
                Value::TemplateUsage { .. } => bail!(&self.builder.funding, "template usage not supported yet"),
            },
        }
    }

    fn do_token_expr(&mut self, expr: Expr) -> Result<RegexId> {
        let _frame = self.enter::<frames::TokenExpr<'_, 't, 'f>>()?;
        let atom = self.do_token_atom(expr.atom)?;
        if let Some(range) = &expr.range {
            ensure!(&self.builder.funding, expr.op.is_none(), "ranges not supported with operators");
            ensure!(&self.builder.funding, range.0 >= 0, "range start must be >= 0, got {:?}", range);
            ensure!(&self.builder.funding,
                range.1 >= range.0,
                "range end must be >= start, got {:?}",
                range
            );
            Ok(self.builder.regex.repeat(
                atom,
                range.0 as u32,
                if range.1 == i32::MAX {
                    None
                } else {
                    Some(range.1 as u32)
                },
            )?)
        } else {
            match &expr.op {
                Some(op) => match op.0.as_str() {
                    "*" => Ok(self.builder.regex.zero_or_more(atom)?),
                    "+" => Ok(self.builder.regex.one_or_more(atom)?),
                    "?" => Ok(self.builder.regex.optional(atom)?),
                    _ => {
                        bail!(&self.builder.funding, "unsupported operator: {:?}", op.0);
                    }
                },
                None => Ok(atom),
            }
        }
    }

    fn do_token_expansions(&mut self, expansions: Expansions) -> Result<RegexId> {
        let _frame = self.enter::<frames::TokenExpansions<'_, 't, 'f>>()?;
        self.builder.check_limits()?;
        let mut options = Vec::new();
        for alias in expansions.1 {
            ensure!(&self.builder.funding, alias.param_cond.is_true(), "'%if' is not supported in terminals");
            let mut args = Vec::new();
            for exp in alias.conjuncts {
                let mut concat = Vec::new();
                for expr in exp.0 {
                    let expr = self.do_token_expr(expr).map_err(|error| expansions.0.augment(error))?;
                    self.builder.funding.try_push(&mut concat, expr)?;
                }
                let expr = self.builder.regex.concat(&concat)?;
                self.builder.funding.try_push(&mut args, expr)?;
            }
            let expr = self.builder.regex.and(&args)?;
            self.builder.funding.try_push(&mut options, expr)?;
        }
        Ok(self.builder.regex.select(&options)?)
    }

    fn lift_regex(&mut self, rx_id: RegexId) -> Result<NodeRef> {
        let _frame = self.enter::<frames::Lift<'_, 't, 'f>>()?;
        Ok(self.builder.lexeme(rx_id)?)
    }

    fn do_nested(
        &mut self,
        loc: &Location,
        v: Value,
        temperature: Option<f32>,
        props: NodeProps,
    ) -> Result<NodeRef> {
        let _frame = self.enter::<frames::Nested<'_, 't, 'f>>()?;
        let inner = match v {
            Value::NestedLark(items) => PendingGrammar::Lark(items),
            Value::Json(json) => PendingGrammar::Json(json),
            _ => bail!(&self.builder.funding, "expected %lark or %json, got {:?}", v),
        };
        let name = self.builder.funding.try_format(format_args!("%nested---{}", self.builder.num_nodes()))?;
        let gg = self.builder.gen_grammar(
            GenGrammarOptions {
                grammar: GrammarId::Name(name),
                temperature,
            },
            props,
        )?;
        self.builder.funding.try_push(&mut self.pending_grammars, (gg, loc.clone(), inner))?;
        Ok(gg)
    }

    fn do_atom(&mut self, loc: &Location, expr: Atom) -> Result<NodeRef> {
        let _frame = self.enter::<frames::AtomFrame<'_, 't, 'f>>()?;
        match expr {
            Atom::Group(expansions) => self.do_expansions(expansions),
            Atom::Maybe(expansions) => {
                let id = self.do_expansions(expansions)?;
                Ok(self.builder.optional(id)?)
            }
            Atom::Not(_) => {
                // treat as token
                let rx = self.do_token_atom(expr)?;
                Ok(self.lift_regex(rx)?)
            }
            Atom::Value(value) => {
                match &value {
                    Value::Name(n) => {
                        if self.is_rule(n) {
                            return self.do_rule(n, None);
                        } else {
                            // OK -> treat as token
                        }
                    }
                    Value::NameParam(name, param) => {
                        return self.do_rule(name, Some(param.clone()));
                    }
                    Value::SpecialToken(s) => {
                        if s.starts_with("<[") && s.ends_with("]>") {
                            let s = &s[2..s.len() - 2];
                            let negate = s.starts_with("^");
                            let s = if negate { &s[1..] } else { s };
                            if s == "*" {
                                if negate {
                                    bail!(&self.builder.funding, "negated wildcard token <[^*]> is not supported");
                                }
                                return self.builder.any_token();
                            } else if s.contains('*') {
                                bail!(&self.builder.funding,
                                    "wildcard token range '*' must not contain additional tokens"
                                );
                            }
                            let mut ranges = vec![];
                            for range in s.split(",") {
                                let mut ends = range.split('-').map(str::trim);
                                let first = ends.next().unwrap();
                                let second = ends.next();
                                ensure!(&self.builder.funding, ends.next().is_none(), "invalid token range: {:?}", range);
                                if second.is_none() && first.is_empty() { continue; }
                                let start = first.parse::<u32>().map_err(|error| derivre::ParserError::cause(error, &self.builder.funding))?;
                                let end = match second { Some(end) => end.parse::<u32>().map_err(|error| derivre::ParserError::cause(error, &self.builder.funding))?, None => start };
                                ensure!(&self.builder.funding, start <= end, "invalid token range: {:?}", range);
                                self.builder.funding.try_push(&mut ranges, start..=end)?;
                            }
                            ensure!(&self.builder.funding, !ranges.is_empty(), "empty token range");
                            return if negate {
                                self.builder.negated_token_ranges(ranges)
                            } else {
                                self.builder.token_ranges(ranges)
                            };
                        }
                        return self.builder.special_token(s);
                    }
                    Value::GrammarRef(g) => {
                        return self.gen_grammar(g, None, NodeProps::default());
                    }
                    Value::NestedLark(_) | Value::Json(_) => {
                        return self.do_nested(loc, value, None, NodeProps::default());
                    }
                    // special case "" literal, so it doesn't pollute grammar with epsilon regex
                    Value::LiteralString(s, _) if s.is_empty() => {
                        return Ok(self.builder.empty()?)
                    }
                    Value::RegexExt(_)
                    | Value::LiteralRange(_, _)
                    | Value::LiteralString(_, _)
                    | Value::LiteralRegex(_, _) => {
                        // treat as token
                    }
                    Value::TemplateUsage { .. } => {
                        bail!(&self.builder.funding, "template usage not supported yet");
                    }
                };
                let rx = self.do_token_atom(Atom::Value(value))?;
                Ok(self.lift_regex(rx)?)
            }
        }
    }

    fn do_expr(&mut self, loc: &Location, expr: Expr) -> Result<NodeRef> {
        let _frame = self.enter::<frames::ExprFrame<'_, 't, 'f>>()?;
        let atom = self.do_atom(loc, expr.atom)?;

        if let Some((a, b)) = expr.range {
            ensure!(&self.builder.funding, expr.op.is_none(), "ranges not supported with operators");
            ensure!(&self.builder.funding, a <= b, "range end must be >= start, got {:?}", (a, b));
            ensure!(&self.builder.funding, a >= 0, "range start must be >= 0, got {:?}", a);
            Ok(self.builder.repeat(
                atom,
                a as usize,
                if b == i32::MAX {
                    None
                } else {
                    Some(b as usize)
                },
            )?)
        } else {
            match &expr.op {
                Some(op) => match op.0.as_str() {
                    "*" => Ok(self.builder.zero_or_more(atom)?),
                    "+" => Ok(self.builder.one_or_more(atom)?),
                    "?" => Ok(self.builder.optional(atom)?),
                    _ => {
                        bail!(&self.builder.funding, "unsupported operator: {}", op.0);
                    }
                },
                None => Ok(atom),
            }
        }
    }

    fn do_expansions(&mut self, expansions: Expansions) -> Result<NodeRef> {
        let _frame = self.enter::<frames::ExpansionsFrame<'_, 't, 'f>>()?;
        self.builder.check_limits()?;
        let loc = expansions.0;
        let mut conds = vec![];
        let needs_cond = expansions.1.iter().any(|alias| !alias.param_cond.is_true());
        let mut options = Vec::new();
        for mut alias in expansions.1 {
            ensure!(&self.builder.funding, alias.conjuncts.len() == 1,
                "& is only supported for tokens, not rules; try renaming the rule to UPPERCASE");
            let mut args = Vec::new();
            for expr in alias.conjuncts.pop().unwrap().0 {
                let expr = self.do_expr(&loc, expr).map_err(|error| loc.augment(error))?;
                self.builder.funding.try_push(&mut args, expr)?;
            }
            if needs_cond { self.builder.funding.try_push(&mut conds, alias.param_cond)?; }
            let option = self.builder.join_props(&args, NodeProps::default())?;
            self.builder.funding.try_push(&mut options, option)?;
        }
        Ok(self.builder.select_with_cond(&options, conds)?)
    }

    fn is_rule(&self, name: &str) -> bool {
        self.node_ids.contains_key(name)
            || self.in_progress.contains_key(name)
            || self.grammar.rules.contains_key(name)
    }

    fn do_rule(&mut self, name: &str, param: Option<ParamExpr>) -> Result<NodeRef> {
        let _frame = self.enter::<frames::Rule<'_, 't, 'f>>()?;
        if let Some(id) = self.node_ids.get(name) {
            return self.builder.apply(*id, param);
        }
        if let Some(&is_param) = self.in_progress.get(name) {
            let id = self.builder.new_param_node(&self.builder.funding.try_format(format_args!("{name}_"))?, is_param)?;
            self.builder.funding.try_insert(&mut self.node_ids, self.builder.funding.try_copy_str(name)?, id)?;
            return self.builder.apply(id, param);
        }

        debug!("BEG rule {}", name);
        let id = self.do_rule_core(name)?;

        if let Some(placeholder) = self.node_ids.get(name) {
            self.builder.set_placeholder(*placeholder, id)?;
        }
        self.builder.funding.try_insert(&mut self.node_ids, self.builder.funding.try_copy_str(name)?, id)?;
        self.in_progress.remove(name);
        self.builder.rename(id, name)?;
        debug!("END rule {}", name);
        self.builder.apply(id, param)
    }

    fn gen_grammar(
        &mut self,
        name: &str,
        temperature: Option<f32>,
        props: NodeProps,
    ) -> Result<NodeRef> {
        let _frame = self.enter::<frames::GenGrammar<'_, 't, 'f>>()?;
        assert!(name.starts_with("@"));
        // see if name[1..] is an integer
        let name = if name[1..].parse::<usize>().is_ok() {
            bail!(&self.builder.funding, "numeric grammar references no longer supported");
        } else {
            self.builder.funding.try_copy_str(&name[1..])?
        };
        let id = self.builder.gen_grammar(
            GenGrammarOptions {
                grammar: GrammarId::Name(name),
                temperature,
            },
            props,
        )?;
        Ok(id)
    }

    fn do_rule_core(&mut self, name: &str) -> Result<NodeRef> {
        let _frame = self.enter::<frames::RuleCore<'_, 't, 'f>>()?;
        let mut rule = self
            .grammar
            .rules
            .remove(name)
            .ok_or_else(|| anyhow!(&self.builder.funding, "rule {:?} not found", name))?;

        self.builder.funding.try_insert(&mut self.in_progress,
            self.builder.funding.try_copy_str(name)?, rule.is_parametric)?;

        let has_capture = rule.capture_name.is_some();
        let props = NodeProps {
            max_tokens: rule.max_tokens,
            capture_name: rule.capture_name.take(),
            ..Default::default()
        };

        if rule.stop.is_some() && rule.suffix.is_some() {
            bail!(&self.builder.funding, "stop= and suffix= cannot be used together");
        }

        if rule.is_parametric && rule.stop_like().is_some() {
            bail!(&self.builder.funding, "stop-like is not supported for parametric rules");
        }

        if rule.is_parametric && rule.temperature.is_some() {
            bail!(&self.builder.funding, "temperature= is not supported for parametric rules");
        }

        if rule.is_parametric && rule.max_tokens.is_some() {
            bail!(&self.builder.funding, "max_tokens= is not supported for parametric rules");
        }

        if rule.max_tokens == Some(0) {
            // max_tokens=N caps a rule at N emitted tokens, so max_tokens=0
            // forces zero emitted tokens: the rule can only match the empty
            // string (epsilon). Compiling it as a token-limited lexeme (or
            // subgrammar) instead produces broken runtime output -- the matcher
            // emits an opening token and then never terminates. Treat it the
            // same as an empty-string body `""`.
            // See https://github.com/guidance-ai/llguidance/issues/236
            return Ok(self.builder.string("")?);
        }

        let id = if let Some(stop) = rule.stop_like() {
            let is_suffix = rule.suffix.is_some();
            let is_empty = matches!(stop, Value::LiteralString(s, _) if s.is_empty());
            let lazy = rule.is_lazy();
            let stop_val = Atom::Value(rule.take_stop_like().unwrap());
            let rx_id = self.do_token_expansions(rule.expansions)?;
            let stop_id = self.do_token_atom(stop_val)?;

            self.builder.gen(
                GenOptions {
                    body_rx: RegexAst::ExprRef(rx_id),
                    stop_rx: if is_empty {
                        RegexAst::EmptyString
                    } else {
                        RegexAst::ExprRef(stop_id)
                    },
                    stop_capture_name: rule.stop_capture_name.take(),
                    lazy: Some(lazy),
                    temperature: rule.temperature,
                    is_suffix: Some(is_suffix),
                },
                props,
            )?
        } else {
            ensure!(&self.builder.funding,
                rule.stop_capture_name.is_none(),
                "stop_capture_name requires stop= or suffix="
            );
            if rule.temperature.is_some() || rule.max_tokens.is_some() {
                match rule.expansions.single_atom() {
                    Some(Atom::Value(Value::GrammarRef(g))) => {
                        return self.gen_grammar(g, rule.temperature, props);
                    }
                    Some(Atom::Value(Value::Json(_) | Value::NestedLark(_))) => {
                        if let Some(Atom::Value(x)) = rule.expansions.take_single_atom() {
                            return self.do_nested(&rule.expansions.0, x, rule.temperature, props);
                        } else {
                            unreachable!();
                        }
                    }
                    _ => {
                        // try as terminal
                        let rx_id = self.do_token_expansions(rule.expansions).map_err(|e| {
                            if crate::earley::is_grammar_storage_failure(&e) { e } else {
                                e.context("temperature= and max_tokens= only supported on TERMINALS and @subgrammars", &self.builder.funding)
                            }
                        })?;
                        return Ok(self.builder.lexeme_ext(rx_id, rule.temperature, props)?);
                    }
                }
            }

            let inner = self.do_expansions(rule.expansions)?;

            let inner_needs_param = self.builder.needs_param(inner);

            if rule.is_parametric && !inner_needs_param {
                // TODO unclear if this should be an error or not
                bail!(&self.builder.funding,
                    "rule {:?} is parametric, but its body doesn't need parameters",
                    name
                );
            }
            if !rule.is_parametric && inner_needs_param {
                //println!("inner {} needs parameters", self.builder.node_to_string(inner));
                bail!(&self.builder.funding,
                    "rule {:?} is not parametric, but its body requires parameters",
                    name
                );
            }

            #[allow(clippy::assertions_on_constants)]
            if let Some(max_tokens) = rule.max_tokens {
                assert!(false, "max_tokens handled above for now");
                self.builder.join_props(
                    &[inner],
                    NodeProps {
                        max_tokens: Some(max_tokens),
                        // assume the user also wants capture
                        capture_name: Some(self.builder.funding.try_copy_str(name)?),
                        ..Default::default()
                    },
                )?
            } else if has_capture || (inner.is_parametric() && !rule.is_parametric)
            {
                self.builder.join_props(&[inner], props)?
            } else {
                inner
            }
        };
        Ok(id)
    }

    fn execute(mut self) -> Result<GrammarResult<'t>> {
        let _frame = self.enter::<frames::Execute<'_, 't, 'f>>()?;
        let mut grm = Grammar::default();
        for item in std::mem::take(&mut self.parsed.items) {
            let loc = item.location().clone();
            grm.process_item(item, &self.builder.funding, self.frames).map_err(|e| loc.augment(e))?;
        }
        let start_name = "start";
        ensure!(&self.builder.funding,
            grm.rules.contains_key(start_name),
            "no {} rule found",
            start_name
        );
        let ignore = std::mem::take(&mut grm.ignore);
        self.grammar = grm;

        let opts = std::mem::take(&mut self.grammar.llguidance_options);

        let mut skip_exprs = Vec::new();
        for exp in ignore {
            let expr = RegexAst::ExprRef(self.do_token_expansions(exp)?);
            self.builder.funding.try_push(&mut skip_exprs, expr)?;
        }
        let skip_regex = RegexAst::Or(skip_exprs);
        let skip = if opts.ignore_once {
            SkipSpec::once(skip_regex)
        } else {
            SkipSpec::unbounded(skip_regex)
        };
        let id = self.builder.add_grammar_with_skip(opts.general, skip)?;

        let start = self.do_rule(start_name, None)?;
        self.builder.set_start_node(start)?;

        let mut builder = self.builder;
        for (gg, loc, grm) in self.pending_grammars {
            let res = match grm {
                PendingGrammar::Json(json_schema) => JsonCompileOptions::new(&builder.funding)?
                    .json_to_llg_with_overrides(builder, json_schema)
                    .map_err(|e| loc.augment(e))?,
                PendingGrammar::Lark(items) => compile_lark(builder, ParsedLark { items }, self.frames)?,
            };
            builder = res.builder;
            builder.link_gen_grammar(gg, res.start_node)?;
        }

        Ok(builder.finalize(id))
    }
}

impl Grammar {
    fn add_token_def(&mut self, loc: &Location, local_name: String, regex: &str, funding: &ParserAllocationFunding, frames: &FundingScope<'_>) -> Result<()> {
        let _frame = frames::enter::<frames::AddToken<'_>>(frames, funding)?;
        ensure!(funding,
            !self.tokens.contains_key(&local_name),
            "duplicate token (in import): {:?}",
            local_name
        );

        let mut exprs = Vec::new();
        funding.try_push(&mut exprs, Expr {
            atom: Atom::Value(Value::LiteralRegex(funding.try_copy_str(regex)?, String::new())),
            op: None, range: None,
        })?;
        let mut conjuncts = Vec::new();
        funding.try_push(&mut conjuncts, Expansion(exprs))?;
        let mut aliases = Vec::new();
        funding.try_push(&mut aliases, Alias { conjuncts, param_cond: ParamCond::True, alias: None })?;
        let t = TokenDef {
            name: local_name, params: None, priority: None,
            expansions: Expansions(loc.clone(), aliases),
        };
        funding.try_insert(&mut self.tokens, funding.try_copy_str(&t.name)?, t)?;
        Ok(())
    }

    fn do_statement(&mut self, loc: &Location, statement: Statement, funding: &ParserAllocationFunding, frames: &FundingScope<'_>) -> Result<()> {
        let _frame = frames::enter::<frames::StatementFrame<'_>>(frames, funding)?;
        match statement {
            Statement::Ignore(exp) => {
                funding.try_push(&mut self.ignore, exp)?;
            }
            Statement::Import { path, alias } => {
                let regex = lookup_common_regex(&path, funding)?;
                let local_name = match alias {
                    Some(alias) => alias,
                    None => funding.try_copy_str(path.split('.').next_back().unwrap())?,
                };
                self.add_token_def(loc, local_name, regex, funding, frames)?;
            }
            Statement::MultiImport { path, names } => {
                for n in names {
                    let qname = funding.try_format(format_args!("{path}.{n}"))?;
                    let regex = lookup_common_regex(&qname, funding)?;
                    self.add_token_def(loc, n, regex, funding, frames)?;
                }
            }
            Statement::LLGuidance(json_value) => {
                self.llguidance_options.apply_value(json_value, funding, frames)?;
            }
            Statement::OverrideRule(_) => {
                bail!(funding, "override statement not supported yet");
            }
            Statement::Declare(_) => {
                bail!(funding, "declare statement not supported yet");
            }
        }
        Ok(())
    }

    fn process_item(&mut self, item: Item, funding: &ParserAllocationFunding, frames: &FundingScope<'_>) -> Result<()> {
        let _frame = frames::enter::<frames::ItemFrame<'_>>(frames, funding)?;
        match item {
            Item::Rule(rule) => {
                ensure!(funding, rule.params.is_none(), "params not supported yet");
                ensure!(funding, rule.priority.is_none(), "priority not supported yet");
                ensure!(funding,
                    !self.rules.contains_key(&rule.name),
                    "duplicate rule: {:?}",
                    rule.name
                );
                funding.try_insert(&mut self.rules, funding.try_copy_str(&rule.name)?, rule)?;
            }
            Item::Token(token_def) => {
                ensure!(funding, token_def.params.is_none(), "params not supported yet");
                ensure!(funding, token_def.priority.is_none(), "priority not supported yet");
                ensure!(funding,
                    !self.tokens.contains_key(&token_def.name),
                    "duplicate token: {:?}",
                    token_def.name
                );
                funding.try_insert(&mut self.tokens, funding.try_copy_str(&token_def.name)?, token_def)?;
            }
            Item::Statement(loc, statement) => {
                self.do_statement(&loc, statement, funding, frames)?;
            }
        }
        Ok(())
    }
}

fn compile_lark_regex(builder: &mut GrammarBuilder, l: RegexExt, frames: &FundingScope<'_>) -> Result<RegexId> {
    let _frame = frames::enter::<frames::ExtendedRegex<'_, '_>>(frames, &builder.funding)?;
    let mut fields_set = [""; 3];
    let mut fields_len = 0;
    if l.substring_chunks.is_some() {
        fields_set[fields_len] = "substring_chunks"; fields_len += 1;
    }
    if l.substring_words.is_some() {
        fields_set[fields_len] = "substring_words"; fields_len += 1;
    }
    if l.substring_chars.is_some() {
        fields_set[fields_len] = "substring_chars"; fields_len += 1;
    }
    if fields_len == 0 {
        bail!(&builder.funding, "no fields set on %regex");
    }
    if fields_len > 1 {
        bail!(&builder.funding, "only one field can be set on %regex; got {:?}", &fields_set[..fields_len]);
    }

    let bld = &mut builder.regex.spec.regex_builder;

    let eref = if let Some(s) = l.substring_words {
        substring(bld, chunk_into_words(&s))?
    } else if let Some(s) = l.substring_chars {
        substring(bld, chunk_into_chars(&s))?
    } else if let Some(s) = l.substring_chunks {
        substring(bld, s.iter().map(|s| s.as_str()))?
    } else {
        unreachable!()
    };

    Ok(eref)
}

struct Escaped<'a>(&'a str);
impl std::fmt::Display for Escaped<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for ch in self.0.chars() { std::fmt::Display::fmt(&EscapedChar(ch), f)?; }
        Ok(())
    }
}
struct EscapedChar(char);
impl std::fmt::Display for EscapedChar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if regex_syntax::is_meta_character(self.0) { f.write_str("\\")?; }
        std::fmt::Write::write_char(f, self.0)
    }
}
