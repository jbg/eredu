//! Copies actual compiled grammar declarations, never recompiling source syntax.
use super::{
    CGrammar, CSymIdx, CSymbol, GenGrammarOptions, GrammarId, ParamCond, ParamExpr, RhsPtr,
    SymFlags, SymbolProps,
};
use crate::earley::lexerspec::{LexerSpec, LexerSpecCopyFailure};
use std::{
    alloc::Layout,
    collections::TryReserveError,
    fmt,
    mem::{size_of, size_of_val},
};
/// Source condition geometry cannot be represented.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConditionSourceError;
impl fmt::Display for ConditionSourceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("condition source geometry overflow")
    }
}
impl std::error::Error for ConditionSourceError {}
/// Actual boxed descendants and bounded constructor/inspection depth.
#[derive(Clone, Copy, Debug)]
pub struct ConditionCopyRequirements {
    buffers: usize,
    controls: usize,
    total: usize,
}
impl ConditionCopyRequirements {
    /// Exact boxed descendant population.
    pub fn buffer_bytes(&self) -> usize {
        self.buffers
    }
    /// Actual maximum live constructor/inspection frames.
    pub fn control_bytes(&self) -> usize {
        self.controls
    }
    /// Complete local condition copy requirement.
    pub fn required_bytes(&self) -> usize {
        self.total
    }
}
/// A loan of one actual condition; it cannot fabricate parameter semantics.
pub struct ConditionCopyPlan<'a> {
    source: &'a ParamCond,
    requirements: ConditionCopyRequirements,
}
impl fmt::Debug for ConditionCopyPlan<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ConditionCopyPlan")
            .field("requirements", &self.requirements)
            .finish()
    }
}
fn condition_geometry(
    c: &ParamCond,
    depth: usize,
    boxes: &mut usize,
    max_depth: &mut usize,
) -> Option<()> {
    *max_depth = (*max_depth).max(depth);
    match c {
        ParamCond::And(a, b) | ParamCond::Or(a, b) => {
            *boxes = boxes.checked_add(2)?;
            condition_geometry(a, depth.checked_add(1)?, boxes, max_depth)?;
            condition_geometry(b, depth.checked_add(1)?, boxes, max_depth)?;
        }
        ParamCond::Not(a) => {
            *boxes = boxes.checked_add(1)?;
            condition_geometry(a, depth.checked_add(1)?, boxes, max_depth)?;
        }
        _ => {}
    }
    Some(())
}
fn condition_copy(c: &ParamCond) -> ParamCond {
    match c {
        ParamCond::True => ParamCond::True,
        ParamCond::NE(a, b) => ParamCond::NE(*a, *b),
        ParamCond::EQ(a, b) => ParamCond::EQ(*a, *b),
        ParamCond::LE(a, b) => ParamCond::LE(*a, *b),
        ParamCond::LT(a, b) => ParamCond::LT(*a, *b),
        ParamCond::GE(a, b) => ParamCond::GE(*a, *b),
        ParamCond::GT(a, b) => ParamCond::GT(*a, *b),
        ParamCond::BitCountNE(a, b) => ParamCond::BitCountNE(*a, *b),
        ParamCond::BitCountEQ(a, b) => ParamCond::BitCountEQ(*a, *b),
        ParamCond::BitCountLE(a, b) => ParamCond::BitCountLE(*a, *b),
        ParamCond::BitCountLT(a, b) => ParamCond::BitCountLT(*a, *b),
        ParamCond::BitCountGE(a, b) => ParamCond::BitCountGE(*a, *b),
        ParamCond::BitCountGT(a, b) => ParamCond::BitCountGT(*a, *b),
        ParamCond::And(a, b) => {
            ParamCond::And(Box::new(condition_copy(a)), Box::new(condition_copy(b)))
        }
        ParamCond::Or(a, b) => {
            ParamCond::Or(Box::new(condition_copy(a)), Box::new(condition_copy(b)))
        }
        ParamCond::Not(a) => ParamCond::Not(Box::new(condition_copy(a))),
    }
}
impl ParamCond {
    /// Quotes actual immutable condition topology without evaluating parameters.
    pub fn source_copy_plan(&self) -> Result<ConditionCopyPlan<'_>, ConditionSourceError> {
        let mut boxes = 0;
        let mut depth = 0;
        condition_geometry(self, 1, &mut boxes, &mut depth).ok_or(ConditionSourceError)?;
        let buffers = Layout::array::<ParamCond>(boxes)
            .map_err(|_| ConditionSourceError)?
            .size();
        let frame = size_of::<(
            &ParamCond,
            usize,
            &mut usize,
            &mut usize,
            Option<()>,
            ParamCond,
            Box<ParamCond>,
            Box<ParamCond>,
        )>();
        let parts = [
            size_of::<ConditionCopyPlan<'_>>(),
            size_of::<ConditionCopyRequirements>(),
            size_of::<ConditionSourceError>(),
            size_of::<Result<ConditionCopyPlan<'_>, ConditionSourceError>>(),
            size_of::<Result<Layout, std::alloc::LayoutError>>(),
        ];
        let controls = parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
            .and_then(|n| n.checked_add(frame.checked_mul(depth)?))
            .ok_or(ConditionSourceError)?;
        let total = buffers.checked_add(controls).ok_or(ConditionSourceError)?;
        Ok(ConditionCopyPlan {
            source: self,
            requirements: ConditionCopyRequirements {
                buffers,
                controls,
                total,
            },
        })
    }
}
impl ConditionCopyPlan<'_> {
    /// Exact local quote for this condition.
    pub fn requirements(&self) -> ConditionCopyRequirements {
        self.requirements
    }
    /// The same ordinary constructor. Every Box has its actual source-node quote;
    /// no runtime condition evaluation or expanding cache is involved.
    pub fn compile(self) -> ParamCond {
        condition_copy(self.source)
    }
}
/// Finite complete copied grammar population.
#[derive(Clone, Copy, Debug)]
pub struct CompiledGrammarCopyRequirements {
    buffers: usize,
    retained: usize,
    controls: usize,
    total: usize,
}
impl CompiledGrammarCopyRequirements {
    /// Exact completed destination population. Constructor scratch and fixed
    /// frames are excluded from persistent source/residency accounting.
    pub fn retained_bytes(&self) -> usize {
        self.retained
    }
    /// All copied arrays, strings, conditions and lexer declarations.
    pub fn buffer_bytes(&self) -> usize {
        self.buffers
    }
    /// Constructor, failure and actual child frames.
    pub fn control_bytes(&self) -> usize {
        self.controls
    }
    /// Complete local source-copy requirement.
    pub fn required_bytes(&self) -> usize {
        self.total
    }
}
#[derive(Debug)]
enum Cause {
    Overflow,
    Capacity,
    Vector(TryReserveError),
    Lexer(LexerSpecCopyFailure),
    Condition(ConditionSourceError),
}
impl fmt::Display for Cause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Overflow => f.write_str("compiled grammar source geometry overflow"),
            Self::Capacity => f.write_str("compiled grammar destination differs from source"),
            Self::Vector(e) => fmt::Display::fmt(e, f),
            Self::Lexer(e) => fmt::Display::fmt(e, f),
            Self::Condition(e) => fmt::Display::fmt(e, f),
        }
    }
}
impl std::error::Error for Cause {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Vector(e) => Some(e),
            Self::Lexer(e) => Some(e),
            Self::Condition(e) => Some(e),
            _ => None,
        }
    }
}
#[derive(Default)]
struct SymbolPartial {
    name: Vec<u8>,
    capture: Vec<u8>,
    stop: Vec<u8>,
    grammar: Vec<u8>,
    nullable: Vec<ParamCond>,
    rules: Vec<RhsPtr>,
    conditions: Vec<ParamCond>,
}
#[derive(Default)]
struct Partial {
    lexer: Option<LexerSpec>,
    symbols: Vec<CSymbol>,
    symbol: SymbolPartial,
    rhs: Vec<CSymIdx>,
    params: Vec<ParamExpr>,
    index: Vec<CSymIdx>,
    flags: Vec<SymFlags>,
}
/// Owns all completed grammar/lexer/symbol prefixes through failure retirement.
pub struct CompiledGrammarCopyFailure {
    cause: Cause,
    partial: Option<Partial>,
}
impl fmt::Debug for CompiledGrammarCopyFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CompiledGrammarCopyFailure")
            .field("cause", &self.cause)
            .field("retains_prefix", &self.partial.is_some())
            .finish()
    }
}
impl fmt::Display for CompiledGrammarCopyFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.cause, f)
    }
}
impl std::error::Error for CompiledGrammarCopyFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.cause)
    }
}
/// Exact compiled source loan. No rule renumbering, grammar resolution or parser
/// creation is performed by this constructor.
pub struct CompiledGrammarCopyPlan<'a> {
    source: &'a CGrammar,
    requirements: CompiledGrammarCopyRequirements,
    text_bytes: usize,
}
impl fmt::Debug for CompiledGrammarCopyPlan<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CompiledGrammarCopyPlan")
            .field("requirements", &self.requirements)
            .finish()
    }
}
fn bytes<T>(n: usize) -> Result<usize, Cause> {
    Layout::array::<T>(n)
        .map(|l| l.size())
        .map_err(|_| Cause::Overflow)
}
fn add(n: &mut usize, value: usize) -> Result<(), Cause> {
    *n = n.checked_add(value).ok_or(Cause::Overflow)?;
    Ok(())
}
fn reserve<T>(v: &mut Vec<T>, n: usize) -> Result<(), Cause> {
    v.try_reserve_exact(n).map_err(Cause::Vector)?;
    if v.capacity() != n {
        return Err(Cause::Capacity);
    }
    Ok(())
}
fn text(v: Vec<u8>) -> String {
    String::from_utf8(v).expect("actual source String bytes")
}
fn copy_text(source: &str, dst: &mut Vec<u8>, left: &mut usize) -> Result<(), Cause> {
    *left = left.checked_sub(source.len()).ok_or(Cause::Capacity)?;
    reserve(dst, source.len())?;
    dst.extend_from_slice(source.as_bytes());
    Ok(())
}
fn condition_list(source: &[ParamCond], dst: &mut Vec<ParamCond>) -> Result<(), Cause> {
    reserve(dst, source.len())?;
    for c in source {
        dst.push(c.source_copy_plan().map_err(Cause::Condition)?.compile());
    }
    Ok(())
}
fn symbol_copy(s: &CSymbol, p: &mut SymbolPartial, left: &mut usize) -> Result<CSymbol, Cause> {
    copy_text(&s.name, &mut p.name, left)?;
    if let Some(v) = &s.props.capture_name {
        copy_text(v, &mut p.capture, left)?;
    }
    if let Some(v) = &s.props.stop_capture_name {
        copy_text(v, &mut p.stop, left)?;
    }
    if let Some(g) = &s.gen_grammar {
        let GrammarId::Name(v) = &g.grammar;
        copy_text(v, &mut p.grammar, left)?;
    }
    condition_list(&s.cond_nullable, &mut p.nullable)?;
    reserve(&mut p.rules, s.rules.len())?;
    p.rules.extend_from_slice(&s.rules);
    condition_list(&s.rules_cond, &mut p.conditions)?;
    let p = std::mem::take(p);
    Ok(CSymbol {
        idx: s.idx,
        name: text(p.name),
        is_terminal: s.is_terminal,
        is_nullable: s.is_nullable,
        cond_nullable: p.nullable,
        props: SymbolProps {
            max_tokens: s.props.max_tokens,
            capture_name: s.props.capture_name.as_ref().map(|_| text(p.capture)),
            stop_capture_name: s.props.stop_capture_name.as_ref().map(|_| text(p.stop)),
            temperature: s.props.temperature,
            grammar_id: s.props.grammar_id,
            is_start: s.props.is_start,
            parametric: s.props.parametric,
        },
        gen_grammar: s.gen_grammar.as_ref().map(|g| GenGrammarOptions {
            grammar: GrammarId::Name(text(p.grammar)),
            temperature: g.temperature,
        }),
        rules: p.rules,
        rules_cond: p.conditions,
        sym_flags: s.sym_flags,
        lexeme: s.lexeme,
    })
}
impl CGrammar {
    /// Quotes the exact compiled symbols/rules and their retained lexer source.
    pub fn source_copy_plan(
        &self,
    ) -> Result<CompiledGrammarCopyPlan<'_>, CompiledGrammarCopyFailure> {
        CompiledGrammarCopyPlan::prepare(self).map_err(|cause| CompiledGrammarCopyFailure {
            cause,
            partial: None,
        })
    }
}
impl<'a> CompiledGrammarCopyPlan<'a> {
    fn prepare(s: &'a CGrammar) -> Result<Self, Cause> {
        let lexer = s
            .lexer_spec
            .source_copy_plan()
            .map_err(Cause::Lexer)?
            .requirements();
        let mut buffers = lexer.buffer_bytes();
        let scratch = lexer
            .buffer_bytes()
            .checked_sub(lexer.retained_bytes())
            .ok_or(Cause::Overflow)?;
        let mut controls = lexer.control_bytes();
        let mut text_bytes = 0;
        add(&mut buffers, bytes::<CSymbol>(s.symbols.len())?)?;
        add(&mut buffers, bytes::<CSymIdx>(s.rhs_elements.len())?)?;
        add(&mut buffers, bytes::<ParamExpr>(s.rhs_params.len())?)?;
        add(&mut buffers, bytes::<CSymIdx>(s.rhs_ptr_to_sym_idx.len())?)?;
        add(
            &mut buffers,
            bytes::<SymFlags>(s.rhs_ptr_to_sym_flags.len())?,
        )?;
        for symbol in &s.symbols {
            add(&mut text_bytes, symbol.name.len())?;
            if let Some(v) = &symbol.props.capture_name {
                add(&mut text_bytes, v.len())?;
            }
            if let Some(v) = &symbol.props.stop_capture_name {
                add(&mut text_bytes, v.len())?;
            }
            if let Some(g) = &symbol.gen_grammar {
                let GrammarId::Name(v) = &g.grammar;
                add(&mut text_bytes, v.len())?;
            }
            add(&mut buffers, bytes::<RhsPtr>(symbol.rules.len())?)?;
            add(
                &mut buffers,
                bytes::<ParamCond>(symbol.cond_nullable.len())?,
            )?;
            add(&mut buffers, bytes::<ParamCond>(symbol.rules_cond.len())?)?;
            for condition in symbol.cond_nullable.iter().chain(&symbol.rules_cond) {
                let q = condition
                    .source_copy_plan()
                    .map_err(Cause::Condition)?
                    .requirements();
                add(&mut buffers, q.buffer_bytes())?;
                add(&mut controls, q.control_bytes())?;
            }
        }
        add(&mut buffers, text_bytes)?;
        let parts = [
            size_of::<Self>(),
            size_of::<CompiledGrammarCopyRequirements>(),
            size_of::<CompiledGrammarCopyFailure>(),
            size_of::<Cause>(),
            size_of::<Partial>(),
            size_of::<SymbolPartial>(),
            size_of::<CGrammar>(),
            size_of::<CSymbol>(),
            size_of::<SymbolProps>(),
            size_of::<Option<GenGrammarOptions>>(),
            size_of::<GrammarId>(),
            size_of::<Result<Self, Cause>>(),
            size_of::<Result<Self, CompiledGrammarCopyFailure>>(),
            size_of::<Result<CGrammar, CompiledGrammarCopyFailure>>(),
            size_of::<Result<(), Cause>>(),
            size_of::<Result<CSymbol, Cause>>(),
            size_of::<Result<(), TryReserveError>>(),
            size_of::<Result<String, std::string::FromUtf8Error>>(),
            size_of::<Result<Layout, std::alloc::LayoutError>>(),
            size_of::<std::slice::Iter<'_, CSymbol>>(),
            size_of::<
                std::iter::Chain<std::slice::Iter<'_, ParamCond>, std::slice::Iter<'_, ParamCond>>,
            >(),
            size_of::<(&CSymbol, &mut SymbolPartial, &mut usize)>(),
            size_of::<(&CGrammar, &mut Partial, &mut usize)>(),
        ];
        controls = parts
            .into_iter()
            .try_fold(
                controls
                    .checked_add(size_of_val(&parts))
                    .ok_or(Cause::Overflow)?,
                usize::checked_add,
            )
            .ok_or(Cause::Overflow)?;
        let total = buffers.checked_add(controls).ok_or(Cause::Overflow)?;
        let retained = buffers.checked_sub(scratch).ok_or(Cause::Overflow)?;
        Ok(Self {
            source: s,
            text_bytes,
            requirements: CompiledGrammarCopyRequirements {
                buffers,
                retained,
                controls,
                total,
            },
        })
    }
    /// The complete local constructor population.
    pub fn requirements(&self) -> CompiledGrammarCopyRequirements {
        self.requirements
    }
    /// Same source constructor as ordinary Clone; it preserves all symbol/rule
    /// indices, conditions and capture semantics without an inference engine.
    pub fn compile(self) -> Result<CGrammar, CompiledGrammarCopyFailure> {
        let s = self.source;
        let mut p = Partial::default();
        let mut left = self.text_bytes;
        let result = (|| -> Result<(), Cause> {
            p.lexer = Some(
                s.lexer_spec
                    .source_copy_plan()
                    .map_err(Cause::Lexer)?
                    .compile()
                    .map_err(Cause::Lexer)?,
            );
            reserve(&mut p.symbols, s.symbols.len())?;
            for symbol in &s.symbols {
                p.symbols
                    .push(symbol_copy(symbol, &mut p.symbol, &mut left)?);
            }
            reserve(&mut p.rhs, s.rhs_elements.len())?;
            p.rhs.extend_from_slice(&s.rhs_elements);
            reserve(&mut p.params, s.rhs_params.len())?;
            p.params.extend_from_slice(&s.rhs_params);
            reserve(&mut p.index, s.rhs_ptr_to_sym_idx.len())?;
            p.index.extend_from_slice(&s.rhs_ptr_to_sym_idx);
            reserve(&mut p.flags, s.rhs_ptr_to_sym_flags.len())?;
            p.flags.extend_from_slice(&s.rhs_ptr_to_sym_flags);
            if left != 0 {
                return Err(Cause::Capacity);
            }
            Ok(())
        })();
        if let Err(cause) = result {
            return Err(CompiledGrammarCopyFailure {
                cause,
                partial: Some(p),
            });
        }
        Ok(CGrammar {
            parametric: s.parametric,
            start_symbol: s.start_symbol,
            lexer_spec: p.lexer.take().expect("completed lexer declaration"),
            symbols: p.symbols,
            rhs_elements: p.rhs,
            rhs_params: p.params,
            rhs_ptr_to_sym_idx: p.index,
            rhs_ptr_to_sym_flags: p.flags,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        api::ParserLimits,
        earley::grammar::{Grammar, ParamRef, ParamValue},
    };
    use derivre::RegexAst;
    #[test]
    fn compiled_source_copy_preserves_parametric_rules_captures_and_failed_symbol_prefix() {
        let condition = ParamCond::And(
            Box::new(ParamCond::GE(ParamRef::full(), ParamValue(2))),
            Box::new(ParamCond::Not(Box::new(ParamCond::BitCountEQ(
                ParamRef::full(),
                1,
            )))),
        );
        let condition_copy = condition.source_copy_plan().unwrap().compile();
        assert_eq!(condition, condition_copy);
        for m in [0, 1, 2, 3, 7, 8, u64::MAX] {
            assert_eq!(
                condition.eval(ParamValue(m)),
                condition_copy.eval(ParamValue(m))
            );
        }
        let mut lexer = LexerSpec::new().unwrap();
        lexer.setup_lexeme_class(RegexAst::NoMatch).unwrap();
        let terminal = lexer
            .add_simple_literal("word".into(), "done", false)
            .unwrap();
        let mut grammar = Grammar::new(Some("actual source".into()));
        let start = grammar.fresh_symbol_ext(
            "start",
            SymbolProps {
                is_start: true,
                parametric: true,
                ..SymbolProps::default()
            },
        );
        let word = grammar.fresh_symbol_ext(
            "word",
            SymbolProps {
                capture_name: Some("word-capture".into()),
                stop_capture_name: Some("word-stop".into()),
                temperature: 0.25,
                max_tokens: 17,
                ..SymbolProps::default()
            },
        );
        grammar.make_terminal(word, terminal, &lexer).unwrap();
        grammar
            .add_rule_ext(start, condition, vec![(word, ParamExpr::Null)])
            .unwrap();
        grammar
            .add_rule_ext(start, condition_copy, Vec::new())
            .unwrap();
        let source = grammar.compile(lexer, &ParserLimits::default()).unwrap();
        assert!(source.parametric);
        let plan = source.source_copy_plan().unwrap();
        let quote = plan.requirements();
        let copied = plan.compile().unwrap();
        assert!(quote.required_bytes() > quote.buffer_bytes());
        assert!(quote.retained_bytes() < quote.buffer_bytes());
        assert_eq!(copied.start_symbol, source.start_symbol);
        assert_eq!(copied.rhs_elements, source.rhs_elements);
        assert_eq!(copied.rhs_params, source.rhs_params);
        assert_eq!(copied.rhs_ptr_to_sym_idx, source.rhs_ptr_to_sym_idx);
        assert_eq!(
            copied
                .rhs_ptr_to_sym_flags
                .iter()
                .map(|v| v.0)
                .collect::<Vec<_>>(),
            source
                .rhs_ptr_to_sym_flags
                .iter()
                .map(|v| v.0)
                .collect::<Vec<_>>()
        );
        assert_eq!(copied.symbols.len(), source.symbols.len());
        for (a, b) in copied.symbols.iter().zip(&source.symbols) {
            assert_eq!(a.idx, b.idx);
            assert_eq!(a.name, b.name);
            assert_eq!(a.props, b.props);
            assert_eq!(a.cond_nullable, b.cond_nullable);
            assert_eq!(a.rules_cond, b.rules_cond);
            assert_eq!(a.rules, b.rules);
            assert_eq!(a.gen_grammar, b.gen_grammar);
            assert_eq!(a.sym_flags.0, b.sym_flags.0);
        }
        assert!(copied.symbols.iter().any(|s| !s.cond_nullable.is_empty()));
        assert!(
            copied
                .symbols
                .iter()
                .any(|s| s.props.capture_name.as_deref() == Some("word-capture"))
        );
        let mut failing = source.source_copy_plan().unwrap();
        failing.text_bytes = source.symbols[0].name.len();
        let failure = match failing.compile() {
            Err(e) => e,
            Ok(_) => panic!("incomplete symbol text accepted"),
        };
        assert!(matches!(failure.cause, Cause::Capacity));
        drop(grammar);
        drop(source);
        drop(copied);
        let p = failure.partial.as_ref().unwrap();
        assert!(p.lexer.is_some());
        assert_eq!(p.symbols.len(), 1);
        assert_eq!(p.symbols[0].name, "NULL");
    }
}
