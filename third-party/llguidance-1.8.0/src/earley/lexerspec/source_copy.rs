//! Exact copied lexer declarations, preserving the ordinary semantic owners.
use super::{LexemeClass, LexemeIdx, LexemeSpec, LexerSpec, MatchingLexemes, SkipRepetition};
use derivre::{
    ParserAllocationFunding, ParserAllocationFailure,
    prepared_funding::{PreparedFunding, FrameError},
    ExprRef, JsonQuoteOptions, RegexAst, RegexAstCopyFailure, RegexAstCopyPlan, RegexBuilder,
    RegexBuilderCopyFailure, SourceHashMap, SourceMapReserveError,
};
use std::{
    alloc::Layout,
    collections::TryReserveError,
    fmt,
    mem::{size_of, size_of_val},
    ops::RangeInclusive,
};
use toktrie::TokenId;
/// Complete source-copy population; no future lexer/DFA allocation authority.
#[derive(Clone, Copy, Debug)]
pub struct LexerSourceCopyRequirements {
    buffers: usize,
    retained: usize,
    controls: usize,
    total: usize,
}
impl LexerSourceCopyRequirements {
    /// Actual destination buffers after all child constructor scratch retires.
    pub fn retained_bytes(&self) -> usize {
        self.retained
    }
    /// Actual independent payload/child/table bytes.
    pub fn buffer_bytes(&self) -> usize {
        self.buffers
    }
    /// Fixed source/constructor/failure frames and child frames.
    pub fn control_bytes(&self) -> usize {
        self.controls
    }
    /// Complete local copy requirement.
    pub fn required_bytes(&self) -> usize {
        self.total
    }
}
#[derive(Debug)]
pub(super) enum Cause {
    Overflow,
    Funding(ParserAllocationFailure),
    Capacity,
    Vector(TryReserveError),
    Table(SourceMapReserveError),
    Ast(RegexAstCopyFailure),
    Builder(RegexBuilderCopyFailure),
}
impl fmt::Display for Cause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Overflow => f.write_str("lexer source-copy geometry overflow"),
            Self::Funding(error) => fmt::Display::fmt(error, f),
            Self::Capacity => f.write_str("lexer copy destination differs from source"),
            Self::Vector(e) => fmt::Display::fmt(e, f),
            Self::Table(e) => fmt::Display::fmt(e, f),
            Self::Ast(e) => fmt::Display::fmt(e, f),
            Self::Builder(e) => fmt::Display::fmt(e, f),
        }
    }
}
impl std::error::Error for Cause {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Funding(error) => Some(error),
            Self::Vector(e) => Some(e),
            Self::Table(e) => Some(e),
            Self::Ast(e) => Some(e),
            Self::Builder(e) => Some(e),
            _ => None,
        }
    }
}
pub(super) fn frame(error: FrameError<ParserAllocationFailure>) -> Cause {
    match error { FrameError::Overflow => Cause::Overflow, FrameError::Funding(error) => Cause::Funding(error) }
}
fn bytes<T>(n: usize) -> Result<usize, Cause> {
    Layout::array::<T>(n)
        .map(|l| l.size())
        .map_err(|_| Cause::Overflow)
}
fn add(total: &mut usize, n: usize) -> Result<(), Cause> {
    *total = total.checked_add(n).ok_or(Cause::Overflow)?;
    Ok(())
}
fn reserve<T>(v: &mut Vec<T>, n: usize) -> Result<(), Cause> {
    v.try_reserve_exact(n).map_err(Cause::Vector)?;
    if v.capacity() != n {
        return Err(Cause::Capacity);
    }
    Ok(())
}
fn text(bytes: Vec<u8>) -> String {
    String::from_utf8(bytes).expect("copied actual source String")
}
#[derive(Default)]
struct LexemePartial {
    name: Vec<u8>,
    set: Vec<LexemeIdx>,
    rx: Option<RegexAst>,
    options: Vec<u8>,
    ranges: Vec<RangeInclusive<TokenId>>,
}
/// Failed lexeme copies retain actual sibling/child destinations.
pub struct LexemeCopyFailure {
    cause: Cause,
    partial: LexemePartial,
}
impl fmt::Debug for LexemeCopyFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LexemeCopyFailure")
            .field("cause", &self.cause)
            .field("retains_ast", &self.partial.rx.is_some())
            .finish()
    }
}
impl fmt::Display for LexemeCopyFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.cause, f)
    }
}
impl std::error::Error for LexemeCopyFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.cause)
    }
}
/// Loan of one actual lexeme and its existing AST source constructor.
pub struct LexemeCopyPlan<'a> {
    source: &'a LexemeSpec,
    ast: RegexAstCopyPlan<'a>,
    requirements: LexerSourceCopyRequirements,
}
impl fmt::Debug for LexemeCopyPlan<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LexemeCopyPlan")
            .field("requirements", &self.requirements)
            .finish()
    }
}
impl LexemeSpec {
    /// Copies actual source metadata and AST; no expression compilation occurs.
    pub fn source_copy_plan<F: PreparedFunding<Error = ParserAllocationFailure>>(
        &self, funding: &F,
    ) -> Result<LexemeCopyPlan<'_>, LexemeCopyFailure> {
        LexemeCopyPlan::prepare(self, funding).map_err(|cause| LexemeCopyFailure {
            cause,
            partial: LexemePartial::default(),
        })
    }
}
type LexemeInspectionFrame<'a, F> = (
    &'a LexemeSpec, &'a F, LexemeCopyPlan<'a>, LexerSourceCopyRequirements, LexemeCopyFailure,
    usize, usize, usize, usize, usize, [usize; 16],
    Result<LexemeCopyPlan<'a>, Cause>, Result<LexemeCopyPlan<'a>, LexemeCopyFailure>,
    Result<Layout, std::alloc::LayoutError>, Option<usize>,
);
impl<'a> LexemeCopyPlan<'a> {
    fn prepare<F: PreparedFunding<Error = ParserAllocationFailure>>(
        source: &'a LexemeSpec, funding: &F,
    ) -> Result<Self, Cause> {
        let _frame = funding.frame(size_of::<LexemeInspectionFrame<'_, F>>()).map_err(frame)?;
        let ast = source.rx.source_copy_plan(funding).map_err(Cause::Ast)?;
        let mut buffers = ast.requirements().buffer_bytes();
        let scratch = ast
            .requirements()
            .buffer_bytes()
            .checked_sub(ast.requirements().retained_bytes())
            .ok_or(Cause::Overflow)?;
        add(&mut buffers, source.name.len())?;
        if let MatchingLexemes::Many(v) = &source.single_set {
            add(&mut buffers, bytes::<LexemeIdx>(v.len())?)?;
        }
        add(
            &mut buffers,
            bytes::<RangeInclusive<TokenId>>(source.token_ranges.len())?,
        )?;
        if let Some(o) = &source.json_options {
            add(&mut buffers, o.allowed_escapes.len())?;
        }
        let parts = [
            size_of::<Self>(),
            size_of::<LexerSourceCopyRequirements>(),
            size_of::<LexemeCopyFailure>(),
            size_of::<LexemePartial>(),
            size_of::<Cause>(),
            size_of::<LexemeSpec>(),
            size_of::<MatchingLexemes>(),
            size_of::<Option<JsonQuoteOptions>>(),
            size_of::<Result<Self, Cause>>(),
            size_of::<Result<Self, LexemeCopyFailure>>(),
            size_of::<Result<LexemeSpec, LexemeCopyFailure>>(),
            size_of::<Result<(), Cause>>(),
            size_of::<Result<(), TryReserveError>>(),
            size_of::<Result<String, std::string::FromUtf8Error>>(),
            size_of::<Result<Layout, std::alloc::LayoutError>>(),
            size_of::<(&LexemeSpec, &mut LexemePartial)>(),
        ];
        let mut controls = parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
            .ok_or(Cause::Overflow)?;
        add(&mut controls, ast.requirements().control_bytes())?;
        // LexerSpec's finite constructor reuses this exact child inspection worker.
        // Its entire fixed entry/guard population is paid in the child's quote.
        add(&mut controls, derivre::prepared_funding::frame_control_bytes::<ParserAllocationFailure>(
            size_of::<LexemeInspectionFrame<'_, F>>()
        ).ok_or(Cause::Overflow)?)?;
        let total = buffers.checked_add(controls).ok_or(Cause::Overflow)?;
        let retained = buffers.checked_sub(scratch).ok_or(Cause::Overflow)?;
        Ok(Self {
            source,
            ast,
            requirements: LexerSourceCopyRequirements {
                buffers,
                retained,
                controls,
                total,
            },
        })
    }
    /// Actual independent local-copy population.
    pub fn requirements(&self) -> LexerSourceCopyRequirements {
        self.requirements
    }
    /// Same source constructor consumed by ordinary Clone.
    pub fn compile(self) -> Result<LexemeSpec, LexemeCopyFailure> {
        let s = self.source;
        let mut p = LexemePartial::default();
        let result = (|| -> Result<(), Cause> {
            reserve(&mut p.name, s.name.len())?;
            p.name.extend_from_slice(s.name.as_bytes());
            if let MatchingLexemes::Many(v) = &s.single_set {
                reserve(&mut p.set, v.len())?;
                p.set.extend_from_slice(v);
            }
            p.rx = Some(self.ast.compile().map_err(Cause::Ast)?);
            if let Some(o) = &s.json_options {
                reserve(&mut p.options, o.allowed_escapes.len())?;
                p.options.extend_from_slice(o.allowed_escapes.as_bytes());
            }
            reserve(&mut p.ranges, s.token_ranges.len())?;
            p.ranges.extend_from_slice(&s.token_ranges);
            Ok(())
        })();
        if let Err(cause) = result {
            return Err(LexemeCopyFailure { cause, partial: p });
        }
        let single_set = match &s.single_set {
            MatchingLexemes::None => MatchingLexemes::None,
            MatchingLexemes::One(v) => MatchingLexemes::One(*v),
            MatchingLexemes::Two(v) => MatchingLexemes::Two(*v),
            MatchingLexemes::Many(_) => MatchingLexemes::Many(p.set),
        };
        Ok(LexemeSpec {
            idx: s.idx,
            single_set,
            name: text(p.name),
            rx: p.rx.take().expect("completed actual AST"),
            class: s.class,
            compiled_rx: s.compiled_rx,
            ends_at_eos: s.ends_at_eos,
            lazy: s.lazy,
            contextual: s.contextual,
            max_tokens: s.max_tokens,
            is_extra: s.is_extra,
            is_suffix: s.is_suffix,
            is_skip: s.is_skip,
            skip_repetition: s.skip_repetition,
            json_options: s.json_options.as_ref().map(|o| JsonQuoteOptions {
                allowed_escapes: text(p.options),
                raw_mode: o.raw_mode,
            }),
            token_ranges: p.ranges,
        })
    }
}
type Classes = SourceHashMap<(ExprRef, SkipRepetition), LexemeClass>;
struct Partial {
    lexemes: Vec<LexemeSpec>,
    builder: Option<RegexBuilder>,
    skip: Vec<LexemeIdx>,
    classes: Classes,
    warnings: Vec<(String, usize)>,
    pending_warning: Vec<u8>,
}
/// Retains the actual completed declaration prefix plus a failed child source.
pub struct LexerSpecCopyFailure {
    cause: Option<Cause>,
    lexeme: Option<LexemeCopyFailure>,
    partial: Option<Partial>,
}
impl LexerSpecCopyFailure {
    pub(super) fn inspection(cause: Cause) -> Self {
        Self { cause: Some(cause), lexeme: None, partial: None }
    }
}
impl fmt::Debug for LexerSpecCopyFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LexerSpecCopyFailure")
            .field("cause", &self.cause)
            .field("lexeme", &self.lexeme)
            .field("retains_prefix", &self.partial.is_some())
            .finish()
    }
}
impl fmt::Display for LexerSpecCopyFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(c) = &self.cause {
            fmt::Display::fmt(c, f)
        } else if let Some(c) = &self.lexeme {
            fmt::Display::fmt(c, f)
        } else {
            f.write_str("lexer source-copy failure")
        }
    }
}
impl std::error::Error for LexerSpecCopyFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        if let Some(c) = &self.cause {
            Some(c)
        } else {
            self.lexeme.as_ref().map(|c| c as _)
        }
    }
}
/// Loan of the compiled ordinary declaration; all per-lexeme plans use this same
/// immutable owner when the final constructor runs.
pub struct LexerSpecCopyPlan<'a> {
    source: &'a LexerSpec,
    requirements: LexerSourceCopyRequirements,
    warning_bytes: usize,
}
impl fmt::Debug for LexerSpecCopyPlan<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LexerSpecCopyPlan")
            .field("requirements", &self.requirements)
            .finish()
    }
}
impl LexerSpec {
    /// Quotes copying one actual declaration, without running the lexer compiler.
    pub fn source_copy_plan<F: PreparedFunding<Error = ParserAllocationFailure>>(
        &self, funding: &F,
    ) -> Result<LexerSpecCopyPlan<'_>, LexerSpecCopyFailure> {
        LexerSpecCopyPlan::prepare(self, funding)
    }
}
impl<'a> LexerSpecCopyPlan<'a> {
    fn prepare<F: PreparedFunding<Error = ParserAllocationFailure>>(
        source: &'a LexerSpec, funding: &F,
    ) -> Result<Self, LexerSpecCopyFailure> {
        let _frame = funding.frame(size_of::<(
            &LexerSpec, &F, Self, LexerSourceCopyRequirements, LexerSpecCopyFailure,
            usize, usize, usize, usize, usize, [usize; 24],
            std::slice::Iter<'_, LexemeSpec>, std::slice::Iter<'_, (String, usize)>,
            Result<Self, LexerSpecCopyFailure>, Result<(), Cause>, Option<usize>,
        )>()).map_err(|error| LexerSpecCopyFailure {
            cause: Some(frame(error)), lexeme: None, partial: None,
        })?;
        let mut buffers = 0;
        let mut controls = 0;
        let mut warning_bytes = 0;
        let mut scratch = 0usize;
        let quote = (|| -> Result<(), Cause> {
            add(&mut buffers, bytes::<LexemeSpec>(source.lexemes.len())?)?;
            add(
                &mut buffers,
                bytes::<LexemeIdx>(source.skip_by_class.len())?,
            )?;
            add(
                &mut buffers,
                bytes::<(String, usize)>(source.grammar_warnings.len())?,
            )?;
            add(&mut buffers, source.class_by_skip.allocation_size())?;
            for (text, _) in &source.grammar_warnings {
                add(&mut warning_bytes, text.len())?;
            }
            add(&mut buffers, warning_bytes)?;
            let _builder_frame = funding.frame(
                RegexBuilder::source_inspection_control_bytes().ok_or(Cause::Overflow)?
            ).map_err(frame)?;
            let builder = source
                .regex_builder
                .source_copy_plan()
                .map_err(Cause::Builder)?
                .requirements();
            add(&mut buffers, builder.buffer_bytes())?;
            add(&mut controls, builder.control_bytes())?;
            Ok(())
        })();
        if let Err(cause) = quote {
            return Err(LexerSpecCopyFailure {
                cause: Some(cause),
                lexeme: None,
                partial: None,
            });
        }
        for lexeme in &source.lexemes {
            let quote = lexeme
                .source_copy_plan(funding)
                .map_err(|lexeme| LexerSpecCopyFailure {
                    cause: None,
                    lexeme: Some(lexeme),
                    partial: None,
                })?
                .requirements();
            if add(&mut buffers, quote.buffer_bytes()).is_err()
                || add(&mut controls, quote.control_bytes()).is_err()
                || quote
                    .buffer_bytes()
                    .checked_sub(quote.retained_bytes())
                    .and_then(|n| scratch.checked_add(n))
                    .map(|n| {
                        scratch = n;
                    })
                    .is_none()
            {
                return Err(LexerSpecCopyFailure {
                    cause: Some(Cause::Overflow),
                    lexeme: None,
                    partial: None,
                });
            }
        }
        let parts = [
            size_of::<Self>(),
            size_of::<LexerSourceCopyRequirements>(),
            size_of::<LexerSpecCopyFailure>(),
            size_of::<Partial>(),
            size_of::<Cause>(),
            size_of::<LexerSpec>(),
            size_of::<Option<Partial>>(),
            size_of::<Classes>(),
            size_of_val(&source.class_by_skip.iter()),
            size_of_val(source.class_by_skip.hasher()),
            size_of::<<derivre::RandomState as std::hash::BuildHasher>::Hasher>(),
            size_of::<Result<Self, LexerSpecCopyFailure>>(),
            size_of::<Result<LexerSpec, LexerSpecCopyFailure>>(),
            size_of::<Result<(), Cause>>(),
            size_of::<Result<(), SourceMapReserveError>>(),
            size_of::<Result<(), TryReserveError>>(),
            size_of::<Option<LexemeClass>>(),
            size_of::<Result<String, std::string::FromUtf8Error>>(),
            size_of::<Result<Layout, std::alloc::LayoutError>>(),
            size_of::<std::slice::Iter<'_, LexemeSpec>>(),
            size_of::<std::slice::Iter<'_, (String, usize)>>(),
            size_of::<(&LexerSpec, &mut Partial)>(),
        ];
        controls = parts
            .into_iter()
            .try_fold(
                controls
                    .checked_add(size_of_val(&parts))
                    .ok_or_else(|| LexerSpecCopyFailure {
                        cause: Some(Cause::Overflow),
                        lexeme: None,
                        partial: None,
                    })?,
                usize::checked_add,
            )
            .ok_or_else(|| LexerSpecCopyFailure {
                cause: Some(Cause::Overflow),
                lexeme: None,
                partial: None,
            })?;
        let total = buffers
            .checked_add(controls)
            .ok_or_else(|| LexerSpecCopyFailure {
                cause: Some(Cause::Overflow),
                lexeme: None,
                partial: None,
            })?;
        let retained = buffers
            .checked_sub(scratch)
            .ok_or_else(|| LexerSpecCopyFailure {
                cause: Some(Cause::Overflow),
                lexeme: None,
                partial: None,
            })?;
        Ok(Self {
            source,
            warning_bytes,
            requirements: LexerSourceCopyRequirements {
                buffers,
                retained,
                controls,
                total,
            },
        })
    }
    /// Actual complete local constructor population.
    pub fn requirements(&self) -> LexerSourceCopyRequirements {
        self.requirements
    }
    /// Uses the same ordinary source/child constructors; no new lexemes or regex
    /// states are invented. All partial destinations are retained on refusal.
    pub fn compile(self) -> Result<LexerSpec, LexerSpecCopyFailure> {
        let s = self.source;
        let mut p = Partial {
            lexemes: Vec::new(),
            builder: None,
            skip: Vec::new(),
            classes: Classes::with_hasher(s.class_by_skip.hasher().clone()),
            warnings: Vec::new(),
            pending_warning: Vec::new(),
        };
        let mut warning_bytes = self.warning_bytes;
        macro_rules! run {
            ($e:expr) => {
                if let Err(cause) = $e {
                    return Err(LexerSpecCopyFailure {
                        cause: Some(cause),
                        lexeme: None,
                        partial: Some(p),
                    });
                }
            };
        }
        run!(reserve(&mut p.lexemes, s.lexemes.len()));
        for source in &s.lexemes {
            // The enclosing copy quote already includes this child inspection.
            match source.source_copy_plan(&ParserAllocationFunding::unenforced()).and_then(|plan| plan.compile()) {
                Ok(lexeme) => p.lexemes.push(lexeme),
                Err(lexeme) => {
                    return Err(LexerSpecCopyFailure {
                        cause: None,
                        lexeme: Some(lexeme),
                        partial: Some(p),
                    });
                }
            }
        }
        match s
            .regex_builder
            .source_copy_plan()
            .and_then(|plan| plan.compile())
        {
            Ok(builder) => p.builder = Some(builder),
            Err(cause) => {
                return Err(LexerSpecCopyFailure {
                    cause: Some(Cause::Builder(cause)),
                    lexeme: None,
                    partial: Some(p),
                });
            }
        }
        run!(reserve(&mut p.skip, s.skip_by_class.len()));
        p.skip.extend_from_slice(&s.skip_by_class);
        run!(
            p.classes
                .try_reserve(s.class_by_skip.capacity())
                .map_err(Cause::Table)
        );
        if p.classes.capacity() != s.class_by_skip.capacity()
            || p.classes.allocation_size() != s.class_by_skip.allocation_size()
        {
            run!(Err::<(), _>(Cause::Capacity));
        }
        for (&key, &value) in &s.class_by_skip {
            if p.classes.len() == p.classes.capacity() {
                run!(Err::<(), _>(Cause::Capacity));
            }
            if p.classes.insert(key, value).is_some() {
                run!(Err::<(), _>(Cause::Capacity));
            }
        }
        run!(reserve(&mut p.warnings, s.grammar_warnings.len()));
        for (source, count) in &s.grammar_warnings {
            match warning_bytes.checked_sub(source.len()) {
                Some(n) => warning_bytes = n,
                None => {
                    run!(Err::<(), _>(Cause::Capacity));
                }
            }
            run!(reserve(&mut p.pending_warning, source.len()));
            p.pending_warning.extend_from_slice(source.as_bytes());
            p.warnings
                .push((text(std::mem::take(&mut p.pending_warning)), *count));
        }
        if warning_bytes != 0 {
            run!(Err::<(), _>(Cause::Capacity));
        }
        Ok(LexerSpec {
            lexemes: p.lexemes,
            regex_builder: p.builder.take().expect("completed actual builder"),
            no_forcing: s.no_forcing,
            allow_initial_skip: s.allow_initial_skip,
            num_extra_lexemes: s.num_extra_lexemes,
            skip_by_class: p.skip,
            class_by_skip: p.classes,
            current_class: s.current_class,
            special_token_rx: s.special_token_rx,
            has_stop: s.has_stop,
            has_max_tokens: s.has_max_tokens,
            has_temperature: s.has_temperature,
            grammar_warnings: p.warnings,
            funding: derivre::ParserAllocationFunding::unenforced(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        api::{ParserLimits, SkipSpec},
        earley::lexer::Lexer,
    };
    #[test]
    fn lexer_source_inspection_refuses_each_reached_ast_frame_with_original_error_custody() {
        use derivre::prepared_funding::Scope;
        use std::sync::{Arc, atomic::{AtomicUsize, Ordering}};
        let mut source = LexerSpec::new(ParserAllocationFunding::unenforced()).unwrap();
        source.setup_lexeme_class(RegexAst::NoMatch).unwrap();
        let mut ast = RegexAst::Literal("source".into());
        // Extend beyond the builder inspection peak so AST recursion itself
        // reaches additional prospective frame growth.
        for _ in 0..256 { ast = RegexAst::Repeat(Box::new(ast), 1, 1); }
        source.add_greedy_lexeme("deep".into(), ast, true, None, 31).unwrap();
        for retained in [false, true] {
            let run = |cut: usize| {
                let calls = Arc::new(AtomicUsize::new(0));
                let count = calls.clone();
                let owner = Arc::new(());
                let weak = Arc::downgrade(&owner);
                let funding = ParserAllocationFunding::prepare(move |_| {
                    let _keep = &owner;
                    let n = count.fetch_add(1, Ordering::SeqCst) + 1;
                    if n == cut { Err(std::io::Error::from_raw_os_error(n as i32)) } else { Ok(()) }
                }).unwrap();
                let scope = Scope::new(&funding);
                let failure = match scope {
                    Err(FrameError::Funding(error)) => Some(LexerSpecCopyFailure::inspection(Cause::Funding(error))),
                    Err(FrameError::Overflow) => panic!("finite scope"),
                    Ok(scope) => {
                        if retained { source.retained_capacity_bytes(&scope).err() }
                        else { source.source_copy_plan(&scope).err() }
                    }
                };
                let n = calls.load(Ordering::SeqCst);
                drop(funding);
                if let Some(error) = failure {
                    assert!(error.partial.is_none());
                    let mut cause: &dyn std::error::Error = &error;
                    let original = loop {
                        if let Some(original) = cause.downcast_ref::<std::io::Error>() { break original; }
                        cause = cause.source().expect("original typed inspection refusal");
                    };
                    assert_eq!(original.raw_os_error(), Some(cut as i32));
                    assert_eq!(n, cut);
                    assert!(weak.upgrade().is_some());
                    drop(error);
                } else { assert_eq!(cut, usize::MAX); }
                assert!(weak.upgrade().is_none());
                n
            };
            let reached = run(usize::MAX);
            assert!(reached > 12, "retained={retained}, reached={reached}");
            for cut in 2..=reached { run(cut); }
        }
    }

    #[test]
    fn copied_lexer_source_preserves_skip_classes_ranges_regexes_and_failed_warning_prefix() {
        let mut source = LexerSpec::new(derivre::ParserAllocationFunding::unenforced()).unwrap();
        let root = source.setup_lexeme_class(RegexAst::NoMatch).unwrap();
        source
            .add_simple_literal("word".into(), "abc", false)
            .unwrap();
        source
            .add_greedy_lexeme(
                "greek".into(),
                RegexAst::Regex("[α-ω]+".into()),
                true,
                Some(JsonQuoteOptions::regular()),
                73,
            )
            .unwrap();
        source
            .add_special_token("tool".into(), vec![7..=9, 17..=17])
            .unwrap();
        let secondary = source
            .setup_lexeme_class_with_skip(SkipSpec::new(
                RegexAst::Literal(" ".into()),
                SkipRepetition::Once,
            ))
            .unwrap();
        assert_ne!(root, secondary);
        source
            .add_rx_and_stop(
                "until".into(),
                RegexAst::Regex("[a-z]+".into()),
                RegexAst::Literal("!".into()),
                true,
                11,
                true,
            )
            .unwrap();
        source.add_extra_lexemes(&["[0-9]+".into()]).unwrap();
        source.grammar_warnings = vec![
            ("first source warning".into(), 3),
            ("second source warning".into(), 1),
        ];
        let plan = source.source_copy_plan(&ParserAllocationFunding::unenforced()).unwrap();
        let quote = plan.requirements();
        let mut copied = plan.compile().unwrap();
        assert!(quote.required_bytes() > quote.buffer_bytes());
        assert_eq!(copied.class_by_skip, source.class_by_skip);
        assert_eq!(
            copied.class_by_skip.allocation_size(),
            source.class_by_skip.allocation_size()
        );
        assert_eq!(copied.skip_by_class, source.skip_by_class);
        assert_eq!(copied.grammar_warnings, source.grammar_warnings);
        assert_eq!(copied.special_token_rx, source.special_token_rx);
        assert_eq!(copied.has_stop, source.has_stop);
        assert_eq!(copied.num_extra_lexemes, source.num_extra_lexemes);
        for i in 0..source.lexemes.len() {
            let a = &source.lexemes[i];
            let b = &copied.lexemes[i];
            assert_eq!(a.compiled_rx, b.compiled_rx);
            assert_eq!(a.name, b.name);
            assert_eq!(a.token_ranges, b.token_ranges);
            assert_eq!(a.max_tokens, b.max_tokens);
            assert_eq!(a.single_set.as_slice(), b.single_set.as_slice());
            assert_eq!(format!("{:?}", a.rx), format!("{:?}", b.rx));
        }
        let mut original_lexer = Lexer::from(&source, &mut ParserLimits::default(), false).unwrap();
        let mut copied_lexer = Lexer::from(&copied, &mut ParserLimits::default(), false).unwrap();
        let mut a = original_lexer.start_state(&source.all_lexemes());
        let mut b = copied_lexer.start_state(&copied.all_lexemes());
        assert_eq!(a, b);
        for byte in b"abc" {
            a = original_lexer.dfa.transition(a, *byte);
            b = copied_lexer.dfa.transition(b, *byte);
            assert_eq!(a, b);
        }
        assert_eq!(
            original_lexer.dfa.stats().num_states,
            copied_lexer.dfa.stats().num_states
        );
        assert!(source.has_max_tokens || source.has_temperature);
        // Ordinary policy deliberately allocates a new class once either flag
        // is present. Compare the same reached source policy after copying.
        let source_next = source.setup_lexeme_class(RegexAst::NoMatch).unwrap();
        let copied_next = copied.setup_lexeme_class(RegexAst::NoMatch).unwrap();
        assert_eq!(copied_next, source_next);
        assert_ne!(source_next, root);

        let mut failing = source.source_copy_plan(&ParserAllocationFunding::unenforced()).unwrap();
        failing.warning_bytes = source.grammar_warnings[0].0.len();
        let failure = match failing.compile() {
            Err(e) => e,
            Ok(_) => panic!("incomplete source-warning population accepted"),
        };
        assert!(matches!(failure.cause, Some(Cause::Capacity)));
        drop(source);
        drop(copied);
        drop(original_lexer);
        drop(copied_lexer);
        let prefix = failure.partial.as_ref().unwrap();
        assert!(prefix.builder.is_some());
        assert!(prefix.classes.allocation_size() > 0);
        assert!(prefix.lexemes.iter().any(|l| l.name == "greek"));
        assert_eq!(prefix.warnings.len(), 1);
        assert_eq!(prefix.warnings[0].0, "first source warning");
        assert_eq!(prefix.warnings[0].1, 3);
    }
}
