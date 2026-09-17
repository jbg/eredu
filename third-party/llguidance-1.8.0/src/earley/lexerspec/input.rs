//! Exact lexer scalar rows joined to the shared alphabet constructor.
use super::{LexemeSpec, LexerSpec};
use crate::earley::regexvec::RxLexeme;
use derivre::{AlphabetInfo, ExprRef, raw::ExprSet};
use std::{
    alloc::Layout,
    collections::TryReserveError,
    fmt,
    mem::{size_of, size_of_val},
};

/// Fixed root/scalar destinations from an actual lexer declaration.
#[derive(Clone, Copy, Debug)]
pub struct LexerRootRequirements {
    buffers: usize,
    controls: usize,
    total: usize,
}
impl LexerRootRequirements {
    /// Actual scalar rows and copied root references.
    pub fn buffer_bytes(&self) -> usize {
        self.buffers
    }
    /// Complete local fixed preparation, copy and completion frames.
    pub fn control_bytes(&self) -> usize {
        self.controls
    }
    /// Complete constructor reservation, without alphabet/DFA construction.
    pub fn required_bytes(&self) -> usize {
        self.total
    }
}
/// Loan of one actual lexer declaration; no invented lexical policy or roots.
pub struct LexerRootPlan<'a> {
    source: &'a LexerSpec,
    requirements: LexerRootRequirements,
}
impl fmt::Debug for LexerRootPlan<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LexerRootPlan")
            .field("requirements", &self.requirements)
            .finish()
    }
}
/// Complete root/priority/lazy rows, with the exact special-token occurrence.
#[derive(Debug)]
pub struct LexerRootSource {
    rows: Vec<RxLexeme>,
    roots: Vec<ExprRef>,
    special_position: Option<usize>,
}
/// Immutable input for the shared regex-vector constructor. This is storage,
/// not permission for mutable derivative, relevance-cache or DFA execution.
pub struct RegexVectorInput {
    pub(crate) alpha: AlphabetInfo,
    pub(crate) expressions: ExprSet,
    pub(crate) rows: Vec<RxLexeme>,
    pub(crate) roots: Vec<ExprRef>,
    pub(crate) special_token_rx: Option<ExprRef>,
}
impl fmt::Debug for RegexVectorInput {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RegexVectorInput")
            .field("roots", &self.roots.len())
            .field("alphabet", &self.alpha.len())
            .finish()
    }
}
impl RegexVectorInput {
    /// The exact constructed alphabet.
    pub fn alphabet(&self) -> &AlphabetInfo {
        &self.alpha
    }
    /// Independently owned expression declaration. Subsequent growth is separate.
    pub fn expressions(&self) -> &ExprSet {
        &self.expressions
    }
    /// Root order consumed by the shared regex-vector worker.
    pub fn roots(&self) -> &[ExprRef] {
        &self.roots
    }
}
#[derive(Debug)]
enum Cause {
    Overflow,
    Capacity,
    Source,
    Vector(TryReserveError),
}
/// Keeps actual root/scalar prefixes and any completed alphabet destination.
#[derive(Debug)]
pub struct LexerInputFailure {
    cause: Cause,
    source: Option<LexerRootSource>,
    input: Option<RegexVectorInput>,
}
impl fmt::Display for LexerInputFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.cause {
            Cause::Overflow => f.write_str("lexer root geometry overflow"),
            Cause::Capacity => {
                f.write_str("lexer root destination differs from its exact capacity")
            }
            Cause::Source => f.write_str("lexer alphabet output differs from its root declaration"),
            Cause::Vector(e) => fmt::Display::fmt(e, f),
        }
    }
}
impl std::error::Error for LexerInputFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.cause {
            Cause::Vector(e) => Some(e),
            _ => None,
        }
    }
}
fn failure(cause: Cause) -> LexerInputFailure {
    LexerInputFailure {
        cause,
        source: None,
        input: None,
    }
}
fn row(source: &LexemeSpec) -> RxLexeme {
    RxLexeme {
        rx: source.compiled_rx,
        priority: 0,
        lazy: source.lazy,
    }
}
fn reserve<T>(v: &mut Vec<T>, n: usize) -> Result<(), Cause> {
    v.try_reserve_exact(n).map_err(Cause::Vector)?;
    if v.capacity() != n {
        return Err(Cause::Capacity);
    }
    Ok(())
}
impl LexerSpec {
    /// Fixed inspection frames; source rows are borrowed and no parser is run.
    pub fn root_source_inspection_control_bytes() -> Option<usize> {
        let parts = [
            size_of::<LexerRootPlan<'_>>(),
            size_of::<LexerRootSource>(),
            size_of::<LexerRootRequirements>(),
            size_of::<LexerInputFailure>(),
            size_of::<RegexVectorInput>(),
            size_of::<Cause>(),
            size_of::<RxLexeme>(),
            size_of::<Option<usize>>(),
            size_of::<Option<ExprRef>>(),
            size_of::<Result<LexerRootPlan<'_>, LexerInputFailure>>(),
            size_of::<Result<LexerRootSource, LexerInputFailure>>(),
            size_of::<Result<RegexVectorInput, LexerInputFailure>>(),
            size_of::<Result<(), Cause>>(),
            size_of::<Result<(), TryReserveError>>(),
            size_of::<Result<Layout, std::alloc::LayoutError>>(),
            size_of::<std::slice::Iter<'_, LexemeSpec>>(),
            size_of::<std::slice::Iter<'_, RxLexeme>>(),
            size_of::<(AlphabetInfo, ExprSet, Vec<ExprRef>)>(),
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
    /// Exact scalar/root copy quote. Caller reserves it before compilation.
    pub fn root_source_plan(&self) -> Result<LexerRootPlan<'_>, LexerInputFailure> {
        let result = (|| -> Result<_, Cause> {
            let buffers = Layout::array::<RxLexeme>(self.lexemes.len())
                .map_err(|_| Cause::Overflow)?
                .size()
                .checked_add(
                    Layout::array::<ExprRef>(self.lexemes.len())
                        .map_err(|_| Cause::Overflow)?
                        .size(),
                )
                .ok_or(Cause::Overflow)?;
            let controls = Self::root_source_inspection_control_bytes().ok_or(Cause::Overflow)?;
            let total = buffers.checked_add(controls).ok_or(Cause::Overflow)?;
            Ok(LexerRootPlan {
                source: self,
                requirements: LexerRootRequirements {
                    buffers,
                    controls,
                    total,
                },
            })
        })();
        result.map_err(failure)
    }
}
impl LexerRootPlan<'_> {
    /// Root/scalar destination and fixed frame population.
    pub fn requirements(&self) -> LexerRootRequirements {
        self.requirements
    }
    /// Uses the same scalar worker as ordinary lexer construction.
    pub fn compile(self) -> Result<LexerRootSource, LexerInputFailure> {
        let mut source = LexerRootSource {
            rows: Vec::new(),
            roots: Vec::new(),
            special_position: None,
        };
        let result = (|| -> Result<(), Cause> {
            reserve(&mut source.rows, self.source.lexemes.len())?;
            for lexeme in &self.source.lexemes {
                source.rows.push(row(lexeme));
            }
            source.fill_roots(self.source.special_token_rx)
        })();
        match result {
            Ok(()) => Ok(source),
            Err(cause) => Err(LexerInputFailure {
                cause,
                source: Some(source),
                input: None,
            }),
        }
    }
}
impl LexerRootSource {
    pub(crate) fn from_rows(
        rows: Vec<RxLexeme>,
        special: Option<ExprRef>,
    ) -> Result<Self, LexerInputFailure> {
        let mut source = Self {
            rows,
            roots: Vec::new(),
            special_position: None,
        };
        match source.fill_roots(special) {
            Ok(()) => Ok(source),
            Err(cause) => Err(LexerInputFailure {
                cause,
                source: Some(source),
                input: None,
            }),
        }
    }
    fn fill_roots(&mut self, special: Option<ExprRef>) -> Result<(), Cause> {
        reserve(&mut self.roots, self.rows.len())?;
        self.roots.extend(self.rows.iter().map(|r| r.rx));
        self.special_position = special.and_then(|rx| self.roots.iter().position(|r| *r == rx));
        Ok(())
    }
    /// Exact roots borrowed by the independent alphabet-copy plan.
    pub fn roots(&self) -> &[ExprRef] {
        &self.roots
    }
    /// Closes the same root update/special occurrence handoff as ordinary code.
    /// The returned input still supplies no mutable lexer execution authority.
    pub fn finish(
        self,
        alpha: AlphabetInfo,
        expressions: ExprSet,
        roots: Vec<ExprRef>,
    ) -> Result<RegexVectorInput, LexerInputFailure> {
        let mut input = RegexVectorInput {
            alpha,
            expressions,
            rows: Vec::new(),
            roots,
            special_token_rx: None,
        };
        if input.roots != self.roots || input.roots.iter().any(|r| !input.expressions.is_valid(*r))
        {
            return Err(LexerInputFailure {
                cause: Cause::Source,
                source: Some(self),
                input: Some(input),
            });
        }
        input.special_token_rx = self.special_position.map(|i| input.roots[i]);
        input.rows = self.rows;
        for (row, root) in input.rows.iter_mut().zip(&input.roots) {
            row.rx = *root;
        }
        Ok(input)
    }
    pub(crate) fn ordinary(
        self,
        expressions: ExprSet,
    ) -> Result<RegexVectorInput, LexerInputFailure> {
        let (alpha, expressions, roots) = AlphabetInfo::from_exprset(expressions, &self.roots);
        self.finish(alpha, expressions, roots)
    }
}
