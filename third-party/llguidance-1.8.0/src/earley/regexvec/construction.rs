//! One root/fuel/classification worker for both actual lexer constructors.
use super::{ExprRef, ExprSet, LexemeIdx, LexemeSet, RelevanceCache, RxLexeme};
pub(super) trait Expressions {
    type Error;
    fn source(&self) -> &ExprSet;
    fn non_empty(&mut self, root: ExprRef, fuel: u64) -> Result<bool, Self::Error>;
    fn has_repeat(&mut self, root: ExprRef) -> Result<bool, Self::Error>;
}
pub(super) fn roots<C: Expressions>(
    expressions: &mut C,
    roots: &mut [ExprRef],
    fuel: &mut u64,
) -> Result<(), C::Error> {
    for root in roots {
        let cost = expressions.source().cost();
        if !expressions.non_empty(*root, *fuel)? {
            *root = ExprRef::NO_MATCH;
        }
        *fuel = fuel.saturating_sub(expressions.source().cost() - cost);
    }
    Ok(())
}
pub(super) fn classify<C: Expressions>(
    expressions: &mut C,
    rows: &[RxLexeme],
    lazy: &mut LexemeSet,
    subsumable: &mut LexemeSet,
) -> Result<(), C::Error> {
    for (index, row) in rows.iter().enumerate() {
        if row.lazy {
            lazy.add(LexemeIdx::new(index));
        } else if expressions.has_repeat(row.rx)? {
            subsumable.add(LexemeIdx::new(index));
        }
    }
    Ok(())
}
pub(super) struct Ordinary<'a> {
    pub source: &'a mut ExprSet,
    pub relevance: &'a mut RelevanceCache,
}
impl Expressions for Ordinary<'_> {
    type Error = derivre::ParserError;
    fn source(&self) -> &ExprSet {
        self.source
    }
    fn non_empty(&mut self, root: ExprRef, fuel: u64) -> derivre::ParserResult<bool> {
        self.relevance.is_non_empty_limited(self.source, root, fuel)
    }
    fn has_repeat(&mut self, root: ExprRef) -> derivre::ParserResult<bool> {
        Ok(self.source.attr_has_repeat(root)?)
    }
}

/// Same selected-root order for ordinary and funded initial state candidates.
pub(super) fn selected<E>(
    roots: &[ExprRef],
    selected: &LexemeSet,
    mut emit: impl FnMut(LexemeIdx, ExprRef) -> Result<(), E>,
) -> Result<(), E> {
    for index in selected.iter() {
        let root = roots[index.as_usize()];
        if root != ExprRef::NO_MATCH {
            emit(index, root)?;
        }
    }
    Ok(())
}
