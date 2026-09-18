//! Same derivative, relevance, candidate and fuel order for actual DFA edges.
use super::{descriptor, DerivCache, ExprRef, ExprSet, LexemeIdx, RelevanceCache};
pub(super) trait Context {
    type Error;
    fn cost(&self) -> u64;
    fn derivative(&mut self, root: ExprRef, byte: u8) -> Result<ExprRef, Self::Error>;
    fn non_empty(&mut self, root: ExprRef, fuel: u64) -> Result<bool, Self::Error>;
    fn is_fuel(error: &Self::Error) -> bool;
    fn emit(&mut self, index: LexemeIdx, root: ExprRef) -> Result<(), Self::Error>;
}
pub(super) fn build<C: Context>(
    context: &mut C,
    words: &[u32],
    byte: u8,
    fuel: &mut u64,
) -> Result<(), C::Error> {
    let cost = context.cost();
    let result = (|| {
        for (index, root) in descriptor::pairs(words) {
            let derived = context.derivative(root, byte)?;
            let available = fuel.saturating_sub(context.cost() - cost);
            let live = match context.non_empty(derived, available) {
                Ok(value) => value,
                Err(error) => {
                    if C::is_fuel(&error) {
                        *fuel = 0;
                    }
                    return Err(error);
                }
            };
            if live && derived != ExprRef::NO_MATCH {
                context.emit(index, derived)?;
            }
        }
        Ok(())
    })();
    *fuel = fuel.saturating_sub(context.cost() - cost);
    result
}
pub(super) struct Ordinary<'a> {
    pub source: &'a mut ExprSet,
    pub derivative: &'a mut DerivCache,
    pub relevance: &'a mut RelevanceCache,
    pub candidate: &'a mut Vec<u32>,
}
impl Context for Ordinary<'_> {
    type Error = derivre::ParserError;
    fn cost(&self) -> u64 {
        self.source.cost()
    }
    fn derivative(&mut self, root: ExprRef, byte: u8) -> derivre::ParserResult<ExprRef> {
        self.derivative.derivative(self.source, root, byte)
    }
    fn non_empty(&mut self, root: ExprRef, fuel: u64) -> derivre::ParserResult<bool> {
        self.relevance.is_non_empty_limited(self.source, root, fuel)
    }
    fn is_fuel(_: &Self::Error) -> bool {
        true
    }
    fn emit(&mut self, index: LexemeIdx, root: ExprRef) -> derivre::ParserResult<()> {
        self.candidate.push(index.as_usize() as u32);
        self.candidate.push(root.as_u32());
        Ok(())
    }
}
