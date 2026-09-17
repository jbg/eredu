//! Shared subsumption selection, cache-failure choice and actual-cost budget.
use super::{descriptor, DerivCache, ExprRef, ExprSet, LexemeSet, RelevanceCache};
pub(super) trait Context {
    type Error;
    fn cost(&self) -> u64;
    fn contains(
        &mut self,
        small: ExprRef,
        big: ExprRef,
        fuel: u64,
        cache_failures: bool,
    ) -> Result<bool, Self::Error>;
    fn recover_refusal(error: Self::Error) -> Result<bool, Self::Error>;
}
pub(super) fn run<C: Context>(
    context: &mut C,
    words: &[u32],
    subsumable: &LexemeSet,
    small: ExprRef,
    mut budget: u64,
) -> Result<bool, C::Error> {
    let original = budget;
    for (index, root) in descriptor::pairs(words) {
        if !subsumable.contains(index) {
            continue;
        }
        let cost = context.cost();
        let result = context.contains(small, root, budget, budget > original / 2);
        let contains = match result {
            Ok(value) => value,
            Err(error) => C::recover_refusal(error)?,
        };
        if contains {
            return Ok(true);
        }
        budget = budget.saturating_sub(context.cost() - cost);
    }
    Ok(false)
}
pub(super) struct Ordinary<'a> {
    pub source: &'a mut ExprSet,
    pub derivative: &'a mut DerivCache,
    pub relevance: &'a mut RelevanceCache,
}
impl Context for Ordinary<'_> {
    type Error = anyhow::Error;
    fn cost(&self) -> u64 {
        self.source.cost()
    }
    fn contains(
        &mut self,
        small: ExprRef,
        big: ExprRef,
        fuel: u64,
        cache_failures: bool,
    ) -> anyhow::Result<bool> {
        self.relevance.is_contained_in_prefixes(
            self.source,
            self.derivative,
            small,
            big,
            fuel,
            cache_failures,
        )
    }
    fn recover_refusal(_: Self::Error) -> anyhow::Result<bool> {
        Ok(false)
    }
}
