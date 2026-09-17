//! Parser startup's ordinary large-lexeme order and threshold.
use super::super::lexerspec::LexemeIdx;
use crate::api::ParserLimits;
pub(super) trait Context {
    type Error;
    fn len(&self) -> usize;
    fn lexeme(&self, index: usize) -> LexemeIdx;
    fn set_fuel(&mut self, fuel: u64);
    fn weight(&mut self, lexeme: LexemeIdx) -> Result<u32, Self::Error>;
    fn precompute(&mut self, lexeme: LexemeIdx) -> Result<(), Self::Error>;
}
pub(super) fn run<C: Context>(context: &mut C, limits: &ParserLimits) -> Result<(), C::Error> {
    if !limits.precompute_large_lexemes {
        return Ok(());
    }
    context.set_fuel(limits.initial_lexer_fuel);
    for index in 0..context.len() {
        let lexeme = context.lexeme(index);
        if context.weight(lexeme)? > 1000 {
            context.precompute(lexeme)?;
        }
    }
    Ok(())
}
