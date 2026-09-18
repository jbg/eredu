//! The ordinary disjoint-selector equations with explicit destinations.
pub(crate) mod storage;
use super::SymRes;
use crate::ast::{ExprRef, ExprSet};
use crate::ParserError as ConstructionError;

pub(crate) trait Construction {
    type Error;
    fn union(&mut self, left: ExprRef, right: ExprRef) -> Result<ExprRef, Self::Error>;
    fn intersection(&mut self, left: ExprRef, right: ExprRef) -> Result<ExprRef, Self::Error>;
    fn subtract(&mut self, left: ExprRef, right: ExprRef) -> Result<ExprRef, Self::Error>;
    fn push(&mut self, output: &mut SymRes, pair: (ExprRef, ExprRef)) -> Result<(), Self::Error>;
    fn invalid(&self) -> Self::Error;
}
pub(crate) fn run<C: Construction>(
    input: &SymRes,
    output: &mut SymRes,
    c: &mut C,
) -> Result<(), C::Error> {
    'input: for &(mut selector, value) in input {
        let count = output.len();
        for index in 0..count {
            let (previous, old_value) = output[index];
            if selector == previous {
                output[index] = (selector, c.union(value, old_value)?);
                continue 'input;
            }
            let intersection = c.intersection(selector, previous)?;
            if intersection == ExprRef::NO_MATCH {
                continue;
            }
            output[index] = (intersection, c.union(value, old_value)?);
            let old_remaining = c.subtract(previous, selector)?;
            if old_remaining != ExprRef::NO_MATCH {
                c.push(output, (old_remaining, old_value))?;
            }
            selector = c.subtract(selector, intersection)?;
            if selector == ExprRef::NO_MATCH {
                continue 'input;
            }
        }
        if selector == ExprRef::NO_MATCH {
            return Err(c.invalid());
        }
        c.push(output, (selector, value))?;
    }
    Ok(())
}
struct Ordinary<'a>(&'a mut ExprSet);
impl Construction for Ordinary<'_> {
    type Error = ConstructionError;
    fn union(&mut self, left: ExprRef, right: ExprRef) -> Result<ExprRef, ConstructionError> {
        Ok(self.0.mk_or_pair(left, right)?)
    }
    fn intersection(&mut self, left: ExprRef, right: ExprRef) -> Result<ExprRef, ConstructionError> {
        Ok(self.0.mk_byte_set_and(left, right)?)
    }
    fn subtract(&mut self, left: ExprRef, right: ExprRef) -> Result<ExprRef, ConstructionError> {
        Ok(self.0.mk_byte_set_sub(left, right)?)
    }
    fn push(&mut self, output: &mut SymRes, pair: (ExprRef, ExprRef)) -> Result<(), ConstructionError> {
        self.0.construction_funding()?.try_push(output, pair)?;
        Ok(())
    }
    fn invalid(&self) -> ConstructionError {
        crate::raw::PreparedExprError::Source.into()
    }
}
pub(super) fn ordinary(source: &mut ExprSet, input: &SymRes) -> crate::ParserResult<SymRes> {
    let mut output = Vec::new();
    run(input, &mut output, &mut Ordinary(source))?;
    Ok(output)
}
