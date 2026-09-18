//! One concatenation flatten/fold worker with explicit storage destinations.
use super::{
    scalar::{self, Emission},
    ConcatElement, OwnedConcatElement,
};
use crate::ast::{Expr, ExprFlags, ExprRef, ExprSet};
use crate::{ParserAllocationFunding, raw::PreparedExprError};

pub(crate) mod storage;

trait Destination {
    type Error;
    fn last_is_bytes(&self) -> bool;
    fn extend_bytes(&mut self, bytes: &[u8]) -> Result<(), Self::Error>;
    fn push_bytes(&mut self, bytes: &[u8]) -> Result<(), Self::Error>;
    fn push_expr(&mut self, expr: ExprRef) -> Result<(), Self::Error>;
}
fn append<D: Destination>(
    destination: &mut D,
    value: &ConcatElement<'_>,
) -> Result<bool, D::Error> {
    match value {
        ConcatElement::Bytes(bytes) => {
            if destination.last_is_bytes() {
                destination.extend_bytes(bytes)?;
            } else {
                destination.push_bytes(bytes)?;
            }
        }
        ConcatElement::Expr(expr) => {
            if *expr == ExprRef::NO_MATCH {
                return Ok(false);
            }
            if *expr != ExprRef::EMPTY_STRING {
                destination.push_expr(*expr)?;
            }
        }
    }
    Ok(true)
}
struct OwnedDestination<'a>(&'a mut Vec<OwnedConcatElement>, &'a ParserAllocationFunding);
impl Destination for OwnedDestination<'_> {
    type Error = PreparedExprError;
    fn last_is_bytes(&self) -> bool {
        matches!(self.0.last(), Some(OwnedConcatElement::Bytes(_)))
    }
    fn extend_bytes(&mut self, bytes: &[u8]) -> Result<(), PreparedExprError> {
        let Some(OwnedConcatElement::Bytes(current)) = self.0.last_mut() else {
            unreachable!("checked last byte group")
        };
        self.1.try_extend_copy(current, bytes)?;
        Ok(())
    }
    fn push_bytes(&mut self, bytes: &[u8]) -> Result<(), PreparedExprError> {
        let mut copy = Vec::new();
        self.1.try_extend_copy(&mut copy, bytes)?;
        self.1.try_push(self.0, OwnedConcatElement::Bytes(copy))?;
        Ok(())
    }
    fn push_expr(&mut self, expr: ExprRef) -> Result<(), PreparedExprError> {
        self.1.try_push(self.0, OwnedConcatElement::Expr(expr))?;
        Ok(())
    }
}
pub(super) fn push_owned(
    destination: &mut Vec<OwnedConcatElement>,
    value: &ConcatElement<'_>,
    funding: &ParserAllocationFunding,
) -> Result<bool, PreparedExprError> {
    append(&mut OwnedDestination(destination, funding), value)
}

trait Memory {
    type Error;
    fn begin(&mut self) -> Result<usize, Self::Error>;
    fn append(&mut self, frame: usize, element: &ConcatElement<'_>) -> Result<bool, Self::Error>;
    fn push_tail(&mut self, frame: usize, tail: ExprRef) -> Result<(), Self::Error>;
    fn len(&self, frame: usize) -> usize;
    fn part(&self, frame: usize, index: usize) -> ConcatElement<'_>;
    fn finish(&mut self, frame: usize);
}
struct GrowingMemory {
    funding: ParserAllocationFunding,
    frames: Vec<Vec<OwnedConcatElement>>,
}
impl Memory for GrowingMemory {
    type Error = PreparedExprError;
    fn begin(&mut self) -> Result<usize, PreparedExprError> {
        let index = self.frames.len();
        self.funding.try_push(&mut self.frames, Vec::new())?;
        Ok(index)
    }
    fn append(&mut self, frame: usize, value: &ConcatElement<'_>) -> Result<bool, PreparedExprError> {
        append(&mut OwnedDestination(&mut self.frames[frame], &self.funding), value)
    }
    fn push_tail(&mut self, frame: usize, tail: ExprRef) -> Result<(), PreparedExprError> {
        self.funding.try_push(&mut self.frames[frame], OwnedConcatElement::Expr(tail))?;
        Ok(())
    }
    fn len(&self, frame: usize) -> usize {
        self.frames[frame].len()
    }
    fn part(&self, frame: usize, index: usize) -> ConcatElement<'_> {
        match &self.frames[frame][index] {
            OwnedConcatElement::Expr(expr) => ConcatElement::Expr(*expr),
            OwnedConcatElement::Bytes(bytes) => ConcatElement::Bytes(bytes),
        }
    }
    fn finish(&mut self, frame: usize) {
        assert_eq!(frame + 1, self.frames.len());
        self.frames.pop();
    }
}
pub(super) fn growing(source: &mut ExprSet, left: ExprRef, right: ExprRef) -> Result<ExprRef, PreparedExprError> {
    let mut memory = GrowingMemory { funding: source.construction_funding()?.clone(), frames: Vec::new() };
    concat(&mut scalar::Prepared(source), &mut memory, left, right)
}
pub(super) fn growing_fold(source: &mut ExprSet, parts: Vec<OwnedConcatElement>) -> Result<ExprRef, PreparedExprError> {
    let mut memory = GrowingMemory { funding: source.construction_funding()?.clone(), frames: Vec::new() };
    memory.funding.try_push(&mut memory.frames, parts)?;
    fold(&mut scalar::Prepared(source), &mut memory, 0)
}

fn fold<S: Emission, M: Memory<Error = S::Error>>(
    sink: &mut S,
    memory: &mut M,
    frame: usize,
) -> Result<ExprRef, S::Error> {
    let count = memory.len(frame);
    if count == 0 {
        memory.finish(frame);
        return Ok(ExprRef::EMPTY_STRING);
    }
    let mut result = match memory.part(frame, count - 1) {
        ConcatElement::Expr(expr) => expr,
        ConcatElement::Bytes(bytes) => scalar::byte_concat(sink, bytes, ExprRef::EMPTY_STRING)?,
    };
    for index in (0..count - 1).rev() {
        result = match memory.part(frame, index) {
            ConcatElement::Expr(expr) => concat(sink, memory, expr, result)?,
            ConcatElement::Bytes(bytes) => scalar::byte_concat(sink, bytes, result)?,
        };
    }
    memory.finish(frame);
    Ok(result)
}
fn concat<S: Emission, M: Memory<Error = S::Error>>(
    sink: &mut S,
    memory: &mut M,
    left: ExprRef,
    right: ExprRef,
) -> Result<ExprRef, S::Error> {
    sink.pay(2)?;
    if left == ExprRef::EMPTY_STRING {
        return Ok(right);
    }
    if right == ExprRef::EMPTY_STRING {
        return Ok(left);
    }
    if left == ExprRef::NO_MATCH || right == ExprRef::NO_MATCH {
        return Ok(ExprRef::NO_MATCH);
    }
    if sink.source().is_concat(left) {
        let frame = memory.begin()?;
        for element in sink.source().iter_concat(left) {
            if !memory.append(frame, &element)? {
                memory.finish(frame);
                return Ok(ExprRef::NO_MATCH);
            }
        }
        memory.push_tail(frame, right)?;
        return fold(sink, memory, frame);
    }
    let left_flags = sink.source().get_flags(left);
    let right_flags = sink.source().get_flags(right);
    let flags = ExprFlags::from_nullable_positive(
        left_flags.is_nullable() && right_flags.is_nullable(),
        left_flags.is_positive() && right_flags.is_positive(),
    );
    sink.emit(Expr::Concat(flags, [left, right]))
}

// The filtered relevance list contains only expression references. Seed the
// same frame representation and use the ordinary concatenation fold unchanged.
fn fold_expressions<S: Emission, M: Memory<Error = S::Error>>(
    sink: &mut S,
    memory: &mut M,
    args: &[ExprRef],
) -> Result<ExprRef, S::Error> {
    let frame = memory.begin()?;
    for &arg in args {
        memory.push_tail(frame, arg)?;
    }
    fold(sink, memory, frame)
}
pub(crate) fn growing_fold_expressions(source: &mut ExprSet, args: &[ExprRef]) -> Result<ExprRef, PreparedExprError> {
    let mut memory = GrowingMemory { funding: source.construction_funding()?.clone(), frames: Vec::new() };
    fold_expressions(&mut scalar::Prepared(source), &mut memory, args)
}
