//! Scalar constructors share equations; their emitter owns actual storage.
use crate::ast::{Expr, ExprFlags, ExprRef, ExprSet, PreparedExprError};
use std::convert::Infallible;

pub(super) trait Emission {
    type Error;
    fn source(&self) -> &ExprSet;
    fn pay(&mut self, cost: usize) -> Result<(), Self::Error>;
    fn emit(&mut self, expression: Expr<'_>) -> Result<ExprRef, Self::Error>;
    fn invalid_repeat(&mut self) -> Self::Error;
    fn power10(&mut self, scale: u32) -> Result<u32, Self::Error>;
    fn optimized_or(
        &mut self,
        flags: ExprFlags,
        args: &mut [ExprRef],
    ) -> Result<ExprRef, Self::Error>;
}
pub(super) struct Ordinary<'a>(pub(super) &'a mut ExprSet);
impl Emission for Ordinary<'_> {
    type Error = Infallible;
    fn source(&self) -> &ExprSet {
        self.0
    }
    fn pay(&mut self, cost: usize) -> Result<(), Infallible> {
        self.0.pay(cost);
        Ok(())
    }
    fn emit(&mut self, expression: Expr<'_>) -> Result<ExprRef, Infallible> {
        Ok(self.0.mk(expression))
    }
    fn optimized_or(
        &mut self,
        flags: ExprFlags,
        args: &mut [ExprRef],
    ) -> Result<ExprRef, Infallible> {
        Ok(self.0.or_optimized(flags, args))
    }
    fn power10(&mut self, scale: u32) -> Result<u32, Infallible> {
        Ok(10u32.pow(scale))
    }
    fn invalid_repeat(&mut self) -> Infallible {
        panic!("invalid repetition interval")
    }
}
pub(super) fn ordinary(
    source: &mut ExprSet,
    operation: impl FnOnce(&mut Ordinary<'_>) -> Result<ExprRef, Infallible>,
) -> ExprRef {
    match operation(&mut Ordinary(source)) {
        Ok(value) => value,
        Err(never) => match never {},
    }
}
pub(super) struct Prepared<'a>(pub(super) &'a mut ExprSet);
impl Emission for Prepared<'_> {
    type Error = PreparedExprError;
    fn source(&self) -> &ExprSet {
        self.0
    }
    fn pay(&mut self, cost: usize) -> Result<(), PreparedExprError> {
        self.0.pay_prepared(cost)
    }
    fn emit(&mut self, expression: Expr<'_>) -> Result<ExprRef, PreparedExprError> {
        self.0.emit_prepared(expression)
    }
    fn optimized_or(
        &mut self,
        _: ExprFlags,
        _: &mut [ExprRef],
    ) -> Result<ExprRef, PreparedExprError> {
        Err(PreparedExprError::Optimized)
    }
    fn power10(&mut self, scale: u32) -> Result<u32, PreparedExprError> {
        10u32.checked_pow(scale).ok_or(PreparedExprError::Source)
    }
    fn invalid_repeat(&mut self) -> PreparedExprError {
        PreparedExprError::Source
    }
}

pub(super) fn byte<S: Emission>(sink: &mut S, byte: u8) -> Result<ExprRef, S::Error> {
    sink.pay(1)?;
    sink.emit(Expr::Byte(byte))
}

pub(super) fn byte_set<S: Emission>(sink: &mut S, bits: &[u32]) -> Result<ExprRef, S::Error> {
    assert_eq!(bits.len(), sink.source().alphabet_words);
    sink.pay(sink.source().alphabet_words)?;
    let count: u32 = bits.iter().map(|value| value.count_ones()).sum();
    if count == 0 {
        return Ok(ExprRef::NO_MATCH);
    }
    if count == 1 {
        for i in 0..sink.source().alphabet_size {
            if crate::ast::byteset_contains(bits, i) {
                return byte(sink, i as u8);
            }
        }
        unreachable!("single byte-set member must belong to the source alphabet");
    }
    sink.emit(Expr::ByteSet(bits))
}
pub(super) fn repeat<S: Emission>(
    sink: &mut S,
    arg: ExprRef,
    min: u32,
    max: u32,
) -> Result<ExprRef, S::Error> {
    sink.pay(2)?;
    if arg == ExprRef::NO_MATCH {
        Ok(if min == 0 {
            ExprRef::EMPTY_STRING
        } else {
            ExprRef::NO_MATCH
        })
    } else if arg == ExprRef::EMPTY_STRING {
        Ok(ExprRef::EMPTY_STRING)
    } else if min > max {
        Err(sink.invalid_repeat())
    } else if max == 0 {
        Ok(ExprRef::EMPTY_STRING)
    } else if min == 1 && max == 1 {
        Ok(arg)
    } else {
        let flags = sink.source().get_flags(arg);
        let min = if flags.is_nullable() { 0 } else { min };
        let flags = ExprFlags::from_nullable_positive(min == 0, flags.is_positive());
        sink.emit(Expr::Repeat(flags, arg, min, max))
    }
}
pub(super) fn byte_concat<S: Emission>(
    sink: &mut S,
    mut bytes: &[u8],
    mut tail: ExprRef,
) -> Result<ExprRef, S::Error> {
    if bytes.is_empty() {
        return Ok(tail);
    }
    if bytes.len() == 1 && tail == ExprRef::EMPTY_STRING {
        return byte(sink, bytes[0]);
    }
    sink.pay(2 + bytes.len() / ExprRef::MAX_BYTE_CONCAT)?;
    let flags = ExprFlags::from_nullable_positive(false, sink.source().is_positive(tail));
    loop {
        if bytes.len() <= ExprRef::MAX_BYTE_CONCAT {
            return sink.emit(Expr::ByteConcat(flags, bytes, tail));
        }
        let split = bytes.len() - ExprRef::MAX_BYTE_CONCAT;
        tail = sink.emit(Expr::ByteConcat(flags, &bytes[split..], tail))?;
        bytes = &bytes[..split];
    }
}
pub(super) fn not<S: Emission>(sink: &mut S, arg: ExprRef) -> Result<ExprRef, S::Error> {
    sink.pay(2)?;
    if arg == ExprRef::EMPTY_STRING {
        Ok(ExprRef::NON_EMPTY_BYTE_STRING)
    } else if arg == ExprRef::NON_EMPTY_BYTE_STRING {
        Ok(ExprRef::EMPTY_STRING)
    } else if arg == ExprRef::ANY_BYTE_STRING {
        Ok(ExprRef::NO_MATCH)
    } else if arg == ExprRef::NO_MATCH {
        Ok(ExprRef::ANY_BYTE_STRING)
    } else {
        let expression = sink.source().get(arg);
        if let Expr::Not(_, inner) = expression {
            return Ok(inner);
        }
        let nullable_positive = !expression.nullable();
        let flags = ExprFlags::from_nullable_positive(nullable_positive, nullable_positive);
        sink.emit(Expr::Not(flags, arg))
    }
}
pub(super) fn lookahead<S: Emission>(
    sink: &mut S,
    mut arg: ExprRef,
    offset: u32,
) -> Result<ExprRef, S::Error> {
    sink.pay(2)?;
    if arg == ExprRef::NO_MATCH {
        return Ok(ExprRef::NO_MATCH);
    }
    let flags = sink.source().get_flags(arg);
    if flags.is_nullable() {
        arg = ExprRef::EMPTY_STRING;
    }
    sink.emit(Expr::Lookahead(flags, arg, offset))
}

impl ExprSet {
    fn prepared_child(&self, arg: ExprRef) -> Result<(), PreparedExprError> {
        self.require_prepared()?;
        if self.is_valid(arg) {
            Ok(())
        } else {
            Err(PreparedExprError::Source)
        }
    }
    pub(crate) fn try_mk_byte(&mut self, byte_value: u8) -> Result<ExprRef, PreparedExprError> {
        self.require_prepared()?;
        if byte_value as usize >= self.alphabet_size {
            return Err(PreparedExprError::Source);
        }
        byte(&mut Prepared(self), byte_value)
    }
    pub(crate) fn try_mk_repeat(
        &mut self,
        arg: ExprRef,
        min: u32,
        max: u32,
    ) -> Result<ExprRef, PreparedExprError> {
        self.prepared_child(arg)?;
        repeat(&mut Prepared(self), arg, min, max)
    }
    pub(crate) fn try_mk_byte_concat(
        &mut self,
        bytes: &[u8],
        tail: ExprRef,
    ) -> Result<ExprRef, PreparedExprError> {
        self.prepared_child(tail)?;
        if bytes.iter().any(|&b| b as usize >= self.alphabet_size) {
            return Err(PreparedExprError::Source);
        }
        byte_concat(&mut Prepared(self), bytes, tail)
    }
    pub(crate) fn try_mk_not(&mut self, arg: ExprRef) -> Result<ExprRef, PreparedExprError> {
        self.prepared_child(arg)?;
        not(&mut Prepared(self), arg)
    }
    pub(crate) fn try_mk_lookahead(
        &mut self,
        arg: ExprRef,
        offset: u32,
    ) -> Result<ExprRef, PreparedExprError> {
        self.prepared_child(arg)?;
        lookahead(&mut Prepared(self), arg, offset)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ast::ExprEncodingError, raw::HashConsCapacityError};

    #[test]
    fn prepared_expression_source_uses_shared_scalars_and_preserves_exhausted_owner() {
        let mut source = ExprSet::new(256);
        source.mk_byte_literal(&[b'x'; ExprRef::MAX_BYTE_CONCAT + 1]);
        // Historical ordinary construction owns this actual table layout. Its
        // completed source supplies finite slots; the prepared copy never grows.
        source.reserve(24);
        let mut ordinary = source.clone();
        let plan = source.prepared_source_plan().unwrap();
        let requirements = plan.requirements();
        assert_eq!(
            requirements.required_bytes(),
            requirements.buffer_bytes() + requirements.control_bytes()
        );
        let mut prepared = plan.compile().unwrap();
        drop(source);

        let a = prepared.source_mut().try_mk_byte(b'Q').unwrap();
        assert_eq!(a, ordinary.mk_byte(b'Q'));
        let repeat = prepared.source_mut().try_mk_repeat(a, 0, 3).unwrap();
        assert_eq!(repeat, ordinary.mk_repeat(a, 0, 3));
        assert!(prepared.source().is_nullable(repeat));
        let not = prepared.source_mut().try_mk_not(repeat).unwrap();
        assert_eq!(not, ordinary.mk_not(repeat));
        assert_eq!(
            prepared.source_mut().try_mk_not(not).unwrap(),
            ordinary.mk_not(not)
        );
        let ahead = prepared.source_mut().try_mk_lookahead(repeat, 7).unwrap();
        assert_eq!(ahead, ordinary.mk_lookahead(repeat, 7));
        assert!(matches!(
            prepared.source().get(ahead),
            Expr::Lookahead(_, ExprRef::EMPTY_STRING, 7)
        ));
        let bytes = [b'y'; ExprRef::MAX_BYTE_CONCAT + 3];
        let literal = prepared.source_mut().try_mk_byte_concat(&bytes, a).unwrap();
        assert_eq!(literal, ordinary.mk_byte_concat(&bytes, a));
        for id in [a, repeat, not, ahead, literal] {
            assert_eq!(
                prepared.source().expr_to_string(id),
                ordinary.expr_to_string(id)
            );
        }
        assert_eq!(prepared.source().cost(), ordinary.cost());

        let cost = prepared.source().cost();
        assert!(matches!(
            prepared
                .source_mut()
                .try_mk_repeat(ExprRef::new(u32::MAX), 1, 2),
            Err(PreparedExprError::Source)
        ));
        assert_eq!(prepared.source().cost(), cost);
        let mut exhausted = false;
        for byte in 0..=u8::MAX {
            let before = prepared.source().len();
            let cost = prepared.source().cost();
            match prepared.source_mut().try_mk_byte(byte) {
                Ok(_) => {}
                Err(PreparedExprError::Encoding(ExprEncodingError::Storage(
                    HashConsCapacityError::EntriesExceeded { .. }
                    | HashConsCapacityError::WordsExceeded { .. },
                ))) => {
                    assert_eq!(prepared.source().len(), before);
                    assert_eq!(prepared.source().cost(), cost + 1);
                    exhausted = true;
                    break;
                }
                Err(other) => panic!("unexpected source refusal: {other}"),
            }
        }
        assert!(exhausted);
        assert_eq!(prepared.source_mut().try_mk_byte(b'Q').unwrap(), a);
        assert!(prepared.source().is_valid(literal));
        assert_eq!(
            prepared.source().expr_to_string(literal),
            ordinary.expr_to_string(literal)
        );
        assert!(matches!(
            ordinary.try_mk_byte(b'R'),
            Err(PreparedExprError::Storage)
        ));
    }
}

pub(super) fn and2<S: Emission>(sink: &mut S, a: ExprRef, b: ExprRef) -> Result<ExprRef, S::Error> {
    sink.pay(2)?;
    let (a, b) = if a < b { (a, b) } else { (b, a) };
    let nullable = sink.source().is_nullable(a) && sink.source().is_nullable(b);
    sink.emit(Expr::And(
        ExprFlags::from_nullable_positive(nullable, nullable),
        &[a, b],
    ))
}
impl ExprSet {
    pub(crate) fn try_mk_and2(
        &mut self,
        a: ExprRef,
        b: ExprRef,
    ) -> Result<ExprRef, PreparedExprError> {
        self.prepared_child(a)?;
        self.prepared_child(b)?;
        and2(&mut Prepared(self), a, b)
    }
}
