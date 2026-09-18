//! One derivative node worker with explicit fallible construction boundaries.
//!
//! This interface supplies no storage or execution authority. Original callers
//! must qualify every constructor, including its scratch and expression growth.
use crate::ast::{Expr, ExprRef, ExprSet};
use crate::ParserError as ConstructionError;

pub(crate) trait Construction {
    type Error;
    fn source(&self) -> &ExprSet;
    fn multiply(&mut self, left: u32, right: u32) -> Result<u32, Self::Error>;
    fn add(&mut self, left: u32, right: u32) -> Result<u32, Self::Error>;
    fn power10(&mut self, scale: u32) -> Result<u32, Self::Error>;
    fn modulo(&mut self, value: u32, divisor: u32) -> Result<u32, Self::Error>;
    fn byte_suffix(&mut self, source: ExprRef, tail: ExprRef) -> Result<ExprRef, Self::Error>;
    fn remainder(
        &mut self,
        divisor: u32,
        remainder: u32,
        scale: u32,
        fractional: bool,
    ) -> Result<ExprRef, Self::Error>;
    fn and(&mut self, args: &mut Vec<ExprRef>) -> Result<ExprRef, Self::Error>;
    fn or(&mut self, args: &mut Vec<ExprRef>) -> Result<ExprRef, Self::Error>;
    fn not(&mut self, arg: ExprRef) -> Result<ExprRef, Self::Error>;
    fn repeat(&mut self, arg: ExprRef, min: u32, max: u32) -> Result<ExprRef, Self::Error>;
    fn concat(&mut self, left: ExprRef, right: ExprRef) -> Result<ExprRef, Self::Error>;
    fn alternatives(&mut self, left: ExprRef, right: ExprRef) -> Result<ExprRef, Self::Error>;
    fn lookahead(&mut self, arg: ExprRef, offset: u32) -> Result<ExprRef, Self::Error>;
}

pub(crate) fn surely_no_match(source: &ExprSet, root: ExprRef, byte: u8) -> bool {
    match source.get(root) {
        Expr::Concat(_, args) => source.get(args[0]).surely_no_match(byte),
        expression => expression.surely_no_match(byte),
    }
}

/// Shared equations and constructor order. A refusal stops at that constructor;
/// prior expression mutations and caller-owned child values remain untouched.
pub(crate) fn node<C: Construction>(
    construction: &mut C,
    derivatives: &mut Vec<ExprRef>,
    root: ExprRef,
    byte: u8,
) -> Result<ExprRef, C::Error> {
    let expression = construction.source().get(root);
    match expression {
        Expr::EmptyString | Expr::NoMatch | Expr::ByteSet(_) | Expr::Byte(_) => {
            Ok(if expression.matches_byte(byte) {
                ExprRef::EMPTY_STRING
            } else {
                ExprRef::NO_MATCH
            })
        }
        Expr::ByteConcat(_, bytes, tail) => {
            if bytes[0] == byte {
                construction.byte_suffix(root, tail)
            } else {
                Ok(ExprRef::NO_MATCH)
            }
        }
        Expr::RemainderIs {
            divisor,
            remainder,
            scale,
            fractional_part,
        } => {
            if let Some(idx) = construction.source().digits.iter().position(|&x| x == byte) {
                let (remainder, scale) = if !fractional_part {
                    (construction.multiply(remainder, 10)?, scale)
                } else {
                    if scale == 0 {
                        return Ok(ExprRef::NO_MATCH);
                    }
                    (remainder, scale - 1)
                };
                let power = construction.power10(scale)?;
                let term = construction.multiply(idx as u32, power)?;
                let value = construction.add(remainder, term)?;
                let remainder = construction.modulo(value, divisor)?;
                construction.remainder(divisor, remainder, scale, fractional_part)
            } else if byte == construction.source().digit_dot && !fractional_part && scale > 0 {
                construction.remainder(divisor, remainder, scale, true)
            } else {
                Ok(ExprRef::NO_MATCH)
            }
        }
        Expr::And(_, _) => construction.and(derivatives),
        Expr::Or(_, _) => construction.or(derivatives),
        Expr::Not(_, _) => construction.not(derivatives[0]),
        Expr::Repeat(_, arg, min, max) => {
            if derivatives[0] == ExprRef::NO_MATCH {
                return Ok(ExprRef::NO_MATCH);
            }
            let max = if max == u32::MAX {
                u32::MAX
            } else {
                max.saturating_sub(1)
            };
            let tail = construction.repeat(arg, min.saturating_sub(1), max)?;
            construction.concat(derivatives[0], tail)
        }
        Expr::Concat(_, [left, right]) => {
            let first = construction.concat(derivatives[0], right)?;
            if construction.source().is_nullable(left) {
                construction.alternatives(first, derivatives[1])
            } else {
                Ok(first)
            }
        }
        Expr::Lookahead(_, arg, offset) => {
            if arg == ExprRef::EMPTY_STRING {
                Ok(ExprRef::NO_MATCH)
            } else {
                let offset = construction.add(offset, 1)?;
                construction.lookahead(derivatives[0], offset)
            }
        }
    }
}

pub(super) struct Ordinary<'a> {
    pub(super) expressions: &'a mut ExprSet,
    pub(super) alternatives: &'a mut Vec<ExprRef>,
}
impl Construction for Ordinary<'_> {
    type Error = ConstructionError;
    fn source(&self) -> &ExprSet {
        self.expressions
    }
    fn multiply(&mut self, left: u32, right: u32) -> Result<u32, ConstructionError> {
        Ok(left.checked_mul(right).ok_or(crate::raw::PreparedExprError::Source)?)
    }
    fn add(&mut self, left: u32, right: u32) -> Result<u32, ConstructionError> {
        Ok(left.checked_add(right).ok_or(crate::raw::PreparedExprError::Source)?)
    }
    fn power10(&mut self, scale: u32) -> Result<u32, ConstructionError> {
        Ok(10u32.checked_pow(scale).ok_or(crate::raw::PreparedExprError::Source)?)
    }
    fn modulo(&mut self, value: u32, divisor: u32) -> Result<u32, ConstructionError> {
        Ok(value.checked_rem(divisor).ok_or(crate::raw::PreparedExprError::Source)?)
    }

    fn byte_suffix(&mut self, root: ExprRef, tail: ExprRef) -> Result<ExprRef, ConstructionError> {
        let Expr::ByteConcat(_, bytes, _) = self.expressions.get(root) else {
            unreachable!("derivative byte suffix requires its source node")
        };
        let mut copy = [0; ExprRef::MAX_BYTE_CONCAT];
        let len = bytes.len() - 1;
        copy[..len].copy_from_slice(&bytes[1..]);
        Ok(self.expressions.mk_byte_concat(&copy[..len], tail)?)
    }
    fn remainder(
        &mut self,
        divisor: u32,
        remainder: u32,
        scale: u32,
        fractional: bool,
    ) -> Result<ExprRef, ConstructionError> {
        Ok(self
            .expressions
            .mk_remainder_is(divisor, remainder, scale, fractional)?)
    }
    fn and(&mut self, args: &mut Vec<ExprRef>) -> Result<ExprRef, ConstructionError> {
        Ok(self.expressions.mk_and(args)?)
    }
    fn or(&mut self, args: &mut Vec<ExprRef>) -> Result<ExprRef, ConstructionError> {
        Ok(self.expressions.mk_or(args)?)
    }
    fn not(&mut self, arg: ExprRef) -> Result<ExprRef, ConstructionError> {
        Ok(self.expressions.mk_not(arg)?)
    }
    fn repeat(&mut self, arg: ExprRef, min: u32, max: u32) -> Result<ExprRef, ConstructionError> {
        Ok(self.expressions.mk_repeat(arg, min, max)?)
    }
    fn concat(&mut self, left: ExprRef, right: ExprRef) -> Result<ExprRef, ConstructionError> {
        Ok(self.expressions.mk_concat(left, right)?)
    }
    fn alternatives(&mut self, left: ExprRef, right: ExprRef) -> Result<ExprRef, ConstructionError> {
        self.alternatives.clear();
        self.expressions.construction_funding()?.try_extend_copy(self.alternatives, &[left, right])?;
        Ok(self.expressions.mk_or(self.alternatives)?)
    }
    fn lookahead(&mut self, arg: ExprRef, offset: u32) -> Result<ExprRef, ConstructionError> {
        Ok(self.expressions.mk_lookahead(arg, offset)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ast::ExprFlags, raw::DerivCache};

    #[derive(Debug)]
    struct Failure {
        at: &'static str,
        completed: Option<ExprRef>,
    }
    struct Refusing<'a> {
        ordinary: Ordinary<'a>,
        refuse: &'static str,
        completed: Option<ExprRef>,
    }
    macro_rules! constructor {
        ($name:ident($($arg:ident: $ty:ty),*)) => {
            fn $name(&mut self, $($arg: $ty),*) -> Result<ExprRef, Failure> {
                if self.refuse == stringify!($name) {
                    return Err(Failure { at: self.refuse, completed: self.completed });
                }
                let value = match self.ordinary.$name($($arg),*) {
                    Ok(value) => value,
                    Err(error) => panic!("unenforced fixture constructor: {error}"),
                };
                self.completed = Some(value);
                Ok(value)
            }
        };
    }
    impl Construction for Refusing<'_> {
        type Error = Failure;
        fn source(&self) -> &ExprSet {
            self.ordinary.source()
        }
        fn multiply(&mut self, left: u32, right: u32) -> Result<u32, Failure> {
            Ok(left * right)
        }
        fn add(&mut self, left: u32, right: u32) -> Result<u32, Failure> {
            Ok(left + right)
        }
        fn power10(&mut self, scale: u32) -> Result<u32, Failure> {
            Ok(10u32.pow(scale))
        }
        fn modulo(&mut self, value: u32, divisor: u32) -> Result<u32, Failure> {
            Ok(value % divisor)
        }

        constructor!(byte_suffix(root: ExprRef, tail: ExprRef));
        constructor!(remainder(divisor: u32, remainder: u32, scale: u32, fractional: bool));
        constructor!(and(args: &mut Vec<ExprRef>));
        constructor!(or(args: &mut Vec<ExprRef>));
        constructor!(not(arg: ExprRef));
        constructor!(repeat(arg: ExprRef, min: u32, max: u32));
        constructor!(concat(left: ExprRef, right: ExprRef));
        constructor!(alternatives(left: ExprRef, right: ExprRef));
        constructor!(lookahead(arg: ExprRef, offset: u32));
    }

    #[test]
    fn shared_derivative_preserves_partial_constructors_nullable_order_and_actual_suffix() {
        let mut source = ExprSet::new(256, crate::ParserAllocationFunding::unenforced()).unwrap();
        let a = source.mk_byte(b'a').unwrap();
        let repeat = source.mk_repeat(a, 2, 4).unwrap();
        let before = source.len();
        let mut alternatives = Vec::new();
        let mut derivatives = vec![ExprRef::EMPTY_STRING];
        let error = node(
            &mut Refusing {
                ordinary: Ordinary {
                    expressions: &mut source,
                    alternatives: &mut alternatives,
                },
                refuse: "concat",
                completed: None,
            },
            &mut derivatives,
            repeat,
            b'a',
        )
        .unwrap_err();
        assert_eq!(error.at, "concat");
        let tail = error.completed.unwrap();
        assert!(tail.as_usize() >= before);
        assert!(matches!(source.get(tail), Expr::Repeat(_, arg, 1, 3) if arg == a));
        assert_eq!(derivatives, vec![ExprRef::EMPTY_STRING]);
        assert!(alternatives.is_empty());
        assert_eq!(
            DerivCache::new().derivative(&mut source, repeat, b'a').unwrap(),
            tail
        );

        let nullable = source.mk_repeat(a, 0, 2).unwrap();
        let c = source.mk_byte(b'c').unwrap();
        let concat = source.mk(Expr::Concat(ExprFlags::POSITIVE, [nullable, c])).unwrap();
        let before = source.len();
        let mut derivatives = vec![a, ExprRef::EMPTY_STRING];
        let error = node(
            &mut Refusing {
                ordinary: Ordinary {
                    expressions: &mut source,
                    alternatives: &mut alternatives,
                },
                refuse: "alternatives",
                completed: None,
            },
            &mut derivatives,
            concat,
            b'a',
        )
        .unwrap_err();
        assert_eq!(error.at, "alternatives");
        let first = error.completed.unwrap();
        assert!(first.as_usize() >= before);
        assert!(matches!(source.get(first), Expr::Concat(_, args) if args == [a, c]));
        assert_eq!(derivatives, vec![a, ExprRef::EMPTY_STRING]);
        assert!(alternatives.is_empty());

        let literal = source.mk_byte_literal(b"abc").unwrap();
        let suffix = DerivCache::new().derivative(&mut source, literal, b'a').unwrap();
        assert!(
            matches!(source.get(suffix), Expr::ByteConcat(_, bytes, tail)
            if bytes == b"bc" && tail == ExprRef::EMPTY_STRING)
        );
        assert_eq!(
            DerivCache::new().derivative(&mut source, literal, b'x').unwrap(),
            ExprRef::NO_MATCH
        );
        assert!(surely_no_match(&source, literal, b'x'));
    }
}
