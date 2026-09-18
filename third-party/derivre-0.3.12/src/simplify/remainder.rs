//! Same remainder equations with fixed mapped-decimal destinations.
use super::scalar::{self, Emission};
use crate::ast::{Expr, ExprRef, ExprSet, PreparedExprError};
use std::mem::{size_of, size_of_val};

// A u32 has at most ten decimal digits. In the legacy wrapping-power build,
// 10^scale is zero for scale >= 32, so the forcing branch can only request a
// width below 32. The checked producer refuses arithmetic overflow earlier.
struct Digits {
    bytes: [u8; u32::BITS as usize],
    start: usize,
}

pub(crate) fn fixed_control_bytes() -> Option<usize> {
    let frames = [
        size_of::<Digits>(),
        size_of::<scalar::Prepared<'_>>(),
        size_of::<(u32, u32, u32, bool)>(),
        size_of::<(u32, u32)>(),
        size_of::<(&[u8; 10], usize)>(),
        size_of::<Result<ExprRef, PreparedExprError>>(),
        size_of::<Result<u32, PreparedExprError>>(),
    ];
    frames
        .into_iter()
        .try_fold(size_of_val(&frames), usize::checked_add)
}
impl Digits {
    fn mapped(mut value: u32, width: u32, digits: &[u8; 10]) -> Self {
        assert!(
            width < u32::BITS,
            "forcing decimal width follows its nonzero u32 power"
        );
        let mut result = Self {
            bytes: [0; u32::BITS as usize],
            start: u32::BITS as usize,
        };
        loop {
            result.start -= 1;
            result.bytes[result.start] = digits[(value % 10) as usize];
            value /= 10;
            if value == 0 {
                break;
            }
        }
        while result.bytes.len() - result.start < width as usize {
            result.start -= 1;
            result.bytes[result.start] = digits[0];
        }
        result
    }
    fn as_bytes(&self) -> &[u8] {
        &self.bytes[self.start..]
    }
}

pub(super) fn construct<S: Emission>(
    sink: &mut S,
    divisor: u32,
    remainder: u32,
    scale: u32,
    fractional: bool,
) -> Result<ExprRef, S::Error> {
    sink.pay(1)?;
    if !fractional {
        return sink.emit(Expr::RemainderIs {
            divisor,
            remainder,
            scale,
            fractional_part: false,
        });
    }
    if scale == 0 && remainder == 0 {
        return Ok(ExprRef::EMPTY_STRING);
    }
    let multiplier = sink.power10(scale)?;
    let remaining = (divisor - remainder) % divisor;
    if remaining < multiplier {
        if multiplier <= divisor {
            let digits = Digits::mapped(remaining, scale, &sink.source().digits);
            scalar::byte_concat(sink, digits.as_bytes(), ExprRef::EMPTY_STRING)
        } else {
            sink.emit(Expr::RemainderIs {
                divisor,
                remainder,
                scale,
                fractional_part: fractional,
            })
        }
    } else {
        Ok(ExprRef::NO_MATCH)
    }
}
impl ExprSet {
    pub(crate) fn try_mk_remainder_is(
        &mut self,
        divisor: u32,
        remainder: u32,
        scale: u32,
        fractional: bool,
    ) -> Result<ExprRef, PreparedExprError> {
        if divisor == 0 || remainder > divisor {
            return Err(PreparedExprError::Source);
        }
        construct(
            &mut scalar::Prepared(self),
            divisor,
            remainder,
            scale,
            fractional,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_remainder_digits_preserve_width_mapping_cost_and_checked_arithmetic() {
        let mapping = [40, 41, 42, 43, 44, 45, 46, 47, 48, 49];
        for value in [0, 1, 25, 999, u32::MAX] {
            for width in 0..u32::BITS {
                let ordinary = format!("{:0>width$}", value, width = width as usize);
                let expected: Vec<_> = ordinary
                    .bytes()
                    .map(|byte| mapping[(byte - b'0') as usize])
                    .collect();
                assert_eq!(
                    Digits::mapped(value, width, &mapping).as_bytes(),
                    expected.as_slice()
                );
            }
        }
        let mut source = ExprSet::new(256, crate::ParserAllocationFunding::unenforced()).unwrap();
        source.reserve(24).unwrap();
        source.digits = mapping;
        let mut ordinary = source.clone();
        let mut prepared = source.prepared_source_plan().unwrap().compile().unwrap();
        drop(source);
        let cases = [
            (1000, 999, 3, true),
            (10, 10, 0, true),
            (10, 0, 0, true),
            (250, 125, 2, true),
            (3, 1, 2, true),
            (13, 3, 1, false),
        ];
        for (divisor, remainder, scale, fractional) in cases {
            let expected = ordinary.mk_remainder_is(divisor, remainder, scale, fractional).unwrap();
            let actual = prepared
                .source_mut()
                .try_mk_remainder_is(divisor, remainder, scale, fractional)
                .unwrap();
            assert_eq!(actual, expected);
            assert_eq!(
                prepared.source().expr_to_string(actual),
                ordinary.expr_to_string(expected)
            );
            assert_eq!(prepared.source().cost(), ordinary.cost());
        }
        let entries = prepared.source().len();
        let cost = prepared.source().cost();
        assert!(matches!(
            prepared.source_mut().try_mk_remainder_is(7, 1, 10, true),
            Err(PreparedExprError::Source)
        ));
        assert_eq!(prepared.source().len(), entries);
        assert_eq!(prepared.source().cost(), cost + 1);
        let cost = prepared.source().cost();
        assert!(matches!(
            prepared.source_mut().try_mk_remainder_is(0, 0, 0, false),
            Err(PreparedExprError::Source)
        ));
        assert_eq!(prepared.source().cost(), cost);
    }
}
