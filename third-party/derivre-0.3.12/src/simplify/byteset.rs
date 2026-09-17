//! One byte-set equation worker with ordinary and fixed scratch destinations.
use super::scalar::{self, Emission};
use crate::ast::{
    byteset_clear, byteset_contains, byteset_set, byteset_union, Expr, ExprRef, ExprSet,
    PreparedExprError,
};
use std::{
    convert::Infallible,
    mem::{size_of, size_of_val},
};

pub(super) trait Memory {
    type Error;
    fn words(&mut self, count: usize) -> Result<&mut [u32], Self::Error>;
    fn invalid() -> Self::Error;
}
pub(super) struct Ordinary(Vec<u32>);
impl Memory for Ordinary {
    type Error = Infallible;
    fn words(&mut self, count: usize) -> Result<&mut [u32], Infallible> {
        self.0.resize(count, 0);
        self.0.fill(0);
        Ok(&mut self.0)
    }
    fn invalid() -> Infallible {
        panic!("byte-set operation requires a byte or byte set")
    }
}
pub(super) fn ordinary(
    source: &mut ExprSet,
    work: impl FnOnce(&mut scalar::Ordinary<'_>, &mut Ordinary) -> Result<ExprRef, Infallible>,
) -> ExprRef {
    let result = work(&mut scalar::Ordinary(source), &mut Ordinary(Vec::new()));
    match result {
        Ok(value) => value,
        Err(never) => match never {},
    }
}

pub(super) fn not<S: Emission, M: Memory<Error = S::Error>>(
    sink: &mut S,
    memory: &mut M,
    x: ExprRef,
) -> Result<ExprRef, S::Error> {
    let expression = sink.source().get(x);
    match expression {
        Expr::Byte(byte) => {
            let words = memory.words(sink.source().alphabet_words)?;
            words.fill(!0);
            byteset_clear(words, byte as usize);
            scalar::byte_set(sink, words)
        }
        Expr::ByteSet(input) => {
            let words = memory.words(input.len())?;
            for (out, &value) in words.iter_mut().zip(input) {
                *out = !value;
            }
            scalar::byte_set(sink, words)
        }
        _ => Err(M::invalid()),
    }
}
pub(super) fn union<S: Emission, M: Memory<Error = S::Error>>(
    sink: &mut S,
    memory: &mut M,
    args: &[ExprRef],
    negated: bool,
) -> Result<ExprRef, S::Error> {
    let words = memory.words(sink.source().alphabet_words)?;
    for &arg in args {
        match sink.source().get(arg) {
            Expr::Byte(byte) => byteset_set(words, byte as usize),
            Expr::ByteSet(input) => byteset_union(words, input),
            _ => return Err(M::invalid()),
        }
    }
    if negated {
        for word in words.iter_mut() {
            *word = !*word;
        }
        for bit in sink.source().alphabet_size..sink.source().alphabet_words * 32 {
            byteset_clear(words, bit);
        }
    }
    scalar::byte_set(sink, words)
}
pub(super) fn intersection<S: Emission, M: Memory<Error = S::Error>>(
    sink: &mut S,
    memory: &mut M,
    a: ExprRef,
    b: ExprRef,
) -> Result<ExprRef, S::Error> {
    if a == b {
        return Ok(a);
    }
    match (sink.source().get(a), sink.source().get(b)) {
        (Expr::Byte(_), Expr::Byte(_)) => Ok(ExprRef::NO_MATCH),
        (Expr::Byte(byte), Expr::ByteSet(set)) => Ok(if byteset_contains(set, byte as usize) {
            a
        } else {
            ExprRef::NO_MATCH
        }),
        (Expr::ByteSet(set), Expr::Byte(byte)) => Ok(if byteset_contains(set, byte as usize) {
            b
        } else {
            ExprRef::NO_MATCH
        }),
        (Expr::ByteSet(left), Expr::ByteSet(right)) => {
            let words = memory.words(left.len())?;
            for index in 0..words.len() {
                words[index] = left[index] & right[index];
            }
            scalar::byte_set(sink, words)
        }
        _ => Err(M::invalid()),
    }
}
pub(super) fn subtract<S: Emission, M: Memory<Error = S::Error>>(
    sink: &mut S,
    memory: &mut M,
    a: ExprRef,
    b: ExprRef,
) -> Result<ExprRef, S::Error> {
    match (sink.source().get(a), sink.source().get(b)) {
        (Expr::Byte(x), Expr::Byte(y)) => Ok(if x == y { ExprRef::NO_MATCH } else { a }),
        (Expr::Byte(byte), Expr::ByteSet(set)) => Ok(if byteset_contains(set, byte as usize) {
            ExprRef::NO_MATCH
        } else {
            a
        }),
        (Expr::ByteSet(set), Expr::Byte(byte)) => {
            if !byteset_contains(set, byte as usize) {
                return Ok(a);
            }
            let words = memory.words(set.len())?;
            words.copy_from_slice(set);
            byteset_clear(words, byte as usize);
            scalar::byte_set(sink, words)
        }
        (Expr::ByteSet(left), Expr::ByteSet(right)) => {
            let words = memory.words(left.len())?;
            for index in 0..words.len() {
                words[index] = left[index] & !right[index];
            }
            scalar::byte_set(sink, words)
        }
        _ => Err(M::invalid()),
    }
}

// Byte selectors are u8: the maximum complete domain is exactly eight u32s.
// Larger ordinary expression alphabets remain on their ordinary destination.
struct Fixed {
    words: [u32; 8],
}
impl Memory for Fixed {
    type Error = PreparedExprError;
    fn words(&mut self, count: usize) -> Result<&mut [u32], Self::Error> {
        let words = self
            .words
            .get_mut(..count)
            .ok_or(PreparedExprError::Source)?;
        words.fill(0);
        Ok(words)
    }
    fn invalid() -> Self::Error {
        PreparedExprError::Source
    }
}
fn validate(source: &ExprSet, args: &[ExprRef]) -> Result<(), PreparedExprError> {
    source.require_prepared()?;
    if source.alphabet_size == 0
        || source.alphabet_size > 256
        || source.alphabet_words != source.alphabet_size.div_ceil(32)
    {
        return Err(PreparedExprError::Source);
    }
    for &arg in args {
        if !source.is_valid(arg) {
            return Err(PreparedExprError::Source);
        }
        match source.get(arg) {
            Expr::Byte(byte) if usize::from(byte) < source.alphabet_size => {}
            Expr::ByteSet(words) if words.len() == source.alphabet_words => {}
            _ => return Err(PreparedExprError::Source),
        }
    }
    Ok(())
}
pub(crate) fn fixed_control_bytes() -> Option<usize> {
    let frames = [
        size_of::<Fixed>(),
        size_of::<scalar::Prepared<'_>>(),
        size_of::<Expr<'_>>() * 2,
        size_of::<&mut [u32]>(),
        size_of::<Result<ExprRef, PreparedExprError>>(),
        size_of::<[ExprRef; 2]>(),
    ];
    frames
        .into_iter()
        .try_fold(size_of_val(&frames), usize::checked_add)
}
impl ExprSet {
    pub(crate) fn try_mk_byte_set_not(
        &mut self,
        arg: ExprRef,
    ) -> Result<ExprRef, PreparedExprError> {
        validate(self, &[arg])?;
        not(
            &mut scalar::Prepared(self),
            &mut Fixed { words: [0; 8] },
            arg,
        )
    }
    pub(crate) fn try_mk_byte_set_or(
        &mut self,
        args: &[ExprRef],
    ) -> Result<ExprRef, PreparedExprError> {
        validate(self, args)?;
        union(
            &mut scalar::Prepared(self),
            &mut Fixed { words: [0; 8] },
            args,
            false,
        )
    }
    pub(crate) fn try_mk_byte_set_neg_or(
        &mut self,
        args: &[ExprRef],
    ) -> Result<ExprRef, PreparedExprError> {
        validate(self, args)?;
        union(
            &mut scalar::Prepared(self),
            &mut Fixed { words: [0; 8] },
            args,
            true,
        )
    }
    pub(crate) fn try_mk_byte_set_and(
        &mut self,
        a: ExprRef,
        b: ExprRef,
    ) -> Result<ExprRef, PreparedExprError> {
        validate(self, &[a, b])?;
        intersection(
            &mut scalar::Prepared(self),
            &mut Fixed { words: [0; 8] },
            a,
            b,
        )
    }
    pub(crate) fn try_mk_byte_set_sub(
        &mut self,
        a: ExprRef,
        b: ExprRef,
    ) -> Result<ExprRef, PreparedExprError> {
        validate(self, &[a, b])?;
        subtract(
            &mut scalar::Prepared(self),
            &mut Fixed { words: [0; 8] },
            a,
            b,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_byte_selectors_preserve_shared_tail_bits_shortcuts_cost_and_source_refusal() {
        assert!(fixed_control_bytes().unwrap() >= size_of::<Fixed>());
        for alphabet in [1, 31, 32, 33, 256] {
            let mut source = ExprSet::new(alphabet);
            let a = source.mk_byte(0);
            let b = source.mk_byte((alphabet - 1) as u8);
            let c = source.mk_byte((alphabet / 2) as u8);
            let left = source.mk_byte_set_or(&[a, b]);
            let right = source.mk_byte_set_or(&[a, c]);
            source.reserve(32);
            let mut ordinary = source.clone();
            let mut prepared = source.prepared_source_plan().unwrap().compile().unwrap();
            drop(source);
            let extent = prepared.source().prepared_extents().unwrap();
            for operation in 0..12 {
                let expected = match operation {
                    0 => ordinary.mk_byte_set_not(left),
                    1 => ordinary.mk_byte_set_or(&[]),
                    2 => ordinary.mk_byte_set_or(&[left, right]),
                    3 => ordinary.mk_byte_set_neg_or(&[left, right]),
                    4 => ordinary.mk_byte_set_and(left, right),
                    5 => ordinary.mk_byte_set_sub(left, right),
                    6 => ordinary.mk_byte_set_sub(left, a),
                    7 => ordinary.mk_byte_set_and(a, right),
                    8 => ordinary.mk_byte_set_not(a),
                    9 => ordinary.mk_byte_set_sub(a, right),
                    10 => ordinary.mk_byte_set_and(left, left),
                    11 => ordinary.mk_byte_set_neg_or(&[]),
                    _ => unreachable!(),
                };
                let p = prepared.source_mut();
                let actual = match operation {
                    0 => p.try_mk_byte_set_not(left),
                    1 => p.try_mk_byte_set_or(&[]),
                    2 => p.try_mk_byte_set_or(&[left, right]),
                    3 => p.try_mk_byte_set_neg_or(&[left, right]),
                    4 => p.try_mk_byte_set_and(left, right),
                    5 => p.try_mk_byte_set_sub(left, right),
                    6 => p.try_mk_byte_set_sub(left, a),
                    7 => p.try_mk_byte_set_and(a, right),
                    8 => p.try_mk_byte_set_not(a),
                    9 => p.try_mk_byte_set_sub(a, right),
                    10 => p.try_mk_byte_set_and(left, left),
                    11 => p.try_mk_byte_set_neg_or(&[]),
                    _ => unreachable!(),
                }
                .unwrap();
                assert_eq!(actual, expected);
                assert_eq!(p.cost(), ordinary.cost());
                // The legacy simple printer assumes all 256 byte bits. These
                // fixtures intentionally retain smaller symbol alphabets; use
                // their exact selector domain and encoded tail bits instead.
                assert_eq!(p.get_tag(actual), ordinary.get_tag(expected));
                for byte in 0..alphabet {
                    assert_eq!(
                        p.get(actual).matches_byte(byte as u8),
                        ordinary.get(expected).matches_byte(byte as u8)
                    );
                }
                assert_eq!(p.prepared_extents().unwrap(), extent);
                if let (Expr::ByteSet(actual), Expr::ByteSet(expected)) =
                    (p.get(actual), ordinary.get(expected))
                {
                    assert_eq!(actual, expected);
                }
            }
            let cost = prepared.source().cost();
            let entries = prepared.source().len();
            assert!(matches!(
                prepared
                    .source_mut()
                    .try_mk_byte_set_sub(left, ExprRef::new(u32::MAX)),
                Err(PreparedExprError::Source)
            ));
            assert!(matches!(
                prepared
                    .source_mut()
                    .try_mk_byte_set_or(&[ExprRef::EMPTY_STRING]),
                Err(PreparedExprError::Source)
            ));
            assert_eq!(prepared.source().cost(), cost);
            assert_eq!(prepared.source().len(), entries);
            assert!(prepared.source().is_valid(left));
        }
    }
}
