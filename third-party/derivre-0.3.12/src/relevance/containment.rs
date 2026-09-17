//! Existing conservative prefix-containment policy over shared source workers.
pub(crate) mod branches;
pub(crate) mod length;
use super::{entry, RelevanceCache};
use crate::{
    ast::{Expr, ExprRef, ExprSet},
    nextbyte::next_byte_simple,
    raw::DerivCache,
    NextByte,
};
pub(crate) trait Context {
    type Error;
    fn source(&self) -> &ExprSet;
    fn derivative(&mut self, root: ExprRef, byte: u8) -> Result<ExprRef, Self::Error>;
    fn not(&mut self, root: ExprRef) -> Result<ExprRef, Self::Error>;
    fn and2(&mut self, left: ExprRef, right: ExprRef) -> Result<ExprRef, Self::Error>;
    fn repeat(&mut self, root: ExprRef, min: u32, max: u32) -> Result<ExprRef, Self::Error>;
    fn non_empty(&mut self, root: ExprRef) -> Result<bool, Self::Error>;
    fn max_length(&mut self, root: ExprRef) -> Result<Option<usize>, Self::Error>;
    fn branches(&mut self, main: ExprRef, head: ExprRef) -> Result<Option<ExprRef>, Self::Error>;
    fn leftovers(&self) -> &[(ExprRef, u32)];
}
fn contains<C: Context>(c: &mut C, small: ExprRef, big: ExprRef) -> Result<bool, C::Error> {
    if small == big {
        return Ok(true);
    }
    let not_big = c.not(big)?;
    let difference = c.and2(small, not_big)?;
    c.non_empty(difference).map(|r| !r)
}
fn alternative(source: &ExprSet, root: ExprRef, index: usize) -> Option<ExprRef> {
    match source.get(root) {
        Expr::Or(_, args) => args.get(index).copied(),
        Expr::NoMatch => None,
        _ => {
            if index == 0 {
                Some(root)
            } else {
                None
            }
        }
    }
}
fn repeats<C: Context>(
    c: &mut C,
    small: ExprRef,
    main: ExprRef,
    except: ExprRef,
) -> Result<bool, C::Error> {
    match (c.source().get(small), c.source().get(main)) {
        (
            Expr::Repeat(_, small_child, _, small_high),
            Expr::Repeat(_, main_child, _, main_high),
        ) if small_high <= main_high && 2 <= main_high => {
            let mut index = 0;
            while let Some(child) = alternative(c.source(), except, index) {
                if c.max_length(child)?.unwrap_or(usize::MAX) >= main_high as usize {
                    return Ok(false);
                }
                index += 1;
            }
            contains(c, small_child, main_child)
        }
        _ => Ok(false),
    }
}
pub(crate) fn run<C: Context>(
    c: &mut C,
    mut small: ExprRef,
    mut big: ExprRef,
) -> Result<bool, C::Error> {
    for _ in 0..10 {
        match next_byte_simple(c.source(), small) {
            NextByte::ForcedByte(byte) => {
                small = c.derivative(small, byte)?;
                big = c.derivative(big, byte)?;
            }
            _ => break,
        }
    }
    let (main, except) = a_and_not_b(c.source(), big).unwrap_or((big, ExprRef::NO_MATCH));
    let main = match c.source().get(main) {
        Expr::Concat(_, [left, right]) => match c.source().get(left) {
            Expr::Repeat(_, _, 0, 1) => right,
            _ => main,
        },
        _ => main,
    };
    if let Expr::Concat(_, [head, tail]) = c.source().get(small) {
        if matches!(c.source().get(tail), Expr::Repeat(_, _, _, _)) {
            if let Some(rest) = c.branches(main, head)? {
                return repeats(c, tail, rest, except);
            }
            for index in 0..c.leftovers().len() {
                let (child, maximum) = c.leftovers()[index];
                if contains(c, head, child)? {
                    let next = if maximum == u32::MAX {
                        maximum
                    } else {
                        maximum - 1
                    };
                    let next_main = c.repeat(child, 0, next)?;
                    if repeats(c, tail, next_main, except)? {
                        return Ok(true);
                    }
                }
            }
            return Ok(false);
        }
    }
    let main = match c.source().get(main) {
        Expr::Or(_, args) => args
            .iter()
            .find(|&&root| matches!(c.source().get(root), Expr::Repeat(_, _, _, _)))
            .copied()
            .unwrap_or(main),
        _ => main,
    };
    let (main, except) = strip_common_suffix(c.source(), main, except);
    repeats(c, small, main, except)
}
fn a_and_not_b(source: &ExprSet, root: ExprRef) -> Option<(ExprRef, ExprRef)> {
    match source.get(root) {
        Expr::And(_, [a, b]) => match (source.get(*a), source.get(*b)) {
            (Expr::Not(_, inner), _) => Some((*b, inner)),
            (_, Expr::Not(_, inner)) => Some((*a, inner)),
            _ => None,
        },
        _ => None,
    }
}
fn strip_common_suffix(source: &ExprSet, a: ExprRef, b: ExprRef) -> (ExprRef, ExprRef) {
    match (source.get(a), source.get(b)) {
        (Expr::Concat(_, [a0, a1]), Expr::Concat(_, [b0, b1])) if a1 == b1 => (a0, b0),
        (Expr::Concat(_, [a0, _]), _) => (a0, b),
        _ => (a, b),
    }
}
struct Ordinary<'a> {
    source: &'a mut ExprSet,
    cache: &'a mut RelevanceCache,
    derivative: &'a mut DerivCache,
    lengths: Vec<length::Task>,
    branches: branches::Ordinary,
}
impl Context for Ordinary<'_> {
    type Error = anyhow::Error;
    fn source(&self) -> &ExprSet {
        self.source
    }
    fn derivative(&mut self, root: ExprRef, byte: u8) -> anyhow::Result<ExprRef> {
        Ok(self.derivative.derivative(self.source, root, byte))
    }
    fn not(&mut self, root: ExprRef) -> anyhow::Result<ExprRef> {
        Ok(self.source.mk_not(root))
    }
    fn and2(&mut self, a: ExprRef, b: ExprRef) -> anyhow::Result<ExprRef> {
        Ok(self.source.mk_and2(a, b))
    }
    fn repeat(&mut self, root: ExprRef, min: u32, max: u32) -> anyhow::Result<ExprRef> {
        Ok(self.source.mk_repeat(root, min, max))
    }
    fn non_empty(&mut self, root: ExprRef) -> anyhow::Result<bool> {
        entry::ordinary(self.cache, self.source, root)
    }
    fn max_length(&mut self, root: ExprRef) -> anyhow::Result<Option<usize>> {
        length::run(self.source, root, &mut self.lengths, &mut length::Ordinary)
    }
    fn branches(&mut self, main: ExprRef, head: ExprRef) -> anyhow::Result<Option<ExprRef>> {
        self.branches.stack.clear();
        self.branches.rows.clear();
        branches::run(self.source, main, head, &mut self.branches)
    }
    fn leftovers(&self) -> &[(ExprRef, u32)] {
        &self.branches.rows
    }
}
pub(crate) fn ordinary(
    cache: &mut RelevanceCache,
    source: &mut ExprSet,
    derivative: &mut DerivCache,
    small: ExprRef,
    big: ExprRef,
) -> anyhow::Result<bool> {
    run(
        &mut Ordinary {
            source,
            cache,
            derivative,
            lengths: Vec::new(),
            branches: branches::Ordinary {
                stack: Vec::new(),
                rows: Vec::new(),
            },
        },
        small,
        big,
    )
}
