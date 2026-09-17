//! Shared fold callbacks over fixed scalar/source descriptors and flat materialization recipes.
#![forbid(unsafe_code)]
use super::{materialize::Operation, reader::Style};
use super::{view::Packed, Cause, Collection, Descriptor, Scalar, Workspace};
use crate::bounded::expression::store::{NodeId, Sequence};
use crate::compiler::{
    ast::{BinOpKind, CompareOpKind},
    fold::{Continuation, Store},
};
use crate::value::primitive::scalar::{self, Arithmetic};
use crate::value::primitive::text;
use crate::value::primitive::{self, Integer, Truth};
fn truth(owner: &Workspace<'_, '_>, value: Descriptor) -> Truth {
    match value {
        Descriptor::Text(value) => Truth::Length(value.bytes),
        Descriptor::List(id) => Truth::Length(owner.list_recipe(id).0.len),
        Descriptor::Scalar(value) => match value {
            Scalar::None => Truth::False,
            Scalar::Bool(x) => Truth::Bool(x),
            Scalar::U64(x) => Truth::U64(x),
            Scalar::U128(x) => Truth::U128(x),
            Scalar::I64(x) => Truth::I64(x),
            Scalar::I128(x) => Truth::I128(x),
            Scalar::F64(x) => Truth::F64(x),
        },
    }
}
fn integer(value: Integer) -> Descriptor {
    Descriptor::Scalar(match value {
        Integer::I64(x) => Scalar::I64(x),
        Integer::I128(x) => Scalar::I128(x),
    })
}
fn negative(value: Descriptor) -> Option<Descriptor> {
    let Descriptor::Scalar(value) = value else {
        return None;
    };
    let signed = match value {
        Scalar::F64(x) => {
            return Some(Descriptor::Scalar(Scalar::F64(primitive::negative_float(
                x,
            ))))
        }
        Scalar::U128(x) => {
            if let Some(x) = primitive::negative_special_u128(x) {
                return Some(Descriptor::Scalar(Scalar::U128(x)));
            }
            i128::try_from(x).ok()?
        }
        Scalar::U64(x) => i128::from(x),
        Scalar::I64(x) => i128::from(x),
        Scalar::I128(x) => x,
        Scalar::None | Scalar::Bool(_) => return None,
    };
    primitive::negative_integer(signed).map(integer)
}
impl<'o, 's: 'o> Store<Packed<'o, 's>> for Workspace<'o, 's> {
    type Value = Descriptor;
    type Error = Cause;
    fn push(&mut self, frame: Continuation<Packed<'o, 's>, Descriptor>) -> Result<(), Cause> {
        if self.frames.len() >= self.limit || self.frames.len() == self.frames.capacity() {
            self.pending = Some(frame);
            return Err(Cause::Capacity);
        }
        self.frames.push(frame);
        #[cfg(test)]
        {
            self.peak = self.peak.max(self.frames.len());
            self.pushes += 1;
        }
        Ok(())
    }
    fn pop(&mut self) -> Option<Continuation<Packed<'o, 's>, Descriptor>> {
        self.frames.pop()
    }
    fn constant(&mut self, expr: NodeId) -> Descriptor {
        self.literal(expr)
    }
    fn boolean(&mut self, value: bool) -> Descriptor {
        Descriptor::Scalar(Scalar::Bool(value))
    }
    fn not(&mut self, value: &Descriptor) -> Result<Descriptor, Cause> {
        Ok(self.boolean(!truth(self, *value).get()))
    }
    fn neg(&mut self, value: &Descriptor) -> Result<Option<Descriptor>, Cause> {
        Ok(negative(*value))
    }
    fn binary(
        &mut self,
        op: BinOpKind,
        left: &Descriptor,
        right: &Descriptor,
    ) -> Result<Option<Descriptor>, Cause> {
        if matches!(op, BinOpKind::ScAnd | BinOpKind::ScOr) {
            let op = if matches!(op, BinOpKind::ScAnd) {
                scalar::Logical::And
            } else {
                scalar::Logical::Or
            };
            return Ok(Some(
                match scalar::logical(
                    op,
                    || truth(self, *left).get(),
                    || truth(self, *right).get(),
                ) {
                    scalar::Selection::Left => *left,
                    scalar::Selection::Right => *right,
                    scalar::Selection::False => Descriptor::Scalar(Scalar::Bool(false)),
                },
            ));
        }
        // Own the operation's actual operands before any checked recipe/count failure.
        self.operation = Some((*left, *right));
        let result = self.binary_materialized(op, *left, *right);
        if result.is_ok() {
            self.operation = None;
        }
        result
    }
    fn compare(
        &mut self,
        op: CompareOpKind,
        left: &Descriptor,
        right: &Descriptor,
    ) -> Result<Option<bool>, Cause> {
        self.operation = Some((*left, *right));
        let result = self.comparison(op, *left, *right);
        if result.is_ok() {
            self.operation = None;
        }
        result
    }
    fn list(&mut self, items: Sequence) -> Result<Descriptor, Cause> {
        self.collection = Some((items, None));
        let mut tail = items;
        let mut display_bytes = 2usize;
        let mut count = 0usize;
        while let Some(value) = self.leaf(&mut tail) {
            if count != 0 {
                display_bytes = display_bytes.checked_add(2).ok_or(Cause::Overflow)?;
            }
            display_bytes = display_bytes
                .checked_add(self.display_len(value, Style::Debug)?)
                .ok_or(Cause::Overflow)?;
            count = count.checked_add(1).ok_or(Cause::Overflow)?;
        }
        if count != items.len {
            return Err(Cause::Source);
        }
        let result = self.append(
            Operation::DirectList {
                items,
                display_bytes,
            },
            count,
        )?;
        self.collection = None;
        Ok(result)
    }
    fn map(&mut self, keys: Sequence, values: Sequence) -> Result<Descriptor, Cause> {
        self.collection = Some((keys, Some(values)));
        Err(Cause::NeedsMaterialization(Collection::Map))
    }
}

pub(super) fn project(value: Scalar) -> scalar::Scalar {
    match value {
        Scalar::None => scalar::Scalar::None,
        Scalar::Bool(x) => scalar::Scalar::Bool(x),
        Scalar::U64(x) => scalar::Scalar::U64(x),
        Scalar::U128(x) => scalar::Scalar::U128(x),
        Scalar::I64(x) => scalar::Scalar::I64(x),
        Scalar::I128(x) => scalar::Scalar::I128(x),
        Scalar::F64(x) => scalar::Scalar::F64(x),
    }
}
fn descriptor(value: scalar::Scalar) -> Descriptor {
    Descriptor::Scalar(match value {
        scalar::Scalar::None => Scalar::None,
        scalar::Scalar::Bool(x) => Scalar::Bool(x),
        scalar::Scalar::U64(x) => Scalar::U64(x),
        scalar::Scalar::U128(x) => Scalar::U128(x),
        scalar::Scalar::I64(x) => Scalar::I64(x),
        scalar::Scalar::I128(x) => Scalar::I128(x),
        scalar::Scalar::F64(x) => Scalar::F64(x),
    })
}
impl Workspace<'_, '_> {
    fn binary_materialized(
        &mut self,
        op: BinOpKind,
        left: Descriptor,
        right: Descriptor,
    ) -> Result<Option<Descriptor>, Cause> {
        if let Some(comparison) = match op {
            BinOpKind::Eq => Some(CompareOpKind::Eq),
            BinOpKind::Ne => Some(CompareOpKind::Ne),
            BinOpKind::Lt => Some(CompareOpKind::Lt),
            BinOpKind::Lte => Some(CompareOpKind::Lte),
            BinOpKind::Gt => Some(CompareOpKind::Gt),
            BinOpKind::Gte => Some(CompareOpKind::Gte),
            BinOpKind::In => Some(CompareOpKind::In),
            _ => None,
        } {
            return Ok(self
                .comparison(comparison, left, right)?
                .map(|x| Descriptor::Scalar(Scalar::Bool(x))));
        }
        if matches!(op, BinOpKind::Concat) {
            let left_bytes = self.display_len(left, Style::Display)?;
            let len = left_bytes
                .checked_add(self.display_len(right, Style::Display)?)
                .ok_or(Cause::Overflow)?;
            return self
                .append(
                    Operation::DisplayPair {
                        left,
                        right,
                        left_bytes,
                    },
                    len,
                )
                .map(Some);
        }
        if matches!(op, BinOpKind::Add) {
            match (left, right) {
                (Descriptor::List(_), Descriptor::List(_)) => {
                    return Err(Cause::NeedsMaterialization(Collection::List))
                }
                (Descriptor::Text(a), Descriptor::Text(b)) => {
                    let len = a.bytes.checked_add(b.bytes).ok_or(Cause::Overflow)?;
                    return self.append(Operation::Add(a, b), len).map(Some);
                }
                _ => (),
            }
        }
        if matches!(op, BinOpKind::Mul) {
            let pair = match (left, right) {
                (Descriptor::Text(a), b) => Some((a, b)),
                (a, Descriptor::Text(b)) => Some((b, a)),
                _ => None,
            };
            if let Some((a, n)) = pair {
                let n = match n {
                    Descriptor::Scalar(x) => text::fixed_usize(project(x)),
                    _ => None,
                };
                return match text::repeat_length(a.bytes, n) {
                    Ok((n, len)) => self.append(Operation::Repeat(a, n), len).map(Some),
                    Err(_) => Ok(None), // Actual ordinary operation error is swallowed by the shared fold.
                };
            }
            if matches!(left, Descriptor::List(_)) || matches!(right, Descriptor::List(_)) {
                return Err(Cause::NeedsMaterialization(Collection::List));
            }
        }
        if let (Descriptor::Scalar(a), Descriptor::Scalar(b)) = (left, right) {
            let arithmetic = match op {
                BinOpKind::Add => Arithmetic::Add,
                BinOpKind::Sub => Arithmetic::Sub,
                BinOpKind::Mul => Arithmetic::Mul,
                BinOpKind::Div => Arithmetic::Div,
                BinOpKind::FloorDiv => Arithmetic::FloorDiv,
                BinOpKind::Rem => Arithmetic::Rem,
                BinOpKind::Pow => Arithmetic::Pow,
                _ => unreachable!("shared operator selection"),
            };
            let (a, b) = (project(a), project(b));
            return Ok(
                scalar::arithmetic(arithmetic, a, b, scalar::fixed_conversion(a, b))
                    .ok()
                    .map(descriptor),
            );
        }
        // Closed scalar/text/direct-list types have no remaining dynamic conversion callbacks.
        Ok(None)
    }
    fn comparison(
        &self,
        op: CompareOpKind,
        a: Descriptor,
        b: Descriptor,
    ) -> Result<Option<bool>, Cause> {
        Ok(match op {
            CompareOpKind::Eq => Some(self.equal(a, b)?),
            CompareOpKind::Ne => Some(!self.equal(a, b)?),
            CompareOpKind::Lt => Some(self.compare_values(a, b)?.is_lt()),
            CompareOpKind::Lte => Some(!self.compare_values(a, b)?.is_gt()),
            CompareOpKind::Gt => Some(self.compare_values(a, b)?.is_gt()),
            CompareOpKind::Gte => Some(!self.compare_values(a, b)?.is_lt()),
            CompareOpKind::In => self.contains(b, a)?,
            CompareOpKind::NotIn => self.contains(b, a)?.map(|x| !x),
        })
    }
}
