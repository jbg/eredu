//! Flat backward recipes and two payload destinations; no callback or external Value enters this owner.
#![forbid(unsafe_code)]
use super::{Cause, Descriptor, List, MaterializationRequirements, Scalar, Text, Value, Workspace};
use crate::bounded::expression::store::{Joined, NodeId, RecordKind, Scalar as Literal, Sequence};
use crate::compiler::fold::View;
use std::alloc::Layout;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) struct RecipeId(pub(super) usize);
#[derive(Clone, Copy)]
pub(super) enum TextKind {
    Source(Joined),
    Generated(RecipeId),
}
#[derive(Clone, Copy)]
pub(super) struct TextDescriptor {
    pub(super) kind: TextKind,
    pub(super) bytes: usize,
}
impl From<Joined> for TextDescriptor {
    fn from(value: Joined) -> Self {
        Self {
            kind: TextKind::Source(value),
            bytes: value.bytes,
        }
    }
}
#[derive(Clone, Copy)]
pub(super) enum Operation {
    Add(TextDescriptor, TextDescriptor),
    Repeat(TextDescriptor, usize),
    DisplayPair {
        left: Descriptor,
        right: Descriptor,
        left_bytes: usize,
    },
    DirectList {
        items: Sequence,
        display_bytes: usize,
    },
}
#[derive(Clone, Copy)]
pub(super) struct Recipe {
    pub(super) operation: Operation,
    pub(super) start: usize,
    pub(super) len: usize,
}
impl<'o, 's: 'o> Workspace<'o, 's> {
    pub(super) fn literal(&self, expr: NodeId) -> Descriptor {
        match &self.source.storage.expressions.nodes[expr.0].kind {
            RecordKind::Const(value) => match value {
                Literal::None => Descriptor::Scalar(Scalar::None),
                Literal::Bool(x) => Descriptor::Scalar(Scalar::Bool(*x)),
                Literal::Int(x) => Descriptor::Scalar(Scalar::U64(*x)),
                Literal::Int128(x) => Descriptor::Scalar(Scalar::U128(*x)),
                Literal::Float(x) => Descriptor::Scalar(Scalar::F64(*x)),
                Literal::Text(x) => Descriptor::Text((*x).into()),
            },
            _ => unreachable!("shared immediate literal selection"),
        }
    }
    pub(super) fn list_recipe(&self, id: RecipeId) -> (Sequence, usize, usize) {
        let recipe = self.recipes[id.0];
        match recipe.operation {
            Operation::DirectList {
                items,
                display_bytes,
            } => (items, recipe.start, display_bytes),
            _ => unreachable!("private list recipe"),
        }
    }
    pub(super) fn leaf(&self, items: &mut Sequence) -> Option<Descriptor> {
        let (expr, tail) = super::Packed(self.source).next_item(*items)?;
        *items = tail;
        Some(self.literal(expr))
    }
    pub(super) fn append(&mut self, operation: Operation, len: usize) -> Result<Descriptor, Cause> {
        let list = matches!(operation, Operation::DirectList { .. });
        let start = if list {
            self.payload.items
        } else {
            self.payload.bytes
        };
        let recipe = Recipe {
            operation,
            start,
            len,
        };
        self.pending_recipe = Some(recipe);
        if self.recipes.len() >= self.requirements.recipes
            || self.recipes.len() == self.recipes.capacity()
        {
            return Err(Cause::Capacity);
        }
        let end = start.checked_add(len).ok_or(Cause::Overflow)?;
        if list && end > self.requirements.operands {
            return Err(Cause::Capacity);
        }
        let (bytes, items) = if list {
            (self.payload.bytes, end)
        } else {
            (end, self.payload.items)
        };
        // Both layouts and their sum are checked before changing the committed geometry.
        let requested = payload_layout(bytes, items)?;
        let id = RecipeId(self.recipes.len());
        let earlier = |value: Descriptor| match value {
            Descriptor::Text(TextDescriptor {
                kind: TextKind::Generated(child),
                ..
            })
            | Descriptor::List(child) => child.0 < id.0,
            _ => true,
        };
        let valid = match operation {
            Operation::Add(a, b) => earlier(Descriptor::Text(a)) && earlier(Descriptor::Text(b)),
            Operation::Repeat(a, _) => earlier(Descriptor::Text(a)),
            Operation::DisplayPair { left, right, .. } => earlier(left) && earlier(right),
            Operation::DirectList { items, .. } => items.len == len,
        };
        if !valid {
            return Err(Cause::Source);
        }
        self.recipes.push(recipe);
        self.pending_recipe = None;
        self.payload = MaterializationRequirements {
            bytes,
            items,
            requested,
        };
        Ok(if list {
            Descriptor::List(id)
        } else {
            Descriptor::Text(TextDescriptor {
                kind: TextKind::Generated(id),
                bytes: len,
            })
        })
    }
    pub(super) fn fill(&mut self) -> Result<(), Cause> {
        if self.bytes.capacity() < self.payload.bytes || self.items.capacity() < self.payload.items
        {
            return Err(Cause::Capacity);
        }
        for index in 0..self.recipes.len() {
            let recipe = self.recipes[index];
            match recipe.operation {
                Operation::DirectList { mut items, .. } => {
                    if self.items.len() != recipe.start {
                        return Err(Cause::Source);
                    }
                    let end = recipe
                        .start
                        .checked_add(recipe.len)
                        .ok_or(Cause::Overflow)?;
                    while let Some(value) = self.leaf(&mut items) {
                        if self.items.len() >= end || self.items.len() == self.items.capacity() {
                            return Err(Cause::Capacity);
                        }
                        self.items.push(value);
                    }
                    if self.items.len() != end {
                        return Err(Cause::Source);
                    }
                }
                _ => {
                    if self.bytes.len() != recipe.start {
                        return Err(Cause::Source);
                    }
                    let text = Descriptor::Text(TextDescriptor {
                        kind: TextKind::Generated(RecipeId(index)),
                        bytes: recipe.len,
                    });
                    for offset in 0..recipe.len {
                        let byte = self
                            .byte(text, super::reader::Style::Display, offset)?
                            .ok_or(Cause::Source)?;
                        if self.bytes.len() == self.bytes.capacity() {
                            return Err(Cause::Capacity);
                        }
                        self.bytes.push(byte);
                    }
                    // Exact source/format/repeat byte construction must preserve UTF-8; validate the whole range once.
                    let end = recipe
                        .start
                        .checked_add(recipe.len)
                        .ok_or(Cause::Overflow)?;
                    std::str::from_utf8(self.bytes.get(recipe.start..end).ok_or(Cause::Source)?)
                        .map_err(|_| Cause::Source)?;
                }
            }
        }
        Ok(())
    }
    pub(super) fn value_view(&self, value: Descriptor) -> Value<'_, 's> {
        match value {
            Descriptor::Scalar(x) => Value::Scalar(x),
            Descriptor::Text(text) => {
                let generated = match text.kind {
                    TextKind::Source(_) => &[][..],
                    TextKind::Generated(id) => {
                        let recipe = self.recipes[id.0];
                        &self.bytes[recipe.start..recipe.start + recipe.len]
                    }
                };
                Value::Text(Text {
                    source: self.source,
                    text,
                    generated,
                })
            }
            Descriptor::List(id) => {
                let (items, start, _) = self.list_recipe(id);
                Value::List(List {
                    source: self.source,
                    items: &self.items[start..start + items.len],
                })
            }
        }
    }
}
pub(super) fn payload_layout(bytes: usize, items: usize) -> Result<usize, Cause> {
    Layout::array::<u8>(bytes)
        .map_err(|_| Cause::Overflow)?
        .size()
        .checked_add(
            Layout::array::<Descriptor>(items)
                .map_err(|_| Cause::Overflow)?
                .size(),
        )
        .ok_or(Cause::Overflow)
}
