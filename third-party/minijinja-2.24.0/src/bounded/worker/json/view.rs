//! Borrowed and generated values project into one formatter. Nested traversal
//! stores copied handles in its admitted continuation bank; input loans stay here.
use super::*;
use crate::bounded::input::record::ReadValue;
use serde_json::Value as J;
use crate::bounded::json_format::{self, Format, Key, Node, Projection, Sink, View};
use crate::bounded::source::MappingProjection;
pub(super) struct Values<'source, 'input, 'work> {
    pub source: &'source PreparedTemplate,
    pub context: RenderContext<'input>,
    pub borrowed: &'work Borrowed<'input>,
    pub values: &'work [Slot],
    pub prefix: Option<&'work [u8]>,
    pub atoms: Option<&'work [Atom]>,
}
fn invalid(_: Error) -> json_format::Error {
    json_format::Error::Write(std::fmt::Error)
}
fn bad() -> json_format::Error {
    json_format::Error::Write(std::fmt::Error)
}
impl<'input> Values<'_, 'input, '_> {
    fn pair(&self, range: Range, index: usize) -> Result<(Slot, Slot), json_format::Error> {
        if index >= range.len {
            return Err(bad());
        }
        let start = range
            .start
            .checked_add(index.checked_mul(2).ok_or(json_format::Error::Overflow)?)
            .ok_or(json_format::Error::Overflow)?;
        let values = self
            .values
            .get(start..start.checked_add(2).ok_or(json_format::Error::Overflow)?)
            .ok_or_else(bad)?;
        Ok((values[0], values[1]))
    }
    fn mapping(
        &self,
        register: usize,
        index: usize,
    ) -> Result<(&'input str, ReadValue<'input>), json_format::Error> {
        let map = self
            .borrowed
            .get(register)
            .map_err(invalid)?
            .as_object()
            .ok_or_else(bad)?;
        super::super::mapping::Pairs::new(map)
            .nth(index)
            .ok_or_else(bad)
    }
    fn item(&self, value: Slot, index: usize) -> Result<Node<'input, Slot>, json_format::Error> {
        Ok(match value {
            Slot::Sequence(range) => {
                if index >= range.len {
                    return Err(bad());
                }
                Node::Custom(
                    *self
                        .values
                        .get(
                            range
                                .start
                                .checked_add(index)
                                .ok_or(json_format::Error::Overflow)?,
                        )
                        .ok_or_else(bad)?,
                )
            }
            Slot::Messages => {
                if let Some(value) = self.context.messages().value(index) {
                    Node::from(value)
                } else if index < self.context.messages().len() {
                    Node::Custom(Slot::Message(index))
                } else {
                    return Err(bad());
                }
            }
            Slot::Range(value) => Node::Custom(value.item(index).map_err(invalid)?),
            Slot::Slice(view) => {
                return self.item(view.base.slot(), view.plan.index(index).ok_or_else(bad)?);
            }
            Slot::ObjectView(view) => {
                let (key, value) = self.pair(view.range, index)?;
                Node::Custom(match view.projection {
                    MappingProjection::Keys => key,
                    MappingProjection::Values => value,
                    MappingProjection::Items => Slot::Sequence(super::super::values::Value {
                        start: view
                            .range
                            .start
                            .checked_add(index.checked_mul(2).ok_or(json_format::Error::Overflow)?)
                            .ok_or(json_format::Error::Overflow)?,
                        len: 2,
                        depth: 0,
                    }),
                })
            }
            Slot::Mapping(view) => {
                let (key, value) = self.mapping(view.register, index)?;
                match view.projection {
                    MappingProjection::Keys => Node::Text(key),
                    MappingProjection::Values => Node::from(value),
                    MappingProjection::Items => Node::Custom(Slot::MappingPair(MappingPair {
                        register: view.register,
                        index,
                    })),
                }
            }
            Slot::MappingPair(pair) => {
                let (key, value) = self.mapping(pair.register, pair.index)?;
                match index {
                    0 => Node::Text(key),
                    1 => Node::from(value),
                    _ => return Err(bad()),
                }
            }
            Slot::Structured { register, .. } => Node::from(
                self.borrowed
                    .get(register)
                    .map_err(invalid)?
                    .as_array()
                    .and_then(|v| v.get(index))
                    .ok_or_else(bad)?,
            ),
            Slot::Split(view) => {
                let value = atom_text(
                    self.source,
                    self.context,
                    self.borrowed,
                    self.prefix,
                    view.value,
                )
                .map_err(invalid)?;
                let separator = view
                    .separator
                    .map(|r| r.text(&self.source.bytes).ok_or_else(bad))
                    .transpose()?;
                let part = crate::filters::split_parts_limit(value, separator, view.limit)
                    .nth(index)
                    .ok_or_else(bad)?;
                let start = (part.as_ptr() as usize)
                    .checked_sub(value.as_ptr() as usize)
                    .ok_or_else(bad)?;
                let end = start
                    .checked_add(part.len())
                    .ok_or(json_format::Error::Overflow)?;
                if value.get(start..end) != Some(part) {
                    return Err(bad());
                }
                Node::Custom(Slot::Text(Text::Atom(
                    slice_atom(view.value, start, part.len()).map_err(invalid)?,
                )))
            }
            _ => return Err(bad()),
        })
    }
}
impl<'input> View<'input, Slot> for Values<'_, 'input, '_> {
    fn project(&self, value: Slot) -> Result<Projection<'input>, json_format::Error> {
        let (object, length) = match value {
            Slot::Structured { register, .. } => {
                return Ok(Projection::from(
                    self.borrowed.get(register).map_err(invalid)?,
                ));
            }
            Slot::Message(index) => {
                if let Some(value) = self.context.messages().value(index) {
                    return Ok(Projection::from(value));
                }
                self.context.messages().get(index).ok_or_else(bad)?;
                (true, 2)
            }
            Slot::Object(range) => (true, range.len),
            Slot::Sequence(value) => (false, value.len),
            Slot::ObjectView(view) => (false, view.range.len),
            Slot::Messages => (false, self.context.messages().len()),
            Slot::EmptySequence => (false, 0),
            Slot::Range(value) => (false, value.plan.length),
            Slot::Slice(view) => (false, view.plan.length),
            Slot::Split(view) => (false, view.length),
            Slot::Mapping(view) => (false, view.length),
            Slot::MappingPair(_) => (false, 2),
            Slot::Text(_) | Slot::Undefined => return Ok(Projection::Scalar),
            value if scalar(value).is_some() => return Ok(Projection::Scalar),
            _ => return Err(bad()),
        };
        Ok(Projection::Container { object, length })
    }
    fn entry(
        &self,
        value: Slot,
        index: usize,
    ) -> Result<(Option<Key<'input, Slot>>, Node<'input, Slot>), json_format::Error> {
        match value {
            Slot::Object(range) => {
                let (key, value) = self.pair(range, index)?;
                Ok((Some(Key::Custom(key)), Node::Custom(value)))
            }
            Slot::Message(message) => {
                let (key, value) = match index {
                    0 => ("role", Atom::Role(message)),
                    1 => ("content", Atom::Content(message)),
                    _ => return Err(bad()),
                };
                Ok((
                    Some(Key::Text(key)),
                    Node::Custom(Slot::Text(Text::Atom(value))),
                ))
            }
            _ => Ok((None, self.item(value, index)?)),
        }
    }
    fn key(&self, value: Slot) -> Result<&str, json_format::Error> {
        match value {
            Slot::Text(Text::Atom(atom)) => {
                atom_text(self.source, self.context, self.borrowed, self.prefix, atom)
                    .map_err(invalid)
            }
            _ => Err(bad()),
        }
    }
    fn scalar<W: Sink + ?Sized>(
        &self,
        format: &Format<'_>,
        output: &mut W,
        value: Slot,
    ) -> Result<(), json_format::Error> {
        match value {
            Slot::Text(text) => super::consumer::write_generated_text(
                format,
                output,
                text,
                self.source,
                self.context,
                self.borrowed,
                self.prefix,
                self.atoms,
            ),
            Slot::Undefined => format.write_scalar(output, &J::Null),
            other => format.write_scalar(
                output,
                &super::consumer::number(scalar(other).ok_or_else(bad)?).map_err(invalid)?,
            ),
        }
    }
}
pub(super) fn control_bytes() -> Option<usize> {
    use std::mem::{size_of, size_of_val};
    let parts = [
        size_of::<Values<'_, '_, '_>>(),
        size_of::<(Range, usize, usize)>(),
        size_of::<[Slot; 3]>(),
        size_of::<(Slot, usize)>(),
        size_of::<(Slot, usize)>(),
        size_of::<Node<'_, Slot>>(),
        size_of::<Key<'_, Slot>>(),
        size_of::<Projection<'_>>(),
        size_of::<Result<Projection<'_>, json_format::Error>>(),
        size_of::<Result<Node<'_, Slot>, json_format::Error>>(),
        size_of::<Result<(Option<Key<'_, Slot>>, Node<'_, Slot>), json_format::Error>>(),
        size_of::<Result<(&str, ReadValue<'_>), json_format::Error>>(),
        size_of::<super::super::mapping::Pairs<'_>>(),
        size_of::<crate::filters::SplitParts<'_>>(),
        size_of::<(&str, Option<&str>, &str, usize, usize)>(),
        size_of::<Result<(Slot, Slot), json_format::Error>>(),
    ];
    parts
        .into_iter()
        .try_fold(size_of_val(&parts), usize::checked_add)
}
