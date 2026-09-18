//! Borrowed plain-join inputs and the existing paid text/atom destinations.
//! Generated output carries its measured metadata inline: measurement never
//! reads a not-yet-allocated destination or retains a temporary input register.
use super::text_segments::Segments;
use super::{generated::Output, measured::Measured, *};
use std::fmt::{self, Write};

// Only the actual closed input population is enumerable here. Object callbacks,
// nested value formatting, and generated-text iteration remain separate profiles.
enum Items<'a> {
    Empty,
    Slots {
        values: std::slice::Iter<'a, Slot>,
        segments: Segments<'a>,
    },
    Range(primitive::range::Iter),
    Text(std::str::Chars<'a>),
    Split(crate::filters::SplitParts<'a>),
    #[cfg(feature = "json")]
    Json(crate::bounded::input::record::InputArrayIter<'a>),
    #[cfg(feature = "json")]
    MapKeys(mapping::Pairs<'a>),
    #[cfg(feature = "json")]
    MapValues(mapping::Pairs<'a>),
}
enum Item<'a> {
    Text(&'a str),
    Generated(Text, Segments<'a>),
    Character(char),
    Scalar(Scalar),
    Invalid,
}
impl<'a> Iterator for Items<'a> {
    type Item = Item<'a>;
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Empty => None,
            Self::Slots { values, segments } => values.next().map(|value| match value {
                Slot::Text(text) => Item::Generated(*text, *segments),
                Slot::Undefined => Item::Text(""),
                value => scalar(*value).map_or(Item::Invalid, Item::Scalar),
            }),
            Self::Range(values) => values
                .next()
                .map(|value| Item::Scalar(Scalar::I64(value as i64))),
            Self::Text(chars) => chars.next().map(Item::Character),
            Self::Split(parts) => parts.next().map(Item::Text),
            #[cfg(feature = "json")]
            Self::Json(items) => items.next().map(json_item),
            #[cfg(feature = "json")]
            Self::MapKeys(items) => items.next().map(|(key, _)| Item::Text(key)),
            #[cfg(feature = "json")]
            Self::MapValues(items) => items.next().map(|(_, value)| json_item(value)),
        }
    }
}
#[cfg(feature = "json")]
fn json_item(value: crate::bounded::input::record::ReadValue<'_>) -> Item<'_> {
    use crate::bounded::input::record::ReadKind as J;
    match value.kind() {
        J::Null => Item::Scalar(Scalar::None),
        J::Bool(value) => Item::Scalar(Scalar::Bool(value)),
        J::Text(value) => Item::Text(value),
        J::Number(value) => {
            if let Some(value) = value.as_u64() {
                Item::Scalar(Scalar::U64(value))
            } else if let Some(value) = value.as_i64() {
                Item::Scalar(Scalar::I64(value))
            } else if let Some(value) = value.as_f64() {
                Item::Scalar(Scalar::F64(value))
            } else {
                Item::Invalid
            }
        }
        J::Array(_) | J::Object(_) => Item::Invalid,
    }
}
fn write_item(output: &mut Output<'_>, item: Item<'_>) -> fmt::Result {
    match item {
        Item::Text(text) => output.write_str(text),
        Item::Generated(text, segments) => segments
            .visit(text, |text| {
                output.write_str(text).map_err(|_| Error::Geometry)
            })
            .map_err(|cause| {
                if output.error.is_none() {
                    output.error = Some(cause);
                }
                fmt::Error
            }),
        Item::Character(character) => output.write_char(character),
        Item::Scalar(value) => primitive::text::write_scalar(output, value, false),
        Item::Invalid => {
            output.error = Some(Error::Geometry);
            Err(fmt::Error)
        }
    }
}
impl Engine<'_, '_, '_> {
    pub(super) fn apply_join(&mut self, arguments: Option<u16>) -> Result<(), Error> {
        let arguments = arguments
            .filter(|n| (1..=2).contains(n))
            .ok_or(Error::Geometry)?;
        let separator = if arguments == 2 {
            let value = self.pop()?;
            self.text_atom(value)?
        } else {
            Atom::Source(Range::new(0, 0))
        };
        let value = self.pop()?;
        let value = if matches!(value, Slot::Text(_)) {
            Slot::Text(Text::Atom(self.text_atom(value)?))
        } else {
            value
        };
        let needed = super::super::render::TextCapacity {
            atoms: self.counts.concat,
            bytes: self.counts.context_text,
        };
        if self.workspace.concat.as_ref().map_or(0, |v| v.len()) < needed.atoms
            || self.workspace.context_text.as_ref().map_or(0, |v| v.len()) < needed.bytes
        {
            return Err(Error::TextCapacity(needed));
        }
        let (start, text_start) = self.generated_start()?;
        let (prefix, destination) = match self.workspace.context_text.as_deref_mut() {
            Some(bytes) => {
                let (prefix, tail) = bytes.split_at_mut(text_start);
                (Some(&*prefix), Some(tail))
            }
            None => (None, None),
        };
        let segments = Segments {
            source: self.source,
            context: self.context,
            borrowed: self.borrowed,
            prefix,
            atoms: self.workspace.concat.as_deref(),
        };
        let items = match value {
            Slot::Sequence(range) => Items::Slots {
                values: self
                    .workspace
                    .values
                    .get(range.start..range.start.checked_add(range.len).ok_or(Error::Overflow)?)
                    .ok_or(Error::Geometry)?
                    .iter(),
                segments,
            },
            Slot::Undefined | Slot::EmptySequence | Slot::Scalar(Scalar::None) => Items::Empty,
            Slot::Range(value) => Items::Range(value.plan.iter()),
            Slot::Text(Text::Atom(atom)) => Items::Text(
                atom_text(self.source, self.context, &self.borrowed, prefix, atom)?.chars(),
            ),
            Slot::Split(view) => Items::Split(crate::filters::split_parts_limit(
                atom_text(
                    self.source,
                    self.context,
                    &self.borrowed,
                    prefix,
                    view.value,
                )?,
                view.separator
                    .map(|range| range.text(&self.source.bytes).ok_or(Error::Geometry))
                    .transpose()?,
                view.limit,
            )),
            #[cfg(feature = "json")]
            Slot::Mapping(view) => match view.projection {
                crate::bounded::source::MappingProjection::Keys => {
                    Items::MapKeys(mapping::Pairs::new(
                        self.borrowed
                            .get(view.register)?
                            .as_object()
                            .ok_or(Error::Geometry)?,
                    ))
                }
                crate::bounded::source::MappingProjection::Values => {
                    Items::MapValues(mapping::Pairs::new(
                        self.borrowed
                            .get(view.register)?
                            .as_object()
                            .ok_or(Error::Geometry)?,
                    ))
                }
                crate::bounded::source::MappingProjection::Items => return Err(Error::Geometry),
            },
            #[cfg(feature = "json")]
            Slot::Structured { register, .. } => Items::Json(
                self.borrowed
                    .get(register)?
                    .as_array()
                    .ok_or(Error::Geometry)?
                    .iter(),
            ),
            _ => return Err(Error::Geometry),
        };
        let separator = atom_text(self.source, self.context, &self.borrowed, prefix, separator)?;
        let mut output = Output {
            bytes: destination,
            start: 0,
            measured: Measured::default(),
            error: None,
        };
        // This is the same ordered separator/format loop as ordinary join_plain.
        let formatter: fn(&mut Output<'_>, Item<'_>) -> fmt::Result = write_item;
        primitive::join::plain(&mut output, items, separator, formatter)
            .map_err(|_| output.error.unwrap_or(Error::Geometry))?;
        let measured = output.finish().map_err(|cause| match cause {
            Error::TextCapacity(next) => text_start
                .checked_add(next.bytes)
                .map(|bytes| {
                    Error::TextCapacity(super::super::render::TextCapacity {
                        atoms: needed.atoms,
                        bytes,
                    })
                })
                .unwrap_or(Error::Overflow),
            other => other,
        })?;
        self.finish_generated(start, text_start, measured)
    }
}
pub(super) fn control_bytes() -> Option<usize> {
    use std::mem::{size_of, size_of_val};
    let controls = [
        size_of::<Option<u16>>(),
        size_of::<Segments<'_>>(),
        size_of::<super::super::render::TextCapacity>(),
        size_of::<(Option<&[u8]>, Option<&mut [u8]>)>(),
        size_of::<std::slice::Iter<'_, Slot>>(),
        size_of::<u16>(),
        size_of::<[Slot; 2]>(),
        size_of::<(Atom, &str)>(),
        size_of::<[usize; 6]>(),
        size_of::<(Item<'_>, Option<Item<'_>>, char)>(),
        size_of::<(Option<u64>, Option<i64>, Option<f64>)>(),
        size_of::<(&mut Output<'_>, Item<'_>)>(),
    ];
    let total = controls
        .into_iter()
        .try_fold(size_of_val(&controls), usize::checked_add)?;
    let total = total
        .checked_add(super::text_segments::control_bytes()?)?
        .checked_add(primitive::text::scalar_control_bytes::<Output<'_>>()?)?
        .checked_add(primitive::join::control_bytes::<
            Output<'_>,
            Items<'_>,
            fn(&mut Output<'_>, Item<'_>) -> fmt::Result,
        >()?)?;
    #[cfg(feature = "json")]
    let total = total
        .checked_add(crate::bounded::input::RecordValue::control_bytes()?)?
        .checked_add(size_of::<(&serde_json::Value, Option<&serde_json::Value>)>())?;
    Some(total)
}
