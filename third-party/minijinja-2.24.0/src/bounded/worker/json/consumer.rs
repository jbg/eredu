//! Shared registered JSON formatting; no Value construction for borrowed trees.
use super::super::{generated::Output, measured::Measured};
use super::*;
use crate::bounded::json_format::{self, Format, Indent};
use serde_json::Value as J;
use std::{
    fmt::Write,
    mem::{size_of, size_of_val},
};
fn text<'a>(
    source: &'a PreparedTemplate,
    context: RenderContext<'a>,
    borrowed: &Borrowed<'a>,
    prefix: Option<&'a [u8]>,
    value: Slot,
) -> Result<&'a str, Error> {
    let Slot::Text(Text::Atom(atom)) = value else {
        return Err(Error::Geometry);
    };
    atom_text(source, context, borrowed, prefix, atom)
}
pub(super) fn number(value: Scalar) -> Result<J, Error> {
    Ok(match value {
        Scalar::None => J::Null,
        Scalar::Bool(v) => J::Bool(v),
        Scalar::I64(v) => J::Number(v.into()),
        Scalar::U64(v) => J::Number(v.into()),
        Scalar::I128(v) => {
            if let Ok(v) = i64::try_from(v) {
                J::Number(v.into())
            } else {
                J::Number(u64::try_from(v).map_err(|_| Error::Geometry)?.into())
            }
        }
        Scalar::U128(v) => J::Number(u64::try_from(v).map_err(|_| Error::Geometry)?.into()),
        Scalar::F64(v) => serde_json::Number::from_f64(v).map_or(J::Null, J::Number),
    })
}
fn failed(cause: json_format::Error, output: &Output<'_>) -> Error {
    match cause {
        json_format::Error::Capacity(next) => Error::JsonCapacity(next),
        json_format::Error::Overflow => Error::Overflow,
        _ => output.error.unwrap_or(Error::Geometry),
    }
}
impl Engine<'_, '_, '_> {
    pub(in crate::bounded::worker) fn apply_json(
        &mut self,
        arguments: Option<u16>,
    ) -> Result<(), Error> {
        let options = self.json_options.take().ok_or(Error::Geometry)?;
        if arguments != Some(u16::from(options.arguments) + 1) || options.arguments > 5 {
            return Err(Error::Geometry);
        }
        let mut values = [Slot::Undefined; 5];
        for index in (0..usize::from(options.arguments)).rev() {
            values[index] = self.pop()?;
        }
        let value = self.pop()?;
        // All generated inputs are immutable prefixes. A fixed attempt must
        // fund those bytes and atom descriptors before any consumer reads them.
        let needed = super::super::super::render::TextCapacity {
            atoms: self.counts.concat,
            bytes: self.counts.context_text,
        };
        if self.workspace.concat.as_ref().map_or(0, |v| v.len()) < needed.atoms
            || self.workspace.context_text.as_ref().map_or(0, |v| v.len()) < needed.bytes
        {
            return Err(Error::TextCapacity(needed));
        }
        let option = |index: usize| -> Result<Slot, Error> {
            let position = options.order[index];
            if position == u8::MAX {
                Ok(Slot::Undefined)
            } else {
                values
                    .get(usize::from(position))
                    .copied()
                    .ok_or(Error::Geometry)
            }
        };
        let ascii = option(0)?;
        let indent = option(1)?;
        let separators = option(2)?;
        let sort = option(3)?;
        let ensure_ascii = self.truth(&ascii)?;
        let sort_keys = self.truth(&sort)?;
        let (start, text_start) = self.generated_start()?;
        let (prefix, destination) = match self.workspace.context_text.as_deref_mut() {
            Some(bytes) => {
                let (prefix, tail) = bytes.split_at_mut(text_start);
                (Some(&*prefix), Some(tail))
            }
            None => (None, None),
        };
        let indent = match indent {
            Slot::Undefined | Slot::Scalar(Scalar::None) => None,
            Slot::Text(_) => Some(Indent::Text(text(
                self.source,
                self.context,
                self.borrowed,
                prefix,
                indent,
            )?)),
            other => Some(Indent::Spaces(
                match scalar(other).ok_or(Error::Geometry)? {
                    Scalar::Bool(value) => usize::from(value),
                    Scalar::I64(value) => {
                        usize::try_from(value.max(0)).map_err(|_| Error::Overflow)?
                    }
                    Scalar::U64(value) if value <= i64::MAX as u64 => {
                        usize::try_from(value).map_err(|_| Error::Overflow)?
                    }
                    Scalar::I128(value) if i64::try_from(value).is_ok() => {
                        usize::try_from(value.max(0)).map_err(|_| Error::Overflow)?
                    }
                    Scalar::U128(value) if value <= i64::MAX as u128 => {
                        usize::try_from(value).map_err(|_| Error::Overflow)?
                    }
                    _ => return Err(Error::Geometry),
                },
            )),
        };
        let (item_separator, key_separator) = if options.separator_pair {
            let second = values
                .get(usize::from(options.order[2]) + 1)
                .copied()
                .ok_or(Error::Geometry)?;
            (
                text(self.source, self.context, self.borrowed, prefix, separators)?,
                text(self.source, self.context, self.borrowed, prefix, second)?,
            )
        } else {
            match separators {
                Slot::Undefined | Slot::Scalar(Scalar::None) => {
                    (if indent.is_some() { "," } else { ", " }, ": ")
                }
                Slot::Sequence(range) if range.len == 2 => {
                    let values = self
                        .workspace
                        .values
                        .get(range.start..range.start.checked_add(2).ok_or(Error::Overflow)?)
                        .ok_or(Error::Geometry)?;
                    (
                        text(self.source, self.context, self.borrowed, prefix, values[0])?,
                        text(self.source, self.context, self.borrowed, prefix, values[1])?,
                    )
                }
                Slot::Structured { register, .. } => {
                    let Some(values) = self.borrowed.get(register)?.as_array() else {
                        return Err(Error::Geometry);
                    };
                    if values.len() != 2 {
                        return Err(Error::Geometry);
                    }
                    let first = values.get(0).ok_or(Error::Geometry)?;
                    let second = values.get(1).ok_or(Error::Geometry)?;
                    (
                        first.as_str().ok_or(Error::Geometry)?,
                        second.as_str().ok_or(Error::Geometry)?,
                    )
                }
                Slot::Text(_) => {
                    let value = text(self.source, self.context, self.borrowed, prefix, separators)?;
                    let mut chars = value.char_indices();
                    chars.next().ok_or(Error::Geometry)?;
                    let (at, _) = chars.next().ok_or(Error::Geometry)?;
                    if chars.next().is_some() {
                        return Err(Error::Geometry);
                    }
                    (&value[..at], &value[at..])
                }
                _ => return Err(Error::Geometry),
            }
        };
        let format = Format {
            ensure_ascii,
            sort_keys,
            indent,
            item_separator,
            key_separator,
        };
        let mut output = Output {
            bytes: destination,
            start: 0,
            measured: Measured::default(),
            error: None,
        };
        let view = super::view::Values {
            source: self.source,
            context: self.context,
            borrowed: self.borrowed,
            values: self.workspace.values,
            prefix,
            atoms: self.workspace.concat.as_deref(),
        };
        let result = format.write_view(
            &mut output,
            json_format::Node::Custom(value),
            self.json,
            &view,
        );
        result.map_err(|cause| failed(cause, &output))?;
        let measured = output.finish().map_err(|cause| match cause {
            Error::TextCapacity(next) => text_start
                .checked_add(next.bytes)
                .map(|bytes| {
                    Error::TextCapacity(super::super::super::render::TextCapacity {
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
pub(super) fn write_generated_text<W: json_format::Sink + ?Sized>(
    format: &Format<'_>,
    output: &mut W,
    value: Text,
    source: &PreparedTemplate,
    context: RenderContext<'_>,
    borrowed: &Borrowed<'_>,
    prefix: Option<&[u8]>,
    atoms: Option<&[Atom]>,
) -> Result<(), json_format::Error> {
    let invalid = |_| json_format::Error::Write(std::fmt::Error);
    output.write_char('"')?;
    let segments = super::super::text_segments::Segments {
        source,
        context,
        borrowed,
        prefix,
        atoms,
    };
    segments
        .visit(value, |segment| {
            format
                .write_text_segment(output, segment)
                .map_err(|_| Error::Geometry)
        })
        .map_err(invalid)?;
    output.write_char('"')?;
    Ok(())
}
pub(super) fn control_bytes() -> Option<usize> {
    let parts = [
        size_of::<(Option<&[u8]>, Option<&mut [u8]>)>(),
        size_of::<super::super::super::render::TextCapacity>(),
        size_of::<(Text, Atom, usize, usize, usize)>(),
        size_of::<std::slice::Iter<'_, Atom>>(),
        size_of::<(
            &Format<'_>,
            &mut Output<'_>,
            Text,
            &PreparedTemplate,
            RenderContext<'_>,
            &Borrowed<'_>,
            Option<&[u8]>,
            Option<&[Atom]>,
        )>(),
        size_of::<[Slot; 5]>(),
        size_of::<[Slot; 5]>(),
        size_of::<crate::bounded::source::JsonOptions>(),
        size_of::<(Option<u16>, usize, usize)>(),
        size_of::<[&str; 4]>(),
        size_of::<Option<Indent<'_>>>(),
        size_of::<J>(),
        size_of::<Result<J, Error>>(),
        size_of::<std::str::CharIndices<'_>>(),
        size_of::<Option<(usize, char)>>(),
        size_of::<Output<'_>>(),
        size_of::<Measured>(),
        size_of::<crate::bounded::input::Messages<'_>>(),
        size_of::<crate::bounded::input::TextMessage<'_>>(),
        size_of::<std::ops::Range<usize>>(),
        size_of::<Result<(), json_format::Error>>(),
        json_format::view_control_bytes::<Output<'_>, Slot>()?,
        super::view::control_bytes()?,
        super::super::text_segments::control_bytes()?,
    ];
    parts
        .into_iter()
        .try_fold(size_of_val(&parts), usize::checked_add)
}
