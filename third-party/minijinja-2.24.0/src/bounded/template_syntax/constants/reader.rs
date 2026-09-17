//! Fixed byte-at-offset reader. Every generated edge moves to an earlier recipe; no return stack.
#![forbid(unsafe_code)]
use super::{
    materialize::{Operation, TextKind},
    storage::project,
    Cause, Descriptor, Scalar, Segments, Workspace,
};
use crate::value::{
    primitive::{scalar, text},
    ValueKind,
};
use std::{
    cmp::Ordering,
    fmt::{self, Write},
    mem::size_of,
};
#[derive(Clone, Copy)]
pub(super) enum Style {
    Display,
    Debug,
}
#[derive(Default)]
struct Count {
    length: usize,
}
impl Write for Count {
    fn write_str(&mut self, text: &str) -> fmt::Result {
        self.length = self.length.checked_add(text.len()).ok_or(fmt::Error)?;
        Ok(())
    }
}
struct Byte {
    offset: usize,
    result: Option<u8>,
}
impl Write for Byte {
    fn write_str(&mut self, text: &str) -> fmt::Result {
        if self.result.is_none() {
            if self.offset < text.len() {
                self.result = Some(text.as_bytes()[self.offset]);
            } else {
                self.offset -= text.len();
            }
        }
        Ok(())
    }
}
/// Selected named local shells only; not a complete Reader, call frame or producer stack bound.
pub(super) fn control_bytes() -> usize {
    // Two concurrent readers for comparisons/containment, plus current pure-format sinks.
    2 * (size_of::<Descriptor>() + size_of::<Style>() + size_of::<usize>())
        + size_of::<Count>()
        + size_of::<Byte>()
        + size_of::<fmt::Arguments<'static>>()
        + size_of::<fmt::Result>()
}
impl Workspace<'_, '_> {
    fn write_leaf(&self, out: &mut impl Write, value: Descriptor, style: Style) -> fmt::Result {
        match value {
            Descriptor::Scalar(x) => {
                text::write_scalar(out, project(x), matches!(style, Style::Debug))
            }
            Descriptor::Text(value) => {
                let TextKind::Source(joined) = value.kind else {
                    unreachable!("immediate literal text")
                };
                let mut segments = Segments {
                    source: self.source,
                    next: joined.sequence.head,
                    generated: None,
                };
                if matches!(style, Style::Debug) {
                    text::write_debug_text(out, segments)
                } else {
                    segments.try_for_each(|text| out.write_str(text))
                }
            }
            Descriptor::List(_) => unreachable!("immediate scalar/text leaf"),
        }
    }
    pub(super) fn display_len(&self, value: Descriptor, style: Style) -> Result<usize, Cause> {
        match value {
            Descriptor::Text(text) if matches!(style, Style::Display) => Ok(text.bytes),
            Descriptor::List(id) => Ok(self.list_recipe(id).2),
            _ => {
                let mut count = Count::default();
                self.write_leaf(&mut count, value, style)
                    .map_err(|_| Cause::Overflow)?;
                Ok(count.length)
            }
        }
    }
    pub(super) fn byte(
        &self,
        mut value: Descriptor,
        mut style: Style,
        mut offset: usize,
    ) -> Result<Option<u8>, Cause> {
        // Each loop transition either terminates at a literal, or selects one strictly earlier recipe.
        // Lists select an immediate literal, never a nested list or generated text.
        loop {
            match value {
                Descriptor::Scalar(_) => {
                    let mut output = Byte {
                        offset,
                        result: None,
                    };
                    self.write_leaf(&mut output, value, style)
                        .map_err(|_| Cause::Overflow)?;
                    return Ok(output.result);
                }
                Descriptor::Text(text) => {
                    if matches!(style, Style::Debug) {
                        let mut output = Byte {
                            offset,
                            result: None,
                        };
                        self.write_leaf(&mut output, value, style)
                            .map_err(|_| Cause::Overflow)?;
                        return Ok(output.result);
                    }
                    if offset >= text.bytes {
                        return Ok(None);
                    }
                    match text.kind {
                        TextKind::Source(joined) => {
                            for segment in (Segments {
                                source: self.source,
                                next: joined.sequence.head,
                                generated: None,
                            }) {
                                if offset < segment.len() {
                                    return Ok(Some(segment.as_bytes()[offset]));
                                }
                                offset -= segment.len();
                            }
                            return Err(Cause::Source);
                        }
                        TextKind::Generated(id) => match self.recipes[id.0].operation {
                            Operation::Add(left, right) => {
                                value = Descriptor::Text(if offset < left.bytes {
                                    left
                                } else {
                                    offset -= left.bytes;
                                    right
                                });
                            }
                            Operation::Repeat(text, count) => {
                                // Out-of-range/empty text returned above. Never iterate count, including usize::MAX empty repeats.
                                if text.bytes == 0 || count == 0 {
                                    return Err(Cause::Source);
                                }
                                offset %= text.bytes;
                                value = Descriptor::Text(text);
                            }
                            Operation::DisplayPair {
                                left,
                                right,
                                left_bytes,
                            } => {
                                value = if offset < left_bytes {
                                    left
                                } else {
                                    offset -= left_bytes;
                                    right
                                };
                                style = Style::Display;
                            }
                            Operation::DirectList { .. } => return Err(Cause::Source),
                        },
                    }
                }
                Descriptor::List(id) => {
                    let (mut items, _, length) = self.list_recipe(id);
                    if offset >= length {
                        return Ok(None);
                    }
                    if offset == 0 {
                        return Ok(Some(b'['));
                    }
                    offset -= 1;
                    let mut first = true;
                    let mut selected = None;
                    while let Some(leaf) = self.leaf(&mut items) {
                        if !first {
                            if offset < 2 {
                                return Ok(Some(if offset == 0 { b',' } else { b' ' }));
                            }
                            offset -= 2;
                        }
                        first = false;
                        let length = self.display_len(leaf, Style::Debug)?;
                        if offset < length {
                            selected = Some(leaf);
                            break;
                        }
                        offset -= length;
                    }
                    if let Some(leaf) = selected {
                        value = leaf;
                        style = Style::Debug;
                    } else {
                        return if offset == 0 {
                            Ok(Some(b']'))
                        } else {
                            Err(Cause::Source)
                        };
                    }
                }
            }
        }
    }
    fn bytes_cmp(&self, a: Descriptor, b: Descriptor) -> Result<Ordering, Cause> {
        let na = self.display_len(a, Style::Display)?;
        let nb = self.display_len(b, Style::Display)?;
        for offset in 0..na.min(nb) {
            let order =
                self.byte(a, Style::Display, offset)?
                    .cmp(&self.byte(b, Style::Display, offset)?);
            if order != Ordering::Equal {
                return Ok(order);
            }
        }
        Ok(na.cmp(&nb))
    }
    fn equal_leaf(&self, a: Descriptor, b: Descriptor) -> Result<bool, Cause> {
        Ok(match (a, b) {
            (Descriptor::Scalar(a), Descriptor::Scalar(b)) => {
                let (a, b) = (project(a), project(b));
                scalar::equal(a, b, scalar::fixed_conversion(a, b))
            }
            (Descriptor::Text(_), Descriptor::Text(_)) => self.bytes_cmp(a, b)?.is_eq(),
            _ => false,
        })
    }
    pub(super) fn equal(&self, a: Descriptor, b: Descriptor) -> Result<bool, Cause> {
        if let (Descriptor::List(a), Descriptor::List(b)) = (a, b) {
            if a == b {
                return Ok(true);
            }
            let (mut a, _, _) = self.list_recipe(a);
            let (mut b, _, _) = self.list_recipe(b);
            loop {
                match (self.leaf(&mut a), self.leaf(&mut b)) {
                    (None, None) => return Ok(true),
                    (Some(a), Some(b)) if self.equal_leaf(a, b)? => (),
                    _ => return Ok(false),
                }
            }
        }
        self.equal_leaf(a, b)
    }
    fn compare_leaf(&self, a: Descriptor, b: Descriptor) -> Result<Ordering, Cause> {
        if let (Descriptor::Scalar(a), Descriptor::Scalar(b)) = (a, b) {
            let (a, b) = (project(a), project(b));
            return Ok(scalar::compare(a, b, scalar::fixed_conversion(a, b)));
        }
        let order = kind(a).cmp(&kind(b));
        if !order.is_eq() {
            return Ok(order);
        }
        self.bytes_cmp(a, b)
    }
    pub(super) fn compare_values(&self, a: Descriptor, b: Descriptor) -> Result<Ordering, Cause> {
        if let (Descriptor::List(a), Descriptor::List(b)) = (a, b) {
            if a == b {
                return Ok(Ordering::Equal);
            }
            let (mut a, _, _) = self.list_recipe(a);
            let (mut b, _, _) = self.list_recipe(b);
            loop {
                match (self.leaf(&mut a), self.leaf(&mut b)) {
                    (None, None) => return Ok(Ordering::Equal),
                    (None, Some(_)) => return Ok(Ordering::Less),
                    (Some(_), None) => return Ok(Ordering::Greater),
                    (Some(a), Some(b)) => {
                        let order = self.compare_leaf(a, b)?;
                        if !order.is_eq() {
                            return Ok(order);
                        }
                    }
                }
            }
        }
        self.compare_leaf(a, b)
    }
    pub(super) fn contains(
        &self,
        container: Descriptor,
        needle: Descriptor,
    ) -> Result<Option<bool>, Cause> {
        match container {
            Descriptor::Scalar(_) => Ok(None),
            Descriptor::List(id) => {
                let (mut items, _, _) = self.list_recipe(id);
                while let Some(value) = self.leaf(&mut items) {
                    if self.equal_leaf(value, needle)? {
                        return Ok(Some(true));
                    }
                }
                Ok(Some(false))
            }
            Descriptor::Text(text) => {
                let n = self.display_len(needle, Style::Display)?;
                if n > text.bytes {
                    return Ok(Some(false));
                }
                for start in 0..=text.bytes - n {
                    let mut matches = true;
                    for offset in 0..n {
                        if self.byte(container, Style::Display, start + offset)?
                            != self.byte(needle, Style::Display, offset)?
                        {
                            matches = false;
                            break;
                        }
                    }
                    if matches {
                        return Ok(Some(true));
                    }
                }
                Ok(Some(false))
            }
        }
    }
}
fn kind(value: Descriptor) -> ValueKind {
    match value {
        Descriptor::Scalar(Scalar::None) => ValueKind::None,
        Descriptor::Scalar(Scalar::Bool(_)) => ValueKind::Bool,
        Descriptor::Scalar(_) => ValueKind::Number,
        Descriptor::Text(_) => ValueKind::String,
        Descriptor::List(_) => ValueKind::Seq,
    }
}
