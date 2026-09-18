//! The registered chat filter's Python JSON spelling, shared by ordinary and
//! counted writers. This module supplies no host-memory or source authority.
use serde_json::Value;
use std::{
    collections::TryReserveError,
    fmt::{self, Write},
    mem::{size_of, size_of_val},
};
mod storage;
pub use storage::{Capacity, Scratch};
mod scalar;
mod view;
pub(crate) use view::{Key, Node, Projection, View};

/// Exact accepted option order of the registered filter.
pub const OPTION_NAMES: [&str; 4] = crate::bounded::source::JsonOptions::NAMES;

/// Borrowed indentation, or the actual integer-width space sequence.
#[derive(Clone, Copy, Debug)]
pub enum Indent<'a> {
    /// Repeat the caller's string once per nesting level.
    Text(&'a str),
    /// Repeat this exact number of spaces once per level.
    Spaces(usize),
}
/// Actual options after the registered filter's argument conversion.
#[derive(Clone, Copy, Debug)]
pub struct Format<'a> {
    /// Escape non-ASCII scalars using lower-case UTF-16 JSON escapes.
    pub ensure_ascii: bool,
    /// Order object keys lexically.
    pub sort_keys: bool,
    /// A newline and this indentation at nonempty container boundaries.
    pub indent: Option<Indent<'a>>,
    /// Exact separator between array members or object entries.
    pub item_separator: &'a str,
    /// Exact separator between object key and value.
    pub key_separator: &'a str,
}

/// A writer can count a repeated segment without constructing or visiting each
/// byte. The ordinary default writes the same bytes in order.
pub trait Sink: Write {
    /// Write the exact immutable repeated string.
    fn repeat(&mut self, text: &str, count: usize) -> fmt::Result {
        for _ in 0..count {
            self.write_str(text)?;
        }
        Ok(())
    }
}
impl Sink for String {}

/// Fixed traversal/output errors; allocation failures preserve their source.
#[derive(Debug)]
pub enum Error {
    /// The exact next reached container needs these continuation/key slots.
    Capacity(Capacity),
    /// Ordinary or prepaid scratch allocation failed.
    Reserve(TryReserveError),
    /// Checked slot/depth arithmetic overflowed.
    Overflow,
    /// The supplied output refused a write.
    Write(fmt::Error),
}
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Capacity(_) => "JSON traversal needs more counted storage",
            Self::Reserve(_) => "JSON traversal storage allocation failed",
            Self::Overflow => "JSON traversal layout overflow",
            Self::Write(_) => "JSON output write failed",
        })
    }
}
impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Reserve(e) => Some(e),
            Self::Write(e) => Some(e),
            _ => None,
        }
    }
}
impl From<fmt::Error> for Error {
    fn from(value: fmt::Error) -> Self {
        Self::Write(value)
    }
}

impl Format<'_> {
    pub(crate) fn newline(
        &self,
        output: &mut (impl Sink + ?Sized),
        depth: usize,
    ) -> Result<(), Error> {
        if let Some(indent) = self.indent {
            output.write_char('\n')?;
            match indent {
                Indent::Text(text) => output.repeat(text, depth)?,
                Indent::Spaces(width) => {
                    output.repeat(" ", width.checked_mul(depth).ok_or(Error::Overflow)?)?
                }
            }
        }
        Ok(())
    }
    /// Write a string using the same escaping as every object key/value.
    pub fn write_text(&self, output: &mut (impl Sink + ?Sized), value: &str) -> Result<(), Error> {
        scalar::string(output, value, self.ensure_ascii).map_err(Error::Write)
    }
    /// Write one immutable UTF-8 segment inside an already opened JSON string.
    /// Segment boundaries do not alter escaping; callers retain both quotes.
    pub(crate) fn write_text_segment(
        &self,
        output: &mut (impl Sink + ?Sized),
        value: &str,
    ) -> Result<(), Error> {
        scalar::string_content(output, value, self.ensure_ascii).map_err(Error::Write)
    }
    /// Write one already-converted JSON scalar without a traversal lifetime.
    pub fn write_scalar(
        &self,
        output: &mut (impl Sink + ?Sized),
        value: &Value,
    ) -> Result<(), Error> {
        scalar::write(output, value, self.ensure_ascii).map_err(Error::Write)
    }
    /// Exact two-string record used by the existing typed role/content view.
    pub fn write_text_pair_object(
        &self,
        output: &mut (impl Sink + ?Sized),
        mut fields: [(&str, &str); 2],
        depth: usize,
    ) -> Result<(), Error> {
        if self.sort_keys && fields[0].0 > fields[1].0 {
            fields.swap(0, 1);
        }
        output.write_char('{')?;
        for (index, (key, value)) in fields.into_iter().enumerate() {
            if index != 0 {
                output.write_str(self.item_separator)?;
            }
            self.newline(output, depth.checked_add(1).ok_or(Error::Overflow)?)?;
            self.write_text(output, key)?;
            output.write_str(self.key_separator)?;
            self.write_text(output, value)?;
        }
        self.newline(output, depth)?;
        output.write_char('}')?;
        Ok(())
    }
    /// Stream the same closed JSON tree through explicit continuation storage.
    /// Ordinary callers use growable scratch; admitted callers pass exact fixed
    /// capacities and handle Capacity before requesting larger storage.
    pub fn write<'a, W: Sink + ?Sized>(
        &self,
        output: &mut W,
        value: &'a Value,
        scratch: &mut Scratch<'a>,
    ) -> Result<(), Error> {
        scratch.clear();
        self.write_inner(output, Some(value), scratch)
    }
    /// Borrow an existing JSON slice as a root array without constructing a Value.
    pub fn write_array<'a, W: Sink + ?Sized>(
        &self,
        output: &mut W,
        values: &'a [Value],
        scratch: &mut Scratch<'a>,
    ) -> Result<(), Error> {
        scratch.clear();
        if values.is_empty() {
            output.write_str("[]")?;
            return Ok(());
        }
        scratch.push_array(values)?;
        output.write_char('[')?;
        self.write_inner(output, None, scratch)
    }

    /// Borrow a closed indexed view through the ordinary formatter and its
    /// actual fixed continuation/key destinations.
    pub(crate) fn write_view<'a, V: Copy, W: Sink + ?Sized>(
        &self,
        output: &mut W,
        value: Node<'a, V>,
        scratch: &mut Scratch<'a, V>,
        view: &impl View<'a, V>,
    ) -> Result<(), Error> {
        scratch.clear();
        self.write_inner_view(output, Some(value), scratch, view)
    }
    fn write_inner<'a, W: Sink + ?Sized>(
        &self,
        output: &mut W,
        next: Option<&'a Value>,
        scratch: &mut Scratch<'a>,
    ) -> Result<(), Error> {
        self.write_inner_view(output, next.map(Node::Json), scratch, &view::Closed)
    }
    fn write_inner_view<'a, V: Copy, W: Sink + ?Sized>(
        &self,
        output: &mut W,
        mut next: Option<Node<'a, V>>,
        scratch: &mut Scratch<'a, V>,
        view: &impl View<'a, V>,
    ) -> Result<(), Error> {
        loop {
            if let Some(mut value) = next.take() {
                if let Node::Custom(custom) = value {
                    match view.project(custom)? {
                        Projection::Json(json) => value = Node::Json(json),
                        Projection::JsonArray(values) => value = Node::JsonArray(values),
                        Projection::Record(record) => value = Node::Record(record),
                        Projection::Scalar => view.scalar(self, output, custom)?,
                        Projection::Container { object, length } => {
                            if length == 0 {
                                output.write_str(if object { "{}" } else { "[]" })?;
                            } else {
                                scratch.push_view(custom, object, length, self.sort_keys, view)?;
                                output.write_char(if object { '{' } else { '[' })?;
                            }
                        }
                    }
                }
                match value {
                    Node::JsonArray(values) => {
                        if values.is_empty() {
                            output.write_str("[]")?;
                        } else {
                            scratch.push_array(values)?;
                            output.write_char('[')?;
                        }
                    }
                    Node::Json(Value::Array(values)) if values.is_empty() => {
                        output.write_str("[]")?
                    }
                    Node::Json(Value::Object(values)) if values.is_empty() => {
                        output.write_str("{}")?
                    }
                    Node::Json(json @ (Value::Array(_) | Value::Object(_))) => {
                        scratch.push(json, self.sort_keys)?;
                        output.write_char(if json.is_array() { '[' } else { '{' })?;
                    }
                    Node::Json(json) => scalar::write(output, json, self.ensure_ascii)?,
                    Node::Record(record) => match record {
                        crate::bounded::input::RecordValue::Null => output.write_str("null")?,
                        crate::bounded::input::RecordValue::Bool(value) => {
                            output.write_str(if value { "true" } else { "false" })?
                        }
                        crate::bounded::input::RecordValue::Text(value) => {
                            self.write_text(output, value)?
                        }
                        record => {
                            let object = record.is_object();
                            if record.length() == Some(0) {
                                output.write_str(if object { "{}" } else { "[]" })?;
                            } else {
                                scratch.push_record(record, self.sort_keys)?;
                                output.write_char(if object { '{' } else { '[' })?;
                            }
                        }
                    },
                    Node::Text(text) => self.write_text(output, text)?,
                    Node::Custom(_) => {}
                }
            }
            match scratch.step(view)? {
                storage::Step::Done => return Ok(()),
                storage::Step::Entry {
                    key,
                    value,
                    first,
                    depth,
                } => {
                    if !first {
                        output.write_str(self.item_separator)?;
                    }
                    self.newline(output, depth)?;
                    if let Some(key) = key {
                        self.write_text(output, key.text(view)?)?;
                        output.write_str(self.key_separator)?;
                    }
                    next = Some(value);
                }
                storage::Step::Close { delimiter, depth } => {
                    self.newline(output, depth)?;
                    output.write_char(delimiter)?;
                }
            }
        }
    }
}

/// Fixed formatter arguments, loop/selection/escape and standard formatting
/// controls. Scratch heap slots and output bytes are reported separately.
pub fn control_bytes<W: Sink>() -> Option<usize> {
    view_control_bytes::<W, ()>()
}
pub(crate) fn view_control_bytes<W: Sink, V: Copy>() -> Option<usize> {
    let parts = [
        size_of::<Key<'_, V>>(),
        size_of::<Projection<'_>>(),
        size_of::<Format<'_>>(),
        size_of::<Indent<'_>>(),
        size_of::<Error>(),
        size_of::<Result<(), Error>>(),
        size_of::<(&mut W, &Value, &mut Scratch<'_, V>)>(),
        size_of::<Option<Node<'_, V>>>(),
        size_of::<storage::Step<'_, V>>(),
        size_of::<(bool, usize, Option<&str>)>(),
        size_of::<fmt::Arguments<'_>>(),
        size_of::<fmt::Result>(),
        size_of::<(&str, usize)>(),
        size_of::<[(&str, &str); 2]>(),
        size_of::<std::iter::Enumerate<std::array::IntoIter<(&str, &str), 2>>>(),
        scalar::control_bytes::<W>()?,
        storage::control_bytes::<V>()?,
    ];
    parts
        .into_iter()
        .try_fold(size_of_val(&parts), usize::checked_add)
}
