//! Concrete storage for the same instruction dispatcher as the ordinary VM.
use crate::vm::shared::{self, Next, Op, Storage};

use super::input::{ContextValue, RenderContext, VariableKey};
use super::source::{Instruction, PreparedTemplate, Range};
use crate::value::primitive::{self, scalar::Scalar};
mod capture;
mod case;
#[cfg(feature = "chat-clock")]
mod clock;
mod collection;
mod default;
mod dictsort;
mod equality;
mod generated;
mod join;
pub(super) mod json;
mod length;
mod map_filter;
mod measured;
mod membership;
mod methods;
mod object;
mod ordering;
mod range;
mod replace;
mod selection;
mod slice;
mod string;
mod string_views;
mod text_segments;
mod values;
use string_views::SplitView;
mod mapping;
use mapping::{MappingPair, MappingView};
mod structured;
pub(super) use structured::Borrowed;
mod loops;
pub(super) mod macros;
use macros::{CallFrame, MacroValue};
pub(super) mod namespace;
use namespace::{Namespace, NamespaceField};

#[derive(Clone, Copy, Debug)]
pub(super) enum Atom {
    Source(Range),
    Variable(VariableKey, Range),
    Borrowed(usize, Range),
    Owned(Range),
    Role(usize),
    Content(usize),
    RoleSlice(usize, Range),
    ContentSlice(usize, Range),
}
#[derive(Clone, Copy, Debug)]
pub(super) enum Text {
    Atom(Atom),
    Concat {
        start: usize,
        count: usize,
        bytes: usize,
        skip: usize,
        leading: usize,
        trailing: usize,
        characters: usize,
        leading_characters: usize,
        trailing_characters: usize,
    },
}
#[derive(Clone, Copy, Debug)]
pub(super) enum Slot {
    Undefined,
    Bool(bool),
    Zero,
    Scalar(Scalar),
    EmptySequence,
    Sequence(values::Value),
    Object(Range),
    ObjectView(object::View),
    Range(range::Value),
    Slice(slice::View),
    Namespace(usize),
    Kwargs(usize),
    Macro(MacroValue),
    #[cfg(feature = "chat-clock")]
    Clock,
    Closure(Option<usize>),
    Arguments {
        start: u32,
        count: usize,
    },
    Split(SplitView),
    Mapping(MappingView),
    MappingPair(MappingPair),
    Structured {
        register: usize,
        length: usize,
    },
    Text(Text),
    Messages,
    Message(usize),
    Loop(usize),
}
impl Default for Slot {
    fn default() -> Self {
        Self::Undefined
    }
}
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct Local {
    name: Option<Range>,
    value: Slot,
}
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct Frame {
    input: Slot,
    length: usize,
    next: usize,
    current: Option<usize>,
    locals_start: usize,
    locals_end: usize,
    iterate: u32,
    remaining: usize,
    owns_closure: bool,
    closure_len: usize,
    visible_loop: bool,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Error {
    Geometry,
    UnknownFunction,
    Overflow,
    JsonCapacity(super::render::JsonCapacity),
    TextCapacity(super::render::TextCapacity),
    ValueCapacity(super::render::ValueCapacity),
}
#[derive(Clone, Copy, Debug)]
pub(super) struct Failure {
    pub kind: Error,
    pub instruction: u32,
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct Counts {
    pub concat: usize,
    pub values: usize,
    pub output: usize,
    pub context_text: usize,
}

pub(super) struct Workspace<'a> {
    pub operands: &'a mut [Slot],
    pub values: &'a mut [Slot],
    pub value_register_start: usize,
    pub frames: &'a mut [Frame],
    pub locals: &'a mut [Local],
    pub namespaces: &'a mut [Namespace],
    pub namespace_fields: &'a mut [NamespaceField],
    pub calls: &'a mut [CallFrame],
    pub arguments: &'a mut [Slot],
    pub closure_fields: &'a mut [Local],
    // Measurement counts the same append operations without allocating them.
    pub concat: Option<&'a mut [Atom]>,
    pub output: Option<&'a mut [u8]>,
    pub context_text: Option<&'a mut [u8]>,
}
struct Engine<'source, 'input, 'work> {
    source: &'source PreparedTemplate,
    context: RenderContext<'input>,
    generation: bool,
    borrowed: &'work mut Borrowed<'input>,
    json: &'work mut json::Scratch<'input>,
    json_options: Option<super::source::JsonOptions>,
    workspace: Workspace<'work>,
    depth: usize,
    loops: usize,
    locals: usize,
    counts: Counts,
    remaining_instructions: usize,
    calls: usize,
    capture: Option<usize>,
    collection: Option<usize>,
    closure_active: bool,
    closure_len: usize,
}
fn atom_text<'a>(
    source: &'a PreparedTemplate,
    context: RenderContext<'a>,
    borrowed: &Borrowed<'a>,
    context_text: Option<&'a [u8]>,
    atom: Atom,
) -> Result<&'a str, Error> {
    match atom {
        Atom::Borrowed(index, range) => range
            .text(borrowed.text(index).ok_or(Error::Geometry)?.as_bytes())
            .ok_or(Error::Geometry),
        Atom::Owned(range) => range
            .text(context_text.unwrap_or(&[]))
            .ok_or(Error::Geometry),
        Atom::Source(range) => range.text(&source.bytes).ok_or(Error::Geometry),
        Atom::Variable(key, range) => range
            .text(context.text(key).ok_or(Error::Geometry)?.as_bytes())
            .ok_or(Error::Geometry),
        Atom::Role(index) => Ok(context.messages().get(index).ok_or(Error::Geometry)?.role),
        Atom::Content(index) => Ok(context
            .messages()
            .get(index)
            .ok_or(Error::Geometry)?
            .content),
        Atom::RoleSlice(index, range) => range
            .text(
                context
                    .messages()
                    .get(index)
                    .ok_or(Error::Geometry)?
                    .role
                    .as_bytes(),
            )
            .ok_or(Error::Geometry),
        Atom::ContentSlice(index, range) => range
            .text(
                context
                    .messages()
                    .get(index)
                    .ok_or(Error::Geometry)?
                    .content
                    .as_bytes(),
            )
            .ok_or(Error::Geometry),
    }
}
fn slice_atom(atom: Atom, start: usize, bytes: usize) -> Result<Atom, Error> {
    let range = |offset: usize| -> Result<Range, Error> {
        Ok(Range::new(
            offset.checked_add(start).ok_or(Error::Overflow)?,
            bytes,
        ))
    };
    Ok(match atom {
        Atom::Source(original) => Atom::Source(range(original.start)?),
        Atom::Borrowed(index, original) => Atom::Borrowed(index, range(original.start)?),
        Atom::Owned(original) => Atom::Owned(range(original.start)?),
        Atom::Variable(key, original) => Atom::Variable(key, range(original.start)?),
        Atom::Role(index) => Atom::RoleSlice(index, range(0)?),
        Atom::Content(index) => Atom::ContentSlice(index, range(0)?),
        Atom::RoleSlice(index, original) => Atom::RoleSlice(index, range(original.start)?),
        Atom::ContentSlice(index, original) => Atom::ContentSlice(index, range(original.start)?),
    })
}
// Clip one borrowed atom against a concatenation's retained byte window. Keep
// empty atoms too, so the exact same finite slot count is known before H.
fn clip_atom(
    source: &PreparedTemplate,
    context: RenderContext<'_>,
    borrowed: &Borrowed<'_>,
    context_text: Option<&[u8]>,
    atom: Atom,
    skip: &mut usize,
    remaining: &mut usize,
) -> Result<Atom, Error> {
    let text = atom_text(source, context, borrowed, context_text, atom)?;
    let start = (*skip).min(text.len());
    *skip -= start;
    let bytes = (text.len() - start).min(*remaining);
    text.get(start..start + bytes).ok_or(Error::Geometry)?;
    *remaining -= bytes;
    slice_atom(atom, start, bytes)
}
fn scalar(value: Slot) -> Option<Scalar> {
    Some(match value {
        Slot::Bool(value) => Scalar::Bool(value),
        Slot::Zero => Scalar::U64(0),
        Slot::Scalar(value) => value,
        _ => return None,
    })
}
fn scalar_truth(value: Scalar) -> bool {
    use primitive::Truth;
    match value {
        Scalar::None => Truth::False,
        Scalar::Bool(v) => Truth::Bool(v),
        Scalar::U64(v) => Truth::U64(v),
        Scalar::U128(v) => Truth::U128(v),
        Scalar::I64(v) => Truth::I64(v),
        Scalar::I128(v) => Truth::I128(v),
        Scalar::F64(v) => Truth::F64(v),
    }
    .get()
}
struct ScalarOutput<'a> {
    output: Option<&'a mut [u8]>,
    cursor: &'a mut usize,
    error: Option<Error>,
}
impl std::fmt::Write for ScalarOutput<'_> {
    fn write_str(&mut self, text: &str) -> std::fmt::Result {
        let result: Result<(), Error> = (|| {
            let end = self.cursor.checked_add(text.len()).ok_or(Error::Overflow)?;
            if let Some(output) = self.output.as_deref_mut() {
                output
                    .get_mut(*self.cursor..end)
                    .ok_or(Error::Geometry)?
                    .copy_from_slice(text.as_bytes());
            }
            *self.cursor = end;
            Ok(())
        })();
        result.map_err(|error| {
            self.error = Some(error);
            std::fmt::Error
        })
    }
}
impl Engine<'_, '_, '_> {
    fn emit_scalar(&mut self, value: Scalar) -> Result<(), Error> {
        let mut writer = ScalarOutput {
            output: self.workspace.output.as_deref_mut(),
            cursor: &mut self.counts.output,
            error: None,
        };
        primitive::text::write_scalar(&mut writer, value, false)
            .map_err(|_| writer.error.unwrap_or(Error::Geometry))
    }
    fn emit_text(&mut self, text: &str) -> Result<(), Error> {
        use std::fmt::Write;
        let mut writer = ScalarOutput {
            output: self.workspace.output.as_deref_mut(),
            cursor: &mut self.counts.output,
            error: None,
        };
        writer
            .write_str(text)
            .map_err(|_| writer.error.unwrap_or(Error::Geometry))
    }
    fn parts(&self, text: Text) -> Result<(usize, usize), Error> {
        match text {
            Text::Atom(Atom::Owned(range)) => Ok((1, range.len)),
            Text::Atom(atom) => Ok((
                1,
                atom_text(
                    self.source,
                    self.context,
                    &self.borrowed,
                    self.workspace.context_text.as_deref(),
                    atom,
                )?
                .len(),
            )),
            Text::Concat { count, bytes, .. } => Ok((count, bytes)),
        }
    }
    fn edges(&self, text: Text) -> Result<(usize, usize, usize), Error> {
        match text {
            Text::Atom(atom) => {
                let text = atom_text(
                    self.source,
                    self.context,
                    &self.borrowed,
                    self.workspace.context_text.as_deref(),
                    atom,
                )?;
                let kept = crate::filters::whitespace_trim_range(text);
                Ok((text.len(), kept.start, text.len() - kept.end))
            }
            Text::Concat {
                bytes,
                leading,
                trailing,
                ..
            } => Ok((bytes, leading, trailing)),
        }
    }
    fn append(&mut self, text: Text) -> Result<(), Error> {
        let text = match text {
            Text::Atom(atom) => Text::Atom(self.own_atom(atom)?),
            other => other,
        };
        // Borrowed text length remains known during measurement after snapshot.
        let count = self.parts(text)?.0;
        let end = self
            .counts
            .concat
            .checked_add(count)
            .ok_or(Error::Overflow)?;
        if let Some(storage) = self.workspace.concat.as_deref_mut() {
            if end > storage.len() {
                return Err(Error::TextCapacity(super::render::TextCapacity {
                    atoms: end,
                    bytes: self.counts.context_text,
                }));
            }
            match text {
                Text::Atom(atom) => storage[self.counts.concat] = atom,
                Text::Concat {
                    start,
                    count,
                    bytes,
                    mut skip,
                    ..
                } => {
                    let previous_end = start.checked_add(count).ok_or(Error::Overflow)?;
                    if previous_end > self.counts.concat {
                        return Err(Error::Geometry);
                    }
                    let mut remaining = bytes;
                    for offset in 0..count {
                        let atom = clip_atom(
                            self.source,
                            self.context,
                            &self.borrowed,
                            self.workspace.context_text.as_deref(),
                            storage[start + offset],
                            &mut skip,
                            &mut remaining,
                        )?;
                        storage[self.counts.concat + offset] = atom;
                    }
                    if skip != 0 || remaining != 0 {
                        return Err(Error::Geometry);
                    }
                }
            }
        }
        self.counts.concat = end;
        Ok(())
    }
    fn emit_atom(&mut self, atom: Atom) -> Result<(), Error> {
        let text = atom_text(
            self.source,
            self.context,
            &self.borrowed,
            self.workspace.context_text.as_deref(),
            atom,
        )?;
        let end = self
            .counts
            .output
            .checked_add(text.len())
            .ok_or(Error::Overflow)?;
        if let Some(output) = self.workspace.output.as_deref_mut() {
            output
                .get_mut(self.counts.output..end)
                .ok_or(Error::Geometry)?
                .copy_from_slice(text.as_bytes());
        }
        self.counts.output = end;
        Ok(())
    }
}
impl<'source> Storage<'source> for Engine<'source, '_, '_> {
    type Value = Slot;
    type Error = Error;
    fn pop(&mut self) -> Result<Slot, Error> {
        self.depth = self.depth.checked_sub(1).ok_or(Error::Geometry)?;
        let slot = *self
            .workspace
            .operands
            .get(self.depth)
            .ok_or(Error::Geometry)?;
        self.workspace.operands[self.depth] = Slot::Undefined;
        Ok(slot)
    }
    fn peek(&self) -> Result<&Slot, Error> {
        self.workspace
            .operands
            .get(self.depth.checked_sub(1).ok_or(Error::Geometry)?)
            .ok_or(Error::Geometry)
    }
    fn push(&mut self, value: Slot) -> Result<(), Error> {
        let value = self.borrowed.push(value, self.depth)?;
        *self
            .workspace
            .operands
            .get_mut(self.depth)
            .ok_or(Error::Geometry)? = value;
        self.depth = self.depth.checked_add(1).ok_or(Error::Overflow)?;
        Ok(())
    }
    fn lookup(&mut self, name: &'source str) -> Result<Slot, Error> {
        for local in self
            .workspace
            .locals
            .get(self.call_locals_start()..self.locals)
            .ok_or(Error::Geometry)?
            .iter()
            .rev()
        {
            if local.name.and_then(|range| range.text(&self.source.bytes)) == Some(name) {
                return Ok(local.value);
            }
        }
        if name == "loop" {
            if let Some(index) = (self.call_loops_start()..self.loops)
                .rev()
                .find(|index| self.workspace.frames[*index].visible_loop)
            {
                return Ok(Slot::Loop(index));
            }
        }
        if let Some(closure) = self
            .calls
            .checked_sub(1)
            .and_then(|i| self.workspace.calls[i].closure)
        {
            if let Some(value) = self.macro_capture(closure, name)? {
                return Ok(value);
            }
        }
        Ok(match self.context.lookup(name, self.generation) {
            ContextValue::Text(key, text) => {
                Slot::Text(Text::Atom(Atom::Variable(key, Range::new(0, text.len()))))
            }
            ContextValue::Scalar(value) => Slot::Scalar(value),
            ContextValue::Messages => Slot::Messages,
            ContextValue::EmptySequence => Slot::EmptySequence,
            #[cfg(feature = "chat-clock")]
            ContextValue::Clock => Slot::Clock,
            #[cfg(feature = "json")]
            ContextValue::Structured(length, value) => self.structured(length, value.into()),
            #[cfg(feature = "json")]
            ContextValue::Records(length, value) => self.structured(length, value),
            ContextValue::Undefined => Slot::Undefined,
            ContextValue::Unsupported => return Err(Error::Geometry),
        })
    }

    fn store(&mut self, name: &'source str, value: Slot) -> Result<(), Error> {
        self.store_scoped_local(name, value)
    }
    fn set_attr(&mut self, object: Slot, name: &'source str, value: Slot) -> Result<(), Error> {
        let Slot::Namespace(index) = object else {
            return Err(Error::Geometry);
        };
        self.namespace_set(index, name, value)
    }
    fn build_kwargs(&mut self, count: usize) -> Result<(), Error> {
        self.namespace_keywords(count)
    }
    fn call_function(
        &mut self,
        name: &'source str,
        arguments: Option<u16>,
        function: Option<Slot>,
        pc: u32,
    ) -> Result<Option<u32>, Error> {
        if function.is_some() {
            return Err(Error::Geometry);
        }
        let function = self.lookup(name)?;
        if let Slot::Macro(value) = function {
            return self.call_macro(value, arguments, pc).map(Some);
        }
        #[cfg(feature = "chat-clock")]
        if matches!(function, Slot::Clock) {
            self.clock_call(arguments)?;
            return Ok(None);
        }
        // Known but unimplemented builtins retain their explicit geometry
        // refusal. Only an actually absent callable is an ordinary nonmatch.
        // dict and strftime_now are registered by this closed text profile;
        // super is a contextual VM callable and cannot be classified here.
        if matches!(function, Slot::Undefined)
            && !crate::defaults::is_builtin_global(name)
            && !matches!(name, "dict" | "strftime_now" | "super")
        {
            return Err(Error::UnknownFunction);
        }
        if name == "range" {
            if !matches!(function, Slot::Undefined) {
                return Err(Error::Geometry);
            }
            self.range_call(arguments)?;
        } else {
            self.namespace_call(name, arguments)?;
        }
        Ok(None)
    }
    fn duplicate(&mut self) -> Result<(), Error> {
        let value = *self.peek()?;
        self.push(value)
    }
    fn enclose(&mut self, name: &'source str) -> Result<(), Error> {
        self.enclose_macro(name)
    }
    fn closure(&mut self) -> Result<Slot, Error> {
        Ok(Slot::Closure(self.current_macro_closure()?))
    }
    fn build_macro(&mut self, name: &'source str, offset: u32, flags: u8) -> Result<(), Error> {
        self.create_macro(name, offset, flags)
    }
    fn return_macro(&mut self) -> Result<Option<u32>, Error> {
        self.finish_macro().map(Some)
    }
    fn unpack(&mut self, count: usize) -> Result<(), Error> {
        self.unpack_loop_value(count)
    }
    fn attr(&mut self, value: Slot, name: &str) -> Result<Slot, Error> {
        if let Slot::Namespace(index) = value {
            return self.namespace_attr(index, name);
        }
        if let Slot::Macro(value) = value {
            return self.macro_attr(value, name);
        }
        if let Slot::Message(index) = value {
            if self.context.messages().get(index).is_none() {
                return Err(Error::Geometry);
            }
            return match name {
                "role" => Ok(Slot::Text(Text::Atom(Atom::Role(index)))),
                "content" => Ok(Slot::Text(Text::Atom(Atom::Content(index)))),
                _ => Ok(Slot::Undefined),
            };
        }
        if let Slot::Loop(index) = value {
            return self.loop_attr_at(index, name);
        }
        self.structured_attr(value, name)
    }
    fn slice(&mut self, value: Slot, start: Slot, stop: Slot, step: Slot) -> Result<Slot, Error> {
        self.slice_value(value, start, stop, step)
    }
    fn item(&mut self, value: Slot, key: Slot) -> Result<Slot, Error> {
        if let Slot::Namespace(index) = value {
            let Slot::Text(Text::Atom(atom)) = key else {
                return Ok(Slot::Undefined);
            };
            let key = atom_text(
                self.source,
                self.context,
                &self.borrowed,
                self.workspace.context_text.as_deref(),
                atom,
            )?;
            return self.namespace_attr(index, key);
        }
        match (value, key) {
            (Slot::Message(index), Slot::Text(Text::Atom(atom))) => {
                if self.context.messages().get(index).is_none() {
                    return Err(Error::Geometry);
                }
                match atom_text(
                    self.source,
                    self.context,
                    &self.borrowed,
                    self.workspace.context_text.as_deref(),
                    atom,
                )? {
                    "role" => Ok(Slot::Text(Text::Atom(Atom::Role(index)))),
                    "content" => Ok(Slot::Text(Text::Atom(Atom::Content(index)))),
                    _ => Ok(Slot::Undefined),
                }
            }
            (Slot::Message(_), Slot::Text(Text::Concat { .. })) => Err(Error::Geometry),
            (value, key) => self.selected_item(value, key),
        }
    }
    fn call_method(&mut self, name: &'source str, arguments: Option<u16>) -> Result<(), Error> {
        match name {
            "get" => self.mapping_get(arguments),
            "keys" | "values" | "items" => self.mapping_method(name, arguments),
            "startswith" => self.string_edge(arguments, true),
            "endswith" => self.string_edge(arguments, false),
            "split" => self.string_split(arguments),
            "replace" => self.apply_replace(arguments),
            "strip" => self.string_strip(arguments, true, true),
            "lstrip" => self.string_strip(arguments, true, false),
            "rstrip" => self.string_strip(arguments, false, true),
            _ => Err(Error::Geometry),
        }
    }
    fn build_map(&mut self, count: usize) -> Result<(), Error> {
        self.build_object(count)
    }
    fn begin_collection(&mut self) -> Result<(), Error> {
        self.start_collection()
    }
    fn append_collection(&mut self, value: Slot) -> Result<(), Error> {
        self.collect_value(value)
    }
    fn end_collection(&mut self) -> Result<Slot, Error> {
        self.finish_collection()
    }
    fn build_list(&mut self, count: usize) -> Result<(), Error> {
        self.build_sequence(count)
    }
    fn add(&mut self, left: Slot, right: Slot) -> Result<Slot, Error> {
        if matches!(left, Slot::Sequence(_) | Slot::EmptySequence)
            && matches!(right, Slot::Sequence(_) | Slot::EmptySequence)
        {
            return self.concat_sequence(left, right);
        }
        if let (Some(left), Some(right)) = (scalar(left), scalar(right)) {
            return primitive::scalar::arithmetic(
                primitive::scalar::Arithmetic::Add,
                left,
                right,
                primitive::scalar::fixed_conversion(left, right),
            )
            .map(Slot::Scalar)
            .map_err(|_| Error::Geometry);
        }
        let (Slot::Text(left), Slot::Text(right)) = (left, right) else {
            return Err(Error::Geometry);
        };
        let (ln, lb) = self.parts(left)?;
        let (rn, rb) = self.parts(right)?;
        let count = ln.checked_add(rn).ok_or(Error::Overflow)?;
        let bytes = lb.checked_add(rb).ok_or(Error::Overflow)?;
        let measured::Measured {
            leading,
            trailing,
            characters,
            leading_characters,
            trailing_characters,
            ..
        } = self.measure_text(left)?.append(self.measure_text(right)?)?;
        let start = self.counts.concat;
        self.append(left)?;
        self.append(right)?;
        Ok(Slot::Text(Text::Concat {
            start,
            count,
            bytes,
            skip: 0,
            leading,
            trailing,
            characters,
            leading_characters,
            trailing_characters,
        }))
    }
    fn string_concat(&mut self, left: Slot, right: Slot) -> Result<Slot, Error> {
        let left = self.string_value(left)?;
        let right = self.string_value(right)?;
        self.add(left, right)
    }
    fn arithmetic(
        &mut self,
        operation: primitive::scalar::Arithmetic,
        left: Slot,
        right: Slot,
    ) -> Result<Slot, Error> {
        let left = scalar(left).ok_or(Error::Geometry)?;
        let right = scalar(right).ok_or(Error::Geometry)?;
        primitive::scalar::arithmetic(
            operation,
            left,
            right,
            primitive::scalar::fixed_conversion(left, right),
        )
        .map(Slot::Scalar)
        .map_err(|_| Error::Geometry)
    }
    fn ne(&mut self, left: Slot, right: Slot) -> Result<Slot, Error> {
        self.structural_equal(left, right)
            .map(|equal| Slot::Bool(!equal))
    }
    fn eq(&mut self, left: Slot, right: Slot) -> Result<Slot, Error> {
        let Slot::Bool(value) = self.ne(left, right)? else {
            return Err(Error::Geometry);
        };
        Ok(Slot::Bool(!value))
    }
    fn order(&mut self, left: Slot, right: Slot, order: shared::Order) -> Result<Slot, Error> {
        let (left, right) = if matches!((left, right), (Slot::Text(_), Slot::Text(_))) {
            (
                Slot::Text(Text::Atom(self.text_atom(left)?)),
                Slot::Text(Text::Atom(self.text_atom(right)?)),
            )
        } else {
            (left, right)
        };
        self.ordered(left, right, order)
    }
    fn contains(&mut self, container: Slot, value: Slot) -> Result<Slot, Error> {
        self.membership(container, value)
    }
    fn not(&mut self, value: Slot) -> Result<Slot, Error> {
        Ok(Slot::Bool(!self.truth(&value)?))
    }
    fn neg(&mut self, value: Slot) -> Result<Slot, Error> {
        let value = scalar(value).ok_or(Error::Geometry)?;
        let value = match value {
            Scalar::None | Scalar::Bool(_) => return Err(Error::Geometry),
            Scalar::F64(value) => Scalar::F64(primitive::negative_float(value)),
            Scalar::U128(value) if primitive::negative_special_u128(value).is_some() => {
                Scalar::U128(primitive::negative_special_u128(value).ok_or(Error::Geometry)?)
            }
            value => match primitive::negative_integer(
                primitive::scalar::signed(value).ok_or(Error::Geometry)?,
            )
            .ok_or(Error::Geometry)?
            {
                primitive::Integer::I64(value) => Scalar::I64(value),
                primitive::Integer::I128(value) => Scalar::I128(value),
            },
        };
        Ok(Slot::Scalar(value))
    }
    fn perform_test(
        &mut self,
        name: &'source str,
        arguments: Option<u16>,
        _local: u8,
    ) -> Result<(), Error> {
        if arguments != Some(1) {
            return Err(Error::Geometry);
        }
        let value = self.pop()?;
        if let Some(kind) = primitive::type_tests::TypeTest::from_name(name) {
            let result = self.type_test(kind, value)?;
            return self.push(Slot::Bool(result));
        }
        use primitive::Presence;
        let kind = match name {
            "defined" => Presence::Defined,
            "undefined" => Presence::Undefined,
            "none" => Presence::None,
            _ => return Err(Error::Geometry),
        };
        self.push(Slot::Bool(kind.check(
            matches!(value, Slot::Undefined),
            matches!(scalar(value), Some(Scalar::None)),
        )))
    }
    fn truth(&self, value: &Slot) -> Result<bool, Error> {
        match value {
            Slot::Undefined | Slot::Zero => Ok(false),
            Slot::Bool(value) => Ok(*value),
            Slot::Scalar(value) => Ok(scalar_truth(*value)),
            Slot::EmptySequence => Ok(false),
            Slot::Sequence(value) => Ok(value.len != 0),
            Slot::Object(range) => Ok(range.len != 0),
            Slot::ObjectView(view) => Ok(view.range.len != 0),
            Slot::Range(value) => Ok(value.plan.length != 0),
            Slot::Slice(value) => Ok(value.plan.length != 0),
            Slot::Namespace(index) | Slot::Kwargs(index) => Ok(self.namespace_len(*index)? != 0),
            Slot::Macro(_) => Ok(true),
            #[cfg(feature = "chat-clock")]
            Slot::Clock => Ok(true),
            Slot::Closure(_) | Slot::Arguments { .. } => Err(Error::Geometry),
            Slot::Structured { length, .. } => Ok(*length != 0),
            Slot::Split(view) => Ok(view.length != 0),
            Slot::Mapping(view) => Ok(view.length != 0),
            Slot::MappingPair(_) => Ok(true),
            Slot::Text(text) => Ok(self.parts(*text)?.1 != 0),
            Slot::Messages => Ok(!self.context.messages().is_empty()),
            Slot::Message(_) | Slot::Loop(_) => Ok(true),
        }
    }
    fn apply_filter(
        &mut self,
        name: &str,
        arguments: Option<u16>,
        _local: u8,
    ) -> Result<(), Error> {
        if matches!(name, "items" | "list") {
            return self.mapping_filter(name, arguments);
        }
        if matches!(name, "select" | "reject" | "selectattr" | "rejectattr") {
            return self.apply_selection(name, arguments);
        }
        if name == "tojson" {
            return self.apply_json(arguments);
        }
        if name == "string" {
            return self.apply_string(arguments);
        }
        if name == "float" {
            return self.apply_float(arguments);
        }
        if name == "dictsort" {
            return self.apply_dictsort(arguments);
        }
        if name == "map" {
            return self.apply_map_filter(arguments);
        }
        if name == "upper" {
            return self.apply_upper(arguments);
        }
        if name == "replace" {
            return self.apply_replace(arguments);
        }
        if name == "join" {
            return self.apply_join(arguments);
        }
        if matches!(name, "default" | "d") {
            return self.apply_default(arguments);
        }
        if matches!(name, "length" | "count") {
            if arguments != Some(1) {
                return Err(Error::Geometry);
            }
            let value = self.pop()?;
            let length = self.value_length(value)?;
            return self.push(Slot::Scalar(Scalar::U64(
                u64::try_from(length).map_err(|_| Error::Overflow)?,
            )));
        }
        if name != "trim" {
            return Err(Error::Geometry);
        }
        if arguments == Some(2) {
            let Slot::Text(Text::Atom(characters)) = self.pop()? else {
                return Err(Error::Geometry);
            };
            let Slot::Text(Text::Atom(value)) = self.pop()? else {
                return Err(Error::Geometry);
            };
            let kept = crate::filters::character_trim_range(
                atom_text(
                    self.source,
                    self.context,
                    &self.borrowed,
                    self.workspace.context_text.as_deref(),
                    value,
                )?,
                atom_text(
                    self.source,
                    self.context,
                    &self.borrowed,
                    self.workspace.context_text.as_deref(),
                    characters,
                )?,
            );
            return self.push(Slot::Text(Text::Atom(slice_atom(
                value,
                kept.start,
                kept.len(),
            )?)));
        }
        if arguments != Some(1) {
            return Err(Error::Geometry);
        }
        let Slot::Text(text) = self.pop()? else {
            return Err(Error::Geometry);
        };
        let (bytes, leading, trailing) = self.edges(text)?;
        let bytes = bytes
            .checked_sub(leading)
            .and_then(|n| n.checked_sub(trailing))
            .ok_or(Error::Geometry)?;
        let (characters, leading_characters, trailing_characters) = self.character_edges(text)?;
        let characters = characters
            .checked_sub(leading_characters)
            .and_then(|n| n.checked_sub(trailing_characters))
            .ok_or(Error::Geometry)?;
        let trimmed = match text {
            Text::Atom(atom) => Text::Atom(slice_atom(atom, leading, bytes)?),
            Text::Concat {
                start, count, skip, ..
            } => Text::Concat {
                start,
                count,
                bytes,
                skip: skip.checked_add(leading).ok_or(Error::Overflow)?,
                leading: 0,
                trailing: 0,
                characters,
                leading_characters: 0,
                trailing_characters: 0,
            },
        };
        self.push(Slot::Text(trimmed))
    }
    fn begin_capture(&mut self, mode: crate::output::CaptureMode) -> Result<(), Error> {
        self.begin_output_capture(mode)
    }
    fn end_capture(&mut self) -> Result<Slot, Error> {
        self.end_output_capture()
    }
    fn emit(&mut self, value: Slot) -> Result<(), Error> {
        if self.emit_into_capture(value)? {
            return Ok(());
        }
        if self.calls != 0 {
            return self.macro_emit(value);
        }
        if let Some(value) = scalar(value) {
            return self.emit_scalar(value);
        }
        if matches!(value, Slot::Undefined) {
            return Ok(());
        }
        if matches!(value, Slot::EmptySequence) {
            return self.emit_text("[]");
        }
        let Slot::Text(text) = value else {
            return Err(Error::Geometry);
        };
        match text {
            Text::Atom(atom) => self.emit_atom(atom),
            Text::Concat {
                start,
                count,
                bytes,
                mut skip,
                ..
            } => {
                if self.workspace.output.is_none() {
                    self.counts.output = self
                        .counts
                        .output
                        .checked_add(bytes)
                        .ok_or(Error::Overflow)?;
                    return Ok(());
                }
                let end = start.checked_add(count).ok_or(Error::Overflow)?;
                if end > self.counts.concat {
                    return Err(Error::Geometry);
                }
                let mut remaining = bytes;
                for index in start..end {
                    let atom = *self
                        .workspace
                        .concat
                        .as_deref()
                        .and_then(|v| v.get(index))
                        .ok_or(Error::Geometry)?;
                    let atom = clip_atom(
                        self.source,
                        self.context,
                        &self.borrowed,
                        self.workspace.context_text.as_deref(),
                        atom,
                        &mut skip,
                        &mut remaining,
                    )?;
                    self.emit_atom(atom)?;
                }
                if skip != 0 || remaining != 0 {
                    return Err(Error::Geometry);
                }
                Ok(())
            }
        }
    }
    fn push_loop(&mut self, value: Slot, flags: u8, pc: u32) -> Result<(), Error> {
        self.enter_loop(value, flags, pc)
    }
    fn next(&mut self) -> Result<Option<Slot>, Error> {
        self.advance_loop()
    }
    fn pop_loop(&mut self) -> Result<Option<u32>, Error> {
        self.leave_loop()
    }
}

pub(super) fn run<'input>(
    source: &PreparedTemplate,
    context: RenderContext<'input>,
    generation: bool,
    workspace: Workspace<'_>,
    borrowed: &mut Borrowed<'input>,
    json: &mut json::Scratch<'input>,
) -> Result<Counts, Failure> {
    // Each source-structured frame consumes a finite forward segment. Only
    // advancing its immutable input iterator replenishes that frame's budget;
    // nested frames have independent source-sized budgets. No guessed product.
    let remaining_instructions = source.instructions.len();
    borrowed.clear();
    let mut engine = Engine {
        remaining_instructions,
        source,
        context,
        generation,
        borrowed,
        json,
        json_options: None,
        workspace,
        depth: 0,
        loops: 0,
        locals: 0,
        counts: Counts::default(),
        calls: 0,
        capture: None,
        collection: None,
        closure_active: false,
        closure_len: 0,
    };
    engine.workspace.operands.fill(Slot::Undefined);
    engine.workspace.frames.fill(Frame::default());
    engine.workspace.locals.fill(Local::default());
    engine.workspace.namespaces.fill(Namespace::default());
    engine
        .workspace
        .namespace_fields
        .fill(NamespaceField::default());
    engine.workspace.calls.fill(CallFrame::default());
    engine.workspace.arguments.fill(Slot::Undefined);
    engine.workspace.closure_fields.fill(Local::default());
    let mut pc = 0_u32;
    let result = (|| {
        while let Some(instruction) = source.instructions.get(pc as usize) {
            let remaining = if engine.loops == engine.call_loops_start() {
                if engine.calls == 0 {
                    &mut engine.remaining_instructions
                } else {
                    &mut engine.workspace.calls[engine.calls - 1].remaining
                }
            } else {
                &mut engine
                    .workspace
                    .frames
                    .get_mut(engine.loops - 1)
                    .ok_or(Error::Geometry)?
                    .remaining
            };
            *remaining = remaining.checked_sub(1).ok_or(Error::Geometry)?;
            let op = match *instruction {
                Instruction::DupTop => Op::DupTop,
                Instruction::DiscardTop => Op::DiscardTop,
                Instruction::Enclose(name) => {
                    Op::Enclose(name.text(&source.bytes).ok_or(Error::Geometry)?)
                }
                Instruction::GetClosure => Op::GetClosure,
                Instruction::Arguments { start, count } => {
                    Op::LoadConst(Slot::Arguments { start, count })
                }
                Instruction::BuildMacro(name, offset, flags) => Op::BuildMacro(
                    name.text(&source.bytes).ok_or(Error::Geometry)?,
                    offset,
                    flags,
                ),
                Instruction::Return => Op::Return,
                Instruction::Argument(_) => return Err(Error::Geometry),
                Instruction::Lookup(r) => Op::Lookup(r.text(&source.bytes).ok_or(Error::Geometry)?),
                Instruction::StoreLocal(r) => {
                    Op::StoreLocal(r.text(&source.bytes).ok_or(Error::Geometry)?)
                }
                Instruction::SetAttr(r) => {
                    Op::SetAttr(r.text(&source.bytes).ok_or(Error::Geometry)?)
                }
                Instruction::BuildKwargs(count) => Op::BuildKwargs(count),
                Instruction::CallFunction(r, count) => Op::CallFunction(
                    r.text(&source.bytes).ok_or(Error::Geometry)?,
                    Some(count),
                    None,
                ),
                Instruction::UnpackList(count) => Op::UnpackList(count),
                Instruction::BuildList(count) => Op::BuildList(count),
                Instruction::BeginCollection => Op::BeginCollection,
                Instruction::AppendCollection => Op::AppendCollection,
                Instruction::EndCollection => Op::EndCollection,
                Instruction::BuildMap(count) => Op::BuildMap(count),
                Instruction::GetAttr(r) => {
                    Op::GetAttr(r.text(&source.bytes).ok_or(Error::Geometry)?)
                }
                Instruction::GetItem => Op::GetItem,
                Instruction::Slice => Op::Slice,
                Instruction::MappingView(projection) => {
                    Op::CallMethod(projection.method(), Some(1))
                }
                Instruction::Items => Op::ApplyFilter("items", Some(1), 0),
                Instruction::List => Op::ApplyFilter("list", Some(1), 0),
                Instruction::Selection {
                    invert,
                    attribute,
                    args,
                } => Op::ApplyFilter(
                    match (invert, attribute) {
                        (false, false) => "select",
                        (true, false) => "reject",
                        (false, true) => "selectattr",
                        (true, true) => "rejectattr",
                    },
                    Some(u16::from(args) + 1),
                    0,
                ),
                Instruction::MappingGet(args) => Op::CallMethod("get", Some(u16::from(args) + 1)),
                Instruction::StartsWith => Op::CallMethod("startswith", Some(2)),
                Instruction::EndsWith => Op::CallMethod("endswith", Some(2)),
                Instruction::StringSplit(args) => {
                    Op::CallMethod("split", Some(u16::from(args) + 1))
                }
                Instruction::StringStrip { args, left, right } => Op::CallMethod(
                    match (left, right) {
                        (true, true) => "strip",
                        (true, false) => "lstrip",
                        (false, true) => "rstrip",
                        _ => return Err(Error::Geometry),
                    },
                    Some(u16::from(args) + 1),
                ),
                Instruction::Literal(r) => Op::LoadConst(Slot::Text(Text::Atom(Atom::Source(r)))),
                Instruction::Undefined => Op::LoadConst(Slot::Undefined),
                Instruction::Zero => Op::LoadConst(Slot::Zero),
                Instruction::Scalar(value) => Op::LoadConst(Slot::Scalar(value)),
                Instruction::Add => Op::Add,
                Instruction::StringConcat => Op::StringConcat,
                Instruction::Arithmetic(operation) => Op::Arithmetic(operation),
                Instruction::Ne => Op::Ne,
                Instruction::Eq => Op::Eq,
                Instruction::In => Op::In,
                Instruction::Order(order) => Op::Order(order),
                Instruction::Not => Op::Not,
                Instruction::Neg => Op::Neg,
                Instruction::Presence(kind) => Op::PerformTest(
                    match kind {
                        primitive::Presence::Defined => "defined",
                        primitive::Presence::Undefined => "undefined",
                        primitive::Presence::None => "none",
                    },
                    Some(1),
                    0,
                ),
                Instruction::TypeTest(kind) => Op::PerformTest(kind.name(), Some(1), 0),
                Instruction::Trim => Op::ApplyFilter("trim", Some(1), 0),
                Instruction::DictSort => Op::ApplyFilter("dictsort", Some(1), 0),
                Instruction::MapFilter => Op::ApplyFilter("map", Some(2), 0),
                Instruction::Upper => Op::ApplyFilter("upper", Some(1), 0),
                Instruction::TrimCharacters => Op::ApplyFilter("trim", Some(2), 0),
                Instruction::Length => Op::ApplyFilter("length", Some(1), 0),
                Instruction::Stringify => Op::ApplyFilter("string", Some(1), 0),
                Instruction::Floatify => Op::ApplyFilter("float", Some(1), 0),
                Instruction::ToJson(options) => {
                    engine.json_options = Some(options);
                    Op::ApplyFilter("tojson", Some(u16::from(options.arguments) + 1), 0)
                }
                Instruction::Replace => Op::ApplyFilter("replace", Some(3), 0),
                Instruction::StringReplace => Op::CallMethod("replace", Some(3)),
                Instruction::Join(args) => Op::ApplyFilter("join", Some(u16::from(args) + 1), 0),
                Instruction::Default(args) => {
                    Op::ApplyFilter("default", Some(u16::from(args) + 1), 0)
                }
                Instruction::EmptySequence => Op::LoadConst(Slot::EmptySequence),
                Instruction::BeginCapture => Op::BeginCapture(crate::output::CaptureMode::Capture),
                Instruction::EndCapture => Op::EndCapture,
                Instruction::Emit => Op::Emit,
                Instruction::PushLoop(flags) => Op::PushLoop(flags),
                Instruction::Iterate(target) => Op::Iterate(target),
                Instruction::PopLoopFrame => Op::PopLoopFrame,
                Instruction::Jump(target) => Op::Jump(target),
                Instruction::JumpIfFalse(target) => Op::JumpIfFalse(target),
                Instruction::JumpIfFalseOrPop(target) => Op::JumpIfFalseOrPop(target),
                Instruction::JumpIfTrueOrPop(target) => Op::JumpIfTrueOrPop(target),
            };
            pc = match shared::step(op, pc, &mut engine)? {
                Next::Advance => pc.checked_add(1).ok_or(Error::Overflow)?,
                Next::Return => return Err(Error::Geometry),
                Next::Jump(target) => {
                    if target <= pc
                        && !matches!(
                            instruction,
                            Instruction::CallFunction(..) | Instruction::Return
                        )
                        && engine
                            .workspace
                            .frames
                            .get(engine.loops.checked_sub(1).ok_or(Error::Geometry)?)
                            .ok_or(Error::Geometry)?
                            .iterate
                            != target
                    {
                        return Err(Error::Geometry);
                    }
                    target
                }
            };
        }
        if pc as usize != source.instructions.len()
            || engine.depth != 0
            || engine.loops != 0
            || engine.calls != 0
            || engine.capture.is_some()
            || engine.collection.is_some()
            || engine.locals > source.geometry.locals
        {
            return Err(Error::Geometry);
        }
        // The root lexical scope ends with this render. Drop every actual
        // borrowed local before returning the pointer-free output workspace.
        engine.clear_locals(0, engine.locals)?;
        Ok(engine.counts)
    })();
    result.map_err(|kind| Failure {
        kind,
        instruction: pc,
    })
}

/// Concrete transient controls in addition to the paid source-derived destinations.
pub(super) fn control_bytes() -> Option<usize> {
    use std::mem::size_of;
    let clock_bytes = {
        #[cfg(feature = "chat-clock")]
        {
            clock::control_bytes()?
        }
        #[cfg(not(feature = "chat-clock"))]
        {
            0usize
        }
    };
    RenderContext::control_bytes()?
        .checked_add(clock_bytes)?
        .checked_add(structured::control_bytes()?)?
        .checked_add(values::control_bytes()?)?
        .checked_add(selection::control_bytes()?)?
        .checked_add(map_filter::control_bytes()?)?
        .checked_add(dictsort::control_bytes()?)?
        .checked_add(capture::control_bytes()?)?
        .checked_add(collection::control_bytes()?)?
        .checked_add(object::control_bytes()?)?
        .checked_add(loops::control_bytes()?)?
        .checked_add(namespace::control_bytes()?)?
        .checked_add(macros::control_bytes()?)?
        .checked_add(length::control_bytes()?)?
        .checked_add(default::control_bytes()?)?
        .checked_add(join::control_bytes()?)?
        .checked_add(generated::control_bytes()?)?
        .checked_add(slice::control_bytes()?)?
        .checked_add(json::control_bytes()?)?
        .checked_add(replace::control_bytes()?)?
        .checked_add(case::control_bytes()?)?
        .checked_add(measured::control_bytes()?)?
        .checked_add(ordering::control_bytes()?)?
        .checked_add(equality::control_bytes()?)?
        .checked_add(membership::control_bytes()?)?
        .checked_add(methods::control_bytes()?)?
        .checked_add(range::control_bytes()?)?
        .checked_add(crate::defaults::builtin_global_control_bytes())?
        .checked_add(string::control_bytes()?)?
        .checked_add(string_views::control_bytes()?)?
        .checked_add(mapping::control_bytes()?)?
        .checked_add(size_of::<(
            primitive::Presence,
            bool,
            bool,
            Option<primitive::Integer>,
        )>())?
        .checked_add(primitive::text::scalar_control_bytes::<ScalarOutput<'_>>()?)?
        .checked_add(size_of::<ScalarOutput<'_>>())?
        .checked_add(size_of::<Scalar>())?
        // Same scalar equality/arithmetic/truth helpers, including the two-value
        // fixed conversion closure and exact returned coercion/error variants.
        .checked_add(size_of::<[Scalar; 2]>())?
        // Existing numeric worker's checked integer/floating controls and exact
        // conversion closure; dynamic repetition remains a distinct producer.
        .checked_add(primitive::scalar::control_bytes())?
        .checked_add(size_of::<(primitive::scalar::Arithmetic, Slot, Slot)>())?
        .checked_add(size_of::<Result<Slot, Error>>())?
        .checked_add(size_of::<primitive::Truth>())?
        .checked_add(size_of::<Option<primitive::scalar::Pair>>())?
        .checked_add(size_of::<Result<Scalar, primitive::scalar::Failure>>())?
        .checked_add(size_of::<(
            primitive::scalar::Arithmetic,
            primitive::scalar::Side,
        )>())?
        .checked_add(size_of::<std::fmt::Arguments<'_>>())?
        .checked_add(size_of::<Result<(), std::fmt::Error>>())?
        .checked_add(size_of::<[Slot; 2]>())?
        // Compared generated operands retain their paid materialized descriptors.
        .checked_add(size_of::<[Slot; 4]>())?
        .checked_add(size_of::<Engine<'_, '_, '_>>())?
        .checked_add(size_of::<Workspace<'_>>())?
        .checked_add(size_of::<Op<'_, Slot>>())?
        .checked_add(size_of::<Result<Next, Error>>())?
        .checked_add(size_of::<Counts>())?
        .checked_add(size_of::<Result<Counts, Failure>>())?
        .checked_add(size_of::<super::source::Geometry>())?
        .checked_add(size_of::<Local>())?
        .checked_add(size_of::<Frame>())?
        // add: both text operands, four parts, count/bytes, five measured
        // edge/count fields and the retained start.
        .checked_add(size_of::<(Text, Text)>())?
        .checked_add(size_of::<[usize; 12]>())?
        // append/emit: input text, retained interval, cursor, and atom transport.
        .checked_add(size_of::<Text>())?
        .checked_add(size_of::<[usize; 8]>())?
        .checked_add(size_of::<(Atom, Result<Atom, Error>)>())?
        // trim/clip/edge helpers: borrowed values, UTF-8 windows and results.
        .checked_add(size_of::<(Slot, Text, Text, Atom, Range)>())?
        .checked_add(size_of::<[usize; 8]>())?
        .checked_add(size_of::<(&str, &str, std::ops::Range<usize>)>())?
        .checked_add(size_of::<Result<Range, Error>>())?
        // Character-set trim's two borrowed operands and same ordinary helper.
        .checked_add(size_of::<(Slot, Slot, Atom, Atom, std::ops::Range<usize>)>())?
        .checked_add(crate::filters::character_trim_control_bytes()?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bounded::{
        CompilePlan, TemplateName, TemplateSettings, TextMessage, supported_source,
    };

    #[test]
    fn actual_concat_and_output_checked_frontiers_reject_before_writing() {
        let source = CompilePlan::prepare(
            supported_source(),
            TemplateName::Single("frontier"),
            TemplateSettings::text_chat(),
        )
        .unwrap()
        .compile()
        .unwrap();
        let messages = [TextMessage {
            role: "user",
            content: "é",
        }];
        let mut operands = [Slot::Undefined; 3];
        let mut frames = [Frame::default(); 1];
        let mut locals = [Local::default(); 1];
        let mut concat = [Atom::Role(0); 2];
        let mut output = [0xA5; 4];
        let mut borrowed = Borrowed::prepare(
            Borrowed::count(source.geometry).unwrap(),
            source.geometry.operands,
        )
        .unwrap();
        let mut engine = Engine {
            source: &source,
            context: RenderContext::from_messages(crate::bounded::Messages::from_text(&messages)),
            generation: false,
            borrowed: &mut borrowed,
            workspace: Workspace {
                values: &mut [],
                value_register_start: 0,
                operands: &mut operands,
                frames: &mut frames,
                locals: &mut locals,
                namespaces: &mut [],
                namespace_fields: &mut [],
                calls: &mut [],
                arguments: &mut [],
                closure_fields: &mut [],
                concat: Some(&mut concat),
                output: Some(&mut output),
                context_text: None,
            },
            depth: 0,
            loops: 0,
            locals: 0,
            counts: Counts {
                concat: usize::MAX,
                output: usize::MAX,
                context_text: 0,
            },
            remaining_instructions: 1,
            capture: None,
            collection: None,
        };
        // Private unit setup targets the real checked accumulators. No public
        // capacity, source/input length or successful fake profile is supplied.
        assert_eq!(
            engine.append(Text::Atom(Atom::Content(0))),
            Err(Error::Overflow)
        );
        assert_eq!(engine.emit_atom(Atom::Content(0)), Err(Error::Overflow));
        let huge = Slot::Text(Text::Concat {
            start: 0,
            count: 1,
            bytes: usize::MAX,
            skip: 0,
            leading: 0,
            trailing: 0,
            characters: usize::MAX,
            leading_characters: 0,
            trailing_characters: 0,
        });
        assert!(matches!(
            engine.add(huge, Slot::Text(Text::Atom(Atom::Role(0)))),
            Err(Error::Overflow)
        ));
        assert_eq!(
            engine.counts,
            Counts {
                concat: usize::MAX,
                output: usize::MAX,
                context_text: 0
            }
        );
        drop(engine);
        assert!(concat.iter().all(|atom| matches!(atom, Atom::Role(0))));
        assert_eq!(output, [0xA5; 4]);
    }
}
