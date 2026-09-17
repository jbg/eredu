//! Paid storage for the ordinary call/default/capture/return protocol.
//! Root declarations share the exact capture table. Calls retain their own
//! lexical boundary and output, so recursive calls never borrow caller locals.
use super::*;
mod closures;
use crate::vm::shared::macro_arguments::{self, View};
use std::mem::{size_of, size_of_val};
#[derive(Clone, Copy, Debug)]
pub(super) struct MacroValue {
    name: Range,
    entry: u32,
    arguments: u32,
    count: usize,
    closure: Option<usize>,
    caller: bool,
}
#[derive(Clone, Copy, Debug, Default)]
pub(in crate::bounded) struct CallFrame {
    return_pc: u32,
    operands: usize,
    locals: usize,
    loops: usize,
    depth: usize,
    pub(super) remaining: usize,
    pub(super) closure: Option<usize>,
    pub(super) owns_closure: bool,
    pub(super) closure_len: usize,
    output: Option<Text>,
}
/// Ordinary eval_macro installs two context frames and its existing recursion
/// cost; this is the configured text-chat VM limit, not an added render limit.
pub(in crate::bounded) const fn call_capacity() -> usize {
    (crate::environment::MAX_RECURSION - 1) / (crate::vm::MACRO_RECURSION_COST + 2)
}
fn closure_fields_per_scope(source: &PreparedTemplate) -> Option<usize> {
    source.instructions.iter().try_fold(0usize, |count, op| {
        count.checked_add(usize::from(matches!(
            op,
            Instruction::StoreLocal(_) | Instruction::Enclose(_)
        )))
    })
}
fn closure_scopes(source: &PreparedTemplate) -> Option<usize> {
    if !source
        .instructions
        .iter()
        .any(|op| matches!(op, Instruction::BuildMacro(..)))
    {
        return Some(0);
    }
    1usize
        .checked_add(call_capacity())?
        .checked_add(source.geometry.frames)
}
pub(in crate::bounded) fn closure_capacity(source: &PreparedTemplate) -> Option<usize> {
    closure_scopes(source)?.checked_mul(closure_fields_per_scope(source)?)
}
#[derive(Clone, Copy, Debug, Default)]
pub(in crate::bounded) struct Layout {
    pub calls: usize,
    pub arguments: usize,
    pub closure_fields: usize,
    pub register_start: usize,
}
impl Layout {
    pub(in crate::bounded) fn inspect(source: &PreparedTemplate) -> Option<Self> {
        let present = source
            .instructions
            .iter()
            .any(|op| matches!(op, Instruction::BuildMacro(..)));
        let arguments = source
            .instructions
            .iter()
            .filter_map(|op| match op {
                Instruction::Arguments { count, .. } => Some(*count),
                _ => None,
            })
            .max()
            .unwrap_or(0);
        let namespaces = namespace::Layout::inspect(source)?;
        Some(Self {
            calls: if present { call_capacity() } else { 0 },
            arguments,
            closure_fields: closure_capacity(source)?,
            register_start: namespaces.register_start.checked_add(namespaces.fields)?,
        })
    }
    pub(in crate::bounded) fn borrowed_count(self) -> Option<usize> {
        self.register_start
            .checked_add(self.arguments)?
            .checked_add(self.closure_fields)
    }
}
struct Arguments<'a> {
    source: &'a PreparedTemplate,
    signature: MacroValue,
    positional: &'a [Slot],
    keywords: &'a [Local],
}
impl<'a> View<'a> for Arguments<'a> {
    type Value = Slot;
    fn parameters(&self) -> usize {
        self.signature.count
    }
    fn parameter(&self, index: usize) -> Option<&'a str> {
        match self
            .source
            .instructions
            .get((self.signature.arguments as usize).checked_add(index)?)?
        {
            Instruction::Argument(name) => name.text(&self.source.bytes),
            _ => None,
        }
    }
    fn positionals(&self) -> usize {
        self.positional.len()
    }
    fn positional(&self, index: usize) -> Option<Slot> {
        self.positional.get(index).copied()
    }
    fn keyword(&self, name: &str) -> Option<Slot> {
        self.keywords
            .iter()
            .find(|v| v.name.and_then(|r| r.text(&self.source.bytes)) == Some(name))
            .map(|v| v.value)
    }
    fn keywords(&self) -> usize {
        self.keywords.len()
    }
    fn keyword_name(&self, index: usize) -> Option<&'a str> {
        self.keywords.get(index)?.name?.text(&self.source.bytes)
    }
    fn undefined(&self) -> Slot {
        Slot::Undefined
    }
    fn caller(&self) -> bool {
        self.signature.caller
    }
}
impl<'source> Engine<'source, '_, '_> {
    pub(super) fn macro_attr(&self, value: MacroValue, name: &str) -> Result<Slot, Error> {
        Ok(match name {
            "name" => Slot::Text(Text::Atom(Atom::Source(value.name))),
            "caller" => Slot::Bool(value.caller),
            _ => Slot::Undefined,
        })
    }
    pub(super) fn call_locals_start(&self) -> usize {
        self.calls
            .checked_sub(1)
            .map_or(0, |i| self.workspace.calls[i].locals)
    }
    pub(super) fn call_loops_start(&self) -> usize {
        self.calls
            .checked_sub(1)
            .map_or(0, |i| self.workspace.calls[i].loops)
    }
    pub(super) fn runtime_depth(&self) -> Result<usize, Error> {
        let base = self
            .calls
            .checked_sub(1)
            .map_or(1, |i| self.workspace.calls[i].depth);
        base.checked_add(
            self.loops
                .checked_sub(self.call_loops_start())
                .ok_or(Error::Geometry)?,
        )
        .ok_or(Error::Overflow)
    }
    pub(super) fn create_macro(&mut self, name: &str, entry: u32, flags: u8) -> Result<(), Error> {
        if flags & !1 != 0 {
            return Err(Error::Geometry);
        }
        let Slot::Arguments { start, count } = self.pop()? else {
            return Err(Error::Geometry);
        };
        let Slot::Closure(closure) = self.pop()? else {
            return Err(Error::Geometry);
        };
        let end = (start as usize).checked_add(count).ok_or(Error::Overflow)?;
        if end != entry as usize || end >= self.source.instructions.len() {
            return Err(Error::Geometry);
        }
        for op in self
            .source
            .instructions
            .get(start as usize..end)
            .ok_or(Error::Geometry)?
        {
            if !matches!(op, Instruction::Argument(_)) {
                return Err(Error::Geometry);
            }
        }
        let name = self
            .source
            .instructions
            .iter()
            .find_map(|op| match op {
                Instruction::BuildMacro(range, offset, _)
                    if *offset == entry && range.text(&self.source.bytes) == Some(name) =>
                {
                    Some(*range)
                }
                _ => None,
            })
            .ok_or(Error::Geometry)?;
        self.push(Slot::Macro(MacroValue {
            name,
            entry,
            arguments: start,
            count,
            closure,
            caller: flags & 1 != 0,
        }))
    }
    pub(super) fn call_macro(
        &mut self,
        value: MacroValue,
        count: Option<u16>,
        pc: u32,
    ) -> Result<u32, Error> {
        let count = usize::from(count.ok_or(Error::Geometry)?);
        let start = self.depth.checked_sub(count).ok_or(Error::Geometry)?;
        let has_keywords =
            count != 0 && matches!(self.workspace.operands[self.depth - 1], Slot::Kwargs(_));
        let positional_end = self.depth - usize::from(has_keywords);
        let keyword_range = if has_keywords {
            let Slot::Kwargs(index) = self.workspace.operands[self.depth - 1] else {
                unreachable!()
            };
            self.keyword_range(index)?
        } else {
            0..0
        };
        let arguments = Arguments {
            source: self.source,
            signature: value,
            positional: &self.workspace.operands[start..positional_end],
            keywords: &self.workspace.namespace_fields[keyword_range],
        };
        let mut written = 0usize;
        // The destination is separate from source operands/keyword fields. A
        // borrowed value's own register is retained before either input retires.
        let destination = &mut self.workspace.arguments;
        let caller = macro_arguments::bind(&arguments, |slot| {
            *destination.get_mut(written).ok_or(Error::Geometry)? = slot;
            written = written.checked_add(1).ok_or(Error::Overflow)?;
            Ok::<(), Error>(())
        })
        .map_err(|_| Error::Geometry)?;
        if written != value.count {
            return Err(Error::Geometry);
        }
        let layout = Layout::inspect(self.source).ok_or(Error::Overflow)?;
        for index in 0..written {
            if matches!(
                self.workspace.arguments[index],
                Slot::Closure(_) | Slot::Arguments { .. }
            ) {
                return Err(Error::Geometry);
            }
            self.workspace.arguments[index] = self.borrowed.push(
                self.workspace.arguments[index],
                layout
                    .register_start
                    .checked_add(index)
                    .ok_or(Error::Overflow)?,
            )?;
        }
        // caller is a macro value, never a borrowed JSON input; reject any
        // unsupported value before the caller's argument registers are cleared.
        if caller.is_some_and(|slot| !matches!(slot, Slot::Macro(_) | Slot::Undefined)) {
            return Err(Error::Geometry);
        }
        let depth = self
            .runtime_depth()?
            .checked_add(crate::vm::MACRO_RECURSION_COST + 2)
            .ok_or(Error::Overflow)?;
        if depth > crate::environment::MAX_RECURSION || self.calls >= self.workspace.calls.len() {
            return Err(Error::Geometry);
        }
        while self.depth > start {
            self.pop()?;
        }
        let frame = CallFrame {
            return_pc: pc.checked_add(1).ok_or(Error::Overflow)?,
            operands: start,
            locals: self.locals,
            loops: self.loops,
            depth,
            remaining: self.source.instructions.len(),
            closure: value.closure,
            owns_closure: false,
            closure_len: 0,
            output: None,
        };
        self.workspace.calls[self.calls] = frame;
        self.calls += 1;
        if let Some(caller) = caller {
            self.store_scoped_local("caller", caller)?;
        }
        for index in 0..written {
            let slot = self.workspace.arguments[index];
            self.push(slot)?;
        }
        self.workspace.arguments[..written].fill(Slot::Undefined);
        for index in 0..written {
            self.borrowed.push(
                Slot::Undefined,
                layout
                    .register_start
                    .checked_add(index)
                    .ok_or(Error::Overflow)?,
            )?;
        }
        Ok(value.entry)
    }
    pub(super) fn macro_emit(&mut self, value: Slot) -> Result<(), Error> {
        let index=self.calls.checked_sub(1).ok_or(Error::Geometry)?;
        let previous=self.workspace.calls[index].output;
        self.workspace.calls[index].output=self.append_captured_output(previous,value)?;
        Ok(())
    }
    pub(super) fn append_captured_output(&mut self,previous:Option<Text>,value:Slot)->Result<Option<Text>,Error> {
        let text = if matches!(value, Slot::Undefined) {
            return Ok(previous);
        } else if let Slot::Text(text) = value {
            text
        } else {
            let (start, text_start) = self.generated_start()?;
            let mut output = generated::Output {
                bytes: self.workspace.context_text.as_deref_mut(),
                start: text_start,
                measured: measured::Measured::default(),
                error: None,
            };
            let result = if let Some(scalar) = scalar(value) {
                primitive::text::write_scalar(&mut output, scalar, false)
            } else if matches!(value, Slot::EmptySequence) {
                std::fmt::Write::write_str(&mut output, "[]")
            } else {
                return Err(Error::Geometry);
            };
            result.map_err(|_| output.error.unwrap_or(Error::Geometry))?;
            let measured = output.finish()?;
            self.finish_generated(start, text_start, measured)?;
            let Slot::Text(text) = self.pop()? else {
                return Err(Error::Geometry);
            };
            text
        };
        let text = if let Some(previous) = previous {
            let Slot::Text(text) = self.add(Slot::Text(previous), Slot::Text(text))? else {
                return Err(Error::Geometry);
            };
            text
        } else {
            // Even one emitted borrowed atom must survive argument/local reuse.
            let measured = self.measure_text(text)?;
            let start = self.counts.concat;
            let count = self.parts(text)?.0;
            self.append(text)?;
            measured.concat(start, count)
        };
        Ok(Some(text))
    }
    pub(super) fn finish_macro(&mut self) -> Result<u32, Error> {
        let index = self.calls.checked_sub(1).ok_or(Error::Geometry)?;
        let frame = self.workspace.calls[index];
        if self.depth != frame.operands || self.loops != frame.loops {
            return Err(Error::Geometry);
        }
        self.retire_macro_scope(index.checked_add(1).ok_or(Error::Overflow)?)?;
        self.clear_locals(frame.locals, self.locals)?;
        self.calls = index;
        self.workspace.calls[index] = CallFrame::default();
        let output = frame
            .output
            .unwrap_or_else(|| measured::Measured::default().concat(self.counts.concat, 0));
        self.push(Slot::Text(output))?;
        Ok(frame.return_pc)
    }
}
pub(super) fn control_bytes() -> Option<usize> {
    let parts = [
        size_of::<Layout>(),
        size_of::<MacroValue>(),
        size_of::<CallFrame>(),
        size_of::<Arguments<'_>>(),
        size_of::<macro_arguments::Failure<'_, Error>>(),
        size_of::<Result<Option<Slot>, macro_arguments::Failure<'_, Error>>>(),
        size_of::<(&mut [Slot], usize)>(),
        size_of::<[usize; 8]>(),
        size_of::<[Slot; 3]>(),
        size_of::<Option<Text>>(),
        size_of::<Result<u32, Error>>(),
        size_of::<std::ops::Range<usize>>(),
        size_of::<std::slice::Iter<'_, Local>>(),
        size_of::<std::slice::Iter<'_, Instruction>>(),
        size_of::<Option<Range>>(),
        size_of::<Result<Option<Slot>, Error>>(),
    ];
    parts
        .into_iter()
        .try_fold(size_of_val(&parts), usize::checked_add)?
        .checked_add(macro_arguments::control_bytes::<Slot>()?)?
        .checked_add(closures::control_bytes()?)
}
