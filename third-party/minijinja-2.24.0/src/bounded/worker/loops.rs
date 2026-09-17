//! Source-derived lexical frames retain exact immutable iterator/local loans.
use super::*;
impl<'source, 'input, 'work> Engine<'source, 'input, 'work> {
    pub(super) fn iterator(&self, value: Slot) -> Result<(Slot, usize), Error> {
        let value = match value {
            Slot::Undefined | Slot::Scalar(Scalar::None) => Slot::EmptySequence,
            Slot::Object(range)=>Slot::ObjectView(object::View{range,projection:super::super::source::MappingProjection::Keys,listed:false}),
            #[cfg(feature = "json")]
            Slot::Structured { register, length } if self.borrowed.get(register)?.is_object() => {
                Slot::Mapping(MappingView {
                    register,
                    length,
                    projection: super::super::source::MappingProjection::Keys,
                    listed: false,
                })
            }
            other => other,
        };
        // Strings use a separate character iterator; compact message objects
        // have no retained arbitrary key inventory. Neither is inferred here.
        if !matches!(
            value,
            Slot::EmptySequence
                | Slot::Sequence(_)
                | Slot::ObjectView(_)
                | Slot::Messages
                | Slot::Range(_)
                | Slot::Slice(_)
                | Slot::Split(_)
                | Slot::Structured { .. }
                | Slot::Mapping(_)
                | Slot::MappingPair(_)
        ) {
            return Err(Error::Geometry);
        }
        Ok((value, self.value_length(value)?))
    }
    pub(super) fn iterator_item(&mut self, input: Slot, index: usize) -> Result<Slot, Error> {
        if let Slot::Range(value)=input { return value.item(index) }
        let key = Slot::Scalar(Scalar::U64(
            u64::try_from(index).map_err(|_| Error::Overflow)?,
        ));
        self.selected_item(input, key)
    }
    fn frame_register(&self, index: usize) -> Result<usize, Error> {
        self.source
            .geometry
            .operands
            .checked_add(2)
            .and_then(|n| n.checked_add(index))
            .ok_or(Error::Overflow)
    }
    fn local_register(&self, index: usize) -> Result<usize, Error> {
        self.frame_register(self.source.geometry.frames)?
            .checked_add(index)
            .ok_or(Error::Overflow)
    }
    pub(super) fn enter_loop(&mut self, value: Slot, flags: u8, pc: u32) -> Result<(), Error> {
        if self
            .runtime_depth()?
            .checked_add(1)
            .ok_or(Error::Overflow)?
            > crate::environment::MAX_RECURSION
        {
            return Err(Error::Geometry);
        }
        if flags > 1 {
            return Err(Error::Geometry);
        }
        let (input, length) = self.iterator(value)?;
        let register = self.frame_register(self.loops)?;
        let input = self.borrowed.push(input, register)?;
        let iterate = pc.checked_add(1).ok_or(Error::Overflow)?;
        if !matches!(
            self.source.instructions.get(iterate as usize),
            Some(Instruction::Iterate(_))
        ) {
            return Err(Error::Geometry);
        }
        *self
            .workspace
            .frames
            .get_mut(self.loops)
            .ok_or(Error::Geometry)? = Frame {
            input,
            length,
            next: 0,
            current: None,
            locals_start: self.locals,
            locals_end: self.locals,
            iterate,
            remaining: self.source.instructions.len(),
            owns_closure: false,
            closure_len: 0,
            visible_loop: flags == 1,
        };
        self.loops = self.loops.checked_add(1).ok_or(Error::Overflow)?;
        Ok(())
    }
    pub(super) fn advance_loop(&mut self) -> Result<Option<Slot>, Error> {
        let index = self.loops.checked_sub(1).ok_or(Error::Geometry)?;
        let frame = *self.workspace.frames.get(index).ok_or(Error::Geometry)?;
        if frame.next == frame.length {
            return Ok(None);
        }
        if frame.next > frame.length {
            return Err(Error::Geometry);
        }
        let value = self.iterator_item(frame.input, frame.next)?;
        // Match ordinary Context::next_loop_item: a successful advance clears
        // only this iteration's locals, retaining its outer lexical scope.
        self.clear_locals(frame.locals_start, frame.locals_end)?;
        self.workspace.frames[index].locals_end = self.locals;
        let current = self
            .workspace
            .frames
            .get_mut(index)
            .ok_or(Error::Geometry)?;
        current.next = frame.next.checked_add(1).ok_or(Error::Overflow)?;
        current.current = Some(frame.next);
        current.remaining = self.source.instructions.len();
        Ok(Some(value))
    }
    pub(super) fn leave_loop(&mut self) -> Result<Option<u32>, Error> {
        let index = self.loops.checked_sub(1).ok_or(Error::Geometry)?;
        let frame = *self.workspace.frames.get(index).ok_or(Error::Geometry)?;
        if frame.next != frame.length || frame.locals_end != self.locals {
            return Err(Error::Geometry);
        }
        self.retire_macro_scope(
            macros::call_capacity()
                .checked_add(index + 1)
                .ok_or(Error::Overflow)?,
        )?;
        self.clear_locals(frame.locals_start, frame.locals_end)?;
        self.loops = index;
        self.workspace.frames[index] = Frame::default();
        let register = self.frame_register(index)?;
        self.borrowed.push(Slot::Undefined, register)?;
        Ok(None)
    }
    pub(super) fn clear_locals(&mut self, start: usize, end: usize) -> Result<(), Error> {
        if start > end || end != self.locals {
            return Err(Error::Geometry);
        }
        for local in start..end {
            *self
                .workspace
                .locals
                .get_mut(local)
                .ok_or(Error::Geometry)? = Local::default();
            let register = self.local_register(local)?;
            self.borrowed.push(Slot::Undefined, register)?;
        }
        self.locals = start;
        Ok(())
    }
    pub(super) fn store_scoped_local(
        &mut self,
        name: &'source str,
        value: Slot,
    ) -> Result<(), Error> {
        let frame_index = (self.loops > self.call_loops_start()).then(|| self.loops - 1);
        let (start, end) = if let Some(index) = frame_index {
            let frame = self.workspace.frames.get(index).ok_or(Error::Geometry)?;
            if frame.current.is_none() || frame.locals_end != self.locals {
                return Err(Error::Geometry);
            }
            (frame.locals_start, frame.locals_end)
        } else {
            (self.call_locals_start(), self.locals)
        };
        let mut index = end;
        for candidate in start..end {
            if self.workspace.locals[candidate]
                .name
                .and_then(|r| r.text(&self.source.bytes))
                == Some(name)
            {
                index = candidate;
                break;
            }
        }
        // Name ranges originate in this immutable instruction image; no owned
        // String or environment map is created on an iteration.
        let range = self
            .source
            .instructions
            .iter()
            .find_map(|instruction| match instruction {
                Instruction::StoreLocal(range)
                | Instruction::CallFunction(range, _)
                | Instruction::Lookup(range)
                | Instruction::Enclose(range)
                    if range.text(&self.source.bytes) == Some(name) =>
                {
                    Some(*range)
                }
                _ => None,
            })
            .ok_or(Error::Geometry)?;
        if self.current_macro_closure()?.is_some() {
            self.store_macro_capture(name, value)?;
        }
        let register = self.local_register(index)?;
        let value = self.borrowed.push(value, register)?;
        *self
            .workspace
            .locals
            .get_mut(index)
            .ok_or(Error::Geometry)? = Local {
            name: Some(range),
            value,
        };
        if index == end {
            self.locals = self.locals.checked_add(1).ok_or(Error::Overflow)?;
            if let Some(frame_index) = frame_index {
                self.workspace.frames[frame_index].locals_end = self.locals;
            }
        }
        Ok(())
    }
    pub(super) fn unpack_loop_value(&mut self, count: usize) -> Result<(), Error> {
        let value = self.pop()?;
        // Ordinary assignment unpacks objects, not string or null iteration.
        if matches!(value, Slot::Undefined | Slot::Scalar(_)) {
            return Err(Error::Geometry);
        }
        let (value, length) = self.iterator(value)?;
        if length != count {
            return Err(Error::Geometry);
        }
        let register = self
            .source
            .geometry
            .operands
            .checked_add(1)
            .ok_or(Error::Overflow)?;
        let value = self.borrowed.push(value, register)?;
        // Shared StoreLocal consumes the first element first, as the ordinary
        // unpack worker's reverse_top does. This input stays in its own slot.
        for index in (0..count).rev() {
            let item = self.iterator_item(value, index)?;
            self.push(item)?;
        }
        self.borrowed.push(Slot::Undefined, register)?;
        Ok(())
    }
    pub(super) fn loop_attr_at(&mut self, index: usize, name: &str) -> Result<Slot, Error> {
        if index >= self.loops {
            return Err(Error::Geometry);
        }
        let frame = *self.workspace.frames.get(index).ok_or(Error::Geometry)?;
        let current = frame.current.ok_or(Error::Geometry)?;
        #[cfg(feature = "adjacent_loop_items")]
        if matches!(name, "previtem" | "nextitem") {
            let adjacent = if name == "previtem" { current.checked_sub(1) }
                else { Some(frame.next).filter(|next| *next < frame.length) };
            return match adjacent {
                Some(index) => self.iterator_item(frame.input, index),
                None => Ok(Slot::Undefined),
            };
        }
        let number = match name {
            "first" => return Ok(Slot::Bool(current == 0)),
            "last" => return Ok(Slot::Bool(frame.next == frame.length)),
            "index0" => current,
            "index" => frame.next,
            "length" => frame.length,
            "revindex" => frame.length.checked_sub(current).ok_or(Error::Geometry)?,
            "revindex0" => frame
                .length
                .checked_sub(frame.next)
                .ok_or(Error::Geometry)?,
            _ => return Err(Error::Geometry),
        };
        Ok(Slot::Scalar(Scalar::U64(
            u64::try_from(number).map_err(|_| Error::Overflow)?,
        )))
    }
}
pub(super) fn control_bytes() -> Option<usize> {
    let controls = [
        std::mem::size_of::<Frame>(),
        std::mem::size_of::<Local>(),
        std::mem::size_of::<(Option<usize>, usize, usize, usize)>(),
        std::mem::size_of::<std::ops::Range<usize>>(),
        std::mem::size_of::<Result<(), Error>>(),
        std::mem::size_of::<(Slot, Slot, usize, usize, u32)>(),
        std::mem::size_of::<Result<(Slot, usize), Error>>(),
        std::mem::size_of::<Result<Option<Slot>, Error>>(),
        std::mem::size_of::<Result<Option<u32>, Error>>(),
        std::mem::size_of::<std::ops::Range<usize>>(),
        std::mem::size_of::<std::iter::Rev<std::ops::Range<usize>>>(),
        std::mem::size_of::<std::slice::Iter<'_, Instruction>>(),
        std::mem::size_of::<std::iter::Rev<std::slice::Iter<'_, Local>>>(),
        std::mem::size_of::<Option<Range>>(),
    ];
    controls
        .into_iter()
        .try_fold(std::mem::size_of_val(&controls), usize::checked_add)
}
