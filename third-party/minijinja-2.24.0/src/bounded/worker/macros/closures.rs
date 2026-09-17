//! One closure destination per actual ordinary lexical frame. Macro results
//! are text and mutable namespace fields still reject escaping macro objects.
//! Thus a closure cannot outlive its frame; aliases share it until that frame
//! retires, including the ordinary loop's repeated-iteration closure identity.
use super::*;
impl<'source> Engine<'source, '_, '_> {
    fn closure_scope(&self) -> Result<usize, Error> {
        if self.loops > self.call_loops_start() {
            call_capacity()
                .checked_add(self.loops)
                .ok_or(Error::Overflow)
        } else {
            Ok(self.calls)
        }
    }
    fn closure_state(&self, index: usize) -> Result<(bool, usize), Error> {
        if index == 0 {
            return Ok((self.closure_active, self.closure_len));
        }
        if index <= call_capacity() {
            if index > self.calls {
                return Err(Error::Geometry);
            }
            let frame = self.workspace.calls.get(index - 1).ok_or(Error::Geometry)?;
            return Ok((frame.owns_closure, frame.closure_len));
        }
        let index = index
            .checked_sub(call_capacity() + 1)
            .ok_or(Error::Geometry)?;
        if index >= self.loops {
            return Err(Error::Geometry);
        }
        let frame = self.workspace.frames.get(index).ok_or(Error::Geometry)?;
        Ok((frame.owns_closure, frame.closure_len))
    }
    fn set_closure_state(&mut self, index: usize, active: bool, len: usize) -> Result<(), Error> {
        if index == 0 {
            self.closure_active = active;
            self.closure_len = len;
            return Ok(());
        }
        if index <= call_capacity() {
            if index > self.calls {
                return Err(Error::Geometry);
            }
            let frame = self
                .workspace
                .calls
                .get_mut(index - 1)
                .ok_or(Error::Geometry)?;
            frame.owns_closure = active;
            frame.closure_len = len;
            return Ok(());
        }
        let index = index
            .checked_sub(call_capacity() + 1)
            .ok_or(Error::Geometry)?;
        if index >= self.loops {
            return Err(Error::Geometry);
        }
        let frame = self
            .workspace
            .frames
            .get_mut(index)
            .ok_or(Error::Geometry)?;
        frame.owns_closure = active;
        frame.closure_len = len;
        Ok(())
    }
    fn closure_fields(&self, index: usize, len: usize) -> Result<std::ops::Range<usize>, Error> {
        let capacity = closure_fields_per_scope(self.source).ok_or(Error::Overflow)?;
        if len > capacity {
            return Err(Error::Geometry);
        }
        let start = index.checked_mul(capacity).ok_or(Error::Overflow)?;
        let end = start.checked_add(len).ok_or(Error::Overflow)?;
        if end > self.workspace.closure_fields.len() {
            return Err(Error::Geometry);
        }
        Ok(start..end)
    }
    pub(in crate::bounded::worker) fn current_macro_closure(&self) -> Result<Option<usize>, Error> {
        let index = self.closure_scope()?;
        Ok(self.closure_state(index)?.0.then_some(index))
    }
    pub(in crate::bounded::worker) fn macro_capture(
        &self,
        index: usize,
        name: &str,
    ) -> Result<Option<Slot>, Error> {
        let (active, len) = self.closure_state(index)?;
        if !active {
            return Err(Error::Geometry);
        }
        for field in &self.workspace.closure_fields[self.closure_fields(index, len)?] {
            if field.name.and_then(|r| r.text(&self.source.bytes)) == Some(name) {
                return Ok(Some(field.value));
            }
        }
        Ok(None)
    }
    pub(in crate::bounded::worker) fn store_macro_capture(
        &mut self,
        name: &str,
        value: Slot,
    ) -> Result<(), Error> {
        if matches!(value, Slot::Closure(_) | Slot::Arguments { .. }) {
            return Err(Error::Geometry);
        }
        let scope = self.closure_scope()?;
        let (active, len) = self.closure_state(scope)?;
        if !active {
            return Err(Error::Geometry);
        }
        let row = self.closure_fields(scope, len)?;
        let index = self.workspace.closure_fields[row.clone()]
            .iter()
            .position(|field| field.name.and_then(|r| r.text(&self.source.bytes)) == Some(name))
            .unwrap_or(len);
        let address = row.start.checked_add(index).ok_or(Error::Overflow)?;
        if index == len {
            self.closure_fields(scope, len.checked_add(1).ok_or(Error::Overflow)?)?;
        }
        let range = self
            .source
            .instructions
            .iter()
            .find_map(|op| match op {
                Instruction::StoreLocal(r) | Instruction::Enclose(r) | Instruction::Lookup(r)
                    if r.text(&self.source.bytes) == Some(name) =>
                {
                    Some(*r)
                }
                _ => None,
            })
            .ok_or(Error::Geometry)?;
        let layout = Layout::inspect(self.source).ok_or(Error::Overflow)?;
        let register = layout
            .register_start
            .checked_add(layout.arguments)
            .and_then(|n| n.checked_add(address))
            .ok_or(Error::Overflow)?;
        let value = self.borrowed.push(value, register)?;
        self.workspace.closure_fields[address] = Local {
            name: Some(range),
            value,
        };
        if index == len {
            self.set_closure_state(scope, true, len.checked_add(1).ok_or(Error::Overflow)?)?;
        }
        Ok(())
    }
    pub(in crate::bounded::worker) fn enclose_macro(
        &mut self,
        name: &'source str,
    ) -> Result<(), Error> {
        let scope = self.closure_scope()?;
        let (active, len) = self.closure_state(scope)?;
        if !active {
            if len != 0 {
                return Err(Error::Geometry);
            }
            self.set_closure_state(scope, true, 0)?;
        }
        if self.macro_capture(scope, name)?.is_none() {
            let value = self.lookup(name)?;
            self.store_macro_capture(name, value)?;
        }
        Ok(())
    }
    pub(in crate::bounded::worker) fn retire_macro_scope(
        &mut self,
        index: usize,
    ) -> Result<(), Error> {
        let (active, len) = self.closure_state(index)?;
        if !active {
            return if len == 0 {
                Ok(())
            } else {
                Err(Error::Geometry)
            };
        }
        let layout = Layout::inspect(self.source).ok_or(Error::Overflow)?;
        for address in self.closure_fields(index, len)? {
            self.workspace.closure_fields[address] = Local::default();
            let register = layout
                .register_start
                .checked_add(layout.arguments)
                .and_then(|n| n.checked_add(address))
                .ok_or(Error::Overflow)?;
            self.borrowed.push(Slot::Undefined, register)?;
        }
        self.set_closure_state(index, false, 0)
    }
}
pub(super) fn control_bytes() -> Option<usize> {
    let parts = [
        size_of::<(usize, usize, bool)>(),
        size_of::<(usize, usize, bool)>(),
        size_of::<std::ops::Range<usize>>(),
        size_of::<std::slice::Iter<'_, Local>>(),
        size_of::<(Layout, &str, Slot)>(),
        size_of::<Option<usize>>(),
        size_of::<Option<Range>>(),
        size_of::<Result<(bool, usize), Error>>(),
        size_of::<Result<Option<Slot>, Error>>(),
    ];
    parts
        .into_iter()
        .try_fold(size_of_val(&parts), usize::checked_add)
}
