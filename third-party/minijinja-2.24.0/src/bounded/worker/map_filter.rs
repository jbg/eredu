//! Filter mapping retains its iterator and each result in the paid value arena.
use super::*;
use crate::filters::mapping::{self, Driver};
struct Mapped<'a, 's, 'i, 'w> {
    engine: &'a mut Engine<'s, 'i, 'w>,
    base: usize,
    length: usize,
    next: usize,
}
impl Driver for Mapped<'_, '_, '_, '_> {
    type Item = Slot;
    type Error = Error;
    fn next(&mut self) -> Result<Option<Slot>, Error> {
        if self.next == self.length {
            return Ok(None);
        }
        let input = self.engine.workspace.values[self.base];
        let value = self.engine.iterator_item(input, self.next)?;
        self.next += 1;
        Ok(Some(value))
    }
    fn transform(&mut self, value: Slot) -> Result<Slot, Error> {
        self.engine.push(value)?;
        self.engine.apply_upper(Some(1))?;
        self.engine.pop()
    }
    fn retain(&mut self, value: Slot) -> Result<(), Error> {
        self.engine.retain_value(self.base + self.next, value)
    }
}
impl Engine<'_, '_, '_> {
    pub(super) fn apply_map_filter(&mut self, arguments: Option<u16>) -> Result<(), Error> {
        if arguments != Some(2) {
            return Err(Error::Geometry);
        }
        let name = self.pop()?;
        let name = self.text_atom(name)?;
        let name = atom_text(
            self.source,
            self.context,
            self.borrowed,
            self.workspace.context_text.as_deref(),
            name,
        )?;
        if name != "upper" {
            return Err(Error::Geometry);
        }
        let input = self.pop()?;
        let (input, length) = self.iterator(input)?;
        let range = self.value_range(length.checked_add(1).ok_or(Error::Overflow)?)?;
        self.retain_value(range.start, input)?;
        self.counts.values = range.start.checked_add(range.len).ok_or(Error::Overflow)?;
        mapping::run(&mut Mapped {
            engine: self,
            base: range.start,
            length,
            next: 0,
        })?;
        self.push(Slot::Sequence(values::Value {
            start: range.start + 1,
            len: length,
            depth: 0,
        }))
    }
}
pub(super) fn control_bytes() -> Option<usize> {
    use std::mem::{size_of, size_of_val};
    let frames = [
        mapping::control_bytes::<Mapped<'_, '_, '_, '_>>()?,
        size_of::<Option<u16>>(),
        size_of::<[Slot; 3]>(),
        size_of::<Atom>(),
        size_of::<&str>(),
        size_of::<(Slot, usize)>(),
        size_of::<Range>(),
        size_of::<Result<&str, Error>>(),
        size_of::<values::Value>(),
    ];
    frames
        .into_iter()
        .try_fold(size_of_val(&frames), usize::checked_add)
}
