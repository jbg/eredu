//! Immutable generated mappings; construction preserves insertion order and
//! replaces duplicate values before publishing the finished map index.
use super::super::source::MappingProjection;
use super::*;
#[derive(Clone, Copy, Debug)]
pub(super) struct View {
    pub range: Range,
    pub projection: MappingProjection,
    pub listed: bool,
}
impl Engine<'_, '_, '_> {
    fn object_pair(&self, range: Range, index: usize) -> Result<(Slot, Slot), Error> {
        if index >= range.len {
            return Err(Error::Geometry);
        }
        let start = range
            .start
            .checked_add(index.checked_mul(2).ok_or(Error::Overflow)?)
            .ok_or(Error::Overflow)?;
        let pair = self
            .workspace
            .values
            .get(start..start.checked_add(2).ok_or(Error::Overflow)?)
            .ok_or(Error::Geometry)?;
        Ok((pair[0], pair[1]))
    }
    pub(super) fn object_position(
        &mut self,
        range: Range,
        key: Slot,
    ) -> Result<Option<usize>, Error> {
        for index in 0..range.len {
            let (candidate, _) = self.object_pair(range, index)?;
            let Slot::Bool(equal) = self.eq(candidate, key)? else {
                return Err(Error::Geometry);
            };
            if equal {
                return Ok(Some(index));
            }
        }
        Ok(None)
    }
    pub(super) fn object_get(&mut self, range: Range, key: Slot) -> Result<Slot, Error> {
        match self.object_position(range, key)? {
            Some(index) => Ok(self.object_pair(range, index)?.1),
            None => Ok(Slot::Undefined),
        }
    }
    pub(super) fn object_attr(&self, range: Range, name: &str) -> Result<Slot, Error> {
        for index in 0..range.len {
            let (key, value) = self.object_pair(range, index)?;
            if let Slot::Text(Text::Atom(atom)) = key {
                if atom_text(
                    self.source,
                    self.context,
                    self.borrowed,
                    self.workspace.context_text.as_deref(),
                    atom,
                )? == name
                {
                    return Ok(value);
                }
            }
        }
        Ok(Slot::Undefined)
    }
    pub(super) fn build_object(&mut self, count: usize) -> Result<(), Error> {
        let slots = count.checked_mul(2).ok_or(Error::Overflow)?;
        let operands = self.depth.checked_sub(slots).ok_or(Error::Geometry)?;
        let reserved = self.value_range(slots)?;
        let mut range = Range::new(reserved.start, 0);
        for index in 0..count {
            let position = operands + index * 2;
            let key = self.workspace.operands[position];
            let key=if matches!(key,Slot::Text(_)){Slot::Text(Text::Atom(self.text_atom(key)?))}else{key};
            let value = self.workspace.operands[position + 1];
            // These are the closed scalar key forms. No custom hash/equality
            // object or mutable namespace is accepted as a generated key.
            if scalar(key).is_none() && !matches!(key, Slot::Text(Text::Atom(_))) {
                return Err(Error::Geometry);
            }
            let previous = self.object_position(range, key)?;
            let target = previous.unwrap_or(range.len);
            let destination = range.start + target * 2;
            if previous.is_none() {
                self.retain_value(destination, key)?;
                range.len = range.len.checked_add(1).ok_or(Error::Overflow)?;
            }
            self.retain_value(destination + 1, value)?;
        }
        self.counts.values = reserved
            .start
            .checked_add(reserved.len)
            .ok_or(Error::Overflow)?;
        while self.depth > operands {
            self.pop()?;
        }
        self.push(Slot::Object(range))
    }
    pub(super) fn object_view_item(&self, view: View, index: usize) -> Result<Slot, Error> {
        let (key, value) = self.object_pair(view.range, index)?;
        Ok(match view.projection {
            MappingProjection::Keys => key,
            MappingProjection::Values => value,
            MappingProjection::Items => Slot::Sequence(values::Value {
                start: view
                    .range
                    .start
                    .checked_add(index.checked_mul(2).ok_or(Error::Overflow)?)
                    .ok_or(Error::Overflow)?,
                len: 2,
                depth: 0,
            }),
        })
    }
}
pub(super) fn control_bytes() -> Option<usize> {
    use std::mem::{size_of, size_of_val};
    let frames = [
        size_of::<[Range; 3]>(),
        size_of::<View>(),
        size_of::<[Slot; 4]>(),
        size_of::<[usize; 10]>(),
        size_of::<(Slot, Slot)>(),
        size_of::<Result<(Slot, Slot), Error>>(),
        size_of::<Result<Slot, Error>>(),
        size_of::<Result<Option<usize>, Error>>(),
        size_of::<Option<usize>>(),
        size_of::<(&mut Engine<'_, '_, '_>, Range, Slot)>(),
        size_of::<(&Engine<'_, '_, '_>, Range, &str)>(),
        size_of::<std::ops::Range<usize>>(),
    ];
    frames
        .into_iter()
        .try_fold(size_of_val(&frames), usize::checked_add)
}
