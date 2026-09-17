//! Immutable generated values in the admitted arena. References remain indices;
//! each borrowed input retained in a value owns a distinct temporary register.
use super::*;
/// The materialization spelling is observable through the ordinary type tests.
/// Values are physically retained in the bounded arena in either case.
#[derive(Clone,Copy,Debug)]
pub(super) struct Value {
    pub start:usize,
    pub len:usize,
    pub depth:usize,
}
impl Value {
    fn materialized(range:Range)->Self {Self{start:range.start,len:range.len,depth:0}}
}

impl Engine<'_, '_, '_> {
    pub(super) fn value_range(&self, count: usize) -> Result<Range, Error> {
        let start = self.counts.values;
        let end = start.checked_add(count).ok_or(Error::Overflow)?;
        if end > self.workspace.values.len() {
            return Err(Error::ValueCapacity(super::super::render::ValueCapacity {
                slots: end,
            }));
        }
        Ok(Range::new(start, count))
    }
    pub(super) fn retain_value(&mut self, position: usize, value: Slot) -> Result<(), Error> {
        if matches!(
            value,
            Slot::Namespace(_)
                | Slot::Kwargs(_)
                | Slot::Macro(_)
                | Slot::Closure(_)
                | Slot::Arguments { .. }
                | Slot::Loop(_)
        ) {
            return Err(Error::Geometry);
        }
        let register = self
            .workspace
            .value_register_start
            .checked_add(position)
            .ok_or(Error::Overflow)?;
        let value = self.borrowed.push(value, register)?;
        *self
            .workspace
            .values
            .get_mut(position)
            .ok_or(Error::Geometry)? = value;
        Ok(())
    }
    pub(super) fn build_sequence(&mut self, count: usize) -> Result<(), Error> {
        let operand_start = self.depth.checked_sub(count).ok_or(Error::Geometry)?;
        let range = self.value_range(count)?;
        for index in 0..count {
            let value = self.workspace.operands[operand_start + index];
            self.retain_value(range.start + index, value)?;
        }
        self.counts.values = range.start.checked_add(range.len).ok_or(Error::Overflow)?;
        while self.depth > operand_start {
            self.pop()?;
        }
        self.push(Slot::Sequence(Value::materialized(range)))
    }
    pub(super) fn sequence_item(&self, range: Value, index: usize) -> Result<Slot, Error> {
        if index >= range.len {
            return Ok(Slot::Undefined);
        }
        self.workspace
            .values
            .get(range.start.checked_add(index).ok_or(Error::Overflow)?)
            .copied()
            .ok_or(Error::Geometry)
    }
    pub(super) fn concat_sequence(&mut self, left: Slot, right: Slot) -> Result<Slot, Error> {
        let as_range = |value| match value {
            Slot::Sequence(range) => Ok(range),
            Slot::EmptySequence => Ok(Value::materialized(Range::new(0, 0))),
            _ => Err(Error::Geometry),
        };
        let left = as_range(left)?;
        let right = as_range(right)?;
        let range = self.value_range(left.len.checked_add(right.len).ok_or(Error::Overflow)?)?;
        let mut position = range.start;
        for source in [left, right] {
            for index in 0..source.len {
                let value = self.sequence_item(source, index)?;
                self.retain_value(position, value)?;
                position = position.checked_add(1).ok_or(Error::Overflow)?;
            }
        }
        self.counts.values = position;
        let depth=left.depth.max(right.depth).checked_add(1).ok_or(Error::Overflow)?;
        let depth=if depth>crate::value::merge_object::MergeSeq::MAX_DEPTH {0}else{depth};
        Ok(Slot::Sequence(Value{start:range.start,len:range.len,depth}))
    }
}
pub(super) fn control_bytes() -> Option<usize> {
    use std::mem::{size_of, size_of_val};
    let frames = [
        size_of::<[Range; 4]>(),
        size_of::<[Value; 3]>(),
        size_of::<[Slot; 3]>(),
        size_of::<[usize; 8]>(),
        size_of::<super::super::render::ValueCapacity>(),
        size_of::<Result<Range, Error>>(),
        size_of::<Result<Slot, Error>>(),
        size_of::<Result<(), Error>>(),
        size_of::<std::ops::Range<usize>>(),
        size_of::<std::array::IntoIter<Value, 2>>(),
        size_of::<(&mut Engine<'_, '_, '_>, usize, Slot)>(),
    ];
    frames
        .into_iter()
        .try_fold(size_of_val(&frames), usize::checked_add)
}
