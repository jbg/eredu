//! Filtered iteration retains accepted values in a separately admitted arena.
use super::*;
impl Engine<'_, '_, '_> {
    fn collection_number(&self, index: usize) -> Result<usize, Error> {
        match self.workspace.values.get(index) {
            Some(Slot::Scalar(Scalar::U64(value))) => {
                usize::try_from(*value).map_err(|_| Error::Overflow)
            }
            _ => Err(Error::Geometry),
        }
    }
    fn collection_store_number(&mut self, index: usize, value: usize) -> Result<(), Error> {
        self.retain_value(
            index,
            Slot::Scalar(Scalar::U64(
                u64::try_from(value).map_err(|_| Error::Overflow)?,
            )),
        )
    }
    pub(super) fn start_collection(&mut self) -> Result<(), Error> {
        let input = *self.peek()?;
        let (_, capacity) = self.iterator(input)?;
        let range = self.value_range(capacity.checked_add(4).ok_or(Error::Overflow)?)?;
        let parent = match self.collection {
            Some(index) => Slot::Scalar(Scalar::U64(
                u64::try_from(index).map_err(|_| Error::Overflow)?,
            )),
            None => Slot::Undefined,
        };
        self.retain_value(range.start, parent)?;
        self.collection_store_number(range.start + 1, self.calls)?;
        self.collection_store_number(range.start + 2, capacity)?;
        self.collection_store_number(range.start + 3, 0)?;
        self.counts.values = range.start.checked_add(range.len).ok_or(Error::Overflow)?;
        self.collection = Some(range.start);
        Ok(())
    }
    pub(super) fn collect_value(&mut self, value: Slot) -> Result<(), Error> {
        let index = self.collection.ok_or(Error::Geometry)?;
        if self.collection_number(index + 1)? != self.calls {
            return Err(Error::Geometry);
        }
        let length = self.collection_number(index + 3)?;
        if length >= self.collection_number(index + 2)? {
            return Err(Error::Geometry);
        }
        self.retain_value(index + 4 + length, value)?;
        self.collection_store_number(index + 3, length + 1)
    }
    pub(super) fn finish_collection(&mut self) -> Result<Slot, Error> {
        let index = self.collection.ok_or(Error::Geometry)?;
        if self.collection_number(index + 1)? != self.calls {
            return Err(Error::Geometry);
        }
        let length = self.collection_number(index + 3)?;
        if length > self.collection_number(index + 2)? {
            return Err(Error::Geometry);
        }
        let parent = match self.workspace.values[index] {
            Slot::Undefined => None,
            Slot::Scalar(Scalar::U64(parent)) => {
                Some(usize::try_from(parent).map_err(|_| Error::Overflow)?)
            }
            _ => return Err(Error::Geometry),
        };
        if parent.is_some_and(|parent| parent >= index) {
            return Err(Error::Geometry);
        }
        self.collection = parent;
        Ok(Slot::Sequence(values::Value {
            start: index + 4,
            len: length,
            depth: 0,
        }))
    }
}
pub(super) fn control_bytes() -> Option<usize> {
    use std::mem::{size_of, size_of_val};
    let frames = [
        size_of::<[usize; 10]>(),
        size_of::<[Slot; 4]>(),
        size_of::<Range>(),
        size_of::<[Option<usize>; 2]>(),
        size_of::<Result<usize, Error>>(),
        size_of::<Result<Slot, Error>>(),
        size_of::<Result<(), Error>>(),
        size_of::<Result<(Slot, usize), Error>>(),
        size_of::<(&mut Engine<'_, '_, '_>, usize, Slot)>(),
    ];
    frames
        .into_iter()
        .try_fold(size_of_val(&frames), usize::checked_add)
}
