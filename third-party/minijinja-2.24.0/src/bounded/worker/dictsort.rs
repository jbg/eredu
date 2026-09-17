//! Default dictionary sorting retains original key/value identities and order ties.
use super::*;
use crate::filters::dictsort::{self, Driver};
use std::cmp::Ordering;
struct Sorted<'a, 's, 'i, 'w> {
    engine: &'a mut Engine<'s, 'i, 'w>,
    base: usize,
    len: usize,
}
impl Sorted<'_, '_, '_, '_> {
    fn ordinal(&self, index: usize) -> Result<usize, Error> {
        match self.engine.workspace.values.get(self.base + 1 + index) {
            Some(Slot::Scalar(Scalar::U64(value))) => {
                usize::try_from(*value).map_err(|_| Error::Overflow)
            }
            _ => Err(Error::Geometry),
        }
    }
}
impl Driver for Sorted<'_, '_, '_, '_> {
    type Error = Error;
    fn len(&self) -> usize {
        self.len
    }
    fn compare(&mut self, left: usize, right: usize) -> Result<Ordering, Error> {
        let (left, right) = (self.ordinal(left)?, self.ordinal(right)?);
        let view = self.engine.workspace.values[self.base];
        let pair = self.engine.iterator_item(view, left)?;
        let key = self
            .engine
            .selected_item(pair, Slot::Scalar(Scalar::U64(0)))?;
        let key = self.engine.text_atom(key)?;
        // Mapping item lookup uses the temporary borrowed slot. Retain the left
        // key before looking up the right, including generated object keys.
        self.engine
            .retain_value(self.base + 1 + self.len, Slot::Text(Text::Atom(key)))?;
        let pair = self.engine.iterator_item(view, right)?;
        let key = self
            .engine
            .selected_item(pair, Slot::Scalar(Scalar::U64(0)))?;
        let right_key = self.engine.text_atom(key)?;
        let Slot::Text(Text::Atom(left_key)) =
            self.engine.workspace.values[self.base + 1 + self.len]
        else {
            return Err(Error::Geometry);
        };
        let a = atom_text(
            self.engine.source,
            self.engine.context,
            self.engine.borrowed,
            self.engine.workspace.context_text.as_deref(),
            left_key,
        )?;
        let b = atom_text(
            self.engine.source,
            self.engine.context,
            self.engine.borrowed,
            self.engine.workspace.context_text.as_deref(),
            right_key,
        )?;
        Ok(primitive::case::ascii_fold_cmp(a, b).then_with(|| left.cmp(&right)))
    }
    fn swap(&mut self, left: usize, right: usize) {
        self.engine
            .workspace
            .values
            .swap(self.base + 1 + left, self.base + 1 + right);
    }
}
impl Engine<'_, '_, '_> {
    pub(super) fn apply_dictsort(&mut self, arguments: Option<u16>) -> Result<(), Error> {
        if arguments != Some(1) || cfg!(feature = "unicode") {
            return Err(Error::Geometry);
        }
        let input = self.pop()?;
        let input =
            self.mapping_view(input, super::super::source::MappingProjection::Items, false)?;
        let (input, length) = self.iterator(input)?;
        // One source register, one ordinal per entry, one key comparison loan,
        // and one distinct result loan per actual original pair.
        let range = self.value_range(
            length
                .checked_mul(2)
                .and_then(|n| n.checked_add(2))
                .ok_or(Error::Overflow)?,
        )?;
        self.retain_value(range.start, input)?;
        for index in 0..length {
            self.retain_value(
                range.start + 1 + index,
                Slot::Scalar(Scalar::U64(
                    u64::try_from(index).map_err(|_| Error::Overflow)?,
                )),
            )?;
        }
        self.counts.values = range.start.checked_add(range.len).ok_or(Error::Overflow)?;
        let result_start = range.start + length + 2;
        let mut driver = Sorted {
            engine: self,
            base: range.start,
            len: length,
        };
        dictsort::run(&mut driver)?;
        for index in 0..length {
            let ordinal = driver.ordinal(index)?;
            let input = driver.engine.workspace.values[range.start];
            let pair = driver.engine.iterator_item(input, ordinal)?;
            driver.engine.retain_value(result_start + index, pair)?;
        }
        self.push(Slot::Sequence(values::Value {
            start: result_start,
            len: length,
            depth: 0,
        }))
    }
}
pub(super) fn control_bytes() -> Option<usize> {
    use std::mem::{size_of, size_of_val};
    let frames = [
        dictsort::control_bytes::<Sorted<'_, '_, '_, '_>>()?,
        primitive::case::ascii_fold_control_bytes()?,
        size_of::<[Slot; 5]>(),
        size_of::<[Atom; 3]>(),
        size_of::<[&str; 2]>(),
        size_of::<[usize; 8]>(),
        size_of::<Option<u16>>(),
        size_of::<Range>(),
        size_of::<Result<usize, Error>>(),
        size_of::<Ordering>(),
        size_of::<values::Value>(),
    ];
    frames
        .into_iter()
        .try_fold(size_of_val(&frames), usize::checked_add)
}
