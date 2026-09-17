//! Retained exact ordinary range facts on existing operand/loop/local storage.
use super::*;
#[derive(Clone, Copy, Debug)]
pub(super) struct Value {
    pub(super) plan: primitive::range::Range,
    pub(super) listed: bool,
}
impl Value {
    pub(super) fn item(self, index: usize) -> Result<Slot, Error> {
        Ok(self.plan.get(index).map_or(Slot::Undefined, |value| {
            Slot::Scalar(Scalar::I64(value as i64))
        }))
    }
}
fn integer(value: Slot) -> Result<isize, Error> {
    scalar(value)
        .and_then(primitive::scalar::signed)
        .and_then(|value| isize::try_from(value).ok())
        .ok_or(Error::Geometry)
}
fn optional(value: Slot) -> Result<Option<isize>, Error> {
    if matches!(value, Slot::Undefined | Slot::Scalar(Scalar::None)) {
        Ok(None)
    } else {
        integer(value).map(Some)
    }
}
impl Engine<'_, '_, '_> {
    pub(super) fn range_call(&mut self, arguments: Option<u16>) -> Result<(), Error> {
        if !cfg!(feature = "builtins") {
            return Err(Error::Geometry);
        }
        let count = usize::from(
            arguments
                .filter(|n| (1..=3).contains(n))
                .ok_or(Error::Geometry)?,
        );
        let mut values = [Slot::Undefined; 3];
        for index in (0..count).rev() {
            values[index] = self.pop()?;
        }
        let lower = integer(values[0])?;
        let upper = optional(values[1])?;
        let step = optional(values[2])?;
        let plan =
            primitive::range::Range::prepare(lower, upper, step).map_err(|_| Error::Geometry)?;
        self.push(Slot::Range(Value {
            plan,
            listed: false,
        }))
    }
}
pub(super) fn control_bytes() -> Option<usize> {
    use std::mem::{size_of, size_of_val};
    let parts = [
        size_of::<[Slot; 3]>(),
        size_of::<Option<u16>>(),
        size_of::<Value>(),
        size_of::<(usize, isize, Option<isize>, Option<isize>)>(),
        size_of::<Result<(), Error>>(),
        size_of::<Result<Option<isize>, Error>>(),
        size_of::<Result<Slot, Error>>(),
        size_of::<std::iter::Rev<std::ops::Range<usize>>>(),
    ];
    parts
        .into_iter()
        .try_fold(size_of_val(&parts), usize::checked_add)?
        .checked_add(primitive::range::control_bytes())
}
