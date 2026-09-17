//! Structural comparison uses the ordinary driver and admitted value continuations.
use super::*;
use crate::value::structural::{self, Next, Storage as CompareStorage};
struct Driver<'a, 's, 'i, 'w> {
    engine: &'a mut Engine<'s, 'i, 'w>,
    top: Option<usize>,
}
impl Driver<'_, '_, '_, '_> {
    fn number(&self, index: usize) -> Result<usize, Error> {
        match self.engine.workspace.values.get(index) {
            Some(Slot::Scalar(Scalar::U64(value))) => {
                usize::try_from(*value).map_err(|_| Error::Overflow)
            }
            _ => Err(Error::Geometry),
        }
    }
    fn number_at(&mut self, index: usize, value: usize) -> Result<(), Error> {
        self.engine.retain_value(
            index,
            Slot::Scalar(Scalar::U64(
                u64::try_from(value).map_err(|_| Error::Overflow)?,
            )),
        )
    }
    fn parent(&self, index: usize) -> Result<Option<usize>, Error> {
        let parent = match self.engine.workspace.values[index + 2] {
            Slot::Undefined => None,
            Slot::Scalar(Scalar::U64(value)) => {
                Some(usize::try_from(value).map_err(|_| Error::Overflow)?)
            }
            _ => return Err(Error::Geometry),
        };
        if parent.is_some_and(|parent| parent >= index) {
            return Err(Error::Geometry);
        }
        Ok(parent)
    }
    fn shape(&self, value: Slot) -> Result<Option<bool>, Error> {
        Ok(match value {
            Slot::Object(_) | Slot::Message(_) => Some(true),
            #[cfg(feature = "json")]
            Slot::Structured { register, .. } => {
                Some(self.engine.borrowed.get(register)?.is_object())
            }
            Slot::EmptySequence
            | Slot::Sequence(_)
            | Slot::Messages
            | Slot::ObjectView(_)
            | Slot::Mapping(_)
            | Slot::MappingPair(_)
            | Slot::Range(_)
            | Slot::Slice(_)
            | Slot::Split(_) => Some(false),
            _ => None,
        })
    }
    fn same(&self, left: Slot, right: Slot) -> Result<bool, Error> {
        Ok(match (left, right) {
            #[cfg(feature = "json")]
            (Slot::Structured { register: a, .. }, Slot::Structured { register: b, .. }) => self
                .engine
                .borrowed
                .get(a)?
                .same_source(self.engine.borrowed.get(b)?),
            (Slot::Object(a), Slot::Object(b)) => a.start == b.start && a.len == b.len,
            (Slot::Sequence(a), Slot::Sequence(b)) => a.start == b.start && a.len == b.len,
            (Slot::Messages, Slot::Messages) => true,
            (Slot::Message(a), Slot::Message(b)) => a == b,
            _ => false,
        })
    }
}
impl CompareStorage for Driver<'_, '_, '_, '_> {
    type Value = Slot;
    type Error = Error;
    fn open(&mut self, left: Slot, right: Slot) -> Result<bool, Error> {
        match (self.shape(left)?, self.shape(right)?) {
            (None, None) => {
                let Slot::Bool(different) = self.engine.leaf_ne(left, right)? else {
                    return Err(Error::Geometry);
                };
                return Ok(!different);
            }
            (Some(a), Some(b)) if a == b => {}
            _ => return Ok(false),
        }
        if self.same(left, right)? {
            return Ok(true);
        }
        let length = self.engine.value_length(left)?;
        if length != self.engine.value_length(right)? {
            return Ok(false);
        }
        if length == 0 {
            return Ok(true);
        }
        // Two retained containers, parent, next index, key and left child.
        // The right child remains in the existing result register until open.
        let range = self.engine.value_range(6)?;
        self.engine.retain_value(range.start, left)?;
        self.engine.retain_value(range.start + 1, right)?;
        let parent = match self.top {
            Some(index) => Slot::Scalar(Scalar::U64(
                u64::try_from(index).map_err(|_| Error::Overflow)?,
            )),
            None => Slot::Undefined,
        };
        self.engine.retain_value(range.start + 2, parent)?;
        self.number_at(range.start + 3, 0)?;
        self.engine.counts.values = range.start.checked_add(6).ok_or(Error::Overflow)?;
        self.top = Some(range.start);
        Ok(true)
    }
    fn next(&mut self) -> Result<Next<Slot>, Error> {
        loop {
            let Some(frame) = self.top else {
                return Ok(Next::Done);
            };
            let left = self.engine.workspace.values[frame];
            let right = self.engine.workspace.values[frame + 1];
            let index = self.number(frame + 3)?;
            let length = self.engine.value_length(left)?;
            if index == length {
                self.top = self.parent(frame)?;
                continue;
            }
            if index > length {
                return Err(Error::Geometry);
            }
            self.number_at(frame + 3, index + 1)?;
            let (a, b) = if self.shape(left)? == Some(true) {
                let key = if matches!(left, Slot::Message(_)) {
                    self.engine.comparison_message_key(index)?
                } else {
                    let (keys, _) = self.engine.iterator(left)?;
                    self.engine.iterator_item(keys, index)?
                };
                self.engine.retain_value(frame + 4, key)?;
                let key = self.engine.workspace.values[frame + 4];
                let Slot::Bool(present) = self.engine.membership(right, key)? else {
                    return Err(Error::Geometry);
                };
                if !present {
                    return Ok(Next::Different);
                }
                let a = self.engine.item(left, key)?;
                self.engine.retain_value(frame + 5, a)?;
                let b = self.engine.item(right, key)?;
                (self.engine.workspace.values[frame + 5], b)
            } else {
                let a = self.engine.iterator_item(left, index)?;
                self.engine.retain_value(frame + 5, a)?;
                let b = self.engine.iterator_item(right, index)?;
                (self.engine.workspace.values[frame + 5], b)
            };
            return Ok(Next::Pair(a, b));
        }
    }
}
impl Engine<'_, '_, '_> {
    pub(super) fn structural_equal(&mut self, left: Slot, right: Slot) -> Result<bool, Error> {
        structural::equal(
            &mut Driver {
                engine: self,
                top: None,
            },
            left,
            right,
        )
    }
    fn comparison_message_key(&mut self, index: usize) -> Result<Slot, Error> {
        use std::fmt::Write;
        let text = match index {
            0 => "role",
            1 => "content",
            _ => return Err(Error::Geometry),
        };
        let (start, text_start) = self.generated_start()?;
        let mut output = generated::Output {
            bytes: self.workspace.context_text.as_deref_mut(),
            start: text_start,
            measured: measured::Measured::default(),
            error: None,
        };
        if output.write_str(text).is_err() {
            return Err(output.error.unwrap_or(Error::Geometry));
        }
        let measured = output.finish()?;
        self.generated_value(start, text_start, measured)
    }
    pub(super) fn leaf_ne(&mut self, left: Slot, right: Slot) -> Result<Slot, Error> {
        let (left, right) = if matches!((left, right), (Slot::Text(_), Slot::Text(_))) {
            (
                Slot::Text(Text::Atom(self.text_atom(left)?)),
                Slot::Text(Text::Atom(self.text_atom(right)?)),
            )
        } else {
            (left, right)
        };
        if let (Some(left), Some(right)) = (scalar(left), scalar(right)) {
            return Ok(Slot::Bool(!primitive::scalar::equal(
                left,
                right,
                primitive::scalar::fixed_conversion(left, right),
            )));
        }
        match (left, right) {
            (Slot::Text(Text::Atom(left)), Slot::Text(Text::Atom(right))) => Ok(Slot::Bool(
                atom_text(
                    self.source,
                    self.context,
                    &self.borrowed,
                    self.workspace.context_text.as_deref(),
                    left,
                )? != atom_text(
                    self.source,
                    self.context,
                    &self.borrowed,
                    self.workspace.context_text.as_deref(),
                    right,
                )?,
            )),
            (Slot::Undefined, Slot::Undefined) => Ok(Slot::Bool(false)),
            (Slot::Undefined, _) | (_, Slot::Undefined) => Ok(Slot::Bool(true)),
            (Slot::Structured { .. }, other) | (other, Slot::Structured { .. })
                if scalar(other).is_some() || matches!(other, Slot::Text(_)) =>
            {
                Ok(Slot::Bool(true))
            }
            (Slot::Text(_), other) | (other, Slot::Text(_)) if scalar(other).is_some() => {
                Ok(Slot::Bool(true))
            }
            _ => Err(Error::Geometry),
        }
    }
}
pub(super) fn control_bytes() -> Option<usize> {
    use std::mem::{size_of, size_of_val};
    let frames = [
        size_of::<Driver<'_, '_, '_, '_>>(),
        size_of::<Next<Slot>>(),
        size_of::<[Slot; 10]>(),
        size_of::<[usize; 10]>(),
        size_of::<[Option<usize>; 3]>(),
        size_of::<[Option<bool>; 2]>(),
        size_of::<Range>(),
        size_of::<Result<bool, Error>>(),
        size_of::<Result<Next<Slot>, Error>>(),
        size_of::<Result<(Slot, usize), Error>>(),
        size_of::<Result<Option<usize>, Error>>(),
        size_of::<(&mut Engine<'_, '_, '_>, Slot, Slot)>(),
        size_of::<(&str, measured::Measured)>(),
    ];
    // Generated-map key lookup can reenter equality once. Closed keys are
    // scalars/text, so that inner comparison cannot open another container.
    frames
        .into_iter()
        .try_fold(size_of_val(&frames), usize::checked_add)?
        .checked_mul(2)
}
