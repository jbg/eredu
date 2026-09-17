//! Selection owns immutable source/value registers in the same fixed arena.
use super::*;
use crate::filters::selection::{self, Driver};
struct Selected<'a, 's, 'i, 'w> {
    engine: &'a mut Engine<'s, 'i, 'w>,
    base: usize,
    length: usize,
    next: usize,
    kept: usize,
    equality: bool,
    attribute: bool,
}
impl Driver for Selected<'_, '_, '_, '_> {
    type Item = Slot;
    type Error = Error;
    fn next(&mut self) -> Result<Option<Slot>, Error> {
        if self.next == self.length {
            return Ok(None);
        }
        let input = self.engine.workspace.values[self.base];
        let value = self.engine.iterator_item(input, self.next)?;
        self.next += 1;
        // The predicate may overwrite the shared temporary loan. Preserve the
        // actual candidate before inspecting its attributes, then retain every
        // selected result in its own arena register.
        self.engine.retain_value(self.base + 3, value)?;
        Ok(Some(self.engine.workspace.values[self.base + 3]))
    }
    fn test(&mut self, value: &Slot) -> Result<bool, Error> {
        let mut value = *value;
        if self.attribute {
            let Slot::Text(Text::Atom(path)) = self.engine.workspace.values[self.base + 2] else {
                return Err(Error::Geometry);
            };
            let mut offset = 0;
            loop {
                let (end, key, done) = {
                    let text = atom_text(
                        self.engine.source,
                        self.engine.context,
                        self.engine.borrowed,
                        self.engine.workspace.context_text.as_deref(),
                        path,
                    )?;
                    let rest = text.get(offset..).ok_or(Error::Geometry)?;
                    let length = rest.find('.').unwrap_or(rest.len());
                    let part = &rest[..length];
                    let key = match part.parse::<usize>() {
                        Ok(index) => Slot::Scalar(Scalar::U64(
                            u64::try_from(index).map_err(|_| Error::Overflow)?,
                        )),
                        Err(_) => Slot::Text(Text::Atom(slice_atom(path, offset, length)?)),
                    };
                    (offset + length, key, length == rest.len())
                };
                value = self.engine.item(value, key)?;
                if done {
                    break;
                }
                offset = end.checked_add(1).ok_or(Error::Overflow)?;
            }
        }
        if self.equality {
            let expected = self.engine.workspace.values[self.base + 1];
            let value = if matches!(value, Slot::Text(_)) {
                Slot::Text(Text::Atom(self.engine.text_atom(value)?))
            } else {
                value
            };
            let Slot::Bool(result) = self.engine.eq(value, expected)? else {
                return Err(Error::Geometry);
            };
            Ok(result)
        } else {
            self.engine.truth(&value)
        }
    }
    fn retain(&mut self, value: Slot) -> Result<(), Error> {
        self.engine.retain_value(self.base + 4 + self.kept, value)?;
        self.kept += 1;
        Ok(())
    }
}
impl Engine<'_, '_, '_> {
    pub(super) fn apply_selection(
        &mut self,
        name: &str,
        arguments: Option<u16>,
    ) -> Result<(), Error> {
        let attribute = matches!(name, "selectattr" | "rejectattr");
        let invert = matches!(name, "reject" | "rejectattr");
        let extra = usize::from(attribute);
        let count = usize::from(arguments.ok_or(Error::Geometry)?);
        let equality = count == extra + 3;
        if count != extra + 1 && !equality {
            return Err(Error::Geometry);
        }
        let expected = if equality {
            let value = self.pop()?;
            let value = if matches!(value, Slot::Text(_)) {
                Slot::Text(Text::Atom(self.text_atom(value)?))
            } else {
                value
            };
            let test = self.pop()?;
            let test = self.text_atom(test)?;
            let name = atom_text(
                self.source,
                self.context,
                self.borrowed,
                self.workspace.context_text.as_deref(),
                test,
            )?;
            if !matches!(name, "equalto" | "eq" | "==") {
                return Err(Error::Geometry);
            }
            value
        } else {
            Slot::Undefined
        };
        let path = if attribute {
            let value = self.pop()?;
            Slot::Text(Text::Atom(self.text_atom(value)?))
        } else {
            Slot::Undefined
        };
        let input = self.pop()?;
        let (input, length) = self.iterator(input)?;
        let range = self.value_range(length.checked_add(4).ok_or(Error::Overflow)?)?;
        self.retain_value(range.start, input)?;
        self.retain_value(range.start + 1, expected)?;
        self.retain_value(range.start + 2, path)?;
        self.counts.values = range.start.checked_add(range.len).ok_or(Error::Overflow)?;
        let mut driver = Selected {
            engine: self,
            base: range.start,
            length,
            next: 0,
            kept: 0,
            equality,
            attribute,
        };
        selection::run(&mut driver, invert)?;
        let count = driver.kept;
        self.push(Slot::Sequence(values::Value {
            start: range.start + 4,
            len: count,
            depth: 0,
        }))
    }
}
pub(super) fn control_bytes() -> Option<usize> {
    use std::mem::{size_of, size_of_val};
    let parts = [
        selection::control_bytes::<Selected<'_, '_, '_, '_>>()?,
        size_of::<[Slot; 6]>(),
        size_of::<[usize; 8]>(),
        size_of::<[bool; 3]>(),
        size_of::<Range>(),
        size_of::<Atom>(),
        size_of::<(&str, &str, &str)>(),
        size_of::<Result<usize, std::num::ParseIntError>>(),
        size_of::<(&mut Engine<'_, '_, '_>, &str, Option<u16>)>(),
    ];
    parts
        .into_iter()
        .try_fold(size_of_val(&parts), usize::checked_add)
}
