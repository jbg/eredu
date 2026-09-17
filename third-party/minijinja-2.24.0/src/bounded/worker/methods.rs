//! The installed Python-compatible mapping get, on existing borrowed operands.
use super::*;
impl Engine<'_, '_, '_> {
    pub(super) fn mapping_get(&mut self, arguments: Option<u16>) -> Result<(), Error> {
        let count = arguments
            .filter(|count| (2..=3).contains(count))
            .ok_or(Error::Geometry)?;
        let fallback = if count == 3 {
            self.pop()?
        } else {
            Slot::Scalar(Scalar::None)
        };
        // minijinja-contrib parses the default as Option<Value>, which treats
        // both undefined and None as the implicit None. False/zero/empty survive.
        let fallback = if matches!(fallback, Slot::Undefined) {
            Slot::Scalar(Scalar::None)
        } else {
            fallback
        };
        let key = self.pop()?;
        let value = self.pop()?;
        match value {
            Slot::Object(_) => {},
            Slot::Message(index) if self.context.messages().get(index).is_some() => {}
            #[cfg(feature = "json")]
            Slot::Structured { register, .. } if self.borrowed.get(register)?.is_object() => {}
            _ => return Err(Error::Geometry),
        }
        let selected = self.item(value, key)?;
        // Existing lookup distinguishes absence from an actual JSON null.
        // push moves the selected borrowed result into its paid operand slot.
        self.push(if matches!(selected, Slot::Undefined) {
            fallback
        } else {
            selected
        })
    }
}
fn edge(value: &str, needle: &str, prefix: bool) -> bool {
    if prefix {
        value.starts_with(needle)
    } else {
        value.ends_with(needle)
    }
}
impl Engine<'_, '_, '_> {
    fn method_text(&self, slot: Slot) -> Result<&str, Error> {
        let Slot::Text(Text::Atom(atom)) = slot else {
            return Err(Error::Geometry);
        };
        atom_text(
            self.source,
            self.context,
            &self.borrowed,
            self.workspace.context_text.as_deref(),
            atom,
        )
    }
    pub(super) fn string_edge(
        &mut self,
        arguments: Option<u16>,
        prefix: bool,
    ) -> Result<(), Error> {
        if arguments != Some(2) {
            return Err(Error::Geometry);
        }
        let needle = self.pop()?;
        let value = self.pop()?;
        let value = Slot::Text(Text::Atom(self.text_atom(value)?));
        let matches = match needle {
            Slot::Text(_) => {
                let needle = Slot::Text(Text::Atom(self.text_atom(needle)?));
                edge(self.method_text(value)?, self.method_text(needle)?, prefix)
            }
            Slot::Sequence(range) => {
                let mut found = false;
                for index in 0..range.len {
                    let needle = self.sequence_item(range, index)?;
                    let needle = Slot::Text(Text::Atom(self.text_atom(needle)?));
                    if edge(self.method_text(value)?, self.method_text(needle)?, prefix) {
                        found = true;
                        break;
                    }
                }
                found
            }
            Slot::EmptySequence => false,
            #[cfg(feature = "json")]
            Slot::Structured { register, .. } => {
                let values = self
                    .borrowed
                    .get(register)?
                    .as_array()
                    .ok_or(Error::Geometry)?;
                // The installed compatibility method stops at the first match;
                // a later invalid element is not read after that match.
                let mut found = false;
                for item in values.iter() {
                    let needle = item.as_str().ok_or(Error::Geometry)?;
                    if edge(self.method_text(value)?, needle, prefix) {
                        found = true;
                        break;
                    }
                }
                found
            }
            _ => return Err(Error::Geometry),
        };
        self.push(Slot::Bool(matches))
    }
}
pub(super) fn control_bytes() -> Option<usize> {
    use std::mem::{size_of, size_of_val};
    let controls = [
        size_of::<Option<u16>>(),
        size_of::<u16>(),
        size_of::<[Slot; 4]>(),
        size_of::<values::Value>(),
        size_of::<std::ops::Range<usize>>(),
        size_of::<usize>(),
        size_of::<(&mut Engine<'_, '_, '_>, &str)>(),
        size_of::<Result<(), Error>>(),
        size_of::<Result<Slot, Error>>(),
        size_of::<bool>(),
    ];
    let total = controls
        .into_iter()
        .try_fold(size_of_val(&controls), usize::checked_add)?
        .checked_add(size_of::<(&str, &str, bool, Atom)>())?;
    #[cfg(feature = "json")]
    let total = total
        .checked_add(size_of::<std::slice::Iter<'_, serde_json::Value>>())?
        .checked_add(size_of::<Option<&serde_json::Value>>())?
        .checked_add(size_of::<Result<&str, Error>>())?;
    Some(total)
}
