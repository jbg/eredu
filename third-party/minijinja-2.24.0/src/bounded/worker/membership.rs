//! Membership borrows actual source/context containers; no Value or iterator
//! heap is created; generated text first uses its paid contiguous materializer.
use super::*;

fn scalar_needle(value: Slot) -> bool {
    scalar(value).is_some() || matches!(value, Slot::Undefined | Slot::Text(Text::Atom(_)))
}
impl Engine<'_, '_, '_> {
    fn membership_text(&self, value: Slot) -> Result<&str, Error> {
        let Slot::Text(Text::Atom(atom)) = value else {
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
    pub(super) fn membership(&mut self, container: Slot, value: Slot) -> Result<Slot, Error> {
        let value = if matches!(value, Slot::Text(_)) {
            Slot::Text(Text::Atom(self.text_atom(value)?))
        } else {
            value
        };
        let container = if matches!(container, Slot::Text(_)) {
            Slot::Text(Text::Atom(self.text_atom(container)?))
        } else {
            container
        };
        let result = match container {
            Slot::Undefined | Slot::EmptySequence => false,
            Slot::Object(range) => self.object_position(range, value)?.is_some(),
            Slot::Sequence(range) => {
                let mut found = false;
                for index in 0..range.len {
                    let item = self.sequence_item(range, index)?;
                    let Slot::Bool(equal) = self.eq(value, item)? else {
                        return Err(Error::Geometry);
                    };
                    if equal {
                        found = true;
                        break;
                    }
                }
                found
            }
            Slot::Mapping(view) => self.mapping_membership(view, value)?,
            Slot::Range(range) => {
                primitive::membership::sequence_contains(range.plan.iter(), |item| {
                    let Slot::Bool(equal) =
                        self.eq(value, Slot::Scalar(Scalar::I64(item as i64)))?
                    else {
                        return Err(Error::Geometry);
                    };
                    Ok(equal)
                })?
            }
            Slot::Text(Text::Atom(_)) => primitive::membership::text_contains(
                self.membership_text(container)?,
                self.membership_text(value)?,
            ),
            Slot::Message(index) => {
                if self.context.messages().get(index).is_none() {
                    return Err(Error::Geometry);
                }
                match value {
                    Slot::Text(Text::Atom(_)) => {
                        matches!(self.membership_text(value)?, "role" | "content")
                    }
                    value if scalar_needle(value) => false,
                    _ => return Err(Error::Geometry),
                }
            }
            Slot::Split(view) => match value {
                Slot::Text(Text::Atom(_)) => {
                    let needle = self.membership_text(value)?;
                    primitive::membership::sequence_contains(self.split_parts(view)?, |part| {
                        Ok::<bool, Error>(part == needle)
                    })?
                }
                value if scalar_needle(value) => false,
                _ => return Err(Error::Geometry),
            },
            Slot::Messages if scalar_needle(value) => false, // every element is a map
            #[cfg(feature = "json")]
            Slot::Structured { register, .. } => {
                let actual = self.borrowed.get(register)?;
                match actual.kind() {
                    crate::bounded::input::record::ReadKind::Object(values) => match value {
                        Slot::Text(Text::Atom(_)) => {
                            values.get(self.membership_text(value)?).is_some()
                        }
                        value if scalar_needle(value) => false,
                        _ => return Err(Error::Geometry),
                    },
                    crate::bounded::input::record::ReadKind::Array(values)
                        if scalar_needle(value) =>
                    {
                        primitive::membership::sequence_contains(values.iter(), |item| {
                            let selected = self.selected(Some(item))?;
                            let Slot::Bool(equal) = self.eq(value, selected)? else {
                                return Err(Error::Geometry);
                            };
                            Ok(equal)
                        })?
                    }
                    _ => return Err(Error::Geometry),
                }
            }
            _ => return Err(Error::Geometry),
        };
        Ok(Slot::Bool(result))
    }
}
pub(super) fn control_bytes() -> Option<usize> {
    use std::mem::size_of;
    let total = primitive::membership::text_control_bytes()
        .checked_add(primitive::scalar::control_bytes())?
        .checked_add(size_of::<(Slot, Slot, bool)>())?
        .checked_add(size_of::<[Slot; 2]>())?
        .checked_add(size_of::<(&mut Engine<'_, '_, '_>, Slot)>())?
        .checked_add(size_of::<(Atom, &str)>())?
        .checked_add(size_of::<Result<bool, Error>>())?
        .checked_add(size_of::<Result<Slot, Error>>())?;
    #[cfg(feature = "json")]
    let total = total
        .checked_add(crate::bounded::input::RecordValue::control_bytes()?)?
        .checked_add(size_of::<std::slice::Iter<'_, serde_json::Value>>())?
        .checked_add(size_of::<Option<&serde_json::Value>>())?
        .checked_add(size_of::<(
            &serde_json::Value,
            &serde_json::Map<String, serde_json::Value>,
        )>())?;
    Some(total)
}
