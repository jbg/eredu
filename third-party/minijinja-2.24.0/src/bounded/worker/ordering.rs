//! Borrowed scalar/string ordering through the ordinary comparison kernels.
use super::*;
use crate::value::ValueKind;
use std::cmp::Ordering;

fn kind(value: Slot) -> Result<ValueKind, Error> {
    if let Some(value) = scalar(value) {
        return Ok(primitive::scalar::kind(value));
    }
    match value {
        Slot::Undefined => Ok(ValueKind::Undefined),
        Slot::Text(_) => Ok(ValueKind::String),
        // The caller supplies paid materialized text only when kinds match.
        // Recursive container ordering needs a separate finite traversal.
        _ => Err(Error::Geometry),
    }
}

impl Engine<'_, '_, '_> {
    pub(super) fn ordered(
        &self,
        left: Slot,
        right: Slot,
        order: shared::Order,
    ) -> Result<Slot, Error> {
        let comparison = if let (Some(left), Some(right)) = (scalar(left), scalar(right)) {
            primitive::scalar::compare(
                left,
                right,
                primitive::scalar::fixed_conversion(left, right),
            )
        } else {
            let comparison = kind(left)?.cmp(&kind(right)?);
            if comparison != Ordering::Equal {
                comparison
            } else {
                match (left, right) {
                    (Slot::Undefined, Slot::Undefined) => Ordering::Equal,
                    (Slot::Text(Text::Atom(left)), Slot::Text(Text::Atom(right))) => {
                        let left = atom_text(
                            self.source,
                            self.context,
                            &self.borrowed,
                            self.workspace.context_text.as_deref(),
                            left,
                        )?;
                        let right = atom_text(
                            self.source,
                            self.context,
                            &self.borrowed,
                            self.workspace.context_text.as_deref(),
                            right,
                        )?;
                        left.cmp(right)
                    }
                    _ => return Err(Error::Geometry),
                }
            }
        };
        Ok(Slot::Bool(order.accepts(comparison)))
    }
}

pub(super) fn control_bytes() -> Option<usize> {
    use std::mem::size_of;
    primitive::scalar::control_bytes()
        .checked_add(size_of::<(Slot, Slot, shared::Order)>())?
        .checked_add(size_of::<(ValueKind, ValueKind, Ordering)>())?
        .checked_add(size_of::<(Atom, Atom, &str, &str)>())?
        .checked_add(size_of::<[Option<Scalar>; 2]>())?
        .checked_add(size_of::<Result<ValueKind, Error>>())?
        .checked_add(size_of::<Result<Slot, Error>>())
}
