//! Exact character counts through borrowed atoms and retained concatenations.
use super::*;
use primitive::length::Length;
impl Engine<'_, '_, '_> {
    pub(super) fn character_edges(&self, text: Text) -> Result<(usize, usize, usize), Error> {
        let measured = self.measure_text(text)?;
        Ok((
            measured.characters,
            measured.leading_characters,
            measured.trailing_characters,
        ))
    }
    pub(super) fn value_length(&self, value: Slot) -> Result<usize, Error> {
        let known = match value {
            Slot::Text(text) => Some(self.character_edges(text)?.0),
            Slot::EmptySequence => Some(0),
            Slot::Sequence(value) => Some(value.len),
            Slot::Object(range)=>Some(range.len),
            Slot::ObjectView(view)=>Some(view.range.len),
            Slot::Range(value) => Some(value.plan.length),
            Slot::Slice(value) => Some(value.plan.length),
            // Ordinary Macro::enumerate exposes name, arguments and caller.
            Slot::Macro(_) => Some(3),
            #[cfg(feature = "chat-clock")]
            Slot::Clock => None,
            Slot::Closure(_) | Slot::Arguments { .. } => return Err(Error::Geometry),
            Slot::Namespace(index) | Slot::Kwargs(index) => Some(self.namespace_len(index)?),
            Slot::Split(view) => Some(view.length),
            Slot::Mapping(view) => Some(view.length),
            Slot::MappingPair(_) => Some(2),
            Slot::Structured { length, .. } => Some(length),
            Slot::Messages => Some(self.context.messages().len()),
            Slot::Message(index) => {
                self.context.messages().get(index).ok_or(Error::Geometry)?;
                Some(2)
            }
            // The ordinary Value rule has no length for undefined, None, bool,
            // or numeric values. Dynamic loop-object enumeration is separate.
            Slot::Undefined | Slot::Bool(_) | Slot::Zero | Slot::Scalar(_) | Slot::Loop(_) => None,
        };
        Length::Known(known).get().ok_or(Error::Geometry)
    }
}
pub(super) fn control_bytes() -> Option<usize> {
    use std::mem::{size_of, size_of_val};
    let controls = [
        size_of::<Length<'_>>(),
        size_of::<std::str::Chars<'_>>(),
        size_of::<(Text, Slot, Atom)>(),
        size_of::<(&str, std::ops::Range<usize>)>(),
        size_of::<Result<(usize, usize, usize), Error>>(),
        size_of::<Option<usize>>(),
        size_of::<Result<usize, Error>>(),
        // The helper's three returned Unicode counts.
        size_of::<[usize; 3]>(),
        size_of::<(Slot, Option<u16>, &str, usize)>(),
        // Trim holds previous counts and the resulting character count.
        size_of::<[usize; 4]>(),
        size_of::<Result<u64, std::num::TryFromIntError>>(),
    ];
    controls
        .into_iter()
        .try_fold(size_of_val(&controls), usize::checked_add)
}
