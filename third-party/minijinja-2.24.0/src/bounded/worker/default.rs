//! Built-in default selection on the same three paid operand positions.
use super::*;
impl Engine<'_, '_, '_> {
    pub(super) fn apply_default(&mut self, arguments: Option<u16>) -> Result<(), Error> {
        let arguments = arguments
            .filter(|n| (1..=3).contains(n))
            .ok_or(Error::Geometry)?;
        let lax = if arguments == 3 {
            let value = self.pop()?;
            self.truth(&value)?
        } else {
            false
        };
        let fallback = if arguments >= 2 {
            self.pop()?
        } else {
            // Ordinary default's implicit empty string needs no new payload.
            Slot::Text(Text::Atom(Atom::Source(Range::new(0, 0))))
        };
        let value = self.pop()?;
        let selected =
            primitive::default::use_fallback(matches!(value, Slot::Undefined), lax, || {
                self.truth(&value)
            })?;
        // push rebinds the chosen borrowed operand before reusing its slot.
        self.push(if selected { fallback } else { value })
    }
}
pub(super) fn control_bytes() -> Option<usize> {
    use std::mem::{size_of, size_of_val};
    let controls = [
        size_of::<Option<u16>>(),
        size_of::<u16>(),
        size_of::<[Slot; 3]>(),
        size_of::<(bool, bool)>(),
        size_of::<(&Engine<'_, '_, '_>, &Slot)>(),
        size_of::<Result<bool, Error>>(),
        size_of::<Result<(), Error>>(),
        size_of::<(Atom, Range)>(),
    ];
    controls
        .into_iter()
        .try_fold(size_of_val(&controls), usize::checked_add)
}
