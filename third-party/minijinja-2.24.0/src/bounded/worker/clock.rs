//! Retained clock facts into the existing admitted generated-text destination.
use super::{generated::Output, measured::Measured, *};
impl Engine<'_, '_, '_> {
    pub(super) fn clock_call(&mut self, arguments: Option<u16>) -> Result<(), Error> {
        if arguments != Some(1) {
            return Err(Error::Geometry);
        }
        let value = self.pop()?;
        if !matches!(value, Slot::Text(_)) {
            return Err(Error::Geometry);
        }
        let format = self.text_atom(value)?;
        let clock = self.context.clock().ok_or(Error::Geometry)?;
        let (start, text_start) = self.generated_start()?;
        let (prefix, destination) = match self.workspace.context_text.as_deref_mut() {
            Some(bytes) => {
                let (prefix, tail) = bytes.split_at_mut(text_start);
                (Some(&*prefix), Some(tail))
            }
            None => (None, None),
        };
        let format = atom_text(self.source, self.context, &self.borrowed, prefix, format)?;
        let mut output = Output {
            bytes: destination,
            start: 0,
            measured: Measured::default(),
            error: None,
        };
        clock
            .write(&mut output, format)
            .map_err(|_| output.error.unwrap_or(Error::Geometry))?;
        let measured = output.finish().map_err(|cause| match cause {
            Error::TextCapacity(next) => text_start
                .checked_add(next.bytes)
                .map(|bytes| {
                    Error::TextCapacity(super::super::render::TextCapacity {
                        atoms: self.counts.concat,
                        bytes,
                    })
                })
                .unwrap_or(Error::Overflow),
            other => other,
        })?;
        self.finish_generated(start, text_start, measured)
    }
}
pub(super) fn control_bytes() -> Option<usize> {
    use std::mem::{size_of, size_of_val};
    let frames = [
        size_of::<Option<u16>>(),
        size_of::<Slot>(),
        size_of::<Atom>(),
        size_of::<(&str, usize, usize)>(),
        size_of::<(Option<&[u8]>, Option<&mut [u8]>)>(),
        size_of::<super::super::render::TextCapacity>(),
        size_of::<Result<&str, Error>>(),
    ];
    frames
        .into_iter()
        .try_fold(size_of_val(&frames), usize::checked_add)?
        .checked_add(super::super::clock::Snapshot::control_bytes::<Output<'_>>()?)
}
