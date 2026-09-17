//! One immutable generated-text traversal for plain and escaped consumers.
use super::*;
#[derive(Clone, Copy)]
pub(super) struct Segments<'a> {
    pub source: &'a PreparedTemplate,
    pub context: RenderContext<'a>,
    pub borrowed: &'a Borrowed<'a>,
    pub prefix: Option<&'a [u8]>,
    pub atoms: Option<&'a [Atom]>,
}
impl Segments<'_> {
    pub(super) fn visit(
        &self,
        text: Text,
        mut write: impl FnMut(&str) -> Result<(), Error>,
    ) -> Result<(), Error> {
        match text {
            Text::Atom(atom) => write(atom_text(
                self.source,
                self.context,
                self.borrowed,
                self.prefix,
                atom,
            )?),
            Text::Concat {
                start,
                count,
                bytes,
                skip,
                ..
            } => {
                if count==0 {
                    return if bytes==0 && skip==0 {Ok(())}else{Err(Error::Geometry)};
                }
                let end = start.checked_add(count).ok_or(Error::Overflow)?;
                let atoms = self
                    .atoms
                    .and_then(|values| values.get(start..end))
                    .ok_or(Error::Geometry)?;
                let (mut skip, mut remaining) = (skip, bytes);
                for atom in atoms {
                    let atom = clip_atom(
                        self.source,
                        self.context,
                        self.borrowed,
                        self.prefix,
                        *atom,
                        &mut skip,
                        &mut remaining,
                    )?;
                    write(atom_text(
                        self.source,
                        self.context,
                        self.borrowed,
                        self.prefix,
                        atom,
                    )?)?;
                }
                if skip != 0 || remaining != 0 {
                    return Err(Error::Geometry);
                }
                Ok(())
            }
        }
    }
}
pub(super) fn control_bytes() -> Option<usize> {
    use std::mem::{size_of, size_of_val};
    let parts = [
        size_of::<Segments<'_>>(),
        size_of::<super::super::render::TextCapacity>(),
        size_of::<generated::Output<'_>>(),
        size_of::<measured::Measured>(),
        size_of::<(Option<&[u8]>, Option<&mut [u8]>)>(),
        size_of::<[usize; 5]>(),
        size_of::<Result<Atom, Error>>(),
        size_of::<(&mut Engine<'_, '_, '_>, Slot)>(),
        size_of::<&mut generated::Output<'_>>(),
        size_of::<Text>(),
        size_of::<Atom>(),
        size_of::<[usize; 5]>(),
        size_of::<&str>(),
        size_of::<std::slice::Iter<'_, Atom>>(),
        size_of::<Result<(), Error>>(),
        size_of::<(&Segments<'_>, Text)>(),
        size_of::<Option<&[Atom]>>(),
    ];
    parts
        .into_iter()
        .try_fold(size_of_val(&parts), usize::checked_add)
}

impl Engine<'_, '_, '_> {
    /// A library string consumer needs one contiguous UTF-8 loan. Reuse an
    /// existing one-segment atom; otherwise materialize the actual bytes once
    /// into a counted destination and retain its ordinary generated descriptor.
    pub(super) fn text_atom(&mut self, value: Slot) -> Result<Atom, Error> {
        let Slot::Text(text) = value else {
            return Err(Error::Geometry);
        };
        if let Text::Atom(atom) = text {
            return Ok(atom);
        }
        if matches!(text,Text::Concat{count:0,bytes:0,skip:0,..}) {
            return Ok(Atom::Source(Range::new(0,0)));
        }
        let needed = super::super::render::TextCapacity {
            atoms: self.counts.concat,
            bytes: self.counts.context_text,
        };
        if self.workspace.concat.as_ref().map_or(0, |v| v.len()) < needed.atoms
            || self.workspace.context_text.as_ref().map_or(0, |v| v.len()) < needed.bytes
        {
            return Err(Error::TextCapacity(needed));
        }
        if let Text::Concat {
            start,
            count: 1,
            bytes,
            skip,
            ..
        } = text
        {
            let atom = *self
                .workspace
                .concat
                .as_ref()
                .and_then(|v| v.get(start))
                .ok_or(Error::Geometry)?;
            let (mut skip, mut remaining) = (skip, bytes);
            let atom = clip_atom(
                self.source,
                self.context,
                self.borrowed,
                self.workspace.context_text.as_deref(),
                atom,
                &mut skip,
                &mut remaining,
            )?;
            if skip != 0 || remaining != 0 {
                return Err(Error::Geometry);
            }
            return Ok(atom);
        }
        let (start, text_start) = self.generated_start()?;
        let (prefix, destination) = match self.workspace.context_text.as_deref_mut() {
            Some(bytes) => {
                let (prefix, tail) = bytes.split_at_mut(text_start);
                (Some(&*prefix), Some(tail))
            }
            None => (None, None),
        };
        let segments = Segments {
            source: self.source,
            context: self.context,
            borrowed: self.borrowed,
            prefix,
            atoms: self.workspace.concat.as_deref(),
        };
        let mut output = generated::Output {
            bytes: destination,
            start: 0,
            measured: measured::Measured::default(),
            error: None,
        };
        use std::fmt::Write;
        let result = segments.visit(text, |text| {
            output.write_str(text).map_err(|_| Error::Geometry)
        });
        if let Err(error) = result {
            return Err(match output.error.unwrap_or(error) {
                Error::TextCapacity(next) => {
                    Error::TextCapacity(super::super::render::TextCapacity {
                        atoms: needed.atoms,
                        bytes: text_start.checked_add(next.bytes).ok_or(Error::Overflow)?,
                    })
                }
                other => other,
            });
        }
        let measured = output.finish().map_err(|cause|match cause {
            Error::TextCapacity(next)=>text_start.checked_add(next.bytes)
                .map(|bytes|Error::TextCapacity(super::super::render::TextCapacity{atoms:needed.atoms,bytes}))
                .unwrap_or(Error::Overflow),
            other=>other,
        })?;
        let bytes = measured.bytes;
        self.generated_value(start, text_start, measured)?;
        Ok(Atom::Owned(Range::new(text_start, bytes)))
    }
}
