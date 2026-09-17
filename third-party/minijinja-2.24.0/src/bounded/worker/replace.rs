//! Plain UTF-8 replacement into the existing counted generated-text destination.
use super::{generated::Output, measured::Measured, *};
impl Engine<'_, '_, '_> {
    pub(super) fn apply_replace(&mut self, arguments: Option<u16>) -> Result<(), Error> {
        if arguments != Some(3) {
            return Err(Error::Geometry);
        }
        let input=self.pop()?;
        let to=self.text_atom(input)?;
        let input=self.pop()?;
        let from=self.text_atom(input)?;
        let input=self.pop()?;
        let value=self.text_atom(input)?;
        let (start, text_start) = self.generated_start()?;
        let (prefix,destination)=match self.workspace.context_text.as_deref_mut(){
            Some(bytes)=>{let (prefix,tail)=bytes.split_at_mut(text_start);(Some(&*prefix),Some(tail))},
            None=>(None,None),
        };
        let value = atom_text(self.source, self.context, &self.borrowed, prefix, value)?;
        let from = atom_text(self.source, self.context, &self.borrowed, prefix, from)?;
        let to = atom_text(self.source, self.context, &self.borrowed, prefix, to)?;
        let mut output = Output {
            bytes: destination,
            start: 0,
            measured: Measured::default(),
            error: None,
        };
        primitive::replace::plain(&mut output, value, from, to)
            .map_err(|_| output.error.unwrap_or(Error::Geometry))?;
        let measured = output.finish().map_err(|cause|match cause {
            Error::TextCapacity(next)=>text_start.checked_add(next.bytes).map(|bytes|Error::TextCapacity(super::super::render::TextCapacity{atoms:self.counts.concat,bytes})).unwrap_or(Error::Overflow),
            other=>other,
        })?;
        self.finish_generated(start, text_start, measured)
    }
}
pub(super) fn control_bytes() -> Option<usize> {
    use std::mem::{size_of, size_of_val};
    let controls = [
        size_of::<Option<u16>>(),
        size_of::<(Option<&[u8]>,Option<&mut [u8]>)>(),
        size_of::<super::super::render::TextCapacity>(),
        size_of::<[Slot; 3]>(),
        size_of::<[Atom; 3]>(),
        size_of::<[&str; 3]>(),
        size_of::<(usize, usize)>(),
        size_of::<Result<&str, Error>>(),
    ];
    controls
        .into_iter()
        .try_fold(size_of_val(&controls), usize::checked_add)?
        .checked_add(primitive::replace::control_bytes::<Output<'_>>()?)
}
