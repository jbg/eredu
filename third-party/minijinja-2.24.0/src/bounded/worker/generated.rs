//! Existing generated-text byte destination shared by text-producing filters.
use super::{measured::Measured, *};
use std::fmt::{self, Write};
pub(super) struct Output<'a> {
    pub(super) bytes: Option<&'a mut [u8]>,
    pub(super) start: usize,
    pub(super) measured: Measured,
    pub(super) error: Option<Error>,
}
impl Output<'_> {
    pub(super) fn finish(self) -> Result<Measured,Error> {
        let end=self.start.checked_add(self.measured.bytes).ok_or(Error::Overflow)?;
        if self.bytes.as_ref().is_some_and(|bytes|bytes.len()<end) {
            return Err(Error::TextCapacity(super::super::render::TextCapacity {atoms:0,bytes:end}));
        }
        Ok(self.measured)
    }
}
impl Write for Output<'_> {
    fn write_str(&mut self, text: &str) -> fmt::Result {
        let result: Result<(), Error> = (|| {
            let next = self.measured.append(Measured::text(text))?;
            let begin = self
                .start
                .checked_add(self.measured.bytes)
                .ok_or(Error::Overflow)?;
            let end = self.start.checked_add(next.bytes).ok_or(Error::Overflow)?;
            if let Some(bytes) = self.bytes.as_deref_mut() {
                if let Some(destination)=bytes.get_mut(begin..end) {
                    destination.copy_from_slice(text.as_bytes());
                }
            }
            self.measured = next;
            Ok(())
        })();
        result.map_err(|error| {
            self.error = Some(error);
            fmt::Error
        })
    }
}

#[cfg(feature="json")]
impl crate::bounded::json_format::Sink for Output<'_> {
    fn repeat(&mut self,text:&str,count:usize)->fmt::Result {
        let result:Result<(),Error>=(||{
            let mut count_left=count;
            let mut block=Measured::text(text);
            let mut repeated=Measured::default();
            while count_left!=0 {
                if count_left&1!=0 {repeated=repeated.append(block)?;}
                count_left>>=1;
                if count_left!=0 {block=block.append(block)?;}
            }
            let next=self.measured.append(repeated)?;
            let begin=self.start.checked_add(self.measured.bytes).ok_or(Error::Overflow)?;
            let end=self.start.checked_add(next.bytes).ok_or(Error::Overflow)?;
            if let Some(bytes)=self.bytes.as_deref_mut() {
                if let Some(destination)=bytes.get_mut(begin..end) {
                    if !text.is_empty(){for chunk in destination.chunks_exact_mut(text.len()){chunk.copy_from_slice(text.as_bytes());}}
                }
            }
            self.measured=next;Ok(())
        })();
        result.map_err(|error|{self.error=Some(error);fmt::Error})
    }
}

impl Engine<'_, '_, '_> {
    pub(super) fn generated_start(&self) -> Result<(usize, usize), Error> {
        let start = self.counts.concat;
        start.checked_add(1).ok_or(Error::Overflow)?;
        Ok((start, self.counts.context_text))
    }
    pub(super) fn finish_generated(
        &mut self,
        start: usize,
        text_start: usize,
        measured: Measured,
    ) -> Result<(), Error> {
        let value = self.generated_value(start, text_start, measured)?;
        self.push(value)
    }
    /// Finish the text producer independently of the VM operand destination.
    /// Internal consumers can retain its atom while the operand stack is full.
    pub(super) fn generated_value(
        &mut self,
        start: usize,
        text_start: usize,
        measured: Measured,
    ) -> Result<Slot, Error> {
        let end = text_start
            .checked_add(measured.bytes)
            .ok_or(Error::Overflow)?;
        let next = start.checked_add(1).ok_or(Error::Overflow)?;
        if let Some(atoms) = self.workspace.concat.as_deref_mut() {
            *atoms.get_mut(start).ok_or(Error::TextCapacity(super::super::render::TextCapacity {atoms:next,bytes:end}))? =
                Atom::Owned(Range::new(text_start, measured.bytes));
        }
        self.counts.context_text = end;
        self.counts.concat = next;
        Ok(Slot::Text(measured.concat(start, 1)))
    }
}
pub(super) fn control_bytes() -> Option<usize> {
    use std::mem::{size_of, size_of_val};
    let parts = [
        size_of::<Output<'_>>(),
        size_of::<Measured>(),
        size_of::<Result<(), Error>>(),
        size_of::<(Measured, usize, usize)>(),
        size_of::<Result<Measured, Error>>(),
        size_of::<Result<Slot, Error>>(),
        size_of::<Slot>(),
        size_of::<(Option<&mut [u8]>, &str)>(),
        size_of::<[usize; 6]>(),
        size_of::<[Measured; 3]>(),
        size_of::<std::slice::ChunksExactMut<'_,u8>>(),
        size_of::<Result<(usize, usize), Error>>(),
        size_of::<[u8; 4]>(),
        size_of::<fmt::Result>(),
        size_of::<(&Engine<'_, '_, '_>, usize, usize)>(),
        size_of::<(&mut Engine<'_, '_, '_>, usize, usize, Measured)>(),
    ];
    parts
        .into_iter()
        .try_fold(size_of_val(&parts), usize::checked_add)
}
