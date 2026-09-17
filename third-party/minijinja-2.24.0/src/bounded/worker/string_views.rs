//! Split/strip keep views into the actual retained source or render input.
//! Split separators are source-owned literals; dynamic separator custody and
//! recursive sequence formatting are separate, explicitly refused profiles.
use super::*;

#[derive(Clone, Copy, Debug)]
pub(in crate::bounded) struct SplitView {
    pub value: Atom,
    pub separator: Option<Range>,
    pub length: usize,
    pub limit:Option<usize>,
}
impl Engine<'_, '_, '_> {
    pub(super) fn split_parts(
        &self,
        view: SplitView,
    ) -> Result<crate::filters::SplitParts<'_>, Error> {
        let value = atom_text(
            self.source,
            self.context,
            &self.borrowed,
            self.workspace.context_text.as_deref(),
            view.value,
        )?;
        let separator = view
            .separator
            .map(|range| range.text(&self.source.bytes).ok_or(Error::Geometry))
            .transpose()?;
        Ok(crate::filters::split_parts_limit(value, separator,view.limit))
    }
    pub(super) fn string_split(&mut self, arguments: Option<u16>) -> Result<(), Error> {
        let count = arguments
            .filter(|count| (1..=3).contains(count))
            .ok_or(Error::Geometry)?;
        let limit=if count==3 {
            match scalar(self.pop()?).ok_or(Error::Geometry)? {
                Scalar::None=>None,
                value=>{
                    let value=i64::try_from(primitive::scalar::signed(value).ok_or(Error::Geometry)?).map_err(|_|Error::Geometry)?;
                    if value<0 {None}else{Some(usize::try_from(value).map_err(|_|Error::Overflow)?.checked_add(1).ok_or(Error::Overflow)?)}
                }
            }
        }else{None};
        let separator = if count >= 2 {
            match self.pop()? {
                Slot::Text(Text::Atom(Atom::Source(range))) => Some(range),
                Slot::Undefined | Slot::Scalar(Scalar::None) => None,
                _ => return Err(Error::Geometry),
            }
        } else {
            None
        };
        let input=self.pop()?;
        let value=self.text_atom(input)?;
        let mut view = SplitView {
            value,
            separator,
            length: 0,
            limit,
        };
        // Checked enumeration shares the ordinary iterator, including empty
        // separator behavior. No substring is copied or materialized here.
        view.length = self.split_parts(view)?.try_fold(0usize, |count, _| {
            count.checked_add(1).ok_or(Error::Overflow)
        })?;
        self.push(Slot::Split(view))
    }
    pub(super) fn split_item(&self, view: SplitView, index: Option<usize>) -> Result<Slot, Error> {
        let Some(index) = index.filter(|index| *index < view.length) else {
            return Ok(Slot::Undefined);
        };
        let value = atom_text(
            self.source,
            self.context,
            &self.borrowed,
            self.workspace.context_text.as_deref(),
            view.value,
        )?;
        let part = self.split_parts(view)?.nth(index).ok_or(Error::Geometry)?;
        // Each iterator item is a slice of this exact value, even empty items.
        let start = (part.as_ptr() as usize)
            .checked_sub(value.as_ptr() as usize)
            .ok_or(Error::Geometry)?;
        let end = start.checked_add(part.len()).ok_or(Error::Overflow)?;
        if value.get(start..end) != Some(part) {
            return Err(Error::Geometry);
        }
        Ok(Slot::Text(Text::Atom(slice_atom(
            view.value,
            start,
            part.len(),
        )?)))
    }
    pub(super) fn string_strip(
        &mut self,
        arguments: Option<u16>,
        left: bool,
        right: bool,
    ) -> Result<(), Error> {
        let count = arguments
            .filter(|count| (1..=2).contains(count))
            .ok_or(Error::Geometry)?;
        let characters = if count == 2 {
            match self.pop()? {
                value @ Slot::Text(_) => Some(self.text_atom(value)?),
                Slot::Undefined | Slot::Scalar(Scalar::None) => None,
                _ => return Err(Error::Geometry),
            }
        } else {
            None
        };
        let input=self.pop()?;
        let value=self.text_atom(input)?;
        let text = atom_text(
            self.source,
            self.context,
            &self.borrowed,
            self.workspace.context_text.as_deref(),
            value,
        )?;
        let characters = characters
            .map(|atom| {
                atom_text(
                    self.source,
                    self.context,
                    &self.borrowed,
                    self.workspace.context_text.as_deref(),
                    atom,
                )
            })
            .transpose()?;
        let kept = crate::filters::strip_range(text, characters, left, right);
        self.push(Slot::Text(Text::Atom(slice_atom(
            value,
            kept.start,
            kept.len(),
        )?)))
    }
}
pub(super) fn control_bytes() -> Option<usize> {
    use std::mem::{size_of, size_of_val};
    let controls = [
        size_of::<SplitView>(),
        size_of::<(Option<u16>, u16, Slot, Atom, Option<Atom>)>(),
        size_of::<(Slot, SplitView, Option<usize>, usize, usize)>(),
        size_of::<(Option<Range>, Option<&str>, &str, &str)>(),
        size_of::<(bool, bool, std::ops::Range<usize>)>(),
        size_of::<Result<crate::filters::SplitParts<'_>, Error>>(),
        size_of::<Result<usize, Error>>(),
        size_of::<Result<Slot, Error>>(),
        // Count/index iterator adapters, source comparison and membership's
        // borrowed predicate have no heap destination.
        size_of::<(usize, &str, Option<&str>, bool)>(),
        size_of::<(&str, Result<bool, Error>)>(),
    ];
    controls
        .into_iter()
        .try_fold(size_of_val(&controls), usize::checked_add)?
        .checked_add(crate::filters::string_view_control_bytes()?)
}
