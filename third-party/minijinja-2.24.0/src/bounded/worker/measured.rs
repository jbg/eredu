//! Shared byte/Unicode/trim metadata for concatenation and generated text.
use super::*;
#[derive(Clone, Copy, Default)]
pub(super) struct Measured {
    pub(super) bytes: usize,
    pub(super) leading: usize,
    pub(super) trailing: usize,
    pub(super) characters: usize,
    pub(super) leading_characters: usize,
    pub(super) trailing_characters: usize,
}
impl Measured {
    pub(super) fn text(text: &str) -> Self {
        let kept = crate::filters::whitespace_trim_range(text);
        Self {
            bytes: text.len(),
            leading: kept.start,
            trailing: text.len() - kept.end,
            characters: primitive::length::Length::Text(text)
                .get()
                .expect("text length exists"),
            leading_characters: primitive::length::Length::Text(&text[..kept.start])
                .get()
                .expect("text length exists"),
            trailing_characters: primitive::length::Length::Text(&text[kept.end..])
                .get()
                .expect("text length exists"),
        }
    }
    pub(super) fn append(self, right: Self) -> Result<Self, Error> {
        let bytes = self.bytes.checked_add(right.bytes).ok_or(Error::Overflow)?;
        let characters = self
            .characters
            .checked_add(right.characters)
            .ok_or(Error::Overflow)?;
        let leading = if self.leading == self.bytes {
            self.bytes
                .checked_add(right.leading)
                .ok_or(Error::Overflow)?
        } else {
            self.leading
        };
        let leading_characters = if self.leading == self.bytes {
            self.characters
                .checked_add(right.leading_characters)
                .ok_or(Error::Overflow)?
        } else {
            self.leading_characters
        };
        let (trailing, trailing_characters) = if leading == bytes {
            (0, 0)
        } else if right.leading == right.bytes {
            (
                right
                    .bytes
                    .checked_add(self.trailing)
                    .ok_or(Error::Overflow)?,
                right
                    .characters
                    .checked_add(self.trailing_characters)
                    .ok_or(Error::Overflow)?,
            )
        } else {
            (right.trailing, right.trailing_characters)
        };
        Ok(Self {
            bytes,
            leading,
            trailing,
            characters,
            leading_characters,
            trailing_characters,
        })
    }
    pub(super) fn concat(self, start: usize, count: usize) -> Text {
        Text::Concat {
            start,
            count,
            skip: 0,
            bytes: self.bytes,
            leading: self.leading,
            trailing: self.trailing,
            characters: self.characters,
            leading_characters: self.leading_characters,
            trailing_characters: self.trailing_characters,
        }
    }
}
impl Engine<'_, '_, '_> {
    pub(super) fn measure_text(&self, text: Text) -> Result<Measured, Error> {
        Ok(match text {
            Text::Atom(atom) => Measured::text(atom_text(
                self.source,
                self.context,
                &self.borrowed,
                self.workspace.context_text.as_deref(),
                atom,
            )?),
            Text::Concat {
                bytes,
                leading,
                trailing,
                characters,
                leading_characters,
                trailing_characters,
                ..
            } => Measured {
                bytes,
                leading,
                trailing,
                characters,
                leading_characters,
                trailing_characters,
            },
        })
    }
}
pub(super) fn control_bytes() -> Option<usize> {
    use std::mem::{size_of, size_of_val};
    let controls = [
        size_of::<[Measured; 3]>(),
        size_of::<Result<Measured, Error>>(),
        size_of::<[usize; 6]>(),
        size_of::<std::str::Chars<'_>>(),
        size_of::<(&str, std::ops::Range<usize>)>(),
        size_of::<(Text, usize, usize)>(),
        size_of::<Result<Text, Error>>(),
    ];
    controls
        .into_iter()
        .try_fold(size_of_val(&controls), usize::checked_add)
}
