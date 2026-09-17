//! Same character/collection length rule for ordinary and borrowed operands.
#![forbid(unsafe_code)]
#[derive(Clone, Copy)]
pub(crate) enum Length<'a> {
    Text(&'a str),
    Known(Option<usize>),
}
impl Length<'_> {
    pub(crate) fn get(self) -> Option<usize> {
        match self {
            Self::Text(text) => Some(text.chars().count()),
            Self::Known(length) => length,
        }
    }
}
