//! Read-only projection into the same iterative JSON traversal. Custom nodes
//! are copied handles, never a second object tree or an allocation authority.
use super::*;
#[derive(Clone, Copy, Debug)]
pub(crate) enum Node<'a, V> {
    Json(&'a Value),
    Record(crate::bounded::input::RecordValue<'a>),
    Text(&'a str),
    Custom(V),
}
#[derive(Clone, Copy, Debug)]
pub(crate) enum Key<'a, V> {
    Text(&'a str),
    Custom(V),
}
impl<V: Copy> Key<'_, V> {
    pub(super) fn text<'v, 'a>(&'v self, view: &'v impl View<'a, V>) -> Result<&'v str, Error> {
        match self {
            Self::Text(text) => Ok(text),
            Self::Custom(value) => view.key(*value),
        }
    }
}
pub(crate) enum Projection<'a> {
    Json(&'a Value),
    Record(crate::bounded::input::RecordValue<'a>),
    Container { object: bool, length: usize },
    Scalar,
}
pub(crate) trait View<'a, V: Copy> {
    fn project(&self, value: V) -> Result<Projection<'a>, Error>;
    fn entry(&self, value: V, index: usize) -> Result<(Option<Key<'a, V>>, Node<'a, V>), Error>;
    fn key(&self, value: V) -> Result<&str, Error>;
    fn scalar<W: Sink + ?Sized>(
        &self,
        format: &Format<'_>,
        output: &mut W,
        value: V,
    ) -> Result<(), Error>;
}
pub(super) struct Closed;
impl<'a> View<'a, ()> for Closed {
    fn project(&self, _: ()) -> Result<Projection<'a>, Error> {
        Err(Error::Write(fmt::Error))
    }
    fn entry(&self, _: (), _: usize) -> Result<(Option<Key<'a, ()>>, Node<'a, ()>), Error> {
        Err(Error::Write(fmt::Error))
    }
    fn key(&self, _: ()) -> Result<&str, Error> {
        Err(Error::Write(fmt::Error))
    }
    fn scalar<W: Sink + ?Sized>(&self, _: &Format<'_>, _: &mut W, _: ()) -> Result<(), Error> {
        Err(Error::Write(fmt::Error))
    }
}

impl<'a, V> From<crate::bounded::input::record::ReadValue<'a>> for Node<'a, V> {
    fn from(value: crate::bounded::input::record::ReadValue<'a>) -> Self {
        match value {
            crate::bounded::input::record::ReadValue::Json(value) => Self::Json(value),
            crate::bounded::input::record::ReadValue::Record(value) => Self::Record(value),
        }
    }
}
impl<'a> From<crate::bounded::input::record::ReadValue<'a>> for Projection<'a> {
    fn from(value: crate::bounded::input::record::ReadValue<'a>) -> Self {
        match value {
            crate::bounded::input::record::ReadValue::Json(value) => Self::Json(value),
            crate::bounded::input::record::ReadValue::Record(value) => Self::Record(value),
        }
    }
}
