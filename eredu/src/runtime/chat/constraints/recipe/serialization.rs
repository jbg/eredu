//! Borrowed JSON serialization into a measured, already-funded destination.
use llguidance::derivre::{ParserAllocationFunding, ParserStorageError};
use serde::Serialize;
use serde_json::Value;
use std::{io::{self, Write}, mem::{size_of, size_of_val}};

#[derive(Debug, thiserror::Error)]
pub(super) enum Cause {
    #[error("constraint recipe extent overflow")]
    Overflow,
    #[error("constraint recipe structural token names and IDs differ in length")]
    Structural,
    #[error("constraint recipe serialization changed its measured extent")]
    Destination,
    #[error(transparent)]
    Funding(#[from] llguidance::derivre::ParserAllocationFailure),
    #[error(transparent)]
    Storage(#[from] ParserStorageError),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

#[derive(Default)]
pub(super) struct Count(usize);
impl Count {
    pub(super) fn bytes(&self) -> usize { self.0 }
}
impl Write for Count {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.0 = self.0.checked_add(bytes.len())
            .ok_or_else(|| io::Error::from(io::ErrorKind::OutOfMemory))?;
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> { Ok(()) }
}

pub(super) fn value<W: Write, T: Serialize + ?Sized>(
    writer: &mut W, value: &T, funding: &ParserAllocationFunding,
) -> Result<(), Cause> {
    // The error destination is paid before serde can wrap a fixed I/O refusal.
    // Supported values serialize directly; no owned value or JSON string is built.
    let parts = [
        size_of::<serde_json::Serializer<&mut W>>(),
        size_of::<Result<(), serde_json::Error>>(),
        size_of::<(&mut W, &T, &ParserAllocationFunding)>(),
        serde_json::Error::io_storage_bytes(),
    ];
    funding.reserve(parts.into_iter().try_fold(size_of_val(&parts), usize::checked_add)
        .ok_or(Cause::Overflow)?)?;
    serde_json::to_writer(writer, value).map_err(Into::into)
}

fn raw(writer: &mut impl Write, bytes: &[u8]) -> Result<(), Cause> {
    writer.write_all(bytes).map_err(|_| Cause::Destination)
}

pub(super) fn tools<W: Write>(
    writer: &mut W, tools: &[Value], funding: &ParserAllocationFunding,
) -> Result<(), Cause> {
    raw(writer, b"[")?;
    for (index, item) in tools.iter().enumerate() {
        if index != 0 { raw(writer, b",")?; }
        canonical(writer, item, funding)?;
    }
    raw(writer, b"]")
}

fn canonical<W: Write>(
    writer: &mut W, input: &Value, funding: &ParserAllocationFunding,
) -> Result<(), Cause> {
    let parts = [
        size_of::<(&mut W, &Value, &ParserAllocationFunding)>(),
        size_of::<serde_json::map::Iter<'_>>(),
        size_of::<std::slice::Iter<'_, Value>>(),
        size_of::<Vec<(&str, &Value)>>(),
        size_of::<Result<(), Cause>>(),
    ];
    funding.reserve(parts.into_iter().try_fold(size_of_val(&parts), usize::checked_add)
        .ok_or(Cause::Overflow)?)?;
    match input {
        Value::Object(object) => {
            raw(writer, b"{")?;
            // BTreeMap is already ordered. The optional insertion-order map uses
            // one paid row vector of borrowed fields, never a cloned JSON tree.
            if serde_json::bounded_events::preserves_object_order() {
                let mut rows = Vec::new();
                funding.try_grow_vec(&mut rows, object.len())?;
                rows.extend(object.iter().map(|(key, value)| (key.as_str(), value)));
                rows.sort_unstable_by_key(|(key, _)| *key);
                for (index, (key, item)) in rows.into_iter().enumerate() {
                    if index != 0 { raw(writer, b",")?; }
                    value(writer, key, funding)?;
                    raw(writer, b":")?;
                    canonical(writer, item, funding)?;
                }
            } else {
                for (index, (key, item)) in object.iter().enumerate() {
                    if index != 0 { raw(writer, b",")?; }
                    value(writer, key, funding)?;
                    raw(writer, b":")?;
                    canonical(writer, item, funding)?;
                }
            }
            raw(writer, b"}")
        }
        Value::Array(items) => tools(writer, items, funding),
        Value::Number(number) => {
            // Preserve ordinary Value equality for signed floating zero, while
            // leaving arbitrary-precision textual number distinctions intact.
            if number.equals_float_zero() { raw(writer, b"0.0") }
            else { value(writer, number, funding) }
        }
        scalar => value(writer, scalar, funding),
    }
}
