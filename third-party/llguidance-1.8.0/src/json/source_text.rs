//! Exact serialization destinations for the original JSON grammar producer.
use derivre::{ParserResult as Result, ParserError};
use derivre::ParserAllocationFunding;
use serde::Serialize;
use std::io;

pub(super) fn serialize<T: Serialize + ?Sized>(value: &T, funding: &ParserAllocationFunding) -> Result<String> {
    struct Count(usize);
    impl io::Write for Count {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.0 = self.0.checked_add(bytes.len()).ok_or(io::ErrorKind::OutOfMemory)?;
            Ok(bytes.len())
        }
        fn flush(&mut self) -> io::Result<()> { Ok(()) }
    }
    struct Output<'a>(&'a mut Vec<u8>);
    impl io::Write for Output<'_> {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            if bytes.len() > self.0.capacity() - self.0.len() { return Err(io::ErrorKind::OutOfMemory.into()); }
            self.0.extend_from_slice(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> io::Result<()> { Ok(()) }
    }
    funding.reserve(std::mem::size_of::<(Count, Output<'_>, serde_json::Serializer<Count>, serde_json::Serializer<Output<'_>>, Vec<u8>, String, Result<(), serde_json::Error>)>())?;
    let mut count = Count(0);
    serde_json::to_writer(&mut count, value).map_err(|error| derivre::ParserError::cause(error, funding))?;
    let mut bytes = Vec::new();
    funding.try_grow_vec(&mut bytes, count.0)?;
    serde_json::to_writer(Output(&mut bytes), value).map_err(|error| derivre::ParserError::cause(error, funding))?;
    Ok(String::from_utf8(bytes).expect("JSON serialization emits UTF-8"))
}
