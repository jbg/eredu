//! Checked logical storage measurement for owned, fully serialized host DTOs.
//!
//! This traverses values without allocating a JSON buffer. It counts owned
//! sequence/map entries, strings and bytes; inline root storage is added by the
//! caller. Composite sequence elements receive a conservative 1024-byte inline
//! allowance, checked below against the supported admission DTOs. Serde frequently
//! passes references to sequence elements, so size_of_val on its arguments would
//! undercount owned String/Vec/struct entries. Scalar/string/vector sizes instead
//! follow the serialized value category. Allocator spare capacity/node overhead
//! is excluded. This is only for the audited admission DTOs, never native handles
//! or objects with skipped owned fields.

use serde::ser::{self, Serialize};

#[derive(Debug, thiserror::Error)]
#[error("logical host storage estimate is unavailable")]
pub(crate) struct StorageError;
impl ser::Error for StorageError {
    fn custom<T: std::fmt::Display>(_: T) -> Self {
        Self
    }
}

pub(crate) fn heap_bytes<T: Serialize + ?Sized>(value: &T) -> Option<u64> {
    value.serialize(Storage { inline: false }).ok()
}

const COMPOSITE_INLINE: u64 = 1024;
const _: () = {
    use eredu_core::{capture::*, discovery::*, intervention::*};
    assert!(std::mem::size_of::<CaptureSelection>() <= COMPOSITE_INLINE as usize);
    assert!(std::mem::size_of::<CaptureSlice>() <= COMPOSITE_INLINE as usize);
    assert!(std::mem::size_of::<ObservationPoint>() <= COMPOSITE_INLINE as usize);
    assert!(std::mem::size_of::<TensorAxis>() <= COMPOSITE_INLINE as usize);
    assert!(std::mem::size_of::<SymbolicDimension>() <= COMPOSITE_INLINE as usize);
    assert!(std::mem::size_of::<InterventionOperation>() <= COMPOSITE_INLINE as usize);
    assert!(std::mem::size_of::<InterventionPoint>() <= COMPOSITE_INLINE as usize);
    assert!(std::mem::size_of::<InterventionAction>() <= COMPOSITE_INLINE as usize);
};

struct Storage {
    inline: bool,
}
impl Storage {
    fn bytes(self, dynamic: usize, inline: u64) -> Result<u64, StorageError> {
        u64::try_from(dynamic)
            .map_err(|_| StorageError)?
            .checked_add(if self.inline { inline } else { 0 })
            .ok_or(StorageError)
    }
}
struct Fields {
    bytes: u64,
    elements: bool,
}
impl Fields {
    fn add<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), StorageError> {
        self.bytes = self
            .bytes
            .checked_add(value.serialize(Storage {
                inline: self.elements,
            })?)
            .ok_or(StorageError)?;
        Ok(())
    }
}

macro_rules! scalar {
    ($($method:ident($ty:ty)),*) => { $(fn $method(self, _: $ty) -> Result<u64, StorageError> { self.bytes(0, std::mem::size_of::<$ty>() as u64) })* };
}
impl ser::Serializer for Storage {
    type Ok = u64;
    type Error = StorageError;
    type SerializeSeq = Fields;
    type SerializeTuple = Fields;
    type SerializeTupleStruct = Fields;
    type SerializeTupleVariant = Fields;
    type SerializeMap = Fields;
    type SerializeStruct = Fields;
    type SerializeStructVariant = Fields;
    scalar!(
        serialize_bool(bool),
        serialize_i8(i8),
        serialize_i16(i16),
        serialize_i32(i32),
        serialize_i64(i64),
        serialize_i128(i128),
        serialize_u8(u8),
        serialize_u16(u16),
        serialize_u32(u32),
        serialize_u64(u64),
        serialize_u128(u128),
        serialize_f32(f32),
        serialize_f64(f64),
        serialize_char(char)
    );
    fn serialize_str(self, value: &str) -> Result<u64, StorageError> {
        self.bytes(value.len(), std::mem::size_of::<String>() as u64)
    }
    fn serialize_bytes(self, value: &[u8]) -> Result<u64, StorageError> {
        self.bytes(value.len(), std::mem::size_of::<Vec<u8>>() as u64)
    }
    fn serialize_none(self) -> Result<u64, StorageError> {
        self.bytes(0, COMPOSITE_INLINE)
    }
    fn serialize_some<T: Serialize + ?Sized>(self, value: &T) -> Result<u64, StorageError> {
        let tag = if self.inline { 16 } else { 0 };
        value.serialize(self)?.checked_add(tag).ok_or(StorageError)
    }
    fn serialize_unit(self) -> Result<u64, StorageError> {
        Ok(0)
    }
    fn serialize_unit_struct(self, _: &'static str) -> Result<u64, StorageError> {
        self.bytes(0, COMPOSITE_INLINE)
    }
    fn serialize_unit_variant(
        self,
        _: &'static str,
        _: u32,
        _: &'static str,
    ) -> Result<u64, StorageError> {
        self.bytes(0, COMPOSITE_INLINE)
    }
    fn serialize_newtype_struct<T: Serialize + ?Sized>(
        self,
        _: &'static str,
        value: &T,
    ) -> Result<u64, StorageError> {
        self.bytes(0, COMPOSITE_INLINE)?
            .checked_add(value.serialize(Storage { inline: false })?)
            .ok_or(StorageError)
    }
    fn serialize_newtype_variant<T: Serialize + ?Sized>(
        self,
        _: &'static str,
        _: u32,
        _: &'static str,
        value: &T,
    ) -> Result<u64, StorageError> {
        self.bytes(0, COMPOSITE_INLINE)?
            .checked_add(value.serialize(Storage { inline: false })?)
            .ok_or(StorageError)
    }
    fn serialize_seq(self, _: Option<usize>) -> Result<Fields, StorageError> {
        Ok(Fields {
            bytes: self.bytes(0, std::mem::size_of::<Vec<()>>() as u64)?,
            elements: true,
        })
    }
    fn serialize_tuple(self, _: usize) -> Result<Fields, StorageError> {
        Ok(Fields {
            bytes: 0,
            elements: self.inline,
        })
    }
    fn serialize_tuple_struct(self, _: &'static str, _: usize) -> Result<Fields, StorageError> {
        self.serialize_tuple(0)
    }
    fn serialize_tuple_variant(
        self,
        _: &'static str,
        _: u32,
        _: &'static str,
        _: usize,
    ) -> Result<Fields, StorageError> {
        self.serialize_tuple(0)
    }
    fn serialize_map(self, _: Option<usize>) -> Result<Fields, StorageError> {
        self.serialize_seq(None)
    }
    fn serialize_struct(self, _: &'static str, _: usize) -> Result<Fields, StorageError> {
        Ok(Fields {
            bytes: self.bytes(0, COMPOSITE_INLINE)?,
            elements: false,
        })
    }
    fn serialize_struct_variant(
        self,
        _: &'static str,
        _: u32,
        _: &'static str,
        _: usize,
    ) -> Result<Fields, StorageError> {
        self.serialize_struct("variant", 0)
    }
    fn collect_str<T: std::fmt::Display + ?Sized>(self, value: &T) -> Result<u64, StorageError> {
        struct Count(u64);
        impl std::fmt::Write for Count {
            fn write_str(&mut self, text: &str) -> std::fmt::Result {
                self.0 = self
                    .0
                    .checked_add(u64::try_from(text.len()).map_err(|_| std::fmt::Error)?)
                    .ok_or(std::fmt::Error)?;
                Ok(())
            }
        }
        use std::fmt::Write;
        let mut count = Count(0);
        write!(&mut count, "{value}").map_err(|_| StorageError)?;
        count
            .0
            .checked_add(self.bytes(0, std::mem::size_of::<String>() as u64)?)
            .ok_or(StorageError)
    }
}

macro_rules! sequence {
    ($trait:ident, $method:ident) => {
        impl ser::$trait for Fields {
            type Ok = u64;
            type Error = StorageError;
            fn $method<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), StorageError> {
                self.add(value)
            }
            fn end(self) -> Result<u64, StorageError> {
                Ok(self.bytes)
            }
        }
    };
}
sequence!(SerializeSeq, serialize_element);
sequence!(SerializeTuple, serialize_element);
sequence!(SerializeTupleStruct, serialize_field);
sequence!(SerializeTupleVariant, serialize_field);
impl ser::SerializeMap for Fields {
    type Ok = u64;
    type Error = StorageError;
    fn serialize_key<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), StorageError> {
        self.add(value)
    }
    fn serialize_value<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), StorageError> {
        self.add(value)
    }
    fn end(self) -> Result<u64, StorageError> {
        Ok(self.bytes)
    }
}
macro_rules! structure {
    ($trait:ident) => {
        impl ser::$trait for Fields {
            type Ok = u64;
            type Error = StorageError;
            fn serialize_field<T: Serialize + ?Sized>(
                &mut self,
                _: &'static str,
                value: &T,
            ) -> Result<(), StorageError> {
                self.add(value)
            }
            fn end(self) -> Result<u64, StorageError> {
                Ok(self.bytes)
            }
        }
    };
}
structure!(SerializeStruct);
structure!(SerializeStructVariant);

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn owned_nested_payloads_charge_inline_entries_and_dynamic_contents() {
        #[derive(serde::Serialize)]
        struct Payload {
            values: Vec<String>,
            indices: Vec<Vec<u64>>,
        }
        let value = Payload {
            values: vec!["small".into(), "x".repeat(1024)],
            indices: vec![vec![1, 2, 3]],
        };
        let expected =
            2 * std::mem::size_of::<String>() + 1029 + std::mem::size_of::<Vec<u64>>() + 24;
        assert_eq!(heap_bytes(&value), Some(expected as u64));
    }
}
