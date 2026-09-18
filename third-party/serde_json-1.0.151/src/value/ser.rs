use crate::error::{Error, ErrorCode, Result};
use crate::map::Map;
use crate::value::{to_value, Value};
use alloc::borrow::ToOwned;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::fmt::Display;
use core::result;
use serde::ser::{Impossible, Serialize};

impl Serialize for Value {
    #[inline]
    fn serialize<S>(&self, serializer: S) -> result::Result<S::Ok, S::Error>
    where S: ::serde::Serializer {
        ValueRef { value: self, context: None }.serialize(serializer)
    }
}
struct ValueRef<'a> {
    value: &'a Value,
    context: Option<&'a Serialization<'a>>,
}
impl Serialize for ValueRef<'_> {
    fn serialize<S>(&self, serializer: S) -> result::Result<S::Ok, S::Error>
    where S: ::serde::Serializer {
        if let Some(context) = self.context {
            let parts = [
                core::mem::size_of::<Self>(), core::mem::size_of::<S>(),
                core::mem::size_of::<S::SerializeSeq>(), core::mem::size_of::<S::SerializeMap>(),
                core::mem::size_of::<result::Result<S::Ok, S::Error>>(),
                core::mem::size_of::<crate::map::Iter<'_>>(),
                core::mem::size_of::<core::slice::Iter<'_, Value>>(),
                core::mem::size_of::<(&str, &Value)>(),
            ];
            let control = parts.iter().try_fold(core::mem::size_of_val(&parts), |a,b| a.checked_add(*b));
            let result = control.ok_or(crate::allocation::AllocationError::SizeOverflow)
                .and_then(|bytes| context.funding.reserve(bytes));
            if let Err(error) = result {
                context.failure.set(Some(error));
                // This wrapper is used with the paid writer only. Its next write
                // returns the pre-admitted fixed I/O error without a diagnostic.
                return serializer.serialize_unit();
            }
        }
        match self.value {
            Value::Null => serializer.serialize_unit(),
            Value::Bool(b) => serializer.serialize_bool(*b),
            Value::Number(n) => n.serialize(serializer),
            Value::String(s) => serializer.serialize_str(s),
            Value::Array(values) => {
                use serde::ser::SerializeSeq;
                let mut seq = tri!(serializer.serialize_seq(Some(values.len())));
                for value in values {
                    tri!(seq.serialize_element(&ValueRef { value, context: self.context }));
                }
                seq.end()
            }
            #[cfg(any(feature = "std", feature = "alloc"))]
            Value::Object(values) => {
                use serde::ser::SerializeMap;
                let mut map = tri!(serializer.serialize_map(Some(values.len())));
                for (key, value) in values {
                    tri!(map.serialize_entry(key, &ValueRef { value, context: self.context }));
                }
                map.end()
            }
            #[cfg(not(any(feature = "std", feature = "alloc")))]
            Value::Object(_) => unreachable!(),
        }
    }
}
struct Serialization<'a> {
    funding: &'a dyn crate::allocation::Allocation,
    failure: core::cell::Cell<Option<crate::allocation::AllocationError>>,
}
/// A serialization refusal retains no unfunded owned diagnostic.
#[derive(Debug)]
pub enum ValueWriteError {
    /// The caller refused a reached writer or traversal producer.
    Allocation(crate::allocation::AllocationError),
    /// The ordinary serializer rejected the value.
    Serialization(Error),
}
impl core::fmt::Display for ValueWriteError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self { Self::Allocation(e) => e.fmt(f), Self::Serialization(e) => e.fmt(f) }
    }
}
#[cfg(feature = "std")]
impl std::error::Error for ValueWriteError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(match self { Self::Allocation(e) => e, Self::Serialization(e) => e })
    }
}
#[cfg(feature = "std")]
impl Value {
    /// Serialize through the ordinary Value/serde worker with prospective
    /// traversal controls and output growth. The enclosing owner retains funding.
    pub fn to_string_with_allocations(&self, funding: &dyn crate::allocation::Allocation)
        -> result::Result<String, ValueWriteError>
    {
        use crate::allocation::{AllocationError, Allocator};
        struct Writer<'a> { context: &'a Serialization<'a>, bytes: Vec<u8> }
        impl std::io::Write for Writer<'_> {
            fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
                let result = if let Some(error) = self.context.failure.get() { Err(error) }
                else {
                    self.bytes.len().checked_add(bytes.len()).ok_or(AllocationError::SizeOverflow)
                        .and_then(|n| Allocator::new(self.context.funding).grow(&mut self.bytes, n))
                };
                if let Err(error) = result {
                    self.context.failure.set(Some(error));
                    return Err(std::io::ErrorKind::Other.into());
                }
                self.bytes.extend_from_slice(bytes);
                Ok(bytes.len())
            }
            fn flush(&mut self) -> std::io::Result<()> { Ok(()) }
        }
        let controls = [Error::io_storage_bytes(), core::mem::size_of::<Serialization<'_>>(),
            core::mem::size_of::<Writer<'_>>(), core::mem::size_of::<ValueWriteError>(),
            core::mem::size_of::<crate::Serializer<&mut Writer<'_>>>(),
            core::mem::size_of::<result::Result<String, ValueWriteError>>()];
        let controls = controls.iter().try_fold(core::mem::size_of_val(&controls), |a,b| a.checked_add(*b))
            .ok_or(ValueWriteError::Allocation(AllocationError::SizeOverflow))?;
        funding.reserve(controls).map_err(ValueWriteError::Allocation)?;
        let context = Serialization { funding, failure: core::cell::Cell::new(None) };
        let mut writer = Writer { context: &context, bytes: Vec::new() };
        let result = crate::to_writer(&mut writer, &ValueRef { value: self, context: Some(&context) });
        if let Some(error) = context.failure.get() { return Err(ValueWriteError::Allocation(error)); }
        result.map_err(ValueWriteError::Serialization)?;
        Ok(String::from_utf8(writer.bytes).expect("serde serialized UTF-8"))
    }
}

/// Serializer whose output is a `Value`.
///
/// This is the serializer that backs [`serde_json::to_value`][crate::to_value].
/// Unlike the main serde_json serializer which goes from some serializable
/// value of type `T` to JSON text, this one goes from `T` to
/// `serde_json::Value`.
///
/// The `to_value` function is implementable as:
///
/// ```
/// use serde::Serialize;
/// use serde_json::{Error, Value};
///
/// pub fn to_value<T>(input: T) -> Result<Value, Error>
/// where
///     T: Serialize,
/// {
///     input.serialize(serde_json::value::Serializer)
/// }
/// ```
pub struct Serializer;

impl serde::Serializer for Serializer {
    type Ok = Value;
    type Error = Error;

    type SerializeSeq = SerializeVec;
    type SerializeTuple = SerializeVec;
    type SerializeTupleStruct = SerializeVec;
    type SerializeTupleVariant = SerializeTupleVariant;
    type SerializeMap = SerializeMap;
    type SerializeStruct = SerializeMap;
    type SerializeStructVariant = SerializeStructVariant;

    #[inline]
    fn serialize_bool(self, value: bool) -> Result<Value> {
        Ok(Value::Bool(value))
    }

    #[inline]
    fn serialize_i8(self, value: i8) -> Result<Value> {
        self.serialize_i64(value as i64)
    }

    #[inline]
    fn serialize_i16(self, value: i16) -> Result<Value> {
        self.serialize_i64(value as i64)
    }

    #[inline]
    fn serialize_i32(self, value: i32) -> Result<Value> {
        self.serialize_i64(value as i64)
    }

    fn serialize_i64(self, value: i64) -> Result<Value> {
        Ok(Value::Number(value.into()))
    }

    fn serialize_i128(self, value: i128) -> Result<Value> {
        #[cfg(feature = "arbitrary_precision")]
        {
            Ok(Value::Number(value.into()))
        }

        #[cfg(not(feature = "arbitrary_precision"))]
        {
            if let Ok(value) = u64::try_from(value) {
                Ok(Value::Number(value.into()))
            } else if let Ok(value) = i64::try_from(value) {
                Ok(Value::Number(value.into()))
            } else {
                Err(Error::syntax(ErrorCode::NumberOutOfRange, 0, 0))
            }
        }
    }

    #[inline]
    fn serialize_u8(self, value: u8) -> Result<Value> {
        self.serialize_u64(value as u64)
    }

    #[inline]
    fn serialize_u16(self, value: u16) -> Result<Value> {
        self.serialize_u64(value as u64)
    }

    #[inline]
    fn serialize_u32(self, value: u32) -> Result<Value> {
        self.serialize_u64(value as u64)
    }

    #[inline]
    fn serialize_u64(self, value: u64) -> Result<Value> {
        Ok(Value::Number(value.into()))
    }

    fn serialize_u128(self, value: u128) -> Result<Value> {
        #[cfg(feature = "arbitrary_precision")]
        {
            Ok(Value::Number(value.into()))
        }

        #[cfg(not(feature = "arbitrary_precision"))]
        {
            if let Ok(value) = u64::try_from(value) {
                Ok(Value::Number(value.into()))
            } else {
                Err(Error::syntax(ErrorCode::NumberOutOfRange, 0, 0))
            }
        }
    }

    #[inline]
    fn serialize_f32(self, float: f32) -> Result<Value> {
        Ok(Value::from(float))
    }

    #[inline]
    fn serialize_f64(self, float: f64) -> Result<Value> {
        Ok(Value::from(float))
    }

    #[inline]
    fn serialize_char(self, value: char) -> Result<Value> {
        let mut s = String::new();
        s.push(value);
        Ok(Value::String(s))
    }

    #[inline]
    fn serialize_str(self, value: &str) -> Result<Value> {
        Ok(Value::String(value.to_owned()))
    }

    fn serialize_bytes(self, value: &[u8]) -> Result<Value> {
        let vec = value.iter().map(|&b| Value::Number(b.into())).collect();
        Ok(Value::Array(vec))
    }

    #[inline]
    fn serialize_unit(self) -> Result<Value> {
        Ok(Value::Null)
    }

    #[inline]
    fn serialize_unit_struct(self, _name: &'static str) -> Result<Value> {
        self.serialize_unit()
    }

    #[inline]
    fn serialize_unit_variant(
        self,
        _name: &'static str,
        _variant_index: u32,
        variant: &'static str,
    ) -> Result<Value> {
        self.serialize_str(variant)
    }

    #[inline]
    fn serialize_newtype_struct<T>(self, _name: &'static str, value: &T) -> Result<Value>
    where
        T: ?Sized + Serialize,
    {
        value.serialize(self)
    }

    fn serialize_newtype_variant<T>(
        self,
        _name: &'static str,
        _variant_index: u32,
        variant: &'static str,
        value: &T,
    ) -> Result<Value>
    where
        T: ?Sized + Serialize,
    {
        let mut values = Map::new();
        values.insert(String::from(variant), tri!(to_value(value)));
        Ok(Value::Object(values))
    }

    #[inline]
    fn serialize_none(self) -> Result<Value> {
        self.serialize_unit()
    }

    #[inline]
    fn serialize_some<T>(self, value: &T) -> Result<Value>
    where
        T: ?Sized + Serialize,
    {
        value.serialize(self)
    }

    fn serialize_seq(self, len: Option<usize>) -> Result<Self::SerializeSeq> {
        Ok(SerializeVec {
            vec: Vec::with_capacity(len.unwrap_or(0)),
        })
    }

    fn serialize_tuple(self, len: usize) -> Result<Self::SerializeTuple> {
        self.serialize_seq(Some(len))
    }

    fn serialize_tuple_struct(
        self,
        _name: &'static str,
        len: usize,
    ) -> Result<Self::SerializeTupleStruct> {
        self.serialize_seq(Some(len))
    }

    fn serialize_tuple_variant(
        self,
        _name: &'static str,
        _variant_index: u32,
        variant: &'static str,
        len: usize,
    ) -> Result<Self::SerializeTupleVariant> {
        Ok(SerializeTupleVariant {
            name: String::from(variant),
            vec: Vec::with_capacity(len),
        })
    }

    fn serialize_map(self, len: Option<usize>) -> Result<Self::SerializeMap> {
        Ok(SerializeMap::Map {
            map: Map::with_capacity(len.unwrap_or(0)),
            next_key: None,
        })
    }

    fn serialize_struct(self, name: &'static str, len: usize) -> Result<Self::SerializeStruct> {
        match name {
            #[cfg(feature = "arbitrary_precision")]
            crate::number::TOKEN => Ok(SerializeMap::Number { out_value: None }),
            #[cfg(feature = "raw_value")]
            crate::raw::TOKEN => Ok(SerializeMap::RawValue { out_value: None }),
            _ => self.serialize_map(Some(len)),
        }
    }

    fn serialize_struct_variant(
        self,
        _name: &'static str,
        _variant_index: u32,
        variant: &'static str,
        _len: usize,
    ) -> Result<Self::SerializeStructVariant> {
        Ok(SerializeStructVariant {
            name: String::from(variant),
            map: Map::new(),
        })
    }

    fn collect_str<T>(self, value: &T) -> Result<Value>
    where
        T: ?Sized + Display,
    {
        Ok(Value::String(value.to_string()))
    }
}

pub struct SerializeVec {
    vec: Vec<Value>,
}

pub struct SerializeTupleVariant {
    name: String,
    vec: Vec<Value>,
}

pub enum SerializeMap {
    Map {
        map: Map<String, Value>,
        next_key: Option<String>,
    },
    #[cfg(feature = "arbitrary_precision")]
    Number { out_value: Option<Value> },
    #[cfg(feature = "raw_value")]
    RawValue { out_value: Option<Value> },
}

pub struct SerializeStructVariant {
    name: String,
    map: Map<String, Value>,
}

impl serde::ser::SerializeSeq for SerializeVec {
    type Ok = Value;
    type Error = Error;

    fn serialize_element<T>(&mut self, value: &T) -> Result<()>
    where
        T: ?Sized + Serialize,
    {
        self.vec.push(tri!(to_value(value)));
        Ok(())
    }

    fn end(self) -> Result<Value> {
        Ok(Value::Array(self.vec))
    }
}

impl serde::ser::SerializeTuple for SerializeVec {
    type Ok = Value;
    type Error = Error;

    fn serialize_element<T>(&mut self, value: &T) -> Result<()>
    where
        T: ?Sized + Serialize,
    {
        serde::ser::SerializeSeq::serialize_element(self, value)
    }

    fn end(self) -> Result<Value> {
        serde::ser::SerializeSeq::end(self)
    }
}

impl serde::ser::SerializeTupleStruct for SerializeVec {
    type Ok = Value;
    type Error = Error;

    fn serialize_field<T>(&mut self, value: &T) -> Result<()>
    where
        T: ?Sized + Serialize,
    {
        serde::ser::SerializeSeq::serialize_element(self, value)
    }

    fn end(self) -> Result<Value> {
        serde::ser::SerializeSeq::end(self)
    }
}

impl serde::ser::SerializeTupleVariant for SerializeTupleVariant {
    type Ok = Value;
    type Error = Error;

    fn serialize_field<T>(&mut self, value: &T) -> Result<()>
    where
        T: ?Sized + Serialize,
    {
        self.vec.push(tri!(to_value(value)));
        Ok(())
    }

    fn end(self) -> Result<Value> {
        let mut object = Map::new();

        object.insert(self.name, Value::Array(self.vec));

        Ok(Value::Object(object))
    }
}

impl serde::ser::SerializeMap for SerializeMap {
    type Ok = Value;
    type Error = Error;

    fn serialize_key<T>(&mut self, key: &T) -> Result<()>
    where
        T: ?Sized + Serialize,
    {
        match self {
            SerializeMap::Map { next_key, .. } => {
                *next_key = Some(tri!(key.serialize(MapKeySerializer)));
                Ok(())
            }
            #[cfg(feature = "arbitrary_precision")]
            SerializeMap::Number { .. } => unreachable!(),
            #[cfg(feature = "raw_value")]
            SerializeMap::RawValue { .. } => unreachable!(),
        }
    }

    fn serialize_value<T>(&mut self, value: &T) -> Result<()>
    where
        T: ?Sized + Serialize,
    {
        match self {
            SerializeMap::Map { map, next_key } => {
                let key = next_key.take();
                // Panic because this indicates a bug in the program rather than an
                // expected failure.
                let key = key.expect("serialize_value called before serialize_key");
                map.insert(key, tri!(to_value(value)));
                Ok(())
            }
            #[cfg(feature = "arbitrary_precision")]
            SerializeMap::Number { .. } => unreachable!(),
            #[cfg(feature = "raw_value")]
            SerializeMap::RawValue { .. } => unreachable!(),
        }
    }

    fn end(self) -> Result<Value> {
        match self {
            SerializeMap::Map { map, .. } => Ok(Value::Object(map)),
            #[cfg(feature = "arbitrary_precision")]
            SerializeMap::Number { .. } => unreachable!(),
            #[cfg(feature = "raw_value")]
            SerializeMap::RawValue { .. } => unreachable!(),
        }
    }
}

struct MapKeySerializer;

fn key_must_be_a_string() -> Error {
    Error::syntax(ErrorCode::KeyMustBeAString, 0, 0)
}

fn float_key_must_be_finite() -> Error {
    Error::syntax(ErrorCode::FloatKeyMustBeFinite, 0, 0)
}

impl serde::Serializer for MapKeySerializer {
    type Ok = String;
    type Error = Error;

    type SerializeSeq = Impossible<String, Error>;
    type SerializeTuple = Impossible<String, Error>;
    type SerializeTupleStruct = Impossible<String, Error>;
    type SerializeTupleVariant = Impossible<String, Error>;
    type SerializeMap = Impossible<String, Error>;
    type SerializeStruct = Impossible<String, Error>;
    type SerializeStructVariant = Impossible<String, Error>;

    #[inline]
    fn serialize_unit_variant(
        self,
        _name: &'static str,
        _variant_index: u32,
        variant: &'static str,
    ) -> Result<String> {
        Ok(variant.to_owned())
    }

    #[inline]
    fn serialize_newtype_struct<T>(self, _name: &'static str, value: &T) -> Result<String>
    where
        T: ?Sized + Serialize,
    {
        value.serialize(self)
    }

    fn serialize_bool(self, value: bool) -> Result<String> {
        Ok(if value { "true" } else { "false" }.to_owned())
    }

    fn serialize_i8(self, value: i8) -> Result<String> {
        Ok(itoa::Buffer::new().format(value).to_owned())
    }

    fn serialize_i16(self, value: i16) -> Result<String> {
        Ok(itoa::Buffer::new().format(value).to_owned())
    }

    fn serialize_i32(self, value: i32) -> Result<String> {
        Ok(itoa::Buffer::new().format(value).to_owned())
    }

    fn serialize_i64(self, value: i64) -> Result<String> {
        Ok(itoa::Buffer::new().format(value).to_owned())
    }

    fn serialize_i128(self, value: i128) -> Result<String> {
        Ok(itoa::Buffer::new().format(value).to_owned())
    }

    fn serialize_u8(self, value: u8) -> Result<String> {
        Ok(itoa::Buffer::new().format(value).to_owned())
    }

    fn serialize_u16(self, value: u16) -> Result<String> {
        Ok(itoa::Buffer::new().format(value).to_owned())
    }

    fn serialize_u32(self, value: u32) -> Result<String> {
        Ok(itoa::Buffer::new().format(value).to_owned())
    }

    fn serialize_u64(self, value: u64) -> Result<String> {
        Ok(itoa::Buffer::new().format(value).to_owned())
    }

    fn serialize_u128(self, value: u128) -> Result<String> {
        Ok(itoa::Buffer::new().format(value).to_owned())
    }

    fn serialize_f32(self, value: f32) -> Result<String> {
        if value.is_finite() {
            Ok(zmij::Buffer::new().format_finite(value).to_owned())
        } else {
            Err(float_key_must_be_finite())
        }
    }

    fn serialize_f64(self, value: f64) -> Result<String> {
        if value.is_finite() {
            Ok(zmij::Buffer::new().format_finite(value).to_owned())
        } else {
            Err(float_key_must_be_finite())
        }
    }

    #[inline]
    fn serialize_char(self, value: char) -> Result<String> {
        Ok({
            let mut s = String::new();
            s.push(value);
            s
        })
    }

    #[inline]
    fn serialize_str(self, value: &str) -> Result<String> {
        Ok(value.to_owned())
    }

    fn serialize_bytes(self, _value: &[u8]) -> Result<String> {
        Err(key_must_be_a_string())
    }

    fn serialize_unit(self) -> Result<String> {
        Err(key_must_be_a_string())
    }

    fn serialize_unit_struct(self, _name: &'static str) -> Result<String> {
        Err(key_must_be_a_string())
    }

    fn serialize_newtype_variant<T>(
        self,
        _name: &'static str,
        _variant_index: u32,
        _variant: &'static str,
        _value: &T,
    ) -> Result<String>
    where
        T: ?Sized + Serialize,
    {
        Err(key_must_be_a_string())
    }

    fn serialize_none(self) -> Result<String> {
        Err(key_must_be_a_string())
    }

    fn serialize_some<T>(self, _value: &T) -> Result<String>
    where
        T: ?Sized + Serialize,
    {
        Err(key_must_be_a_string())
    }

    fn serialize_seq(self, _len: Option<usize>) -> Result<Self::SerializeSeq> {
        Err(key_must_be_a_string())
    }

    fn serialize_tuple(self, _len: usize) -> Result<Self::SerializeTuple> {
        Err(key_must_be_a_string())
    }

    fn serialize_tuple_struct(
        self,
        _name: &'static str,
        _len: usize,
    ) -> Result<Self::SerializeTupleStruct> {
        Err(key_must_be_a_string())
    }

    fn serialize_tuple_variant(
        self,
        _name: &'static str,
        _variant_index: u32,
        _variant: &'static str,
        _len: usize,
    ) -> Result<Self::SerializeTupleVariant> {
        Err(key_must_be_a_string())
    }

    fn serialize_map(self, _len: Option<usize>) -> Result<Self::SerializeMap> {
        Err(key_must_be_a_string())
    }

    fn serialize_struct(self, _name: &'static str, _len: usize) -> Result<Self::SerializeStruct> {
        Err(key_must_be_a_string())
    }

    fn serialize_struct_variant(
        self,
        _name: &'static str,
        _variant_index: u32,
        _variant: &'static str,
        _len: usize,
    ) -> Result<Self::SerializeStructVariant> {
        Err(key_must_be_a_string())
    }

    fn collect_str<T>(self, value: &T) -> Result<String>
    where
        T: ?Sized + Display,
    {
        Ok(value.to_string())
    }
}

impl serde::ser::SerializeStruct for SerializeMap {
    type Ok = Value;
    type Error = Error;

    fn serialize_field<T>(&mut self, key: &'static str, value: &T) -> Result<()>
    where
        T: ?Sized + Serialize,
    {
        match self {
            SerializeMap::Map { .. } => serde::ser::SerializeMap::serialize_entry(self, key, value),
            #[cfg(feature = "arbitrary_precision")]
            SerializeMap::Number { out_value } => {
                if key == crate::number::TOKEN {
                    *out_value = Some(tri!(value.serialize(NumberValueEmitter)));
                    Ok(())
                } else {
                    Err(invalid_number())
                }
            }
            #[cfg(feature = "raw_value")]
            SerializeMap::RawValue { out_value } => {
                if key == crate::raw::TOKEN {
                    *out_value = Some(tri!(value.serialize(RawValueEmitter)));
                    Ok(())
                } else {
                    Err(invalid_raw_value())
                }
            }
        }
    }

    fn end(self) -> Result<Value> {
        match self {
            SerializeMap::Map { .. } => serde::ser::SerializeMap::end(self),
            #[cfg(feature = "arbitrary_precision")]
            SerializeMap::Number { out_value, .. } => {
                Ok(out_value.expect("number value was not emitted"))
            }
            #[cfg(feature = "raw_value")]
            SerializeMap::RawValue { out_value, .. } => {
                Ok(out_value.expect("raw value was not emitted"))
            }
        }
    }
}

impl serde::ser::SerializeStructVariant for SerializeStructVariant {
    type Ok = Value;
    type Error = Error;

    fn serialize_field<T>(&mut self, key: &'static str, value: &T) -> Result<()>
    where
        T: ?Sized + Serialize,
    {
        self.map.insert(String::from(key), tri!(to_value(value)));
        Ok(())
    }

    fn end(self) -> Result<Value> {
        let mut object = Map::new();

        object.insert(self.name, Value::Object(self.map));

        Ok(Value::Object(object))
    }
}

#[cfg(feature = "arbitrary_precision")]
struct NumberValueEmitter;

#[cfg(feature = "arbitrary_precision")]
fn invalid_number() -> Error {
    Error::syntax(ErrorCode::InvalidNumber, 0, 0)
}

#[cfg(feature = "arbitrary_precision")]
impl serde::ser::Serializer for NumberValueEmitter {
    type Ok = Value;
    type Error = Error;

    type SerializeSeq = Impossible<Value, Error>;
    type SerializeTuple = Impossible<Value, Error>;
    type SerializeTupleStruct = Impossible<Value, Error>;
    type SerializeTupleVariant = Impossible<Value, Error>;
    type SerializeMap = Impossible<Value, Error>;
    type SerializeStruct = Impossible<Value, Error>;
    type SerializeStructVariant = Impossible<Value, Error>;

    fn serialize_bool(self, _v: bool) -> Result<Value> {
        Err(invalid_number())
    }

    fn serialize_i8(self, _v: i8) -> Result<Value> {
        Err(invalid_number())
    }

    fn serialize_i16(self, _v: i16) -> Result<Value> {
        Err(invalid_number())
    }

    fn serialize_i32(self, _v: i32) -> Result<Value> {
        Err(invalid_number())
    }

    fn serialize_i64(self, _v: i64) -> Result<Value> {
        Err(invalid_number())
    }

    fn serialize_u8(self, _v: u8) -> Result<Value> {
        Err(invalid_number())
    }

    fn serialize_u16(self, _v: u16) -> Result<Value> {
        Err(invalid_number())
    }

    fn serialize_u32(self, _v: u32) -> Result<Value> {
        Err(invalid_number())
    }

    fn serialize_u64(self, _v: u64) -> Result<Value> {
        Err(invalid_number())
    }

    fn serialize_f32(self, _v: f32) -> Result<Value> {
        Err(invalid_number())
    }

    fn serialize_f64(self, _v: f64) -> Result<Value> {
        Err(invalid_number())
    }

    fn serialize_char(self, _v: char) -> Result<Value> {
        Err(invalid_number())
    }

    fn serialize_str(self, value: &str) -> Result<Value> {
        let n = tri!(value.to_owned().parse());
        Ok(Value::Number(n))
    }

    fn serialize_bytes(self, _value: &[u8]) -> Result<Value> {
        Err(invalid_number())
    }

    fn serialize_none(self) -> Result<Value> {
        Err(invalid_number())
    }

    fn serialize_some<T>(self, _value: &T) -> Result<Value>
    where
        T: ?Sized + Serialize,
    {
        Err(invalid_number())
    }

    fn serialize_unit(self) -> Result<Value> {
        Err(invalid_number())
    }

    fn serialize_unit_struct(self, _name: &'static str) -> Result<Value> {
        Err(invalid_number())
    }

    fn serialize_unit_variant(
        self,
        _name: &'static str,
        _variant_index: u32,
        _variant: &'static str,
    ) -> Result<Value> {
        Err(invalid_number())
    }

    fn serialize_newtype_struct<T>(self, _name: &'static str, _value: &T) -> Result<Value>
    where
        T: ?Sized + Serialize,
    {
        Err(invalid_number())
    }

    fn serialize_newtype_variant<T>(
        self,
        _name: &'static str,
        _variant_index: u32,
        _variant: &'static str,
        _value: &T,
    ) -> Result<Value>
    where
        T: ?Sized + Serialize,
    {
        Err(invalid_number())
    }

    fn serialize_seq(self, _len: Option<usize>) -> Result<Self::SerializeSeq> {
        Err(invalid_number())
    }

    fn serialize_tuple(self, _len: usize) -> Result<Self::SerializeTuple> {
        Err(invalid_number())
    }

    fn serialize_tuple_struct(
        self,
        _name: &'static str,
        _len: usize,
    ) -> Result<Self::SerializeTupleStruct> {
        Err(invalid_number())
    }

    fn serialize_tuple_variant(
        self,
        _name: &'static str,
        _variant_index: u32,
        _variant: &'static str,
        _len: usize,
    ) -> Result<Self::SerializeTupleVariant> {
        Err(invalid_number())
    }

    fn serialize_map(self, _len: Option<usize>) -> Result<Self::SerializeMap> {
        Err(invalid_number())
    }

    fn serialize_struct(self, _name: &'static str, _len: usize) -> Result<Self::SerializeStruct> {
        Err(invalid_number())
    }

    fn serialize_struct_variant(
        self,
        _name: &'static str,
        _variant_index: u32,
        _variant: &'static str,
        _len: usize,
    ) -> Result<Self::SerializeStructVariant> {
        Err(invalid_number())
    }
}

#[cfg(feature = "raw_value")]
struct RawValueEmitter;

#[cfg(feature = "raw_value")]
fn invalid_raw_value() -> Error {
    Error::syntax(ErrorCode::ExpectedSomeValue, 0, 0)
}

#[cfg(feature = "raw_value")]
impl serde::ser::Serializer for RawValueEmitter {
    type Ok = Value;
    type Error = Error;

    type SerializeSeq = Impossible<Value, Error>;
    type SerializeTuple = Impossible<Value, Error>;
    type SerializeTupleStruct = Impossible<Value, Error>;
    type SerializeTupleVariant = Impossible<Value, Error>;
    type SerializeMap = Impossible<Value, Error>;
    type SerializeStruct = Impossible<Value, Error>;
    type SerializeStructVariant = Impossible<Value, Error>;

    fn serialize_bool(self, _v: bool) -> Result<Value> {
        Err(invalid_raw_value())
    }

    fn serialize_i8(self, _v: i8) -> Result<Value> {
        Err(invalid_raw_value())
    }

    fn serialize_i16(self, _v: i16) -> Result<Value> {
        Err(invalid_raw_value())
    }

    fn serialize_i32(self, _v: i32) -> Result<Value> {
        Err(invalid_raw_value())
    }

    fn serialize_i64(self, _v: i64) -> Result<Value> {
        Err(invalid_raw_value())
    }

    fn serialize_u8(self, _v: u8) -> Result<Value> {
        Err(invalid_raw_value())
    }

    fn serialize_u16(self, _v: u16) -> Result<Value> {
        Err(invalid_raw_value())
    }

    fn serialize_u32(self, _v: u32) -> Result<Value> {
        Err(invalid_raw_value())
    }

    fn serialize_u64(self, _v: u64) -> Result<Value> {
        Err(invalid_raw_value())
    }

    fn serialize_f32(self, _v: f32) -> Result<Value> {
        Err(invalid_raw_value())
    }

    fn serialize_f64(self, _v: f64) -> Result<Value> {
        Err(invalid_raw_value())
    }

    fn serialize_char(self, _v: char) -> Result<Value> {
        Err(invalid_raw_value())
    }

    fn serialize_str(self, value: &str) -> Result<Value> {
        crate::from_str(value)
    }

    fn serialize_bytes(self, _value: &[u8]) -> Result<Value> {
        Err(invalid_raw_value())
    }

    fn serialize_none(self) -> Result<Value> {
        Err(invalid_raw_value())
    }

    fn serialize_some<T>(self, _value: &T) -> Result<Value>
    where
        T: ?Sized + Serialize,
    {
        Err(invalid_raw_value())
    }

    fn serialize_unit(self) -> Result<Value> {
        Err(invalid_raw_value())
    }

    fn serialize_unit_struct(self, _name: &'static str) -> Result<Value> {
        Err(invalid_raw_value())
    }

    fn serialize_unit_variant(
        self,
        _name: &'static str,
        _variant_index: u32,
        _variant: &'static str,
    ) -> Result<Value> {
        Err(invalid_raw_value())
    }

    fn serialize_newtype_struct<T>(self, _name: &'static str, _value: &T) -> Result<Value>
    where
        T: ?Sized + Serialize,
    {
        Err(invalid_raw_value())
    }

    fn serialize_newtype_variant<T>(
        self,
        _name: &'static str,
        _variant_index: u32,
        _variant: &'static str,
        _value: &T,
    ) -> Result<Value>
    where
        T: ?Sized + Serialize,
    {
        Err(invalid_raw_value())
    }

    fn serialize_seq(self, _len: Option<usize>) -> Result<Self::SerializeSeq> {
        Err(invalid_raw_value())
    }

    fn serialize_tuple(self, _len: usize) -> Result<Self::SerializeTuple> {
        Err(invalid_raw_value())
    }

    fn serialize_tuple_struct(
        self,
        _name: &'static str,
        _len: usize,
    ) -> Result<Self::SerializeTupleStruct> {
        Err(invalid_raw_value())
    }

    fn serialize_tuple_variant(
        self,
        _name: &'static str,
        _variant_index: u32,
        _variant: &'static str,
        _len: usize,
    ) -> Result<Self::SerializeTupleVariant> {
        Err(invalid_raw_value())
    }

    fn serialize_map(self, _len: Option<usize>) -> Result<Self::SerializeMap> {
        Err(invalid_raw_value())
    }

    fn serialize_struct(self, _name: &'static str, _len: usize) -> Result<Self::SerializeStruct> {
        Err(invalid_raw_value())
    }

    fn serialize_struct_variant(
        self,
        _name: &'static str,
        _variant_index: u32,
        _variant: &'static str,
        _len: usize,
    ) -> Result<Self::SerializeStructVariant> {
        Err(invalid_raw_value())
    }

    fn collect_str<T>(self, value: &T) -> Result<Self::Ok>
    where
        T: ?Sized + Display,
    {
        self.serialize_str(&value.to_string())
    }
}
