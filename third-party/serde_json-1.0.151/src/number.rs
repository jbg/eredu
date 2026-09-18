use crate::de::ParserNumber;
use crate::error::Error;
#[cfg(feature = "arbitrary_precision")]
use crate::error::ErrorCode;
#[cfg(feature = "arbitrary_precision")]
use alloc::borrow::ToOwned;
#[cfg(feature = "arbitrary_precision")]
use alloc::string::{String, ToString};
use core::fmt::{self, Debug, Display};
#[cfg(not(feature = "arbitrary_precision"))]
use core::hash::{Hash, Hasher};
use serde::de::{self, Unexpected, Visitor};
#[cfg(feature = "arbitrary_precision")]
use serde::de::{IntoDeserializer, MapAccess};
use serde::{forward_to_deserialize_any, Deserialize, Deserializer, Serialize, Serializer};

#[cfg(feature = "arbitrary_precision")]
pub(crate) const TOKEN: &str = "$serde_json::private::Number";

/// Represents a JSON number, whether integer or floating point.
#[derive(PartialEq, Eq, Hash)]
pub struct Number {
    n: N,
}

impl Clone for Number {
    fn clone(&self) -> Self {
        self.try_clone_with_allocations(&crate::allocation::Unenforced)
            .expect("ordinary JSON number clone")
    }
}

#[cfg(test)]
mod allocation_free_zero_tests {
    #[test]
    fn equality_fact_preserves_the_selected_number_representation() {
        let zero = super::Number::from_f64(0.0).unwrap();
        for text in ["0", "-0", "0.0", "-0.0", "0e0", "0.00", "1", "-1", "0.001"] {
            let number: super::Number = crate::from_str(text).unwrap();
            assert_eq!(number.equals_float_zero(), number == zero, "{text}");
        }
    }
}

#[cfg(not(feature = "arbitrary_precision"))]
#[derive(Copy, Clone)]
enum N {
    PosInt(u64),
    /// Always less than zero.
    NegInt(i64),
    /// Always finite.
    Float(f64),
}

#[cfg(not(feature = "arbitrary_precision"))]
impl PartialEq for N {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (N::PosInt(a), N::PosInt(b)) => a == b,
            (N::NegInt(a), N::NegInt(b)) => a == b,
            (N::Float(a), N::Float(b)) => a == b,
            _ => false,
        }
    }
}

// Implementing Eq is fine since any float values are always finite.
#[cfg(not(feature = "arbitrary_precision"))]
impl Eq for N {}

#[cfg(not(feature = "arbitrary_precision"))]
impl Hash for N {
    fn hash<H: Hasher>(&self, h: &mut H) {
        match *self {
            N::PosInt(i) => i.hash(h),
            N::NegInt(i) => i.hash(h),
            N::Float(f) => {
                if f == 0.0f64 {
                    // There are 2 zero representations, +0 and -0, which
                    // compare equal but have different bits. We use the +0 hash
                    // for both so that hash(+0) == hash(-0).
                    0.0f64.to_bits().hash(h);
                } else {
                    f.to_bits().hash(h);
                }
            }
        }
    }
}

#[cfg(feature = "arbitrary_precision")]
type N = String;

impl Number {
    /// Actual owned backing of the selected number representation.
    pub fn allocation_size(&self) -> usize {
        #[cfg(feature = "arbitrary_precision")]
        {
            self.n.capacity()
        }
        #[cfg(not(feature = "arbitrary_precision"))]
        {
            0
        }
    }

    /// Copy the selected number representation after admitting its backing.
    pub fn try_clone_with_allocations(
        &self,
        funding: &dyn crate::allocation::Allocation,
    ) -> Result<Self, crate::allocation::AllocationError> {
        #[cfg(feature = "arbitrary_precision")]
        let n = crate::allocation::Allocator::new(funding).copy_string(&self.n)?;
        #[cfg(not(feature = "arbitrary_precision"))]
        let n = {
            let _ = funding;
            self.n
        };
        Ok(Self { n })
    }
    /// Whether this number compares equal to `Number::from_f64(0.0)` without
    /// constructing that value. Arbitrary-precision textual distinctions are
    /// preserved exactly as in ordinary `Number` equality.
    pub fn equals_float_zero(&self) -> bool {
        #[cfg(not(feature = "arbitrary_precision"))]
        {
            matches!(self.n, N::Float(value) if value == 0.0)
        }
        #[cfg(feature = "arbitrary_precision")]
        {
            self.n == "0.0"
        }
    }

    /// Returns true if the `Number` is an integer between `i64::MIN` and
    /// `i64::MAX`.
    ///
    /// For any Number on which `is_i64` returns true, `as_i64` is guaranteed to
    /// return the integer value.
    pub fn is_i64(&self) -> bool {
        #[cfg(not(feature = "arbitrary_precision"))]
        match self.n {
            N::PosInt(v) => v <= i64::MAX as u64,
            N::NegInt(_) => true,
            N::Float(_) => false,
        }
        #[cfg(feature = "arbitrary_precision")]
        self.as_i64().is_some()
    }

    /// Returns true if the `Number` is an integer between zero and `u64::MAX`.
    ///
    /// For any Number on which `is_u64` returns true, `as_u64` is guaranteed to
    /// return the integer value.
    pub fn is_u64(&self) -> bool {
        #[cfg(not(feature = "arbitrary_precision"))]
        match self.n {
            N::PosInt(_) => true,
            N::NegInt(_) | N::Float(_) => false,
        }
        #[cfg(feature = "arbitrary_precision")]
        self.as_u64().is_some()
    }

    /// Returns true if the `Number` can be represented by f64.
    ///
    /// For any Number on which `is_f64` returns true, `as_f64` is guaranteed to
    /// return the floating point value.
    ///
    /// Currently this function returns true if and only if both `is_i64` and
    /// `is_u64` return false but this is not a guarantee in the future.
    pub fn is_f64(&self) -> bool {
        #[cfg(not(feature = "arbitrary_precision"))]
        match self.n {
            N::Float(_) => true,
            N::PosInt(_) | N::NegInt(_) => false,
        }
        #[cfg(feature = "arbitrary_precision")]
        {
            for c in self.n.chars() {
                if c == '.' || c == 'e' || c == 'E' {
                    return self.n.parse::<f64>().is_ok_and(f64::is_finite);
                }
            }
            false
        }
    }

    /// If the `Number` is an integer, represent it as i64 if possible. Returns
    /// None otherwise.
    pub fn as_i64(&self) -> Option<i64> {
        #[cfg(not(feature = "arbitrary_precision"))]
        match self.n {
            N::PosInt(n) => {
                if n <= i64::MAX as u64 {
                    Some(n as i64)
                } else {
                    None
                }
            }
            N::NegInt(n) => Some(n),
            N::Float(_) => None,
        }
        #[cfg(feature = "arbitrary_precision")]
        self.n.parse().ok()
    }

    /// If the `Number` is an integer, represent it as u64 if possible. Returns
    /// None otherwise.
    pub fn as_u64(&self) -> Option<u64> {
        #[cfg(not(feature = "arbitrary_precision"))]
        match self.n {
            N::PosInt(n) => Some(n),
            N::NegInt(_) | N::Float(_) => None,
        }
        #[cfg(feature = "arbitrary_precision")]
        self.n.parse().ok()
    }

    /// Represents the number as f64 if possible. Returns None otherwise.
    pub fn as_f64(&self) -> Option<f64> {
        #[cfg(not(feature = "arbitrary_precision"))]
        match self.n {
            N::PosInt(n) => Some(n as f64),
            N::NegInt(n) => Some(n as f64),
            N::Float(n) => Some(n),
        }
        #[cfg(feature = "arbitrary_precision")]
        self.n.parse::<f64>().ok().filter(|float| float.is_finite())
    }

    /// Converts a finite `f64` to a `Number`. Infinite or NaN values are not JSON
    /// numbers.
    ///
    /// ```
    /// # use serde_json::Number;
    /// #
    /// assert!(Number::from_f64(256.0).is_some());
    ///
    /// assert!(Number::from_f64(f64::NAN).is_none());
    /// ```
    pub fn from_f64(f: f64) -> Option<Number> {
        Self::from_f64_with_allocations(f, &crate::allocation::Unenforced)
            .expect("ordinary JSON number allocation")
    }
    /// Construct the same finite numeric representation with prospective storage.
    pub fn from_f64_with_allocations(
        f: f64,
        allocation: &dyn crate::allocation::Allocation,
    ) -> Result<Option<Number>, crate::allocation::AllocationError> {
        if f.is_finite() {
            Self::from_parser_with_allocations(ParserNumber::F64(f), allocation).map(Some)
        } else {
            Ok(None)
        }
    }
    /// Construct a signed scalar with the ordinary representation and explicit source.
    pub fn from_i64_with_allocations(
        value: i64,
        allocation: &dyn crate::allocation::Allocation,
    ) -> Result<Number, crate::allocation::AllocationError> {
        if value < 0 {
            Self::from_parser_with_allocations(ParserNumber::I64(value), allocation)
        } else {
            Self::from_parser_with_allocations(ParserNumber::U64(value as u64), allocation)
        }
    }
    /// Construct an unsigned scalar with the ordinary representation and explicit source.
    pub fn from_u64_with_allocations(
        value: u64,
        allocation: &dyn crate::allocation::Allocation,
    ) -> Result<Number, crate::allocation::AllocationError> {
        Self::from_parser_with_allocations(ParserNumber::U64(value), allocation)
    }

    /// If the `Number` is an integer, represent it as i128 if possible. Returns
    /// None otherwise.
    pub fn as_i128(&self) -> Option<i128> {
        #[cfg(not(feature = "arbitrary_precision"))]
        match self.n {
            N::PosInt(n) => Some(n as i128),
            N::NegInt(n) => Some(n as i128),
            N::Float(_) => None,
        }
        #[cfg(feature = "arbitrary_precision")]
        self.n.parse().ok()
    }

    /// If the `Number` is an integer, represent it as u128 if possible. Returns
    /// None otherwise.
    pub fn as_u128(&self) -> Option<u128> {
        #[cfg(not(feature = "arbitrary_precision"))]
        match self.n {
            N::PosInt(n) => Some(n as u128),
            N::NegInt(_) | N::Float(_) => None,
        }
        #[cfg(feature = "arbitrary_precision")]
        self.n.parse().ok()
    }

    /// Converts an `i128` to a `Number`. Numbers smaller than i64::MIN or
    /// larger than u64::MAX can only be represented in `Number` if serde_json's
    /// "arbitrary_precision" feature is enabled.
    ///
    /// ```
    /// # use serde_json::Number;
    /// #
    /// assert!(Number::from_i128(256).is_some());
    /// ```
    pub fn from_i128(i: i128) -> Option<Number> {
        Self::from_i128_with_allocations(i, &crate::allocation::Unenforced)
            .expect("ordinary JSON number allocation")
    }
    /// Construct the same wide scalar representation after admitting owned text.
    pub fn from_i128_with_allocations(
        i: i128,
        allocation: &dyn crate::allocation::Allocation,
    ) -> Result<Option<Number>, crate::allocation::AllocationError> {
        #[cfg(not(feature = "arbitrary_precision"))]
        let _ = allocation;
        let n = {
            #[cfg(not(feature = "arbitrary_precision"))]
            {
                if let Ok(u) = u64::try_from(i) {
                    N::PosInt(u)
                } else if let Ok(i) = i64::try_from(i) {
                    N::NegInt(i)
                } else {
                    return Ok(None);
                }
            }
            #[cfg(feature = "arbitrary_precision")]
            {
                crate::allocation::Allocator::new(allocation)
                    .copy_string(itoa::Buffer::new().format(i))?
            }
        };
        Ok(Some(Number { n }))
    }

    /// Converts a `u128` to a `Number`. Numbers greater than u64::MAX can only
    /// be represented in `Number` if serde_json's "arbitrary_precision" feature
    /// is enabled.
    ///
    /// ```
    /// # use serde_json::Number;
    /// #
    /// assert!(Number::from_u128(256).is_some());
    /// ```
    pub fn from_u128(i: u128) -> Option<Number> {
        Self::from_u128_with_allocations(i, &crate::allocation::Unenforced)
            .expect("ordinary JSON number allocation")
    }
    /// Construct the same wide scalar representation after admitting owned text.
    pub fn from_u128_with_allocations(
        i: u128,
        allocation: &dyn crate::allocation::Allocation,
    ) -> Result<Option<Number>, crate::allocation::AllocationError> {
        #[cfg(not(feature = "arbitrary_precision"))]
        let _ = allocation;
        let n = {
            #[cfg(not(feature = "arbitrary_precision"))]
            {
                if let Ok(u) = u64::try_from(i) {
                    N::PosInt(u)
                } else {
                    return Ok(None);
                }
            }
            #[cfg(feature = "arbitrary_precision")]
            {
                crate::allocation::Allocator::new(allocation)
                    .copy_string(itoa::Buffer::new().format(i))?
            }
        };
        Ok(Some(Number { n }))
    }

    /// Returns the JSON representation that this Number was parsed from.
    ///
    /// When parsing with serde_json's `arbitrary_precision` feature enabled,
    /// positive exponents are normalized to include an explicit `+` sign so
    /// that `1e140` becomes `1e+140`.
    ///
    /// For numbers constructed not via parsing, such as by `From<i32>`, returns
    /// the JSON representation that serde\_json would serialize for this
    /// number.
    ///
    /// ```
    /// # use serde_json::Number;
    /// for value in [
    ///     "7",
    ///     "12.34",
    ///     "34e-56789",
    ///     "0.0123456789000000012345678900000001234567890000123456789",
    ///     "343412345678910111213141516171819202122232425262728293034",
    ///     "-343412345678910111213141516171819202122232425262728293031",
    /// ] {
    ///     let number: Number = serde_json::from_str(value).unwrap();
    ///     assert_eq!(number.as_str(), value);
    /// }
    /// ```
    #[cfg(feature = "arbitrary_precision")]
    #[cfg_attr(docsrs, doc(cfg(feature = "arbitrary_precision")))]
    pub fn as_str(&self) -> &str {
        &self.n
    }

    pub(crate) fn as_f32(&self) -> Option<f32> {
        #[cfg(not(feature = "arbitrary_precision"))]
        match self.n {
            N::PosInt(n) => Some(n as f32),
            N::NegInt(n) => Some(n as f32),
            N::Float(n) => Some(n as f32),
        }
        #[cfg(feature = "arbitrary_precision")]
        self.n.parse::<f32>().ok().filter(|float| float.is_finite())
    }

    pub(crate) fn from_f32(f: f32) -> Option<Number> {
        if f.is_finite() {
            let n = {
                #[cfg(not(feature = "arbitrary_precision"))]
                {
                    N::Float(f as f64)
                }
                #[cfg(feature = "arbitrary_precision")]
                {
                    zmij::Buffer::new().format_finite(f).to_owned()
                }
            };
            Some(Number { n })
        } else {
            None
        }
    }

    #[cfg(feature = "arbitrary_precision")]
    /// Not public API. Only tests use this.
    #[doc(hidden)]
    #[inline]
    pub fn from_string_unchecked(n: String) -> Self {
        Number { n }
    }
}

impl Display for Number {
    #[cfg(not(feature = "arbitrary_precision"))]
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        match self.n {
            N::PosInt(u) => formatter.write_str(itoa::Buffer::new().format(u)),
            N::NegInt(i) => formatter.write_str(itoa::Buffer::new().format(i)),
            N::Float(f) => formatter.write_str(zmij::Buffer::new().format_finite(f)),
        }
    }

    #[cfg(feature = "arbitrary_precision")]
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        Display::fmt(&self.n, formatter)
    }
}

impl Debug for Number {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(formatter, "Number({})", self)
    }
}

impl Serialize for Number {
    #[cfg(not(feature = "arbitrary_precision"))]
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self.n {
            N::PosInt(u) => serializer.serialize_u64(u),
            N::NegInt(i) => serializer.serialize_i64(i),
            N::Float(f) => serializer.serialize_f64(f),
        }
    }

    #[cfg(feature = "arbitrary_precision")]
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::ser::SerializeStruct;

        let mut s = tri!(serializer.serialize_struct(TOKEN, 1));
        tri!(s.serialize_field(TOKEN, &self.n));
        s.end()
    }
}

impl<'de> Deserialize<'de> for Number {
    #[inline]
    fn deserialize<D>(deserializer: D) -> Result<Number, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct NumberVisitor;

        impl<'de> Visitor<'de> for NumberVisitor {
            type Value = Number;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("a JSON number")
            }

            fn visit_i64<E>(self, value: i64) -> Result<Number, E> {
                Ok(value.into())
            }

            fn visit_i128<E>(self, value: i128) -> Result<Number, E>
            where
                E: de::Error,
            {
                Number::from_i128(value)
                    .ok_or_else(|| de::Error::custom("JSON number out of range"))
            }

            fn visit_u64<E>(self, value: u64) -> Result<Number, E> {
                Ok(value.into())
            }

            fn visit_u128<E>(self, value: u128) -> Result<Number, E>
            where
                E: de::Error,
            {
                Number::from_u128(value)
                    .ok_or_else(|| de::Error::custom("JSON number out of range"))
            }

            fn visit_f64<E>(self, value: f64) -> Result<Number, E>
            where
                E: de::Error,
            {
                Number::from_f64(value).ok_or_else(|| de::Error::custom("not a JSON number"))
            }

            #[cfg(feature = "arbitrary_precision")]
            fn visit_map<V>(self, mut visitor: V) -> Result<Number, V::Error>
            where
                V: de::MapAccess<'de>,
            {
                let value = tri!(visitor.next_key::<NumberKey>());
                if value.is_none() {
                    return Err(de::Error::invalid_type(Unexpected::Map, &self));
                }
                let v: NumberFromString = tri!(visitor.next_value());
                Ok(v.value)
            }
        }

        deserializer.deserialize_any(NumberVisitor)
    }
}

#[cfg(feature = "arbitrary_precision")]
struct NumberKey;

#[cfg(feature = "arbitrary_precision")]
impl<'de> de::Deserialize<'de> for NumberKey {
    fn deserialize<D>(deserializer: D) -> Result<NumberKey, D::Error>
    where
        D: de::Deserializer<'de>,
    {
        struct FieldVisitor;

        impl<'de> de::Visitor<'de> for FieldVisitor {
            type Value = ();

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("a valid number field")
            }

            fn visit_str<E>(self, s: &str) -> Result<(), E>
            where
                E: de::Error,
            {
                if s == TOKEN {
                    Ok(())
                } else {
                    Err(de::Error::custom("expected field with custom name"))
                }
            }
        }

        tri!(deserializer.deserialize_identifier(FieldVisitor));
        Ok(NumberKey)
    }
}

#[cfg(feature = "arbitrary_precision")]
pub struct NumberFromString {
    pub value: Number,
}

#[cfg(feature = "arbitrary_precision")]
pub(crate) struct NumberFromStringSeed<'a> {
    pub allocation: &'a dyn crate::allocation::Allocation,
    pub failure: Option<&'a core::cell::RefCell<Option<NumberSourceError>>>,
}
#[cfg(feature = "arbitrary_precision")]
impl<'de> de::Deserialize<'de> for NumberFromString {
    fn deserialize<D: de::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        de::DeserializeSeed::deserialize(
            NumberFromStringSeed {
                allocation: &crate::allocation::Unenforced,
                failure: None,
            },
            deserializer,
        )
    }
}
#[cfg(feature = "arbitrary_precision")]
impl<'de> de::DeserializeSeed<'de> for NumberFromStringSeed<'_> {
    type Value = NumberFromString;
    fn deserialize<D: de::Deserializer<'de>>(
        self,
        deserializer: D,
    ) -> Result<Self::Value, D::Error> {
        deserializer.deserialize_str(self)
    }
}
#[cfg(feature = "arbitrary_precision")]
impl<'de> de::Visitor<'de> for NumberFromStringSeed<'_> {
    type Value = NumberFromString;
    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("string containing a number")
    }
    fn visit_str<E: de::Error>(self, source: &str) -> Result<Self::Value, E> {
        match Number::from_str_with_allocations(source, self.allocation) {
            Ok(value) => Ok(NumberFromString { value }),
            Err(error) => match self.failure {
                Some(failure) => {
                    *failure.borrow_mut() = Some(error);
                    Err(E::custom(""))
                }
                None => Err(E::custom(error)),
            },
        }
    }
}

#[cfg(feature = "arbitrary_precision")]
fn invalid_number() -> Error {
    Error::syntax(ErrorCode::InvalidNumber, 0, 0)
}

macro_rules! deserialize_any {
    (@expand [$($num_string:tt)*]) => {
        #[cfg(not(feature = "arbitrary_precision"))]
        fn deserialize_any<V>(self, visitor: V) -> Result<V::Value, Error>
        where
            V: Visitor<'de>,
        {
            match self.n {
                N::PosInt(u) => visitor.visit_u64(u),
                N::NegInt(i) => visitor.visit_i64(i),
                N::Float(f) => visitor.visit_f64(f),
            }
        }

        #[cfg(feature = "arbitrary_precision")]
        fn deserialize_any<V>(self, visitor: V) -> Result<V::Value, Error>
            where V: Visitor<'de>
        {
            if let Some(u) = self.as_u64() {
                return visitor.visit_u64(u);
            } else if let Some(i) = self.as_i64() {
                return visitor.visit_i64(i);
            } else if let Some(u) = self.as_u128() {
                return visitor.visit_u128(u);
            } else if let Some(i) = self.as_i128() {
                return visitor.visit_i128(i);
            } else if let Some(f) = self.as_f64() {
                if zmij::Buffer::new().format_finite(f) == self.n || f.to_string() == self.n {
                    return visitor.visit_f64(f);
                }
            }

            visitor.visit_map(NumberDeserializer {
                number: Some(self.$($num_string)*),
            })
        }
    };

    (owned) => {
        deserialize_any!(@expand [n]);
    };

    (ref) => {
        deserialize_any!(@expand [n.clone()]);
    };
}

macro_rules! deserialize_number {
    ($deserialize:ident => $visit:ident) => {
        #[cfg(not(feature = "arbitrary_precision"))]
        fn $deserialize<V>(self, visitor: V) -> Result<V::Value, Error>
        where
            V: Visitor<'de>,
        {
            self.deserialize_any(visitor)
        }

        #[cfg(feature = "arbitrary_precision")]
        fn $deserialize<V>(self, visitor: V) -> Result<V::Value, Error>
        where
            V: de::Visitor<'de>,
        {
            visitor.$visit(tri!(self.n.parse().map_err(|_| invalid_number())))
        }
    };
}

impl<'de> Deserializer<'de> for Number {
    type Error = Error;

    deserialize_any!(owned);

    deserialize_number!(deserialize_i8 => visit_i8);
    deserialize_number!(deserialize_i16 => visit_i16);
    deserialize_number!(deserialize_i32 => visit_i32);
    deserialize_number!(deserialize_i64 => visit_i64);
    deserialize_number!(deserialize_i128 => visit_i128);
    deserialize_number!(deserialize_u8 => visit_u8);
    deserialize_number!(deserialize_u16 => visit_u16);
    deserialize_number!(deserialize_u32 => visit_u32);
    deserialize_number!(deserialize_u64 => visit_u64);
    deserialize_number!(deserialize_u128 => visit_u128);
    deserialize_number!(deserialize_f32 => visit_f32);
    deserialize_number!(deserialize_f64 => visit_f64);

    forward_to_deserialize_any! {
        bool char str string bytes byte_buf option unit unit_struct
        newtype_struct seq tuple tuple_struct map struct enum identifier
        ignored_any
    }
}

impl<'de> Deserializer<'de> for &Number {
    type Error = Error;

    deserialize_any!(ref);

    deserialize_number!(deserialize_i8 => visit_i8);
    deserialize_number!(deserialize_i16 => visit_i16);
    deserialize_number!(deserialize_i32 => visit_i32);
    deserialize_number!(deserialize_i64 => visit_i64);
    deserialize_number!(deserialize_i128 => visit_i128);
    deserialize_number!(deserialize_u8 => visit_u8);
    deserialize_number!(deserialize_u16 => visit_u16);
    deserialize_number!(deserialize_u32 => visit_u32);
    deserialize_number!(deserialize_u64 => visit_u64);
    deserialize_number!(deserialize_u128 => visit_u128);
    deserialize_number!(deserialize_f32 => visit_f32);
    deserialize_number!(deserialize_f64 => visit_f64);

    forward_to_deserialize_any! {
        bool char str string bytes byte_buf option unit unit_struct
        newtype_struct seq tuple tuple_struct map struct enum identifier
        ignored_any
    }
}

#[cfg(feature = "arbitrary_precision")]
pub(crate) struct NumberDeserializer {
    pub number: Option<String>,
}

#[cfg(feature = "arbitrary_precision")]
impl<'de> MapAccess<'de> for NumberDeserializer {
    type Error = Error;

    fn next_key_seed<K>(&mut self, seed: K) -> Result<Option<K::Value>, Error>
    where
        K: de::DeserializeSeed<'de>,
    {
        if self.number.is_none() {
            return Ok(None);
        }
        seed.deserialize(NumberFieldDeserializer).map(Some)
    }

    fn next_value_seed<V>(&mut self, seed: V) -> Result<V::Value, Error>
    where
        V: de::DeserializeSeed<'de>,
    {
        seed.deserialize(self.number.take().unwrap().into_deserializer())
    }
}

#[cfg(feature = "arbitrary_precision")]
struct NumberFieldDeserializer;

#[cfg(feature = "arbitrary_precision")]
impl<'de> Deserializer<'de> for NumberFieldDeserializer {
    type Error = Error;

    fn deserialize_any<V>(self, visitor: V) -> Result<V::Value, Error>
    where
        V: de::Visitor<'de>,
    {
        visitor.visit_borrowed_str(TOKEN)
    }

    forward_to_deserialize_any! {
        bool u8 u16 u32 u64 u128 i8 i16 i32 i64 i128 f32 f64 char str string seq
        bytes byte_buf map struct option unit newtype_struct ignored_any
        unit_struct tuple_struct tuple enum identifier
    }
}

impl From<ParserNumber> for Number {
    fn from(value: ParserNumber) -> Self {
        Self::from_parser_with_allocations(value, &crate::allocation::Unenforced)
            .expect("ordinary JSON number allocation")
    }
}

macro_rules! impl_from_unsigned {
    ($($ty:ty),*)=>{$(impl From<$ty> for Number {
        fn from(value:$ty)->Self { Self::from_u128_with_allocations(value as u128,&crate::allocation::Unenforced).expect("ordinary JSON number allocation").expect("representable scalar") }
    })*};
}
macro_rules! impl_from_signed {
    ($($ty:ty),*)=>{$(impl From<$ty> for Number {
        fn from(value:$ty)->Self { Self::from_i128_with_allocations(value as i128,&crate::allocation::Unenforced).expect("ordinary JSON number allocation").expect("representable scalar") }
    })*};
}

impl_from_unsigned!(u8, u16, u32, u64, usize);
impl_from_signed!(i8, i16, i32, i64, isize);

#[cfg(feature = "arbitrary_precision")]
impl_from_unsigned!(u128);
#[cfg(feature = "arbitrary_precision")]
impl_from_signed!(i128);

impl Number {
    #[cfg(not(feature = "arbitrary_precision"))]
    #[cold]
    pub(crate) fn unexpected(&self) -> Unexpected {
        match self.n {
            N::PosInt(u) => Unexpected::Unsigned(u),
            N::NegInt(i) => Unexpected::Signed(i),
            N::Float(f) => Unexpected::Float(f),
        }
    }

    #[cfg(feature = "arbitrary_precision")]
    #[cold]
    pub(crate) fn unexpected(&self) -> Unexpected {
        Unexpected::Other("number")
    }
}

impl Number {
    pub(crate) fn from_parser_with_allocations(
        value: ParserNumber,
        allocation: &dyn crate::allocation::Allocation,
    ) -> Result<Self, crate::allocation::AllocationError> {
        #[cfg(not(feature = "arbitrary_precision"))]
        let _ = allocation;
        #[cfg(feature = "arbitrary_precision")]
        let allocator = crate::allocation::Allocator::new(allocation);
        let n = match value {
            ParserNumber::F64(f) => {
                #[cfg(not(feature = "arbitrary_precision"))]
                {
                    N::Float(f)
                }
                #[cfg(feature = "arbitrary_precision")]
                {
                    allocator.copy_string(zmij::Buffer::new().format_finite(f))?
                }
            }
            ParserNumber::U64(u) => {
                #[cfg(not(feature = "arbitrary_precision"))]
                {
                    N::PosInt(u)
                }
                #[cfg(feature = "arbitrary_precision")]
                {
                    allocator.copy_string(itoa::Buffer::new().format(u))?
                }
            }
            ParserNumber::I64(i) => {
                #[cfg(not(feature = "arbitrary_precision"))]
                {
                    N::NegInt(i)
                }
                #[cfg(feature = "arbitrary_precision")]
                {
                    allocator.copy_string(itoa::Buffer::new().format(i))?
                }
            }
            #[cfg(feature = "arbitrary_precision")]
            ParserNumber::String(s) => s,
        };
        Ok(Number { n })
    }
}

/// Failure of the original Number source compiler. Its caller retains the
/// allocation source until the number or diagnostic storage has retired.
#[derive(Debug)]
pub enum NumberSourceError {
    /// A producer was refused before its allocation.
    Allocation(crate::allocation::AllocationError),
    /// The original Number parser rejected the source.
    Syntax(crate::Error),
}
impl fmt::Display for NumberSourceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Allocation(error) => fmt::Display::fmt(error, f),
            Self::Syntax(error) => fmt::Display::fmt(error, f),
        }
    }
}
#[cfg(feature = "std")]
impl std::error::Error for NumberSourceError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Allocation(error) => Some(error),
            Self::Syntax(error) => Some(error),
        }
    }
}
impl From<crate::allocation::AllocationError> for NumberSourceError {
    fn from(error: crate::allocation::AllocationError) -> Self {
        Self::Allocation(error)
    }
}
impl Number {
    /// Borrow exact retained decimal text when the selected representation has it.
    pub fn source_text(&self) -> Option<&str> {
        #[cfg(feature = "arbitrary_precision")]
        {
            Some(&self.n)
        }
        #[cfg(not(feature = "arbitrary_precision"))]
        {
            None
        }
    }
    /// Compile the original numeric source, preserving exact optional decimal
    /// spelling and admitting its scanner, destination and diagnostic storage.
    pub fn from_str_with_allocations(
        source: &str,
        allocation: &dyn crate::allocation::Allocation,
    ) -> Result<Self, NumberSourceError> {
        crate::de::number_from_str_with_allocations(source, allocation)
    }
}

#[cfg(test)]
mod allocation_tests {
    use super::Number;
    use crate::{
        allocation::{Allocation, AllocationError},
        NumberSourceError,
    };
    use alloc::string::ToString;
    use core::cell::Cell;
    struct Account {
        calls: Cell<usize>,
        refuse_at: usize,
        refused: Cell<bool>,
    }
    impl Account {
        fn new(refuse_at: usize) -> Self {
            Self {
                calls: Cell::new(0),
                refuse_at,
                refused: Cell::new(false),
            }
        }
    }
    impl Allocation for Account {
        fn reserve(&self, _: usize) -> Result<(), AllocationError> {
            assert!(
                !self.refused.get(),
                "numeric source continued after refusal"
            );
            let call = self.calls.get();
            self.calls.set(call + 1);
            if call == self.refuse_at {
                self.refused.set(true);
                Err(AllocationError::Refused)
            } else {
                Ok(())
            }
        }
    }
    #[test]
    fn original_number_source_preserves_spelling_errors_and_each_refusal() {
        for source in [
            "0",
            "-0",
            "42",
            "-42",
            "18446744073709551616",
            "1e400",
            "1E12",
            "3.14159265358979323846264338327950288419716939937510",
            "-1e-400",
            "01",
            "1e",
            "1.2e+q",
            "123456789012345678901234567890123456789012345678901234567890junk",
            "",
        ] {
            let expected = source.parse::<Number>();
            let account = Account::new(usize::MAX);
            let value = Number::from_str_with_allocations(source, &account);
            match (&expected, &value) {
                (Ok(expected), Ok(value)) => assert_eq!(expected, value, "{source}"),
                (Err(expected), Err(NumberSourceError::Syntax(value))) => {
                    assert_eq!(expected.to_string(), value.to_string(), "{source}")
                }
                _ => panic!("changed number source result {source}: {value:?}"),
            }
            for refuse_at in 0..account.calls.get() {
                let refused = Account::new(refuse_at);
                assert!(
                    matches!(
                        Number::from_str_with_allocations(source, &refused),
                        Err(NumberSourceError::Allocation(AllocationError::Refused))
                    ),
                    "{source} at {refuse_at}"
                );
                assert_eq!(refused.calls.get(), refuse_at + 1);
            }
        }
        #[cfg(feature = "arbitrary_precision")]
        {
            assert_eq!("1E12".parse::<Number>().unwrap().as_str(), "1e+12");
            assert_eq!("-0".parse::<Number>().unwrap().as_str(), "0");
            assert_eq!("1e400".parse::<Number>().unwrap().as_str(), "1e+400");
        }
    }
}
