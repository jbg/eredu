//! Source-planned events from the ordinary serde JSON deserializer.
//!
//! This does not create owned Values or caller record DTOs. A consumer may fill
//! separately admitted fixed destinations, or merely compare borrowed fields.
//! It must account its own state/allocations; this parser grants no authority.
mod value;
pub use value::{from_prefix_with_allocations, from_slice_with_allocations, ValueError};

use crate::{Error, Result};
use alloc::vec::Vec;
use core::{fmt, mem::size_of};
use serde::de::{DeserializeSeed, MapAccess, SeqAccess, Visitor};

/// One lexical callback. Strings borrow parser/input storage only for the call.
#[derive(Debug)]
pub enum Event<'a> {
    /// Enter an object in source order.
    Object,
    /// One decoded object key.
    Key(&'a str),
    /// Leave that object.
    EndObject,
    /// Enter an array in source order.
    Array,
    /// Leave that array.
    EndArray,
    /// A decoded UTF-8 value.
    String(&'a str),
    /// A signed integer.
    I64(i64),
    /// An unsigned integer.
    U64(u64),
    /// An ordinary finite parsed floating value.
    F64(f64),
    /// An exact retained numeric representation, borrowed for this callback.
    Number(&'a crate::Number),
    /// A boolean value.
    Bool(bool),
    /// JSON null.
    Null,
}
/// Fixed consumer of parser events. Store validation status in the consumer;
/// returning a custom serde error would create an unrelated diagnostic producer.
pub trait Sink {
    /// Process one event before its borrowed string storage can be reused.
    fn event(&mut self, event: Event<'_>);
    /// Whether the consumer retained a terminal destination failure. Parsing
    /// stops before another numeric producer can reach the original account.
    fn stopped(&self) -> bool {
        false
    }
}
/// Fixed planning refusal, before parser allocation or callback entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlanError {
    /// Source exceeds ordinary recursion/input counters or an allocation bound.
    Source,
    /// A concrete storage sum overflowed.
    Overflow,
}
impl fmt::Display for PlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Source => "JSON event source exceeds the ordinary parser profile",
            Self::Overflow => "JSON event parser storage overflow",
        })
    }
}
#[cfg(feature = "std")]
impl std::error::Error for PlanError {}
/// Concrete parser storage; caller destination and callback state are separate.
#[derive(Clone, Copy, Debug)]
pub struct Requirements {
    scratch: usize,
    temporary: usize,
    controls: usize,
    failure: usize,
}
impl Requirements {
    /// Exact preallocated parser scratch request.
    pub fn scratch_bytes(self) -> usize {
        self.scratch
    }
    /// Finite numeric scratch across possible source tokens; no refund assumed.
    pub fn temporary_bytes(self) -> usize {
        self.temporary
    }
    /// Named parser, traversal and transport controls for the selected sink.
    pub fn control_bytes(self) -> usize {
        self.controls
    }
    /// The actual private syntax error allocation.
    pub fn failure_bytes(self) -> usize {
        self.failure
    }
    /// Checked total to admit before invoking the ordinary parser.
    pub fn required_bytes(self) -> usize {
        self.scratch + self.temporary + self.controls + self.failure
    }
}
/// Whether this compiled serde map retains first insertion order rather than
/// lexical key order. Duplicate insertion replaces the value at that position.
/// Event consumers can match ordinary Value construction without allocating a
/// second map or guessing another crate's selected feature flags.
pub const fn preserves_object_order() -> bool {
    cfg!(feature = "preserve_order")
}

/// Move-only plan borrowing the exact immutable receipt bytes.
#[derive(Debug)]
pub struct Plan<'a> {
    input: &'a [u8],
    scratch: usize,
    depth: usize,
    numbers: usize,
}
impl<'a> Plan<'a> {
    /// A nonallocating storage census, not JSON validation. String escapes cannot
    /// expand beyond source bytes; UTF-8 emission additionally reserves four
    /// scratch slots in the ordinary worker. Syntax validation stays in parse.
    pub fn prepare(input: &'a [u8]) -> core::result::Result<Self, PlanError> {
        Self::prepare_source(input, false)
    }
    /// Plan one JSON prefix followed by another language or value. Scratch and
    /// numeric storage cover the supplied source; parser traversal is capped by
    /// its actual recursion limit rather than braces in unrelated trailing text.
    pub fn prepare_prefix(input: &'a [u8]) -> core::result::Result<Self, PlanError> {
        Self::prepare_source(input, true)
    }
    fn prepare_source(input: &'a [u8], _prefix: bool) -> core::result::Result<Self, PlanError> {
        if input.len() > i32::MAX as usize {
            return Err(PlanError::Source);
        }
        let scratch = input.len().checked_add(4).ok_or(PlanError::Overflow)?;
        if scratch > isize::MAX as usize {
            return Err(PlanError::Source);
        }
        let (mut string, mut escaped, mut depth, mut maximum, mut numbers, mut number) =
            (false, false, 0usize, 0usize, 0usize, false);
        for &byte in input {
            if string {
                if escaped {
                    escaped = false;
                } else if byte == b'\\' {
                    escaped = true;
                } else if byte == b'"' {
                    string = false;
                }
                continue;
            }
            if byte == b'"' {
                string = true;
                number = false;
                continue;
            }
            match byte {
                b'{' | b'[' => {
                    depth = depth.checked_add(1).ok_or(PlanError::Overflow)?;
                    maximum = maximum.max(depth);
                }
                b'}' | b']' => depth = depth.saturating_sub(1),
                _ => (),
            }
            let numeric = matches!(byte, b'-' | b'+' | b'0'..=b'9' | b'.' | b'e' | b'E');
            if numeric && !number {
                numbers = numbers.checked_add(1).ok_or(PlanError::Overflow)?;
            }
            number = numeric;
        }
        // The ordinary parser diagnoses nesting at its recursion limit. The
        // source census caps retained frames at that same limit, including a
        // malformed trailing suffix; it does not reinterpret depth as syntax.
        Ok(Self {
            input,
            scratch,
            depth: maximum.min(128) + 1,
            numbers,
        })
    }
    /// Source-derived parser requirement before any callback or payload work.
    pub fn requirements<S: Sink>(&self) -> core::result::Result<Requirements, PlanError> {
        let frames = size_of::<Value<'_, '_, S>>()
            .checked_add(size_of::<Key<'_, S>>())
            .and_then(|n| n.checked_add(size_of::<Event<'_>>()))
            .and_then(|n| n.checked_add(size_of::<Result<()>>()))
            .and_then(|n| n.checked_add(size_of::<(usize, bool, &mut S)>()))
            .and_then(|n| n.checked_mul(self.depth))
            .ok_or(PlanError::Overflow)?;
        let controls = crate::de::bounded_number_control_bytes()
            .ok_or(PlanError::Overflow)?
            .checked_add(frames)
            .and_then(|n| n.checked_add(crate::de::bounded_event_access_control_bytes(self.depth)?))
            .and_then(|n| n.checked_add(size_of::<Self>()))
            .and_then(|n| n.checked_add(size_of::<Vec<u8>>()))
            .and_then(|n| {
                n.checked_add(size_of::<
                    crate::StreamDeserializer<
                        '_,
                        crate::read::SliceRead<'_>,
                        serde::de::IgnoredAny,
                    >,
                >())
            })
            .and_then(|n| n.checked_add(size_of::<Option<Result<()>>>()))
            .and_then(|n| n.checked_add(size_of::<Requirements>()))
            .and_then(|n| n.checked_add(size_of::<Context<'_>>()))
            .and_then(|n| n.checked_add(size_of::<core::result::Result<Requirements, PlanError>>()))
            .ok_or(PlanError::Overflow)?;
        #[cfg(all(feature = "float_roundtrip", not(feature = "arbitrary_precision")))]
        let temporary = crate::lexical::bounded::temporary_bytes()
            .ok_or(PlanError::Overflow)?
            .checked_mul(self.numbers)
            .ok_or(PlanError::Overflow)?;
        #[cfg(not(all(feature = "float_roundtrip", not(feature = "arbitrary_precision"))))]
        let temporary = {
            let _ = self.numbers;
            0
        };
        let failure = Error::bounded_number_storage_bytes()
            .checked_mul(2)
            .ok_or(PlanError::Overflow)?;
        self.scratch
            .checked_add(temporary)
            .and_then(|n| n.checked_add(controls))
            .and_then(|n| n.checked_add(failure))
            .ok_or(PlanError::Overflow)?;
        Ok(Requirements {
            scratch: self.scratch,
            temporary,
            controls,
            failure,
        })
    }
    /// Consume the source through the existing deserializer. The sink must
    /// already own any destination/custody; event arrival does not certify a
    /// complete record until this returns success and its own checks pass.
    pub fn parse<S: Sink>(
        self,
        sink: &mut S,
        allocation: &dyn crate::allocation::Allocation,
    ) -> Result<()> {
        let context = Context::new(allocation);
        let mut parser =
            crate::de::bounded_event_deserializer(self.input, self.scratch, &context.allocation);
        let parsed = Value(sink, &context)
            .deserialize(&mut parser)
            .and_then(|()| parser.end());
        context.finish(parsed)
    }
    /// Parse one prefix through the ordinary stream worker. The returned byte
    /// count has exactly the stream deserializer's boundary semantics. No value
    /// is returned for an empty or whitespace-only source.
    pub fn parse_prefix<S: Sink>(
        self,
        sink: &mut S,
        allocation: &dyn crate::allocation::Allocation,
    ) -> Result<Option<usize>> {
        let context = Context::new(allocation);
        let parser =
            crate::de::bounded_event_deserializer(self.input, self.scratch, &context.allocation);
        let mut stream = parser.into_iter::<serde::de::IgnoredAny>();
        let result = match stream.next_seed(Value(sink, &context)) {
            Some(result) => result.map(|()| Some(stream.byte_offset())),
            None => Ok(None),
        };
        context.finish(result)
    }
}
struct Context<'a> {
    allocation: crate::allocation::Session<'a>,
    #[cfg(feature = "arbitrary_precision")]
    number_failure: core::cell::RefCell<Option<crate::NumberSourceError>>,
}
impl<'a> Context<'a> {
    fn new(allocation: &'a dyn crate::allocation::Allocation) -> Self {
        Self {
            allocation: crate::allocation::Session::new(allocation),
            #[cfg(feature = "arbitrary_precision")]
            number_failure: core::cell::RefCell::new(None),
        }
    }
    fn finish<T>(&self, result: Result<T>) -> Result<T> {
        if let Some(error) = self.allocation.failure() {
            return Err(Error::syntax(
                crate::error::ErrorCode::Allocation(error),
                0,
                0,
            ));
        }
        #[cfg(feature = "arbitrary_precision")]
        if let Some(failure) = self.number_failure.borrow_mut().take() {
            return Err(match failure {
                crate::NumberSourceError::Allocation(error) => {
                    Error::syntax(crate::error::ErrorCode::Allocation(error), 0, 0)
                }
                crate::NumberSourceError::Syntax(error) => {
                    Error::custom_with_allocations(error, &self.allocation).unwrap_or_else(
                        |error| Error::syntax(crate::error::ErrorCode::Allocation(error), 0, 0),
                    )
                }
            });
        }
        result
    }
}
struct Value<'a, 'b, S>(&'a mut S, &'a Context<'b>);
impl<'de, S: Sink> DeserializeSeed<'de> for Value<'_, '_, S> {
    type Value = ();
    fn deserialize<D: serde::Deserializer<'de>>(
        self,
        deserializer: D,
    ) -> core::result::Result<(), D::Error> {
        deserializer.deserialize_any(self)
    }
}
impl<S: Sink> Value<'_, '_, S> {
    fn emit<E: serde::de::Error>(&mut self, event: Event<'_>) -> core::result::Result<(), E> {
        self.0.event(event);
        if self.0.stopped() {
            Err(E::custom(""))
        } else {
            Ok(())
        }
    }
}
impl<'de, S: Sink> Visitor<'de> for Value<'_, '_, S> {
    type Value = ();
    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("a JSON value")
    }
    fn visit_unit<E: serde::de::Error>(mut self) -> core::result::Result<(), E> {
        self.emit(Event::Null)?;
        Ok(())
    }
    fn visit_bool<E: serde::de::Error>(mut self, value: bool) -> core::result::Result<(), E> {
        self.emit(Event::Bool(value))?;
        Ok(())
    }
    fn visit_i64<E: serde::de::Error>(mut self, value: i64) -> core::result::Result<(), E> {
        self.emit(Event::I64(value))?;
        Ok(())
    }
    fn visit_u64<E: serde::de::Error>(mut self, value: u64) -> core::result::Result<(), E> {
        self.emit(Event::U64(value))?;
        Ok(())
    }
    fn visit_f64<E: serde::de::Error>(mut self, value: f64) -> core::result::Result<(), E> {
        self.emit(Event::F64(value))?;
        Ok(())
    }
    fn visit_str<E: serde::de::Error>(mut self, value: &str) -> core::result::Result<(), E> {
        self.emit(Event::String(value))?;
        Ok(())
    }
    fn visit_borrowed_str<E: serde::de::Error>(
        mut self,
        value: &'de str,
    ) -> core::result::Result<(), E> {
        self.visit_str(value)
    }
    fn visit_seq<A: SeqAccess<'de>>(mut self, mut seq: A) -> core::result::Result<(), A::Error> {
        self.emit(Event::Array)?;
        while seq
            .next_element_seed(Value(&mut *self.0, self.1))?
            .is_some()
        {}
        self.emit(Event::EndArray)?;
        Ok(())
    }
    fn visit_map<A: MapAccess<'de>>(mut self, mut map: A) -> core::result::Result<(), A::Error> {
        #[cfg(feature = "arbitrary_precision")]
        {
            match map.next_key_seed(FirstKey(&mut *self.0))? {
                Some(true) => {
                    let number = map.next_value_seed(crate::number::NumberFromStringSeed {
                        allocation: &self.1.allocation,
                        failure: Some(&self.1.number_failure),
                    })?;
                    self.emit(Event::Number(&number.value))?;
                    return Ok(());
                }
                Some(false) => map.next_value_seed(Value(&mut *self.0, self.1))?,
                None => {
                    self.emit(Event::Object)?;
                    self.emit(Event::EndObject)?;
                    return Ok(());
                }
            }
        }
        #[cfg(not(feature = "arbitrary_precision"))]
        self.emit(Event::Object)?;
        while map.next_key_seed(Key(&mut *self.0))?.is_some() {
            map.next_value_seed(Value(&mut *self.0, self.1))?;
        }
        self.emit(Event::EndObject)?;
        Ok(())
    }
}
struct Key<'a, S>(&'a mut S);
impl<'de, S: Sink> DeserializeSeed<'de> for Key<'_, S> {
    type Value = ();
    fn deserialize<D: serde::Deserializer<'de>>(
        self,
        deserializer: D,
    ) -> core::result::Result<(), D::Error> {
        deserializer.deserialize_str(self)
    }
}
impl<'de, S: Sink> Visitor<'de> for Key<'_, S> {
    type Value = ();
    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("a JSON object key")
    }
    fn visit_str<E: serde::de::Error>(self, value: &str) -> core::result::Result<(), E> {
        self.0.event(Event::Key(value));
        if self.0.stopped() {
            Err(E::custom(""))
        } else {
            Ok(())
        }
    }
    fn visit_borrowed_str<E: serde::de::Error>(
        self,
        value: &'de str,
    ) -> core::result::Result<(), E> {
        self.visit_str(value)
    }
}

#[cfg(feature = "arbitrary_precision")]
struct FirstKey<'a, S>(&'a mut S);
#[cfg(feature = "arbitrary_precision")]
impl<'de, S: Sink> DeserializeSeed<'de> for FirstKey<'_, S> {
    type Value = bool;
    fn deserialize<D: serde::Deserializer<'de>>(
        self,
        deserializer: D,
    ) -> core::result::Result<bool, D::Error> {
        deserializer.deserialize_str(self)
    }
}
#[cfg(feature = "arbitrary_precision")]
impl<'de, S: Sink> Visitor<'de> for FirstKey<'_, S> {
    type Value = bool;
    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("a JSON object key")
    }
    fn visit_str<E: serde::de::Error>(self, key: &str) -> core::result::Result<bool, E> {
        if key == crate::number::TOKEN {
            Ok(true)
        } else {
            self.0.event(Event::Object);
            if self.0.stopped() {
                return Err(E::custom(""));
            }
            self.0.event(Event::Key(key));
            if self.0.stopped() {
                Err(E::custom(""))
            } else {
                Ok(false)
            }
        }
    }
    fn visit_borrowed_str<E: serde::de::Error>(
        self,
        key: &'de str,
    ) -> core::result::Result<bool, E> {
        self.visit_str(key)
    }
}

#[cfg(test)]
mod prefix_tests {
    use super::*;
    struct Count(usize);
    impl Sink for Count {
        fn event(&mut self, _: Event<'_>) {
            self.0 += 1;
        }
    }

    #[test]
    fn event_prefix_uses_ordinary_stream_boundaries_and_errors() {
        for input in [
            "",
            " \n",
            "null suffix",
            "123 suffix",
            "123suffix",
            "truefalse",
            "\"ready\"suffix",
            "[1,2]suffix",
            "{\"a\":true}suffix",
            "{\"a\":",
            "[1,]",
            "false:next",
            "-12.3e4,rest",
            "\"\\ud83d\\ude00\" ",
        ] {
            let mut stream =
                crate::Deserializer::from_slice(input.as_bytes()).into_iter::<crate::Value>();
            let expected = stream.next();
            let expected_offset = stream.byte_offset();
            let plan = Plan::prepare_prefix(input.as_bytes()).unwrap();
            let _paid = plan.requirements::<Count>().unwrap();
            let got = plan.parse_prefix(&mut Count(0), &crate::allocation::Unenforced);
            match (expected, got) {
                (None, Ok(None)) => (),
                (Some(Ok(_)), Ok(Some(offset))) => assert_eq!(offset, expected_offset, "{input:?}"),
                (Some(Err(expected)), Err(got)) => {
                    assert_eq!(expected.classify(), got.classify(), "{input:?}");
                    assert_eq!(
                        (expected.line(), expected.column()),
                        (got.line(), got.column()),
                        "{input:?}"
                    );
                }
                (expected, got) => panic!("prefix mismatch {input:?}: {expected:?}, {got:?}"),
            }
        }
    }

    #[test]
    fn unrelated_trailing_braces_do_not_change_prefix_admission() {
        let mut input = alloc::string::String::from("{} ");
        for _ in 0..256 {
            input.push('{');
        }
        let plan = Plan::prepare_prefix(input.as_bytes()).unwrap();
        assert_eq!(
            plan.parse_prefix(&mut Count(0), &crate::allocation::Unenforced)
                .unwrap(),
            Some(2)
        );
    }
}
