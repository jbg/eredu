//! Source-planned events from the ordinary serde JSON deserializer.
//!
//! This does not create owned Values or caller record DTOs. A consumer may fill
//! separately admitted fixed destinations, or merely compare borrowed fields.
//! It must account its own state/allocations; this parser grants no authority.
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
}
/// Fixed planning refusal, before parser allocation or callback entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlanError {
    /// Arbitrary precision introduces a separate retained numeric representation.
    Features,
    /// Source exceeds ordinary recursion/input counters or an allocation bound.
    Source,
    /// A concrete storage sum overflowed.
    Overflow,
}
impl fmt::Display for PlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Features => "JSON event numeric feature profile is unqualified",
            Self::Source => "JSON event source exceeds the ordinary parser profile",
            Self::Overflow => "JSON event parser storage overflow",
        })
    }
}
#[cfg(feature="std")]
impl std::error::Error for PlanError {}
/// Concrete parser storage; caller destination and callback state are separate.
#[derive(Clone, Copy, Debug)]
pub struct Requirements { scratch: usize, temporary: usize, controls: usize, failure: usize }
impl Requirements {
    /// Exact preallocated parser scratch request.
    pub fn scratch_bytes(self) -> usize { self.scratch }
    /// Finite numeric scratch across possible source tokens; no refund assumed.
    pub fn temporary_bytes(self) -> usize { self.temporary }
    /// Named parser, traversal and transport controls for the selected sink.
    pub fn control_bytes(self) -> usize { self.controls }
    /// The actual private syntax error allocation.
    pub fn failure_bytes(self) -> usize { self.failure }
    /// Checked total to admit before invoking the ordinary parser.
    pub fn required_bytes(self) -> usize { self.scratch + self.temporary + self.controls + self.failure }
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
pub struct Plan<'a> { input: &'a [u8], scratch: usize, depth: usize, numbers: usize }
impl<'a> Plan<'a> {
    /// A nonallocating storage census, not JSON validation. String escapes cannot
    /// expand beyond source bytes; UTF-8 emission additionally reserves four
    /// scratch slots in the ordinary worker. Syntax validation stays in parse.
    pub fn prepare(input: &'a [u8]) -> core::result::Result<Self, PlanError> {
        if cfg!(feature="arbitrary_precision") { return Err(PlanError::Features); }
        if input.len() > i32::MAX as usize { return Err(PlanError::Source); }
        let scratch = input.len().checked_add(4).ok_or(PlanError::Overflow)?;
        if scratch > isize::MAX as usize { return Err(PlanError::Source); }
        let (mut string, mut escaped, mut depth, mut maximum, mut numbers, mut number) = (false, false, 0usize, 0usize, 0usize, false);
        for &byte in input {
            if string {
                if escaped { escaped = false; }
                else if byte == b'\\' { escaped = true; }
                else if byte == b'"' { string = false; }
                continue;
            }
            if byte == b'"' { string = true; number = false; continue; }
            match byte {
                b'{' | b'[' => { depth = depth.checked_add(1).ok_or(PlanError::Overflow)?; maximum = maximum.max(depth); }
                b'}' | b']' => depth = depth.saturating_sub(1),
                _ => (),
            }
            let numeric = matches!(byte, b'-' | b'+' | b'0'..=b'9' | b'.' | b'e' | b'E');
            if numeric && !number { numbers = numbers.checked_add(1).ok_or(PlanError::Overflow)?; }
            number = numeric;
        }
        if maximum > 128 { return Err(PlanError::Source); }
        Ok(Self { input, scratch, depth: maximum + 1, numbers })
    }
    /// Source-derived parser requirement before any callback or payload work.
    pub fn requirements<S: Sink>(&self) -> core::result::Result<Requirements, PlanError> {
        let frames = size_of::<Value<'_, S>>()
            .checked_add(size_of::<Key<'_, S>>()).and_then(|n|n.checked_add(size_of::<Event<'_>>()))
            .and_then(|n|n.checked_add(size_of::<Result<()>>()))
            .and_then(|n|n.checked_add(size_of::<(usize, bool, &mut S)>()))
            .and_then(|n|n.checked_mul(self.depth)).ok_or(PlanError::Overflow)?;
        let controls = crate::de::bounded_number_control_bytes().ok_or(PlanError::Overflow)?
            .checked_add(frames)
            .and_then(|n|n.checked_add(crate::de::bounded_event_access_control_bytes(self.depth)?))
            .and_then(|n|n.checked_add(size_of::<Self>()))
            .and_then(|n|n.checked_add(size_of::<Vec<u8>>()))
            .and_then(|n|n.checked_add(size_of::<Requirements>()))
            .and_then(|n|n.checked_add(size_of::<core::result::Result<Requirements, PlanError>>()))
            .ok_or(PlanError::Overflow)?;
        #[cfg(feature="float_roundtrip")]
        let temporary = crate::lexical::bounded::temporary_bytes().ok_or(PlanError::Overflow)?
            .checked_mul(self.numbers).ok_or(PlanError::Overflow)?;
        #[cfg(not(feature="float_roundtrip"))]
        let temporary = { let _ = self.numbers; 0 };
        let failure = Error::bounded_number_storage_bytes();
        self.scratch.checked_add(temporary).and_then(|n|n.checked_add(controls))
            .and_then(|n|n.checked_add(failure)).ok_or(PlanError::Overflow)?;
        Ok(Requirements { scratch: self.scratch, temporary, controls, failure })
    }
    /// Consume the source through the existing deserializer. The sink must
    /// already own any destination/custody; event arrival does not certify a
    /// complete record until this returns success and its own checks pass.
    pub fn parse<S: Sink>(self, sink: &mut S) -> Result<()> {
        let mut parser = crate::de::bounded_event_deserializer(self.input, self.scratch);
        Value(sink).deserialize(&mut parser)?;
        parser.end()
    }
}
struct Value<'a, S>(&'a mut S);
impl<'de, S: Sink> DeserializeSeed<'de> for Value<'_, S> {
    type Value = ();
    fn deserialize<D: serde::Deserializer<'de>>(self, deserializer: D) -> core::result::Result<(), D::Error> {
        deserializer.deserialize_any(self)
    }
}
impl<'de, S: Sink> Visitor<'de> for Value<'_, S> {
    type Value = ();
    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result { f.write_str("a JSON value") }
    fn visit_unit<E: serde::de::Error>(self) -> core::result::Result<(), E> { self.0.event(Event::Null); Ok(()) }
    fn visit_bool<E: serde::de::Error>(self, value: bool) -> core::result::Result<(), E> { self.0.event(Event::Bool(value)); Ok(()) }
    fn visit_i64<E: serde::de::Error>(self, value: i64) -> core::result::Result<(), E> { self.0.event(Event::I64(value)); Ok(()) }
    fn visit_u64<E: serde::de::Error>(self, value: u64) -> core::result::Result<(), E> { self.0.event(Event::U64(value)); Ok(()) }
    fn visit_f64<E: serde::de::Error>(self, value: f64) -> core::result::Result<(), E> { self.0.event(Event::F64(value)); Ok(()) }
    fn visit_str<E: serde::de::Error>(self, value: &str) -> core::result::Result<(), E> { self.0.event(Event::String(value)); Ok(()) }
    fn visit_borrowed_str<E: serde::de::Error>(self, value: &'de str) -> core::result::Result<(), E> { self.visit_str(value) }
    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> core::result::Result<(), A::Error> {
        self.0.event(Event::Array);
        while seq.next_element_seed(Value(&mut *self.0))?.is_some() {}
        self.0.event(Event::EndArray); Ok(())
    }
    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> core::result::Result<(), A::Error> {
        self.0.event(Event::Object);
        while map.next_key_seed(Key(&mut *self.0))?.is_some() { map.next_value_seed(Value(&mut *self.0))?; }
        self.0.event(Event::EndObject); Ok(())
    }
}
struct Key<'a, S>(&'a mut S);
impl<'de, S: Sink> DeserializeSeed<'de> for Key<'_, S> {
    type Value = ();
    fn deserialize<D: serde::Deserializer<'de>>(self, deserializer: D) -> core::result::Result<(), D::Error> { deserializer.deserialize_str(self) }
}
impl<'de, S: Sink> Visitor<'de> for Key<'_, S> {
    type Value = ();
    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result { f.write_str("a JSON object key") }
    fn visit_str<E: serde::de::Error>(self, value: &str) -> core::result::Result<(), E> { self.0.event(Event::Key(value)); Ok(()) }
    fn visit_borrowed_str<E: serde::de::Error>(self, value: &'de str) -> core::result::Result<(), E> { self.visit_str(value) }
}
