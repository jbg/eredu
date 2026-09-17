//! Closed projections over the existing fixed-stack JSON validator.
//! No public span/reader constructor can skip complete source validation.
use super::borrowed_json as json;
use std::fmt;

/// Fixed JSON syntax/depth/UTF-8 diagnostic without an allocated message.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Error {
    /// Source byte offset at which validation failed.
    pub offset: usize,
    /// Whether the validator's fixed nesting stack was exceeded.
    pub depth_limit: bool,
}
impl From<json::Error> for Error {
    fn from(error: json::Error) -> Self {
        Self {
            offset: error.offset,
            depth_limit: error.kind == json::Kind::DepthLimit,
        }
    }
}
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("invalid borrowed JSON document")
    }
}
impl std::error::Error for Error {}

/// A valid JSON number outside the explicitly selected fixed integer profile.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IntegerProfileErrorKind {
    /// Fractional and exponent forms require a separate construction profile.
    NonInteger,
    /// The value is outside signed i64 / unsigned u64 representation.
    OutOfRange,
}
/// Fixed source-positioned numeric profile error, distinct from JSON grammar.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IntegerProfileError {
    /// Start byte of the actual numeric value, never a numeric-looking string.
    pub offset: usize,
    /// Actual representation that the integer-only profile cannot accept.
    pub kind: IntegerProfileErrorKind,
}
impl fmt::Display for IntegerProfileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("JSON numeric construction profile requires bounded integers")
    }
}
impl std::error::Error for IntegerProfileError {}

/// Whole validated immutable JSON source.
#[derive(Clone, Copy, Debug)]
pub struct Document<'a> {
    root: Value<'a>,
}
impl<'a> Document<'a> {
    /// Validate UTF-8, all JSON values, escapes and trailing input, using the
    /// existing fixed-stack parser before making any readonly projection.
    pub fn parse(input: &'a [u8]) -> Result<Self, Error> {
        let input = std::str::from_utf8(input).map_err(|error| Error {
            offset: error.valid_up_to(),
            depth_limit: false,
        })?;
        let mut reader = json::Reader::new(
            input,
            json::Span {
                start: 0,
                end: input.len(),
            },
        );
        let span = reader.value()?;
        reader.finish()?;
        Ok(Self {
            root: Value { input, span },
        })
    }
    /// Root value bound to this complete validated source.
    pub fn root(self) -> Value<'a> {
        self.root
    }
    /// Actual immutable complete source bytes.
    pub fn source(self) -> &'a str {
        self.root.input
    }
    /// Check the separate integer-only scalar profile. Full grammar/UTF-8 was
    /// already validated. Quoted regions use the unchanged string scanner;
    /// no number inside a key or string value participates in this profile.
    pub fn validate_integer_profile(self) -> Result<(), IntegerProfileError> {
        for number in self.numbers() {
            let raw = number.spelling().as_bytes();
            let negative = raw[0] == b'-';
            let digits = &raw[usize::from(negative)..];
            if !digits.iter().all(u8::is_ascii_digit) {
                return Err(IntegerProfileError {
                    offset: number.offset(),
                    kind: IntegerProfileErrorKind::NonInteger,
                });
            }
            let limit = if negative { 1u64 << 63 } else { u64::MAX };
            let mut value = 0u64;
            for &digit in digits {
                value = value
                    .checked_mul(10)
                    .and_then(|v| v.checked_add(u64::from(digit - b'0')))
                    .filter(|&v| v <= limit)
                    .ok_or(IntegerProfileError {
                        offset: number.offset(),
                        kind: IntegerProfileErrorKind::OutOfRange,
                    })?;
            }
        }
        Ok(())
    }
    /// Borrow all actual numeric tokens in document order, excluding strings
    /// and keys. Every returned slice belongs to this fully validated document.
    pub fn numbers(self) -> Numbers<'a> {
        Numbers {
            input: self.source(),
            at: 0,
        }
    }
    /// Checked fixed parser/projection/numeric-validation controls. These are
    /// concrete live view/iterator populations, not a source-length multiplier.
    pub fn control_bytes() -> Option<usize> {
        use std::mem::size_of;
        IntoIterator::into_iter([
            json::stack_bytes(),
            size_of::<[json::Reader<'a>; 2]>(),
            size_of::<Self>(),
            size_of::<Result<Self, Error>>(),
            size_of::<[Value<'a>; 3]>(),
            size_of::<[Object<'a>; 2]>(),
            size_of::<[Array<'a>; 2]>(),
            size_of::<[Entries<'a>; 2]>(),
            size_of::<[Option<(StringValue<'a>, Value<'a>)>; 2]>(),
            size_of::<[StringValue<'a>; 4]>(),
            size_of::<[Bytes<'a>; 2]>(),
            size_of::<[Option<u8>; 2]>(),
            size_of::<[usize; 4]>(),
            size_of::<[u64; 2]>(),
            size_of::<bool>(),
            size_of::<Numbers<'a>>(),
            size_of::<NumberValue<'a>>(),
            size_of::<Option<NumberValue<'a>>>(),
            size_of::<IntegerProfileError>(),
            size_of::<Result<(), IntegerProfileError>>(),
            size_of::<json::Error>(),
            size_of::<Error>(),
        ])
        .try_fold(0usize, usize::checked_add)
    }
}
/// Borrowed scalar spelling proven to be an actual numeric JSON value.
#[derive(Clone, Copy, Debug)]
pub struct NumberValue<'a> {
    input: &'a str,
    start: usize,
    end: usize,
}
impl<'a> NumberValue<'a> {
    /// Exact original spelling, without conversion or normalized storage.
    pub fn spelling(self) -> &'a str {
        &self.input[self.start..self.end]
    }
    /// Original source byte offset.
    pub fn offset(self) -> usize {
        self.start
    }
    /// Original source byte range for a parser that retains document positions.
    pub fn range(self) -> std::ops::Range<usize> {
        self.start..self.end
    }
}
/// Allocation-free numeric projection over a fully validated document.
#[derive(Clone, Debug)]
pub struct Numbers<'a> {
    input: &'a str,
    at: usize,
}
impl<'a> Iterator for Numbers<'a> {
    type Item = NumberValue<'a>;
    fn next(&mut self) -> Option<Self::Item> {
        let bytes = self.input.as_bytes();
        while self.at < bytes.len() {
            if bytes[self.at] == b'"' {
                let mut reader = json::Reader::new(
                    self.input,
                    json::Span {
                        start: self.at,
                        end: self.input.len(),
                    },
                );
                reader.string().expect("whole document validated");
                self.at = reader.pos;
            } else if matches!(bytes[self.at], b'-' | b'0'..=b'9') {
                let start = self.at;
                while self.at < bytes.len()
                    && matches!(
                        bytes[self.at],
                        b'-' | b'+' | b'.' | b'e' | b'E' | b'0'..=b'9'
                    )
                {
                    self.at += 1;
                }
                return Some(NumberValue {
                    input: self.input,
                    start,
                    end: self.at,
                });
            } else {
                self.at += 1;
            }
        }
        None
    }
}
/// Validated subvalue; ranges can be obtained only from a validated document.
///
/// ```compile_fail
/// use tokenizers::utils::json_view::Value;
/// fn unchecked(input: &str) { let _ = Value::new(input, 0, input.len()); }
/// ```
#[derive(Clone, Copy, Debug)]
pub struct Value<'a> {
    input: &'a str,
    span: json::Span,
}
impl<'a> Value<'a> {
    fn reader(self) -> json::Reader<'a> {
        json::Reader::new(self.input, self.span)
    }
    fn first(self) -> u8 {
        self.input.as_bytes()[self.span.start]
    }
    /// Null test without constructing a serde value.
    pub fn is_null(self) -> bool {
        self.first() == b'n'
    }
    /// Borrow a decoded-string iterator after the original escape validation.
    pub fn string(self) -> Option<StringValue<'a>> {
        (self.first() == b'"')
            .then(|| StringValue(self.reader().string().expect("validated JSON string")))
    }
    /// Borrow an array iterator; no owned collection is created.
    pub fn array(self) -> Option<Array<'a>> {
        (self.first() == b'[').then(|| Array {
            input: self.input,
            inner: self.reader().array().expect("validated JSON array"),
        })
    }
    /// Borrow an object view preserving duplicate-key last-value semantics.
    pub fn object(self) -> Option<Object<'a>> {
        (self.first() == b'{').then_some(Object { value: self })
    }
}
/// Immutable decoded UTF-8 string, including escaped surrogate pairs.
#[derive(Clone, Copy, Debug)]
pub struct StringValue<'a>(json::Text<'a>);
impl<'a> StringValue<'a> {
    /// Stream the actual decoded bytes with fixed inline UTF-8 scratch.
    pub fn bytes(self) -> Bytes<'a> {
        Bytes(self.0.bytes())
    }
    /// Compare decoded bytes directly with a UTF-8 string.
    pub fn is(self, text: &str) -> bool {
        self.0.is(text)
    }
    /// Compare two decoded keys without an owned allocation.
    pub fn same(self, other: Self) -> bool {
        let mut left = self.bytes();
        let mut right = other.bytes();
        loop {
            match (left.next(), right.next()) {
                (None, None) => return true,
                (Some(a), Some(b)) if a == b => {}
                _ => return false,
            }
        }
    }
    /// Exact decoded byte count.
    pub fn len(self) -> usize {
        self.0.len()
    }
    /// Whether the decoded string is empty.
    pub fn is_empty(self) -> bool {
        self.bytes().next().is_none()
    }
}
/// Fixed-state decoded byte iterator; callers cannot supply arbitrary spans.
#[derive(Clone)]
pub struct Bytes<'a>(json::Bytes<'a>);
impl Iterator for Bytes<'_> {
    type Item = u8;
    fn next(&mut self) -> Option<u8> {
        self.0.next()
    }
}
/// Streaming readonly array from a validated complete source.
pub struct Array<'a> {
    input: &'a str,
    inner: json::Array<'a>,
}
impl<'a> Iterator for Array<'a> {
    type Item = Value<'a>;
    fn next(&mut self) -> Option<Self::Item> {
        self.inner
            .next()
            .expect("validated JSON array")
            .map(|span| Value {
                input: self.input,
                span,
            })
    }
}
/// Readonly object over actual validated source, with no map construction.
#[derive(Clone, Copy, Debug)]
pub struct Object<'a> {
    value: Value<'a>,
}
impl<'a> Object<'a> {
    /// Visit actual key occurrences in source order, including duplicate keys.
    pub fn entries(self) -> Entries<'a> {
        Entries {
            input: self.value.input,
            inner: self.value.reader().object().expect("validated JSON object"),
        }
    }
    /// Last value of an exact decoded key, matching ordinary serde maps.
    pub fn get(self, name: &str) -> Option<Value<'a>> {
        let mut entries = self.entries();
        let mut found = None;
        while let Some((key, value)) = entries.next() {
            if key.is(name) {
                found = Some(value);
            }
        }
        found
    }
    /// Number of distinct decoded keys. Rescans preserve ordinary object-size
    /// semantics without allocating a hash table, including escaped duplicates.
    pub fn unique_len(self) -> usize {
        let mut entries = self.entries();
        let mut index = 0;
        let mut count = 0;
        while let Some((key, _)) = entries.next() {
            let mut earlier = self.entries();
            let mut duplicate = false;
            for _ in 0..index {
                let (prior, _) = earlier.next().expect("previous validated entry");
                if prior.same(key) {
                    duplicate = true;
                    break;
                }
            }
            if !duplicate {
                count += 1;
            }
            // Each occurrence consumes at least one distinct source byte, so
            // these counters cannot exceed the original slice's isize bound.
            index += 1;
        }
        count
    }
}
/// Actual object entries from a checked source; fixed parser controls only.
pub struct Entries<'a> {
    input: &'a str,
    inner: json::Object<'a>,
}
impl<'a> Iterator for Entries<'a> {
    type Item = (StringValue<'a>, Value<'a>);
    fn next(&mut self) -> Option<Self::Item> {
        self.inner
            .next()
            .expect("validated JSON object")
            .map(|(text, span)| {
                (
                    StringValue(text),
                    Value {
                        input: self.input,
                        span,
                    },
                )
            })
    }
}

#[cfg(test)]
mod tests;
