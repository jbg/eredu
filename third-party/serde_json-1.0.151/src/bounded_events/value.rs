//! Owned values from the same event/deserializer worker.
use super::{Event, Plan, PlanError, Sink};
use crate::{
    allocation::{Allocation, AllocationError, Allocator},
    Map, Value,
};
use alloc::{string::String, vec::Vec};
use core::{fmt, mem::size_of};

/// A JSON producer failure. The enclosing source or error owner must retain the
/// original allocation account until its values and this error have retired.
#[derive(Debug)]
pub enum ValueError {
    /// The concrete parser source cannot be planned.
    Plan(PlanError),
    /// Reached source storage was refused before construction.
    Allocation(AllocationError),
    /// The ordinary JSON deserializer rejected the source.
    Syntax(crate::Error),
    /// A prefix source did not contain a JSON value.
    Empty,
}
impl fmt::Display for ValueError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Plan(error) => error.fmt(f),
            Self::Allocation(error) => error.fmt(f),
            Self::Syntax(error) => error.fmt(f),
            Self::Empty => f.write_str("empty JSON source"),
        }
    }
}
#[cfg(feature = "std")]
impl std::error::Error for ValueError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Plan(error) => Some(error),
            Self::Allocation(error) => Some(error),
            Self::Syntax(error) => Some(error),
            Self::Empty => None,
        }
    }
}
impl From<PlanError> for ValueError {
    fn from(error: PlanError) -> Self {
        Self::Plan(error)
    }
}
impl From<AllocationError> for ValueError {
    fn from(error: AllocationError) -> Self {
        Self::Allocation(error)
    }
}
impl From<crate::Error> for ValueError {
    fn from(error: crate::Error) -> Self {
        Self::Syntax(error)
    }
}

enum Frame {
    Array(Vec<Value>),
    Object(Map<String, Value>, Option<String>),
}
struct Builder<'a> {
    allocation: &'a dyn Allocation,
    frames: Vec<Frame>,
    root: Option<Value>,
    failed: Option<AllocationError>,
}
impl Builder<'_> {
    fn push_value(&mut self, value: Value) -> Result<(), AllocationError> {
        match self.frames.last_mut() {
            Some(Frame::Array(values)) => Allocator::new(self.allocation).push(values, value),
            Some(Frame::Object(values, key)) => {
                values.try_insert_with_allocations(
                    key.take().expect("ordinary JSON key event"),
                    value,
                    self.allocation,
                )?;
                Ok(())
            }
            None => {
                assert!(self.root.is_none(), "one ordinary JSON root");
                self.root = Some(value);
                Ok(())
            }
        }
    }
    fn accept(&mut self, event: Event<'_>) -> Result<(), AllocationError> {
        let allocator = Allocator::new(self.allocation);
        match event {
            Event::Object => allocator.push(&mut self.frames, Frame::Object(Map::new(), None)),
            Event::Array => allocator.push(&mut self.frames, Frame::Array(Vec::new())),
            Event::Key(key) => {
                let key = allocator.copy_string(key)?;
                let Some(Frame::Object(_, current)) = self.frames.last_mut() else {
                    unreachable!("ordinary JSON object event")
                };
                assert!(current.is_none(), "ordinary JSON key followed by value");
                *current = Some(key);
                Ok(())
            }
            Event::EndObject => {
                let Some(Frame::Object(values, None)) = self.frames.pop() else {
                    unreachable!("ordinary JSON object end")
                };
                self.push_value(Value::Object(values))
            }
            Event::EndArray => {
                let Some(Frame::Array(values)) = self.frames.pop() else {
                    unreachable!("ordinary JSON array end")
                };
                self.push_value(Value::Array(values))
            }
            Event::String(value) => {
                let value = allocator.copy_string(value)?;
                self.push_value(Value::String(value))
            }
            Event::I64(value) => self.push_value(Value::Number(
                crate::Number::from_i64_with_allocations(value, self.allocation)?,
            )),
            Event::U64(value) => self.push_value(Value::Number(
                crate::Number::from_u64_with_allocations(value, self.allocation)?,
            )),
            Event::F64(value) => self.push_value(Value::Number(
                crate::Number::from_f64_with_allocations(value, self.allocation)?
                    .expect("ordinary finite JSON number"),
            )),
            Event::Number(value) => self.push_value(Value::Number(
                value.try_clone_with_allocations(self.allocation)?,
            )),
            Event::Bool(value) => self.push_value(Value::Bool(value)),
            Event::Null => self.push_value(Value::Null),
        }
    }
}
impl Sink for Builder<'_> {
    fn stopped(&self) -> bool {
        self.failed.is_some()
    }
    fn event(&mut self, event: Event<'_>) {
        if self.failed.is_some() {
            return;
        }
        if let Err(error) = self.accept(event) {
            self.failed = Some(error);
        }
    }
}

fn prepare<'a>(plan: &Plan<'_>, allocation: &'a dyn Allocation) -> Result<Builder<'a>, ValueError> {
    // The parser owns and reports traversal/scratch/error storage. This producer
    // adds its concrete callback, return and destination controls; every reached
    // frame, string, map node and array backing is paid by its own worker.
    let controls = [
        size_of::<Builder<'_>>(),
        size_of::<Allocator<'_>>(),
        size_of::<Frame>(),
        size_of::<Value>(),
        size_of::<ValueError>(),
        size_of::<Result<Value, ValueError>>(),
        size_of::<Result<(Value, usize), ValueError>>(),
        size_of::<Result<(), AllocationError>>(),
    ];
    let parser = plan.requirements::<Builder<'_>>()?.required_bytes();
    let bytes = controls
        .into_iter()
        .try_fold(size_of::<[usize; 8]>(), usize::checked_add)
        .and_then(|bytes| bytes.checked_add(parser))
        .ok_or(ValueError::Allocation(AllocationError::SizeOverflow))?;
    allocation.reserve(bytes)?;
    Ok(Builder {
        allocation,
        frames: Vec::new(),
        root: None,
        failed: None,
    })
}

/// Parse a complete owned JSON value using the ordinary deserializer and exact
/// prospective destination producers. The caller retains allocation custody.
pub fn from_slice_with_allocations(
    input: &[u8],
    allocation: &dyn Allocation,
) -> Result<Value, ValueError> {
    let session = crate::allocation::Session::new(allocation);
    let allocation = &session as &dyn Allocation;
    let plan = Plan::prepare(input)?;
    let mut builder = prepare(&plan, allocation)?;
    let parsed = plan.parse(&mut builder, allocation);
    if let Some(error) = builder.failed {
        return Err(ValueError::Allocation(error));
    }
    parsed.map_err(|error| match error.allocation_error() {
        Some(error) => ValueError::Allocation(error),
        None => ValueError::Syntax(error),
    })?;
    Ok(builder.root.take().expect("ordinary complete JSON root"))
}

/// Parse the first owned JSON value and its exact consumed byte count through
/// the ordinary stream worker. Trailing input remains untouched.
pub fn from_prefix_with_allocations(
    input: &[u8],
    allocation: &dyn Allocation,
) -> Result<(Value, usize), ValueError> {
    let session = crate::allocation::Session::new(allocation);
    let allocation = &session as &dyn Allocation;
    let plan = Plan::prepare_prefix(input)?;
    let mut builder = prepare(&plan, allocation)?;
    let parsed = plan.parse_prefix(&mut builder, allocation);
    if let Some(error) = builder.failed {
        return Err(ValueError::Allocation(error));
    }
    let consumed = parsed
        .map_err(|error| match error.allocation_error() {
            Some(error) => ValueError::Allocation(error),
            None => ValueError::Syntax(error),
        })?
        .ok_or(ValueError::Empty)?;
    Ok((
        builder.root.take().expect("ordinary complete JSON prefix"),
        consumed,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::sync::atomic::{AtomicUsize, Ordering};

    struct Account {
        calls: AtomicUsize,
        stop: usize,
    }
    impl Allocation for Account {
        fn reserve(&self, _: usize) -> Result<(), AllocationError> {
            if self.calls.fetch_add(1, Ordering::SeqCst) == self.stop {
                Err(AllocationError::Refused)
            } else {
                Ok(())
            }
        }
    }
    const SOURCE: &[u8] = br#"{"duplicate":1,"nested":[true,null,{"text":"escaped\n\ud83d\ude00","number":1.25e-200}],"duplicate":2,"negative":-17,"large":18446744073709551615}"#;

    #[test]
    fn funded_values_preserve_ordinary_numbers_escapes_duplicate_keys_and_prefix() {
        let expected: Value = crate::from_slice(SOURCE).unwrap();
        let actual = from_slice_with_allocations(SOURCE, &crate::allocation::Unenforced).unwrap();
        assert_eq!(actual, expected);
        assert_eq!(
            crate::to_string(&actual).unwrap(),
            crate::to_string(&expected).unwrap()
        );
        let mut prefix = SOURCE.to_vec();
        prefix.extend_from_slice(b" grammar: /[a-z]+/");
        let (value, consumed) =
            from_prefix_with_allocations(&prefix, &crate::allocation::Unenforced).unwrap();
        assert_eq!(value, expected);
        assert_eq!(consumed, SOURCE.len());
    }

    #[test]
    fn each_reached_value_allocation_refuses_before_later_destinations() {
        let account = Account {
            calls: AtomicUsize::new(0),
            stop: usize::MAX,
        };
        let value = from_slice_with_allocations(SOURCE, &account).unwrap();
        assert!(value.is_object());
        let count = account.calls.load(Ordering::SeqCst);
        assert!(count > 1);
        for stop in 0..count {
            let account = Account {
                calls: AtomicUsize::new(0),
                stop,
            };
            assert!(
                matches!(
                    from_slice_with_allocations(SOURCE, &account),
                    Err(ValueError::Allocation(AllocationError::Refused))
                ),
                "allocation {stop}"
            );
            assert_eq!(account.calls.load(Ordering::SeqCst), stop + 1);
        }
    }

    #[test]
    fn value_refusal_keeps_precedence_over_later_invalid_json() {
        let account = Account {
            calls: AtomicUsize::new(0),
            stop: 1,
        };
        assert!(matches!(
            from_slice_with_allocations(br#"{"field":["text", invalid]}"#, &account),
            Err(ValueError::Allocation(AllocationError::Refused))
        ));
        assert!(matches!(
            from_slice_with_allocations(b"[1,]", &crate::allocation::Unenforced),
            Err(ValueError::Syntax(_))
        ));
    }
}

#[cfg(all(test, feature = "arbitrary_precision"))]
mod arbitrary_tests {
    use super::*;
    use alloc::string::ToString;
    use core::cell::Cell;
    struct Account {
        calls: Cell<usize>,
        stop: usize,
        failed: Cell<bool>,
    }
    impl Allocation for Account {
        fn reserve(&self, _: usize) -> Result<(), AllocationError> {
            assert!(!self.failed.get(), "original account reached after refusal");
            let call = self.calls.get();
            self.calls.set(call + 1);
            if call == self.stop {
                self.failed.set(true);
                Err(AllocationError::Refused)
            } else {
                Ok(())
            }
        }
    }
    #[test]
    fn exact_numbers_private_transport_and_each_reached_refusal() {
        for source in [
            "1E400",
            "-0",
            "-0.0",
            "1844674407370955161600000",
            "[1e-400,1.2345678901234567890123456789,2]",
            r#"{"$serde_json::private::Number":"1E400"}"#,
            r#"{"$serde_json::private::Number":"bad"}"#,
            r#"{"$serde_json::private::Number":"1e","suffix":true}"#,
            r#"{"$serde_json::private::Number":42}"#,
            r#"{"$serde_json::private::Number":true}"#,
            r#"{"$serde_json::private::Number":null}"#,
            r#"{"$serde_json::private::Number":[]}"#,
            r#"{"$serde_json::private::Number":{}}"#,
            r#"{"$serde_json::private::Number":"-0","suffix":true}"#,
            r#"{"normal":1,"$serde_json::private::Number":"bad"}"#,
            r#"{"\u0024serde_json::private::Number":"1e+12"}"#,
            "[123456789012345678901234567890123456789012345678901234567890,]",
        ] {
            let expected = crate::from_str::<Value>(source);
            let account = Account {
                calls: Cell::new(0),
                stop: usize::MAX,
                failed: Cell::new(false),
            };
            let actual = from_slice_with_allocations(source.as_bytes(), &account);
            match (&expected, &actual) {
                (Ok(expected), Ok(actual)) => assert_eq!(expected, actual, "{source}"),
                (Err(expected), Err(ValueError::Syntax(actual))) => {
                    assert_eq!(expected.to_string(), actual.to_string(), "{source}")
                }
                _ => {
                    panic!("changed original numeric transport {source}: {expected:?} / {actual:?}")
                }
            }
            for stop in 0..account.calls.get() {
                let account = Account {
                    calls: Cell::new(0),
                    stop,
                    failed: Cell::new(false),
                };
                let value = from_slice_with_allocations(source.as_bytes(), &account);
                assert!(
                    matches!(value, Err(ValueError::Allocation(AllocationError::Refused))),
                    "{source} at {stop}: {value:?}"
                );
                assert_eq!(account.calls.get(), stop + 1);
            }
        }
    }
}
