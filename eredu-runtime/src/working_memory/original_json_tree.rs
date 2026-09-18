//! Immutable paid values produced by the actual serde event deserializer.
use super::OriginalJsonValueKind;
use eredu_core::{HostPreparationAuthority, SpeculativeBuffer, SpeculativeBufferAllocationError};
use eredu_nn::workspace::{HostMetadataFunding, HostMetadataFundingError};
use serde_json::bounded_events::{Event, Plan, PlanError, Sink};
use std::{
    cmp::Ordering,
    mem::{size_of, size_of_val},
};

/// Exact primitive number emitted by the selected ordinary serde parser.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum OriginalJsonNumber<'a> {
    /// Signed integer token.
    I64(i64),
    /// Unsigned integer token.
    U64(u64),
    /// Finite floating token, including its signed zero.
    F64(f64),
    /// Borrowed exact number from the original optional numeric representation.
    Exact(&'a serde_json::Number),
}
#[derive(Clone, Copy, Debug)]
enum StoredNumber {
    I64(i64),
    U64(u64),
    F64(f64),
    Exact(usize),
}
#[derive(Clone, Copy, Debug)]
struct Span {
    start: usize,
    length: usize,
}
#[derive(Clone, Copy, Debug, Default)]
struct Children {
    first: Option<usize>,
    last: Option<usize>,
    length: usize,
}
#[derive(Clone, Copy, Debug)]
enum Value {
    Object {
        children: Children,
        pending: Option<Span>,
    },
    Array(Children),
    String(Span),
    Number(StoredNumber),
    Bool(bool),
    Null,
}
#[derive(Clone, Copy, Debug)]
struct Record {
    value: Value,
    key: Option<Span>,
    parent: Option<usize>,
    next: Option<usize>,
}
#[derive(Debug, thiserror::Error)]
enum Cause {
    #[error(transparent)]
    Plan(#[from] PlanError),
    #[error(transparent)]
    Syntax(#[from] serde_json::Error),
    #[error(transparent)]
    Funding(#[from] HostMetadataFundingError),
    #[error(transparent)]
    Allocation(#[from] SpeculativeBufferAllocationError),
    #[error(transparent)]
    JsonAllocation(#[from] serde_json::allocation::AllocationError),
    #[error("original JSON tree destination is full")]
    Capacity,
    #[error("original JSON tree source extent overflow")]
    Overflow,
    #[error("original JSON tree event source is inconsistent")]
    Source,
}
/// An immutable JSON value with finite record and decoded-string destinations.
/// Duplicate object keys use the ordinary last-value rule. Object iteration
/// follows the selected serde map: first insertion order with preserve_order,
/// lexicographic otherwise. Superseded rows remain owned and charged.
#[derive(Debug)]
pub struct OriginalJsonTree {
    records: SpeculativeBuffer<Record>,
    text: SpeculativeBuffer<u8>,
    numbers: SpeculativeBuffer<serde_json::Number>,
    funding: HostMetadataFunding,
}
/// Failure retains all produced rows and decoded strings, including an escaped
/// partial root. The parser's first callback failure remains independently owned.
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
pub struct OriginalJsonTreeError {
    #[source]
    cause: Cause,
    event_failure: Option<Cause>,
    prefix: OriginalJsonTree,
}
struct Builder {
    tree: OriginalJsonTree,
    current: Option<usize>,
    failure: Option<Cause>,
}
impl OriginalJsonTree {
    fn text(&self, span: Span) -> &str {
        std::str::from_utf8(&self.text[span.start..span.start + span.length])
            .expect("complete decoded strings from serde")
    }
    fn buffer<T>(
        capacity: usize,
        funding: &HostMetadataFunding,
    ) -> Result<SpeculativeBuffer<T>, Cause> {
        let bytes = SpeculativeBuffer::<T>::retained_control_bytes(capacity)
            .and_then(|n| {
                n.checked_add(HostPreparationAuthority::retention_bytes::<
                    HostMetadataFunding,
                >()?)
            })
            .and_then(|n| n.checked_add(size_of::<Result<SpeculativeBuffer<T>, Cause>>()))
            .and_then(|n| n.checked_add(size_of::<(usize, &HostMetadataFunding)>()))
            .ok_or(Cause::Overflow)?;
        funding.reserve_metadata(bytes)?;
        Ok(SpeculativeBuffer::try_new_retained(
            capacity,
            HostPreparationAuthority::retain(funding.clone()),
        )?)
    }
    fn controls(input_bytes: usize) -> Option<usize> {
        let fixed = [
            size_of::<Self>(),
            size_of::<Builder>(),
            size_of::<OriginalJsonTreeError>(),
            size_of::<Cause>(),
            size_of::<Option<Cause>>(),
            size_of::<HostMetadataFunding>(),
            size_of::<Plan<'_>>(),
            size_of::<Result<Plan<'_>, PlanError>>(),
            size_of::<serde_json::bounded_events::Requirements>(),
            size_of::<Result<serde_json::bounded_events::Requirements, PlanError>>(),
            size_of::<Result<Self, OriginalJsonTreeError>>(),
            size_of::<Result<(), Cause>>(),
            size_of::<Result<(), serde_json::Error>>(),
            size_of::<(&str, &HostMetadataFunding)>(),
        ];
        let event = [
            size_of::<Event<'_>>(),
            size_of::<Record>(),
            size_of::<Value>(),
            size_of::<Children>(),
            size_of::<Option<Span>>(),
            size_of::<Span>(),
            size_of::<Result<Span, Cause>>(),
            size_of::<Result<(), Cause>>(),
            size_of::<Option<usize>>(),
            size_of::<(usize, usize, bool, bool)>(),
            size_of::<(&mut Builder, Event<'_>)>(),
            size_of::<(&mut Builder, Value)>(),
            size_of::<(&mut Builder, &str)>(),
            size_of::<OriginalJsonNode<'_>>(),
            size_of::<OriginalJsonChildren<'_>>(),
            size_of::<Ordering>(),
            size_of::<eredu_core::GenerationError>(),
            size_of::<Result<(), eredu_core::GenerationError>>(),
        ];
        fixed
            .into_iter()
            .try_fold(size_of_val(&fixed), usize::checked_add)?
            .checked_add(
                event
                    .into_iter()
                    .try_fold(size_of_val(&event), usize::checked_add)?
                    .checked_mul(input_bytes)?,
            )
    }
    /// Parses with the existing serde deserializer after paying its parser and
    /// exact finite output destinations. Every callback/record consumes at least
    /// one source byte; decoded UTF-8 strings cannot exceed their source spelling.
    pub fn parse(
        input: &str,
        funding: &HostMetadataFunding,
    ) -> Result<Self, OriginalJsonTreeError> {
        let mut builder = Builder {
            tree: Self {
                records: SpeculativeBuffer::default(),
                text: SpeculativeBuffer::default(),
                numbers: SpeculativeBuffer::default(),
                funding: funding.clone(),
            },
            current: None,
            failure: None,
        };
        let result = (|| {
            funding.reserve_metadata(Self::controls(input.len()).ok_or(Cause::Overflow)?)?;
            let plan = Plan::prepare(input.as_bytes())?;
            let requirements = plan.requirements::<Builder>()?;
            funding.reserve_metadata(requirements.required_bytes())?;
            builder.tree.records = Self::buffer(input.len(), funding)?;
            builder.tree.text = Self::buffer(input.len(), funding)?;
            let allocation = super::original_json_allocation::JsonAllocation::new(funding)?;
            let parsed = plan.parse(&mut builder, &allocation);
            if let Some(cause) = allocation.failure() {
                return Err(cause.into());
            }
            if let Some(cause) = builder.failure.take() {
                return Err(cause);
            }
            parsed?;
            if builder.current.is_some() || builder.tree.records.is_empty() {
                return Err(Cause::Source);
            }
            Ok::<_, Cause>(())
        })();
        match result {
            Ok(()) => Ok(builder.tree),
            Err(cause) => Err(OriginalJsonTreeError {
                cause,
                event_failure: builder.failure,
                prefix: builder.tree,
            }),
        }
    }
    /// The complete root borrowed from this exact immutable owner.
    pub fn root(&self) -> OriginalJsonNode<'_> {
        OriginalJsonNode {
            tree: self,
            index: 0,
        }
    }
}
impl Builder {
    fn text(&mut self, value: &str) -> Result<Span, Cause> {
        let span = Span {
            start: self.tree.text.len(),
            length: value.len(),
        };
        if value.len() > self.tree.text.capacity() - self.tree.text.len() {
            return Err(Cause::Capacity);
        }
        self.tree
            .text
            .try_extend(value.bytes())
            .map_err(|_| Cause::Capacity)?;
        Ok(span)
    }
    fn append(&mut self, value: Value) -> Result<(), Cause> {
        if self.current.is_none() && !self.tree.records.is_empty() {
            return Err(Cause::Source);
        }
        let mut key = None;
        if let Some(parent) = self.current {
            match &mut self.tree.records[parent].value {
                Value::Object { pending, .. } => {
                    key = Some(pending.take().ok_or(Cause::Source)?);
                }
                Value::Array(_) => (),
                _ => return Err(Cause::Source),
            }
        }
        let index = self.tree.records.len();
        self.tree
            .records
            .try_push(Record {
                value,
                key,
                parent: self.current,
                next: None,
            })
            .map_err(|_| Cause::Capacity)?;
        if let Some(parent) = self.current {
            match self.tree.records[parent].value {
                Value::Array(mut children) => {
                    if let Some(last) = children.last {
                        self.tree.records[last].next = Some(index);
                    } else {
                        children.first = Some(index);
                    }
                    children.last = Some(index);
                    children.length = children.length.checked_add(1).ok_or(Cause::Overflow)?;
                    self.tree.records[parent].value = Value::Array(children);
                }
                Value::Object {
                    mut children,
                    pending,
                } => {
                    let key = key.ok_or(Cause::Source)?;
                    let mut previous = None;
                    let mut next = children.first;
                    let mut replace = false;
                    while let Some(other) = next {
                        let other_key = self.tree.records[other].key.ok_or(Cause::Source)?;
                        match self.tree.text(other_key).cmp(self.tree.text(key)) {
                            Ordering::Equal => {
                                replace = true;
                                break;
                            }
                            Ordering::Greater
                                if !serde_json::bounded_events::preserves_object_order() =>
                            {
                                break
                            }
                            _ => {
                                previous = Some(other);
                                next = self.tree.records[other].next;
                            }
                        }
                    }
                    let following = if replace {
                        self.tree.records[next.ok_or(Cause::Source)?].next
                    } else {
                        next
                    };
                    self.tree.records[index].next = following;
                    if let Some(previous) = previous {
                        self.tree.records[previous].next = Some(index);
                    } else {
                        children.first = Some(index);
                    }
                    if following.is_none() {
                        children.last = Some(index);
                    }
                    if !replace {
                        children.length = children.length.checked_add(1).ok_or(Cause::Overflow)?;
                    }
                    self.tree.records[parent].value = Value::Object { children, pending };
                }
                _ => return Err(Cause::Source),
            }
        }
        if matches!(value, Value::Object { .. } | Value::Array(_)) {
            self.current = Some(index);
        }
        Ok(())
    }
    fn event_inner(&mut self, event: Event<'_>) -> Result<(), Cause> {
        match event {
            Event::Object => self.append(Value::Object {
                children: Children::default(),
                pending: None,
            }),
            Event::Array => self.append(Value::Array(Children::default())),
            Event::Key(value) => {
                let key = self.text(value)?;
                let current = self.current.ok_or(Cause::Source)?;
                let Value::Object { pending, .. } = &mut self.tree.records[current].value else {
                    return Err(Cause::Source);
                };
                if pending.replace(key).is_some() {
                    return Err(Cause::Source);
                }
                Ok(())
            }
            Event::EndObject | Event::EndArray => {
                let current = self.current.ok_or(Cause::Source)?;
                let valid = matches!(
                    (&event, self.tree.records[current].value),
                    (Event::EndObject, Value::Object { pending: None, .. })
                        | (Event::EndArray, Value::Array(_))
                );
                if !valid {
                    return Err(Cause::Source);
                }
                self.current = self.tree.records[current].parent;
                Ok(())
            }
            Event::String(value) => {
                let span = self.text(value)?;
                self.append(Value::String(span))
            }
            Event::I64(value) => self.append(Value::Number(StoredNumber::I64(value))),
            Event::U64(value) => self.append(Value::Number(StoredNumber::U64(value))),
            Event::F64(value) => self.append(Value::Number(StoredNumber::F64(value))),
            Event::Number(value) => {
                if self.tree.numbers.capacity() == 0 {
                    self.tree.numbers =
                        OriginalJsonTree::buffer(self.tree.records.capacity(), &self.tree.funding)?;
                }
                let allocation =
                    super::original_json_allocation::JsonAllocation::new(&self.tree.funding)?;
                let cloned = value.try_clone_with_allocations(&allocation);
                if let Some(cause) = allocation.failure() {
                    return Err(cause.into());
                }
                let index = self.tree.numbers.len();
                self.tree
                    .numbers
                    .try_push(cloned?)
                    .map_err(|_| Cause::Capacity)?;
                self.append(Value::Number(StoredNumber::Exact(index)))
            }
            Event::Bool(value) => self.append(Value::Bool(value)),
            Event::Null => self.append(Value::Null),
        }
    }
}
impl Sink for Builder {
    fn stopped(&self) -> bool {
        self.failure.is_some()
    }
    fn event(&mut self, event: Event<'_>) {
        if self.failure.is_none() {
            if let Err(cause) = self.event_inner(event) {
                self.failure = Some(cause);
            }
        }
    }
}

/// A borrowed immutable value; it cannot outlive its exact tree and payer.
#[derive(Clone, Copy, Debug)]
pub struct OriginalJsonNode<'a> {
    tree: &'a OriginalJsonTree,
    index: usize,
}
impl<'a> OriginalJsonNode<'a> {
    fn record(self) -> &'a Record {
        &self.tree.records[self.index]
    }
    /// Root kind from the actual parser event.
    pub fn kind(self) -> OriginalJsonValueKind {
        match self.record().value {
            Value::Object { .. } => OriginalJsonValueKind::Object,
            Value::Array(_) => OriginalJsonValueKind::Array,
            Value::String(_) => OriginalJsonValueKind::String,
            Value::Number(_) => OriginalJsonValueKind::Number,
            Value::Bool(_) => OriginalJsonValueKind::Bool,
            Value::Null => OriginalJsonValueKind::Null,
        }
    }
    /// Decoded string from the retained byte destination.
    pub fn string(self) -> Option<&'a str> {
        if let Value::String(span) = self.record().value {
            Some(self.tree.text(span))
        } else {
            None
        }
    }
    /// Exact primitive parsed number.
    pub fn number(self) -> Option<OriginalJsonNumber<'a>> {
        if let Value::Number(value) = self.record().value {
            Some(match value {
                StoredNumber::I64(n) => OriginalJsonNumber::I64(n),
                StoredNumber::U64(n) => OriginalJsonNumber::U64(n),
                StoredNumber::F64(n) => OriginalJsonNumber::F64(n),
                StoredNumber::Exact(index) => OriginalJsonNumber::Exact(&self.tree.numbers[index]),
            })
        } else {
            None
        }
    }
    /// Boolean value, absent for other kinds.
    pub fn boolean(self) -> Option<bool> {
        if let Value::Bool(value) = self.record().value {
            Some(value)
        } else {
            None
        }
    }
    /// Stable storage identity for this record while the borrowed owner lives.
    pub fn storage_identity(self) -> usize {
        std::ptr::from_ref(self.record()) as usize
    }
    /// Array elements or object members in the selected ordinary serde order.
    /// Array entries have no key. Duplicate keys retain their original position
    /// and expose only their final value.
    pub fn children(self) -> Option<OriginalJsonChildren<'a>> {
        let children = match self.record().value {
            Value::Object { children, .. } | Value::Array(children) => children,
            _ => return None,
        };
        Some(OriginalJsonChildren {
            tree: self.tree,
            next: children.first,
            remaining: children.length,
        })
    }
    /// Borrows a decoded object member by name without allocating.
    pub fn get(self, name: &str) -> Option<Self> {
        if !matches!(self.record().value, Value::Object { .. }) {
            return None;
        }
        for (key, value) in self.children()? {
            match key?.cmp(name) {
                Ordering::Equal => return Some(value),
                Ordering::Greater if !serde_json::bounded_events::preserves_object_order() => break,
                _ => (),
            }
        }
        None
    }
}
/// Allocation-free borrowed child traversal.
#[derive(Clone, Debug)]
pub struct OriginalJsonChildren<'a> {
    tree: &'a OriginalJsonTree,
    next: Option<usize>,
    remaining: usize,
}
impl<'a> Iterator for OriginalJsonChildren<'a> {
    type Item = (Option<&'a str>, OriginalJsonNode<'a>);
    fn next(&mut self) -> Option<Self::Item> {
        let index = self.next?;
        let record = &self.tree.records[index];
        self.next = record.next;
        self.remaining -= 1;
        Some((
            record.key.map(|key| self.tree.text(key)),
            OriginalJsonNode {
                tree: self.tree,
                index,
            },
        ))
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}
impl ExactSizeIterator for OriginalJsonChildren<'_> {}
impl std::iter::FusedIterator for OriginalJsonChildren<'_> {}

#[cfg(test)]
mod numeric_tests {
    use super::*;
    use eredu_core::HostMetadataAccount;
    use std::sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc,
    };
    #[derive(Debug)]
    struct State {
        calls: AtomicUsize,
        stop: AtomicUsize,
        failed: AtomicBool,
        retired: AtomicBool,
    }
    #[derive(Debug)]
    struct Account(Arc<State>);
    impl HostMetadataAccount for Account {
        fn reserve_metadata(&self, bytes: usize) -> Result<(), HostMetadataFundingError> {
            assert!(
                !self.0.failed.load(Ordering::SeqCst),
                "tree reached original payer after refusal"
            );
            let call = self.0.calls.fetch_add(1, Ordering::SeqCst);
            if call == self.0.stop.load(Ordering::SeqCst) {
                self.0.failed.store(true, Ordering::SeqCst);
                Err(HostMetadataFundingError::Capacity {
                    required: bytes as u64,
                    available: 0,
                })
            } else {
                Ok(())
            }
        }
    }
    impl Drop for Account {
        fn drop(&mut self) {
            self.0.retired.store(true, Ordering::SeqCst);
        }
    }
    fn account(stop: usize) -> (HostMetadataFunding, Arc<State>) {
        let state = Arc::new(State {
            calls: AtomicUsize::new(0),
            stop: AtomicUsize::new(usize::MAX),
            failed: AtomicBool::new(false),
            retired: AtomicBool::new(false),
        });
        let funding = HostMetadataFunding::new(Account(state.clone())).unwrap();
        state.calls.store(0, Ordering::SeqCst);
        state.stop.store(stop, Ordering::SeqCst);
        (funding, state)
    }
    #[test]
    fn original_numeric_carrier_preserves_values_and_failure_custody() {
        let input = r#"[1.234567890123456789012345678901234567890,-0.0,18446744073709551616,{"nested":2.25e-30}]"#;
        let expected: serde_json::Value = serde_json::from_str(input).unwrap();
        let (funding, state) = account(usize::MAX);
        let tree = OriginalJsonTree::parse(input, &funding).unwrap();
        for ((_, actual), expected) in tree
            .root()
            .children()
            .unwrap()
            .take(3)
            .zip(expected.as_array().unwrap())
        {
            let actual = match actual.number().unwrap() {
                OriginalJsonNumber::I64(n) => serde_json::Number::from(n),
                OriginalJsonNumber::U64(n) => serde_json::Number::from(n),
                OriginalJsonNumber::F64(n) => serde_json::Number::from_f64(n).unwrap(),
                OriginalJsonNumber::Exact(n) => n.clone(),
            };
            assert_eq!(&actual, expected.as_number().unwrap());
            assert_eq!(
                actual.source_text(),
                expected.as_number().unwrap().source_text()
            );
        }
        let count = state.calls.load(Ordering::SeqCst);
        drop(funding);
        assert!(!state.retired.load(Ordering::SeqCst));
        drop(tree);
        assert!(state.retired.load(Ordering::SeqCst));
        for stop in 0..count {
            let (funding, state) = account(stop);
            let error = OriginalJsonTree::parse(input, &funding).unwrap_err();
            assert!(
                matches!(
                    error.cause,
                    Cause::Funding(HostMetadataFundingError::Capacity { .. })
                ),
                "refusal {stop}: {error:?}"
            );
            assert_eq!(state.calls.load(Ordering::SeqCst), stop + 1);
            drop(funding);
            assert!(!state.retired.load(Ordering::SeqCst));
            drop(error);
            assert!(state.retired.load(Ordering::SeqCst));
        }
    }
}
