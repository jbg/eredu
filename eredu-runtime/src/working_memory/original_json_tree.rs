//! Immutable paid values produced by the actual serde event deserializer.
use super::OriginalJsonValueKind;
use eredu_core::{HostPreparationAuthority, SpeculativeBuffer, SpeculativeBufferAllocationError};
use eredu_nn::workspace::{WorkspaceMetadataFunding, WorkspaceMetadataFundingError};
use serde_json::bounded_events::{Event, Plan, PlanError, Sink};
use std::{
    cmp::Ordering,
    mem::{size_of, size_of_val},
};

/// Exact primitive number emitted by the selected ordinary serde parser.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum OriginalJsonNumber {
    /// Signed integer token.
    I64(i64),
    /// Unsigned integer token.
    U64(u64),
    /// Finite floating token, including its signed zero.
    F64(f64),
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
    Number(OriginalJsonNumber),
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
    Funding(#[from] WorkspaceMetadataFundingError),
    #[error(transparent)]
    Allocation(#[from] SpeculativeBufferAllocationError),
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
    funding: WorkspaceMetadataFunding,
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
        funding: &WorkspaceMetadataFunding,
    ) -> Result<SpeculativeBuffer<T>, Cause> {
        let bytes = SpeculativeBuffer::<T>::retained_control_bytes(capacity)
            .and_then(|n| {
                n.checked_add(HostPreparationAuthority::retention_bytes::<
                    WorkspaceMetadataFunding,
                >()?)
            })
            .and_then(|n| n.checked_add(size_of::<Result<SpeculativeBuffer<T>, Cause>>()))
            .and_then(|n| n.checked_add(size_of::<(usize, &WorkspaceMetadataFunding)>()))
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
            size_of::<WorkspaceMetadataFunding>(),
            size_of::<Plan<'_>>(),
            size_of::<Result<Plan<'_>, PlanError>>(),
            size_of::<serde_json::bounded_events::Requirements>(),
            size_of::<Result<serde_json::bounded_events::Requirements, PlanError>>(),
            size_of::<Result<Self, OriginalJsonTreeError>>(),
            size_of::<Result<(), Cause>>(),
            size_of::<Result<(), serde_json::Error>>(),
            size_of::<(&str, &WorkspaceMetadataFunding)>(),
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
        funding: &WorkspaceMetadataFunding,
    ) -> Result<Self, OriginalJsonTreeError> {
        let mut builder = Builder {
            tree: Self {
                records: SpeculativeBuffer::default(),
                text: SpeculativeBuffer::default(),
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
            plan.parse(&mut builder)?;
            if let Some(cause) = builder.failure.take() {
                return Err(cause);
            }
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
                            Ordering::Greater if !serde_json::bounded_events::preserves_object_order() => break,
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
            Event::I64(value) => self.append(Value::Number(OriginalJsonNumber::I64(value))),
            Event::U64(value) => self.append(Value::Number(OriginalJsonNumber::U64(value))),
            Event::F64(value) => self.append(Value::Number(OriginalJsonNumber::F64(value))),
            Event::Bool(value) => self.append(Value::Bool(value)),
            Event::Null => self.append(Value::Null),
        }
    }
}
impl Sink for Builder {
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
    pub fn number(self) -> Option<OriginalJsonNumber> {
        if let Value::Number(value) = self.record().value {
            Some(value)
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
