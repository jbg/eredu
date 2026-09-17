//! Immutable source records. The borrowed tree owns no object callbacks, VM
//! state, allocation authority or mutable handles. Original producers retain
//! and price their actual record/string storage through the lexical render.

/// Readonly scalar/container declaration used by a prepared input producer.
#[derive(Clone, Copy, Debug)]
pub enum RecordValue<'a> {
    /// JSON null.
    Null,
    /// JSON Boolean.
    Bool(bool),
    /// Exact borrowed UTF-8 spelling.
    Text(&'a str),
    /// Ordered borrowed sequence.
    Array(&'a [RecordValue<'a>]),
    /// Ordered immutable fields. Repeated keys use ordinary map-insertion
    /// semantics: the first position is retained and the final value wins.
    Object(&'a [RecordField<'a>]),
}
/// One immutable object member; both the key and any payload remain borrowed.
#[derive(Clone, Copy, Debug)]
pub struct RecordField<'a> {
    /// Exact case-sensitive field spelling.
    pub key: &'a str,
    /// Scalar or container supplied by the same source owner.
    pub value: RecordValue<'a>,
}
impl<'a> RecordValue<'a> {
    /// Whether the declaration is an immutable object.
    pub fn is_object(self) -> bool {
        matches!(self, Self::Object(_))
    }
    /// Exact object lookup, sharing ordinary replacement precedence.
    pub fn field(self, key: &str) -> Option<Self> {
        match self {
            Self::Object(fields) => fields
                .iter()
                .rev()
                .find(|field| field.key == key)
                .map(|field| field.value),
            _ => None,
        }
    }
    /// Declared sequence item, with no conversion or copied child ownership.
    pub fn item(self, index: usize) -> Option<Self> {
        match self {
            Self::Array(items) => items.get(index).copied(),
            _ => None,
        }
    }
    /// Effective object field count or exact array item count.
    pub fn length(self) -> Option<usize> {
        match self {
            Self::Array(items) => Some(items.len()),
            Self::Object(fields) => Some(
                fields
                    .iter()
                    .enumerate()
                    .filter(|(index, field)| {
                        !fields[..*index].iter().any(|prior| prior.key == field.key)
                    })
                    .count(),
            ),
            _ => None,
        }
    }
    /// Object fields in ordinary insertion order; duplicate values replace
    /// their first position. This creates no sorted/copied key table.
    pub fn fields(self) -> Option<RecordFields<'a>> {
        match self {
            Self::Object(fields) => Some(RecordFields { fields, next: 0 }),
            _ => None,
        }
    }
    /// Fixed projection/access frames; actual input descriptors and text bytes
    /// belong to the producer, separately from renderer working storage.
    pub fn control_bytes() -> Option<usize> {
        use std::mem::size_of;
        let parts = [
            size_of::<ReadValue<'a>>(),
            size_of::<ReadKind<'a>>(),
            size_of::<ReadArray<'a>>(),
            size_of::<ReadArrayIter<'a>>(),
            size_of::<ReadObject<'a>>(),
            size_of::<ReadPairs<'a>>(),
            size_of::<NaturalPairs<'a>>(),
            size_of::<Option<ReadValue<'a>>>(),
            size_of::<Option<ReadKind<'a>>>(),
            size_of::<Self>(),
            size_of::<RecordField<'a>>(),
            size_of::<RecordFields<'a>>(),
            size_of::<Option<Self>>(),
            size_of::<Option<RecordFields<'a>>>(),
            size_of::<Option<(&str, Self)>>(),
            size_of::<std::slice::Iter<'a, RecordField<'a>>>(),
            size_of::<std::iter::Enumerate<std::slice::Iter<'a, RecordField<'a>>>>(),
            size_of::<std::iter::Rev<std::slice::Iter<'a, RecordField<'a>>>>(),
            size_of::<(usize, &'a str, &'a [RecordField<'a>])>(),
        ];
        parts
            .into_iter()
            .try_fold(std::mem::size_of_val(&parts), usize::checked_add)
    }
}
/// Borrowed immutable insertion-order object projection.
#[derive(Clone, Debug)]
pub struct RecordFields<'a> {
    fields: &'a [RecordField<'a>],
    next: usize,
}
impl<'a> Iterator for RecordFields<'a> {
    type Item = (&'a str, RecordValue<'a>);
    fn next(&mut self) -> Option<Self::Item> {
        while let Some(field) = self.fields.get(self.next) {
            let index = self.next;
            self.next += 1;
            if self.fields[..index]
                .iter()
                .any(|prior| prior.key == field.key)
            {
                continue;
            }
            let value = self.fields[index..]
                .iter()
                .rev()
                .find(|later| later.key == field.key)?
                .value;
            return Some((field.key, value));
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn borrowed_records_match_ordinary_replacement_order_and_keep_nested_values() {
        let text = String::from("λ\0value");
        let nested = [RecordValue::Text(&text), RecordValue::Bool(true)];
        let fields = [
            RecordField {
                key: "content",
                value: RecordValue::Text("first"),
            },
            RecordField {
                key: "role",
                value: RecordValue::Text("assistant"),
            },
            RecordField {
                key: "content",
                value: RecordValue::Array(&nested),
            },
        ];
        let record = RecordValue::Object(&fields);
        let keys: Vec<_> = record.fields().unwrap().map(|(key, _)| key).collect();
        let mut ordinary = serde_json::Map::new();
        ordinary.insert("content".into(), serde_json::Value::String("first".into()));
        ordinary.insert("role".into(), serde_json::Value::String("assistant".into()));
        ordinary.insert("content".into(), serde_json::json!([text, true]));
        assert_eq!(record.length(), Some(ordinary.len()));
        assert_eq!(keys, ["content", "role"]);
        assert!(
            matches!(record.field("content").unwrap().item(0), Some(RecordValue::Text(value)) if value == text)
        );
        assert!(record.field("missing").is_none());
    }
}

/// Lexical operand shared by JSON inputs and source-owned immutable records.
#[derive(Clone, Copy, Debug)]
pub(in crate::bounded) enum ReadValue<'a> {
    Json(&'a serde_json::Value),
    Record(RecordValue<'a>),
}
impl<'a> From<&'a serde_json::Value> for ReadValue<'a> {
    fn from(value: &'a serde_json::Value) -> Self {
        Self::Json(value)
    }
}
#[derive(Clone, Copy, Debug)]
pub(in crate::bounded) enum ReadKind<'a> {
    Null,
    Bool(bool),
    Text(&'a str),
    Number(&'a serde_json::Number),
    Array(ReadArray<'a>),
    Object(ReadObject<'a>),
}
impl<'a> ReadValue<'a> {
    pub(in crate::bounded) fn kind(self) -> ReadKind<'a> {
        use serde_json::Value as J;
        match self {
            Self::Json(J::Null) | Self::Record(RecordValue::Null) => ReadKind::Null,
            Self::Json(J::Bool(value)) => ReadKind::Bool(*value),
            Self::Record(RecordValue::Bool(value)) => ReadKind::Bool(value),
            Self::Json(J::String(value)) => ReadKind::Text(value),
            Self::Record(RecordValue::Text(value)) => ReadKind::Text(value),
            Self::Json(J::Number(value)) => ReadKind::Number(value),
            Self::Json(J::Array(value)) => ReadKind::Array(ReadArray::Json(value)),
            Self::Record(RecordValue::Array(value)) => ReadKind::Array(ReadArray::Record(value)),
            Self::Json(J::Object(value)) => ReadKind::Object(ReadObject::Json(value)),
            Self::Record(RecordValue::Object(value)) => ReadKind::Object(ReadObject::Record(value)),
        }
    }
    pub(in crate::bounded) fn as_str(self) -> Option<&'a str> {
        match self.kind() {
            ReadKind::Text(value) => Some(value),
            _ => None,
        }
    }
    pub(in crate::bounded) fn as_array(self) -> Option<ReadArray<'a>> {
        match self.kind() {
            ReadKind::Array(value) => Some(value),
            _ => None,
        }
    }
    pub(in crate::bounded) fn as_object(self) -> Option<ReadObject<'a>> {
        match self.kind() {
            ReadKind::Object(value) => Some(value),
            _ => None,
        }
    }
    pub(in crate::bounded) fn is_array(self) -> bool {
        self.as_array().is_some()
    }
    pub(in crate::bounded) fn is_object(self) -> bool {
        self.as_object().is_some()
    }
    pub(in crate::bounded) fn same_source(self, other: Self) -> bool {
        match (self, other) {
            (Self::Json(a), Self::Json(b)) => std::ptr::eq(a, b),
            (Self::Record(RecordValue::Array(a)), Self::Record(RecordValue::Array(b))) => {
                std::ptr::eq(a, b)
            }
            (Self::Record(RecordValue::Object(a)), Self::Record(RecordValue::Object(b))) => {
                std::ptr::eq(a, b)
            }
            _ => false,
        }
    }
}
#[derive(Clone, Copy, Debug)]
pub(in crate::bounded) enum ReadArray<'a> {
    Json(&'a [serde_json::Value]),
    Record(&'a [RecordValue<'a>]),
}
impl<'a> ReadArray<'a> {
    pub(in crate::bounded) fn len(self) -> usize {
        match self {
            Self::Json(a) => a.len(),
            Self::Record(a) => a.len(),
        }
    }
    pub(in crate::bounded) fn get(self, index: usize) -> Option<ReadValue<'a>> {
        match self {
            Self::Json(a) => a.get(index).map(ReadValue::Json),
            Self::Record(a) => a.get(index).copied().map(ReadValue::Record),
        }
    }
    pub(in crate::bounded) fn iter(self) -> ReadArrayIter<'a> {
        ReadArrayIter {
            array: self,
            next: 0,
        }
    }
}
#[derive(Clone, Debug)]
pub(in crate::bounded) struct ReadArrayIter<'a> {
    array: ReadArray<'a>,
    next: usize,
}
impl<'a> Iterator for ReadArrayIter<'a> {
    type Item = ReadValue<'a>;
    fn next(&mut self) -> Option<Self::Item> {
        let result = self.array.get(self.next)?;
        self.next += 1;
        Some(result)
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.array.len() - self.next;
        (n, Some(n))
    }
}
impl ExactSizeIterator for ReadArrayIter<'_> {}
#[derive(Clone, Copy, Debug)]
pub(in crate::bounded) enum ReadObject<'a> {
    Json(&'a serde_json::Map<String, serde_json::Value>),
    Record(&'a [RecordField<'a>]),
}
impl<'a> ReadObject<'a> {
    pub(in crate::bounded) fn len(self) -> usize {
        match self {
            Self::Json(value) => value.len(),
            Self::Record(value) => RecordValue::Object(value).length().expect("object"),
        }
    }
    pub(in crate::bounded) fn get(self, key: &str) -> Option<ReadValue<'a>> {
        match self {
            Self::Json(value) => value.get(key).map(ReadValue::Json),
            Self::Record(value) => RecordValue::Object(value).field(key).map(ReadValue::Record),
        }
    }
    pub(in crate::bounded) fn pairs(self) -> ReadPairs<'a> {
        ReadPairs {
            object: self,
            next: 0,
            after: None,
            natural: match self {
                Self::Json(values) => NaturalPairs::Json(values.iter()),
                Self::Record(values) => NaturalPairs::Record(RecordFields {
                    fields: values,
                    next: 0,
                }),
            },
        }
    }
    fn entry(self, index: usize) -> Option<(&'a str, ReadValue<'a>)> {
        match self {
            Self::Json(value) => value
                .iter()
                .nth(index)
                .map(|(key, value)| (key.as_str(), ReadValue::Json(value))),
            Self::Record(value) => RecordValue::Object(value)
                .fields()?
                .nth(index)
                .map(|(key, value)| (key, ReadValue::Record(value))),
        }
    }
}
#[derive(Clone, Debug)]
enum NaturalPairs<'a> {
    Json(serde_json::map::Iter<'a>),
    Record(RecordFields<'a>),
}
impl<'a> Iterator for NaturalPairs<'a> {
    type Item = (&'a str, ReadValue<'a>);
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Json(values) => values
                .next()
                .map(|(key, value)| (key.as_str(), ReadValue::Json(value))),
            Self::Record(values) => values
                .next()
                .map(|(key, value)| (key, ReadValue::Record(value))),
        }
    }
}
#[derive(Clone, Debug)]
pub(in crate::bounded) struct ReadPairs<'a> {
    natural: NaturalPairs<'a>,
    object: ReadObject<'a>,
    next: usize,
    after: Option<&'a str>,
}
impl<'a> ReadPairs<'a> {
    pub(in crate::bounded) fn new(object: ReadObject<'a>) -> Self {
        object.pairs()
    }
}
impl<'a> Iterator for ReadPairs<'a> {
    type Item = (&'a str, ReadValue<'a>);
    fn next(&mut self) -> Option<Self::Item> {
        #[cfg(feature = "preserve_order")]
        let result = self.natural.next()?;
        #[cfg(not(feature = "preserve_order"))]
        let result = (0..self.object.len())
            .filter_map(|index| self.object.entry(index))
            .filter(|(key, _)| self.after.is_none_or(|after| *key > after))
            .min_by(|a, b| a.0.cmp(b.0))?;
        self.next += 1;
        self.after = Some(result.0);
        Some(result)
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.object.len() - self.next;
        (n, Some(n))
    }
}
impl ExactSizeIterator for ReadPairs<'_> {}
