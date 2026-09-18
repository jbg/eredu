//! Paid object fields over the same cursor used by the ordinary tool parser.
use super::{OriginalJsonValue, OriginalJsonValueError, OriginalJsonValueKind};
use eredu_core::{HostPreparationAuthority, SemanticText, SpeculativeBuffer};
use eredu_nn::workspace::{HostMetadataFunding, HostMetadataFundingError};
use eredu_text::json_fragments::{JsonFieldNames, JsonFieldError, ObjectContext, ObjectCursor, ObjectSyntaxError};
use std::{
    mem::{size_of, size_of_val},
    ops::Range,
};

#[derive(Debug, thiserror::Error)]
enum Cause {
    #[error(transparent)] Field(#[from] JsonFieldError),
    #[error(transparent)]
    Funding(#[from] HostMetadataFundingError),
    #[error(transparent)]
    Buffer(#[from] eredu_core::SpeculativeBufferAllocationError),
    #[error(transparent)]
    Value(#[from] OriginalJsonValueError),
    #[error(transparent)]
    Syntax(#[from] ObjectSyntaxError),
    #[error("JSON object contains a duplicate decoded key: {0:?}")]
    Duplicate(SemanticText),
    #[error("prepared JSON object destination is full")]
    Capacity,
    #[error("prepared JSON object source extent overflow")]
    Overflow,
    #[error("prepared JSON object source is inconsistent")]
    Source,
}
/// One decoded key and its completed root value. Payload aliases retain their
/// original payer; the exact raw spelling remains in the enclosing object.
#[derive(Debug, Clone)]
pub struct OriginalJsonField {
    key: SemanticText,
    raw: Option<Range<usize>>,
    kind: Option<OriginalJsonValueKind>,
    text: Option<SemanticText>,
}
impl OriginalJsonField {
    /// Exact decoded field name, including JSON escapes.
    pub fn key(&self) -> &SemanticText {
        &self.key
    }
    /// Absent until the complete field has passed the actual JSON parser.
    pub fn kind(&self) -> Option<OriginalJsonValueKind> {
        self.kind
    }
    /// Paid decoded string for a completed string value.
    pub fn string(&self) -> Option<&SemanticText> {
        self.text.as_ref()
    }
}
#[derive(Debug)]
struct Data {
    raw: SpeculativeBuffer<u8>,
    fields: SpeculativeBuffer<OriginalJsonField>,
    funding: HostMetadataFunding,
}
impl Data {
    fn source(&self) -> &str {
        std::str::from_utf8(&self.raw).expect("whole UTF-8 characters")
    }
}
impl ObjectContext for Data {
    type Key = usize;
    type Error = Cause;
    fn raw_len(&self) -> usize {
        self.raw.len()
    }
    fn append(&mut self, character: char) -> Result<(), Cause> {
        if self.raw.capacity() - self.raw.len() < character.len_utf8() {
            return Err(Cause::Capacity);
        }
        let mut bytes = [0; 4];
        self.raw
            .try_extend(character.encode_utf8(&mut bytes).bytes())
            .map_err(|_| Cause::Capacity)
    }
    fn key(&mut self, raw: Range<usize>) -> Result<usize, Cause> {
        let key = OriginalJsonValue::parse(&self.source()[raw], &self.funding)?
            .into_string()
            .ok_or(Cause::Source)?;
        if self.fields.iter().any(|field| field.key == key) {
            return Err(Cause::Duplicate(key));
        }
        if self.fields.len() == self.fields.capacity() {
            return Err(Cause::Capacity);
        }
        let index = self.fields.len();
        self.fields
            .try_push(OriginalJsonField {
                key,
                raw: None,
                kind: None,
                text: None,
            })
            .map_err(|_| Cause::Capacity)?;
        Ok(index)
    }
    fn value(&mut self, key: usize, raw: Range<usize>) -> Result<(), Cause> {
        self.value_with_fields(key, raw, None)
    }
    fn syntax(&self, cause: ObjectSyntaxError) -> Cause {
        Cause::Syntax(cause)
    }
}
impl Data {
    fn value_with_fields(&mut self, key: usize, raw: Range<usize>, fields: Option<JsonFieldNames<'_>>) -> Result<(), Cause> {
        let value = OriginalJsonValue::parse(&self.source()[raw.clone()], &self.funding)?;
        if let Some(fields) = fields {
            fields.inspect(self.fields.get(key).ok_or(Cause::Source)?.key.as_str(), value.kind(), value.string())?;
        }
        let kind = value.kind();
        let text = value.into_string();
        let field = self.fields.get_mut(key).ok_or(Cause::Source)?;
        field.raw = Some(raw);
        field.kind = Some(kind);
        field.text = text;
        Ok(())
    }
}
struct FieldContext<'a, 'p> { data: &'a mut Data, fields: JsonFieldNames<'p> }
impl ObjectContext for FieldContext<'_, '_> {
    type Key = usize;
    type Error = Cause;
    fn raw_len(&self) -> usize { self.data.raw_len() }
    fn append(&mut self, character: char) -> Result<(), Cause> { self.data.append(character) }
    fn key(&mut self, raw: Range<usize>) -> Result<usize, Cause> { self.data.key(raw) }
    fn value(&mut self, key: usize, raw: Range<usize>) -> Result<(), Cause> { self.data.value_with_fields(key, raw, Some(self.fields)) }
    fn syntax(&self, cause: ObjectSyntaxError) -> Cause { self.data.syntax(cause) }
}
/// Finite raw/key/value destinations and the actual shared object cursor.
/// This mechanism validates JSON syntax and fields, not tool names or schemas.
#[derive(Debug)]
pub struct OriginalJsonObject {
    cursor: ObjectCursor<usize>,
    data: Data,
    complete: bool,
}
/// An operation failure owns the complete successful prefix until destruction.
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
pub struct OriginalJsonObjectError {
    #[source]
    cause: Cause,
    prefix: Option<OriginalJsonObject>,
    funding: HostMetadataFunding,
}
impl OriginalJsonObject {
    fn controls() -> Option<usize> {
        let parts = [
            ObjectCursor::<usize>::control_bytes::<Data>()?,
            ObjectCursor::<usize>::control_bytes::<FieldContext<'_, '_>>()?,
            JsonFieldNames::control_bytes()?,
            size_of::<FieldContext<'_, '_>>(), size_of::<Option<JsonFieldNames<'_>>>(),
            size_of::<Self>(),
            size_of::<Data>(),
            size_of::<OriginalJsonField>(),
            size_of::<Cause>(),
            size_of::<OriginalJsonObjectError>(),
            size_of::<Option<Self>>(),
            size_of::<Result<Self, OriginalJsonObjectError>>(),
            size_of::<Result<(Self, usize, bool), OriginalJsonObjectError>>(),
            size_of::<Result<(usize, bool), Cause>>(),
            size_of::<Result<(), Cause>>(),
            size_of::<Result<(), HostMetadataFundingError>>(),
            size_of::<Result<(), eredu_core::GenerationError>>(),
            size_of::<(&Self, &HostMetadataFunding)>(),
            size_of::<(&mut Data, Range<usize>, usize)>(),
            size_of::<[u8; 4]>(),
            size_of::<std::str::Bytes<'_>>(),
            size_of::<std::str::Chars<'_>>(),
            size_of::<std::slice::Iter<'_, OriginalJsonField>>(),
            size_of::<std::iter::Cloned<std::slice::Iter<'_, OriginalJsonField>>>(),
            size_of::<std::iter::Copied<std::slice::Iter<'_, u8>>>(),
            size_of::<Option<&OriginalJsonField>>(),
            size_of::<Option<&str>>(),
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
    fn buffer_bytes<T>(capacity: usize) -> Option<usize> {
        SpeculativeBuffer::<T>::retained_control_bytes(capacity)
            .and_then(|n| {
                n.checked_add(HostPreparationAuthority::retention_bytes::<
                    HostMetadataFunding,
                >()?)
            })
            .and_then(|n| n.checked_add(size_of::<Result<SpeculativeBuffer<T>, Cause>>()))
            .and_then(|n| n.checked_add(size_of::<(usize, &HostMetadataFunding)>()))
    }
    fn buffer<T>(capacity: usize, funding: &HostMetadataFunding) -> Result<SpeculativeBuffer<T>, Cause> {
        let bytes = Self::buffer_bytes::<T>(capacity).ok_or(Cause::Overflow)?;
        funding.reserve_metadata(bytes)?;
        Ok(SpeculativeBuffer::try_new_retained(
            capacity,
            HostPreparationAuthority::retain(funding.clone()),
        )?)
    }
    /// Pays each actual destination before allocating it. Every field needs at
    /// least one input byte, so input_bytes also bounds the retained field rows.
    pub fn prepare(
        input_bytes: usize,
        funding: &HostMetadataFunding,
    ) -> Result<Self, OriginalJsonObjectError> {
        let mut owner = Self {
            cursor: ObjectCursor::default(),
            data: Data {
                raw: SpeculativeBuffer::default(),
                fields: SpeculativeBuffer::default(),
                funding: funding.clone(),
            },
            complete: false,
        };
        let result = (|| {
            funding.reserve_metadata(Self::controls().ok_or(Cause::Overflow)?)?;
            owner.data.raw = Self::buffer(input_bytes, funding)?;
            owner.data.fields = Self::buffer(input_bytes, funding)?;
            Ok::<_, Cause>(())
        })();
        match result {
            Ok(()) => Ok(owner),
            Err(cause) => Err(owner.fail(cause)),
        }
    }
    fn fail(self, cause: Cause) -> OriginalJsonObjectError {
        OriginalJsonObjectError {
            funding: self.data.funding.clone(),
            cause,
            prefix: Some(self),
        }
    }
    /// Consumes a split input through the shared cursor. The returned consumed
    /// byte count excludes an enclosing envelope's suffix. Failure keeps every
    /// raw byte, decoded key and completed value already produced.
    pub fn push(self, input: &str) -> Result<(Self, usize, bool), OriginalJsonObjectError> {
        self.push_inner(input, None, true)
    }
    pub(super) fn push_fields(self, input: &str, fields: JsonFieldNames<'_>) -> Result<(Self, usize, bool), OriginalJsonObjectError> {
        self.push_inner(input, Some(fields), false)
    }
    fn push_inner(mut self, input: &str, fields: Option<JsonFieldNames<'_>>, validate_object: bool) -> Result<(Self, usize, bool), OriginalJsonObjectError> {
        let result = (|| {
            self.data
                .funding
                .reserve_metadata(Self::controls().ok_or(Cause::Overflow)?)?;
            let (consumed, complete) = if let Some(fields) = fields {
                self.cursor.push(input, &mut FieldContext { data: &mut self.data, fields })?
            } else { self.cursor.push(input, &mut self.data)? };
            if validate_object && complete && !self.complete {
                let value =
                    OriginalJsonValue::parse(self.data.source().trim(), &self.data.funding)?;
                if value.kind() != OriginalJsonValueKind::Object {
                    return Err(Cause::Source);
                }
                self.complete = true;
            }
            Ok::<_, Cause>((consumed, complete))
        })();
        match result {
            Ok((consumed, complete)) => Ok((self, consumed, complete)),
            Err(cause) => Err(self.fail(cause)),
        }
    }
    pub(super) fn finish_object(mut self) -> Result<Self, OriginalJsonObjectError> {
        let result = (|| {
            self.data.funding.reserve_metadata(Self::controls().ok_or(Cause::Overflow)?)?;
            let value = OriginalJsonValue::parse(self.data.source().trim(), &self.data.funding)?;
            if value.kind() != OriginalJsonValueKind::Object { return Err(Cause::Source); }
            self.complete = true;
            Ok::<_, Cause>(())
        })();
        match result { Ok(()) => Ok(self), Err(cause) => Err(self.fail(cause)) }
    }
    /// Exact raw spelling retained by the same mutable parser owner.
    pub fn source(&self) -> &str {
        self.data.source()
    }
    /// Actual decoded field rows, including at most one pending value.
    pub fn fields(&self) -> &[OriginalJsonField] {
        &self.data.fields
    }
    /// Exact completed value spelling for this object's field index.
    pub fn field_value(&self, index: usize) -> Option<&str> {
        Some(&self.source()[self.data.fields.get(index)?.raw.clone()?])
    }
    /// Exact growing value span for streaming, before full validation completes.
    pub fn active_value(&self) -> Option<(&SemanticText, &str)> {
        let (key, raw) = self.cursor.active_value(self.data.raw.len())?;
        Some((&self.data.fields.get(*key)?.key, &self.source()[raw]))
    }
    pub(super) fn copy_bytes(&self) -> Option<usize> {
        Self::controls()?.checked_add(Self::buffer_bytes::<u8>(self.data.raw.capacity())?)?
            .checked_add(Self::buffer_bytes::<OriginalJsonField>(self.data.fields.capacity())?)
    }
    /// Copies fixed destinations under new funding. Existing immutable decoded
    /// text aliases retain their original account as well as the new copy owner.
    pub fn try_copy(
        &self,
        funding: &HostMetadataFunding,
    ) -> Result<Self, OriginalJsonObjectError> {
        let bytes = self.copy_bytes().ok_or_else(|| OriginalJsonObjectError { cause: Cause::Overflow, prefix: None, funding: funding.clone() })?;
        funding.reserve_metadata(bytes).map_err(|cause| OriginalJsonObjectError { cause: cause.into(), prefix: None, funding: funding.clone() })?;
        self.copy_prepaid(HostPreparationAuthority::retain(funding.clone()), funding)
    }
    pub(super) fn rebind_funding(&mut self, funding: &HostMetadataFunding) {
        self.data.funding = funding.clone();
    }
    /// Same fixed-destination copy worker after its owner has prepaid copy_bytes.
    pub(super) fn copy_prepaid(&self, host: HostPreparationAuthority, funding: &HostMetadataFunding) -> Result<Self, OriginalJsonObjectError> {
        let mut copy = Self {
            cursor: self.cursor.clone(), complete: self.complete,
            data: Data { raw: SpeculativeBuffer::default(), fields: SpeculativeBuffer::default(), funding: funding.clone() },
        };
        let result = (|| {
            copy.data.raw = SpeculativeBuffer::try_new_retained(self.data.raw.capacity(), host.clone())?;
            copy.data.fields = SpeculativeBuffer::try_new_retained(self.data.fields.capacity(), host)?;
            copy.data.raw.try_extend(self.data.raw.iter().copied()).map_err(|_| Cause::Capacity)?;
            copy.data.fields.try_extend(self.data.fields.iter().cloned()).map_err(|_| Cause::Capacity)?;
            Ok::<_, Cause>(())
        })();
        match result { Ok(()) => Ok(copy), Err(cause) => Err(copy.fail(cause)) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eredu_nn::workspace::HostMetadataAccount;
    use std::sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    };
    #[derive(Debug)]
    struct Account {
        refused: Arc<AtomicBool>,
        retired: Arc<AtomicBool>,
    }
    impl HostMetadataAccount for Account {
        fn reserve_metadata(&self, bytes: usize) -> Result<(), HostMetadataFundingError> {
            if self.refused.load(Ordering::SeqCst) {
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
            self.retired.store(true, Ordering::SeqCst);
        }
    }
    fn funding() -> (HostMetadataFunding, Arc<AtomicBool>, Arc<AtomicBool>) {
        let refused = Arc::new(AtomicBool::new(false));
        let retired = Arc::new(AtomicBool::new(false));
        (
            HostMetadataFunding::new(Account {
                refused: refused.clone(),
                retired: retired.clone(),
            })
            .unwrap(),
            refused,
            retired,
        )
    }
    #[test]
    fn paid_object_fields_share_split_spelling_copy_and_failed_prefix_custody() {
        use super::super::{OriginalJsonNode, OriginalJsonNumber, OriginalJsonTree};
        fn to_value(node: OriginalJsonNode<'_>) -> serde_json::Value {
            use serde_json::Value;
            match node.kind() {
                OriginalJsonValueKind::Object => Value::Object(
                    node.children()
                        .unwrap()
                        .map(|(key, child)| (key.unwrap().to_owned(), to_value(child)))
                        .collect(),
                ),
                OriginalJsonValueKind::Array => Value::Array(
                    node.children()
                        .unwrap()
                        .map(|(key, child)| {
                            assert!(key.is_none());
                            to_value(child)
                        })
                        .collect(),
                ),
                OriginalJsonValueKind::String => Value::String(node.string().unwrap().to_owned()),
                OriginalJsonValueKind::Number => match node.number().unwrap() {
                    OriginalJsonNumber::I64(value) => Value::from(value),
                    OriginalJsonNumber::U64(value) => Value::from(value),
                    OriginalJsonNumber::F64(value) => Value::from(value),
                    OriginalJsonNumber::Exact(value)=>Value::Number(value.clone()),
                },
                OriginalJsonValueKind::Bool => Value::Bool(node.boolean().unwrap()),
                OriginalJsonValueKind::Null => Value::Null,
            }
        }
        let tree_input = r#"{"z":{"old":[false]},"a":"é🙂","n":-0.0,"z":[1,18446744073709551615,-2,null],"\u0061":"last"}"#;
        let (payer, _, retired) = funding();
        let tree = OriginalJsonTree::parse(tree_input, &payer).unwrap();
        assert_eq!(
            to_value(tree.root()),
            serde_json::from_str::<serde_json::Value>(tree_input).unwrap()
        );
        assert_eq!(tree.root().children().unwrap().len(), 3);
        assert_eq!(
            tree.root()
                .children()
                .unwrap()
                .map(|(key, _)| key.unwrap())
                .collect::<Vec<_>>(),
            serde_json::from_str::<serde_json::Value>(tree_input).unwrap()
                .as_object().unwrap().keys().map(String::as_str).collect::<Vec<_>>()
        );
        assert_eq!(tree.root().get("a").unwrap().string(), Some("last"));
        assert_eq!(tree.root().get("z").unwrap().children().unwrap().len(), 4);
        assert_ne!(
            tree.root().storage_identity(),
            tree.root().get("z").unwrap().storage_identity()
        );
        assert_eq!(
            to_value(tree.root().get("n").unwrap()),
            serde_json::from_str::<serde_json::Value>("-0.0").unwrap()
        );
        drop(payer);
        assert!(!retired.load(Ordering::SeqCst));
        drop(tree);
        assert!(retired.load(Ordering::SeqCst));
        let (payer, _, retired) = funding();
        let failure = OriginalJsonTree::parse(r#"{"complete":{"s":"kept"},"partial":[1,"#, &payer)
            .unwrap_err();
        drop(payer);
        assert!(!retired.load(Ordering::SeqCst));
        drop(failure);
        assert!(retired.load(Ordering::SeqCst));

        let input = r#" {"arguments":{"v":[1,{"s":"a\"🙂"}]},"name":"ch\u0065ck","extra":true}"#;
        let expected: serde_json::Value = serde_json::from_str(input).unwrap();
        let (payer, _, retired) = funding();
        let mut escaped = None;
        for split in input.char_indices().map(|(i, _)| i).chain([input.len()]) {
            let owner = OriginalJsonObject::prepare(input.len(), &payer).unwrap();
            let (owner, consumed, _) = owner.push(&input[..split]).unwrap();
            assert_eq!(consumed, split);
            let (copy_payer, _, copy_retired) = funding();
            let copy = owner.try_copy(&copy_payer).unwrap();
            assert_ne!(owner.data.raw.as_ptr(), copy.data.raw.as_ptr());
            let (copy, consumed, complete) = copy.push(&input[split..]).unwrap();
            assert_eq!(consumed, input.len() - split);
            assert!(complete);
            assert_eq!(copy.source(), input);
            for (index, field) in copy.fields().iter().enumerate() {
                let actual: serde_json::Value =
                    serde_json::from_str(copy.field_value(index).unwrap()).unwrap();
                assert_eq!(actual, expected[field.key().as_str()]);
            }
            let name = copy
                .fields()
                .iter()
                .find(|f| f.key().as_str() == "name")
                .unwrap()
                .string()
                .unwrap();
            assert_eq!(name.as_str(), "check");
            drop(owner);
            drop(copy_payer);
            assert!(!copy_retired.load(Ordering::SeqCst));
            // Retain a decoded output alias from the original payer when the
            // complete input was copied; a destination alias does not relabel it.
            if split == input.len() {
                escaped = Some(name.clone());
            }
            drop(copy);
            assert!(copy_retired.load(Ordering::SeqCst));
        }
        drop(payer);
        assert!(!retired.load(Ordering::SeqCst));
        drop(escaped);
        assert!(retired.load(Ordering::SeqCst));

        let (payer, _, retired) = funding();
        let duplicate = r#"{"a":1,"\u0061":2}"#;
        let failure = OriginalJsonObject::prepare(duplicate.len(), &payer)
            .unwrap()
            .push(duplicate)
            .unwrap_err();
        assert!(matches!(&failure.cause, Cause::Duplicate(key) if key.as_str() == "a"));
        assert_eq!(failure.prefix.as_ref().unwrap().field_value(0), Some("1"));
        drop(payer);
        assert!(!retired.load(Ordering::SeqCst));
        drop(failure);
        assert!(retired.load(Ordering::SeqCst));

        let (payer, refused, retired) = funding();
        let owner = OriginalJsonObject::prepare(64, &payer).unwrap();
        let (owner, _, _) = owner.push(r#"{"name":"check", "#).unwrap();
        refused.store(true, Ordering::SeqCst);
        let failure = owner.push(r#""arguments":{}}"#).unwrap_err();
        assert!(matches!(&failure.cause, Cause::Funding(_)));
        assert_eq!(
            failure.prefix.as_ref().unwrap().fields()[0]
                .string()
                .unwrap()
                .as_str(),
            "check"
        );
        drop(payer);
        assert!(!retired.load(Ordering::SeqCst));
        drop(failure);
        assert!(retired.load(Ordering::SeqCst));

        let (payer, _, retired) = funding();
        let failure = OriginalJsonValue::parse(r#""escaped" trailing"#, &payer).unwrap_err();
        drop(payer);
        assert!(!retired.load(Ordering::SeqCst));
        drop(failure);
        assert!(retired.load(Ordering::SeqCst));
    }
}
