//! Facade representation adapter over the paid ordinary serde event tree.
//! Compiled sources are cold declarations, separate from invocation authority.
use eredu_core::HostPreparationAuthority;
use eredu_nn::workspace::{WorkspaceMetadataFunding, WorkspaceMetadataFundingError};
use eredu_runtime::working_memory::{
    OriginalJsonChildren, OriginalJsonNode, OriginalJsonNumber, OriginalJsonTree,
    OriginalJsonTreeError, OriginalJsonValueKind,
};
use jsonschema::{
    json::{Array, Json, JsonNumber, Node, NodeIdentity, Object},
    JsonType, OriginalJson, OriginalValidationError, OriginalValidationFunding,
    OriginalValidationSource,
};
use std::{
    borrow::Cow,
    mem::{size_of, size_of_val},
    sync::Arc,
};

#[derive(Debug)]
pub(crate) struct Representation;
#[derive(Clone, Copy)]
pub(crate) enum Input<'a> {
    Tree(OriginalJsonNode<'a>),
    String(&'a str),
}
pub(crate) struct Members<'a>(OriginalJsonChildren<'a>);
pub(crate) struct Elements<'a>(OriginalJsonChildren<'a>);
pub(crate) struct Container<'a>(OriginalJsonNode<'a>);
pub(crate) struct Number(OriginalJsonNumber);
impl Json for Representation {
    type Node<'a> = Input<'a>;
    type PreparedKey = String;
    type StringBuffer = ();
    fn prepare_key(key: &str) -> String {
        key.to_owned()
    }
    fn with_string_node<T>(_: &mut (), string: &str, f: impl FnOnce(Input<'_>) -> T) -> T {
        f(Input::String(string))
    }
}
impl OriginalJson for Representation {
    fn original_string_controls() -> Option<(usize, usize)> {
        // The actual buffer is unit and the string node borrows its key.
        // Callback and recursive validation frames belong to the caller.
        let prepare = size_of::<()>();
        let invoke = size_of::<(&mut (), &str, Input<'_>)>();
        Some((prepare, invoke))
    }
    fn original_key_bytes(key: &String) -> Option<usize> { Some(key.capacity()) }
    fn original_number_bytes(_: &serde_json::Number) -> Option<usize> {
        // The actual serde plan refuses arbitrary-precision representation.
        serde_json::bounded_events::Plan::prepare(b"0").ok().map(|_| 0)
    }
    fn original_input_controls() -> Option<usize> {
        // Only borrowed primitive operations are reached by qualified bodies.
        // to_value/as_str/default equality and uniqueness remain cold methods.
        let parts = [
            size_of::<Input<'_>>(),
            size_of::<Container<'_>>(),
            size_of::<Members<'_>>(),
            size_of::<Elements<'_>>(),
            size_of::<Number>(),
            size_of::<OriginalJsonNode<'_>>(),
            size_of::<OriginalJsonChildren<'_>>(),
            size_of::<Option<(&str, Input<'_>)>>(),
            size_of::<Option<Input<'_>>>(),
            size_of::<Option<Number>>(),
            size_of::<Option<NodeIdentity>>(),
            size_of::<Option<Cow<'_, str>>>(),
            size_of::<std::str::Chars<'_>>(),
            size_of::<(&Container<'_>, &String)>(),
            size_of::<(&Input<'_>,)>(),
            size_of::<(usize, usize, bool)>(),
            size_of::<std::cmp::Ordering>(),
            size_of::<Option<i64>>(),
            size_of::<Option<u64>>(),
            size_of::<Option<f64>>(),
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
}
impl JsonNumber for Number {
    fn as_u64(&self) -> Option<u64> {
        match self.0 {
            OriginalJsonNumber::U64(n) => Some(n),
            OriginalJsonNumber::I64(n) => n.try_into().ok(),
            OriginalJsonNumber::F64(_) => None,
        }
    }
    fn as_i64(&self) -> Option<i64> {
        match self.0 {
            OriginalJsonNumber::I64(n) => Some(n),
            OriginalJsonNumber::U64(n) => n.try_into().ok(),
            OriginalJsonNumber::F64(_) => None,
        }
    }
    fn as_f64(&self) -> Option<f64> {
        Some(match self.0 {
            OriginalJsonNumber::I64(n) => n as f64,
            OriginalJsonNumber::U64(n) => n as f64,
            OriginalJsonNumber::F64(n) => n,
        })
    }
    // These cold conversions remain unqualified in original dispatch.
    fn as_str(&self) -> Cow<'_, str> {
        Cow::Owned(self.to_number().to_string())
    }
    fn to_number(&self) -> Cow<'_, serde_json::Number> {
        Cow::Owned(match self.0 {
            OriginalJsonNumber::I64(n) => n.into(),
            OriginalJsonNumber::U64(n) => n.into(),
            OriginalJsonNumber::F64(n) => {
                serde_json::Number::from_f64(n).expect("finite serde event")
            }
        })
    }
    fn is_integer(&self) -> bool {
        match self.0 {
            OriginalJsonNumber::I64(_) | OriginalJsonNumber::U64(_) => true,
            OriginalJsonNumber::F64(n) => n.fract() == 0.0,
        }
    }
}
impl<'a> Node<'a, Representation> for Input<'a> {
    type Object = Container<'a>;
    type Array = Container<'a>;
    type Number = Number;
    fn as_object(&self) -> Option<Container<'a>> {
        match self {
            Self::Tree(node) if node.kind() == OriginalJsonValueKind::Object => {
                Some(Container(*node))
            }
            _ => None,
        }
    }
    fn as_array(&self) -> Option<Container<'a>> {
        match self {
            Self::Tree(node) if node.kind() == OriginalJsonValueKind::Array => {
                Some(Container(*node))
            }
            _ => None,
        }
    }
    fn as_string(&self) -> Option<Cow<'a, str>> {
        match self {
            Self::String(s) => Some(Cow::Borrowed(s)),
            Self::Tree(node) => node.string().map(Cow::Borrowed),
        }
    }
    fn as_number(&self) -> Option<Number> {
        match self {
            Self::Tree(node) => node.number().map(Number),
            Self::String(_) => None,
        }
    }
    fn as_boolean(&self) -> Option<bool> {
        match self {
            Self::Tree(node) => node.boolean(),
            Self::String(_) => None,
        }
    }
    fn is_null(&self) -> bool {
        self.json_type() == JsonType::Null
    }
    fn json_type(&self) -> JsonType {
        match self {
            Self::String(_) => JsonType::String,
            Self::Tree(node) => match node.kind() {
                OriginalJsonValueKind::Object => JsonType::Object,
                OriginalJsonValueKind::Array => JsonType::Array,
                OriginalJsonValueKind::String => JsonType::String,
                OriginalJsonValueKind::Number => JsonType::Number,
                OriginalJsonValueKind::Bool => JsonType::Boolean,
                OriginalJsonValueKind::Null => JsonType::Null,
            },
        }
    }
    fn to_value(&self) -> Cow<'a, serde_json::Value> {
        // Preserve the complete ordinary representation contract for cold
        // callers. No qualified original body calls this allocating conversion.
        let value = match self.json_type() {
            JsonType::Object => serde_json::Value::Object(
                self.as_object()
                    .unwrap()
                    .members()
                    .map(|(key, value)| (key.to_owned(), value.to_value().into_owned()))
                    .collect(),
            ),
            JsonType::Array => serde_json::Value::Array(
                self.as_array()
                    .unwrap()
                    .elements()
                    .map(|value| value.to_value().into_owned())
                    .collect(),
            ),
            JsonType::String => serde_json::Value::String(self.as_string().unwrap().into_owned()),
            JsonType::Number | JsonType::Integer => {
                serde_json::Value::Number(self.as_number().unwrap().to_number().into_owned())
            }
            JsonType::Boolean => serde_json::Value::Bool(self.as_boolean().unwrap()),
            JsonType::Null => serde_json::Value::Null,
        };
        Cow::Owned(value)
    }
    fn identity(&self) -> Option<NodeIdentity> {
        match self {
            Self::Tree(node) => Some(NodeIdentity::new(node.storage_identity())),
            // Transient property-name nodes do not enter the persistent container cache.
            Self::String(_) => None,
        }
    }
}
impl<'a> Iterator for Members<'a> {
    type Item = (&'a str, Input<'a>);
    fn next(&mut self) -> Option<Self::Item> {
        self.0
            .next()
            .map(|(key, value)| (key.expect("object member"), Input::Tree(value)))
    }
}
impl<'a> Iterator for Elements<'a> {
    type Item = Input<'a>;
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(|(_, value)| Input::Tree(value))
    }
}
impl<'a> Object<'a, Representation> for Container<'a> {
    type Node = Input<'a>;
    type MemberName = &'a str;
    type MembersIter = Members<'a>;
    fn len(&self) -> usize {
        self.0.children().expect("container").len()
    }
    fn get(&self, key: &String) -> Option<Input<'a>> {
        self.0.get(key).map(Input::Tree)
    }
    fn members(&self) -> Members<'a> {
        Members(self.0.children().expect("object"))
    }
}
impl<'a> Array<'a, Representation> for Container<'a> {
    type Node = Input<'a>;
    type ElementsIter = Elements<'a>;
    fn len(&self) -> usize {
        self.0.children().expect("container").len()
    }
    fn elements(&self) -> Elements<'a> {
        Elements(self.0.children().expect("array"))
    }
}

#[derive(Debug)]
struct Payload {
    validator: OriginalValidationSource<Representation>,
    authority: HostPreparationAuthority,
}
/// Cold typed source. Construction is never called by the original completion.
#[derive(Debug, Clone)]
pub(crate) struct Source(Arc<Payload>);
#[derive(Debug, thiserror::Error)]
enum Cause {
    #[error(transparent)]
    Funding(#[from] WorkspaceMetadataFundingError),
    #[error(transparent)]
    Validation(#[from] OriginalValidationError),
    #[error("tool arguments must be a JSON object")]
    Object,
    #[error("tool arguments do not match their original full schema")]
    Mismatch,
    #[error("original tool input parse failed")]
    Parse,
    #[error("original tool validation extent overflow")]
    Overflow,
}
/// All actual arguments (including parse prefixes), compiled source and funding
/// survive a callback failure until the caller releases this error.
#[derive(Debug)]
pub(crate) struct Failure {
    cause: Cause,
    tree: Option<OriginalJsonTree>,
    parse_failure: Option<OriginalJsonTreeError>,
    source: Source,
    funding: WorkspaceMetadataFunding,
}
impl std::fmt::Display for Failure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match (&self.cause, &self.parse_failure) {
            (Cause::Parse, Some(error)) => std::fmt::Display::fmt(error, f),
            _ => std::fmt::Display::fmt(&self.cause, f),
        }
    }
}
impl std::error::Error for Failure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match (&self.cause, &self.parse_failure) {
            (Cause::Parse, Some(error)) => Some(error),
            (Cause::Funding(error), _) => Some(error),
            (Cause::Validation(error), _) => Some(error),
            _ => Some(&self.cause),
        }
    }
}
struct Loan<'a> {
    funding: &'a WorkspaceMetadataFunding,
    failure: Option<WorkspaceMetadataFundingError>,
}
impl OriginalValidationFunding for Loan<'_> {
    fn reserve(&mut self, bytes: usize) -> bool {
        if self.failure.is_some() {
            return false;
        }
        match self.funding.reserve_metadata(bytes) {
            Ok(()) => true,
            Err(cause) => {
                self.failure = Some(cause);
                false
            }
        }
    }
}
impl Source {
    /// Ordinary immutable source construction. The caller supplies and retains
    /// source authority; this method makes no finite compilation bound claim.
    pub(crate) fn compile(
        schema: &serde_json::Value,
        authority: &HostPreparationAuthority,
    ) -> Result<Self, String> {
        let options = jsonschema::ValidationOptions::<
            '_,
            Arc<dyn jsonschema::Retrieve>,
            Representation,
        >::default();
        let validator = options.build(schema).map_err(|error| error.to_string())?;
        Ok(Self(Arc::new(Payload {
            validator: OriginalValidationSource::new(validator),
            authority: authority.clone(),
        })))
    }
    /// Cold immutable backing census. This is called before original admission;
    /// completion consumes a stored receipt and never walks this graph.
    pub(crate) fn capacity_bytes(&self) -> Result<usize, OriginalValidationError> {
        let shell = std::alloc::Layout::new::<[std::sync::atomic::AtomicUsize; 2]>()
            .extend(std::alloc::Layout::new::<Payload>()).map_err(|_| OriginalValidationError::Overflow)?
            .0.pad_to_align().size();
        shell.checked_add(self.0.validator.retained_bytes()?).ok_or(OriginalValidationError::Overflow)
    }
    pub(crate) fn validate(
        &self,
        arguments: &str,
        funding: &WorkspaceMetadataFunding,
    ) -> Result<(), Failure> {
        let mut failure = Failure {
            cause: Cause::Parse,
            tree: None,
            parse_failure: None,
            source: self.clone(),
            funding: funding.clone(),
        };
        let result = (|| -> Result<(), Cause> {
            let parts = [
                size_of::<Self>(),
                size_of::<Failure>(),
                size_of::<Cause>(),
                size_of::<Loan<'_>>(),
                size_of::<Option<WorkspaceMetadataFundingError>>(),
                size_of::<Result<(), Failure>>(),
                size_of::<Result<(), Cause>>(),
                size_of::<Result<OriginalJsonTree, OriginalJsonTreeError>>(),
                size_of::<(&Self, &str, &WorkspaceMetadataFunding)>(),
                OriginalValidationSource::<Representation>::control_bytes()
                    .ok_or(Cause::Overflow)?,
            ];
            funding.reserve_metadata(
                parts
                    .into_iter()
                    .try_fold(size_of_val(&parts), usize::checked_add)
                    .ok_or(Cause::Overflow)?,
            )?;
            let tree = match OriginalJsonTree::parse(arguments, funding) {
                Ok(tree) => tree,
                Err(cause) => {
                    failure.parse_failure = Some(cause);
                    return Err(Cause::Parse);
                }
            };
            failure.tree = Some(tree);
            let root = failure.tree.as_ref().expect("stored input").root();
            if root.kind() != OriginalJsonValueKind::Object {
                return Err(Cause::Object);
            }
            let mut loan = Loan {
                funding,
                failure: None,
            };
            let result = self.0.validator.is_valid(Input::Tree(root), &mut loan);
            // Preserve the original payer cause, rather than its Boolean bridge.
            if let Some(cause) = loan.failure {
                return Err(cause.into());
            }
            if !result? {
                return Err(Cause::Mismatch);
            }
            Ok(())
        })();
        match result {
            Ok(()) => Ok(()),
            Err(cause) => {
                failure.cause = cause;
                Err(failure)
            }
        }
    }
}
