//! Facade representation adapter over the paid ordinary serde event tree.
//! Compiled sources are cold declarations, separate from invocation authority.
use eredu_core::HostPreparationAuthority;
use eredu_nn::workspace::{HostMetadataFunding, HostMetadataFundingError};
use eredu_runtime::working_memory::{
    OriginalJsonChildren, OriginalJsonNode, OriginalJsonNumber, OriginalJsonTree,
    OriginalJsonTreeError, OriginalJsonValueKind,
};
use jsonschema::{
    JsonType, OriginalJson, OriginalValidationError, OriginalValidationFunding,
    OriginalValidationSource,
    json::{Array, Json, JsonNumber, Node, NodeIdentity, Object},
};
use llguidance::derivre::{ParserAllocationFailure, ParserAllocationFunding};
use std::{
    borrow::Cow,
    mem::{size_of, size_of_val},
    sync::{Arc, OnceLock, atomic::AtomicUsize},
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
pub(crate) struct Number<'a>(OriginalJsonNumber<'a>);
impl Json for Representation {
    type Node<'a> = Input<'a>;
    type PreparedKey = String;
    type StringBuffer = ();
    fn prepare_key_with_allocations(
        key: &str,
        allocations: &dyn serde_json::allocation::Allocation,
    ) -> Result<String, jsonschema::json::KeyPreparationError> {
        Ok(serde_json::allocation::Allocator::new(allocations).copy_string(key)?)
    }
    fn prepare_string_node_with_allocations<'a>(
        _: &'a mut (),
        string: &'a str,
        _: &dyn serde_json::allocation::Allocation,
    ) -> Result<Input<'a>, jsonschema::json::KeyPreparationError> {
        Ok(Input::String(string))
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
    fn original_key_bytes(key: &String) -> Option<usize> {
        Some(key.capacity())
    }
    fn original_number_bytes(number: &serde_json::Number) -> Option<usize> {
        Some(number.allocation_size())
    }
    fn original_input_controls() -> Option<usize> {
        // Only borrowed primitive operations are reached by qualified bodies.
        // to_value/as_str/default equality and uniqueness remain cold methods.
        let parts = [
            size_of::<Input<'_>>(),
            size_of::<Container<'_>>(),
            size_of::<Members<'_>>(),
            size_of::<Elements<'_>>(),
            size_of::<Number<'_>>(),
            size_of::<OriginalJsonNode<'_>>(),
            size_of::<OriginalJsonChildren<'_>>(),
            size_of::<Option<(&str, Input<'_>)>>(),
            size_of::<Option<Input<'_>>>(),
            size_of::<Option<Number<'_>>>(),
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
impl JsonNumber for Number<'_> {
    fn as_u64(&self) -> Option<u64> {
        match self.0 {
            OriginalJsonNumber::U64(n) => Some(n),
            OriginalJsonNumber::I64(n) => n.try_into().ok(),
            OriginalJsonNumber::F64(_) => None,
            OriginalJsonNumber::Exact(n)=>n.as_u64(),
        }
    }
    fn as_i64(&self) -> Option<i64> {
        match self.0 {
            OriginalJsonNumber::I64(n) => Some(n),
            OriginalJsonNumber::U64(n) => n.try_into().ok(),
            OriginalJsonNumber::F64(_) => None,
            OriginalJsonNumber::Exact(n)=>n.as_i64(),
        }
    }
    fn as_f64(&self) -> Option<f64> {
        Some(match self.0 {
            OriginalJsonNumber::I64(n) => n as f64,
            OriginalJsonNumber::U64(n) => n as f64,
            OriginalJsonNumber::F64(n) => n,
            OriginalJsonNumber::Exact(n)=>return n.as_f64(),
        })
    }
    // These cold conversions remain unqualified in original dispatch.
    fn as_str(&self) -> Cow<'_, str> {
        if let OriginalJsonNumber::Exact(number)=self.0 {if let Some(text)=number.source_text(){return Cow::Borrowed(text);}}
        Cow::Owned(self.to_number().to_string())
    }
    fn to_number(&self) -> Cow<'_, serde_json::Number> {
        Cow::Owned(match self.0 {
            OriginalJsonNumber::Exact(n)=>return Cow::Borrowed(n),
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
            OriginalJsonNumber::Exact(n)=>jsonschema::json::JsonNumber::is_integer(&n),
        }
    }
}
impl<'a> Node<'a, Representation> for Input<'a> {
    type Object = Container<'a>;
    type Array = Container<'a>;
    type Number = Number<'a>;
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
    fn as_number(&self) -> Option<Number<'a>> {
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
struct CompilationAccount {
    failure: OnceLock<ParserAllocationFailure>,
    funding: ParserAllocationFunding,
}
impl jsonschema::CompilationFunding for CompilationAccount {
    fn reserve(&self, bytes: usize) -> Result<(), jsonschema::CompilationAllocationError> {
        if self.failure.get().is_some() {
            return Err(jsonschema::CompilationAllocationError::Refused);
        }
        self.funding.reserve(bytes).map_err(|cause| {
            let _ = self.failure.set(cause);
            jsonschema::CompilationAllocationError::Refused
        })
    }
    fn source_error(&self) -> Option<&(dyn std::error::Error + Send + Sync + 'static)> {
        self.failure.get().map(|cause| cause as _)
    }
}
#[derive(Debug, thiserror::Error)]
enum CompilationCause {
    #[error(transparent)]
    Funding(#[from] ParserAllocationFailure),
    #[error(transparent)]
    Schema(#[from] jsonschema::CompilationError),
    #[error("tool schema compiler extent overflow")]
    Overflow,
}
/// The dependency error and callback shell retire before their original payer.
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
pub(crate) struct CompilationFailure {
    #[source]
    cause: CompilationCause,
    authority: HostPreparationAuthority,
    funding: ParserAllocationFunding,
}

impl CompilationFailure {
    pub(crate) fn is_local_property_error(&self) -> bool {
        matches!(&self.cause, CompilationCause::Schema(error)
            if error.schema_error().is_some() || matches!(error.reference_error(),
                Some(jsonschema::ReferencingError::PointerToNowhere { .. }
                    | jsonschema::ReferencingError::NoSuchAnchor { .. })))
    }
    pub(crate) fn is_unqualified(&self) -> bool {
        use std::error::Error;
        let mut cause: &(dyn Error + 'static) = self;
        loop {
            if matches!(
                cause.downcast_ref::<OriginalValidationError>(),
                Some(OriginalValidationError::Unqualified(_))
            ) || matches!(
                cause.downcast_ref::<jsonschema::CompilationAllocationError>(),
                Some(jsonschema::CompilationAllocationError::Unqualified(_))
            ) {
                return true;
            }
            match cause.source() {
                Some(next) => cause = next,
                None => return false,
            }
        }
    }
}

#[derive(Debug)]
struct Payload {
    validator: OriginalValidationSource<Representation>,
    authority: HostPreparationAuthority,
    funding: ParserAllocationFunding,
}
/// Cold typed source. Construction is never called by the original completion.
#[derive(Debug, Clone)]
pub(crate) struct Source(Option<Arc<Payload>>);
impl Drop for Source {
    fn drop(&mut self) {
        if let Some(owner) = self.0.take() {
            // The last alias frees its shell before the graph and source payer.
            drop(Arc::into_inner(owner));
        }
    }
}
#[derive(Debug, thiserror::Error)]
enum Cause {
    #[error(transparent)]
    Funding(#[from] HostMetadataFundingError),
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
    funding: HostMetadataFunding,
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
    funding: &'a HostMetadataFunding,
    failure: Option<HostMetadataFundingError>,
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
    fn payload(&self) -> &Payload {
        self.0.as_deref().expect("live schema source")
    }
    /// One schema compiler for funded and explicitly unenforced preparation.
    /// The dependency owns its default options, graph and diagnostic producers.
    pub(crate) fn compile(
        schema: &serde_json::Value,
        authority: &HostPreparationAuthority,
        funding: &ParserAllocationFunding,
    ) -> Result<Self, CompilationFailure> {
        let result = (|| -> Result<Self, CompilationCause> {
            let source_shell = std::alloc::Layout::new::<[AtomicUsize; 2]>()
                .extend(std::alloc::Layout::new::<Payload>())
                .map_err(|_| CompilationCause::Overflow)?
                .0
                .pad_to_align()
                .size();
            let callback_shell = if funding.is_enforced() {
                std::alloc::Layout::new::<[AtomicUsize; 2]>()
                    .extend(std::alloc::Layout::new::<CompilationAccount>())
                    .map_err(|_| CompilationCause::Overflow)?
                    .0
                    .pad_to_align()
                    .size()
            } else {
                0
            };
            let controls = [
                source_shell,
                callback_shell,
                size_of::<Self>(),
                size_of::<Payload>(),
                size_of::<Option<Arc<Payload>>>(),
                size_of::<Option<Payload>>(),
                size_of::<CompilationAccount>(),
                size_of::<Option<Arc<dyn jsonschema::CompilationFunding>>>(),
                size_of::<CompilationCause>(),
                size_of::<CompilationFailure>(),
                size_of::<Result<Self, CompilationFailure>>(),
                size_of::<Result<Self, CompilationCause>>(),
                size_of::<
                    Result<jsonschema::Validator<Representation>, jsonschema::CompilationError>,
                >(),
                size_of::<jsonschema::Validator<Representation>>(),
                size_of::<OriginalValidationSource<Representation>>(),
                size_of::<(
                    &serde_json::Value,
                    &HostPreparationAuthority,
                    &ParserAllocationFunding,
                )>(),
            ];
            let bytes = controls
                .into_iter()
                .try_fold(size_of_val(&controls), usize::checked_add)
                .ok_or(CompilationCause::Overflow)?;
            funding.reserve(bytes)?;
            let source = if funding.is_enforced() {
                Some(Arc::new(CompilationAccount {
                    failure: OnceLock::new(),
                    funding: funding.clone(),
                })
                    as Arc<dyn jsonschema::CompilationFunding>)
            } else {
                None
            };
            let validator =
                jsonschema::Validator::<Representation>::build_with_funding(schema, source)?;
            Ok(Self(Some(Arc::new(Payload {
                validator: OriginalValidationSource::new(validator)?,
                authority: authority.clone(),
                funding: funding.clone(),
            }))))
        })();
        result.map_err(|cause| CompilationFailure {
            cause,
            authority: authority.clone(),
            funding: funding.clone(),
        })
    }
    /// Cold immutable backing census. This is called before original admission;
    /// completion consumes a stored receipt and never walks this graph.
    pub(crate) fn capacity_bytes(&self) -> Result<usize, CompilationFailure> {
        let payload = self.payload();
        let result = (|| -> Result<_, CompilationCause> {
            let parts = [
                size_of::<&Self>(),
                size_of::<std::alloc::Layout>(),
                size_of::<CompilationCause>(),
                size_of::<CompilationFailure>(),
                size_of::<Result<usize, CompilationFailure>>(),
                size_of::<Result<usize, CompilationCause>>(),
                size_of::<Result<usize, jsonschema::CompilationError>>(),
            ];
            let controls = parts
                .into_iter()
                .try_fold(size_of_val(&parts), usize::checked_add)
                .ok_or(CompilationCause::Overflow)?;
            payload.funding.reserve(controls)?;
            let shell = std::alloc::Layout::new::<[AtomicUsize; 2]>()
                .extend(std::alloc::Layout::new::<Payload>())
                .map_err(|_| CompilationCause::Overflow)?
                .0
                .pad_to_align()
                .size();
            shell
                .checked_add(payload.validator.retained_bytes()?)
                .ok_or(CompilationCause::Overflow)
        })();
        result.map_err(|cause| CompilationFailure {
            cause,
            authority: payload.authority.clone(),
            funding: payload.funding.clone(),
        })
    }
    pub(crate) fn validate(&self, arguments: &str, funding: &HostMetadataFunding) -> Result<(), Failure> {
        self.evaluate(arguments, true, funding).map(|_| ())
    }
    pub(crate) fn matches(&self, value: &str, funding: &HostMetadataFunding) -> Result<bool, Failure> {
        self.evaluate(value, false, funding)
    }
    fn evaluate(&self, arguments: &str, require_object: bool, funding: &HostMetadataFunding) -> Result<bool, Failure> {
        let mut failure = Failure {
            cause: Cause::Parse,
            tree: None,
            parse_failure: None,
            source: self.clone(),
            funding: funding.clone(),
        };
        let result = (|| -> Result<bool, Cause> {
            let parts = [
                size_of::<Self>(),
                size_of::<Failure>(),
                size_of::<Cause>(),
                size_of::<Loan<'_>>(),
                size_of::<Option<HostMetadataFundingError>>(),
                size_of::<Result<bool, Failure>>(),
                size_of::<Result<bool, Cause>>(),
                size_of::<Result<OriginalJsonTree, OriginalJsonTreeError>>(),
                size_of::<(&Self, &str, bool, &HostMetadataFunding)>(),
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
            if require_object && root.kind() != OriginalJsonValueKind::Object {
                return Err(Cause::Object);
            }
            let mut loan = Loan {
                funding,
                failure: None,
            };
            let result = self
                .payload()
                .validator
                .is_valid(Input::Tree(root), &mut loan);
            // Preserve the original payer cause, rather than its Boolean bridge.
            if let Some(cause) = loan.failure {
                return Err(cause.into());
            }
            let valid = result?;
            if require_object && !valid { return Err(Cause::Mismatch); }
            Ok(valid)
        })();
        match result {
            Ok(valid) => Ok(valid),
            Err(cause) => {
                failure.cause = cause;
                Err(failure)
            }
        }
    }
}
