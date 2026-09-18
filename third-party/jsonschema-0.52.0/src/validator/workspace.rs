//! Explicit original allocation loan, used by the ordinary validation context.
use std::{
    alloc::Layout,
    collections::TryReserveError,
    fmt,
    mem::{size_of, size_of_val},
};
/// A caller-owned payer. Refusal is recorded by the caller in its existing
/// error storage; this borrow neither allocates nor manufactures authority.
pub trait Funding {
    fn reserve(&mut self, bytes: usize) -> bool;
}
#[derive(Debug, Clone, Copy)]
pub enum Component {
    RegexSyntax,
    Source(&'static str),
    Validator(&'static str),
}
#[derive(Debug)]
pub enum Error {
    Funding,
    Overflow,
    Allocation(TryReserveError),
    Table(hashbrown::TryReserveError),
    Capacity,
    HostAllocation,
    Unqualified(Component),
}
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Funding => f.write_str("schema workspace funding refused"),
            Self::Overflow => f.write_str("schema workspace extent overflow"),
            Self::Allocation(e) => fmt::Display::fmt(e, f),
            Self::Table(e) => fmt::Display::fmt(e, f),
            Self::HostAllocation => f.write_str("schema workspace host allocation failed"),
            Self::Capacity => f.write_str("schema workspace allocation exceeds its exact request"),
            Self::Unqualified(component) => write!(
                f,
                "schema workspace source is not yet qualified: {component:?}"
            ),
        }
    }
}
impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Allocation(error) => Some(error),
            Self::Table(error) => Some(error),
            _ => None,
        }
    }
}
/// Where the same regex search worker retains its mutable scratch.
#[derive(Default)]
enum RegexCachePolicy {
    #[default]
    SharedPool,
    Invocation,
}
#[derive(Default)]
pub(crate) struct Workspace<'a> {
    regex_caches: Vec<fancy_regex::OwnedSearchWorkspace>,
    standard_caches: Vec<regex::OwnedSearchWorkspace>,
    regex_cache_policy: RegexCachePolicy,
    failure: Option<Error>,
    funding: Option<&'a mut dyn Funding>,
    hasher: Option<&'a ahash::RandomState>,
    input_controls: usize,
    string_controls: Option<(usize, usize)>,
}
/// A scoped loan used by dependency-owned producers; no destination retains it.
struct AllocationLoan<'a, 'f>(std::cell::RefCell<&'a mut Workspace<'f>>);
impl serde_json::allocation::Allocation for AllocationLoan<'_, '_> {
    fn reserve(&self, bytes: usize) -> Result<(), serde_json::allocation::AllocationError> {
        if self.0.borrow_mut().reserve(Some(bytes)) {
            Ok(())
        } else {
            Err(serde_json::allocation::AllocationError::Refused)
        }
    }
    fn is_enforced(&self) -> bool {
        self.0.borrow().original()
    }
}
impl jsonschema_regex::allocation::Allocation for AllocationLoan<'_, '_> {
    fn reserve(&self, bytes: usize) -> Result<(), jsonschema_regex::allocation::AllocationError> {
        if self.0.borrow_mut().reserve(Some(bytes)) {
            Ok(())
        } else {
            Err(jsonschema_regex::allocation::AllocationError::Refused)
        }
    }
}
impl fraction::operations::Allocation for AllocationLoan<'_, '_> {
    fn reserve(&self, bytes: usize) -> Result<(), fraction::operations::AllocationError> {
        if self.0.borrow_mut().reserve(Some(bytes)) {
            Ok(())
        } else {
            Err(fraction::operations::AllocationError::Refused)
        }
    }
    fn is_enforced(&self) -> bool {
        self.0.borrow().original()
    }
}
impl regex::allocation::Allocation for AllocationLoan<'_, '_> {
    fn reserve(&self, bytes: usize) -> Result<(), regex::allocation::AllocationError> {
        <Self as jsonschema_regex::allocation::Allocation>::reserve(self, bytes).map_err(Into::into)
    }
}
impl<'a> Workspace<'a> {
    pub(crate) fn new(funding: &'a mut dyn Funding, hasher: &'a ahash::RandomState) -> Self {
        Self {
            regex_caches: Vec::new(),
            standard_caches: Vec::new(),
            regex_cache_policy: RegexCachePolicy::Invocation,
            failure: None,
            funding: Some(funding),
            hasher: Some(hasher),
            input_controls: 0,
            string_controls: None,
        }
    }
    pub(crate) fn use_invocation_regex_cache(&mut self) {
        self.regex_cache_policy = RegexCachePolicy::Invocation;
    }
    pub(crate) fn set_input_controls(&mut self, bytes: usize) {
        self.input_controls = bytes;
    }
    pub(crate) fn input_controls(&self) -> usize {
        self.input_controls
    }
    pub(crate) fn set_string_controls(&mut self, controls: Option<(usize, usize)>) {
        self.string_controls = controls;
    }
    /// Refuse before an undeclared representation constructor can run.
    pub(crate) fn string_constructor(&mut self, buffer: bool) -> bool {
        if self.failed() {
            return false;
        }
        if !self.original() {
            return true;
        }
        let Some((prepare, invoke)) = self.string_controls else {
            return self.refuse(Error::Unqualified(Component::Source(
                "JSON string constructor",
            )));
        };
        let bytes = if buffer { prepare } else { invoke };
        self.reserve(bytes.checked_add(size_of::<(&mut Self, bool, usize, usize, usize)>()))
    }
    /// The original source owns this initialized seed; cloning it allocates nothing.
    pub(crate) fn hasher(&self) -> ahash::RandomState {
        self.hasher.cloned().unwrap_or_else(|| {
            crate::compilation::Funding::default()
                .random_state()
                .expect("ordinary schema hash seed")
        })
    }
    pub(crate) fn original(&self) -> bool {
        self.funding.is_some()
    }
    pub(crate) fn failed(&self) -> bool {
        self.failure.is_some()
    }
    pub(crate) fn into_failure(self) -> Option<Error> {
        self.failure
    }
    pub(crate) fn refuse(&mut self, error: Error) -> bool {
        if self.failure.is_none() {
            self.failure = Some(error);
        }
        false
    }
    pub(crate) fn reserve(&mut self, bytes: Option<usize>) -> bool {
        if self.failed() {
            return false;
        }
        let Some(funding) = self.funding.as_mut() else {
            return true;
        };
        let Some(bytes) = bytes else {
            return self.refuse(Error::Overflow);
        };
        if !funding.reserve(bytes) {
            return self.refuse(Error::Funding);
        }
        true
    }
    /// Prepare one representation-owned string before recursive validation borrows
    /// this workspace again. The allocation policy is a stack-only mutable loan.
    pub(crate) fn prepare_string_node<'n, F: crate::Json>(
        &mut self,
        buffer: &'n mut F::StringBuffer,
        string: &'n str,
    ) -> Option<F::Node<'n>> {
        if self.failed() {
            return None;
        }
        let controls = size_of::<AllocationLoan<'_, '_>>()
            .checked_add(size_of::<
                Result<F::Node<'n>, jsonschema_value::KeyPreparationError>,
            >())
            .and_then(|bytes| {
                bytes.checked_add(size_of::<(&mut Self, &mut F::StringBuffer, &str)>())
            });
        if !self.reserve(controls) {
            return None;
        }
        let result = F::prepare_string_node_with_allocations(
            buffer,
            string,
            &AllocationLoan(std::cell::RefCell::new(&mut *self)),
        );
        match result {
            Ok(node) => Some(node),
            Err(error) => {
                use jsonschema_value::KeyPreparationError;
                use serde_json::allocation::AllocationError;
                let error = match error {
                    KeyPreparationError::Unqualified(source) => {
                        Error::Unqualified(Component::Source(source))
                    }
                    KeyPreparationError::Allocation(AllocationError::Refused) => Error::Funding,
                    KeyPreparationError::Allocation(AllocationError::SizeOverflow) => {
                        Error::Overflow
                    }
                    KeyPreparationError::Allocation(AllocationError::HostAllocation) => {
                        Error::HostAllocation
                    }
                };
                self.refuse(error);
                None
            }
        }
    }
    pub(crate) fn with_json_allocations<T>(
        &mut self,
        operation: impl FnOnce(
            &dyn serde_json::allocation::Allocation,
        ) -> Result<T, serde_json::allocation::AllocationError>,
    ) -> Option<T> {
        if !self.reserve(
            size_of::<AllocationLoan<'_, '_>>()
                .checked_add(size_of::<Result<T, serde_json::allocation::AllocationError>>())
                .and_then(|bytes| bytes.checked_add(size_of_val(&operation)))
                .and_then(|bytes| bytes.checked_add(size_of::<&mut Self>())),
        ) {
            return None;
        }
        let result = operation(&AllocationLoan(std::cell::RefCell::new(&mut *self)));
        match result {
            Ok(value) => Some(value),
            Err(error) => {
                use serde_json::allocation::AllocationError;
                self.refuse(match error {
                    AllocationError::Refused => Error::Funding,
                    AllocationError::SizeOverflow => Error::Overflow,
                    AllocationError::HostAllocation => Error::HostAllocation,
                });
                None
            }
        }
    }
    pub(crate) fn with_numeric_allocations<T>(
        &mut self,
        operation: impl FnOnce(
            &dyn fraction::operations::Allocation,
        ) -> Result<T, fraction::operations::AllocationError>,
    ) -> Option<T> {
        if !self.reserve(
            size_of::<AllocationLoan<'_, '_>>()
                .checked_add(size_of::<Result<T, fraction::operations::AllocationError>>())
                .and_then(|bytes| bytes.checked_add(size_of_val(&operation)))
                .and_then(|bytes| bytes.checked_add(size_of::<&mut Self>())),
        ) {
            return None;
        }
        let result = operation(&AllocationLoan(std::cell::RefCell::new(&mut *self)));
        match result {
            Ok(value) => Some(value),
            Err(error) => {
                use fraction::operations::AllocationError;
                self.refuse(match error {
                    AllocationError::Refused => Error::Funding,
                    AllocationError::SizeOverflow => Error::Overflow,
                    AllocationError::HostAllocation => Error::HostAllocation,
                    AllocationError::Unqualified(source) => {
                        Error::Unqualified(Component::Validator(source))
                    }
                });
                None
            }
        }
    }
    pub(crate) fn copy_string(&mut self, source: &str) -> Option<String> {
        self.with_json_allocations(|allocation| {
            serde_json::allocation::Allocator::new(allocation).copy_string(source)
        })
    }
    fn match_regex_cache<C, E>(
        &mut self,
        caches: &mut Vec<C>,
        matches: impl Fn(&C) -> bool,
        create: impl FnOnce(&AllocationLoan<'_, '_>) -> Result<C, E>,
        run: impl FnOnce(&mut C, &AllocationLoan<'_, '_>) -> Result<bool, E>,
        pooled: impl FnOnce() -> Result<bool, E>,
        refuse: impl Fn(regex::allocation::AllocationError) -> E,
        allocation_error: impl Fn(&E) -> Option<regex::allocation::AllocationError>,
    ) -> Result<bool, E> {
        use regex::allocation::AllocationError;
        if matches!(self.regex_cache_policy, RegexCachePolicy::SharedPool) {
            return pooled();
        }
        let controls = [
            size_of::<(Option<usize>, C, AllocationLoan<'_, '_>, Result<bool, E>)>(),
            size_of_val(&matches),
            size_of_val(&create),
            size_of_val(&run),
            size_of_val(&pooled),
            size_of_val(&refuse),
            size_of_val(&allocation_error),
        ];
        if !self.reserve(
            controls
                .into_iter()
                .try_fold(size_of_val(&controls), usize::checked_add),
        ) {
            return Err(refuse(AllocationError::Refused));
        }
        let result = (|| {
            let index = match caches.iter().position(matches) {
                Some(index) => index,
                None => {
                    let cache = create(&AllocationLoan(std::cell::RefCell::new(&mut *self)))?;
                    if !self.push(caches, cache) {
                        return Err(refuse(AllocationError::Refused));
                    }
                    caches.len() - 1
                }
            };
            run(
                &mut caches[index],
                &AllocationLoan(std::cell::RefCell::new(&mut *self)),
            )
        })();
        if let Err(error) = &result {
            if let Some(error) = allocation_error(error) {
                self.refuse(match error {
                    AllocationError::Refused => Error::Funding,
                    AllocationError::SizeOverflow => Error::Overflow,
                    AllocationError::HostAllocation => Error::HostAllocation,
                });
            }
        }
        result
    }
    pub(crate) fn fancy_match(
        &mut self,
        regex: &std::sync::Arc<fancy_regex::Regex>,
        text: &str,
    ) -> Result<bool, fancy_regex::Error> {
        let mut caches = std::mem::take(&mut self.regex_caches);
        let result = self.match_regex_cache(
            &mut caches,
            |cache| cache.matches_source(regex),
            |loan| fancy_regex::OwnedSearchWorkspace::new_with_allocations(regex.clone(), loan),
            |cache, loan| cache.is_match(text, loan),
            || regex.is_match(text),
            |error| fancy_regex::Error::Allocation(error.into()),
            |error| match error {
                fancy_regex::Error::Allocation(error) => Some((*error).into()),
                _ => None,
            },
        );
        self.regex_caches = caches;
        result
    }
    pub(crate) fn standard_match(
        &mut self,
        regex: &std::sync::Arc<regex::Regex>,
        text: &str,
    ) -> Result<bool, regex::SearchError> {
        let mut caches = std::mem::take(&mut self.standard_caches);
        let result = self.match_regex_cache(
            &mut caches,
            |cache| cache.matches_source(regex),
            |loan| regex::OwnedSearchWorkspace::new_with_allocations(regex.clone(), loan),
            |cache, loan| cache.is_match(text, loan),
            || Ok(regex.is_match(text)),
            Into::into,
            regex::SearchError::allocation_error,
        );
        self.standard_caches = caches;
        result
    }
    pub(crate) fn ecma_regex(&mut self, pattern: &str) -> Option<bool> {
        if !self.reserve(
            size_of::<AllocationLoan<'_, '_>>()
                .checked_add(size_of::<
                    Result<bool, jsonschema_regex::allocation::AllocationError>,
                >())
                .and_then(|bytes| bytes.checked_add(size_of::<(&mut Self, &str)>())),
        ) {
            return None;
        }
        let result = jsonschema_regex::is_valid_ecma_regex_with_allocations(
            pattern,
            &AllocationLoan(std::cell::RefCell::new(&mut *self)),
        );
        match result {
            Ok(value) => Some(value),
            Err(error) => {
                use jsonschema_regex::allocation::AllocationError;
                self.refuse(match error {
                    AllocationError::Refused => Error::Funding,
                    AllocationError::SizeOverflow => Error::Overflow,
                    AllocationError::HostAllocation => Error::HostAllocation,
                });
                None
            }
        }
    }
    /// The ordinary initialized mask allocation, with exact paid capacity when loaned.
    pub(crate) fn filled<T: Copy>(&mut self, length: usize, value: T) -> Option<Vec<T>> {
        if !self.original() {
            return Some(vec![value; length]);
        }
        let parts = [
            size_of::<Self>(),
            size_of::<Vec<T>>(),
            size_of::<T>(),
            size_of::<Result<(), TryReserveError>>(),
            size_of::<Error>(),
            size_of::<Result<Layout, std::alloc::LayoutError>>(),
            size_of::<Option<Vec<T>>>(),
            size_of::<(&mut Self, usize, T)>(),
        ];
        if !self.reserve(
            parts
                .into_iter()
                .try_fold(size_of_val(&parts), usize::checked_add),
        ) {
            return None;
        }
        let Ok(layout) = Layout::array::<T>(length) else {
            self.refuse(Error::Overflow);
            return None;
        };
        if !self.reserve(Some(layout.size())) {
            return None;
        }
        let mut values = Vec::new();
        if let Err(error) = values.try_reserve_exact(length) {
            self.refuse(Error::Allocation(error));
            return None;
        }
        if size_of::<T>() != 0 && values.capacity() > length {
            self.refuse(Error::Capacity);
            return None;
        }
        values.resize(length, value);
        Some(values)
    }
    /// Existing ordinary push, or a source-sized replacement reserved before
    /// growth. All prefix elements remain with the actual Vec on refusal.
    pub(crate) fn push<T>(&mut self, values: &mut Vec<T>, value: T) -> bool {
        if !self.original() {
            values.push(value);
            return true;
        }
        let parts = [
            size_of::<Self>(),
            size_of::<Vec<T>>(),
            size_of::<T>(),
            size_of::<TryReserveError>(),
            size_of::<Result<(), TryReserveError>>(),
            size_of::<Error>(),
            size_of::<(usize, usize)>(),
            size_of::<(&mut Self, &mut Vec<T>, T)>(),
            size_of::<Result<Layout, std::alloc::LayoutError>>(),
        ];
        if !self.reserve(
            parts
                .into_iter()
                .try_fold(size_of_val(&parts), usize::checked_add),
        ) {
            return false;
        }
        if values.len() == values.capacity() {
            let Some(required) = values.len().checked_add(1) else {
                return self.refuse(Error::Overflow);
            };
            let Ok(layout) = Layout::array::<T>(required) else {
                return self.refuse(Error::Overflow);
            };
            if !self.reserve(Some(layout.size())) {
                return false;
            }
            if let Err(error) = values.try_reserve_exact(required - values.len()) {
                return self.refuse(Error::Allocation(error));
            }
            if size_of::<T>() != 0 && values.capacity() > required {
                return self.refuse(Error::Capacity);
            }
        }
        values.push(value);
        true
    }
}

/// Common borrowed invocation and exact selected representation frames.
/// Concrete bodies name their remaining locals at their own implementation.
pub(crate) fn body_controls<F: crate::Json, T>(locals: &[usize]) -> Result<usize, Error> {
    use crate::{Array, Node, Object};
    let parts = [
        size_of::<&T>(),
        size_of::<&F::Node<'_>>(),
        size_of::<F::Node<'_>>(),
        size_of::<<F::Node<'_> as Node<'_, F>>::Object>(),
        size_of::<<F::Node<'_> as Node<'_, F>>::Array>(),
        size_of::<<F::Node<'_> as Node<'_, F>>::Number>(),
        size_of::<<<F::Node<'_> as Node<'_, F>>::Object as Object<'_, F>>::MembersIter>(),
        size_of::<<<F::Node<'_> as Node<'_, F>>::Object as Object<'_, F>>::MemberName>(),
        size_of::<<<F::Node<'_> as Node<'_, F>>::Array as Array<'_, F>>::ElementsIter>(),
        size_of::<Option<F::Node<'_>>>(),
        size_of::<Option<crate::NodeIdentity>>(),
        size_of::<Option<bool>>(),
        size_of::<Option<std::borrow::Cow<'_, str>>>(),
        size_of::<(&[usize], usize)>(),
        size_of::<Result<usize, Error>>(),
    ];
    parts
        .into_iter()
        .chain(locals.iter().copied())
        .try_fold(
            size_of_val(&parts)
                .checked_add(size_of_val(locals))
                .ok_or(Error::Overflow)?,
            usize::checked_add,
        )
        .ok_or(Error::Overflow)
}

#[cfg(test)]
mod tests {
    use super::{Error, Funding};
    use crate::{
        validator::{Validate, ValidationContext},
        NodeIdentity,
    };
    use serde_json::json;

    #[derive(Default)]
    struct Payer {
        calls: usize,
        fail_at: Option<usize>,
        charged: usize,
    }
    impl Funding for Payer {
        fn reserve(&mut self, bytes: usize) -> bool {
            let call = self.calls;
            self.calls += 1;
            if self.fail_at == Some(call) {
                return false;
            }
            self.charged = self.charged.checked_add(bytes).unwrap();
            true
        }
    }

    #[test]
    fn regex_caches_reuse_exact_sources_within_the_invocation() {
        use std::{cell::Cell, sync::Arc};
        struct Meter<'a>(&'a Cell<usize>);
        impl Funding for Meter<'_> {
            fn reserve(&mut self, _: usize) -> bool {
                self.0.set(self.0.get() + 1);
                true
            }
        }
        let source = Arc::new(fancy_regex::Regex::new(r"^[a-z][a-z0-9_-]{1,32}$").unwrap());
        let equal_source = Arc::new(fancy_regex::Regex::new(source.as_str()).unwrap());
        let weak = Arc::downgrade(&source);
        let calls = Cell::new(0);
        let seed = ahash::RandomState::with_seeds(1, 2, 3, 4);
        let mut meter = Meter(&calls);
        let mut workspace = super::Workspace::new(&mut meter, &seed);
        assert!(workspace.fancy_match(&source, "property_name_42").unwrap());
        assert_eq!(workspace.regex_caches.len(), 1);
        let first = calls.get();
        assert!(workspace.fancy_match(&source, "property_name_42").unwrap());
        assert_eq!(
            calls.get(),
            first + 1,
            "only the fixed invocation frame is charged after cache warmup"
        );
        assert!(workspace
            .fancy_match(&equal_source, "property_name_42")
            .unwrap());
        assert_eq!(
            workspace.regex_caches.len(),
            2,
            "equal text does not authenticate a distinct source owner"
        );
        assert!(calls.get() > first + 2);
        drop(source);
        assert!(
            weak.upgrade().is_some(),
            "invocation retains the actual source alias"
        );
        drop(workspace);
        assert!(
            weak.upgrade().is_none(),
            "source alias retires with invocation scratch"
        );
    }

    #[test]
    fn standard_caches_reuse_exact_sources_within_the_invocation() {
        use std::{cell::Cell, sync::Arc};
        struct Meter<'a>(&'a Cell<usize>);
        impl Funding for Meter<'_> {
            fn reserve(&mut self, _: usize) -> bool {
                self.0.set(self.0.get() + 1);
                true
            }
        }
        let source = Arc::new(regex::Regex::new(r"^[a-z][a-z0-9_-]{1,32}$").unwrap());
        let equal_source = Arc::new(regex::Regex::new(source.as_str()).unwrap());
        let weak = Arc::downgrade(&source);
        let calls = Cell::new(0);
        let seed = ahash::RandomState::with_seeds(1, 2, 3, 4);
        let mut meter = Meter(&calls);
        let mut workspace = super::Workspace::new(&mut meter, &seed);
        assert!(workspace
            .standard_match(&source, "property_name_42")
            .unwrap());
        assert_eq!(workspace.standard_caches.len(), 1);
        let first = calls.get();
        assert!(workspace
            .standard_match(&source, "property_name_42")
            .unwrap());
        assert_eq!(
            calls.get(),
            first + 1,
            "only the fixed invocation frame is charged after cache warmup"
        );
        assert!(workspace
            .standard_match(&equal_source, "property_name_42")
            .unwrap());
        assert_eq!(
            workspace.standard_caches.len(),
            2,
            "equal text does not authenticate a distinct source owner"
        );
        assert!(calls.get() > first + 2);
        drop(source);
        assert!(
            weak.upgrade().is_some(),
            "invocation retains the actual source alias"
        );
        drop(workspace);
        assert!(
            weak.upgrade().is_none(),
            "source alias retires with invocation scratch"
        );
    }

    #[test]
    fn paid_recursive_context_shares_actual_validation_and_keeps_refused_prefix() {
        // Schema, parsed input, and initialized hash source preexist this loan.
        // This exercises the context producer, not yet a full-schema admission API.
        let seed = ahash::RandomState::new();
        let schema = json!({
            "$defs": {"node": {"type": "object", "properties": {
                "value": {"type": "integer"}, "next": {"$ref": "#/$defs/node"}
            }}}, "$ref": "#/$defs/node"
        });
        let validator = crate::validator_for(&schema).unwrap();
        let valid = json!({"value": 1, "next": {"value": 2, "next": {"value": 3}}});
        let invalid = json!({"value": 1, "next": {"value": "bad"}});
        for input in [&valid, &invalid] {
            let ordinary = validator.is_valid(input);
            let mut payer = Payer::default();
            let mut context = ValidationContext::with_workspace(&mut payer, &seed);
            assert_eq!(validator.root.is_valid(&input, &mut context), ordinary);
            assert!(context.validating.is_empty());
            assert!(context.workspace_failure().is_none());
            assert!(payer.charged > 0);
        }
        let mut payer = Payer::default();
        let mut context = ValidationContext::with_workspace(&mut payer, &seed);
        assert!(validator.root.is_valid(&&valid, &mut context));
        assert!(context.workspace_failure().is_none());
        let calls = payer.calls;
        for fail_at in 0..calls {
            let mut payer = Payer {
                fail_at: Some(fail_at),
                ..Payer::default()
            };
            let mut context = ValidationContext::with_workspace(&mut payer, &seed);
            let _ = validator.root.is_valid(&&valid, &mut context);
            assert!(context.validating.is_empty());
            assert!(matches!(context.workspace_failure(), Some(Error::Funding)));
            assert_eq!(payer.calls, fail_at + 1);
        }

        // The actual unevaluated traversals use initialized item masks and
        // borrowed property names through the same context and hash source.
        for (schema, valid, invalid) in [
            (
                json!({"prefixItems": [{"type": "integer"}],
                "contains": {"type": "string"}, "unevaluatedItems": false}),
                json!([1, "x", "y"]),
                json!([1, "x", 2]),
            ),
            (
                json!({"allOf": [{"properties": {"a": {"type": "integer"}}}],
                "if": {"required": ["b"]},
                "then": {"properties": {"b": {"type": "string"}}},
                "unevaluatedProperties": false}),
                json!({"a": 1, "b": "x"}),
                json!({"a": 1, "b": "x", "extra": true}),
            ),
        ] {
            let validator = crate::validator_for(&schema).unwrap();
            for (input, expected) in [(&valid, true), (&invalid, false)] {
                assert_eq!(validator.is_valid(input), expected);
                let mut payer = Payer::default();
                let mut context = ValidationContext::with_workspace(&mut payer, &seed);
                assert_eq!(validator.root.is_valid(&input, &mut context), expected);
                assert!(context.marking.is_empty());
                assert!(context.workspace_failure().is_none());
                assert!(payer.charged > 0);
            }
            let mut payer = Payer::default();
            let mut context = ValidationContext::with_workspace(&mut payer, &seed);
            assert!(validator.root.is_valid(&&valid, &mut context));
            assert!(context.workspace_failure().is_none());
            for fail_at in 0..payer.calls {
                let mut payer = Payer {
                    fail_at: Some(fail_at),
                    ..Payer::default()
                };
                let mut context = ValidationContext::with_workspace(&mut payer, &seed);
                let _ = validator.root.is_valid(&&valid, &mut context);
                assert!(context.marking.is_empty());
                assert!(context.validating.is_empty());
                assert!(matches!(context.workspace_failure(), Some(Error::Funding)));
                assert_eq!(payer.calls, fail_at + 1);
            }
        }

        // An unqualified intrinsic refuses before its body can execute. This
        // does not turn an ordinary valid schema into a successful paid result.
        struct Unqualified(std::sync::atomic::AtomicBool);
        impl crate::validator::Validate for Unqualified {
            fn is_valid_body(&self, _: &&serde_json::Value, _: &mut ValidationContext) -> bool {
                self.0.store(true, std::sync::atomic::Ordering::SeqCst);
                true
            }
            fn validate_body<'i>(
                &self,
                _: &<crate::SerdeJson as crate::Json>::Node<'i>,
                _: &crate::paths::LazyLocation,
                _: Option<&crate::paths::RefTracker>,
                _: &mut ValidationContext,
            ) -> Result<(), crate::ValidationError<'i>> {
                Ok(())
            }
        }
        let unqualified = Unqualified(std::sync::atomic::AtomicBool::new(false));
        let mut payer = Payer::default();
        let mut context = ValidationContext::with_workspace(&mut payer, &seed);
        assert!(!unqualified.is_valid(&&valid, &mut context));
        assert!(!unqualified.0.load(std::sync::atomic::Ordering::SeqCst));
        assert!(matches!(
            context.workspace_failure(),
            Some(Error::Unqualified(super::Component::Validator(_)))
        ));

        // Ordinary representations receive no implicit buffer-construction
        // authority merely because the property-name body is qualified.
        let names_schema = json!({"propertyNames": {"minLength": 1}});
        let names = crate::validator_for(&names_schema).unwrap();
        let named = json!({"key": 1});
        assert!(names.is_valid(&named));
        let mut payer = Payer::default();
        let mut context = ValidationContext::with_workspace(&mut payer, &seed);
        assert!(!names.root.is_valid(&&named, &mut context));
        assert!(matches!(
            context.workspace_failure(),
            Some(Error::Unqualified(super::Component::Source(
                "JSON string constructor"
            )))
        ));

        // A later refused insertion cannot erase previously published results.
        // Duplicate replacement consumes controls but no new table allocation.
        let first = Some(NodeIdentity::new(1));
        let next = Some(NodeIdentity::new(2));
        struct Switch<'a>(&'a std::cell::Cell<bool>);
        impl Funding for Switch<'_> {
            fn reserve(&mut self, _: usize) -> bool {
                !self.0.get()
            }
        }
        let denied = std::cell::Cell::new(false);
        let mut payer = Switch(&denied);
        let mut context = ValidationContext::with_workspace(&mut payer, &seed);
        context.cache_result(7, first, true);
        context.cache_result(7, first, false);
        context.cache_result(8, next, true);
        assert_eq!(context.get_cached_result(7, first), Some(false));
        assert_eq!(context.get_cached_result(8, next), Some(true));
        denied.set(true);
        context.cache_result(9, Some(NodeIdentity::new(3)), true); // denied controls
        assert_eq!(context.get_cached_result(7, first), Some(false));
        assert_eq!(context.get_cached_result(8, next), Some(true));
        assert_eq!(
            context.get_cached_result(9, Some(NodeIdentity::new(3))),
            None
        );
        assert!(matches!(context.workspace_failure(), Some(Error::Funding)));
    }
}
