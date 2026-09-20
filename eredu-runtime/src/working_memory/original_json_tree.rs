//! Immutable stock JSON values retaining their original host-admission account.
use super::{DependencyMemoryPolicy, OriginalJsonValueKind};
use eredu_nn::workspace::{HostMetadataFunding, HostMetadataFundingError};
use serde::Deserialize;
use serde_json::Value;
use std::mem::size_of;

/// A parsed JSON number. Exact borrows the stock representation, including
/// any numeric precision selected by upstream serde_json features.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum OriginalJsonNumber<'a> {
    /// Signed integer.
    I64(i64),
    /// Unsigned integer.
    U64(u64),
    /// Finite floating value, including signed zero.
    F64(f64),
    /// Original upstream number without conversion or copying.
    Exact(&'a serde_json::Number),
}
#[derive(Debug, thiserror::Error)]
enum Cause {
    #[error(transparent)]
    Syntax(#[from] serde_json::Error),
    #[error(transparent)]
    Funding(#[from] HostMetadataFundingError),
    #[error("JSON tree admission estimate overflow")]
    Overflow,
}
/// An immutable stock JSON tree with retained admission headroom. Object order,
/// duplicate keys, numbers and syntax follow the configured serde_json release.
#[derive(Debug)]
pub struct OriginalJsonTree {
    value: Value,
    funding: HostMetadataFunding,
}
/// A syntax or funding failure retaining its account and a completed root if
/// trailing bytes are invalid. Upstream retires its incomplete parse temporaries.
#[derive(Debug)]
pub struct OriginalJsonTreeError {
    cause: Cause,
    completed: Option<Value>,
    funding: HostMetadataFunding,
}
impl std::fmt::Display for OriginalJsonTreeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.cause, f)
    }
}
impl std::error::Error for OriginalJsonTreeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(match &self.cause {
            Cause::Funding(error) => error,
            Cause::Syntax(error) => error,
            Cause::Overflow => &self.cause,
        })
    }
}
impl OriginalJsonTree {
    /// Admits default dependency headroom, then parses the complete JSON input.
    pub fn parse(
        input: &str,
        funding: &HostMetadataFunding,
    ) -> Result<Self, OriginalJsonTreeError> {
        Self::parse_with_memory_policy(input, funding, DependencyMemoryPolicy::default())
    }
    /// Configures upstream construction headroom. It is an estimate, not a
    /// measured heap inventory or enforceable dependency/process memory ceiling.
    pub fn parse_with_memory_policy(
        input: &str,
        funding: &HostMetadataFunding,
        policy: DependencyMemoryPolicy,
    ) -> Result<Self, OriginalJsonTreeError> {
        let mut completed = None;
        let result = (|| -> Result<(), Cause> {
            let bytes = policy
                .estimate(input.len())
                .and_then(|n| n.checked_add(size_of::<Self>() + size_of::<OriginalJsonTreeError>()))
                .ok_or(Cause::Overflow)?;
            funding.reserve_metadata(bytes)?;
            let mut parser = serde_json::Deserializer::from_str(input);
            completed = Some(Value::deserialize(&mut parser)?);
            parser.end()?;
            Ok(())
        })();
        match result {
            Ok(()) => Ok(Self {
                value: completed.expect("parsed root"),
                funding: funding.clone(),
            }),
            Err(cause) => Err(OriginalJsonTreeError {
                cause,
                completed,
                funding: funding.clone(),
            }),
        }
    }
    /// Borrows the stock value directly for upstream validation APIs.
    pub fn value(&self) -> &Value {
        &self.value
    }
    /// Borrows the complete root and its immutable children.
    pub fn root(&self) -> OriginalJsonNode<'_> {
        OriginalJsonNode(&self.value)
    }
}
/// A borrowed immutable value that cannot outlive its tree and payer.
#[derive(Clone, Copy, Debug)]
pub struct OriginalJsonNode<'a>(&'a Value);
impl<'a> OriginalJsonNode<'a> {
    /// Kind of the actual parsed value.
    pub fn kind(self) -> OriginalJsonValueKind {
        OriginalJsonValueKind::of(self.0)
    }
    /// Decoded UTF-8 string, absent for every other kind.
    pub fn string(self) -> Option<&'a str> {
        self.0.as_str()
    }
    /// Original numeric representation without narrowing.
    pub fn number(self) -> Option<OriginalJsonNumber<'a>> {
        self.0.as_number().map(OriginalJsonNumber::Exact)
    }
    /// Boolean value, absent for other kinds.
    pub fn boolean(self) -> Option<bool> {
        self.0.as_bool()
    }
    /// Stable address while this immutable borrow is alive.
    pub fn storage_identity(self) -> usize {
        std::ptr::from_ref(self.0) as usize
    }
    /// Object members in stock serde order or array values with no key.
    pub fn children(self) -> Option<OriginalJsonChildren<'a>> {
        match self.0 {
            Value::Array(values) => Some(OriginalJsonChildren(Children::Array(values.iter()))),
            Value::Object(values) => Some(OriginalJsonChildren(Children::Object(values.iter()))),
            _ => None,
        }
    }
    /// Looks up one decoded object key without allocating.
    pub fn get(self, name: &str) -> Option<Self> {
        self.0.as_object()?.get(name).map(Self)
    }
}
#[derive(Clone, Debug)]
enum Children<'a> {
    Array(std::slice::Iter<'a, Value>),
    Object(serde_json::map::Iter<'a>),
}
/// Allocation-free borrowed child traversal with an exact remaining length.
#[derive(Clone, Debug)]
pub struct OriginalJsonChildren<'a>(Children<'a>);
impl<'a> Iterator for OriginalJsonChildren<'a> {
    type Item = (Option<&'a str>, OriginalJsonNode<'a>);
    fn next(&mut self) -> Option<Self::Item> {
        match &mut self.0 {
            Children::Array(values) => values.next().map(|v| (None, OriginalJsonNode(v))),
            Children::Object(values) => values
                .next()
                .map(|(k, v)| (Some(k.as_str()), OriginalJsonNode(v))),
        }
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        match &self.0 {
            Children::Array(values) => values.size_hint(),
            Children::Object(values) => values.size_hint(),
        }
    }
}
impl ExactSizeIterator for OriginalJsonChildren<'_> {}
impl std::iter::FusedIterator for OriginalJsonChildren<'_> {}

#[cfg(test)]
mod numeric_tests {
    use super::*;
    use eredu_core::HostMetadataAccount;
    use std::sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
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
