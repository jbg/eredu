//! Stock schema compilation and validation with retained admission headroom.
use crate::runtime::chat::preparation_memory::{PreparationFailure, PreparationFunding};
use eredu_core::HostPreparationAuthority;
use eredu_nn::workspace::{HostMetadataFunding, HostMetadataFundingError};
use eredu_runtime::working_memory::{OriginalJsonTree, OriginalJsonTreeError};
use std::{
    alloc::Layout,
    mem::size_of,
    sync::{Arc, atomic::AtomicUsize},
};

#[derive(Debug, thiserror::Error)]
enum CompilationCause {
    #[error(transparent)]
    Funding(#[from] PreparationFailure),
    #[error(transparent)]
    Schema(#[from] jsonschema::ValidationError<'static>),
    #[error("tool schema admission estimate overflow")]
    Overflow,
}
/// The dependency diagnostic retires before its source authority and account.
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
pub(crate) struct CompilationFailure {
    #[source]
    cause: CompilationCause,
    authority: HostPreparationAuthority,
    funding: PreparationFunding,
}
impl CompilationFailure {
    pub(crate) fn is_local_property_error(&self) -> bool {
        let CompilationCause::Schema(error) = &self.cause else {
            return false;
        };
        // A property fragment can refer to definitions owned by the full object.
        // Other reference failures still propagate; the full object is always
        // compiled and validated independently.
        match error.kind() {
            jsonschema::error::ValidationErrorKind::Referencing(error) => matches!(
                error,
                jsonschema::ReferencingError::PointerToNowhere { .. }
                    | jsonschema::ReferencingError::NoSuchAnchor { .. }
            ),
            _ => true,
        }
    }
}
#[derive(Debug)]
struct Payload {
    validator: jsonschema::Validator,
    admission_bytes: usize,
    authority: HostPreparationAuthority,
    funding: PreparationFunding,
}
/// Immutable compiled schema; aliases retain its construction account.
#[derive(Debug, Clone)]
pub(crate) struct Source(Option<Arc<Payload>>);
impl Drop for Source {
    fn drop(&mut self) {
        if let Some(owner) = self.0.take() {
            drop(Arc::into_inner(owner));
        }
    }
}
#[derive(Debug, thiserror::Error)]
enum Cause {
    #[error(transparent)]
    Funding(#[from] HostMetadataFundingError),
    #[error("tool arguments must be a JSON object")]
    Object,
    #[error("tool arguments do not match their original full schema")]
    Mismatch,
    #[error("original tool input parse failed")]
    Parse,
    #[error("tool validation admission estimate overflow")]
    Overflow,
}
/// Parsed arguments and compilation sources retire before the invocation payer.
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
            _ => Some(&self.cause),
        }
    }
}
impl Source {
    fn payload(&self) -> &Payload {
        self.0.as_deref().expect("live schema source")
    }

    pub(crate) fn compile(
        schema: &serde_json::Value,
        authority: &HostPreparationAuthority,
        funding: &PreparationFunding,
    ) -> Result<Self, CompilationFailure> {
        let result = (|| -> Result<Self, CompilationCause> {
            let shell = Layout::new::<[AtomicUsize; 2]>()
                .extend(Layout::new::<Payload>())
                .map_err(|_| CompilationCause::Overflow)?
                .0
                .pad_to_align()
                .size();
            // Stock serde's counting writer and schema compiler have separate
            // headroom. Neither reservation measures their internal allocation.
            funding.reserve_dependency(0)?;
            let estimate = funding
                .memory_policy()
                .estimate_json(schema)
                .ok_or(CompilationCause::Overflow)?;
            let admission_bytes = shell
                .checked_add(estimate)
                .ok_or(CompilationCause::Overflow)?;
            funding.reserve(admission_bytes)?;
            let validator = jsonschema::Validator::new(schema)?;
            Ok(Self(Some(Arc::new(Payload {
                validator,
                admission_bytes,
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
    /// Configured immutable-source planning allowance, not a heap census.
    pub(crate) fn admission_bytes(&self) -> usize {
        self.payload().admission_bytes
    }

    pub(crate) fn validate(
        &self,
        arguments: &str,
        funding: &HostMetadataFunding,
    ) -> Result<(), Failure> {
        self.evaluate(arguments, true, funding).map(|_| ())
    }
    pub(crate) fn matches(
        &self,
        value: &str,
        funding: &HostMetadataFunding,
    ) -> Result<bool, Failure> {
        self.evaluate(value, false, funding)
    }
    fn evaluate(
        &self,
        arguments: &str,
        require_object: bool,
        funding: &HostMetadataFunding,
    ) -> Result<bool, Failure> {
        let mut failure = Failure {
            cause: Cause::Parse,
            tree: None,
            parse_failure: None,
            source: self.clone(),
            funding: funding.clone(),
        };
        let result = (|| -> Result<bool, Cause> {
            let policy = self.payload().funding.memory_policy();
            let bytes = policy
                .estimate(arguments.len())
                .and_then(|n| n.checked_add(size_of::<Failure>()))
                .ok_or(Cause::Overflow)?;
            funding.reserve_metadata(bytes)?;
            let tree = match OriginalJsonTree::parse_with_memory_policy(arguments, funding, policy)
            {
                Ok(tree) => tree,
                Err(cause) => {
                    failure.parse_failure = Some(cause);
                    return Err(Cause::Parse);
                }
            };
            failure.tree = Some(tree);
            let value = failure.tree.as_ref().expect("stored input").value();
            if require_object && !value.is_object() {
                return Err(Cause::Object);
            }
            let valid = self.payload().validator.is_valid(value);
            if require_object && !valid {
                return Err(Cause::Mismatch);
            }
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

#[cfg(test)]
mod tests;
