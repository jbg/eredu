use core::fmt;

use crate::allocation::{Allocation, AllocationError};
use fluent_uri::Uri;
use serde_json::Value;

#[cfg(target_family = "wasm")]
pub trait RetrieverBounds {}

#[cfg(target_family = "wasm")]
impl<T: ?Sized> RetrieverBounds for T {}

#[cfg(not(target_family = "wasm"))]
pub trait RetrieverBounds: Send + Sync {}

#[cfg(not(target_family = "wasm"))]
impl<T: ?Sized + Send + Sync> RetrieverBounds for T {}

/// Failure from a retrieval producer's prospective storage contract.
#[derive(Debug)]
pub enum RetrievalError {
    /// The retriever returned its existing semantic or transport failure.
    External(Box<dyn std::error::Error + Send + Sync>),
    /// A reached storage request failed before the producer allocated it.
    Allocation(AllocationError),
    /// The external producer has no prospective allocation contract.
    Unqualified(&'static str),
}
impl fmt::Display for RetrievalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::External(error) => error.fmt(f),
            Self::Allocation(error) => error.fmt(f),
            Self::Unqualified(producer) => write!(
                f,
                "reference retrieval allocation is unqualified: {producer}"
            ),
        }
    }
}
impl std::error::Error for RetrievalError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::External(error) => Some(&**error),
            Self::Allocation(error) => Some(error),
            Self::Unqualified(_) => None,
        }
    }
}
impl From<AllocationError> for RetrievalError {
    fn from(error: AllocationError) -> Self {
        Self::Allocation(error)
    }
}

/// Trait for retrieving resources from external sources.
///
/// Implementors of this trait can be used to fetch resources that are not
/// initially present in a [`crate::Registry`].
pub trait Retrieve: RetrieverBounds {
    /// Attempt to retrieve a resource from the given URI.
    ///
    /// # Arguments
    ///
    /// * `uri` - The URI of the resource to retrieve.
    ///
    /// # Errors
    ///
    /// This method can fail for various reasons:
    /// - Resource not found
    /// - Network errors (for remote resources)
    /// - Permission errors
    fn retrieve(
        &self,
        uri: &Uri<String>,
    ) -> Result<Value, Box<dyn std::error::Error + Send + Sync>>;

    /// Retrieves through a prospective allocation producer. The default refuses an
    /// enforced opaque callback before invoking it; ordinary policy retains the same worker.
    /// An implementation accepting enforcement must admit every reached owned allocation
    /// and leave all returned values and failures under the caller's retained owner.
    fn retrieve_with_allocations(
        &self,
        uri: &Uri<String>,
        allocation: &dyn Allocation,
    ) -> Result<Value, RetrievalError> {
        if allocation.is_enforced() {
            return Err(RetrievalError::Unqualified("external retriever"));
        }
        Retrieve::retrieve(self, uri).map_err(RetrievalError::External)
    }
}

#[derive(Debug, Clone)]
struct DefaultRetrieverError;

impl fmt::Display for DefaultRetrieverError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Default retriever does not fetch resources")
    }
}

impl std::error::Error for DefaultRetrieverError {}

/// A retriever that always fails, used as a default when external resource fetching is not needed.
#[derive(Debug, PartialEq, Eq)]
pub struct DefaultRetriever;

impl Retrieve for DefaultRetriever {
    fn retrieve(&self, _: &Uri<String>) -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
        Err(Box::new(DefaultRetrieverError))
    }
    fn retrieve_with_allocations(
        &self,
        uri: &Uri<String>,
        _: &dyn Allocation,
    ) -> Result<Value, RetrievalError> {
        // The existing error is zero-sized, so its box performs no heap allocation.
        Retrieve::retrieve(self, uri).map_err(RetrievalError::External)
    }
}

#[cfg(feature = "retrieve-async")]
#[cfg_attr(target_family = "wasm", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_family = "wasm"), async_trait::async_trait)]
pub trait AsyncRetrieve: RetrieverBounds {
    /// Asynchronously retrieve a resource from the given URI.
    ///
    /// This is the non-blocking equivalent of [`Retrieve::retrieve`].
    ///
    /// # Arguments
    ///
    /// * `uri` - The URI of the resource to retrieve.
    ///
    /// # Errors
    ///
    /// This method can fail for various reasons:
    /// - Resource not found
    /// - Network errors (for remote resources)
    /// - Permission errors
    async fn retrieve(
        &self,
        uri: &Uri<String>,
    ) -> Result<Value, Box<dyn std::error::Error + Send + Sync>>;
}

#[cfg(feature = "retrieve-async")]
#[cfg_attr(target_family = "wasm", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_family = "wasm"), async_trait::async_trait)]
impl AsyncRetrieve for DefaultRetriever {
    async fn retrieve(
        &self,
        _: &Uri<String>,
    ) -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
        Err(Box::new(DefaultRetrieverError))
    }
}
