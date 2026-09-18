//! Input normalization for resources entering the registry.
use std::sync::Arc;

use serde_json::Value;

use crate::{
    allocation::{Allocation, AllocationError, Allocator},
    Resource, ResourceRef, Retrieve,
};

/// A resource waiting to enter the registry.
pub(crate) enum PendingResource<'a> {
    Value(Value),
    ValueRef(&'a Value),
    SharedValue(Arc<Value>),
    Resource(Resource),
    ResourceRef(ResourceRef<'a>),
}

pub(crate) mod private {
    use super::PendingResource;
    pub(crate) trait Sealed<'a> {
        fn into_pending(self) -> PendingResource<'a>;
    }
}
#[allow(private_bounds)]
pub trait IntoRegistryResource<'a>: private::Sealed<'a> {}
impl<'a, T> IntoRegistryResource<'a> for T where T: private::Sealed<'a> {}

impl<'a> private::Sealed<'a> for Resource {
    fn into_pending(self) -> PendingResource<'a> {
        PendingResource::Resource(self)
    }
}

impl<'a> private::Sealed<'a> for &'a Resource {
    fn into_pending(self) -> PendingResource<'a> {
        PendingResource::ResourceRef(ResourceRef::new(self.contents(), self.draft()))
    }
}

impl<'a> private::Sealed<'a> for &'a Value {
    fn into_pending(self) -> PendingResource<'a> {
        PendingResource::ValueRef(self)
    }
}

impl<'a> private::Sealed<'a> for ResourceRef<'a> {
    fn into_pending(self) -> PendingResource<'a> {
        PendingResource::ResourceRef(self)
    }
}

impl<'a> private::Sealed<'a> for Value {
    fn into_pending(self) -> PendingResource<'a> {
        PendingResource::Value(self)
    }
}

impl<'a> private::Sealed<'a> for Arc<Value> {
    fn into_pending(self) -> PendingResource<'a> {
        PendingResource::SharedValue(self)
    }
}
pub trait IntoRetriever {
    fn into_retriever(
        self,
        allocation: &dyn Allocation,
    ) -> Result<Arc<dyn Retrieve>, AllocationError>;
}

impl<T: Retrieve + 'static> IntoRetriever for T {
    fn into_retriever(
        self,
        allocation: &dyn Allocation,
    ) -> Result<Arc<dyn Retrieve>, AllocationError> {
        Ok(Allocator(allocation).arc(self)?)
    }
}

impl IntoRetriever for Arc<dyn Retrieve> {
    fn into_retriever(self, _: &dyn Allocation) -> Result<Arc<dyn Retrieve>, AllocationError> {
        Ok(self)
    }
}

#[cfg(feature = "retrieve-async")]
pub trait IntoAsyncRetriever {
    fn into_retriever(
        self,
        allocation: &dyn Allocation,
    ) -> Result<Arc<dyn crate::AsyncRetrieve>, AllocationError>;
}

#[cfg(feature = "retrieve-async")]
impl<T: crate::AsyncRetrieve + 'static> IntoAsyncRetriever for T {
    fn into_retriever(
        self,
        allocation: &dyn Allocation,
    ) -> Result<Arc<dyn crate::AsyncRetrieve>, AllocationError> {
        Ok(Allocator(allocation).arc(self)?)
    }
}

#[cfg(feature = "retrieve-async")]
impl IntoAsyncRetriever for Arc<dyn crate::AsyncRetrieve> {
    fn into_retriever(
        self,
        _: &dyn Allocation,
    ) -> Result<Arc<dyn crate::AsyncRetrieve>, AllocationError> {
        Ok(self)
    }
}
