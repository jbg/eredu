//! Nonescaping handoff of an actual prepared addressable invocation.
use crate::workspace::HostMetadataFunding;
use std::any::Any;

/// An outer producer lends its actual move-only invocation to the existing
/// indexed movement owner. The native receiver authenticates its concrete type,
/// source identity, stream and funding before taking it. This is no byte grant.
pub struct PreparedIndexedInvocationLoan<'a> {
    source: &'a mut dyn Any,
    funding: &'a HostMetadataFunding,
}
impl<'a> PreparedIndexedInvocationLoan<'a> {
    pub fn new(source: &'a mut dyn Any, funding: &'a HostMetadataFunding) -> Self {
        Self { source, funding }
    }
    pub fn source_mut(&mut self) -> &mut dyn Any { self.source }
    pub fn funding(&self) -> &'a HostMetadataFunding { self.funding }
}
