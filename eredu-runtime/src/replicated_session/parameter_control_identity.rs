//! Retained session origin and its checked parameter-publication generation.
use std::sync::Arc;

#[derive(Clone, Debug)]
pub(crate) struct ParameterControlIdentity {
    owner: Arc<()>,
    pub(crate) generation: u64,
}
impl ParameterControlIdentity {
    pub(crate) fn new() -> Self {
        Self {
            owner: Arc::new(()),
            generation: 0,
        }
    }
    pub(crate) fn matches(&self, other: &Self) -> bool {
        self.same_owner(other) && self.generation == other.generation
    }
    pub(crate) fn same_owner(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.owner, &other.owner)
    }
    pub(crate) fn visit<T>(
        &mut self,
        visitor: &mut dyn crate::parameter_operations::ParameterPublication<T>,
    ) {
        let generation = self.generation;
        visitor.counter(&mut self.generation, &mut |_, _| {
            generation
                .checked_add(1)
                .ok_or_else(|| eredu_nn::workspace::WorkspaceMetadataError::Overflow.into())
        });
    }
}
