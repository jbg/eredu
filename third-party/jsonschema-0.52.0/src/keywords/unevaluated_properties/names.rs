//! Evaluated property names in the shared ordinary validation traversal.
use crate::validator::{cache::Cache, ValidationContext};
use std::{
    borrow::Cow,
    mem::{size_of, size_of_val},
};

pub(super) struct Names<'i> {
    rows: Cache<Cow<'i, str>, ()>,
}
impl<'i> Names<'i> {
    pub(super) fn new(capacity: usize, context: &mut ValidationContext<'_>) -> Self {
        let parts = [
            size_of::<Self>(),
            size_of::<usize>(),
            size_of::<(&mut ValidationContext<'_>, usize)>(),
            size_of::<ahash::RandomState>(),
        ];
        let mut names = Self {
            rows: Cache::new(context.workspace.hasher()),
        };
        if context.workspace.reserve(
            parts
                .into_iter()
                .try_fold(size_of_val(&parts), usize::checked_add),
        ) {
            names.rows.reserve(capacity, &mut context.workspace);
        }
        names
    }
    pub(super) fn insert(
        &mut self,
        name: Cow<'i, str>,
        context: &mut ValidationContext<'_>,
    ) -> bool {
        self.rows.insert(name, (), &mut context.workspace)
    }
    pub(super) fn contains(&self, name: &str) -> bool {
        self.rows.get(name).is_some()
    }
    pub(super) fn len(&self) -> usize {
        self.rows.len()
    }
}
