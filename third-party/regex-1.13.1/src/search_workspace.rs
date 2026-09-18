use crate::{allocation::Allocation, Regex};
use alloc::sync::Arc;
use regex_automata::{meta, Input, MatchError};

/// Invocation-owned search cache retaining the exact original source alias.
/// The caller keeps invocation funding alive until this cache is dropped.
#[derive(Debug)]
pub struct OwnedSearchWorkspace {
    cache: meta::Cache,
    regex: Arc<Regex>,
}
impl OwnedSearchWorkspace {
    /// Create source-bound storage through the original meta-engine cache worker.
    pub fn new_with_allocations(
        regex: Arc<Regex>,
        funding: &dyn Allocation,
    ) -> Result<Self, MatchError> {
        let cache = regex.meta.create_cache_with_allocations(funding)?;
        Ok(Self { cache, regex })
    }
    /// Borrow the exact retained source owner.
    pub fn source(&self) -> &Arc<Regex> {
        &self.regex
    }
    /// Test exact source identity, independently of pattern text equality.
    pub fn matches_source(&self, source: &Arc<Regex>) -> bool {
        Arc::ptr_eq(&self.regex, source)
    }
    /// Test for a match with prospective funding for any reached cache growth.
    pub fn is_match(&mut self, text: &str, funding: &dyn Allocation) -> Result<bool, MatchError> {
        self.regex
            .meta
            .search_half_with_allocations(
                &mut self.cache,
                &Input::new(text).earliest(true),
                funding,
            )
            .map(|found| found.is_some())
    }
}
