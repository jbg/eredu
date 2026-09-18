//! Borrowed census of actual source allocation owners, separate from admission totals.
use regex_automata::util::source_storage as storage;
pub use regex_automata::util::source_storage::{Error, Visitor};
macro_rules! source {
    ($type:ty) => {
        impl $type {
            /// Visit actual retained source allocations, deduplicated by the caller.
            /// Used ordinary search pools require a separate mutable census.
            pub fn visit_source_storage(&self, visitor: &mut dyn Visitor) -> Result<(), Error> {
                storage::arc(&self.pattern, 0, visitor)?;
                self.meta.visit_source_storage(visitor)
            }
        }
    };
}
source!(crate::Regex);
source!(crate::bytes::Regex);
macro_rules! set_source {
    ($type:ty) => {
        impl $type {
            /// Visit actual retained source allocations, deduplicated by the caller.
            pub fn visit_source_storage(&self, visitor: &mut dyn Visitor) -> Result<(), Error> {
                if storage::arc(&self.patterns, 0, visitor)? {
                    for pattern in self.patterns.iter() {
                        storage::string(pattern, visitor);
                    }
                }
                self.meta.visit_source_storage(visitor)
            }
        }
    };
}
set_source!(crate::RegexSet);
set_source!(crate::bytes::RegexSet);
