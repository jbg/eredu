//! Borrowed retained-source census, separate from prospective admission totals.
use crate::{Regex, RegexImpl};
use regex_automata::util::source_storage as storage;
pub use storage::{Error, Visitor};

impl Regex {
    /// Visit actual source allocation groups, preserving shared owner identity.
    /// The caller owns deduplication and may return false to skip a seen owner.
    /// Ordinary warmed search pools reject; scoped search workspaces leave this
    /// source census valid and are counted by their invocation owner separately.
    pub fn visit_source_storage(&self, visitor: &mut dyn Visitor) -> Result<(), Error> {
        let mut names = self.named_groups.allocation_size();
        for name in self.named_groups.keys() {
            names = names
                .checked_add(name.capacity())
                .ok_or(Error::SizeOverflow)?;
        }
        storage::arc(&self.named_groups, names, visitor)?;
        match &self.inner {
            RegexImpl::Wrap {
                inner,
                pattern,
                delegated_pattern,
                ..
            } => {
                storage::string(pattern, visitor);
                storage::string(delegated_pattern, visitor);
                inner.visit_source_storage(visitor)?;
            }
            RegexImpl::Fancy { prog, pattern, .. } => {
                storage::string(pattern, visitor);
                if storage::arc(prog, 0, visitor)? {
                    prog.visit_source_storage(visitor)?;
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        allocation::{Allocation, AllocationError, Unenforced},
        RegexOptionsBuilder,
    };
    use core::cell::Cell;
    use hashbrown::HashMap;
    struct Funding(Cell<usize>);
    impl Allocation for Funding {
        fn reserve(&self, bytes: usize) -> Result<(), AllocationError> {
            self.0.set(self.0.get().checked_add(bytes).unwrap());
            Ok(())
        }
    }
    fn census(regex: &Regex, seen: &mut HashMap<usize, usize>) -> Result<usize, Error> {
        let mut total = 0;
        regex.visit_source_storage(&mut |id: *const (), bytes| {
            assert!(bytes > 0);
            if let Some(previous) = seen.insert(id as usize, bytes) {
                assert_eq!(previous, bytes);
                false
            } else {
                total += bytes;
                true
            }
        })?;
        Ok(total)
    }
    #[test]
    fn retained_source_is_stable_across_scoped_searches_and_deduplicates_aliases() {
        for (pattern, text) in [
            ("needle", "a needle"),
            (r"(?=a+)(a+)\1", "aaaa"),
            (r"[α-ω]+\d+", "αβ12"),
            (r"\d+@!\w+", "12@!xyz"),
            #[cfg(feature = "variable-lookbehinds")]
            (r"(?<=(a+))b", "aaab"),
        ] {
            let funding = Funding(Cell::new(0));
            let regex = RegexOptionsBuilder::new()
                .build_with_allocations(pattern, &funding)
                .unwrap();
            let mut before = HashMap::new();
            let retained = census(&regex, &mut before).unwrap();
            assert!(
                retained > 0 && retained <= funding.0.get(),
                "{}: retained={} constructed={}",
                pattern,
                retained,
                funding.0.get()
            );
            assert_eq!(census(&regex, &mut before).unwrap(), 0);
            {
                let mut scoped = regex
                    .search_workspace_with_allocations(&Unenforced)
                    .unwrap();
                assert!(scoped.is_match(text, &Unenforced).unwrap());
            }
            let mut after = HashMap::new();
            assert_eq!(census(&regex, &mut after).unwrap(), retained);
            assert_eq!(before, after);
        }
        let regex = Regex::new(r"(?=a+)(a+)\1").unwrap();
        let clone = regex.clone();
        let mut seen = HashMap::new();
        census(&regex, &mut seen).unwrap();
        assert_eq!(
            census(&clone, &mut seen).unwrap(),
            regex.as_str().len(),
            "VM clone aliases the program and names; only its actual pattern copy is new"
        );
    }
    #[test]
    fn ordinary_used_pool_requires_a_separate_mutable_census() {
        let regex = Regex::new(r"(?=a+)(a+)\1").unwrap();
        assert!(regex.is_match("aaaa").unwrap());
        assert_eq!(
            regex.visit_source_storage(&mut |_, _| true),
            Err(Error::WarmedPool)
        );
    }
}
