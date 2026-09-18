#[cfg(feature = "alloc")]
use crate::util::allocation::{Allocation, AllocationError};
use crate::util::{
    prefilter::PrefilterI,
    search::{MatchKind, Span},
};

#[derive(Clone, Debug)]
pub(crate) struct Memmem {
    #[cfg(not(all(feature = "std", feature = "perf-literal-substring")))]
    _unused: (),
    #[cfg(all(feature = "std", feature = "perf-literal-substring"))]
    finder: memchr::memmem::Finder<'static>,
}

impl Memmem {
    #[cfg(feature = "alloc")]
    pub(crate) fn new_with_allocations<B: AsRef<[u8]>>(
        _kind: MatchKind,
        needles: &[B],
        funding: &dyn Allocation,
    ) -> Result<Option<Memmem>, AllocationError> {
        #[cfg(not(all(feature = "std", feature = "perf-literal-substring")))]
        {
            Ok(None)
        }
        #[cfg(all(feature = "std", feature = "perf-literal-substring"))]
        {
            if needles.len() != 1 {
                return Ok(None);
            }
            let needle = needles[0].as_ref();
            let finder = memchr::memmem::Finder::new(needle)
                .into_owned_with_allocations(&super::funding::Funding(funding))
                .map_err(super::funding::memchr)?;
            Ok(Some(Memmem { finder }))
        }
    }
}

impl PrefilterI for Memmem {
    #[cfg(feature = "alloc")]
    fn visit_source_storage(
        &self,
        visitor: &mut dyn crate::util::source_storage::Visitor,
    ) -> Result<(), crate::util::source_storage::Error> {
        #[cfg(all(feature = "std", feature = "perf-literal-substring"))]
        self.finder
            .visit_source_storage(&mut |id, bytes| visitor.visit(id, bytes));
        Ok(())
    }

    fn name(&self) -> &'static str {
        "memmem"
    }

    fn find(&self, haystack: &[u8], span: Span) -> Option<Span> {
        #[cfg(not(all(feature = "std", feature = "perf-literal-substring")))]
        {
            unreachable!()
        }
        #[cfg(all(feature = "std", feature = "perf-literal-substring"))]
        {
            self.finder.find(&haystack[span]).map(|i| {
                let start = span.start + i;
                let end = start + self.finder.needle().len();
                Span { start, end }
            })
        }
    }

    fn prefix(&self, haystack: &[u8], span: Span) -> Option<Span> {
        #[cfg(not(all(feature = "std", feature = "perf-literal-substring")))]
        {
            unreachable!()
        }
        #[cfg(all(feature = "std", feature = "perf-literal-substring"))]
        {
            let needle = self.finder.needle();
            if haystack[span].starts_with(needle) {
                Some(Span {
                    end: span.start + needle.len(),
                    ..span
                })
            } else {
                None
            }
        }
    }

    fn memory_usage(&self) -> usize {
        #[cfg(not(all(feature = "std", feature = "perf-literal-substring")))]
        {
            unreachable!()
        }
        #[cfg(all(feature = "std", feature = "perf-literal-substring"))]
        {
            self.finder.needle().len()
        }
    }

    fn is_fast(&self) -> bool {
        #[cfg(not(all(feature = "std", feature = "perf-literal-substring")))]
        {
            unreachable!()
        }
        #[cfg(all(feature = "std", feature = "perf-literal-substring"))]
        {
            true
        }
    }
}
