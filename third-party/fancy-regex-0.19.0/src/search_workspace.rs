//! Caller-owned storage for the original selected regex and VM workers.
use crate::{
    allocation::{Allocation, Context},
    input::Input,
    Regex, RegexImpl, RegexInput, Result,
};
use alloc::sync::Arc;

/// Search storage bound to one compiled regex. Keep its funding owner alive
/// until this workspace is dropped. Each operation borrows a policy; neither
/// the workspace nor the regex stores that borrowed callback.
#[derive(Debug)]
pub struct SearchWorkspace<'r> {
    regex: &'r Regex,
    storage: Storage,
}
#[derive(Debug)]
pub(crate) enum Storage {
    Wrap(regex_automata::meta::Cache),
    Fancy(crate::vm::ScopedScratch),
}

impl Regex {
    /// Construct scoped search storage without entering persistent cache pools.
    /// This uses the same selected engines and search workers as ordinary calls.
    pub fn search_workspace_with_allocations(
        &self,
        funding: &dyn Allocation,
    ) -> Result<SearchWorkspace<'_>> {
        let storage = Storage::new(self, Context::new(funding))?;
        Ok(SearchWorkspace {
            regex: self,
            storage,
        })
    }
}

impl Storage {
    fn new(regex: &Regex, allocation: Context<'_>) -> Result<Self> {
        Ok(match &regex.inner {
            RegexImpl::Wrap { inner, .. } => Self::Wrap(
                inner
                    .create_cache_with_allocations(&allocation)
                    .map_err(crate::vm::search_error)?,
            ),
            RegexImpl::Fancy { .. } => Self::Fancy(crate::vm::ScopedScratch::new()),
        })
    }
}

/// Invocation storage that retains an alias of the exact compiled source.
/// No pattern or automaton is cloned. This permits a heterogeneous validation
/// invocation to keep regex caches without borrowing transient callback inputs.
#[derive(Debug)]
pub struct OwnedSearchWorkspace {
    regex: Arc<Regex>,
    storage: Storage,
}
impl OwnedSearchWorkspace {
    /// Bind caller-owned search storage to the supplied existing source owner.
    /// The caller keeps source and invocation funding alive until their storage
    /// retires; no borrowed funding callback is retained here.
    pub fn new_with_allocations(regex: Arc<Regex>, funding: &dyn Allocation) -> Result<Self> {
        let storage = Storage::new(&regex, Context::new(funding))?;
        Ok(Self { regex, storage })
    }
    /// Borrow the actual retained source owner for invocation cache lookup.
    pub fn source(&self) -> &Arc<Regex> {
        &self.regex
    }
    /// Authenticate an exact shared source, independently of pattern equality.
    pub fn matches_source(&self, source: &Arc<Regex>) -> bool {
        Arc::ptr_eq(&self.regex, source)
    }
    /// Test for a match using this invocation's reusable storage.
    pub fn is_match<S: Input + ?Sized>(
        &mut self,
        input: &S,
        funding: &dyn Allocation,
    ) -> Result<bool> {
        self.is_match_input(RegexInput::new(input), funding)
    }
    /// Test the supplied range and assertion configuration.
    pub fn is_match_input<S: Input + ?Sized>(
        &mut self,
        input: RegexInput<'_, S>,
        funding: &dyn Allocation,
    ) -> Result<bool> {
        is_match(
            &self.regex,
            Some(&mut self.storage),
            &input,
            Context::new(funding),
        )
    }
    /// Find the first match in the full original input.
    pub fn find<'t, S: Input + ?Sized>(
        &mut self,
        input: &'t S,
        funding: &dyn Allocation,
    ) -> Result<Option<S::Match<'t>>> {
        self.find_input(RegexInput::new(input), funding)
    }
    /// Find a match while retaining the surrounding assertion context.
    pub fn find_input<'t, S: Input + ?Sized>(
        &mut self,
        input: RegexInput<'t, S>,
        funding: &dyn Allocation,
    ) -> Result<Option<S::Match<'t>>> {
        Ok(find_spans(
            &self.regex,
            Some(&mut self.storage),
            &input,
            0,
            Context::new(funding),
        )?
        .map(|(start, end)| input.haystack().make_match(start, end)))
    }
}
impl SearchWorkspace<'_> {
    /// Test for a match, funding any reached cache or VM growth.
    pub fn is_match<S: Input + ?Sized>(
        &mut self,
        input: &S,
        funding: &dyn Allocation,
    ) -> Result<bool> {
        self.is_match_input(RegexInput::new(input), funding)
    }
    /// Test the supplied range and assertion configuration.
    pub fn is_match_input<S: Input + ?Sized>(
        &mut self,
        input: RegexInput<'_, S>,
        funding: &dyn Allocation,
    ) -> Result<bool> {
        is_match(
            self.regex,
            Some(&mut self.storage),
            &input,
            Context::new(funding),
        )
    }
    /// Find the first match with absolute offsets into the supplied input.
    pub fn find<'t, S: Input + ?Sized>(
        &mut self,
        input: &'t S,
        funding: &dyn Allocation,
    ) -> Result<Option<S::Match<'t>>> {
        self.find_input(RegexInput::new(input), funding)
    }
    /// Search a bounded range without slicing away surrounding context.
    pub fn find_input<'t, S: Input + ?Sized>(
        &mut self,
        input: RegexInput<'t, S>,
        funding: &dyn Allocation,
    ) -> Result<Option<S::Match<'t>>> {
        Ok(self
            .find_spans(&input, funding)?
            .map(|(start, end)| input.haystack().make_match(start, end)))
    }
    fn find_spans<S: Input + ?Sized>(
        &mut self,
        input: &RegexInput<'_, S>,
        funding: &dyn Allocation,
    ) -> Result<Option<(usize, usize)>> {
        find_spans(
            self.regex,
            Some(&mut self.storage),
            input,
            0,
            Context::new(funding),
        )
    }
}

pub(crate) fn is_match<S: Input + ?Sized>(
    regex: &Regex,
    storage: Option<&mut Storage>,
    input: &RegexInput<'_, S>,
    allocation: Context<'_>,
) -> Result<bool> {
    if input.is_done() {
        return Ok(false);
    }
    if let RegexImpl::Wrap { inner, .. } = &regex.inner {
        return match storage {
            Some(Storage::Wrap(cache)) => inner
                .is_match_with_allocations(cache, &crate::ra_input(input), &allocation)
                .map_err(crate::vm::search_error),
            None => Ok(inner.is_match(crate::ra_input(input))),
            _ => unreachable!("workspace source kind"),
        };
    }
    find_spans(regex, storage, input, 0, allocation).map(|found| found.is_some())
}

pub(crate) fn find_spans<S: Input + ?Sized>(
    regex: &Regex,
    storage: Option<&mut Storage>,
    input: &RegexInput<'_, S>,
    option_flags: u32,
    allocation: Context<'_>,
) -> Result<Option<(usize, usize)>> {
    if input.is_done() {
        return Ok(None);
    }
    match &regex.inner {
        RegexImpl::Wrap {
            inner,
            explicit_capture_group_0,
            ..
        } => {
            let mut delegated_input = crate::ra_input(input);
            if input.is_anchored() {
                delegated_input = delegated_input.anchored(regex_automata::Anchored::Yes);
            }
            if !*explicit_capture_group_0 {
                let found = match storage {
                    Some(Storage::Wrap(cache)) => inner
                        .search_with_allocations(cache, &delegated_input, &allocation)
                        .map_err(crate::vm::search_error)?,
                    None => inner.search(&delegated_input),
                    _ => unreachable!("workspace source kind"),
                };
                return Ok(found.map(|m| (m.start(), m.end())));
            }
            let mut slots = [None; 4];
            let found = match storage {
                Some(Storage::Wrap(cache)) => inner
                    .search_slots_with_allocations(cache, &delegated_input, &mut slots, &allocation)
                    .map_err(crate::vm::search_error)?,
                None => inner.search_slots(&delegated_input, &mut slots),
                _ => unreachable!("workspace source kind"),
            };
            if found.is_some() {
                Ok(slots[2]
                    .zip(slots[3])
                    .map(|(start, end)| (start.get(), end.get())))
            } else {
                Ok(None)
            }
        }
        RegexImpl::Fancy { prog, options, .. } => {
            let flags = option_flags
                | if options.find_not_empty {
                    crate::vm::OPTION_FIND_NOT_EMPTY
                } else {
                    0
                };
            match storage {
                Some(Storage::Fancy(scratch)) => {
                    scratch.run_spans(prog, input, flags, options, allocation)
                }
                None => crate::vm::run_spans(prog, input, flags, options),
                _ => unreachable!("workspace source kind"),
            }
        }
    }
}

#[cfg(test)]
mod tests;
