//! Cache ownership for the same delegated search workers.
use super::*;
use crate::allocation::AllocationError;
use regex_automata::{HalfMatch, Match, PatternID};

#[derive(Debug)]
enum Cache {
    Forward(Box<regex_automata::meta::Cache>),
    #[cfg(feature = "variable-lookbehinds")]
    Reverse(Box<regex_automata::hybrid::dfa::Cache>),
}

#[derive(Debug, Default)]
pub(super) struct OwnedCaches {
    rows: hashbrown::HashMap<(usize, bool), Cache>,
}

pub(super) enum Caches<'a> {
    Pooled,
    Scoped(&'a mut OwnedCaches),
    Fixed,
}

pub(crate) fn search_error(error: regex_automata::MatchError) -> Error {
    match error.allocation_error() {
        Some(error) => Error::Allocation(error.into()),
        None => Error::RuntimeError(RuntimeError::DelegateError(error)),
    }
}

impl OwnedCaches {
    fn forward(
        &mut self,
        pc: usize,
        regex: &Regex,
        allocation: Context<'_>,
    ) -> Result<&mut regex_automata::meta::Cache> {
        let key = (pc, false);
        if !self.rows.contains_key(&key) {
            let cache = regex
                .create_cache_with_allocations(&allocation)
                .map_err(search_error)?;
            let cache = allocation.storage.boxed(cache)?;
            allocation.insert(&mut self.rows, key, Cache::Forward(cache))?;
        }
        match self.rows.get_mut(&key).expect("inserted cache") {
            Cache::Forward(cache) => Ok(cache),
            #[cfg(feature = "variable-lookbehinds")]
            _ => unreachable!("cache kind follows instruction and direction"),
        }
    }

    #[cfg(feature = "variable-lookbehinds")]
    fn reverse(
        &mut self,
        pc: usize,
        dfa: &regex_automata::hybrid::dfa::DFA,
        allocation: Context<'_>,
    ) -> Result<&mut regex_automata::hybrid::dfa::Cache> {
        let key = (pc, true);
        if !self.rows.contains_key(&key) {
            let cache = regex_automata::hybrid::dfa::Cache::new_with_allocations(dfa, &allocation)
                .map_err(|error| match error.allocation_error() {
                    Some(error) => Error::Allocation(error.into()),
                    None => Error::RuntimeError(RuntimeError::DelegateError(regex_automata::MatchError::gave_up(0))),
                })?;
            let cache = allocation.storage.boxed(cache)?;
            allocation.insert(&mut self.rows, key, Cache::Reverse(cache))?;
        }
        match self.rows.get_mut(&key).expect("inserted cache") {
            Cache::Reverse(cache) => Ok(cache),
            _ => unreachable!("cache kind follows instruction and direction"),
        }
    }
}

impl Caches<'_> {
    pub(super) fn search(
        &mut self,
        pc: usize,
        regex: &Regex,
        input: &Input<'_>,
        allocation: Context<'_>,
    ) -> Result<Option<Match>> {
        match self {
            Self::Pooled => Ok(regex.search(input)),
            Self::Scoped(caches) => regex
                .search_with_allocations(caches.forward(pc, regex, allocation)?, input, &allocation)
                .map_err(search_error),
            Self::Fixed => Err(Error::Allocation(AllocationError::Refused)),
        }
    }
    pub(super) fn half(
        &mut self,
        pc: usize,
        regex: &Regex,
        input: &Input<'_>,
        allocation: Context<'_>,
    ) -> Result<Option<HalfMatch>> {
        match self {
            Self::Pooled => Ok(regex.search_half(input)),
            Self::Scoped(caches) => regex
                .search_half_with_allocations(
                    caches.forward(pc, regex, allocation)?,
                    input,
                    &allocation,
                )
                .map_err(search_error),
            Self::Fixed => Err(Error::Allocation(AllocationError::Refused)),
        }
    }
    pub(super) fn slots(
        &mut self,
        pc: usize,
        regex: &Regex,
        input: &Input<'_>,
        slots: &mut [Option<NonMaxUsize>],
        allocation: Context<'_>,
    ) -> Result<Option<PatternID>> {
        match self {
            Self::Pooled => Ok(regex.search_slots(input, slots)),
            Self::Scoped(caches) => regex
                .search_slots_with_allocations(
                    caches.forward(pc, regex, allocation)?,
                    input,
                    slots,
                    &allocation,
                )
                .map_err(search_error),
            Self::Fixed => Err(Error::Allocation(AllocationError::Refused)),
        }
    }
    #[cfg(feature = "variable-lookbehinds")]
    pub(super) fn reverse(
        &mut self,
        pc: usize,
        delegate: &ReverseBackwardsDelegate,
        input: &Input<'_>,
        allocation: Context<'_>,
    ) -> Result<Option<HalfMatch>> {
        let result = match self {
            Self::Pooled => delegate
                .dfa
                .try_search_rev(&mut delegate.cache_pool.get(), input),
            Self::Scoped(caches) => delegate.dfa.try_search_rev_with_allocations(
                caches.reverse(pc, &delegate.dfa, allocation)?,
                input,
                &allocation,
            ),
            Self::Fixed => return Err(Error::Allocation(AllocationError::Refused)),
        };
        // The original reverse-lookbehind worker treats intrinsic engine exits
        // as a failed assertion. A funding refusal must leave that worker.
        match result {
            Ok(found) => Ok(found),
            Err(error) if error.allocation_error().is_some() => Err(search_error(error)),
            Err(_) => Ok(None),
        }
    }
}

/// Per-invocation VM storage. It never inserts a borrowed funding loan in a
/// compiled program's persistent pools.
#[derive(Debug)]
pub(crate) struct ScopedScratch {
    scratch: Scratch,
    caches: OwnedCaches,
}
impl ScopedScratch {
    pub(crate) fn new() -> Self {
        Self {
            scratch: new_scratch(),
            caches: OwnedCaches::default(),
        }
    }
    pub(crate) fn run_spans<S: HaystackInput + ?Sized>(
        &mut self,
        prog: &Prog,
        input: &RegexInput<'_, S>,
        option_flags: u32,
        options: &HardRegexRuntimeOptions,
        allocation: Context<'_>,
    ) -> Result<Option<(usize, usize)>> {
        if input.is_done() {
            return Ok(None);
        }
        let Scratch { state, inner_slots } = &mut self.scratch;
        state.reset(prog.n_saves, option_flags, allocation)?;
        inner_slots.clear();
        run_inner(
            prog,
            input,
            option_flags,
            options,
            state,
            inner_slots,
            &mut [],
            &mut Caches::Scoped(&mut self.caches),
            allocation,
            |state| Ok((state.get(0), state.get(1))),
        )
    }
}
