//! Exact trie-walk destinations. DFA mutation has a separate storage contract.
use super::{LexemeSet, Lexer, LexerResult, StateID};
use crate::toktrie::{
    Recognizer, SimpleVob, TokTrie, TokenMaskConstructionFailure, TokenMaskConstructionPlan,
    TokenMaskSourceError,
};
use std::{
    alloc::Layout,
    collections::TryReserveError,
    convert::Infallible,
    fmt,
    mem::{size_of, size_of_val},
};

#[derive(Debug)]
enum Cause {
    Overflow,
    Capacity,
    WalkCapacity,
    Allocation(TryReserveError),
    MaskSource(TokenMaskSourceError),
    Mask(TokenMaskConstructionFailure),
    Lexer(anyhow::Error),
}
/// Owns every completed local destination on failure. The enclosing original
/// owner must retain its source/account; this is not a DFA admission contract.
#[derive(Debug)]
pub struct LexerPrecomputeFailure {
    cause: Cause,
    states: Vec<StateID>,
    mask: Option<SimpleVob>,
}
impl fmt::Display for LexerPrecomputeFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.cause {
            Cause::Overflow => f.write_str("lexer precompute trie geometry overflow"),
            Cause::Capacity => f.write_str("lexer precompute destination exceeded its extent"),
            Cause::WalkCapacity => f.write_str("lexer precompute walk exceeded its source depth"),
            Cause::Allocation(e) => fmt::Display::fmt(e, f),
            Cause::MaskSource(e) => fmt::Display::fmt(e, f),
            Cause::Mask(e) => fmt::Display::fmt(e, f),
            Cause::Lexer(e) => fmt::Display::fmt(e, f),
        }
    }
}
impl std::error::Error for LexerPrecomputeFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.cause {
            Cause::Allocation(e) => Some(e),
            Cause::MaskSource(e) => Some(e),
            Cause::Mask(e) => Some(e),
            Cause::Lexer(e) => Some(e.as_ref()),
            _ => None,
        }
    }
}
impl LexerPrecomputeFailure {
    // Ordinary callers have no original account loan: tear down local buffers,
    // then return the same original typed DFA error without changing downcasts.
    pub(super) fn into_ordinary(self) -> anyhow::Error {
        let Self {
            cause,
            states,
            mask,
        } = self;
        drop(states);
        drop(mask);
        match cause {
            Cause::Lexer(e) => e,
            cause => anyhow::Error::new(Self {
                cause,
                states: Vec::new(),
                mask: None,
            }),
        }
    }
}
fn rejected(cause: Cause) -> LexerPrecomputeFailure {
    LexerPrecomputeFailure {
        cause,
        states: Vec::new(),
        mask: None,
    }
}
/// Source-derived state-stack and token-mask storage for one lexer trie walk.
#[derive(Debug, Clone, Copy)]
pub struct LexerPrecomputeRequirements {
    states: usize,
    buffers: usize,
    controls: usize,
    total: usize,
}
impl LexerPrecomputeRequirements {
    /// One initial state plus the actual trie's maximum token depth.
    pub fn state_capacity(&self) -> usize {
        self.states
    }
    /// Exact state stack and actual trie-walker mask payload.
    pub fn buffer_bytes(&self) -> usize {
        self.buffers
    }
    /// Fixed constructor, walker and failure representations.
    pub fn control_bytes(&self) -> usize {
        self.controls
    }
    /// Complete local quote; DFA construction/mutation is not included.
    pub fn required_bytes(&self) -> usize {
        self.total
    }
}
/// Closed actual trie loan; no caller capacity can replace source geometry.
pub struct LexerPrecomputePlan<'a> {
    trie: &'a TokTrie,
    mask: TokenMaskConstructionPlan<'a>,
    requirements: LexerPrecomputeRequirements,
}
impl fmt::Debug for LexerPrecomputePlan<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LexerPrecomputePlan")
            .field("requirements", &self.requirements)
            .finish()
    }
}
impl<'a> LexerPrecomputePlan<'a> {
    /// Quotes the same state stack and mask consumed by ordinary precomputation.
    pub fn prepare(trie: &'a TokTrie) -> Result<Self, LexerPrecomputeFailure> {
        Self::inspect(trie).map_err(rejected)
    }
    fn inspect(trie: &'a TokTrie) -> Result<Self, Cause> {
        let states = trie.max_token_len().checked_add(1).ok_or(Cause::Overflow)?;
        let bytes = Layout::array::<StateID>(states)
            .map_err(|_| Cause::Overflow)?
            .size();
        let mask = TokenMaskConstructionPlan::for_trie(trie).map_err(Cause::MaskSource)?;
        let buffers = bytes
            .checked_add(mask.requirements().buffer_bytes())
            .ok_or(Cause::Overflow)?;
        let controls = Self::inspection_control_bytes().ok_or(Cause::Overflow)?;
        let total = buffers.checked_add(controls).ok_or(Cause::Overflow)?;
        Ok(Self {
            trie,
            mask,
            requirements: LexerPrecomputeRequirements {
                states,
                buffers,
                controls,
                total,
            },
        })
    }
    /// Fixed source/constructor frames, reserved before preparing the actual
    /// trie geometry. Generic callback execution is separately quoted.
    pub fn inspection_control_bytes() -> Option<usize> {
        let parts = [
            TokenMaskConstructionPlan::inspection_control_bytes()?,
            size_of::<Self>(),
            size_of::<LexerPrecomputeRequirements>(),
            size_of::<LexerPrecompute<'_>>(),
            size_of::<LexerPrecomputeFailure>(),
            size_of::<Cause>(),
            size_of::<Walker<'_, fn(StateID, u8) -> Result<LexerResult, Infallible>, Infallible>>(),
            size_of::<StateID>(),
            size_of::<LexerResult>(),
            size_of::<Vec<StateID>>(),
            size_of::<Option<SimpleVob>>(),
            size_of::<Result<Self, Cause>>(),
            size_of::<Result<Self, LexerPrecomputeFailure>>(),
            size_of::<Result<LexerPrecompute<'_>, LexerPrecomputeFailure>>(),
            size_of::<Result<(), anyhow::Error>>(),
            size_of::<Result<(), TryReserveError>>(),
            size_of::<Result<Layout, std::alloc::LayoutError>>(),
            size_of::<(&mut Lexer, &LexemeSet)>(),
            size_of::<(
                &TokTrie,
                &mut Walker<'_, fn(StateID, u8) -> Result<LexerResult, Infallible>, Infallible>,
                &mut SimpleVob,
            )>(),
            size_of::<(usize, StateID, bool)>(),
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
    /// Immutable complete local quote for this same source loan.
    pub fn requirements(&self) -> LexerPrecomputeRequirements {
        self.requirements
    }
    /// Performs one mask constructor and one fallible state-stack reservation.
    pub fn compile(self) -> Result<LexerPrecompute<'a>, LexerPrecomputeFailure> {
        let mask = self.mask.compile().map_err(|e| rejected(Cause::Mask(e)))?;
        let mut states = Vec::new();
        if let Err(e) = states.try_reserve_exact(self.requirements.states) {
            return Err(LexerPrecomputeFailure {
                cause: Cause::Allocation(e),
                states,
                mask: Some(mask),
            });
        }
        if states.capacity() > self.requirements.states {
            return Err(LexerPrecomputeFailure {
                cause: Cause::Capacity,
                states,
                mask: Some(mask),
            });
        }
        Ok(LexerPrecompute {
            trie: self.trie,
            states,
            mask,
        })
    }
}
/// Prepared local storage retaining its exact immutable trie loan.
pub struct LexerPrecompute<'a> {
    trie: &'a TokTrie,
    states: Vec<StateID>,
    mask: SimpleVob,
}
impl fmt::Debug for LexerPrecompute<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LexerPrecompute")
            .field("states", &self.states.capacity())
            .finish_non_exhaustive()
    }
}
struct Walker<'a, F, E> {
    states: &'a mut Vec<StateID>,
    advance: &'a mut F,
    exceeded: bool,
    error: Option<E>,
}
impl<F, E> Recognizer for Walker<'_, F, E>
where
    F: FnMut(StateID, u8) -> Result<LexerResult, E>,
{
    fn collapse(&mut self) {}
    fn trie_finished(&mut self) {}
    fn pop_bytes(&mut self, num: usize) {
        self.states.truncate(self.states.len() - num);
    }
    fn try_push_byte(&mut self, byte: u8) -> bool {
        if self.error.is_some() || self.exceeded {
            return false;
        }
        let state = *self.states.last().expect("initial precompute state");
        match (self.advance)(state, byte) {
            Ok(LexerResult::State(next, _)) => {
                if self.states.len() == self.states.capacity() {
                    self.exceeded = true;
                    false
                } else {
                    self.states.push(next);
                    true
                }
            }
            Ok(_) => false,
            Err(error) => {
                self.error = Some(error);
                false
            }
        }
    }
}
#[derive(Debug)]
enum RunCause<E> {
    Capacity,
    Worker(E),
}
/// Owns completed local trie-walk buffers together with the exact typed worker
/// failure. No borrowed trie or erased ordinary error escapes this value.
#[derive(Debug)]
pub struct LexerPrecomputeRunFailure<E> {
    cause: RunCause<E>,
    states: Vec<StateID>,
    mask: SimpleVob,
}
impl<E: fmt::Display> fmt::Display for LexerPrecomputeRunFailure<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.cause {
            RunCause::Capacity => f.write_str("lexer precompute walk exceeded its source depth"),
            RunCause::Worker(e) => fmt::Display::fmt(e, f),
        }
    }
}
impl<E: std::error::Error + 'static> std::error::Error for LexerPrecomputeRunFailure<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.cause {
            RunCause::Worker(e) => Some(e),
            _ => None,
        }
    }
}
impl<'a> LexerPrecompute<'a> {
    fn fail(self, cause: Cause) -> LexerPrecomputeFailure {
        LexerPrecomputeFailure {
            cause,
            states: self.states,
            mask: Some(self.mask),
        }
    }
    /// Uses the ordinary start-state and trie walker. The supplied DFA must have
    /// its own construction/mutation authority; this local quote does not grant it.
    pub fn run(
        self,
        lexer: &mut Lexer,
        allowed: &LexemeSet,
    ) -> Result<Self, LexerPrecomputeFailure> {
        let state = lexer.start_state(allowed);
        if let Err(cause) = lexer.check_error() {
            return Err(self.fail(Cause::Lexer(cause)));
        }
        self.run_from_state(lexer, state)
    }
    pub(super) fn run_from_state(
        self,
        lexer: &mut Lexer,
        state: StateID,
    ) -> Result<Self, LexerPrecomputeFailure> {
        let storage = self
            .run_with(state, |state, byte| {
                Ok::<_, Infallible>(lexer.advance(state, byte, false))
            })
            .map_err(|error| {
                let cause = match error.cause {
                    RunCause::Capacity => Cause::WalkCapacity,
                    RunCause::Worker(never) => match never {},
                };
                LexerPrecomputeFailure {
                    cause,
                    states: error.states,
                    mask: Some(error.mask),
                }
            })?;
        if let Err(cause) = lexer.check_error() {
            return Err(storage.fail(Cause::Lexer(cause)));
        }
        Ok(storage)
    }
    /// Actual callback, result and recognizer frames for the shared trie walk.
    /// This is storage accounting, not authority to run the supplied worker.
    pub fn execution_control_bytes<F, E>(_: &F) -> Option<usize>
    where
        F: FnMut(StateID, u8) -> Result<LexerResult, E>,
    {
        let parts = [
            size_of::<F>(),
            size_of::<Walker<'_, F, E>>(),
            size_of::<Option<E>>(),
            size_of::<LexerPrecomputeRunFailure<E>>(),
            size_of::<RunCause<E>>(),
            size_of::<Option<RunCause<E>>>(),
            size_of::<Result<LexerResult, E>>(),
            size_of::<Result<Self, LexerPrecomputeRunFailure<E>>>(),
            size_of::<(&TokTrie, &mut Walker<'_, F, E>, &mut SimpleVob)>(),
            size_of::<(StateID, StateID, u8, bool)>(),
            size_of::<LexerResult>(),
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
    /// Runs the same ordinary TokTrie traversal with a separately admitted
    /// lexical worker. The first typed failure stops further worker calls while
    /// the trie restores its stack; every reached local destination is retained.
    pub fn run_with<F, E>(
        mut self,
        state: StateID,
        mut advance: F,
    ) -> Result<Self, LexerPrecomputeRunFailure<E>>
    where
        F: FnMut(StateID, u8) -> Result<LexerResult, E>,
    {
        self.states.clear();
        self.states.push(state);
        self.mask.set_all(false);
        let cause = {
            let mut walker = Walker {
                states: &mut self.states,
                advance: &mut advance,
                exceeded: false,
                error: None,
            };
            self.trie.add_bias(&mut walker, &mut self.mask, &[]);
            if let Some(error) = walker.error {
                Some(RunCause::Worker(error))
            } else if walker.exceeded {
                Some(RunCause::Capacity)
            } else {
                None
            }
        };
        match cause {
            Some(cause) => Err(LexerPrecomputeRunFailure {
                cause,
                states: self.states,
                mask: self.mask,
            }),
            None => Ok(self),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        api::ParserLimits,
        earley::{lexerspec::LexerSpec, ParserError},
        toktrie::TokRxInfo,
    };
    use derivre::RegexAst;

    #[test]
    fn actual_long_trie_precompute_reuses_exact_stack_and_keeps_failed_dfa_cause() {
        let word = format!("{}!", "a".repeat(640));
        let tokens = vec![
            word.as_bytes().to_vec(),
            b"a".to_vec(),
            b"b".to_vec(),
            Vec::new(),
        ];
        let trie = TokTrie::from(&TokRxInfo::new(tokens.len() as u32, 3), &tokens);
        let mut spec = LexerSpec::new().unwrap();
        spec.setup_lexeme_class(RegexAst::NoMatch).unwrap();
        spec.add_simple_literal("long".into(), &word, false)
            .unwrap();
        let mut limits = ParserLimits {
            max_lexer_states: 4096,
            precompute_large_lexemes: false,
            ..ParserLimits::default()
        };
        let baseline = Lexer::from(&spec, &mut limits, false).unwrap();
        let allowed = spec.all_lexemes();
        let mut ordinary = baseline.clone();
        let mut prepared = baseline.clone();
        ordinary.precompute_for(&trie, &allowed).unwrap();
        let plan = LexerPrecomputePlan::prepare(&trie).unwrap();
        assert_eq!(plan.requirements().state_capacity(), word.len() + 1);
        assert!(plan.requirements().required_bytes() > plan.requirements().buffer_bytes());
        let storage = plan.compile().unwrap();
        let states_ptr = storage.states.as_ptr();
        let mask_ptr = storage.mask.as_slice().as_ptr();
        let storage = storage.run(&mut prepared, &allowed).unwrap();
        assert_eq!(storage.states.capacity(), word.len() + 1);
        assert_eq!(storage.states.as_ptr(), states_ptr);
        assert_eq!(storage.mask.as_slice().as_ptr(), mask_ptr);
        assert!(prepared.dfa.stats().num_states > 300);
        assert_eq!(
            prepared.dfa.stats().num_states,
            ordinary.dfa.stats().num_states
        );
        assert_eq!(
            prepared.dfa.stats().num_transitions,
            ordinary.dfa.stats().num_transitions
        );
        let storage = storage.run(&mut prepared, &allowed).unwrap();
        assert_eq!(storage.states.as_ptr(), states_ptr);
        assert_eq!(storage.mask.as_slice().as_ptr(), mask_ptr);
        drop(storage);

        let mut ordinary_failed = baseline.clone();
        let mut prepared_failed = baseline;
        let cap = prepared_failed.dfa.stats().num_states;
        ordinary_failed.dfa.set_max_states(cap);
        prepared_failed.dfa.set_max_states(cap);
        let expected = ordinary_failed.precompute_for(&trie, &allowed).unwrap_err();
        let actual = LexerPrecomputePlan::prepare(&trie)
            .unwrap()
            .compile()
            .unwrap()
            .run(&mut prepared_failed, &allowed)
            .unwrap_err();
        let Cause::Lexer(cause) = &actual.cause else {
            panic!("wrong failure {actual:?}");
        };
        assert_eq!(
            cause.downcast_ref::<ParserError>().unwrap().message(),
            expected.downcast_ref::<ParserError>().unwrap().message()
        );
        assert_eq!(actual.states.capacity(), word.len() + 1);
        assert!(!actual.mask.as_ref().unwrap().as_slice().is_empty());
        let calls = std::cell::Cell::new(0usize);
        let callback_failure = LexerPrecomputePlan::prepare(&trie)
            .unwrap()
            .compile()
            .unwrap()
            .run_with(StateID::DEAD, |_state, byte| {
                calls.set(calls.get() + 1);
                if calls.get() == 3 {
                    Err("actual callback failure")
                } else {
                    Ok(LexerResult::State(StateID::DEAD, byte))
                }
            })
            .unwrap_err();
        assert_eq!(calls.get(), 3);
        assert_eq!(callback_failure.to_string(), "actual callback failure");
        assert_eq!(callback_failure.states.capacity(), word.len() + 1);
        assert!(!callback_failure.mask.as_slice().is_empty());
        drop((
            prepared,
            ordinary,
            prepared_failed,
            ordinary_failed,
            spec,
            trie,
            tokens,
        ));
        // The real failed walk destinations remain owned after all inputs retire.
        assert!(!actual.states.is_empty());
        assert!(!callback_failure.states.is_empty());
        drop((actual, callback_failure));
    }
}
