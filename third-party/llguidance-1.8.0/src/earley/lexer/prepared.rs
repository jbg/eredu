//! Paid lexer warm-up and shared lexical decisions over the exact owning DFA.
mod copy;
pub use copy::PreparedLexerCopyFailure;
use super::{policy, LexemeSet, LexerResult, NextByte, PreLexeme, StateDesc, StateID};
use crate::earley::{
    lexerspec::LexemeIdx,
    regexvec::prepared::{PreparedRegexVector, PreparedRegexVectorOperationError},
};
use std::{
    fmt,
    mem::{size_of, size_of_val},
};
use toktrie::{
    SimpleVob, TokenMaskConstructionFailure, TokenMaskConstructionPlan, TokenMaskSourceError,
};

#[derive(Debug)]
enum Cause<E> {
    Overflow,
    Source,
    Failed,
    Funding(E),
    Vector(PreparedRegexVectorOperationError<E>),
    MaskSource(TokenMaskSourceError),
    Mask(TokenMaskConstructionFailure),
    Precompute(super::precompute::LexerPrecomputeFailure),
    Walk(super::precompute::LexerPrecomputeRunFailure<PreparedRegexVectorOperationError<E>>),
}
/// A failed warm-up retains all reached masks and the actual vector. The
/// enclosing owner retains source/H. Mutable operations use a separate error.
pub struct PreparedLexerError<E> {
    cause: Cause<E>,
    prefix: Option<PreparedLexer>,
}
impl<E> PreparedLexerError<E> {
    /// Actual vector retained after a partially completed lexer constructor.
    pub fn retained_vector(&self) -> Option<&PreparedRegexVector> {
        self.prefix.as_ref().map(|owner| &owner.vector)
    }
}
impl<E: fmt::Debug> fmt::Debug for PreparedLexerError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PreparedLexerError")
            .field("cause", &self.cause)
            .field("retains_prefix", &self.prefix.is_some())
            .finish()
    }
}
impl<E: fmt::Display> fmt::Display for Cause<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Cause::Overflow => f.write_str("prepared lexer control geometry overflow"),
            Cause::Source => f.write_str("prepared lexer state is unavailable"),
            Cause::Failed => f.write_str("prepared lexer retains a failed operation"),
            Cause::Funding(e) => fmt::Display::fmt(e, f),
            Cause::Vector(e) => fmt::Display::fmt(e, f),
            Cause::MaskSource(e) => fmt::Display::fmt(e, f),
            Cause::Mask(e) => fmt::Display::fmt(e, f),
            Cause::Precompute(e) => fmt::Display::fmt(e, f),
            Cause::Walk(e) => fmt::Display::fmt(e, f),
        }
    }
}
impl<E: std::error::Error + 'static> std::error::Error for Cause<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Cause::Funding(e) => Some(e),
            Cause::Vector(e) => Some(e),
            Cause::MaskSource(e) => Some(e),
            Cause::Mask(e) => Some(e),
            Cause::Precompute(e) => Some(e),
            Cause::Walk(e) => Some(e),
            _ => None,
        }
    }
}
impl<E: fmt::Display> fmt::Display for PreparedLexerError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.cause, f)
    }
}
impl<E: std::error::Error + 'static> std::error::Error for PreparedLexerError<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.cause)
    }
}
/// A mutable lexer failure carries its typed cause; the failed lexer retains all
/// reached storage, so no constructor-prefix payload crosses the parser stack.
#[derive(Debug)]
pub struct PreparedLexerOperationError<E> {
    cause: Cause<E>,
}
impl<E: fmt::Display> fmt::Display for PreparedLexerOperationError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.cause, f)
    }
}
impl<E: std::error::Error + 'static> std::error::Error for PreparedLexerOperationError<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.cause)
    }
}
/// Move-only mutable lexer mechanisms. The enclosing grammar owner holds its
/// already copied immutable LexerSpec and original source identity/funding.
/// This type cannot create an ordinary Lexer or bypass parser admission.
#[derive(Debug)]
pub struct PreparedLexer {
    vector: PreparedRegexVector,
    allowed_first_byte: Option<SimpleVob>,
    selected: Option<LexemeSet>,
    failed: bool,
}
fn mask<F: Fn(usize) -> Result<(), E>, E>(bits: usize, reserve: &F) -> Result<SimpleVob, Cause<E>> {
    let plan = TokenMaskConstructionPlan::zeroed(bits).map_err(Cause::MaskSource)?;
    reserve(plan.requirements().required_bytes()).map_err(Cause::Funding)?;
    plan.compile().map_err(Cause::Mask)
}
impl PreparedLexer {
    fn controls<F, E>() -> Option<usize> {
        let parts = [
            TokenMaskConstructionPlan::inspection_control_bytes()?,
            size_of::<Self>(),
            size_of::<PreparedLexerError<E>>(),
            size_of::<PreparedLexerOperationError<E>>(),
            size_of::<Cause<E>>(),
            size_of::<PreparedRegexVectorOperationError<E>>(),
            size_of::<F>(),
            size_of::<TokenMaskConstructionPlan<'_>>(),
            size_of::<TokenMaskConstructionFailure>(),
            size_of::<TokenMaskSourceError>(),
            size_of::<Result<Self, PreparedLexerError<E>>>(),
            size_of::<Result<(), E>>(),
            size_of::<Result<(), Cause<E>>>(),
            size_of::<Result<SimpleVob, Cause<E>>>(),
            size_of::<Result<SimpleVob, TokenMaskConstructionFailure>>(),
            size_of::<Result<TokenMaskConstructionPlan<'_>, TokenMaskSourceError>>(),
            size_of::<Result<StateID, PreparedRegexVectorOperationError<E>>>(),
            size_of::<Option<SimpleVob>>(),
            size_of::<Option<LexemeSet>>(),
            size_of::<(usize, u8, StateID, StateID)>(),
            size_of::<(&mut PreparedRegexVector, &F, StateID)>(),
            size_of::<(
                &mut PreparedRegexVector,
                &SimpleVob,
                &toktrie::TokTrie,
                &LexemeSet,
                &F,
            )>(),
            size_of::<std::ops::Range<usize>>(),
            size_of::<std::ops::RangeInclusive<u8>>(),
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
    /// Consumes the existing paid vector, selects its actual lexical rows, and
    /// uses the same full first-byte warm-up as ordinary Lexer construction.
    pub fn prepare<F: Fn(usize) -> Result<(), E>, E>(
        vector: PreparedRegexVector,
        reserve: &F,
    ) -> Result<Self, PreparedLexerError<E>> {
        let mut owner = Self {
            vector,
            allowed_first_byte: None,
            selected: None,
            failed: true,
        };
        let result = (|| -> Result<(), Cause<E>> {
            reserve(Self::controls::<F, E>().ok_or(Cause::Overflow)?).map_err(Cause::Funding)?;
            owner.selected = Some(LexemeSet::from_owned_vob(mask(
                owner.vector.roots().len(),
                reserve,
            )?));
            for index in 0..owner.vector.roots().len() {
                owner
                    .selected
                    .as_mut()
                    .expect("selection")
                    .add(LexemeIdx::new(index));
            }
            let state = owner
                .vector
                .initial_state(owner.selected.as_ref().expect("selection"), reserve)
                .map_err(Cause::Vector)?;
            owner.allowed_first_byte = Some(mask(256, reserve)?);
            let vector = &mut owner.vector;
            policy::warm_first_bytes(
                owner.allowed_first_byte.as_mut().expect("first bytes"),
                |byte| {
                    vector
                        .transition(state, byte, reserve)
                        .map_err(Cause::Vector)
                },
            )?;
            owner.selected = None;
            Ok(())
        })();
        match result {
            Ok(()) => {
                owner.failed = false;
                Ok(owner)
            }
            Err(cause) => Err(PreparedLexerError {
                cause,
                prefix: Some(owner),
            }),
        }
    }
    fn run<T, F, E, G>(&mut self, reserve: &F, run: G) -> Result<T, PreparedLexerOperationError<E>>
    where
        F: Fn(usize) -> Result<(), E>,
        G: FnOnce(&mut Self, &F) -> Result<T, Cause<E>>,
    {
        if self.failed {
            return Err(PreparedLexerOperationError {
                cause: Cause::Failed,
            });
        }
        self.failed = true;
        let result = (|| {
            let frames = [
                Self::controls::<F, E>().ok_or(Cause::Overflow)?,
                size_of::<G>(),
                size_of::<T>(),
                size_of::<Result<T, Cause<E>>>(),
                size_of::<Result<T, PreparedLexerOperationError<E>>>(),
                size_of::<(&mut Self, &F)>(),
                size_of::<(&StateDesc, &StateDesc, &SimpleVob)>(),
                size_of::<LexerResult>(),
                size_of::<PreLexeme>(),
                size_of::<NextByte>(),
            ];
            reserve(
                frames
                    .into_iter()
                    .try_fold(size_of_val(&frames), usize::checked_add)
                    .ok_or(Cause::Overflow)?,
            )
            .map_err(Cause::Funding)?;
            run(self, reserve)
        })();
        match result {
            Ok(value) => {
                self.failed = false;
                Ok(value)
            }
            Err(cause) => Err(PreparedLexerOperationError { cause }),
        }
    }
    fn info<E>(&self, state: StateID) -> Result<&StateDesc, Cause<E>> {
        if state.as_u32() == StateID::MISSING.as_u32() {
            return Err(Cause::Source);
        }
        self.vector.state_desc(state).ok_or(Cause::Source)
    }
    /// Same selected initial-state worker and state interning as the ordinary lexer.
    pub fn start_state<F: Fn(usize) -> Result<(), E>, E>(
        &mut self,
        selected: &LexemeSet,
        reserve: &F,
    ) -> Result<StateID, PreparedLexerOperationError<E>> {
        self.run(reserve, |owner, reserve| {
            owner
                .vector
                .initial_state(selected, reserve)
                .map_err(Cause::Vector)
        })
    }
    /// One real derivative transition followed by the shared greedy/lazy/special decision.
    pub fn advance<F: Fn(usize) -> Result<(), E>, E>(
        &mut self,
        prev: StateID,
        byte: u8,
        reserve: &F,
    ) -> Result<LexerResult, PreparedLexerOperationError<E>> {
        self.run(reserve, |owner, reserve| {
            owner.info::<E>(prev)?;
            let state = owner
                .vector
                .transition(prev, byte, reserve)
                .map_err(Cause::Vector)?;
            Ok(policy::advance(
                prev,
                byte,
                state,
                owner
                    .allowed_first_byte
                    .as_ref()
                    .expect("completed warm-up"),
                owner.info::<E>(prev)?,
                owner.info::<E>(state)?,
            ))
        })
    }
    /// Same raw initial-row transition used by ordinary lexical handoff.
    pub(crate) fn transition_start_state<F: Fn(usize) -> Result<(), E>, E>(
        &mut self,
        state: StateID,
        byte: Option<u8>,
        reserve: &F,
    ) -> Result<StateID, PreparedLexerOperationError<E>> {
        self.run(reserve, |owner, reserve| {
            owner.info::<E>(state)?;
            match byte {
                None => Ok(state),
                Some(byte) => owner
                    .vector
                    .transition(state, byte, reserve)
                    .map_err(Cause::Vector),
            }
        })
    }
    /// Restricts an existing state through the same paid state-interning worker.
    pub(crate) fn limit_state_to<F: Fn(usize) -> Result<(), E>, E>(
        &mut self,
        state: StateID,
        selected: &LexemeSet,
        reserve: &F,
    ) -> Result<StateID, PreparedLexerOperationError<E>> {
        self.run(reserve, |owner, reserve| {
            owner
                .vector
                .limit_state_to(state, selected, reserve)
                .map_err(Cause::Vector)
        })
    }
    /// Actual next-byte query, including greedy-match fuzzing used by ordinary Lexer.
    pub fn next_byte<F: Fn(usize) -> Result<(), E>, E>(
        &mut self,
        state: StateID,
        reserve: &F,
    ) -> Result<NextByte, PreparedLexerOperationError<E>> {
        self.run(reserve, |owner, reserve| {
            if state.has_lowest_match() {
                return Err(Cause::Source);
            }
            let next = owner
                .vector
                .next_byte(state, reserve)
                .map_err(Cause::Vector)?;
            Ok(policy::next_byte(next, owner.info::<E>(state)?))
        })
    }
    /// Shared ordinary forced-end decision on the same completed descriptor.
    pub fn force_lexeme_end<F: Fn(usize) -> Result<(), E>, E>(
        &mut self,
        state: StateID,
        reserve: &F,
    ) -> Result<LexerResult, PreparedLexerOperationError<E>> {
        self.run(reserve, |owner, _| {
            Ok(policy::force_end(owner.info::<E>(state)?))
        })
    }
    /// Shared ordinary greedy accepting end decision.
    pub fn try_lexeme_end<F: Fn(usize) -> Result<(), E>, E>(
        &mut self,
        state: StateID,
        reserve: &F,
    ) -> Result<LexerResult, PreparedLexerOperationError<E>> {
        self.run(reserve, |owner, _| {
            Ok(policy::try_end(state, owner.info::<E>(state)?))
        })
    }
    /// Shared single-byte ForcedEOI decision without greedy fuzzing.
    pub fn check_for_single_byte_lexeme<F: Fn(usize) -> Result<(), E>, E>(
        &mut self,
        state: StateID,
        byte: u8,
        reserve: &F,
    ) -> Result<Option<PreLexeme>, PreparedLexerOperationError<E>> {
        self.run(reserve, |owner, reserve| {
            let next = owner
                .vector
                .next_byte(state, reserve)
                .map_err(Cause::Vector)?;
            Ok(policy::single_byte(state, byte, next))
        })
    }
    /// The actual source-trie precompute uses the same fixed-depth stack,
    /// TokTrie traversal and lexical decision worker as ordinary Lexer.
    pub fn precompute_for<F: Fn(usize) -> Result<(), E>, E>(
        &mut self,
        trie: &toktrie::TokTrie,
        selected: &LexemeSet,
        reserve: &F,
    ) -> Result<(), PreparedLexerOperationError<E>> {
        self.run(reserve, |owner, reserve| {
            Self::precompute_parts(
                &mut owner.vector,
                owner
                    .allowed_first_byte
                    .as_ref()
                    .expect("completed warm-up"),
                trie,
                selected,
                reserve,
            )
        })
    }
    fn precompute_parts<F: Fn(usize) -> Result<(), E>, E>(
        vector: &mut PreparedRegexVector,
        first: &SimpleVob,
        trie: &toktrie::TokTrie,
        selected: &LexemeSet,
        reserve: &F,
    ) -> Result<(), Cause<E>> {
        use super::precompute::{LexerPrecompute, LexerPrecomputePlan};
        reserve(LexerPrecomputePlan::inspection_control_bytes().ok_or(Cause::Overflow)?)
            .map_err(Cause::Funding)?;
        let state = vector
            .initial_state(selected, reserve)
            .map_err(Cause::Vector)?;
        let advance = |previous: StateID, byte: u8| {
            let state = vector.transition(previous, byte, reserve)?;
            Ok(policy::advance(
                previous,
                byte,
                state,
                first,
                vector
                    .state_desc(previous)
                    .expect("closed initial/trie predecessor"),
                vector.state_desc(state).expect("completed transition"),
            ))
        };
        reserve(LexerPrecompute::execution_control_bytes(&advance).ok_or(Cause::Overflow)?)
            .map_err(Cause::Funding)?;
        let plan = LexerPrecomputePlan::prepare(trie).map_err(Cause::Precompute)?;
        reserve(plan.requirements().required_bytes()).map_err(Cause::Funding)?;
        let storage = plan.compile().map_err(Cause::Precompute)?;
        storage
            .run_with(state, advance)
            .map(drop)
            .map_err(Cause::Walk)
    }
    /// Same parser startup schedule, with paid single-lexeme selections and the
    /// exact original trie. Failure retains any reached selection in this owner.
    pub fn prepare_large_lexemes<F: Fn(usize) -> Result<(), E>, E>(
        &mut self,
        trie: &toktrie::TokTrie,
        limits: &crate::api::ParserLimits,
        reserve: &F,
    ) -> Result<(), PreparedLexerOperationError<E>> {
        self.run(reserve, |owner, reserve| {
            let parts = [
                size_of::<Scheduled<'_, F, E>>(),
                size_of::<&crate::api::ParserLimits>(),
                size_of::<std::ops::Range<usize>>(),
                size_of::<(usize, LexemeIdx, u32, u64)>(),
                size_of::<Result<u32, Cause<E>>>(),
                size_of::<Result<(), Cause<E>>>(),
                size_of::<(
                    &mut PreparedRegexVector,
                    &SimpleVob,
                    &toktrie::TokTrie,
                    &LexemeSet,
                    &F,
                )>(),
            ];
            reserve(
                parts
                    .into_iter()
                    .try_fold(size_of_val(&parts), usize::checked_add)
                    .ok_or(Cause::Overflow)?,
            )
            .map_err(Cause::Funding)?;
            super::schedule::run(
                &mut Scheduled {
                    owner,
                    trie,
                    reserve,
                    marker: std::marker::PhantomData,
                },
                limits,
            )
        })
    }
    pub(crate) fn begin_mask<F: Fn(usize) -> Result<(), E>, E>(
        &mut self,
        limits: &crate::api::ParserLimits,
        funding: &F,
    ) -> Result<(), PreparedLexerOperationError<E>> {
        self.run(funding, |owner, _| {
            owner.vector.set_fuel(limits.step_lexer_fuel);
            owner.vector.set_max_states(limits.max_lexer_states);
            Ok(())
        })
    }
    /// Borrowed completed vector for descriptor observations, never mutable execution.
    pub fn vector(&self) -> &PreparedRegexVector {
        &self.vector
    }
    /// Exact completed first-byte warm-up result.
    pub fn allows_first_byte(&self, byte: u8) -> bool {
        self.allowed_first_byte
            .as_ref()
            .is_some_and(|set| set.is_allowed(byte as u32))
    }
}

struct Scheduled<'a, F, E> {
    owner: &'a mut PreparedLexer,
    trie: &'a toktrie::TokTrie,
    reserve: &'a F,
    marker: std::marker::PhantomData<E>,
}
impl<F: Fn(usize) -> Result<(), E>, E> super::schedule::Context for Scheduled<'_, F, E> {
    type Error = Cause<E>;
    fn len(&self) -> usize {
        self.owner.vector.roots().len()
    }
    fn lexeme(&self, index: usize) -> LexemeIdx {
        LexemeIdx::new(index)
    }
    fn set_fuel(&mut self, fuel: u64) {
        self.owner.vector.set_fuel(fuel);
    }
    fn weight(&mut self, lexeme: LexemeIdx) -> Result<u32, Cause<E>> {
        self.owner
            .vector
            .lexeme_weight(lexeme, self.reserve)
            .map_err(Cause::Vector)
    }
    fn precompute(&mut self, lexeme: LexemeIdx) -> Result<(), Cause<E>> {
        self.owner.selected = Some(LexemeSet::from_owned_vob(mask(
            self.owner.vector.roots().len(),
            self.reserve,
        )?));
        self.owner.selected.as_mut().expect("selection").add(lexeme);
        PreparedLexer::precompute_parts(
            &mut self.owner.vector,
            self.owner
                .allowed_first_byte
                .as_ref()
                .expect("completed warm-up"),
            self.trie,
            self.owner.selected.as_ref().expect("selection"),
            self.reserve,
        )?;
        self.owner.selected = None;
        Ok(())
    }
}
