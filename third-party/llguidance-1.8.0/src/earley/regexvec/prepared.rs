//! Owning paid lexical constructor and actual state publication using shared workers.
mod copy;
pub use copy::PreparedRegexVectorCopyFailure;
use super::{
    checked_state_table_growth, construction, descriptor, subsumption, transition, ExprRef,
    ExprSet, LexemeIdx, LexemeSet, MatchingLexemes, NextByte, NextByteCache, StateDesc,
    StateGrowthError, StateID,
};
use crate::{
    api::{InvalidLexerStateLimit, ParserLimits},
    earley::lexerspec::RegexVectorInput,
};
use derivre::raw::{
    ExprSetCopyFailure, ExprSetPreparedSourcePlan, HashConsCapacityError, HashConsCopyFailure,
    HashConsEmptySourcePlan, PreparedExprError, PreparedExprSet, PreparedExpressionFailure,
    PreparedExpressionMachine, PreparedExpressionOperationError, PreparedExpressionPlan,
    ParserAllocationFunding, PreparedRelevanceError, PreparedVecHashCons, PreparedWeightError,
};
use std::{
    alloc::Layout,
    collections::TryReserveError,
    fmt,
    marker::PhantomData,
    mem::{size_of, size_of_val},
};
use toktrie::{TokenMaskConstructionFailure, TokenMaskConstructionPlan, TokenMaskSourceError};

#[derive(Debug)]
enum Cause<E> {
    Overflow,
    Capacity,
    Source,
    Failed,
    State(StateGrowthError),
    Funding(E),
    Allocation(TryReserveError),
    Relevance(PreparedRelevanceError<E>),
    Weight(PreparedWeightError),
    Derivative(PreparedExpressionOperationError),
    Table(HashConsCapacityError),
    MaskSource(TokenMaskSourceError),
    Mask(TokenMaskConstructionFailure),
}
// Construction-only failures keep their reached sources. They never travel
// through mutable DFA operations, whose destinations remain in the vector.
#[derive(Debug)]
enum ConstructionCause<E> {
    Operation(Cause<E>),
    Limits(InvalidLexerStateLimit),
    Expressions(ExprSetCopyFailure),
    Machine(PreparedExpressionFailure),
    TableSource(HashConsCopyFailure),
    Binding(PreparedExprError),
}
impl<E> From<Cause<E>> for ConstructionCause<E> {
    fn from(cause: Cause<E>) -> Self {
        Self::Operation(cause)
    }
}
impl<E: fmt::Display> fmt::Display for Cause<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Cause::Overflow => f.write_str("prepared lexer destination geometry overflow"),
            Cause::Capacity => {
                f.write_str("prepared lexer destination differs from its exact capacity")
            }
            Cause::Source => f.write_str("prepared lexer source or selection differs"),
            Cause::Failed => f.write_str("prepared lexer retains a failed operation"),
            Cause::State(e) => fmt::Display::fmt(e, f),
            Cause::Funding(e) => fmt::Display::fmt(e, f),
            Cause::Allocation(e) => fmt::Display::fmt(e, f),
            Cause::Relevance(e) => fmt::Display::fmt(e, f),
            Cause::Weight(e) => fmt::Display::fmt(e, f),
            Cause::Derivative(e) => fmt::Display::fmt(e, f),
            Cause::Table(e) => fmt::Display::fmt(e, f),
            Cause::MaskSource(e) => fmt::Display::fmt(e, f),
            Cause::Mask(e) => fmt::Display::fmt(e, f),
        }
    }
}
impl<E: std::error::Error + 'static> std::error::Error for Cause<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(match self {
            Cause::State(e) => e,
            Cause::Funding(e) => e,
            Cause::Allocation(e) => e,
            Cause::Relevance(e) => e,
            Cause::Weight(e) => e,
            Cause::Derivative(e) => e,
            Cause::Table(e) => e,
            Cause::MaskSource(e) => e,
            Cause::Mask(e) => e,
            _ => return None,
        })
    }
}
impl<E: fmt::Display> fmt::Display for ConstructionCause<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Operation(cause) => fmt::Display::fmt(cause, f),
            Self::Limits(cause) => fmt::Display::fmt(cause, f),
            Self::Expressions(cause) => fmt::Display::fmt(cause, f),
            Self::Machine(cause) => fmt::Display::fmt(cause, f),
            Self::TableSource(cause) => fmt::Display::fmt(cause, f),
            Self::Binding(cause) => fmt::Display::fmt(cause, f),
        }
    }
}
impl<E: std::error::Error + 'static> std::error::Error for ConstructionCause<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(match self {
            Self::Operation(cause) => cause,
            Self::Limits(cause) => cause,
            Self::Expressions(cause) => cause,
            Self::Machine(cause) => cause,
            Self::TableSource(cause) => cause,
            Self::Binding(cause) => cause,
        })
    }
}
/// A construction failure owns the entire reached lexical prefix.
pub struct PreparedRegexVectorError<E> {
    cause: ConstructionCause<E>,
    prefix: Option<PreparedRegexVector>,
}
impl<E: fmt::Debug> fmt::Debug for PreparedRegexVectorError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PreparedRegexVectorError")
            .field("cause", &self.cause)
            .field("retains_prefix", &self.prefix.is_some())
            .finish()
    }
}
impl<E: fmt::Display> fmt::Display for PreparedRegexVectorError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.cause, f)
    }
}
impl<E: std::error::Error + 'static> std::error::Error for PreparedRegexVectorError<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.cause)
    }
}
/// A mutable DFA operation failure. All reached destinations remain in the
/// failed vector, so the error transports only the original typed cause.
#[derive(Debug)]
pub struct PreparedRegexVectorOperationError<E> {
    cause: Cause<E>,
}
impl<E: fmt::Display> fmt::Display for PreparedRegexVectorOperationError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.cause, f)
    }
}
impl<E: std::error::Error + 'static> std::error::Error for PreparedRegexVectorOperationError<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.cause)
    }
}
struct PendingState {
    desc: Option<StateDesc>,
    selection: descriptor::Selection,
}
/// Exact lexical input, independently owned expression mechanisms and fixed
/// intern-table storage. It grants no ordinary Clone or raw mutable source.
/// The enclosing grammar owner retains its funding and immutable declaration.
pub struct PreparedRegexVector {
    machine: Option<PreparedExpressionMachine>,
    prepared: Option<PreparedExprSet>,
    plan: Option<PreparedExpressionPlan>,
    table: Option<PreparedVecHashCons>,
    lazy: Option<LexemeSet>,
    subsumable: Option<LexemeSet>,
    states: Vec<StateDesc>,
    transitions: Vec<StateID>,
    next: Vec<Option<NextByte>>,
    candidate: Vec<u32>,
    pending: Option<PendingState>,
    max_states: usize,
    fuel: u64,
    transitions_attempted: usize,
    failed: bool,
    input: RegexVectorInput,
}
impl fmt::Debug for PreparedRegexVector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PreparedRegexVector")
            .field("states", &self.states.len())
            .field("roots", &self.input.roots.len())
            .field("failed", &self.failed)
            .finish()
    }
}
fn allocation_bytes<T>(total: usize) -> Option<usize> {
    Some(Layout::array::<T>(total).ok()?.size())
}
fn grow<T, F: crate::earley::PreparedFunding<Error = E>, E>(
    values: &mut Vec<T>,
    total: usize,
    reserve: &F,
) -> Result<(), Cause<E>> {
    if total <= values.capacity() {
        return Ok(());
    }
    let bytes = allocation_bytes::<T>(total).ok_or(Cause::Overflow)?;
    reserve.reserve(bytes).map_err(Cause::Funding)?;
    values
        .try_reserve_exact(total - values.len())
        .map_err(Cause::Allocation)?;
    if values.capacity() != total {
        return Err(Cause::Capacity);
    }
    Ok(())
}
fn mask<F: crate::earley::PreparedFunding<Error = E>, E>(bits: usize, reserve: &F) -> Result<LexemeSet, Cause<E>> {
    let plan = TokenMaskConstructionPlan::zeroed(bits).map_err(Cause::MaskSource)?;
    reserve.reserve(plan.requirements().required_bytes()).map_err(Cause::Funding)?;
    Ok(LexemeSet {
        vob: plan.compile().map_err(Cause::Mask)?,
    })
}
struct Expressions<'a, F, E> {
    machine: &'a mut PreparedExpressionMachine,
    reserve: &'a F,
    marker: PhantomData<E>,
}
impl<F: crate::earley::PreparedFunding<Error = E>, E> construction::Expressions for Expressions<'_, F, E> {
    type Error = Cause<E>;
    fn source(&self) -> &ExprSet {
        self.machine.source()
    }
    fn non_empty(&mut self, root: ExprRef, fuel: u64) -> Result<bool, Self::Error> {
        self.machine
            .is_non_empty(root, fuel, &|bytes| self.reserve.reserve(bytes))
            .map_err(Cause::Relevance)
    }
    fn has_repeat(&mut self, root: ExprRef) -> Result<bool, Self::Error> {
        self.reserve.reserve(
            PreparedExpressionMachine::weight_operation_control_bytes().ok_or(Cause::Overflow)?,
        )
        .map_err(Cause::Funding)?;
        self.machine.has_repeat(root).map_err(Cause::Weight)
    }
}
struct Descriptor<'a, F, E> {
    source: &'a ExprSet,
    next: &'a mut Vec<Option<NextByte>>,
    reserve: &'a F,
    marker: PhantomData<E>,
}
impl<F: crate::earley::PreparedFunding<Error = E>, E> descriptor::Context for Descriptor<'_, F, E> {
    type Error = Cause<E>;
    fn source(&self) -> &ExprSet {
        self.source
    }
    fn next_byte(&mut self, root: ExprRef) -> Result<NextByte, Self::Error> {
        if !self.source.is_valid(root) {
            return Err(Cause::Source);
        }
        if let Some(Some(value)) = self.next.get(root.as_usize()) {
            return Ok(*value);
        }
        let count = root.as_usize().checked_add(1).ok_or(Cause::Overflow)?;
        grow(self.next, count, self.reserve)?;
        if self.next.len() < count {
            self.next.resize(count, None);
        }
        let value = NextByteCache::analyze(self.source, root);
        self.next[root.as_usize()] = Some(value);
        Ok(value)
    }
    fn prepare_matches(
        &mut self,
        values: &mut Vec<LexemeIdx>,
        total: usize,
    ) -> Result<(), Self::Error> {
        grow(values, total, self.reserve)
    }
}
struct Advance<'a, F, E> {
    machine: &'a mut PreparedExpressionMachine,
    candidate: &'a mut Vec<u32>,
    reserve: &'a F,
    marker: PhantomData<E>,
}
impl<F: crate::earley::PreparedFunding<Error = E>, E> transition::Context for Advance<'_, F, E> {
    type Error = Cause<E>;
    fn cost(&self) -> u64 {
        self.machine.source().cost()
    }
    fn derivative(&mut self, root: ExprRef, byte: u8) -> Result<ExprRef, Self::Error> {
        self.reserve.reserve(
            PreparedExpressionMachine::operation_control_bytes().ok_or(Cause::Overflow)?,
        )
        .map_err(Cause::Funding)?;
        self.machine
            .derivative(root, byte)
            .map_err(Cause::Derivative)
    }
    fn non_empty(&mut self, root: ExprRef, fuel: u64) -> Result<bool, Self::Error> {
        self.machine
            .is_non_empty(root, fuel, &|bytes| self.reserve.reserve(bytes))
            .map_err(Cause::Relevance)
    }
    fn is_fuel(error: &Self::Error) -> bool {
        matches!(error, Cause::Relevance(PreparedRelevanceError::Fuel { .. }))
    }
    fn emit(&mut self, index: LexemeIdx, root: ExprRef) -> Result<(), Self::Error> {
        if self.candidate.len().checked_add(2).ok_or(Cause::Overflow)? > self.candidate.capacity() {
            return Err(Cause::Capacity);
        }
        let index = u32::try_from(index.as_usize()).map_err(|_| Cause::Overflow)?;
        self.candidate.push(index);
        self.candidate.push(root.as_u32());
        Ok(())
    }
}
impl<F: crate::earley::PreparedFunding<Error = E>, E> subsumption::Context for Advance<'_, F, E> {
    type Error = Cause<E>;
    fn cost(&self) -> u64 {
        self.machine.source().cost()
    }
    fn contains(
        &mut self,
        small: ExprRef,
        big: ExprRef,
        fuel: u64,
        cache_failures: bool,
    ) -> Result<bool, Self::Error> {
        self.machine
            .is_contained_in_prefixes(small, big, fuel, cache_failures, &|bytes| self.reserve.reserve(bytes))
            .map_err(Cause::Relevance)
    }
    fn recover_refusal(error: Self::Error) -> Result<bool, Self::Error> {
        match error {
            Cause::Relevance(PreparedRelevanceError::Fuel { .. }) => Ok(false),
            error => Err(error),
        }
    }
}
impl PreparedRegexVector {
    /// Complete fixed constructor, source inspection, descriptor, and failure
    /// frames. Caller reserves these before consuming the input.
    pub fn construction_control_bytes<F, E>() -> Option<usize> {
        let parts = [
            ExprSet::prepared_source_inspection_control_bytes()?,
            PreparedExpressionPlan::inspection_control_bytes()?,
            HashConsEmptySourcePlan::inspection_control_bytes()?,
            TokenMaskConstructionPlan::inspection_control_bytes()?,
            size_of::<Self>(),
            size_of::<PreparedRegexVectorError<E>>(),
            size_of::<ConstructionCause<E>>(),
            size_of::<Result<(), ConstructionCause<E>>>(),
            size_of::<Cause<E>>(),
            size_of::<RegexVectorInput>(),
            size_of::<PreparedExprSet>(),
            size_of::<PreparedExpressionPlan>(),
            size_of::<ExprSetPreparedSourcePlan<'_>>(),
            size_of::<HashConsEmptySourcePlan<'_>>(),
            size_of::<TokenMaskConstructionPlan<'_>>(),
            size_of::<TokenMaskSourceError>(),
            size_of::<TokenMaskConstructionFailure>(),
            size_of::<StateDesc>(),
            size_of::<LexemeSet>(),
            size_of::<MatchingLexemes>() * 3,
            size_of::<PendingState>(),
            size_of::<Option<ParserAllocationFunding>>(),
            size_of::<ParserAllocationFunding>(),
            size_of::<Result<(), PreparedExprError>>(),
            size_of::<descriptor::Selection>(),
            size_of::<Expressions<'_, F, E>>(),
            size_of::<Descriptor<'_, F, E>>(),
            size_of::<Advance<'_, F, E>>(),
            size_of::<Result<ExprRef, PreparedExpressionOperationError>>(),
            size_of::<Result<bool, Cause<E>>>(),
            size_of::<u64>() * 4,
            size_of::<Option<StateID>>(),
            size_of::<Option<usize>>(),
            size_of::<usize>() * 10,
            size_of::<u64>() * 4,
            size_of::<ExprRef>() * 4,
            size_of::<[LexemeIdx; 2]>(),
            size_of::<NextByte>(),
            size_of::<Layout>(),
            size_of::<std::alloc::LayoutError>(),
            size_of::<std::slice::ChunksExact<'_, u32>>(),
            size_of::<std::slice::IterMut<'_, ExprRef>>(),
            size_of::<std::slice::Iter<'_, super::RxLexeme>>(),
            size_of::<toktrie::SimpleVobIter<'_>>(),
            size_of::<Result<Self, PreparedRegexVectorError<E>>>(),
            size_of::<Result<(), Cause<E>>>(),
            size_of::<Result<StateID, Cause<E>>>(),
            size_of::<Result<LexemeSet, Cause<E>>>(),
            size_of::<Result<PreparedExprSet, ExprSetCopyFailure>>(),
            size_of::<Result<PreparedExpressionPlan, PreparedExpressionFailure>>(),
            size_of::<Result<PreparedExpressionMachine, PreparedExpressionFailure>>(),
            size_of::<Result<PreparedVecHashCons, HashConsCopyFailure>>(),
            size_of::<Result<(), HashConsCapacityError>>(),
            size_of::<Result<u32, HashConsCapacityError>>(),
            size_of::<Result<(), TryReserveError>>(),
            size_of::<Result<(), InvalidLexerStateLimit>>(),
            size_of::<Result<(), E>>(),
            size_of::<Result<bool, PreparedRelevanceError<E>>>(),
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
    /// Consumes actual compiled lexical input and the same caller-selected
    /// limits. Every allocated destination is reserved before construction.
    pub fn prepare<F: crate::earley::PreparedFunding<Error = E>, E>(
        input: RegexVectorInput,
        limits: &mut ParserLimits,
        reserve: &F,
    ) -> Result<Self, PreparedRegexVectorError<E>> {
        Self::prepare_with_backing(input, limits, None, reserve)
    }
    /// Same source constructor, with a separately paid actual backing owner.
    /// It permits reached word/scratch allocations, never unchecked entry-table
    /// growth, an ordinary source alias, or lexer/parser admission.
    pub fn prepare_with_backing<F: crate::earley::PreparedFunding<Error = E>, E>(
        input: RegexVectorInput,
        limits: &mut ParserLimits,
        backing: Option<ParserAllocationFunding>,
        reserve: &F,
    ) -> Result<Self, PreparedRegexVectorError<E>> {
        let mut owner = Self {
            machine: None,
            prepared: None,
            plan: None,
            table: None,
            lazy: None,
            subsumable: None,
            states: Vec::new(),
            transitions: Vec::new(),
            next: Vec::new(),
            candidate: Vec::new(),
            pending: None,
            max_states: limits.max_lexer_states,
            fuel: u64::MAX,
            transitions_attempted: 0,
            failed: true,
            input,
        };
        let result = (|| -> Result<(), ConstructionCause<E>> {
            reserve.reserve(Self::construction_control_bytes::<F, E>().ok_or(Cause::Overflow)?)
                .map_err(Cause::Funding)?;
            limits
                .validate_lexer_state_limit()
                .map_err(ConstructionCause::Limits)?;
            if owner.input.rows.len() != owner.input.roots.len()
                || owner.input.alpha.is_empty()
                || owner
                    .input
                    .roots
                    .iter()
                    .any(|r| !owner.input.expressions.is_valid(*r))
            {
                return Err(Cause::Source.into());
            }
            let plan = owner
                .input
                .expressions
                .prepared_source_plan()
                .map_err(ConstructionCause::Expressions)?;
            reserve.reserve(plan.requirements().required_bytes()).map_err(Cause::Funding)?;
            owner.prepared = Some(plan.compile().map_err(ConstructionCause::Expressions)?);
            if let Some(funding) = &backing {
                owner
                    .prepared
                    .as_mut()
                    .expect("prepared source")
                    .bind_backing_funding(funding.clone())
                    .map_err(ConstructionCause::Binding)?;
            }
            owner.plan = Some(
                PreparedExpressionPlan::prepare(owner.prepared.take().expect("prepared source"))
                    .map_err(ConstructionCause::Machine)?,
            );
            reserve.reserve(
                owner
                    .plan
                    .as_ref()
                    .expect("expression plan")
                    .requirements()
                    .required_bytes(),
            )
            .map_err(Cause::Funding)?;
            owner.machine = Some(
                owner
                    .plan
                    .take()
                    .expect("funded plan")
                    .compile()
                    .map_err(ConstructionCause::Machine)?,
            );
            let mut expressions = Expressions {
                machine: owner.machine.as_mut().expect("machine"),
                reserve,
                marker: PhantomData,
            };
            construction::roots(
                &mut expressions,
                &mut owner.input.roots,
                &mut limits.initial_lexer_fuel,
            )?;
            owner.lazy = Some(mask(owner.input.rows.len(), reserve)?);
            owner.subsumable = Some(mask(owner.input.rows.len(), reserve)?);
            construction::classify(
                &mut expressions,
                &owner.input.rows,
                owner.lazy.as_mut().expect("lazy mask"),
                owner.subsumable.as_mut().expect("repeat mask"),
            )?;
            let width = owner
                .input
                .rows
                .len()
                .checked_mul(2)
                .ok_or(Cause::Overflow)?
                .max(1);
            let plan = owner
                .input
                .expressions
                .empty_table_source_plan(width)
                .map_err(ConstructionCause::TableSource)?;
            reserve.reserve(plan.requirements().required_bytes()).map_err(Cause::Funding)?;
            owner.table = Some(plan.compile().map_err(ConstructionCause::TableSource)?);
            if let Some(funding) = &backing {
                owner
                    .table
                    .as_mut()
                    .expect("state table")
                    .bind_backing_funding(funding.clone())
                    .map_err(Cause::Table)?;
            }
            StateID::seed_hash_cons(|words| {
                owner.table.as_mut().expect("state table").try_insert(words)
            })
            .map_err(Cause::Table)?;
            // The ordinary MISSING descriptor is a copy of DEAD, with DEAD's
            // state field. Both have independent possible-mask storage.
            for _ in 0..2 {
                let table_len = checked_state_table_growth(
                    owner.states.len(),
                    owner.input.alpha.len(),
                    owner.max_states,
                )
                .map_err(Cause::State)?;
                let state_count = owner.states.len().checked_add(1).ok_or(Cause::Overflow)?;
                grow(&mut owner.states, state_count, reserve)?;
                grow(&mut owner.transitions, table_len, reserve)?;
                owner.pending = Some(PendingState {
                    desc: None,
                    selection: descriptor::Selection::new(),
                });
                owner.pending.as_mut().expect("state prefix").desc = Some(StateDesc::empty(
                    StateID::DEAD,
                    mask(owner.input.rows.len(), reserve)?,
                ));
                owner.finish_descriptor(&[], reserve)?;
                owner.transitions.resize(table_len, StateID::DEAD);
                owner.states.push(
                    owner
                        .pending
                        .as_mut()
                        .expect("state prefix")
                        .desc
                        .take()
                        .expect("descriptor"),
                );
                owner.pending = None;
            }
            Ok(())
        })();
        match result {
            Ok(()) => {
                owner.failed = false;
                Ok(owner)
            }
            Err(cause) => Err(PreparedRegexVectorError {
                cause,
                prefix: Some(owner),
            }),
        }
    }
    fn finish_descriptor<F: crate::earley::PreparedFunding<Error = E>, E>(
        &mut self,
        words: &[u32],
        reserve: &F,
    ) -> Result<(), Cause<E>> {
        let pending = self.pending.as_mut().ok_or(Cause::Failed)?;
        let desc = pending.desc.as_mut().ok_or(Cause::Failed)?;
        let mut context = Descriptor {
            source: self.machine.as_ref().ok_or(Cause::Failed)?.source(),
            next: &mut self.next,
            reserve,
            marker: PhantomData,
        };
        descriptor::possible(&mut context, words, desc)?;
        descriptor::lowest(
            &mut context,
            words,
            &self.input.roots,
            self.input.special_token_rx,
            self.lazy.as_ref().ok_or(Cause::Failed)?,
            desc,
            &mut pending.selection,
        )
    }
    /// Fixed candidate, descriptor, publication and typed-refusal frames, before
    /// any operation walks selected roots or touches its paid destinations.
    pub fn state_operation_control_bytes<F, E>() -> Option<usize> {
        let parts = [
            // State operations retain the vector in place. Its owned machine,
            // source and table controls were paid by prepare, not moved onto
            // every transition's stack.
            size_of::<&mut Self>(),
            size_of::<PreparedRegexVectorOperationError<E>>(),
            size_of::<Cause<E>>(),
            size_of::<PendingState>(),
            size_of::<StateDesc>(),
            size_of::<descriptor::Selection>(),
            size_of::<Descriptor<'_, F, E>>(),
            size_of::<Advance<'_, F, E>>(),
            size_of::<Result<ExprRef, PreparedExpressionOperationError>>(),
            size_of::<Result<bool, Cause<E>>>(),
            size_of::<u64>() * 4,
            size_of::<MatchingLexemes>() * 3,
            size_of::<Vec<LexemeIdx>>(),
            size_of::<Vec<u32>>(),
            size_of::<[LexemeIdx; 2]>(),
            size_of::<toktrie::SimpleVobIter<'_>>(),
            size_of::<std::slice::ChunksExact<'_, u32>>(),
            size_of::<std::slice::Iter<'_, ExprRef>>(),
            size_of::<Option<NextByte>>(),
            size_of::<Option<StateID>>(),
            size_of::<ExprRef>() * 5,
            size_of::<usize>() * 12,
            size_of::<derivre::raw::Expr<'_>>() * 2,
            size_of::<StateGrowthError>(),
            size_of::<Result<StateID, Cause<E>>>(),
            size_of::<Result<StateID, PreparedRegexVectorOperationError<E>>>(),
            size_of::<Result<(), Cause<E>>>(),
            size_of::<Result<(), E>>(),
            size_of::<Result<(), TryReserveError>>(),
            size_of::<Layout>(),
            TokenMaskConstructionPlan::inspection_control_bytes()?,
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
    /// Builds the actual selected-root DFA state under paid candidate/descriptor
    /// storage. Existing intern hits keep their state ID and lowest-match bit.
    pub fn initial_state<F: crate::earley::PreparedFunding<Error = E>, E>(
        &mut self,
        selected: &LexemeSet,
        reserve: &F,
    ) -> Result<StateID, PreparedRegexVectorOperationError<E>> {
        if self.failed {
            return Err(PreparedRegexVectorOperationError {
                cause: Cause::Failed,
            });
        }
        self.failed = true;
        let result = (|| -> Result<StateID, Cause<E>> {
            let _frame = reserve.frame(Self::state_operation_control_bytes::<F, E>().ok_or(Cause::Overflow)?)
                .map_err(Cause::frame)?;
            if selected.len() != self.input.roots.len() || self.pending.is_some() {
                return Err(Cause::Source);
            }
            self.candidate.clear();
            let mut count = 0usize;
            construction::selected(&self.input.roots, selected, |_, _| {
                count = count.checked_add(2).ok_or(Cause::Overflow)?;
                Ok::<(), Cause<E>>(())
            })?;
            grow(&mut self.candidate, count, reserve)?;
            construction::selected(&self.input.roots, selected, |index, root| {
                let index = u32::try_from(index.as_usize()).map_err(|_| Cause::Overflow)?;
                self.candidate.push(index);
                self.candidate.push(root.as_u32());
                Ok::<(), Cause<E>>(())
            })?;
            if self.candidate.len() != count {
                return Err(Cause::Source);
            }
            self.publish_candidate(reserve)
        })();
        match result {
            Ok(state) => {
                self.failed = false;
                Ok(state)
            }
            Err(cause) => Err(PreparedRegexVectorOperationError { cause }),
        }
    }
    fn publish_candidate<F: crate::earley::PreparedFunding<Error = E>, E>(
        &mut self,
        reserve: &F,
    ) -> Result<StateID, Cause<E>> {
        let table = self.table.as_mut().ok_or(Cause::Failed)?;
        if let Some(id) = table.lookup(&self.candidate) {
            let state = StateID::new(id);
            let desc = self.states.get(id as usize).ok_or(Cause::Source)?;
            self.candidate.clear();
            return Ok(if desc.lazy_accepting.is_some() {
                state._set_lowest_match()
            } else {
                state
            });
        }
        let table_len =
            checked_state_table_growth(self.states.len(), self.input.alpha.len(), self.max_states)
                .map_err(Cause::State)?;
        let state_count = self.states.len().checked_add(1).ok_or(Cause::Overflow)?;
        grow(&mut self.states, state_count, reserve)?;
        grow(&mut self.transitions, table_len, reserve)?;
        let id = table.try_insert(&self.candidate).map_err(Cause::Table)?;
        if id as usize != self.states.len() {
            return Err(Cause::Source);
        }
        let state = StateID::new(id);
        self.pending = Some(PendingState {
            desc: None,
            selection: descriptor::Selection::new(),
        });
        self.pending.as_mut().expect("publication prefix").desc = Some(StateDesc::empty(
            state,
            mask(self.input.rows.len(), reserve)?,
        ));
        let pending = self.pending.as_mut().expect("publication prefix");
        let desc = pending.desc.as_mut().expect("paid descriptor");
        let mut context = Descriptor {
            source: self.machine.as_ref().ok_or(Cause::Failed)?.source(),
            next: &mut self.next,
            reserve,
            marker: PhantomData,
        };
        descriptor::possible(&mut context, &self.candidate, desc)?;
        descriptor::lowest(
            &mut context,
            &self.candidate,
            &self.input.roots,
            self.input.special_token_rx,
            self.lazy.as_ref().ok_or(Cause::Failed)?,
            desc,
            &mut pending.selection,
        )?;
        let state = if desc.lazy_accepting.is_some() {
            state._set_lowest_match()
        } else {
            state
        };
        self.transitions.resize(table_len, StateID::MISSING);
        self.states
            .push(pending.desc.take().expect("completed descriptor"));
        self.pending = None;
        self.candidate.clear();
        Ok(state)
    }
    /// Existing per-edge expression-fuel policy. It never clears a failure or
    /// grants expression/state storage beyond the retained finite sources.
    pub fn set_fuel(&mut self, fuel: u64) {
        if !self.failed {
            self.fuel = fuel;
        }
    }
    pub(crate) fn set_max_states(&mut self, maximum: usize) {
        if !self.failed {
            self.max_states = maximum;
        }
    }
    /// Remaining ordinary expression-fuel allowance after reached work.
    pub fn fuel(&self) -> u64 {
        self.fuel
    }
    /// Actual attempted uncached transitions, including a failed derivative prefix.
    pub fn transitions_attempted(&self) -> usize {
        self.transitions_attempted
    }
    /// Computes one real DFA edge with the same derivative/relevance ordering.
    /// A refusal retains the source, candidate and any completed publication.
    pub fn transition<F: crate::earley::PreparedFunding<Error = E>, E>(
        &mut self,
        state: StateID,
        byte: u8,
        reserve: &F,
    ) -> Result<StateID, PreparedRegexVectorOperationError<E>> {
        if self.failed {
            return Err(PreparedRegexVectorOperationError {
                cause: Cause::Failed,
            });
        }
        self.failed = true;
        let result = (|| -> Result<StateID, Cause<E>> {
            let _frame = reserve.frame(Self::state_operation_control_bytes::<F, E>().ok_or(Cause::Overflow)?)
                .map_err(Cause::frame)?;
            if state.as_u32() == StateID::MISSING.as_u32()
                || state.as_usize() >= self.states.len()
                || self.pending.is_some()
            {
                return Err(Cause::Source);
            }
            let index = self.input.alpha.map_state(state, byte);
            let next = *self.transitions.get(index).ok_or(Cause::Source)?;
            if next != StateID::MISSING {
                return Ok(next);
            }
            let words = self
                .table
                .as_ref()
                .ok_or(Cause::Failed)?
                .get(state.as_u32());
            self.candidate.clear();
            grow(&mut self.candidate, words.len(), reserve)?;
            self.transitions_attempted = self
                .transitions_attempted
                .checked_add(1)
                .ok_or(Cause::Overflow)?;
            let fuel_before = self.fuel;
            transition::build(
                &mut Advance {
                    machine: self.machine.as_mut().ok_or(Cause::Failed)?,
                    candidate: &mut self.candidate,
                    reserve,
                    marker: PhantomData,
                },
                words,
                byte,
                &mut self.fuel,
            )?;
            if self.fuel == 0 {
                return Err(Cause::Relevance(PreparedRelevanceError::Fuel {
                    max_fuel: fuel_before,
                }));
            }
            let state = self.publish_candidate(reserve)?;
            self.transitions[index] = state;
            Ok(state)
        })();
        match result {
            Ok(state) => {
                self.failed = false;
                Ok(state)
            }
            Err(cause) => Err(PreparedRegexVectorOperationError { cause }),
        }
    }
    /// Existing lazy/dead-state applicability predicate over actual retained
    /// state rows; malformed or failed owners never supply subsumption authority.
    pub fn subsume_possible(&self, state: StateID) -> bool {
        !self.failed && self.subsume_geometry(state)
    }
    fn subsume_geometry(&self, state: StateID) -> bool {
        if state.is_dead()
            || state.as_u32() == StateID::MISSING.as_u32()
            || state.as_usize() >= self.states.len()
        {
            return false;
        }
        let Some(table) = self.table.as_ref() else {
            return false;
        };
        let Some(lazy) = self.lazy.as_ref() else {
            return false;
        };
        !descriptor::pairs(table.get(state.as_u32())).any(|(index, _)| lazy.contains(index))
    }
    /// Uses the ordinary budget/cache-failure selection and containment worker.
    /// Fuel alone is recoverable; funding/source/equation failures are terminal.
    pub fn check_subsume<F: crate::earley::PreparedFunding<Error = E>, E>(
        &mut self,
        state: StateID,
        lexeme: LexemeIdx,
        budget: u64,
        reserve: &F,
    ) -> Result<bool, PreparedRegexVectorOperationError<E>> {
        if self.failed {
            return Err(PreparedRegexVectorOperationError {
                cause: Cause::Failed,
            });
        }
        self.failed = true;
        let result = (|| -> Result<bool, Cause<E>> {
            let _frame = reserve.frame(Self::state_operation_control_bytes::<F, E>().ok_or(Cause::Overflow)?)
                .map_err(Cause::frame)?;
            if !self.subsume_geometry(state) || self.pending.is_some() {
                return Err(Cause::Source);
            }
            let small = *self
                .input
                .roots
                .get(lexeme.as_usize())
                .ok_or(Cause::Source)?;
            subsumption::run(
                &mut Advance {
                    machine: self.machine.as_mut().ok_or(Cause::Failed)?,
                    candidate: &mut self.candidate,
                    reserve,
                    marker: PhantomData,
                },
                self.table
                    .as_ref()
                    .ok_or(Cause::Failed)?
                    .get(state.as_u32()),
                self.subsumable.as_ref().ok_or(Cause::Failed)?,
                small,
                budget,
            )
        })();
        match result {
            Ok(value) => {
                self.failed = false;
                Ok(value)
            }
            Err(cause) => Err(PreparedRegexVectorOperationError { cause }),
        }
    }
    fn query<T, F, E, G>(
        &mut self,
        reserve: &F,
        run: G,
    ) -> Result<T, PreparedRegexVectorOperationError<E>>
    where
        F: crate::earley::PreparedFunding<Error = E>,
        G: FnOnce(&mut Self, &F) -> Result<T, Cause<E>>,
    {
        if self.failed {
            return Err(PreparedRegexVectorOperationError {
                cause: Cause::Failed,
            });
        }
        self.failed = true;
        let result = (|| {
            let frames = [
                Self::state_operation_control_bytes::<F, E>().ok_or(Cause::Overflow)?,
                size_of::<G>(),
                size_of::<Result<T, Cause<E>>>(),
                size_of::<Result<T, PreparedRegexVectorOperationError<E>>>(),
                size_of::<(&mut Self, &F)>(),
            ];
            let _frame = reserve.frame(
                frames
                    .into_iter()
                    .try_fold(size_of_val(&frames), usize::checked_add)
                    .ok_or(Cause::Overflow)?,
            )
            .map_err(Cause::frame)?;
            run(self, reserve)
        })();
        match result {
            Ok(value) => {
                self.failed = false;
                Ok(value)
            }
            Err(cause) => Err(PreparedRegexVectorOperationError { cause }),
        }
    }
    fn state_index(&self, state: StateID) -> Option<usize> {
        (state.as_u32() != StateID::MISSING.as_u32()
            && state.as_usize() < self.states.len()
            && self.pending.is_none())
        .then_some(state.as_usize())
    }
    /// Same cached maximal hidden/lookahead length over actual state expressions.
    pub fn possible_lookahead_len<F: crate::earley::PreparedFunding<Error = E>, E>(
        &mut self,
        state: StateID,
        reserve: &F,
    ) -> Result<usize, PreparedRegexVectorOperationError<E>> {
        self.query(reserve, |owner, _| {
            let index = owner.state_index(state).ok_or(Cause::Source)?;
            if let Some(value) = owner.states[index].possible_lookahead_len {
                return Ok(value);
            }
            let value = descriptor::possible_hidden(
                owner.machine.as_ref().ok_or(Cause::Failed)?.source(),
                owner
                    .table
                    .as_ref()
                    .ok_or(Cause::Failed)?
                    .get(state.as_u32()),
            );
            owner.states[index].possible_lookahead_len = Some(value);
            Ok(value)
        })
    }
    /// Same first accepting lookahead length, retaining the ordinary optional cache.
    pub fn lookahead_len_for_state<F: crate::earley::PreparedFunding<Error = E>, E>(
        &mut self,
        state: StateID,
        reserve: &F,
    ) -> Result<Option<usize>, PreparedRegexVectorOperationError<E>> {
        self.query(reserve, |owner, _| {
            let index = owner.state_index(state).ok_or(Cause::Source)?;
            let desc = &mut owner.states[index];
            if desc.greedy_accepting.is_none() {
                return Ok(None);
            }
            if let Some(value) = desc.lookahead_len {
                return Ok(value);
            }
            let value = descriptor::lookahead(
                owner.machine.as_ref().ok_or(Cause::Failed)?.source(),
                owner
                    .table
                    .as_ref()
                    .ok_or(Cause::Failed)?
                    .get(state.as_u32()),
                &desc.greedy_accepting,
            );
            desc.lookahead_len = Some(value);
            Ok(value)
        })
    }
    /// Same next-byte approximation, with paid reached expression-cache slots.
    pub fn next_byte<F: crate::earley::PreparedFunding<Error = E>, E>(
        &mut self,
        state: StateID,
        reserve: &F,
    ) -> Result<NextByte, PreparedRegexVectorOperationError<E>> {
        self.query(reserve, |owner, reserve| {
            let index = owner.state_index(state).ok_or(Cause::Source)?;
            if let Some(value) = owner.states[index].next_byte {
                return Ok(value);
            }
            let value = descriptor::next_bytes(
                &mut Descriptor {
                    source: owner.machine.as_ref().ok_or(Cause::Failed)?.source(),
                    next: &mut owner.next,
                    reserve,
                    marker: PhantomData,
                },
                owner
                    .table
                    .as_ref()
                    .ok_or(Cause::Failed)?
                    .get(state.as_u32()),
            )?;
            owner.states[index].next_byte = Some(value);
            Ok(value)
        })
    }
    /// Same actual state subset, published through the shared fixed intern and
    /// descriptor path; no caller-provided row or root replaces the source.
    pub fn limit_state_to<F: crate::earley::PreparedFunding<Error = E>, E>(
        &mut self,
        state: StateID,
        allowed: &LexemeSet,
        reserve: &F,
    ) -> Result<StateID, PreparedRegexVectorOperationError<E>> {
        self.query(reserve, |owner, reserve| {
            owner.state_index(state).ok_or(Cause::Source)?;
            if allowed.len() != owner.input.roots.len() {
                return Err(Cause::Source);
            }
            let words = owner
                .table
                .as_ref()
                .ok_or(Cause::Failed)?
                .get(state.as_u32());
            let mut count = 0usize;
            descriptor::restricted(words, allowed, |_, _| {
                count = count.checked_add(2).ok_or(Cause::Overflow)?;
                Ok::<(), Cause<E>>(())
            })?;
            owner.candidate.clear();
            grow(&mut owner.candidate, count, reserve)?;
            descriptor::restricted(words, allowed, |index, root| {
                owner
                    .candidate
                    .push(u32::try_from(index.as_usize()).map_err(|_| Cause::Overflow)?);
                owner.candidate.push(root.as_u32());
                Ok::<(), Cause<E>>(())
            })?;
            if owner.candidate.len() != count {
                return Err(Cause::Source);
            }
            owner.publish_candidate(reserve)
        })
    }
    /// Same source-qualified weight used by the existing large-lexeme precompute policy.
    pub fn lexeme_weight<F: crate::earley::PreparedFunding<Error = E>, E>(
        &mut self,
        lexeme: LexemeIdx,
        reserve: &F,
    ) -> Result<u32, PreparedRegexVectorOperationError<E>> {
        self.query(reserve, |owner, reserve| {
            let root = *owner
                .input
                .roots
                .get(lexeme.as_usize())
                .ok_or(Cause::Source)?;
            reserve.reserve(
                PreparedExpressionMachine::weight_operation_control_bytes()
                    .ok_or(Cause::Overflow)?,
            )
            .map_err(Cause::Funding)?;
            owner
                .machine
                .as_mut()
                .ok_or(Cause::Failed)?
                .weight(root)
                .map_err(Cause::Weight)
        })
    }
    /// Actual retained expression source, including committed startup work.
    pub fn expressions(&self) -> &ExprSet {
        self.machine
            .as_ref()
            .map_or(&self.input.expressions, |m| m.source())
    }
    /// Same filtered root order selected by the ordinary constructor.
    pub fn roots(&self) -> &[ExprRef] {
        &self.input.roots
    }
    /// Same source-derived lazy lexeme selection.
    pub fn lazy_regexes(&self) -> &LexemeSet {
        self.lazy.as_ref().expect("completed constructor")
    }
    /// Same source-derived repeat eligibility.
    pub fn subsumable_regexes(&self) -> &LexemeSet {
        self.subsumable.as_ref().expect("completed constructor")
    }
    /// Borrow one actual completed descriptor. No arbitrary StateID indexing.
    pub fn state_desc(&self, state: StateID) -> Option<&StateDesc> {
        self.states.get(state.as_usize())
    }
}

#[cfg(test)]
mod tests {
    use crate::earley::PreparedFunding;
    use super::*;
    use crate::earley::{
        lexerspec::LexerRootSource,
        regexvec::{RegexVec, RxLexeme},
    };
    use derivre::RegexBuilder;
    use std::cell::Cell;
    fn input() -> RegexVectorInput {
        let mut builder = RegexBuilder::new(derivre::ParserAllocationFunding::unenforced()).unwrap();
        let mut rows = Vec::new();
        for (pattern, lazy) in [
            ("[ab]{0,2}", false),
            ("", false),
            ("", false),
            ("", false),
            ("", true),
            ("", true),
            ("", true),
            ("[^\\s\\S]", false),
            ("((abc|def|ghi|jkl|mno|pqr|stu|vwx){1,3})", false),
        ] {
            rows.push(RxLexeme {
                rx: builder.mk_regex(pattern).unwrap(),
                lazy,
                priority: 0,
            });
        }
        LexerRootSource::from_rows(rows, None)
            .unwrap()
            .ordinary(builder.exprset().clone())
            .unwrap()
    }
    fn selection(count: usize, indices: &[usize]) -> LexemeSet {
        let mut selected = LexemeSet::new(count);
        for &index in indices {
            selected.add(LexemeIdx::new(index));
        }
        selected
    }
    fn compare(left: &StateDesc, right: &StateDesc) {
        assert_eq!(left.state, right.state);
        assert_eq!(
            left.greedy_accepting.as_slice(),
            right.greedy_accepting.as_slice()
        );
        assert_eq!(
            left.lazy_accepting.as_slice(),
            right.lazy_accepting.as_slice()
        );
        assert_eq!(left.possible.vob.as_slice(), right.possible.vob.as_slice());
        assert_eq!(left.lazy_hidden_len, right.lazy_hidden_len);
        assert_eq!(left.has_special_token, right.has_special_token);
    }
    #[test]
    fn owning_vector_preserves_real_constructor_states_and_failed_publication_custody() {
        let calls = Cell::new(0usize);
        let spent = Cell::new(0usize);
        let reserve = |bytes: usize| -> Result<(), &'static str> {
            calls.set(calls.get() + 1);
            let next = spent.get().checked_add(bytes).ok_or("overflow")?;
            if next > 512 * 1024 * 1024 {
                return Err("fixture allowance");
            }
            spent.set(next);
            Ok(())
        };
        let mut limits = ParserLimits::default();
        let mut expected_limits = limits.clone();
        let mut ordinary = RegexVec::new_with_input(input(), &mut expected_limits).unwrap();
        let mut prepared = PreparedRegexVector::prepare_with_backing(input(), &mut limits, Some(ParserAllocationFunding::unenforced()), &reserve).unwrap();
        let constructor_calls = calls.get();
        assert_eq!(
            limits.initial_lexer_fuel,
            expected_limits.initial_lexer_fuel
        );
        assert_eq!(prepared.roots(), ordinary.rx_list);
        assert_eq!(prepared.expressions().cost(), ordinary.exprs.cost());
        assert_eq!(prepared.lazy_regexes().vob, ordinary.lazy.vob);
        assert_eq!(prepared.subsumable_regexes().vob, ordinary.subsumable.vob);
        assert_eq!(prepared.transitions, ordinary.state_table);
        compare(
            prepared.state_desc(StateID::DEAD).unwrap(),
            ordinary.state_desc(StateID::DEAD),
        );
        compare(
            prepared.state_desc(StateID::MISSING).unwrap(),
            ordinary.state_desc(StateID::MISSING),
        );
        for indices in [
            &[0, 1, 2, 3, 4, 5, 6, 7][..],
            &[1, 2, 3][..],
            &[0][..],
            &[][..],
        ] {
            let selected = selection(prepared.roots().len(), indices);
            let expected = ordinary.initial_state(&selected);
            let state = prepared.initial_state(&selected, &reserve).unwrap();
            assert_eq!(state, expected);
            compare(
                prepared.state_desc(state).unwrap(),
                ordinary.state_desc(expected),
            );
            assert_eq!(
                prepared.next_byte(state, &reserve).unwrap(),
                ordinary.next_byte(expected)
            );
            assert_eq!(
                prepared.possible_lookahead_len(state, &reserve).unwrap(),
                ordinary.possible_lookahead_len(expected)
            );
            assert_eq!(
                prepared.lookahead_len_for_state(state, &reserve).unwrap(),
                ordinary.lookahead_len_for_state(expected)
            );
            assert_eq!(
                prepared.table.as_ref().unwrap().get(state.as_u32()),
                ordinary.rx_sets.get(expected.as_u32())
            );
            assert_eq!(prepared.initial_state(&selected, &reserve).unwrap(), state);
        }
        assert_eq!(prepared.transitions, ordinary.state_table);
        assert_eq!(prepared.states.len(), ordinary.state_descs.len());
        let all = selection(prepared.roots().len(), &[0, 1, 2, 3, 4, 5, 6]);
        let start = prepared.initial_state(&all, &reserve).unwrap();
        let expected_start = ordinary.initial_state(&all);
        let selected = selection(prepared.roots().len(), &[0, 1]);
        assert_eq!(
            prepared.limit_state_to(start, &selected, &reserve).unwrap(),
            ordinary.limit_state_to(expected_start, &selected)
        );
        assert_eq!(
            prepared.lexeme_weight(LexemeIdx::new(0), &reserve).unwrap(),
            ordinary.lexeme_weight(LexemeIdx::new(0)).unwrap()
        );
        let selected = selection(prepared.roots().len(), &[0]);
        let mut state = prepared.initial_state(&selected, &reserve).unwrap();
        let mut expected = ordinary.initial_state(&selected);
        assert!(prepared.subsume_possible(state));
        assert_eq!(
            prepared
                .check_subsume(state, LexemeIdx::new(0), 100, &reserve)
                .unwrap(),
            ordinary
                .check_subsume(expected, LexemeIdx::new(0), 100)
                .unwrap()
        );
        for byte in b"aba" {
            let previous = state;
            state = prepared.transition(state, *byte, &reserve).unwrap();
            expected = ordinary.transition(expected, *byte);
            assert_eq!(state, expected);
            compare(
                prepared.state_desc(state).unwrap(),
                ordinary.state_desc(expected),
            );
            assert_eq!(
                prepared.next_byte(state, &reserve).unwrap(),
                ordinary.next_byte(expected)
            );
            assert_eq!(
                prepared.possible_lookahead_len(state, &reserve).unwrap(),
                ordinary.possible_lookahead_len(expected)
            );
            assert_eq!(
                prepared.lookahead_len_for_state(state, &reserve).unwrap(),
                ordinary.lookahead_len_for_state(expected)
            );
            assert_eq!(prepared.expressions().cost(), ordinary.exprs.cost());
            assert_eq!(prepared.fuel(), ordinary.get_fuel());
            let attempts = prepared.transitions_attempted();
            assert_eq!(
                prepared.transition(previous, *byte, &reserve).unwrap(),
                state
            );
            assert_eq!(prepared.transitions_attempted(), attempts);
            if prepared.subsume_possible(state) {
                assert_eq!(
                    prepared
                        .check_subsume(state, LexemeIdx::new(0), 0, &reserve)
                        .unwrap(),
                    ordinary
                        .check_subsume(expected, LexemeIdx::new(0), 0)
                        .unwrap()
                );
            }
        }
        for fail_at in [1, 2, constructor_calls] {
            let calls = Cell::new(0);
            let reject = |bytes| {
                calls.set(calls.get() + 1);
                if calls.get() == fail_at {
                    Err("constructor destination")
                } else {
                    reserve.reserve(bytes)
                }
            };
            let failure =
                PreparedRegexVector::prepare(input(), &mut ParserLimits::default(), &reject)
                    .unwrap_err();
            assert!(matches!(
                failure.cause,
                ConstructionCause::Operation(Cause::Funding("constructor destination"))
            ));
            let prefix = failure.prefix.as_ref().unwrap();
            assert_eq!(prefix.input.roots.len(), prepared.roots().len());
            if fail_at == constructor_calls {
                assert!(prefix.machine.is_some());
                assert!(prefix.table.is_some());
                assert_eq!(prefix.states.len(), 1);
            }
        }
        let mut failed =
            PreparedRegexVector::prepare(input(), &mut ParserLimits::default(), &reserve).unwrap();
        let selected = selection(failed.roots().len(), &[0, 1, 2, 3, 4, 5, 6]);
        let calls = Cell::new(0);
        let reject = |bytes| {
            calls.set(calls.get() + 1);
            if calls.get() == 5 {
                Err("descriptor destination")
            } else {
                reserve.reserve(bytes)
            }
        };
        let failure = failed.initial_state(&selected, &reject).unwrap_err();
        assert!(matches!(
            failure.cause,
            Cause::Funding("descriptor destination")
        ));
        assert_eq!(failed.table.as_ref().unwrap().len(), 3);
        assert_eq!(failed.states.len(), 2);
        assert!(failed.pending.as_ref().unwrap().desc.is_none());
        let before = spent.get();
        assert!(matches!(
            failed.initial_state(&selected, &reserve).unwrap_err().cause,
            Cause::Failed
        ));
        assert_eq!(spent.get(), before);
        let mut exhausted = PreparedRegexVector::prepare_with_backing(input(), &mut ParserLimits::default(), Some(ParserAllocationFunding::unenforced()), &reserve).unwrap();
        let selected = selection(exhausted.roots().len(), &[0]);
        let start = exhausted.initial_state(&selected, &reserve).unwrap();
        exhausted.set_fuel(0);
        let error = exhausted.transition(start, b'a', &reserve).unwrap_err();
        assert!(matches!(
            error.cause,
            Cause::Relevance(PreparedRelevanceError::Fuel { .. })
        ));
        assert_eq!(exhausted.fuel(), 0);
        assert_eq!(exhausted.transitions_attempted(), 1);
        let before = spent.get();
        assert!(matches!(
            exhausted
                .transition(start, b'a', &reserve)
                .unwrap_err()
                .cause,
            Cause::Failed
        ));
        assert_eq!(spent.get(), before);
        drop(ordinary);
        assert!(failed.expressions().is_valid(failed.roots()[0]));
        assert!(!prepared.failed);
    }
}

impl<E> Cause<E> {
 fn frame(error: crate::earley::FrameError<E>) -> Self { match error { crate::earley::FrameError::Overflow => Self::Overflow, crate::earley::FrameError::Funding(error) => Self::Funding(error) } }
}
