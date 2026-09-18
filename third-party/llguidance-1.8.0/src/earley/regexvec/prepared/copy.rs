//! Copies the actual lexical DFA and its owning expression machine.
use super::{grow, Cause as OperationCause, PreparedRegexVector, StateDesc, MatchingLexemes, LexemeSet};
use crate::earley::lexerspec::RegexVectorInput;
use derivre::{AlphabetInfo, raw::{ExprSet, ExprSetCopyFailure, HashConsCopyFailure, HashConsCapacityError, PreparedExprError, PreparedExpressionCopyFailure, ParserAllocationFunding}};
use std::{fmt, mem::{size_of, size_of_val}};
use toktrie::{SimpleVob, TokenMaskConstructionPlan};
#[derive(Debug)]
enum Cause<E> {
    Operation(OperationCause<E>), Input(ExprSetCopyFailure),
    Machine(PreparedExpressionCopyFailure<E>), Table(HashConsCopyFailure),
    Binding(HashConsCapacityError),
}
impl<E> From<OperationCause<E>> for Cause<E> {
    fn from(cause: OperationCause<E>) -> Self { Self::Operation(cause) }
}
/// A failed independent DFA copy owns every reached destination and the new
/// backing account. The borrowed original remains unchanged.
pub struct PreparedRegexVectorCopyFailure<E> {
    cause: Cause<E>, prefix: Option<PreparedRegexVector>, backing: Option<ParserAllocationFunding>,
}
impl<E: fmt::Debug> fmt::Debug for PreparedRegexVectorCopyFailure<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PreparedRegexVectorCopyFailure").field("cause", &self.cause)
            .field("retains_prefix", &self.prefix.is_some()).finish()
    }
}
impl<E: fmt::Display> fmt::Display for PreparedRegexVectorCopyFailure<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.cause { Cause::Operation(e) => fmt::Display::fmt(e,f), Cause::Input(e) => fmt::Display::fmt(e,f),
            Cause::Machine(e) => fmt::Display::fmt(e,f), Cause::Table(e) => fmt::Display::fmt(e,f), Cause::Binding(e) => fmt::Display::fmt(e,f) }
    }
}
impl<E: std::error::Error + 'static> std::error::Error for PreparedRegexVectorCopyFailure<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(match &self.cause { Cause::Operation(e) => e, Cause::Input(e) => e, Cause::Machine(e) => e, Cause::Table(e) => e, Cause::Binding(e) => e })
    }
}
fn fixed_controls<T: Copy, E>() -> Option<usize> {
    let frames = [size_of::<(&mut Vec<T>, &Vec<T>, &())>(), size_of::<T>(),
        size_of::<Result<(), OperationCause<E>>>(), size_of::<std::slice::Iter<'_, T>>()];
    frames.into_iter().try_fold(size_of_val(&frames), usize::checked_add)
}
pub(super) fn fixed_required_bytes<T: Copy, E>(source: &Vec<T>) -> Option<usize> {
    fixed_controls::<T, E>()?.checked_add(super::allocation_bytes::<T>(source.capacity())?)
}
fn fixed<T: Copy, F: Fn(usize) -> Result<(), E>, E>(
    target: &mut Vec<T>, source: &Vec<T>, funding: &F,
) -> Result<(), OperationCause<E>> {
    funding(fixed_controls::<T, E>().ok_or(OperationCause::Overflow)?)
        .map_err(OperationCause::Funding)?;
    grow(target, source.capacity(), funding)?;
    target.extend_from_slice(source);
    Ok(())
}
fn mask<F: Fn(usize) -> Result<(), E>, E>(
    source: &LexemeSet, funding: &F,
) -> Result<LexemeSet, OperationCause<E>> {
    let plan = source.copy_plan().map_err(OperationCause::MaskSource)?;
    funding(plan.requirements().required_bytes()).map_err(OperationCause::Funding)?;
    Ok(LexemeSet::from_owned_vob(plan.compile().map_err(OperationCause::Mask)?))
}
fn matching<F: Fn(usize) -> Result<(), E>, E>(
    target: &mut MatchingLexemes, source: &MatchingLexemes, funding: &F,
) -> Result<(), OperationCause<E>> {
    *target = match source {
        MatchingLexemes::None => MatchingLexemes::None,
        MatchingLexemes::One(v) => MatchingLexemes::One(*v),
        MatchingLexemes::Two(v) => MatchingLexemes::Two(*v),
        MatchingLexemes::Many(_) => MatchingLexemes::Many(Vec::new()),
    };
    if let (MatchingLexemes::Many(target), MatchingLexemes::Many(source)) = (target, source) {
        fixed(target, source, funding)?;
    }
    Ok(())
}
impl PreparedRegexVector {
    fn copy_ready(&self) -> bool {
        !self.failed && self.pending.is_none() && self.prepared.is_none() && self.plan.is_none()
            && self.machine.is_some() && self.table.is_some()
    }
    /// Exact prospective payment for the existing independent DFA copy. Reads
    /// only retained capacities and source plans; never constructs a destination
    /// or invokes a funding callback. None preserves incomplete-source refusal.
    pub fn copy_required_bytes<E>(&self) -> Option<usize> {
        if !self.copy_ready() { return None; }
        let mut bytes = Self::copy_controls::<E>()?
            .checked_add(self.input.expressions.source_copy_plan().ok()?.requirements().required_bytes())?
            .checked_add(fixed_required_bytes::<_, E>(&self.input.rows)?)?
            .checked_add(fixed_required_bytes::<_, E>(&self.input.roots)?)?
            .checked_add(self.machine.as_ref()?.copy_required_bytes::<E>()?)?
            .checked_add(self.table.as_ref()?.prepared_source_plan().ok()?.requirements().required_bytes())?;
        for source in [&self.lazy, &self.subsumable].into_iter().flatten() {
            bytes = bytes.checked_add(source.copy_plan().ok()?.requirements().required_bytes())?;
        }
        bytes = bytes.checked_add(super::allocation_bytes::<StateDesc>(self.states.capacity())?)?;
        for source in &self.states {
            bytes = bytes.checked_add(source.possible.copy_plan().ok()?.requirements().required_bytes())?;
            for matching in [&source.greedy_accepting, &source.lazy_accepting] {
                if let MatchingLexemes::Many(values) = matching {
                    bytes = bytes.checked_add(fixed_required_bytes::<_, E>(values)?)?;
                }
            }
        }
        bytes.checked_add(fixed_required_bytes::<_, E>(&self.transitions)?)?
            .checked_add(fixed_required_bytes::<_, E>(&self.next)?)?
            .checked_add(fixed_required_bytes::<_, E>(&self.candidate)?)
    }
    fn copy_controls<E>() -> Option<usize> {
            let parts = [
                Self::construction_control_bytes::<(), E>()?,
                size_of::<Self>(), size_of::<&Self>(), size_of::<Cause<E>>(),
                size_of::<PreparedRegexVectorCopyFailure<E>>(),
                size_of::<Result<Self, PreparedRegexVectorCopyFailure<E>>>(),
                size_of::<Result<(), Cause<E>>>(), size_of::<Option<Self>>(),
                size_of::<AlphabetInfo>(), size_of::<RegexVectorInput>(),
                size_of::<Result<ExprSet, ExprSetCopyFailure>>(),
                size_of::<Result<derivre::raw::PreparedExpressionMachine, PreparedExpressionCopyFailure<E>>>(),
                size_of::<std::slice::Iter<'_, StateDesc>>(),
                size_of::<(&mut StateDesc, &StateDesc)>(),
                size_of::<(&mut MatchingLexemes, &MatchingLexemes, &())>(),
                size_of::<Result<(), OperationCause<E>>>(),
                size_of::<Result<LexemeSet, OperationCause<E>>>(),
                size_of::<PreparedExprError>(), size_of::<SimpleVob>(),
            ];
        parts.into_iter().try_fold(size_of_val(&parts), usize::checked_add)
    }
    /// Copies the exact graph, state IDs, intern table, descriptors, transition
    /// cache and source configuration. Future growth uses only `backing`, never
    /// the original account. The enclosing owner retains copy funding.
    pub fn try_copy<F: Fn(usize) -> Result<(), E>, E>(
        &self, backing: Option<ParserAllocationFunding>, funding: &F,
    ) -> Result<Self, PreparedRegexVectorCopyFailure<E>> {
        let mut prefix = None;
        let result = (|| -> Result<(), Cause<E>> {
            funding(Self::copy_controls::<E>().ok_or(OperationCause::Overflow)?)
                .map_err(OperationCause::Funding)?;
            if !self.copy_ready() { return Err(OperationCause::Source.into()); }
            let input = self.input.expressions.source_copy_plan().map_err(Cause::Input)?;
            funding(input.requirements().required_bytes()).map_err(OperationCause::Funding)?;
            let input = RegexVectorInput {
                alpha: self.input.alpha.clone(), expressions: input.compile().map_err(Cause::Input)?,
                rows: Vec::new(), roots: Vec::new(), special_token_rx: self.input.special_token_rx,
            };
            prefix = Some(Self {
                machine: None, prepared: None, plan: None, table: None, lazy: None, subsumable: None,
                states: Vec::new(), transitions: Vec::new(), next: Vec::new(), candidate: Vec::new(), pending: None,
                max_states: self.max_states, fuel: self.fuel, transitions_attempted: self.transitions_attempted,
                failed: true, input,
            });
            let target = prefix.as_mut().expect("copy vector prefix");
            fixed(&mut target.input.rows, &self.input.rows, funding)?;
            fixed(&mut target.input.roots, &self.input.roots, funding)?;
            target.machine = Some(self.machine.as_ref().expect("checked machine").try_copy(backing.clone(), funding).map_err(Cause::Machine)?);
            let table = self.table.as_ref().expect("checked table").prepared_source_plan().map_err(Cause::Table)?;
            funding(table.requirements().required_bytes()).map_err(OperationCause::Funding)?;
            target.table = Some(table.compile().map_err(Cause::Table)?);
            if let Some(backing) = &backing {
                target.table.as_mut().expect("copied table").bind_backing_funding(backing.clone()).map_err(Cause::Binding)?;
            }
            if let Some(source) = &self.lazy { target.lazy = Some(mask(source, funding)?); }
            if let Some(source) = &self.subsumable { target.subsumable = Some(mask(source, funding)?); }
            grow(&mut target.states, self.states.capacity(), funding)?;
            for source in &self.states {
                let possible = mask(&source.possible, funding)?;
                target.states.push(StateDesc::empty(source.state, possible));
                let desc = target.states.last_mut().expect("copied state");
                matching(&mut desc.greedy_accepting, &source.greedy_accepting, funding)?;
                matching(&mut desc.lazy_accepting, &source.lazy_accepting, funding)?;
                desc.lazy_hidden_len = source.lazy_hidden_len;
                desc.has_special_token = source.has_special_token;
                desc.possible_lookahead_len = source.possible_lookahead_len;
                desc.lookahead_len = source.lookahead_len;
                desc.next_byte = source.next_byte;
            }
            fixed(&mut target.transitions, &self.transitions, funding)?;
            fixed(&mut target.next, &self.next, funding)?;
            fixed(&mut target.candidate, &self.candidate, funding)?;
            target.failed = self.failed;
            Ok(())
        })();
        match result { Ok(()) => Ok(prefix.expect("complete vector copy")),
            Err(cause) => Err(PreparedRegexVectorCopyFailure { cause, prefix, backing }) }
    }
}
