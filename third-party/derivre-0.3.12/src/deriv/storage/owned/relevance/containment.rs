//! Source-qualified containment consumer with paid reached destinations.
use super::{allocate, entry, Frame, Parts, PreparedRelevanceError, Search};
use crate::{
    ast::{ExprRef, ExprSet},
    relevance::containment::{self, branches, length},
};
use std::{
    alloc::Layout,
    marker::PhantomData,
    mem::{size_of, size_of_val},
};
pub(super) struct State {
    stack: Vec<ExprRef>,
    rows: Vec<(ExprRef, u32)>,
    lengths: Vec<length::Task>,
    nodes: usize,
    arguments: usize,
    fuel_refusal: Option<u64>,
}
impl State {
    fn new(nodes: usize, arguments: usize) -> Self {
        Self {
            stack: Vec::new(),
            rows: Vec::new(),
            lengths: Vec::new(),
            nodes,
            arguments,
            fuel_refusal: None,
        }
    }
}
struct Branches<'a, E> {
    stack: &'a mut Vec<ExprRef>,
    rows: Option<&'a mut Vec<(ExprRef, u32)>>,
    count: &'a mut usize,
    marker: PhantomData<E>,
}
impl<E> branches::Destination for Branches<'_, E> {
    type Error = PreparedRelevanceError<E>;
    fn push(&mut self, node: ExprRef) -> Result<(), Self::Error> {
        if self.stack.len() == self.stack.capacity() {
            return Err(Self::Error::Capacity);
        }
        self.stack.push(node);
        Ok(())
    }
    fn pop(&mut self) -> Option<ExprRef> {
        self.stack.pop()
    }
    fn leftover(&mut self, node: ExprRef, maximum: u32) -> Result<(), Self::Error> {
        *self.count = self.count.checked_add(1).ok_or(Self::Error::Overflow)?;
        if let Some(rows) = self.rows.as_mut() {
            if rows.len() == rows.capacity() {
                return Err(Self::Error::Capacity);
            }
            rows.push((node, maximum));
        }
        Ok(())
    }
}
struct Length<E>(PhantomData<E>);
impl<E> length::Destination for Length<E> {
    type Error = PreparedRelevanceError<E>;
    fn push(
        &mut self,
        todo: &mut Vec<length::Task>,
        task: length::Task,
    ) -> Result<(), Self::Error> {
        if todo.len() == todo.capacity() {
            return Err(Self::Error::Capacity);
        }
        todo.push(task);
        Ok(())
    }
    fn add(&mut self, left: usize, right: usize) -> Result<usize, Self::Error> {
        left.checked_add(right).ok_or(Self::Error::Overflow)
    }
}
struct Consumer<'a, F, E> {
    source: &'a mut ExprSet,
    derivative: &'a mut Option<super::DerivativeParts>,
    symbolic: &'a mut Option<super::symbolic::Parts>,
    weights: &'a mut super::weights::Parts,
    cache: &'a mut Vec<Option<bool>>,
    search: &'a mut Option<Search>,
    entry: &'a mut Option<entry::State>,
    state: &'a mut State,
    reserve: &'a F,
    max_fuel: u64,
    cost_limit: u64,
    marker: PhantomData<E>,
}
impl<F: Fn(usize) -> Result<(), E>, E> containment::Context for Consumer<'_, F, E> {
    type Error = PreparedRelevanceError<E>;
    fn source(&self) -> &ExprSet {
        self.source
    }
    fn derivative(&mut self, root: ExprRef, byte: u8) -> Result<ExprRef, Self::Error> {
        (self.reserve)(
            super::super::PreparedExpressionMachine::operation_control_bytes()
                .ok_or(Self::Error::Overflow)?,
        )
        .map_err(Self::Error::Funding)?;
        let mut parts = self.derivative.take().ok_or(Self::Error::Failed)?;
        if let Err(error) = parts.prepare_reached(self.source) {
            *self.derivative = Some(parts);
            return Err(Self::Error::Expression(error));
        }
        let mut storage = parts.bind(self.source);
        let result = storage.derivative_prepared(root, byte);
        *self.derivative = Some(storage.into_parts());
        result.map_err(|cause| {
            Self::Error::Derivative(super::super::PreparedExpressionOperationError { cause })
        })
    }
    fn not(&mut self, root: ExprRef) -> Result<ExprRef, Self::Error> {
        self.source
            .try_mk_not(root)
            .map_err(Self::Error::Expression)
    }
    fn and2(&mut self, a: ExprRef, b: ExprRef) -> Result<ExprRef, Self::Error> {
        self.source
            .try_mk_and2(a, b)
            .map_err(Self::Error::Expression)
    }
    fn repeat(&mut self, root: ExprRef, min: u32, max: u32) -> Result<ExprRef, Self::Error> {
        self.source
            .try_mk_repeat(root, min, max)
            .map_err(Self::Error::Expression)
    }
    fn non_empty(&mut self, root: ExprRef) -> Result<bool, Self::Error> {
        (self.reserve)(entry::operation_control_bytes::<F, E>().ok_or(Self::Error::Overflow)?)
            .map_err(Self::Error::Funding)?;
        entry::run_entry(
            &mut Frame {
                source: self.source,
                derivative: self.derivative.as_mut().ok_or(Self::Error::Failed)?,
                symbolic: self.symbolic,
                weights: self.weights,
                cache: self.cache,
                reserve: self.reserve,
                max_fuel: self.max_fuel,
                cost_limit: self.cost_limit,
                marker: PhantomData,
            },
            self.search,
            self.entry,
            root,
        )
    }
    fn max_length(&mut self, root: ExprRef) -> Result<Option<usize>, Self::Error> {
        if self.state.lengths.capacity() == 0 {
            let count = self
                .state
                .nodes
                .checked_mul(2)
                .and_then(|n| n.checked_add(1))
                .ok_or(Self::Error::Overflow)?;
            (self.reserve)(
                Layout::array::<length::Task>(count)
                    .map_err(|_| Self::Error::Overflow)?
                    .size(),
            )
            .map_err(Self::Error::Funding)?;
            allocate(&mut self.state.lengths, count)?;
        }
        length::run(
            self.source,
            root,
            &mut self.state.lengths,
            &mut Length(PhantomData),
        )
    }
    fn branches(&mut self, main: ExprRef, head: ExprRef) -> Result<Option<ExprRef>, Self::Error> {
        if self.state.stack.capacity() == 0 {
            let count = self
                .state
                .nodes
                .checked_mul(self.state.arguments)
                .and_then(|n| n.checked_add(1))
                .ok_or(Self::Error::Overflow)?;
            (self.reserve)(
                Layout::array::<ExprRef>(count)
                    .map_err(|_| Self::Error::Overflow)?
                    .size(),
            )
            .map_err(Self::Error::Funding)?;
            allocate(&mut self.state.stack, count)?;
        }
        self.state.stack.clear();
        let mut count = 0;
        let matched = branches::run(
            self.source,
            main,
            head,
            &mut Branches {
                stack: &mut self.state.stack,
                rows: None,
                count: &mut count,
                marker: PhantomData::<E>,
            },
        )?;
        if matched.is_some() {
            return Ok(matched);
        }
        (self.reserve)(
            Layout::array::<(ExprRef, u32)>(count)
                .map_err(|_| Self::Error::Overflow)?
                .size(),
        )
        .map_err(Self::Error::Funding)?;
        self.state.rows = Vec::new();
        allocate(&mut self.state.rows, count)?;
        self.state.stack.clear();
        let mut filled = 0;
        let matched = branches::run(
            self.source,
            main,
            head,
            &mut Branches {
                stack: &mut self.state.stack,
                rows: Some(&mut self.state.rows),
                count: &mut filled,
                marker: PhantomData::<E>,
            },
        )?;
        if matched.is_some() || filled != count {
            return Err(Self::Error::Source);
        }
        Ok(None)
    }
    fn leftovers(&self) -> &[(ExprRef, u32)] {
        &self.state.rows
    }
}
pub(crate) fn operation_control_bytes<F, E>() -> Option<usize> {
    let frames = [
        super::operation_control_bytes::<F, E>()?,
        entry::operation_control_bytes::<F, E>()?,
        size_of::<State>(),
        size_of::<Consumer<'_, F, E>>(),
        size_of::<Branches<'_, E>>(),
        size_of::<Length<E>>(),
        size_of::<length::Task>() * 2,
        size_of::<crate::ast::Expr<'_>>() * 4,
        size_of::<ExprRef>() * 12,
        size_of::<usize>() * 8,
        size_of::<u32>() * 4,
        size_of::<Option<usize>>(),
        size_of::<Option<ExprRef>>(),
        size_of::<crate::NextByte>(),
        size_of::<Result<Option<usize>, PreparedRelevanceError<E>>>(),
        size_of::<Result<Option<ExprRef>, PreparedRelevanceError<E>>>(),
    ];
    frames
        .into_iter()
        .try_fold(size_of_val(&frames), usize::checked_add)
}
impl Parts {
    fn cache_containment<F: Fn(usize) -> Result<(), E>, E>(
        &mut self,
        small: ExprRef,
        big: ExprRef,
        value: bool,
        reserve: &F,
    ) -> Result<(), PreparedRelevanceError<E>> {
        let count = self
            .contained
            .len()
            .checked_add(1)
            .ok_or(PreparedRelevanceError::Overflow)?;
        reserve(
            Layout::array::<(ExprRef, ExprRef, bool)>(count)
                .map_err(|_| PreparedRelevanceError::Overflow)?
                .size(),
        )
        .map_err(PreparedRelevanceError::Funding)?;
        self.contained
            .try_reserve_exact(1)
            .map_err(PreparedRelevanceError::Allocation)?;
        if self.contained.capacity() != count {
            return Err(PreparedRelevanceError::Capacity);
        }
        self.contained.push((small, big, value));
        Ok(())
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn contained<F: Fn(usize) -> Result<(), E>, E>(
        &mut self,
        source: &mut ExprSet,
        derivative: &mut Option<super::DerivativeParts>,
        symbolic: &mut Option<super::symbolic::Parts>,
        weights: &mut super::weights::Parts,
        small: ExprRef,
        big: ExprRef,
        max_fuel: u64,
        cache_failures: bool,
        reserve: &F,
    ) -> Result<bool, PreparedRelevanceError<E>> {
        if self.search.is_some() || self.entry.is_some() || self.containment.is_some() {
            return Err(PreparedRelevanceError::Failed);
        }
        if !source.is_valid(small) || !source.is_valid(big) {
            return Err(PreparedRelevanceError::Source);
        }
        if let Some(&(_, _, value)) = self
            .contained
            .iter()
            .find(|&&(a, b, _)| a == small && b == big)
        {
            return Ok(value);
        }
        let (_, nodes, encoded) = source
            .storage_extents();
        self.containment = Some(State::new(nodes, encoded));
        let cost_limit = source.cost().saturating_add(max_fuel);
        let result = containment::run(
            &mut Consumer {
                source,
                derivative,
                symbolic,
                weights,
                cache: &mut self.cache,
                search: &mut self.search,
                entry: &mut self.entry,
                state: self.containment.as_mut().expect("containment state"),
                reserve,
                max_fuel,
                cost_limit,
                marker: PhantomData,
            },
            small,
            big,
        );
        match result {
            Ok(value) => {
                self.cache_containment(small, big, value, reserve)?;
                self.containment = None;
                Ok(value)
            }
            Err(PreparedRelevanceError::Fuel { max_fuel }) => {
                self.containment
                    .as_mut()
                    .expect("retained containment")
                    .fuel_refusal = Some(max_fuel);
                if cache_failures {
                    self.cache_containment(small, big, false, reserve)?;
                }
                // Completed host-only temporaries may retire for ordinary
                // subsumption recovery. The source and charged account remain.
                self.search = None;
                self.entry = None;
                self.containment = None;
                Err(PreparedRelevanceError::Fuel { max_fuel })
            }
            Err(error) => Err(error),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ast::{Expr, ExprFlags},
        raw::{DerivCache, PreparedExpressionPlan, RelevanceCache},
        AlphabetInfo,
    };
    use std::cell::Cell;

    #[test]
    fn owning_containment_preserves_prefix_branches_lengths_cache_and_fuel_recovery() {
        let mut source = ExprSet::new(256, crate::ParserAllocationFunding::unenforced()).unwrap();
        let a = source.mk_byte(b'a').unwrap();
        let b = source.mk_byte(b'b').unwrap();
        let c = source.mk_byte(b'c').unwrap();
        let ab = source.mk_byte_set_or(&[a, b]).unwrap();
        let bc = source.mk_byte_set_or(&[b, c]).unwrap();
        let abc = source.mk_byte_set_or(&[a, b, c]).unwrap();
        let small = source.mk_repeat(ab, 1, 2).unwrap();
        let main = source.mk_repeat(abc, 1, 4).unwrap();
        let partial = source.mk_repeat(bc, 1, 4).unwrap();
        let tail = source.mk_repeat(ab, 0, 2).unwrap();
        let head_tail = source.mk_concat(ab, tail).unwrap();
        let main_tail = source.mk_repeat(abc, 0, 4).unwrap();
        let matching_head = source.mk_concat(ab, main_tail).unwrap();
        let prefix = source.mk_byte_literal(b"xy").unwrap();
        let prefixed_small = source.mk_concat(prefix, small).unwrap();
        let prefixed_big = source.mk_concat(prefix, main).unwrap();
        let short = source.mk_byte_literal(b"pq").unwrap();
        let choices = source.mk_or(&mut vec![short, c]).unwrap();
        let not_choices = source.mk_not(choices).unwrap();
        let excepted = source.mk_and2(main, not_choices).unwrap();
        let not_repeat = source.mk_not(partial).unwrap();
        let unbounded_except = source.mk_and2(main, not_repeat).unwrap();
        let suffix = source.mk_byte(b'z').unwrap();
        let main_suffix = source.mk_concat(main, suffix).unwrap();
        let except_suffix = source.mk_concat(choices, suffix).unwrap();
        let not_suffix = source.mk_not(except_suffix).unwrap();
        let suffixed = source.mk_and2(main_suffix, not_suffix).unwrap();
        let optional = source.mk_repeat(bc, 0, 1).unwrap();
        let optional_main = source.mk_concat(optional, main).unwrap();
        let cases = [
            (small, main, true),
            (small, partial, false),
            (head_tail, matching_head, true),
            (head_tail, main_tail, true),
            (prefixed_small, prefixed_big, true),
            (small, excepted, true),
            (small, unbounded_except, false),
            (small, suffixed, true),
            (small, optional_main, true),
        ];
        // This retained declaration supplies real initialized source backing;
        // reserve below changes only the existing ordinary table geometry.
        let declaration = source.mk(Expr::Or(ExprFlags::POSITIVE, &[main; 128])).unwrap();
        let mut roots = Vec::new();
        for &(left, right, _) in &cases {
            roots.extend([left, right]);
        }
        roots.push(declaration);
        let (_, mut source, _) = AlphabetInfo::from_exprset(source, &roots).unwrap();
        source.reserve(128).unwrap();
        let spent = Cell::new(0usize);
        let reserve = |bytes: usize| -> Result<(), &'static str> {
            let next = spent.get().checked_add(bytes).ok_or("overflow")?;
            if next > 512 * 1024 * 1024 {
                return Err("fixture allowance");
            }
            spent.set(next);
            Ok(())
        };
        let compile = || {
            let prepared = source.prepared_source_plan().unwrap().compile().unwrap();
            let plan = PreparedExpressionPlan::prepare(prepared).unwrap();
            reserve(plan.requirements().required_bytes()).unwrap();
            plan.compile().unwrap()
        };
        let mut machine = compile();
        let mut ordinary = source.clone();
        let mut cache = RelevanceCache::new();
        let mut derivative = DerivCache::new();
        for (left, right, expected) in cases {
            assert_eq!(
                cache
                    .is_contained_in_prefixes(
                        &mut ordinary,
                        &mut derivative,
                        left,
                        right,
                        u64::MAX,
                        false
                    )
                    .unwrap(),
                expected
            );
            assert_eq!(
                machine
                    .is_contained_in_prefixes(left, right, u64::MAX, false, &reserve)
                    .unwrap(),
                expected
            );
            assert_eq!(machine.source().cost(), ordinary.cost());
        }
        let cost = machine.source().cost();
        let pairs = machine.relevance.as_ref().unwrap().contained.len();
        assert!(machine
            .is_contained_in_prefixes(small, main, 0, false, &reserve)
            .unwrap());
        assert_eq!(machine.source().cost(), cost);
        assert_eq!(machine.relevance.as_ref().unwrap().contained.len(), pairs);
        assert!(machine.relevance.as_ref().unwrap().containment.is_none());

        for cache_failures in [false, true] {
            let mut machine = compile();
            let mut ordinary = source.clone();
            let mut cache = RelevanceCache::new();
            let mut derivative = DerivCache::new();
            assert!(cache
                .is_contained_in_prefixes(
                    &mut ordinary,
                    &mut derivative,
                    small,
                    main,
                    0,
                    cache_failures
                )
                .is_err());
            assert!(matches!(
                machine.is_contained_in_prefixes(small, main, 0, cache_failures, &reserve),
                Err(PreparedRelevanceError::Fuel { max_fuel: 0 })
            ));
            assert_eq!(machine.source().cost(), ordinary.cost());
            assert!(!machine.failed);
            let expected = cache
                .is_contained_in_prefixes(
                    &mut ordinary,
                    &mut derivative,
                    small,
                    main,
                    u64::MAX,
                    cache_failures,
                )
                .unwrap();
            assert_eq!(expected, !cache_failures);
            assert_eq!(
                machine
                    .is_contained_in_prefixes(small, main, u64::MAX, cache_failures, &reserve)
                    .unwrap(),
                expected
            );
            assert_eq!(machine.source().cost(), ordinary.cost());
        }

        let mut failed = compile();
        let calls = Cell::new(0);
        let reject = |bytes| {
            calls.set(calls.get() + 1);
            if calls.get() == 2 {
                Err("branch destination refused")
            } else {
                reserve(bytes)
            }
        };
        assert!(matches!(
            failed.is_contained_in_prefixes(head_tail, main_tail, u64::MAX, true, &reject),
            Err(PreparedRelevanceError::Funding(
                "branch destination refused"
            ))
        ));
        assert!(failed.relevance.as_ref().unwrap().containment.is_some());
        assert!(failed.relevance.as_ref().unwrap().contained.is_empty());
        let spent_before = spent.get();
        assert!(matches!(
            failed.is_contained_in_prefixes(small, main, u64::MAX, false, &reserve),
            Err(PreparedRelevanceError::Failed)
        ));
        assert_eq!(spent.get(), spent_before);
        drop(source);
        drop(ordinary);
        assert!(failed.source().is_valid(head_tail));
        assert!(machine.source().is_valid(suffixed));
    }
}

impl State {
    pub(super) fn copy_required_bytes<E>(&self) -> Option<usize> {
        use crate::copy_storage as copy;
        copy::frame_bytes::<(&mut Option<Self>, &Self, Self), E>()?
            .checked_add(copy::fixed_required_bytes::<_, E>(&self.stack)?)?
            .checked_add(copy::fixed_required_bytes::<_, E>(&self.rows)?)?
            .checked_add(copy::fixed_required_bytes::<_, E>(&self.lengths)?)
    }
}
impl State {
    pub(super) fn copy_into<F: Fn(usize) -> Result<(), E>, E>(
        target: &mut Option<Self>, source: &Self, funding: &F,
    ) -> Result<(), crate::copy_storage::Error<E>> {
        use crate::copy_storage as copy;
        copy::frame::<(&mut Option<Self>, &Self, Self), _, _>(funding)?;
        let target = target.get_or_insert_with(|| Self::new(source.nodes, source.arguments));
        copy::fixed(&mut target.stack, &source.stack, funding)?;
        copy::fixed(&mut target.rows, &source.rows, funding)?;
        copy::fixed(&mut target.lengths, &source.lengths, funding)?;
        target.nodes = source.nodes;
        target.arguments = source.arguments;
        target.fuel_refusal = source.fuel_refusal;
        Ok(())
    }
}
