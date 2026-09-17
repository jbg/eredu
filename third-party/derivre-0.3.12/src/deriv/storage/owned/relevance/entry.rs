//! Paid full entry over the existing owning source and search contexts.
use super::{allocate, run_search, Frame, Parts, PreparedRelevanceError, Search};
use crate::{
    ast::{ExprRef, ExprSet},
    relevance::{entry, walk::Context as WalkContext},
};
use std::{
    alloc::Layout,
    marker::PhantomData,
    mem::{size_of, size_of_val},
};
pub(super) struct State {
    concat: Vec<ExprRef>,
    probes: Vec<ExprRef>,
    nodes: usize,
}
impl State {
    fn empty(nodes: usize) -> Self {
        Self {
            concat: Vec::new(),
            probes: Vec::new(),
            nodes,
        }
    }
}
struct Consumer<'a, 's, F, E> {
    frame: &'a mut Frame<'s, F, E>,
    search: &'a mut Option<Search>,
    state: &'a mut State,
}
impl<F: Fn(usize) -> Result<(), E>, E> entry::Context for Consumer<'_, '_, F, E> {
    type Error = PreparedRelevanceError<E>;
    fn source(&self) -> &ExprSet {
        self.frame.source
    }
    fn cached(&self, root: ExprRef) -> Option<bool> {
        self.frame.cached(root)
    }
    fn cache_true(&mut self, root: ExprRef) -> Result<(), Self::Error> {
        self.frame.cache(root, true)
    }
    fn begin_concat(&mut self, count: usize) -> Result<(), Self::Error> {
        let bytes = Layout::array::<ExprRef>(count)
            .map_err(|_| Self::Error::Overflow)?
            .size();
        (self.frame.reserve)(bytes).map_err(Self::Error::Funding)?;
        self.state.concat = Vec::new();
        allocate(&mut self.state.concat, count)
    }
    fn copy_concat(&mut self, root: ExprRef) -> Result<(), Self::Error> {
        let dest = &mut self.state.concat;
        entry::copy_filtered(self.frame.source, root, |child| {
            if dest.len() == dest.capacity() {
                return Err(Self::Error::Capacity);
            }
            dest.push(child);
            Ok(())
        })
    }
    fn fold_concat(&mut self) -> Result<ExprRef, Self::Error> {
        self.frame
            .derivative
            .buffers
            .concat
            .as_mut()
            .ok_or(Self::Error::Failed)?
            .fold_expressions(self.frame.source, &self.state.concat)
            .map_err(Self::Error::Expression)
    }
    fn and(&mut self, left: ExprRef, right: ExprRef) -> Result<ExprRef, Self::Error> {
        let args = &mut self.frame.derivative.buffers.alternatives;
        if args.capacity() < 2 {
            return Err(Self::Error::Capacity);
        }
        args.clear();
        args.push(left);
        args.push(right);
        self.frame
            .derivative
            .buffers
            .nary
            .as_mut()
            .ok_or(Self::Error::Failed)?
            .apply_to_vec(self.frame.source, args, true)
            .map_err(Self::Error::Expression)
    }
    fn push_probe(&mut self, root: ExprRef) -> Result<(), Self::Error> {
        if self.state.probes.capacity() == 0 {
            let bytes = Layout::array::<ExprRef>(self.state.nodes)
                .map_err(|_| Self::Error::Overflow)?
                .size();
            (self.frame.reserve)(bytes).map_err(Self::Error::Funding)?;
            allocate(&mut self.state.probes, self.state.nodes)?;
        }
        if self.state.probes.len() == self.state.probes.capacity() {
            let total = self
                .state
                .probes
                .len()
                .checked_add(1)
                .ok_or(Self::Error::Overflow)?;
            self.frame
                .source
                .grow_prepared_workspace(&mut self.state.probes, total)
                .map_err(Self::Error::Expression)?;
        }
        self.state.probes.push(root);
        Ok(())
    }
    fn pop_probe(&mut self) -> Option<ExprRef> {
        self.state.probes.pop()
    }
    fn walk(&mut self, root: ExprRef) -> Result<bool, Self::Error> {
        run_search(self.frame, self.search, root)
    }
    fn probe_refusal(&mut self, error: Self::Error) -> Result<(), Self::Error> {
        if matches!(error, Self::Error::Fuel { .. }) {
            // Only temporary failed-search buffers retire. Source expressions,
            // memo entries and cumulative reservations are never rolled back.
            *self.search = None;
            Ok(())
        } else {
            Err(error)
        }
    }
}
pub(crate) fn operation_control_bytes<F, E>() -> Option<usize> {
    let frames = [
        super::operation_control_bytes::<F, E>()?,
        crate::deriv::storage::Storage::prepared_operation_control_bytes()?,
        size_of::<State>(),
        size_of::<Consumer<'_, '_, F, E>>(),
        size_of::<crate::simplify::ConcatIter<'_>>(),
        size_of::<crate::simplify::ConcatElement<'_>>(),
        size_of::<crate::ast::Expr<'_>>() * 3,
        size_of::<ExprRef>() * 6,
        size_of::<Option<ExprRef>>(),
        size_of::<usize>() * 2,
        size_of::<Result<Option<ExprRef>, PreparedRelevanceError<E>>>(),
    ];
    frames
        .into_iter()
        .try_fold(size_of_val(&frames), usize::checked_add)
}
impl Parts {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn non_empty<F: Fn(usize) -> Result<(), E>, E>(
        &mut self,
        source: &mut ExprSet,
        derivative: &mut super::DerivativeParts,
        symbolic: &mut Option<super::symbolic::Parts>,
        weights: &mut super::weights::Parts,
        root: ExprRef,
        max_fuel: u64,
        reserve: &F,
    ) -> Result<bool, PreparedRelevanceError<E>> {
        let cost_limit = source.cost().saturating_add(max_fuel);
        run_entry(
            &mut Frame {
                source,
                derivative,
                symbolic,
                weights,
                cache: &mut self.cache,
                reserve,
                max_fuel,
                cost_limit,
                marker: PhantomData,
            },
            &mut self.search,
            &mut self.entry,
            root,
        )
    }
}

pub(super) fn run_entry<F: Fn(usize) -> Result<(), E>, E>(
    frame: &mut Frame<'_, F, E>,
    search: &mut Option<Search>,
    state: &mut Option<State>,
    root: ExprRef,
) -> Result<bool, PreparedRelevanceError<E>> {
    if state.is_some() || search.is_some() {
        return Err(PreparedRelevanceError::Failed);
    }
    if !frame.source.is_valid(root) {
        return Err(PreparedRelevanceError::Source);
    }
    frame
        .source
        .grow_prepared_workspace(frame.cache, frame.source.len())
        .map_err(PreparedRelevanceError::Expression)?;
    if frame.cache.len() < frame.source.len() {
        frame.cache.resize(frame.source.len(), None);
    }
    *state = Some(State::empty(frame.cache.len()));
    let result = entry::run(
        &mut Consumer {
            frame,
            search,
            state: state.as_mut().expect("entry destination"),
        },
        root,
    );
    if result.is_ok() {
        *state = None;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ast::{Expr, ExprFlags},
        raw::{PreparedExpressionPlan, RelevanceCache},
        AlphabetInfo,
    };
    use std::cell::Cell;
    #[test]
    fn full_entry_preserves_concat_nested_repeat_probe_and_recoverable_fuel_order() {
        let mut source = ExprSet::new(256);
        let a = source.mk_byte(b'a');
        let b = source.mk_byte(b'b');
        let c = source.mk_byte(b'c');
        let d = source.mk_byte(b'd');
        let ab = source.mk_byte_set_or(&[a, b]);
        let bc = source.mk_byte_set_or(&[b, c]);
        let cd = source.mk_byte_set_or(&[c, d]);
        let left = source.mk_repeat(ab, 1, 2);
        let right = source.mk_repeat(bc, 1, 3);
        let disjoint = source.mk_repeat(cd, 1, 2);
        let repeated = source.mk_and(&mut vec![left, right]);
        let empty = source.mk_and(&mut vec![left, disjoint]);
        let nested_left = source.mk_repeat(left, 1, 2);
        let nested_right = source.mk_repeat(right, 1, 2);
        let nested = source.mk_and(&mut vec![nested_left, nested_right]);
        let prefix = source.mk_byte_literal(b"prefix");
        let suffix = source.mk_byte_literal(b"suffix");
        let prefixed = source.mk_concat(prefix, repeated);
        let concatenated = source.mk_concat(prefixed, suffix);
        let declaration = source.mk(Expr::Or(ExprFlags::POSITIVE, &[nested; 128]));
        let (_, mut source, _) = AlphabetInfo::from_exprset(
            source,
            &[repeated, empty, nested, concatenated, declaration],
        );
        source.reserve(128);
        let mut ordinary = source.clone();
        let mut reference = RelevanceCache::new();
        let prepared = source.prepared_source_plan().unwrap().compile().unwrap();
        let plan = PreparedExpressionPlan::prepare(prepared).unwrap();
        let spent = Cell::new(plan.requirements().required_bytes());
        let reserve = |bytes: usize| -> Result<(), &'static str> {
            let next = spent.get().checked_add(bytes).ok_or("overflow")?;
            if next > 512 * 1024 * 1024 {
                return Err("fixture allowance");
            }
            spent.set(next);
            Ok(())
        };
        let mut machine = plan.compile().unwrap();
        for (root, expected) in [(nested, true), (concatenated, true), (empty, false)] {
            assert_eq!(
                reference
                    .is_non_empty_limited(&mut ordinary, root, u64::MAX)
                    .unwrap(),
                expected
            );
            assert_eq!(
                machine.is_non_empty(root, u64::MAX, &reserve).unwrap(),
                expected
            );
            assert_eq!(machine.source().cost(), ordinary.cost());
        }
        let cost = machine.source().cost();
        let nodes = machine.num_symbolic_nodes();
        assert!(machine.is_non_empty(nested, u64::MAX, &reserve).unwrap());
        assert_eq!(machine.source().cost(), cost);
        assert_eq!(machine.num_symbolic_nodes(), nodes);
        assert!(machine.relevance.as_ref().unwrap().entry.is_none());
        let mut expected_source = source.clone();
        let mut expected_cache = RelevanceCache::new();
        assert!(expected_cache
            .is_non_empty_limited(&mut expected_source, empty, 0)
            .is_err());
        let prepared = source.prepared_source_plan().unwrap().compile().unwrap();
        let plan = PreparedExpressionPlan::prepare(prepared).unwrap();
        reserve(plan.requirements().required_bytes()).unwrap();
        let mut failed = plan.compile().unwrap();
        assert!(matches!(
            failed.is_non_empty(empty, 0, &reserve),
            Err(PreparedRelevanceError::Fuel { max_fuel: 0 })
        ));
        assert_eq!(failed.source().cost(), expected_source.cost());
        let retained = failed.relevance.as_ref().unwrap();
        assert!(retained.entry.is_some());
        assert!(retained.search.as_ref().unwrap().visited[empty.as_usize()]);
        let cost = failed.source().cost();
        let spent_before = spent.get();
        assert!(matches!(
            failed.is_non_empty(repeated, u64::MAX, &reserve),
            Err(PreparedRelevanceError::Failed)
        ));
        assert_eq!(spent.get(), spent_before);
        assert_eq!(failed.source().cost(), cost);
        drop(source);
        drop(ordinary);
        drop(expected_source);
        assert!(failed.source().is_valid(empty));
        assert!(machine.source().is_valid(concatenated));
    }
}

impl State {
    pub(super) fn copy_required_bytes<E>(&self) -> Option<usize> {
        use crate::copy_storage as copy;
        copy::frame_bytes::<(&mut Option<Self>, &Self, Self), E>()?
            .checked_add(copy::fixed_required_bytes::<_, E>(&self.concat)?)?
            .checked_add(copy::fixed_required_bytes::<_, E>(&self.probes)?)
    }
}
impl State {
    pub(super) fn copy_into<F: Fn(usize) -> Result<(), E>, E>(
        target: &mut Option<Self>, source: &Self, funding: &F,
    ) -> Result<(), crate::copy_storage::Error<E>> {
        use crate::copy_storage as copy;
        copy::frame::<(&mut Option<Self>, &Self, Self), _, _>(funding)?;
        let target = target.get_or_insert_with(|| Self::empty(source.nodes));
        copy::fixed(&mut target.concat, &source.concat, funding)?;
        copy::fixed(&mut target.probes, &source.probes, funding)?;
        target.nodes = source.nodes;
        Ok(())
    }
}
