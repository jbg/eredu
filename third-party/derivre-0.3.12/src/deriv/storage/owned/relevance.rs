//! Actual finite relevance-search destinations and existing symbolic consumers.
mod containment;
mod entry;
use super::super::Parts as DerivativeParts;
use super::{symbolic, weights};
use crate::{
    ast::{ExprRef, ExprSet, PreparedExprError},
    relevance::walk::{self, Context, Memory, StackFrame},
};
pub(super) use containment::operation_control_bytes as containment_operation_control_bytes;
pub(super) use entry::operation_control_bytes as entry_operation_control_bytes;
use std::{
    alloc::Layout,
    collections::TryReserveError,
    fmt,
    marker::PhantomData,
    mem::{size_of, size_of_val},
};

/// A relevance search stopped with its exact source, caches and work prefix
/// still retained in the expression machine. This is not controller admission.
#[derive(Debug)]
pub enum PreparedRelevanceError<E> {
    /// The queried expression or finite source geometry does not match.
    Source,
    /// Source-derived buffer or control arithmetic overflowed.
    Overflow,
    /// A destination differs from its exact prepared capacity.
    Capacity,
    /// An earlier operation failed or was interrupted.
    Failed,
    /// The same ordinary expression-cost ceiling was exceeded.
    Fuel { max_fuel: u64 },
    /// The enclosing owner refused a reservation before allocation.
    Funding(E),
    /// A destination allocation failed; the machine retains the prefix.
    Allocation(TryReserveError),
    /// The shared expression equation could not commit its result.
    Expression(PreparedExprError),
    /// The existing prepared symbolic derivative failed.
    Symbolic(symbolic::PreparedSymbolicError<E>),
    /// The existing prepared scalar derivative failed.
    Derivative(super::PreparedExpressionOperationError),
    /// The shared weight worker failed.
    Weight(weights::PreparedWeightError),
}
impl<E: fmt::Display> fmt::Display for PreparedRelevanceError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Source => f.write_str("relevance source differs from its retained arena"),
            Self::Overflow => f.write_str("relevance destination geometry overflow"),
            Self::Capacity => f.write_str("relevance destination exceeds exact capacity"),
            Self::Failed => f.write_str("relevance machine retains a failed prefix"),
            Self::Fuel { max_fuel } => {
                write!(f, "maximum relevance check fuel {max_fuel} exceeded")
            }
            Self::Funding(e) => fmt::Display::fmt(e, f),
            Self::Allocation(e) => fmt::Display::fmt(e, f),
            Self::Expression(e) => fmt::Display::fmt(e, f),
            Self::Symbolic(e) => fmt::Display::fmt(e, f),
            Self::Weight(e) => fmt::Display::fmt(e, f),
            Self::Derivative(e) => fmt::Display::fmt(e, f),
        }
    }
}
impl<E: std::error::Error + 'static> std::error::Error for PreparedRelevanceError<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Funding(e) => Some(e),
            Self::Allocation(e) => Some(e),
            Self::Expression(e) => Some(e),
            Self::Symbolic(e) => Some(e),
            Self::Weight(e) => Some(e),
            Self::Derivative(e) => Some(e),
            _ => None,
        }
    }
}
#[derive(Clone, Copy, Debug)]
pub(super) struct Requirements {
    pub(super) buffers: usize,
    pub(super) controls: usize,
}
#[derive(Debug)]
enum InitCause {
    Source,
    Overflow,
    Capacity,
    Allocation(TryReserveError),
}
pub(super) struct InitFailure {
    cause: InitCause,
    cache: Vec<Option<bool>>,
}
impl fmt::Debug for InitFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RelevanceCacheFailure")
            .field("cause", &self.cause)
            .field("cache_capacity", &self.cache.capacity())
            .finish()
    }
}
impl fmt::Display for InitFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.cause {
            InitCause::Source => f.write_str("relevance cache requires finite expression source"),
            InitCause::Overflow => f.write_str("relevance cache geometry overflow"),
            InitCause::Capacity => f.write_str("relevance cache differs from exact destination"),
            InitCause::Allocation(e) => fmt::Display::fmt(e, f),
        }
    }
}
impl std::error::Error for InitFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.cause {
            InitCause::Allocation(e) => Some(e),
            _ => None,
        }
    }
}
pub(super) struct Plan {
    nodes: usize,
    requirements: Requirements,
}
pub(super) struct Parts {
    cache: Vec<Option<bool>>,
    search: Option<Search>,
    entry: Option<entry::State>,
    contained: Vec<(ExprRef, ExprRef, bool)>,
    containment: Option<containment::State>,
}
impl Plan {
    pub(super) fn inspection_control_bytes() -> Option<usize> {
        let frames = [
            size_of::<Self>(),
            size_of::<Parts>(),
            size_of::<Requirements>(),
            size_of::<InitFailure>(),
            size_of::<InitCause>(),
            size_of::<Result<Self, InitFailure>>(),
            size_of::<Result<Parts, InitFailure>>(),
            size_of::<Result<(), TryReserveError>>(),
            size_of::<Layout>(),
        ];
        frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)
    }
    pub(super) fn prepare(source: &ExprSet) -> Result<Self, InitFailure> {
        let fail = |cause| InitFailure {
            cause,
            cache: Vec::new(),
        };
        let (_, nodes, _) = source
            .prepared_extents()
            .map_err(|_| fail(InitCause::Source))?;
        if source.len() > nodes || u32::try_from(nodes).is_err() {
            return Err(fail(InitCause::Source));
        }
        let buffers = Layout::array::<Option<bool>>(nodes)
            .map_err(|_| fail(InitCause::Overflow))?
            .size();
        let controls = Self::inspection_control_bytes().ok_or_else(|| fail(InitCause::Overflow))?;
        Ok(Self {
            nodes,
            requirements: Requirements { buffers, controls },
        })
    }
    pub(super) fn requirements(&self) -> Requirements {
        self.requirements
    }
    pub(super) fn compile(self) -> Result<Parts, InitFailure> {
        let mut cache = Vec::new();
        if let Err(e) = cache.try_reserve_exact(self.nodes) {
            return Err(InitFailure {
                cause: InitCause::Allocation(e),
                cache,
            });
        }
        if cache.capacity() != self.nodes {
            return Err(InitFailure {
                cause: InitCause::Capacity,
                cache,
            });
        }
        cache.resize(self.nodes, None);
        Ok(Parts {
            cache,
            search: None,
            entry: None,
            contained: Vec::new(),
            containment: None,
        })
    }
}
struct Search {
    visited: Vec<bool>,
    stack: Vec<StackFrame>,
    pending: Vec<ExprRef>,
}
impl Search {
    fn empty() -> Self {
        Self {
            visited: Vec::new(),
            stack: Vec::new(),
            pending: Vec::new(),
        }
    }
    fn requirement(nodes: usize) -> Option<usize> {
        Layout::array::<bool>(nodes)
            .ok()?
            .size()
            .checked_add(
                Layout::array::<StackFrame>(nodes.checked_add(1)?)
                    .ok()?
                    .size(),
            )?
            .checked_add(Layout::array::<ExprRef>(1).ok()?.size())
    }
    fn init<E>(&mut self, nodes: usize, root: ExprRef) -> Result<(), PreparedRelevanceError<E>> {
        allocate(&mut self.visited, nodes)?;
        self.visited.resize(nodes, false);
        allocate(
            &mut self.stack,
            nodes
                .checked_add(1)
                .ok_or(PreparedRelevanceError::Overflow)?,
        )?;
        allocate(&mut self.pending, 1)?;
        self.pending.push(root);
        self.stack.push(StackFrame {
            expression: root,
            cursor: 0,
            children: std::mem::take(&mut self.pending),
        });
        Ok(())
    }
}
fn allocate<T, E>(values: &mut Vec<T>, count: usize) -> Result<(), PreparedRelevanceError<E>> {
    values
        .try_reserve_exact(count)
        .map_err(PreparedRelevanceError::Allocation)?;
    if values.capacity() != count {
        return Err(PreparedRelevanceError::Capacity);
    }
    Ok(())
}
fn grow_reached<T, F: Fn(usize) -> Result<(), E>, E>(
    values: &mut Vec<T>,
    total: usize,
    reserve: &F,
) -> Result<(), PreparedRelevanceError<E>> {
    if total <= values.capacity() {
        return Ok(());
    }
    let layout = Layout::array::<T>(total).map_err(|_| PreparedRelevanceError::Overflow)?;
    let frames = [
        size_of::<(&mut Vec<T>, usize, &F)>(),
        size_of::<Layout>(),
        size_of::<Result<(), TryReserveError>>(),
        size_of::<Result<(), E>>(),
        size_of::<Result<(), PreparedRelevanceError<E>>>(),
        size_of::<Option<usize>>(),
    ];
    let bytes = frames
        .into_iter()
        .try_fold(size_of_val(&frames), usize::checked_add)
        .and_then(|n| n.checked_add(layout.size()))
        .ok_or(PreparedRelevanceError::Overflow)?;
    reserve(bytes).map_err(PreparedRelevanceError::Funding)?;
    values
        .try_reserve_exact(total - values.len())
        .map_err(PreparedRelevanceError::Allocation)?;
    if values.capacity() != total {
        return Err(PreparedRelevanceError::Capacity);
    }
    Ok(())
}
struct Storage<'a, F, E> {
    search: &'a mut Search,
    reserve: &'a F,
    marker: PhantomData<E>,
}
impl<F: Fn(usize) -> Result<(), E>, E> Memory for Storage<'_, F, E> {
    type Error = PreparedRelevanceError<E>;
    fn next(&mut self) -> Option<ExprRef> {
        walk::next(&mut self.search.stack)
    }
    fn visited(&self, node: ExprRef) -> bool {
        self.search
            .visited
            .get(node.as_usize())
            .copied()
            .unwrap_or(false)
    }
    fn visit(&mut self, node: ExprRef) -> Result<(), Self::Error> {
        if !node.is_valid() {
            return Err(Self::Error::Source);
        }
        let total = node
            .as_usize()
            .checked_add(1)
            .ok_or(Self::Error::Overflow)?;
        grow_reached(&mut self.search.visited, total, self.reserve)?;
        if self.search.visited.len() < total {
            self.search.visited.resize(total, false);
        }

        *self
            .search
            .visited
            .get_mut(node.as_usize())
            .ok_or(Self::Error::Source)? = true;
        Ok(())
    }
    fn begin_children(&mut self, count: usize) -> Result<(), Self::Error> {
        if !self.search.pending.is_empty() {
            return Err(Self::Error::Failed);
        }
        let bytes = Layout::array::<ExprRef>(count)
            .map_err(|_| Self::Error::Overflow)?
            .size();
        (self.reserve)(bytes).map_err(Self::Error::Funding)?;
        self.search.pending = Vec::new();
        allocate(&mut self.search.pending, count)
    }
    fn child(&mut self, node: ExprRef) -> Result<(), Self::Error> {
        if self.search.pending.len() == self.search.pending.capacity() {
            return Err(Self::Error::Capacity);
        }
        self.search.pending.push(node);
        Ok(())
    }
    fn children(&mut self) -> &mut [ExprRef] {
        &mut self.search.pending
    }
    fn push(&mut self, node: ExprRef) -> Result<(), Self::Error> {
        if self.search.pending.is_empty() {
            return Ok(());
        }
        let total = self
            .search
            .stack
            .len()
            .checked_add(1)
            .ok_or(Self::Error::Overflow)?;
        grow_reached(&mut self.search.stack, total, self.reserve)?;
        self.search.stack.push(StackFrame {
            expression: node,
            cursor: 0,
            children: std::mem::take(&mut self.search.pending),
        });
        Ok(())
    }
    fn pop_expression(&mut self) -> Option<ExprRef> {
        self.search.stack.pop().map(|f| f.expression)
    }
    fn publish_empty<C: Context<Error = Self::Error>>(
        &self,
        context: &mut C,
    ) -> Result<(), Self::Error> {
        for (index, &visited) in self.search.visited.iter().enumerate() {
            if visited {
                context.cache(ExprRef::new(index as u32), false)?;
            }
        }
        Ok(())
    }
}
struct Frame<'a, F, E> {
    source: &'a mut ExprSet,
    derivative: &'a mut DerivativeParts,
    symbolic: &'a mut Option<symbolic::Parts>,
    weights: &'a mut weights::Parts,
    cache: &'a mut Vec<Option<bool>>,
    reserve: &'a F,
    max_fuel: u64,
    cost_limit: u64,
    marker: PhantomData<E>,
}
impl<F: Fn(usize) -> Result<(), E>, E> Context for Frame<'_, F, E> {
    type Error = PreparedRelevanceError<E>;
    fn cached(&self, node: ExprRef) -> Option<bool> {
        self.cache.get(node.as_usize()).copied().flatten()
    }
    fn cache(&mut self, node: ExprRef, value: bool) -> Result<(), Self::Error> {
        if !self.source.is_valid(node) {
            return Err(Self::Error::Source);
        }
        self.source
            .grow_prepared_workspace(self.cache, self.source.len())
            .map_err(Self::Error::Expression)?;
        if self.cache.len() < self.source.len() {
            self.cache.resize(self.source.len(), None);
        }

        *self
            .cache
            .get_mut(node.as_usize())
            .ok_or(Self::Error::Source)? = Some(value);
        Ok(())
    }
    fn positive(&self, node: ExprRef) -> bool {
        self.source.is_positive(node)
    }
    fn derivative(&mut self, node: ExprRef) -> Result<super::symbolic::Values, Self::Error> {
        (self.reserve)(symbolic::operation_control_bytes::<F, E>().ok_or(Self::Error::Overflow)?)
            .map_err(Self::Error::Funding)?;
        let parts = self.symbolic.take().ok_or(Self::Error::Failed)?;
        let (parts, result) = parts.execute(self.source, self.derivative, node, self.reserve);
        *self.symbolic = Some(parts);
        result.map_err(Self::Error::Symbolic)
    }
    fn pay(&mut self, count: usize) -> Result<(), Self::Error> {
        self.source
            .pay_prepared(count)
            .map_err(Self::Error::Expression)
    }
    fn prepare_weight(&mut self, node: ExprRef) -> Result<(), Self::Error> {
        (self.reserve)(weights::operation_control_bytes().ok_or(Self::Error::Overflow)?)
            .map_err(Self::Error::Funding)?;
        self.weights
            .attributes(self.source, node)
            .map(|_| ())
            .map_err(Self::Error::Weight)
    }
    fn weight(&self, node: ExprRef) -> u32 {
        self.weights.cached_weight(node)
    }
    fn check_fuel(&self) -> Result<(), Self::Error> {
        if self.source.cost() > self.cost_limit {
            return Err(Self::Error::Fuel {
                max_fuel: self.max_fuel,
            });
        }
        Ok(())
    }
}
pub(super) fn operation_control_bytes<F, E>() -> Option<usize> {
    let frames = [
        size_of::<Search>(),
        size_of::<Storage<'_, F, E>>(),
        size_of::<Frame<'_, F, E>>(),
        size_of::<(usize, usize, usize)>(),
        size_of::<Result<(usize, usize, usize), PreparedExprError>>(),
        size_of::<Result<(), PreparedExprError>>(),
        size_of::<StackFrame>(),
        size_of::<Option<StackFrame>>(),
        size_of::<Option<ExprRef>>(),
        size_of::<symbolic::Values>(),
        size_of::<ExprRef>() * 4,
        size_of::<usize>() * 3,
        size_of::<PreparedRelevanceError<E>>(),
        size_of::<Result<bool, PreparedRelevanceError<E>>>(),
        size_of::<Result<(), PreparedRelevanceError<E>>>(),
        size_of::<Result<(), TryReserveError>>(),
        size_of::<Layout>(),
        size_of::<std::slice::Iter<'_, (ExprRef, ExprRef)>>(),
        size_of::<std::slice::Iter<'_, ExprRef>>(),
    ];
    frames
        .into_iter()
        .try_fold(size_of_val(&frames), usize::checked_add)
}
impl Parts {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn execute<F: Fn(usize) -> Result<(), E>, E>(
        &mut self,
        source: &mut ExprSet,
        derivative: &mut DerivativeParts,
        symbolic: &mut Option<symbolic::Parts>,
        weights: &mut weights::Parts,
        root: ExprRef,
        max_fuel: u64,
        reserve: &F,
    ) -> Result<bool, PreparedRelevanceError<E>> {
        let cost_limit = source.cost().saturating_add(max_fuel);
        run_search(
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
            root,
        )
    }
}

fn run_search<F: Fn(usize) -> Result<(), E>, E>(
    frame: &mut Frame<'_, F, E>,
    search: &mut Option<Search>,
    root: ExprRef,
) -> Result<bool, PreparedRelevanceError<E>> {
    if search.is_some() {
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
    let reserve = frame.reserve;
    reserve(Search::requirement(frame.cache.len()).ok_or(PreparedRelevanceError::Overflow)?)
        .map_err(PreparedRelevanceError::Funding)?;
    *search = Some(Search::empty());
    let current = search.as_mut().expect("installed search destination");
    current.init(frame.cache.len(), root)?;
    let result = walk::run(
        frame,
        &mut Storage {
            search: current,
            reserve,
            marker: PhantomData,
        },
    );
    if result.is_ok() {
        *search = None;
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
    fn owning_relevance_walk_preserves_shared_search_fuel_cache_and_failed_destinations() {
        let mut source = ExprSet::new(256);
        let a = source.mk_byte(b'a');
        let b = source.mk_byte(b'b');
        let c = source.mk_byte(b'c');
        let ab = source.mk_byte_set_or(&[a, b]);
        let bc = source.mk_byte_set_or(&[b, c]);
        let xy = source.mk_byte_literal(b"xy");
        let xz = source.mk_byte_literal(b"xz");
        let left = source.mk_concat(ab, xy);
        let right_empty = source.mk_concat(bc, xz);
        let right_full = source.mk_concat(bc, xy);
        let empty = source.mk_and(&mut vec![left, right_empty]);
        let full = source.mk_and(&mut vec![left, right_full]);
        let declaration = source.mk(Expr::Or(ExprFlags::POSITIVE, &[empty; 128]));
        let (_, mut source, _) = AlphabetInfo::from_exprset(source, &[empty, full, declaration]);
        source.reserve(128);
        assert!(!source.is_positive(empty));
        assert!(!source.is_positive(full));
        let mut ordinary = source.clone();
        let mut ordinary_cache = RelevanceCache::new();
        let prepared = source.prepared_source_plan().unwrap().compile().unwrap();
        let plan = PreparedExpressionPlan::prepare(prepared).unwrap();
        let spent = Cell::new(plan.requirements().required_bytes());
        let limit = 128 * 1024 * 1024usize;
        let reserve = |bytes: usize| -> Result<(), &'static str> {
            let next = spent.get().checked_add(bytes).ok_or("overflow")?;
            if next > limit {
                return Err("fixture allowance");
            }
            spent.set(next);
            Ok(())
        };
        let mut machine = plan.compile().unwrap();
        for (root, expected) in [(empty, false), (full, true)] {
            assert_eq!(
                ordinary_cache
                    .is_non_empty_limited(&mut ordinary, root, u64::MAX)
                    .unwrap(),
                expected
            );
            assert_eq!(
                machine.relevance_walk(root, u64::MAX, &reserve).unwrap(),
                expected
            );
            assert_eq!(machine.source().cost(), ordinary.cost());
            assert_eq!(
                machine.relevance.as_ref().unwrap().cache[root.as_usize()],
                Some(expected)
            );
            let nodes = machine.num_symbolic_nodes();
            let cost = machine.source().cost();
            assert_eq!(
                machine.relevance_walk(root, u64::MAX, &reserve).unwrap(),
                expected
            );
            assert_eq!(machine.num_symbolic_nodes(), nodes);
            assert_eq!(machine.source().cost(), cost);
        }
        assert!(machine.relevance.as_ref().unwrap().search.is_none());
        // Fuel is tested after the same committed derivative/visited/child-sort
        // prefix; the failed machine retains that actual search and all caches.
        let mut ordinary_failed = source.clone();
        let mut failed_cache = RelevanceCache::new();
        assert!(failed_cache
            .is_non_empty_limited(&mut ordinary_failed, empty, 0)
            .is_err());
        let prepared = source.prepared_source_plan().unwrap().compile().unwrap();
        let plan = PreparedExpressionPlan::prepare(prepared).unwrap();
        reserve(plan.requirements().required_bytes()).unwrap();
        let mut failed = plan.compile().unwrap();
        let before = spent.get();
        assert!(matches!(
            failed.relevance_walk(empty, 0, &reserve),
            Err(PreparedRelevanceError::Fuel { max_fuel: 0 })
        ));
        assert!(spent.get() > before);
        assert_eq!(failed.source().cost(), ordinary_failed.cost());
        let retained = failed.relevance.as_ref().unwrap();
        assert_eq!(retained.cache[empty.as_usize()], None);
        let search = retained.search.as_ref().unwrap();
        assert!(search.visited[empty.as_usize()]);
        assert!(!search.stack.is_empty());
        let spent_before = spent.get();
        let cost = failed.source().cost();
        let nodes = failed.num_symbolic_nodes();
        assert!(matches!(
            failed.relevance_walk(full, u64::MAX, &reserve),
            Err(PreparedRelevanceError::Failed)
        ));
        assert!(failed.derivative(empty, b'b').is_err());
        assert_eq!(spent.get(), spent_before);
        assert_eq!(failed.source().cost(), cost);
        assert_eq!(failed.num_symbolic_nodes(), nodes);
        drop(source);
        drop(ordinary);
        drop(ordinary_failed);
        assert!(machine.source().is_valid(full));
        assert!(failed.source().is_valid(empty));
    }
}

impl Parts {
    pub(super) fn copy_destination_controls<E>(&self) -> Option<usize> {
        crate::copy_storage::frame_bytes::<(&Self, Self), E>()
    }
    pub(super) fn copy_destination(&self) -> Self {
        Self { cache: Vec::new(), search: None, entry: None, contained: Vec::new(), containment: None }
    }
    pub(super) fn copy_required_bytes<E>(&self) -> Option<usize> {
        use crate::copy_storage as copy;
        let mut bytes = copy::frame_bytes::<(&mut Self, &Self), E>()?
            .checked_add(copy::fixed_required_bytes::<_, E>(&self.cache)?)?
            .checked_add(copy::fixed_required_bytes::<_, E>(&self.contained)?)?;
        if let Some(source) = &self.search {
            bytes = bytes.checked_add(copy::frame_bytes::<(Search, &Search, StackFrame, std::slice::Iter<'_, StackFrame>), E>()?)?
                .checked_add(copy::fixed_required_bytes::<_, E>(&source.visited)?)?
                .checked_add(copy::fixed_required_bytes::<_, E>(&source.pending)?)?
                .checked_add(copy::reserve_required_bytes::<StackFrame, E>(source.stack.capacity())?)?;
            for frame in &source.stack {
                bytes = bytes.checked_add(copy::fixed_required_bytes::<_, E>(&frame.children)?)?;
            }
        }
        if let Some(source) = &self.entry { bytes = bytes.checked_add(source.copy_required_bytes::<E>()?)?; }
        if let Some(source) = &self.containment { bytes = bytes.checked_add(source.copy_required_bytes::<E>()?)?; }
        Some(bytes)
    }
}
impl Parts {

    pub(crate) fn restore_copy<F: Fn(usize) -> Result<(), E>, E>(
        &mut self, source: &Self, funding: &F,
    ) -> Result<(), crate::copy_storage::Error<E>> {
        use crate::copy_storage as copy;
        copy::frame::<(&mut Self, &Self), _, _>(funding)?;
        copy::fixed(&mut self.cache, &source.cache, funding)?;
        copy::fixed(&mut self.contained, &source.contained, funding)?;
        if let Some(source) = &source.search {
            copy::frame::<(Search, &Search, StackFrame, std::slice::Iter<'_, StackFrame>), _, _>(funding)?;
            let target = self.search.get_or_insert_with(|| Search { visited: Vec::new(), stack: Vec::new(), pending: Vec::new() });
            copy::fixed(&mut target.visited, &source.visited, funding)?;
            copy::fixed(&mut target.pending, &source.pending, funding)?;
            copy::reserve(&mut target.stack, source.stack.capacity(), funding)?;
            target.stack.clear();
            for frame in &source.stack {
                target.stack.push(StackFrame { expression: frame.expression, cursor: frame.cursor, children: Vec::new() });
                copy::fixed(&mut target.stack.last_mut().expect("copy search child").children, &frame.children, funding)?;
            }
        } else { self.search = None; }
        if let Some(source) = &source.entry { entry::State::copy_into(&mut self.entry, source, funding)?; }
        else { self.entry = None; }
        if let Some(source) = &source.containment { containment::State::copy_into(&mut self.containment, source, funding)?; }
        else { self.containment = None; }
        Ok(())
    }
}
