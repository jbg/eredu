//! Finite cache/scratch destinations beside the same owning expression source.
use super::{cached, compute, Attrs, Destination, Scratch};
use crate::ast::{ExprRef, ExprSet, PreparedExprError, ATTR_HAS_REPEAT};
use std::{
    alloc::Layout,
    collections::TryReserveError,
    fmt,
    mem::{size_of, size_of_val},
};

/// A weight traversal stopped without discarding its completed cache prefix.
#[derive(Debug)]
pub enum PreparedWeightError {
    /// The root is not a committed expression in this retained source.
    Source,
    /// The fixed cache or traversal destination cannot hold the next entry.
    Capacity,
    /// A shared weight equation overflowed before storing its result.
    Arithmetic,
    /// This machine retains a failed operation prefix.
    Failed,
    /// Exact source-bound backing payment or allocation failed; the existing
    /// cache and traversal prefix remain on the expression machine.
    Storage(crate::ast::PreparedExprError),
}
impl fmt::Display for PreparedWeightError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Self::Storage(error) = self {
            return fmt::Display::fmt(error, f);
        }
        f.write_str(match self {
            Self::Source => "weight root is outside the retained expression source",
            Self::Capacity => "weight traversal exceeds its prepared destination",
            Self::Arithmetic => "expression weight arithmetic overflow",
            Self::Failed => "expression weight machine retains a failed prefix",
            Self::Storage(_) => unreachable!(),
        })
    }
}
impl std::error::Error for PreparedWeightError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Storage(error) => Some(error),
            _ => None,
        }
    }
}
struct Checked;
impl Destination for Checked {
    type Error = PreparedWeightError;
    fn push<T>(to: &mut Vec<T>, value: T) -> Result<(), Self::Error> {
        if to.len() == to.capacity() {
            return Err(Self::Error::Capacity);
        }
        to.push(value);
        Ok(())
    }
    fn store(cache: &mut Vec<Attrs>, index: usize, value: Attrs) -> Result<(), Self::Error> {
        *cache.get_mut(index).ok_or(Self::Error::Capacity)? = value;
        Ok(())
    }
    fn add(left: u32, right: u32) -> Result<u32, Self::Error> {
        left.checked_add(right).ok_or(Self::Error::Arithmetic)
    }
}
#[derive(Clone, Copy, Debug)]
pub(crate) struct Requirements {
    pub(crate) buffers: usize,
    pub(crate) controls: usize,
}
#[derive(Clone, Copy, Debug)]
struct Geometry {
    nodes: usize,
    slots: usize,
    todo: usize,
    mapped: usize,
}
#[derive(Debug)]
enum InitCause {
    Source,
    Overflow,
    Capacity,
    Allocation(TryReserveError),
}
pub(crate) struct InitFailure {
    cause: InitCause,
    cache: Vec<Attrs>,
    scratch: Scratch,
}
impl fmt::Debug for InitFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WeightConstructionFailure")
            .field("cause", &self.cause)
            .field("cache_capacity", &self.cache.capacity())
            .field("todo_capacity", &self.scratch.todo.capacity())
            .field("mapped_capacity", &self.scratch.mapped.capacity())
            .finish()
    }
}
impl fmt::Display for InitFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.cause {
            InitCause::Source => f.write_str("weight source lacks finite expression geometry"),
            InitCause::Overflow => f.write_str("weight destination geometry overflow"),
            InitCause::Capacity => f.write_str("weight allocation differs from exact destination"),
            InitCause::Allocation(error) => fmt::Display::fmt(error, f),
        }
    }
}
impl std::error::Error for InitFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.cause {
            InitCause::Allocation(error) => Some(error),
            _ => None,
        }
    }
}
fn failure(cause: InitCause) -> InitFailure {
    InitFailure {
        cause,
        cache: Vec::new(),
        scratch: Scratch::empty(),
    }
}
pub(crate) struct Plan<'a> {
    source: &'a ExprSet,
    geometry: Geometry,
    requirements: Requirements,
}
pub(crate) struct Parts {
    nodes: usize,
    cache: Vec<Attrs>,
    scratch: Scratch,
    failed: bool,
}
impl Plan<'_> {
    pub(crate) fn inspection_control_bytes() -> Option<usize> {
        let frames = [
            size_of::<Self>(),
            size_of::<Parts>(),
            size_of::<InitFailure>(),
            size_of::<InitCause>(),
            size_of::<Geometry>(),
            size_of::<Requirements>(),
            size_of::<Result<Self, InitFailure>>(),
            size_of::<Result<Parts, InitFailure>>(),
            size_of::<Result<(), TryReserveError>>(),
            size_of::<Layout>(),
        ];
        frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)
    }
    pub(crate) fn prepare(source: &ExprSet) -> Result<Plan<'_>, InitFailure> {
        let (_, nodes, encoded) = source
            .prepared_extents()
            .map_err(|_| failure(InitCause::Source))?;
        let mapped = encoded
            .checked_sub(1)
            .ok_or_else(|| failure(InitCause::Source))?;
        if source.len() > nodes {
            return Err(failure(InitCause::Source));
        }
        // Each deferred ancestor contributes at most one requeued parent and
        // `mapped` child entries. Committed source children are topological,
        // so active ancestor depth cannot exceed the finite arena entry bound.
        let todo = mapped
            .checked_add(1)
            .and_then(|n| nodes.checked_mul(n))
            .and_then(|n| n.checked_add(1))
            .ok_or_else(|| failure(InitCause::Overflow))?;
        // Ordinary cache growth pads the historical last entry by 100; preserve
        // that entire immutable prefix even when it exceeds the arena slots.
        let slots = nodes.max(source.expr_weight.len());
        let geometry = Geometry {
            nodes,
            slots,
            todo,
            mapped,
        };
        let buffers = Layout::array::<Attrs>(slots)
            .ok()
            .map(|l| l.size())
            .and_then(|n| n.checked_add(Layout::array::<ExprRef>(todo).ok()?.size()))
            .and_then(|n| n.checked_add(Layout::array::<Attrs>(mapped).ok()?.size()))
            .ok_or_else(|| failure(InitCause::Overflow))?;
        let controls =
            Self::inspection_control_bytes().ok_or_else(|| failure(InitCause::Overflow))?;
        Ok(Plan {
            source,
            geometry,
            requirements: Requirements { buffers, controls },
        })
    }
    pub(crate) fn requirements(&self) -> Requirements {
        self.requirements
    }
    pub(crate) fn compile(self) -> Result<Parts, InitFailure> {
        let mut partial = failure(InitCause::Capacity);
        fn allocate<T>(to: &mut Vec<T>, size: usize) -> Result<(), InitCause> {
            to.try_reserve_exact(size).map_err(InitCause::Allocation)?;
            if to.capacity() != size {
                return Err(InitCause::Capacity);
            }
            Ok(())
        }
        let result = (|| {
            allocate(&mut partial.cache, self.geometry.slots)?;
            partial.cache.extend_from_slice(&self.source.expr_weight);
            partial.cache.resize(self.geometry.slots, (0, 0));
            allocate(&mut partial.scratch.todo, self.geometry.todo)?;
            allocate(&mut partial.scratch.mapped, self.geometry.mapped)?;
            Ok::<_, InitCause>(())
        })();
        match result {
            Ok(()) => Ok(Parts {
                nodes: self.geometry.nodes,
                cache: partial.cache,
                scratch: partial.scratch,
                failed: false,
            }),
            Err(cause) => {
                partial.cause = cause;
                Err(partial)
            }
        }
    }
}
impl Parts {
    pub(crate) fn cached_weight(&self, root: ExprRef) -> u32 {
        cached(&self.cache, root).0
    }
    pub(crate) fn attributes(
        &mut self,
        source: &ExprSet,
        root: ExprRef,
    ) -> Result<Attrs, PreparedWeightError> {
        if self.failed {
            return Err(PreparedWeightError::Failed);
        }
        // Fence before entering the shared worker, including unwind paths.
        self.failed = true;
        if !source.is_valid(root) {
            return Err(PreparedWeightError::Source);
        }
        if source.len() > self.nodes {
            let (nodes, edges, width) = source
                .reached_workspace_geometry()
                .map_err(PreparedWeightError::Storage)?;
            let todo = edges
                .checked_add(nodes)
                .ok_or(PreparedWeightError::Capacity)?;
            source
                .grow_prepared_workspace(&mut self.cache, nodes)
                .map_err(PreparedWeightError::Storage)?;
            if self.cache.len() < nodes {
                self.cache.resize(nodes, (0, 0));
            }
            source
                .grow_prepared_workspace(&mut self.scratch.todo, todo)
                .map_err(PreparedWeightError::Storage)?;
            source
                .grow_prepared_workspace(&mut self.scratch.mapped, width)
                .map_err(PreparedWeightError::Storage)?;
            self.nodes = nodes;
        }
        if cached(&self.cache, root).0 == 0 {
            compute::<Checked>(&source.exprs, &mut self.cache, &mut self.scratch, root)?;
        }
        let result = cached(&self.cache, root);
        self.failed = false;
        Ok(result)
    }
}

pub(crate) fn operation_control_bytes() -> Option<usize> {
    let frames = [
        size_of::<&ExprSet>(),
        size_of::<(usize, usize, usize)>(),
        size_of::<Result<(usize, usize, usize), PreparedExprError>>(),
        size_of::<Result<(), PreparedExprError>>(),
        size_of::<&mut Parts>(),
        size_of::<&mut Vec<Attrs>>(),
        size_of::<&mut Scratch>(),
        size_of::<ExprRef>(),
        size_of::<crate::ast::Expr<'_>>(),
        size_of::<bool>(),
        size_of::<u32>() * 4,
        size_of::<Attrs>(),
        size_of::<PreparedWeightError>(),
        size_of::<Result<(), PreparedWeightError>>(),
        size_of::<Result<Attrs, PreparedWeightError>>(),
    ];
    frames
        .into_iter()
        .try_fold(size_of_val(&frames), usize::checked_add)
}
impl Parts {
    pub(crate) fn has_repeat(
        &mut self,
        source: &ExprSet,
        root: ExprRef,
    ) -> Result<bool, PreparedWeightError> {
        self.attributes(source, root)
            .map(|v| v.1 & ATTR_HAS_REPEAT != 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ast::{Expr, ExprFlags},
        raw::{DerivCache, PreparedExpressionPlan},
        AlphabetInfo,
    };
    #[test]
    fn owning_weights_preserve_cached_attributes_successors_threshold_and_failed_prefix() {
        let mut source = ExprSet::new(256);
        let byte = source.mk_byte(b'a');
        let other = source.mk_byte(b'b');
        let selector = source.mk_byte_set_or(&[byte, other]);
        let repeat = source.mk_repeat(selector, 2, 12);
        let lookahead = source.mk_lookahead(repeat, 3);
        let complement = source.mk_not(lookahead);
        let literal = source.mk_byte_literal(&[b'a'; 64]);
        let mut dag = repeat;
        for _ in 0..20 {
            dag = source.mk(Expr::Concat(ExprFlags::POSITIVE, [dag, dag]));
        }
        let declaration = source.mk(Expr::Or(ExprFlags::POSITIVE, &[dag; 128]));
        let fresh = source.mk_byte(b'z');
        let last = source.mk_not(fresh);
        let roots = [
            repeat,
            lookahead,
            complement,
            literal,
            dag,
            declaration,
            last,
        ];
        let (_, mut source, _) = AlphabetInfo::from_exprset(source, &roots);
        source.reserve(96);
        assert!(source.get_weight(repeat) > 0);
        assert!(source.attr_has_repeat(repeat));
        let historical = source.expr_weight.clone();
        let mut ordinary = source.clone();
        let prepared = source.prepared_source_plan().unwrap().compile().unwrap();
        let plan = Plan::prepare(prepared.source()).unwrap();
        assert!(plan.requirements().buffers > 0);
        let mut parts = plan.compile().unwrap();
        assert_eq!(&parts.cache[..historical.len()], historical.as_slice());
        assert_eq!(
            parts.attributes(prepared.source(), repeat).unwrap(),
            ordinary.get_attrs(repeat)
        );
        // A reached destination refusal keeps the successfully computed child
        // cache and exact source. It cannot silently use ordinary cache growth.
        parts.cache.truncate(last.as_usize());
        let before_cost = prepared.source().cost();
        assert!(matches!(
            parts.attributes(prepared.source(), last),
            Err(PreparedWeightError::Capacity)
        ));
        assert_eq!(cached(&parts.cache, fresh).0, 1);
        assert_eq!(prepared.source().cost(), before_cost);
        assert!(matches!(
            parts.attributes(prepared.source(), repeat),
            Err(PreparedWeightError::Failed)
        ));
        drop(parts);
        // The complete owning machine retains the original historical source
        // while its private cache computes actual future derivative IDs.
        let mut machine = PreparedExpressionPlan::prepare(prepared)
            .unwrap()
            .compile()
            .unwrap();
        for root in roots {
            assert_eq!(machine.weight(root).unwrap(), ordinary.get_weight(root));
            assert_eq!(
                machine.has_repeat(root).unwrap(),
                ordinary.attr_has_repeat(root)
            );
        }
        assert!(machine.weight(dag).unwrap() >= 1_000_000);
        let cost = machine.source().cost();
        let entries = machine.source().len();
        assert_eq!(machine.weight(dag).unwrap(), ordinary.get_weight(dag));
        assert_eq!(machine.source().cost(), cost);
        assert_eq!(machine.source().len(), entries);
        let mut derivative = DerivCache::new();
        let next = machine.derivative(literal, b'a').unwrap();
        assert_eq!(next, derivative.derivative(&mut ordinary, literal, b'a'));
        assert_eq!(machine.weight(next).unwrap(), ordinary.get_weight(next));
        assert_eq!(machine.source().cost(), ordinary.cost());
        assert_eq!(&machine.source().expr_weight, &historical);
        let cost = machine.source().cost();
        let entries = machine.source().len();
        assert!(matches!(
            machine.weight(ExprRef::INVALID),
            Err(PreparedWeightError::Source)
        ));
        assert!(matches!(
            machine.has_repeat(repeat),
            Err(PreparedWeightError::Failed)
        ));
        assert!(machine.derivative(next, b'a').is_err());
        assert_eq!(machine.source().cost(), cost);
        assert_eq!(machine.source().len(), entries);
        drop(source);
        assert!(machine.source().is_valid(next));
    }
}

impl Parts {
    pub(crate) fn copy_destination_controls<E>(&self) -> Option<usize> {
        crate::copy_storage::frame_bytes::<(&Self, Self), E>()
    }
    pub(crate) fn copy_destination(&self) -> Self {
        Self { nodes: self.nodes, cache: Vec::new(), scratch: Scratch::empty(), failed: self.failed }
    }
    pub(crate) fn copy_required_bytes<E>(&self) -> Option<usize> {
        use crate::copy_storage as copy;
        copy::frame_bytes::<(&mut Self, &Self), E>()?
            .checked_add(copy::fixed_required_bytes::<_, E>(&self.cache)?)?
            .checked_add(copy::fixed_required_bytes::<_, E>(&self.scratch.todo)?)?
            .checked_add(copy::fixed_required_bytes::<_, E>(&self.scratch.mapped)?)
    }
}
impl Parts {

    pub(crate) fn restore_copy<F: Fn(usize) -> Result<(), E>, E>(
        &mut self, source: &Self, funding: &F,
    ) -> Result<(), crate::copy_storage::Error<E>> {
        use crate::copy_storage as copy;
        copy::frame::<(&mut Self, &Self), _, _>(funding)?;
        copy::fixed(&mut self.cache, &source.cache, funding)?;
        copy::fixed(&mut self.scratch.todo, &source.scratch.todo, funding)?;
        copy::fixed(&mut self.scratch.mapped, &source.scratch.mapped, funding)?;
        self.nodes = source.nodes;
        self.failed = source.failed;
        Ok(())
    }
}
