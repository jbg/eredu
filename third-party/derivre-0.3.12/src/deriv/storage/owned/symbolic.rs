//! Persistent symbolic memo and the actual shared node consumer.
use super::super::{workspace, Parts as DerivativeParts};
use crate::{
    ast::{mapping::Cache, Expr, ExprRef, ExprSet, PreparedExprError},
    relevance::{
        disjoint, grouping,
        symbolic::{self, storage as lists, Construction},
    },
    simplify::{concat::storage as concatenation, nary::storage as nary},
};
use std::{
    alloc::Layout,
    collections::TryReserveError,
    fmt,
    marker::PhantomData,
    mem::{size_of, size_of_val},
};
pub(super) type Values = Vec<(ExprRef, ExprRef)>;
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
    Mapping(workspace::ConstructionFailure<Values>),
}
pub(super) struct InitFailure {
    cause: InitCause,
    memo: Vec<Option<Values>>,
}
impl fmt::Debug for InitFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SymbolicMemoFailure")
            .field("cause", &self.cause)
            .field("capacity", &self.memo.capacity())
            .finish()
    }
}
impl fmt::Display for InitFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.cause {
            InitCause::Source => f.write_str("symbolic memo requires its finite expression source"),
            InitCause::Overflow => f.write_str("symbolic memo geometry overflow"),
            InitCause::Capacity => {
                f.write_str("symbolic memo destination differs from its exact capacity")
            }
            InitCause::Allocation(e) => fmt::Display::fmt(e, f),
            InitCause::Mapping(e) => fmt::Display::fmt(e, f),
        }
    }
}
impl std::error::Error for InitFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.cause {
            InitCause::Allocation(e) => Some(e),
            InitCause::Mapping(e) => Some(e),
            _ => None,
        }
    }
}
pub(super) struct Plan<'a> {
    mapping: workspace::Plan<'a, Values>,
    nodes: usize,
    requirements: Requirements,
}
pub(super) struct Parts {
    mapping: workspace::Parts<Values>,
    memo: Vec<Option<Values>>,
    pub(super) nodes: usize,
}
impl Plan<'_> {
    pub(super) fn prepare(source: &mut ExprSet) -> Result<Plan<'_>, InitFailure> {
        let failure = |cause| InitFailure {
            cause,
            memo: Vec::new(),
        };
        let (_, nodes, _) = source
            .prepared_extents()
            .map_err(|_| failure(InitCause::Source))?;
        let mapping = source
            .mapping_workspace_plan()
            .map_err(|e| failure(InitCause::Mapping(e)))?
            .prepared_bounds()
            .map_err(|e| failure(InitCause::Mapping(e)))?;
        let mapped = mapping.requirements();
        let bytes = Layout::array::<Option<Values>>(nodes)
            .map_err(|_| failure(InitCause::Overflow))?
            .size();
        let frames = [
            size_of::<Plan<'_>>(),
            size_of::<Parts>(),
            size_of::<InitFailure>(),
            size_of::<InitCause>(),
            size_of::<Requirements>(),
            size_of::<Result<Parts, InitFailure>>(),
            size_of::<Result<(), TryReserveError>>(),
        ];
        let own = frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)
            .ok_or_else(|| failure(InitCause::Overflow))?;
        let requirements = Requirements {
            buffers: mapped
                .buffers
                .checked_add(bytes)
                .ok_or_else(|| failure(InitCause::Overflow))?,
            controls: mapped
                .controls
                .checked_add(own)
                .ok_or_else(|| failure(InitCause::Overflow))?,
        };
        Ok(Plan {
            mapping,
            nodes,
            requirements,
        })
    }
    pub(super) fn requirements(&self) -> Requirements {
        self.requirements
    }
    pub(super) fn compile(self) -> Result<Parts, InitFailure> {
        let mut memo = Vec::new();
        if let Err(error) = memo.try_reserve_exact(self.nodes) {
            return Err(InitFailure {
                cause: InitCause::Allocation(error),
                memo,
            });
        }
        if memo.capacity() != self.nodes {
            return Err(InitFailure {
                cause: InitCause::Capacity,
                memo,
            });
        }
        memo.resize_with(self.nodes, || None);
        match self.mapping.compile() {
            Ok(mapping) => Ok(Parts {
                mapping: mapping.into_parts(),
                memo,
                nodes: 0,
            }),
            Err(error) => Err(InitFailure {
                cause: InitCause::Mapping(error),
                memo,
            }),
        }
    }
}
#[derive(Debug)]
enum Cause<E> {
    Overflow,
    Source,
    Failed,
    Funding(E),
    Expression(PreparedExprError),
    Node(lists::Failure),
    Grouping(grouping::storage::Failure),
    Disjoint(disjoint::storage::Failure),
    List(lists::Issue),
    Allocation {
        error: TryReserveError,
        values: Values,
        selectors: Vec<ExprRef>,
    },
}
impl<E: fmt::Display> fmt::Display for Cause<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Overflow => f.write_str("symbolic operation geometry overflow"),
            Self::Source => f.write_str("symbolic operation differs from its expression source"),
            Self::Failed => f.write_str("symbolic expression machine retains a failed prefix"),
            Self::Funding(e) => fmt::Display::fmt(e, f),
            Self::Expression(e) => fmt::Display::fmt(e, f),
            Self::Node(e) => fmt::Display::fmt(e, f),
            Self::Grouping(e) => fmt::Display::fmt(e, f),
            Self::Disjoint(e) => fmt::Display::fmt(e, f),
            Self::List(e) => fmt::Display::fmt(e, f),
            Self::Allocation { error, .. } => fmt::Display::fmt(error, f),
        }
    }
}
impl<E: std::error::Error + 'static> std::error::Error for Cause<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Funding(e) => Some(e),
            Self::Expression(e) => Some(e),
            Self::Node(e) => Some(e),
            Self::Grouping(e) => Some(e),
            Self::Disjoint(e) => Some(e),
            Self::List(e) => Some(e),
            Self::Allocation { error, .. } => Some(error),
            _ => None,
        }
    }
}
/// The actual reservation, constructor or source error from a symbolic step.
/// The enclosing machine retains its source and completed memo prefix.
#[derive(Debug)]
pub struct PreparedSymbolicError<E> {
    cause: workspace::OperationError<Cause<E>>,
}
impl<E: fmt::Display> fmt::Display for PreparedSymbolicError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.cause, f)
    }
}
impl<E: std::error::Error + 'static> std::error::Error for PreparedSymbolicError<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.cause)
    }
}
impl<E> PreparedSymbolicError<E> {
    pub(super) fn failed() -> Self {
        Self {
            cause: workspace::OperationError::Process(Cause::Failed),
        }
    }
    pub(super) fn overflow() -> Self {
        Self {
            cause: workspace::OperationError::Process(Cause::Overflow),
        }
    }
    pub(super) fn funding(error: E) -> Self {
        Self {
            cause: workspace::OperationError::Process(Cause::Funding(error)),
        }
    }
}
pub(super) fn operation_control_bytes<F, E>() -> Option<usize> {
    let frames = [
        size_of::<(usize, usize, usize)>(),
        size_of::<Result<(usize, usize, usize), PreparedExprError>>(),
        size_of::<Result<(), PreparedExprError>>(),
        size_of::<Parts>(),
        size_of::<super::super::Parts>(),
        size_of::<workspace::Workspace<'_, Values>>(),
        size_of::<Memo<'_, E>>(),
        size_of::<Frame<'_, F, E>>(),
        size_of::<Cause<E>>(),
        size_of::<PreparedSymbolicError<E>>(),
        size_of::<Result<Values, PreparedSymbolicError<E>>>(),
        size_of::<Result<Values, Cause<E>>>(),
        size_of::<&F>(),
        size_of::<Expr<'_>>(),
        size_of::<[u8; ExprRef::MAX_BYTE_CONCAT]>(),
        size_of::<(ExprRef, ExprRef)>() * 4,
        crate::simplify::byteset::fixed_control_bytes()?,
        crate::simplify::remainder::fixed_control_bytes()?,
    ];
    frames
        .into_iter()
        .try_fold(size_of_val(&frames), usize::checked_add)
}
struct Memo<'a, E> {
    slots: &'a mut [Option<Values>],
    marker: PhantomData<E>,
}
impl<E> Cache<ExprRef, Values> for Memo<'_, E> {
    type Error = Cause<E>;
    fn get(&self, key: &ExprRef) -> Option<&Values> {
        self.slots.get(key.as_usize())?.as_ref()
    }
    fn contains_key(&self, key: &ExprRef) -> bool {
        self.get(key).is_some()
    }
    fn insert(&mut self, key: ExprRef, value: Values) -> Result<(), Self::Error> {
        let slot = self.slots.get_mut(key.as_usize()).ok_or(Cause::Source)?;
        *slot = Some(value);
        Ok(())
    }
}
fn value_copy<E>(
    input: &Values,
    fund: &impl Fn(usize) -> Result<(), E>,
) -> Result<Values, Cause<E>> {
    let frames = [
        size_of::<Values>(),
        size_of::<Result<Values, Cause<E>>>(),
        size_of::<Result<(), TryReserveError>>(),
        size_of::<Layout>(),
    ];
    let controls = frames
        .into_iter()
        .try_fold(size_of_val(&frames), usize::checked_add)
        .ok_or(Cause::Overflow)?;
    let total = Layout::array::<(ExprRef, ExprRef)>(input.len())
        .map_err(|_| Cause::Overflow)?
        .size()
        .checked_add(controls)
        .ok_or(Cause::Overflow)?;
    fund(total).map_err(Cause::Funding)?;
    let mut values = Vec::new();
    if let Err(error) = values.try_reserve_exact(input.len()) {
        return Err(Cause::Allocation {
            error,
            values,
            selectors: Vec::new(),
        });
    }
    if values.capacity() != input.len() {
        return Err(Cause::Source);
    }
    values.extend_from_slice(input);
    Ok(values)
}
struct Frame<'a, F, E> {
    source: &'a mut ExprSet,
    nary: &'a mut nary::Parts,
    concat: &'a mut concatenation::Parts,
    pair: &'a mut Vec<ExprRef>,
    lists: lists::Lists,
    fund: &'a F,
    marker: PhantomData<E>,
}
impl<F: Fn(usize) -> Result<(), E>, E> Construction for Frame<'_, F, E> {
    type Error = Cause<E>;
    fn source(&self) -> &ExprSet {
        self.source
    }
    fn list(&mut self, capacity: usize) -> Result<Values, Self::Error> {
        self.lists.take(capacity).map_err(Cause::List)
    }
    fn push(&mut self, list: &mut Values, value: (ExprRef, ExprRef)) -> Result<(), Self::Error> {
        if list.len() == list.capacity() {
            return Err(Cause::Source);
        }
        list.push(value);
        Ok(())
    }
    fn overflow(&self) -> Self::Error {
        Cause::Overflow
    }
    fn pay(&mut self, amount: usize) -> Result<(), Self::Error> {
        self.source.pay_prepared(amount).map_err(Cause::Expression)
    }
    fn literal_derivative(&mut self, root: ExprRef) -> Result<(ExprRef, ExprRef), Self::Error> {
        let Expr::ByteConcat(_, bytes, tail) = self.source.get(root) else {
            return Err(Cause::Source);
        };
        let first = *bytes.first().ok_or(Cause::Source)?;
        let suffix = bytes.get(1..).ok_or(Cause::Source)?;
        let length = suffix.len();
        let mut copy = [0u8; ExprRef::MAX_BYTE_CONCAT];
        copy.get_mut(..length)
            .ok_or(Cause::Source)?
            .copy_from_slice(suffix);
        let selector = self.source.try_mk_byte(first).map_err(Cause::Expression)?;
        let expression = self
            .source
            .try_mk_byte_concat(&copy[..length], tail)
            .map_err(Cause::Expression)?;
        Ok((selector, expression))
    }
    fn byte(&mut self, value: u8) -> Result<ExprRef, Self::Error> {
        self.source.try_mk_byte(value).map_err(Cause::Expression)
    }
    fn remainder(
        &mut self,
        divisor: u32,
        remainder: u32,
        scale: u32,
        fractional: bool,
    ) -> Result<ExprRef, Self::Error> {
        self.source
            .try_mk_remainder_is(divisor, remainder, scale, fractional)
            .map_err(Cause::Expression)
    }
    fn multiply(&mut self, left: u32, right: u32) -> Result<u32, Self::Error> {
        left.checked_mul(right).ok_or(Cause::Overflow)
    }
    fn add(&mut self, left: u32, right: u32) -> Result<u32, Self::Error> {
        left.checked_add(right).ok_or(Cause::Overflow)
    }
    fn power10(&mut self, scale: u32) -> Result<u32, Self::Error> {
        10u32.checked_pow(scale).ok_or(Cause::Overflow)
    }
    fn modulo(&mut self, value: u32, divisor: u32) -> Result<u32, Self::Error> {
        value.checked_rem(divisor).ok_or(Cause::Source)
    }
    fn byte_and(&mut self, a: ExprRef, b: ExprRef) -> Result<ExprRef, Self::Error> {
        self.source
            .try_mk_byte_set_and(a, b)
            .map_err(Cause::Expression)
    }
    fn and(&mut self, a: ExprRef, b: ExprRef) -> Result<ExprRef, Self::Error> {
        if self.pair.capacity() < 2 {
            return Err(Cause::Source);
        }
        self.pair.clear();
        self.pair.push(a);
        self.pair.push(b);
        self.nary
            .apply_to_vec(self.source, self.pair, true)
            .map_err(Cause::Expression)
    }
    fn not(&mut self, arg: ExprRef) -> Result<ExprRef, Self::Error> {
        self.source.try_mk_not(arg).map_err(Cause::Expression)
    }
    fn repeat(&mut self, arg: ExprRef, min: u32, max: u32) -> Result<ExprRef, Self::Error> {
        self.source
            .try_mk_repeat(arg, min, max)
            .map_err(Cause::Expression)
    }
    fn concat(&mut self, left: ExprRef, right: ExprRef) -> Result<ExprRef, Self::Error> {
        self.concat
            .concat(self.source, left, right)
            .map_err(Cause::Expression)
    }
    fn simplify(&mut self, list: Values) -> Result<Values, Self::Error> {
        (self.fund)(grouping::storage::Plan::control_bytes().ok_or(Cause::Overflow)?)
            .map_err(Cause::Funding)?;
        let plan = self
            .source
            .symbolic_grouping_plan(&list)
            .map_err(Cause::Grouping)?;
        (self.fund)(plan.requirements().total).map_err(Cause::Funding)?;
        plan.compile()
            .map_err(Cause::Grouping)?
            .run(list)
            .map_err(Cause::Expression)
    }
    fn disjoint(&mut self, list: &Values) -> Result<Values, Self::Error> {
        (self.fund)(disjoint::storage::Plan::control_bytes().ok_or(Cause::Overflow)?)
            .map_err(Cause::Funding)?;
        let plan = self
            .source
            .symbolic_disjoint_plan(list)
            .map_err(Cause::Disjoint)?;
        (self.fund)(plan.requirements().total).map_err(Cause::Funding)?;
        plan.compile()
            .map_err(Cause::Disjoint)?
            .run(list)
            .map_err(Cause::Expression)
    }
    fn negated_union(&mut self, list: &Values) -> Result<ExprRef, Self::Error> {
        let frames = [
            size_of::<Vec<ExprRef>>(),
            size_of::<Result<ExprRef, Cause<E>>>(),
            size_of::<Result<(), TryReserveError>>(),
        ];
        let controls = frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)
            .ok_or(Cause::Overflow)?;
        let total = Layout::array::<ExprRef>(list.len())
            .map_err(|_| Cause::Overflow)?
            .size()
            .checked_add(controls)
            .ok_or(Cause::Overflow)?;
        (self.fund)(total).map_err(Cause::Funding)?;
        let mut selectors = Vec::new();
        if let Err(error) = selectors.try_reserve_exact(list.len()) {
            return Err(Cause::Allocation {
                error,
                values: Vec::new(),
                selectors,
            });
        }
        if selectors.capacity() != list.len() {
            return Err(Cause::Source);
        }
        for &(selector, _) in list {
            selectors.push(selector);
        }
        self.source
            .try_mk_byte_set_neg_or(&selectors)
            .map_err(Cause::Expression)
    }
}
impl Parts {
    pub(super) fn execute<F: Fn(usize) -> Result<(), E>, E>(
        self,
        source: &mut ExprSet,
        derivative: &mut DerivativeParts,
        root: ExprRef,
        fund: &F,
    ) -> (Self, Result<Values, PreparedSymbolicError<E>>) {
        let Self {
            mut mapping,
            mut memo,
            mut nodes,
        } = self;
        let prepared = (|| {
            mapping.prepare_reached(source)?;
            source.grow_prepared_workspace(&mut memo, source.len())?;
            if memo.len() < source.len() {
                memo.resize_with(source.len(), || None);
            }
            derivative.prepare_reached(source)
        })();
        if let Err(error) = prepared {
            return (
                Self {
                    mapping,
                    memo,
                    nodes,
                },
                Err(PreparedSymbolicError {
                    cause: workspace::OperationError::Process(Cause::Expression(error)),
                }),
            );
        }
        let mut mapping = mapping.bind(source);
        let result = (|| {
            let nary = derivative
                .buffers
                .nary
                .as_mut()
                .ok_or(workspace::OperationError::Process(Cause::Failed))?;
            let concat = derivative
                .buffers
                .concat
                .as_mut()
                .ok_or(workspace::OperationError::Process(Cause::Failed))?;
            let pair = &mut derivative.buffers.alternatives;
            mapping.map(
                root,
                &mut Memo {
                    slots: &mut memo,
                    marker: PhantomData,
                },
                true,
                |r| r,
                |values| value_copy(values, fund),
                |source, children, root| {
                    nodes = nodes.checked_add(1).ok_or(Cause::Overflow)?;
                    fund(lists::Plan::control_bytes().ok_or(Cause::Overflow)?)
                        .map_err(Cause::Funding)?;
                    let plan = lists::Plan::prepare(source, root, children).map_err(Cause::Node)?;
                    fund(plan.requirements().total).map_err(Cause::Funding)?;
                    let lists = plan.compile().map_err(Cause::Node)?;
                    symbolic::node(
                        &mut Frame {
                            source,
                            nary: &mut *nary,
                            concat: &mut *concat,
                            pair: &mut *pair,
                            lists,
                            fund,
                            marker: PhantomData,
                        },
                        children,
                        root,
                    )
                },
            )
        })();
        (
            Self {
                mapping: mapping.into_parts(),
                memo,
                nodes,
            },
            result.map_err(|cause| PreparedSymbolicError { cause }),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ast::ExprFlags, raw::RelevanceCache, AlphabetInfo};
    use std::cell::Cell;
    #[derive(Debug)]
    struct Refusal {
        required: usize,
        limit: usize,
    }
    impl fmt::Display for Refusal {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "required {} exceeds {}", self.required, self.limit)
        }
    }
    impl std::error::Error for Refusal {}
    fn controls<F: Fn(usize) -> Result<(), Refusal>>(_: &F) -> usize {
        operation_control_bytes::<F, Refusal>().unwrap()
    }
    #[test]
    fn owning_symbolic_worker_preserves_shared_rows_memo_copy_funding_and_failed_prefix() {
        let mut source = ExprSet::new(256);
        let a = source.mk_byte(b'a');
        let b = source.mk_byte(b'b');
        let c = source.mk_byte(b'c');
        let ab = source.mk_byte_set_or(&[a, b]);
        let bc = source.mk_byte_set_or(&[b, c]);
        let x = source.mk_byte(b'x');
        let y = source.mk_byte(b'y');
        let ax = source.mk_concat(ab, x);
        let by = source.mk_concat(bc, y);
        let either = source.mk_or(&mut vec![ax, by]);
        let root = source.mk_not(either);
        // Retained structural declaration, not an inferred/default expression arena.
        let declaration = source.mk(Expr::Or(ExprFlags::POSITIVE, &[either; 128]));
        let (_, mut source, _) = AlphabetInfo::from_exprset(source, &[root, declaration]);
        source.reserve(96);
        let mut ordinary = source.clone();
        let mut ordinary_cache = RelevanceCache::new();
        let expected = ordinary_cache.deriv(&mut ordinary, root);
        let prepared = source.prepared_source_plan().unwrap().compile().unwrap();
        drop(source);
        let mut machine = super::super::PreparedExpressionPlan::prepare(prepared)
            .unwrap()
            .compile()
            .unwrap();
        let spent = Cell::new(0usize);
        let limit = Cell::new(64 * 1024 * 1024usize);
        let reserve = |bytes: usize| {
            let required = spent.get().checked_add(bytes).ok_or(Refusal {
                required: usize::MAX,
                limit: limit.get(),
            })?;
            if required > limit.get() {
                return Err(Refusal {
                    required,
                    limit: limit.get(),
                });
            }
            spent.set(required);
            Ok(())
        };
        let result = machine.symbolic_derivative(root, &reserve).unwrap();
        assert_eq!(result, expected);
        assert_eq!(machine.source().cost(), ordinary.cost());
        assert!(!result.is_empty());
        let nodes = machine.num_symbolic_nodes();
        let cost = machine.source().cost();
        let before = spent.get();
        let copied = machine.symbolic_derivative(root, &reserve).unwrap();
        assert_eq!(copied, result);
        assert_ne!(copied.as_ptr(), result.as_ptr());
        assert!(spent.get() > before);
        assert_eq!(machine.num_symbolic_nodes(), nodes);
        assert_eq!(machine.source().cost(), cost);
        let entries = machine.source().len();
        let before = spent.get();
        let control = controls(&reserve);
        limit.set(before + control);
        let error = machine.symbolic_derivative(root, &reserve).unwrap_err();
        assert!(matches!(
            error.cause,
            workspace::OperationError::Copy(Cause::Funding(Refusal { .. }))
        ));
        assert_eq!(spent.get(), before + control);
        assert_eq!(machine.num_symbolic_nodes(), nodes);
        assert_eq!(machine.source().cost(), cost);
        assert_eq!(machine.source().len(), entries);
        assert_eq!(result, copied);
        assert!(machine.derivative(root, b'a').is_err());
        let before = spent.get();
        assert!(matches!(
            machine
                .symbolic_derivative(root, &reserve)
                .unwrap_err()
                .cause,
            workspace::OperationError::Process(Cause::Failed)
        ));
        assert_eq!(spent.get(), before);
    }
}

impl Parts {
    pub(super) fn copy_destination_controls<E>(&self) -> Option<usize> {
        crate::copy_storage::frame_bytes::<(&Self, Self), E>()?.checked_add(self.mapping.copy_destination_controls::<E>()?)
    }
    pub(super) fn copy_destination(&self) -> Self {
        Self { mapping: self.mapping.copy_destination(), memo: Vec::new(), nodes: self.nodes }
    }
    pub(super) fn copy_required_bytes<E>(&self) -> Option<usize> {
        use crate::copy_storage as copy;
        copy::frame_bytes::<(&mut Self, &Self), E>()?
            .checked_add(self.mapping.copy_required_bytes::<E, _>(copy::vectors_required_bytes::<_, E>)?)?
            .checked_add(copy::optional_vectors_required_bytes::<_, E>(&self.memo)?)
    }
}
impl Parts {

    pub(crate) fn restore_copy<F: Fn(usize) -> Result<(), E>, E>(
        &mut self, source: &Self, funding: &F,
    ) -> Result<(), crate::copy_storage::Error<E>> {
        use crate::copy_storage as copy;
        copy::frame::<(&mut Self, &Self), _, _>(funding)?;
        self.mapping.restore_copy(&source.mapping, funding, copy::vectors)?;
        copy::optional_vectors(&mut self.memo, &source.memo, funding)?;
        self.nodes = source.nodes;
        Ok(())
    }
}
