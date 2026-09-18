//! Source-bound derivative destinations and the shared finite expression consumer.
mod owned;
use super::construction::{self, Construction};
use crate::ast::{
    mapping::{workspace, Cache},
    Expr, ExprRef, ExprSet, PreparedExprError, PreparedExprSet,
};
use crate::simplify::{concat::storage as concatenation, nary::storage as nary};
pub use owned::{
    PreparedExpressionCopyFailure, PreparedExpressionFailure, PreparedExpressionMachine, PreparedExpressionOperationError,
    PreparedExpressionPlan, PreparedExpressionRequirements, PreparedRelevanceError,
    PreparedSymbolicError, PreparedWeightError,
};
use std::{
    alloc::Layout,
    collections::TryReserveError,
    fmt,
    marker::PhantomData,
    mem::{size_of, size_of_val},
};

/// A qualified expression/simplifier producer must implement these mutations.
/// This interface has no ordinary fallback and supplies no allocation allowance.
pub(crate) trait Arena {
    type Error;
    fn multiply(&mut self, left: u32, right: u32) -> Result<u32, Self::Error>;
    fn add(&mut self, left: u32, right: u32) -> Result<u32, Self::Error>;
    fn power10(&mut self, scale: u32) -> Result<u32, Self::Error>;
    fn modulo(&mut self, value: u32, divisor: u32) -> Result<u32, Self::Error>;
    fn byte_concat(
        &mut self,
        source: &mut ExprSet,
        bytes: &[u8],
        tail: ExprRef,
    ) -> Result<ExprRef, Self::Error>;
    fn remainder(
        &mut self,
        source: &mut ExprSet,
        divisor: u32,
        remainder: u32,
        scale: u32,
        fractional: bool,
    ) -> Result<ExprRef, Self::Error>;
    fn and(
        &mut self,
        source: &mut ExprSet,
        args: &mut Vec<ExprRef>,
    ) -> Result<ExprRef, Self::Error>;
    fn or(&mut self, source: &mut ExprSet, args: &mut Vec<ExprRef>)
        -> Result<ExprRef, Self::Error>;
    fn not(&mut self, source: &mut ExprSet, arg: ExprRef) -> Result<ExprRef, Self::Error>;
    fn repeat(
        &mut self,
        source: &mut ExprSet,
        arg: ExprRef,
        min: u32,
        max: u32,
    ) -> Result<ExprRef, Self::Error>;
    fn concat(
        &mut self,
        source: &mut ExprSet,
        left: ExprRef,
        right: ExprRef,
    ) -> Result<ExprRef, Self::Error>;
    fn lookahead(
        &mut self,
        source: &mut ExprSet,
        arg: ExprRef,
        offset: u32,
    ) -> Result<ExprRef, Self::Error>;
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Requirements {
    pub(crate) buffers: usize,
    pub(crate) controls: usize,
    pub(crate) total: usize,
}
#[derive(Clone, Copy, Debug)]
struct Geometry {
    nodes: usize,
    alphabet: usize,
    memo: usize,
    suffix: usize,
    alternatives: usize,
}
struct Buffers {
    memo: Vec<Option<ExprRef>>,
    suffix: Vec<u8>,
    alternatives: Vec<ExprRef>,
    nary: Option<nary::Parts>,
    concat: Option<concatenation::Parts>,
}
impl Buffers {
    fn new() -> Self {
        Self {
            memo: Vec::new(),
            suffix: Vec::new(),
            alternatives: Vec::new(),
            nary: None,
            concat: None,
        }
    }
}
#[derive(Debug)]
enum Cause {
    Geometry,
    Capacity,
    Allocation(TryReserveError),
    Mapping(workspace::ConstructionFailure<ExprRef>),
    Nary(nary::Failure),
    Concat(concatenation::Failure),
}
pub(crate) struct Failure {
    cause: Cause,
    buffers: Buffers,
}
impl fmt::Debug for Failure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DerivativeStorageFailure")
            .field("cause", &self.cause)
            .field("memo_capacity", &self.buffers.memo.capacity())
            .field("suffix_capacity", &self.buffers.suffix.capacity())
            .field(
                "alternative_capacity",
                &self.buffers.alternatives.capacity(),
            )
            .finish()
    }
}
impl fmt::Display for Failure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.cause {
            Cause::Geometry => f.write_str("derivative source geometry is invalid"),
            Cause::Capacity => {
                f.write_str("derivative destination differs from its fixed capacity")
            }
            Cause::Allocation(e) => fmt::Display::fmt(e, f),
            Cause::Mapping(e) => fmt::Display::fmt(e, f),
            Cause::Nary(e) => fmt::Display::fmt(e, f),
            Cause::Concat(e) => fmt::Display::fmt(e, f),
        }
    }
}
impl std::error::Error for Failure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.cause {
            Cause::Allocation(e) => Some(e),
            Cause::Mapping(e) => Some(e),
            Cause::Nary(e) => Some(e),
            Cause::Concat(e) => Some(e),
            _ => None,
        }
    }
}
pub(crate) struct Plan<'a> {
    mapping: workspace::Plan<'a, ExprRef>,
    geometry: Geometry,
    requirements: Requirements,
    nary: nary::Blueprint,
    concat: concatenation::Blueprint,
}
impl fmt::Debug for Plan<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DerivativeStoragePlan")
            .field("requirements", &self.requirements)
            .finish()
    }
}
pub(crate) struct Storage<'a> {
    mapping: workspace::Workspace<'a, ExprRef>,
    buffers: Buffers,
    geometry: Geometry,
    failed: bool,
    num_deriv: usize,
}
struct Parts {
    mapping: workspace::Parts<ExprRef>,
    buffers: Buffers,
    geometry: Geometry,
    failed: bool,
    num_deriv: usize,
}
impl Parts {
    pub(super) fn prepare_reached(&mut self, source: &ExprSet) -> Result<(), PreparedExprError> {
        if source.len() <= self.geometry.nodes {
            return Ok(());
        }
        let memo = source
            .len()
            .checked_mul(self.geometry.alphabet)
            .ok_or(PreparedExprError::Capacity)?;
        source.grow_prepared_workspace(&mut self.buffers.memo, memo)?;
        self.buffers.memo.resize(memo, None);
        self.mapping.prepare_reached(source)?;
        self.geometry.nodes = source.len();
        self.geometry.memo = memo;
        Ok(())
    }
    fn bind(self, source: &mut ExprSet) -> Storage<'_> {
        Storage {
            mapping: self.mapping.bind(source),
            buffers: self.buffers,
            geometry: self.geometry,
            failed: self.failed,
            num_deriv: self.num_deriv,
        }
    }
}
impl Storage<'_> {
    fn into_parts(self) -> Parts {
        Parts {
            mapping: self.mapping.into_parts(),
            buffers: self.buffers,
            geometry: self.geometry,
            failed: self.failed,
            num_deriv: self.num_deriv,
        }
    }
}
impl fmt::Debug for Storage<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DerivativeStorage")
            .field("geometry", &self.geometry)
            .field("failed", &self.failed)
            .finish()
    }
}
impl ExprSet {
    /// Quotes derivative and traversal destinations from the actual owner. A
    /// prepared source adds its finite successor-node and simplifier buffers;
    /// a plain source has no expression-growth or simplification authority.
    pub(crate) fn derivative_storage_plan(&mut self) -> Result<Plan<'_>, Failure> {
        let failure = |cause| Failure {
            cause,
            buffers: Buffers::new(),
        };
        if self.alphabet_size == 0 || self.alphabet_size > 256 {
            return Err(failure(Cause::Geometry));
        }
        let mut geometry = Geometry {
            nodes: self.len(),
            alphabet: self.alphabet_size,
            memo: self
                .len()
                .checked_mul(self.alphabet_size)
                .ok_or_else(|| failure(Cause::Geometry))?,
            suffix: 0,
            alternatives: 2,
        };
        for i in 1..self.len() {
            let id = ExprRef::new(u32::try_from(i).map_err(|_| failure(Cause::Geometry))?);
            if let Expr::ByteConcat(_, bytes, _) = self.get(id) {
                let suffix = bytes
                    .len()
                    .checked_sub(1)
                    .ok_or_else(|| failure(Cause::Geometry))?;
                geometry.suffix = geometry.suffix.max(suffix);
            }
        }
        let (_, nodes, _) = self.storage_extents();
        let nary = nary::Blueprint::prepare(self).map_err(|e| failure(Cause::Nary(e)))?;
        let concat = concatenation::Blueprint::prepare(self).map_err(|e| failure(Cause::Concat(e)))?;
        geometry.nodes = nodes;
        geometry.memo = nodes.checked_mul(geometry.alphabet).ok_or_else(|| failure(Cause::Geometry))?;
        geometry.suffix = ExprRef::MAX_BYTE_CONCAT;
        geometry.alternatives = nary.argument_capacity();
        let mut buffers = Layout::array::<Option<ExprRef>>(geometry.memo)
            .map_err(|_| failure(Cause::Geometry))?
            .size();
        buffers = buffers
            .checked_add(
                Layout::array::<u8>(geometry.suffix)
                    .map_err(|_| failure(Cause::Geometry))?
                    .size(),
            )
            .and_then(|n| n.checked_add(geometry.alternatives.checked_mul(size_of::<ExprRef>())?))
            .ok_or_else(|| failure(Cause::Geometry))?;
        let mut controls =
            Plan::inspection_control_bytes().ok_or_else(|| failure(Cause::Geometry))?;
        let mapping = self
            .mapping_workspace_plan()
            .map_err(|e| failure(Cause::Mapping(e)))?;
        let mapping = mapping.prepared_bounds().map_err(|e| failure(Cause::Mapping(e)))?;
        let n = nary.requirements();
        let c = concat.requirements();
        buffers = buffers.checked_add(n.buffers).and_then(|x| x.checked_add(c.buffers))
            .ok_or_else(|| failure(Cause::Geometry))?;
        controls = controls.checked_add(n.controls).and_then(|x| x.checked_add(c.controls))
            .ok_or_else(|| failure(Cause::Geometry))?;
        let mapped = mapping.requirements();
        let buffers = buffers
            .checked_add(mapped.buffers)
            .ok_or_else(|| failure(Cause::Geometry))?;
        let controls = controls
            .checked_add(mapped.controls)
            .ok_or_else(|| failure(Cause::Geometry))?;
        let total = buffers
            .checked_add(controls)
            .ok_or_else(|| failure(Cause::Geometry))?;
        Ok(Plan {
            mapping,
            nary,
            concat,
            geometry,
            requirements: Requirements {
                buffers,
                controls,
                total,
            },
        })
    }
}
impl<'a> Plan<'a> {
    pub(crate) fn inspection_control_bytes() -> Option<usize> {
        let parts = [
            size_of::<Self>(),
            size_of::<Storage<'a>>(),
            size_of::<Parts>(),
            size_of::<Requirements>(),
            size_of::<Geometry>(),
            size_of::<Buffers>(),
            size_of::<Failure>(),
            size_of::<Cause>(),
            size_of::<Expr<'_>>(),
            size_of::<Result<Self, Failure>>(),
            size_of::<Result<Storage<'a>, Failure>>(),
            size_of::<Result<(), Cause>>(),
            size_of::<Result<(), TryReserveError>>(),
            size_of::<Result<Layout, std::alloc::LayoutError>>(),
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
    pub(crate) fn requirements(&self) -> Requirements {
        self.requirements
    }
    pub(crate) fn compile(self) -> Result<Storage<'a>, Failure> {
        let mut buffers = Buffers::new();
        let result = (|| -> Result<(), Cause> {
            buffers
                .memo
                .try_reserve_exact(self.geometry.memo)
                .map_err(Cause::Allocation)?;
            if buffers.memo.capacity() != self.geometry.memo {
                return Err(Cause::Capacity);
            }
            buffers.memo.resize(self.geometry.memo, None);
            buffers
                .suffix
                .try_reserve_exact(self.geometry.suffix)
                .map_err(Cause::Allocation)?;
            if buffers.suffix.capacity() != self.geometry.suffix {
                return Err(Cause::Capacity);
            }
            buffers
                .alternatives
                .try_reserve_exact(self.geometry.alternatives)
                .map_err(Cause::Allocation)?;
            if buffers.alternatives.capacity() != self.geometry.alternatives {
                return Err(Cause::Capacity);
            }
            buffers.nary = Some(self.nary.compile().map_err(Cause::Nary)?);
            buffers.concat = Some(self.concat.compile().map_err(Cause::Concat)?);
            Ok(())
        })();
        if let Err(cause) = result {
            return Err(Failure { cause, buffers });
        }
        match self.mapping.compile() {
            Ok(mapping) => Ok(Storage {
                mapping,
                buffers,
                geometry: self.geometry,
                failed: false,
                num_deriv: 0,
            }),
            Err(error) => Err(Failure {
                cause: Cause::Mapping(error),
                buffers,
            }),
        }
    }
}

#[derive(Debug)]
pub(crate) enum OperationError<E> {
    Failed,
    Source,
    Suffix,
    Alternatives,
    CounterOverflow,
    Arena(E),
}
impl<E: fmt::Display> fmt::Display for OperationError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Failed => f.write_str("derivative storage retains a failed prefix"),
            Self::Source => f.write_str("derivative occurrence is outside its source"),
            Self::Suffix => f.write_str("derivative suffix exceeds its actual source destination"),
            Self::Alternatives => {
                f.write_str("derivative alternatives lack their fixed destination")
            }
            Self::CounterOverflow => f.write_str("derivative operation count overflow"),
            Self::Arena(e) => fmt::Display::fmt(e, f),
        }
    }
}
impl<E: std::error::Error + 'static> std::error::Error for OperationError<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Arena(e) => Some(e),
            _ => None,
        }
    }
}
struct Memo<'a, E> {
    slots: &'a mut [Option<ExprRef>],
    geometry: Geometry,
    marker: PhantomData<E>,
}
impl<E> Memo<'_, E> {
    fn index(&self, key: &(ExprRef, u8)) -> Option<usize> {
        (key.0.is_valid()
            && key.0.as_usize() < self.geometry.nodes
            && (key.1 as usize) < self.geometry.alphabet)
            .then(|| key.0.as_usize() * self.geometry.alphabet + key.1 as usize)
    }
}
impl<E> Cache<(ExprRef, u8), ExprRef> for Memo<'_, E> {
    type Error = OperationError<E>;
    fn get(&self, key: &(ExprRef, u8)) -> Option<&ExprRef> {
        self.slots.get(self.index(key)?)?.as_ref()
    }
    fn contains_key(&self, key: &(ExprRef, u8)) -> bool {
        self.get(key).is_some()
    }
    fn insert(&mut self, key: (ExprRef, u8), value: ExprRef) -> Result<(), Self::Error> {
        let index = self.index(&key).ok_or(OperationError::Source)?;
        let slot = self.slots.get_mut(index).ok_or(OperationError::Source)?;
        *slot = Some(value);
        Ok(())
    }
}
struct Bound<'a, A> {
    source: &'a mut ExprSet,
    arena: &'a mut A,
    suffix: &'a mut Vec<u8>,
    alternatives: &'a mut Vec<ExprRef>,
    geometry: Geometry,
}
impl<A: Arena> Construction for Bound<'_, A> {
    type Error = OperationError<A::Error>;
    fn source(&self) -> &ExprSet {
        self.source
    }
    fn multiply(&mut self, left: u32, right: u32) -> Result<u32, Self::Error> {
        self.arena
            .multiply(left, right)
            .map_err(OperationError::Arena)
    }
    fn add(&mut self, left: u32, right: u32) -> Result<u32, Self::Error> {
        self.arena.add(left, right).map_err(OperationError::Arena)
    }
    fn power10(&mut self, scale: u32) -> Result<u32, Self::Error> {
        self.arena.power10(scale).map_err(OperationError::Arena)
    }
    fn modulo(&mut self, value: u32, divisor: u32) -> Result<u32, Self::Error> {
        self.arena
            .modulo(value, divisor)
            .map_err(OperationError::Arena)
    }

    fn byte_suffix(&mut self, root: ExprRef, tail: ExprRef) -> Result<ExprRef, Self::Error> {
        let Expr::ByteConcat(_, bytes, actual_tail) = self.source.get(root) else {
            return Err(OperationError::Source);
        };
        let suffix = bytes.get(1..).ok_or(OperationError::Source)?;
        if actual_tail != tail {
            return Err(OperationError::Source);
        }
        if suffix.len() > self.geometry.suffix || suffix.len() > self.suffix.capacity() {
            return Err(OperationError::Suffix);
        }
        self.suffix.clear();
        self.suffix.extend_from_slice(suffix);
        self.arena
            .byte_concat(self.source, self.suffix, tail)
            .map_err(OperationError::Arena)
    }
    fn remainder(
        &mut self,
        divisor: u32,
        remainder: u32,
        scale: u32,
        fractional: bool,
    ) -> Result<ExprRef, Self::Error> {
        self.arena
            .remainder(self.source, divisor, remainder, scale, fractional)
            .map_err(OperationError::Arena)
    }
    fn and(&mut self, args: &mut Vec<ExprRef>) -> Result<ExprRef, Self::Error> {
        self.arena
            .and(self.source, args)
            .map_err(OperationError::Arena)
    }
    fn or(&mut self, args: &mut Vec<ExprRef>) -> Result<ExprRef, Self::Error> {
        self.arena
            .or(self.source, args)
            .map_err(OperationError::Arena)
    }
    fn not(&mut self, arg: ExprRef) -> Result<ExprRef, Self::Error> {
        self.arena
            .not(self.source, arg)
            .map_err(OperationError::Arena)
    }
    fn repeat(&mut self, arg: ExprRef, min: u32, max: u32) -> Result<ExprRef, Self::Error> {
        self.arena
            .repeat(self.source, arg, min, max)
            .map_err(OperationError::Arena)
    }
    fn concat(&mut self, left: ExprRef, right: ExprRef) -> Result<ExprRef, Self::Error> {
        self.arena
            .concat(self.source, left, right)
            .map_err(OperationError::Arena)
    }
    fn alternatives(&mut self, left: ExprRef, right: ExprRef) -> Result<ExprRef, Self::Error> {
        if self.alternatives.capacity() < 2 {
            return Err(OperationError::Alternatives);
        }
        self.alternatives.clear();
        self.alternatives.push(left);
        self.alternatives.push(right);
        self.arena
            .or(self.source, self.alternatives)
            .map_err(OperationError::Arena)
    }
    fn lookahead(&mut self, arg: ExprRef, offset: u32) -> Result<ExprRef, Self::Error> {
        self.arena
            .lookahead(self.source, arg, offset)
            .map_err(OperationError::Arena)
    }
}
struct Frame<'a, A> {
    arena: &'a mut A,
    suffix: &'a mut Vec<u8>,
    alternatives: &'a mut Vec<ExprRef>,
    num_deriv: &'a mut usize,
    geometry: Geometry,
    byte: u8,
}
impl<A: Arena> Frame<'_, A> {
    fn node(
        &mut self,
        source: &mut ExprSet,
        derivatives: &mut Vec<ExprRef>,
        root: ExprRef,
    ) -> Result<ExprRef, OperationError<A::Error>> {
        *self.num_deriv = self
            .num_deriv
            .checked_add(1)
            .ok_or(OperationError::CounterOverflow)?;
        construction::node(
            &mut Bound {
                source,
                arena: self.arena,
                suffix: self.suffix,
                alternatives: self.alternatives,
                geometry: self.geometry,
            },
            derivatives,
            root,
            self.byte,
        )
    }
}
impl Storage<'_> {
    /// Local typed processor/error frames; the enclosing mapping-call transport
    /// and Arena constructor quote remain additional admission requirements.
    pub(crate) fn operation_control_bytes<A: Arena>() -> Option<usize> {
        let parts = [
            size_of::<Bound<'_, A>>(),
            size_of::<Frame<'_, A>>(),
            size_of::<&mut Frame<'_, A>>(),
            size_of::<Memo<'_, A::Error>>(),
            size_of::<OperationError<A::Error>>(),
            size_of::<workspace::OperationError<OperationError<A::Error>>>(),
            size_of::<Result<ExprRef, OperationError<A::Error>>>(),
            size_of::<Result<ExprRef, workspace::OperationError<OperationError<A::Error>>>>(),
            size_of::<Expr<'_>>(),
            size_of::<&mut A>(),
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
    pub(crate) fn derivative<A: Arena>(
        &mut self,
        arena: &mut A,
        root: ExprRef,
        byte: u8,
    ) -> Result<ExprRef, workspace::OperationError<OperationError<A::Error>>> {
        let refuse = |e| workspace::OperationError::Process(e);
        if self.failed {
            return Err(refuse(OperationError::Failed));
        }
        if !self.mapping.source().is_valid(root)
            || root.as_usize() >= self.geometry.nodes
            || (byte as usize) >= self.geometry.alphabet
        {
            self.failed = true;
            return Err(refuse(OperationError::Source));
        }
        if construction::surely_no_match(self.mapping.source(), root, byte) {
            return Ok(ExprRef::NO_MATCH);
        }
        let Buffers {
            memo,
            suffix,
            alternatives,
            ..
        } = &mut self.buffers;
        let geometry = self.geometry;
        let mut frame = Frame {
            arena,
            suffix,
            alternatives,
            num_deriv: &mut self.num_deriv,
            geometry,
            byte,
        };
        let result = self.mapping.map(
            root,
            &mut Memo {
                slots: memo,
                geometry,
                marker: PhantomData,
            },
            true,
            |r| (r, byte),
            |r| Ok(*r),
            |source, derivatives, root| frame.node(source, derivatives, root),
        );
        if result.is_err() {
            self.failed = true;
        }
        result
    }
}

/// These bodies are borrowed only by the storage holding the same expression
/// loan. Geometry receipts never supply an expression or allocation authority.
struct Builtin {
    nary: nary::Parts,
    concat: concatenation::Parts,
}
impl Arena for Builtin {
    type Error = PreparedExprError;
    fn multiply(&mut self, left: u32, right: u32) -> Result<u32, Self::Error> {
        left.checked_mul(right).ok_or(PreparedExprError::Source)
    }
    fn add(&mut self, left: u32, right: u32) -> Result<u32, Self::Error> {
        left.checked_add(right).ok_or(PreparedExprError::Source)
    }
    fn power10(&mut self, scale: u32) -> Result<u32, Self::Error> {
        10u32.checked_pow(scale).ok_or(PreparedExprError::Source)
    }
    fn modulo(&mut self, value: u32, divisor: u32) -> Result<u32, Self::Error> {
        value.checked_rem(divisor).ok_or(PreparedExprError::Source)
    }
    fn byte_concat(
        &mut self,
        source: &mut ExprSet,
        bytes: &[u8],
        tail: ExprRef,
    ) -> Result<ExprRef, Self::Error> {
        source.try_mk_byte_concat(bytes, tail)
    }
    fn remainder(
        &mut self,
        source: &mut ExprSet,
        divisor: u32,
        remainder: u32,
        scale: u32,
        fractional: bool,
    ) -> Result<ExprRef, Self::Error> {
        source.try_mk_remainder_is(divisor, remainder, scale, fractional)
    }
    fn and(
        &mut self,
        source: &mut ExprSet,
        args: &mut Vec<ExprRef>,
    ) -> Result<ExprRef, Self::Error> {
        self.nary.apply_to_vec(source, args, true)
    }
    fn or(
        &mut self,
        source: &mut ExprSet,
        args: &mut Vec<ExprRef>,
    ) -> Result<ExprRef, Self::Error> {
        self.nary.apply_to_vec(source, args, false)
    }
    fn not(&mut self, source: &mut ExprSet, arg: ExprRef) -> Result<ExprRef, Self::Error> {
        source.try_mk_not(arg)
    }
    fn repeat(
        &mut self,
        source: &mut ExprSet,
        arg: ExprRef,
        min: u32,
        max: u32,
    ) -> Result<ExprRef, Self::Error> {
        source.try_mk_repeat(arg, min, max)
    }
    fn concat(
        &mut self,
        source: &mut ExprSet,
        left: ExprRef,
        right: ExprRef,
    ) -> Result<ExprRef, Self::Error> {
        self.concat.concat(source, left, right)
    }
    fn lookahead(
        &mut self,
        source: &mut ExprSet,
        arg: ExprRef,
        offset: u32,
    ) -> Result<ExprRef, Self::Error> {
        source.try_mk_lookahead(arg, offset)
    }
}
impl PreparedExprSet {
    /// The actual prepared arena and all scratch use one mutable source loan.
    pub(crate) fn derivative_scope_plan(&mut self) -> Result<Plan<'_>, Failure> {
        self.source_mut().derivative_storage_plan()
    }
}
impl Storage<'_> {
    pub(crate) fn source(&self) -> &ExprSet {
        self.mapping.source()
    }
    pub(crate) fn num_derivatives(&self) -> usize {
        self.num_deriv
    }
    /// Fixed adapter/body frames. Enclosing mapping and recursive concat call
    /// controls are still supplied by the grammar operation's admission owner.
    pub(crate) fn prepared_operation_control_bytes() -> Option<usize> {
        let parts = [
            Self::operation_control_bytes::<Builtin>()?,
            size_of::<Builtin>(),
            size_of::<Option<nary::Parts>>(),
            size_of::<Option<concatenation::Parts>>(),
            size_of::<Result<ExprRef, PreparedExprError>>(),
            size_of::<&[ExprRef]>(),
            size_of::<u32>() * 8,
            crate::simplify::remainder::fixed_control_bytes()?,
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
    pub(crate) fn derivative_prepared(
        &mut self,
        root: ExprRef,
        byte: u8,
    ) -> Result<ExprRef, workspace::OperationError<OperationError<PreparedExprError>>> {
        if self.failed {
            return Err(workspace::OperationError::Process(OperationError::Failed));
        }
        // Plain-source storage has no prepared simplifier authority. Taking
        // both bodies also fences an interrupted invocation against reuse.
        if self.buffers.nary.is_none() || self.buffers.concat.is_none() {
            self.failed = true;
            return Err(workspace::OperationError::Process(OperationError::Source));
        }
        let mut arena = Builtin {
            nary: self
                .buffers
                .nary
                .take()
                .expect("checked prepared n-ary body"),
            concat: self
                .buffers
                .concat
                .take()
                .expect("checked prepared concat body"),
        };
        let result = self.derivative(&mut arena, root, byte);
        self.buffers.nary = Some(arena.nary);
        self.buffers.concat = Some(arena.concat);
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::raw::DerivCache;

    // Ordinary arena construction is explicit test input, never an available
    // production fallback for this prepared storage adapter.
    struct TestArena {
        suffix_calls: usize,
        fail_suffix: bool,
        completed: Option<ExprRef>,
    }
    #[derive(Debug)]
    struct Refusal;
    impl Arena for TestArena {
        type Error = Refusal;
        fn multiply(&mut self, left: u32, right: u32) -> Result<u32, Refusal> {
            Ok(left * right)
        }
        fn add(&mut self, left: u32, right: u32) -> Result<u32, Refusal> {
            Ok(left + right)
        }
        fn power10(&mut self, scale: u32) -> Result<u32, Refusal> {
            Ok(10u32.pow(scale))
        }
        fn modulo(&mut self, value: u32, divisor: u32) -> Result<u32, Refusal> {
            Ok(value % divisor)
        }

        fn byte_concat(
            &mut self,
            source: &mut ExprSet,
            bytes: &[u8],
            tail: ExprRef,
        ) -> Result<ExprRef, Refusal> {
            self.suffix_calls += 1;
            let value = source.mk_byte_concat(bytes, tail).unwrap();
            self.completed = Some(value);
            if self.fail_suffix {
                Err(Refusal)
            } else {
                Ok(value)
            }
        }
        fn remainder(
            &mut self,
            source: &mut ExprSet,
            divisor: u32,
            remainder: u32,
            scale: u32,
            fractional: bool,
        ) -> Result<ExprRef, Refusal> {
            Ok(source.mk_remainder_is(divisor, remainder, scale, fractional).unwrap())
        }
        fn and(
            &mut self,
            source: &mut ExprSet,
            args: &mut Vec<ExprRef>,
        ) -> Result<ExprRef, Refusal> {
            Ok(source.mk_and(args).unwrap())
        }
        fn or(
            &mut self,
            source: &mut ExprSet,
            args: &mut Vec<ExprRef>,
        ) -> Result<ExprRef, Refusal> {
            Ok(source.mk_or(args).unwrap())
        }
        fn not(&mut self, source: &mut ExprSet, arg: ExprRef) -> Result<ExprRef, Refusal> {
            Ok(source.mk_not(arg).unwrap())
        }
        fn repeat(
            &mut self,
            source: &mut ExprSet,
            arg: ExprRef,
            min: u32,
            max: u32,
        ) -> Result<ExprRef, Refusal> {
            Ok(source.mk_repeat(arg, min, max).unwrap())
        }
        fn concat(
            &mut self,
            source: &mut ExprSet,
            left: ExprRef,
            right: ExprRef,
        ) -> Result<ExprRef, Refusal> {
            Ok(source.mk_concat(left, right).unwrap())
        }
        fn lookahead(
            &mut self,
            source: &mut ExprSet,
            arg: ExprRef,
            offset: u32,
        ) -> Result<ExprRef, Refusal> {
            Ok(source.mk_lookahead(arg, offset).unwrap())
        }
    }
    fn capacities(storage: &Storage<'_>) -> (usize, usize, usize) {
        (
            storage.buffers.memo.capacity(),
            storage.buffers.suffix.capacity(),
            storage.buffers.alternatives.capacity(),
        )
    }

    #[test]
    fn actual_derivative_source_slots_preserve_hits_new_nodes_and_failed_arena_prefixes() {
        let mut source = ExprSet::new(256, crate::ParserAllocationFunding::unenforced()).unwrap();
        let root = source.mk_byte_literal(b"abcdef").unwrap();
        let mut ordinary = source.clone();
        let ordinary_result = DerivCache::new().derivative(&mut ordinary, root, b'a').unwrap();
        let nodes = source.len();
        let plan = source.derivative_storage_plan().unwrap();
        let requirements = plan.requirements();
        assert_eq!(
            requirements.total,
            requirements.buffers + requirements.controls
        );
        assert!(requirements.buffers >= nodes * 256 * size_of::<Option<ExprRef>>());
        assert!(Storage::operation_control_bytes::<TestArena>().unwrap() > 0);
        let mut storage = plan.compile().unwrap();
        let original_capacities = capacities(&storage);
        assert_eq!(original_capacities.0, nodes * 256);
        assert!(original_capacities.1 >= b"bcdef".len());
        assert!(original_capacities.2 >= 2);
        let mut arena = TestArena {
            suffix_calls: 0,
            fail_suffix: false,
            completed: None,
        };
        let result = storage.derivative(&mut arena, root, b'a').unwrap();
        assert_eq!(result, ordinary_result);
        assert_eq!(storage.buffers.suffix.as_slice(), b"bcdef");
        assert_eq!(storage.derivative(&mut arena, root, b'a').unwrap(), result);
        assert_eq!(
            storage.derivative(&mut arena, root, b'z').unwrap(),
            ExprRef::NO_MATCH
        );
        assert_eq!(storage.num_deriv, 1);
        assert_eq!(arena.suffix_calls, 1);
        assert_eq!(capacities(&storage), original_capacities);
        assert!(result.as_usize() >= nodes);
        assert!(matches!(
            storage.derivative(&mut arena, result, b'b'),
            Err(workspace::OperationError::Process(OperationError::Source))
        ));
        assert!(matches!(
            storage.derivative(&mut arena, root, b'z'),
            Err(workspace::OperationError::Process(OperationError::Failed))
        ));
        drop(storage);
        assert!(source.is_valid(result));

        let nodes = source.len();
        let mut storage = source.derivative_storage_plan().unwrap().compile().unwrap();
        let original_capacities = capacities(&storage);
        arena.fail_suffix = true;
        let error = storage.derivative(&mut arena, result, b'b').unwrap_err();
        assert!(matches!(
            error,
            workspace::OperationError::Process(OperationError::Arena(Refusal))
        ));
        let prefix = arena.completed.unwrap();
        assert!(prefix.as_usize() >= nodes);
        assert_eq!(storage.buffers.suffix.as_slice(), b"cdef");
        assert_eq!(storage.num_deriv, 1);
        assert!(storage.buffers.memo.iter().all(Option::is_none));
        assert_eq!(capacities(&storage), original_capacities);
        assert!(matches!(
            storage.derivative(&mut arena, root, b'z'),
            Err(workspace::OperationError::Process(OperationError::Failed))
        ));
        drop(storage);
        assert!(
            matches!(source.get(prefix), Expr::ByteConcat(_, bytes, tail)
            if bytes == b"cdef" && tail == ExprRef::EMPTY_STRING)
        );
    }

    #[test]
    fn prepared_derivatives_use_actual_simplifiers_successor_slots_and_failed_prefix_custody() {
        use crate::AlphabetInfo;

        let mut source = ExprSet::new(256, crate::ParserAllocationFunding::unenforced()).unwrap();
        let literal = source.mk_byte_literal(b"abcdef").unwrap();
        // This is actual historical encoding/storage, inspected by the source
        // plan; neither a parser fuel value nor a default arena allowance.
        let encoding_source = source.mk_byte_literal(&[b'q'; 256]).unwrap();
        let (_, mut source, _) = AlphabetInfo::from_exprset(source, &[literal, encoding_source]).unwrap();
        assert!(!source.optimize);
        let a = source.mk_byte(b'a').unwrap();
        let optional = source.mk_repeat(a, 0, 2).unwrap();
        let tail = source.mk_byte_literal(b"bcdef").unwrap();
        let chain = source.mk_concat(optional, tail).unwrap();
        let alternatives = source.mk_or(&mut vec![chain, literal]).unwrap();
        let excluded = source.mk_byte_literal(b"ax").unwrap();
        let not = source.mk_not(excluded).unwrap();
        let intersection = source.mk_and(&mut vec![alternatives, not]).unwrap();
        let ahead = source.mk_lookahead(literal, 0).unwrap();
        let remainder = source.mk_remainder_is(13, 3, 1, false).unwrap();
        let overflow = source.mk_remainder_is(u32::MAX, u32::MAX - 1, 1, false).unwrap();
        source.reserve(96).unwrap();
        let initial_nodes = source.len();
        let mut ordinary = source.clone();
        let mut prepared = source.prepared_source_plan().unwrap().compile().unwrap();
        drop(source);
        let extents = prepared.source().storage_extents();
        let plan = prepared.derivative_scope_plan().unwrap();
        let requirements = plan.requirements();
        assert_eq!(
            requirements.total,
            requirements.buffers + requirements.controls
        );
        assert!(requirements.buffers >= extents.1 * 256 * size_of::<Option<ExprRef>>());
        assert!(Storage::prepared_operation_control_bytes().unwrap() > 0);
        let mut scope = plan.compile().unwrap();
        let capacity = capacities(&scope);
        let mut ordinary_cache = DerivCache::new();
        let mut created_successor = false;
        let cases: &[(ExprRef, &[u8])] = &[
            (literal, b"abcdef"),
            (chain, b"aabcdef"),
            (alternatives, b"abcdef"),
            (intersection, b"abcdef"),
            (ahead, b"abc"),
            (remainder, b"2.3"),
        ];
        let mut retained = ExprRef::EMPTY_STRING;
        for &(mut root, bytes) in cases {
            for &byte in bytes {
                let expected = ordinary_cache.derivative(&mut ordinary, root, byte).unwrap();
                let result = scope.derivative_prepared(root, byte).unwrap();
                assert_eq!(result, expected);
                assert_eq!(scope.source().cost(), ordinary.cost());
                assert_eq!(scope.num_derivatives(), ordinary_cache.num_deriv);
                assert_eq!(
                    scope.source().expr_to_string(result),
                    ordinary.expr_to_string(expected)
                );
                let calls = scope.num_derivatives();
                assert_eq!(scope.derivative_prepared(root, byte).unwrap(), result);
                assert_eq!(scope.num_derivatives(), calls);
                assert_eq!(capacities(&scope), capacity);
                assert_eq!(scope.source().storage_extents(), extents);
                created_successor |= result.as_usize() >= initial_nodes;
                root = result;
                retained = result;
            }
        }
        assert!(created_successor);
        let entries = scope.source().len();
        let cost = scope.source().cost();
        let calls = scope.num_derivatives();
        let completed = scope
            .buffers
            .memo
            .iter()
            .filter(|value| value.is_some())
            .count();
        assert!(completed > 0);
        let error = scope.derivative_prepared(overflow, b'1').unwrap_err();
        assert!(matches!(
            error,
            workspace::OperationError::Process(OperationError::Arena(PreparedExprError::Source))
        ));
        assert_eq!(scope.num_derivatives(), calls + 1);
        assert_eq!(scope.source().cost(), cost);
        assert_eq!(scope.source().len(), entries);
        assert_eq!(
            scope
                .buffers
                .memo
                .iter()
                .filter(|value| value.is_some())
                .count(),
            completed
        );
        assert_eq!(capacities(&scope), capacity);
        assert!(scope.buffers.nary.is_some() && scope.buffers.concat.is_some());
        assert!(matches!(
            scope.derivative_prepared(literal, b'a'),
            Err(workspace::OperationError::Process(OperationError::Failed))
        ));
        drop(scope);
        assert!(prepared.source().is_valid(retained));
        assert_eq!(prepared.source().len(), entries);

        // Original and independently copied lexical sources share mutation bodies.
        let mut plain = ExprSet::new(256, crate::ParserAllocationFunding::unenforced()).unwrap();
        let mut plain_scope = plain.derivative_storage_plan().unwrap().compile().unwrap();
        assert_eq!(plain_scope.derivative_prepared(ExprRef::EMPTY_STRING, b'a').unwrap(), ExprRef::NO_MATCH);
    }
}

impl Parts {
    fn copy_destination_controls<E>(&self) -> Option<usize> {
        let mut bytes = crate::copy_storage::frame_bytes::<(&Self, Self), E>()?
            .checked_add(self.mapping.copy_destination_controls::<E>()?)?;
        if let Some(source) = &self.buffers.nary { bytes = bytes.checked_add(source.copy_destination_controls::<E>()?)?; }
        if let Some(source) = &self.buffers.concat { bytes = bytes.checked_add(source.copy_destination_controls::<E>()?)?; }
        Some(bytes)
    }
    fn copy_destination(&self) -> Self {
        Self {
            mapping: self.mapping.copy_destination(),
            buffers: Buffers {
                nary: self.buffers.nary.as_ref().map(nary::Parts::copy_destination),
                concat: self.buffers.concat.as_ref().map(concatenation::Parts::copy_destination),
                ..Buffers::new()
            },
            geometry: self.geometry, failed: self.failed, num_deriv: self.num_deriv,
        }
    }
    fn copy_required_bytes<E>(&self) -> Option<usize> {
        use crate::copy_storage as copy;
        let mut bytes = copy::frame_bytes::<(&mut Self, &Self), E>()?
            .checked_add(self.mapping.copy_required_bytes::<E, _>(copy::fixed_required_bytes::<_, E>)?)?
            .checked_add(copy::fixed_required_bytes::<_, E>(&self.buffers.memo)?)?
            .checked_add(copy::fixed_required_bytes::<_, E>(&self.buffers.suffix)?)?
            .checked_add(copy::fixed_required_bytes::<_, E>(&self.buffers.alternatives)?)?;
        if let Some(source) = &self.buffers.nary { bytes = bytes.checked_add(source.copy_required_bytes::<E>()?)?; }
        if let Some(source) = &self.buffers.concat { bytes = bytes.checked_add(source.copy_required_bytes::<E>()?)?; }
        Some(bytes)
    }
}
impl Parts {

    pub(crate) fn restore_copy<F: Fn(usize) -> Result<(), E>, E>(
        &mut self, source: &Self, funding: &F,
    ) -> Result<(), crate::copy_storage::Error<E>> {
        use crate::copy_storage as copy;
        copy::frame::<(&mut Self, &Self), _, _>(funding)?;
        self.mapping.restore_copy(&source.mapping, funding, copy::fixed)?;
        copy::fixed(&mut self.buffers.memo, &source.buffers.memo, funding)?;
        copy::fixed(&mut self.buffers.suffix, &source.buffers.suffix, funding)?;
        copy::fixed(&mut self.buffers.alternatives, &source.buffers.alternatives, funding)?;
        if let Some(source) = &source.buffers.nary {
            self.buffers.nary.as_mut().ok_or(copy::Error::Source)?.restore_copy(source, funding)?;
        } else { self.buffers.nary = None; }
        if let Some(source) = &source.buffers.concat {
            self.buffers.concat.as_mut().ok_or(copy::Error::Source)?.restore_copy(source, funding)?;
        } else { self.buffers.concat = None; }
        self.geometry = source.geometry;
        self.failed = source.failed;
        self.num_deriv = source.num_deriv;
        Ok(())
    }
}
