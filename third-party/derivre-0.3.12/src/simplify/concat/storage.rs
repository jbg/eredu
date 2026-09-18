//! Finite concat frames/parts/bytes under the same prepared source loan.
use super::{append, concat, scalar, ConcatElement, Destination, Memory};
use crate::ast::{ExprRef, ExprSet, PreparedExprError, PreparedExprSet};
use std::{
    alloc::Layout,
    collections::TryReserveError,
    fmt,
    mem::{size_of, size_of_val},
};

#[derive(Clone, Copy, Debug)]
struct Geometry {
    frames: usize,
    parts: usize,
    bytes: usize,
}
#[derive(Clone, Copy, Debug)]
struct Mark {
    parts: usize,
    bytes: usize,
    len: usize,
}
#[derive(Clone, Copy, Debug)]
enum Part {
    Expr(ExprRef),
    Bytes { start: usize, len: usize },
}
#[derive(Default)]
struct Buffers {
    frames: Vec<Mark>,
    parts: Vec<Part>,
    bytes: Vec<u8>,
}
#[derive(Clone, Copy, Debug)]
pub(crate) struct Requirements {
    pub(crate) buffers: usize,
    pub(crate) controls: usize,
    pub(crate) total: usize,
}
#[derive(Debug)]
enum Cause {
    Source(PreparedExprError),
    Geometry,
    Capacity,
    Allocation(TryReserveError),
}
pub(crate) struct Failure {
    cause: Cause,
    buffers: Buffers,
}
impl fmt::Debug for Failure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ConcatWorkspaceFailure")
            .field("cause", &self.cause)
            .field("frame_capacity", &self.buffers.frames.capacity())
            .field("part_capacity", &self.buffers.parts.capacity())
            .field("byte_capacity", &self.buffers.bytes.capacity())
            .finish()
    }
}
impl fmt::Display for Failure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.cause {
            Cause::Source(e) => fmt::Display::fmt(e, f),
            Cause::Geometry => f.write_str("concat source workspace geometry overflow"),
            Cause::Capacity => f.write_str("concat workspace differs from its finite capacity"),
            Cause::Allocation(e) => fmt::Display::fmt(e, f),
        }
    }
}
impl std::error::Error for Failure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.cause {
            Cause::Source(e) => Some(e),
            Cause::Allocation(e) => Some(e),
            _ => None,
        }
    }
}
/// Geometry receipt only. The enclosing driver retains the exact source loan.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Blueprint {
    geometry: Geometry,
    requirements: Requirements,
}
pub(crate) struct Parts {
    geometry: Geometry,
    buffers: Buffers,
    failed: bool,
}
impl Blueprint {
    pub(crate) fn prepare(source: &ExprSet) -> Result<Self, Failure> {
        let fail = |cause| Failure {
            cause,
            buffers: Buffers::default(),
        };
        let (_, frames, _) = source
            .storage_extents();
        let parts = frames
            .checked_add(1)
            .and_then(|n| n.checked_mul(frames))
            .ok_or_else(|| fail(Cause::Geometry))?;
        let bytes = frames
            .checked_mul(frames)
            .and_then(|n| n.checked_mul(ExprRef::MAX_BYTE_CONCAT))
            .ok_or_else(|| fail(Cause::Geometry))?;
        let geometry = Geometry {
            frames,
            parts,
            bytes,
        };
        let layouts = [
            Layout::array::<Mark>(frames),
            Layout::array::<Part>(parts),
            Layout::array::<u8>(bytes),
        ];
        let layout_controls = size_of_val(&layouts);
        let buffers = layouts
            .into_iter()
            .try_fold(0usize, |total, layout| {
                total
                    .checked_add(layout.map_err(|_| Cause::Geometry)?.size())
                    .ok_or(Cause::Geometry)
            })
            .map_err(fail)?;
        let controls = Plan::control_bytes()
            .and_then(|n| n.checked_add(layout_controls))
            .ok_or_else(|| fail(Cause::Geometry))?;
        let total = buffers
            .checked_add(controls)
            .ok_or_else(|| fail(Cause::Geometry))?;
        Ok(Self {
            geometry,
            requirements: Requirements {
                buffers,
                controls,
                total,
            },
        })
    }
    pub(crate) fn requirements(&self) -> Requirements {
        self.requirements
    }
    pub(crate) fn compile(self) -> Result<Parts, Failure> {
        let buffers = allocate(self.geometry)?;
        Ok(Parts {
            geometry: self.geometry,
            buffers,
            failed: false,
        })
    }
}
fn allocate(geometry: Geometry) -> Result<Buffers, Failure> {
    let mut buffers = Buffers::default();
    fn reserve<T>(values: &mut Vec<T>, capacity: usize) -> Result<(), Cause> {
        values
            .try_reserve_exact(capacity)
            .map_err(Cause::Allocation)?;
        if values.capacity() != capacity {
            return Err(Cause::Capacity);
        }
        Ok(())
    }
    let result = (|| {
        reserve(&mut buffers.frames, geometry.frames)?;
        reserve(&mut buffers.parts, geometry.parts)?;
        reserve(&mut buffers.bytes, geometry.bytes)
    })();
    match result {
        Ok(()) => Ok(buffers),
        Err(cause) => Err(Failure { cause, buffers }),
    }
}
pub(crate) struct Plan<'a> {
    source: &'a mut ExprSet,
    geometry: Geometry,
    requirements: Requirements,
}
impl fmt::Debug for Plan<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ConcatWorkspacePlan")
            .field("requirements", &self.requirements)
            .finish()
    }
}
pub(crate) struct Scope<'a> {
    source: &'a mut ExprSet,
    geometry: Geometry,
    buffers: Buffers,
    failed: bool,
}
impl fmt::Debug for Scope<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ConcatWorkspace")
            .field("geometry", &self.geometry)
            .field("failed", &self.failed)
            .finish()
    }
}
impl PreparedExprSet {
    /// All source/emitted children precede their parent IDs. The finite entry
    /// limit therefore bounds both a concat chain and simultaneous recursive
    /// frames; each iterator item supplies at most MAX_BYTE_CONCAT source bytes.
    pub(crate) fn concat_scope_plan(&mut self) -> Result<Plan<'_>, Failure> {
        let source = self.source_mut();
        let Blueprint {
            geometry,
            requirements,
        } = Blueprint::prepare(source)?;
        Ok(Plan {
            source,
            geometry,
            requirements,
        })
    }
}
impl<'a> Plan<'a> {
    fn control_bytes() -> Option<usize> {
        let frames = [
            size_of::<Self>(),
            size_of::<Blueprint>(),
            size_of::<Parts>(),
            size_of::<Result<Parts, Failure>>(),
            size_of::<Scope<'a>>(),
            size_of::<Requirements>(),
            size_of::<Geometry>(),
            size_of::<Mark>(),
            size_of::<Part>(),
            size_of::<Buffers>(),
            size_of::<Fixed<'a>>(),
            size_of::<FrameDestination<'a, 'a>>(),
            size_of::<Failure>(),
            size_of::<Cause>(),
            size_of::<Result<Self, Failure>>(),
            size_of::<Result<Scope<'a>, Failure>>(),
            size_of::<Result<(), Cause>>(),
            size_of::<Result<(), TryReserveError>>(),
            size_of::<scalar::Prepared<'a>>(),
        ];
        frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)
    }
    pub(crate) fn requirements(&self) -> Requirements {
        self.requirements
    }
    pub(crate) fn compile(self) -> Result<Scope<'a>, Failure> {
        let buffers = allocate(self.geometry)?;
        Ok(Scope {
            source: self.source,
            geometry: self.geometry,
            buffers,
            failed: false,
        })
    }
}
struct Fixed<'a> {
    geometry: Geometry,
    buffers: &'a mut Buffers,
}
struct FrameDestination<'a, 'b> {
    memory: &'a mut Fixed<'b>,
    frame: usize,
}
impl FrameDestination<'_, '_> {
    fn check(&self, parts: usize, bytes: usize) -> Result<(), PreparedExprError> {
        if self.frame + 1 != self.memory.buffers.frames.len()
            || parts > self.memory.geometry.parts
            || parts > self.memory.buffers.parts.capacity()
            || bytes > self.memory.geometry.bytes
            || bytes > self.memory.buffers.bytes.capacity()
        {
            Err(PreparedExprError::Capacity)
        } else {
            Ok(())
        }
    }
}
impl Destination for FrameDestination<'_, '_> {
    type Error = PreparedExprError;
    fn last_is_bytes(&self) -> bool {
        let frame = self.memory.buffers.frames[self.frame];
        frame.len != 0 && matches!(self.memory.buffers.parts.last(), Some(Part::Bytes { .. }))
    }
    fn extend_bytes(&mut self, input: &[u8]) -> Result<(), Self::Error> {
        let end = self
            .memory
            .buffers
            .bytes
            .len()
            .checked_add(input.len())
            .ok_or(PreparedExprError::Capacity)?;
        self.check(self.memory.buffers.parts.len(), end)?;
        let Some(Part::Bytes { start, len }) = self.memory.buffers.parts.last_mut() else {
            return Err(PreparedExprError::Capacity);
        };
        if start.checked_add(*len) != Some(self.memory.buffers.bytes.len()) {
            return Err(PreparedExprError::Capacity);
        }
        let next = len
            .checked_add(input.len())
            .ok_or(PreparedExprError::Capacity)?;
        self.memory.buffers.bytes.extend_from_slice(input);
        *len = next;
        Ok(())
    }
    fn push_bytes(&mut self, input: &[u8]) -> Result<(), Self::Error> {
        let start = self.memory.buffers.bytes.len();
        let end = start
            .checked_add(input.len())
            .ok_or(PreparedExprError::Capacity)?;
        let parts = self
            .memory
            .buffers
            .parts
            .len()
            .checked_add(1)
            .ok_or(PreparedExprError::Capacity)?;
        self.check(parts, end)?;
        self.memory.buffers.bytes.extend_from_slice(input);
        self.memory.buffers.parts.push(Part::Bytes {
            start,
            len: input.len(),
        });
        self.memory.buffers.frames[self.frame].len += 1;
        Ok(())
    }
    fn push_expr(&mut self, expr: ExprRef) -> Result<(), Self::Error> {
        let parts = self
            .memory
            .buffers
            .parts
            .len()
            .checked_add(1)
            .ok_or(PreparedExprError::Capacity)?;
        self.check(parts, self.memory.buffers.bytes.len())?;
        self.memory.buffers.parts.push(Part::Expr(expr));
        self.memory.buffers.frames[self.frame].len += 1;
        Ok(())
    }
}
impl Memory for Fixed<'_> {
    type Error = PreparedExprError;
    fn begin(&mut self) -> Result<usize, Self::Error> {
        let index = self.buffers.frames.len();
        if index >= self.geometry.frames || index == self.buffers.frames.capacity() {
            return Err(PreparedExprError::Capacity);
        }
        self.buffers.frames.push(Mark {
            parts: self.buffers.parts.len(),
            bytes: self.buffers.bytes.len(),
            len: 0,
        });
        Ok(index)
    }
    fn append(&mut self, frame: usize, element: &ConcatElement<'_>) -> Result<bool, Self::Error> {
        append(
            &mut FrameDestination {
                memory: self,
                frame,
            },
            element,
        )
    }
    fn push_tail(&mut self, frame: usize, tail: ExprRef) -> Result<(), Self::Error> {
        FrameDestination {
            memory: self,
            frame,
        }
        .push_expr(tail)
    }
    fn len(&self, frame: usize) -> usize {
        self.buffers.frames[frame].len
    }
    fn part(&self, frame: usize, index: usize) -> ConcatElement<'_> {
        let mark = self.buffers.frames[frame];
        assert!(index < mark.len);
        match self.buffers.parts[mark.parts + index] {
            Part::Expr(expr) => ConcatElement::Expr(expr),
            Part::Bytes { start, len } => {
                ConcatElement::Bytes(&self.buffers.bytes[start..start + len])
            }
        }
    }
    fn finish(&mut self, frame: usize) {
        assert_eq!(frame + 1, self.buffers.frames.len());
        let mark = self.buffers.frames.pop().expect("existing concat frame");
        self.buffers.parts.truncate(mark.parts);
        self.buffers.bytes.truncate(mark.bytes);
    }
}
impl Scope<'_> {
    pub(crate) fn concat(
        &mut self,
        left: ExprRef,
        right: ExprRef,
    ) -> Result<ExprRef, PreparedExprError> {
        if self.failed {
            return Err(PreparedExprError::Failed);
        }
        if !self.source.is_valid(left) || !self.source.is_valid(right) {
            self.failed = true;
            return Err(PreparedExprError::Source);
        }
        let mut sink = scalar::Prepared(&mut *self.source);
        let mut memory = Fixed {
            geometry: self.geometry,
            buffers: &mut self.buffers,
        };
        let result = concat(&mut sink, &mut memory, left, right);
        if result.is_err() {
            self.failed = true;
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ast::{Expr, ExprEncodingError, ExprFlags},
        raw::HashConsCapacityError,
    };

    fn capacities(scope: &Scope<'_>) -> (usize, usize, usize) {
        (
            scope.buffers.frames.capacity(),
            scope.buffers.parts.capacity(),
            scope.buffers.bytes.capacity(),
        )
    }
    #[test]
    fn shared_concat_source_preserves_recursive_byte_groups_and_failed_frame_custody() {
        let mut source = ExprSet::new(256, crate::ParserAllocationFunding::unenforced()).unwrap();
        let a = source.mk_byte(b'a').unwrap();
        let b = source.mk_byte(b'b').unwrap();
        let c = source.mk_byte(b'c').unwrap();
        let atom = source.mk_repeat(a, 1, 2).unwrap();
        let literal = source.mk_byte_literal(b"0123456789abcdefghijklmnopqrstuv").unwrap();
        let inner = source.mk(Expr::Concat(ExprFlags::POSITIVE, [atom, literal])).unwrap();
        let root = source.mk(Expr::Concat(ExprFlags::POSITIVE, [inner, b])).unwrap();
        let existing_tail = source.mk_byte_concat(b"b", c).unwrap();
        source.reserve(64).unwrap();
        let mut ordinary = source.clone();
        let mut prepared = source.prepared_source_plan().unwrap().compile().unwrap();
        drop(source);
        let plan = prepared.concat_scope_plan().unwrap();
        let requirements = plan.requirements();
        assert_eq!(
            requirements.total,
            requirements.buffers + requirements.controls
        );
        let mut scope = plan.compile().unwrap();
        let capacity = capacities(&scope);
        let result = scope.concat(root, atom).unwrap();
        let expected = ordinary.mk_concat(root, atom).unwrap();
        assert_eq!(result, expected);
        assert_eq!(
            scope.source.expr_to_string(result),
            ordinary.expr_to_string(expected)
        );
        assert_eq!(scope.source.cost(), ordinary.cost());
        assert!(
            scope.buffers.frames.is_empty()
                && scope.buffers.parts.is_empty()
                && scope.buffers.bytes.is_empty()
        );
        assert_eq!(capacities(&scope), capacity);
        assert_eq!(
            scope.concat(ExprRef::EMPTY_STRING, result).unwrap(),
            ordinary.mk_concat(ExprRef::EMPTY_STRING, expected).unwrap()
        );
        assert_eq!(scope.source.cost(), ordinary.cost());

        let mut exhausted = false;
        for offset in 100..1000 {
            match scope.source.try_mk_lookahead(a, offset) {
                Ok(_) => {}
                Err(PreparedExprError::Encoding(ExprEncodingError::Storage(
                    HashConsCapacityError::EntriesExceeded { .. }
                    | HashConsCapacityError::WordsExceeded { .. },
                ))) => {
                    exhausted = true;
                    break;
                }
                Err(error) => panic!("unexpected fixed-arena failure: {error}"),
            }
        }
        assert!(exhausted);
        let entries = scope.source.len();
        let cost = scope.source.cost();
        let failure = scope.concat(root, c).unwrap_err();
        assert!(matches!(
            failure,
            PreparedExprError::Encoding(ExprEncodingError::Storage(
                HashConsCapacityError::EntriesExceeded { .. }
                    | HashConsCapacityError::WordsExceeded { .. }
            ))
        ));
        assert_eq!(scope.source.len(), entries);
        assert!(scope.source.cost() > cost);
        assert!(scope.source.is_valid(existing_tail));
        assert!(scope.buffers.frames.len() >= 2);
        assert!(!scope.buffers.parts.is_empty());
        assert!(scope.buffers.bytes.len() > 1);
        assert_eq!(capacities(&scope), capacity);
        let cost = scope.source.cost();
        assert!(matches!(
            scope.concat(ExprRef::EMPTY_STRING, a),
            Err(PreparedExprError::Failed)
        ));
        assert_eq!(scope.source.cost(), cost);
        drop(scope);
        assert!(prepared.source().is_valid(result));
    }
}

impl Parts {
    pub(crate) fn concat(
        &mut self,
        source: &mut ExprSet,
        left: ExprRef,
        right: ExprRef,
    ) -> Result<ExprRef, PreparedExprError> {
        if self.failed {
            return Err(PreparedExprError::Failed);
        }
        self.failed = true;
        let mut scope = Scope {
            source,
            geometry: self.geometry,
            buffers: std::mem::take(&mut self.buffers),
            failed: false,
        };
        let result = scope.concat(left, right);
        let Scope {
            geometry,
            buffers,
            failed,
            ..
        } = scope;
        self.geometry = geometry;
        self.buffers = buffers;
        self.failed = failed;
        result
    }
}

impl Parts {
    pub(crate) fn fold_expressions(
        &mut self,
        source: &mut ExprSet,
        args: &[ExprRef],
    ) -> Result<ExprRef, PreparedExprError> {
        if self.failed {
            return Err(PreparedExprError::Failed);
        }
        self.failed = true;
        if args.iter().any(|&arg| !source.is_valid(arg)) {
            return Err(PreparedExprError::Source);
        }
        let result = super::fold_expressions(
            &mut scalar::Prepared(source),
            &mut Fixed {
                geometry: self.geometry,
                buffers: &mut self.buffers,
            },
            args,
        );
        if result.is_ok() {
            self.failed = false;
        }
        result
    }
}

impl Parts {
    pub(crate) fn copy_destination_controls<E>(&self) -> Option<usize> {
        crate::copy_storage::frame_bytes::<(&Self, Self), E>()
    }
    pub(crate) fn copy_destination(&self) -> Self {
        Self { geometry: self.geometry, buffers: Buffers::default(), failed: self.failed }
    }
    pub(crate) fn copy_required_bytes<E>(&self) -> Option<usize> {
        use crate::copy_storage as copy;
        copy::frame_bytes::<(&mut Self, &Self), E>()?
            .checked_add(copy::fixed_required_bytes::<_, E>(&self.buffers.frames)?)?
            .checked_add(copy::fixed_required_bytes::<_, E>(&self.buffers.parts)?)?
            .checked_add(copy::fixed_required_bytes::<_, E>(&self.buffers.bytes)?)
    }
}
impl Parts {

    pub(crate) fn restore_copy<F: Fn(usize) -> Result<(), E>, E>(
        &mut self, source: &Self, funding: &F,
    ) -> Result<(), crate::copy_storage::Error<E>> {
        use crate::copy_storage as copy;
        copy::frame::<(&mut Self, &Self), _, _>(funding)?;
        copy::fixed(&mut self.buffers.frames, &source.buffers.frames, funding)?;
        copy::fixed(&mut self.buffers.parts, &source.buffers.parts, funding)?;
        copy::fixed(&mut self.buffers.bytes, &source.buffers.bytes, funding)?;
        self.geometry = source.geometry;
        self.failed = source.failed;
        Ok(())
    }
}
