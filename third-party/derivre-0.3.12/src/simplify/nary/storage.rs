//! Paid finite n-ary destinations bound to one prepared expression owner.
use super::{Lookahead, Memory, and, or, scalar};
use crate::ast::{Expr, ExprRef, ExprSet, ExprTag, PreparedExprError, PreparedExprSet};
use std::{
    alloc::Layout,
    collections::TryReserveError,
    fmt,
    mem::{size_of, size_of_val},
};

#[derive(Clone, Copy, Debug)]
struct Geometry {
    input: usize,
    flat: usize,
    words: usize,
}
#[derive(Clone, Copy, Debug)]
pub(crate) struct Requirements {
    pub(crate) buffers: usize,
    pub(crate) controls: usize,
    pub(crate) total: usize,
}
#[derive(Default)]
struct Buffers {
    args: Vec<ExprRef>,
    suffix: Vec<ExprRef>,
    bytes: Vec<u32>,
    lookahead: Vec<Lookahead>,
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
        f.debug_struct("NaryWorkspaceFailure")
            .field("cause", &self.cause)
            .field("argument_capacity", &self.buffers.args.capacity())
            .field("suffix_capacity", &self.buffers.suffix.capacity())
            .field("byte_capacity", &self.buffers.bytes.capacity())
            .field("lookahead_capacity", &self.buffers.lookahead.capacity())
            .finish()
    }
}
impl fmt::Display for Failure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.cause {
            Cause::Source(e) => fmt::Display::fmt(e, f),
            Cause::Geometry => f.write_str("n-ary source geometry overflow"),
            Cause::Capacity => f.write_str("n-ary destination differs from its fixed capacity"),
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
        Self::prepare_inputs(source, 0)
    }
    pub(crate) fn prepare_inputs(source: &ExprSet, required: usize) -> Result<Self, Failure> {
        let fail = |cause| Failure {
            cause,
            buffers: Buffers::default(),
        };
        let (_, _, encoded) = source
            .prepared_extents()
            .map_err(|e| fail(Cause::Source(e)))?;
        let input = encoded
            .checked_sub(1)
            .filter(|&n| n != 0)
            .ok_or_else(|| fail(Cause::Geometry))?;
        let encoded_args = input;
        let input = input.max(required);
        let geometry = Geometry {
            input,
            flat: input
                .checked_mul(encoded_args)
                .ok_or_else(|| fail(Cause::Geometry))?,
            words: source.alphabet_words,
        };
        let layouts = [
            Layout::array::<ExprRef>(geometry.flat),
            Layout::array::<ExprRef>(geometry.input),
            Layout::array::<u32>(geometry.words),
            Layout::array::<Lookahead>(geometry.flat),
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
        reserve(&mut buffers.args, geometry.flat)?;
        reserve(&mut buffers.suffix, geometry.input)?;
        reserve(&mut buffers.bytes, geometry.words)?;
        buffers.bytes.resize(geometry.words, 0);
        reserve(&mut buffers.lookahead, geometry.flat)
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
        f.debug_struct("NaryWorkspacePlan")
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
        f.debug_struct("NaryWorkspace")
            .field("geometry", &self.geometry)
            .field("failed", &self.failed)
            .finish()
    }
}
impl PreparedExprSet {
    /// Source-owned encoded-word limits bound each n-ary argument list and each
    /// one-level expansion. There is no default/parser-fuel capacity substitute.
    pub(crate) fn nary_scope_plan(&mut self) -> Result<Plan<'_>, Failure> {
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
            size_of::<Geometry>(),
            size_of::<Requirements>(),
            size_of::<Buffers>(),
            size_of::<Fixed<'a>>(),
            size_of::<Failure>(),
            size_of::<Cause>(),
            size_of::<Result<Self, Failure>>(),
            size_of::<Result<Scope<'a>, Failure>>(),
            size_of::<Result<(), Cause>>(),
            size_of::<Result<(), TryReserveError>>(),
            size_of::<Result<ExprRef, PreparedExprError>>(),
            size_of::<scalar::Prepared<'a>>(),
            size_of::<Expr<'a>>(),
            size_of::<Lookahead>(),
            size_of::<std::slice::Iter<'a, ExprRef>>(),
            size_of::<std::slice::Iter<'a, Lookahead>>(),
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
    suffix: &'a mut Vec<ExprRef>,
    bytes: &'a mut Vec<u32>,
    lookahead: &'a mut Vec<Lookahead>,
}
impl Memory for Fixed<'_> {
    type Error = PreparedExprError;
    fn copy_suffix(&mut self, input: &[ExprRef]) -> Result<(), Self::Error> {
        if input.len() > self.geometry.input || input.len() > self.suffix.capacity() {
            return Err(PreparedExprError::Capacity);
        }
        self.suffix.clear();
        self.suffix.extend_from_slice(input);
        Ok(())
    }
    fn suffix(&self) -> &[ExprRef] {
        self.suffix
    }
    fn finish_suffix(&mut self) {
        self.suffix.clear();
    }
    fn ensure_args(&mut self, args: &mut Vec<ExprRef>, required: usize) -> Result<(), Self::Error> {
        if required > self.geometry.flat || required > args.capacity() {
            Err(PreparedExprError::Capacity)
        } else {
            Ok(())
        }
    }
    fn bytes(&mut self, words: usize) -> Result<&mut [u32], Self::Error> {
        if words != self.geometry.words || words != self.bytes.len() {
            return Err(PreparedExprError::Capacity);
        }
        self.bytes.fill(0);
        Ok(self.bytes)
    }
    fn finish_bytes(&mut self) {
        self.bytes.fill(0);
    }
    fn lookahead(&mut self, count: usize) -> Result<&mut Vec<Lookahead>, Self::Error> {
        if count > self.geometry.flat || count > self.lookahead.capacity() {
            return Err(PreparedExprError::Capacity);
        }
        self.lookahead.clear();
        Ok(self.lookahead)
    }
    fn finish_lookahead(&mut self) {
        self.lookahead.clear();
    }
    fn overflow(&mut self) -> Self::Error {
        PreparedExprError::Capacity
    }
}
impl Scope<'_> {
    pub(crate) fn and(&mut self, args: &[ExprRef]) -> Result<ExprRef, PreparedExprError> {
        self.run(args, ExprTag::And)
    }
    pub(crate) fn or(&mut self, args: &[ExprRef]) -> Result<ExprRef, PreparedExprError> {
        self.run(args, ExprTag::Or)
    }
    pub(crate) fn normalized_args(&self) -> &[ExprRef] {
        &self.buffers.args
    }
    fn run(&mut self, input: &[ExprRef], tag: ExprTag) -> Result<ExprRef, PreparedExprError> {
        if self.failed {
            return Err(PreparedExprError::Failed);
        }
        if input.len() > self.geometry.input || input.len() > self.buffers.args.capacity() {
            self.failed = true;
            return Err(PreparedExprError::Capacity);
        }
        if input.iter().any(|&arg| !self.source.is_valid(arg)) {
            self.failed = true;
            return Err(PreparedExprError::Source);
        }
        let Buffers {
            args,
            suffix,
            bytes,
            lookahead,
        } = &mut self.buffers;
        args.clear();
        args.extend_from_slice(input);
        let mut memory = Fixed {
            geometry: self.geometry,
            suffix,
            bytes,
            lookahead,
        };
        let mut sink = scalar::Prepared(&mut *self.source);
        let result = match tag {
            ExprTag::And => and(&mut sink, &mut memory, args),
            ExprTag::Or => or(&mut sink, &mut memory, args),
            _ => unreachable!("closed n-ary scope operation"),
        };
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
        AlphabetInfo,
        ast::{ExprEncodingError, ExprFlags},
        raw::HashConsCapacityError,
    };

    fn capacities(scope: &Scope<'_>) -> (usize, usize, usize, usize) {
        (
            scope.buffers.args.capacity(),
            scope.buffers.suffix.capacity(),
            scope.buffers.bytes.capacity(),
            scope.buffers.lookahead.capacity(),
        )
    }
    #[test]
    fn actual_lexer_source_nary_scope_preserves_normalization_cost_and_failed_prefix() {
        let mut source = ExprSet::new(256);
        let a = source.mk_byte(b'a');
        let b = source.mk_byte(b'b');
        let c = source.mk_byte(b'c');
        let high = source.mk_lookahead(a, 3);
        let low = source.mk_lookahead(a, 1);
        let other = source.mk_lookahead(b, 2);
        let nested_or = source.mk(Expr::Or(ExprFlags::POSITIVE, &[a, high, low]));
        let nested_and = source.mk(Expr::And(
            ExprFlags::from_nullable_positive(false, false),
            &[a, c],
        ));
        let outer = source.mk(Expr::Or(
            ExprFlags::POSITIVE,
            &[nested_or, other, b, high, ExprRef::NO_MATCH],
        ));
        let (_, mut source, roots) = AlphabetInfo::from_exprset(
            source,
            &[a, b, c, high, low, other, nested_or, nested_and, outer],
        );
        assert!(!source.optimize);
        source.reserve(64);
        let mut ordinary = source.clone();
        let mut prepared = source.prepared_source_plan().unwrap().compile().unwrap();
        drop(source);
        let plan = prepared.nary_scope_plan().unwrap();
        let requirements = plan.requirements();
        assert_eq!(
            requirements.total,
            requirements.buffers + requirements.controls
        );
        let mut scope = plan.compile().unwrap();
        let capacity = capacities(&scope);

        let input = [roots[6], roots[5], roots[1], roots[3], ExprRef::NO_MATCH];
        let mut expected_args = input.to_vec();
        let expected = ordinary.mk_or(&mut expected_args);
        let result = scope.or(&input).unwrap();
        assert_eq!(result, expected);
        assert_eq!(scope.normalized_args(), expected_args.as_slice());
        assert!(scope.normalized_args().contains(&roots[4]));
        assert!(!scope.normalized_args().contains(&roots[3]));
        assert_eq!(scope.source.cost(), ordinary.cost());
        assert_eq!(
            scope.source.expr_to_string(result),
            ordinary.expr_to_string(expected)
        );
        assert_eq!(capacities(&scope), capacity);

        let input = [roots[7], roots[0], ExprRef::ANY_BYTE_STRING];
        let mut expected_args = input.to_vec();
        let expected = ordinary.mk_and(&mut expected_args);
        assert_eq!(scope.and(&input).unwrap(), expected);
        assert_eq!(scope.normalized_args(), expected_args.as_slice());
        assert_eq!(scope.source.cost(), ordinary.cost());
        assert_eq!(capacities(&scope), capacity);

        let mut exhausted = false;
        for offset in 100..1000 {
            match scope.source.try_mk_lookahead(roots[0], offset) {
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
        let failure = scope.or(&[roots[0], roots[2]]).unwrap_err();
        assert!(matches!(
            failure,
            PreparedExprError::Encoding(ExprEncodingError::Storage(
                HashConsCapacityError::EntriesExceeded { .. }
                    | HashConsCapacityError::WordsExceeded { .. }
            ))
        ));
        assert_eq!(scope.source.len(), entries);
        assert!(scope.normalized_args().is_empty());
        assert_eq!(
            scope
                .buffers
                .bytes
                .iter()
                .map(|word| word.count_ones())
                .sum::<u32>(),
            2
        );
        assert_eq!(capacities(&scope), capacity);
        let cost = scope.source.cost();
        assert!(matches!(
            scope.and(&[roots[0]]),
            Err(PreparedExprError::Failed)
        ));
        assert_eq!(scope.source.cost(), cost);
        drop(scope);
        assert!(prepared.source().is_valid(result));
    }
}

impl Blueprint {
    pub(crate) fn argument_capacity(&self) -> usize {
        self.geometry.flat
    }
}
impl Parts {
    pub(crate) fn normalized_args(&self) -> &[ExprRef] {
        &self.buffers.args
    }
    pub(crate) fn and(
        &mut self,
        source: &mut ExprSet,
        input: &[ExprRef],
    ) -> Result<ExprRef, PreparedExprError> {
        self.run(source, input, ExprTag::And)
    }
    pub(crate) fn or(
        &mut self,
        source: &mut ExprSet,
        input: &[ExprRef],
    ) -> Result<ExprRef, PreparedExprError> {
        self.run(source, input, ExprTag::Or)
    }
    fn run(
        &mut self,
        source: &mut ExprSet,
        input: &[ExprRef],
        tag: ExprTag,
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
        let result = scope.run(input, tag);
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
    pub(crate) fn apply_to_vec(
        &mut self,
        source: &mut ExprSet,
        args: &mut Vec<ExprRef>,
        intersection: bool,
    ) -> Result<ExprRef, PreparedExprError> {
        let result = if intersection {
            self.and(source, args)
        } else {
            self.or(source, args)
        };
        let normalized = self.normalized_args();
        if normalized.len() > args.capacity() {
            return match result {
                Err(error) => Err(error),
                Ok(_) => Err(PreparedExprError::Capacity),
            };
        }
        args.clear();
        args.extend_from_slice(normalized);
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
            .checked_add(copy::fixed_required_bytes::<_, E>(&self.buffers.args)?)?
            .checked_add(copy::fixed_required_bytes::<_, E>(&self.buffers.suffix)?)?
            .checked_add(copy::fixed_required_bytes::<_, E>(&self.buffers.bytes)?)?
            .checked_add(copy::fixed_required_bytes::<_, E>(&self.buffers.lookahead)?)
    }
}
impl Parts {

    pub(crate) fn restore_copy<F: Fn(usize) -> Result<(), E>, E>(
        &mut self, source: &Self, funding: &F,
    ) -> Result<(), crate::copy_storage::Error<E>> {
        use crate::copy_storage as copy;
        copy::frame::<(&mut Self, &Self), _, _>(funding)?;
        copy::fixed(&mut self.buffers.args, &source.buffers.args, funding)?;
        copy::fixed(&mut self.buffers.suffix, &source.buffers.suffix, funding)?;
        copy::fixed(&mut self.buffers.bytes, &source.buffers.bytes, funding)?;
        copy::fixed(&mut self.buffers.lookahead, &source.buffers.lookahead, funding)?;
        self.geometry = source.geometry;
        self.failed = source.failed;
        Ok(())
    }
}
