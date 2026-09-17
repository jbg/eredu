//! Paid disjoint rows and n-ary scratch under one prepared source loan.
use super::{Construction, SymRes};
use crate::{
    ast::{Expr, ExprRef, ExprSet, PreparedExprError, PreparedExprSet},
    simplify::{byteset, nary::storage as nary},
};
use std::{
    alloc::Layout,
    collections::TryReserveError,
    fmt,
    mem::{size_of, size_of_val},
};
#[derive(Clone, Copy, Debug)]
struct Geometry {
    rows: usize,
    arguments: usize,
}
#[derive(Clone, Copy, Debug)]
pub(crate) struct Requirements {
    pub(crate) buffers: usize,
    pub(crate) controls: usize,
    pub(crate) total: usize,
}
#[derive(Default)]
struct Buffers {
    output: SymRes,
    arguments: Vec<ExprRef>,
}
#[derive(Debug)]
enum Cause {
    Source(PreparedExprError),
    Geometry,
    Capacity,
    Allocation(TryReserveError),
    Nary(nary::Failure),
}
pub(crate) struct Failure {
    cause: Cause,
    buffers: Buffers,
}
impl fmt::Debug for Failure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DisjointSourceFailure")
            .field("cause", &self.cause)
            .field("row_capacity", &self.buffers.output.capacity())
            .field("argument_capacity", &self.buffers.arguments.capacity())
            .finish()
    }
}
impl fmt::Display for Failure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.cause {
            Cause::Source(e) => fmt::Display::fmt(e, f),
            Cause::Geometry => f.write_str("disjoint source geometry overflow"),
            Cause::Capacity => f.write_str("disjoint destination differs from its exact capacity"),
            Cause::Allocation(e) => fmt::Display::fmt(e, f),
            Cause::Nary(e) => fmt::Display::fmt(e, f),
        }
    }
}
impl std::error::Error for Failure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.cause {
            Cause::Source(e) => Some(e),
            Cause::Allocation(e) => Some(e),
            Cause::Nary(e) => Some(e),
            _ => None,
        }
    }
}
fn fail(cause: Cause) -> Failure {
    Failure {
        cause,
        buffers: Buffers::default(),
    }
}
fn validate(source: &ExprSet, input: &SymRes) -> Result<(), PreparedExprError> {
    source.require_prepared()?;
    if source.alphabet_size == 0
        || source.alphabet_size > 256
        || source.alphabet_words != source.alphabet_size.div_ceil(32)
    {
        return Err(PreparedExprError::Source);
    }
    for &(selector, value) in input {
        if !source.is_valid(selector) || !source.is_valid(value) {
            return Err(PreparedExprError::Source);
        }
        match source.get(selector) {
            Expr::Byte(byte) if usize::from(byte) < source.alphabet_size => {}
            Expr::ByteSet(words)
                if words.len() == source.alphabet_words && words.iter().any(|&w| w != 0) => {}
            _ => return Err(PreparedExprError::Source),
        }
    }
    Ok(())
}
pub(crate) struct Plan<'a> {
    source: &'a mut ExprSet,
    geometry: Geometry,
    nary: nary::Blueprint,
    requirements: Requirements,
}
impl fmt::Debug for Plan<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DisjointSourcePlan")
            .field("requirements", &self.requirements)
            .finish()
    }
}
pub(crate) struct Scope<'a> {
    source: &'a mut ExprSet,
    geometry: Geometry,
    buffers: Buffers,
    nary: nary::Parts,
    attempted: bool,
}
impl fmt::Debug for Scope<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DisjointSourceScope")
            .field("geometry", &self.geometry)
            .field("attempted", &self.attempted)
            .finish()
    }
}
impl PreparedExprSet {
    pub(crate) fn symbolic_disjoint_plan(&mut self, input: &SymRes) -> Result<Plan<'_>, Failure> {
        self.source_mut().symbolic_disjoint_plan(input)
    }
}
impl ExprSet {
    pub(crate) fn symbolic_disjoint_plan(&mut self, input: &SymRes) -> Result<Plan<'_>, Failure> {
        let source = self;
        validate(source, input).map_err(|e| fail(Cause::Source(e)))?;
        let (_, _, encoded) = source
            .prepared_extents()
            .map_err(|e| fail(Cause::Source(e)))?;
        let arguments = encoded
            .checked_sub(1)
            .and_then(|n| n.checked_mul(2))
            .ok_or_else(|| fail(Cause::Geometry))?;
        // Ordinary ByteSet complements may retain the final word's padding bits.
        // Count that complete stored universe, rather than silently masking it.
        let rows = source
            .alphabet_words
            .checked_mul(32)
            .ok_or_else(|| fail(Cause::Geometry))?;
        let geometry = Geometry { rows, arguments };
        let nary = nary::Blueprint::prepare_inputs(source, 2).map_err(|e| fail(Cause::Nary(e)))?;
        let own = Layout::array::<(ExprRef, ExprRef)>(rows)
            .map_err(|_| fail(Cause::Geometry))?
            .size()
            .checked_add(
                Layout::array::<ExprRef>(arguments)
                    .map_err(|_| fail(Cause::Geometry))?
                    .size(),
            )
            .ok_or_else(|| fail(Cause::Geometry))?;
        let child = nary.requirements();
        let buffers = own
            .checked_add(child.buffers)
            .ok_or_else(|| fail(Cause::Geometry))?;
        let controls = Plan::control_bytes()
            .and_then(|n| n.checked_add(child.controls))
            .and_then(|n| n.checked_add(byteset::fixed_control_bytes()?))
            .ok_or_else(|| fail(Cause::Geometry))?;
        let total = buffers
            .checked_add(controls)
            .ok_or_else(|| fail(Cause::Geometry))?;
        Ok(Plan {
            source,
            geometry,
            nary,
            requirements: Requirements {
                buffers,
                controls,
                total,
            },
        })
    }
}
impl<'a> Plan<'a> {
    pub(crate) fn control_bytes() -> Option<usize> {
        let frames = [
            size_of::<Self>(),
            size_of::<Scope<'a>>(),
            size_of::<Geometry>(),
            size_of::<Requirements>(),
            size_of::<Buffers>(),
            size_of::<Failure>(),
            size_of::<Cause>(),
            size_of::<Bound<'a>>(),
            size_of::<Result<Self, Failure>>(),
            size_of::<Result<Scope<'a>, Failure>>(),
            size_of::<Result<(), Cause>>(),
            size_of::<Result<(), TryReserveError>>(),
            size_of::<Result<Layout, std::alloc::LayoutError>>(),
            size_of::<Result<SymRes, PreparedExprError>>(),
            size_of::<Expr<'a>>(),
            size_of::<std::slice::Iter<'a, (ExprRef, ExprRef)>>(),
        ];
        frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)
    }
    pub(crate) fn requirements(&self) -> Requirements {
        self.requirements
    }
    pub(crate) fn compile(self) -> Result<Scope<'a>, Failure> {
        let mut buffers = Buffers::default();
        fn reserve<T>(values: &mut Vec<T>, count: usize) -> Result<(), Cause> {
            values.try_reserve_exact(count).map_err(Cause::Allocation)?;
            if values.capacity() != count {
                return Err(Cause::Capacity);
            }
            Ok(())
        }
        let result = (|| {
            reserve(&mut buffers.output, self.geometry.rows)?;
            reserve(&mut buffers.arguments, self.geometry.arguments)?;
            self.nary.compile().map_err(Cause::Nary)
        })();
        match result {
            Ok(nary) => Ok(Scope {
                source: self.source,
                geometry: self.geometry,
                buffers,
                nary,
                attempted: false,
            }),
            Err(cause) => Err(Failure { cause, buffers }),
        }
    }
}
struct Bound<'a> {
    source: &'a mut ExprSet,
    nary: &'a mut nary::Parts,
    arguments: &'a mut Vec<ExprRef>,
    rows: usize,
}
impl Construction for Bound<'_> {
    type Error = PreparedExprError;
    fn union(&mut self, left: ExprRef, right: ExprRef) -> Result<ExprRef, Self::Error> {
        if self.arguments.capacity() < 2 {
            return Err(PreparedExprError::Capacity);
        }
        self.arguments.clear();
        self.arguments.push(left);
        self.arguments.push(right);
        self.nary.apply_to_vec(self.source, self.arguments, false)
    }
    fn intersection(&mut self, left: ExprRef, right: ExprRef) -> Result<ExprRef, Self::Error> {
        self.source.try_mk_byte_set_and(left, right)
    }
    fn subtract(&mut self, left: ExprRef, right: ExprRef) -> Result<ExprRef, Self::Error> {
        self.source.try_mk_byte_set_sub(left, right)
    }
    fn push(&mut self, output: &mut SymRes, pair: (ExprRef, ExprRef)) -> Result<(), Self::Error> {
        if output.len() >= self.rows || output.len() == output.capacity() {
            return Err(PreparedExprError::Capacity);
        }
        output.push(pair);
        Ok(())
    }
    fn invalid(&self) -> Self::Error {
        PreparedExprError::Source
    }
}
impl Scope<'_> {
    pub(crate) fn run(&mut self, input: &SymRes) -> Result<SymRes, PreparedExprError> {
        if self.attempted {
            return Err(PreparedExprError::Failed);
        }
        self.attempted = true;
        validate(self.source, input)?;
        super::run(
            input,
            &mut self.buffers.output,
            &mut Bound {
                source: &mut *self.source,
                nary: &mut self.nary,
                arguments: &mut self.buffers.arguments,
                rows: self.geometry.rows,
            },
        )?;
        Ok(std::mem::take(&mut self.buffers.output))
    }
    pub(crate) fn source(&self) -> &ExprSet {
        self.source
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AlphabetInfo, ast::ExprEncodingError, raw::HashConsCapacityError};
    #[test]
    fn prepared_disjoint_rows_preserve_overlap_order_and_retained_failure_prefix() {
        let mut source = ExprSet::new(256);
        let a = source.mk_byte(b'a');
        let b = source.mk_byte(b'b');
        let c = source.mk_byte(b'c');
        let d = source.mk_byte(b'd');
        let ab = source.mk_byte_set_or(&[a, b]);
        let bc = source.mk_byte_set_or(&[b, c]);
        let cd = source.mk_byte_set_or(&[c, d]);
        let v1 = source.mk_byte_literal(b"one");
        let v2 = source.mk_byte_literal(b"two");
        let v3 = source.mk_byte_literal(b"three");
        let literal = source.mk_byte_literal(&[b'q'; 256]);
        let (_, mut source, _) =
            AlphabetInfo::from_exprset(source, &[ab, bc, cd, v1, v2, v3, literal]);
        source.reserve(48);
        let input = vec![(ab, v1), (bc, v2), (cd, v3)];
        let mut ordinary = source.clone();
        let expected = super::super::ordinary(&mut ordinary, &input);
        assert_eq!(expected.len(), 4);
        let mut prepared = source.prepared_source_plan().unwrap().compile().unwrap();
        drop(source);
        let plan = prepared.symbolic_disjoint_plan(&input).unwrap();
        let requirements = plan.requirements();
        assert_eq!(
            requirements.total,
            requirements.buffers + requirements.controls
        );
        let mut scope = plan.compile().unwrap();
        let result = scope.run(&input).unwrap();
        assert_eq!(result, expected);
        assert_eq!(scope.source().cost(), ordinary.cost());
        for (row, byte) in result.iter().zip(b"bacd") {
            assert!(scope.source().get(row.0).matches_byte(*byte));
        }
        assert!(matches!(scope.run(&input), Err(PreparedExprError::Failed)));
        drop(scope);
        let failed_input = vec![(ab, v1), (bc, v3)];
        let mut exhausted = false;
        for offset in 100..2000 {
            match prepared.source_mut().try_mk_lookahead(v1, offset) {
                Ok(_) => {}
                Err(PreparedExprError::Encoding(ExprEncodingError::Storage(
                    HashConsCapacityError::EntriesExceeded { .. }
                    | HashConsCapacityError::WordsExceeded { .. },
                ))) => {
                    exhausted = true;
                    break;
                }
                Err(error) => panic!("unexpected arena failure: {error}"),
            }
        }
        assert!(exhausted);
        let cost = prepared.source().cost();
        let entries = prepared.source().len();
        let mut scope = prepared
            .symbolic_disjoint_plan(&failed_input)
            .unwrap()
            .compile()
            .unwrap();
        let error = scope.run(&failed_input).unwrap_err();
        assert!(matches!(
            error,
            PreparedExprError::Encoding(ExprEncodingError::Storage(
                HashConsCapacityError::EntriesExceeded { .. }
                    | HashConsCapacityError::WordsExceeded { .. }
            ))
        ));
        assert_eq!(scope.buffers.output, vec![(ab, v1)]);
        assert!(!scope.buffers.arguments.is_empty());
        assert!(scope.source().cost() > cost);
        assert_eq!(scope.source().len(), entries);
        assert!(matches!(scope.run(&input), Err(PreparedExprError::Failed)));
        drop(scope);
        assert!(prepared.source().is_valid(result[0].0));
        assert!(prepared.source().is_valid(result[0].1));
    }
}
