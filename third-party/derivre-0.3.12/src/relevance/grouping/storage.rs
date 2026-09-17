//! Finite two-pass grouping storage under one actual prepared expression loan.
use super::{Combine, Memory, SymRes};
use crate::{
    ast::{Expr, ExprRef, ExprSet, PreparedExprError, PreparedExprSet},
    simplify::nary::storage as nary,
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
    nodes: usize,
    arguments: usize,
}
#[derive(Clone, Copy, Debug)]
pub(crate) struct Requirements {
    pub(crate) buffers: usize,
    pub(crate) controls: usize,
    pub(crate) total: usize,
}
struct Group {
    key: ExprRef,
    values: Vec<ExprRef>,
}
#[derive(Default)]
struct Buffers {
    index: Vec<Option<usize>>,
    groups: Vec<Group>,
    output: Option<SymRes>,
    used: usize,
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
    pending: Vec<ExprRef>,
}
impl fmt::Debug for Failure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SymbolicGroupingFailure")
            .field("cause", &self.cause)
            .field("groups", &self.buffers.groups.len())
            .field("pending_capacity", &self.pending.capacity())
            .finish()
    }
}
impl fmt::Display for Failure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.cause {
            Cause::Source(e) => fmt::Display::fmt(e, f),
            Cause::Geometry => f.write_str("symbolic grouping geometry overflow"),
            Cause::Capacity => {
                f.write_str("symbolic grouping destination differs from its exact capacity")
            }
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
        pending: Vec::new(),
    }
}
fn validate(source: &ExprSet, input: &SymRes) -> Result<(), PreparedExprError> {
    source.require_prepared()?;
    for &(selector, value) in input {
        if !source.is_valid(selector) || !source.is_valid(value) {
            return Err(PreparedExprError::Source);
        }
        if !matches!(source.get(selector), Expr::Byte(_) | Expr::ByteSet(_)) {
            return Err(PreparedExprError::Source);
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
        f.debug_struct("SymbolicGroupingPlan")
            .field("requirements", &self.requirements)
            .finish()
    }
}
pub(crate) struct Scope<'a> {
    source: &'a mut ExprSet,
    geometry: Geometry,
    buffers: Buffers,
    nary: nary::Parts,
    failed: bool,
}
impl fmt::Debug for Scope<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SymbolicGroupingScope")
            .field("geometry", &self.geometry)
            .field("failed", &self.failed)
            .finish()
    }
}
impl PreparedExprSet {
    pub(crate) fn symbolic_grouping_plan(&mut self, input: &SymRes) -> Result<Plan<'_>, Failure> {
        self.source_mut().symbolic_grouping_plan(input)
    }
}
impl ExprSet {
    pub(crate) fn symbolic_grouping_plan(&mut self, input: &SymRes) -> Result<Plan<'_>, Failure> {
        let source = self;
        validate(source, input).map_err(|e| fail(Cause::Source(e)))?;
        let (_, nodes, encoded) = source
            .prepared_extents()
            .map_err(|e| fail(Cause::Source(e)))?;
        let per_arg = encoded
            .checked_sub(1)
            .ok_or_else(|| fail(Cause::Geometry))?;
        let geometry = Geometry {
            rows: input.len(),
            nodes,
            arguments: input
                .len()
                .checked_mul(per_arg)
                .ok_or_else(|| fail(Cause::Geometry))?,
        };
        let nary = nary::Blueprint::prepare_inputs(source, input.len())
            .map_err(|e| fail(Cause::Nary(e)))?;
        let layouts = [
            Layout::array::<Option<usize>>(nodes),
            Layout::array::<Group>(geometry.rows),
            Layout::array::<ExprRef>(geometry.arguments),
            Layout::array::<(ExprRef, ExprRef)>(geometry.rows),
        ];
        let layout_controls = size_of_val(&layouts);
        let [index, groups, args, out] =
            layouts.map(|layout| layout.map(|l| l.size()).map_err(|_| fail(Cause::Geometry)));
        let index = index?;
        let groups = groups?;
        let args = args?;
        let out = out?;
        let own = index
            .checked_add(groups)
            .and_then(|n| n.checked_add(out))
            .and_then(|n| n.checked_add(args.checked_mul(geometry.rows)?))
            .ok_or_else(|| fail(Cause::Geometry))?;
        let child = nary.requirements();
        let buffers = own
            .checked_add(child.buffers)
            .ok_or_else(|| fail(Cause::Geometry))?;
        let controls = Plan::control_bytes()
            .and_then(|n| n.checked_add(layout_controls))
            .and_then(|n| n.checked_add(child.controls))
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
            size_of::<Group>(),
            size_of::<Failure>(),
            size_of::<Cause>(),
            size_of::<Fixed<'a>>(),
            size_of::<Source<'a>>(),
            size_of::<Result<Self, Failure>>(),
            size_of::<Result<Scope<'a>, Failure>>(),
            size_of::<Result<(), Cause>>(),
            size_of::<Result<(), TryReserveError>>(),
            size_of::<Result<Layout, std::alloc::LayoutError>>(),
            size_of::<Result<SymRes, PreparedExprError>>(),
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
        let mut pending = Vec::new();
        fn reserve<T>(values: &mut Vec<T>, count: usize) -> Result<(), Cause> {
            values.try_reserve_exact(count).map_err(Cause::Allocation)?;
            if values.capacity() != count {
                return Err(Cause::Capacity);
            }
            Ok(())
        }
        let result = (|| {
            reserve(&mut buffers.index, self.geometry.nodes)?;
            buffers.index.resize(self.geometry.nodes, None);
            reserve(&mut buffers.groups, self.geometry.rows)?;
            for _ in 0..self.geometry.rows {
                reserve(&mut pending, self.geometry.arguments)?;
                buffers.groups.push(Group {
                    key: ExprRef::NO_MATCH,
                    values: std::mem::take(&mut pending),
                });
            }
            buffers.output = Some(Vec::new());
            reserve(
                buffers
                    .output
                    .as_mut()
                    .expect("installed output destination"),
                self.geometry.rows,
            )?;
            self.nary.compile().map_err(Cause::Nary)
        })();
        match result {
            Ok(nary) => Ok(Scope {
                source: self.source,
                geometry: self.geometry,
                buffers,
                nary,
                failed: false,
            }),
            Err(cause) => Err(Failure {
                cause,
                buffers,
                pending,
            }),
        }
    }
}
struct Fixed<'a> {
    buffers: &'a mut Buffers,
    geometry: Geometry,
}
impl Memory for Fixed<'_> {
    type Error = PreparedExprError;
    fn begin(&mut self) -> Result<(), Self::Error> {
        self.buffers.index.fill(None);
        for group in &mut self.buffers.groups {
            group.values.clear();
        }
        self.buffers.used = 0;
        Ok(())
    }
    fn get(&self, key: ExprRef) -> Option<usize> {
        self.buffers.index.get(key.as_usize()).copied().flatten()
    }
    fn index(&mut self, key: ExprRef, index: usize) -> Result<(), Self::Error> {
        let slot = self
            .buffers
            .index
            .get_mut(key.as_usize())
            .ok_or(PreparedExprError::Source)?;
        *slot = Some(index);
        Ok(())
    }
    fn clear_index(&mut self) {
        self.buffers.index.fill(None);
    }
    fn groups(&self) -> usize {
        self.buffers.used
    }
    fn add_group(&mut self, key: ExprRef, value: ExprRef) -> Result<(), Self::Error> {
        let index = self.buffers.used;
        let group = self
            .buffers
            .groups
            .get_mut(index)
            .ok_or(PreparedExprError::Capacity)?;
        if group.values.capacity() == 0 {
            return Err(PreparedExprError::Capacity);
        }
        group.key = key;
        group.values.push(value);
        self.buffers.used += 1;
        Ok(())
    }
    fn append(&mut self, index: usize, value: ExprRef) -> Result<(), Self::Error> {
        let values = &mut self
            .buffers
            .groups
            .get_mut(index)
            .ok_or(PreparedExprError::Capacity)?
            .values;
        if values.len() >= self.geometry.arguments || values.len() == values.capacity() {
            return Err(PreparedExprError::Capacity);
        }
        values.push(value);
        Ok(())
    }
    fn group(&mut self, index: usize) -> (ExprRef, &mut Vec<ExprRef>) {
        let group = &mut self.buffers.groups[index];
        (group.key, &mut group.values)
    }
    fn release_group(&mut self, index: usize) {
        self.buffers.groups[index].values.clear();
    }
    fn output(&mut self, count: usize) -> Result<SymRes, Self::Error> {
        let mut output = self
            .buffers
            .output
            .take()
            .ok_or(PreparedExprError::Failed)?;
        if count > self.geometry.rows || count > output.capacity() {
            self.buffers.output = Some(output);
            return Err(PreparedExprError::Capacity);
        }
        output.clear();
        Ok(output)
    }
    fn push(&mut self, out: &mut SymRes, value: (ExprRef, ExprRef)) -> Result<(), Self::Error> {
        if out.len() >= self.geometry.rows || out.len() == out.capacity() {
            return Err(PreparedExprError::Capacity);
        }
        out.push(value);
        Ok(())
    }
    fn retire_input(&mut self, mut input: SymRes) {
        input.clear();
        self.buffers.output = Some(input);
    }
}
struct Source<'a> {
    source: &'a mut ExprSet,
    nary: &'a mut nary::Parts,
}
impl Combine for Source<'_> {
    type Error = PreparedExprError;
    fn pay(&mut self, amount: usize) -> Result<(), Self::Error> {
        self.source.pay_prepared(amount)
    }
    fn regex(&mut self, args: &mut Vec<ExprRef>) -> Result<ExprRef, Self::Error> {
        self.nary.apply_to_vec(self.source, args, false)
    }
    fn selectors(&mut self, args: &mut Vec<ExprRef>) -> Result<ExprRef, Self::Error> {
        self.source.try_mk_byte_set_or(args)
    }
}
impl Scope<'_> {
    pub(crate) fn run(&mut self, input: SymRes) -> Result<SymRes, PreparedExprError> {
        if self.failed {
            return Err(PreparedExprError::Failed);
        }
        self.failed = true;
        if input.len() > self.geometry.rows {
            return Err(PreparedExprError::Capacity);
        }
        validate(self.source, &input)?;
        let result = super::simplify(
            input,
            &mut Fixed {
                buffers: &mut self.buffers,
                geometry: self.geometry,
            },
            &mut Source {
                source: &mut *self.source,
                nary: &mut self.nary,
            },
        );
        if result.is_ok() {
            self.failed = false;
        }
        result
    }
    pub(crate) fn source(&self) -> &ExprSet {
        self.source
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ast::{Expr, ExprEncodingError, ExprFlags},
        raw::HashConsCapacityError,
        AlphabetInfo,
    };
    #[test]
    fn prepared_symbolic_grouping_preserves_both_passes_and_failed_source_prefix() {
        let mut source = ExprSet::new(256);
        let a = source.mk_byte(b'a');
        let b = source.mk_byte(b'b');
        let c = source.mk_byte(b'c');
        let v1 = source.mk_byte_literal(b"ab");
        let v2 = source.mk_byte_literal(b"ac");
        let v3 = source.mk_byte_literal(b"bb");
        let literal = source.mk_byte_literal(&[b'q'; 256]);
        let (_, mut source, _) =
            AlphabetInfo::from_exprset(source, &[a, b, c, v1, v2, v3, literal]);
        let combined = source.mk_or(&mut vec![v1, v2]);
        // Table reserve alone does not create initialized expression words.
        // Retain a real encoded declaration which supplies both that source
        // backing and the source-derived maximum encoding size.
        let declaration = source.mk(Expr::Or(ExprFlags::POSITIVE, &[combined; 128]));
        assert!(source.is_valid(declaration));
        source.reserve(48);
        let input = vec![(a, v1), (b, combined), (a, v2), (c, v1), (c, v2)];
        let mut ordinary = source.clone();
        let expected = super::super::ordinary_simplify(&mut ordinary, input.clone());
        assert_eq!(expected.len(), 1);
        assert_eq!(expected[0].1, combined);
        let mut prepared = source.prepared_source_plan().unwrap().compile().unwrap();
        drop(source);
        let plan = prepared.symbolic_grouping_plan(&input).unwrap();
        let requirements = plan.requirements();
        assert_eq!(
            requirements.total,
            requirements.buffers + requirements.controls
        );
        let mut scope = plan.compile().unwrap();
        let index_capacity = scope.buffers.index.capacity();
        let result = scope.run(input).unwrap();
        assert_eq!(result, expected);
        assert_eq!(scope.source().cost(), ordinary.cost());
        for byte in b"abc" {
            assert!(scope.source().get(result[0].0).matches_byte(*byte));
        }
        assert_eq!(scope.buffers.index.capacity(), index_capacity);
        let mut exhausted = false;
        for offset in 100..2000 {
            match scope.source.try_mk_lookahead(v1, offset) {
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
        let cost = scope.source().cost();
        let entries = scope.source().len();
        let error = scope.run(vec![(a, v1), (a, v3), (b, v2)]).unwrap_err();
        assert!(matches!(
            error,
            PreparedExprError::Encoding(ExprEncodingError::Storage(
                HashConsCapacityError::EntriesExceeded { .. }
                    | HashConsCapacityError::WordsExceeded { .. }
            ))
        ));
        assert!(scope.source().cost() > cost);
        assert_eq!(scope.source().len(), entries);
        assert!(!scope.buffers.groups[0].values.is_empty());
        assert_eq!(scope.buffers.index.capacity(), index_capacity);
        assert!(matches!(
            scope.run(Vec::new()),
            Err(PreparedExprError::Failed)
        ));
        drop(scope);
        assert!(prepared.source().is_valid(result[0].0));
        assert!(prepared.source().is_valid(result[0].1));
    }
}
