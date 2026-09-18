//! One finite byte-equivalence worker for ordinary and original construction.
use crate::{
    AlphabetInfo, ExprRef,
    ast::{Expr, ExprSet, ExprSetCopyFailure, ExprSetCopyPlan, byteset_contains},
    pp::PrettyPrinter,
};
use std::{
    alloc::Layout,
    collections::TryReserveError,
    fmt,
    mem::{size_of, size_of_val},
};

#[derive(Clone, Copy, Debug)]
struct Geometry {
    todo: usize,
    visited: usize,
    sets: usize,
    signature_words: usize,
    signatures: usize,
    output_roots: usize,
    printer: usize,
    cost_after: u64,
}
/// Exact source-derived copy, traversal, signature and output destinations.
#[derive(Clone, Copy, Debug)]
pub struct AlphabetCopyRequirements {
    buffers: usize,
    controls: usize,
    total: usize,
}
impl AlphabetCopyRequirements {
    /// All source copies, fixed traversal scratch and independent outputs.
    pub fn buffer_bytes(&self) -> usize {
        self.buffers
    }
    /// Complete local construction/inspection frames and child frames.
    pub fn control_bytes(&self) -> usize {
        self.controls
    }
    /// Complete local requirement before constructor execution.
    pub fn required_bytes(&self) -> usize {
        self.total
    }
}
#[derive(Debug)]
enum Cause {
    Overflow,
    Source,
    Capacity,
    Vector(TryReserveError),
    Expressions(ExprSetCopyFailure),
}
#[derive(Default)]
struct Partial {
    todo: Vec<ExprRef>,
    visited: Vec<bool>,
    sets: Vec<ExprRef>,
    signatures: Vec<u64>,
    printer: Vec<u8>,
    roots: Vec<ExprRef>,
}
/// Retains every allocated prefix, including the copied expression source.
pub struct AlphabetCopyFailure {
    cause: Cause,
    partial: Option<Partial>,
    expressions: Option<ExprSet>,
}
impl fmt::Debug for AlphabetCopyFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AlphabetCopyFailure")
            .field("cause", &self.cause)
            .field("retains_destinations", &self.partial.is_some())
            .field("retains_expressions", &self.expressions.is_some())
            .finish()
    }
}
impl fmt::Display for AlphabetCopyFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.cause {
            Cause::Overflow => f.write_str("alphabet source geometry overflow"),
            Cause::Source => f.write_str(
                "alphabet source contains an invalid byte alphabet or expression reference",
            ),
            Cause::Capacity => {
                f.write_str("alphabet destination differs from its prepared capacity")
            }
            Cause::Vector(cause) => fmt::Display::fmt(cause, f),
            Cause::Expressions(cause) => fmt::Display::fmt(cause, f),
        }
    }
}
impl std::error::Error for AlphabetCopyFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.cause {
            Cause::Vector(cause) => Some(cause),
            Cause::Expressions(cause) => Some(cause),
            _ => None,
        }
    }
}
fn failure(cause: Cause) -> AlphabetCopyFailure {
    AlphabetCopyFailure {
        cause,
        partial: None,
        expressions: None,
    }
}
fn bytes<T>(n: usize) -> Result<usize, Cause> {
    Ok(Layout::array::<T>(n).map_err(|_| Cause::Overflow)?.size())
}
fn reserve<T>(dst: &mut Vec<T>, n: usize) -> Result<(), Cause> {
    dst.try_reserve_exact(n).map_err(Cause::Vector)?;
    if dst.capacity() != n {
        return Err(Cause::Capacity);
    }
    Ok(())
}
fn push<T>(dst: &mut Vec<T>, value: T) -> Result<(), Cause> {
    if dst.len() == dst.capacity() {
        return Err(Cause::Capacity);
    }
    dst.push(value);
    Ok(())
}

// The maximum alphabet has 256 bytes. Fixed representatives avoid a separately
// allocated hash-table population; signatures and first-seen numbering retain
// the same equivalence predicate and ascending unassigned-byte order.
pub(crate) struct MappingPlan<'a> {
    source: &'a ExprSet,
    roots: &'a [ExprRef],
    geometry: Geometry,
    requirements: AlphabetCopyRequirements,
}
impl<'a> MappingPlan<'a> {
    pub(crate) fn requirements(&self) -> AlphabetCopyRequirements { self.requirements }

    pub(crate) fn prepare(
        source: &'a ExprSet,
        roots: &'a [ExprRef],
    ) -> Result<Self, AlphabetCopyFailure> {
        let result = (|| -> Result<_, Cause> {
            if source.alphabet_size != 256 {
                return Err(Cause::Source);
            }
            let valid = |e: ExprRef| e.is_valid() && e.as_usize() < source.len();
            if roots.iter().any(|e| !valid(*e)) {
                return Err(Cause::Source);
            }
            let mut geometry = Geometry {
                todo: 0,
                visited: 0,
                sets: 0,
                signature_words: 0,
                signatures: 0,
                output_roots: roots.len(),
                printer: 0,
                cost_after: source.cost,
            };
            if cfg!(feature = "compress") {
                geometry.todo = roots.len();
                geometry.visited = source.len();
                geometry.printer = 256;
                geometry.cost_after = source
                    .cost
                    .checked_add(source.cost)
                    .ok_or(Cause::Overflow)?;
                for id in 1..source.len() {
                    let e = ExprRef::new(u32::try_from(id).map_err(|_| Cause::Overflow)?);
                    let args = source.get_args(e);
                    if args.iter().any(|e| !valid(*e)) {
                        return Err(Cause::Source);
                    }
                    geometry.todo = geometry
                        .todo
                        .checked_add(args.len())
                        .ok_or(Cause::Overflow)?;
                    if matches!(source.get(e), Expr::ByteSet(_)) {
                        geometry.sets = geometry.sets.checked_add(1).ok_or(Cause::Overflow)?;
                    }
                }
                geometry.signature_words = geometry.sets.div_ceil(64);
                if geometry.signature_words > 1 {
                    geometry.signatures = 256usize
                        .checked_mul(geometry.signature_words)
                        .ok_or(Cause::Overflow)?;
                }
            }
            let buffers = [
                bytes::<ExprRef>(geometry.todo)?,
                bytes::<bool>(geometry.visited)?,
                bytes::<ExprRef>(geometry.sets)?,
                bytes::<u64>(geometry.signatures)?,
                bytes::<u8>(geometry.printer)?,
                bytes::<ExprRef>(geometry.output_roots)?,
            ]
            .into_iter()
            .try_fold(0usize, usize::checked_add)
            .ok_or(Cause::Overflow)?;
            let controls = Self::inspection_control_bytes().ok_or(Cause::Overflow)?;
            let total = buffers.checked_add(controls).ok_or(Cause::Overflow)?;
            Ok(MappingPlan {
                source,
                roots,
                geometry,
                requirements: AlphabetCopyRequirements {
                    buffers,
                    controls,
                    total,
                },
            })
        })();
        result.map_err(failure)
    }
    pub(crate) fn inspection_control_bytes() -> Option<usize> {
        let parts = [
            size_of::<MappingPlan<'_>>(),
            size_of::<Geometry>(),
            size_of::<AlphabetCopyRequirements>(),
            size_of::<Partial>(),
            size_of::<Cause>(),
            size_of::<AlphabetCopyFailure>(),
            size_of::<Mapping>(),
            size_of::<[u8; 256]>(),
            size_of::<[u8; 256]>(),
            size_of::<[bool; 256]>(),
            size_of::<[u64; 256]>(),
            size_of::<Result<MappingPlan<'_>, AlphabetCopyFailure>>(),
            size_of::<Result<Mapping, AlphabetCopyFailure>>(),
            size_of::<Result<(), Cause>>(),
            size_of::<Result<(), TryReserveError>>(),
            size_of::<Result<Layout, std::alloc::LayoutError>>(),
            size_of::<Result<u32, std::num::TryFromIntError>>(),
            size_of::<(AlphabetInfo, ExprSet, Vec<ExprRef>)>(),
            size_of::<PrettyPrinter>(),
            size_of::<std::slice::Iter<'_, ExprRef>>(),
            size_of::<(usize, usize, usize, usize, u8, u64)>(),
            size_of::<Expr<'_>>(),
            size_of::<(&ExprSet, &[ExprRef])>(),
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
    pub(crate) fn compile(self) -> Result<Mapping, AlphabetCopyFailure> {
        let mut partial = Partial::default();
        let result = self.fill(&mut partial);
        match result {
            Ok((mapping, size)) => Ok(Mapping {
                mapping,
                size,
                cost_after: self.geometry.cost_after,
                printer: partial.printer,
                roots: partial.roots,
            }),
            Err(cause) => Err(AlphabetCopyFailure {
                cause,
                partial: Some(partial),
                expressions: None,
            }),
        }
    }
    fn fill(&self, p: &mut Partial) -> Result<([u8; 256], usize), Cause> {
        reserve(&mut p.todo, self.geometry.todo)?;
        reserve(&mut p.visited, self.geometry.visited)?;
        reserve(&mut p.sets, self.geometry.sets)?;
        reserve(&mut p.signatures, self.geometry.signatures)?;
        reserve(&mut p.printer, self.geometry.printer)?;
        reserve(&mut p.roots, self.geometry.output_roots)?;
        for &root in self.roots {
            push(&mut p.roots, root)?;
        }
        let mut mapping = [0u8; 256];
        if !cfg!(feature = "compress") {
            for (b, v) in mapping.iter_mut().enumerate() {
                *v = b as u8;
            }
            return Ok((mapping, 256));
        }
        p.visited.resize(self.geometry.visited, false);
        p.signatures.resize(self.geometry.signatures, 0);
        for &root in self.roots {
            push(&mut p.todo, root)?;
        }
        let mut assigned = [false; 256];
        let mut alphabet_size = 0usize;
        while let Some(e) = p.todo.pop() {
            if p.visited[e.as_usize()] {
                continue;
            }
            p.visited[e.as_usize()] = true;
            for &arg in self.source.get_args(e) {
                push(&mut p.todo, arg)?;
            }
            match self.source.get(e) {
                Expr::Byte(b) => assign(b, &mut assigned, &mut mapping, &mut alphabet_size)?,
                Expr::ByteConcat(_, bs, _) => {
                    for &b in bs {
                        assign(b, &mut assigned, &mut mapping, &mut alphabet_size)?;
                    }
                }
                Expr::ByteSet(_) => push(&mut p.sets, e)?,
                Expr::RemainderIs { scale, .. } => {
                    for b in self.source.digits {
                        assign(b, &mut assigned, &mut mapping, &mut alphabet_size)?;
                    }
                    if scale > 0 {
                        assign(
                            self.source.digit_dot,
                            &mut assigned,
                            &mut mapping,
                            &mut alphabet_size,
                        )?;
                    }
                }
                _ => {}
            }
        }
        let mut single_signatures = [0u64; 256];
        for b in 0..256 {
            if assigned[b] {
                continue;
            }
            for (idx, &e) in p.sets.iter().enumerate() {
                let Expr::ByteSet(bs) = self.source.get(e) else {
                    return Err(Cause::Source);
                };
                if byteset_contains(bs, b) {
                    if self.geometry.signature_words <= 1 {
                        single_signatures[b] |= 1u64 << idx;
                    } else {
                        p.signatures[b * self.geometry.signature_words + idx / 64] |=
                            1u64 << (idx % 64);
                    }
                }
            }
        }
        let mut representatives = [0u8; 256];
        let mut count = 0;
        for b in 0..256 {
            if assigned[b] {
                continue;
            }
            let same = representatives[..count].iter().copied().find(|previous| {
                let previous = usize::from(*previous);
                if self.geometry.signature_words <= 1 {
                    single_signatures[previous] == single_signatures[b]
                } else {
                    let width = self.geometry.signature_words;
                    p.signatures[previous * width..(previous + 1) * width]
                        == p.signatures[b * width..(b + 1) * width]
                }
            });
            if let Some(previous) = same {
                mapping[b] = mapping[usize::from(previous)];
                assigned[b] = true;
            } else {
                assign(b as u8, &mut assigned, &mut mapping, &mut alphabet_size)?;
                representatives[count] = b as u8;
                count += 1;
            }
        }
        for b in mapping {
            push(&mut p.printer, b)?;
        }
        Ok((mapping, alphabet_size))
    }
}
fn assign(
    b: u8,
    assigned: &mut [bool; 256],
    mapping: &mut [u8; 256],
    count: &mut usize,
) -> Result<(), Cause> {
    let b = usize::from(b);
    if !assigned[b] {
        mapping[b] = u8::try_from(*count).map_err(|_| Cause::Capacity)?;
        *count += 1;
        assigned[b] = true;
    }
    Ok(())
}
pub(crate) struct Mapping {
    mapping: [u8; 256],
    size: usize,
    cost_after: u64,
    printer: Vec<u8>,
    roots: Vec<ExprRef>,
}
impl Mapping {
    pub(crate) fn finish(self, mut expressions: ExprSet) -> (AlphabetInfo, ExprSet, Vec<ExprRef>) {
        if cfg!(feature = "compress") {
            expressions.cost = self.cost_after;
            expressions.set_pp(PrettyPrinter::new(self.printer, self.size));
        }
        expressions.disable_optimizations();
        (
            AlphabetInfo::from_mapping(self.mapping, self.size),
            expressions,
            self.roots,
        )
    }
}

/// Immutable exact source loan; no lexer, regex parsing or mutable DFA is run.
pub struct AlphabetCopyPlan<'a> {
    mapping: MappingPlan<'a>,
    expressions: ExprSetCopyPlan<'a>,
    requirements: AlphabetCopyRequirements,
}
impl fmt::Debug for AlphabetCopyPlan<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AlphabetCopyPlan")
            .field("requirements", &self.requirements)
            .finish()
    }
}
impl AlphabetInfo {
    /// Fixed mapping/adapter inspection frames before walking a source. The
    /// caller also prices its exact expression-copy source inspection frames.
    /// This is a layout query only and creates no source or allocation authority.
    pub fn source_inspection_control_bytes() -> Option<usize> {
        let parts = [
            MappingPlan::inspection_control_bytes()?,
            size_of::<AlphabetCopyPlan<'_>>(),
            size_of::<Result<AlphabetCopyPlan<'_>, AlphabetCopyFailure>>(),
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
    /// Source-derived finite construction using the same ordinary grouping worker.
    /// Reserve returned requirements before compiling; this quote grants no authority.
    pub fn source_copy_plan<'a>(
        expressions: &'a ExprSet,
        roots: &'a [ExprRef],
    ) -> Result<AlphabetCopyPlan<'a>, AlphabetCopyFailure> {
        let mapping = MappingPlan::prepare(expressions, roots)?;
        let expressions = expressions
            .source_copy_plan()
            .map_err(|e| failure(Cause::Expressions(e)))?;
        let q = expressions.requirements();
        let buffers = q
            .buffer_bytes()
            .checked_add(mapping.requirements.buffers)
            .ok_or_else(|| failure(Cause::Overflow))?;
        let local = [
            size_of::<AlphabetCopyPlan<'_>>(),
            size_of::<Result<AlphabetCopyPlan<'_>, AlphabetCopyFailure>>(),
            size_of::<Result<ExprSet, ExprSetCopyFailure>>(),
            size_of::<Result<(AlphabetInfo, ExprSet, Vec<ExprRef>), AlphabetCopyFailure>>(),
        ];
        let controls = local
            .into_iter()
            .try_fold(
                q.control_bytes()
                    .checked_add(mapping.requirements.controls)
                    .and_then(|v| v.checked_add(size_of_val(&local)))
                    .ok_or_else(|| failure(Cause::Overflow))?,
                usize::checked_add,
            )
            .ok_or_else(|| failure(Cause::Overflow))?;
        let total = buffers
            .checked_add(controls)
            .ok_or_else(|| failure(Cause::Overflow))?;
        Ok(AlphabetCopyPlan {
            mapping,
            expressions,
            requirements: AlphabetCopyRequirements {
                buffers,
                controls,
                total,
            },
        })
    }
}
impl AlphabetCopyPlan<'_> {
    /// Complete source-copy, mapping scratch, roots and printer destinations.
    pub fn requirements(&self) -> AlphabetCopyRequirements {
        self.requirements
    }
    /// Builds independent input for the shared lexer constructor. Mutable DFA,
    /// derivative caches and future expression growth need separate admission.
    pub fn compile(self) -> Result<(AlphabetInfo, ExprSet, Vec<ExprRef>), AlphabetCopyFailure> {
        let expressions = self
            .expressions
            .compile()
            .map_err(|e| failure(Cause::Expressions(e)))?;
        match self.mapping.compile() {
            Ok(mapping) => Ok(mapping.finish(expressions)),
            Err(mut error) => {
                error.expressions = Some(expressions);
                Err(error)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn finite_alphabet_copy_preserves_group_order_all_256_classes_and_failed_destinations() {
        let mut expressions = ExprSet::new(256, crate::ParserAllocationFunding::unenforced()).unwrap();
        let mut roots = (0..=255)
            .map(|b| expressions.mk(Expr::Byte(b)).unwrap())
            .collect::<Vec<_>>();
        // This repeated final root is visited first, fixing the ordinary DFS order.
        roots.push(roots[0]);
        let before_cost = expressions.cost();
        let plan = AlphabetInfo::source_copy_plan(&expressions, &roots).unwrap();
        assert!(plan.requirements().buffer_bytes() > 0);
        let (copied, copy, copied_roots) = plan.compile().unwrap();
        let (ordinary, ordinary_copy, ordinary_roots) =
            AlphabetInfo::from_exprset(expressions.clone(), &roots).unwrap();
        assert_eq!(copied_roots, roots);
        assert_eq!(ordinary_roots, roots);
        assert_eq!(copied.len(), 256);
        assert_eq!(ordinary.len(), 256);
        let mut seen = [false; 256];
        for b in 0..=255 {
            let class = copied.map(b);
            assert!(!seen[class]);
            seen[class] = true;
            assert_eq!(class, ordinary.map(b));
            let expected = if cfg!(feature = "compress") {
                if b == 0 { 0 } else { 256 - usize::from(b) }
            } else {
                usize::from(b)
            };
            assert_eq!(class, expected);
        }
        let expected_cost = if cfg!(feature = "compress") {
            before_cost * 2
        } else {
            before_cost
        };
        assert_eq!(copy.cost(), expected_cost);
        assert_eq!(ordinary_copy.cost(), expected_cost);
        assert!(!copy.optimize);

        let mut many = ExprSet::new(256, crate::ParserAllocationFunding::unenforced()).unwrap();
        let mut many_roots = Vec::new();
        for bit in 1..=70 {
            let mut words = [0u32; 8];
            words[0] |= 1;
            words[bit / 32] |= 1 << (bit % 32);
            many_roots.push(many.mk(Expr::ByteSet(&words)).unwrap());
        }
        let many_plan = AlphabetInfo::source_copy_plan(&many, &many_roots).unwrap();
        if cfg!(feature = "compress") {
            assert!(many_plan.mapping.geometry.signature_words > 1);
        }
        let (alpha, many_copy, actual_roots) = many_plan.compile().unwrap();
        assert_eq!(actual_roots, many_roots);
        if cfg!(feature = "compress") {
            assert_eq!(alpha.len(), 72);
            assert_eq!(alpha.map(71), alpha.map(255));
            assert_ne!(alpha.map(0), alpha.map(1));
            assert_ne!(alpha.map(1), alpha.map(2));
        }
        let mut failed = AlphabetInfo::source_copy_plan(&many, &many_roots).unwrap();
        failed.mapping.geometry.output_roots = 1;
        let failed = match failed.compile() {
            Err(error) => error,
            Ok(_) => panic!("short destination must refuse"),
        };
        assert!(matches!(failed.cause, Cause::Capacity));
        assert_eq!(
            failed.partial.as_ref().unwrap().roots.as_slice(),
            &many_roots[..1]
        );
        assert_eq!(failed.expressions.as_ref().unwrap().len(), many.len());
        let invalid = AlphabetInfo::source_copy_plan(&many, &[ExprRef::new(u32::MAX)]).unwrap_err();
        assert!(matches!(invalid.cause, Cause::Source));
        drop((many, many_roots, copy, ordinary_copy, many_copy));
        assert!(!failed.expressions.as_ref().unwrap().is_empty());
        assert_eq!(failed.partial.as_ref().unwrap().roots.len(), 1);
    }
}
