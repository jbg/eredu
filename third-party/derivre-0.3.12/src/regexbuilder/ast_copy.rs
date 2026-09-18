//! Source-derived AST copies using finite traversal and partial destinations.
use super::{JsonQuoteOptions, RegexAst};
use crate::{ParserAllocationFailure, prepared_funding::{FrameError, PreparedFunding}};
use std::{
    alloc::Layout,
    collections::TryReserveError,
    fmt,
    mem::{size_of, size_of_val},
    string::FromUtf8Error,
};
/// Exact source tree constructor population, including simultaneous traversal.
#[derive(Clone, Copy, Debug)]
pub struct RegexAstCopyRequirements {
    buffers: usize,
    retained: usize,
    controls: usize,
    total: usize,
    nodes: usize,
    depth: usize,
}
impl RegexAstCopyRequirements {
    /// Exact destination payload after constructor stacks retire. Only this
    /// population may enter persistent shared-source accounting.
    pub fn retained_bytes(&self) -> usize {
        self.retained
    }
    /// Tree payloads, boxed children and finite traversal/destination stacks.
    pub fn buffer_bytes(&self) -> usize {
        self.buffers
    }
    /// Fixed inspection/copy/error frames, including recursive source inspection.
    pub fn control_bytes(&self) -> usize {
        self.controls
    }
    /// Complete checked local copy requirement.
    pub fn required_bytes(&self) -> usize {
        self.total
    }
    /// Actual source nodes.
    pub fn nodes(&self) -> usize {
        self.nodes
    }
    /// Actual maximum source nesting, including the root.
    pub fn depth(&self) -> usize {
        self.depth
    }
}
/// Loan of an existing AST; it cannot certify regex compilation or parser growth.
pub struct RegexAstCopyPlan<'a> {
    source: &'a RegexAst,
    requirements: RegexAstCopyRequirements,
    byte_words: usize,
}
impl fmt::Debug for RegexAstCopyPlan<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RegexAstCopyPlan")
            .field("requirements", &self.requirements)
            .finish()
    }
}
#[derive(Debug)]
enum Cause {
    Overflow,
    Funding(ParserAllocationFailure),
    Capacity,
    Vector(TryReserveError),
    Utf8(FromUtf8Error),
}
#[derive(Default)]
struct Destination {
    children: Vec<RegexAst>,
    child: Option<RegexAst>,
    bytes: Vec<u8>,
    words: Vec<u32>,
    options: Vec<u8>,
}
struct Frame<'a> {
    source: &'a RegexAst,
    next: usize,
}
/// Retains every unfinished destination and completed sibling. Borrowed traversal
/// scratch is safely retired before returning; no source references are adopted.
pub struct RegexAstCopyFailure {
    cause: Cause,
    destinations: Vec<Destination>,
    completed: Option<RegexAst>,
}
impl fmt::Debug for RegexAstCopyFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RegexAstCopyFailure")
            .field("cause", &self.cause)
            .field("partial_depth", &self.destinations.len())
            .field("completed", &self.completed.is_some())
            .finish()
    }
}
impl fmt::Display for RegexAstCopyFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.cause {
            Cause::Overflow => f.write_str("regex AST source geometry overflow"),
            Cause::Funding(error) => fmt::Display::fmt(error, f),
            Cause::Capacity => f.write_str("regex AST destination differs from source"),
            Cause::Vector(e) => fmt::Display::fmt(e, f),
            Cause::Utf8(e) => fmt::Display::fmt(e, f),
        }
    }
}
impl std::error::Error for RegexAstCopyFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.cause {
            Cause::Funding(error) => Some(error),
            Cause::Vector(e) => Some(e),
            Cause::Utf8(e) => Some(e),
            _ => None,
        }
    }
}
#[derive(Default)]
struct Geometry {
    buffers: usize,
    byte_words: usize,
    nodes: usize,
    depth: usize,
}
fn array<T>(n: usize) -> Result<usize, Cause> {
    Layout::array::<T>(n)
        .map(|l| l.size())
        .map_err(|_| Cause::Overflow)
}
fn add(total: &mut usize, amount: usize) -> Result<(), Cause> {
    *total = total.checked_add(amount).ok_or(Cause::Overflow)?;
    Ok(())
}
fn payload(source: &RegexAst) -> &[u8] {
    match source {
        RegexAst::Regex(v) | RegexAst::SearchRegex(v) | RegexAst::Literal(v) => v.as_bytes(),
        RegexAst::ByteLiteral(v) => v,
        _ => &[],
    }
}
fn frame(error: FrameError<ParserAllocationFailure>) -> Cause {
    match error { FrameError::Overflow => Cause::Overflow, FrameError::Funding(error) => Cause::Funding(error) }
}
type InspectFrame<'a, F> = (
    &'a RegexAst, usize, &'a mut Geometry, &'a F, &'a [RegexAst], usize,
    std::slice::Iter<'a, RegexAst>, Result<(), Cause>,
    Result<Layout, std::alloc::LayoutError>, Option<usize>,
);
type PlanFrame<'a, F> = (
    &'a RegexAst, &'a F, Geometry, RegexAstCopyPlan<'a>, RegexAstCopyRequirements,
    RegexAstCopyFailure, Cause, Result<RegexAstCopyPlan<'a>, Cause>,
    Result<RegexAstCopyPlan<'a>, RegexAstCopyFailure>, [usize; 20],
    usize, usize, usize, usize, Option<usize>, Result<Layout, std::alloc::LayoutError>,
);
fn inspect<F: PreparedFunding<Error = ParserAllocationFailure>>(
    source: &RegexAst, depth: usize, g: &mut Geometry, funding: &F,
) -> Result<(), Cause> {
    let _frame = funding.frame(size_of::<InspectFrame<'_, F>>()).map_err(frame)?;
    add(&mut g.nodes, 1)?;
    g.depth = g.depth.max(depth);
    let args = source.get_args();
    match source {
        RegexAst::And(_) | RegexAst::Or(_) | RegexAst::Concat(_) => {
            add(&mut g.buffers, array::<RegexAst>(args.len())?)?
        }
        RegexAst::LookAhead(_)
        | RegexAst::Not(_)
        | RegexAst::Repeat(_, _, _)
        | RegexAst::JsonQuote(_, _) => add(&mut g.buffers, size_of::<RegexAst>())?,
        RegexAst::ByteSet(v) => add(&mut g.buffers, array::<u32>(v.len())?)?,
        _ => {}
    }
    let bytes = payload(source).len();
    add(&mut g.buffers, bytes)?;
    add(&mut g.byte_words, bytes)?;
    if let RegexAst::JsonQuote(_, o) = source {
        add(&mut g.buffers, o.allowed_escapes.len())?;
        add(&mut g.byte_words, o.allowed_escapes.len())?;
    }
    for child in args {
        inspect(child, depth.checked_add(1).ok_or(Cause::Overflow)?, g, funding)?;
    }
    Ok(())
}
impl RegexAst {
    /// Actual descendant allocations retained by this tree, including spare
    /// vector/string capacity. This creates no source or compilation allowance.
    pub fn retained_capacity_bytes<F: PreparedFunding<Error = ParserAllocationFailure>>(
        &self, funding: &F,
    ) -> Result<usize, RegexAstCopyFailure> {
        retained(self, funding).map_err(failure)
    }

    /// Counts this actual tree without parsing names or compiling expressions.
    /// Every reached inspection frame uses the caller's same live scope.
    pub fn source_copy_plan<F: PreparedFunding<Error = ParserAllocationFailure>>(
        &self, funding: &F,
    ) -> Result<RegexAstCopyPlan<'_>, RegexAstCopyFailure> {
        RegexAstCopyPlan::prepare(self, funding).map_err(failure)
    }

}
fn failure(cause: Cause) -> RegexAstCopyFailure {
    RegexAstCopyFailure { cause, destinations: Vec::new(), completed: None }
}
fn retained<F: PreparedFunding<Error = ParserAllocationFailure>>(
    source: &RegexAst, funding: &F,
) -> Result<usize, Cause> {
    let _frame = funding.frame(size_of::<(
        &RegexAst, &F, usize, Option<usize>, Result<usize, Cause>,
        std::slice::Iter<'_, RegexAst>,
    )>()).map_err(frame)?;
    let mut own = (|| match source {
        RegexAst::And(v) | RegexAst::Or(v) | RegexAst::Concat(v) => v.capacity().checked_mul(size_of::<RegexAst>()),
        RegexAst::LookAhead(_) | RegexAst::Not(_) | RegexAst::Repeat(_, _, _) => Some(size_of::<RegexAst>()),
        RegexAst::JsonQuote(_, options) => size_of::<RegexAst>().checked_add(options.allowed_escapes.capacity()),
        RegexAst::ByteSet(v) => v.capacity().checked_mul(size_of::<u32>()),
        RegexAst::Regex(v) | RegexAst::SearchRegex(v) | RegexAst::Literal(v) => Some(v.capacity()),
        RegexAst::ByteLiteral(v) => Some(v.capacity()),
        _ => Some(0),
    })().ok_or(Cause::Overflow)?;
    for child in source.get_args() { add(&mut own, retained(child, funding)?)?; }
    Ok(own)
}
fn reserve<T>(v: &mut Vec<T>, n: usize) -> Result<(), Cause> {
    v.try_reserve_exact(n).map_err(Cause::Vector)?;
    if v.capacity() != n {
        return Err(Cause::Capacity);
    }
    Ok(())
}
fn string(bytes: &mut Vec<u8>) -> Result<String, Cause> {
    String::from_utf8(std::mem::take(bytes)).map_err(Cause::Utf8)
}
fn assemble(source: &RegexAst, d: &mut Destination) -> Result<RegexAst, Cause> {
    Ok(match source {
        RegexAst::And(_) => RegexAst::And(std::mem::take(&mut d.children)),
        RegexAst::Or(_) => RegexAst::Or(std::mem::take(&mut d.children)),
        RegexAst::Concat(_) => RegexAst::Concat(std::mem::take(&mut d.children)),
        RegexAst::LookAhead(_) => {
            RegexAst::LookAhead(Box::new(d.child.take().ok_or(Cause::Capacity)?))
        }
        RegexAst::Not(_) => RegexAst::Not(Box::new(d.child.take().ok_or(Cause::Capacity)?)),
        RegexAst::Repeat(_, a, b) => {
            RegexAst::Repeat(Box::new(d.child.take().ok_or(Cause::Capacity)?), *a, *b)
        }
        RegexAst::JsonQuote(_, o) => RegexAst::JsonQuote(
            Box::new(d.child.take().ok_or(Cause::Capacity)?),
            JsonQuoteOptions {
                allowed_escapes: string(&mut d.options)?,
                raw_mode: o.raw_mode,
            },
        ),
        RegexAst::Regex(_) => RegexAst::Regex(string(&mut d.bytes)?),
        RegexAst::SearchRegex(_) => RegexAst::SearchRegex(string(&mut d.bytes)?),
        RegexAst::Literal(_) => RegexAst::Literal(string(&mut d.bytes)?),
        RegexAst::ByteLiteral(_) => RegexAst::ByteLiteral(std::mem::take(&mut d.bytes)),
        RegexAst::ByteSet(_) => RegexAst::ByteSet(std::mem::take(&mut d.words)),
        RegexAst::MultipleOf(a, b) => RegexAst::MultipleOf(*a, *b),
        RegexAst::EmptyString => RegexAst::EmptyString,
        RegexAst::NoMatch => RegexAst::NoMatch,
        RegexAst::Byte(b) => RegexAst::Byte(*b),
        RegexAst::ExprRef(e) => RegexAst::ExprRef(*e),
    })
}
impl<'a> RegexAstCopyPlan<'a> {
    fn prepare<F: PreparedFunding<Error = ParserAllocationFailure>>(
        source: &'a RegexAst, funding: &F,
    ) -> Result<Self, Cause> {
        let _frame = funding.frame(size_of::<PlanFrame<'_, F>>()).map_err(frame)?;
        let mut g = Geometry::default();
        inspect(source, 1, &mut g, funding)?;
        let retained = g.buffers;
        add(&mut g.buffers, array::<Frame<'_>>(g.depth)?)?;
        add(&mut g.buffers, array::<Destination>(g.depth)?)?;
        let parts = [
            size_of::<Self>(),
            size_of::<Geometry>(),
            size_of::<RegexAstCopyRequirements>(),
            size_of::<RegexAstCopyFailure>(),
            size_of::<Cause>(),
            size_of::<Frame<'_>>(),
            size_of::<Destination>(),
            size_of::<Vec<Frame<'_>>>(),
            size_of::<Vec<Destination>>(),
            size_of::<Result<Self, Cause>>(),
            size_of::<Result<Self, RegexAstCopyFailure>>(),
            size_of::<Result<RegexAst, RegexAstCopyFailure>>(),
            size_of::<Result<RegexAst, Cause>>(),
            size_of::<Result<(), Cause>>(),
            size_of::<Result<(), TryReserveError>>(),
            size_of::<Result<String, FromUtf8Error>>(),
            size_of::<Option<RegexAst>>(),
            size_of::<Box<RegexAst>>(),
            size_of::<JsonQuoteOptions>(),
            size_of::<Result<Layout, std::alloc::LayoutError>>(),
        ];
        let mut controls = parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
            .ok_or(Cause::Overflow)?;
        // Nested copy workers re-inspect their existing source under this prepaid
        // requirement; count the same entry/guard layout, not only the old walk locals.
        let recursive = crate::prepared_funding::frame_control_bytes::<ParserAllocationFailure>(
            size_of::<InspectFrame<'_, F>>()
        ).ok_or(Cause::Overflow)?;
        add(&mut controls, recursive.checked_mul(g.depth).ok_or(Cause::Overflow)?)?;
        add(&mut controls, crate::prepared_funding::frame_control_bytes::<ParserAllocationFailure>(
            size_of::<PlanFrame<'_, F>>()
        ).ok_or(Cause::Overflow)?)?;
        let total = g.buffers.checked_add(controls).ok_or(Cause::Overflow)?;
        Ok(Self {
            source,
            byte_words: g.byte_words,
            requirements: RegexAstCopyRequirements {
                buffers: g.buffers,
                retained,
                controls,
                total,
                nodes: g.nodes,
                depth: g.depth,
            },
        })
    }
    /// Complete local source-constructor requirement.
    pub fn requirements(&self) -> RegexAstCopyRequirements {
        self.requirements
    }
    /// Copies the actual tree through finite destination/parent stacks. Ordinary
    /// Clone uses this same worker; every Box has its exact source-node quote.
    pub fn compile(self) -> Result<RegexAst, RegexAstCopyFailure> {
        let mut frames = Vec::new();
        let mut destinations = Vec::new();
        let mut completed = None;
        let mut byte_words = self.byte_words;
        let mut visited = 0usize;
        let result = (|| -> Result<(), Cause> {
            reserve(&mut frames, self.requirements.depth)?;
            reserve(&mut destinations, self.requirements.depth)?;
            frames.push(Frame {
                source: self.source,
                next: 0,
            });
            destinations.push(Destination::default());
            loop {
                let frame = frames.last_mut().ok_or(Cause::Capacity)?;
                let source = frame.source;
                let args = source.get_args();
                if frame.next == 0 {
                    visited = visited.checked_add(1).ok_or(Cause::Overflow)?;
                    if visited > self.requirements.nodes {
                        return Err(Cause::Capacity);
                    }
                    let d = destinations.last_mut().ok_or(Cause::Capacity)?;
                    if matches!(
                        source,
                        RegexAst::And(_) | RegexAst::Or(_) | RegexAst::Concat(_)
                    ) {
                        reserve(&mut d.children, args.len())?;
                    }
                    let bytes = payload(source);
                    byte_words = byte_words.checked_sub(bytes.len()).ok_or(Cause::Capacity)?;
                    reserve(&mut d.bytes, bytes.len())?;
                    d.bytes.extend_from_slice(bytes);
                    if let RegexAst::ByteSet(words) = source {
                        reserve(&mut d.words, words.len())?;
                        d.words.extend_from_slice(words);
                    }
                    if let RegexAst::JsonQuote(_, o) = source {
                        byte_words = byte_words
                            .checked_sub(o.allowed_escapes.len())
                            .ok_or(Cause::Capacity)?;
                        reserve(&mut d.options, o.allowed_escapes.len())?;
                        d.options.extend_from_slice(o.allowed_escapes.as_bytes());
                    }
                }
                if frame.next < args.len() {
                    let child = &args[frame.next];
                    frame.next += 1;
                    if frames.len() == frames.capacity()
                        || destinations.len() == destinations.capacity()
                    {
                        return Err(Cause::Capacity);
                    }
                    frames.push(Frame {
                        source: child,
                        next: 0,
                    });
                    destinations.push(Destination::default());
                    continue;
                }
                let d = destinations.last_mut().ok_or(Cause::Capacity)?;
                completed = Some(assemble(source, d)?);
                frames.pop();
                destinations.pop();
                let Some(parent) = frames.last() else { break };
                let d = destinations.last_mut().ok_or(Cause::Capacity)?;
                if matches!(
                    parent.source,
                    RegexAst::And(_) | RegexAst::Or(_) | RegexAst::Concat(_)
                ) {
                    if d.children.len() == d.children.capacity() {
                        return Err(Cause::Capacity);
                    }
                    d.children.push(completed.take().ok_or(Cause::Capacity)?);
                } else {
                    if d.child.is_some() {
                        return Err(Cause::Capacity);
                    }
                    d.child = completed.take();
                }
            }
            if byte_words != 0 || visited != self.requirements.nodes {
                return Err(Cause::Capacity);
            }
            Ok(())
        })();
        match result {
            Ok(()) => Ok(completed.take().expect("completed actual AST source")),
            Err(cause) => Err(RegexAstCopyFailure {
                cause,
                destinations,
                completed,
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RegexBuilder;
    #[test]
    fn ast_source_copy_keeps_all_payloads_deep_unary_nodes_and_failed_siblings() {
        let source = RegexAst::Concat(vec![
            RegexAst::Literal("kept".into()),
            RegexAst::JsonQuote(
                Box::new(RegexAst::Or(vec![
                    RegexAst::Literal("α\n".into()),
                    RegexAst::ByteLiteral(vec![b'b', b'c']),
                ])),
                JsonQuoteOptions::regular(),
            ),
            RegexAst::Repeat(Box::new(RegexAst::Byte(b'!')), 1, 3),
            RegexAst::LookAhead(Box::new(RegexAst::Regex("[0-9]".into()))),
        ]);
        let plan = source.source_copy_plan(&crate::ParserAllocationFunding::unenforced()).unwrap();
        let requirements = plan.requirements();
        let copied = plan.compile().unwrap();
        assert!(requirements.nodes() > requirements.depth());
        assert!(requirements.required_bytes() > requirements.buffer_bytes());
        assert_eq!(format!("{source:?}"), format!("{copied:?}"));
        let mut builder = RegexBuilder::new(crate::ParserAllocationFunding::unenforced()).unwrap();
        let id = builder.mk(&source).unwrap();
        assert_eq!(builder.mk(&copied).unwrap(), id);
        let mut matcher = builder.to_regex(id).unwrap();
        assert!(matcher.is_match("kept\"bc\"!!5").unwrap());
        assert!(!matcher.is_match("kept\"bc\"!!x").unwrap());

        let other = RegexAst::And(vec![
            RegexAst::SearchRegex("α".into()),
            RegexAst::Not(Box::new(RegexAst::NoMatch)),
            RegexAst::EmptyString,
            RegexAst::MultipleOf(7, 2),
            RegexAst::ByteSet(vec![0x55, 0xaa]),
            RegexAst::ExprRef(crate::ExprRef::ANY_BYTE),
        ]);
        let other_copy = other.source_copy_plan(&crate::ParserAllocationFunding::unenforced()).unwrap().compile().unwrap();
        assert_eq!(format!("{other:?}"), format!("{other_copy:?}"));

        let mut deep = RegexAst::Byte(b'x');
        for _ in 0..640 {
            deep = RegexAst::Repeat(Box::new(deep), 1, 1);
        }
        let deep_plan = deep.source_copy_plan(&crate::ParserAllocationFunding::unenforced()).unwrap();
        assert_eq!(deep_plan.requirements().depth(), 641);
        assert_eq!(
            deep_plan.requirements().retained_bytes(),
            640 * size_of::<RegexAst>()
        );
        assert!(
            deep_plan.requirements().buffer_bytes() > deep_plan.requirements().retained_bytes()
        );
        let deep_copy = deep_plan.compile().unwrap();
        let mut node = &deep_copy;
        for _ in 0..640 {
            let RegexAst::Repeat(child, 1, 1) = node else {
                panic!("lost unary source")
            };
            node = child;
        }
        assert!(matches!(node, RegexAst::Byte(b'x')));

        let mut failing = source.source_copy_plan(&crate::ParserAllocationFunding::unenforced()).unwrap();
        failing.byte_words = 4;
        let failure = match failing.compile() {
            Err(e) => e,
            Ok(_) => panic!("incomplete byte population accepted"),
        };
        assert!(matches!(failure.cause, Cause::Capacity));
        assert_eq!(failure.destinations.len(), 2);
        drop(source);
        drop(copied);
        drop(builder);
        drop(matcher);
        let siblings = &failure.destinations[0].children;
        assert!(siblings.capacity() > 1);
        assert!(matches!(siblings.as_slice(),[RegexAst::Literal(value)] if value=="kept"));
        assert!(failure.completed.is_none());
    }
}

#[cfg(test)]
mod inspection_tests;
