//! Source-bound expression syntax through the ordinary shared parser.
//!
//! This mechanism constructs syntax, not instructions or a runnable template.
//! It grants no admission and accepts no caller capacities.
//!
//! ```compile_fail
//! use minijinja::bounded::expression::Plan;
//! let parsed = {
//!     let source = String::from("user.name");
//!     Plan::inspect(&source, "example").unwrap().construct().unwrap()
//! };
//! println!("{}", parsed.source());
//! ```
//!
//! Failure storage retains the same source lifetime:
//! ```compile_fail
//! use minijinja::bounded::expression::Plan;
//! let failure = {
//!     let source = String::from("[value, ] +");
//!     Plan::inspect(&source, "example").unwrap().construct().unwrap_err()
//! };
//! println!("{failure}");
//! ```
//!
//! Borrowed projections cannot survive their specific owner:
//! ```compile_fail
//! use minijinja::bounded::expression::Plan;
//! let source = String::from("value");
//! let view = {
//!     let owner = Plan::inspect(&source, "example").unwrap().construct().unwrap();
//!     owner.source()
//! };
//! println!("{view}");
//! ```
#![forbid(unsafe_code)]

use std::alloc::Layout;
use std::collections::TryReserveError;
use std::fmt;
use std::mem::size_of;

use crate::compiler::lexer::literal::LexError;
use crate::compiler::parser::{shared, storage};
pub use crate::compiler::tokens::Span;
use crate::ErrorKind;

pub(super) mod input;
pub(super) mod store;
use input::{Buffers, ParserInput};
use store::Storage;

/// A source or requested layout cannot be represented.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlanError {
    /// Checked source offsets, counts or array layouts overflow.
    Geometry,
}
impl fmt::Display for PlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("expression source geometry cannot be represented")
    }
}
impl std::error::Error for PlanError {}

/// One real destination, reserved once before parsing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Buffer {
    /// Flat syntax nodes, including consumed temporary nodes.
    Nodes,
    /// Variable-length child, argument and comparison records.
    Operands,
    /// Plain/decoded pieces of adjacent string literals.
    Segments,
    /// Decoded escaped bytes, including failed or prefetched literals.
    Literals,
    /// Reused underscore-stripped number conversion bytes.
    NumericScratch,
    /// Explicit parser continuation frames.
    Frames,
    /// Flat full-template statements, used only by template syntax construction.
    Statements,
    /// Full-template body links.
    Bodies,
    /// Full-template duplicate-block name storage.
    BlockNames,
    /// Full-template statement/assignment continuations.
    StatementFrames,
}

/// Checked requested capacities and actual concrete control representations.
#[derive(Clone, Copy, Debug)]
pub struct Requirements {
    events: usize,
    literal_bytes: usize,
    numeric_bytes: usize,
    frames: usize,
    heap_bytes: usize,
}
impl Requirements {
    /// Source fetch-event bound for each flat record population. Advancing
    /// scanner failures and one terminal failure are included conservatively.
    pub fn records(self) -> usize {
        self.events
    }
    /// Decoded bytes over all reachable literal fetches, including partial failures.
    pub fn literal_bytes(self) -> usize {
        self.literal_bytes
    }
    /// Maximum underscore-stripped numeric length.
    pub fn numeric_bytes(self) -> usize {
        self.numeric_bytes
    }
    /// Explicit frame bound from the finite zero-consumption call graph.
    pub fn frames(self) -> usize {
        self.frames
    }
    /// Sum of checked requested array layouts; no allocator-rounding claim.
    pub fn heap_bytes(self) -> usize {
        self.heap_bytes
    }
    /// Concrete source-plan shell, separate from construction results.
    pub fn plan_control_bytes(self) -> usize {
        size_of::<Plan<'static>>()
    }
    /// Larger retained success/error shell, including all Vec controls.
    pub fn retained_control_bytes(self) -> usize {
        size_of::<ParsedExpression<'static>>().max(size_of::<ConstructError<'static>>())
    }
    /// Actual scanner/input, stack view, depth and movable dispatcher controls.
    /// These are typed representations, not a whole-thread stack bound.
    pub fn worker_control_bytes(self) -> usize {
        size_of::<ParserInput<'static, 'static>>()
            + size_of::<FixedStack<'static, 'static>>()
            + size_of::<usize>()
            + shared::control_bytes::<Storage<'static>, ParserInput<'static, 'static>>()
    }
}

fn requirements(
    events: usize,
    literal_bytes: usize,
    numeric_bytes: usize,
) -> Result<Requirements, PlanError> {
    let frames = events
        .checked_add(1)
        .and_then(|n| n.checked_mul(shared::ZERO_RUN_FRAMES))
        .ok_or(PlanError::Geometry)?;
    let sizes = [
        Layout::array::<store::Record<'static>>(events),
        Layout::array::<store::Operand<'static>>(events),
        Layout::array::<store::Segment<'static>>(events),
        Layout::array::<u8>(literal_bytes),
        Layout::array::<u8>(numeric_bytes),
        Layout::array::<shared::Frame<'static, Storage<'static>>>(frames),
    ];
    let mut heap_bytes = 0usize;
    for layout in sizes {
        heap_bytes = heap_bytes
            .checked_add(layout.map_err(|_| PlanError::Geometry)?.size())
            .ok_or(PlanError::Geometry)?;
    }
    Ok(Requirements {
        events,
        literal_bytes,
        numeric_bytes,
        frames,
        heap_bytes,
    })
}

/// Immutable source-derived geometry for the default standalone expression syntax.
#[derive(Debug)]
pub struct Plan<'s> {
    source: &'s str,
    filename: &'s str,
    requirements: Requirements,
    #[cfg(test)]
    ceiling: Option<(Buffer, usize)>,
}
impl<'s> Plan<'s> {
    /// Inspect source without allocating or reporting its later syntax/literal
    /// error early. Syntax errors retain real construction prefixes instead.
    pub fn inspect(source: &'s str, filename: &'s str) -> Result<Self, PlanError> {
        super::frontend::check_source_geometry(source).map_err(|_| PlanError::Geometry)?;
        let (events, bytes, numeric) = input::inspect(source, filename)?;
        Ok(Self {
            source,
            filename,
            requirements: requirements(events, bytes, numeric)?,
            #[cfg(test)]
            ceiling: None,
        })
    }
    /// Checked requested layouts; these values cannot construct another plan.
    pub fn requirements(&self) -> Requirements {
        self.requirements
    }
    /// Reserve each destination once and run the shared expression worker.
    pub fn construct(self) -> Result<ParsedExpression<'s>, ConstructError<'s>> {
        self.construct_inner(None)
    }
    fn construct_inner(
        self,
        failure: Option<Buffer>,
    ) -> Result<ParsedExpression<'s>, ConstructError<'s>> {
        let mut owner = ParsedExpression {
            source: self.source,
            filename: self.filename,
            requirements: self.requirements,
            storage: Storage::new(self.requirements.events),
            buffers: Buffers::new(
                self.requirements.literal_bytes,
                self.requirements.numeric_bytes,
            ),
            frames: Vec::new(),
            root: None,
        };
        macro_rules! reserve {
            ($destination:expr, $count:expr, $which:expr) => {{
                let count = if failure == Some($which) {
                    usize::MAX
                } else {
                    $count
                };
                if let Err(error) = ($destination).try_reserve_exact(count) {
                    return Err(ConstructError {
                        owner,
                        cause: Cause::Reserve {
                            buffer: $which,
                            error,
                        },
                    });
                }
            }};
        }
        reserve!(owner.storage.nodes, self.requirements.events, Buffer::Nodes);
        reserve!(
            owner.storage.operands,
            self.requirements.events,
            Buffer::Operands
        );
        reserve!(
            owner.storage.segments,
            self.requirements.events,
            Buffer::Segments
        );
        reserve!(
            owner.buffers.literals,
            self.requirements.literal_bytes,
            Buffer::Literals
        );
        reserve!(
            owner.buffers.numeric,
            self.requirements.numeric_bytes,
            Buffer::NumericScratch
        );
        reserve!(owner.frames, self.requirements.frames, Buffer::Frames);
        #[cfg(test)]
        if let Some((buffer, limit)) = self.ceiling {
            match buffer {
                Buffer::Nodes => owner.storage.node_limit = limit,
                Buffer::Operands => owner.storage.operand_limit = limit,
                Buffer::Segments => owner.storage.segment_limit = limit,
                Buffer::Literals => owner.buffers.literal_limit = limit,
                Buffer::NumericScratch => owner.buffers.numeric_limit = limit,
                Buffer::Frames
                | Buffer::Statements
                | Buffer::Bodies
                | Buffer::BlockNames
                | Buffer::StatementFrames => (),
            }
        }
        let frame_limit = self.requirements.frames;
        #[cfg(test)]
        let frame_limit = match self.ceiling {
            Some((Buffer::Frames, limit)) => limit,
            _ => frame_limit,
        };
        let result = {
            let mut input = ParserInput::new(self.source, self.filename, &mut owner.buffers);
            let mut frames = FixedStack {
                values: &mut owner.frames,
                limit: frame_limit,
            };
            let mut depth = 0;
            let result = shared::run(
                shared::Method::Expr,
                &mut input,
                &mut owner.storage,
                &mut frames,
                &mut depth,
            )
            .and_then(|value| {
                use storage::Input;
                if input.next()?.is_some() {
                    return Err(Cause::Syntax(SyntaxError {
                        detail: storage::SyntaxFailure::Static("unexpected input after expression"),
                        span: None,
                    }));
                }
                match value {
                    shared::Output::Expr(root) => Ok(root),
                    _ => unreachable!("standalone expression output"),
                }
            });
            result.map_err(|mut error| {
                error.attach(storage::Input::last_span(&input));
                error
            })
        };
        match result {
            Ok(root) => {
                owner.root = Some(root);
                Ok(owner)
            }
            Err(cause) => Err(ConstructError { owner, cause }),
        }
    }
}

struct FixedStack<'a, 's> {
    values: &'a mut Vec<shared::Frame<'s, Storage<'s>>>,
    limit: usize,
}
impl<'s> shared::Stack<'s, Storage<'s>> for FixedStack<'_, 's> {
    fn push(&mut self, frame: shared::Frame<'s, Storage<'s>>) -> Result<(), Cause<'s>> {
        if self.values.len() >= self.limit || self.values.len() == self.values.capacity() {
            return Err(Cause::Capacity(Buffer::Frames));
        }
        self.values.push(frame);
        Ok(())
    }
    fn pop(&mut self) -> Option<shared::Frame<'s, Storage<'s>>> {
        self.values.pop()
    }
}

/// Closed syntax storage retaining its source and all construction destinations.
/// It has no ordinary AST/Value, mutable storage or clone escape.
pub struct ParsedExpression<'s> {
    source: &'s str,
    filename: &'s str,
    requirements: Requirements,
    storage: Storage<'s>,
    buffers: Buffers,
    frames: Vec<shared::Frame<'s, Storage<'s>>>,
    root: Option<store::NodeId>,
}
impl ParsedExpression<'_> {
    /// Exact borrowed source inspected by the plan.
    pub fn source(&self) -> &str {
        self.source
    }
    /// Exact borrowed diagnostic filename.
    pub fn filename(&self) -> &str {
        self.filename
    }
    /// Requested layouts retained by this owner.
    pub fn requirements(&self) -> Requirements {
        self.requirements
    }
    /// Actual constructed node records, including discarded grammar temporaries.
    pub fn node_count(&self) -> usize {
        self.storage.nodes.len()
    }
    /// Full source span of the root expression.
    pub fn root_span(&self) -> Span {
        self.storage.nodes[self.root.expect("successful root").0].span
    }
}
impl fmt::Debug for ParsedExpression<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ParsedExpression")
            .field("requirements", &self.requirements)
            .field("nodes", &self.storage.nodes.len())
            .field("operands", &self.storage.operands.len())
            .field("segments", &self.storage.segments.len())
            .field("decoded_bytes", &self.buffers.literals.len())
            .finish()
    }
}

/// Nonallocating parser diagnostic, with the same semantic detail and full span.
#[derive(Debug)]
pub struct SyntaxError<'s> {
    detail: storage::SyntaxFailure<'s>,
    span: Option<Span>,
}
impl SyntaxError<'_> {
    /// Full span attached at the ordinary parser's corresponding boundary.
    pub fn span(&self) -> Option<Span> {
        self.span
    }
}
impl fmt::Display for SyntaxError<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.detail.fmt(f)
    }
}

/// Nonallocating scanner/literal error preserving kind, detail and location.
#[derive(Debug)]
pub struct LexicalError(LexError);
impl LexicalError {
    /// Ordinary semantic error category.
    pub fn kind(&self) -> ErrorKind {
        self.0.kind
    }
    /// Static ordinary detail, if present.
    pub fn detail(&self) -> Option<&'static str> {
        self.0.detail
    }
    /// Full ordinary span, including parser attachment for literal errors.
    pub fn span(&self) -> Option<Span> {
        self.0.span
    }
}
impl fmt::Display for LexicalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0.detail {
            Some(detail) => f.write_str(detail),
            None => self.0.kind.fmt(f),
        }
    }
}

/// Actual cause retained alongside all already reserved storage.
#[derive(Debug)]
pub enum Cause<'s> {
    /// The selected real destination's fallible reserve failed.
    Reserve {
        /// Actual target buffer.
        buffer: Buffer,
        /// Actual allocation/capacity error.
        error: TryReserveError,
    },
    /// A real guarded write exceeded its pre-reserved population.
    Capacity(Buffer),
    /// Ordinary scanner or literal error.
    Lexical(LexicalError),
    /// Ordinary expression grammar error.
    Syntax(SyntaxError<'s>),
    /// The ordinary standalone lexer reached its empty-stack panic state;
    /// the closed profile reports this state without panicking.
    ExpressionEnd,
}
impl<'s> Cause<'s> {
    fn lexical(mut error: LexError) -> Self {
        if error.empty_stack {
            return Self::ExpressionEnd;
        }
        if let Some(span) = error.span.as_mut() {
            if span.start_col == span.end_col {
                span.end_col += 1;
                span.end_offset += 1;
            }
        }
        Self::Lexical(LexicalError(error))
    }
    pub(super) fn attach(&mut self, span: Span) {
        match self {
            Self::Lexical(error) if error.0.span.is_none() => error.0.span = Some(span),
            Self::Syntax(error) if error.span.is_none() => error.span = Some(span),
            _ => (),
        }
    }
}
impl fmt::Display for Cause<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Reserve { buffer, error } => write!(f, "{buffer:?} reserve: {error}"),
            Self::Capacity(buffer) => write!(f, "{buffer:?} population exceeded"),
            Self::Lexical(error) => error.fmt(f),
            Self::Syntax(error) => error.fmt(f),
            Self::ExpressionEnd => {
                f.write_str("standalone expression lexer has no remaining state")
            }
        }
    }
}

/// Construction failure retaining the real cause, source and partial destinations.
#[derive(Debug)]
pub struct ConstructError<'s> {
    owner: ParsedExpression<'s>,
    cause: Cause<'s>,
}
impl ConstructError<'_> {
    /// Actual reserve, syntax, lexical or capacity cause.
    pub fn cause(&self) -> &Cause<'_> {
        &self.cause
    }
    /// Retained requested population/layout diagnostics.
    pub fn requirements(&self) -> Requirements {
        self.owner.requirements
    }
    /// Exact source kept alive through failure retirement.
    pub fn source_text(&self) -> &str {
        self.owner.source
    }
}
impl fmt::Display for ConstructError<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.cause.fmt(f)
    }
}
impl std::error::Error for ConstructError<'_> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.cause {
            Cause::Reserve { error, .. } => Some(error),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests;
