//! Closed, source-derived full-template syntax through the ordinary parser.
//!
//! No instructions, environment, original grant or rendering activation is created.
//! The exact borrowed source and default-delimiter whitespace settings determine
//! all requested capacities. Construction retains every destination on failure.
//!
//! ```compile_fail
//! use minijinja::bounded::template_syntax::{Plan, WhitespaceConfig};
//! let parsed = {
//!     let source = String::from("{% set x = 1 %}{{ x }}");
//!     Plan::inspect(&source, "example", WhitespaceConfig::default()).unwrap().construct().unwrap()
//! };
//! println!("{}", parsed.source());
//! ```
//!
//! ```compile_fail
//! use minijinja::bounded::template_syntax::{Plan, WhitespaceConfig};
//! let failure = {
//!     let source = String::from("{% for x in %}");
//!     Plan::inspect(&source, "example", WhitespaceConfig::default()).unwrap().construct().unwrap_err()
//! };
//! println!("{failure}");
//! ```
//!
//! ```compile_fail
//! use minijinja::bounded::template_syntax::{Plan, WhitespaceConfig};
//! let source = String::from("hello");
//! let view = {
//!     let owner = Plan::inspect(&source, "example", WhitespaceConfig::default()).unwrap().construct().unwrap();
//!     owner.source()
//! };
//! println!("{view}");
//! ```
#![forbid(unsafe_code)]

use super::expression::{input, store as expr};
pub use super::expression::{Buffer, Cause, PlanError};
pub use crate::compiler::lexer::WhitespaceConfig;
use crate::compiler::parser::{shared, statements, storage::Input};
pub use crate::compiler::tokens::Span;
use input::{Buffers, ParserInput};
use std::alloc::Layout;
use std::fmt;
use std::mem::size_of;
/// Reusable capture membership over this exact immutable syntax owner.
#[cfg(feature = "macros")]
pub mod captures;
/// Reusable constant-fold control over this exact immutable syntax owner.
pub mod constants;
pub(in crate::bounded) mod emit;
mod store;
use store::{BlockNames, Storage};

/// Checked requested capacities and concrete control layouts.
#[derive(Clone, Copy, Debug)]
pub struct Requirements {
    events: usize,
    literal_bytes: usize,
    numeric_bytes: usize,
    expression_frames: usize,
    statement_frames: usize,
    statements: usize,
    heap_bytes: usize,
}
impl Requirements {
    /// Fetch events, including advancing scanner errors and one stationary error.
    pub fn records(self) -> usize {
        self.events
    }
    /// Statement population, including the source-less root template.
    pub fn statements(self) -> usize {
        self.statements
    }
    /// Maximum cumulative decoded literal prefix, including lookahead/failure.
    pub fn literal_bytes(self) -> usize {
        self.literal_bytes
    }
    /// Maximum reused stripped-number scratch length.
    pub fn numeric_bytes(self) -> usize {
        self.numeric_bytes
    }
    /// Reusable expression continuation capacity.
    pub fn expression_frames(self) -> usize {
        self.expression_frames
    }
    /// Full-template continuation capacity from the actual zero-edge graph.
    pub fn statement_frames(self) -> usize {
        self.statement_frames
    }
    /// Sum of ten requested array layouts, excluding allocator rounding.
    pub fn heap_bytes(self) -> usize {
        self.heap_bytes
    }
    /// Actual source plan shell, with no allocation of its own.
    pub fn plan_control_bytes(self) -> usize {
        size_of::<Plan<'static>>()
    }
    /// Larger retained result/error shell, including all Vec controls.
    pub fn retained_control_bytes(self) -> usize {
        size_of::<ParsedTemplate<'static>>().max(size_of::<ConstructError<'static>>())
    }
    /// Concrete worker/input/stack/return shells; not a whole-thread stack bound.
    pub fn worker_control_bytes(self) -> usize {
        size_of::<ParserInput<'static, 'static>>()
            + size_of::<ExpressionStack<'static, 'static>>()
            + size_of::<StatementStack<'static, 'static>>()
            + shared::control_bytes::<Storage<'static>, ParserInput<'static, 'static>>()
            + statements::control_bytes::<
                Storage<'static>,
                ParserInput<'static, 'static>,
                ExpressionStack<'static, 'static>,
                BlockNames<'static>,
            >()
    }
}
// With default delimiters every successful scanner token consumes source
// bytes; a stationary scanner error contributes at most one final fetch. Raw
// blocks consume their delimiters even when their emitted body is empty. String
// unescaping and underscore removal never exceed their source-byte population.
// This bound lets an encoded source obtain J before its decoded buffer exists.
pub(in crate::bounded) fn byte_requirements(bytes: usize) -> Result<Requirements, PlanError> {
    requirements(
        bytes.checked_add(1).ok_or(PlanError::Geometry)?,
        bytes,
        bytes,
    )
}

fn requirements(
    events: usize,
    literal_bytes: usize,
    numeric_bytes: usize,
) -> Result<Requirements, PlanError> {
    let statements = events.checked_add(1).ok_or(PlanError::Geometry)?;
    let expression_frames = statements
        .checked_mul(shared::ZERO_RUN_FRAMES)
        .ok_or(PlanError::Geometry)?;
    let statement_frames = statements
        .checked_mul(statements::ZERO_RUN_FRAMES)
        .ok_or(PlanError::Geometry)?;
    let arrays = [
        Layout::array::<expr::Record<'static>>(events),
        Layout::array::<expr::Operand<'static>>(events),
        Layout::array::<expr::Segment<'static>>(events),
        Layout::array::<u8>(literal_bytes),
        Layout::array::<u8>(numeric_bytes),
        Layout::array::<shared::Frame<'static, Storage<'static>>>(expression_frames),
        Layout::array::<store::Record<'static>>(statements),
        Layout::array::<store::Link>(events),
        Layout::array::<&'static str>(events),
        Layout::array::<statements::Frame<'static, Storage<'static>>>(statement_frames),
    ];
    let mut heap_bytes = 0usize;
    for value in arrays {
        heap_bytes = heap_bytes
            .checked_add(value.map_err(|_| PlanError::Geometry)?.size())
            .ok_or(PlanError::Geometry)?;
    }
    Ok(Requirements {
        events,
        literal_bytes,
        numeric_bytes,
        expression_frames,
        statement_frames,
        statements,
        heap_bytes,
    })
}

/// Immutable geometry for one exact source and copied whitespace configuration.
#[derive(Debug)]
pub struct Plan<'s> {
    source: &'s str,
    filename: &'s str,
    whitespace: WhitespaceConfig,
    requirements: Requirements,
    #[cfg(test)]
    ceiling: Option<(Buffer, usize)>,
}
impl<'s> Plan<'s> {
    /// Inspect all reachable scanner fetches without eagerly reporting semantic errors.
    pub fn inspect(
        source: &'s str,
        filename: &'s str,
        whitespace: WhitespaceConfig,
    ) -> Result<Self, PlanError> {
        super::frontend::check_source_geometry(source).map_err(|_| PlanError::Geometry)?;
        let (events, bytes, numeric) = input::inspect_config(source, filename, false, whitespace)?;
        Ok(Self {
            source,
            filename,
            whitespace,
            requirements: requirements(events, bytes, numeric)?,
            #[cfg(test)]
            ceiling: None,
        })
    }
    /// Checked layouts; these diagnostics cannot create another plan.
    pub fn requirements(&self) -> Requirements {
        self.requirements
    }
    /// Reserve every destination before lazy input creation, then run the shared parser.
    pub fn construct(self) -> Result<ParsedTemplate<'s>, ConstructError<'s>> {
        self.construct_inner(None)
    }
    fn construct_inner(
        self,
        failure: Option<Buffer>,
    ) -> Result<ParsedTemplate<'s>, ConstructError<'s>> {
        let q = self.requirements;
        let mut owner = ParsedTemplate {
            source: self.source,
            filename: self.filename,
            requirements: q,
            whitespace: self.whitespace,
            storage: Storage::new(q.events, q.statements),
            buffers: Buffers::new(q.literal_bytes, q.numeric_bytes),
            expressions: Vec::new(),
            statements: Vec::new(),
            names: BlockNames {
                values: Vec::new(),
                limit: q.events,
            },
            root: None,
            depth: 0,
            in_loop: false,
            in_macro: false,
        };
        macro_rules! reserve {
            ($target:expr,$count:expr,$buffer:expr) => {{
                let count = if failure == Some($buffer) {
                    usize::MAX
                } else {
                    $count
                };
                if let Err(error) = ($target).try_reserve_exact(count) {
                    return Err(ConstructError {
                        owner,
                        cause: Cause::Reserve {
                            buffer: $buffer,
                            error,
                        },
                    });
                }
            }};
        }
        reserve!(owner.storage.expressions.nodes, q.events, Buffer::Nodes);
        reserve!(
            owner.storage.expressions.operands,
            q.events,
            Buffer::Operands
        );
        reserve!(
            owner.storage.expressions.segments,
            q.events,
            Buffer::Segments
        );
        reserve!(owner.buffers.literals, q.literal_bytes, Buffer::Literals);
        reserve!(
            owner.buffers.numeric,
            q.numeric_bytes,
            Buffer::NumericScratch
        );
        reserve!(owner.expressions, q.expression_frames, Buffer::Frames);
        reserve!(owner.storage.statements, q.statements, Buffer::Statements);
        reserve!(owner.storage.links, q.events, Buffer::Bodies);
        reserve!(owner.names.values, q.events, Buffer::BlockNames);
        reserve!(
            owner.statements,
            q.statement_frames,
            Buffer::StatementFrames
        );
        let expression_limit = q.expression_frames;
        let statement_limit = q.statement_frames;
        #[cfg(test)]
        let (expression_limit, statement_limit) = {
            let mut e = expression_limit;
            let mut s = statement_limit;
            if let Some((which, limit)) = self.ceiling {
                match which {
                    Buffer::Nodes => owner.storage.expressions.node_limit = limit,
                    Buffer::Operands => owner.storage.expressions.operand_limit = limit,
                    Buffer::Segments => owner.storage.expressions.segment_limit = limit,
                    Buffer::Literals => owner.buffers.literal_limit = limit,
                    Buffer::NumericScratch => owner.buffers.numeric_limit = limit,
                    Buffer::Frames => e = limit,
                    Buffer::Statements => owner.storage.statement_limit = limit,
                    Buffer::Bodies => owner.storage.link_limit = limit,
                    Buffer::BlockNames => owner.names.limit = limit,
                    Buffer::StatementFrames => s = limit,
                }
            }
            (e, s)
        };
        let result = {
            let mut input = ParserInput::configured(
                self.source,
                self.filename,
                &mut owner.buffers,
                false,
                self.whitespace,
            );
            let mut expressions = ExpressionStack {
                values: &mut owner.expressions,
                limit: expression_limit,
            };
            let mut stack = StatementStack {
                values: &mut owner.statements,
                limit: statement_limit,
            };
            statements::run(
                &mut input,
                &mut owner.storage,
                &mut expressions,
                &mut owner.names,
                &mut stack,
                statements::State {
                    depth: &mut owner.depth,
                    in_loop: &mut owner.in_loop,
                    in_macro: &mut owner.in_macro,
                },
            )
            .map_err(|mut cause| {
                cause.attach(input.last_span());
                cause
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
struct ExpressionStack<'a, 's> {
    values: &'a mut Vec<shared::Frame<'s, Storage<'s>>>,
    limit: usize,
}
impl<'s> shared::Stack<'s, Storage<'s>> for ExpressionStack<'_, 's> {
    fn push(&mut self, value: shared::Frame<'s, Storage<'s>>) -> Result<(), Cause<'s>> {
        if self.values.len() >= self.limit || self.values.len() == self.values.capacity() {
            return Err(Cause::Capacity(Buffer::Frames));
        }
        self.values.push(value);
        Ok(())
    }
    fn pop(&mut self) -> Option<shared::Frame<'s, Storage<'s>>> {
        self.values.pop()
    }
}
struct StatementStack<'a, 's> {
    values: &'a mut Vec<statements::Frame<'s, Storage<'s>>>,
    limit: usize,
}
impl<'s> statements::Stack<'s, Storage<'s>> for StatementStack<'_, 's> {
    fn push(&mut self, value: statements::Frame<'s, Storage<'s>>) -> Result<(), Cause<'s>> {
        if self.values.len() >= self.limit || self.values.len() == self.values.capacity() {
            return Err(Cause::Capacity(Buffer::StatementFrames));
        }
        self.values.push(value);
        Ok(())
    }
    fn pop(&mut self) -> Option<statements::Frame<'s, Storage<'s>>> {
        self.values.pop()
    }
}

/// Flat syntax owner, with private IDs and no AST/Value/clone escape.
pub struct ParsedTemplate<'s> {
    source: &'s str,
    filename: &'s str,
    requirements: Requirements,
    whitespace: WhitespaceConfig,
    storage: Storage<'s>,
    buffers: Buffers,
    expressions: Vec<shared::Frame<'s, Storage<'s>>>,
    statements: Vec<statements::Frame<'s, Storage<'s>>>,
    names: BlockNames<'s>,
    root: Option<store::StmtId>,
    depth: usize,
    in_loop: bool,
    in_macro: bool,
}
impl ParsedTemplate<'_> {
    /// Source borrow tied to this specific retained owner.
    pub fn source(&self) -> &str {
        self.source
    }
    /// Exact diagnostic filename.
    pub fn filename(&self) -> &str {
        self.filename
    }
    /// Copied whitespace settings used by both inspection and parsing.
    pub fn whitespace(&self) -> WhitespaceConfig {
        self.whitespace
    }
    /// Requested storage retained by this owner.
    pub fn requirements(&self) -> Requirements {
        self.requirements
    }
    /// Constructed expression nodes, including discarded assignment/call intermediates.
    pub fn expression_count(&self) -> usize {
        self.storage.expressions.nodes.len()
    }
    /// Constructed statements, including the root.
    pub fn statement_count(&self) -> usize {
        self.storage.statements.len()
    }
    /// Ordinary full-template root span.
    pub fn root_span(&self) -> Span {
        self.storage.statements[self.root.expect("successful template").0].span
    }
}
impl fmt::Debug for ParsedTemplate<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ParsedTemplate")
            .field("requirements", &self.requirements)
            .field("statements", &self.storage.statements.len())
            .field("expressions", &self.storage.expressions.nodes.len())
            .field("literal_bytes", &self.buffers.literals.len())
            .finish()
    }
}
/// Real failure plus all already reserved storage and original source borrows.
#[derive(Debug)]
pub struct ConstructError<'s> {
    owner: ParsedTemplate<'s>,
    cause: Cause<'s>,
}
impl ConstructError<'_> {
    pub(in crate::bounded) fn into_compile_cause(self) -> super::source::CompileCause {
        let Self { owner, cause } = self;
        let result = match cause {
            Cause::Reserve { error, .. } => super::source::CompileCause::SyntaxReserve(error),
            Cause::Capacity(_) => {
                super::source::CompileCause::Source(super::source::SourceError::Geometry)
            }
            _ => super::source::CompileCause::Source(super::source::SourceError::Syntax),
        };
        drop(owner);
        result
    }
    /// Original reserve or semantic/capacity cause.
    pub fn cause(&self) -> &Cause<'_> {
        &self.cause
    }
    /// Exact source retained on failure.
    pub fn source_text(&self) -> &str {
        self.owner.source
    }
    /// Checked requested capacities retained on failure.
    pub fn requirements(&self) -> Requirements {
        self.owner.requirements
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
