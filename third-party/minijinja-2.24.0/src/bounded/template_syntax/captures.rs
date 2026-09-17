//! Reusable source-bound macro capture membership, without instruction ordering.
//!
//! A workspace cannot outlive its actual syntax owner:
//! ```compile_fail,E0505
//! use minijinja::bounded::template_syntax::{Plan as Syntax, WhitespaceConfig};
//! use minijinja::bounded::template_syntax::captures::Plan;
//! let owner = Syntax::inspect("{% macro m() %}{{ x }}{% endmacro %}", "x", WhitespaceConfig::default()).unwrap().construct().unwrap();
//! let workspace = Plan::for_macro(owner.macros().next().unwrap()).unwrap().construct().unwrap();
//! drop(owner);
//! let _ = workspace.capacities();
//! ```
//! A completed membership view prevents scratch reuse:
//! ```compile_fail,E0505
//! use minijinja::bounded::template_syntax::{Plan as Syntax, WhitespaceConfig};
//! use minijinja::bounded::template_syntax::captures::Plan;
//! let owner = Syntax::inspect("{% macro m() %}{{ x }}{% endmacro %}", "x", WhitespaceConfig::default()).unwrap().construct().unwrap();
//! let handle = owner.macros().next().unwrap();
//! let result = Plan::for_macro(handle).unwrap().construct().unwrap().analyze(handle).unwrap();
//! let mut names = result.names();
//! let _workspace = result.into_workspace();
//! assert_eq!(names.next(), Some("x"));
//! ```
//! A returned operation error still borrows its original syntax owner:
//! ```compile_fail
//! use minijinja::bounded::template_syntax::{Plan as Syntax, WhitespaceConfig};
//! use minijinja::bounded::template_syntax::captures::Plan;
//! let other = Syntax::inspect("{% macro m() %}{{ x }}{% endmacro %}", "x", WhitespaceConfig::default()).unwrap().construct().unwrap();
//! let failure = {
//!     let owner = Syntax::inspect("{% macro m() %}{{ x }}{% endmacro %}", "x", WhitespaceConfig::default()).unwrap().construct().unwrap();
//!     let workspace = Plan::for_macro(owner.macros().next().unwrap()).unwrap().construct().unwrap();
//!     workspace.analyze(other.macros().next().unwrap()).unwrap_err()
//! };
//! let _ = failure.capacities();
//! ```
//! Source IDs cannot be read or replaced by callers:
//! ```compile_fail,E0616
//! use minijinja::bounded::template_syntax::{Plan as Syntax, WhitespaceConfig};
//! let owner = Syntax::inspect("{% macro m() %}{{ x }}{% endmacro %}", "x", WhitespaceConfig::default()).unwrap().construct().unwrap();
//! let mut handle = owner.macros().next().unwrap();
//! handle.id = handle.id;
//! ```
#![forbid(unsafe_code)]
use super::{ParsedTemplate, store::StmtId};
use crate::bounded::expression::store::{OperandKind, RecordKind};
use crate::compiler::meta::{shared, view as walk};
use crate::compiler::parser::statements::storage::Statement;
use crate::compiler::tokens::Span;
use std::{
    alloc::Layout,
    collections::TryReserveError,
    fmt,
    mem::{size_of, size_of_val},
};

mod storage;
mod view;
use storage::Scratch;
use view::Packed;

/// One concrete reserve target. It grants no storage or source authority.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Buffer {
    /// Active scope-local assignments.
    Assigned,
    /// Assignment entry lengths at scope entry.
    Scopes,
    /// Unique output membership names.
    Captures,
    /// Explicit traversal continuations.
    Tasks,
}
impl Buffer {
    fn index(self) -> usize {
        match self {
            Self::Assigned => 0,
            Self::Scopes => 1,
            Self::Captures => 2,
            Self::Tasks => 3,
        }
    }
}

/// A checked source/count/layout planning refusal, before any reserve.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlanError {
    /// The requested concrete count or layout cannot be represented.
    Overflow,
    /// The private location does not identify a macro in this source.
    Source,
}
impl fmt::Display for PlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Overflow => "macro capture layout overflows",
            Self::Source => "macro capture source mismatch",
        })
    }
}
impl std::error::Error for PlanError {}

/// Fixed requested geometry, derived only from an actual closed syntax owner.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Requirements {
    counts: [usize; 4],
    bytes: [usize; 4],
    total: usize,
}
impl Requirements {
    /// Requested element capacity of one concrete buffer.
    pub fn capacity(self, buffer: Buffer) -> usize {
        self.counts[buffer.index()]
    }
    /// Requested allocation layout bytes for one buffer.
    pub fn buffer_bytes(self, buffer: Buffer) -> usize {
        self.bytes[buffer.index()]
    }
    /// Sum of the four requested allocation layouts; excludes fixed controls.
    pub const fn requested_bytes(self) -> usize {
        self.total
    }
}

/// A private macro location borrowing its exact immutable syntax owner.
#[derive(Clone, Copy)]
pub struct MacroRef<'o, 's> {
    source: &'o ParsedTemplate<'s>,
    id: StmtId,
}
impl<'o, 's> MacroRef<'o, 's> {
    pub(super) fn from_statement(source: &'o ParsedTemplate<'s>, id: StmtId) -> Self {
        Self { source, id }
    }

    /// Original macro or inline call-block declaration span.
    pub fn span(self) -> Span {
        match &self.source.storage.statements[self.id.0].value {
            Statement::CallBlock { macro_span, .. } => *macro_span,
            _ => self.source.storage.statements[self.id.0].span,
        }
    }
    /// Whether this is a call block's inline caller declaration.
    pub fn is_caller(self) -> bool {
        matches!(
            self.source.storage.statements[self.id.0].value,
            Statement::CallBlock { .. }
        )
    }
}
impl fmt::Debug for MacroRef<'_, '_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MacroRef")
            .field("span", &self.span())
            .field("is_caller", &self.is_caller())
            .finish()
    }
}
impl<'s> ParsedTemplate<'s> {
    /// Iterates actual macro declarations without allocation. Discovery order is
    /// private source-record order, not an instruction or closure lookup order.
    pub fn macros(&self) -> impl Iterator<Item = MacroRef<'_, 's>> {
        self.storage
            .statements
            .iter()
            .enumerate()
            .filter_map(|(i, record)| {
                matches!(
                    record.value,
                    Statement::Macro { .. } | Statement::CallBlock { .. }
                )
                .then_some(MacroRef {
                    source: self,
                    id: StmtId(i),
                })
            })
    }
}

/// Geometry and source binding for one reusable macro workspace.
pub struct Plan<'o, 's> {
    source: &'o ParsedTemplate<'s>,
    requirements: Requirements,
    #[cfg(feature = "development-closed-chat")]
    failure: Option<Buffer>,
}
fn add(value: &mut usize, extra: usize) -> Result<(), PlanError> {
    *value = value.checked_add(extra).ok_or(PlanError::Overflow)?;
    Ok(())
}
impl<'o, 's: 'o> Plan<'o, 's> {
    /// Derives all four capacities from this genuine macro's syntax owner.
    /// The same workspace may subsequently analyze any macro in that owner.
    pub fn for_macro(value: MacroRef<'o, 's>) -> Result<Self, PlanError> {
        if !matches!(
            value
                .source
                .storage
                .statements
                .get(value.id.0)
                .map(|r| &r.value),
            Some(Statement::Macro { .. } | Statement::CallBlock { .. })
        ) {
            return Err(PlanError::Source);
        }
        let mut counts = [0usize, 1, 0, 1];
        for node in &value.source.storage.expressions.nodes {
            let tasks = match node.kind {
                RecordKind::Var(_) => {
                    add(&mut counts[0], 1)?;
                    add(&mut counts[2], 1)?;
                    0
                }
                RecordKind::Const(_) => 0,
                RecordKind::Unary(..)
                | RecordKind::Attr(..)
                | RecordKind::List(..)
                | RecordKind::Map(..) => 1,
                RecordKind::Binary(..)
                | RecordKind::Compare(..)
                | RecordKind::Filter(..)
                | RecordKind::Test(..)
                | RecordKind::Item(..)
                | RecordKind::Call(..) => 2,
                RecordKind::If(..) | RecordKind::Slice { .. } => 3,
            };
            add(&mut counts[3], tasks)?;
        }
        for record in &value.source.storage.statements {
            let (assigned, scopes, tasks) = match record.value {
                Statement::Template(_) => (1, 0, 2),
                Statement::EmitExpr(_) => (0, 0, 1),
                Statement::EmitRaw(_) => (0, 0, 0),
                Statement::For { .. } => (1, 2, 10),
                Statement::If { .. } => (0, 2, 7),
                Statement::With(..) => (0, 1, 4),
                Statement::Set(..) => (0, 0, 2),
                Statement::SetBlock(..) => (0, 1, 4),
                Statement::AutoEscape(..) | Statement::FilterBlock(..) => (0, 1, 3),
                #[cfg(feature = "multi_template")]
                Statement::Block { .. } => (1, 1, 4),
                #[cfg(feature = "multi_template")]
                Statement::Extends(_) | Statement::Include(..) => (0, 0, 0),
                #[cfg(feature = "multi_template")]
                Statement::Import(..) | Statement::FromImport(..) => (0, 0, 1),
                // Includes the separate macro-entry task's four children.
                Statement::Macro { .. } => (2, 1, 8),
                Statement::CallBlock { .. } => (1, 1, 9),
                #[cfg(feature = "loop_controls")]
                Statement::Continue | Statement::Break => (0, 0, 0),
                Statement::Do(_) => (0, 0, 2),
            };
            add(&mut counts[0], assigned)?;
            add(&mut counts[1], scopes)?;
            add(&mut counts[3], tasks)?;
        }
        add(
            &mut counts[3],
            value
                .source
                .storage
                .links
                .len()
                .checked_mul(2)
                .ok_or(PlanError::Overflow)?,
        )?;
        add(
            &mut counts[3],
            value
                .source
                .storage
                .expressions
                .operands
                .len()
                .checked_mul(2)
                .ok_or(PlanError::Overflow)?,
        )?;
        for entry in &value.source.storage.expressions.operands {
            if matches!(entry.kind, OperandKind::Binding(..)) {
                add(&mut counts[3], 1)?;
            }
        }
        let requirements = Requirements::checked(counts)?;
        Ok(Self {
            source: value.source,
            requirements,
            #[cfg(feature = "development-closed-chat")]
            failure: None,
        })
    }
    /// Concrete source-derived requested capacities and layout sizes.
    pub const fn requirements(&self) -> Requirements {
        self.requirements
    }
    /// Selects a real capacity-overflow failure at one actual reserve site.
    /// Disabled in ordinary builds; cannot choose a successful allocation size.
    #[cfg(feature = "development-closed-chat")]
    pub fn fail_reservation(mut self, buffer: Buffer) -> Self {
        self.failure = Some(buffer);
        self
    }
    /// Reserves all four concrete arrays once, retaining every partial prefix on failure.
    pub fn construct(self) -> Result<Workspace<'o, 's>, ConstructError<'o, 's>> {
        let mut workspace = Workspace {
            source: self.source,
            requirements: self.requirements,
            scratch: Scratch::new(self.requirements.counts),
        };
        macro_rules! reserve {
            ($buffer:ident, $field:ident) => {{
                let requested = self.requirements.capacity(Buffer::$buffer);
                #[cfg(feature = "development-closed-chat")]
                let requested = if self.failure == Some(Buffer::$buffer) {
                    usize::MAX
                } else {
                    requested
                };
                if let Err(error) = workspace.scratch.$field.try_reserve_exact(requested) {
                    return Err(ConstructError {
                        workspace,
                        cause: Cause::Reserve {
                            buffer: Buffer::$buffer,
                            error,
                        },
                    });
                }
            }};
        }
        reserve!(Assigned, assigned);
        reserve!(Scopes, scopes);
        reserve!(Captures, captures);
        reserve!(Tasks, tasks);
        Ok(workspace)
    }
}

/// Actual construction or operation failure, retained without allocating an error wrapper.
#[derive(Debug)]
pub enum Cause {
    /// The original reserve error from the selected concrete buffer.
    Reserve {
        /// Actual target.
        buffer: Buffer,
        /// Original allocator/capacity error.
        error: TryReserveError,
    },
    /// A real append would exceed its private request or actual capacity.
    Capacity(Buffer),
    /// The macro belongs to a different syntax owner; prior state is intact.
    SourceMismatch,
    /// An internal operation is inapplicable to the closed macro-only profile.
    Geometry,
}
impl fmt::Display for Cause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Reserve { buffer, error } => {
                write!(f, "macro capture {buffer:?} reserve failed: {error}")
            }
            Self::Capacity(b) => write!(f, "macro capture {b:?} capacity exhausted"),
            Self::SourceMismatch => f.write_str("macro capture source owner differs"),
            Self::Geometry => f.write_str("macro capture geometry differs"),
        }
    }
}
impl std::error::Error for Cause {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Reserve { error, .. } => Some(error),
            _ => None,
        }
    }
}

/// Four owned reusable arrays borrowing exactly one immutable syntax owner.
pub struct Workspace<'o, 's: 'o> {
    source: &'o ParsedTemplate<'s>,
    requirements: Requirements,
    scratch: Scratch<'o, 's>,
}
impl fmt::Debug for Workspace<'_, '_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MacroWorkspace")
            .field("requirements", &self.requirements)
            .field("capacities", &self.capacities())
            .finish()
    }
}
impl<'o, 's: 'o> Workspace<'o, 's> {
    /// Original concrete requested layouts.
    pub const fn requirements(&self) -> Requirements {
        self.requirements
    }
    /// Actual retained element capacities, in Buffer order.
    pub fn capacities(&self) -> [usize; 4] {
        self.scratch.capacities()
    }
    /// Analyzes a genuine macro of this same source without another reserve.
    /// A foreign handle is rejected before clearing any previous state.
    pub fn analyze(
        mut self,
        value: MacroRef<'o, 's>,
    ) -> Result<Analysis<'o, 's>, AnalysisError<'o, 's>> {
        if !std::ptr::eq(self.source, value.source)
            || !matches!(
                self.source
                    .storage
                    .statements
                    .get(value.id.0)
                    .map(|r| &r.value),
                Some(Statement::Macro { .. } | Statement::CallBlock { .. })
            )
        {
            return Err(AnalysisError {
                workspace: self,
                cause: Cause::SourceMismatch,
            });
        }
        self.scratch.clear();
        let result = walk::Storage::push_scope(&mut self.scratch).and_then(|()| {
            shared::run(
                Packed(self.source),
                &mut self.scratch,
                walk::Task::Macro(value.id, false),
            )
        });
        match result {
            Ok(()) => Ok(Analysis { workspace: self }),
            Err(cause) => Err(AnalysisError {
                workspace: self,
                cause,
            }),
        }
    }
}

/// Completed capture membership. This is not a closure instruction-order policy.
pub struct Analysis<'o, 's: 'o> {
    workspace: Workspace<'o, 's>,
}
impl<'o, 's: 'o> Analysis<'o, 's> {
    /// Unique captured names borrowed only for this result-view lifetime.
    pub fn names<'a>(&'a self) -> impl ExactSizeIterator<Item = &'a str> + 'a {
        self.workspace.scratch.captures.iter().copied()
    }
    /// Tests membership without copying or changing lookup order elsewhere.
    pub fn contains(&self, name: &str) -> bool {
        self.workspace.scratch.captures.contains(&name)
    }
    /// Returns the same four allocations for another macro after all views end.
    pub fn into_workspace(self) -> Workspace<'o, 's> {
        self.workspace
    }
}
impl fmt::Debug for Analysis<'_, '_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MacroAnalysis")
            .field("names", &self.workspace.scratch.captures)
            .finish()
    }
}

/// Owns all completed reserves and the real cause of construction failure.
#[derive(Debug)]
pub struct ConstructError<'o, 's: 'o> {
    workspace: Workspace<'o, 's>,
    cause: Cause,
}
/// Owns all scratch/source borrows after an operation failure.
#[derive(Debug)]
pub struct AnalysisError<'o, 's: 'o> {
    workspace: Workspace<'o, 's>,
    cause: Cause,
}
macro_rules! error_accessors {
    ($ty:ident) => {
        impl $ty<'_, '_> {
            /// Original fixed or reserve cause.
            pub fn cause(&self) -> &Cause {
                &self.cause
            }
            /// Actual retained partial element capacities, in Buffer order.
            pub fn capacities(&self) -> [usize; 4] {
                self.workspace.capacities()
            }
        }
        impl fmt::Display for $ty<'_, '_> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                self.cause.fmt(f)
            }
        }
        impl std::error::Error for $ty<'_, '_> {
            fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
                Some(&self.cause)
            }
        }
    };
}
error_accessors!(ConstructError);
error_accessors!(AnalysisError);
impl<'o, 's: 'o> AnalysisError<'o, 's> {
    /// Recovers the already-reserved storage without discarding its source binding.
    pub fn into_workspace(self) -> Workspace<'o, 's> {
        self.workspace
    }
}

/// Concrete owned/control representations; requested array backing is separate.
/// This is not an original source grant or a whole-rendering allocation bound.
pub fn control_bytes() -> Option<usize> {
    // Named source-level controls across inspect, reserve, run and return.
    // Sum distinct phases conservatively; no compiler-stack size is inferred.
    let sizes = [
        size_of::<Plan<'static, 'static>>(),
        size_of::<Requirements>(),
        size_of::<Workspace<'static, 'static>>(),
        size_of::<Analysis<'static, 'static>>(),
        size_of::<ConstructError<'static, 'static>>(),
        size_of::<AnalysisError<'static, 'static>>(),
        size_of::<Result<Plan<'static, 'static>, PlanError>>(),
        size_of::<Result<Workspace<'static, 'static>, ConstructError<'static, 'static>>>(),
        size_of::<Result<Analysis<'static, 'static>, AnalysisError<'static, 'static>>>(),
        size_of::<MacroRef<'static, 'static>>(),
        size_of::<Cause>(),
        size_of::<PlanError>(),
        size_of::<Buffer>(),
        size_of::<TryReserveError>(),
        size_of::<Result<(), TryReserveError>>(),
        size_of::<Result<(), Cause>>(),
        size_of::<Result<(), PlanError>>(),
        size_of::<Result<Layout, PlanError>>(),
        size_of::<Result<usize, PlanError>>(),
        // Root argument, popped current task, and one child push argument.
        size_of::<walk::Task<'static, Packed<'static, 'static>>>(),
        size_of::<walk::Task<'static, Packed<'static, 'static>>>(),
        size_of::<walk::Task<'static, Packed<'static, 'static>>>(),
        size_of::<Option<walk::Task<'static, Packed<'static, 'static>>>>(),
        size_of::<walk::Expression<'static, Packed<'static, 'static>>>(),
        size_of::<walk::Statement<'static, Packed<'static, 'static>>>(),
        size_of::<walk::Item<'static, Packed<'static, 'static>>>(),
        size_of::<walk::MacroParts<'static, Packed<'static, 'static>>>(),
        size_of::<Option<(walk::Item<'static, Packed<'static, 'static>>, view::Cursor)>>(),
        size_of::<walk::SequenceKind>(),
        size_of::<view::Cursor>(),
        size_of::<Packed<'static, 'static>>(),
        size_of::<&mut Scratch<'static, 'static>>(),
        // Closed nested builder is unit; the false branch has no Vec/String.
        size_of::<()>(),
        size_of::<bool>(),
        size_of::<crate::bounded::expression::store::NodeId>(),
        size_of::<Option<crate::bounded::expression::store::NodeId>>(),
        size_of::<StmtId>(),
        size_of::<&str>(),
        // Planner counts, checked layout sizes and layout-construction result.
        size_of::<[usize; 4]>(),
        size_of::<[usize; 4]>(),
        size_of::<Layout>(),
        size_of::<Result<Layout, std::alloc::LayoutError>>(),
        size_of::<std::slice::Iter<'static, crate::bounded::expression::store::Record<'static>>>(),
        size_of::<std::slice::Iter<'static, super::store::Record<'static>>>(),
        size_of::<std::slice::Iter<'static, crate::bounded::expression::store::Operand<'static>>>(),
        size_of::<std::slice::Iter<'static, usize>>(),
        size_of::<&crate::bounded::expression::store::Record<'static>>(),
        size_of::<&super::store::Record<'static>>(),
        size_of::<&crate::bounded::expression::store::Operand<'static>>(),
        size_of::<(usize, usize, usize)>(), // per-statement assignment/scope/task increments
        size_of::<usize>(),                 // per-expression task increment
        size_of::<usize>(),                 // requested reserve count
        size_of::<usize>(),                 // checked layout total
        size_of::<&mut usize>(),            // checked accumulator borrow
        size_of::<usize>(),                 // checked increment
        size_of::<Option<usize>>(),         // checked arithmetic / cursor option
        size_of::<usize>(),                 // append length
        size_of::<usize>(),                 // append actual capacity
        size_of::<usize>(),                 // current scope start
        size_of::<Option<&usize>>(),        // scope last
    ];
    sizes
        .iter()
        .try_fold(size_of_val(&sizes), |sum, value| sum.checked_add(*value))
}

#[cfg(test)]
mod proof;
#[cfg(test)]
mod tests;

impl Requirements {
    fn checked(counts: [usize; 4]) -> Result<Self, PlanError> {
        let bytes = [
            Layout::array::<&str>(counts[0])
                .map_err(|_| PlanError::Overflow)?
                .size(),
            Layout::array::<usize>(counts[1])
                .map_err(|_| PlanError::Overflow)?
                .size(),
            Layout::array::<&str>(counts[2])
                .map_err(|_| PlanError::Overflow)?
                .size(),
            Layout::array::<walk::Task<'static, Packed<'static, 'static>>>(counts[3])
                .map_err(|_| PlanError::Overflow)?
                .size(),
        ];
        let total = bytes.iter().try_fold(0usize, |sum, value| {
            sum.checked_add(*value).ok_or(PlanError::Overflow)
        })?;
        Ok(Self {
            counts,
            bytes,
            total,
        })
    }
}

/// Preparse upper population for the same four arrays. Parser-produced nodes,
/// links and operands each fit the source event count; no new capture walker.
pub(in crate::bounded) fn compile_bound(nodes: usize, statements: usize) -> Option<usize> {
    let counts = [
        nodes.checked_add(statements.checked_mul(2)?)?,
        1usize.checked_add(statements.checked_mul(2)?)?,
        nodes,
        1usize
            .checked_add(nodes.checked_mul(8)?)?
            .checked_add(statements.checked_mul(12)?)?,
    ];
    Some(Requirements::checked(counts).ok()?.requested_bytes())
}

impl ConstructError<'_, '_> {
    pub(in crate::bounded) fn into_compile_cause(self) -> crate::bounded::source::CompileCause {
        match self.cause {
            Cause::Reserve { error, .. } => {
                crate::bounded::source::CompileCause::SyntaxReserve(error)
            }
            _ => crate::bounded::source::CompileCause::Source(
                crate::bounded::source::SourceError::Geometry,
            ),
        }
    }
}
