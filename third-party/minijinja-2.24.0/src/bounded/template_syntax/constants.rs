//! Closed constant-fold descriptors over an actual immutable syntax owner.
//!
//! Shared scalar arithmetic/comparisons and logical selection preserve ordinary Value semantics.
//! Text and direct literal lists use source-bound recipes and separately reserved payloads.
//! Maps and lazy iterable operations remain explicit owning refusals.
//! No general compiler, captured Value, original grant or chat activation is created.
//!
//! ```compile_fail
//! use minijinja::bounded::template_syntax::{Plan as Syntax, WhitespaceConfig};
//! use minijinja::bounded::template_syntax::constants::Plan;
//! let source = Syntax::inspect("{{ 1 }}", "x", WhitespaceConfig::default()).unwrap().construct().unwrap();
//! let workspace = Plan::for_expression(source.expressions().next().unwrap()).unwrap().construct().unwrap();
//! drop(source);
//! println!("{}", workspace.capacity());
//! ```
//!
//! ```compile_fail
//! use minijinja::bounded::template_syntax::{Plan as Syntax, WhitespaceConfig};
//! use minijinja::bounded::template_syntax::constants::{Plan, Value};
//! let source = Syntax::inspect("{{ 'a' 'b' }}", "x", WhitespaceConfig::default()).unwrap().construct().unwrap();
//! let expr = source.expressions().next().unwrap();
//! let result = Plan::for_expression(expr).unwrap().construct().unwrap().fold(expr).unwrap();
//! let Some(Value::Text(text)) = result.value() else { panic!() };
//! let _workspace = result.into_workspace();
//! println!("{}", text.byte_len());
//! ```
//!
//! ```compile_fail
//! use minijinja::bounded::template_syntax::{Plan as Syntax, WhitespaceConfig};
//! use minijinja::bounded::template_syntax::constants::Plan;
//! let other = Syntax::inspect("{{ 1 }}", "x", WhitespaceConfig::default()).unwrap().construct().unwrap();
//! let failure = {
//!     let source = Syntax::inspect("{{ 1 }}", "x", WhitespaceConfig::default()).unwrap().construct().unwrap();
//!     Plan::for_expression(source.expressions().next().unwrap()).unwrap().construct().unwrap().fold(other.expressions().next().unwrap()).unwrap_err()
//! };
//! println!("{}", failure.capacity());
//! ```
//!
//! ```compile_fail
//! use minijinja::bounded::template_syntax::{Plan as Syntax, WhitespaceConfig};
//! let source = Syntax::inspect("{{ 1 }}", "x", WhitespaceConfig::default()).unwrap().construct().unwrap();
//! let mut expr = source.expressions().next().unwrap();
//! expr.id = expr.id;
//! ```
//! A planned payload cannot outlive its source:
//! ```compile_fail
//! use minijinja::bounded::template_syntax::{Plan as Syntax, WhitespaceConfig};
//! use minijinja::bounded::template_syntax::constants::Plan;
//! let source = Syntax::inspect("{{ 'a'~[1] }}", "x", WhitespaceConfig::default()).unwrap().construct().unwrap();
//! let expr = source.expressions().last().unwrap();
//! let planned = Plan::for_expression(expr).unwrap().construct().unwrap().prepare(expr).unwrap();
//! drop(source);
//! let _ = planned.materialize();
//! ```
//! A borrowed list prevents consuming its payload owner:
//! ```compile_fail
//! use minijinja::bounded::template_syntax::{Plan as Syntax, WhitespaceConfig};
//! use minijinja::bounded::template_syntax::constants::{Plan, Value};
//! let source = Syntax::inspect("{{ [1,'a'] }}", "x", WhitespaceConfig::default()).unwrap().construct().unwrap();
//! let expr = source.expressions().last().unwrap();
//! let result = Plan::for_expression(expr).unwrap().construct().unwrap().fold(expr).unwrap();
//! let Some(Value::List(list)) = result.value() else { panic!() };
//! let _ = result.into_workspace();
//! println!("{}", list.len());
//! ```
#![forbid(unsafe_code)]
use super::ParsedTemplate;
use crate::bounded::expression::store::{NodeId, SegmentKind, Sequence};
use crate::compiler::fold::{self, Continuation, View};
use crate::compiler::tokens::Span;
use std::{alloc::Layout, collections::TryReserveError, fmt, mem::size_of};
mod materialize;
mod reader;
mod storage;
use materialize::{Recipe, RecipeId, TextDescriptor};
mod view;
use view::Packed;

/// A folded scalar's exact representation, without a Value allocation.
#[derive(Clone, Copy, Debug)]
pub enum Scalar {
    /// The None singleton.
    None,
    /// A boolean, distinct from integers.
    Bool(bool),
    /// An unsigned 64-bit literal.
    U64(u64),
    /// An unsigned 128-bit literal, including the ordinary negation special case.
    U128(u128),
    /// A signed narrow operation result.
    I64(i64),
    /// A signed wide operation result.
    I128(i128),
    /// Floating point; payload/sign bits are preserved.
    F64(f64),
}
#[derive(Clone, Copy)]
enum Descriptor {
    Scalar(Scalar),
    Text(TextDescriptor),
    List(RecipeId),
}
/// A collection operation whose ordinary ownership remains unfinished here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Collection {
    /// Lazy list/iterable addition or repetition; immediate literal lists are supported.
    List,
    /// Direct-constant map, including empty.
    Map,
}
/// Fixed planning failure before the reserve.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlanError {
    /// The concrete count or allocation layout cannot be represented.
    Overflow,
    /// The private handle is not an expression in this syntax owner.
    Source,
}
impl fmt::Display for PlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "constant-fold plan {self:?}")
    }
}
impl std::error::Error for PlanError {}
/// A private expression location borrowing its exact closed syntax object.
#[derive(Clone, Copy)]
pub struct ExpressionRef<'o, 's> {
    source: &'o ParsedTemplate<'s>,
    id: NodeId,
}
impl ExpressionRef<'_, '_> {
    /// Original expression span.
    pub fn span(self) -> Span {
        self.source.storage.expressions.nodes[self.id.0].span
    }
}
impl fmt::Debug for ExpressionRef<'_, '_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExpressionRef")
            .field("span", &self.span())
            .finish()
    }
}
impl<'s> ParsedTemplate<'s> {
    /// Genuine expression locations in private construction order, including intermediates.
    pub fn expressions(&self) -> impl Iterator<Item = ExpressionRef<'_, 's>> {
        (0..self.storage.expressions.nodes.len()).map(|i| ExpressionRef {
            source: self,
            id: NodeId(i),
        })
    }
}
/// Requested geometry, not caller storage authority.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Requirements {
    nodes: usize,
    frames: usize,
    bytes: usize,
    recipes: usize,
    operands: usize,
}
impl Requirements {
    /// Actual successful source expression records, including conservative unused records.
    pub fn expressions(self) -> usize {
        self.nodes
    }
    /// Checked continuation capacity.
    pub fn frames(self) -> usize {
        self.frames
    }
    /// Maximum recipe records, derived from actual expression nodes.
    pub fn recipes(self) -> usize {
        self.recipes
    }
    /// Requested metadata-array layouts, excluding allocator rounding and controls.
    pub fn requested_bytes(self) -> usize {
        self.bytes
    }
    /// Concrete feature-dependent plan shell.
    pub fn plan_control_bytes(self) -> usize {
        size_of::<Plan<'static, 'static>>()
    }
    /// Largest concrete retained workspace/result/error shell.
    pub fn retained_control_bytes(self) -> usize {
        size_of::<Workspace<'static, 'static>>()
            .max(size_of::<Completed<'static, 'static>>())
            .max(size_of::<Failure<'static, 'static>>())
            .max(size_of::<Planned<'static, 'static>>())
    }
    /// Named worker/control shells, not a whole-thread native stack bound.
    pub fn worker_control_bytes(self) -> usize {
        fold::control_bytes::<Packed<'static, 'static>, Workspace<'static, 'static>>()
            + size_of::<ExpressionRef<'static, 'static>>()
            + size_of::<Segments<'static, 'static>>()
            + crate::value::primitive::scalar::control_bytes()
            + reader::control_bytes()
            + size_of::<Recipe>()
            + size_of::<MaterializationRequirements>()
    }
}
fn requirements(nodes: usize, operands: usize) -> Result<Requirements, PlanError> {
    let frames = nodes.checked_add(1).ok_or(PlanError::Overflow)?;
    let bytes = Layout::array::<Continuation<Packed<'static, 'static>, Descriptor>>(frames)
        .map_err(|_| PlanError::Overflow)?
        .size()
        .checked_add(
            Layout::array::<Recipe>(nodes)
                .map_err(|_| PlanError::Overflow)?
                .size(),
        )
        .ok_or(PlanError::Overflow)?;
    Ok(Requirements {
        nodes,
        frames,
        bytes,
        recipes: nodes,
        operands,
    })
}
/// Source-bound checked geometry for one once-reserved fold workspace.
pub struct Plan<'o, 's> {
    source: &'o ParsedTemplate<'s>,
    requirements: Requirements,
    #[cfg(feature = "development-closed-chat")]
    failure: Option<Buffer>,
}
impl<'o, 's: 'o> Plan<'o, 's> {
    /// Derives capacity from this genuine expression's immutable syntax owner.
    pub fn for_expression(expr: ExpressionRef<'o, 's>) -> Result<Self, PlanError> {
        if expr
            .source
            .storage
            .expressions
            .nodes
            .get(expr.id.0)
            .is_none()
        {
            return Err(PlanError::Source);
        }
        Ok(Self {
            source: expr.source,
            requirements: requirements(
                expr.source.expression_count(),
                expr.source.storage.expressions.operands.len(),
            )?,
            #[cfg(feature = "development-closed-chat")]
            failure: None,
        })
    }
    /// Exact requested layout diagnostics.
    pub fn requirements(&self) -> Requirements {
        self.requirements
    }
    /// Development-only fixed failure selection at the actual continuation reserve.
    #[cfg(feature = "development-closed-chat")]
    pub fn fail_reservation(mut self) -> Self {
        self.failure = Some(Buffer::Frames);
        self
    }
    /// Select a real metadata or payload reserve for development failure injection.
    #[cfg(feature = "development-closed-chat")]
    pub fn fail_buffer(mut self, buffer: Buffer) -> Self {
        self.failure = Some(buffer);
        self
    }
    /// Reserve the actual metadata destinations, preserving real failure and source.
    pub fn construct(self) -> Result<Workspace<'o, 's>, Failure<'o, 's>> {
        #[cfg(feature = "development-closed-chat")]
        let fail = self.failure;
        #[cfg(not(feature = "development-closed-chat"))]
        let fail = None;
        self.construct_inner(fail)
    }
    fn construct_inner(self, fail: Option<Buffer>) -> Result<Workspace<'o, 's>, Failure<'o, 's>> {
        let mut owner = Workspace {
            source: self.source,
            requirements: self.requirements,
            limit: self.requirements.frames,
            frames: Vec::new(),
            pending: None,
            operation: None,
            collection: None,
            value: None,
            recipes: Vec::new(),
            bytes: Vec::new(),
            items: Vec::new(),
            payload: MaterializationRequirements::default(),
            reserve_failure: fail,
            pending_recipe: None,
            #[cfg(test)]
            peak: 0,
            #[cfg(test)]
            pushes: 0,
        };
        if let Err(error) =
            owner
                .frames
                .try_reserve_exact(request(fail, Buffer::Frames, owner.requirements.frames))
        {
            return Err(Failure {
                owner,
                cause: Cause::Reserve(error),
            });
        }
        if let Err(error) = owner.recipes.try_reserve_exact(request(
            fail,
            Buffer::Recipes,
            owner.requirements.recipes,
        )) {
            return Err(Failure {
                owner,
                cause: Cause::Reserve(error),
            });
        }
        Ok(owner)
    }
}
impl fmt::Debug for Plan<'_, '_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Plan")
            .field("requirements", &self.requirements)
            .finish()
    }
}
/// Actual failure cause, distinguishing unfinished work from semantic nonconstancy.
#[derive(Debug)]
pub enum Cause {
    /// Actual reserve failure at the real destination.
    Reserve(TryReserveError),
    /// An equal-content or other foreign syntax owner cannot reuse this workspace.
    Source,
    /// The source-derived logical or real destination capacity was reached.
    Capacity,
    /// Checked text/item count or concrete payload layout cannot be represented.
    Overflow,
    /// The reached operator has no closed ownership implementation yet.
    NeedsOperation(&'static str),
    /// The reached ordinary materializer has no closed ownership implementation yet.
    NeedsMaterialization(Collection),
}
/// Actual destination selected for development-only failure injection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Buffer {
    /// Shared fold continuation records.
    Frames,
    /// Flat text/list recipes.
    Recipes,
    /// Generated UTF-8 payload.
    Bytes,
    /// Direct-list leaf descriptors.
    Items,
}
fn request(fail: Option<Buffer>, selected: Buffer, count: usize) -> usize {
    if fail == Some(selected) {
        usize::MAX
    } else {
        count
    }
}
/// Payload geometry discovered by the actual shared fold after metadata admission.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MaterializationRequirements {
    bytes: usize,
    items: usize,
    requested: usize,
}
impl MaterializationRequirements {
    /// Sum of all generated text recipe byte lengths, including intermediates.
    pub fn text_bytes(self) -> usize {
        self.bytes
    }
    /// Sum of all direct-list leaf slots, including intermediate lists.
    pub fn items(self) -> usize {
        self.items
    }
    /// Checked concrete byte and descriptor allocation layouts.
    pub fn requested_bytes(self) -> usize {
        self.requested
    }
}
/// Reusable four-array owner; metadata planning precedes separate payload reservation.
pub struct Workspace<'o, 's> {
    source: &'o ParsedTemplate<'s>,
    requirements: Requirements,
    limit: usize,
    frames: Vec<Continuation<Packed<'o, 's>, Descriptor>>,
    pending: Option<Continuation<Packed<'o, 's>, Descriptor>>,
    operation: Option<(Descriptor, Descriptor)>,
    collection: Option<(Sequence, Option<Sequence>)>,
    value: Option<Descriptor>,
    recipes: Vec<Recipe>,
    bytes: Vec<u8>,
    items: Vec<Descriptor>,
    payload: MaterializationRequirements,
    reserve_failure: Option<Buffer>,
    pending_recipe: Option<Recipe>,
    #[cfg(test)]
    peak: usize,
    #[cfg(test)]
    pushes: usize,
}
impl<'o, 's: 'o> Workspace<'o, 's> {
    /// Requested concrete geometry.
    pub fn requirements(&self) -> Requirements {
        self.requirements
    }
    /// Actual reserved element capacity; excess capacity never enlarges the logical bound.
    pub fn capacity(&self) -> usize {
        self.frames.capacity()
    }
    /// Actual capacities in Frames, Recipes, Bytes, Items order.
    pub fn capacities(&self) -> [usize; 4] {
        [
            self.frames.capacity(),
            self.recipes.capacity(),
            self.bytes.capacity(),
            self.items.capacity(),
        ]
    }
    /// Actual retained Vec allocation-layout total, separate from source-derived requested bounds.
    pub fn retained_layout_bytes(&self) -> Option<usize> {
        self.frames
            .capacity()
            .checked_mul(size_of::<Continuation<Packed<'o, 's>, Descriptor>>())?
            .checked_add(self.recipes.capacity().checked_mul(size_of::<Recipe>())?)?
            .checked_add(self.bytes.capacity())?
            .checked_add(self.items.capacity().checked_mul(size_of::<Descriptor>())?)
    }
    /// Compose source-bound metadata planning with separately reserved payload construction.
    pub fn fold(self, expr: ExpressionRef<'o, 's>) -> Result<Completed<'o, 's>, Failure<'o, 's>> {
        self.prepare(expr)?.materialize()
    }
    /// Plan one genuine expression in the same owner; mismatch preserves previous state.
    pub fn prepare(
        mut self,
        expr: ExpressionRef<'o, 's>,
    ) -> Result<Planned<'o, 's>, Failure<'o, 's>> {
        if !std::ptr::eq(self.source, expr.source)
            || self
                .source
                .storage
                .expressions
                .nodes
                .get(expr.id.0)
                .is_none()
        {
            return Err(Failure {
                owner: self,
                cause: Cause::Source,
            });
        }
        self.frames.clear();
        self.recipes.clear();
        self.bytes.clear();
        self.items.clear();
        self.payload = MaterializationRequirements::default();
        self.pending_recipe = None;
        self.pending = None;
        self.operation = None;
        self.collection = None;
        self.value = None;
        #[cfg(test)]
        {
            self.peak = 0;
            self.pushes = 0;
        }
        let source = Packed(self.source);
        match fold::run(source, &mut self, source.node(expr.id)) {
            Ok(value) => {
                self.value = value;
                Ok(Planned { owner: self })
            }
            Err(cause) => Err(Failure { owner: self, cause }),
        }
    }
}
impl fmt::Debug for Workspace<'_, '_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Workspace")
            .field("requirements", &self.requirements)
            .field("capacity", &self.frames.capacity())
            .field("pending_frames", &self.frames.len())
            .finish()
    }
}
/// Planned value and exact payload demand, retaining all metadata and source custody.
#[derive(Debug)]
pub struct Planned<'o, 's> {
    owner: Workspace<'o, 's>,
}
impl<'o, 's: 'o> Planned<'o, 's> {
    /// Actual semantic payload geometry; no caller-selected size authorizes it.
    pub fn requirements(&self) -> MaterializationRequirements {
        self.owner.payload
    }
    /// Actual retained capacities before payload construction.
    pub fn capacities(&self) -> [usize; 4] {
        self.owner.capacities()
    }
    /// Reserve both real payload destinations before filling either.
    pub fn materialize(mut self) -> Result<Completed<'o, 's>, Failure<'o, 's>> {
        let fail = self.owner.reserve_failure;
        // Reuse preserves existing allocation, but injection still reaches the selected real reserve.
        let bytes = self
            .owner
            .payload
            .bytes
            .saturating_sub(self.owner.bytes.len());
        if let Err(error) = self
            .owner
            .bytes
            .try_reserve_exact(request(fail, Buffer::Bytes, bytes))
        {
            return Err(Failure {
                owner: self.owner,
                cause: Cause::Reserve(error),
            });
        }
        let items = self
            .owner
            .payload
            .items
            .saturating_sub(self.owner.items.len());
        if let Err(error) = self
            .owner
            .items
            .try_reserve_exact(request(fail, Buffer::Items, items))
        {
            return Err(Failure {
                owner: self.owner,
                cause: Cause::Reserve(error),
            });
        }
        match self.owner.fill() {
            Ok(()) => Ok(Completed { owner: self.owner }),
            Err(cause) => Err(Failure {
                owner: self.owner,
                cause,
            }),
        }
    }
    /// Recover the same workspace without constructing payloads.
    pub fn into_workspace(self) -> Workspace<'o, 's> {
        self.owner
    }
}
/// Completed value or actual semantic nonconstancy, retaining the reusable owner.
#[derive(Debug)]
pub struct Completed<'o, 's> {
    owner: Workspace<'o, 's>,
}
impl<'o, 's: 'o> Completed<'o, 's> {
    /// None is true ordinary nonconstancy, never an unfinished operator/materializer.
    pub fn value(&self) -> Option<Value<'_, 's>> {
        self.owner.value.map(|v| self.owner.value_view(v))
    }
    /// Return the same storage after all result views have ended.
    pub fn into_workspace(self) -> Workspace<'o, 's> {
        self.owner
    }
}
/// Borrowed result view retaining source segments or separately reserved generated payload.
pub enum Value<'a, 's> {
    /// Exact inline primitive.
    Scalar(Scalar),
    /// Source-owned joined segments or a generated UTF-8 range in the retained payload.
    Text(Text<'a, 's>),
    /// An actual direct-literal list borrowing retained leaf slots.
    List(List<'a, 's>),
}
/// Borrowed text; source-only text remains segmented and generated text is a retained byte range.
pub struct Text<'a, 's> {
    source: &'a ParsedTemplate<'s>,
    text: TextDescriptor,
    generated: &'a [u8],
}
impl<'a, 's: 'a> Text<'a, 's> {
    /// Exact checked UTF-8 byte count.
    pub fn byte_len(&self) -> usize {
        self.text.bytes
    }
    /// Source segments or the one generated UTF-8 range; never joins implicitly.
    pub fn segments(&self) -> Segments<'_, 's> {
        Segments {
            source: self.source,
            next: match self.text.kind {
                materialize::TextKind::Source(x) => x.sequence.head,
                _ => None,
            },
            generated: match self.text.kind {
                materialize::TextKind::Generated(_) => {
                    Some(std::str::from_utf8(self.generated).expect("private generated UTF-8"))
                }
                _ => None,
            },
        }
    }
}
/// Borrowed direct-literal list with no external Value or callback.
pub struct List<'a, 's> {
    source: &'a ParsedTemplate<'s>,
    items: &'a [Descriptor],
}
impl<'a, 's: 'a> List<'a, 's> {
    /// Actual retained leaf count.
    pub fn len(&self) -> usize {
        self.items.len()
    }
    /// Whether this list is empty.
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
    /// Borrow one immediate scalar or source-text literal.
    pub fn get(&self, index: usize) -> Option<Value<'_, 's>> {
        self.items.get(index).map(|value| match *value {
            Descriptor::Scalar(x) => Value::Scalar(x),
            Descriptor::Text(text) => Value::Text(Text {
                source: self.source,
                text,
                generated: &[],
            }),
            Descriptor::List(_) => unreachable!("direct literals never contain a list"),
        })
    }
}
/// Borrowed flat segment traversal.
pub struct Segments<'a, 's> {
    source: &'a ParsedTemplate<'s>,
    next: Option<usize>,
    generated: Option<&'a str>,
}
impl<'a, 's: 'a> Iterator for Segments<'a, 's> {
    type Item = &'a str;
    fn next(&mut self) -> Option<Self::Item> {
        if let Some(text) = self.generated.take() {
            return Some(text);
        }
        let record = self
            .source
            .storage
            .expressions
            .segments
            .get(self.next?)
            .expect("private validated text segment");
        self.next = record.next;
        Some(match record.kind {
            SegmentKind::Source(text) => text,
            SegmentKind::Decoded(bytes) => std::str::from_utf8(
                self.source
                    .buffers
                    .literals
                    .get(bytes.start..bytes.end)
                    .expect("private validated literal range"),
            )
            .expect("private validated UTF-8 literal"),
        })
    }
}
/// Owning failure; the same source, actual array and partial controls stay retained.
#[derive(Debug)]
pub struct Failure<'o, 's> {
    owner: Workspace<'o, 's>,
    cause: Cause,
}
impl<'o, 's: 'o> Failure<'o, 's> {
    /// Actual cause, including the original reserve error.
    pub fn cause(&self) -> &Cause {
        &self.cause
    }
    /// Actual retained element capacity.
    pub fn capacity(&self) -> usize {
        self.owner.frames.capacity()
    }
    /// Actual capacities in Frames, Recipes, Bytes, Items order.
    pub fn capacities(&self) -> [usize; 4] {
        self.owner.capacities()
    }
    /// Return the same source-bound storage for subsequent use.
    pub fn into_workspace(self) -> Workspace<'o, 's> {
        self.owner
    }
}
impl fmt::Display for Failure<'_, '_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "constant-fold failure: {:?}", self.cause)
    }
}
impl std::error::Error for Failure<'_, '_> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        if let Cause::Reserve(e) = &self.cause {
            Some(e)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests;
