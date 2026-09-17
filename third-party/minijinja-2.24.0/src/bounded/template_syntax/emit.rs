//! Packed syntax adapter for the ordinary jump protocol and existing shared VM.
//! The explicit continuation vector replaces recursive compiler traversal.
#![forbid(unsafe_code)]
use super::{ParsedTemplate, store::StmtId};
use crate::bounded::expression::store::{
    NodeId, Operand, OperandKind, RecordKind, Scalar, SegmentKind, Sequence,
};
use crate::bounded::source::{
    Instruction, Location, MappingProjection, Range, SourceError, Span as OutputSpan,
};
use crate::compiler::parser::statements::storage::Statement;
use crate::compiler::{
    ast::{BinOpKind, CompareOpKind, UnaryOpKind},
    codegen::patch::{self, Emitter, Jump},
    tokens::Span,
};
use crate::vm::shared::Order;
use std::{alloc::Layout, mem::size_of};
mod macros;
mod json;

#[derive(Clone, Copy)]
enum Action {
    Body(Sequence),
    Statement(StmtId),
    Expression(NodeId),
    Assign(NodeId, Span),
    KeywordArguments(Sequence, Span),
    OrderedOperands(Sequence, Span),
    MappingEntries(Sequence, Sequence, Span),
    MacroExpression(StmtId),
    MacroParameters(StmtId, usize),
    MacroDefault(u32, Range, StmtId, usize, Span),
    MacroEnd(macros::Declaration),
    Emit(Instruction, Span),
    Conditional(NodeId, Option<NodeId>, Span),
    ConditionalElse(u32, Option<NodeId>, usize, Span),
    ConditionalEnd(u32, usize),
    If(Sequence, Sequence, Span),
    Else(u32, Sequence, Span),
    End(u32),
    Loop(NodeId, Sequence, Span),
    FilterLoop(NodeId, NodeId, Sequence, Span),
    FilterTest(NodeId, Sequence, u32, usize, Span),
    EndLoop(u32, usize, Span),
    ShortCircuit(Jump, NodeId, Span),
}
#[derive(Clone, Copy)]
pub(in crate::bounded) struct Requirements {
    pub instructions: usize,
    pub actions: usize,
    pub heap: usize,
    pub controls: usize,
}
pub(in crate::bounded) fn requirements(
    nodes: usize,
    statements: usize,
) -> Result<Requirements, SourceError> {
    // Source nodes cover ordinary expressions plus macro formal/default,
    // capture, signature and call instructions. A formal can emit its name,
    // duplicate/presence/jump/discard/default/store protocol; statements add
    // their declaration/return or loop/assignment protocol. Every continuation
    // expansion is covered by four actions per distinct syntax node. Filtered
    // loops add a bounded collection pass before the ordinary body loop; the
    // statement instruction allowance covers both passes. A JSON
    // filter's two own actions plus each actual option/value child are covered
    // by those source nodes; absent options emit no synthetic child values.
    let instructions = nodes
        .checked_mul(8)
        .and_then(|n| statements.checked_mul(18)?.checked_add(n))
        .ok_or(SourceError::Overflow)?;
    u32::try_from(instructions).map_err(|_| SourceError::Overflow)?;
    let actions = nodes
        .checked_add(statements)
        .and_then(|n| n.checked_mul(4))
        .and_then(|n| n.checked_add(1))
        .ok_or(SourceError::Overflow)?;
    let mut heap = Layout::array::<Action>(actions)
        .map_err(|_| SourceError::Overflow)?
        .size();
    #[cfg(feature = "macros")]
    {
        heap = heap
            .checked_add(
                super::captures::compile_bound(nodes, statements).ok_or(SourceError::Overflow)?,
            )
            .ok_or(SourceError::Overflow)?;
    }
    let controls = [
        size_of::<Compiler<'static, 'static, 'static>>(),
        size_of::<Requirements>(),
        size_of::<Action>(),
        size_of::<Option<Action>>(),
        size_of::<Sequence>(),
        size_of::<crate::bounded::source::Geometry>(),
        size_of::<Result<crate::bounded::source::Geometry, crate::bounded::source::CompileCause>>(),
        size_of::<(usize, usize, usize, Option<usize>, NodeId)>(),
        size_of::<NodeId>(),
        size_of::<StmtId>(),
        size_of::<Instruction>(),
        size_of::<Span>(),
        size_of::<Range>(),
        size_of::<Jump>(),
        size_of::<(BinOpKind, CompareOpKind, Order)>(),
        size_of::<crate::value::primitive::type_tests::TypeTest>(),
        size_of::<Option<crate::value::primitive::type_tests::TypeTest>>(),
        size_of::<&str>(),
        size_of::<Result<u32, SourceError>>(),
        size_of::<Result<(), SourceError>>(),
        json::control_bytes(),
        // Default/replace's exact two-argument compiler destination and traversal.
        size_of::<[Option<NodeId>; 2]>(),
        size_of::<Option<usize>>(),
        size_of::<Option<&Operand<'static>>>(),
        size_of::<std::iter::Take<std::slice::IterMut<'static, Option<NodeId>>>>(),
        size_of::<std::iter::Rev<std::iter::Flatten<std::array::IntoIter<Option<NodeId>, 2>>>>(),
        // Character trim/join/replace's borrowed argument and atom-profile scan controls.
        size_of::<(
            NodeId,
            NodeId,
            &Operand<'static>,
            &RecordKind<'static>,
            bool,
        )>(),
    ]
    .into_iter()
    .try_fold(0usize, usize::checked_add)
    .ok_or(SourceError::Overflow)?;
    let controls = controls
        .checked_add(macros::control_bytes().ok_or(SourceError::Overflow)?)
        .ok_or(SourceError::Overflow)?;
    Ok(Requirements {
        instructions,
        actions,
        heap,
        controls,
    })
}
struct Compiler<'o, 's, 'w> {
    source: &'o ParsedTemplate<'s>,
    instructions: &'w mut Vec<Instruction>,
    bytes: &'w mut Vec<u8>,
    locations: &'w mut Vec<Location>,
    actions: Vec<Action>,
    limits: Requirements,
    text_limit: usize,
    span: Span,
    depth: usize,
    loop_depth: usize,
    locals_depth: usize,
    in_macro: bool,
    capture_failure: Option<crate::bounded::source::CompileCause>,
    geometry: crate::bounded::source::Geometry,
}
impl Emitter for Compiler<'_, '_, '_> {
    type Error = SourceError;
    fn position(&self) -> u32 {
        self.instructions.len() as u32
    }
    fn jump(&mut self, kind: Jump) -> Result<u32, SourceError> {
        self.op(
            match kind {
                Jump::Always => Instruction::Jump(!0),
                Jump::IfFalse => Instruction::JumpIfFalse(!0),
                Jump::IfFalseOrPop => Instruction::JumpIfFalseOrPop(!0),
                Jump::IfTrueOrPop => Instruction::JumpIfTrueOrPop(!0),
            },
            self.span,
        )
    }
    fn patch(&mut self, at: u32, target: u32) -> Result<(), SourceError> {
        if target as usize > self.instructions.len() {
            return Err(SourceError::Geometry);
        }
        match self.instructions.get_mut(at as usize) {
            Some(
                Instruction::Jump(value)
                | Instruction::JumpIfFalse(value)
                | Instruction::JumpIfFalseOrPop(value)
                | Instruction::JumpIfTrueOrPop(value)
                | Instruction::Iterate(value),
            ) => {
                *value = target;
                Ok(())
            }
            _ => Err(SourceError::Geometry),
        }
    }
}
impl<'o, 's, 'w> Compiler<'o, 's, 'w> {
    fn push(&mut self, action: Action) -> Result<(), SourceError> {
        if self.actions.len() == self.limits.actions
            || self.actions.len() == self.actions.capacity()
        {
            return Err(SourceError::Geometry);
        }
        self.actions.push(action);
        Ok(())
    }
    fn intern(&mut self, text: &str) -> Result<Range, SourceError> {
        if text.is_empty() {
            return Ok(Range::new(0, 0));
        }
        if let Some(start) = self
            .bytes
            .windows(text.len())
            .position(|v| v == text.as_bytes())
        {
            return Ok(Range::new(start, text.len()));
        }
        self.text(text)
    }
    fn text(&mut self, text: &str) -> Result<Range, SourceError> {
        let start = self.bytes.len();
        let end = start.checked_add(text.len()).ok_or(SourceError::Overflow)?;
        if end > self.text_limit || end > self.bytes.capacity() {
            return Err(SourceError::Geometry);
        }
        self.bytes.extend_from_slice(text.as_bytes());
        Ok(Range::new(start, text.len()))
    }
    fn op(&mut self, op: Instruction, span: Span) -> Result<u32, SourceError> {
        let (need, delta) = match op {
            Instruction::DupTop | Instruction::GetClosure | Instruction::Arguments { .. } => (0, 1),
            Instruction::Argument(_) | Instruction::Enclose(_) | Instruction::Return => (0, 0),
            Instruction::BuildMacro(..) => (2, -1),
            Instruction::DiscardTop => (1, -1),
            Instruction::Lookup(_)
            | Instruction::Literal(_)
            | Instruction::Undefined
            | Instruction::Zero
            | Instruction::Scalar(_)
            | Instruction::EmptySequence
            | Instruction::Iterate(_)
            | Instruction::EndCapture
            | Instruction::EndCollection => (0, 1isize),
            Instruction::BeginCapture => (0,0),
            Instruction::BeginCollection => (1,0),
            Instruction::BuildMap(count) => {let n=count.checked_mul(2).ok_or(SourceError::Overflow)?;(n,1isize.checked_sub(isize::try_from(n).map_err(|_|SourceError::Overflow)?).ok_or(SourceError::Overflow)?)},
            Instruction::BuildList(count) => (count, 1isize.checked_sub(isize::try_from(count).map_err(|_|SourceError::Overflow)?).ok_or(SourceError::Overflow)?),
            Instruction::UnpackList(count) => (
                1,
                isize::try_from(count).map_err(|_| SourceError::Overflow)? - 1,
            ),
            Instruction::SetAttr(_) => (2, -2),
            Instruction::BuildKwargs(count) => {
                let count = count.checked_mul(2).ok_or(SourceError::Overflow)?;
                (
                    count,
                    1 - isize::try_from(count).map_err(|_| SourceError::Overflow)?,
                )
            }
            Instruction::CallFunction(_, count) => (
                usize::from(count),
                1 - isize::try_from(count).map_err(|_| SourceError::Overflow)?,
            ),
            Instruction::StoreLocal(_)
            | Instruction::Emit
            | Instruction::AppendCollection
            | Instruction::PushLoop(_)
            | Instruction::JumpIfFalse(_)
            | Instruction::JumpIfFalseOrPop(_)
            | Instruction::JumpIfTrueOrPop(_) => (1, -1),
            Instruction::GetAttr(_)
            | Instruction::Not
            | Instruction::Neg
            | Instruction::Presence(_)
            | Instruction::TypeTest(_)
            | Instruction::Trim
            | Instruction::Upper
            | Instruction::DictSort
            | Instruction::Length
            | Instruction::Stringify
            | Instruction::Floatify
            | Instruction::MappingView(_)
            | Instruction::Items
            | Instruction::List => (1, 0),
            Instruction::GetItem
            | Instruction::Add
            | Instruction::StringConcat
            | Instruction::Arithmetic(_)
            | Instruction::Ne
            | Instruction::Eq
            | Instruction::In
            | Instruction::StartsWith
            | Instruction::EndsWith
            | Instruction::Order(_)
            | Instruction::TrimCharacters
            | Instruction::MapFilter => (2, -1),
            Instruction::Replace | Instruction::StringReplace => (3, -2),
            Instruction::Slice => (4, -3),
            Instruction::Selection{args,..} => (usize::from(args)+1,-isize::from(args)),
            Instruction::ToJson(options) => (usize::from(options.arguments) + 1, -(isize::from(options.arguments))),
            Instruction::Join(args) if args <= 1 => (usize::from(args) + 1, -isize::from(args)),
            Instruction::Join(_) => return Err(SourceError::Geometry),
            Instruction::StringSplit(args) if args<=2 => (usize::from(args)+1,-isize::from(args)),
            Instruction::StringStrip { args, .. } if args <= 1 => {
                (usize::from(args) + 1, -isize::from(args))
            }
            Instruction::StringSplit(_) | Instruction::StringStrip { .. } => {
                return Err(SourceError::Geometry);
            }
            Instruction::MappingGet(args) if (1..=2).contains(&args) => {
                (usize::from(args) + 1, -isize::from(args))
            }
            Instruction::MappingGet(_) => return Err(SourceError::Geometry),
            Instruction::Default(args) if args <= 2 => (usize::from(args) + 1, -isize::from(args)),
            Instruction::Default(_) => return Err(SourceError::Geometry),
            _ => (0, 0),
        };
        if self.depth < need {
            return Err(SourceError::Geometry);
        }
        self.depth = self
            .depth
            .checked_add_signed(delta)
            .ok_or(SourceError::Geometry)?;
        self.geometry.operands = self.geometry.operands.max(self.depth);
        if self.instructions.len() == self.limits.instructions
            || self.instructions.len() == self.instructions.capacity()
            || self.locations.len() == self.locations.capacity()
        {
            return Err(SourceError::Geometry);
        }
        let at = self.position();
        self.instructions.push(op);
        self.locations.push(Location {
            line: usize::from(span.start_line),
            span: Some(OutputSpan {
                start_line: usize::from(span.start_line),
                start_col: usize::from(span.start_col),
                start_offset: span.start_offset as usize,
                end_line: usize::from(span.end_line),
                end_col: usize::from(span.end_col),
                end_offset: span.end_offset as usize,
            }),
        });
        Ok(at)
    }
    // Measurement stores whitespace edges for concatenations, but has no atom
    // destination yet. Filters consuming bytes during measurement therefore
    // require one borrowed atom, including prior trims of such an atom.
    fn atom_expression(&self, mut id: NodeId) -> bool {
        loop {
            match &self.source.storage.expressions.nodes[id.0].kind {
                RecordKind::Const(Scalar::Text(_))
                | RecordKind::Var(_)
                | RecordKind::Attr(_, _)
                | RecordKind::Item(_, _) => return true,
                RecordKind::Filter("trim", Some(value), arguments) if arguments.len <= 1 => {
                    id = *value;
                }
                _ => return false,
            }
        }
    }
    fn expression(&mut self, id: NodeId) -> Result<(), SourceError> {
        let source = self.source;
        let record = &source.storage.expressions.nodes[id.0];
        let span = record.span;
        match &record.kind {
            RecordKind::Var(name) => {
                // External lookup is represented by the borrowed render context.
                // Values and actual operations are qualified during measurement.
                let name = self.text(name)?;
                self.op(Instruction::Lookup(name), span)?;
            }
            RecordKind::Const(Scalar::Int(0)) => {
                self.op(Instruction::Zero, span)?;
            }
            RecordKind::Const(Scalar::None) => {
                self.op(
                    Instruction::Scalar(crate::value::primitive::scalar::Scalar::None),
                    span,
                )?;
            }
            RecordKind::Const(Scalar::Bool(value)) => {
                self.op(
                    Instruction::Scalar(crate::value::primitive::scalar::Scalar::Bool(*value)),
                    span,
                )?;
            }
            RecordKind::Const(Scalar::Int(value)) => {
                self.op(
                    Instruction::Scalar(crate::value::primitive::scalar::Scalar::U64(*value)),
                    span,
                )?;
            }
            RecordKind::Const(Scalar::Int128(value)) => {
                self.op(
                    Instruction::Scalar(crate::value::primitive::scalar::Scalar::U128(*value)),
                    span,
                )?;
            }
            RecordKind::Const(Scalar::Float(value)) => {
                self.op(
                    Instruction::Scalar(crate::value::primitive::scalar::Scalar::F64(*value)),
                    span,
                )?;
            }
            RecordKind::Const(Scalar::Text(joined)) => {
                let start = self.bytes.len();
                let mut current = joined.sequence.head;
                let mut seen = 0;
                while let Some(index) = current {
                    let segment = &source.storage.expressions.segments[index];
                    match segment.kind {
                        SegmentKind::Source(text) => {
                            self.text(text)?;
                        }
                        SegmentKind::Decoded(range) => {
                            let text = std::str::from_utf8(
                                &source.buffers.literals[range.start..range.end],
                            )
                            .map_err(|_| SourceError::Geometry)?;
                            self.text(text)?;
                        }
                    }
                    seen += 1;
                    if seen > joined.sequence.len {
                        return Err(SourceError::Geometry);
                    }
                    current = segment.next;
                }
                if seen != joined.sequence.len || self.bytes.len() - start != joined.bytes {
                    return Err(SourceError::Geometry);
                }
                self.op(Instruction::Literal(Range::new(start, joined.bytes)), span)?;
            }
            RecordKind::If(test, yes, no) => {
                self.push(Action::Conditional(*yes, *no, span))?;
                self.push(Action::Expression(*test))?;
            }
            RecordKind::Attr(value, name) => {
                let name = self.text(name)?;
                self.push(Action::Emit(Instruction::GetAttr(name), span))?;
                self.push(Action::Expression(*value))?;
            }
            RecordKind::Slice { expr, start, stop, step } => {
                self.push(Action::Emit(Instruction::Slice, span))?;
                for argument in [step, stop, start] {
                    self.push(match argument {
                        Some(value) => Action::Expression(*value),
                        None => Action::Emit(Instruction::Scalar(crate::value::primitive::scalar::Scalar::None), span),
                    })?;
                }
                self.push(Action::Expression(*expr))?;
            }
            RecordKind::Item(value, key) => {
                self.push(Action::Emit(Instruction::GetItem, span))?;
                self.push(Action::Expression(*key))?;
                self.push(Action::Expression(*value))?;
            }
            RecordKind::Call(callee, arguments)
                if (cfg!(feature = "macros") || (cfg!(feature = "builtins") &&
                    matches!(source.storage.expressions.nodes[callee.0].kind,RecordKind::Var("range"))))
                    && matches!(source.storage.expressions.nodes[callee.0].kind,RecordKind::Var(name) if name!="namespace") =>
            {
                self.macro_call(*callee, *arguments, None, span)?;
            }
            RecordKind::Call(callee, arguments)
                if cfg!(feature = "builtins")
                    && matches!(
                        source.storage.expressions.nodes[callee.0].kind,
                        RecordKind::Var("namespace")
                    ) =>
            {
                let name = self.text("namespace")?;
                self.push(Action::Emit(
                    Instruction::CallFunction(name, u16::from(arguments.len != 0)),
                    span,
                ))?;
                if arguments.len != 0 {
                    self.push(Action::Emit(Instruction::BuildKwargs(arguments.len), span))?;
                    self.push(Action::KeywordArguments(*arguments, span))?;
                }
            }
            RecordKind::Call(callee, arguments) if arguments.len <= 2 => {
                let RecordKind::Attr(value, name) =
                    &source.storage.expressions.nodes[callee.0].kind
                else {
                    return Err(SourceError::Profile);
                };
                let operation = match (*name, arguments.len) {
                    ("keys", 0) => Instruction::MappingView(MappingProjection::Keys),
                    ("values", 0) => Instruction::MappingView(MappingProjection::Values),
                    ("items", 0) => Instruction::MappingView(MappingProjection::Items),
                    ("get", 1 | 2) => Instruction::MappingGet(arguments.len as u8),
                    ("startswith", 1) => Instruction::StartsWith,
                    ("endswith", 1) => Instruction::EndsWith,
                    ("replace", 2) => Instruction::StringReplace,
                    ("split", 0 | 1 | 2) => Instruction::StringSplit(arguments.len as u8),
                    ("strip" | "lstrip" | "rstrip", 0 | 1) => Instruction::StringStrip {
                        args: arguments.len as u8,
                        left: *name != "rstrip",
                        right: *name != "lstrip",
                    },
                    _ => return Err(SourceError::Profile),
                };
                let mut args = [None; 2];
                let mut next = arguments.head;
                for destination in args.iter_mut().take(arguments.len) {
                    let argument = source
                        .storage
                        .expressions
                        .operands
                        .get(next.ok_or(SourceError::Geometry)?)
                        .ok_or(SourceError::Geometry)?;
                    let OperandKind::Pos(value) = argument.kind else {
                        return Err(SourceError::Profile);
                    };
                    *destination = Some(value);
                    next = argument.next;
                }
                if next.is_some() {
                    return Err(SourceError::Geometry);
                }
                self.push(Action::Emit(operation, span))?;
                for argument in args.into_iter().flatten().rev() {
                    self.push(Action::Expression(argument))?;
                }
                self.push(Action::Expression(*value))?;
            }
            RecordKind::Filter(name @ ("select"|"reject"|"selectattr"|"rejectattr"),Some(value),arguments)
                if cfg!(feature="builtins") && arguments.len<=3 => {
                let attribute=matches!(*name,"selectattr"|"rejectattr");
                let extra=usize::from(attribute);
                if !(arguments.len==extra || arguments.len==extra+2) {return Err(SourceError::Profile);}
                let mut args=[None;3];let mut next=arguments.head;
                for destination in args.iter_mut().take(arguments.len) {
                    let argument=source.storage.expressions.operands.get(next.ok_or(SourceError::Geometry)?).ok_or(SourceError::Geometry)?;
                    let OperandKind::Pos(value)=argument.kind else{return Err(SourceError::Profile);};
                    *destination=Some(value);next=argument.next;
                }
                if next.is_some(){return Err(SourceError::Geometry);}
                self.push(Action::Emit(Instruction::Selection{invert:matches!(*name,"reject"|"rejectattr"),attribute,args:arguments.len as u8},span))?;
                for argument in args.into_iter().flatten().rev(){self.push(Action::Expression(argument))?;}
                self.push(Action::Expression(*value))?;
            }
            RecordKind::Filter(name @ ("items" | "list"), Some(value), arguments)
                if arguments.len == 0 && cfg!(feature = "builtins") =>
            {
                self.push(Action::Emit(
                    if *name == "items" {
                        Instruction::Items
                    } else {
                        Instruction::List
                    },
                    span,
                ))?;
                self.push(Action::Expression(*value))?;
            }
            RecordKind::Filter("tojson", Some(value), arguments) if cfg!(feature = "json") => {
                self.json_filter(*value, *arguments, span)?;
            }
            RecordKind::Filter(name @ ("string"|"float"|"safe"), Some(value), arguments)
                if arguments.len == 0 && cfg!(feature = "builtins") =>
            {
                // Source admission fixes plain-text autoescape. The safe filter
                // uses the same string conversion; its escape marker cannot
                // change any accepted output consumer.
                self.push(Action::Emit(if *name=="float" {Instruction::Floatify}else{Instruction::Stringify}, span))?;
                self.push(Action::Expression(*value))?;
            }
            RecordKind::Filter("length" | "count", Some(value), arguments)
                if arguments.len == 0 =>
            {
                self.push(Action::Emit(Instruction::Length, span))?;
                self.push(Action::Expression(*value))?;
            }
            RecordKind::Filter("default" | "d", Some(value), arguments)
                if arguments.len <= 2 && cfg!(feature = "builtins") =>
            {
                // Actual argument order is preserved, including unselected
                // fallback work. The compiler's continuation stack is LIFO.
                let mut args = [None; 2];
                let mut next = arguments.head;
                for destination in args.iter_mut().take(arguments.len) {
                    let argument = source
                        .storage
                        .expressions
                        .operands
                        .get(next.ok_or(SourceError::Geometry)?)
                        .ok_or(SourceError::Geometry)?;
                    let OperandKind::Pos(value) = argument.kind else {
                        return Err(SourceError::Profile);
                    };
                    *destination = Some(value);
                    next = argument.next;
                }
                if next.is_some() {
                    return Err(SourceError::Geometry);
                }
                self.push(Action::Emit(
                    Instruction::Default(arguments.len as u8),
                    span,
                ))?;
                for argument in args.into_iter().flatten().rev() {
                    self.push(Action::Expression(argument))?;
                }
                self.push(Action::Expression(*value))?;
            }
            RecordKind::Map(keys,values) => {
                if keys.len!=values.len {return Err(SourceError::Geometry);}
                self.push(Action::Emit(Instruction::BuildMap(keys.len),span))?;
                if keys.len!=0 {self.push(Action::MappingEntries(*keys,*values,span))?;}
            }
            RecordKind::List(values) => {
                if values.len == 0 {
                    self.op(Instruction::EmptySequence, span)?;
                } else {
                    self.push(Action::Emit(Instruction::BuildList(values.len), span))?;
                    self.push(Action::OrderedOperands(*values, span))?;
                }
            }
            RecordKind::Filter("replace", Some(value), arguments)
                if arguments.len == 2 && cfg!(feature = "builtins") =>
            {
                let mut args = [None; 2];
                let mut next = arguments.head;
                for destination in args.iter_mut().take(arguments.len) {
                    let argument = source
                        .storage
                        .expressions
                        .operands
                        .get(next.ok_or(SourceError::Geometry)?)
                        .ok_or(SourceError::Geometry)?;
                    let OperandKind::Pos(value) = argument.kind else {
                        return Err(SourceError::Profile);
                    };
                    *destination = Some(value);
                    next = argument.next;
                }
                if next.is_some() {
                    return Err(SourceError::Geometry);
                }
                self.push(Action::Emit(Instruction::Replace, span))?;
                for argument in args.into_iter().flatten().rev() {
                    self.push(Action::Expression(argument))?;
                }
                self.push(Action::Expression(*value))?;
            }
            RecordKind::Filter("join", Some(value), arguments)
                if arguments.len <= 1 && cfg!(feature = "builtins") =>
            {
                self.push(Action::Emit(Instruction::Join(arguments.len as u8), span))?;
                if arguments.len == 1 {
                    let argument = source
                        .storage
                        .expressions
                        .operands
                        .get(arguments.head.ok_or(SourceError::Geometry)?)
                        .ok_or(SourceError::Geometry)?;
                    let OperandKind::Pos(separator) = argument.kind else {
                        return Err(SourceError::Profile);
                    };
                    if argument.next.is_some() {
                        return Err(SourceError::Geometry);
                    }
                    self.push(Action::Expression(separator))?;
                } else if arguments.head.is_some() {
                    return Err(SourceError::Geometry);
                }
                self.push(Action::Expression(*value))?;
            }
            RecordKind::Filter("map", Some(value), arguments)
                if arguments.len == 1 && cfg!(feature = "builtins") =>
            {
                let argument = source.storage.expressions.operands
                    .get(arguments.head.ok_or(SourceError::Geometry)?)
                    .ok_or(SourceError::Geometry)?;
                let OperandKind::Pos(filter) = argument.kind else {
                    return Err(SourceError::Profile);
                };
                if argument.next.is_some() { return Err(SourceError::Geometry); }
                self.push(Action::Emit(Instruction::MapFilter, span))?;
                self.push(Action::Expression(filter))?;
                self.push(Action::Expression(*value))?;
            }
            RecordKind::Filter("dictsort", Some(value), arguments)
                if arguments.len == 0 && cfg!(feature = "builtins") && !cfg!(feature = "unicode") =>
            {
                self.push(Action::Emit(Instruction::DictSort, span))?;
                self.push(Action::Expression(*value))?;
            }
            RecordKind::Filter("upper", Some(value), arguments)
                if arguments.len == 0 && cfg!(feature = "builtins") =>
            {
                self.push(Action::Emit(Instruction::Upper, span))?;
                self.push(Action::Expression(*value))?;
            }
            RecordKind::Filter("trim", Some(value), arguments) if arguments.len == 0 => {
                self.push(Action::Emit(Instruction::Trim, span))?;
                self.push(Action::Expression(*value))?;
            }
            RecordKind::Filter("trim", Some(value), arguments) if arguments.len == 1 => {
                let argument = &source.storage.expressions.operands
                    [arguments.head.ok_or(SourceError::Geometry)?];
                let OperandKind::Pos(characters) = argument.kind else {
                    return Err(SourceError::Profile);
                };
                if !self.atom_expression(*value) || !self.atom_expression(characters) {
                    return Err(SourceError::Profile);
                }
                self.push(Action::Emit(Instruction::TrimCharacters, span))?;
                self.push(Action::Expression(characters))?;
                self.push(Action::Expression(*value))?;
            }
            RecordKind::Unary(kind, value) => {
                self.push(Action::Emit(
                    match kind {
                        UnaryOpKind::Not => Instruction::Not,
                        UnaryOpKind::Neg => Instruction::Neg,
                    },
                    span,
                ))?;
                self.push(Action::Expression(*value))?;
            }
            RecordKind::Test(name, value, arguments) if arguments.len == 0 => {
                use crate::value::primitive::{Presence, type_tests::TypeTest};
                let instruction = match *name {
                    "defined" => Instruction::Presence(Presence::Defined),
                    "undefined" => Instruction::Presence(Presence::Undefined),
                    "none" => Instruction::Presence(Presence::None),
                    other => Instruction::TypeTest(
                        TypeTest::from_name(other).ok_or(SourceError::Profile)?,
                    ),
                };
                self.push(Action::Emit(instruction, span))?;
                self.push(Action::Expression(*value))?;
            }
            RecordKind::Binary(BinOpKind::Eq, left, right) => {
                self.push(Action::Emit(Instruction::Eq, span))?;
                self.push(Action::Expression(*right))?;
                self.push(Action::Expression(*left))?;
            }
            RecordKind::Binary(BinOpKind::Concat, left, right) => {
                self.push(Action::Emit(Instruction::StringConcat, span))?;
                self.push(Action::Expression(*right))?;
                self.push(Action::Expression(*left))?;
            }
            RecordKind::Binary(BinOpKind::Add, left, right) => {
                self.push(Action::Emit(Instruction::Add, span))?;
                self.push(Action::Expression(*right))?;
                self.push(Action::Expression(*left))?;
            }
            RecordKind::Binary(
                kind @ (BinOpKind::Sub
                | BinOpKind::Mul
                | BinOpKind::Div
                | BinOpKind::FloorDiv
                | BinOpKind::Rem
                | BinOpKind::Pow),
                left,
                right,
            ) => {
                use crate::value::primitive::scalar::Arithmetic;
                let operation = match kind {
                    BinOpKind::Sub => Arithmetic::Sub,
                    BinOpKind::Mul => Arithmetic::Mul,
                    BinOpKind::Div => Arithmetic::Div,
                    BinOpKind::FloorDiv => Arithmetic::FloorDiv,
                    BinOpKind::Rem => Arithmetic::Rem,
                    BinOpKind::Pow => Arithmetic::Pow,
                    _ => return Err(SourceError::Geometry),
                };
                self.push(Action::Emit(Instruction::Arithmetic(operation), span))?;
                self.push(Action::Expression(*right))?;
                self.push(Action::Expression(*left))?;
            }
            RecordKind::Binary(BinOpKind::Ne, left, right) => {
                self.push(Action::Emit(Instruction::Ne, span))?;
                self.push(Action::Expression(*right))?;
                self.push(Action::Expression(*left))?;
            }
            RecordKind::Binary(
                kind @ (BinOpKind::Lt | BinOpKind::Lte | BinOpKind::Gt | BinOpKind::Gte),
                left,
                right,
            ) => {
                let order = match kind {
                    BinOpKind::Lt => Order::Lt,
                    BinOpKind::Lte => Order::Lte,
                    BinOpKind::Gt => Order::Gt,
                    BinOpKind::Gte => Order::Gte,
                    _ => return Err(SourceError::Profile),
                };
                self.push(Action::Emit(Instruction::Order(order), span))?;
                self.push(Action::Expression(*right))?;
                self.push(Action::Expression(*left))?;
            }
            RecordKind::Binary(BinOpKind::In, left, right) => {
                self.push(Action::Emit(Instruction::In, span))?;
                self.push(Action::Expression(*right))?;
                self.push(Action::Expression(*left))?;
            }
            RecordKind::Binary(BinOpKind::ScAnd, left, right) => {
                self.push(Action::ShortCircuit(Jump::IfFalseOrPop, *right, span))?;
                self.push(Action::Expression(*left))?;
            }
            RecordKind::Binary(BinOpKind::ScOr, left, right) => {
                self.push(Action::ShortCircuit(Jump::IfTrueOrPop, *right, span))?;
                self.push(Action::Expression(*left))?;
            }
            RecordKind::Compare(left, sequence) if sequence.len == 1 => {
                let operand = &source.storage.expressions.operands
                    [sequence.head.ok_or(SourceError::Geometry)?];
                let OperandKind::Compare(kind, right) = operand.kind else {
                    return Err(SourceError::Profile);
                };
                if matches!(kind, CompareOpKind::NotIn) {
                    self.push(Action::Emit(Instruction::Not, span))?;
                }
                let op = match kind {
                    CompareOpKind::Ne => Instruction::Ne,
                    CompareOpKind::Eq => Instruction::Eq,
                    CompareOpKind::In | CompareOpKind::NotIn => Instruction::In,
                    CompareOpKind::Lt => Instruction::Order(Order::Lt),
                    CompareOpKind::Lte => Instruction::Order(Order::Lte),
                    CompareOpKind::Gt => Instruction::Order(Order::Gt),
                    CompareOpKind::Gte => Instruction::Order(Order::Gte),
                    _ => return Err(SourceError::Profile),
                };
                self.push(Action::Emit(op, span))?;
                self.push(Action::Expression(right))?;
                self.push(Action::Expression(*left))?;
            }
            _ => return Err(SourceError::Profile),
        }
        Ok(())
    }
    fn assignment(&mut self, target: NodeId, span: Span) -> Result<(), SourceError> {
        let RecordKind::Var(name) = self.source.storage.expressions.nodes[target.0].kind else {
            return Err(SourceError::Profile);
        };
        if name == "loop" {
            return Err(SourceError::Profile);
        }
        let range = self.text(name)?;
        self.op(Instruction::StoreLocal(range), span)?;
        Ok(())
    }
    // Count each actual source target in its lexical scope, including both
    // conditional branches. Reassignment can reuse a runtime slot; it cannot
    // exceed this source-derived destination population.
    fn assignments(&mut self, target: NodeId, span: Span) -> Result<(), SourceError> {
        let count = match &self.source.storage.expressions.nodes[target.0].kind {
            RecordKind::Var(_) => {
                self.assignment(target, span)?;
                1
            }
            RecordKind::Attr(object, name) => {
                let RecordKind::Var(object) = self.source.storage.expressions.nodes[object.0].kind
                else {
                    return Err(SourceError::Profile);
                };
                let object = self.text(object)?;
                let name = self.text(name)?;
                self.op(Instruction::Lookup(object), span)?;
                self.op(Instruction::SetAttr(name), span)?;
                0
            }
            RecordKind::List(sequence) => {
                let sequence = *sequence;
                if sequence.len == 0 {
                    return Err(SourceError::Profile);
                }
                self.op(Instruction::UnpackList(sequence.len), span)?;
                let mut link = sequence.head;
                for _ in 0..sequence.len {
                    let entry = self
                        .source
                        .storage
                        .expressions
                        .operands
                        .get(link.ok_or(SourceError::Geometry)?)
                        .ok_or(SourceError::Geometry)?;
                    let OperandKind::Expr(node) = entry.kind else {
                        return Err(SourceError::Profile);
                    };
                    link = entry.next;
                    self.assignment(node, span)?;
                }
                if link.is_some() {
                    return Err(SourceError::Geometry);
                }
                sequence.len
            }
            _ => return Err(SourceError::Profile),
        };
        self.locals_depth = self
            .locals_depth
            .checked_add(count)
            .ok_or(SourceError::Overflow)?;
        self.geometry.locals = self.geometry.locals.max(self.locals_depth);
        Ok(())
    }
    fn statement(&mut self, id: StmtId) -> Result<(), SourceError> {
        if self.depth != 0 {
            return Err(SourceError::Geometry);
        }
        let source = self.source;
        let record = &source.storage.statements[id.0];
        let span = record.span;
        match &record.value {
            Statement::Template(body) => self.push(Action::Body(*body))?,
            #[cfg(feature = "macros")]
            Statement::Macro { .. } => self.begin_macro(id)?,
            #[cfg(feature = "macros")]
            Statement::CallBlock { call, .. } => {
                let RecordKind::Call(callee, arguments) =
                    source.storage.expressions.nodes[call.0].kind
                else {
                    return Err(SourceError::Geometry);
                };
                self.push(Action::Emit(Instruction::Emit, span))?;
                self.macro_call(callee, arguments, Some(id), span)?;
            }
            Statement::EmitRaw(text) => {
                let text = self.text(text)?;
                self.op(Instruction::Literal(text), span)?;
                self.op(Instruction::Emit, span)?;
            }
            Statement::EmitExpr(expr) => {
                self.push(Action::Emit(Instruction::Emit, span))?;
                self.push(Action::Expression(*expr))?;
            }
            Statement::SetBlock(target, filter, body) => {
                if filter.is_some(){return Err(SourceError::Profile);}
                self.op(Instruction::BeginCapture,span)?;
                self.push(Action::Assign(*target,span))?;
                self.push(Action::Emit(Instruction::EndCapture,span))?;
                self.push(Action::Body(*body))?;
            }
            Statement::Set(target, value) => {
                self.push(Action::Assign(*target, span))?;
                self.push(Action::Expression(*value))?;
            }
            Statement::If { test, yes, no } => {
                self.push(Action::If(*yes, *no, span))?;
                self.push(Action::Expression(*test))?;
            }
            Statement::For {
                target,
                iter,
                filter,
                recursive,
                body,
                otherwise,
            } => {
                if *recursive || otherwise.len != 0 {
                    return Err(SourceError::Profile);
                }
                self.push(match filter {
                    Some(filter)=>Action::FilterLoop(*target,*filter,*body,span),
                    None=>Action::Loop(*target,*body,span),
                })?;
                self.push(Action::Expression(*iter))?;
            }
            _ => return Err(SourceError::Profile),
        }
        Ok(())
    }
    fn run(&mut self) -> Result<(), SourceError> {
        self.push(Action::Statement(
            self.source.root.ok_or(SourceError::Geometry)?,
        ))?;
        while let Some(action) = self.actions.pop() {
            match action {
                Action::MacroExpression(id) => self.begin_macro(id)?,
                Action::MacroParameters(id, index) => self.macro_parameter(id, index)?,
                Action::MacroDefault(jump, name, id, index, span) => {
                    self.patch(jump, self.position())?;
                    self.op(Instruction::StoreLocal(name), span)?;
                    self.local_added()?;
                    if index != 0 {
                        self.push(Action::MacroParameters(id, index - 1))?;
                    }
                }
                Action::MacroEnd(declaration) => self.end_macro(declaration)?,
                Action::MappingEntries(keys,values,span) => {
                    if keys.len!=values.len || keys.len==0 {return Err(SourceError::Geometry);}
                    let operands=&self.source.storage.expressions.operands;
                    let key=operands.get(keys.head.ok_or(SourceError::Geometry)?).ok_or(SourceError::Geometry)?;
                    let value=operands.get(values.head.ok_or(SourceError::Geometry)?).ok_or(SourceError::Geometry)?;
                    let (OperandKind::Expr(key_node),OperandKind::Expr(value_node))=(&key.kind,&value.kind) else{return Err(SourceError::Geometry);};
                    if (keys.len==1)!=key.next.is_none() || (values.len==1)!=value.next.is_none() {return Err(SourceError::Geometry);}
                    if keys.len>1 {
                        self.push(Action::MappingEntries(
                            Sequence{head:key.next,tail:keys.tail,len:keys.len-1},
                            Sequence{head:value.next,tail:values.tail,len:values.len-1},span))?;
                    }
                    self.push(Action::Expression(*value_node))?;
                    self.push(Action::Expression(*key_node))?;
                }
                Action::OrderedOperands(sequence, span) => {
                    let entry = self
                        .source
                        .storage
                        .expressions
                        .operands
                        .get(sequence.head.ok_or(SourceError::Geometry)?)
                        .ok_or(SourceError::Geometry)?;
                    let next = entry.next;
                    if sequence.len == 0 || (sequence.len == 1) != next.is_none() {
                        return Err(SourceError::Geometry);
                    }
                    let (name, value) = match entry.kind {
                        OperandKind::Pos(value) | OperandKind::Expr(value) => (None, value),
                        OperandKind::Kwarg(name, value) => (Some(name), value),
                        _ => return Err(SourceError::Profile),
                    };
                    if sequence.len > 1 {
                        self.push(Action::OrderedOperands(
                            Sequence {
                                head: next,
                                tail: sequence.tail,
                                len: sequence.len - 1,
                            },
                            span,
                        ))?;
                    }
                    self.push(Action::Expression(value))?;
                    if let Some(name) = name {
                        let range = self.text(name)?;
                        self.push(Action::Emit(Instruction::Literal(range), span))?;
                    }
                }
                Action::KeywordArguments(sequence, span) => {
                    let entry = self
                        .source
                        .storage
                        .expressions
                        .operands
                        .get(sequence.head.ok_or(SourceError::Geometry)?)
                        .ok_or(SourceError::Geometry)?;
                    let OperandKind::Kwarg(name, value) = entry.kind else {
                        return Err(SourceError::Profile);
                    };
                    let next = entry.next;
                    if sequence.len == 0 || (sequence.len == 1) != next.is_none() {
                        return Err(SourceError::Geometry);
                    }
                    let name = self.text(name)?;
                    if sequence.len > 1 {
                        self.push(Action::KeywordArguments(
                            Sequence {
                                head: next,
                                tail: sequence.tail,
                                len: sequence.len - 1,
                            },
                            span,
                        ))?;
                    }
                    self.push(Action::Expression(value))?;
                    self.push(Action::Emit(Instruction::Literal(name), span))?;
                }
                Action::Body(sequence) => {
                    if let Some(index) = sequence.head {
                        if sequence.len == 0 {
                            return Err(SourceError::Geometry);
                        }
                        let link = &self.source.storage.links[index];
                        let (next, stmt) = (link.next, link.stmt);
                        self.push(Action::Body(Sequence {
                            head: next,
                            len: sequence.len - 1,
                            ..sequence
                        }))?;
                        self.push(Action::Statement(stmt))?;
                    } else if sequence.len != 0 {
                        return Err(SourceError::Geometry);
                    }
                }
                Action::Statement(id) => self.statement(id)?,
                Action::Expression(id) => self.expression(id)?,
                Action::Assign(target, span) => self.assignments(target, span)?,
                Action::Emit(op, span) => {
                    self.op(op, span)?;
                }
                Action::Conditional(yes, no, span) => {
                    let branch = self.op(Instruction::JumpIfFalse(!0), span)?;
                    self.push(Action::ConditionalElse(branch, no, self.depth, span))?;
                    self.push(Action::Expression(yes))?;
                }
                Action::ConditionalElse(branch, no, base, span) => {
                    let expected = base.checked_add(1).ok_or(SourceError::Overflow)?;
                    if self.depth != expected {
                        return Err(SourceError::Geometry);
                    }
                    let end = self.op(Instruction::Jump(!0), span)?;
                    self.patch(branch, self.position())?;
                    self.depth = base;
                    self.push(Action::ConditionalEnd(end, expected))?;
                    match no {
                        Some(no) => self.push(Action::Expression(no))?,
                        None => self.push(Action::Emit(Instruction::Undefined, span))?,
                    }
                }
                Action::ConditionalEnd(end, expected) => {
                    if self.depth != expected {
                        return Err(SourceError::Geometry);
                    }
                    self.patch(end, self.position())?;
                }
                Action::If(yes, no, span) => {
                    self.span = span;
                    let branch = patch::begin(self, Jump::IfFalse)?;
                    self.push(Action::Else(branch, no, span))?;
                    self.push(Action::Body(yes))?;
                }
                Action::Else(branch, no, span) => {
                    if self.depth != 0 {
                        return Err(SourceError::Geometry);
                    }
                    self.span = span;
                    if no.len == 0 {
                        patch::finish(self, branch)?;
                    } else {
                        let end = patch::otherwise(self, branch)?;
                        self.push(Action::End(end))?;
                        self.push(Action::Body(no))?;
                    }
                }
                Action::End(branch) => patch::finish(self, branch)?,
                Action::ShortCircuit(kind, right, span) => {
                    self.span = span;
                    let branch = patch::begin(self, kind)?;
                    self.push(Action::End(branch))?;
                    self.push(Action::Expression(right))?;
                }
                Action::FilterLoop(target, filter, body, span) => {
                    self.op(Instruction::BeginCollection,span)?;
                    self.op(Instruction::PushLoop(0),span)?;
                    let iter=self.op(Instruction::Iterate(!0),span)?;
                    let previous_locals=self.locals_depth;
                    self.op(Instruction::DupTop,span)?;
                    self.assignments(target,span)?;
                    self.loop_depth=self.loop_depth.checked_add(1).ok_or(SourceError::Overflow)?;
                    self.geometry.frames=self.geometry.frames.max(self.loop_depth);
                    self.geometry.locals=self.geometry.locals.max(self.locals_depth);
                    self.push(Action::FilterTest(target,body,iter,previous_locals,span))?;
                    self.push(Action::Expression(filter))?;
                }
                Action::FilterTest(target,body,iter,previous_locals,span)=>{
                    let rejected=self.op(Instruction::JumpIfFalse(!0),span)?;
                    self.op(Instruction::AppendCollection,span)?;
                    let accepted=self.op(Instruction::Jump(!0),span)?;
                    self.patch(rejected,self.position())?;
                    self.depth=1; // rejected branch still owns the candidate
                    self.op(Instruction::DiscardTop,span)?;
                    self.patch(accepted,self.position())?;
                    self.op(Instruction::Jump(iter),span)?;
                    self.patch(iter,self.position())?;
                    self.op(Instruction::PopLoopFrame,span)?;
                    self.locals_depth=previous_locals;
                    self.loop_depth=self.loop_depth.checked_sub(1).ok_or(SourceError::Geometry)?;
                    self.op(Instruction::EndCollection,span)?;
                    self.push(Action::Loop(target,body,span))?;
                }
                Action::Loop(target, body, span) => {
                    self.op(Instruction::PushLoop(1), span)?;
                    let iter = self.op(Instruction::Iterate(!0), span)?;
                    let previous_locals = self.locals_depth;
                    self.assignments(target, span)?;
                    self.loop_depth = self
                        .loop_depth
                        .checked_add(1)
                        .ok_or(SourceError::Overflow)?;
                    self.geometry.frames = self.geometry.frames.max(self.loop_depth);
                    self.geometry.locals = self.geometry.locals.max(self.locals_depth);
                    self.push(Action::EndLoop(iter, previous_locals, span))?;
                    self.push(Action::Body(body))?;
                }
                Action::EndLoop(iter, previous_locals, span) => {
                    if self.depth != 0 {
                        return Err(SourceError::Geometry);
                    }
                    self.op(Instruction::Jump(iter), span)?;
                    self.patch(iter, self.position())?;
                    self.op(Instruction::PopLoopFrame, span)?;
                    self.locals_depth = previous_locals;
                    self.loop_depth = self
                        .loop_depth
                        .checked_sub(1)
                        .ok_or(SourceError::Geometry)?;
                }
            }
        }
        if self.depth != 0 || self.loop_depth != 0 {
            return Err(SourceError::Geometry);
        }
        Ok(())
    }
}
pub(in crate::bounded) fn compile(
    source: &ParsedTemplate<'_>,
    instructions: &mut Vec<Instruction>,
    bytes: &mut Vec<u8>,
    locations: &mut Vec<Location>,
    limits: Requirements,
    text_limit: usize,
) -> Result<crate::bounded::source::Geometry, crate::bounded::source::CompileCause> {
    let mut compiler = Compiler {
        source,
        instructions,
        bytes,
        locations,
        actions: Vec::new(),
        limits,
        text_limit,
        span: Span::default(),
        depth: 0,
        loop_depth: 0,
        locals_depth: 0,
        in_macro: false,
        capture_failure: None,
        geometry: crate::bounded::source::Geometry {
            operands: 0,
            frames: 0,
            locals: 0,
        },
    };
    compiler
        .actions
        .try_reserve_exact(limits.actions)
        .map_err(crate::bounded::source::CompileCause::SyntaxReserve)?;
    if let Err(cause) = compiler.run() {
        return Err(compiler
            .capture_failure
            .take()
            .unwrap_or(crate::bounded::source::CompileCause::Source(cause)));
    }
    if compiler
        .instructions
        .iter()
        .any(|op| matches!(op, Instruction::BuildMacro(..)))
    {
        let scopes = crate::bounded::worker::macros::call_capacity()
            .checked_add(1)
            .ok_or(crate::bounded::source::CompileCause::Source(
                SourceError::Overflow,
            ))?;
        compiler.geometry.operands = compiler.geometry.operands.checked_mul(scopes).ok_or(
            crate::bounded::source::CompileCause::Source(SourceError::Overflow),
        )?;
        compiler.geometry.frames = compiler.geometry.frames.checked_mul(scopes).ok_or(
            crate::bounded::source::CompileCause::Source(SourceError::Overflow),
        )?;
        compiler.geometry.locals = compiler.geometry.locals.checked_mul(scopes).ok_or(
            crate::bounded::source::CompileCause::Source(SourceError::Overflow),
        )?;
    }
    Ok(compiler.geometry)
}
