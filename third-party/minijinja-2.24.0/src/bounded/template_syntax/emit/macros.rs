//! Packed source adapter for the ordinary macro jump/default/capture protocol.
use super::*;
#[derive(Clone, Copy)]
pub(super) struct Declaration {
    id: StmtId,
    skip: u32,
    entry: u32,
    arguments: u32,
    count: usize,
    name: Range,
    locals: usize,
    depth: usize,
    loops: usize,
    in_macro: bool,
    assign: bool,
    span: Span,
}
impl<'o, 's, 'w> Compiler<'o, 's, 'w> {
    pub(super) fn local_added(&mut self) -> Result<(), SourceError> {
        self.locals_depth = self
            .locals_depth
            .checked_add(1)
            .ok_or(SourceError::Overflow)?;
        self.geometry.locals = self.geometry.locals.max(self.locals_depth);
        Ok(())
    }
    pub(super) fn macro_call(
        &mut self,
        callee: NodeId,
        arguments: Sequence,
        caller: Option<StmtId>,
        span: Span,
    ) -> Result<(), SourceError> {
        let source = self.source;
        let RecordKind::Var(name) = source.storage.expressions.nodes[callee.0].kind else {
            return Err(SourceError::Profile);
        };
        let mut positional = 0usize;
        let mut keyword = 0usize;
        let mut next = arguments.head;
        for _ in 0..arguments.len {
            let arg = source
                .storage
                .expressions
                .operands
                .get(next.ok_or(SourceError::Geometry)?)
                .ok_or(SourceError::Geometry)?;
            match arg.kind {
                OperandKind::Pos(_) if keyword == 0 => positional += 1,
                OperandKind::Kwarg(..) => keyword += 1,
                _ => return Err(SourceError::Profile),
            }
            next = arg.next;
        }
        if next.is_some() {
            return Err(SourceError::Geometry);
        }
        keyword = keyword
            .checked_add(usize::from(caller.is_some()))
            .ok_or(SourceError::Overflow)?;
        let count = u16::try_from(positional + usize::from(keyword != 0))
            .map_err(|_| SourceError::Overflow)?;
        let name = self.text(name)?;
        self.push(Action::Emit(Instruction::CallFunction(name, count), span))?;
        if keyword != 0 {
            self.push(Action::Emit(Instruction::BuildKwargs(keyword), span))?;
        }
        if let Some(caller) = caller {
            self.push(Action::MacroExpression(caller))?;
            let name = self.intern("caller")?;
            self.push(Action::Emit(Instruction::Literal(name), span))?;
        }
        if arguments.len != 0 {
            self.push(Action::OrderedOperands(arguments, span))?;
        }
        Ok(())
    }
    #[cfg(feature = "macros")]
    fn macro_parts(
        &self,
        id: StmtId,
    ) -> Result<(&'s str, Sequence, Sequence, Sequence, Span), SourceError> {
        let record = self
            .source
            .storage
            .statements
            .get(id.0)
            .ok_or(SourceError::Geometry)?;
        match &record.value {
            Statement::Macro {
                name,
                args,
                defaults,
                body,
            } => Ok((name, *args, *defaults, *body, record.span)),
            Statement::CallBlock {
                args,
                defaults,
                body,
                macro_span,
                ..
            } => Ok(("caller", *args, *defaults, *body, *macro_span)),
            _ => Err(SourceError::Geometry),
        }
    }

    #[cfg(feature = "macros")]
    fn operand(&self, sequence: Sequence, index: usize) -> Result<NodeId, SourceError> {
        if index >= sequence.len {
            return Err(SourceError::Geometry);
        }
        let mut next = sequence.head;
        for ordinal in 0..=index {
            let arg = self
                .source
                .storage
                .expressions
                .operands
                .get(next.ok_or(SourceError::Geometry)?)
                .ok_or(SourceError::Geometry)?;
            if ordinal == index {
                return match arg.kind {
                    OperandKind::Expr(id) => Ok(id),
                    _ => Err(SourceError::Profile),
                };
            }
            next = arg.next;
        }
        Err(SourceError::Geometry)
    }
    #[cfg(feature = "macros")]
    pub(super) fn begin_macro(&mut self, id: StmtId) -> Result<(), SourceError> {
        let (name, args, _, body, span) = self.macro_parts(id)?;
        let name = self.intern(name)?;
        let assign = matches!(
            self.source.storage.statements[id.0].value,
            Statement::Macro { .. }
        );
        let skip = self.op(Instruction::Jump(!0), span)?;
        let arguments = self.position();
        for index in 0..args.len {
            let arg = self.operand(args, index)?;
            let RecordKind::Var(name) = self.source.storage.expressions.nodes[arg.0].kind else {
                return Err(SourceError::Profile);
            };
            let name = self.text(name)?;
            self.op(Instruction::Argument(name), span)?;
        }
        let declaration = Declaration {
            id,
            skip,
            entry: self.position(),
            arguments,
            count: args.len,
            name,
            locals: self.locals_depth,
            depth: self.depth,
            loops: self.loop_depth,
            in_macro: self.in_macro,
            assign,
            span,
        };
        self.in_macro = true;
        self.loop_depth = 0;
        self.locals_depth = 0;
        self.depth = args.len;
        self.geometry.operands = self.geometry.operands.max(self.depth);
        self.push(Action::MacroEnd(declaration))?;
        self.push(Action::Body(body))?;
        if args.len != 0 {
            self.push(Action::MacroParameters(id, args.len - 1))?;
        }
        Ok(())
    }
    #[cfg(feature = "macros")]
    pub(super) fn macro_parameter(&mut self, id: StmtId, index: usize) -> Result<(), SourceError> {
        let (_, args, defaults, _, span) = self.macro_parts(id)?;
        let arg = self.operand(args, index)?;
        let RecordKind::Var(name) = self.source.storage.expressions.nodes[arg.0].kind else {
            return Err(SourceError::Profile);
        };
        let name = self.intern(name)?;
        let first = args
            .len
            .checked_sub(defaults.len)
            .ok_or(SourceError::Geometry)?;
        if index >= first {
            let default = self.operand(defaults, index - first)?;
            self.op(Instruction::DupTop, span)?;
            self.op(
                Instruction::Presence(crate::value::primitive::Presence::Undefined),
                span,
            )?;
            let jump = self.op(Instruction::JumpIfFalse(!0), span)?;
            self.op(Instruction::DiscardTop, span)?;
            self.push(Action::MacroDefault(jump, name, id, index, span))?;
            self.push(Action::Expression(default))?;
        } else {
            self.op(Instruction::StoreLocal(name), span)?;
            self.local_added()?;
            if index != 0 {
                self.push(Action::MacroParameters(id, index - 1))?;
            }
        }
        Ok(())
    }
    #[cfg(feature = "macros")]
    pub(super) fn end_macro(&mut self, value: Declaration) -> Result<(), SourceError> {
        if self.depth != 0 || self.loop_depth != 0 || !self.in_macro {
            return Err(SourceError::Geometry);
        }
        self.op(Instruction::Return, value.span)?;
        self.patch(value.skip, self.position())?;
        self.in_macro = value.in_macro;
        self.loop_depth = value.loops;
        self.depth = value.depth;
        self.locals_depth = value.locals;
        #[cfg(feature = "macros")]
        let caller = {
            let source = self.source;
            let location = super::super::captures::MacroRef::from_statement(source, value.id);
            let plan = super::super::captures::Plan::for_macro(location)
                .map_err(|_| SourceError::Overflow)?;
            let allowed = super::super::captures::compile_bound(
                source.storage.expressions.nodes.len(),
                source.storage.statements.len(),
            )
            .ok_or(SourceError::Overflow)?;
            if plan.requirements().requested_bytes() > allowed {
                return Err(SourceError::Geometry);
            }
            let workspace = match plan.construct() {
                Ok(workspace) => workspace,
                Err(error) => {
                    self.capture_failure = Some(error.into_compile_cause());
                    return Err(SourceError::Geometry);
                }
            };
            let analysis = workspace
                .analyze(location)
                .map_err(|_| SourceError::Geometry)?;
            // Ordinary capture membership is ordered lexicographically by its
            // BTreeSet; preserve it without another source-name allocation.
            let caller = analysis.contains("caller");
            let mut previous = None;
            while let Some(name) = analysis
                .names()
                .filter(|name| *name != "caller" && previous.is_none_or(|old| *name > old))
                .min()
            {
                let range = self.intern(name)?;
                self.op(Instruction::Enclose(range), value.span)?;
                previous = Some(name);
            }
            caller
        };

        if caller {
            self.geometry.locals = self
                .geometry
                .locals
                .checked_add(1)
                .ok_or(SourceError::Overflow)?;
        }
        self.op(Instruction::GetClosure, value.span)?;
        self.op(
            Instruction::Arguments {
                start: value.arguments,
                count: value.count,
            },
            value.span,
        )?;
        self.op(
            Instruction::BuildMacro(value.name, value.entry, u8::from(caller)),
            value.span,
        )?;
        if value.assign {
            self.op(Instruction::StoreLocal(value.name), value.span)?;
            self.local_added()?;
        }
        Ok(())
    }
}
#[cfg(not(feature = "macros"))]
impl Compiler<'_, '_, '_> {
    pub(super) fn begin_macro(&mut self, _: StmtId) -> Result<(), SourceError> {
        Err(SourceError::Profile)
    }
    pub(super) fn macro_parameter(&mut self, _: StmtId, _: usize) -> Result<(), SourceError> {
        Err(SourceError::Profile)
    }
    pub(super) fn end_macro(&mut self, _: Declaration) -> Result<(), SourceError> {
        Err(SourceError::Profile)
    }
}

pub(super) fn control_bytes() -> Option<usize> {
    let parts = [
        size_of::<Declaration>(),
        size_of::<(StmtId, usize)>(),
        size_of::<(u32, Range, StmtId, usize, Span)>(),
        size_of::<(&str, Sequence, Sequence, Sequence, Span)>(),
        size_of::<Option<&str>>(),
        size_of::<std::ops::Range<usize>>(),
        size_of::<Result<NodeId, SourceError>>(),
        size_of::<Result<(), SourceError>>(),
    ];
    let total = parts
        .into_iter()
        .try_fold(std::mem::size_of_val(&parts), usize::checked_add)?;
    #[cfg(feature = "macros")]
    let total = total.checked_add(super::super::captures::control_bytes()?)?;
    Some(total)
}
