//! Registered option binding preserves the actual argument evaluation order.
use super::*;
use crate::bounded::source::JsonOptions;
impl Compiler<'_, '_, '_> {
    pub(super) fn json_filter(
        &mut self,
        value: NodeId,
        arguments: Sequence,
        span: Span,
    ) -> Result<(), SourceError> {
        if arguments.len > 4 {
            return Err(SourceError::Profile);
        }
        let source = self.source;
        let mut options = JsonOptions {
            order: [u8::MAX; 4],
            arguments: 0,
            separator_pair: false,
        };
        let mut values = [None; 5];
        let mut next = arguments.head;
        let mut positional = 0usize;
        let mut keyword = false;
        for _ in 0..arguments.len {
            let argument = source
                .storage
                .expressions
                .operands
                .get(next.ok_or(SourceError::Geometry)?)
                .ok_or(SourceError::Geometry)?;
            let (index, value) = match argument.kind {
                OperandKind::Pos(value) if !keyword => {
                    let index = positional;
                    positional += 1;
                    (index, value)
                }
                OperandKind::Kwarg(name, value) => {
                    keyword = true;
                    let index = JsonOptions::NAMES
                        .iter()
                        .position(|&candidate| candidate == name)
                        .ok_or(SourceError::Profile)?;
                    (index, value)
                }
                _ => return Err(SourceError::Profile),
            };
            if index >= 4 || options.order[index] != u8::MAX {
                return Err(SourceError::Profile);
            }
            options.order[index] = options.arguments;
            let literal_items = if index == 2 {
                match &source.storage.expressions.nodes
                    .get(value.0).ok_or(SourceError::Geometry)?.kind {
                    RecordKind::List(items) => Some(*items),
                    _ => None,
                }
            } else { None };
            if let Some(items) = literal_items {
                if items.len != 2 {
                    return Err(SourceError::Profile);
                }
                let mut item = items.head;
                for _ in 0..2 {
                    let row = source
                        .storage
                        .expressions
                        .operands
                        .get(item.ok_or(SourceError::Geometry)?)
                        .ok_or(SourceError::Geometry)?;
                    let OperandKind::Expr(value) = row.kind else {
                        return Err(SourceError::Profile);
                    };
                    values[usize::from(options.arguments)] = Some(value);
                    options.arguments += 1;
                    item = row.next;
                }
                if item.is_some() {
                    return Err(SourceError::Geometry);
                }
                options.separator_pair = true;
            } else {
                values[usize::from(options.arguments)] = Some(value);
                options.arguments += 1;
            }
            next = argument.next;
        }
        if next.is_some() {
            return Err(SourceError::Geometry);
        }
        self.push(Action::Emit(Instruction::ToJson(options), span))?;
        for value in values.into_iter().flatten().rev() {
            self.push(Action::Expression(value))?;
        }
        self.push(Action::Expression(value))
    }
}
pub(super) fn control_bytes() -> usize {
    size_of::<JsonOptions>()
        + size_of::<[Option<NodeId>; 5]>()
        + size_of::<[&str; 4]>()
        + size_of::<(Sequence, Span, NodeId, usize, usize, bool, Option<usize>)>()
        + size_of::<(&Operand<'_>, usize, NodeId)>()
        + size_of::<std::slice::Iter<'_, &str>>()
        + size_of::<std::ops::Range<usize>>()
        + size_of::<(usize, NodeId, Option<usize>)>()
        + size_of::<std::iter::Rev<std::iter::Flatten<std::array::IntoIter<Option<NodeId>, 5>>>>()
}
