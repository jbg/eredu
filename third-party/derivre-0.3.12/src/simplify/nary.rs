//! Shared flattening and n-ary simplification; scratch owns allocation policy.
use super::scalar::{self, Emission};
use crate::ast::{Expr, ExprFlags, ExprRef, ExprSet, ExprTag, byteset_set, byteset_union};
use crate::{ParserAllocationFunding, raw::PreparedExprError};

pub(crate) mod storage;
type Lookahead = (ExprRef, ExprRef, u32);

trait Memory {
    type Error;
    fn copy_suffix(&mut self, input: &[ExprRef]) -> Result<(), Self::Error>;
    fn suffix(&self) -> &[ExprRef];
    fn finish_suffix(&mut self);
    fn ensure_args(&mut self, args: &mut Vec<ExprRef>, required: usize) -> Result<(), Self::Error>;
    fn bytes(&mut self, words: usize) -> Result<&mut [u32], Self::Error>;
    fn finish_bytes(&mut self);
    fn lookahead(&mut self, count: usize) -> Result<&mut Vec<Lookahead>, Self::Error>;
    fn finish_lookahead(&mut self);
    fn overflow(&mut self) -> Self::Error;
}
struct GrowingMemory {
    funding: ParserAllocationFunding,
    suffix: Vec<ExprRef>,
    bytes: Vec<u32>,
    lookahead: Vec<Lookahead>,
}
impl Memory for GrowingMemory {
    type Error = PreparedExprError;
    fn copy_suffix(&mut self, input: &[ExprRef]) -> Result<(), PreparedExprError> {
        self.funding.try_extend_copy(&mut self.suffix, input)?;
        Ok(())
    }
    fn suffix(&self) -> &[ExprRef] {
        &self.suffix
    }
    fn finish_suffix(&mut self) {
        self.suffix = Vec::new();
    }
    fn ensure_args(&mut self, args: &mut Vec<ExprRef>, required: usize) -> Result<(), PreparedExprError> {
        self.funding.try_grow_vec(args, required)?;
        Ok(())
    }
    fn bytes(&mut self, words: usize) -> Result<&mut [u32], PreparedExprError> {
        self.funding.try_grow_vec(&mut self.bytes, words)?;
        self.bytes.resize(words, 0);
        Ok(&mut self.bytes)
    }
    fn finish_bytes(&mut self) {
        self.bytes = Vec::new();
    }
    fn lookahead(&mut self, count: usize) -> Result<&mut Vec<Lookahead>, PreparedExprError> {
        self.lookahead = Vec::new();
        self.funding.try_grow_vec(&mut self.lookahead, count)?;
        Ok(&mut self.lookahead)
    }
    fn finish_lookahead(&mut self) {
        self.lookahead = Vec::new();
    }
    fn overflow(&mut self) -> PreparedExprError {
        self.funding.storage_overflow().into()
    }
}

pub(super) fn growing(source: &mut ExprSet, args: &mut Vec<ExprRef>, tag: ExprTag) -> Result<ExprRef, PreparedExprError> {
    let mut memory = GrowingMemory { funding: source.construction_funding()?.clone(), suffix: Vec::new(), bytes: Vec::new(), lookahead: Vec::new() };
    let mut sink = scalar::Prepared(source);
    match tag {
        ExprTag::Or => or(&mut sink, &mut memory, args),
        ExprTag::And => and(&mut sink, &mut memory, args),
        _ => unreachable!("only And and Or have n-ary construction"),
    }
}

fn flatten<M: Memory>(
    source: &ExprSet,
    tag: ExprTag,
    args: &mut Vec<ExprRef>,
    memory: &mut M,
) -> Result<(), M::Error> {
    let Some(first) = args.iter().position(|&arg| source.get_tag(arg) == tag) else {
        return Ok(());
    };
    memory.copy_suffix(&args[first..])?;
    args.truncate(first);
    for i in 0..memory.suffix().len() {
        let arg = memory.suffix()[i];
        if source.get_tag(arg) == tag {
            let children = source.get_args(arg);
            let required = args
                .len()
                .checked_add(children.len())
                .ok_or_else(|| memory.overflow())?;
            memory.ensure_args(args, required)?;
            args.extend_from_slice(children);
        } else {
            let required = args.len().checked_add(1).ok_or_else(|| memory.overflow())?;
            memory.ensure_args(args, required)?;
            args.push(arg);
        }
    }
    memory.finish_suffix();
    Ok(())
}

fn and<S: Emission, M: Memory<Error = S::Error>>(
    sink: &mut S,
    memory: &mut M,
    args: &mut Vec<ExprRef>,
) -> Result<ExprRef, S::Error> {
    flatten(sink.source(), ExprTag::And, args, memory)?;
    sink.pay(2 * args.len())?;
    args.sort_unstable();
    let mut kept = 0;
    let mut previous = ExprRef::ANY_BYTE_STRING;
    let mut had_empty = false;
    let mut nullable = true;
    for i in 0..args.len() {
        let arg = args[i];
        if arg == previous || arg == ExprRef::ANY_BYTE_STRING {
            continue;
        }
        if arg == ExprRef::NO_MATCH {
            return Ok(ExprRef::NO_MATCH);
        }
        if arg == ExprRef::EMPTY_STRING {
            had_empty = true;
        }
        if nullable && !sink.source().is_nullable(arg) {
            nullable = false;
        }
        args[kept] = arg;
        kept += 1;
        previous = arg;
    }
    args.truncate(kept);
    if args.is_empty() {
        Ok(ExprRef::ANY_BYTE_STRING)
    } else if args.len() == 1 {
        Ok(args[0])
    } else if had_empty {
        Ok(if nullable {
            ExprRef::EMPTY_STRING
        } else {
            ExprRef::NO_MATCH
        })
    } else {
        let flags = ExprFlags::from_nullable_positive(nullable, nullable);
        sink.emit(Expr::And(flags, args))
    }
}

fn or<S: Emission, M: Memory<Error = S::Error>>(
    sink: &mut S,
    memory: &mut M,
    args: &mut Vec<ExprRef>,
) -> Result<ExprRef, S::Error> {
    flatten(sink.source(), ExprTag::Or, args, memory)?;
    sink.pay(2 * args.len())?;
    args.sort_unstable();
    let mut kept = 0;
    let mut previous = ExprRef::NO_MATCH;
    let mut nullable = false;
    let mut bytes = 0;
    let mut lookaheads = 0;
    let mut positive = false;
    for i in 0..args.len() {
        let arg = args[i];
        if arg == previous || arg == ExprRef::NO_MATCH {
            continue;
        }
        if arg == ExprRef::ANY_BYTE_STRING {
            return Ok(ExprRef::ANY_BYTE_STRING);
        }
        match sink.source().get(arg) {
            Expr::Byte(_) | Expr::ByteSet(_) => bytes += 1,
            Expr::Lookahead(_, _, _) => lookaheads += 1,
            _ => {}
        }
        let flags = sink.source().get_flags(arg);
        if !nullable && flags.is_nullable() {
            nullable = true;
        }
        if !positive && flags.is_positive() {
            positive = true;
        }
        args[kept] = arg;
        kept += 1;
        previous = arg;
    }
    args.truncate(kept);
    if bytes > 1 {
        let bits = memory.bytes(sink.source().alphabet_words)?;
        sink.pay(args.len())?;
        args.retain(|&arg| match sink.source().get(arg) {
            Expr::Byte(byte) => {
                byteset_set(bits, byte as usize);
                false
            }
            Expr::ByteSet(words) => {
                byteset_union(bits, words);
                false
            }
            _ => true,
        });
        let node = scalar::byte_set(sink, bits)?;
        let required = args.len().checked_add(1).ok_or_else(|| memory.overflow())?;
        memory.ensure_args(args, required)?;
        let index = args.binary_search(&node).unwrap_or_else(|i| i);
        assert!(index == args.len() || args[index] != node);
        args.insert(index, node);
        memory.finish_bytes();
    }
    if lookaheads > 1 {
        let values = memory.lookahead(lookaheads)?;
        sink.pay(args.len())?;
        args.retain(|&arg| match sink.source().get(arg) {
            Expr::Lookahead(_, inner, offset) => {
                values.push((arg, inner, offset));
                false
            }
            _ => true,
        });
        // Input IDs were sorted above. The ID tie-break therefore preserves the
        // former stable key sort's exact order without allocating sort scratch.
        values.sort_unstable_by_key(|&(id, inner, offset)| (inner.as_u32(), offset, id.as_u32()));
        let mut previous = ExprRef::INVALID;
        for &(node, inner, _) in values.iter() {
            if inner == previous {
                continue;
            }
            previous = inner;
            // Each retained lookahead replaces one of the removed rows, so
            // this never exceeds the already checked incoming argument extent.
            args.push(node);
        }
        args.sort_unstable();
        memory.finish_lookahead();
    }
    if args.is_empty() {
        Ok(ExprRef::NO_MATCH)
    } else if args.len() == 1 {
        Ok(args[0])
    } else {
        let flags = ExprFlags::from_nullable_positive(nullable, positive);
        if sink.source().optimize {
            sink.optimized_or(flags, args)
        } else {
            sink.emit(Expr::Or(flags, args))
        }
    }
}
