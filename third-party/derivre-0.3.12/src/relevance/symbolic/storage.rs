//! Direct list destinations from one reached symbolic node and its actual children.
//! Grouping/disjoint scratch and expression-constructor controls remain separate.
use super::SymRes;
use crate::ast::{Expr, ExprRef, ExprSet};
use std::{
    alloc::Layout,
    collections::TryReserveError,
    fmt,
    mem::{size_of, size_of_val},
};

#[derive(Clone, Copy, Debug)]
pub(crate) struct Requirements {
    pub(crate) buffers: usize,
    pub(crate) controls: usize,
    pub(crate) total: usize,
}
#[derive(Debug)]
enum Cause {
    Source,
    Overflow,
    Capacity,
    Allocation(TryReserveError),
}
pub(crate) struct Failure {
    cause: Cause,
    slots: Vec<Option<SymRes>>,
    pending: SymRes,
}
impl fmt::Debug for Failure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SymbolicListFailure")
            .field("cause", &self.cause)
            .field("completed_slots", &self.slots.len())
            .field("pending_capacity", &self.pending.capacity())
            .finish()
    }
}
impl fmt::Display for Failure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.cause {
            Cause::Source => f.write_str("symbolic node children differ from the reached source"),
            Cause::Overflow => f.write_str("symbolic list geometry overflow"),
            Cause::Capacity => {
                f.write_str("symbolic list destination differs from its exact capacity")
            }
            Cause::Allocation(e) => fmt::Display::fmt(e, f),
        }
    }
}
impl std::error::Error for Failure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.cause {
            Cause::Allocation(e) => Some(e),
            _ => None,
        }
    }
}
fn empty_failure(cause: Cause) -> Failure {
    Failure {
        cause,
        slots: Vec::new(),
        pending: Vec::new(),
    }
}

/// This is a buffer sizing loan, not expression or execution authority.
pub(crate) struct Plan<'a> {
    source: &'a ExprSet,
    root: ExprRef,
    children: &'a [SymRes],
    count: usize,
    requirements: Requirements,
}
impl fmt::Debug for Plan<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SymbolicListPlan")
            .field("requirements", &self.requirements)
            .finish()
    }
}
fn capacities(
    source: &ExprSet,
    root: ExprRef,
    children: &[SymRes],
    mut visit: impl FnMut(usize) -> Result<(), Cause>,
) -> Result<(), Cause> {
    match source.get(root) {
        Expr::EmptyString | Expr::NoMatch => visit(0),
        Expr::Byte(_) | Expr::ByteSet(_) | Expr::ByteConcat(_, _, _) => visit(1),
        Expr::Lookahead(_, _, _) => Ok(()),
        Expr::RemainderIs {
            scale,
            fractional_part,
            ..
        } => visit(if !fractional_part && scale > 0 {
            11
        } else {
            10
        }),
        Expr::And(_, _) => {
            let mut input = children.iter().rev();
            let mut bound = input.next().ok_or(Cause::Source)?.len();
            for child in input {
                bound = bound.checked_mul(child.len()).ok_or(Cause::Overflow)?;
                visit(bound)?;
            }
            Ok(())
        }
        Expr::Or(_, _) => visit(
            children
                .iter()
                .try_fold(0usize, |n, c| n.checked_add(c.len()))
                .ok_or(Cause::Overflow)?,
        ),
        Expr::Not(_, _) => visit(
            source
                .alphabet_words
                .checked_mul(32)
                .and_then(|n| n.checked_add(1))
                .ok_or(Cause::Overflow)?,
        ),
        Expr::Repeat(_, _, _, _) => visit(children[0].len()),
        Expr::Concat(_, [left, _]) => visit(
            children[0]
                .len()
                .checked_add(if source.is_nullable(left) {
                    children[1].len()
                } else {
                    0
                })
                .ok_or(Cause::Overflow)?,
        ),
    }
}
impl<'a> Plan<'a> {
    pub(crate) fn prepare(
        source: &'a ExprSet,
        root: ExprRef,
        children: &'a [SymRes],
    ) -> Result<Self, Failure> {
        let inspected = (|| {
            if !source.is_valid(root) || source.alphabet_size == 0 || source.alphabet_size > 256 {
                return Err(Cause::Source);
            }
            let expected = match source.get(root) {
                Expr::ByteConcat(_, _, _) => 0,
                Expr::Concat(_, [left, _]) if !source.is_nullable(left) => 1,
                _ => source.get_args(root).len(),
            };
            if children.len() != expected {
                return Err(Cause::Source);
            }
            for child in children {
                for &(selector, value) in child {
                    if !source.is_valid(selector) || !source.is_valid(value) {
                        return Err(Cause::Source);
                    }
                    match source.get(selector) {
                        Expr::Byte(b) if usize::from(b) < source.alphabet_size => {}
                        Expr::ByteSet(words)
                            if words.len() == source.alphabet_words
                                && words.iter().any(|&w| w != 0) => {}
                        _ => return Err(Cause::Source),
                    }
                }
            }
            let mut count = 0usize;
            let mut bytes = 0usize;
            capacities(source, root, children, |size| {
                count = count.checked_add(1).ok_or(Cause::Overflow)?;
                bytes = bytes
                    .checked_add(
                        Layout::array::<(ExprRef, ExprRef)>(size)
                            .map_err(|_| Cause::Overflow)?
                            .size(),
                    )
                    .ok_or(Cause::Overflow)?;
                Ok(())
            })?;
            bytes = bytes
                .checked_add(
                    Layout::array::<Option<SymRes>>(count)
                        .map_err(|_| Cause::Overflow)?
                        .size(),
                )
                .ok_or(Cause::Overflow)?;
            let controls = Self::control_bytes().ok_or(Cause::Overflow)?;
            Ok((
                count,
                Requirements {
                    buffers: bytes,
                    controls,
                    total: bytes.checked_add(controls).ok_or(Cause::Overflow)?,
                },
            ))
        })();
        let (count, requirements) = inspected.map_err(empty_failure)?;
        Ok(Self {
            source,
            root,
            children,
            count,
            requirements,
        })
    }
    pub(crate) fn control_bytes() -> Option<usize> {
        let frames = [
            size_of::<Self>(),
            size_of::<Lists>(),
            size_of::<Failure>(),
            size_of::<Requirements>(),
            size_of::<Cause>(),
            size_of::<SymRes>(),
            size_of::<Result<Self, Failure>>(),
            size_of::<Result<Lists, Failure>>(),
            size_of::<Result<(), Cause>>(),
            size_of::<Result<(), TryReserveError>>(),
            size_of::<Result<Layout, std::alloc::LayoutError>>(),
            size_of::<std::slice::Iter<'_, SymRes>>(),
            size_of::<std::slice::Iter<'_, (ExprRef, ExprRef)>>(),
            size_of::<Expr<'_>>(),
        ];
        frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)
    }
    pub(crate) fn requirements(&self) -> Requirements {
        self.requirements
    }
    pub(crate) fn compile(self) -> Result<Lists, Failure> {
        let mut slots = Vec::new();
        let mut pending = Vec::new();
        let result = (|| {
            slots
                .try_reserve_exact(self.count)
                .map_err(Cause::Allocation)?;
            if slots.capacity() != self.count {
                return Err(Cause::Capacity);
            }
            capacities(self.source, self.root, self.children, |count| {
                pending
                    .try_reserve_exact(count)
                    .map_err(Cause::Allocation)?;
                if pending.capacity() != count {
                    return Err(Cause::Capacity);
                }
                slots.push(Some(std::mem::take(&mut pending)));
                Ok(())
            })
        })();
        match result {
            Ok(()) => Ok(Lists {
                slots,
                next: 0,
                failed: false,
            }),
            Err(cause) => Err(Failure {
                cause,
                slots,
                pending,
            }),
        }
    }
}
#[derive(Debug)]
pub(crate) enum Issue {
    Exhausted,
    Capacity,
    Failed,
}
impl fmt::Display for Issue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Exhausted => "symbolic node list slots exhausted",
            Self::Capacity => "symbolic node list exceeds its reached child geometry",
            Self::Failed => "symbolic node list bank retains a failed issue",
        })
    }
}
impl std::error::Error for Issue {}
pub(crate) struct Lists {
    slots: Vec<Option<SymRes>>,
    next: usize,
    failed: bool,
}
impl fmt::Debug for Lists {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SymbolicNodeLists")
            .field("slots", &self.slots.len())
            .field("issued", &self.next)
            .field("failed", &self.failed)
            .finish()
    }
}
impl Lists {
    pub(crate) fn take(&mut self, required: usize) -> Result<SymRes, Issue> {
        if self.failed {
            return Err(Issue::Failed);
        }
        self.failed = true;
        let slot = self
            .slots
            .get_mut(self.next)
            .and_then(Option::as_mut)
            .ok_or(Issue::Exhausted)?;
        if required > slot.capacity() {
            return Err(Issue::Capacity);
        }
        let result = self.slots[self.next]
            .take()
            .expect("checked once-only symbolic list");
        self.next += 1;
        self.failed = false;
        Ok(result)
    }
}
