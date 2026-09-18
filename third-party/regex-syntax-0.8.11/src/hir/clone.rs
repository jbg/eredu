//! Exact structural copies with prospective destination and control storage.
use super::*;
use crate::allocation::{AllocationError, Allocator};

impl Clone for Properties {
    fn clone(&self) -> Self {
        self.clone_with_allocations(Allocator::unenforced())
            .expect("ordinary properties allocation")
    }
}
impl Properties {
    /// Copy these scalar facts after funding their original boxed destination.
    pub fn clone_with_allocations(&self, allocation: Allocator<'_>) -> Result<Self, AllocationError> {
        Ok(Self(match &self.0 {
            None => None,
            Some(properties) => Some(allocation.boxed((**properties).clone())?),
        }))
    }
}
impl Clone for Hir {
    fn clone(&self) -> Self {
        self.clone_with_allocations(Allocator::unenforced())
            .expect("ordinary HIR allocation")
    }
}
impl Hir {
    /// Copy the exact HIR structure and properties with an explicit control
    /// stack. Source depth never becomes recursive Rust call depth.
    pub fn clone_with_allocations(&self, allocation: Allocator<'_>) -> Result<Self, AllocationError> {
        enum Frame<'a> { Visit(&'a Hir), Finish(&'a Hir) }
        let mut stack = Vec::new();
        let mut values = Vec::new();
        allocation.push(&mut stack, Frame::Visit(self))?;
        while let Some(frame) = stack.pop() {
            let source = match frame {
                Frame::Visit(source) => {
                    allocation.push(&mut stack, Frame::Finish(source))?;
                    for sub in source.kind.subs().iter().rev() {
                        allocation.push(&mut stack, Frame::Visit(sub))?;
                    }
                    continue;
                }
                Frame::Finish(source) => source,
            };
            let kind = match &source.kind {
                HirKind::Empty => HirKind::Empty,
                HirKind::Literal(literal) => HirKind::Literal(Literal(allocation.boxed_slice(allocation.copy_slice(&literal.0)?)?)),
                HirKind::Class(class) => HirKind::Class(class.clone_with_allocations(allocation)?),
                HirKind::Look(look) => HirKind::Look(*look),
                HirKind::Repetition(rep) => HirKind::Repetition(Repetition {
                    min: rep.min, max: rep.max, greedy: rep.greedy,
                    sub: allocation.boxed(values.pop().expect("postorder repetition child"))?,
                }),
                HirKind::Capture(capture) => HirKind::Capture(Capture {
                    index: capture.index,
                    name: capture.name.as_ref().map(|name| allocation.boxed_str(allocation.copy_str(name)?)).transpose()?,
                    sub: allocation.boxed(values.pop().expect("postorder capture child"))?,
                }),
                HirKind::Concat(children) | HirKind::Alternation(children) => {
                    let start = values.len() - children.len();
                    let mut copied = Vec::new();
                    allocation.grow(&mut copied, children.len())?;
                    copied.extend(values.drain(start..));
                    if matches!(&source.kind, HirKind::Concat(_)) { HirKind::Concat(copied) } else { HirKind::Alternation(copied) }
                }
            };
            // Every copied node can participate in the existing iterative
            // retirement worker, including cleanup after a later refusal.
            allocation.reserve(4 * core::mem::size_of::<Hir>())?;
            let props = source.props.clone_with_allocations(allocation)?;
            allocation.push(&mut values, Hir { kind, props })?;
        }
        Ok(values.pop().expect("one copied root"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::cell::Cell;
    struct Funding { calls: Cell<usize>, limit: usize }
    impl crate::allocation::Allocation for Funding {
        fn reserve(&self, _: usize) -> Result<(), AllocationError> {
            let call = self.calls.get(); self.calls.set(call + 1);
            if call < self.limit { Ok(()) } else { Err(AllocationError::Refused) }
        }
    }
    #[test]
    fn exact_copy_and_union_refuse_each_destination() {
        let hir = crate::parse(r"(?P<name>[a-z]+)(α|β)*?(?:abc|def)").unwrap();
        let paid = Funding { calls: Cell::new(0), limit: usize::MAX };
        let copied = hir.clone_with_allocations(Allocator::new(&paid)).unwrap();
        assert_eq!(hir, copied);
        for limit in 0..paid.calls.get() {
            let refuse = Funding { calls: Cell::new(0), limit };
            assert_eq!(hir.clone_with_allocations(Allocator::new(&refuse)).unwrap_err(), AllocationError::Refused);
            assert_eq!(refuse.calls.get(), limit + 1);
        }
        let refuse = Funding { calls: Cell::new(0), limit: 0 };
        assert_eq!(Properties::union_with_allocations([hir.properties()], Allocator::new(&refuse)).unwrap_err(), AllocationError::Refused);
        assert_eq!(Properties::union([hir.properties()]), Properties::union_with_allocations([hir.properties()], Allocator::unenforced()).unwrap());
    }
    #[cfg(feature = "std")]
    #[test]
    fn deep_copy_uses_paid_control_stack() {
        std::thread::Builder::new().stack_size(64 * 1024).spawn(|| {
            let mut hir = Hir::literal(b"a".as_slice());
            for index in 1..=4096 { hir = Hir::capture(Capture { index, name: None, sub: Box::new(hir) }); }
            let paid = Funding { calls: Cell::new(0), limit: usize::MAX };
            let copy = hir.clone_with_allocations(Allocator::new(&paid)).unwrap();
            let mut cursor = &copy; let mut count = 0;
            while let HirKind::Capture(capture) = cursor.kind() { count += 1; cursor = &capture.sub; }
            assert_eq!(count, 4096);
        }).unwrap().join().unwrap();
    }
}
