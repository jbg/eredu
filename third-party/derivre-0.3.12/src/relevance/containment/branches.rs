//! Shared ordered prefix-branch traversal; the destination may count or fill.
use crate::ast::{Expr, ExprRef, ExprSet};
pub(crate) trait Destination {
    type Error;
    fn push(&mut self, node: ExprRef) -> Result<(), Self::Error>;
    fn pop(&mut self) -> Option<ExprRef>;
    fn leftover(&mut self, node: ExprRef, maximum: u32) -> Result<(), Self::Error>;
}
pub(crate) fn run<D: Destination>(
    source: &ExprSet,
    main: ExprRef,
    head: ExprRef,
    dest: &mut D,
) -> Result<Option<ExprRef>, D::Error> {
    dest.push(main)?;
    while let Some(node) = dest.pop() {
        let singleton = [node];
        let branches = match source.get(node) {
            Expr::Or(_, args) => args,
            Expr::NoMatch => &[],
            _ => &singleton,
        };
        for &branch in branches {
            match source.get(branch) {
                Expr::Repeat(_, child, _, maximum) if maximum > 1 => {
                    dest.leftover(child, maximum)?
                }
                Expr::Concat(_, [first, rest]) => {
                    if head == first {
                        return Ok(Some(rest));
                    }
                    dest.push(first)?;
                }
                _ => {}
            }
        }
    }
    Ok(None)
}
pub(super) struct Ordinary {
    pub(super) funding: crate::ParserAllocationFunding,
    pub(super) stack: Vec<ExprRef>,
    pub(super) rows: Vec<(ExprRef, u32)>,
}
impl Destination for Ordinary {
    type Error = crate::ParserError;
    fn push(&mut self, node: ExprRef) -> crate::ParserResult<()> {
        self.funding.try_push(&mut self.stack, node)?;
        Ok(())
    }
    fn pop(&mut self) -> Option<ExprRef> {
        self.stack.pop()
    }
    fn leftover(&mut self, node: ExprRef, maximum: u32) -> crate::ParserResult<()> {
        self.funding.try_push(&mut self.rows, (node, maximum))?;
        Ok(())
    }
}
