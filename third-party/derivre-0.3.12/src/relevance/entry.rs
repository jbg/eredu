//! Shared non-emptiness entry optimizations and iterative repeat continuations.
use super::{walk, RelevanceCache};
use crate::{
    ast::{Expr, ExprRef, ExprSet},
    simplify::{concat, ConcatElement},
};
pub(crate) trait Context {
    type Error;
    fn source(&self) -> &ExprSet;
    fn cached(&self, root: ExprRef) -> Option<bool>;
    fn cache_true(&mut self, root: ExprRef) -> Result<(), Self::Error>;
    fn begin_concat(&mut self, count: usize) -> Result<(), Self::Error>;
    fn copy_concat(&mut self, root: ExprRef) -> Result<(), Self::Error>;
    fn fold_concat(&mut self) -> Result<ExprRef, Self::Error>;
    fn and(&mut self, left: ExprRef, right: ExprRef) -> Result<ExprRef, Self::Error>;
    fn push_probe(&mut self, root: ExprRef) -> Result<(), Self::Error>;
    fn pop_probe(&mut self) -> Option<ExprRef>;
    fn walk(&mut self, root: ExprRef) -> Result<bool, Self::Error>;
    /// Only an ordinary recoverable fuel refusal may resume the enclosing
    /// repeat probe. An allocation/source failure must remain terminal.
    fn probe_refusal(&mut self, error: Self::Error) -> Result<(), Self::Error>;
}
fn kept(source: &ExprSet, element: ConcatElement<'_>) -> Option<ExprRef> {
    match element {
        ConcatElement::Expr(root) if !source.is_positive(root) => Some(root),
        _ => None,
    }
}
fn filtered_concat<C: Context>(
    context: &mut C,
    root: ExprRef,
) -> Result<Option<ExprRef>, C::Error> {
    let mut count = 0;
    let mut filtered = false;
    for element in context.source().iter_concat(root) {
        if kept(context.source(), element).is_some() {
            count += 1;
        } else {
            filtered = true;
        }
    }
    // The discarded ordinary temporary never affected source cost. Both paths
    // only construct the filtered destination when the optimization applies.
    if !filtered {
        return Ok(None);
    }
    context.begin_concat(count)?;
    context.copy_concat(root)?;
    context.fold_concat().map(Some)
}
pub(crate) fn copy_filtered<E>(
    source: &ExprSet,
    root: ExprRef,
    mut push: impl FnMut(ExprRef) -> Result<(), E>,
) -> Result<(), E> {
    for item in source.iter_concat(root) {
        if let Some(child) = kept(source, item) {
            push(child)?;
        }
    }
    Ok(())
}
/// Tail concat rewrites and repeat probes use explicit caller-owned frames.
/// A failed inner fuel probe follows the same ordinary fallback to the outer DFS.
pub(crate) fn run<C: Context>(context: &mut C, mut root: ExprRef) -> Result<bool, C::Error> {
    loop {
        let mut result = if context.source().is_positive(root) {
            Ok(true)
        } else if let Some(found) = context.cached(root) {
            Ok(found)
        } else {
            match context.source().get(root) {
                Expr::Concat(_, _) | Expr::ByteConcat(_, _, _) => {
                    if let Some(inner) = filtered_concat(context, root)? {
                        root = inner;
                        continue;
                    }
                }
                Expr::And(_, &[left, right]) => {
                    if let (Expr::Repeat(_, a, min_a, max_a), Expr::Repeat(_, b, min_b, max_b)) =
                        (context.source().get(left), context.source().get(right))
                    {
                        if min_a.max(min_b) <= max_a.min(max_b) {
                            let inner = context.and(a, b)?;
                            context.push_probe(root)?;
                            root = inner;
                            continue;
                        }
                    }
                }
                _ => {}
            }
            context.walk(root)
        };
        while let Some(parent) = context.pop_probe() {
            result = match result {
                Ok(true) => {
                    context.cache_true(parent)?;
                    Ok(true)
                }
                Ok(false) => context.walk(parent),
                Err(error) => {
                    context.probe_refusal(error)?;
                    context.walk(parent)
                }
            };
        }
        return result;
    }
}
struct Ordinary<'a> {
    source: &'a mut ExprSet,
    cache: &'a mut RelevanceCache,
    concat: Vec<ExprRef>,
    probes: Vec<ExprRef>,
}
impl Context for Ordinary<'_> {
    type Error = anyhow::Error;
    fn source(&self) -> &ExprSet {
        self.source
    }
    fn cached(&self, root: ExprRef) -> Option<bool> {
        self.cache.relevance_cache.get(&root).copied()
    }
    fn cache_true(&mut self, root: ExprRef) -> anyhow::Result<()> {
        self.cache.relevance_cache.insert(root, true);
        Ok(())
    }
    fn begin_concat(&mut self, count: usize) -> anyhow::Result<()> {
        self.concat = Vec::with_capacity(count);
        Ok(())
    }
    fn copy_concat(&mut self, root: ExprRef) -> anyhow::Result<()> {
        let concat = &mut self.concat;
        copy_filtered(self.source, root, |child| {
            concat.push(child);
            Ok(())
        })
    }
    fn fold_concat(&mut self) -> anyhow::Result<ExprRef> {
        Ok(concat::ordinary_fold_expressions(self.source, &self.concat))
    }
    fn and(&mut self, left: ExprRef, right: ExprRef) -> anyhow::Result<ExprRef> {
        Ok(self.source.mk_and(&mut vec![left, right]))
    }
    fn push_probe(&mut self, root: ExprRef) -> anyhow::Result<()> {
        self.probes.push(root);
        Ok(())
    }
    fn pop_probe(&mut self) -> Option<ExprRef> {
        self.probes.pop()
    }
    fn walk(&mut self, root: ExprRef) -> anyhow::Result<bool> {
        walk::ordinary(self.cache, self.source, root)
    }
    fn probe_refusal(&mut self, _: anyhow::Error) -> anyhow::Result<()> {
        Ok(())
    }
}
pub(crate) fn ordinary(
    cache: &mut RelevanceCache,
    source: &mut ExprSet,
    root: ExprRef,
) -> anyhow::Result<bool> {
    run(
        &mut Ordinary {
            source,
            cache,
            concat: Vec::new(),
            probes: Vec::new(),
        },
        root,
    )
}
