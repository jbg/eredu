//! Shared expression-weight traversal and equations; storage owns growth policy.
pub(crate) mod storage;
use super::{storage::ExpressionStorage, Expr, ExprRef, ATTR_HAS_REPEAT};
use std::convert::Infallible;
type Attrs = (u32, u32);
pub(crate) struct Scratch {
    todo: Vec<ExprRef>,
    mapped: Vec<Attrs>,
}
impl Scratch {
    fn empty() -> Self {
        Self {
            todo: Vec::new(),
            mapped: Vec::new(),
        }
    }
    pub(super) fn ordinary(root: ExprRef) -> Self {
        Self {
            todo: vec![root],
            mapped: Vec::with_capacity(32),
        }
    }
}
pub(super) trait Destination {
    type Error;
    fn push<T>(to: &mut Vec<T>, value: T) -> Result<(), Self::Error>;
    fn store(cache: &mut Vec<Attrs>, index: usize, value: Attrs) -> Result<(), Self::Error>;
    fn add(left: u32, right: u32) -> Result<u32, Self::Error>;
    fn sum(values: &[Attrs]) -> Result<u32, Self::Error> {
        values.iter().try_fold(0, |sum, v| Self::add(sum, v.0))
    }
}
pub(super) struct Ordinary;
impl Destination for Ordinary {
    type Error = Infallible;
    fn push<T>(to: &mut Vec<T>, value: T) -> Result<(), Self::Error> {
        to.push(value);
        Ok(())
    }
    fn store(cache: &mut Vec<Attrs>, index: usize, value: Attrs) -> Result<(), Self::Error> {
        if index >= cache.len() {
            cache.resize(index + 100, (0, 0));
        }
        cache[index] = value;
        Ok(())
    }
    fn add(left: u32, right: u32) -> Result<u32, Self::Error> {
        Ok(left + right)
    }
}
fn cached(cache: &[Attrs], id: ExprRef) -> Attrs {
    cache.get(id.as_usize()).copied().unwrap_or((0, 0))
}
/// Same child push order, deferred parent, inherited attributes and large-DAG
/// depth threshold for ordinary and finite prepared storage.
pub(super) fn compute<D: Destination>(
    expressions: &ExpressionStorage,
    cache: &mut Vec<Attrs>,
    scratch: &mut Scratch,
    root: ExprRef,
) -> Result<(), D::Error> {
    scratch.todo.clear();
    D::push(&mut scratch.todo, root)?;
    while let Some(node) = scratch.todo.pop() {
        if cached(cache, node).0 != 0 {
            continue;
        }
        scratch.mapped.clear();
        let mut needs_more_work = false;
        let mut flags = 0;
        let expression = Expr::from_slice(expressions.get(node.0));
        for &child in expression.args() {
            let attrs = cached(cache, child);
            flags |= attrs.1;
            if attrs.0 == 0 {
                if !needs_more_work {
                    D::push(&mut scratch.todo, node)?;
                    needs_more_work = true;
                }
                D::push(&mut scratch.todo, child)?;
            } else {
                D::push(&mut scratch.mapped, attrs)?;
            }
        }
        if needs_more_work {
            continue;
        }
        let mapped = &scratch.mapped;
        let weight = match expression {
            Expr::EmptyString | Expr::NoMatch | Expr::Byte(_) => 1,
            Expr::ByteSet(_) => 2,
            Expr::RemainderIs { .. } => 100,
            Expr::Lookahead(_, _, _) => D::add(mapped[0].0, 1)?,
            Expr::Not(_, _) => D::add(mapped[0].0, 50)?,
            Expr::Repeat(_, _, min, max) => {
                if max >= 2 {
                    flags |= ATTR_HAS_REPEAT;
                }
                D::add(D::add(mapped[0].0, min.min(10))?, max.min(10))?
            }
            Expr::Concat(_, _) => D::add(D::add(mapped[0].0, mapped[1].0)?, 1)?,
            Expr::Or(_, _) => D::sum(mapped)?,
            Expr::And(_, _) => D::add(D::sum(mapped)?, 20)?,
            Expr::ByteConcat(_, bytes, _) => D::add(bytes.len() as u32, mapped[0].0)?,
        };
        let threshold = 1_000_000;
        let weight = if weight < threshold {
            weight
        } else {
            threshold.max(match expression {
                Expr::Concat(_, _) => D::add(mapped[0].0.max(mapped[1].0), 1)?,
                Expr::Or(_, _) => D::add(mapped.iter().map(|v| v.0).max().unwrap(), 5)?,
                Expr::And(_, _) => D::add(mapped.iter().map(|v| v.0).max().unwrap(), 20)?,
                _ => weight,
            })
        };
        D::store(cache, node.as_usize(), (weight, flags))?;
    }
    Ok(())
}
