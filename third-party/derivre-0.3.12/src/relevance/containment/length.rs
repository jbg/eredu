//! The same simple-length heuristic, with explicit finite continuation storage.
use crate::{
    ast::{Expr, ExprRef, ExprSet},
    simplify::{next_concat, ConcatElement},
};
#[derive(Clone, Copy)]
pub(crate) enum Task {
    Evaluate(ExprRef),
    Alternative {
        root: ExprRef,
        index: usize,
        best: usize,
    },
    Concat {
        current: Option<ExprRef>,
        sum: usize,
        returning: bool,
    },
}
pub(crate) trait Destination {
    type Error;
    fn push(&mut self, todo: &mut Vec<Task>, task: Task) -> Result<(), Self::Error>;
    fn add(&mut self, left: usize, right: usize) -> Result<usize, Self::Error>;
}
pub(crate) fn run<D: Destination>(
    source: &ExprSet,
    root: ExprRef,
    todo: &mut Vec<Task>,
    dest: &mut D,
) -> Result<Option<usize>, D::Error> {
    todo.clear();
    dest.push(todo, Task::Evaluate(root))?;
    let mut value = Some(0);
    while let Some(task) = todo.pop() {
        match task {
            Task::Evaluate(root) => match source.get(root) {
                Expr::ByteSet(_) | Expr::Byte(_) => value = Some(1),
                Expr::EmptyString => value = Some(0),
                Expr::Or(_, args) => {
                    value = Some(0);
                    if let Some(&first) = args.first() {
                        dest.push(
                            todo,
                            Task::Alternative {
                                root,
                                index: 0,
                                best: 0,
                            },
                        )?;
                        dest.push(todo, Task::Evaluate(first))?;
                    }
                }
                Expr::Concat(_, _) | Expr::ByteConcat(_, _, _) => {
                    dest.push(
                        todo,
                        Task::Concat {
                            current: Some(root),
                            sum: 0,
                            returning: false,
                        },
                    )?;
                }
                _ => value = None,
            },
            Task::Alternative { root, index, best } => {
                if let Some(length) = value {
                    let best = best.max(length);
                    let args = source.get_args(root);
                    let next = index + 1;
                    if let Some(&child) = args.get(next) {
                        dest.push(
                            todo,
                            Task::Alternative {
                                root,
                                index: next,
                                best,
                            },
                        )?;
                        dest.push(todo, Task::Evaluate(child))?;
                    } else {
                        value = Some(best);
                    }
                }
            }
            Task::Concat {
                mut current,
                mut sum,
                returning,
            } => {
                if returning {
                    let Some(length) = value else {
                        continue;
                    };
                    sum = dest.add(sum, length)?;
                }
                loop {
                    match next_concat(source, &mut current) {
                        Some(ConcatElement::Bytes(bytes)) => sum = dest.add(sum, bytes.len())?,
                        Some(ConcatElement::Expr(child)) => {
                            dest.push(
                                todo,
                                Task::Concat {
                                    current,
                                    sum,
                                    returning: true,
                                },
                            )?;
                            dest.push(todo, Task::Evaluate(child))?;
                            break;
                        }
                        None => {
                            value = Some(sum);
                            break;
                        }
                    }
                }
            }
        }
    }
    Ok(value)
}
pub(super) struct Ordinary;
impl Destination for Ordinary {
    type Error = anyhow::Error;
    fn push(&mut self, todo: &mut Vec<Task>, task: Task) -> anyhow::Result<()> {
        todo.push(task);
        Ok(())
    }
    fn add(&mut self, left: usize, right: usize) -> anyhow::Result<usize> {
        Ok(left + right)
    }
}
