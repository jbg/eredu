//! Explicit continuations for the original HIR-to-Thompson traversal.
//! Only the active ancestry and one repetition cursor occupy work storage.
use super::*;

#[derive(Clone, Copy)]
enum Task<'h> {
    Expr(&'h Hir),
    Capture {
        expr: &'h Hir,
        index: u32,
        name: Option<&'h str>,
    },
    CaptureEnd {
        start: StateID,
        index: u32,
    },
    Concat {
        remaining: &'h [Hir],
        prefix: Option<ThompsonRef>,
    },
    ConcatEnd {
        remaining: &'h [Hir],
        prefix: Option<ThompsonRef>,
    },
    AltFirst {
        remaining: &'h [Hir],
    },
    AltSecond {
        first: ThompsonRef,
        remaining: &'h [Hir],
    },
    AltRest {
        joined: ThompsonRef,
        remaining: &'h [Hir],
    },
    Exact {
        expr: &'h Hir,
        remaining: u32,
        prefix: Option<ThompsonRef>,
    },
    ExactEnd {
        expr: &'h Hir,
        remaining: u32,
        prefix: Option<ThompsonRef>,
    },
    BoundedPrefix {
        expr: &'h Hir,
        greedy: bool,
        remaining: u32,
    },
    Bounded {
        expr: &'h Hir,
        greedy: bool,
        remaining: u32,
        start: StateID,
        previous: StateID,
        empty: StateID,
    },
    BoundedEnd {
        expr: &'h Hir,
        greedy: bool,
        remaining: u32,
        start: StateID,
        previous: StateID,
        empty: StateID,
        union: StateID,
    },
    AtLeast {
        expr: &'h Hir,
        greedy: bool,
        minimum: u32,
    },
    AtLeastPrefix {
        expr: &'h Hir,
        greedy: bool,
    },
    PlusEnd {
        greedy: bool,
        prefix: Option<ThompsonRef>,
    },
    StarEnd {
        union: StateID,
    },
    NullableStarEnd {
        greedy: bool,
    },
    OptionalEnd {
        union: StateID,
    },
}

struct Machine<'h> {
    work: Vec<Task<'h>>,
    result: Option<ThompsonRef>,
}
impl Machine<'_> {
    fn take(&mut self) -> ThompsonRef {
        self.result.take().expect("completed HIR child")
    }
}

impl Compilation<'_> {
    pub(super) fn c_cap(
        &self,
        index: u32,
        name: Option<&str>,
        expr: &Hir,
    ) -> Result<ThompsonRef, BuildError> {
        self.run(Task::Capture { expr, index, name })
    }
    pub(super) fn c_at_least(
        &self,
        expr: &Hir,
        greedy: bool,
        minimum: u32,
    ) -> Result<ThompsonRef, BuildError> {
        self.run(Task::AtLeast {
            expr,
            greedy,
            minimum,
        })
    }
    fn union(&self, greedy: bool) -> Result<StateID, BuildError> {
        if greedy {
            self.add_union()
        } else {
            self.add_union_reverse()
        }
    }
    fn join(
        &self,
        prefix: Option<ThompsonRef>,
        next: ThompsonRef,
    ) -> Result<ThompsonRef, BuildError> {
        match prefix {
            None => Ok(next),
            Some(prefix) => {
                self.patch(prefix.end, next.start)?;
                Ok(ThompsonRef {
                    start: prefix.start,
                    end: next.end,
                })
            }
        }
    }
    fn run<'h>(&self, initial: Task<'h>) -> Result<ThompsonRef, BuildError> {
        // Fixed machine/dispatch/result controls are admitted before the machine
        // is entered. Every source-dependent continuation lives in the same
        // prospectively grown work vector used by ordinary construction.
        self.allocation.reserve(
            core::mem::size_of::<Machine<'h>>()
                + core::mem::size_of::<Task<'h>>()
                + core::mem::size_of::<Result<ThompsonRef, BuildError>>(),
        )?;
        let mut machine = Machine {
            work: Vec::new(),
            result: None,
        };
        self.allocation.push(&mut machine.work, initial)?;
        while let Some(task) = machine.work.pop() {
            macro_rules! push {
                ($task:expr) => {
                    self.allocation.push(&mut machine.work, $task)?
                };
            }
            match task {
                Task::Expr(expr) => {
                    use hir::HirKind;
                    match expr.kind() {
                        HirKind::Empty => machine.result = Some(self.c_empty()?),
                        HirKind::Literal(hir::Literal(bytes)) => {
                            machine.result = Some(self.c_literal(bytes)?)
                        }
                        HirKind::Class(hir::Class::Bytes(class)) => {
                            machine.result = Some(self.c_byte_class(class)?)
                        }
                        HirKind::Class(hir::Class::Unicode(class)) => {
                            machine.result = Some(self.c_unicode_class(class)?)
                        }
                        HirKind::Look(look) => machine.result = Some(self.c_look(look)?),
                        HirKind::Capture(cap) => push!(Task::Capture {
                            expr: &cap.sub,
                            index: cap.index,
                            name: cap.name.as_deref()
                        }),
                        HirKind::Concat(exprs) => push!(Task::Concat {
                            remaining: exprs,
                            prefix: None
                        }),
                        HirKind::Alternation(exprs) => {
                            if exprs.len() > 1
                                && exprs
                                    .iter()
                                    .all(|expr| matches!(expr.kind(), HirKind::Literal(_)))
                            {
                                machine.result = Some(self.c_literal_alternation(exprs)?);
                            } else if let Some((first, remaining)) = exprs.split_first() {
                                push!(Task::AltFirst { remaining });
                                push!(Task::Expr(first));
                            } else {
                                machine.result = Some(self.c_fail()?);
                            }
                        }
                        HirKind::Repetition(rep) => match (rep.min, rep.max) {
                            (0, Some(1)) => {
                                let union = self.union(rep.greedy)?;
                                push!(Task::OptionalEnd { union });
                                push!(Task::Expr(&rep.sub));
                            }
                            (minimum, None) => push!(Task::AtLeast {
                                expr: &rep.sub,
                                greedy: rep.greedy,
                                minimum
                            }),
                            (minimum, Some(maximum)) => {
                                if minimum != maximum {
                                    push!(Task::BoundedPrefix {
                                        expr: &rep.sub,
                                        greedy: rep.greedy,
                                        remaining: maximum.saturating_sub(minimum)
                                    });
                                }
                                push!(Task::Exact {
                                    expr: &rep.sub,
                                    remaining: minimum,
                                    prefix: None
                                });
                            }
                        },
                    }
                }
                Task::Capture { expr, index, name } => {
                    if matches!(self.config.get_which_captures(), WhichCaptures::All)
                        || matches!(self.config.get_which_captures(), WhichCaptures::Implicit) && index == 0
                    {
                        let start = self.add_capture_start(index, name)?;
                        push!(Task::CaptureEnd { start, index });
                    }
                    push!(Task::Expr(expr));
                }
                Task::CaptureEnd { start, index } => {
                    let inner = machine.take();
                    let end = self.add_capture_end(index)?;
                    self.patch(start, inner.start)?;
                    self.patch(inner.end, end)?;
                    machine.result = Some(ThompsonRef { start, end });
                }
                Task::Concat { remaining, prefix } => {
                    let next = if self.is_reverse() {
                        remaining.split_last().map(|(last, rest)| (last, rest))
                    } else {
                        remaining.split_first()
                    };
                    if let Some((expr, remaining)) = next {
                        push!(Task::ConcatEnd { remaining, prefix });
                        push!(Task::Expr(expr));
                    } else {
                        machine.result = Some(match prefix {
                            Some(prefix) => prefix,
                            None => self.c_empty()?,
                        });
                    }
                }
                Task::ConcatEnd { remaining, prefix } => {
                    let prefix = Some(self.join(prefix, machine.take())?);
                    push!(Task::Concat { remaining, prefix });
                }
                Task::AltFirst { remaining } => {
                    if let Some((second, remaining)) = remaining.split_first() {
                        let first = machine.take();
                        push!(Task::AltSecond { first, remaining });
                        push!(Task::Expr(second));
                    }
                }
                Task::AltSecond { first, remaining } => {
                    let second = machine.take();
                    let start = self.add_union()?;
                    let end = self.add_empty()?;
                    self.patch(start, first.start)?;
                    self.patch(first.end, end)?;
                    self.patch(start, second.start)?;
                    self.patch(second.end, end)?;
                    let joined = ThompsonRef { start, end };
                    if let Some((next, remaining)) = remaining.split_first() {
                        push!(Task::AltRest { joined, remaining });
                        push!(Task::Expr(next));
                    } else {
                        machine.result = Some(joined);
                    }
                }
                Task::AltRest { joined, remaining } => {
                    let next = machine.take();
                    self.patch(joined.start, next.start)?;
                    self.patch(next.end, joined.end)?;
                    if let Some((next, remaining)) = remaining.split_first() {
                        push!(Task::AltRest { joined, remaining });
                        push!(Task::Expr(next));
                    } else {
                        machine.result = Some(joined);
                    }
                }
                Task::Exact {
                    expr,
                    remaining,
                    prefix,
                } => {
                    if remaining == 0 {
                        machine.result = Some(match prefix {
                            Some(prefix) => prefix,
                            None => self.c_empty()?,
                        });
                    } else {
                        push!(Task::ExactEnd {
                            expr,
                            remaining: remaining - 1,
                            prefix
                        });
                        push!(Task::Expr(expr));
                    }
                }
                Task::ExactEnd {
                    expr,
                    remaining,
                    prefix,
                } => {
                    let prefix = Some(self.join(prefix, machine.take())?);
                    push!(Task::Exact {
                        expr,
                        remaining,
                        prefix
                    });
                }
                Task::BoundedPrefix {
                    expr,
                    greedy,
                    remaining,
                } => {
                    let prefix = machine.take();
                    let empty = self.add_empty()?;
                    // Public HIR construction can express max < min. Preserve
                    // the original compiler's empty min..max loop in that case.
                    if remaining == 0 {
                        self.patch(prefix.end, empty)?;
                        machine.result = Some(ThompsonRef { start: prefix.start, end: empty });
                        continue;
                    }
                    push!(Task::Bounded {
                        expr,
                        greedy,
                        remaining,
                        start: prefix.start,
                        previous: prefix.end,
                        empty
                    });
                }
                Task::Bounded {
                    expr,
                    greedy,
                    remaining,
                    start,
                    previous,
                    empty,
                } => {
                    let union = self.union(greedy)?;
                    push!(Task::BoundedEnd {
                        expr,
                        greedy,
                        remaining: remaining - 1,
                        start,
                        previous,
                        empty,
                        union
                    });
                    push!(Task::Expr(expr));
                }
                Task::BoundedEnd {
                    expr,
                    greedy,
                    remaining,
                    start,
                    previous,
                    empty,
                    union,
                } => {
                    let next = machine.take();
                    self.patch(previous, union)?;
                    self.patch(union, next.start)?;
                    self.patch(union, empty)?;
                    if remaining == 0 {
                        self.patch(next.end, empty)?;
                        machine.result = Some(ThompsonRef { start, end: empty });
                    } else {
                        push!(Task::Bounded {
                            expr,
                            greedy,
                            remaining,
                            start,
                            previous: next.end,
                            empty
                        });
                    }
                }
                Task::AtLeast {
                    expr,
                    greedy,
                    minimum,
                } => {
                    if minimum == 0 && expr.properties().minimum_len().map_or(false, |len| len > 0)
                    {
                        let union = self.union(greedy)?;
                        push!(Task::StarEnd { union });
                        push!(Task::Expr(expr));
                    } else if minimum == 0 {
                        push!(Task::NullableStarEnd { greedy });
                        push!(Task::Expr(expr));
                    } else if minimum == 1 {
                        push!(Task::PlusEnd {
                            greedy,
                            prefix: None
                        });
                        push!(Task::Expr(expr));
                    } else {
                        push!(Task::AtLeastPrefix { expr, greedy });
                        push!(Task::Exact {
                            expr,
                            remaining: minimum - 1,
                            prefix: None
                        });
                    }
                }
                Task::AtLeastPrefix { expr, greedy } => {
                    let prefix = Some(machine.take());
                    push!(Task::PlusEnd { greedy, prefix });
                    push!(Task::Expr(expr));
                }
                Task::PlusEnd { greedy, prefix } => {
                    let last = machine.take();
                    let union = self.union(greedy)?;
                    let joined = self.join(prefix, last)?;
                    self.patch(last.end, union)?;
                    self.patch(union, last.start)?;
                    machine.result = Some(ThompsonRef {
                        start: joined.start,
                        end: union,
                    });
                }
                Task::StarEnd { union } => {
                    let inner = machine.take();
                    self.patch(union, inner.start)?;
                    self.patch(inner.end, union)?;
                    machine.result = Some(ThompsonRef {
                        start: union,
                        end: union,
                    });
                }
                Task::NullableStarEnd { greedy } => {
                    let inner = machine.take();
                    let plus = self.union(greedy)?;
                    self.patch(inner.end, plus)?;
                    self.patch(plus, inner.start)?;
                    let question = self.union(greedy)?;
                    let empty = self.add_empty()?;
                    self.patch(question, inner.start)?;
                    self.patch(question, empty)?;
                    self.patch(plus, empty)?;
                    machine.result = Some(ThompsonRef {
                        start: question,
                        end: empty,
                    });
                }
                Task::OptionalEnd { union } => {
                    let inner = machine.take();
                    let empty = self.add_empty()?;
                    self.patch(union, inner.start)?;
                    self.patch(union, empty)?;
                    self.patch(inner.end, empty)?;
                    machine.result = Some(ThompsonRef {
                        start: union,
                        end: empty,
                    });
                }
            }
        }
        Ok(machine.take())
    }
}
