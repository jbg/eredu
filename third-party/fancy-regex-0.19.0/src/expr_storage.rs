//! Storage producers shared by ordinary and admitted expression rewrites.
use crate::allocation::{AllocationError, Context};
use crate::{Absent, AstNode, CaptureGroupTarget, Expr};
use alloc::{boxed::Box, sync::Arc, vec::Vec};

type Result<T> = core::result::Result<T, AllocationError>;

impl Context<'_> {
    pub(crate) fn expressions(self, values: impl IntoIterator<Item = Expr>) -> Result<Vec<Expr>> {
        let values = values.into_iter();
        let mut output = Vec::new();
        self.storage.grow(&mut output, values.size_hint().0)?;
        for value in values {
            self.storage.push(&mut output, value)?;
        }
        Ok(output)
    }
    fn clone_box(self, expr: &Expr) -> Result<Box<Expr>> {
        self.storage.boxed(expr.clone_with_context(self)?)
    }
}

impl Clone for Expr {
    fn clone(&self) -> Self {
        self.clone_with_context(Context::unenforced())
            .expect("expression clone allocation")
    }
}
impl Expr {
    pub(crate) fn clone_with_context(&self, a: Context<'_>) -> Result<Self> {
        Ok(match self {
            Self::Empty => Self::Empty,
            Self::Any { newline, crlf } => Self::Any {
                newline: *newline,
                crlf: *crlf,
            },
            Self::Assertion(value) => Self::Assertion(*value),
            Self::GeneralNewline { unicode } => Self::GeneralNewline { unicode: *unicode },
            Self::Literal { val, casei } => Self::Literal {
                val: a.storage.copy_str(val)?,
                casei: *casei,
            },
            Self::Concat(children) | Self::Alt(children) => {
                let mut copies = Vec::new();
                a.storage.grow(&mut copies, children.len())?;
                for child in children {
                    copies.push(child.clone_with_context(a)?);
                }
                if matches!(self, Self::Concat(_)) {
                    Self::Concat(copies)
                } else {
                    Self::Alt(copies)
                }
            }
            Self::Group(child) => Self::Group(Arc::clone(child)),
            Self::LookAround(child, kind) => Self::LookAround(a.clone_box(child)?, *kind),
            Self::Repeat {
                child,
                lo,
                hi,
                greedy,
            } => Self::Repeat {
                child: a.clone_box(child)?,
                lo: *lo,
                hi: *hi,
                greedy: *greedy,
            },
            Self::Delegate { inner, casei } => Self::Delegate {
                inner: a.storage.copy_str(inner)?,
                casei: *casei,
            },
            Self::Backref { group, casei } => Self::Backref {
                group: *group,
                casei: *casei,
            },
            Self::BackrefWithRelativeRecursionLevel {
                group,
                relative_level,
                casei,
            } => Self::BackrefWithRelativeRecursionLevel {
                group: *group,
                relative_level: *relative_level,
                casei: *casei,
            },
            Self::AtomicGroup(child) => Self::AtomicGroup(a.clone_box(child)?),
            Self::KeepOut => Self::KeepOut,
            Self::ContinueFromPreviousMatchEnd => Self::ContinueFromPreviousMatchEnd,
            Self::BackrefExistsCondition {
                group,
                relative_recursion_level,
            } => Self::BackrefExistsCondition {
                group: *group,
                relative_recursion_level: *relative_recursion_level,
            },
            Self::Conditional {
                condition,
                true_branch,
                false_branch,
            } => Self::Conditional {
                condition: a.clone_box(condition)?,
                true_branch: a.clone_box(true_branch)?,
                false_branch: a.clone_box(false_branch)?,
            },
            Self::SubroutineCall(value) => Self::SubroutineCall(*value),
            Self::BacktrackingControlVerb(value) => Self::BacktrackingControlVerb(*value),
            Self::Absent(value) => Self::Absent(value.clone_with_context(a)?),
            Self::DefineGroup { definitions } => Self::DefineGroup {
                definitions: a.clone_box(definitions)?,
            },
            Self::AstNode(value, position) => {
                Self::AstNode(value.clone_with_context(a)?, *position)
            }
        })
    }
    pub(crate) fn ensure_unique_group(&mut self, a: Context<'_>) -> Result<()> {
        if let Self::Group(child) = self {
            if Arc::get_mut(child).is_none() {
                *child = a.arc(child.clone_with_context(a)?)?;
            }
        }
        Ok(())
    }
}

impl Clone for CaptureGroupTarget {
    fn clone(&self) -> Self {
        self.clone_with_context(Context::unenforced())
            .expect("capture target clone allocation")
    }
}
impl CaptureGroupTarget {
    fn clone_with_context(&self, a: Context<'_>) -> Result<Self> {
        Ok(match self {
            Self::ByNumber(value) => Self::ByNumber(*value),
            Self::ByName(value) => Self::ByName(a.storage.copy_str(value)?),
            Self::Relative(value) => Self::Relative(*value),
        })
    }
}
impl Clone for AstNode {
    fn clone(&self) -> Self {
        self.clone_with_context(Context::unenforced())
            .expect("AST clone allocation")
    }
}
impl AstNode {
    fn clone_with_context(&self, a: Context<'_>) -> Result<Self> {
        Ok(match self {
            Self::AstGroup { name, inner } => Self::AstGroup {
                name: name
                    .as_deref()
                    .map(|name| a.storage.copy_str(name))
                    .transpose()?,
                inner: a.clone_box(inner)?,
            },
            Self::Backref {
                target,
                casei,
                relative_recursion_level,
            } => Self::Backref {
                target: target.clone_with_context(a)?,
                casei: *casei,
                relative_recursion_level: *relative_recursion_level,
            },
            Self::SubroutineCall(target) => Self::SubroutineCall(target.clone_with_context(a)?),
            Self::BackrefExistsCondition {
                target,
                relative_recursion_level,
            } => Self::BackrefExistsCondition {
                target: target.clone_with_context(a)?,
                relative_recursion_level: *relative_recursion_level,
            },
        })
    }
}
impl Clone for Absent {
    fn clone(&self) -> Self {
        self.clone_with_context(Context::unenforced())
            .expect("absent expression clone allocation")
    }
}
impl Absent {
    fn clone_with_context(&self, a: Context<'_>) -> Result<Self> {
        Ok(match self {
            Self::Repeater(child) => Self::Repeater(a.clone_box(child)?),
            Self::Stopper(child) => Self::Stopper(a.clone_box(child)?),
            Self::Expression { absent, exp } => Self::Expression {
                absent: a.clone_box(absent)?,
                exp: a.clone_box(exp)?,
            },
            Self::Clear => Self::Clear,
        })
    }
}
