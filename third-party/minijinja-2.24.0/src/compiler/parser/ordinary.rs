//! Existing boxed AST and token representations for the shared parser.
#![forbid(unsafe_code)]

use super::storage::*;
use super::TokenStream;
use crate::compiler::ast::{self, Spanned};
use crate::compiler::tokens::{Span, Token};
use crate::{Error, ErrorKind, Value};

pub(crate) fn kind<'s>(token: &Token<'s>) -> Kind<'s> {
    match token {
        Token::TemplateData(_) => Kind::TemplateData,
        Token::VariableStart => Kind::VariableStart,
        Token::VariableEnd => Kind::VariableEnd,
        Token::BlockStart => Kind::BlockStart,
        Token::BlockEnd => Kind::BlockEnd,
        Token::Ident(name) => Kind::Ident(name),
        Token::Str(_) => Kind::Str,
        Token::String(_) => Kind::String,
        Token::Int(_) => Kind::Int,
        Token::Int128(_) => Kind::Int128,
        Token::Float(_) => Kind::Float,
        Token::Plus => Kind::Plus,
        Token::Minus => Kind::Minus,
        Token::Mul => Kind::Mul,
        Token::Div => Kind::Div,
        Token::FloorDiv => Kind::FloorDiv,
        Token::Pow => Kind::Pow,
        Token::Mod => Kind::Mod,
        Token::Dot => Kind::Dot,
        Token::Comma => Kind::Comma,
        Token::Colon => Kind::Colon,
        Token::Tilde => Kind::Tilde,
        Token::Assign => Kind::Assign,
        Token::Pipe => Kind::Pipe,
        Token::Eq => Kind::Eq,
        Token::Ne => Kind::Ne,
        Token::Gt => Kind::Gt,
        Token::Gte => Kind::Gte,
        Token::Lt => Kind::Lt,
        Token::Lte => Kind::Lte,
        Token::BracketOpen => Kind::BracketOpen,
        Token::BracketClose => Kind::BracketClose,
        Token::ParenOpen => Kind::ParenOpen,
        Token::ParenClose => Kind::ParenClose,
        Token::BraceOpen => Kind::BraceOpen,
        Token::BraceClose => Kind::BraceClose,
    }
}

impl<'s> Input<'s> for TokenStream<'s> {
    type Text = Box<str>;
    type Error = Error;

    fn next(&mut self) -> Result<Option<(Item<'s, Self::Text>, Span)>, Error> {
        TokenStream::next(self).map(|item| {
            item.map(|(token, span)| {
                let value = match token {
                    Token::TemplateData(value) => Item::TemplateData(value),
                    Token::Str(value) => Item::Str(value),
                    Token::String(value) => Item::String(value),
                    Token::Int(value) => Item::Int(value),
                    Token::Int128(value) => Item::Int128(*value),
                    Token::Float(value) => Item::Float(value),
                    value => Item::Simple(kind(&value)),
                };
                (value, span)
            })
        })
    }

    fn current(&mut self) -> Result<Option<(Kind<'s>, Span)>, Error> {
        TokenStream::current(self).map(|item| item.map(|(token, span)| (kind(token), span)))
    }

    fn current_text(&mut self) -> Result<Option<TextRef<'_, 's, Self::Text>>, Error> {
        TokenStream::current(self).map(|item| match item {
            Some((Token::Str(value), _)) => Some(TextRef::Source(value)),
            Some((Token::String(value), _)) => Some(TextRef::Decoded(value)),
            _ => None,
        })
    }

    fn current_span(&self) -> Span {
        TokenStream::current_span(self)
    }
    fn last_span(&self) -> Span {
        TokenStream::last_span(self)
    }
    fn source(&self) -> &'s str {
        self.tokenizer.source()
    }
}

pub(crate) struct Ordinary;

impl<'s> Store<'s> for Ordinary {
    type Expr = ast::Expr<'s>;
    type Exprs = Vec<ast::Expr<'s>>;
    type Args = Vec<ast::CallArg<'s>>;
    type Comparisons = Vec<ast::CompareOp<'s>>;
    type Text = Box<str>;
    type Concat = String;
    type Error = Error;

    fn node(&mut self, node: Node<'s, Self>, span: Span) -> Result<Self::Expr, Error> {
        Ok(match node {
            Node::Var(id) => ast::Expr::Var(Spanned::new(ast::Var { id }, span)),
            Node::Const(value) => {
                let value = match value {
                    Literal::None => Value::from(()),
                    Literal::Bool(value) => Value::from(value),
                    Literal::Int(value) => Value::from(value),
                    Literal::Int128(value) => Value::from(value),
                    Literal::Float(value) => Value::from(value),
                    Literal::Plain(value) => Value::from(value),
                    Literal::Joined(value) => Value::from(value),
                };
                ast::Expr::Const(Spanned::new(ast::Const { value }, span))
            }
            Node::Slice {
                expr,
                start,
                stop,
                step,
            } => ast::Expr::Slice(Spanned::new(
                ast::Slice {
                    expr,
                    start,
                    stop,
                    step,
                },
                span,
            )),
            Node::Unary(op, expr) => {
                ast::Expr::UnaryOp(Spanned::new(ast::UnaryOp { op, expr }, span))
            }
            Node::Binary(op, left, right) => {
                ast::Expr::BinOp(Spanned::new(ast::BinOp { op, left, right }, span))
            }
            Node::Compare(expr, ops) => {
                ast::Expr::Compare(Spanned::new(ast::Compare { expr, ops }, span))
            }
            Node::If(test_expr, true_expr, false_expr) => ast::Expr::IfExpr(Spanned::new(
                ast::IfExpr {
                    test_expr,
                    true_expr,
                    false_expr,
                },
                span,
            )),
            Node::Filter(name, expr, args) => {
                ast::Expr::Filter(Spanned::new(ast::Filter { name, expr, args }, span))
            }
            Node::Test(name, expr, args) => {
                ast::Expr::Test(Spanned::new(ast::Test { name, expr, args }, span))
            }
            Node::Attr(expr, name) => {
                ast::Expr::GetAttr(Spanned::new(ast::GetAttr { expr, name }, span))
            }
            Node::Item(expr, subscript_expr) => ast::Expr::GetItem(Spanned::new(
                ast::GetItem {
                    expr,
                    subscript_expr,
                },
                span,
            )),
            Node::Call(expr, args) => ast::Expr::Call(Spanned::new(ast::Call { expr, args }, span)),
            Node::List(items) => ast::Expr::List(Spanned::new(ast::List { items }, span)),
            Node::Map(keys, values) => {
                ast::Expr::Map(Spanned::new(ast::Map { keys, values }, span))
            }
        })
    }

    fn variable(&self, expr: &Self::Expr) -> Option<(&'s str, Span)> {
        match expr {
            ast::Expr::Var(var) => Some((var.id, var.span())),
            _ => None,
        }
    }

    fn exprs(&mut self, initial_hint: usize) -> Self::Exprs {
        Vec::with_capacity(initial_hint)
    }
    fn exprs_len(&self, values: &Self::Exprs) -> usize {
        values.len()
    }
    fn push_expr(&mut self, values: &mut Self::Exprs, expr: Self::Expr) -> Result<(), Error> {
        values.push(expr);
        Ok(())
    }
    fn args(&mut self) -> Self::Args {
        Vec::new()
    }
    fn args_len(&self, values: &Self::Args) -> usize {
        values.len()
    }
    fn push_arg(&mut self, values: &mut Self::Args, arg: Arg<'s, Self::Expr>) -> Result<(), Error> {
        values.push(match arg {
            Arg::Pos(expr) => ast::CallArg::Pos(expr),
            Arg::Kwarg(name, expr) => ast::CallArg::Kwarg(name, expr),
            Arg::PosSplat(expr) => ast::CallArg::PosSplat(expr),
            Arg::KwargSplat(expr) => ast::CallArg::KwargSplat(expr),
        });
        Ok(())
    }
    fn comparisons(&mut self) -> Self::Comparisons {
        Vec::new()
    }
    fn comparisons_len(&self, values: &Self::Comparisons) -> usize {
        values.len()
    }
    fn push_comparison(
        &mut self,
        values: &mut Self::Comparisons,
        op: ast::CompareOpKind,
        expr: Self::Expr,
    ) -> Result<(), Error> {
        values.push(ast::CompareOp { op, expr });
        Ok(())
    }
    fn pop_comparison(
        &mut self,
        values: &mut Self::Comparisons,
    ) -> (ast::CompareOpKind, Self::Expr) {
        let value = values.pop().unwrap();
        (value.op, value.expr)
    }
    fn begin_text(&mut self, text: Text<'s, Self::Text>) -> Result<String, Error> {
        Ok(match text {
            Text::Source(value) => value.to_owned(),
            Text::Decoded(value) => value.into_string(),
        })
    }
    fn append_text(
        &mut self,
        joined: &mut String,
        text: TextRef<'_, 's, Self::Text>,
    ) -> Result<(), Error> {
        joined.push_str(match text {
            TextRef::Source(value) => value,
            TextRef::Decoded(value) => value,
        });
        Ok(())
    }
    fn syntax(&self, failure: SyntaxFailure<'s>) -> Error {
        match failure {
            SyntaxFailure::Static(detail) => Error::new(ErrorKind::SyntaxError, detail),
            failure => Error::new(ErrorKind::SyntaxError, failure.to_string()),
        }
    }
}

use super::statements::storage::{Statement, StatementStore};

pub(crate) enum Bindings<'s> {
    With(Vec<(ast::Expr<'s>, ast::Expr<'s>)>),
    Names(Vec<(ast::Expr<'s>, Option<ast::Expr<'s>>)>),
}
impl<'s> StatementStore<'s> for Ordinary {
    type Stmt = ast::Stmt<'s>;
    type Body = Vec<ast::Stmt<'s>>;
    type Bindings = Bindings<'s>;
    type Call = Spanned<ast::Call<'s>>;

    fn statement(&mut self, value: Statement<'s, Self>, span: Span) -> Result<Self::Stmt, Error> {
        macro_rules! spanned {
            ($kind:ident, $value:expr) => {
                ast::Stmt::$kind(Spanned::new($value, span))
            };
        }
        Ok(match value {
            Statement::Template(children) => spanned!(Template, ast::Template { children }),
            Statement::EmitExpr(expr) => spanned!(EmitExpr, ast::EmitExpr { expr }),
            Statement::EmitRaw(raw) => spanned!(EmitRaw, ast::EmitRaw { raw }),
            Statement::For {
                target,
                iter,
                filter,
                recursive,
                body,
                otherwise,
            } => spanned!(
                ForLoop,
                ast::ForLoop {
                    target,
                    iter,
                    filter_expr: filter,
                    recursive,
                    body,
                    else_body: otherwise
                }
            ),
            Statement::If { test, yes, no } => spanned!(
                IfCond,
                ast::IfCond {
                    expr: test,
                    true_body: yes,
                    false_body: no
                }
            ),
            Statement::With(assignments, body) => spanned!(
                WithBlock,
                ast::WithBlock {
                    assignments: match assignments {
                        Bindings::With(values) => values,
                        _ => unreachable!("with bindings"),
                    },
                    body
                }
            ),
            Statement::Set(target, expr) => spanned!(Set, ast::Set { target, expr }),
            Statement::SetBlock(target, filter, body) => spanned!(
                SetBlock,
                ast::SetBlock {
                    target,
                    filter,
                    body
                }
            ),
            Statement::AutoEscape(enabled, body) => {
                spanned!(AutoEscape, ast::AutoEscape { enabled, body })
            }
            Statement::FilterBlock(filter, body) => {
                spanned!(FilterBlock, ast::FilterBlock { filter, body })
            }
            #[cfg(feature = "multi_template")]
            Statement::Block {
                name,
                required,
                body,
            } => spanned!(
                Block,
                ast::Block {
                    name,
                    required,
                    body
                }
            ),
            #[cfg(feature = "multi_template")]
            Statement::Extends(name) => spanned!(Extends, ast::Extends { name }),
            #[cfg(feature = "multi_template")]
            Statement::Include(name, ignore_missing) => spanned!(
                Include,
                ast::Include {
                    name,
                    ignore_missing
                }
            ),
            #[cfg(feature = "multi_template")]
            Statement::Import(expr, name) => spanned!(Import, ast::Import { expr, name }),
            #[cfg(feature = "multi_template")]
            Statement::FromImport(expr, names) => spanned!(
                FromImport,
                ast::FromImport {
                    expr,
                    names: match names {
                        Bindings::Names(values) => values,
                        _ => unreachable!("import bindings"),
                    }
                }
            ),
            #[cfg(feature = "macros")]
            Statement::Macro {
                name,
                args,
                defaults,
                body,
            } => spanned!(
                Macro,
                ast::Macro {
                    name,
                    args,
                    defaults,
                    body
                }
            ),
            #[cfg(feature = "macros")]
            Statement::CallBlock {
                call,
                args,
                defaults,
                body,
                macro_span,
            } => spanned!(
                CallBlock,
                ast::CallBlock {
                    call,
                    macro_decl: Spanned::new(
                        ast::Macro {
                            name: "caller",
                            args,
                            defaults,
                            body
                        },
                        macro_span
                    )
                }
            ),
            #[cfg(feature = "loop_controls")]
            Statement::Continue => spanned!(Continue, ast::Continue),
            #[cfg(feature = "loop_controls")]
            Statement::Break => spanned!(Break, ast::Break),
            Statement::Do(call) => spanned!(Do, ast::Do { call }),
        })
    }
    fn body(&mut self, hint: usize) -> Self::Body {
        Vec::with_capacity(hint)
    }
    fn push_statement(&mut self, body: &mut Self::Body, stmt: Self::Stmt) -> Result<(), Error> {
        body.push(stmt);
        Ok(())
    }
    fn body_is_whitespace(&self, body: &Self::Body) -> bool {
        body.iter()
            .all(|stmt| matches!(stmt, ast::Stmt::EmitRaw(raw) if raw.raw.trim().is_empty()))
    }
    fn first_expr(&mut self, exprs: Self::Exprs) -> Self::Expr {
        exprs.into_iter().next().expect("single assignment")
    }
    fn bindings(&mut self, hint: usize, optional: bool) -> Self::Bindings {
        if optional {
            Bindings::Names(Vec::with_capacity(hint))
        } else {
            Bindings::With(Vec::with_capacity(hint))
        }
    }
    fn bindings_len(&self, bindings: &Self::Bindings) -> usize {
        match bindings {
            Bindings::With(v) => v.len(),
            Bindings::Names(v) => v.len(),
        }
    }
    fn push_binding(
        &mut self,
        bindings: &mut Self::Bindings,
        target: Self::Expr,
        value: Option<Self::Expr>,
    ) -> Result<(), Error> {
        match bindings {
            Bindings::With(v) => v.push((target, value.expect("with value"))),
            Bindings::Names(v) => v.push((target, value)),
        };
        Ok(())
    }
    fn call(&mut self, expr: Self::Expr) -> Result<Self::Call, Error> {
        match expr {
            ast::Expr::Call(call) => Ok(call),
            expr => Err(self.syntax(SyntaxFailure::ExpectedCall(expr.description()))),
        }
    }
}
