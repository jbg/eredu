use minijinja::machinery::{
    Span,
    ast::{self, Expr, Stmt},
};

struct Replacement {
    span: Span,
    text: String,
}
struct Slices<'a> {
    source: &'a str,
    replacements: Vec<Replacement>,
}

// Postfix spans in MiniJinja begin at the previous postfix operator. Include
// the receiver recursively to obtain the complete source expression.
fn full_span(expr: &Expr<'_>) -> Span {
    let mut span = expr.span();
    let mut include = |child: &Expr<'_>| {
        let child = full_span(child);
        span.start_offset = span.start_offset.min(child.start_offset);
        span.end_offset = span.end_offset.max(child.end_offset);
    };
    match expr {
        Expr::Slice(e) => include(&e.expr),
        Expr::GetItem(e) => include(&e.expr),
        Expr::GetAttr(e) => include(&e.expr),
        Expr::Call(e) => include(&e.expr),
        Expr::Filter(e) => {
            if let Some(e) = &e.expr {
                include(e);
            }
        }
        Expr::Test(e) => include(&e.expr),
        Expr::UnaryOp(e) => include(&e.expr),
        Expr::BinOp(e) => {
            include(&e.left);
            include(&e.right);
        }
        Expr::Compare(e) => {
            include(&e.expr);
            for op in &e.ops {
                include(&op.expr);
            }
        }
        Expr::IfExpr(e) => {
            include(&e.test_expr);
            include(&e.true_expr);
            if let Some(e) = &e.false_expr {
                include(e);
            }
        }
        Expr::Var(_) | Expr::Const(_) | Expr::List(_) | Expr::Map(_) => {}
    }
    span
}

pub(in super::super) fn normalize_slices(source: &str) -> String {
    // Malformed input is passed to the ordinary named compiler for its error
    // location. A failed parse never reaches template execution.
    let Ok(tree) =
        minijinja::machinery::parse(source, "chat", Default::default(), Default::default())
    else {
        return source.to_owned();
    };
    let mut slices = Slices {
        source,
        replacements: Vec::new(),
    };
    slices.statement(&tree);
    slices.region(0, source.len())
}

impl Slices<'_> {
    fn region(&self, start: usize, end: usize) -> String {
        let mut replacements: Vec<_> = self
            .replacements
            .iter()
            .filter(|r| r.span.start_offset as usize >= start && r.span.end_offset as usize <= end)
            .collect();
        replacements.sort_by_key(|r| r.span.start_offset);
        let mut output = String::new();
        let mut cursor = start;
        for replacement in replacements {
            let left = replacement.span.start_offset as usize;
            let right = replacement.span.end_offset as usize;
            output.push_str(&self.source[cursor..left]);
            output.push_str(&replacement.text);
            cursor = right;
        }
        output.push_str(&self.source[cursor..end]);
        output
    }
    fn operand(&self, expr: &Expr<'_>) -> String {
        // The public parser's conditional span includes its preceding token
        // (for example '(' or ':') but excludes that token's closing partner.
        // Assemble this expression from its operands to preserve grouping.
        if let Expr::IfExpr(value) = expr {
            let otherwise = value
                .false_expr
                .as_ref()
                .map(|e| format!(" else ({})", self.operand(e)))
                .unwrap_or_default();
            return format!(
                "(({}) if ({}){})",
                self.operand(&value.true_expr),
                self.operand(&value.test_expr),
                otherwise
            );
        }
        let span = full_span(expr);
        self.region(span.start_offset as usize, span.end_offset as usize)
    }
    fn optional(&mut self, expr: &Option<Expr<'_>>) {
        if let Some(expr) = expr {
            self.expression(expr);
        }
    }
    fn arguments(&mut self, args: &[ast::CallArg<'_>]) {
        for arg in args {
            match arg {
                ast::CallArg::Pos(e)
                | ast::CallArg::Kwarg(_, e)
                | ast::CallArg::PosSplat(e)
                | ast::CallArg::KwargSplat(e) => self.expression(e),
            }
        }
    }
    fn call(&mut self, call: &ast::Call<'_>) {
        self.expression(&call.expr);
        self.arguments(&call.args);
    }
    fn expressions(&mut self, expressions: &[Expr<'_>]) {
        for expr in expressions {
            self.expression(expr);
        }
    }
    fn expression(&mut self, expr: &Expr<'_>) {
        match expr {
            Expr::Slice(slice) => {
                self.expression(&slice.expr);
                self.optional(&slice.start);
                self.optional(&slice.stop);
                self.optional(&slice.step);
                let argument = |expr: &Option<Expr<'_>>| {
                    expr.as_ref().map_or("none".to_owned(), |e| self.operand(e))
                };
                let text = format!(
                    "(({})|__eredu_slice(({}),({}),({})))",
                    self.operand(&slice.expr),
                    argument(&slice.start),
                    argument(&slice.stop),
                    argument(&slice.step)
                );
                let span = full_span(expr);
                // This replacement incorporates all of its nested slices.
                self.replacements.retain(|r| {
                    !(r.span.start_offset >= span.start_offset
                        && r.span.end_offset <= span.end_offset)
                });
                self.replacements.push(Replacement { span, text });
            }
            Expr::Var(_) | Expr::Const(_) => {}
            Expr::UnaryOp(e) => self.expression(&e.expr),
            Expr::BinOp(e) => {
                self.expression(&e.left);
                self.expression(&e.right);
            }
            Expr::Compare(e) => {
                self.expression(&e.expr);
                for op in &e.ops {
                    self.expression(&op.expr);
                }
            }
            Expr::IfExpr(e) => {
                self.expression(&e.test_expr);
                self.expression(&e.true_expr);
                self.optional(&e.false_expr);
            }
            Expr::Filter(e) => {
                self.optional(&e.expr);
                self.arguments(&e.args);
            }
            Expr::Test(e) => {
                self.expression(&e.expr);
                self.arguments(&e.args);
            }
            Expr::GetAttr(e) => self.expression(&e.expr),
            Expr::GetItem(e) => {
                self.expression(&e.expr);
                self.expression(&e.subscript_expr);
            }
            Expr::Call(e) => self.call(e),
            Expr::List(e) => self.expressions(&e.items),
            Expr::Map(e) => {
                self.expressions(&e.keys);
                self.expressions(&e.values);
            }
        }
    }
    fn statements(&mut self, statements: &[Stmt<'_>]) {
        for statement in statements {
            self.statement(statement);
        }
    }
    fn macro_decl(&mut self, decl: &ast::Macro<'_>) {
        self.expressions(&decl.args);
        self.expressions(&decl.defaults);
        self.statements(&decl.body);
    }
    #[allow(unreachable_patterns)]
    fn statement(&mut self, statement: &Stmt<'_>) {
        match statement {
            Stmt::Template(s) => self.statements(&s.children),
            Stmt::EmitRaw(_) => {}
            Stmt::EmitExpr(s) => self.expression(&s.expr),
            Stmt::ForLoop(s) => {
                self.expression(&s.target);
                self.expression(&s.iter);
                self.optional(&s.filter_expr);
                self.statements(&s.body);
                self.statements(&s.else_body);
            }
            Stmt::IfCond(s) => {
                self.expression(&s.expr);
                self.statements(&s.true_body);
                self.statements(&s.false_body);
            }
            Stmt::WithBlock(s) => {
                for (target, value) in &s.assignments {
                    self.expression(target);
                    self.expression(value);
                }
                self.statements(&s.body);
            }
            Stmt::Set(s) => {
                self.expression(&s.target);
                self.expression(&s.expr);
            }
            Stmt::SetBlock(s) => {
                self.expression(&s.target);
                self.optional(&s.filter);
                self.statements(&s.body);
            }
            Stmt::AutoEscape(s) => {
                self.expression(&s.enabled);
                self.statements(&s.body);
            }
            Stmt::FilterBlock(s) => {
                self.expression(&s.filter);
                self.statements(&s.body);
            }
            Stmt::Block(s) => self.statements(&s.body),
            Stmt::Import(s) => {
                self.expression(&s.expr);
                self.expression(&s.name);
            }
            Stmt::FromImport(s) => {
                self.expression(&s.expr);
                for (name, alias) in &s.names {
                    self.expression(name);
                    self.optional(alias);
                }
            }
            Stmt::Extends(s) => self.expression(&s.name),
            Stmt::Include(s) => self.expression(&s.name),
            Stmt::Macro(s) => self.macro_decl(s),
            Stmt::CallBlock(s) => {
                self.call(&s.call);
                self.macro_decl(&s.macro_decl);
            }
            Stmt::Do(s) => self.call(&s.call),
            // The optional loop_controls feature adds only Break/Continue,
            // neither of which contains expressions or nested statements.
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render(source: &str) -> String {
        let mut env = crate::tokenizer::chat_environment();
        env.add_template_owned("test", normalize_slices(source))
            .unwrap();
        env.get_template("test").unwrap().render(()).unwrap()
    }

    #[test]
    fn only_expression_slices_are_rewritten_and_grouping_is_preserved() {
        assert_eq!(
            render(
                "{# x[::-1] #}{% raw %}{{ x[::-1] }}{% endraw %}|{{ 'x[::-1]' }}|{{ ('abc' + 'dé')[::-1][1:3] }}"
            ),
            "{{ x[::-1] }}|x[::-1]|dc"
        );
        assert_eq!(
            render("{{ ['a','b','c'][[0,1][::-1][0]:][::-1]|join('') }}"),
            "cb"
        );
        assert_eq!(
            render("{{ ('abcdef'[::-1])[1:] }}|{{ ('a'+'b')[::-1] }}"),
            "edcba|ba"
        );
    }

    #[test]
    fn macros_blocks_and_short_circuit_keep_normal_execution() {
        assert_eq!(
            render(
                "{% macro f(x='xyz'[::-1]) %}{{ x[1:] }}{% endmacro %}{{ f() }}|{{ false and missing[::0] }}|{% set v %}ab{% endset %}{{ v[::-1] }}"
            ),
            "yx|False|ba"
        );
        assert_eq!(
            render(
                "{% for v in ['abc', 'xyz'][::-1] if v[::-1] != 'cba' %}{{ v[::-1] }}{% endfor %}"
            ),
            "zyx"
        );
    }

    #[test]
    fn receiver_spans_cover_chained_calls_filters_attributes_and_conditionals() {
        for receiver in [
            "'abc'",
            "((('abc')))",
            "['abc'][0]",
            "[['abc']][0][0]",
            "{'key':'abc'}.key",
            "({'key':'abc'}).get('key')",
            "({'key':['abc']}).key[0]",
            "('abc' if true else 'wrong')",
            "(('ab'+'c') if true else 'wrong')",
            "('abc' if true else ('wrong' if false else 'other'))",
            "('ab'+'c')",
            "('ABC'|lower)",
            "('wrong,abc'.split(','))[-1]",
            "(['abc','wrong'][0:1]|first)",
            "(['wrong','abc']|reverse|list)[0]",
        ] {
            assert_eq!(
                render(&format!("{{{{ {receiver}[::-1] }}}}")),
                "cba",
                "{receiver}"
            );
        }
        assert_eq!(
            render("{{ 'abc'[(0 if true else 1):(3 if true else 2):(1 if true else 0)] }}"),
            "abc"
        );
    }
}
