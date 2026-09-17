#![forbid(unsafe_code)]
use super::*;
use crate::bounded::expression::store::{NodeId, OperandKind, Sequence};
use crate::compiler::parser::statements::storage::Statement;
use crate::compiler::{ast, parser};
use crate::ErrorKind;
#[path = "image.rs"]
mod image;
use image::Atom;
#[path = "reference.rs"]
mod reference;

include!("../expression/tests/fixtures.rs");

fn ordinary_body(body: &[ast::Stmt<'_>], out: &mut Vec<Atom>) {
    out.push(Atom::Count(body.len()));
    for stmt in body {
        ordinary(stmt, out);
    }
}
fn ordinary_exprs(values: &[ast::Expr<'_>], out: &mut Vec<Atom>) {
    out.push(Atom::Count(values.len()));
    for expr in values {
        image::ordinary(expr, out);
    }
}
fn ordinary_call(call: &ast::Spanned<ast::Call<'_>>, out: &mut Vec<Atom>) {
    out.push(Atom::Node("call", call.span()));
    image::ordinary(&call.expr, out);
    out.push(Atom::Count(call.args.len()));
    for arg in &call.args {
        let (tag, expr) = match arg {
            ast::CallArg::Pos(e) => ("pos", e),
            ast::CallArg::Kwarg(n, e) => {
                out.push(Atom::Name((*n).into()));
                ("kwarg", e)
            }
            ast::CallArg::PosSplat(e) => ("splat", e),
            ast::CallArg::KwargSplat(e) => ("kwargs", e),
        };
        out.push(Atom::Tag(tag));
        image::ordinary(expr, out);
    }
}
#[cfg(feature = "macros")]
fn ordinary_macro(value: &ast::Macro<'_>, span: Span, out: &mut Vec<Atom>) {
    out.push(Atom::Node("macro", span));
    out.push(Atom::Name(value.name.into()));
    ordinary_exprs(&value.args, out);
    ordinary_exprs(&value.defaults, out);
    ordinary_body(&value.body, out);
}
fn ordinary(stmt: &ast::Stmt<'_>, out: &mut Vec<Atom>) {
    macro_rules! start {
        ($s:expr,$tag:literal) => {
            out.push(Atom::Node($tag, $s.span()))
        };
    }
    match stmt {
        ast::Stmt::Template(s) => {
            start!(s, "template");
            ordinary_body(&s.children, out);
        }
        ast::Stmt::EmitRaw(s) => {
            start!(s, "raw");
            out.push(Atom::Text(s.raw.into()));
        }
        ast::Stmt::EmitExpr(s) => {
            start!(s, "emit");
            image::ordinary(&s.expr, out);
        }
        ast::Stmt::ForLoop(s) => {
            start!(s, "for");
            image::ordinary(&s.target, out);
            image::ordinary(&s.iter, out);
            match &s.filter_expr {
                Some(e) => image::ordinary(e, out),
                None => out.push(Atom::None),
            };
            out.push(Atom::Bool(s.recursive));
            ordinary_body(&s.body, out);
            ordinary_body(&s.else_body, out);
        }
        ast::Stmt::IfCond(s) => {
            start!(s, "if");
            image::ordinary(&s.expr, out);
            ordinary_body(&s.true_body, out);
            ordinary_body(&s.false_body, out);
        }
        ast::Stmt::WithBlock(s) => {
            start!(s, "with");
            out.push(Atom::Count(s.assignments.len()));
            for (t, e) in &s.assignments {
                image::ordinary(t, out);
                image::ordinary(e, out);
            }
            ordinary_body(&s.body, out);
        }
        ast::Stmt::Set(s) => {
            start!(s, "set");
            image::ordinary(&s.target, out);
            image::ordinary(&s.expr, out);
        }
        ast::Stmt::SetBlock(s) => {
            start!(s, "setblock");
            image::ordinary(&s.target, out);
            match &s.filter {
                Some(e) => image::ordinary(e, out),
                None => out.push(Atom::None),
            };
            ordinary_body(&s.body, out);
        }
        ast::Stmt::AutoEscape(s) => {
            start!(s, "autoescape");
            image::ordinary(&s.enabled, out);
            ordinary_body(&s.body, out);
        }
        ast::Stmt::FilterBlock(s) => {
            start!(s, "filterblock");
            image::ordinary(&s.filter, out);
            ordinary_body(&s.body, out);
        }
        #[cfg(feature = "multi_template")]
        ast::Stmt::Block(s) => {
            start!(s, "block");
            out.push(Atom::Name(s.name.into()));
            out.push(Atom::Bool(s.required));
            ordinary_body(&s.body, out);
        }
        #[cfg(feature = "multi_template")]
        ast::Stmt::Extends(s) => {
            start!(s, "extends");
            image::ordinary(&s.name, out);
        }
        #[cfg(feature = "multi_template")]
        ast::Stmt::Include(s) => {
            start!(s, "include");
            image::ordinary(&s.name, out);
            out.push(Atom::Bool(s.ignore_missing));
        }
        #[cfg(feature = "multi_template")]
        ast::Stmt::Import(s) => {
            start!(s, "import");
            image::ordinary(&s.expr, out);
            image::ordinary(&s.name, out);
        }
        #[cfg(feature = "multi_template")]
        ast::Stmt::FromImport(s) => {
            start!(s, "from");
            image::ordinary(&s.expr, out);
            out.push(Atom::Count(s.names.len()));
            for (n, a) in &s.names {
                image::ordinary(n, out);
                match a {
                    Some(e) => image::ordinary(e, out),
                    None => out.push(Atom::None),
                }
            }
        }
        #[cfg(feature = "macros")]
        ast::Stmt::Macro(s) => ordinary_macro(s, s.span(), out),
        #[cfg(feature = "macros")]
        ast::Stmt::CallBlock(s) => {
            start!(s, "callblock");
            ordinary_call(&s.call, out);
            ordinary_macro(&s.macro_decl, s.macro_decl.span(), out);
        }
        #[cfg(feature = "loop_controls")]
        ast::Stmt::Continue(s) => start!(s, "continue"),
        #[cfg(feature = "loop_controls")]
        ast::Stmt::Break(s) => start!(s, "break"),
        ast::Stmt::Do(s) => {
            start!(s, "do");
            ordinary_call(&s.call, out);
        }
    }
}
fn closed_body(owner: &ParsedTemplate<'_>, body: Sequence, out: &mut Vec<Atom>) {
    out.push(Atom::Count(body.len));
    let mut cursor = body.head;
    let mut count = 0;
    while let Some(i) = cursor {
        let link = &owner.storage.links[i];
        closed(owner, link.stmt, out);
        cursor = link.next;
        count += 1;
        assert!(count <= body.len);
    }
    assert_eq!(count, body.len);
}
fn closed_exprs(owner: &ParsedTemplate<'_>, seq: Sequence, out: &mut Vec<Atom>) {
    out.push(Atom::Count(seq.len));
    let mut cursor = seq.head;
    let mut count = 0;
    while let Some(i) = cursor {
        let operand = &owner.storage.expressions.operands[i];
        match operand.kind {
            OperandKind::Expr(e) => image::closed(owner, e, out),
            _ => panic!("expression list"),
        };
        cursor = operand.next;
        count += 1;
        assert!(count <= seq.len);
    }
    assert_eq!(count, seq.len);
}
fn closed_bindings(owner: &ParsedTemplate<'_>, seq: Sequence, out: &mut Vec<Atom>) {
    out.push(Atom::Count(seq.len));
    let mut cursor = seq.head;
    let mut count = 0;
    while let Some(i) = cursor {
        let operand = &owner.storage.expressions.operands[i];
        match operand.kind {
            OperandKind::Binding(t, e) => {
                image::closed(owner, t, out);
                match e {
                    Some(e) => image::closed(owner, e, out),
                    None => out.push(Atom::None),
                }
            }
            _ => panic!("binding list"),
        };
        cursor = operand.next;
        count += 1;
        assert!(count <= seq.len);
    }
    assert_eq!(count, seq.len);
}
fn closed(owner: &ParsedTemplate<'_>, id: store::StmtId, out: &mut Vec<Atom>) {
    let record = &owner.storage.statements[id.0];
    macro_rules! start {
        ($tag:literal) => {
            out.push(Atom::Node($tag, record.span))
        };
    }
    match &record.value {
        Statement::Template(body) => {
            start!("template");
            closed_body(owner, *body, out);
        }
        Statement::EmitRaw(raw) => {
            start!("raw");
            out.push(Atom::Text((*raw).into()));
        }
        Statement::EmitExpr(e) => {
            start!("emit");
            image::closed(owner, *e, out);
        }
        Statement::For {
            target,
            iter,
            filter,
            recursive,
            body,
            otherwise,
        } => {
            start!("for");
            image::closed(owner, *target, out);
            image::closed(owner, *iter, out);
            match filter {
                Some(e) => image::closed(owner, *e, out),
                None => out.push(Atom::None),
            };
            out.push(Atom::Bool(*recursive));
            closed_body(owner, *body, out);
            closed_body(owner, *otherwise, out);
        }
        Statement::If { test, yes, no } => {
            start!("if");
            image::closed(owner, *test, out);
            closed_body(owner, *yes, out);
            closed_body(owner, *no, out);
        }
        Statement::With(bindings, body) => {
            start!("with");
            closed_bindings(owner, *bindings, out);
            closed_body(owner, *body, out);
        }
        Statement::Set(t, e) => {
            start!("set");
            image::closed(owner, *t, out);
            image::closed(owner, *e, out);
        }
        Statement::SetBlock(t, filter, body) => {
            start!("setblock");
            image::closed(owner, *t, out);
            match filter {
                Some(e) => image::closed(owner, *e, out),
                None => out.push(Atom::None),
            };
            closed_body(owner, *body, out);
        }
        Statement::AutoEscape(e, body) => {
            start!("autoescape");
            image::closed(owner, *e, out);
            closed_body(owner, *body, out);
        }
        Statement::FilterBlock(e, body) => {
            start!("filterblock");
            image::closed(owner, *e, out);
            closed_body(owner, *body, out);
        }
        #[cfg(feature = "multi_template")]
        Statement::Block {
            name,
            required,
            body,
        } => {
            start!("block");
            out.push(Atom::Name((*name).into()));
            out.push(Atom::Bool(*required));
            closed_body(owner, *body, out);
        }
        #[cfg(feature = "multi_template")]
        Statement::Extends(e) => {
            start!("extends");
            image::closed(owner, *e, out);
        }
        #[cfg(feature = "multi_template")]
        Statement::Include(e, ignore) => {
            start!("include");
            image::closed(owner, *e, out);
            out.push(Atom::Bool(*ignore));
        }
        #[cfg(feature = "multi_template")]
        Statement::Import(e, n) => {
            start!("import");
            image::closed(owner, *e, out);
            image::closed(owner, *n, out);
        }
        #[cfg(feature = "multi_template")]
        Statement::FromImport(e, names) => {
            start!("from");
            image::closed(owner, *e, out);
            closed_bindings(owner, *names, out);
        }
        #[cfg(feature = "macros")]
        Statement::Macro {
            name,
            args,
            defaults,
            body,
        } => {
            start!("macro");
            out.push(Atom::Name((*name).into()));
            closed_exprs(owner, *args, out);
            closed_exprs(owner, *defaults, out);
            closed_body(owner, *body, out);
        }
        #[cfg(feature = "macros")]
        Statement::CallBlock {
            call,
            args,
            defaults,
            body,
            macro_span,
        } => {
            start!("callblock");
            image::closed(owner, *call, out);
            out.push(Atom::Node("macro", *macro_span));
            out.push(Atom::Name("caller".into()));
            closed_exprs(owner, *args, out);
            closed_exprs(owner, *defaults, out);
            closed_body(owner, *body, out);
        }
        #[cfg(feature = "loop_controls")]
        Statement::Continue => start!("continue"),
        #[cfg(feature = "loop_controls")]
        Statement::Break => start!("break"),
        Statement::Do(call) => {
            start!("do");
            image::closed(owner, *call, out);
        }
    }
}
fn error_image(error: &crate::Error) -> (ErrorKind, Option<String>, Option<String>, Option<usize>) {
    (
        error.kind(),
        error.detail().map(str::to_owned),
        error.name().map(str::to_owned),
        error.line(),
    )
}
fn check(source: &str, whitespace: WhitespaceConfig) {
    let old = reference::parse(source, "template", Default::default(), whitespace);
    let new = parser::parse(source, "template", Default::default(), whitespace);
    let packed = Plan::inspect(source, "template", whitespace)
        .unwrap()
        .construct();
    match (old, new, packed) {
        (Ok(a), Ok(b), Ok(c)) => {
            let (mut aa, mut bb, mut cc) = (Vec::new(), Vec::new(), Vec::new());
            ordinary(&a, &mut aa);
            ordinary(&b, &mut bb);
            closed(&c, c.root.unwrap(), &mut cc);
            assert_eq!(aa, bb, "ordinary {source:?}");
            assert_eq!(aa, cc, "closed {source:?}");
            assert_eq!(c.source.as_ptr(), source.as_ptr());
            assert!(c.storage.expressions.nodes.len() <= c.requirements.events);
            assert!(c.storage.expressions.operands.len() <= c.requirements.events);
            assert!(c.storage.links.len() <= c.requirements.events);
        }
        (Err(a), Err(b), Err(c)) => {
            assert_eq!(error_image(&a), error_image(&b), "ordinary {source:?}");
            let (kind, detail, span) = match c.cause() {
                Cause::Syntax(e) => (ErrorKind::SyntaxError, Some(e.to_string()), e.span()),
                Cause::Lexical(e) => (e.kind(), e.detail().map(str::to_owned), e.span()),
                _ => panic!("unexpected cause {source:?}: {c:?}"),
            };
            assert_eq!(kind, a.kind(), "{source:?}");
            assert_eq!(detail.as_deref(), a.detail(), "{source:?}");
            assert_eq!(span.map(|s| s.start_line as usize), a.line(), "{source:?}");
            assert_eq!(a.name(), span.map(|_| "template"));
            #[cfg(feature = "debug")]
            {
                assert_eq!(a.span(), b.span());
                assert_eq!(span, a.span(), "{source:?}");
            }
        }
        _ => panic!("success/error disagreement {source:?}"),
    }
}

#[test]
fn all_32_real_templates_and_eight_whitespace_settings_match_full_syntax() {
    assert_eq!(FIXTURES.len(), 32);
    for (name, source) in FIXTURES {
        for bits in 0..8 {
            let whitespace = WhitespaceConfig {
                keep_trailing_newline: bits & 1 != 0,
                lstrip_blocks: bits & 2 != 0,
                trim_blocks: bits & 4 != 0,
            };
            check(source, whitespace);
            assert!(!name.is_empty());
        }
    }
}
#[test]
fn statements_assignments_spans_and_temporary_operands_match() {
    for source in [
        "",
        "raw\n",
        " {{ 1 + 2 }} ",
        "{% set x=1 %}{{ x }}",
        "{% set (x, (y,z)),w = data %}",
        "{% set (((x)))=1 %}",
        "{% set x,=1, %}",
        "{% set ()=[] %}",
        "{% set x=1,2,3 %}",
        "{% set ns . x . y='é' %}",
        "{% with x=1,y=x,(a,b)=pair %}{{ y }}{% endwith %}",
        "{% set body %}text{{ x }}{% endset %}",
        "{% set body|company . safe(1)|upper %}text{% endset %}",
        "{% filter upper|replace('A','B') %}a{% endfilter %}",
        "{% autoescape true %}{{ x }}{% endautoescape %}",
        "{% do service.run(x) %}",
        "{% for () in [] %}{% endfor %}",
    ] {
        check(source, WhitespaceConfig::default());
    }
    let owner = Plan::inspect("{% set ((((name))))=1 %}", "template", Default::default())
        .unwrap()
        .construct()
        .unwrap();
    assert_eq!(owner.storage.expressions.nodes.len(), 2);
    assert_eq!(owner.storage.expressions.operands.len(), 5);
}
#[test]
fn loop_filter_recursive_else_and_elif_control_match() {
    for source in [
        "{% for x in xs if x recursive %}{{ loop(x.children) }}{% else %}none{% endfor %}",
        "{% if a %}a{% elif b %}b{% elif c %}c{% else %}d{% endif %}",
        "{% for x in (a if b else c) %}{% if x %}y{% endif %}{% endfor %}",
        "{% for x in a if b else c %}{% endfor %}",
        "{% for x in xs %}{% continue %}{% break %}{% endfor %}",
        "{% break %}",
        "{% continue %}",
    ] {
        check(source, Default::default());
    }
}
#[test]
fn blocks_imports_required_names_and_feature_denials_match() {
    for source in [
        "{% block a scoped required %} \n{# comment #}{% endblock a %}",
        "{% block a %}x{% endblock b %}",
        "{% block a required %}x{% endblock %}",
        "{% block a %}{% endblock %}{% block a %}{% endblock %}",
        "{% extends parent %}",
        "{% include name with context ignore missing %}",
        "{% include name ignore missing without context %}",
        "{% include name without context %}",
        "{% import name as ns with context %}",
        "{% from name import a as b,c, with context %}",
        "{% from name import without context %}",
    ] {
        check(source, Default::default());
    }
}
#[test]
fn macros_call_blocks_defaults_and_context_errors_match() {
    for source in [
        "{% macro m(x,y=1,z='é',) %}{{ x+y }}{% endmacro %}",
        "{% macro m(x=1,y) %}{% endmacro %}",
        "{% call(x,y=2) service(3) %}{{ x+y }}{% endcall %}",
        "{% call service() %}{{ caller }}{% endcall %}",
        "{% call value %}{% endcall %}",
        "{% do value %}",
        "{% macro m() %}{% block b %}{% endblock %}{% endmacro %}",
        "{% for x in xs %}{% macro m() %}{% break %}{% endmacro %}{% endfor %}",
    ] {
        check(source, Default::default());
    }
}
#[test]
fn reserved_targets_and_error_detail_order_match() {
    for source in [
        "{% set true=1 %}",
        "{% set (a,loop)=xs %}",
        "{% with self=x %}{% endwith %}",
        "{% unknown name %}",
        "{% 12 %}",
        "{%",
        "{{",
        "{% filter %}{% endfilter %}",
        "{% set x= %}",
        "{% if x %}",
        "{% for x in xs %}",
        "{% macro x() %}",
        "{% from x import a as %}",
        "{% include x ignore %}",
        "{% block b %}{% endblock b c %}",
    ] {
        check(source, Default::default());
    }
}
#[test]
fn reuse_after_failure_preserves_flags_depth_and_block_names() {
    fn panic_message(payload: &(dyn std::any::Any + Send)) -> &str {
        if let Some(message) = payload.downcast_ref::<&str>() { message }
        else if let Some(message) = payload.downcast_ref::<String>() { message.as_str() }
        else { panic!("unexpected parser panic payload") }
    }
    for source in [
        "{% for x wrong xs %}{% break %}",
        "{% for x in xs %}{% block b wrong %}{% continue %}",
        "{% macro m() %}{% set true=1 %}{% block b %}{% endblock %}",
        "{% block b required %}x{% endblock %}{% block b %}{% endblock %}",
        "{% if x %}{% for y wrong %}{% continue %}",
        "{% set x=1 %}",
    ] {
        let old = reference::repeated(source);
        let new = parser::statement_reuse_case(source);
        assert_eq!(old.len(), 3);
        assert_eq!(old.len(), new.len());
        for ((a, al, am, ad, ab), (b, bl, bm, bd, bb)) in old.into_iter().zip(new) {
            assert_eq!((al, am, ad, ab), (bl, bm, bd, bb), "{source:?}");
            match (a, b) {
                (Ok(Ok(a)), Ok(Ok(b))) => {
                    let (mut x, mut y) = (Vec::new(), Vec::new());
                    ordinary(&a, &mut x);
                    ordinary(&b, &mut y);
                    assert_eq!(x, y)
                }
                (Ok(Err(a)), Ok(Err(b))) => assert_eq!(error_image(&a), error_image(&b)),
                (Err(a), Err(b)) => {
                    assert_eq!(a.as_ref().type_id(), b.as_ref().type_id(), "{source:?}");
                    assert_eq!(panic_message(a.as_ref()), panic_message(b.as_ref()), "{source:?}");
                }
                _ => panic!("reuse result {source:?}"),
            }
        }
    }
}
#[test]
fn root_raw_comments_trim_and_lazy_error_fetches_match() {
    for source in [
        "before{# ignored {{ {% #}after\n",
        "{# never closes",
        "{% raw %}{{ '\\xQZ' }}{% endraw %}",
        "{% raw %}never closes",
        " x \n {%- set x=1 -%}\n {{- x -}} \n",
        "{%+ if x +%} y {%+ endif +%}",
        "{{ 'a' '\\xQZ' 'later\\n' 12_3 }}",
        "{% set x='a' '\\xQZ' 'later\\n' 12_3 %}",
        "{% if 'a' @ 'later\\n' 12_3 %}{% endif %}",
        "{% for x in xs %}{{ '\\ud800' }}{% endfor %}{{ 'later\\n' 12_3 }}",
        "{% block b %}x{% endblock @ 'later\\n' 12_3 %}",
    ] {
        for bits in 0..8 {
            check(
                source,
                WhitespaceConfig {
                    keep_trailing_newline: bits & 1 != 0,
                    lstrip_blocks: bits & 2 != 0,
                    trim_blocks: bits & 4 != 0,
                },
            );
        }
    }
}
const FULL: &str = "{% set x=['good\\n',12_3] %}raw{{ x }}{% if x %}yes{% else %}no{% endif %}";
fn capacities(owner: &ParsedTemplate<'_>) -> [usize; 10] {
    [
        owner.storage.expressions.nodes.capacity(),
        owner.storage.expressions.operands.capacity(),
        owner.storage.expressions.segments.capacity(),
        owner.buffers.literals.capacity(),
        owner.buffers.numeric.capacity(),
        owner.expressions.capacity(),
        owner.storage.statements.capacity(),
        owner.storage.links.capacity(),
        owner.names.values.capacity(),
        owner.statements.capacity(),
    ]
}
const BUFFERS: [Buffer; 10] = [
    Buffer::Nodes,
    Buffer::Operands,
    Buffer::Segments,
    Buffer::Literals,
    Buffer::NumericScratch,
    Buffer::Frames,
    Buffer::Statements,
    Buffer::Bodies,
    Buffer::BlockNames,
    Buffer::StatementFrames,
];
#[test]
fn all_ten_actual_reserves_retain_real_prefixes_and_causes() {
    for (index, buffer) in BUFFERS.iter().copied().enumerate() {
        let error = Plan::inspect(FULL, "actual", Default::default())
            .unwrap()
            .construct_inner(Some(buffer))
            .unwrap_err();
        assert!(matches!(error.cause(),Cause::Reserve {buffer:b,..} if *b==buffer));
        assert!(std::error::Error::source(&error)
            .unwrap()
            .is::<std::collections::TryReserveError>());
        for (i, capacity) in capacities(&error.owner).iter().copied().enumerate() {
            if i < index {
                assert!(capacity > 0, "{buffer:?} predecessor {i}")
            } else {
                assert_eq!(capacity, 0, "{buffer:?} successor {i}")
            }
        }
        assert_eq!(error.source_text().as_ptr(), FULL.as_ptr());
        assert!(error.owner.storage.expressions.nodes.is_empty());
    }
}
#[test]
fn each_population_write_guard_is_reached_without_growth() {
    let mut cases = vec![
        (Buffer::Nodes, "{{ x }}"),
        (Buffer::Operands, "{% set (((x)))=1 %}"),
        (Buffer::Segments, "{{ 'x' }}"),
        (Buffer::Literals, "{{ 'x\\n' }}"),
        (Buffer::NumericScratch, "{{ 1_2 }}"),
        (Buffer::Frames, "{{ x }}"),
        (Buffer::Statements, "x"),
        (Buffer::Bodies, "x"),
        (Buffer::StatementFrames, "x"),
    ];
    #[cfg(feature = "multi_template")]
    cases.push((Buffer::BlockNames, "{% block b %}{% endblock %}"));
    for (buffer, source) in cases {
        let mut plan = Plan::inspect(source, "guard", Default::default()).unwrap();
        plan.ceiling = Some((buffer, 0));
        let error = plan.construct().unwrap_err();
        assert!(
            matches!(error.cause(),Cause::Capacity(b) if *b==buffer),
            "{buffer:?}: {error:?}"
        );
        assert!(error.owner.expressions.is_empty());
        assert!(error.owner.statements.is_empty());
    }
}
#[test]
fn late_errors_keep_actual_nodes_links_literal_numeric_and_names() {
    let source = "{% set x=['good\\n',12_3] %}raw{% set y= [x, bad + ] %}";
    let error = Plan::inspect(source, "late", Default::default())
        .unwrap()
        .construct()
        .unwrap_err();
    assert!(matches!(error.cause(), Cause::Syntax(_)));
    assert!(error.owner.storage.expressions.nodes.len() >= 6);
    assert!(!error.owner.storage.statements.is_empty());
    assert!(!error.owner.storage.links.is_empty());
    assert_eq!(error.owner.buffers.literals, b"good\n");
    assert_eq!(error.owner.buffers.numeric, b"123");
    assert_eq!(error.source_text(), source);
    #[cfg(feature = "multi_template")]
    {
        let error = Plan::inspect(
            "{% block b %}{{ 'prefix\\ud800' }}{% endblock %}",
            "late",
            Default::default(),
        )
        .unwrap()
        .construct()
        .unwrap_err();
        assert_eq!(error.owner.names.values, ["b"]);
        assert_eq!(error.owner.buffers.literals, b"prefix");
        assert!(matches!(error.cause(),Cause::Lexical(e) if e.kind()==ErrorKind::BadEscape));
    }
}
#[test]
fn flat_long_assignment_and_elif_destruction_preserve_logical_depth() {
    let source = format!("{{% set {}name{}=1 %}}", "(".repeat(3000), ")".repeat(3000));
    let owner = Plan::inspect(&source, "deep", Default::default())
        .unwrap()
        .construct()
        .unwrap();
    assert_eq!(owner.expression_count(), 2);
    assert_eq!(owner.storage.expressions.operands.len(), 3001);
    assert_eq!(owner.depth, 0);
    drop(owner);
    let source = format!("{{% if x %}}{}{{% endif %}}", "{% elif x %}".repeat(1800));
    let owner = Plan::inspect(&source, "deep", Default::default())
        .unwrap()
        .construct()
        .unwrap();
    assert_eq!(owner.statement_count(), 1802);
    assert_eq!(owner.depth, 0);
    drop(owner);
    for count in [5, 40, 80] {
        let source = format!(
            "{}x{}",
            "{% if x %}".repeat(count),
            "{% endif %}".repeat(count)
        );
        check(&source, Default::default());
    }
}
#[test]
fn empty_overflow_and_exact_source_controls_are_closed() {
    let owner = Plan::inspect("", "empty", Default::default())
        .unwrap()
        .construct()
        .unwrap();
    assert_eq!(owner.expression_count(), 0);
    assert_eq!(owner.statement_count(), 1);
    assert_eq!(owner.requirements.records(), 0);
    assert_eq!(owner.root_span(), Span::default());
    for values in [
        (usize::MAX, 0, 0),
        (0, usize::MAX, 0),
        (0, 0, usize::MAX),
        (isize::MAX as usize, 0, 0),
    ] {
        assert!(requirements(values.0, values.1, values.2).is_err());
    }
    assert!(owner.requirements.worker_control_bytes() > 0);
    assert!(owner.requirements.retained_control_bytes() >= size_of::<ParsedTemplate<'_>>());
}
#[test]
fn independent_concurrent_sources_and_external_unwind_retire_flat_owners() {
    let a = String::from(FULL);
    let b = a.clone();
    assert_ne!(a.as_ptr(), b.as_ptr());
    std::thread::scope(|scope| {
        let x = scope.spawn(|| {
            let owner = Plan::inspect(&a, "a", Default::default())
                .unwrap()
                .construct()
                .unwrap();
            assert_eq!(owner.source().as_ptr(), a.as_ptr());
            owner.statement_count()
        });
        let y = scope.spawn(|| {
            let owner = Plan::inspect(&b, "b", Default::default())
                .unwrap()
                .construct()
                .unwrap();
            assert_eq!(owner.source().as_ptr(), b.as_ptr());
            owner.statement_count()
        });
        assert_eq!(x.join().unwrap(), y.join().unwrap());
    });
    let result = std::panic::catch_unwind(|| {
        let _owner = Plan::inspect(FULL, "unwind", Default::default())
            .unwrap()
            .construct()
            .unwrap();
        panic!("external caller unwind")
    });
    assert!(result.is_err());
    check(FULL, Default::default());
}
#[cfg(feature = "custom_syntax")]
#[test]
fn ordinary_custom_syntax_stays_on_the_same_statement_worker() {
    let syntax = crate::syntax::SyntaxConfig::builder()
        .block_delimiters("<%", "%>")
        .variable_delimiters("<<", ">>")
        .build()
        .unwrap();
    let source = "<% set x=1 %><< x >><% if x %>yes<% endif %>";
    let a = reference::parse(source, "custom", syntax.clone(), Default::default()).unwrap();
    let b = parser::parse(source, "custom", syntax, Default::default()).unwrap();
    let (mut aa, mut bb) = (Vec::new(), Vec::new());
    ordinary(&a, &mut aa);
    ordinary(&b, &mut bb);
    assert_eq!(aa, bb);
}
