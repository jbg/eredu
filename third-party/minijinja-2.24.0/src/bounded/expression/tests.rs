use super::*;
use crate::compiler::{ast, parser};
use crate::value::ValueRepr;

mod reference;
include!("tests/fixtures.rs");

#[derive(Debug, PartialEq)]
enum Atom {
    Node(&'static str, Span),
    Name(String),
    Tag(&'static str),
    Count(usize),
    None,
    Bool(bool),
    Int(u64),
    Wide(u128),
    Float(u64),
    Text(String),
}

fn binary(op: ast::BinOpKind) -> &'static str {
    use ast::BinOpKind::*;
    match op {
        Eq => "eq",
        Ne => "ne",
        Lt => "lt",
        Lte => "lte",
        Gt => "gt",
        Gte => "gte",
        ScAnd => "and",
        ScOr => "or",
        Add => "add",
        Sub => "sub",
        Mul => "mul",
        Div => "div",
        FloorDiv => "floor",
        Rem => "rem",
        Pow => "pow",
        Concat => "concat",
        In => "in",
    }
}
fn comparison(op: ast::CompareOpKind) -> &'static str {
    use ast::CompareOpKind::*;
    match op {
        Eq => "eq",
        Ne => "ne",
        Lt => "lt",
        Lte => "lte",
        Gt => "gt",
        Gte => "gte",
        In => "in",
        NotIn => "not-in",
    }
}
fn unary(op: &ast::UnaryOpKind) -> &'static str {
    match op {
        ast::UnaryOpKind::Not => "not",
        ast::UnaryOpKind::Neg => "neg",
    }
}
fn ordinary_args(args: &[ast::CallArg<'_>], out: &mut Vec<Atom>) {
    out.push(Atom::Count(args.len()));
    for arg in args {
        let (tag, expr) = match arg {
            ast::CallArg::Pos(expr) => ("pos", expr),
            ast::CallArg::Kwarg(name, expr) => {
                out.push(Atom::Name((*name).into()));
                ("kwarg", expr)
            }
            ast::CallArg::PosSplat(expr) => ("splat", expr),
            ast::CallArg::KwargSplat(expr) => ("kwargs", expr),
        };
        out.push(Atom::Tag(tag));
        ordinary(expr, out);
    }
}
fn ordinary(expr: &ast::Expr<'_>, out: &mut Vec<Atom>) {
    use ast::Expr::*;
    let tag = match expr {
        Var(_) => "var",
        Const(_) => "const",
        Slice(_) => "slice",
        UnaryOp(_) => "unary",
        BinOp(_) => "binary",
        Compare(_) => "compare",
        IfExpr(_) => "if",
        Filter(_) => "filter",
        Test(_) => "test",
        GetAttr(_) => "attr",
        GetItem(_) => "item",
        Call(_) => "call",
        List(_) => "list",
        Map(_) => "map",
    };
    out.push(Atom::Node(tag, expr.span()));
    match expr {
        Var(value) => out.push(Atom::Name(value.id.into())),
        Const(value) => out.push(match &value.value.0 {
            ValueRepr::None => Atom::None,
            ValueRepr::Bool(value) => Atom::Bool(*value),
            ValueRepr::U64(value) => Atom::Int(*value),
            ValueRepr::U128(value) => Atom::Wide(value.0),
            ValueRepr::F64(value) => Atom::Float(value.to_bits()),
            ValueRepr::String(value, _) => Atom::Text(value.to_string()),
            ValueRepr::SmallStr(value) => Atom::Text(value.as_str().into()),
            _ => panic!("unexpected parser constant"),
        }),
        Slice(value) => {
            ordinary(&value.expr, out);
            for expr in [&value.start, &value.stop, &value.step] {
                match expr {
                    Some(expr) => ordinary(expr, out),
                    None => out.push(Atom::None),
                }
            }
        }
        UnaryOp(value) => {
            out.push(Atom::Tag(unary(&value.op)));
            ordinary(&value.expr, out);
        }
        BinOp(value) => {
            out.push(Atom::Tag(binary(value.op)));
            ordinary(&value.left, out);
            ordinary(&value.right, out);
        }
        Compare(value) => {
            ordinary(&value.expr, out);
            out.push(Atom::Count(value.ops.len()));
            for op in &value.ops {
                out.push(Atom::Tag(comparison(op.op)));
                ordinary(&op.expr, out);
            }
        }
        IfExpr(value) => {
            ordinary(&value.test_expr, out);
            ordinary(&value.true_expr, out);
            match &value.false_expr {
                Some(expr) => ordinary(expr, out),
                None => out.push(Atom::None),
            }
        }
        Filter(value) => {
            out.push(Atom::Name(value.name.into()));
            ordinary(value.expr.as_ref().unwrap(), out);
            ordinary_args(&value.args, out);
        }
        Test(value) => {
            out.push(Atom::Name(value.name.into()));
            ordinary(&value.expr, out);
            ordinary_args(&value.args, out);
        }
        GetAttr(value) => {
            out.push(Atom::Name(value.name.into()));
            ordinary(&value.expr, out);
        }
        GetItem(value) => {
            ordinary(&value.expr, out);
            ordinary(&value.subscript_expr, out);
        }
        Call(value) => {
            ordinary(&value.expr, out);
            ordinary_args(&value.args, out);
        }
        List(value) => {
            out.push(Atom::Count(value.items.len()));
            for expr in &value.items {
                ordinary(expr, out);
            }
        }
        Map(value) => {
            out.push(Atom::Count(value.keys.len()));
            for expr in &value.keys {
                ordinary(expr, out);
            }
            out.push(Atom::Count(value.values.len()));
            for expr in &value.values {
                ordinary(expr, out);
            }
        }
    }
}
fn sequence(owner: &ParsedExpression<'_>, seq: store::Sequence, out: &mut Vec<Atom>) {
    out.push(Atom::Count(seq.len));
    let mut current = seq.head;
    let mut count = 0;
    while let Some(index) = current {
        let operand = &owner.storage.operands[index];
        use store::OperandKind::*;
        let id = match operand.kind {
            Expr(id) => id,
            Binding(..) => unreachable!("standalone expression has no statement binding"),
            Pos(id) => {
                out.push(Atom::Tag("pos"));
                id
            }
            Kwarg(name, id) => {
                out.push(Atom::Name(name.into()));
                out.push(Atom::Tag("kwarg"));
                id
            }
            PosSplat(id) => {
                out.push(Atom::Tag("splat"));
                id
            }
            KwargSplat(id) => {
                out.push(Atom::Tag("kwargs"));
                id
            }
            Compare(op, id) => {
                out.push(Atom::Tag(comparison(op)));
                id
            }
        };
        packed(owner, id, out);
        current = operand.next;
        count += 1;
        assert!(count <= seq.len);
    }
    assert_eq!(count, seq.len);
}
fn packed(owner: &ParsedExpression<'_>, id: store::NodeId, out: &mut Vec<Atom>) {
    use store::RecordKind::*;
    let record = &owner.storage.nodes[id.0];
    let tag = match &record.kind {
        Var(_) => "var",
        Const(_) => "const",
        Slice { .. } => "slice",
        Unary(..) => "unary",
        Binary(..) => "binary",
        Compare(..) => "compare",
        If(..) => "if",
        Filter(..) => "filter",
        Test(..) => "test",
        Attr(..) => "attr",
        Item(..) => "item",
        Call(..) => "call",
        List(_) => "list",
        Map(..) => "map",
    };
    out.push(Atom::Node(tag, record.span));
    match &record.kind {
        Var(name) => out.push(Atom::Name((*name).into())),
        Const(value) => out.push(match value {
            store::Scalar::None => Atom::None,
            store::Scalar::Bool(value) => Atom::Bool(*value),
            store::Scalar::Int(value) => Atom::Int(*value),
            store::Scalar::Int128(value) => Atom::Wide(*value),
            store::Scalar::Float(value) => Atom::Float(value.to_bits()),
            store::Scalar::Text(joined) => {
                let mut value = String::new();
                let mut current = joined.sequence.head;
                let mut count = 0;
                while let Some(index) = current {
                    let segment = &owner.storage.segments[index];
                    match segment.kind {
                        store::SegmentKind::Source(text) => value.push_str(text),
                        store::SegmentKind::Decoded(bytes) => value.push_str(
                            std::str::from_utf8(&owner.buffers.literals[bytes.start..bytes.end])
                                .unwrap(),
                        ),
                    }
                    current = segment.next;
                    count += 1;
                    assert!(count <= joined.sequence.len);
                }
                assert_eq!(count, joined.sequence.len);
                assert_eq!(value.len(), joined.bytes);
                Atom::Text(value)
            }
        }),
        Slice {
            expr,
            start,
            stop,
            step,
        } => {
            packed(owner, *expr, out);
            for expr in [start, stop, step] {
                match expr {
                    Some(expr) => packed(owner, *expr, out),
                    None => out.push(Atom::None),
                }
            }
        }
        Unary(op, expr) => {
            out.push(Atom::Tag(unary(op)));
            packed(owner, *expr, out);
        }
        Binary(op, left, right) => {
            out.push(Atom::Tag(binary(*op)));
            packed(owner, *left, out);
            packed(owner, *right, out);
        }
        Compare(expr, ops) => {
            packed(owner, *expr, out);
            sequence(owner, *ops, out);
        }
        If(test, yes, no) => {
            packed(owner, *test, out);
            packed(owner, *yes, out);
            match no {
                Some(expr) => packed(owner, *expr, out),
                None => out.push(Atom::None),
            }
        }
        Filter(name, expr, args) => {
            out.push(Atom::Name((*name).into()));
            packed(owner, expr.expect("expression filter receiver"), out);
            sequence(owner, *args, out);
        }
        Test(name, expr, args) => {
            out.push(Atom::Name((*name).into()));
            packed(owner, *expr, out);
            sequence(owner, *args, out);
        }
        Attr(expr, name) => {
            out.push(Atom::Name((*name).into()));
            packed(owner, *expr, out);
        }
        Item(expr, subscript) => {
            packed(owner, *expr, out);
            packed(owner, *subscript, out);
        }
        Call(expr, args) => {
            packed(owner, *expr, out);
            sequence(owner, *args, out);
        }
        List(values) => sequence(owner, *values, out),
        Map(keys, values) => {
            sequence(owner, *keys, out);
            sequence(owner, *values, out);
        }
    }
}
fn errors(a: &crate::Error, b: &crate::Error) {
    assert_eq!(a.kind(), b.kind());
    assert_eq!(a.detail(), b.detail());
    assert_eq!(a.name(), b.name());
    assert_eq!(a.line(), b.line());
    #[cfg(feature = "debug")]
    assert_eq!(a.span(), b.span());
}
fn compare(source: &str) {
    let expected = reference::parse_expr(source);
    let ordinary_new = parser::parse_expr(source);
    let parsed = Plan::inspect(source, "<expression>").unwrap().construct();
    match (expected, ordinary_new, parsed) {
        (Ok(old), Ok(new), Ok(actual)) => {
            let (mut a, mut b, mut c) = (Vec::new(), Vec::new(), Vec::new());
            ordinary(&old, &mut a);
            ordinary(&new, &mut b);
            packed(&actual, actual.root.unwrap(), &mut c);
            assert_eq!(a, b, "ordinary: {source:?}");
            assert_eq!(a, c, "packed: {source:?}");
            assert_eq!(actual.source.as_ptr(), source.as_ptr());
        }
        (Err(old), Err(new), Err(actual)) => {
            errors(&old, &new);
            let (kind, detail, span) = match actual.cause() {
                Cause::Lexical(error) => (
                    error.kind(),
                    error.detail().map(str::to_owned),
                    error.span(),
                ),
                Cause::Syntax(error) => (
                    ErrorKind::SyntaxError,
                    Some(error.to_string()),
                    error.span(),
                ),
                _ => panic!("unexpected closed cause for {source:?}: {actual:?}"),
            };
            assert_eq!(kind, old.kind(), "{source:?}");
            assert_eq!(detail.as_deref(), old.detail(), "{source:?}");
            assert_eq!(span.map(|span| span.start_line as usize).filter(|line| *line > 0), old.line(), "{source:?}");
            assert_eq!(old.name(), span.map(|_| "<expression>"));
            #[cfg(feature = "debug")]
            assert_eq!(span, old.span(), "{source:?}");
        }
        _ => panic!("parser success/error disagreement: {source:?}"),
    }
}

#[test]
fn all_expression_shapes_precedence_spans_and_nested_sequences_match() {
    for source in [
        "user",
        "\n  x + -y\t\n",
        "π + 变量",
        "x|company . safe",
        "'é' '\\ud83d\\ude00'",
        "true",
        "False",
        "None",
        "42",
        "18446744073709551616",
        "1.25",
        "'a' 'b\\n' 'c'",
        "a or b and not c == d + e ~ f * g ** -h",
        "2 ** 3 ** 2",
        "not not not x",
        "---x",
        "x if y else z if q else r",
        "x if y",
        "a < b <= c not in d",
        "a not in b",
        "a == b",
        "user.name.0[1:9:2](v, k=3).result",
        "x[:]",
        "x[::]",
        "x[:2]",
        "x[1:]",
        "x[1::2]",
        "x[2]",
        "x|first|company.safe(a, k=3)",
        "x is not defined",
        "x is equalto 42",
        "x is equalto y.z and q",
        "f(a, *args, k=1, **kwargs)",
        "f(k=1, *args)",
        "f(a,)",
        "f()",
        "[]",
        "{}",
        "()",
        "(x)",
        "(x,)",
        "[x,[a,b],f(c,d),z]",
        "{'a':[1,2], 'b':{'c':3}, 'd':f(q)}",
        "f([a, b], k=g(x, z=2), **{'q': r})",
        "[x is equalto -y, x|f, x if y else z]",
        "x[1:2:3:4]",
        "x[]",
        "[a b]",
        "{'a' 3}",
        "f(k=1,a)",
        "f(*a,k=1,b)",
        "f(**a, b)",
        "f(=)",
    ] {
        compare(source);
    }
}

#[test]
fn literal_values_and_lazy_error_order_match_the_unchanged_parser() {
    for source in [
        "0b1_010",
        "0xFF_FF",
        "340282366920938463463374607431768211455",
        "340282366920938463463374607431768211456",
        "1e999",
        "1e-999",
        "2.2250738585072012e-308",
        "0.1000000000000000055511151231257827021181583404541015625",
        "'\\u+123'",
        "'\\ud83d\\ude00'",
        "'\\x+f'",
        "'\\777'",
        "'\\unknown'",
        "'\\ud800'",
        "'first' '\\ud800' '\\n\\t'",
        "'first' 1_ '\\n\\t'",
        "'first' 1_ 123_456_789",
        "[a, 1_ '\\n\\t']",
        "[a, '\\ud800' 123_456_789]",
        "'first' 1e '\\n\\t'",
        "a + ] '\\ud800'",
        "[a, b, '\\n', c + ]",
        "",
        "a @ '\\n'",
        "'unfinished",
    ] {
        compare(source);
    }
    let source = format!("1.{}1e-1_0", "0".repeat(900));
    compare(&source);
    let owner = Plan::inspect("'first' '\\ud800' '\\n\\t'", "<expression>")
        .unwrap()
        .construct()
        .unwrap();
    assert_eq!(owner.buffers.literals, b"\n\t");
    let owner = Plan::inspect("'first' 1_ 123_456_789", "<expression>")
        .unwrap()
        .construct()
        .unwrap();
    assert_eq!(owner.buffers.numeric, b"123456789");
}

#[test]
fn advancing_scanner_failures_count_and_stationary_failure_cannot_reveal_suffix() {
    use crate::compiler::lexer::scanner::{Scanner, Syntax};
    use crate::compiler::lexer::WhitespaceConfig;
    let source = "'first' 1_ 123_456_789";
    let plan = Plan::inspect(source, "fetches").unwrap();
    assert_eq!(plan.requirements().records(), 3);
    assert_eq!(plan.requirements().numeric_bytes(), 9);
    assert_eq!(plan.requirements().frames(), 4 * shared::ZERO_RUN_FRAMES);
    let mut scanner = Scanner::new(
        source,
        "fetches",
        true,
        Syntax::Default,
        WhitespaceConfig::default(),
    );
    assert!(scanner.next_token().unwrap().is_some());
    let before = scanner.progress();
    assert!(scanner.next_token().is_err());
    assert!(scanner.progress() != before);
    assert!(scanner.next_token().unwrap().is_some());
    assert!(scanner.next_token().unwrap().is_none());

    let mut scanner = Scanner::new(
        "@ '\\n' 12_34",
        "stationary",
        true,
        Syntax::Default,
        WhitespaceConfig::default(),
    );
    let before = scanner.progress();
    for _ in 0..4 {
        assert!(scanner.next_token().is_err());
        assert!(scanner.progress() == before);
    }
    let plan = Plan::inspect("@ '\\n' 12_34", "stationary").unwrap();
    assert_eq!(plan.requirements().records(), 1);
    assert_eq!(plan.requirements().literal_bytes(), 0);
    assert_eq!(plan.requirements().numeric_bytes(), 0);
    compare("'first' @ '\\n' 12_34");
}

#[test]
fn all_six_real_reserve_failures_retain_the_prior_destination_capacities() {
    let source = "f(['a\\n', 1_2], k={'x': other})";
    let buffers = [
        Buffer::Nodes,
        Buffer::Operands,
        Buffer::Segments,
        Buffer::Literals,
        Buffer::NumericScratch,
        Buffer::Frames,
    ];
    for (failed, buffer) in buffers.into_iter().enumerate() {
        let plan = Plan::inspect(source, "fixture").unwrap();
        let req = plan.requirements();
        let error = plan.construct_inner(Some(buffer)).unwrap_err();
        let Cause::Reserve {
            buffer: actual,
            error: cause,
        } = error.cause()
        else {
            panic!("wrong reserve cause")
        };
        assert_eq!(*actual, buffer);
        assert!(!cause.to_string().is_empty());
        let owner = &error.owner;
        let actual = [
            owner.storage.nodes.capacity(),
            owner.storage.operands.capacity(),
            owner.storage.segments.capacity(),
            owner.buffers.literals.capacity(),
            owner.buffers.numeric.capacity(),
            owner.frames.capacity(),
        ];
        let requested = [
            req.events,
            req.events,
            req.events,
            req.literal_bytes,
            req.numeric_bytes,
            req.frames,
        ];
        assert!(requested.iter().all(|&count| count > 0));
        for index in 0..6 {
            if index < failed {
                assert!(actual[index] >= requested[index]);
            } else {
                assert_eq!(actual[index], 0);
            }
        }
        assert_eq!(error.source_text().as_ptr(), source.as_ptr());
    }
}

#[test]
fn actual_write_guards_are_terminal_even_in_swallowing_string_lookahead() {
    for (buffer, source) in [
        (Buffer::Nodes, "v"),
        (Buffer::Operands, "[v]"),
        (Buffer::Segments, "'v'"),
        (Buffer::Literals, "'first' '\\n'"),
        (Buffer::NumericScratch, "'first' 1_2"),
        (Buffer::Frames, "v"),
    ] {
        let mut plan = Plan::inspect(source, "fixture").unwrap();
        plan.ceiling = Some((buffer, 0));
        let error = plan.construct().unwrap_err();
        assert!(
            matches!(error.cause(), Cause::Capacity(actual) if *actual == buffer),
            "{source:?}: {error:?}"
        );
        assert!(error.owner.frames.capacity() >= error.requirements().frames());
    }
}

#[test]
fn late_failures_retain_real_nonzero_syntax_literal_and_operand_prefixes() {
    let error = Plan::inspect("[first, 'good\\n', 1_2, bad + ]", "named")
        .unwrap()
        .construct()
        .unwrap_err();
    assert!(matches!(error.cause(), Cause::Syntax(_)));
    assert!(error.owner.storage.nodes.len() >= 4);
    assert!(error.owner.storage.operands.len() >= 3);
    assert_eq!(error.owner.buffers.literals, b"good\n");
    assert_eq!(error.owner.buffers.numeric, b"12");
    assert_eq!(error.owner.filename, "named");
    let error = Plan::inspect("[first, 'good\\n', 'prefix\\ud800']", "named")
        .unwrap()
        .construct()
        .unwrap_err();
    assert!(matches!(error.cause(), Cause::Lexical(value) if value.kind() == ErrorKind::BadEscape));
    assert_eq!(error.owner.buffers.literals, b"good\nprefix");
    assert!(!error.owner.storage.nodes.is_empty());
}

#[test]
fn logical_depth_and_argument_limits_match_without_recursive_closed_destruction() {
    for initial in [0, 147, 148, 149, 150] {
        for source in ["v", "(v)", "not not v", "v if q else r"] {
            let (a, da) = reference::depth_case(source, initial);
            let (b, db) = parser::expression_depth_case(source, initial);
            assert_eq!(da, db, "{source:?} depth {initial}");
            match (a, b) {
                (Err(a), Err(b)) => errors(&a, &b),
                (Ok(a), Ok(b)) => {
                    let (mut x, mut y) = (Vec::new(), Vec::new());
                    ordinary(&a, &mut x);
                    ordinary(&b, &mut y);
                    assert_eq!(x, y);
                }
                _ => panic!("depth result"),
            }
        }
    }
    for count in [2000, 2001] {
        compare(&format!("f({})", vec!["v"; count].join(",")));
    }
    let source = format!("{}v", "not ".repeat(1200));
    let parsed = Plan::inspect(&source, "deep").unwrap().construct().unwrap();
    assert_eq!(parsed.node_count(), 1201);
    assert!(parsed.frames.is_empty());
    drop(parsed);
    let source = vec!["v"; 1200].join("+");
    let parsed = Plan::inspect(&source, "deep").unwrap().construct().unwrap();
    assert_eq!(parsed.node_count(), 2399);
    drop(parsed);
}

#[test]
fn empty_geometry_overflow_and_independent_borrowed_sources_are_closed() {
    let plan = Plan::inspect("", "empty").unwrap();
    assert_eq!(plan.requirements().records(), 0);
    assert_eq!(plan.requirements().frames(), shared::ZERO_RUN_FRAMES);
    let error = plan.construct().unwrap_err();
    assert!(matches!(error.cause(), Cause::Syntax(_)));
    assert!(error.owner.storage.nodes.is_empty());
    assert!(requirements(usize::MAX, 0, 0).is_err());
    assert!(requirements(0, usize::MAX, 0).is_err());
    assert!(requirements(0, 0, usize::MAX).is_err());
    let a = String::from("f(v, k='a\\n')");
    let b = a.clone();
    assert_ne!(a.as_ptr(), b.as_ptr());
    std::thread::scope(|scope| {
        let x = scope.spawn(|| {
            let p = Plan::inspect(&a, "a").unwrap().construct().unwrap();
            assert_eq!(p.source.as_ptr(), a.as_ptr());
            p.node_count()
        });
        let y = scope.spawn(|| {
            let p = Plan::inspect(&b, "b").unwrap().construct().unwrap();
            assert_eq!(p.source.as_ptr(), b.as_ptr());
            p.node_count()
        });
        assert_eq!(x.join().unwrap(), y.join().unwrap());
    });
}

// Ordinary codegen emits each macro's capture HashSet in unspecified order.
// Preserve every index/line/span and all other operands; only the actual
// contiguous Enclose run immediately before GetClosure is a multiset here.
#[cfg(feature = "internal_debug")]
#[derive(Debug, PartialEq)]
enum CodePart {
    Exact(usize, String),
    #[cfg(feature = "macros")]
    Captures(usize, Vec<String>),
}
#[cfg(feature = "internal_debug")]
#[derive(Debug, PartialEq)]
struct InstructionImage {
    name: String,
    source: String,
    required: bool,
    locations: Vec<(Option<usize>, Option<Span>)>,
    parts: Vec<CodePart>,
}
#[cfg(feature = "internal_debug")]
fn instruction_image(code: &crate::compiler::instructions::Instructions<'_>) -> InstructionImage {
    #[cfg(feature = "multi_template")]
    let required = code.is_required_block();
    #[cfg(not(feature = "multi_template"))]
    let required = false;
    let locations = (0..code.len())
        .map(|i| (code.get_line(i as u32), code.get_span(i as u32)))
        .collect();
    let mut parts = Vec::new();
    let mut index = 0usize;
    while index < code.len() {
        #[cfg(feature = "macros")]
        if matches!(
            code.get(index as u32),
            Some(crate::compiler::instructions::Instruction::Enclose(_))
        ) {
            use crate::compiler::instructions::Instruction;
            let start = index;
            let mut names = Vec::new();
            while let Some(Instruction::Enclose(name)) = code.get(index as u32) {
                names.push((*name).to_owned());
                index += 1;
            }
            assert!(matches!(
                code.get(index as u32),
                Some(Instruction::GetClosure)
            ));
            names.sort_unstable();
            parts.push(CodePart::Captures(start, names));
            continue;
        }
        parts.push(CodePart::Exact(
            index,
            format!("{:?}", code.get(index as u32).unwrap()),
        ));
        index += 1;
    }
    InstructionImage {
        name: code.name().into(),
        source: code.source().into(),
        required,
        locations,
        parts,
    }
}

#[cfg(all(feature = "internal_debug", feature = "macros"))]
fn fixed_local_enclose_order_control() {
    use crate::compiler::instructions::Instruction;
    use crate::template::{CompiledTemplate, CompiledTemplateRef, TemplateConfig};
    let source = "{% set a=7 %}{% set b=11 %}{% set c=13 %}{% macro m() %}{{ a }}:{{ b }}:{{ c }}{% endmacro %}{{ m() }}";
    let config = TemplateConfig::new(std::sync::Arc::new(|_| crate::AutoEscape::None));
    let mut compiled = CompiledTemplate::new("capture-control", source, &config).unwrap();
    let env = crate::Environment::new();
    assert_eq!(
        crate::Template::new(&env, CompiledTemplateRef::Borrowed(&compiled))
            .render(())
            .unwrap(),
        "7:11:13"
    );
    let before = instruction_image(&compiled.instructions);
    let mut index = 0;
    let mut changed = None;
    while index < compiled.instructions.len() {
        let start = index;
        let mut names = Vec::new();
        while let Some(Instruction::Enclose(name)) = compiled.instructions.get(index as u32) {
            names.push(*name);
            index += 1;
        }
        if names.len() > 1 {
            names.reverse();
            for (offset, &name) in names.iter().enumerate() {
                *compiled
                    .instructions
                    .get_mut((start + offset) as u32)
                    .unwrap() = Instruction::Enclose(name);
            }
            changed = Some((start, names));
        }
        index += 1;
    }
    let (start, names) = changed.expect("actual multi-name compiler capture group");
    assert_eq!(names.len(), 3);
    assert_eq!(instruction_image(&compiled.instructions), before);
    assert_eq!(
        crate::Template::new(&env, CompiledTemplateRef::Borrowed(&compiled))
            .render(())
            .unwrap(),
        "7:11:13"
    );
    // Membership/multiplicity and surrounding branch operands remain strict.
    *compiled.instructions.get_mut(start as u32).unwrap() = Instruction::Enclose(names[1]);
    assert_ne!(instruction_image(&compiled.instructions), before);
    *compiled.instructions.get_mut(start as u32).unwrap() = Instruction::Enclose(names[0]);
    let jump = (0..compiled.instructions.len())
        .find(|&i| {
            matches!(
                compiled.instructions.get(i as u32),
                Some(Instruction::Jump(_))
            )
        })
        .unwrap();
    let Instruction::Jump(target) = compiled.instructions.get_mut(jump as u32).unwrap() else {
        unreachable!()
    };
    *target += 1;
    assert_ne!(instruction_image(&compiled.instructions), before);
}

#[test]
#[cfg(feature = "internal_debug")]
fn all_32_existing_full_templates_keep_ordinary_ast_and_instruction_geometry() {
    use crate::compiler::codegen::CodeGenerator;
    use crate::compiler::lexer::WhitespaceConfig;
    assert_eq!(FIXTURES.len(), 32);
    let mut successes = 0;
    for (name, source) in FIXTURES {
        for bits in 0..8 {
            let ws = WhitespaceConfig {
                keep_trailing_newline: bits & 1 != 0,
                lstrip_blocks: bits & 2 != 0,
                trim_blocks: bits & 4 != 0,
            };
            match (
                reference::parse(source, name, Default::default(), ws),
                parser::parse(source, name, Default::default(), ws),
            ) {
                (Ok(a), Ok(b)) => {
                    assert_eq!(format!("{a:?}"), format!("{b:?}"), "{name}");
                    let (mut ca, mut cb) = (
                        CodeGenerator::new(name, source),
                        CodeGenerator::new(name, source),
                    );
                    ca.compile_stmt(&a);
                    cb.compile_stmt(&b);
                    assert_eq!(ca.buffer_size_hint(), cb.buffer_size_hint());
                    let (a, ab) = ca.finish();
                    let (b, bb) = cb.finish();
                    assert_eq!(instruction_image(&a), instruction_image(&b), "{name}");
                    assert_eq!(ab.keys().collect::<Vec<_>>(), bb.keys().collect::<Vec<_>>());
                    for (key, instructions) in &ab {
                        assert_eq!(
                            instruction_image(instructions),
                            instruction_image(&bb[key]),
                            "{name}/{key}"
                        );
                    }
                    successes += 1;
                }
                (Err(a), Err(b)) => errors(&a, &b),
                _ => panic!("full template result: {name}"),
            }
        }
    }
    assert!(successes > 0);
    #[cfg(feature = "macros")]
    fixed_local_enclose_order_control();
}

#[test]
fn actual_ordinary_recursive_loop_formatter_strict_undefined_and_macro_use() {
    use std::fmt::Write as _;
    let mut env = crate::Environment::new();
    env.add_template("recursive", "{% for node in tree recursive %}{{ node.name }}{% if node.children %}({{ loop(node.children) }}){% endif %}{% endfor %}").unwrap();
    let leaf = crate::context! { name => "b", children => Vec::<crate::Value>::new() };
    let tree = vec![
        crate::context! { name => "a", children => vec![leaf] },
        crate::context! { name => "c", children => Vec::<crate::Value>::new() },
    ];
    assert_eq!(
        env.get_template("recursive")
            .unwrap()
            .render(crate::context! { tree => tree })
            .unwrap(),
        "a(b)c"
    );
    #[cfg(feature = "macros")]
    {
        env.add_template(
            "macro",
            "{% macro m(x=2) %}{{ x * 3 }}{% endmacro %}{{ m(4) }}",
        )
        .unwrap();
        assert_eq!(env.get_template("macro").unwrap().render(()).unwrap(), "12");
    }
    env.set_undefined_behavior(crate::UndefinedBehavior::Strict);
    env.set_formatter(|out, _state, value| {
        write!(out, "[{value}]").map_err(|_| crate::Error::from(ErrorKind::InvalidOperation))
    });
    env.add_template("formatted", "{{ 'a\\n' }}{{ f(2, k=3) }}")
        .unwrap();
    env.add_function(
        "f",
        |x: i64, kwargs: crate::value::Kwargs| -> Result<i64, crate::Error> {
            let k: i64 = kwargs.get("k")?;
            kwargs.assert_all_used()?;
            Ok(x + k)
        },
    );
    assert_eq!(
        env.get_template("formatted").unwrap().render(()).unwrap(),
        "[a\n][5]"
    );
    env.add_template("arithmetic", "{{ missing + 1 }}").unwrap();
    let arithmetic = env.get_template("arithmetic").unwrap().render(()).unwrap_err();
    let old = crate::compiler::fold::scalar_reference::ops::add(
        &crate::Value::UNDEFINED,
        &crate::Value::from(1),
    ).unwrap_err();
    assert_eq!(arithmetic.kind(), old.kind());
    assert_eq!(arithmetic.detail(), old.detail());
    assert_eq!(old.kind(), ErrorKind::InvalidOperation);
    env.add_template("strict", "{{ missing.attribute }}").unwrap();
    assert_eq!(
        env.get_template("strict")
            .unwrap()
            .render(())
            .unwrap_err()
            .kind(),
        ErrorKind::UndefinedError
    );
}

#[cfg(all(feature = "custom_syntax", feature = "internal_debug"))]
#[test]
fn ordinary_custom_syntax_uses_the_same_expression_worker_and_original_ast() {
    let syntax = crate::syntax::SyntaxConfig::builder()
        .block_delimiters("<%", "%>")
        .variable_delimiters("<<", ">>")
        .line_statement_prefix("#")
        .line_comment_prefix("##")
        .build()
        .unwrap();
    let source = "## hidden\n# set x = 1_2\n<< x + 3 >> <% if x > 10 %>yes<% endif %>";
    let a = reference::parse(source, "custom", syntax.clone(), Default::default()).unwrap();
    let b = parser::parse(source, "custom", syntax.clone(), Default::default()).unwrap();
    assert_eq!(format!("{a:?}"), format!("{b:?}"));
    let mut env = crate::Environment::new();
    env.set_syntax(syntax);
    env.add_template("custom", source).unwrap();
    assert_eq!(
        env.get_template("custom").unwrap().render(()).unwrap(),
        "15 yes"
    );
}

#[test]
fn empty_lexer_state_stays_a_closed_rejection_and_ordinary_panic() {
    let source = "v}}tail";
    assert!(std::panic::catch_unwind(|| reference::parse_expr(source)).is_err());
    assert!(std::panic::catch_unwind(|| parser::parse_expr(source)).is_err());
    let error = Plan::inspect(source, "fixture")
        .unwrap()
        .construct()
        .unwrap_err();
    assert!(matches!(error.cause(), Cause::ExpressionEnd));
    assert!(error.owner.frames.capacity() > 0);
}
