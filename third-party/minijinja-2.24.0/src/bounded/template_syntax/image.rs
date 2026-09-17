#![forbid(unsafe_code)]
use super::*;
use crate::bounded::expression::store;
use crate::compiler::ast;
use crate::value::ValueRepr;
struct View<'a, 's> {
    storage: &'a store::Storage<'s>,
    buffers: &'a Buffers,
}
#[derive(Debug, PartialEq)]
pub(super) enum Atom {
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
pub(super) fn ordinary(expr: &ast::Expr<'_>, out: &mut Vec<Atom>) {
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
            match &value.expr {
                Some(expr) => ordinary(expr, out),
                None => out.push(Atom::None),
            };
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
fn sequence(owner: &View<'_, '_>, seq: store::Sequence, out: &mut Vec<Atom>) {
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
fn packed(owner: &View<'_, '_>, id: store::NodeId, out: &mut Vec<Atom>) {
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
            match expr {
                Some(expr) => packed(owner, *expr, out),
                None => out.push(Atom::None),
            };
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

pub(super) fn closed(owner: &ParsedTemplate<'_>, id: store::NodeId, out: &mut Vec<Atom>) {
    packed(
        &View {
            storage: &owner.storage.expressions,
            buffers: &owner.buffers,
        },
        id,
        out,
    )
}
