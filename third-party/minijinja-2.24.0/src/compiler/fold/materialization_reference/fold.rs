//! Test-only ordinary pre-refactor folding, with receiver/recursive-name adapters only.
#![forbid(unsafe_code)]
use super::{self as scalar_reference, ops};
use crate::compiler::ast::*;
use crate::value::{value_map_with_capacity, Value, ValueKind, ValueRepr};
use crate::{Error, ErrorKind};
const MIN_I128_AS_POS_U128: u128 = 170141183460469231731687303715884105728;
pub(crate) fn expression(expr: &Expr<'_>) -> Option<Value> {
    match expr {
        Expr::Const(c) => Some(c.value.clone()),
        Expr::List(l) => list(l),
        Expr::Map(m) => map(m),
        Expr::UnaryOp(c) => match c.op {
            UnaryOpKind::Not => expression(&c.expr).map(|value| Value::from(!is_true(&value))),
            UnaryOpKind::Neg => expression(&c.expr).and_then(|v| neg(&v).ok()),
        },
        Expr::BinOp(c) => {
            let (Some(left), Some(right)) = (expression(&c.left), expression(&c.right)) else {
                return None;
            };
            eval_binop(c.op, &left, &right)
        }
        Expr::Compare(c) => {
            let mut left = expression(&c.expr)?;
            for op in &c.ops {
                let right = expression(&op.expr)?;
                if !is_true(&eval_compare(op.op, &left, &right)?) {
                    return Some(Value::from(false));
                }
                left = right;
            }
            Some(Value::from(true))
        }
        _ => None,
    }
}

pub(crate) fn list(list: &List<'_>) -> Option<Value> {
    if !list.items.iter().all(|x| matches!(x, Expr::Const(_))) {
        return None;
    }

    let items = list.items.iter();
    let sequence = items.filter_map(|expr| match expr {
        Expr::Const(v) => Some(v.value.clone()),
        _ => None,
    });

    Some(Value::from(sequence.collect::<Vec<_>>()))
}

pub(crate) fn map(map: &Map<'_>) -> Option<Value> {
    if !map.keys.iter().all(|x| matches!(x, Expr::Const(_)))
        || !map.values.iter().all(|x| matches!(x, Expr::Const(_)))
    {
        return None;
    }

    let mut rv = value_map_with_capacity(map.keys.len());
    for (key, value) in map.keys.iter().zip(map.values.iter()) {
        if let (Expr::Const(maybe_key), Expr::Const(value)) = (key, value) {
            rv.insert(maybe_key.value.clone(), value.value.clone());
        }
    }

    Some(Value::from_object(rv))
}

fn eval_binop(op: BinOpKind, left: &Value, right: &Value) -> Option<Value> {
    match op {
        BinOpKind::Add => ops::add(left, right).ok(),
        BinOpKind::Sub => ops::sub(left, right).ok(),
        BinOpKind::Mul => ops::mul(left, right).ok(),
        BinOpKind::Div => ops::div(left, right).ok(),
        BinOpKind::FloorDiv => ops::int_div(left, right).ok(),
        BinOpKind::Rem => ops::rem(left, right).ok(),
        BinOpKind::Pow => ops::pow(left, right).ok(),
        BinOpKind::Concat => Some(ops::string_concat(left.clone(), right)),
        BinOpKind::Eq => Some(Value::from(scalar_reference::equal(left, right))),
        BinOpKind::Ne => Some(Value::from(!scalar_reference::equal(left, right))),
        BinOpKind::Lt => Some(Value::from(scalar_reference::compare(left, right).is_lt())),
        BinOpKind::Lte => Some(Value::from(!scalar_reference::compare(left, right).is_gt())),
        BinOpKind::Gt => Some(Value::from(scalar_reference::compare(left, right).is_gt())),
        BinOpKind::Gte => Some(Value::from(!scalar_reference::compare(left, right).is_lt())),
        BinOpKind::In => ops::contains(right, left).ok(),
        BinOpKind::ScAnd => Some(if is_true(left) && is_true(right) {
            right.clone()
        } else {
            Value::from(false)
        }),
        BinOpKind::ScOr => Some(if is_true(left) {
            left.clone()
        } else {
            right.clone()
        }),
    }
}

fn eval_compare(op: CompareOpKind, left: &Value, right: &Value) -> Option<Value> {
    match op {
        CompareOpKind::Eq => Some(Value::from(scalar_reference::equal(left, right))),
        CompareOpKind::Ne => Some(Value::from(!scalar_reference::equal(left, right))),
        CompareOpKind::Lt => Some(Value::from(scalar_reference::compare(left, right).is_lt())),
        CompareOpKind::Lte => Some(Value::from(!scalar_reference::compare(left, right).is_gt())),
        CompareOpKind::Gt => Some(Value::from(scalar_reference::compare(left, right).is_gt())),
        CompareOpKind::Gte => Some(Value::from(!scalar_reference::compare(left, right).is_lt())),
        CompareOpKind::In => ops::contains(right, left).ok(),
        CompareOpKind::NotIn => ops::contains(right, left)
            .ok()
            .map(|value| Value::from(!is_true(&value))),
    }
}

pub fn neg(val: &Value) -> Result<Value, Error> {
    if val.kind() == ValueKind::Number {
        match val.0 {
            ValueRepr::F64(x) => Ok((-x).into()),
            // special case for the largest i128 that can still be
            // represented.
            ValueRepr::U128(x) if x.0 == MIN_I128_AS_POS_U128 => {
                Ok(Value::from(MIN_I128_AS_POS_U128))
            }
            _ => {
                if let Ok(x) = i128::try_from(val.clone()) {
                    x.checked_mul(-1)
                        .ok_or_else(|| Error::new(ErrorKind::InvalidOperation, "overflow"))
                        .map(int_as_value)
                } else {
                    Err(Error::from(ErrorKind::InvalidOperation))
                }
            }
        }
    } else {
        Err(Error::from(ErrorKind::InvalidOperation))
    }
}

fn int_as_value(val: i128) -> Value {
    if val as i64 as i128 == val {
        (val as i64).into()
    } else {
        val.into()
    }
}

pub(crate) fn is_true(value: &Value) -> bool {
    match value.0 {
        ValueRepr::Bool(val) => val,
        ValueRepr::U64(x) => x != 0,
        ValueRepr::U128(x) => x.0 != 0,
        ValueRepr::I64(x) => x != 0,
        ValueRepr::I128(x) => x.0 != 0,
        ValueRepr::F64(x) => x != 0.0,
        ValueRepr::String(ref x, _) => !x.is_empty(),
        ValueRepr::SmallStr(ref x) => !x.is_empty(),
        ValueRepr::Bytes(ref x) => !x.is_empty(),
        ValueRepr::None | ValueRepr::Undefined(_) | ValueRepr::Invalid(_) => false,
        ValueRepr::Object(ref x) => x.is_true(),
    }
}
