//! Shared symbolic derivative order with explicit expression/list destinations.
pub(crate) mod storage;
use super::SymRes;
use crate::ast::{Expr, ExprRef, ExprSet};
use crate::ParserError as ConstructionError;

pub(crate) trait Construction {
    type Error;
    fn source(&self) -> &ExprSet;
    fn list(&mut self, capacity: usize) -> Result<SymRes, Self::Error>;
    fn push(&mut self, list: &mut SymRes, value: (ExprRef, ExprRef)) -> Result<(), Self::Error>;
    fn overflow(&self) -> Self::Error;
    fn pay(&mut self, amount: usize) -> Result<(), Self::Error>;
    fn literal_derivative(&mut self, root: ExprRef) -> Result<(ExprRef, ExprRef), Self::Error>;
    fn byte(&mut self, value: u8) -> Result<ExprRef, Self::Error>;
    fn remainder(
        &mut self,
        divisor: u32,
        remainder: u32,
        scale: u32,
        fractional: bool,
    ) -> Result<ExprRef, Self::Error>;
    fn multiply(&mut self, left: u32, right: u32) -> Result<u32, Self::Error>;
    fn add(&mut self, left: u32, right: u32) -> Result<u32, Self::Error>;
    fn power10(&mut self, scale: u32) -> Result<u32, Self::Error>;
    fn modulo(&mut self, value: u32, divisor: u32) -> Result<u32, Self::Error>;
    fn byte_and(&mut self, a: ExprRef, b: ExprRef) -> Result<ExprRef, Self::Error>;
    fn and(&mut self, a: ExprRef, b: ExprRef) -> Result<ExprRef, Self::Error>;
    fn not(&mut self, arg: ExprRef) -> Result<ExprRef, Self::Error>;
    fn repeat(&mut self, arg: ExprRef, min: u32, max: u32) -> Result<ExprRef, Self::Error>;
    fn concat(&mut self, left: ExprRef, right: ExprRef) -> Result<ExprRef, Self::Error>;
    fn simplify(&mut self, list: SymRes) -> Result<SymRes, Self::Error>;
    fn disjoint(&mut self, list: &SymRes) -> Result<SymRes, Self::Error>;
    fn negated_union(&mut self, list: &SymRes) -> Result<ExprRef, Self::Error>;
}

pub(crate) fn node<C: Construction>(
    c: &mut C,
    children: &mut Vec<SymRes>,
    root: ExprRef,
) -> Result<SymRes, C::Error> {
    let result = match c.source().get(root) {
        Expr::EmptyString | Expr::NoMatch => c.list(0)?,
        Expr::Byte(_) | Expr::ByteSet(_) => {
            let mut result = c.list(1)?;
            c.push(&mut result, (root, ExprRef::EMPTY_STRING))?;
            result
        }
        Expr::ByteConcat(_, _, _) => {
            // The ordinary source copies the suffix, creates its selector, then
            // constructs the suffix expression before allocating the result list.
            let pair = c.literal_derivative(root)?;
            let mut result = c.list(1)?;
            c.push(&mut result, pair)?;
            result
        }
        Expr::Lookahead(_, _, _) => children.pop().expect("mapped lookahead child"),
        Expr::RemainderIs {
            divisor,
            remainder,
            scale,
            fractional_part,
        } => {
            let capacity = if !fractional_part && scale > 0 {
                11
            } else {
                10
            };
            let mut result = c.list(capacity)?;
            let digits = c.source().digits;
            for (i, byte) in digits.into_iter().enumerate() {
                let selector = c.byte(byte)?;
                let (value, next_scale) = if !fractional_part {
                    (c.multiply(remainder, 10)?, scale)
                } else {
                    if scale == 0 {
                        c.push(&mut result, (selector, ExprRef::NO_MATCH))?;
                        continue;
                    }
                    (remainder, scale - 1)
                };
                let power = c.power10(next_scale)?;
                let term = c.multiply(i as u32, power)?;
                let sum = c.add(value, term)?;
                let rem = c.modulo(sum, divisor)?;
                let expression = c.remainder(divisor, rem, next_scale, fractional_part)?;
                c.push(&mut result, (selector, expression))?;
            }
            if !fractional_part && scale > 0 {
                let selector = c.byte(c.source().digit_dot)?;
                let expression = c.remainder(divisor, remainder, scale, true)?;
                c.push(&mut result, (selector, expression))?;
            }
            result
        }
        Expr::And(_, _) => {
            let mut accumulator = children.pop().expect("mapped intersection child");
            while let Some(other) = children.pop() {
                let capacity = accumulator
                    .len()
                    .checked_mul(other.len())
                    .ok_or_else(|| c.overflow())?;
                let mut next = c.list(capacity)?;
                for &(b0, r0) in &accumulator {
                    for &(b1, r1) in &other {
                        let selector = c.byte_and(b0, b1)?;
                        if selector != ExprRef::NO_MATCH {
                            let expression = c.and(r0, r1)?;
                            if expression != ExprRef::NO_MATCH {
                                c.push(&mut next, (selector, expression))?;
                            }
                        }
                    }
                }
                accumulator = next;
            }
            c.simplify(accumulator)?
        }
        Expr::Or(_, _) => {
            let capacity = children
                .iter()
                .try_fold(0usize, |n, child| n.checked_add(child.len()))
                .ok_or_else(|| c.overflow())?;
            let mut result = c.list(capacity)?;
            for child in children.drain(..) {
                for pair in child {
                    c.push(&mut result, pair)?;
                }
            }
            c.simplify(result)?
        }
        Expr::Not(_, _) => {
            let disjoint = c.disjoint(&children[0])?;
            let capacity = disjoint.len().checked_add(1).ok_or_else(|| c.overflow())?;
            let mut result = c.list(capacity)?;
            for &(selector, expression) in &disjoint {
                let expression = c.not(expression)?;
                c.push(&mut result, (selector, expression))?;
            }
            let left_over = c.negated_union(&children[0])?;
            if left_over != ExprRef::NO_MATCH {
                c.push(&mut result, (left_over, ExprRef::ANY_BYTE_STRING))?;
            }
            c.simplify(result)?
        }
        Expr::Repeat(_, arg, min, max) => {
            let max = if max == u32::MAX {
                u32::MAX
            } else {
                max.saturating_sub(1)
            };
            let tail = c.repeat(arg, min.saturating_sub(1), max)?;
            let mut result = c.list(children[0].len())?;
            for &(selector, expression) in &children[0] {
                let expression = c.concat(expression, tail)?;
                c.push(&mut result, (selector, expression))?;
            }
            c.simplify(result)?
        }
        Expr::Concat(_, [left, right]) => {
            let nullable = c.source().is_nullable(left);
            let additional = if nullable { children[1].len() } else { 0 };
            let capacity = children[0]
                .len()
                .checked_add(additional)
                .ok_or_else(|| c.overflow())?;
            let mut result = c.list(capacity)?;
            for &(selector, expression) in &children[0] {
                let expression = c.concat(expression, right)?;
                c.push(&mut result, (selector, expression))?;
            }
            if nullable {
                for &pair in &children[1] {
                    c.push(&mut result, pair)?;
                }
            }
            c.simplify(result)?
        }
    };
    c.pay(result.len())?;
    Ok(result)
}

pub(super) struct Ordinary<'a>(pub(super) &'a mut ExprSet);
impl Construction for Ordinary<'_> {
    type Error = ConstructionError;
    fn source(&self) -> &ExprSet {
        self.0
    }
    fn list(&mut self, capacity: usize) -> Result<SymRes, ConstructionError> {
        let mut list = Vec::new();
        self.0.construction_funding()?.try_grow_vec(&mut list, capacity)?;
        Ok(list)
    }
    fn push(&mut self, list: &mut SymRes, value: (ExprRef, ExprRef)) -> Result<(), ConstructionError> {
        self.0.construction_funding()?.try_push(list, value)?;
        Ok(())
    }
    fn overflow(&self) -> ConstructionError {
        crate::raw::PreparedExprError::Capacity.into()
    }
    fn pay(&mut self, amount: usize) -> Result<(), ConstructionError> {
        self.0.pay_prepared(amount)?;
        Ok(())
    }
    fn literal_derivative(&mut self, root: ExprRef) -> Result<(ExprRef, ExprRef), ConstructionError> {
        let Expr::ByteConcat(_, bytes, tail) = self.0.get(root) else {
            unreachable!("symbolic literal source")
        };
        let first = bytes[0];
        let mut copy = [0; ExprRef::MAX_BYTE_CONCAT];
        let len = bytes.len() - 1;
        copy[..len].copy_from_slice(&bytes[1..]);
        let selector = self.0.mk_byte(first)?;
        let expression = self.0.mk_byte_concat(&copy[..len], tail)?;
        Ok((selector, expression))
    }
    fn byte(&mut self, value: u8) -> Result<ExprRef, ConstructionError> {
        Ok(self.0.mk_byte(value)?)
    }
    fn remainder(
        &mut self,
        divisor: u32,
        remainder: u32,
        scale: u32,
        fractional: bool,
    ) -> Result<ExprRef, ConstructionError> {
        Ok(self
            .0
            .mk_remainder_is(divisor, remainder, scale, fractional)?)
    }
    fn multiply(&mut self, left: u32, right: u32) -> Result<u32, ConstructionError> {
        Ok(left.checked_mul(right).ok_or(crate::raw::PreparedExprError::Source)?)
    }
    fn add(&mut self, left: u32, right: u32) -> Result<u32, ConstructionError> {
        Ok(left.checked_add(right).ok_or(crate::raw::PreparedExprError::Source)?)
    }
    fn power10(&mut self, scale: u32) -> Result<u32, ConstructionError> {
        Ok(10u32.checked_pow(scale).ok_or(crate::raw::PreparedExprError::Source)?)
    }
    fn modulo(&mut self, value: u32, divisor: u32) -> Result<u32, ConstructionError> {
        Ok(value.checked_rem(divisor).ok_or(crate::raw::PreparedExprError::Source)?)
    }
    fn byte_and(&mut self, a: ExprRef, b: ExprRef) -> Result<ExprRef, ConstructionError> {
        Ok(self.0.mk_byte_set_and(a, b)?)
    }
    fn and(&mut self, a: ExprRef, b: ExprRef) -> Result<ExprRef, ConstructionError> {
        Ok(self.0.mk_and_pair(a, b)?)
    }
    fn not(&mut self, arg: ExprRef) -> Result<ExprRef, ConstructionError> {
        Ok(self.0.mk_not(arg)?)
    }
    fn repeat(&mut self, arg: ExprRef, min: u32, max: u32) -> Result<ExprRef, ConstructionError> {
        Ok(self.0.mk_repeat(arg, min, max)?)
    }
    fn concat(&mut self, left: ExprRef, right: ExprRef) -> Result<ExprRef, ConstructionError> {
        Ok(self.0.mk_concat(left, right)?)
    }
    fn simplify(&mut self, list: SymRes) -> Result<SymRes, ConstructionError> {
        super::simplify(self.0, list)
    }
    fn disjoint(&mut self, list: &SymRes) -> Result<SymRes, ConstructionError> {
        super::make_disjoint(self.0, list)
    }
    fn negated_union(&mut self, list: &SymRes) -> Result<ExprRef, ConstructionError> {
        let funding = self.0.construction_funding()?.clone();
        let mut selectors = Vec::new();
        funding.try_grow_vec(&mut selectors, list.len())?;
        selectors.extend(list.iter().map(|&(selector, _)| selector));
        Ok(self.0.mk_byte_set_neg_or(&selectors)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Debug)]
    struct Refusal {
        completed: ExprRef,
        issue: Option<storage::Issue>,
    }
    struct Refusing<'a> {
        ordinary: Ordinary<'a>,
        lists: storage::Lists,
        remaining: usize,
        completed: Option<ExprRef>,
    }
    fn lift<T>(result: Result<T, ConstructionError>) -> Result<T, Refusal> {
        match result {
            Ok(value) => Ok(value),
            Err(error) => panic!("unenforced fixture constructor: {error}"),
        }
    }
    macro_rules! forward {
        ($name:ident($($arg:ident:$ty:ty),*) -> $result:ty) => {
            fn $name(&mut self,$($arg:$ty),*)->Result<$result,Refusal> {
                lift(self.ordinary.$name($($arg),*))
            }
        };
    }
    impl Construction for Refusing<'_> {
        type Error = Refusal;
        fn source(&self) -> &ExprSet {
            self.ordinary.source()
        }
        fn overflow(&self) -> Refusal {
            panic!("test source has representable list geometry")
        }
        fn list(&mut self, capacity: usize) -> Result<SymRes, Refusal> {
            self.lists.take(capacity).map_err(|issue| Refusal {
                completed: self.completed.unwrap_or(ExprRef::NO_MATCH),
                issue: Some(issue),
            })
        }
        forward!(push(list:&mut SymRes,value:(ExprRef,ExprRef))->());
        forward!(pay(amount:usize)->());
        forward!(literal_derivative(root:ExprRef)->(ExprRef,ExprRef));
        forward!(byte(value:u8)->ExprRef);
        forward!(multiply(left:u32,right:u32)->u32);
        forward!(add(left:u32,right:u32)->u32);
        forward!(power10(scale:u32)->u32);
        forward!(modulo(value:u32,divisor:u32)->u32);
        forward!(byte_and(a:ExprRef,b:ExprRef)->ExprRef);
        forward!(and(a:ExprRef,b:ExprRef)->ExprRef);
        forward!(not(arg:ExprRef)->ExprRef);
        forward!(repeat(arg:ExprRef,min:u32,max:u32)->ExprRef);
        forward!(concat(left:ExprRef,right:ExprRef)->ExprRef);
        forward!(simplify(list:SymRes)->SymRes);
        forward!(disjoint(list:&SymRes)->SymRes);
        forward!(negated_union(list:&SymRes)->ExprRef);
        fn remainder(
            &mut self,
            divisor: u32,
            remainder: u32,
            scale: u32,
            fractional: bool,
        ) -> Result<ExprRef, Refusal> {
            let completed = lift(
                self.ordinary
                    .remainder(divisor, remainder, scale, fractional),
            )?;
            self.completed = Some(completed);
            self.remaining -= 1;
            if self.remaining == 0 {
                Err(Refusal {
                    completed,
                    issue: None,
                })
            } else {
                Ok(completed)
            }
        }
    }
    #[test]
    fn shared_symbolic_remainder_keeps_digit_order_and_post_constructor_failure_prefix() {
        let mut source = ExprSet::new(256, crate::ParserAllocationFunding::unenforced()).unwrap();
        let root = source.mk_remainder_is(13, 3, 1, false).unwrap();
        let mut failed_source = source.clone();
        let mut cache = super::super::RelevanceCache::new();
        let result = cache.deriv(&mut source, root).unwrap();
        assert_eq!(result.len(), 11);
        for (i, &(selector, expression)) in result[..10].iter().enumerate() {
            assert!(source.get(selector).matches_byte(b'0' + i as u8));
            assert!(
                matches!(source.get(expression),Expr::RemainderIs{divisor:13,remainder,scale:1,fractional_part:false} if remainder==(30+i as u32*10)%13)
            );
        }
        assert!(source.get(result[10].0).matches_byte(b'.'));
        assert_eq!(result[10].1, ExprRef::NO_MATCH);
        let cost = source.cost();
        assert_eq!(cache.deriv(&mut source, root).unwrap(), result);
        assert_eq!(source.cost(), cost);

        // Byte-concat mapping deliberately skips the tail child. Its actual
        // reached source therefore has zero child lists, though the AST has one.
        let literal = source.mk_byte_literal(b"ab").unwrap();
        let mut literal_lists = storage::Plan::prepare(&source, literal, &[])
            .unwrap()
            .compile()
            .unwrap();
        assert_eq!(literal_lists.take(1).unwrap().capacity(), 1);

        let start = failed_source.len();
        let cost = failed_source.cost();
        let plan = storage::Plan::prepare(&failed_source, root, &[]).unwrap();
        let requirements = plan.requirements();
        assert_eq!(
            requirements.total,
            requirements.buffers + requirements.controls
        );
        assert!(requirements.buffers >= 11 * std::mem::size_of::<(ExprRef, ExprRef)>());
        let lists = plan.compile().unwrap();
        let mut consumer = Refusing {
            ordinary: Ordinary(&mut failed_source),
            lists,
            remaining: 3,
            completed: None,
        };
        let failure = node(&mut consumer, &mut Vec::new(), root).unwrap_err();
        assert_eq!(consumer.remaining, 0);
        assert!(failure.issue.is_none());
        assert!(matches!(
            consumer.lists.take(0),
            Err(storage::Issue::Exhausted)
        ));
        assert!(matches!(
            consumer.lists.take(0),
            Err(storage::Issue::Failed)
        ));
        assert_eq!(consumer.completed, Some(failure.completed));
        assert!(consumer.source().len() > start);
        assert!(consumer.source().cost() > cost);
        assert!(matches!(
            consumer.source().get(failure.completed),
            Expr::RemainderIs {
                divisor: 13,
                remainder: 11,
                scale: 1,
                fractional_part: false
            }
        ));
        drop(consumer);
        assert!(failed_source.is_valid(failure.completed));
    }
}
