pub(crate) mod containment;
pub(crate) mod disjoint;
pub(crate) mod entry;
pub(crate) mod grouping;
pub(crate) mod symbolic;
pub(crate) mod walk;
use crate::HashMap;
use anyhow::Result;

use crate::{
    ast::{ExprRef, ExprSet},
    raw::DerivCache,
};

// This is map (ByteSet => RegExp); ByteSet is Expr::Byte or Expr::ByteSet,
// and expresses the condition; RegExp is the derivative under that condition.
// Conditions can be overlapping, but should not repeat.
//
// deriv(R) = [(S0,C0),...,(Sn,Cn)]
//   means that
// deriv(byte, R) = Ca | Cb ... when byte ∈ Sa and byte ∈ Sb[byte]
// There is an implicit Sn+1 = Σ \ ⋃Si with Cn+1 = NO_MATCH
type SymRes = Vec<(ExprRef, ExprRef)>;

const DEBUG: bool = false;
macro_rules! debug {
    ($($arg:tt)*) => {
        if DEBUG {
            eprint!("  ");
            eprintln!($($arg)*);
        }
    };
}

/// Cache for relevance (language non-emptiness) and containment checks.
///
/// Relevance determines whether an expression can still match any string,
/// and containment checks whether one expression's language is a subset
/// of another's. Both are used during simplification.
#[derive(Clone)]
pub struct RelevanceCache {
    relevance_cache: HashMap<ExprRef, bool>,
    containment_cache: HashMap<(ExprRef, ExprRef), bool>,
    sym_deriv: HashMap<ExprRef, SymRes>,
    max_fuel: u64,
    cost_limit: u64,
}

fn swap_each<A: Copy>(v: &mut [(A, A)]) {
    for elem in v.iter_mut() {
        *elem = (elem.1, elem.0);
    }
}

fn simplify(exprs: &mut ExprSet, input: SymRes) -> SymRes {
    grouping::ordinary_simplify(exprs, input)
}

fn make_disjoint(exprs: &mut ExprSet, input: &SymRes) -> SymRes {
    disjoint::ordinary(exprs, input)
}

impl Default for RelevanceCache {
    fn default() -> Self {
        Self::new()
    }
}

impl RelevanceCache {
    pub fn new() -> Self {
        RelevanceCache {
            relevance_cache: HashMap::default(),
            sym_deriv: HashMap::default(),
            containment_cache: HashMap::default(),
            cost_limit: u64::MAX,
            max_fuel: u64::MAX,
        }
    }

    pub fn num_bytes(&self) -> usize {
        self.relevance_cache.len() * 3 * std::mem::size_of::<isize>()
    }

    pub(crate) fn deriv(&mut self, exprs: &mut ExprSet, e: ExprRef) -> SymRes {
        exprs.map(
            e,
            &mut self.sym_deriv,
            true,
            |e| e,
            |exprs, deriv, e| {
                let r = match symbolic::node(&mut symbolic::Ordinary(exprs), deriv, e) {
                    Ok(value) => value,
                    Err(never) => match never {},
                };
                debug!(
                    "deriv: {:?} -> {:?}",
                    exprs.expr_to_string(e),
                    r.iter()
                        .map(|(b, r)| (exprs.expr_to_string(*b), exprs.expr_to_string(*r)))
                        .collect::<Vec<_>>()
                );
                r
            },
        )
    }

    pub fn is_non_empty(&mut self, exprs: &mut ExprSet, top_expr: ExprRef) -> bool {
        self.is_non_empty_limited(exprs, top_expr, u64::MAX)
            .unwrap()
    }

    fn is_contained_in_prefixes_inner(
        &mut self,
        exprs: &mut ExprSet,
        deriv: &mut DerivCache,
        small: ExprRef,
        big: ExprRef,
    ) -> Result<bool> {
        containment::ordinary(self, exprs, deriv, small, big)
    }

    /// Check if `small` is contained in `big` with a limit on the number of steps.
    /// If `cache_failures` is true, then the result of the check is cached,
    /// as `false` in case of error (so future checks will return false not error,
    /// even if max_fuel is increased).
    pub fn is_contained_in_prefixes(
        &mut self,
        exprs: &mut ExprSet,
        deriv: &mut DerivCache,
        small: ExprRef,
        big: ExprRef,
        max_fuel: u64,
        cache_failures: bool,
    ) -> Result<bool> {
        if let Some(r) = self.containment_cache.get(&(small, big)) {
            return Ok(*r);
        }

        self.max_fuel = max_fuel;
        self.cost_limit = exprs.cost().saturating_add(max_fuel);

        match self.is_contained_in_prefixes_inner(exprs, deriv, small, big) {
            Ok(r) => {
                self.containment_cache.insert((small, big), r);
                Ok(r)
            }
            Err(e) => {
                if cache_failures {
                    self.containment_cache.insert((small, big), false);
                }
                Err(e)
            }
        }
    }

    pub fn is_non_empty_limited(
        &mut self,
        exprs: &mut ExprSet,
        top_expr: ExprRef,
        max_fuel: u64,
    ) -> Result<bool> {
        self.max_fuel = max_fuel;
        let cost_0 = exprs.cost();
        self.cost_limit = cost_0.saturating_add(max_fuel);
        let r = self.is_non_empty_inner(exprs, top_expr);
        if false {
            println!("cost: {}", exprs.cost() - cost_0);
        }
        r
    }

    fn is_non_empty_inner(&mut self, exprs: &mut ExprSet, top_expr: ExprRef) -> Result<bool> {
        entry::ordinary(self, exprs, top_expr)
    }
}
