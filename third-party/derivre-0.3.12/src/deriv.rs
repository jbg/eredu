use crate::SourceHashMap as HashMap;

use crate::ast::{ExprRef, ExprSet};

mod construction;
pub(crate) mod storage;

const DEBUG: bool = false;
macro_rules! debug {
    ($($arg:tt)*) => {
        if DEBUG {
            eprint!("  ");
            eprintln!($($arg)*);
        }
    };
}

/// Memoization cache for derivative computations.
///
/// Stores previously computed derivatives keyed by `(ExprRef, byte)` pairs to
/// avoid redundant work during DFA construction.
#[derive(Clone)]
pub struct DerivCache {
    pub num_deriv: usize,
    state_table: HashMap<(ExprRef, u8), ExprRef>,
}

impl Default for DerivCache {
    fn default() -> Self {
        Self::new()
    }
}

impl DerivCache {
    pub fn new() -> Self {
        DerivCache {
            num_deriv: 0,
            state_table: HashMap::default(),
        }
    }

    pub fn derivative(&mut self, exprs: &mut ExprSet, r: ExprRef, b: u8) -> crate::ParserResult<ExprRef> {
        // This kicks in for lexers with lots of keywords, that is regexps that are
        // just concats of single bytes.
        // Most of these do not match, so this provides very significant speedup.
        // TODO add a flag on exprs to see if this even applies?
        if construction::surely_no_match(exprs, r, b) {
            return Ok(ExprRef::NO_MATCH);
        }

        let mut or_branches = vec![];

        // regular path
        exprs.map(
            r,
            &mut self.state_table,
            true,
            |r| (r, b),
            |exprs, deriv, r| {
                self.num_deriv += 1;
                let d = construction::node(
                    &mut construction::Ordinary {
                        expressions: exprs,
                        alternatives: &mut or_branches,
                    },
                    deriv,
                    r,
                    b,
                )?;
                debug!(
                    "deriv({}) via {} = {}",
                    exprs.expr_to_string(r),
                    exprs.pp().byte_to_string(b),
                    exprs.expr_to_string(d)
                );
                Ok(d)
            },
        )
    }

    /// Estimate the size of the regex tables in bytes.
    pub fn num_bytes(&self) -> usize {
        self.state_table.len() * 8 * std::mem::size_of::<isize>()
    }
}
