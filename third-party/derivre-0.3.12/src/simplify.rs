pub(crate) mod byteset;
pub(crate) mod concat;
pub(crate) mod nary;
pub(crate) mod remainder;
mod scalar;
use crate::ast::{byteset_set, Expr, ExprFlags, ExprRef, ExprSet, ExprTag};

impl ExprSet {
    pub(crate) fn pay(&mut self, cost: usize) {
        self.cost += cost as u64;
    }

    pub fn byte_set_from_byte(&self, b: u8) -> Vec<u32> {
        let mut r = vec![0; self.alphabet_words];
        byteset_set(&mut r, b as usize);
        r
    }

    pub fn mk_byte(&mut self, b: u8) -> ExprRef {
        scalar::ordinary(self, |s| scalar::byte(s, b))
    }

    pub fn mk_byte_set(&mut self, s: &[u32]) -> ExprRef {
        scalar::ordinary(self, |sink| scalar::byte_set(sink, s))
    }

    pub fn mk_repeat(&mut self, e: ExprRef, min: u32, max: u32) -> ExprRef {
        scalar::ordinary(self, |s| scalar::repeat(s, e, min, max))
    }

    // Complexity of mk_X(args) is O(n log n) where n = |flatten(X, args)|

    pub fn mk_or(&mut self, args: &mut Vec<ExprRef>) -> ExprRef {
        nary::ordinary(self, args, ExprTag::Or)
    }

    fn or_optimized(&mut self, flags: ExprFlags, args: &mut [ExprRef]) -> ExprRef {
        let args0 = args.to_vec();

        args.sort_unstable_by(|&a, &b| self.iter_concat_bytes(a).cmp(self.iter_concat_bytes(b)));

        let mut prev = None;
        let mut has_double = false;
        for c in args.iter() {
            let c0 = self.iter_concat_bytes(*c).next();
            if c0 == prev {
                has_double = true;
                break;
            }
            prev = c0;
        }
        if !has_double {
            self.mk(Expr::Or(flags, &args0))
        } else {
            self.optimize = false;
            let mut args = args
                .iter()
                .map(|a| ConcatBytePointer::new(*a))
                .collect::<Vec<_>>();
            let r = self.trie_rec(args.as_mut_slice(), 0);
            self.optimize = true;
            r
        }
    }

    pub fn mk_prefix_tree(&mut self, mut branches: Vec<(Vec<u8>, ExprRef)>) -> ExprRef {
        branches.sort_unstable_by(|a, b| a.0.cmp(&b.0));

        let mut prev = None;
        let mut has_double = false;
        for c in branches.iter() {
            let c0 = c.0.first();
            if c0 == prev {
                has_double = true;
                break;
            }
            prev = c0;
        }

        let prev_opt = self.optimize;
        self.optimize = false;

        let r = if !has_double {
            let mut refs = branches
                .iter()
                .map(|(p, e)| self.mk_byte_concat(p, *e))
                .collect::<Vec<_>>();
            self.mk_or(&mut refs)
        } else {
            let mut args = branches
                .into_iter()
                .map(|a| ConcatBytePointer {
                    pending: a.0,
                    pending_ptr: 0,
                    current: Some(a.1),
                })
                .collect::<Vec<_>>();
            self.trie_rec(args.as_mut_slice(), 0)
        };

        self.optimize = prev_opt;

        r
    }

    // The idea is to optimize regexps like identifier1|identifier2|...|identifier50000
    // into a "trie" with shared prefixes;
    // for example: (foo|far|bar|baz) => (ba[rz]|f(oo|ar))
    fn trie_rec(&mut self, args: &mut [ConcatBytePointer], depth: usize) -> ExprRef {
        if args.len() == 1 {
            return args[0].snapshot(self);
        }

        // limit recursion depth
        if depth > 100 {
            let mut args = args.iter().map(|a| a.snapshot(self)).collect::<Vec<_>>();
            return self.mk_or(&mut args);
        }

        let mut common = vec![];
        let last_idx = args.len() - 1;
        loop {
            let a_0 = args[0].clone();
            let a_end = args[last_idx].clone();
            let a = args[0].next(self);
            let b = args[last_idx].next(self);
            if a != b {
                args[0] = a_0;
                args[last_idx] = a_end;
                break;
            }
            let a = a.unwrap();
            let b = b.unwrap();

            a.push_owned_to(&mut common);

            // assert!(a != ExprRef::EMPTY_STRING);
            for arg in &mut args[1..last_idx] {
                let a = arg.next(self).unwrap();
                assert!(a == b);
            }
        }
        assert!(depth == 0 || !common.is_empty());

        let mut idx = 0;

        let mut alternatives = vec![];
        while idx < args.len() {
            let cur = args[idx].peek(self);
            let mut next = idx + 1;
            while next < args.len() && args[next].peek(self) == cur {
                next += 1;
            }

            if cur.is_some() {
                alternatives.push(self.trie_rec(&mut args[idx..next], depth + 1));
            } else {
                alternatives.push(ExprRef::EMPTY_STRING);
            }

            idx = next;
        }

        let alts = self.mk_or(&mut alternatives);
        common.push(OwnedConcatElement::Expr(alts));
        self._mk_concat_vec(common)
    }

    pub fn mk_byte_set_not(&mut self, x: ExprRef) -> ExprRef {
        byteset::ordinary(self, |sink, memory| byteset::not(sink, memory, x))
    }
    pub fn mk_byte_set_or(&mut self, args: &[ExprRef]) -> ExprRef {
        byteset::ordinary(self, |sink, memory| {
            byteset::union(sink, memory, args, false)
        })
    }
    pub fn mk_byte_set_neg_or(&mut self, args: &[ExprRef]) -> ExprRef {
        byteset::ordinary(self, |sink, memory| {
            byteset::union(sink, memory, args, true)
        })
    }
    pub fn mk_byte_set_and(&mut self, a: ExprRef, b: ExprRef) -> ExprRef {
        byteset::ordinary(self, |sink, memory| {
            byteset::intersection(sink, memory, a, b)
        })
    }
    pub fn mk_byte_set_sub(&mut self, a: ExprRef, b: ExprRef) -> ExprRef {
        byteset::ordinary(self, |sink, memory| byteset::subtract(sink, memory, a, b))
    }

    pub fn mk_remainder_is(
        &mut self,
        divisor: u32,
        remainder: u32,
        scale: u32,
        fractional_part: bool,
    ) -> ExprRef {
        assert!(divisor > 0);
        assert!(remainder <= divisor);
        scalar::ordinary(self, |sink| {
            remainder::construct(sink, divisor, remainder, scale, fractional_part)
        })
    }

    // this avoids allocation when hitting the hash-cons
    pub(crate) fn mk_and2(&mut self, a: ExprRef, b: ExprRef) -> ExprRef {
        scalar::ordinary(self, |sink| scalar::and2(sink, a, b))
    }

    pub fn mk_and(&mut self, args: &mut Vec<ExprRef>) -> ExprRef {
        nary::ordinary(self, args, ExprTag::And)
    }

    pub fn iter_concat(&self, root: ExprRef) -> ConcatIter<'_> {
        ConcatIter {
            exprs: self,
            current: Some(root),
        }
    }

    pub fn iter_concat_bytes(&self, root: ExprRef) -> ConcatByteIter<'_> {
        ConcatByteIter {
            exprs: self,
            pointer: ConcatBytePointer::new(root),
        }
    }

    fn is_concat(&self, e: ExprRef) -> bool {
        let tag = self.get_tag(e);
        tag == ExprTag::Concat || tag == ExprTag::ByteConcat
    }

    pub fn mk_concat_vec(&mut self, args: &[ExprRef]) -> ExprRef {
        let mut expanded_args = Vec::with_capacity(args.len());
        for idx in 0..args.len() {
            let arg = args[idx];
            if idx == args.len() - 1 {
                if arg == ExprRef::NO_MATCH {
                    return ExprRef::NO_MATCH;
                } else if arg != ExprRef::EMPTY_STRING {
                    expanded_args.push(OwnedConcatElement::Expr(arg));
                }
            } else {
                // flatten everything except for the last element
                for a in self.iter_concat(arg) {
                    if !a.push_owned_to(&mut expanded_args) {
                        return ExprRef::NO_MATCH;
                    }
                }
            }
        }

        self._mk_concat_vec(expanded_args)
    }

    pub(crate) fn _mk_concat_vec(&mut self, args: Vec<OwnedConcatElement>) -> ExprRef {
        concat::ordinary_fold(self, args)
    }

    pub fn mk_concat(&mut self, a: ExprRef, b: ExprRef) -> ExprRef {
        concat::ordinary(self, a, b)
    }

    pub fn mk_byte_concat(&mut self, s: &[u8], tail: ExprRef) -> ExprRef {
        scalar::ordinary(self, |sink| scalar::byte_concat(sink, s, tail))
    }

    pub fn mk_byte_literal(&mut self, s: &[u8]) -> ExprRef {
        self.mk_byte_concat(s, ExprRef::EMPTY_STRING)
    }

    pub fn mk_literal(&mut self, s: &str) -> ExprRef {
        self.mk_byte_literal(s.as_bytes())
    }

    pub fn mk_not(&mut self, e: ExprRef) -> ExprRef {
        scalar::ordinary(self, |s| scalar::not(s, e))
    }

    pub fn mk_lookahead(&mut self, e: ExprRef, offset: u32) -> ExprRef {
        scalar::ordinary(self, |s| scalar::lookahead(s, e, offset))
    }
}

pub enum ConcatElement<'a> {
    Expr(ExprRef),
    Bytes(&'a [u8]),
}

impl ConcatElement<'_> {
    pub fn push_owned_to(&self, out: &mut Vec<OwnedConcatElement>) -> bool {
        concat::push_owned(out, self)
    }
}

pub enum OwnedConcatElement {
    Expr(ExprRef),
    Bytes(Vec<u8>),
}

#[derive(PartialEq, Eq, Debug, PartialOrd, Ord)]
pub enum ByteConcatElement {
    Byte(u8),
    Expr(ExprRef),
}

impl ByteConcatElement {
    pub fn push_owned_to(&self, out: &mut Vec<OwnedConcatElement>) {
        match self {
            ByteConcatElement::Byte(b) => match out.last_mut() {
                Some(OwnedConcatElement::Bytes(ref mut exp)) => {
                    exp.push(*b);
                }
                _ => {
                    out.push(OwnedConcatElement::Bytes(vec![*b]));
                }
            },
            ByteConcatElement::Expr(e) => {
                if *e == ExprRef::NO_MATCH {
                    panic!();
                }
                if *e != ExprRef::EMPTY_STRING {
                    out.push(OwnedConcatElement::Expr(*e));
                }
            }
        }
    }
}

pub struct ConcatIter<'a> {
    exprs: &'a ExprSet,
    current: Option<ExprRef>,
}

pub struct ConcatByteIter<'a> {
    exprs: &'a ExprSet,
    pointer: ConcatBytePointer,
}

impl Iterator for ConcatByteIter<'_> {
    type Item = ByteConcatElement;

    fn next(&mut self) -> Option<Self::Item> {
        self.pointer.next(self.exprs)
    }
}

#[derive(Clone)]
struct ConcatBytePointer {
    pending_ptr: usize,
    pending: Vec<u8>,
    current: Option<ExprRef>,
}

impl ConcatBytePointer {
    pub fn new(curr: ExprRef) -> Self {
        ConcatBytePointer {
            pending_ptr: 0,
            pending: Vec::new(),
            current: Some(curr),
        }
    }

    pub fn peek(&self, exprset: &ExprSet) -> Option<ByteConcatElement> {
        let mut copy = self.clone();
        copy.next(exprset)
    }

    pub fn next(&mut self, exprset: &ExprSet) -> Option<ByteConcatElement> {
        if self.pending_ptr < self.pending.len() {
            let b = self.pending[self.pending_ptr];
            self.pending_ptr += 1;
            return Some(ByteConcatElement::Byte(b));
        }

        let curr = self.current?;

        let mut it = exprset.iter_concat(curr);
        let tmp = it.next();
        self.current = it.current;
        match tmp {
            Some(ConcatElement::Bytes(bytes)) => {
                let b0 = bytes[0];
                self.pending = bytes[1..].to_vec();
                self.pending_ptr = 0;
                Some(ByteConcatElement::Byte(b0))
            }
            Some(ConcatElement::Expr(expr)) => Some(ByteConcatElement::Expr(expr)),
            None => None,
        }
    }

    pub fn snapshot(&self, exprset: &mut ExprSet) -> ExprRef {
        let tail = self.current.unwrap_or(ExprRef::EMPTY_STRING);
        if self.pending_ptr >= self.pending.len() {
            tail
        } else {
            exprset.mk_byte_concat(&self.pending[self.pending_ptr..], tail)
        }
    }
}

pub(crate) fn next_concat<'a>(
    exprs: &'a ExprSet,
    current: &mut Option<ExprRef>,
) -> Option<ConcatElement<'a>> {
    let curr = (*current)?;
    let expr = exprs.get(curr);
    match expr {
        Expr::Concat(_, [l, r]) => {
            *current = Some(r);
            if let Some(bytes) = exprs.get_bytes(l) {
                Some(ConcatElement::Bytes(bytes))
            } else {
                Some(ConcatElement::Expr(l))
            }
        }
        Expr::ByteConcat(_, bytes, tail) => {
            *current = Some(tail);
            Some(ConcatElement::Bytes(bytes))
        }
        _ => {
            *current = None;
            if let Some(bytes) = exprs.get_bytes(curr) {
                Some(ConcatElement::Bytes(bytes))
            } else {
                Some(ConcatElement::Expr(curr))
            }
        }
    }
}
impl<'a> Iterator for ConcatIter<'a> {
    type Item = ConcatElement<'a>;
    fn next(&mut self) -> Option<Self::Item> {
        next_concat(self.exprs, &mut self.current)
    }
}
