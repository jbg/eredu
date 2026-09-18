pub(crate) mod byteset;
pub(crate) mod concat;
pub(crate) mod nary;
pub(crate) mod remainder;
mod scalar;
use crate::ast::{byteset_set, Expr, ExprFlags, ExprRef, ExprSet, ExprTag, PreparedExprError};
use crate::ParserAllocationFunding;

impl ExprSet {
    pub(crate) fn pay(&mut self, cost: usize) {
        self.cost += cost as u64;
    }

    pub fn byte_set_from_byte(&self, b: u8) -> Vec<u32> {
        let mut r = vec![0; self.alphabet_words];
        byteset_set(&mut r, b as usize);
        r
    }

    pub fn mk_byte(&mut self, b: u8) -> Result<ExprRef, PreparedExprError> {
        scalar::byte(&mut scalar::Prepared(self), b)
    }

    pub fn mk_byte_set(&mut self, s: &[u32]) -> Result<ExprRef, PreparedExprError> {
        scalar::byte_set(&mut scalar::Prepared(self), s)
    }

    pub fn mk_repeat(&mut self, e: ExprRef, min: u32, max: u32) -> Result<ExprRef, PreparedExprError> {
        scalar::repeat(&mut scalar::Prepared(self), e, min, max)
    }

    // Complexity of mk_X(args) is O(n log n) where n = |flatten(X, args)|

    pub(crate) fn mk_or_pair(&mut self, left: ExprRef, right: ExprRef) -> Result<ExprRef, PreparedExprError> {
        let mut args = Vec::new();
        self.construction_funding()?.try_extend_copy(&mut args, &[left, right])?;
        self.mk_or(&mut args)
    }
    pub(crate) fn mk_and_pair(&mut self, left: ExprRef, right: ExprRef) -> Result<ExprRef, PreparedExprError> {
        let mut args = Vec::new();
        self.construction_funding()?.try_extend_copy(&mut args, &[left, right])?;
        self.mk_and(&mut args)
    }
    pub fn mk_or(&mut self, args: &mut Vec<ExprRef>) -> Result<ExprRef, PreparedExprError> {
        nary::growing(self, args, ExprTag::Or)
    }

    fn or_optimized(&mut self, flags: ExprFlags, args: &mut [ExprRef]) -> Result<ExprRef, PreparedExprError> {
        let funding = self.construction_funding()?.clone();
        let mut args0 = Vec::new();
        funding.try_extend_copy(&mut args0, args)?;

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
            self.emit_prepared(Expr::Or(flags, &args0))
        } else {
            let mut pointers = Vec::new();
            funding.try_grow_vec(&mut pointers, args.len())?;
            pointers.extend(args.iter().map(|a| ConcatBytePointer::new(*a)));
            let mut args = pointers;
            self.optimize = false;
            let r = self.trie_rec(args.as_mut_slice(), 0);
            self.optimize = true;
            r
        }
    }

    pub fn mk_prefix_tree(&mut self, mut branches: Vec<(Vec<u8>, ExprRef)>) -> Result<ExprRef, PreparedExprError> {
        let funding = self.construction_funding()?.clone();
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
            (|| {
                let mut refs = Vec::new();
                funding.try_grow_vec(&mut refs, branches.len())?;
                for (p, e) in &branches { refs.push(self.mk_byte_concat(p, *e)?); }
                self.mk_or(&mut refs)
            })()
        } else {
            (|| {
                let mut args = Vec::new();
                funding.try_grow_vec(&mut args, branches.len())?;
                args.extend(branches.into_iter().map(|(prefix, tail)| ConcatBytePointer { prefix, ..ConcatBytePointer::new(tail) }));
                self.trie_rec(args.as_mut_slice(), 0)
            })()
        };

        self.optimize = prev_opt;

        r
    }

    // The idea is to optimize regexps like identifier1|identifier2|...|identifier50000
    // into a "trie" with shared prefixes;
    // for example: (foo|far|bar|baz) => (ba[rz]|f(oo|ar))
    fn trie_rec(&mut self, args: &mut [ConcatBytePointer], depth: usize) -> Result<ExprRef, PreparedExprError> {
        let funding = self.construction_funding()?.clone();
        if args.len() == 1 {
            return args[0].snapshot(self);
        }

        // limit recursion depth
        if depth > 100 {
            let mut snapshots = Vec::new();
            funding.try_grow_vec(&mut snapshots, args.len())?;
            for arg in args { snapshots.push(arg.snapshot(self)?); }
            return self.mk_or(&mut snapshots);
        }

        let mut common = vec![];
        let last_idx = args.len() - 1;
        loop {
            let a_0 = args[0].position;
            let a_end = args[last_idx].position;
            let a = args[0].next(self);
            let b = args[last_idx].next(self);
            if a != b {
                args[0].position = a_0;
                args[last_idx].position = a_end;
                break;
            }
            let a = a.unwrap();
            let b = b.unwrap();

            a.push_owned_to(&mut common, &funding)?;

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
                let alternative = self.trie_rec(&mut args[idx..next], depth + 1)?;
                funding.try_push(&mut alternatives, alternative)?;
            } else {
                funding.try_push(&mut alternatives, ExprRef::EMPTY_STRING)?;
            }

            idx = next;
        }

        let alts = self.mk_or(&mut alternatives)?;
        funding.try_push(&mut common, OwnedConcatElement::Expr(alts))?;
        self._mk_concat_vec(common)
    }

    pub fn mk_byte_set_not(&mut self, x: ExprRef) -> Result<ExprRef, PreparedExprError> { self.try_mk_byte_set_not(x) }
    pub fn mk_byte_set_or(&mut self, args: &[ExprRef]) -> Result<ExprRef, PreparedExprError> { self.try_mk_byte_set_or(args) }
    pub fn mk_byte_set_neg_or(&mut self, args: &[ExprRef]) -> Result<ExprRef, PreparedExprError> { self.try_mk_byte_set_neg_or(args) }
    pub fn mk_byte_set_and(&mut self, a: ExprRef, b: ExprRef) -> Result<ExprRef, PreparedExprError> { self.try_mk_byte_set_and(a, b) }
    pub fn mk_byte_set_sub(&mut self, a: ExprRef, b: ExprRef) -> Result<ExprRef, PreparedExprError> { self.try_mk_byte_set_sub(a, b) }

    pub fn mk_remainder_is(
        &mut self,
        divisor: u32,
        remainder: u32,
        scale: u32,
        fractional_part: bool,
    ) -> Result<ExprRef, PreparedExprError> {
        assert!(divisor > 0);
        assert!(remainder <= divisor);
        remainder::construct(&mut scalar::Prepared(self), divisor, remainder, scale, fractional_part)
    }

    // this avoids allocation when hitting the hash-cons
    pub(crate) fn mk_and2(&mut self, a: ExprRef, b: ExprRef) -> Result<ExprRef, PreparedExprError> {
        scalar::and2(&mut scalar::Prepared(self), a, b)
    }

    pub fn mk_and(&mut self, args: &mut Vec<ExprRef>) -> Result<ExprRef, PreparedExprError> {
        nary::growing(self, args, ExprTag::And)
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

    pub fn mk_concat_vec(&mut self, args: &[ExprRef]) -> Result<ExprRef, PreparedExprError> {
        let funding = self.construction_funding()?.clone();
        let mut expanded_args = Vec::new();
        funding.try_grow_vec(&mut expanded_args, args.len())?;
        for idx in 0..args.len() {
            let arg = args[idx];
            if idx == args.len() - 1 {
                if arg == ExprRef::NO_MATCH {
                    return Ok(ExprRef::NO_MATCH);
                } else if arg != ExprRef::EMPTY_STRING {
                    funding.try_push(&mut expanded_args, OwnedConcatElement::Expr(arg))?;
                }
            } else {
                // flatten everything except for the last element
                for a in self.iter_concat(arg) {
                    if !a.push_owned_to(&mut expanded_args, &funding)? {
                        return Ok(ExprRef::NO_MATCH);
                    }
                }
            }
        }

        self._mk_concat_vec(expanded_args)
    }

    pub(crate) fn _mk_concat_vec(&mut self, args: Vec<OwnedConcatElement>) -> Result<ExprRef, PreparedExprError> {
        concat::growing_fold(self, args)
    }

    pub fn mk_concat(&mut self, a: ExprRef, b: ExprRef) -> Result<ExprRef, PreparedExprError> {
        concat::growing(self, a, b)
    }

    pub fn mk_byte_concat(&mut self, s: &[u8], tail: ExprRef) -> Result<ExprRef, PreparedExprError> {
        scalar::byte_concat(&mut scalar::Prepared(self), s, tail)
    }

    pub fn mk_byte_literal(&mut self, s: &[u8]) -> Result<ExprRef, PreparedExprError> {
        self.mk_byte_concat(s, ExprRef::EMPTY_STRING)
    }

    pub fn mk_literal(&mut self, s: &str) -> Result<ExprRef, PreparedExprError> {
        self.mk_byte_literal(s.as_bytes())
    }

    pub fn mk_not(&mut self, e: ExprRef) -> Result<ExprRef, PreparedExprError> {
        scalar::not(&mut scalar::Prepared(self), e)
    }

    pub fn mk_lookahead(&mut self, e: ExprRef, offset: u32) -> Result<ExprRef, PreparedExprError> {
        scalar::lookahead(&mut scalar::Prepared(self), e, offset)
    }
}

pub enum ConcatElement<'a> {
    Expr(ExprRef),
    Bytes(&'a [u8]),
}

impl ConcatElement<'_> {
    pub fn push_owned_to(&self, out: &mut Vec<OwnedConcatElement>, funding: &ParserAllocationFunding) -> Result<bool, PreparedExprError> {
        concat::push_owned(out, self, funding)
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
    pub fn push_owned_to(&self, out: &mut Vec<OwnedConcatElement>, funding: &ParserAllocationFunding) -> Result<(), PreparedExprError> {
        let value = match self {
            Self::Byte(byte) => ConcatElement::Bytes(std::slice::from_ref(byte)),
            Self::Expr(expr) => ConcatElement::Expr(*expr),
        };
        if concat::push_owned(out, &value, funding)? { Ok(()) } else { Err(PreparedExprError::Source) }
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

// A cursor retains source IDs and offsets. Peeking/sorting/checkpointing never
// copies a byte buffer; an explicitly supplied prefix is moved into the cursor.
#[derive(Clone, Copy, Default)]
struct BytePosition {
    prefix_offset: usize,
    source: Option<ExprRef>,
    source_offset: usize,
    current: Option<ExprRef>,
}
struct ConcatBytePointer {
    prefix: Vec<u8>,
    position: BytePosition,
}
impl ConcatBytePointer {
    pub fn new(curr: ExprRef) -> Self {
        Self { prefix: Vec::new(), position: BytePosition { current: Some(curr), ..BytePosition::default() } }
    }
    pub fn peek(&self, exprs: &ExprSet) -> Option<ByteConcatElement> {
        let mut position = self.position;
        Self::advance(&self.prefix, &mut position, exprs)
    }
    pub fn next(&mut self, exprs: &ExprSet) -> Option<ByteConcatElement> {
        Self::advance(&self.prefix, &mut self.position, exprs)
    }
    fn advance(prefix: &[u8], position: &mut BytePosition, exprs: &ExprSet) -> Option<ByteConcatElement> {
        if let Some(&byte) = prefix.get(position.prefix_offset) {
            position.prefix_offset += 1;
            return Some(ByteConcatElement::Byte(byte));
        }
        if let Some(source) = position.source {
            let bytes = exprs.get_bytes(source).expect("cursor source retains byte encoding");
            if let Some(&byte) = bytes.get(position.source_offset) {
                position.source_offset += 1;
                return Some(ByteConcatElement::Byte(byte));
            }
            position.source = None;
        }
        let current = position.current?;
        let (head, tail) = match exprs.get(current) {
            Expr::Concat(_, [left, right]) => (left, Some(right)),
            Expr::ByteConcat(_, _, tail) => (current, Some(tail)),
            _ => (current, None),
        };
        position.current = tail;
        if let Some(bytes) = exprs.get_bytes(head) {
            position.source = Some(head);
            position.source_offset = 1;
            Some(ByteConcatElement::Byte(bytes[0]))
        } else {
            Some(ByteConcatElement::Expr(head))
        }
    }
    pub fn snapshot(&self, exprs: &mut ExprSet) -> Result<ExprRef, PreparedExprError> {
        let mut tail = self.position.current.unwrap_or(ExprRef::EMPTY_STRING);
        if let Some(source) = self.position.source {
            // ByteConcat encodings hold at most31 bytes. A stack copy allows
            // expression insertion without keeping a borrow into its arena.
            let bytes = &exprs.get_bytes(source).ok_or(PreparedExprError::Source)?[self.position.source_offset..];
            let mut copy = [0; ExprRef::MAX_BYTE_CONCAT];
            copy[..bytes.len()].copy_from_slice(bytes);
            let len = bytes.len();
            tail = exprs.mk_byte_concat(&copy[..len], tail)?;
        }
        exprs.mk_byte_concat(&self.prefix[self.position.prefix_offset..], tail)
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
