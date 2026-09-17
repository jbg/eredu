// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
use smallvec::SmallVec;
use std::fmt::{self, Write};
use std::iter::Fuse;
use std::ops::Range;

#[derive(Clone)]
pub(crate) enum DecompositionType {
    Canonical,
    Compatible,
}

/// External iterator for a string decomposition's characters.
#[derive(Clone)]
pub struct Decompositions<I> {
    kind: DecompositionType,
    iter: Fuse<I>,

    // This buffer stores pairs of (canonical combining class, character),
    // pushed onto the end in text order.
    //
    // It's divided into up to three sections:
    // 1) A prefix that is free space;
    // 2) "Ready" characters which are sorted and ready to emit on demand;
    // 3) A "pending" block which stills needs more characters for us to be able
    //    to sort in canonical order and is not safe to emit.
    buffer: SmallVec<[(u8, char, isize); 4]>,
    ready: Range<usize>,
}

#[inline]
pub fn new_canonical<I: Iterator<Item = char>>(iter: I) -> Decompositions<I> {
    Decompositions {
        kind: self::DecompositionType::Canonical,
        iter: iter.fuse(),
        buffer: SmallVec::new(),
        ready: 0..0,
    }
}

#[inline]
pub fn new_compatible<I: Iterator<Item = char>>(iter: I) -> Decompositions<I> {
    Decompositions {
        kind: self::DecompositionType::Compatible,
        iter: iter.fuse(),
        buffer: SmallVec::new(),
        ready: 0..0,
    }
}

pub(crate) trait DecompositionBuffer {
    fn len(&self) -> usize;
    fn get(&self, index: usize) -> (u8, char, isize);
    fn set(&mut self, index: usize, item: (u8, char, isize));
    fn push(&mut self, item: (u8, char, isize));
    fn truncate(&mut self, len: usize);
    fn sort_pending(&mut self, start: usize);
}
impl DecompositionBuffer for SmallVec<[(u8, char, isize); 4]> {
    fn len(&self) -> usize {
        SmallVec::len(self)
    }
    fn get(&self, index: usize) -> (u8, char, isize) {
        self[index]
    }
    fn set(&mut self, index: usize, item: (u8, char, isize)) {
        self[index] = item;
    }
    fn push(&mut self, item: (u8, char, isize)) {
        SmallVec::push(self, item)
    }
    fn truncate(&mut self, len: usize) {
        SmallVec::truncate(self, len)
    }
    fn sort_pending(&mut self, start: usize) {
        self[start..].sort_by_key(|k| k.0);
    }
}

#[inline]
fn push_back<B: DecompositionBuffer>(
    buffer: &mut B,
    ready: &mut Range<usize>,
    ch: char,
    first: bool,
) {
    let class = super::char::canonical_combining_class(ch);

    if class == 0 {
        sort_pending(buffer, ready);
    }

    buffer.push((class, ch, if first { 0 } else { 1 }));
}

#[inline]
fn sort_pending<B: DecompositionBuffer>(buffer: &mut B, ready: &mut Range<usize>) {
    // NB: `sort_by_key` is stable, so it will preserve the original text's
    // order within a combining class.
    buffer.sort_pending(ready.end);
    ready.end = buffer.len();
}

#[inline]
fn reset_buffer<B: DecompositionBuffer>(buffer: &mut B, ready: &mut Range<usize>) {
    // Equivalent to `buffer.drain(0..ready.end)` (if SmallVec
    // supported this API)
    let pending = buffer.len() - ready.end;
    for i in 0..pending {
        let item = buffer.get(i + ready.end);
        buffer.set(i, item);
    }
    buffer.truncate(pending);
    *ready = 0..0;
}

#[inline]
fn increment_next_ready<B: DecompositionBuffer>(buffer: &mut B, ready: &mut Range<usize>) {
    let next = ready.start + 1;
    if next == ready.end {
        reset_buffer(buffer, ready);
    } else {
        ready.start = next;
    }
}
impl<I: Iterator<Item = char>> Iterator for Decompositions<I> {
    type Item = (char, isize);

    #[inline]
    fn next(&mut self) -> Option<(char, isize)> {
        next(
            &self.kind,
            &mut self.iter,
            &mut self.buffer,
            &mut self.ready,
        )
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let (lower, _) = self.iter.size_hint();
        (lower, None)
    }
}

pub(crate) fn next<I: Iterator<Item = char>, B: DecompositionBuffer>(
    kind: &DecompositionType,
    iter: &mut Fuse<I>,
    buffer: &mut B,
    ready: &mut Range<usize>,
) -> Option<(char, isize)> {
    while ready.end == 0 {
        match (iter.next(), kind) {
            (Some(ch), &DecompositionType::Canonical) => {
                let mut first = true;
                super::char::decompose_canonical(ch, |d| {
                    push_back(buffer, ready, d, first);
                    first = false;
                });
            }
            (Some(ch), &DecompositionType::Compatible) => {
                let mut first = true;
                super::char::decompose_compatible(ch, |d| {
                    push_back(buffer, ready, d, first);
                    first = false;
                });
            }
            (None, _) => {
                if buffer.len() == 0 {
                    return None;
                } else {
                    sort_pending(buffer, ready);

                    // This implementation means that we can call `next`
                    // on an exhausted iterator; the last outer `next` call
                    // will result in an inner `next` call. To make this
                    // safe, we use `fuse`.
                    break;
                }
            }
        }
    }

    // We can assume here that, if `ready.end` is greater than zero,
    // it's also greater than `ready.start`. That's because we only
    // increment `ready.start` inside `increment_next_ready`, and
    // whenever it reaches equality with `ready.end`, we reset both
    // to zero, maintaining the invariant that:
    //      ready.start < ready.end || ready.end == ready.start == 0
    //
    // This less-than-obviously-safe implementation is chosen for performance,
    // minimizing the number & complexity of branches in `next` in the common
    // case of buffering then unbuffering a single character with each call.
    let (_, ch, size) = buffer.get(ready.start);
    increment_next_ready(buffer, ready);
    Some((ch, size))
}

impl<I: Iterator<Item = char> + Clone> fmt::Display for Decompositions<I> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for c in self.clone() {
            f.write_char(c.0)?;
        }
        Ok(())
    }
}
