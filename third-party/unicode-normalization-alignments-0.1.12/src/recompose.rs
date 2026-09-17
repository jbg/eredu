// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use decompose::Decompositions;
use smallvec::SmallVec;
use std::fmt::{self, Write};

#[derive(Clone)]
pub(crate) enum RecompositionState {
    Composing,
    Purging(usize),
    Finished(usize),
}

/// External iterator for a string recomposition's characters.
#[derive(Clone)]
pub struct Recompositions<I> {
    iter: Decompositions<I>,
    state: RecompositionState,
    buffer: SmallVec<[(char, isize); 4]>,
    composee: Option<(char, isize)>,
    last_ccc: Option<u8>,
}

#[inline]
pub fn new_canonical<I: Iterator<Item = char>>(iter: I) -> Recompositions<I> {
    Recompositions {
        iter: super::decompose::new_canonical(iter),
        state: self::RecompositionState::Composing,
        buffer: SmallVec::new(),
        composee: None,
        last_ccc: None,
    }
}

#[inline]
pub fn new_compatible<I: Iterator<Item = char>>(iter: I) -> Recompositions<I> {
    Recompositions {
        iter: super::decompose::new_compatible(iter),
        state: self::RecompositionState::Composing,
        buffer: SmallVec::new(),
        composee: None,
        last_ccc: None,
    }
}

impl<I: Iterator<Item = char>> Iterator for Recompositions<I> {
    type Item = (char, isize);

    #[inline]
    fn next(&mut self) -> Option<(char, isize)> {
        next(
            &mut self.iter,
            &mut self.state,
            &mut self.buffer,
            &mut self.composee,
            &mut self.last_ccc,
        )
    }
}

pub(crate) trait RecompositionBuffer {
    fn get(&self, index: usize) -> Option<(char, isize)>;
    fn push(&mut self, item: (char, isize));
    fn clear(&mut self);
}
impl RecompositionBuffer for SmallVec<[(char, isize); 4]> {
    fn get(&self, index: usize) -> Option<(char, isize)> {
        self.as_slice().get(index).cloned()
    }
    fn push(&mut self, item: (char, isize)) {
        SmallVec::push(self, item)
    }
    fn clear(&mut self) {
        SmallVec::clear(self)
    }
}
pub(crate) fn next<I: Iterator<Item = (char, isize)>, B: RecompositionBuffer>(
    iter: &mut I,
    state: &mut RecompositionState,
    buffer: &mut B,
    composee: &mut Option<(char, isize)>,
    last_ccc: &mut Option<u8>,
) -> Option<(char, isize)> {
    use self::RecompositionState::*;

    loop {
        match *state {
            Composing => {
                for (ch, change) in iter.by_ref() {
                    let ch_class = super::char::canonical_combining_class(ch);
                    let k = match *composee {
                        None => {
                            if ch_class != 0 {
                                return Some((ch, change));
                            }
                            *composee = Some((ch, change));
                            continue;
                        }
                        Some(k) => k,
                    };
                    match *last_ccc {
                        None => match super::char::compose(k.0, ch) {
                            Some(r) => {
                                *composee = Some((r, k.1 + change - 1));
                                continue;
                            }
                            None => {
                                if ch_class == 0 {
                                    *composee = Some((ch, change));
                                    return Some(k);
                                }
                                buffer.push((ch, change));
                                *last_ccc = Some(ch_class);
                            }
                        },
                        Some(l_class) => {
                            if l_class >= ch_class {
                                // `ch` is blocked from `composee`
                                if ch_class == 0 {
                                    *composee = Some((ch, change));
                                    *last_ccc = None;
                                    *state = Purging(0);
                                    return Some(k);
                                }
                                buffer.push((ch, change));
                                *last_ccc = Some(ch_class);
                                continue;
                            }
                            match super::char::compose(k.0, ch) {
                                Some(r) => {
                                    *composee = Some((r, k.1 + change - 1));
                                    continue;
                                }
                                None => {
                                    buffer.push((ch, change));
                                    *last_ccc = Some(ch_class);
                                }
                            }
                        }
                    }
                }
                *state = Finished(0);
                if composee.is_some() {
                    return composee.take();
                }
            }
            Purging(next) => match buffer.get(next) {
                None => {
                    buffer.clear();
                    *state = Composing;
                }
                s => {
                    *state = Purging(next + 1);
                    return s;
                }
            },
            Finished(next) => match buffer.get(next) {
                None => {
                    buffer.clear();
                    return composee.take();
                }
                s => {
                    *state = Finished(next + 1);
                    return s;
                }
            },
        }
    }
}
impl<I: Iterator<Item = char> + Clone> fmt::Display for Recompositions<I> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for c in self.clone() {
            f.write_char(c.0)?;
        }
        Ok(())
    }
}
