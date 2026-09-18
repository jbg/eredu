//! Borrowed normalization planning and one canonical NFC/Prepend producer.
//!
//! NFC is idempotent and preserves canonical equivalence under concatenation.
//! Therefore all NFC operations can be represented by the final NFC, with
//! preceding prefixes inside it and subsequent prefixes outside it. Prepend
//! operations are reversed within each group, exactly as sequential prepending.
use super::{absent, add, err, fields, required, text_is, Cause, Error, K};
use crate::{
    normalizers::{prepend::Prepend, utils::Sequence, NormalizerWrapper as N, NFC},
    utils::borrowed_json::{self as json, Reader, Span, Text},
};
use std::{alloc::Layout, mem::size_of};

#[derive(Clone, Copy, Debug)]
enum Leaf<'a> {
    Nfc,
    Prepend(Text<'a>),
    Lowercase,
    Replace(Text<'a>, Text<'a>),
}
#[derive(Debug)]
pub(super) struct Plan<'a> {
    input: &'a str,
    source: Option<Span>,
    leaves: usize,
    last_nfc: Option<usize>,
    before: usize,
    after: usize,
    literal: Option<(Text<'a>, Text<'a>)>,
    ordered: bool,
    #[cfg(any(test, feature = "tokenizer-compiler-test-support"))]
    failure: Option<usize>,
}
impl<'a> Plan<'a> {
    pub(super) fn inspect(input: &'a str, source: Option<Span>) -> Result<Self, Error> {
        let source = absent(input, source);
        let mut out = Self {
            input,
            source,
            leaves: 0,
            last_nfc: None,
            before: 0,
            after: 0,
            literal: None,
            ordered: false,
            #[cfg(any(test, feature = "tokenizer-compiler-test-support"))]
            failure: None,
        };
        if let Some(span) = source {
            let mut values = Reader::new(input, span).object()?;
            let mut is_literal = false;
            while let Some((key, value)) = values.next()? {
                if key.is("type") {
                    is_literal = text_is(input, value, "Replace")?;
                }
            }
            if is_literal {
                let f = fields(input, span, ["type", "pattern", "content"])?;
                let pattern = required(f[1], span.start)?;
                let p = fields(input, pattern, ["String"])?;
                let read = |span| -> Result<Text<'a>, Error> {
                    let mut reader = Reader::new(input, span);
                    let text = reader.string()?;
                    reader.finish()?;
                    Ok(text)
                };
                out.literal = Some((
                    read(required(p[0], pattern.start)?)?,
                    read(required(f[2], span.start)?)?,
                ));
                return Ok(out);
            }
        }
        let mut leaves = 0;
        let mut last_nfc = None;
        let mut ordered = false;
        out.visit(|leaf| {
            if matches!(leaf, Leaf::Nfc) {
                last_nfc = Some(leaves);
            }
            ordered |= matches!(leaf, Leaf::Lowercase | Leaf::Replace(..));
            leaves = add(leaves, 1)?;
            Ok(())
        })?;
        out.leaves = leaves;
        out.ordered = ordered;
        out.last_nfc = last_nfc;
        let mut at = 0;
        let mut before = 0;
        let mut after = 0;
        out.visit(|leaf| {
            if let Leaf::Prepend(value) = leaf {
                if out.last_nfc.is_some_and(|last| at < last) {
                    before = add(before, value.len())?;
                } else {
                    after = add(after, value.len())?;
                }
            }
            at += 1;
            Ok(())
        })?;
        out.before = before;
        out.after = after;
        Ok(out)
    }
    fn visit(&self, mut visit: impl FnMut(Leaf<'a>) -> Result<(), Error>) -> Result<(), Error> {
        let Some(root) = self.source else {
            return Ok(());
        };
        // JSON validation already limits nesting. These actual reader slots also
        // bound this producer's traversal; no recursive call stack is required.
        let mut stack: [Option<json::Array<'a>>; json::DEPTH] = std::array::from_fn(|_| None);
        let mut depth = 0;
        let mut current = Some(root);
        loop {
            if let Some(span) = current.take() {
                let f = fields(
                    self.input,
                    span,
                    ["type", "normalizers", "prepend", "pattern", "content"],
                )?;
                let typ = required(f[0], span.start)?;
                if text_is(self.input, typ, "Sequence")? {
                    if f[2..].iter().any(Option::is_some) {
                        return Err(err(K::UnsupportedField, span.start));
                    }
                    if depth == stack.len() {
                        return Err(err(K::DepthLimit, span.start));
                    }
                    stack[depth] =
                        Some(Reader::new(self.input, required(f[1], span.start)?).array()?);
                    depth += 1;
                } else if text_is(self.input, typ, "NFC")? {
                    if f[1..].iter().any(Option::is_some) {
                        return Err(err(K::UnsupportedField, span.start));
                    }
                    visit(Leaf::Nfc)?;
                } else if text_is(self.input, typ, "Prepend")? {
                    if f[1].is_some() || f[3..].iter().any(Option::is_some) {
                        return Err(err(K::UnsupportedField, span.start));
                    }
                    let mut reader = Reader::new(self.input, required(f[2], span.start)?);
                    let value = reader.string()?;
                    reader.finish()?;
                    visit(Leaf::Prepend(value))?;
                } else if text_is(self.input, typ, "Lowercase")? {
                    if f[1..].iter().any(Option::is_some) {
                        return Err(err(K::UnsupportedField, span.start));
                    }
                    visit(Leaf::Lowercase)?;
                } else if text_is(self.input, typ, "Replace")? {
                    if f[1].is_some() || f[2].is_some() {
                        return Err(err(K::UnsupportedField, span.start));
                    }
                    let pattern = required(f[3], span.start)?;
                    let p = fields(self.input, pattern, ["String"])?;
                    let read = |span| -> Result<Text<'a>, Error> {
                        let mut reader = Reader::new(self.input, span);
                        let text = reader.string()?;
                        reader.finish()?;
                        Ok(text)
                    };
                    visit(Leaf::Replace(
                        read(required(p[0], pattern.start)?)?,
                        read(required(f[4], span.start)?)?,
                    ))?;
                } else {
                    return Err(err(K::ComponentProfile, span.start));
                }
            }
            while depth != 0 {
                if let Some(span) = stack[depth - 1].as_mut().expect("active array").next()? {
                    current = Some(span);
                    break;
                }
                stack[depth - 1] = None;
                depth -= 1;
            }
            if current.is_none() {
                return Ok(());
            }
        }
    }
    pub(super) fn pattern_bounds(
        &self,
        text: Text<'_>,
    ) -> Result<crate::tokenizer::normalization_source::PatternBounds, Error> {
        use crate::tokenizer::normalization_source::{
            pipeline::{Bounds, Step},
            PatternBounds,
        };
        if self.ordered {
            let mut b = Bounds::new(text.len());
            self.visit(|leaf| {
                b.step(match leaf {
                    Leaf::Nfc => Step::Nfc,
                    Leaf::Lowercase => Step::Lowercase,
                    Leaf::Prepend(text) => Step::Prepend(text.len()),
                    Leaf::Replace(pattern, content) => Step::Replace {
                        pattern: pattern.len(),
                        content: content.len(),
                    },
                })
                .ok_or_else(|| err(K::Overflow, 0))
            })?;
            return Ok(PatternBounds {
                bytes: b.output,
                buffers: b.buffers().ok_or_else(|| err(K::Overflow, 0))?,
                controls: b.controls().ok_or_else(|| err(K::Overflow, 0))?,
            });
        }
        if let Some((pattern, content)) = self.literal {
            let bytes = crate::normalizers::Replace::literal_bound(
                text.len(),
                pattern.len(),
                content.len(),
            )
            .ok_or_else(|| err(K::Overflow, 0))?;
            return Ok(PatternBounds {
                bytes,
                buffers: bytes,
                controls: crate::normalizers::Replace::literal_control_bytes(),
            });
        }
        if text.len() == 0 || self.last_nfc.is_none() && self.after == 0 {
            return Ok(PatternBounds::default());
        }
        if let Some(last) = self.last_nfc {
            let mut inspector = unicode_normalization_alignments::workspace::Inspector::new();
            let mut at = 0;
            self.visit(|leaf| {
                if at < last {
                    if let Leaf::Prepend(value) = leaf {
                        for c in value.chars() {
                            inspector.push(c).map_err(|_| err(K::Overflow, 0))?;
                        }
                    }
                }
                at += 1;
                Ok(())
            })?;
            for c in text.chars() {
                inspector.push(c).map_err(|_| err(K::Overflow, 0))?;
            }
            let r = inspector.finish().map_err(|_| err(K::Overflow, 0))?;
            Ok(PatternBounds {
                bytes: add(r.text_capacity(), self.after)?,
                buffers: r.buffer_bytes(),
                controls: r.control_bytes(),
            })
        } else {
            Ok(PatternBounds {
                bytes: add(text.len(), self.after)?,
                buffers: 0,
                controls: 0,
            })
        }
    }
    fn count(&self) -> usize {
        if self.ordered {
            return self.leaves;
        }
        usize::from(self.before != 0)
            + usize::from(self.last_nfc.is_some())
            + usize::from(self.after != 0)
    }
    pub(super) fn buffer_bytes(&self) -> Result<usize, Error> {
        if let Some((pattern, content)) = self.literal {
            return add(pattern.len(), content.len());
        }
        let leaves = Layout::array::<Leaf<'_>>(self.leaves)
            .map_err(|_| err(K::Overflow, 0))?
            .size();
        let items = if self.count() > 1 {
            Layout::array::<N>(self.count())
                .map_err(|_| err(K::Overflow, 0))?
                .size()
        } else {
            0
        };
        let mut strings = add(self.before, self.after)?;
        if self.ordered {
            self.visit(|leaf| {
                if let Leaf::Replace(pattern, content) = leaf {
                    strings = add(strings, add(pattern.len(), content.len())?)?;
                }
                Ok(())
            })?;
        }
        add(add(leaves, items)?, strings)
    }
    pub(super) fn control_bytes(&self) -> Result<usize, Error> {
        [
            size_of::<Self>(),
            size_of::<[Option<json::Array<'_>>; json::DEPTH]>(),
            size_of::<Option<Span>>(),
            size_of::<Vec<Leaf<'_>>>(),
            size_of::<State>(),
            size_of::<crate::tokenizer::normalization_source::pipeline::Bounds>(),
            size_of::<crate::tokenizer::normalization_source::pipeline::Step>(),
            size_of::<Option<N>>(),
            size_of::<Result<(), Cause>>(),
            size_of::<Result<(), Error>>(),
            size_of::<[Option<Span>; 5]>(),
            size_of::<Reader<'_>>(),
            size_of::<json::Chars<'_>>(),
            size_of::<json::Bytes<'_>>(),
            size_of::<usize>() * 6,
        ]
        .iter()
        .copied()
        .try_fold(0, add)
    }
    #[cfg(any(test, feature = "tokenizer-compiler-test-support"))]
    pub(super) fn fail_reservation(mut self, stage: usize) -> Self {
        self.failure = Some(stage);
        self
    }
    fn capacity(&self, stage: usize, actual: usize) -> usize {
        #[cfg(any(test, feature = "tokenizer-compiler-test-support"))]
        if self.failure == Some(stage) {
            return usize::MAX;
        }
        let _ = stage;
        actual
    }
    pub(super) fn fill(&self, state: &mut State) -> Result<Option<N>, Cause> {
        if let Some((pattern, content)) = self.literal {
            state.literal[0].try_reserve_exact(self.capacity(0, pattern.len()))?;
            state.literal[1].try_reserve_exact(self.capacity(1, content.len()))?;
            state.literal[0].extend(pattern.bytes());
            state.literal[1].extend(content.bytes());
            let pattern = String::from_utf8(std::mem::take(&mut state.literal[0]))
                .expect("validated JSON UTF-8");
            let content = String::from_utf8(std::mem::take(&mut state.literal[1]))
                .expect("validated JSON UTF-8");
            return Ok(Some(N::Replace(
                crate::normalizers::Replace::from_literal_parts(pattern, content),
            )));
        }
        let mut leaves = Vec::new();
        leaves.try_reserve_exact(self.capacity(0, self.leaves))?;
        self.visit(|value| {
            leaves.push(value);
            Ok(())
        })?;
        if self.ordered {
            if self.count() > 1 {
                state
                    .items
                    .try_reserve_exact(self.capacity(3, self.count()))?;
            }
            let mut single = None;
            for leaf in leaves {
                let component = match leaf {
                    Leaf::Nfc => N::NFC(NFC),
                    Leaf::Lowercase => N::Lowercase(crate::normalizers::Lowercase),
                    Leaf::Prepend(text) => {
                        state.literal[0].try_reserve_exact(self.capacity(1, text.len()))?;
                        state.literal[0].extend(text.bytes());
                        N::Prepend(Prepend::new(
                            String::from_utf8(std::mem::take(&mut state.literal[0]))
                                .expect("validated JSON UTF-8"),
                        ))
                    }
                    Leaf::Replace(pattern, content) => {
                        state.literal[0].try_reserve_exact(self.capacity(1, pattern.len()))?;
                        state.literal[1].try_reserve_exact(self.capacity(2, content.len()))?;
                        state.literal[0].extend(pattern.bytes());
                        state.literal[1].extend(content.bytes());
                        N::Replace(crate::normalizers::Replace::from_literal_parts(
                            String::from_utf8(std::mem::take(&mut state.literal[0]))
                                .expect("validated JSON UTF-8"),
                            String::from_utf8(std::mem::take(&mut state.literal[1]))
                                .expect("validated JSON UTF-8"),
                        ))
                    }
                };
                if self.count() > 1 {
                    state.items.push(component);
                } else {
                    single = Some(component);
                }
            }
            return Ok(if self.count() > 1 {
                Some(N::Sequence(Sequence::new(std::mem::take(&mut state.items))))
            } else {
                single
            });
        }
        state
            .before
            .try_reserve_exact(self.capacity(1, self.before))?;
        state
            .after
            .try_reserve_exact(self.capacity(2, self.after))?;
        if self.count() > 1 {
            state
                .items
                .try_reserve_exact(self.capacity(3, self.count()))?;
        }
        if leaves.capacity() > self.leaves
            || state.before.capacity() > self.before
            || state.after.capacity() > self.after
            || state.items.capacity() > if self.count() > 1 { self.count() } else { 0 }
        {
            return Err(err(K::CapacityExceeded, 0).into());
        }
        for (at, leaf) in leaves.into_iter().enumerate().rev() {
            if let Leaf::Prepend(text) = leaf {
                if self.last_nfc.is_some_and(|last| at < last) {
                    state.before.extend(text.bytes());
                } else {
                    state.after.extend(text.bytes());
                }
            }
        }
        let before = if state.before.is_empty() {
            None
        } else {
            Some(N::Prepend(Prepend::new(
                String::from_utf8(std::mem::take(&mut state.before)).expect("validated JSON UTF-8"),
            )))
        };
        let after = if state.after.is_empty() {
            None
        } else {
            Some(N::Prepend(Prepend::new(
                String::from_utf8(std::mem::take(&mut state.after)).expect("validated JSON UTF-8"),
            )))
        };
        let nfc = self.last_nfc.map(|_| N::NFC(NFC));
        if self.count() > 1 {
            state
                .items
                .extend(before.into_iter().chain(nfc).chain(after));
            Ok(Some(N::Sequence(Sequence::new(std::mem::take(
                &mut state.items,
            )))))
        } else {
            Ok(before.or(nfc).or(after))
        }
    }
}
#[derive(Debug, Default)]
pub(super) struct State {
    before: Vec<u8>,
    after: Vec<u8>,
    items: Vec<N>,
    literal: [Vec<u8>; 2],
}

impl State {
    pub(super) fn literal_capacities(&self) -> [usize; 2] {
        [self.literal[0].capacity(), self.literal[1].capacity()]
    }
}
