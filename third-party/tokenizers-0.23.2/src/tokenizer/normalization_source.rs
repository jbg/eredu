//! Borrowed projection of the normalization producer's canonical component form.
use crate::normalizers::NormalizerWrapper as N;
pub(crate) mod pipeline;

#[derive(Debug, Clone, Copy)]
pub(crate) struct Shape<'a> {
    pub(crate) before: &'a str,
    pub(crate) after: &'a str,
    pub(crate) nfc: bool,
    pub(crate) literal: Option<&'a crate::normalizers::Replace>,
    pub(crate) pipeline: Option<&'a N>,
    pub(crate) remove_prefixes: bool,
}
impl<'a> Shape<'a> {
    pub(crate) fn inspect(source: Option<&'a N>) -> Option<Self> {
        let mut out = Self {
            before: "",
            after: "",
            nfc: false,
            literal: None,
            pipeline: None,
            remove_prefixes: false,
        };
        let values = match source {
            None => return Some(out),
            Some(N::Sequence(sequence)) => sequence.as_ref(),
            Some(value) => std::slice::from_ref(value),
        };
        match values {
            [] => {}
            [N::Replace(value)] if value.literal_parts().is_some() => out.literal = Some(value),
            [N::Prepend(value)] => out.after = &value.prepend,
            [N::NFC(_)] => out.nfc = true,
            [N::Prepend(before), N::NFC(_)] => {
                out.before = &before.prepend;
                out.nfc = true;
            }
            [N::NFC(_), N::Prepend(after)] => {
                out.after = &after.prepend;
                out.nfc = true;
            }
            [N::Prepend(before), N::NFC(_), N::Prepend(after)] => {
                out.before = &before.prepend;
                out.after = &after.prepend;
                out.nfc = true;
            }
            _ => {
                for leaf in Leaves::new(source, false) {
                    leaf.ok()?;
                }
                out.pipeline = source;
            }
        }
        Some(out)
    }
    pub(crate) fn prefix_free(self) -> Self {
        Self {
            before: "",
            after: "",
            nfc: self.nfc,
            literal: self.literal,
            pipeline: self.pipeline,
            remove_prefixes: true,
        }
    }
    pub(crate) fn has_prefix(self) -> bool {
        !self.before.is_empty()
            || !self.after.is_empty()
            || self.pipeline.is_some()
                && Leaves::new(self.pipeline, false)
                    .any(|leaf| matches!(leaf, Ok(N::Prepend(value)) if !value.prepend.is_empty()))
    }
    pub(crate) fn is_identity(self) -> bool {
        self.pipeline.is_none()
            && self.literal.is_none()
            && !self.nfc
            && self.before.is_empty()
            && self.after.is_empty()
    }
    pub(crate) fn text_bound(self, input: &str) -> Option<usize> {
        if self.pipeline.is_some() {
            return Some(self.pipeline_bounds(input.len())?.output);
        }
        if let Some(literal) = self.literal {
            let (pattern, content) = literal.literal_parts()?;
            return crate::normalizers::Replace::literal_bound(
                input.len(),
                pattern.len(),
                content.len(),
            );
        }
        if input.is_empty() {
            return Some(0);
        }
        let mut bytes = Some(self.after.len());
        for c in self.before.chars().chain(input.chars()) {
            if self.nfc {
                unicode_normalization_alignments::char::decompose_canonical(c, |c| {
                    bytes = bytes.and_then(|n| n.checked_add(c.len_utf8()));
                });
            } else {
                bytes = bytes.and_then(|n| n.checked_add(c.len_utf8()));
            }
        }
        bytes
    }
}

/// Checked destination facts from actual added spelling and prefix scalars.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct PatternBounds {
    pub(crate) bytes: usize,
    pub(crate) buffers: usize,
    pub(crate) controls: usize,
}
impl Shape<'_> {
    pub(crate) fn pattern_bounds(
        self,
        text: crate::utils::borrowed_json::Text<'_>,
    ) -> Option<PatternBounds> {
        if self.pipeline.is_some() {
            let b = self.pipeline_bounds(text.len())?;
            return Some(PatternBounds {
                bytes: b.output,
                buffers: b.buffers()?,
                controls: b.controls()?,
            });
        }
        if let Some(literal) = self.literal {
            let (pattern, content) = literal.literal_parts()?;
            let bytes = crate::normalizers::Replace::literal_bound(
                text.len(),
                pattern.len(),
                content.len(),
            )?;
            return Some(PatternBounds {
                bytes,
                buffers: bytes,
                controls: crate::normalizers::Replace::literal_control_bytes(),
            });
        }
        if text.len() == 0 || self.is_identity() {
            return Some(PatternBounds::default());
        }
        if !self.nfc {
            return Some(PatternBounds {
                bytes: self.after.len().checked_add(text.len())?,
                buffers: 0,
                controls: 0,
            });
        }
        let mut inspector = unicode_normalization_alignments::workspace::Inspector::new();
        for c in self.before.chars().chain(text.chars()) {
            inspector.push(c).ok()?;
        }
        let r = inspector.finish().ok()?;
        Some(PatternBounds {
            bytes: self.after.len().checked_add(r.text_capacity())?,
            buffers: r.buffer_bytes(),
            controls: r.control_bytes(),
        })
    }
}

/// Fixed traversal over the selected immutable normalizer; all nested sequence
/// forms share the compiler's accepted NFC/Prepend semantics.
struct Leaves<'a> {
    stack: [std::slice::Iter<'a, N>; crate::utils::borrowed_json::DEPTH],
    depth: usize,
    reverse: bool,
}
impl<'a> Leaves<'a> {
    fn new(source: Option<&'a N>, reverse: bool) -> Self {
        let mut stack = std::array::from_fn(|_| [].iter());
        stack[0] = source.map_or(&[][..], std::slice::from_ref).iter();
        Self {
            stack,
            depth: 1,
            reverse,
        }
    }
}
impl<'a> Iterator for Leaves<'a> {
    type Item = Result<&'a N, ()>;
    fn next(&mut self) -> Option<Self::Item> {
        while self.depth != 0 {
            let current = &mut self.stack[self.depth - 1];
            let next = if self.reverse {
                current.next_back()
            } else {
                current.next()
            };
            match next {
                None => self.depth -= 1,
                Some(N::Sequence(sequence)) => {
                    if self.depth == self.stack.len() {
                        self.depth = 0;
                        return Some(Err(()));
                    }
                    self.stack[self.depth] = sequence.as_ref().iter();
                    self.depth += 1;
                }
                Some(value @ (N::NFC(_) | N::Prepend(_) | N::Lowercase(_))) => {
                    return Some(Ok(value))
                }
                Some(value @ N::Replace(replace)) if replace.literal_parts().is_some() => {
                    return Some(Ok(value))
                }
                Some(_) => {
                    self.depth = 0;
                    return Some(Err(()));
                }
            }
        }
        None
    }
}
struct PrefixChars<'a> {
    leaves: Leaves<'a>,
    remaining: usize,
    last_nfc: Option<usize>,
    before: bool,
    current: std::str::Chars<'a>,
}
impl Iterator for PrefixChars<'_> {
    type Item = char;
    fn next(&mut self) -> Option<char> {
        loop {
            if let Some(c) = self.current.next() {
                return Some(c);
            }
            let leaf = self.leaves.next()?.expect("validated immutable normalizer");
            self.remaining -= 1;
            if let N::Prepend(value) = leaf {
                let before = self.last_nfc.is_some_and(|last| self.remaining < last);
                if before == self.before {
                    self.current = value.prepend.chars();
                }
            }
        }
    }
}
/// Compare the compiled component form to its immutable declared normalizer.
/// Empty sequences and repeated NFC are canonical semantic equivalents.
pub(crate) fn same_input(input: super::TokenizerInput<'_>, declared: Option<&N>) -> bool {
    let Some(shape) = Shape::inspect(input.root.get_normalizer()) else {
        return false;
    };
    let shape = if input.remove_input_prefixes {
        shape.prefix_free()
    } else {
        shape
    };
    if shape.pipeline.is_some() {
        let mut left = Leaves::new(shape.pipeline, false)
            .filter(|value| !(shape.remove_prefixes && matches!(value, Ok(N::Prepend(_)))));
        let mut right = Leaves::new(declared, false);
        loop {
            match (left.next(), right.next()) {
                (None, None) => return true,
                (Some(Ok(N::NFC(_))), Some(Ok(N::NFC(_))))
                | (Some(Ok(N::Lowercase(_))), Some(Ok(N::Lowercase(_)))) => {}
                (Some(Ok(N::Prepend(a))), Some(Ok(N::Prepend(b)))) if a.prepend == b.prepend => {}
                (Some(Ok(N::Replace(a))), Some(Ok(N::Replace(b)))) if a == b => {}
                _ => return false,
            }
        }
    }
    if let Some(literal) = shape.literal {
        let Some(other) = Shape::inspect(declared) else {
            return false;
        };
        return other.literal.is_some_and(|other| literal == other);
    }
    let mut count = 0usize;
    let mut last = None;
    for value in Leaves::new(declared, false) {
        let Ok(value) = value else {
            return false;
        };
        if !matches!(value, N::NFC(_) | N::Prepend(_)) {
            return false;
        }
        if matches!(value, N::NFC(_)) {
            last = Some(count);
        }
        count += 1;
    }
    if shape.nfc != last.is_some() {
        return false;
    }
    let chars = |before| PrefixChars {
        leaves: Leaves::new(declared, true),
        remaining: count,
        last_nfc: last,
        before,
        current: "".chars(),
    };
    shape.before.chars().eq(chars(true)) && shape.after.chars().eq(chars(false))
}
pub(crate) fn comparison_control_bytes() -> Option<usize> {
    use std::mem::size_of;
    [
        size_of::<Leaves<'_>>(),
        size_of::<PrefixChars<'_>>(),
        size_of::<Shape<'_>>(),
        size_of::<Option<Shape<'_>>>(),
        size_of::<std::str::Chars<'_>>(),
        size_of::<Option<Result<&N, ()>>>(),
        size_of::<Result<&N, ()>>(),
        size_of::<[Option<&N>; 2]>(),
        size_of::<usize>() * 4,
    ]
    .iter()
    .copied()
    .try_fold(0usize, usize::checked_add)
}
