//! Actual Metaspace fields and fixed-stack flattening of pre-tokenizer sequences.
use super::*;
use crate::pre_tokenizers::metaspace::{Metaspace, PrependScheme};

#[derive(Debug, Clone, Copy)]
pub(super) struct Meta {
    replacement: char,
    scheme: PrependScheme,
    split: bool,
}
pub(super) fn select(input: &str, span: Span) -> Result<Meta, Error> {
    let f = fields(
        input,
        span,
        [
            "type",
            "replacement",
            "prepend_scheme",
            "split",
            "add_prefix_space",
            "str_rep",
        ],
    )?;
    let mut r = Reader::new(input, required(f[1], span.start)?);
    let value = r.string()?;
    r.finish()?;
    let mut chars = value.chars();
    let replacement = chars
        .next()
        .ok_or_else(|| err(K::ComponentProfile, span.start))?;
    if chars.next().is_some() {
        return Err(err(K::ComponentProfile, span.start));
    }
    let scheme = match f[2] {
        None => PrependScheme::Always,
        Some(s) if text_is(input, s, "always")? => PrependScheme::Always,
        Some(s) if text_is(input, s, "first")? => PrependScheme::First,
        Some(s) if text_is(input, s, "never")? => PrependScheme::Never,
        Some(s) => return Err(err(K::ComponentProfile, s.start)),
    };
    if !boolean(input, f[4], true)? && scheme != PrependScheme::Never {
        return Err(err(K::ComponentProfile, span.start));
    }
    if let Some(span) = absent(input, f[5]) {
        let mut r = Reader::new(input, span);
        r.string()?;
        r.finish()?;
    }
    Ok(Meta {
        replacement,
        scheme,
        split: boolean(input, f[3], true)?,
    })
}
impl Meta {
    pub(super) fn decoder_component(self) -> crate::decoders::fixed_profile::Component {
        crate::decoders::fixed_profile::Component::Metaspace {
            replacement: self.replacement,
            remove_first: self.scheme != PrependScheme::Never,
        }
    }
    pub(super) fn bytes(self) -> usize {
        self.replacement.len_utf8()
    }
    pub(super) fn compile(self, partial: &mut Partial, failure: Option<usize>) -> Result<PreTokenizerWrapper, Cause> {
        self.compile_component(partial, failure).map(Into::into)
    }
    pub(super) fn compile_component(self, partial: &mut Partial, failure: Option<usize>) -> Result<Metaspace, Cause> {
        let mut encoded = [0; 4];
        let text = self.replacement.encode_utf8(&mut encoded);
        partial.pre_literal.try_reserve_exact(requested(5, text.len(), failure))?;
        if partial.pre_literal.capacity() > text.len() {
            return Err(err(K::CapacityExceeded, 0).into());
        }
        partial.pre_literal.extend_from_slice(text.as_bytes());
        let text =
            String::from_utf8(std::mem::take(&mut partial.pre_literal)).expect("encoded scalar");
        Ok(Metaspace::from_replacement_parts(self.replacement, self.scheme, self.split, text))
    }
}
pub(super) struct Items<'a> {
    input: &'a str,
    stack: [Option<json::Array<'a>>; json::DEPTH],
    depth: usize,
    nested: bool,
}
impl<'a> Items<'a> {
    pub(super) fn new(input: &'a str, array: Span, role: Role) -> Result<Self, Error> {
        let mut stack = std::array::from_fn(|_| None);
        stack[0] = Some(Reader::new(input, array).array()?);
        Ok(Self {
            input,
            stack,
            depth: 1,
            nested: matches!(role, Role::Pre),
        })
    }
    pub(super) fn next(&mut self) -> Result<Option<Span>, Error> {
        while self.depth != 0 {
            let Some(span) = self.stack[self.depth - 1]
                .as_mut()
                .expect("active sequence")
                .next()?
            else {
                self.stack[self.depth - 1] = None;
                self.depth -= 1;
                continue;
            };
            if self.nested {
                let mut object = Reader::new(self.input, span).object()?;
                let mut typ = None;
                while let Some((key, value)) = object.next()? {
                    if key.is("type") && typ.replace(value).is_some() {
                        return Err(err(K::DuplicateField, key.offset));
                    }
                }
                if text_is(self.input, required(typ, span.start)?, "Sequence")? {
                    let f = fields(self.input, span, ["type", "pretokenizers"])?;
                    if self.depth == self.stack.len() {
                        return Err(err(K::DepthLimit, span.start));
                    }
                    self.stack[self.depth] =
                        Some(Reader::new(self.input, required(f[1], span.start)?).array()?);
                    self.depth += 1;
                    continue;
                }
            }
            return Ok(Some(span));
        }
        Ok(None)
    }
}
pub(super) fn controls() -> Result<usize, Error> {
    [
        size_of::<Items<'_>>(),
        size_of::<Result<Items<'_>, Error>>(),
        size_of::<Meta>(),
        size_of::<PrependScheme>(),
        size_of::<[u8; 4]>(),
        size_of::<[Option<Span>; 6]>(),
        size_of::<Result<[Option<Span>; 6], Error>>(),
        size_of::<json::Chars<'_>>(),
        size_of::<Result<Meta, Error>>(),
        size_of::<Metaspace>(),
        size_of::<String>(),
    ]
    .iter()
    .copied()
    .try_fold(0, add)
}
