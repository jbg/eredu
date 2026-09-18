//! Actual finite decoder component construction under the root's paid destinations.
use super::*;
use crate::decoders::{byte_fallback::ByteFallback, fixed_profile, fuse::Fuse, strip::Strip};
use crate::normalizers::replace::Replace;

#[derive(Debug, Clone, Copy)]
pub(super) enum Inline {
    Metaspace(pre::Meta),
    ByteFallback,
    Fuse,
    Replace {
        pattern: Span,
        content: Span,
        bytes: usize,
    },
    Strip,
}
fn text(input: &str, span: Span) -> Result<json::Text<'_>, Error> {
    let mut reader = Reader::new(input, span);
    let text = reader.string()?;
    reader.finish()?;
    Ok(text)
}
pub(super) fn select(input: &str, span: Span, typ: Span) -> Result<Inline, Error> {
    if text_is(input, typ, "Metaspace")? {
        pre::select(input, span).map(Inline::Metaspace)
    } else if text_is(input, typ, "ByteFallback")? {
        fields(input, span, ["type"])?;
        Ok(Inline::ByteFallback)
    } else if text_is(input, typ, "Fuse")? {
        fields(input, span, ["type"])?;
        Ok(Inline::Fuse)
    } else if text_is(input, typ, "Replace")? {
        let values = fields(input, span, ["type", "pattern", "content"])?;
        let pattern = required(values[1], span.start)?;
        let literal = fields(input, pattern, ["String"])?;
        let pattern = required(literal[0], pattern.start)?;
        let content = required(values[2], span.start)?;
        let pattern_text = text(input, pattern)?;
        let content_text = text(input, content)?;
        if !pattern_text.is("▁") || !content_text.is(" ") {
            return Err(err(K::ComponentProfile, span.start));
        }
        Ok(Inline::Replace {
            pattern,
            content,
            bytes: add(pattern_text.len(), content_text.len())?,
        })
    } else if text_is(input, typ, "Strip")? {
        let values = fields(input, span, ["type", "content", "start", "stop"])?;
        let content = required(values[1], span.start)?;
        let start = required(values[2], span.start)?;
        let stop = required(values[3], span.start)?;
        if !text_is(input, content, " ")?
            || input[start.start..start.end].parse::<usize>().ok() != Some(1)
            || input[stop.start..stop.end].parse::<usize>().ok() != Some(0)
        {
            return Err(err(K::ComponentProfile, span.start));
        }
        Ok(Inline::Strip)
    } else {
        Err(err(K::ComponentProfile, span.start))
    }
}
impl Inline {
    pub(super) fn component(self) -> fixed_profile::Component {
        match self {
            Self::Metaspace(value) => value.decoder_component(),
            Self::ByteFallback => fixed_profile::Component::ByteFallback,
            Self::Fuse => fixed_profile::Component::Fuse,
            Self::Replace { .. } => fixed_profile::Component::MarkerReplace,
            Self::Strip => fixed_profile::Component::InitialSpaceStrip,
        }
    }
    pub(super) fn bytes(self) -> usize {
        match self {
            Self::Metaspace(value) => value.bytes(),
            Self::Replace { bytes, .. } => bytes,
            _ => 0,
        }
    }
    pub(super) fn compile(
        self,
        input: &str,
        failure: Option<usize>,
        partial: &mut Partial,
    ) -> Result<DecoderWrapper, Cause> {
        Ok(match self {
            Self::Metaspace(value) => value.compile_component(partial, failure)?.into(),
            Self::ByteFallback => ByteFallback::new().into(),
            Self::Fuse => Fuse::new().into(),
            Self::Strip => Strip::new(' ', 1, 0).into(),
            Self::Replace {
                pattern, content, ..
            } => {
                for (index, span) in [pattern, content].iter().copied().enumerate() {
                    let value = text(input, span)?;
                    let count = value.len();
                    let destination = &mut partial.decode_strings[index];
                    destination.try_reserve_exact(requested(3 + index, count, failure))?;
                    if destination.capacity() > count {
                        return Err(err(K::CapacityExceeded, span.start).into());
                    }
                    destination.extend(value.bytes());
                }
                // JSON validation and the shared escape iterator guarantee UTF-8.
                // Both conversions reuse the paid Vec allocation unchanged.
                let pattern = String::from_utf8(std::mem::take(&mut partial.decode_strings[0]))
                    .expect("validated JSON literal");
                let content = String::from_utf8(std::mem::take(&mut partial.decode_strings[1]))
                    .expect("validated JSON content");
                Replace::from_literal_parts(pattern, content).into()
            }
        })
    }
}
pub(super) fn control_bytes() -> Result<usize, Error> {
    let parts = [
        fixed_profile::Profile::control_bytes().ok_or(err(K::Overflow, 0))?,
        size_of::<Inline>(),
        size_of::<Result<Inline, Error>>(),
        size_of::<Result<DecoderWrapper, Cause>>(),
        size_of::<DecoderWrapper>(),
        size_of::<Result<[Option<Span>; 3], Error>>(),
        size_of::<Result<[Option<Span>; 1], Error>>(),
        size_of::<[Span; 2]>(),
        size_of::<json::Text<'_>>(),
        size_of::<Result<json::Text<'_>, Error>>(),
        size_of::<json::Bytes<'_>>(),
        size_of::<[String; 2]>(),
        size_of::<Result<String, std::string::FromUtf8Error>>(),
        size_of::<std::string::FromUtf8Error>(),
        size_of::<Result<usize, std::num::ParseIntError>>(),
        size_of::<Result<(), std::collections::TryReserveError>>(),
        size_of::<crate::normalizers::replace::ReplacePattern>(),
        size_of::<Replace>(),
        size_of::<ByteFallback>(),
        size_of::<Fuse>(),
        size_of::<Strip>(),
    ];
    parts
        .iter()
        .try_fold(std::mem::size_of_val(&parts), |sum, &part| add(sum, part))
}
