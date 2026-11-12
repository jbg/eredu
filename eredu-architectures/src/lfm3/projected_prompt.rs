//! Shared projected embedding spans for LFM3 conditional decoding.

use std::sync::Arc;

use eredu_nn::{Error, Index, Tensor};

#[derive(Clone, Copy, Eq, PartialEq)]
enum Source {
    Text,
    Projected(u8),
}

pub(super) struct Span {
    position: i32,
    source: Source,
    start: i32,
    end: i32,
}

pub(super) struct ProjectedPrompt<'a, T, const N: usize> {
    pub(super) tokens: T,
    pub(super) rows: [Option<&'a T>; N],
    pub(super) spans: Arc<[Span]>,
    pub(super) width: i32,
    pub(super) length: i32,
}

impl<'a, T: Tensor, const N: usize> ProjectedPrompt<'a, T, N> {
    pub(super) fn new(
        ids: &[i32],
        markers: [i32; N],
        width: i32,
        rows: [Option<&'a T>; N],
        context: &T::Context,
    ) -> Result<Self, Error> {
        let length = i32::try_from(ids.len()).map_err(Error::backend_source)?;
        if length == 0
            || width <= 0
            || markers
                .iter()
                .enumerate()
                .any(|(index, id)| markers[..index].contains(id))
        {
            return Err(Error::backend("invalid LFM3 projected prompt geometry"));
        }
        let mut counts = [0; N];
        let mut spans: Vec<Span> = Vec::new();
        for (index, &id) in ids.iter().enumerate() {
            let (source, start) =
                if let Some(slot) = markers.iter().position(|&marker| marker == id) {
                    let start = counts[slot];
                    counts[slot] += 1;
                    (
                        Source::Projected(u8::try_from(slot).map_err(Error::backend_source)?),
                        start,
                    )
                } else {
                    (Source::Text, index as i32)
                };
            if let Some(span) = spans.last_mut().filter(|span| span.source == source) {
                span.end += 1;
            } else {
                spans.push(Span {
                    position: index as i32,
                    source,
                    start,
                    end: start + 1,
                });
            }
        }
        for (rows, count) in rows.into_iter().zip(counts) {
            match rows {
                Some(rows) if count > 0 && rows.shape() == [count, width] => {}
                None if count == 0 => {}
                _ => {
                    return Err(Error::backend(
                        "LFM3 markers differ from projected media rows",
                    ))
                }
            }
        }
        Ok(Self {
            tokens: T::from_i32_slice(ids, &[1, length], context)?,
            rows,
            spans: spans.into(),
            width,
            length,
        })
    }

    pub(super) fn tokens(&self) -> &T {
        &self.tokens
    }

    /// Borrows one contiguous prompt range for incremental prefill.
    ///
    /// Text indices are local to the range. Media indices still address the
    /// original projected rows, even when a range starts inside a media span.
    pub(super) fn slice(
        &self,
        range: std::ops::Range<i32>,
        context: &T::Context,
    ) -> Result<Self, Error> {
        if range.start < 0 || range.end <= range.start || range.end > self.length {
            return Err(Error::backend("invalid LFM3 projected prompt range"));
        }
        let first = self
            .spans
            .partition_point(|span| span.position + (span.end - span.start) <= range.start);
        let mut spans = Vec::new();
        for span in &self.spans[first..] {
            let position = span.position;
            if position >= range.end {
                break;
            }
            let start = position.max(range.start);
            let end = (position + (span.end - span.start)).min(range.end);
            let row = match span.source {
                Source::Text => start - range.start,
                Source::Projected(_) => span.start + (start - position),
            };
            spans.push(Span {
                position: start - range.start,
                source: span.source,
                start: row,
                end: row + (end - start),
            });
        }
        Ok(Self {
            tokens: self.tokens.index(
                &[Index::Full, Index::Range(range.start, range.end)],
                context,
            )?,
            rows: self.rows,
            spans: spans.into(),
            width: self.width,
            length: range.end - range.start,
        })
    }

    pub(super) fn update_pipeline_digest(
        &self,
        digest: &mut sha2::Sha256,
        context: &T::Context,
    ) -> Result<(), Error> {
        use sha2::Digest;
        digest.update((self.spans.len() as u64).to_le_bytes());
        for span in self.spans.iter() {
            let code = match span.source {
                Source::Text => 0,
                Source::Projected(slot) => slot
                    .checked_add(1)
                    .ok_or_else(|| Error::backend("too many projected modalities"))?,
            };
            digest.update([code]);
            digest.update(span.position.to_le_bytes());
            digest.update(span.start.to_le_bytes());
            digest.update(span.end.to_le_bytes());
        }
        for rows in self.rows {
            digest.update([u8::from(rows.is_some())]);
            if let Some(rows) = rows {
                super::update_pipeline_tensor_digest(digest, rows, context)?;
            }
        }
        Ok(())
    }

    pub(super) fn projected_roots(&self) -> impl Iterator<Item = &T> {
        self.rows.into_iter().flatten()
    }

    pub(super) fn replace(&self, text: &T, context: &T::Context) -> Result<T, Error> {
        if text.shape() != [1, self.length, self.width] {
            return Err(Error::backend(
                "LFM3 projected prompt differs from decoder embedding shape",
            ));
        }
        if self.rows.iter().all(Option::is_none) {
            return Ok(text.clone());
        }
        let dtype = text
            .element_type()
            .ok_or_else(|| Error::backend("unknown decoder embedding dtype"))?;
        let spans = self
            .spans
            .iter()
            .map(|span| {
                let range = Index::Range(span.start, span.end);
                let media = match span.source {
                    Source::Text => return text.index(&[Index::Full, range], context),
                    Source::Projected(slot) => self.rows[usize::from(slot)],
                };
                media
                    .expect("prompt admission checks marker counts")
                    .index(&[range], context)?
                    .cast_float(dtype, context)?
                    .expand_dims(0, context)
            })
            .collect::<Result<Vec<_>, _>>()?;
        T::concatenate(&spans, 1, context)
    }
}
