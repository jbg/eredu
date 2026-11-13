//! Source alignment and parameter-free span selection for d1.

use std::{collections::VecDeque, ops::Range};

use eredu_core::decision::{SpanExtraction, TextSpan};

use super::{readout::interpolate, DecisionError};

pub(super) struct StatePositions {
    text: String,
    offsets: Vec<Option<Range<usize>>>,
}

impl StatePositions {
    pub(super) fn new(
        text: &str,
        prompt: &str,
        start: usize,
        boundaries: &[usize],
        tokens: &[(usize, usize)],
    ) -> Self {
        let stop = start + boundaries.len() - 1;
        let mut offsets: Vec<_> = tokens
            .iter()
            .map(|&(a, b)| {
                let lo = a.max(start);
                let hi = b.min(stop);
                if lo >= hi
                    || !prompt.get(a..lo)?.trim().is_empty()
                    || !prompt.get(hi..b)?.trim().is_empty()
                {
                    return None;
                }
                let span = boundaries[lo - start]..boundaries[hi - start];
                (!text.get(span.clone())?.trim().is_empty()).then_some(span)
            })
            .collect();
        offsets.truncate(
            offsets
                .iter()
                .rposition(Option::is_some)
                .map_or(1, |i| i + 1),
        );
        Self {
            text: text.into(),
            offsets,
        }
    }

    pub(super) fn text(&self) -> &str {
        &self.text
    }

    pub(super) fn width(&self) -> usize {
        self.offsets.len()
    }

    pub(super) fn extract(
        &self,
        start: &[f32],
        end: &[f32],
        cap: usize,
        calibration: &[[f64; 2]],
    ) -> Result<SpanExtraction, DecisionError> {
        if self.width() > start.len() || self.width() > end.len() || !start[0].is_finite() {
            return Err(DecisionError::Request("invalid pointer score row"));
        }
        let valid = |i: usize| self.offsets[i].is_some();
        let scores = (1..self.width()).filter(|&i| valid(i)).map(|i| start[i]);
        let max = scores.clone().fold(start[0], f32::max);
        let weight = |value: f32| (f64::from(value) - f64::from(max)).exp();
        let found_weight: f64 = scores.map(weight).sum();
        let raw_found = found_weight / (weight(start[0]) + found_weight);
        if !raw_found.is_finite() {
            return Err(DecisionError::Request("non-finite pointer score"));
        }
        let found = interpolate(calibration, raw_found);
        let mut span = None;
        if raw_found >= 0.5 && found >= 0.5 {
            let mut queue = VecDeque::new();
            let mut cursor = 1;
            let mut best = f64::NEG_INFINITY;
            for first in 1..self.width() {
                while queue.front().is_some_and(|&last| last < first) {
                    queue.pop_front();
                }
                while cursor < first.saturating_add(cap).min(self.width()) {
                    if valid(cursor) {
                        if !end[cursor].is_finite() {
                            return Err(DecisionError::Request("non-finite pointer score"));
                        }
                        while queue.back().is_some_and(|&last| end[last] < end[cursor]) {
                            queue.pop_back();
                        }
                        queue.push_back(cursor);
                    }
                    cursor += 1;
                }
                if let (Some(a), Some(&last)) = (&self.offsets[first], queue.front()) {
                    let score = f64::from(start[first]) + f64::from(end[last]);
                    if score > best {
                        best = score;
                        let b = self.offsets[last]
                            .as_ref()
                            .expect("queued valid pointer position");
                        let raw = self.text.get(a.start..b.end).ok_or(DecisionError::Request(
                            "pointer endpoints are not a source span",
                        ))?;
                        let trimmed = raw.trim_start();
                        let offset = a.start + raw.len() - trimmed.len();
                        let trimmed = trimmed.trim_end();
                        span = Some(TextSpan {
                            text: trimmed.into(),
                            bytes: offset..offset + trimmed.len(),
                        });
                    }
                }
            }
        }
        Ok(SpanExtraction { span, found })
    }
}
