//! Box readouts over calibrated presence and four ordered edge distributions.

use std::sync::Arc;

use eredu_core::decision::{BoxDetection, DecisionAnswer, ImageBox};

use super::{
    readout::{interpolate, ChoiceReadout},
    DecisionError,
};

pub(super) const EDGES: [&str; 4] = ["top", "bottom", "left", "right"];
pub(super) const DEFAULT_INSTANCES: usize = 16;

pub(super) struct BoxReadout {
    pub(super) count: usize,
    pub(super) multiple: bool,
    pub(super) step: f64,
    pub(super) presence: ChoiceReadout,
    pub(super) edges: ChoiceReadout,
    pub(super) interpolate_edges: bool,
    pub(super) found: Arc<[[f64; 2]]>,
    pub(super) confidence: Arc<[[f64; 2]]>,
}

impl BoxReadout {
    pub(super) fn answer(
        &self,
        mut next: impl FnMut(&ChoiceReadout) -> Result<Vec<f64>, DecisionError>,
    ) -> Result<DecisionAnswer, DecisionError> {
        let mut presence = Vec::with_capacity(self.count);
        let mut detections = Vec::new();
        let mut below = 1.0_f64;
        for _ in 0..self.count {
            let found = interpolate(&self.found, next(&self.presence)?[0]);
            below = below.min(found);
            presence.push(below);
            let probabilities = [
                next(&self.edges)?,
                next(&self.edges)?,
                next(&self.edges)?,
                next(&self.edges)?,
            ];
            if below >= 0.5 {
                detections.push(self.detect(&probabilities, below));
            }
        }
        if !self.multiple {
            return Ok(DecisionAnswer::Box(detections.pop().unwrap_or(
                BoxDetection {
                    bounds: None,
                    found: presence[0],
                    confidence: 0.0,
                },
            )));
        }
        detections.sort_by(|a, b| {
            let (a, b) = (
                a.bounds.expect("present box"),
                b.bounds.expect("present box"),
            );
            a.left
                .total_cmp(&b.left)
                .then_with(|| a.top.total_cmp(&b.top))
        });
        Ok(DecisionAnswer::Boxes {
            at_least: detections.len() == self.count,
            detections,
            presence,
        })
    }

    fn detect(&self, probabilities: &[Vec<f64>; 4], found: f64) -> BoxDetection {
        let value = |index: usize| index as f64 * self.step;
        let high = value(probabilities[0].len() - 1);
        let cdfs = probabilities.each_ref().map(|row| {
            let total: f64 = row.iter().sum();
            row.iter()
                .scan(0.0, |mass, p| {
                    *mass += p / total;
                    Some(*mass)
                })
                .collect::<Vec<_>>()
        });
        let edges = cdfs.each_ref().map(|cdf| {
            let index = cdf.partition_point(|&mass| mass < 0.5).min(cdf.len() - 1);
            let median = value(index);
            if !self.interpolate_edges {
                return median;
            }
            let previous = index.checked_sub(1).map_or(0.0, |index| cdf[index]);
            let share = cdf[index] - previous;
            if share <= 0.0 || cdf[index] < 0.5 {
                return high;
            }
            (median - self.step / 2.0 + (0.5 - previous) / share * self.step).clamp(0.0, high)
        });
        let bounds = rectangle(edges, self.step);
        let hits = (1..=512)
            .filter(|&sample| {
                let edges = [2, 3, 5, 7].map(|base| radical_inverse(sample, base));
                let sampled = std::array::from_fn(|i| {
                    let cdf = &cdfs[i];
                    value(
                        cdf.partition_point(|&mass| mass < edges[i])
                            .min(cdf.len() - 1),
                    )
                });
                overlap(bounds, rectangle(sampled, self.step)) >= 0.5
            })
            .count();
        BoxDetection {
            bounds: Some(bounds),
            found,
            confidence: interpolate(&self.confidence, hits as f64 / 512.0),
        }
    }
}

fn rectangle([top, bottom, left, right]: [f64; 4], step: f64) -> ImageBox {
    let ordered = |a: f64, b: f64| {
        if a < b {
            return (a, b);
        }
        let half = step / 2.0;
        let middle = ((a + b) / 2.0_f64).clamp(half, 1.0 - half);
        (middle - half, middle + half)
    };
    let (top, bottom) = ordered(top, bottom);
    let (left, right) = ordered(left, right);
    ImageBox {
        top,
        bottom,
        left,
        right,
    }
}

fn overlap(a: ImageBox, b: ImageBox) -> f64 {
    let area = |bounds: ImageBox| (bounds.bottom - bounds.top) * (bounds.right - bounds.left);
    let common = (a.bottom.min(b.bottom) - a.top.max(b.top)).max(0.0)
        * (a.right.min(b.right) - a.left.max(b.left)).max(0.0);
    common / (area(a) + area(b) - common)
}

fn radical_inverse(mut index: usize, base: usize) -> f64 {
    let mut value = 0.0;
    let mut scale = 1.0;
    while index != 0 {
        scale /= base as f64;
        value += (index % base) as f64 * scale;
        index /= base;
    }
    value
}
