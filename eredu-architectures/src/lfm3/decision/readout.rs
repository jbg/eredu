//! d1 vocabulary-slot calibration and typed interpretation.

use std::sync::Arc;

use eredu_core::decision::{ChoiceAnswer, ChoiceProbability, DecisionAnswer};

use super::{pointer::StatePositions, DecisionError};

pub(super) enum Readout {
    Box(super::boxes::BoxReadout),
    Pointer(PointerKind),
    Single(ChoiceReadout),
    YesNo(ChoiceReadout),
    Score(ChoiceReadout),
    Estimate {
        readout: ChoiceReadout,
        values: Vec<f64>,
        levels: [f64; 3],
        unit: Option<String>,
    },
    Multiple {
        labels: Vec<String>,
        binary: ChoiceReadout,
        calibration: Arc<[[f64; 2]]>,
    },
}

pub(super) enum PointerKind {
    Text,
    Number(Option<String>),
    Date,
}

impl PointerKind {
    pub(super) fn name(&self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::Number(_) => "number",
            Self::Date => "date",
        }
    }
}

pub(super) struct ChoiceReadout {
    pub(super) groups: Vec<(String, Vec<u32>)>,
    pub(super) temperature: f64,
}

impl Readout {
    pub(super) fn is_pointer(&self) -> bool {
        matches!(self, Self::Pointer(_))
    }

    pub(super) fn slot_count(&self) -> usize {
        match self {
            Self::Pointer(_) => 2,
            Self::Box(readout) => readout.count * 5,
            Self::Multiple { labels, .. } => labels.len(),
            _ => 1,
        }
    }

    pub(super) fn answer<'a>(
        &self,
        logits: &mut impl Iterator<Item = &'a [f32]>,
        state: Option<&StatePositions>,
        found_calibration: &[Arc<[[f64; 2]]>; 3],
    ) -> Result<DecisionAnswer, DecisionError> {
        if let Self::Pointer(kind) = self {
            use super::parsing;
            let start = logits.next().expect("pointer start slot");
            let end = logits.next().expect("pointer end slot");
            let (index, cap) = match kind {
                PointerKind::Text => (0, 512),
                PointerKind::Number(_) => (1, 32),
                PointerKind::Date => (2, 32),
            };
            let calibration = &found_calibration[index];
            let extraction = match state {
                Some(state) => state.extract(start, end, cap, calibration)?,
                None => eredu_core::decision::SpanExtraction {
                    span: None,
                    found: interpolate(calibration, 0.0),
                },
            };
            let context = state.map_or("", StatePositions::text);
            let text = extraction.span.as_ref().map(|span| span.text.as_str());
            return Ok(match kind {
                PointerKind::Text => DecisionAnswer::Text(extraction),
                PointerKind::Number(unit) => DecisionAnswer::Number {
                    value: text.and_then(|text| parsing::number(text, context, unit.as_deref())),
                    unit: unit
                        .clone()
                        .filter(|unit| !unit.is_empty())
                        .or_else(|| text.and_then(parsing::unit)),
                    extraction,
                },
                PointerKind::Date => DecisionAnswer::Date {
                    value: text.and_then(|text| parsing::date(text, context)),
                    extraction,
                },
            });
        }
        let mut next = |readout: &ChoiceReadout| {
            readout.probabilities(
                logits
                    .next()
                    .expect("packed row has one vocabulary row per slot"),
            )
        };
        Ok(match self {
            Self::Box(readout) => readout.answer(next)?,
            Self::Pointer(_) => unreachable!("pointer readout handled above"),
            Self::Single(readout) => DecisionAnswer::SingleChoice(ChoiceAnswer {
                options: readout.labelled(next(readout)?),
            }),
            Self::YesNo(readout) => DecisionAnswer::YesNo {
                probability_yes: next(readout)?[0],
            },
            Self::Estimate {
                readout,
                values,
                levels,
                unit,
            } => {
                let probabilities = next(readout)?;
                let total: f64 = probabilities.iter().sum();
                let mut mass = 0.0;
                let mut quantiles = [*values.last().expect("nonempty estimate grid"); 3];
                let mut next = 0;
                for (&value, probability) in values.iter().zip(probabilities) {
                    mass += probability / total;
                    while next < levels.len() && mass >= levels[next] {
                        quantiles[next] = value;
                        next += 1;
                    }
                }
                let [low, median, high] = quantiles;
                DecisionAnswer::Estimate {
                    median,
                    interval: [low, high],
                    unit: unit.clone(),
                }
            }
            Self::Score(readout) => {
                let probabilities = next(readout)?;
                let total: f64 = probabilities.iter().sum();
                let mut mass = 0.0;
                let level = probabilities
                    .iter()
                    .position(|p| {
                        mass += p / total;
                        mass >= 0.5
                    })
                    .unwrap_or(probabilities.len() - 1);
                let confidence = probabilities.iter().copied().fold(0.0, f64::max);
                DecisionAnswer::Score {
                    level,
                    confidence,
                    levels: readout.labelled(probabilities),
                }
            }
            Self::Multiple {
                labels,
                binary,
                calibration,
            } => {
                let options = labels
                    .iter()
                    .map(|label| {
                        let probability = interpolate(calibration, next(binary)?[0]);
                        Ok(ChoiceProbability {
                            label: label.clone(),
                            probability,
                        })
                    })
                    .collect::<Result<Vec<_>, DecisionError>>()?;
                let selected = options
                    .iter()
                    .filter(|p| p.probability >= 0.5)
                    .map(|p| p.label.clone())
                    .collect();
                DecisionAnswer::MultipleChoice { selected, options }
            }
        })
    }
}

impl ChoiceReadout {
    pub(super) fn probabilities(&self, row: &[f32]) -> Result<Vec<f64>, DecisionError> {
        let scores = self
            .groups
            .iter()
            .map(|(_, ids)| {
                ids.iter().map(|&id| f64::from(row[id as usize])).try_fold(
                    f64::NEG_INFINITY,
                    |max, score| {
                        if score.is_finite() {
                            Ok(max.max(score))
                        } else {
                            Err(DecisionError::Request("non-finite option logit"))
                        }
                    },
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mut weights: Vec<_> = scores
            .into_iter()
            .map(|score| ((score - max) / self.temperature).exp())
            .collect();
        let total: f64 = weights.iter().sum();
        weights.iter_mut().for_each(|value| *value /= total);
        Ok(weights)
    }

    fn labelled(&self, probabilities: Vec<f64>) -> Vec<ChoiceProbability> {
        self.groups
            .iter()
            .zip(probabilities)
            .map(|((label, _), probability)| ChoiceProbability {
                label: label.clone(),
                probability,
            })
            .collect()
    }
}

pub(super) fn interpolate(knots: &[[f64; 2]], probability: f64) -> f64 {
    let index = knots.partition_point(|&[raw, _]| raw < probability);
    match (
        index.checked_sub(1).and_then(|i| knots.get(i)),
        knots.get(index),
    ) {
        (Some(&[left, a]), Some(&[right, b])) => {
            a + (probability - left) * (b - a) / (right - left)
        }
        (Some(&[_, value]), None) | (None, Some(&[_, value])) => value,
        (None, None) => probability,
    }
}
