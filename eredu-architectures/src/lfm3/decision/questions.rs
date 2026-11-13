//! d1 text prompt rendering and slot admission.

use std::collections::BTreeSet;

use eredu_core::decision::{ChoiceQuestion, DecisionQuestion};

use super::readout::{ChoiceReadout, PointerKind, Readout};
use super::{DecisionError, QuestionPlan, TextChoicePreparation, IM_END, IM_START};

const OPTION_LETTERS: std::ops::RangeInclusive<char> = 'A'..='Z';

impl TextChoicePreparation {
    pub(super) fn single_choice(
        &self,
        question: &ChoiceQuestion,
    ) -> Result<QuestionPlan, DecisionError> {
        validate_choice(question)?;
        let count = question.options.len();
        let labels: Vec<_> = question
            .options
            .iter()
            .map(|option| option.label.trim())
            .collect();
        let native_letters = labels.iter().all(|label| {
            let mut chars = label.chars();
            chars.next().is_some_and(char::is_alphabetic) && chars.next().is_none()
        });
        let codes = if native_letters {
            labels.iter().map(|label| (*label).to_owned()).collect()
        } else {
            if count > 26 + 26 * 26 {
                return Err(DecisionError::Request(
                    "choice exceeds the option code space",
                ));
            }
            letter_codes().take(count).collect::<Vec<_>>()
        };
        let mut used = BTreeSet::new();
        let mut body = format!("{}\n\nOptions:\n", question.instructions);
        let mut groups = Vec::with_capacity(count);
        for (index, (option, code)) in question.options.iter().zip(codes).enumerate() {
            let (code, id) = self.alias(code, &mut used)?;
            let description = option
                .description
                .as_deref()
                .filter(|desc| !desc.is_empty())
                .map(str::to_owned)
                .unwrap_or_else(|| option.label.replace('_', " "));
            if index > 0 {
                body.push('\n');
            }
            body.push_str(&format!("{code} {description}"));
            let ids = self.with_spaced_form(&code, id)?;
            groups.push((option.label.clone(), ids));
        }
        body.push_str("\n\nReply with the option code only.");
        Ok(QuestionPlan::single(
            self.suffix(&body),
            Readout::Single(ChoiceReadout {
                groups,
                temperature: self.calibration.choice_temperature(count),
            }),
        ))
    }

    pub(super) fn decision(
        &self,
        question: &DecisionQuestion,
    ) -> Result<QuestionPlan, DecisionError> {
        match question {
            DecisionQuestion::Box { instructions } => self.box_question(instructions, false, None),
            DecisionQuestion::Boxes { instructions, max } => {
                self.box_question(instructions, true, *max)
            }
            DecisionQuestion::Estimate {
                instructions,
                range,
                step,
                unit,
            } => self.estimate(instructions, *range, *step, unit.as_deref()),
            DecisionQuestion::ExtractText { instructions } => {
                self.pointer_question(instructions, PointerKind::Text)
            }
            DecisionQuestion::ReadNumber { instructions, unit } => {
                self.pointer_question(instructions, PointerKind::Number(unit.clone()))
            }
            DecisionQuestion::ReadDate { instructions } => {
                self.pointer_question(instructions, PointerKind::Date)
            }
            DecisionQuestion::SingleChoice(question) => self.single_choice(question),
            DecisionQuestion::MultipleChoice(question) => self.multiple_choice(question),
            DecisionQuestion::YesNo {
                instructions,
                criteria,
            } => {
                validate_instructions(instructions)?;
                let mut body = instructions.clone();
                if let Some([yes, no]) = criteria {
                    body.push_str(&format!("\nYes: {yes}\nNo: {no}"));
                }
                body.push_str("\n\nReply with yes or no only.");
                Ok(QuestionPlan::single(
                    self.suffix(&body),
                    Readout::YesNo(self.binary_readout(self.calibration.temperature("noul"))?),
                ))
            }
            DecisionQuestion::Score {
                instructions,
                levels,
            } => {
                validate_instructions(instructions)?;
                if !(2..=10).contains(&levels.len()) {
                    return Err(DecisionError::Request(
                        "an ordinal score requires two to ten levels",
                    ));
                }
                let mut body = format!("{instructions}\n");
                let mut groups = Vec::with_capacity(levels.len());
                for (index, label) in levels.iter().enumerate() {
                    body.push_str(&format!("\n{index} {label}"));
                    let id = self.single_token(&index.to_string())?.ok_or(
                        DecisionError::Declaration("a score digit must be one token"),
                    )?;
                    groups.push((label.clone(), vec![id]));
                }
                body.push_str(&format!(
                    "\n\nReply with a single digit 0-{} only.",
                    levels.len() - 1
                ));
                Ok(QuestionPlan::single(
                    self.suffix(&body),
                    Readout::Score(ChoiceReadout {
                        groups,
                        temperature: self
                            .calibration
                            .temperature(&format!("score:{}", levels.len())),
                    }),
                ))
            }
        }
    }

    pub(super) fn spoken_decision(
        &self,
        question: &DecisionQuestion,
        markup: &str,
    ) -> Result<QuestionPlan, DecisionError> {
        // The publisher computes answer offsets with this marker in the instructions,
        // then substitutes the expanded clip before tokenizing the complete suffix.
        const SPOKEN: &str = "\u{e000}";
        let mut question = question.clone();
        let instructions = match &mut question {
            DecisionQuestion::Box { instructions }
            | DecisionQuestion::Boxes { instructions, .. }
            | DecisionQuestion::Estimate { instructions, .. }
            | DecisionQuestion::ExtractText { instructions }
            | DecisionQuestion::ReadNumber { instructions, .. }
            | DecisionQuestion::ReadDate { instructions }
            | DecisionQuestion::YesNo { instructions, .. }
            | DecisionQuestion::Score { instructions, .. } => instructions,
            DecisionQuestion::SingleChoice(question)
            | DecisionQuestion::MultipleChoice(question) => &mut question.instructions,
        };
        SPOKEN.clone_into(instructions);
        let mut plan = self.decision(&question)?;
        let rest = plan
            .suffix
            .strip_prefix(SPOKEN)
            .ok_or(DecisionError::Declaration(
                "spoken marker is missing from rendered question",
            ))?;
        plan.suffix = format!("{markup}{rest}");
        Ok(plan)
    }

    fn box_question(
        &self,
        instructions: &str,
        multiple: bool,
        max: Option<usize>,
    ) -> Result<QuestionPlan, DecisionError> {
        use super::boxes::{BoxReadout, DEFAULT_INSTANCES, EDGES};
        validate_instructions(instructions)?;
        let count = if multiple {
            max.unwrap_or(DEFAULT_INSTANCES)
        } else {
            1
        };
        let limit = if self.box_prompt == 1 {
            DEFAULT_INSTANCES
        } else {
            64
        };
        if !(1..=limit).contains(&count) {
            return Err(DecisionError::Request(
                "box count exceeds the checkpoint prompt limit",
            ));
        }
        let numeric = if self.numeric_codes {
            self.number_groups(&(0..1000).map(|i| i as f64).collect::<Vec<_>>())?
        } else {
            None
        };
        let is_numeric = numeric.is_some();
        let (groups, step, legend) = match numeric {
            Some(groups) => (groups, 0.001, "Each edge is a number from 0 to 999, in thousandths of the picture: top and bottom from its top edge, left and right from its left edge.".into()),
            None => {
                let mut used = BTreeSet::new();
                let groups = letter_codes().take(128).map(|code| {
                    let (code, id) = self.alias(code, &mut used)?;
                    let ids = self.with_spaced_form(&code, id)?;
                    Ok((code, ids))
                }).collect::<Result<Vec<_>, DecisionError>>()?;
                let legend = format!("Each edge is a code on a 128-value scale: {} is 0 (top or left), {} is 1 (bottom or right). Values are equally spaced.", groups[0].0, groups[127].0);
                (groups, 1.0 / 127.0, legend)
            }
        };
        let edges = std::iter::once("present").chain(EDGES);
        let (body, names, after) = if multiple {
            let most = if self.box_prompt == 1 {
                "16 at most, "
            } else {
                ""
            };
            let body = format!("{instructions}\n\nEvery one of them, {most}in reading order (left edge, then top). Present k is yes when there are at least k. {legend} Reply with yes or no at each present, then its top, bottom, left and right.");
            let names = (1..=count)
                .flat_map(|k| edges.clone().map(move |name| format!("{name} {k}")))
                .collect::<Vec<_>>();
            (
                body,
                names,
                count < DEFAULT_INSTANCES || self.box_prompt > 1,
            )
        } else {
            let one = if is_numeric { "one number" } else { "one code" };
            let body = format!("{instructions}\n\nPresence is yes or no. {legend} Reply with yes or no at present, then {one} at top, bottom, left and right.");
            (body, edges.map(str::to_owned).collect(), false)
        };
        let (suffix, offsets) = self.marked_suffix_with_tail(&body, &names, after)?;
        let index = if multiple { 2 } else { 0 };
        Ok(QuestionPlan {
            suffix,
            offsets,
            readout: Readout::Box(BoxReadout {
                count,
                multiple,
                step,
                presence: self.binary_readout(1.0)?,
                edges: ChoiceReadout {
                    groups,
                    temperature: 1.0,
                },
                interpolate_edges: self.interpolate_edges,
                found: self.box_calibration[index].clone(),
                confidence: self.box_calibration[index + 1].clone(),
            }),
        })
    }

    fn estimate(
        &self,
        instructions: &str,
        [low, high]: [f64; 2],
        step: Option<f64>,
        unit: Option<&str>,
    ) -> Result<QuestionPlan, DecisionError> {
        validate_instructions(instructions)?;
        let width = high - low;
        if !width.is_finite() || width <= 0.0 || step.is_some_and(|s| !s.is_finite() || s <= 0.0) {
            return Err(DecisionError::Request(
                "an estimate requires finite increasing bounds and positive spacing",
            ));
        }
        let size = step.map_or(128, |step| {
            ((width / step).round_ties_even() + 1.0).min(128.0) as usize
        });
        if size < 2 {
            return Err(DecisionError::Request(
                "an estimate grid requires at least two values",
            ));
        }
        let spacing = width / (size - 1) as f64;
        let mut values: Vec<_> = (0..size - 1).map(|i| low + i as f64 * spacing).collect();
        values.push(high);
        let unit_text = unit.map_or_else(String::new, |unit| format!(" {unit}"));
        let start = general_number(low);
        let end = general_number(high);
        let step_text = general_number(spacing);
        let numeric = if self.numeric_codes {
            self.number_groups(&values)?
        } else {
            None
        };
        let (body, groups) = if let Some(groups) = numeric {
            (format!("{instructions}\n\nEstimate from {start} to {end}{unit_text}, in steps of {step_text}. Reply with the number only."), groups)
        } else {
            let mut used = BTreeSet::new();
            let groups = if size <= 10 {
                (0..size)
                    .map(|i| {
                        let code = i.to_string();
                        let id = self.single_token(&code)?.ok_or(DecisionError::Declaration(
                            "an estimate digit must be one token",
                        ))?;
                        Ok((code, vec![id]))
                    })
                    .collect::<Result<Vec<_>, DecisionError>>()?
            } else {
                letter_codes()
                    .take(size)
                    .map(|code| {
                        let (code, id) = self.alias(code, &mut used)?;
                        let ids = self.with_spaced_form(&code, id)?;
                        Ok((code, ids))
                    })
                    .collect::<Result<Vec<_>, DecisionError>>()?
            };
            let first = &groups[0].0;
            let last = &groups[size - 1].0;
            (format!("{instructions}\n\nEstimate from {start} to {end}{unit_text} on {size} equally spaced values, step {step_text}: {first} is {start}, {last} is {end}. Reply with the code only."), groups)
        };
        Ok(QuestionPlan::single(
            self.suffix(&body),
            Readout::Estimate {
                readout: ChoiceReadout {
                    groups,
                    temperature: 1.0,
                },
                values,
                levels: self.estimate_levels,
                unit: unit.map(str::to_owned),
            },
        ))
    }

    fn number_groups(
        &self,
        values: &[f64],
    ) -> Result<Option<Vec<(String, Vec<u32>)>>, DecisionError> {
        let mut groups = Vec::with_capacity(values.len());
        for &value in values {
            let integer = value.round_ties_even();
            let tolerance = 1e-6_f64.max(1e-9 * integer.abs().max(value.abs()));
            if (value - integer).abs() > tolerance {
                return Ok(None);
            }
            let code = if integer == 0.0 {
                "0".into()
            } else {
                format!("{integer:.0}")
            };
            let Some(id) = self.single_token(&code)? else {
                return Ok(None);
            };
            let ids = self.with_spaced_form(&code, id)?;
            groups.push((code, ids));
        }
        Ok(Some(groups))
    }

    fn with_spaced_form(&self, code: &str, id: u32) -> Result<Vec<u32>, DecisionError> {
        let mut ids = vec![id];
        if let Some(spaced) = self
            .single_token(&format!(" {code}"))?
            .filter(|&other| other != id)
        {
            ids.push(spaced);
        }
        Ok(ids)
    }

    fn pointer_question(
        &self,
        instructions: &str,
        kind: PointerKind,
    ) -> Result<QuestionPlan, DecisionError> {
        validate_instructions(instructions)?;
        let unit = match &kind {
            PointerKind::Number(Some(unit)) if !unit.is_empty() => format!(" (unit: {unit})"),
            _ => String::new(),
        };
        let body = format!(
            "{instructions}\n\nRead the {}{unit} from the state, or nothing if it is absent.",
            kind.name()
        );
        let (suffix, offsets) = self.marked_suffix(&body, &["start".into(), "end".into()])?;
        Ok(QuestionPlan {
            suffix,
            offsets,
            readout: Readout::Pointer(kind),
        })
    }

    fn suffix(&self, body: &str) -> String {
        format!(
            "{}{IM_END}\n{IM_START}assistant\n<think></think>",
            self.plain(body)
        )
    }

    fn binary_readout(&self, temperature: f64) -> Result<ChoiceReadout, DecisionError> {
        let mut groups = Vec::with_capacity(2);
        for forms in [["yes", "Yes", "YES"], ["no", "No", "NO"]] {
            let mut ids = Vec::new();
            for form in forms {
                if let Some(id) = self.single_token(form)?.filter(|id| !ids.contains(id)) {
                    ids.push(id);
                }
            }
            if ids.is_empty() {
                return Err(DecisionError::Declaration(
                    "missing single-token yes/no forms",
                ));
            }
            groups.push((forms[0].to_owned(), ids));
        }
        Ok(ChoiceReadout {
            groups,
            temperature,
        })
    }

    fn multiple_choice(&self, question: &ChoiceQuestion) -> Result<QuestionPlan, DecisionError> {
        validate_choice(question)?;
        let mut body = format!("{}\n\nOptions:", question.instructions);
        let mut names = Vec::new();
        for option in &question.options {
            let name = serde_json::to_string(&option.label)?;
            let description = option
                .description
                .as_deref()
                .filter(|desc| !desc.is_empty())
                .unwrap_or(&option.label);
            body.push_str(&format!("\n{name}: {description}"));
            names.push(name);
        }
        body.push_str("\n\nReply with yes or no for each option.");
        let (suffix, offsets) = self.marked_suffix(&body, &names)?;
        Ok(QuestionPlan {
            suffix,
            offsets,
            readout: Readout::Multiple {
                labels: question
                    .options
                    .iter()
                    .map(|option| option.label.clone())
                    .collect(),
                binary: self.binary_readout(1.0)?,
                calibration: self.multiple_calibration.clone(),
            },
        })
    }

    fn marked_suffix(
        &self,
        body: &str,
        names: &[String],
    ) -> Result<(String, Vec<usize>), DecisionError> {
        self.marked_suffix_with_tail(body, names, false)
    }

    fn marked_suffix_with_tail(
        &self,
        body: &str,
        names: &[String],
        after: bool,
    ) -> Result<(String, Vec<usize>), DecisionError> {
        let mut suffix = self.suffix(body);
        let mut ends = Vec::with_capacity(names.len());
        for (index, name) in names.iter().enumerate() {
            if index != 0 {
                suffix.push('\n');
            }
            suffix.push_str(&self.plain(name));
            suffix.push(':');
            ends.push(suffix.len());
        }
        if after {
            let candidate = format!("{suffix}\n");
            let ids = self.encode(&candidate)?;
            if let Some(&last) = ids.last() {
                let text = self
                    .tokenizer
                    .decode(&[last], false)
                    .map_err(|error| DecisionError::Tokenizer(error.to_string()))?;
                if text.trim_end().ends_with(':') {
                    suffix = candidate;
                }
            }
        }
        let encoded = self
            .tokenizer
            .encode(suffix.as_str(), false)
            .map_err(|error| DecisionError::Tokenizer(error.to_string()))?;
        let offsets = encoded.get_offsets();
        let ids = encoded.get_ids();
        let positions = ends
            .into_iter()
            .map(|end| {
                let mut matches = offsets
                    .iter()
                    .enumerate()
                    .filter(|(_, (start, stop))| *start < end && end <= *stop);
                let (index, &(_, stop)) = matches.next().ok_or(DecisionError::Request(
                    "a slot must have an exact colon token",
                ))?;
                let decoded = self
                    .tokenizer
                    .decode(&ids[index..index + 1], false)
                    .map_err(|error| DecisionError::Tokenizer(error.to_string()))?;
                if matches.next().is_some()
                    || !suffix
                        .get(end..stop)
                        .is_some_and(|tail| tail.trim().is_empty())
                    || !decoded.trim_end().ends_with(':')
                {
                    return Err(DecisionError::Request(
                        "a slot token must end with its colon",
                    ));
                }
                Ok(ids.len() - 1 - index)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok((suffix, positions))
    }

    fn alias(
        &self,
        code: String,
        used: &mut BTreeSet<u32>,
    ) -> Result<(String, u32), DecisionError> {
        let pool = std::iter::once(code)
            .chain(OPTION_LETTERS.map(|c| c.to_string()))
            .chain((0..100).map(|i| format!("{i:02}")))
            .chain(('a'..='z').map(|c| c.to_string()))
            .chain((0..200).map(|i| format!("#{i}")))
            .chain(letter_pairs());
        for code in pool {
            if let Some(id) = self.single_token(&code)?.filter(|&id| used.insert(id)) {
                return Ok((code, id));
            }
        }
        Err(DecisionError::Request(
            "no distinct single-token choice alias remains",
        ))
    }

    pub(super) fn encode(&self, text: &str) -> Result<Vec<u32>, DecisionError> {
        let encoded = self
            .tokenizer
            .encode(text, false)
            .map_err(|error| DecisionError::Tokenizer(error.to_string()))?;
        self.admit_ids(encoded.get_ids())
    }

    pub(super) fn admit_ids(&self, ids: &[u32]) -> Result<Vec<u32>, DecisionError> {
        if ids.iter().any(|&id| id as usize >= self.vocab) {
            return Err(DecisionError::Request(
                "token ID exceeds the model vocabulary",
            ));
        }
        Ok(ids.to_vec())
    }

    fn single_token(&self, text: &str) -> Result<Option<u32>, DecisionError> {
        Ok(match self.encode(text)?.as_slice() {
            &[id] => Some(id),
            _ => None,
        })
    }

    pub(super) fn plain(&self, text: &str) -> String {
        self.escape(text, |_| {})
    }

    pub(super) fn escape(&self, text: &str, mut inserted: impl FnMut(usize)) -> String {
        let mut text = text.to_owned();
        while let Some((at, token)) = self
            .special_tokens
            .iter()
            .filter_map(|token| text.find(token).map(|at| (at, token)))
            .min_by_key(|&(at, token)| (at, std::cmp::Reverse(token.len())))
        {
            let first = token.chars().next().expect("nonempty special token");
            let at = at + first.len_utf8();
            text.insert(at, '\u{200b}');
            inserted(at);
        }
        text
    }
}

fn letter_codes() -> impl Iterator<Item = String> {
    OPTION_LETTERS.map(|c| c.to_string()).chain(letter_pairs())
}

fn letter_pairs() -> impl Iterator<Item = String> {
    OPTION_LETTERS.flat_map(|a| OPTION_LETTERS.map(move |b| format!("{a}{b}")))
}

fn validate_instructions(instructions: &str) -> Result<(), DecisionError> {
    if instructions.trim().is_empty() {
        return Err(DecisionError::Request("a question requires instructions"));
    }
    Ok(())
}

fn validate_choice(question: &ChoiceQuestion) -> Result<(), DecisionError> {
    validate_instructions(&question.instructions)?;
    let unique: BTreeSet<_> = question
        .options
        .iter()
        .map(|option| &option.label)
        .collect();
    if unique.is_empty() || unique.len() != question.options.len() {
        return Err(DecisionError::Request(
            "a choice requires unique option labels",
        ));
    }
    Ok(())
}

// The publisher renders grid geometry with six significant digits.
fn general_number(value: f64) -> String {
    let scientific = format!("{value:.5e}");
    let (mantissa, exponent) = scientific.split_once('e').expect("scientific number");
    let exponent: i32 = exponent.parse().expect("scientific exponent");
    let trim = |s: &str| -> String {
        if s.contains('.') {
            s.trim_end_matches('0').trim_end_matches('.').into()
        } else {
            s.into()
        }
    };
    if (-4..6).contains(&exponent) {
        trim(&format!(
            "{value:.precision$}",
            precision = (5 - exponent) as usize
        ))
    } else {
        format!("{}e{exponent:+03}", trim(mantissa))
    }
}
