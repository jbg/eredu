//! CLI-owned typed questions compiled into generic pinned token canvases.
use anyhow::{ensure, Context, Result};
use eredu::api::{
    LoadedModel, ObservedGenerationEvent, PreparedChatGenerationSettings, TraceLimits,
};
use eredu::runtime::chat::ChatTemplateRequest;
use eredu_core::{capture::*, refinement::*, GenerationConfigOverrides, TextGenerationBackend};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    io::Read,
    ops::ControlFlow,
    path::Path,
    time::Instant,
};

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Request {
    state: Value,
    questions: Vec<Question>,
    #[serde(default = "one")]
    steps: u32,
    #[serde(default)]
    seed: u64,
}
fn one() -> u32 {
    1
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum Question {
    Choice {
        id: String,
        question: String,
        options: Vec<String>,
    },
    Noul {
        id: String,
        question: String,
    },
    Score {
        id: String,
        question: String,
        levels: Vec<Level>,
    },
}
#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Level {
    label: String,
    value: f64,
    description: String,
}
impl Question {
    fn id(&self) -> &str {
        match self {
            Self::Choice { id, .. } | Self::Noul { id, .. } | Self::Score { id, .. } => id,
        }
    }
    fn text(&self) -> &str {
        match self {
            Self::Choice { question, .. }
            | Self::Noul { question, .. }
            | Self::Score { question, .. } => question,
        }
    }
    fn labels(&self) -> Vec<&str> {
        match self {
            Self::Choice { options, .. } => options.iter().map(String::as_str).collect(),
            Self::Noul { .. } => vec!["yes", "no"],
            Self::Score { levels, .. } => levels.iter().map(|l| l.label.as_str()).collect(),
        }
    }
}
impl Request {
    fn validate(&self) -> Result<()> {
        ensure!((1..=64).contains(&self.steps), "steps must be in 1..=64");
        ensure!(
            (1..=32).contains(&self.questions.len()),
            "questions must contain 1..=32 entries"
        );
        let mut ids = BTreeSet::new();
        for q in &self.questions {
            ensure!(
                !q.id().trim().is_empty() && ids.insert(q.id()),
                "question IDs must be nonempty and unique"
            );
            ensure!(
                !q.text().trim().is_empty(),
                "question instructions must be nonempty"
            );
            let labels = q.labels();
            ensure!(
                (2..=26).contains(&labels.len()),
                "each question requires 2..=26 alternatives"
            );
            ensure!(
                labels.iter().all(|l| !l.trim().is_empty())
                    && labels.iter().collect::<BTreeSet<_>>().len() == labels.len(),
                "alternative labels must be nonempty and unique"
            );
            if let Question::Score { levels, .. } = q {
                ensure!(
                    levels.iter().all(|l| l.value.is_finite()),
                    "score values must be finite"
                );
                ensure!(
                    levels.windows(2).all(|pair| pair[0].value < pair[1].value),
                    "score values must be strictly increasing"
                );
            }
        }
        Ok(())
    }
}
pub(crate) fn read(path: &Path) -> Result<Request> {
    let mut data = Vec::new();
    let reader: Box<dyn Read> = if path == Path::new("-") {
        Box::new(std::io::stdin())
    } else {
        Box::new(std::fs::File::open(path).with_context(|| format!("reading {}", path.display()))?)
    };
    reader.take(1_048_577).read_to_end(&mut data)?;
    ensure!(data.len() <= 1_048_576, "decision request exceeds 1 MiB");
    let request: Request = serde_json::from_slice(&data).context("parsing decision request")?;
    request.validate()?;
    Ok(request)
}

struct Slot {
    position: u64,
    ids: Vec<u32>,
}
struct Compiled {
    prompt: String,
    canvas: RefinementCanvas,
    slots: Vec<Slot>,
}
fn compile(
    request: &Request,
    length: usize,
    mut encode: impl FnMut(&str) -> Result<Vec<u32>>,
) -> Result<Compiled> {
    request.validate()?;
    let padding = encode("\n")?;
    let mut seed = Vec::new();
    let mut slots = Vec::new();
    let mut instructions = Vec::new();
    for (index, q) in request.questions.iter().enumerate() {
        let labels = q.labels();
        let candidates: Vec<_> = (0..labels.len())
            .map(|i| encode(&format!("q{index}: {}\n", (b'A' + i as u8) as char)))
            .collect::<Result<_>>()?;
        let first = &candidates[0];
        ensure!(
            !first.is_empty() && candidates.iter().all(|c| c.len() == first.len()),
            "answer labels have incompatible tokenization"
        );
        let differing: Vec<_> = (0..first.len())
            .filter(|p| candidates.iter().any(|c| c[*p] != first[*p]))
            .collect();
        ensure!(
            differing.len() == 1,
            "answer labels must differ in exactly one token in their template"
        );
        let position = differing[0];
        let ids: Vec<_> = candidates.iter().map(|c| c[position]).collect();
        ensure!(
            ids.iter().collect::<BTreeSet<_>>().len() == labels.len(),
            "answer labels do not have distinct token IDs"
        );
        slots.push(Slot {
            position: (seed.len() + position) as u64,
            ids,
        });
        seed.extend(
            first
                .iter()
                .enumerate()
                .map(|(p, id)| if p == position { None } else { Some(*id) }),
        );
        let alternatives: Vec<_> = labels.iter().enumerate().map(|(i, label)| json!({"code": ((b'A' + i as u8) as char).to_string(), "label": label})).collect();
        let mut instruction = json!({"slot": format!("q{index}"), "question": q.text(), "alternatives": alternatives});
        if let Question::Score { levels, .. } = q {
            instruction["levels"] = serde_json::to_value(levels)?;
        }
        instructions.push(instruction);
    }
    ensure!(
        seed.len() <= length,
        "answer template exceeds the model canvas of {length} tokens"
    );
    let padding = padding
        .first()
        .copied()
        .or_else(|| seed.iter().flatten().copied().last())
        .context("answer template has no fixed padding token")?;
    seed.resize(length, Some(padding));
    let pins = seed
        .iter()
        .enumerate()
        .filter_map(|(p, id)| id.map(|_| p as u32))
        .collect();
    let prompt = format!("Use the supplied evidence to answer each question. Select exactly one listed letter for each slot. Reply only with lines q0: A, q1: B, and so on, using the selected letters. Evidence and questions:\n{}", serde_json::to_string(&json!({"state": request.state, "questions": instructions}))?);
    Ok(Compiled {
        prompt,
        canvas: RefinementCanvas {
            seed_tokens: seed,
            pinned_positions: pins,
        },
        slots,
    })
}

fn probabilities(scores: &[f64]) -> Result<Vec<f64>> {
    ensure!(
        !scores.is_empty() && scores.iter().all(|v| v.is_finite()),
        "option scores must be finite"
    );
    let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let weights: Vec<_> = scores.iter().map(|v| (v - max).exp()).collect();
    let total: f64 = weights.iter().sum();
    Ok(weights.into_iter().map(|v| v / total).collect())
}
fn answer(question: &Question, scores: &[f64]) -> Result<Value> {
    let probabilities = probabilities(scores)?;
    let labels = question.labels();
    ensure!(
        labels.len() == probabilities.len(),
        "score count differs from alternatives"
    );
    let mut winner = 0;
    for i in 1..probabilities.len() {
        if probabilities[i] > probabilities[winner] {
            winner = i;
        }
    }
    let distribution: BTreeMap<_, _> = labels
        .iter()
        .zip(&probabilities)
        .map(|(l, p)| (*l, *p))
        .collect();
    let mut result = json!({"probabilities": distribution, "confidence": probabilities[winner]});
    match question {
        Question::Choice { .. } => {
            result["choice"] = json!(labels[winner]);
        }
        Question::Noul { .. } => {
            result["value"] = json!(winner == 0);
            result["probability"] = json!(probabilities[0]);
        }
        Question::Score { levels, .. } => {
            let expected: f64 = levels
                .iter()
                .zip(&probabilities)
                .map(|(l, p)| l.value * p)
                .sum();
            ensure!(expected.is_finite(), "score expectation overflowed");
            result["score"] = json!(expected);
            result["level"] = json!(labels[winner]);
        }
    }
    Ok(result)
}

pub(crate) fn run<B: TextGenerationBackend>(
    model: &mut LoadedModel<B>,
    request: Request,
    seed_override: Option<u64>,
    report_timing: bool,
) -> Result<Value> {
    let started = Instant::now();
    let length = model
        .refinement_canvas_length()
        .context("the selected model does not support pinned block refinement")?
        .get() as usize;
    let compiled = compile(&request, length, |text| Ok(model.encode(text, false)?))?;
    let canvas_prepared = Instant::now();
    let chat = model.prepare_chat(ChatTemplateRequest {
        messages: vec![json!({"role": "user", "content": compiled.prompt})],
        enable_thinking: Some(false),
        add_generation_prompt: true,
        ..Default::default()
    })?;
    let chat_prepared = Instant::now();
    let count = request.questions.len() as u64;
    // Row-sized native reductions include the full vocabulary; host output is bounded by selected IDs.
    let vocabulary = model
        .capture_declarations()?
        .catalog
        .get(eredu_core::MODEL_LOGITS_OBSERVATION_PATH)
        .and_then(|p| p.axes.as_ref())
        .and_then(|axes| axes.last())
        .and_then(|axis| match axis.dimension {
            eredu_core::SymbolicDimension::Known(n) => Some(n as u64),
            _ => None,
        })
        .context("model score vocabulary is not declared")?;
    let discovery_finished = Instant::now();
    let per_step = CaptureUsage {
        captures: count,
        retained_bytes: count * (vocabulary * 32 + (2 << 20)),
        host_bytes: count * (128 << 10),
        encoded_bytes: count * (128 << 10),
    };
    let plan = CapturePlan {
        schema_version: CAPTURE_SCHEMA_VERSION,
        provenance: eredu_core::capture::CaptureProvenance::Session,
        selections: compiled
            .slots
            .iter()
            .enumerate()
            .map(|(i, slot)| CaptureSelection {
                id: format!("slot-{i}"),
                path: eredu_core::MODEL_LOGITS_OBSERVATION_PATH.into(),
                schedule: CaptureSchedule {
                    prefill: false,
                    ..Default::default()
                },
                slices: vec![],
                transform: CaptureTransform::TokenScores {
                    position: Some(slot.position),
                    token_ids: slot.ids.clone(),
                },
            })
            .collect(),
        limits: CaptureLimits {
            per_step,
            cumulative: per_step.checked_mul(u64::from(request.steps) + 2)?,
            physical_native_bytes: None,
            on_limit: CaptureLimitPolicy::Fail,
        },
    };
    let seed = seed_override.unwrap_or(request.seed);
    let prepared = model
        .prepare_observed_chat(
            &chat,
            PreparedChatGenerationSettings {
                overrides: GenerationConfigOverrides {
                    max_new_tokens: Some(1),
                    ..Default::default()
                },
                seed,
                decoding_mode: Some(eredu_core::decoding::DecodingMode::Diffusion {
                    policy: Some(
                        EntropyRefinementOptions {
                            max_denoising_steps: request.steps,
                            stability_threshold: request.steps,
                            ..Default::default()
                        }
                        .try_into()?,
                    ),
                }),
                ..Default::default()
            },
            plan,
            TraceLimits {
                per_record_bytes: count * (256 << 10) + (1 << 20),
                total_bytes: (count * (256 << 10) + (1 << 20)) * (u64::from(request.steps) + 8),
            },
        )?
        .with_refinement_canvas(compiled.canvas);
    let mut latest = BTreeMap::new();
    let mut iteration = 0;
    let mut collect = |record: eredu::api::ControlledGenerationRecord| {
        if let ObservedGenerationEvent::Progress {
            progress,
            captures: Some(step),
            ..
        } = record.generation.event
        {
            if progress.phase == GenerationPhase::Refinement {
                iteration = progress.iteration;
                latest.clear();
                for capture in step.records {
                    if let Some(CapturePayload::TokenScores(scores)) = capture.payload {
                        latest.insert(capture.selection_id, scores);
                    }
                }
            }
        }
        ControlFlow::Continue(())
    };
    let prompt_tokens = prepared.prompt_token_ids().len();
    let context_started = Instant::now();
    let prepare_seconds = context_started.duration_since(started).as_secs_f64();
    let mut session = model.start_controlled(
        prepared,
        &[],
        Default::default(),
        &mut collect,
        eredu::api::OutputMode::Text,
    )?;
    session.step(&mut collect)?;
    let context_seconds = context_started.elapsed().as_secs_f64();
    let refinement_started = Instant::now();
    for _ in 0..request.steps {
        session.step(&mut collect)?;
    }
    let refinement_seconds = refinement_started.elapsed().as_secs_f64();
    let finish_started = Instant::now();
    session.cancel(&mut collect)?;
    drop(session);

    ensure!(
        iteration == request.steps,
        "refinement ended before the requested readout"
    );
    let mut answers = BTreeMap::new();
    for (i, question) in request.questions.iter().enumerate() {
        let row = latest
            .get(&format!("slot-{i}"))
            .context("missing requested position score capture")?;
        ensure!(
            row.scores
                .iter()
                .map(|s| s.target.token_id)
                .eq(compiled.slots[i].ids.iter().copied()),
            "capture token IDs differ from requested alternatives"
        );
        let scores: Vec<_> = row
            .scores
            .iter()
            .map(|s| f64::from(s.target.score))
            .collect();
        let mut value = answer(question, &scores)?;
        let label_mass: f64 = row.scores.iter().map(|s| s.log_probability.exp()).sum();
        ensure!(label_mass.is_finite(), "invalid full-vocabulary label mass");
        value["label_mass"] = json!(label_mass.clamp(0.0, 1.0));
        answers.insert(question.id(), value);
    }
    if report_timing {
        eprintln!("canvas_prepare_seconds={:.3} chat_prepare_seconds={:.3} discovery_seconds={:.3} admission_seconds={:.3}", canvas_prepared.duration_since(started).as_secs_f64(), chat_prepared.duration_since(canvas_prepared).as_secs_f64(), discovery_finished.duration_since(chat_prepared).as_secs_f64(), context_started.duration_since(discovery_finished).as_secs_f64());
        eprintln!("prepare_seconds={prepare_seconds:.3} context_seconds={context_seconds:.3} refinement_seconds={refinement_seconds:.3} finish_seconds={:.3} prompt_tokens={prompt_tokens} canvas_tokens={length}", finish_started.elapsed().as_secs_f64());
    }
    Ok(
        json!({"steps": request.steps, "seed": seed, "normalization": "allowed_labels", "answers": answers}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn typed_answers_use_label_softmax_and_declared_scale_values() {
        let choice = Question::Choice {
            id: "route".into(),
            question: "route".into(),
            options: vec!["a".into(), "b".into()],
        };
        assert_eq!(
            answer(&choice, &[0.0, 3.0_f64.ln()]).unwrap()["probabilities"]["b"],
            0.75
        );
        let noul = Question::Noul {
            id: "yes".into(),
            question: "yes".into(),
        };
        assert_eq!(
            answer(&noul, &[3.0_f64.ln(), 0.0]).unwrap()["probability"],
            0.75
        );
        let score = Question::Score {
            id: "value".into(),
            question: "value".into(),
            levels: vec![
                Level {
                    label: "low".into(),
                    value: 10.0,
                    description: "low".into(),
                },
                Level {
                    label: "high".into(),
                    value: 30.0,
                    description: "high".into(),
                },
            ],
        };
        assert_eq!(answer(&score, &[0.0, 3.0_f64.ln()]).unwrap()["score"], 25.0);
        assert!(probabilities(&[f64::NAN]).is_err());
    }
    #[test]
    fn compiler_pins_scaffold_and_padding_but_leaves_slots_as_noise() {
        let request: Request = serde_json::from_value(json!({"state": "evidence", "questions": [{"id": "x", "type": "choice", "question": "route", "options": ["a", "b"]}]})).unwrap();
        let compiled = compile(&request, 16, |s| Ok(s.bytes().map(u32::from).collect())).unwrap();
        let slot = compiled.slots[0].position as usize;
        assert_eq!(slot, 4);
        assert_eq!(compiled.slots[0].ids, [65, 66]);
        assert_eq!(compiled.canvas.seed_tokens[slot], None);
        assert_eq!(compiled.canvas.pinned_positions.len(), 15);
        assert!(!compiled.canvas.pinned_positions.contains(&(slot as u32)));
        assert!(compile(&request, 3, |s| Ok(s.bytes().map(u32::from).collect())).is_err());
        assert!(compile(&request, 16, |_| Ok(vec![0])).is_err());
    }
}
