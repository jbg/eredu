//! Typed decisions evaluated with a pinned refinement canvas.
use crate::api::{LoadedModel, PreparedChatGenerationSettings};
use crate::runtime::chat::ChatTemplateRequest;
use anyhow::{ensure, Context, Result};
use eredu_core::{
    refinement::*, GenerationConfigOverrides, TextExecutionSession, TextGenerationBackend,
};
use serde_json::json;
use std::{collections::BTreeSet, ops::ControlFlow};

use super::{DecisionAnswer, DecisionQuestion, DecisionRequest, DecisionResponse, DecisionResult};
use eredu_core::decision::{ChoiceAnswer, ChoiceProbability};

fn alternatives(q: &DecisionQuestion) -> Result<(&str, Vec<ChoiceProbability>)> {
    let option = |label: &str| ChoiceProbability {
        label: label.into(),
        probability: 0.0,
    };
    let (text, options) = match q {
        DecisionQuestion::SingleChoice(q) => (&*q.instructions, q.options.iter().map(|o| option(&o.label)).collect()),
        DecisionQuestion::YesNo { instructions, .. } => (&**instructions, vec![option("yes"), option("no")]),
        DecisionQuestion::Score { instructions, levels } => (&**instructions, levels.iter().map(|l| option(l)).collect()),
        _ => anyhow::bail!("the selected refinement model supports single choice, yes/no and ordinal score questions"),
    };
    ensure!(
        (2..=26).contains(&options.len()),
        "refinement requires 2..=26 alternatives"
    );
    ensure!(
        !text.trim().is_empty(),
        "question instructions must be nonempty"
    );
    ensure!(
        options.iter().all(|o| !o.label.trim().is_empty())
            && options
                .iter()
                .map(|o| &o.label)
                .collect::<BTreeSet<_>>()
                .len()
                == options.len(),
        "alternative labels must be nonempty and unique"
    );
    Ok((text, options))
}

struct Compiled {
    prompt: String,
    canvas: RefinementCanvas,
    slots: Vec<RefinementScoreSelection>,
}
fn compile(
    request: &DecisionRequest,
    length: usize,
    mut encode: impl FnMut(&str) -> Result<Vec<u32>>,
) -> Result<Compiled> {
    let padding = encode("\n")?;
    let mut seed = Vec::new();
    let mut slots = Vec::new();
    let mut instructions = Vec::new();
    for (index, q) in request.questions.iter().enumerate() {
        let (text, options) = alternatives(&q.question)?;
        let labels: Vec<_> = options.iter().map(|o| &o.label).collect();
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
        slots.push(RefinementScoreSelection {
            position: u32::try_from(seed.len() + position)?,
            token_ids: ids,
        });
        seed.extend(
            first
                .iter()
                .enumerate()
                .map(|(p, id)| if p == position { None } else { Some(*id) }),
        );
        let alternatives: Vec<_> = labels.iter().enumerate().map(|(i, label)| json!({"code": ((b'A' + i as u8) as char).to_string(), "label": label})).collect();
        let mut instruction =
            json!({"slot": format!("q{index}"), "question": text, "alternatives": alternatives});
        match &q.question {
            DecisionQuestion::SingleChoice(q) => {
                instruction["descriptions"] =
                    json!(q.options.iter().map(|o| &o.description).collect::<Vec<_>>())
            }
            DecisionQuestion::YesNo { criteria, .. } => {
                instruction["descriptions"] = json!(criteria)
            }
            _ => {}
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
fn answer(question: &DecisionQuestion, scores: &[f64]) -> Result<DecisionAnswer> {
    let probabilities = probabilities(scores)?;
    let (_, mut options) = alternatives(question)?;
    ensure!(
        options.len() == probabilities.len(),
        "score count differs from alternatives"
    );
    for (option, probability) in options.iter_mut().zip(probabilities) {
        option.probability = probability;
    }
    Ok(match question {
        DecisionQuestion::YesNo { .. } => DecisionAnswer::YesNo {
            probability_yes: options[0].probability,
        },
        DecisionQuestion::Score { .. } => {
            let mut cumulative = 0.0;
            let level = options
                .iter()
                .position(|o| {
                    cumulative += o.probability;
                    cumulative >= 0.5
                })
                .unwrap_or(options.len() - 1);
            let confidence = options.iter().map(|o| o.probability).fold(0.0, f64::max);
            DecisionAnswer::Score {
                level,
                confidence,
                levels: options,
            }
        }
        _ => DecisionAnswer::SingleChoice(ChoiceAnswer { options }),
    })
}

/// Evaluates a text request using a loaded model's pinned refinement canvas.
pub fn evaluate_refinement<B: TextGenerationBackend, S: TextExecutionSession<B>>(
    model: &mut LoadedModel<B, S>,
    request: &DecisionRequest,
) -> Result<DecisionResponse> {
    request.validate()?;
    ensure!(
        request.media.is_empty() && request.questions.iter().all(|q| q.spoken.is_none()),
        "the selected refinement model requires text state and written instructions"
    );
    let steps = request.refinement_steps.unwrap_or(1);
    ensure!(
        (1..=64).contains(&steps),
        "refinement_steps must be in 1..=64"
    );
    let capabilities = model.runtime.session().decoding_capabilities();
    let policy = if let Some(mask_token_id) = capabilities
        .mask_token_id
        .filter(|_| capabilities.masked_unmasking)
    {
        // Every alternative is scored while its answer slot is still masked.
        ensure!(
            steps == 1,
            "masked decision readout requires one refinement step"
        );
        RefinementPolicy::MaskedUnmasking {
            config: MaskedRefinementOptions {
                max_denoising_steps: steps,
                ..Default::default()
            }
            .try_into()?,
            mask_token_id,
        }
    } else {
        EntropyRefinementOptions {
            max_denoising_steps: steps,
            stability_threshold: steps,
            ..Default::default()
        }
        .try_into()?
    };
    let length = model
        .refinement_canvas_length()
        .context("the selected model does not support pinned block refinement")?
        .get() as usize;
    let compiled = compile(&request, length, |text| Ok(model.encode(text, false)?))?;
    let chat = model.prepare_chat(ChatTemplateRequest {
        messages: vec![json!({"role": "user", "content": compiled.prompt})],
        enable_thinking: Some(false),
        add_generation_prompt: true,
        ..Default::default()
    })?;
    let seed = request.seed.unwrap_or_default();
    let prepared = model
        .prepare_controlled_chat(
            &chat,
            PreparedChatGenerationSettings {
                overrides: GenerationConfigOverrides {
                    max_new_tokens: Some(1),
                    ..Default::default()
                },
                seed,
                decoding_mode: Some(eredu_core::decoding::DecodingMode::Diffusion {
                    policy: Some(policy),
                }),
                ..Default::default()
            },
        )?
        .with_refinement_canvas(compiled.canvas);
    let mut collect = |_| ControlFlow::Continue(());
    let prompt_tokens = prepared.prompt_token_ids().len();
    let mut session = model.start_controlled(
        prepared,
        &[],
        Default::default(),
        &mut collect,
        crate::api::OutputMode::Text,
    )?;
    session.step(&mut collect)?;
    for _ in 1..steps {
        session.step(&mut collect)?;
    }
    let rows = session.step_with_refinement_readout(compiled.slots.clone(), &mut collect)?;
    let iteration = session.position().iteration;
    session.cancel(&mut collect)?;

    ensure!(
        iteration == steps,
        "refinement ended before the requested readout"
    );
    ensure!(
        rows.len() == request.questions.len(),
        "missing requested refinement score rows"
    );
    let mut answers = Vec::new();
    for ((row, selection), question) in rows.iter().zip(&compiled.slots).zip(&request.questions) {
        ensure!(
            row.iter()
                .map(|score| score.token_id)
                .eq(selection.token_ids.iter().copied()),
            "selected refinement scores differ from the requested token IDs"
        );
        let scores: Vec<_> = row.iter().map(|s| f64::from(s.score)).collect();
        let label_mass: f64 = row.iter().map(|s| s.log_probability.exp()).sum();
        ensure!(label_mass.is_finite(), "invalid full-vocabulary label mass");
        answers.push(DecisionResult {
            id: question.id.clone(),
            answer: answer(&question.question, &scores)?,
            label_mass: Some(label_mass.clamp(0.0, 1.0)),
        });
    }
    Ok(DecisionResponse {
        answers,
        input_tokens: prompt_tokens,
        generated_tokens: 0,
        calibration_sha256: None,
    })
}
