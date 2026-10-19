use super::*;
use eredu_core::refinement::{EntropyRefinementConfig, EntropyRefinementOptions};
use std::cell::Cell;

struct Numeric;

struct Context {
    fail_settlement: Cell<bool>,
}

fn probabilities(row: &[f32]) -> Vec<f32> {
    let maximum = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let weights: Vec<_> = row.iter().map(|x| f32::exp(x - maximum)).collect();
    let total: f32 = weights.iter().sum();
    weights.into_iter().map(|x| x / total).collect()
}

fn entropy(row: &[f32]) -> f32 {
    probabilities(row)
        .into_iter()
        .filter(|p| *p > 0.0)
        .map(|p| -p * p.ln())
        .sum()
}

impl RefinementBackend for Numeric {
    type Logits = Vec<Vec<f32>>;
    type Token = Vec<u32>;
    type RandomState = u64;
    type Context = Context;
    type Error = &'static str;
    type Mask = Vec<bool>;
    type ConditioningFormat = ();

    fn initialize_masked_canvas(
        canvas: NonZeroU32,
        mask_token: u32,
        layout: Option<&eredu_core::refinement::RefinementCanvas>,
        context: &Context,
    ) -> Result<Option<(Self::Token, u32)>, Self::Error> {
        let mut tokens = vec![mask_token; canvas.get() as usize];
        if let Some(layout) = layout {
            tokens = Self::seed_canvas(&tokens, layout, context)?.0;
        }
        let count = tokens.iter().filter(|x| **x == mask_token).count() as u32;
        Ok(Some((tokens, count)))
    }

    fn masked_refinement_step(
        canvas: &Self::Token,
        previous: Option<&Self::Logits>,
        logits: &Self::Logits,
        mask_token: u32,
        config: eredu_core::refinement::MaskedRefinementConfig,
        scheduled: u32,
        random: &mut u64,
        _: &Context,
    ) -> Result<Option<MaskedRefinementStep<Self>>, Self::Error> {
        let options = config.options();
        let eligible: Vec<_> = canvas.iter().map(|x| *x == mask_token).collect();
        let sampled: Vec<_> = logits
            .iter()
            .map(|row| {
                masked::sample_position(row, options.sampling, options.temperature, || {
                    *random += 1;
                    0.5
                })
                .map_err(|_| "invalid masked scores")
            })
            .collect::<Result<_, _>>()?;
        let transfer = masked::transfer_positions(
            logits,
            &sampled,
            &eligible,
            scheduled,
            options.confidence_threshold,
        )
        .map_err(|_| "invalid masked transfer")?;
        let result = Self::select_tokens(&transfer, &sampled, canvas, &context())?;
        let scores = logits
            .iter()
            .enumerate()
            .map(|(p, row)| {
                if !eligible[p] {
                    previous.map_or(row, |old| &old[p]).clone()
                } else {
                    row.clone()
                }
            })
            .collect();
        let remaining = result.iter().filter(|x| **x == mask_token).count() as u32;
        Ok(Some(MaskedRefinementStep {
            canvas: result,
            scores,
            transfer,
            remaining,
        }))
    }

    fn invalidate_masked_suffix(
        canvas: &Self::Token,
        position: usize,
        token: &Self::Token,
        mask_token: u32,
        pins: Option<(&Self::Token, &Self::Mask)>,
        _: &Context,
    ) -> Result<Option<(Self::Token, u32)>, Self::Error> {
        let mut ids = canvas.clone();
        ids[position] = token[0];
        ids[position + 1..].fill(mask_token);
        if let Some((seeds, fixed)) = pins {
            for p in position + 1..ids.len() {
                if fixed[p] {
                    ids[p] = seeds[p];
                }
            }
        }
        let count = ids.iter().filter(|x| **x == mask_token).count() as u32;
        Ok(Some((ids, count)))
    }

    fn seed_canvas(
        noise: &Self::Token,
        layout: &eredu_core::refinement::RefinementCanvas,
        _: &Context,
    ) -> Result<(Self::Token, Self::Mask), Self::Error> {
        let tokens = noise
            .iter()
            .zip(&layout.seed_tokens)
            .map(|(n, s)| s.unwrap_or(*n))
            .collect();
        let pins = (0..noise.len())
            .map(|p| layout.pinned_positions.contains(&(p as u32)))
            .collect();
        Ok((tokens, pins))
    }
    fn pin_scores(
        scores: &Self::Logits,
        tokens: &Self::Token,
        pins: &Self::Mask,
        _: &Context,
    ) -> Result<Self::Logits, Self::Error> {
        Ok(scores
            .iter()
            .enumerate()
            .map(|(p, row)| {
                if pins[p] {
                    (0..row.len())
                        .map(|i| {
                            if i == tokens[p] as usize {
                                0.0
                            } else {
                                f32::NEG_INFINITY
                            }
                        })
                        .collect()
                } else {
                    row.clone()
                }
            })
            .collect())
    }

    fn validate_refinement_logits(
        logits: &Self::Logits,
        canvas: NonZeroU32,
        domain: TokenDomain,
    ) -> Result<(), Self::Error> {
        if logits.len() != canvas.get() as usize
            || logits.iter().any(|row| row.len() != domain.cardinality())
        {
            Err("wrong score geometry")
        } else {
            Ok(())
        }
    }

    fn scale_temperature(
        logits: &Self::Logits,
        temperature: f32,
        _: &Context,
    ) -> Result<Self::Logits, Self::Error> {
        if logits.iter().any(|row| {
            row.iter().any(|v| v.is_nan() || *v == f32::INFINITY)
                || !row.iter().any(|v| v.is_finite())
        }) {
            return Err("invalid scores");
        }
        Ok(logits
            .iter()
            .map(|r| r.iter().map(|x| x / temperature).collect())
            .collect())
    }

    fn uniform_tokens(
        canvas: NonZeroU32,
        domain: TokenDomain,
        random: &mut u64,
        _: &Context,
    ) -> Result<Self::Token, Self::Error> {
        // A supplied deterministic draw tape, independent of model equations.
        let result = (0..canvas.get())
            .map(|_| {
                let token = (*random % domain.cardinality() as u64) as u32;
                *random += 1;
                token
            })
            .collect();
        Ok(result)
    }

    fn categorical_tokens(
        logits: &Self::Logits,
        random: &mut u64,
        _: &Context,
    ) -> Result<Self::Token, Self::Error> {
        Ok(logits
            .iter()
            .map(|row| {
                let draw = 0.95;
                *random += 1;
                let mut cumulative = 0.0;
                probabilities(row)
                    .iter()
                    .position(|p| {
                        cumulative += p;
                        draw < cumulative
                    })
                    .unwrap() as u32
            })
            .collect())
    }

    fn argmax_tokens(logits: &Self::Logits, _: &Context) -> Result<Self::Token, Self::Error> {
        Ok(logits
            .iter()
            .map(|row| {
                let mut best = 0;
                for i in 1..row.len() {
                    if row[i] > row[best] {
                        best = i;
                    }
                }
                best as u32
            })
            .collect())
    }

    fn entropy_bound_mask(
        logits: &Self::Logits,
        bound: f32,
        _: &Context,
    ) -> Result<Self::Mask, Self::Error> {
        let values: Vec<_> = logits.iter().map(|row| entropy(row)).collect();
        let mut indices: Vec<_> = (0..values.len()).collect();
        indices.sort_by(|a, b| values[*a].total_cmp(&values[*b]).then(a.cmp(b)));
        let mut total = 0.0;
        let mut mask = vec![false; logits.len()];
        for i in indices {
            total += values[i];
            mask[i] = total - values[i] <= bound;
        }
        Ok(mask)
    }

    fn select_tokens(
        mask: &Self::Mask,
        accepted: &Self::Token,
        other: &Self::Token,
        _: &Context,
    ) -> Result<Self::Token, Self::Error> {
        Ok(mask
            .iter()
            .zip(accepted)
            .zip(other)
            .map(|((m, a), b)| if *m { *a } else { *b })
            .collect())
    }

    fn retain_self_conditioning(
        logits: &Self::Logits,
        _: &(),
        _: &Context,
    ) -> Result<Self::Logits, Self::Error> {
        Ok(logits.clone())
    }

    fn stable_and_confident(
        logits: &Self::Logits,
        argmax: &Self::Token,
        preceding: &[Self::Token],
        threshold: f32,
        pins: Option<&Self::Mask>,
        _: &Context,
    ) -> Result<bool, Self::Error> {
        let rows: Vec<_> = logits
            .iter()
            .enumerate()
            .filter(|(i, _)| pins.is_none_or(|pins| !pins[*i]))
            .map(|(_, row)| row)
            .collect();
        Ok(preceding.iter().all(|old| old == argmax)
            && rows.iter().map(|row| entropy(row)).sum::<f32>() / (rows.len().max(1) as f32)
                < threshold)
    }

    fn settle_initial_canvas(
        _: &Self::Token,
        _: &u64,
        context: &Context,
    ) -> Result<(), Self::Error> {
        if context.fail_settlement.get() {
            Err("settlement failed")
        } else {
            Ok(())
        }
    }

    fn settle_refinement(
        _: &Self::Token,
        _: &Self::Token,
        _: &Self::Logits,
        _: &Self::Logits,
        _: &Self::Mask,
        _: &u64,
        context: &Context,
    ) -> Result<(), Self::Error> {
        if context.fail_settlement.get() {
            Err("unresolved native failure")
        } else {
            Ok(())
        }
    }
}

fn context() -> Context {
    Context {
        fail_settlement: Cell::new(false),
    }
}

#[test]
fn entropy_acceptance_is_not_commitment_and_final_output_is_argmax() {
    let context = context();
    let policy = EntropyRefinementOptions {
        max_denoising_steps: 1,
        t_min: 1.0,
        t_max: 1.0,
        ..Default::default()
    }
    .try_into()
    .unwrap();
    let mut state = RefinementState::<Numeric>::new(
        policy,
        NonZeroU32::new(3).unwrap(),
        TokenDomain::new(2),
        (),
        0,
        None,
        &context,
    )
    .unwrap();
    // Entropies are approximately [0.693, 0.090, 0.365]. The first and second
    // selected entropies have bound costs 0 and 0.090; adding the third costs
    // 0.455. A plain entropy-sum cutoff would incorrectly reject position 2.
    state
        .advance(
            &vec![vec![0.0, 0.0], vec![4.0, 0.0], vec![2.0, 0.0]],
            &context,
        )
        .unwrap();
    assert_eq!(state.acceptance(), Some(&vec![false, true, true]));
    assert_eq!(state.candidates(), Some(&vec![0, 0, 0]));
    assert_eq!(state.canvas(), &vec![0, 0, 1]);
    assert_eq!(state.into_random(), 9);
}

#[test]
fn adaptive_stop_requires_confidence_and_prior_stability_and_forks_replay() {
    let context = context();
    let mut state = RefinementState::<Numeric>::new(
        Default::default(),
        NonZeroU32::new(2).unwrap(),
        TokenDomain::new(2),
        (),
        1,
        None,
        &context,
    )
    .unwrap();
    let logits = vec![vec![20.0, 0.0], vec![0.0, 20.0]];
    state.advance(&logits, &context).unwrap();
    assert!(!state.is_finished());
    let mut fork = state.clone();
    state.advance(&logits, &context).unwrap();
    fork.advance(&logits, &context).unwrap();
    assert_eq!(state.completed_steps(), 2);
    assert_eq!(state.candidates(), Some(&vec![0, 1]));
    assert_eq!(state.candidates(), fork.candidates());
    assert_eq!(state.into_random(), fork.into_random());
}

#[test]
fn failed_settlement_and_invalid_geometry_do_not_advance_state_or_randomness() {
    let context = context();
    let mut state = RefinementState::<Numeric>::new(
        Default::default(),
        NonZeroU32::new(2).unwrap(),
        TokenDomain::new(2),
        (),
        3,
        None,
        &context,
    )
    .unwrap();
    let initial = state.clone();
    assert!(state.advance(&vec![vec![0.0, 1.0]], &context).is_err());
    context.fail_settlement.set(true);
    assert!(state
        .advance(&vec![vec![0.0, 1.0], vec![2.0, 0.0]], &context)
        .is_err());
    assert_eq!(state.completed_steps(), 0);
    assert!(state.draft().is_none());
    assert_eq!(state.canvas(), initial.canvas());
    assert_eq!(state.into_random(), initial.into_random());
}

#[test]
fn schedule_and_json_admission_preserve_reference_indexing() {
    let policy = EntropyRefinementConfig::default();
    assert_eq!(policy.temperature(0).unwrap(), 0.8);
    assert!((policy.temperature(47).unwrap() - (0.4 + 0.4 / 48.0)).abs() < 1e-7);
    assert!(policy.temperature(48).is_err());
    assert!(
        serde_json::from_str::<EntropyRefinementConfig>(r#"{"max_denoising_steps":0}"#).is_err()
    );
    assert!(serde_json::from_str::<EntropyRefinementConfig>(r#"{"t_min":-1}"#).is_err());
}

#[test]
fn masked_policy_preserves_selected_tokens_scores_and_isolates_failed_steps() {
    use eredu_core::refinement::{MaskedRefinementConfig, MaskedRefinementOptions};
    let context = context();
    let config: MaskedRefinementConfig = MaskedRefinementOptions {
        max_denoising_steps: 2,
        ..Default::default()
    }
    .try_into()
    .unwrap();
    let policy = RefinementPolicy::MaskedUnmasking {
        config,
        mask_token_id: 2,
    };
    let mut state = RefinementState::<Numeric>::new(
        policy,
        NonZeroU32::new(3).unwrap(),
        TokenDomain::new(3),
        (),
        7,
        None,
        &context,
    )
    .unwrap();
    assert_eq!(state.canvas(), &[2, 2, 2]);
    let first = vec![
        vec![4.0, 0.0, -20.0],
        vec![0.0, 3.0, -20.0],
        vec![0.0, 0.1, -20.0],
    ];
    state.advance(&first, &context).unwrap();
    assert_eq!(state.canvas(), &[0, 1, 2]);
    assert!(state.self_conditioning().is_none());
    assert!(!state.is_finished());
    let mut fork = state.clone();
    let second = vec![
        vec![0.0, 9.0, -20.0],
        vec![9.0, 0.0, -20.0],
        vec![5.0, 0.0, -20.0],
    ];
    context.fail_settlement.set(true);
    assert!(matches!(
        state.advance(&second, &context),
        Err(RefinementError::Backend("unresolved native failure"))
    ));
    assert_eq!(state.canvas(), &[0, 1, 2]);
    assert_eq!(state.completed_steps(), 1);
    context.fail_settlement.set(false);
    state.advance(&second, &context).unwrap();
    fork.advance(&second, &context).unwrap();
    assert_eq!(state.candidates(), Some(&vec![0, 1, 0]));
    assert_eq!(state.candidates(), fork.candidates());
    assert_eq!(state.final_scores().unwrap()[0], first[0]);
    assert_eq!(state.final_scores().unwrap()[1], first[1]);
    assert_eq!(state.final_scores().unwrap()[2], second[2]);
    assert_eq!(state.into_random(), 7);
}

#[test]
fn masked_policy_keeps_full_canvas_and_front_loads_short_schedule() {
    use eredu_core::refinement::{
        MaskedRefinementConfig, MaskedRefinementOptions, RefinementCanvas,
    };
    let config: MaskedRefinementConfig = MaskedRefinementOptions {
        max_denoising_steps: 2,
        ..Default::default()
    }
    .try_into()
    .unwrap();
    assert_eq!(
        [
            config.transfer_tokens(3, 0).unwrap(),
            config.transfer_tokens(3, 1).unwrap()
        ],
        [2, 1]
    );
    let layout = RefinementCanvas {
        seed_tokens: vec![Some(1), None, None, None],
        pinned_positions: vec![0],
    };
    let mut state = RefinementState::<Numeric>::new(
        RefinementPolicy::MaskedUnmasking {
            config,
            mask_token_id: 2,
        },
        NonZeroU32::new(4).unwrap(),
        TokenDomain::new(3),
        (),
        0,
        Some(&layout),
        &context(),
    )
    .unwrap();
    let scores = vec![vec![2.0, 0.0, -20.0]; 4];
    state.advance(&scores, &context()).unwrap();
    assert_eq!(state.canvas(), &[1, 0, 0, 2]);
    state.advance(&scores, &context()).unwrap();
    assert_eq!(state.candidates(), Some(&vec![1, 0, 0, 0]));
}

#[test]
fn constrained_masked_replacement_refines_only_the_dependent_suffix() {
    use super::block::BlockRefinementState;
    use eredu_core::refinement::{GenerationPhase, MaskedRefinementOptions, RefinementCanvas};
    let config = MaskedRefinementOptions {
        max_denoising_steps: 1,
        ..Default::default()
    }
    .try_into()
    .unwrap();
    let layout = std::sync::Arc::new(RefinementCanvas {
        seed_tokens: vec![None, None, None, Some(1)],
        pinned_positions: vec![3],
    });
    let mut block = BlockRefinementState::<Numeric>::new(
        RefinementPolicy::MaskedUnmasking {
            config,
            mask_token_id: 2,
        },
        NonZeroU32::new(4).unwrap(),
        TokenDomain::new(3),
        (),
        9,
        Some(layout),
    );
    let context = context();
    block.context_encoded(&context, None).unwrap();
    block
        .refine(&vec![vec![4.0, 0.0, -20.0]; 4], &context)
        .unwrap();
    assert_eq!(block.phase(), GenerationPhase::Finalization);
    assert_eq!(
        block.refinement().unwrap().candidates(),
        Some(&vec![0, 0, 0, 1])
    );
    block.finalize(vec![1], true, &context).unwrap();
    assert_eq!(block.phase(), GenerationPhase::Refinement);
    assert_eq!(block.refinement().unwrap().canvas(), &[1, 2, 2, 1]);
    let mut fork = block.clone();
    let updated = vec![vec![0.0, 5.0, -20.0]; 4];
    block.refine(&updated, &context).unwrap();
    fork.refine(&updated, &context).unwrap();
    assert_eq!(
        block.refinement().unwrap().candidates(),
        Some(&vec![1, 1, 1, 1])
    );
    assert_eq!(block.refinement().unwrap().completed_steps(), 2);
    assert_eq!(block.finalization_input().unwrap().1, 1);
    assert_eq!(
        block.finalization_input().unwrap(),
        fork.finalization_input().unwrap()
    );
}

#[test]
fn block_snapshots_preserve_finalization_and_encode_only_authoritative_tokens() {
    use super::block::BlockRefinementState;
    use eredu_core::refinement::GenerationPhase;
    let context = context();
    let policy = EntropyRefinementOptions {
        max_denoising_steps: 1,
        t_min: 1.0,
        t_max: 1.0,
        ..Default::default()
    }
    .try_into()
    .unwrap();
    let mut state = BlockRefinementState::<Numeric>::new(
        policy,
        NonZeroU32::new(2).unwrap(),
        TokenDomain::new(2),
        (),
        3,
        None,
    );
    assert!(state.finalization_input().is_err());
    context.fail_settlement.set(true);
    assert!(state.context_encoded(&context, None).is_err());
    assert_eq!(state.phase(), GenerationPhase::Context);
    context.fail_settlement.set(false);
    state.context_encoded(&context, None).unwrap();
    assert_eq!(state.refinement().unwrap().canvas(), &vec![1, 0]);
    state
        .refine(&vec![vec![3.0, 0.0], vec![0.0, 4.0]], &context)
        .unwrap();
    assert_eq!(state.phase(), GenerationPhase::Finalization);
    assert_eq!(state.finalization_input().unwrap().1, 0);
    // A constrained first commitment replaces argmax 0. This token, rather than
    // the draft or noisy canvas, is retained for the next context invocation.
    state.finalize(vec![1], false, &context).unwrap();
    assert_eq!(state.continuation_context_tokens(1), Some(0));
    assert_eq!(state.continuation_context_tokens(2), Some(2));
    assert_eq!(state.continuation_context_tokens(4), Some(4));
    let mut restored = state.clone();
    let mut alternate = state.clone();
    alternate.finalize(vec![0], false, &context).unwrap();
    assert_eq!(alternate.authoritative_tokens(), &[vec![1], vec![0]]);
    assert_eq!(state.finalization_input().unwrap().1, 1);
    state.finalize(vec![1], false, &context).unwrap();
    restored.finalize(vec![1], false, &context).unwrap();
    assert_eq!(state.authoritative_tokens(), &[vec![1], vec![1]]);
    assert_eq!(state.phase(), GenerationPhase::Context);
    assert_eq!(state.block(), 1);
    assert_eq!(state.continuation_context_tokens(0), Some(0));
    assert_eq!(state.continuation_context_tokens(1), Some(2));
    assert_eq!(state.continuation_context_tokens(3), Some(4));
    assert!(state.finalize(vec![0], false, &context).is_err());
    state.context_encoded(&context, None).unwrap();
    restored.context_encoded(&context, None).unwrap();
    assert!(state.authoritative_tokens().is_empty());
    assert_eq!(state.refinement().unwrap().canvas(), &vec![1, 0]);
    assert_eq!(
        state.refinement().unwrap().canvas(),
        restored.refinement().unwrap().canvas()
    );
    state
        .refine(&vec![vec![3.0, 0.0], vec![0.0, 4.0]], &context)
        .unwrap();
    restored
        .refine(&vec![vec![3.0, 0.0], vec![0.0, 4.0]], &context)
        .unwrap();
    assert_eq!(
        state.finalization_input().unwrap(),
        restored.finalization_input().unwrap()
    );
}

#[test]
fn pinned_positions_preserve_seed_and_one_hot_feedback_across_iterations_and_forks() {
    let context = context();
    let policy = EntropyRefinementOptions {
        max_denoising_steps: 3,
        stability_threshold: 3,
        t_max: 1.0,
        t_min: 1.0,
        ..Default::default()
    }
    .try_into()
    .unwrap();
    let layout = eredu_core::refinement::RefinementCanvas {
        seed_tokens: vec![Some(1), Some(0), Some(2)],
        pinned_positions: vec![0, 2],
    };
    let mut state = RefinementState::<Numeric>::new(
        policy,
        NonZeroU32::new(3).unwrap(),
        TokenDomain::new(3),
        (),
        0,
        Some(&layout),
        &context,
    )
    .unwrap();
    assert_eq!(state.canvas(), &[1, 0, 2]);
    let logits = vec![
        vec![100.0, -100.0, 0.0],
        vec![0.0, 3.0, 1.0],
        vec![90.0, 0.0, -20.0],
    ];
    let mut invalid_logits = logits.clone();
    invalid_logits[0][0] = f32::NAN;
    assert!(state.advance(&invalid_logits, &context).is_err());
    assert_eq!(state.completed_steps(), 0);
    state.advance(&logits, &context).unwrap();
    assert_eq!(state.draft().unwrap(), &[1, 1, 2]);
    assert_eq!(
        probabilities(&state.self_conditioning().unwrap()[0]),
        &[0.0, 1.0, 0.0]
    );
    assert_eq!(
        probabilities(&state.self_conditioning().unwrap()[2]),
        &[0.0, 0.0, 1.0]
    );
    let mut fork = state.clone();
    for _ in 0..2 {
        state.advance(&logits, &context).unwrap();
        fork.advance(&logits, &context).unwrap();
    }
    assert_eq!(state.candidates().unwrap(), &[1, 1, 2]);
    assert_eq!(state.canvas(), fork.canvas());
    assert_eq!(state.into_random(), fork.into_random());
    let invalid = eredu_core::refinement::RefinementCanvas {
        seed_tokens: vec![Some(1), None, Some(2)],
        pinned_positions: vec![1],
    };
    assert!(RefinementState::<Numeric>::new(
        policy,
        NonZeroU32::new(3).unwrap(),
        TokenDomain::new(3),
        (),
        0,
        Some(&invalid),
        &context
    )
    .is_err());
}

#[test]
fn pinned_template_rows_do_not_dilute_confidence_of_unresolved_positions() {
    let context = context();
    let layout = eredu_core::refinement::RefinementCanvas {
        seed_tokens: vec![Some(0), Some(0), None],
        pinned_positions: vec![0, 1],
    };
    let policy = EntropyRefinementOptions {
        max_denoising_steps: 4,
        confidence_threshold: 0.4,
        stability_threshold: 1,
        t_max: 1.0,
        t_min: 1.0,
        ..Default::default()
    }
    .try_into()
    .unwrap();
    let mut state = RefinementState::<Numeric>::new(
        policy,
        NonZeroU32::new(3).unwrap(),
        TokenDomain::new(2),
        (),
        0,
        Some(&layout),
        &context,
    )
    .unwrap();
    let scores = vec![vec![1.0, 0.0]; 3];
    state.advance(&scores, &context).unwrap();
    state.advance(&scores, &context).unwrap();
    assert!(
        !state.is_finished(),
        "the free row entropy exceeds 0.4 even though whole-canvas mean is below it"
    );
}
