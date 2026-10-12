use super::*;
use eredu_core::refinement::RefinementOptions;
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
        _: &Context,
    ) -> Result<bool, Self::Error> {
        Ok(preceding.iter().all(|old| old == argmax)
            && logits.iter().map(|row| entropy(row)).sum::<f32>() / (logits.len() as f32)
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
    let policy = RefinementOptions {
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
    let policy = RefinementConfig::default();
    assert_eq!(policy.temperature(0).unwrap(), 0.8);
    assert!((policy.temperature(47).unwrap() - (0.4 + 0.4 / 48.0)).abs() < 1e-7);
    assert!(policy.temperature(48).is_err());
    assert!(serde_json::from_str::<RefinementConfig>(r#"{"max_denoising_steps":0}"#).is_err());
    assert!(serde_json::from_str::<RefinementConfig>(r#"{"t_min":-1}"#).is_err());
}

#[test]
fn block_snapshots_preserve_finalization_and_encode_only_authoritative_tokens() {
    use super::block::BlockRefinementState;
    use eredu_core::refinement::GenerationPhase;
    let context = context();
    let policy = RefinementOptions {
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
    );
    assert!(state.finalization_input().is_err());
    context.fail_settlement.set(true);
    assert!(state.context_encoded(&context).is_err());
    assert_eq!(state.phase(), GenerationPhase::Context);
    context.fail_settlement.set(false);
    state.context_encoded(&context).unwrap();
    assert_eq!(state.refinement().unwrap().canvas(), &vec![1, 0]);
    state
        .refine(&vec![vec![3.0, 0.0], vec![0.0, 4.0]], &context)
        .unwrap();
    assert_eq!(state.phase(), GenerationPhase::Finalization);
    assert_eq!(state.finalization_input().unwrap().1, 0);
    // A constrained first commitment replaces argmax 0. This token, rather than
    // the draft or noisy canvas, is retained for the next context invocation.
    state.finalize(vec![1]).unwrap();
    assert_eq!(state.continuation_context_tokens(1), Some(0));
    assert_eq!(state.continuation_context_tokens(2), Some(2));
    assert_eq!(state.continuation_context_tokens(4), Some(4));
    let mut restored = state.clone();
    let mut alternate = state.clone();
    alternate.finalize(vec![0]).unwrap();
    assert_eq!(alternate.authoritative_tokens(), &[vec![1], vec![0]]);
    assert_eq!(state.finalization_input().unwrap().1, 1);
    state.finalize(vec![1]).unwrap();
    restored.finalize(vec![1]).unwrap();
    assert_eq!(state.authoritative_tokens(), &[vec![1], vec![1]]);
    assert_eq!(state.phase(), GenerationPhase::Context);
    assert_eq!(state.block(), 1);
    assert_eq!(state.continuation_context_tokens(0), Some(0));
    assert_eq!(state.continuation_context_tokens(1), Some(2));
    assert_eq!(state.continuation_context_tokens(3), Some(4));
    assert!(state.finalize(vec![0]).is_err());
    state.context_encoded(&context).unwrap();
    restored.context_encoded(&context).unwrap();
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
