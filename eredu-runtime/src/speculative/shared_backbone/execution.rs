//! Portable invocation phases, draft scheduling, and refinement continuation.

use super::{SharedBackboneDraftOptions, SharedBackboneMechanisms};
use crate::{
    executable_binding::retained_session::capture_media_shape,
    generation::refinement::RefinementState,
    inspection::{
        with_speculative_activation, SpeculativeActivationGeometry, SpeculativeActivationObserver,
    },
    speculative::autoregressive::{AutoregressiveMechanisms, AutoregressivePass},
};
use eredu_core::refinement::RefinementBackend;
use eredu_core::sampling::TokenDomain;
use eredu_core::{
    decoding::{AttentionInvocation, DecodingCapabilities, DecodingModeError, SpeculationOptions},
    refinement::{RefinementCanvas, RefinementPolicy},
    speculative::{RefinementBlock, RefinementProgress, SpeculativeActivationPhase as Phase},
};

/// Numerical state over the backend's native values, scheduled by the portable driver.
pub type SharedRefinementState<M> = RefinementState<<M as SharedBackboneMechanisms>::Refinement>;
type Observer<'a, M> = dyn SpeculativeActivationObserver<
        <M as SharedBackboneMechanisms>::ActivationTensor,
        <M as AutoregressiveMechanisms>::Error,
    > + 'a;

pub(in crate::speculative) fn prepare_draft_options<M: SharedBackboneMechanisms>(
    model: &M::Model,
    options: SpeculationOptions,
    temperature: f32,
) -> Result<SharedBackboneDraftOptions, DecodingModeError> {
    M::shared_backbone_policy(model)
        .ok_or(DecodingModeError::Unsupported(
            "no retained shared-backbone declaration",
        ))?
        .prepare(options, temperature)
}

pub(in crate::speculative) fn draft_capabilities<M: SharedBackboneMechanisms>(
    model: &M::Model,
) -> DecodingCapabilities {
    let ordinary = M::ordinary_refinement_policy(model);
    let draft = M::shared_backbone_policy(model);
    DecodingCapabilities {
        autoregressive: ordinary.is_none_or(|policy| policy.causal_sampling),
        diffusion: ordinary.is_some(),
        speculation: draft.is_some(),
        mode_changes: true,
        entropy_feedback: ordinary
            .is_some_and(|x| matches!(x.policy, RefinementPolicy::EntropyFeedback(_))),
        masked_unmasking: ordinary
            .is_some_and(|x| matches!(x.policy, RefinementPolicy::MaskedUnmasking { .. })),
        mask_token_id: ordinary.and_then(|x| match x.policy {
            RefinementPolicy::MaskedUnmasking { mask_token_id, .. } => Some(mask_token_id),
            _ => None,
        }),
        draft_adapter: draft.is_some_and(|x| x.adapter_available()),
        ..Default::default()
    }
}

pub(super) fn prefill_observed<'observer, M: SharedBackboneMechanisms>(
    model: &mut M::Model,
    input: &M::Input,
    state: &mut M::State,
    context: M::Context<'_>,
    observer: Option<&'observer mut Observer<'observer, M>>,
) -> Result<(M::Output, usize), M::Error> {
    if observer.is_none() {
        return M::prefill(
            model,
            input,
            state,
            AutoregressivePass::TargetPrefill,
            context,
        );
    }
    let plans = M::activation_prefill_plans(model, input)?;
    let media = capture_media_shape(&plans).map_err(M::execution_error)?;
    let sequence = plans
        .iter()
        .try_fold(0usize, |sum, plan| {
            sum.checked_add(usize::try_from(plan.decoder_positions()).ok()?)
        })
        .ok_or_else(|| M::invalid("prepared observation geometry overflows"))?;
    with_speculative_activation(
        observer,
        Phase::TargetPrefill,
        SpeculativeActivationGeometry { sequence, media },
        |observer| {
            M::prefill_with_activation(
                model,
                input,
                state,
                observer.expect("admitted prompt observer"),
            )
        },
    )
}

pub(super) fn decode_observed<'observer, M: SharedBackboneMechanisms>(
    model: &mut M::Model,
    tokens: &[u32],
    state: &mut M::State,
    pass: AutoregressivePass,
    context: M::Context<'_>,
    observer: Option<&'observer mut Observer<'observer, M>>,
) -> Result<M::Output, M::Error> {
    if observer.is_none() {
        return M::decode(model, tokens, state, pass, context);
    }
    if tokens.is_empty() {
        return Err(M::invalid("causal observation requires nonempty input"));
    }
    let phase = match pass {
        AutoregressivePass::Verification => Phase::Verification,
        _ => Phase::TargetReplay,
    };
    with_speculative_activation(observer, phase, tokens.len(), |observer| {
        M::decode_with_activation(
            model,
            tokens,
            state,
            observer.expect("admitted causal observer"),
        )
    })
}

#[allow(clippy::too_many_arguments)]
pub(in crate::speculative) fn refine<'observer, M: SharedBackboneMechanisms>(
    model: &mut M::Model,
    prefix: &M::Checkpoint,
    next_logits: &M::Logits,
    committed: &[u32],
    requested: Option<RefinementPolicy>,
    temperature: f32,
    randomness: &mut M::ProposalRandomness,
    paused: &mut Option<SharedRefinementState<M>>,
    context: M::Context<'_>,
    observer: Option<&'observer mut Observer<'observer, M>>,
) -> Result<Option<RefinementProgress<M::Logits>>, M::Error> {
    let declaration = M::ordinary_refinement_policy(model)
        .ok_or_else(|| M::invalid("no retained refinement declaration"))?;
    let policy = declaration
        .prepare(requested, temperature)
        .map_err(M::execution_error)?;
    let geometry = declaration.geometry;
    let width = geometry.canvas_length.get() as usize;
    let vocabulary = geometry.vocabulary_size.get() as usize;
    if committed.len() >= width {
        return Err(M::invalid("refinement prefix exceeds admitted geometry"));
    }
    let queue = M::target_refinement_context(context);
    if paused.is_none() {
        let layout = if let RefinementPolicy::MaskedUnmasking {
            config,
            mask_token_id,
        } = policy
        {
            let mut layout = RefinementCanvas {
                seed_tokens: vec![None; width],
                pinned_positions: (0..committed.len())
                    .map(|position| position as u32)
                    .collect(),
            };
            for (seed, token) in layout.seed_tokens.iter_mut().zip(committed) {
                *seed = Some(*token);
            }
            if committed.is_empty() && declaration.seed_first {
                let canvas = M::Refinement::materialize_tokens(&[mask_token_id], queue)?;
                let sampled = crate::generation::refinement::masked::step::<M::Refinement>(
                    &canvas,
                    None,
                    &M::refinement_scores(next_logits),
                    mask_token_id,
                    config,
                    1,
                    randomness,
                    queue,
                )
                .map_err(M::execution_error)?;
                let seed = M::read_refinement_tokens(&sampled.canvas, 0..1, queue)?
                    .first()
                    .copied()
                    .ok_or_else(|| M::invalid("masked seed returned no token"))?;
                layout.seed_tokens[0] = Some(seed);
                if seed != mask_token_id {
                    layout.pinned_positions.push(0);
                }
            }
            layout
                .validate(geometry.canvas_length.get(), vocabulary)
                .map_err(M::execution_error)?;
            Some(layout)
        } else {
            if !committed.is_empty() {
                return Err(M::invalid(
                    "entropy refinement has no pinned selected prefix",
                ));
            }
            None
        };
        *paused = Some(
            RefinementState::new(
                policy,
                geometry.canvas_length,
                TokenDomain::new(vocabulary),
                M::score_format(next_logits),
                randomness.clone(),
                layout.as_ref(),
                queue,
            )
            .map_err(M::execution_error)?,
        );
    }
    let mut transient_prefix = M::restore(prefix, context)?;
    let refinement = paused.as_mut().expect("initialized numerical continuation");
    if !refinement.is_finished() {
        advance::<M>(
            model,
            &mut transient_prefix,
            refinement,
            geometry,
            AttentionInvocation::DIFFUSION,
            Phase::Refinement {
                iteration: refinement.completed_steps(),
            },
            queue,
            observer,
        )?;
    }
    M::settle_refinement(refinement)?;
    *randomness = refinement.clone().into_random();
    if !refinement.is_finished() {
        return Ok(Some(RefinementProgress::Pending {
            width,
            evaluated_positions: width,
            iterations: 1,
        }));
    }
    let tokens = refinement
        .candidates()
        .ok_or_else(|| M::invalid("refinement has no selected canvas"))?;
    let scores = refinement
        .final_scores()
        .ok_or_else(|| M::invalid("refinement has no position scores"))?;
    let positions = M::read_refinement_tokens(tokens, 0..width, queue)?
        .into_iter()
        .enumerate()
        .map(|(position, token)| Ok((token, M::refinement_score_row(scores, position, queue)?)))
        .collect::<Result<Vec<_>, M::Error>>()?;
    *randomness = paused
        .take()
        .expect("completed numerical continuation")
        .into_random();
    Ok(Some(RefinementProgress::Complete(RefinementBlock {
        positions,
        iterations: 1,
        evaluated_positions: width,
    })))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn draft<'observer, M: SharedBackboneMechanisms>(
    model: &mut M::Model,
    prefix: &M::Checkpoint,
    last_token: u32,
    capacity: usize,
    options: &SharedBackboneDraftOptions,
    randomness: &mut M::ProposalRandomness,
    context: M::Context<'_>,
    mut observer: Option<&'observer mut Observer<'observer, M>>,
) -> Result<Vec<u32>, M::Error> {
    let geometry = options.geometry();
    let width = geometry.canvas_length.get() as usize;
    let vocabulary = geometry.vocabulary_size.get() as usize;
    let incompatible =
        || M::invalid("shared-backbone draft does not match retained canvas or device geometry");
    if M::block_geometry(model) != Some(geometry) || capacity == 0 || capacity >= width {
        return Err(incompatible());
    }
    let queue = M::draft_refinement_context(context).ok_or_else(incompatible)?;
    let mut layout = RefinementCanvas {
        seed_tokens: vec![None; width],
        pinned_positions: vec![0],
    };
    layout.seed_tokens[0] = Some(last_token);
    layout
        .validate(geometry.canvas_length.get(), vocabulary)
        .map_err(M::execution_error)?;
    let mut transient_prefix = M::restore(prefix, context)?;
    let mut refinement = RefinementState::new(
        options.policy(),
        geometry.canvas_length,
        TokenDomain::new(vocabulary),
        M::draft_score_format(),
        randomness.clone(),
        Some(&layout),
        queue,
    )
    .map_err(M::execution_error)?;
    while !refinement.is_finished() {
        let iteration = refinement.completed_steps();
        advance::<M>(
            model,
            &mut transient_prefix,
            &mut refinement,
            geometry,
            options.invocation(),
            Phase::MaskedDraft { iteration },
            queue,
            observer
                .as_mut()
                .map(|observer| &mut **observer as &mut Observer<'_, M>),
        )?;
    }
    let candidates = refinement
        .candidates()
        .ok_or_else(|| M::invalid("masked draft has no settled selections"))?;
    let tokens = M::read_refinement_tokens(candidates, 1..capacity + 1, queue)?;
    *randomness = refinement.into_random();
    Ok(tokens)
}

#[allow(clippy::too_many_arguments)]
fn advance<'observer, M: SharedBackboneMechanisms>(
    model: &mut M::Model,
    prefix: &mut M::State,
    refinement: &mut SharedRefinementState<M>,
    geometry: crate::generation::refinement::architecture::BlockRefinementGeometry,
    invocation: AttentionInvocation,
    phase: Phase,
    queue: &<M::Refinement as RefinementBackend>::Context,
    observer: Option<&'observer mut Observer<'observer, M>>,
) -> Result<(), M::Error> {
    let feedback = if matches!(phase, Phase::MaskedDraft { .. }) {
        None
    } else {
        refinement.self_conditioning()
    };
    let scores = with_speculative_activation(
        observer,
        phase,
        geometry.canvas_length.get() as usize,
        |observer| {
            M::evaluate_refinement(
                model,
                prefix,
                refinement.canvas(),
                feedback,
                geometry,
                invocation,
                queue,
                observer,
            )
        },
    )?;
    refinement
        .advance(&scores, queue)
        .map_err(M::execution_error)?;
    Ok(())
}
