//! Complete logical state for alternating context encoding and block refinement.

use super::{
    finalization::{finalize_refinement_position, FinalizationError, FinalizedPosition},
    RefinementError, RefinementState,
};
use eredu_core::refinement::RefinementBackend;
use eredu_core::refinement::RefinementFinalizationBackend;
use eredu_core::sampling::TokenDomain;
use eredu_core::{
    refinement::{GenerationPhase, RefinementCanvas, RefinementPolicy},
    TextProgress, TokenFilter,
};
use std::num::NonZeroU32;

/// A transition requested outside its architecture-declared phase.
#[derive(Debug, thiserror::Error)]
pub enum BlockRefinementError<E> {
    /// Context, refinement and authoritative finalization have distinct inputs.
    #[error("block refinement transition is invalid in the current phase")]
    Phase,
    /// The next block cannot be represented.
    #[error("block refinement index overflow")]
    Overflow,
    /// Preparing or settling the numerical refinement state failed.
    #[error(transparent)]
    Refinement(#[from] RefinementError<E>),
    /// Position scores or constrained commitment could not be realized.
    #[error(transparent)]
    Finalization(#[from] FinalizationError<E>),
}

/// Snapshotable logical state for one generation's iterative blocks.
///
/// Model caches remain with the architecture session. Only finalized tokens enter
/// the next context operation; tentative canvases and feedback never do. A clone
/// isolates the explicit random stream and all mutable policy/cursor state while
/// sharing immutable numerical values. Native snapshots additionally isolate the
/// architecture's retained context and completion resources.
pub struct BlockRefinementState<B: RefinementBackend> {
    config: RefinementPolicy,
    canvas_layout: Option<std::sync::Arc<eredu_core::refinement::RefinementCanvas>>,
    canvas_length: NonZeroU32,
    domain: TokenDomain,
    conditioning_format: B::ConditioningFormat,
    random: B::RandomState,
    refinement: Option<RefinementState<B>>,
    authoritative: Vec<B::Token>,
    position: usize,
    block: u64,
    phase: GenerationPhase,
}

impl<B: RefinementBackend> Clone for BlockRefinementState<B>
where
    B::RandomState: Clone,
{
    fn clone(&self) -> Self {
        Self {
            config: self.config,
            canvas_layout: self.canvas_layout.clone(),
            canvas_length: self.canvas_length,
            domain: self.domain,
            conditioning_format: self.conditioning_format.clone(),
            random: self.random.clone(),
            refinement: self.refinement.clone(),
            authoritative: self.authoritative.clone(),
            position: self.position,
            block: self.block,
            phase: self.phase,
        }
    }
}

impl<B: RefinementBackend> BlockRefinementState<B>
where
    B::RandomState: Clone,
{
    /// Enumerates immutable refinement, finalized-token and random storage.
    pub fn visit_retained_values(&self, mut visit: impl FnMut(super::RefinementValue<'_, B>)) {
        if let Some(refinement) = &self.refinement {
            refinement.visit_retained_values(&mut visit);
        }
        for token in &self.authoritative {
            visit(super::RefinementValue::Tokens(token));
        }
        visit(super::RefinementValue::Random(&self.random));
    }

    /// Host state copied by snapshots, including the complete finalization cursor.
    pub fn snapshot_host_bytes(&self) -> Option<u64> {
        (std::mem::size_of::<Self>() as u64)
            .checked_add(self.canvas_layout.as_ref().map_or(0, |c| c.host_bytes()))?
            .checked_add(
                (self.authoritative.len() as u64)
                    .checked_mul(std::mem::size_of::<B::Token>() as u64)?,
            )?
            .checked_add(
                self.refinement
                    .as_ref()
                    .map_or(Some(0), RefinementState::snapshot_host_bytes)?,
            )
    }

    /// Installs a new random stream without discarding settled canvas or context.
    pub fn replace_random(&mut self, random: B::RandomState) {
        if let Some(refinement) = &mut self.refinement {
            refinement.replace_random(random.clone());
        }
        self.random = random;
    }

    /// Independent numerical stream at the last settled block boundary.
    pub fn random(&self) -> &B::RandomState {
        &self.random
    }
    /// Carries spent draws through a policy handoff after the block settles.
    pub fn into_random(self) -> B::RandomState {
        self.refinement
            .map_or(self.random, RefinementState::into_random)
    }

    /// Exact authoritative context growth needed to emit a finite future prefix.
    /// A partially finalized block is encoded together, including its already
    /// committed positions, only if generation proceeds beyond that block.
    pub fn continuation_context_tokens(&self, commitments: u64) -> Option<u64> {
        continuation_context_tokens(
            u64::from(self.canvas_length.get()),
            self.pending_context_output()?,
            commitments,
        )
    }

    /// Outputs remaining in the current block before another context ingestion.
    pub fn pending_context_output(&self) -> Option<u64> {
        if self.phase == GenerationPhase::Context {
            Some(0)
        } else {
            u64::from(self.canvas_length.get()).checked_sub(self.position as u64)
        }
    }

    /// Retains validated policy without submitting work or drawing initial noise.
    pub fn new(
        config: RefinementPolicy,
        canvas_length: NonZeroU32,
        domain: TokenDomain,
        conditioning_format: B::ConditioningFormat,
        random: B::RandomState,
        canvas_layout: Option<std::sync::Arc<eredu_core::refinement::RefinementCanvas>>,
    ) -> Self {
        Self {
            config,
            canvas_layout,
            canvas_length,
            domain,
            conditioning_format,
            random,
            refinement: None,
            authoritative: Vec::new(),
            position: 0,
            block: 0,
            phase: GenerationPhase::Context,
        }
    }

    /// Physical work required before another authoritative commitment.
    pub fn phase(&self) -> GenerationPhase {
        self.phase
    }
    /// Current block, independently of physical iteration and token count.
    pub fn block(&self) -> u64 {
        self.block
    }
    /// Completed refinement invocations in the active block.
    pub fn iteration(&self) -> u32 {
        self.refinement
            .as_ref()
            .map_or(0, RefinementState::completed_steps)
    }
    /// Authoritative previous block to encode, or the finalized active prefix.
    /// The first context operation consumes the separately retained prompt.
    pub fn authoritative_tokens(&self) -> &[B::Token] {
        &self.authoritative
    }
    /// Tentative decoder state; absent during context encoding.
    pub fn refinement(&self) -> Option<&RefinementState<B>> {
        self.refinement.as_ref()
    }

    /// Starts a fresh canvas after the architecture has settled context encoding.
    /// Failure preserves the random stream and the authoritative block. Encoding
    /// failure must be fenced by the session before calling this transition.
    pub fn context_encoded(
        &mut self,
        context: &B::Context,
        seeds: Option<&eredu_core::refinement::RefinementCanvas>,
    ) -> Result<(), BlockRefinementError<B::Error>> {
        self.initialize_context(self.random.clone(), seeds, context)
    }

    fn initialize_context(
        &mut self,
        random: B::RandomState,
        seeds: Option<&RefinementCanvas>,
        context: &B::Context,
    ) -> Result<(), BlockRefinementError<B::Error>> {
        if self.phase != GenerationPhase::Context {
            return Err(BlockRefinementError::Phase);
        }
        let refinement = RefinementState::new(
            self.config,
            self.canvas_length,
            self.domain,
            self.conditioning_format.clone(),
            random.clone(),
            seeds.or(self.canvas_layout.as_deref()),
            context,
        )?;
        self.random = random;
        self.refinement = Some(refinement);
        self.authoritative.clear();
        self.position = 0;
        self.phase = GenerationPhase::Refinement;
        Ok(())
    }

    /// Settles a decoder iteration without advancing any commitment controller.
    pub fn refine(
        &mut self,
        scores: &B::Logits,
        context: &B::Context,
    ) -> Result<(), BlockRefinementError<B::Error>> {
        if self.phase != GenerationPhase::Refinement {
            return Err(BlockRefinementError::Phase);
        }
        let refinement = self
            .refinement
            .as_mut()
            .ok_or(BlockRefinementError::Phase)?;
        refinement.advance(scores, context)?;
        if refinement.is_finished() {
            self.phase = GenerationPhase::Finalization;
        }
        Ok(())
    }

    /// Whether the selected architecture seeds its first position from context scores.
    pub fn needs_context_seed(&self, seed_first: bool) -> bool {
        seed_first && self.canvas_layout.is_none()
    }

    /// Applies one context encoding, optionally seeding position zero from causal scores.
    /// The masked seed uses the exact same numerical policy and random stream as the block.
    pub fn accept_context(
        &mut self,
        seed_scores: Option<&B::Logits>,
        context: &B::Context,
    ) -> Result<TextProgress, BlockRefinementError<B::Error>> {
        let mut random = self.random.clone();
        let seeds = if let Some(scores) = seed_scores {
            let RefinementPolicy::MaskedUnmasking {
                config,
                mask_token_id,
            } = self.config
            else {
                return Err(BlockRefinementError::Phase);
            };
            let canvas = B::materialize_tokens(&[mask_token_id], context)
                .map_err(RefinementError::Backend)?;
            let sampled = super::masked::step::<B>(
                &canvas,
                None,
                scores,
                mask_token_id,
                config,
                1,
                &mut random,
                context,
            )?;
            let tokens =
                B::read_tokens(&sampled.canvas, context).map_err(RefinementError::Backend)?;
            let [token] = tokens.as_slice() else {
                return Err(RefinementError::Geometry.into());
            };
            let mut seed_tokens = vec![None; self.canvas_length.get() as usize];
            seed_tokens[0] = Some(*token);
            Some(RefinementCanvas {
                seed_tokens,
                pinned_positions: if *token == mask_token_id {
                    vec![]
                } else {
                    vec![0]
                },
            })
        } else {
            None
        };
        self.initialize_context(random, seeds.as_ref(), context)?;
        Ok(TextProgress {
            phase: GenerationPhase::Context,
            block: self.block,
            iteration: 0,
            tentative_token_ids: None,
        })
    }

    /// Applies one tentative decoder result and describes the new progress.
    pub fn accept_refinement(
        &mut self,
        scores: &B::Logits,
        context: &B::Context,
    ) -> Result<TextProgress, BlockRefinementError<B::Error>> {
        self.refine(scores, context)?;
        Ok(TextProgress {
            phase: GenerationPhase::Refinement,
            block: self.block,
            iteration: self.iteration(),
            tentative_token_ids: None,
        })
    }

    /// Selects and installs the next authoritative candidate using the current
    /// filter. The owning generation driver still publishes the commitment
    /// after proving native completion.
    pub fn commit_candidate(
        &mut self,
        filter: &TokenFilter,
        context: &<B as RefinementBackend>::Context,
    ) -> Result<
        FinalizedPosition<<B as RefinementBackend>::Token, B::PositionScores>,
        BlockRefinementError<<B as RefinementBackend>::Error>,
    >
    where
        B: RefinementFinalizationBackend<
            CanvasTokens = B::Token,
            CanvasScores = B::Logits,
            FinalizationContext = B::Context,
            FinalizationError = B::Error,
        >,
    {
        let (scores, position) = self.finalization_input()?;
        let finalized = finalize_refinement_position::<B>(
            scores,
            position,
            self.refinement()
                .and_then(RefinementState::selected_candidates),
            self.domain,
            filter,
            context,
        )?;
        self.finalize(
            finalized.token.clone(),
            finalized.candidate_replaced,
            context,
        )?;
        Ok(finalized)
    }

    /// Full final scores and the next position for an evolving constraint decision.
    /// Score selection uses `finalize_refinement_position`; no draft is committed.
    pub fn finalization_input(
        &self,
    ) -> Result<(&B::Logits, usize), BlockRefinementError<B::Error>> {
        if self.phase != GenerationPhase::Finalization {
            return Err(BlockRefinementError::Phase);
        }
        let scores = self
            .refinement
            .as_ref()
            .and_then(RefinementState::final_scores)
            .ok_or(BlockRefinementError::Phase)?;
        Ok((scores, self.position))
    }

    /// Retains one settled, constraint-selected token for subsequent context.
    /// Called by the submission adapter once selection succeeds. The shared
    /// generation machine owns controller commitment and terminal failure fencing.
    /// A completed block transitions to context encoding with the exact RNG that
    /// follows the last refinement, including draws from the stopping iteration.
    pub fn finalize(
        &mut self,
        token: B::Token,
        candidate_replaced: bool,
        context: &B::Context,
    ) -> Result<(), BlockRefinementError<B::Error>> {
        if self.phase != GenerationPhase::Finalization {
            return Err(BlockRefinementError::Phase);
        }
        let completed = self.position + 1 == self.canvas_length.get() as usize;
        let next_block = if completed {
            Some(
                self.block
                    .checked_add(1)
                    .ok_or(BlockRefinementError::Overflow)?,
            )
        } else {
            None
        };
        if candidate_replaced && !completed {
            self.refinement
                .as_mut()
                .ok_or(BlockRefinementError::Phase)?
                .invalidate_suffix(self.position, &token, context)?;
        }
        self.authoritative.push(token);
        self.position += 1;
        if let Some(block) = next_block {
            self.random = self
                .refinement
                .take()
                .ok_or(BlockRefinementError::Phase)?
                .into_random();
            self.block = block;
            self.phase = GenerationPhase::Context;
        } else if self
            .refinement
            .as_ref()
            .is_some_and(|state| !state.is_finished())
        {
            self.phase = GenerationPhase::Refinement;
        }
        Ok(())
    }
}

pub(crate) fn continuation_context_tokens(
    canvas: u64,
    pending_positions: u64,
    commitments: u64,
) -> Option<u64> {
    if commitments <= pending_positions {
        return Some(0);
    }
    let further = commitments.checked_sub(pending_positions)?;
    further
        .checked_sub(1)?
        .checked_div(canvas)?
        .checked_add(1)?
        .checked_mul(canvas)
}
