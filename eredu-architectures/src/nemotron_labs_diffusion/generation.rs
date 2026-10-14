//! Request-local generation policies over the admitted shared-backbone geometry.

use super::FamilyConfig;
use crate::decoder::Config;
use eredu_core::{
    decoding::{DecodingModeError, SpeculationOptions},
    refinement::{MaskedRefinementOptions, MaskedSampling, RefinementPolicy},
};
use eredu_runtime::{
    generation::refinement::architecture::BlockRefinementGeometry,
    speculative::shared_backbone::{SharedBackboneDraftOptions, SharedBackboneExecutionPolicy},
};

impl FamilyConfig {
    /// Full trained canvas and exact score vocabulary for every shortened block.
    pub fn generation_geometry(&self) -> BlockRefinementGeometry {
        BlockRefinementGeometry {
            canvas_length: self.block_size,
            vocabulary_size: std::num::NonZeroU32::new(self.vocabulary_size() as u32)
                .expect("validated vocabulary"),
        }
    }

    /// Published ordinary masked unmasking with one transfer step per position.
    pub fn diffusion_policy(
        &self,
        temperature: f32,
    ) -> Result<RefinementPolicy, DecodingModeError> {
        let config = MaskedRefinementOptions {
            max_denoising_steps: self.block_size.get(),
            temperature,
            confidence_threshold: None,
            sampling: MaskedSampling::GumbelPower,
        }
        .try_into()
        .map_err(|_| DecodingModeError::Unsupported("diffusion sampling temperature"))?;
        Ok(RefinementPolicy::MaskedUnmasking {
            config,
            mask_token_id: self.mask_token_id,
        })
    }

    /// Published self-speculative draft selection. A zero threshold transfers all
    /// unresolved positions together; a positive threshold always makes progress.
    /// Target sampling and verification use their own causal policy and RNG stream.
    pub fn self_speculative_draft_options(
        &self,
        options: SpeculationOptions,
        target_temperature: f32,
    ) -> Result<SharedBackboneDraftOptions, DecodingModeError> {
        self.shared_backbone_draft_policy()?
            .prepare(options, target_temperature)
    }

    /// Source-free family defaults retained by prepared execution and cold reports.
    pub fn shared_backbone_draft_policy(
        &self,
    ) -> Result<SharedBackboneExecutionPolicy, DecodingModeError> {
        if self.block_size.get() < 2 {
            return Err(DecodingModeError::Unsupported(
                "seeded canvas has no proposal positions",
            ));
        }
        let defaults = MaskedRefinementOptions {
            max_denoising_steps: self.block_size.get() - 1,
            temperature: 0.0,
            confidence_threshold: Some(0.0),
            sampling: MaskedSampling::Categorical,
        }
        .try_into()
        .map_err(|_| DecodingModeError::Unsupported("draft numerical defaults"))?;
        let RefinementPolicy::MaskedUnmasking {
            config: diffusion, ..
        } = self.diffusion_policy(0.0)?
        else {
            unreachable!()
        };
        SharedBackboneExecutionPolicy::new(
            self.generation_geometry(),
            self.mask_token_id,
            defaults,
            diffusion,
            self.model_identity() == "nemotron_labs_diffusion",
            self.draft_adapter().map(|adapter| eredu_runtime::speculative::shared_backbone::SharedDraftAdapterGeometry {
                rank: std::num::NonZeroU32::new(adapter.rank() as u32).expect("validated positive adapter rank"),
                input: std::num::NonZeroU32::new((self.num_attention_heads()*self.head_dim()) as u32).expect("validated attention channels"),
                output: std::num::NonZeroU32::new(self.hidden_size() as u32).expect("validated residual width"),
                layers: std::num::NonZeroU32::new(self.num_hidden_layers() as u32).expect("validated layer count"),
            }),
            true,
        )
    }
}
