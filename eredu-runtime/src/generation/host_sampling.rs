//! Host sampling and shared text sampler commitment.

use eredu_core::{
    sampling::{
        host::{HostSampling, HostScores},
        host_random::HostRandomState,
        SamplingBackend,
    },
    BackendFailure, TokenFilter,
};

use super::TextSampler;

fn invalid(detail: &str) -> BackendFailure {
    HostSampling::error(detail.into())
}

/// Numerical outcome staged until the shared generation driver accepts the token.
pub(crate) struct HostSample {
    pub token: u32,
    probability: Option<f32>,
}

impl HostSample {
    pub(crate) fn select(
        sampler: &TextSampler,
        scores: Vec<f32>,
        filter: &TokenFilter,
        temperature: f32,
        random: Option<&mut HostRandomState>,
    ) -> Result<Self, BackendFailure> {
        let scores = HostSampling::apply_token_filter(&HostScores::from_row(scores)?, filter, &())?;
        match sampler {
            TextSampler::Standard(sampler) => Ok(Self {
                token: sampler.select::<HostSampling>(&scores, temperature, random, &())?,
                probability: None,
            }),
            TextSampler::MirostatV2(sampler) => {
                let scores = sampler.process_for::<HostSampling>(
                    &scores,
                    temperature,
                    sampler.generated_tokens(),
                    &(),
                )?;
                let token = HostSampling::sample_processed(&scores, temperature, random, &())?;
                let probability = HostSampling::token_probability(&scores, token, &())?;
                if probability <= 0.0 {
                    return Err(invalid("sampled Mirostat probability is below FP32 range"));
                }
                Ok(Self {
                    token,
                    probability: Some(probability),
                })
            }
        }
    }

    pub(crate) fn commit(self, sampler: &mut TextSampler) -> Result<(), BackendFailure> {
        match (sampler, self.probability) {
            (TextSampler::Standard(sampler), None) => {
                sampler.accept_token(self.token);
                Ok(())
            }
            (TextSampler::MirostatV2(sampler), Some(probability)) => sampler
                .accept_token(self.token, probability)
                .map_err(BackendFailure::from_error),
            _ => Err(invalid(
                "sample does not match its committed sampling strategy",
            )),
        }
    }
}
