//! Host numerical sampling with native tensor inputs and outputs.

use std::marker::PhantomData;

use eredu_nn::{Tensor, TensorElementType};

use crate::{BackendFailure, TokenFilter};

use super::{
    host::{HostSampling, HostScores},
    host_random::HostRandomState,
    validate_dense_draw_shape, validate_gumbel_epsilon, DenseSamplingBackend, GumbelTruncation,
    PenaltyConfig, SamplingBackend, TokenDomain,
};

/// Runs the shared host sampler on rows read from a tensor, then materializes
/// the selected values on the same context.
pub struct HostTensorSampling<T>(PhantomData<fn() -> T>);

impl<T: Tensor> HostTensorSampling<T> {
    fn rows(logits: &T, context: &T::Context) -> Result<(Vec<f32>, usize), BackendFailure> {
        let count =
            validate_dense_draw_shape(logits.shape()).map_err(BackendFailure::from_error)?;
        let width = *logits
            .shape()
            .last()
            .ok_or_else(|| Self::error("missing vocabulary axis".into()))?
            as usize;
        let scores = logits
            .to_f32_vec(context)
            .map_err(BackendFailure::from_error)?;
        if scores.len() != count {
            return Err(Self::error(
                "score readback disagrees with tensor shape".into(),
            ));
        }
        Ok((scores, width))
    }

    fn transform(
        logits: &T,
        context: &T::Context,
        mut operation: impl FnMut(HostScores) -> Result<HostScores, BackendFailure>,
    ) -> Result<T, BackendFailure> {
        let (scores, width) = Self::rows(logits, context)?;
        let mut output = Vec::with_capacity(scores.len());
        for row in scores.chunks_exact(width) {
            output.extend_from_slice(operation(HostScores::from_row(row.to_vec())?)?.values());
        }
        T::from_f32_slice(&output, logits.shape(), context).map_err(BackendFailure::from_error)
    }

    fn tokens(token: &T, context: &T::Context) -> Result<Vec<i32>, BackendFailure> {
        match token.element_type() {
            Some(TensorElementType::I32) => token
                .to_i32_vec(context)
                .map_err(BackendFailure::from_error),
            _ => Err(Self::error(
                "sampling tokens require exact 32-bit integer storage".into(),
            )),
        }
    }

    fn select(
        logits: &T,
        random: Option<&mut HostRandomState>,
        context: &T::Context,
        mut operation: impl FnMut(
            &HostScores,
            Option<&mut HostRandomState>,
        ) -> Result<u32, BackendFailure>,
    ) -> Result<T, BackendFailure> {
        let (scores, width) = Self::rows(logits, context)?;
        let mut candidate = random.as_deref().cloned();
        let tokens = scores
            .chunks_exact(width)
            .map(|row| {
                let token = operation(&HostScores::from_row(row.to_vec())?, candidate.as_mut())?;
                i32::try_from(token).map_err(BackendFailure::from_error)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let output = T::from_i32_slice(
            &tokens,
            &logits.shape()[..logits.shape().len() - 1],
            context,
        )
        .map_err(BackendFailure::from_error)?;
        if let Some((random, candidate)) = random.zip(candidate) {
            *random = candidate;
        }
        Ok(output)
    }

    fn draw(
        shape: &[i32],
        random: &mut HostRandomState,
        context: &T::Context,
        mut draw: impl FnMut(&mut HostRandomState) -> f32,
    ) -> Result<T, BackendFailure> {
        let count = validate_dense_draw_shape(shape).map_err(BackendFailure::from_error)?;
        let mut candidate = random.clone();
        let values: Vec<_> = (0..count).map(|_| draw(&mut candidate)).collect();
        let value =
            T::from_f32_slice(&values, shape, context).map_err(BackendFailure::from_error)?;
        *random = candidate;
        Ok(value)
    }
}

impl<T: Tensor> SamplingBackend for HostTensorSampling<T> {
    type Logits = T;
    type Token = T;
    type RandomState = HostRandomState;
    type Context = T::Context;
    type Error = BackendFailure;

    fn error(message: String) -> BackendFailure {
        HostSampling::error(message)
    }

    fn validate_token(
        token: &T,
        domain: TokenDomain,
        ctx: &T::Context,
    ) -> Result<T, BackendFailure> {
        for id in Self::tokens(token, ctx)? {
            HostSampling::validate_token(
                &u32::try_from(id).map_err(BackendFailure::from_error)?,
                domain,
                &(),
            )?;
        }
        Ok(token.clone())
    }
    fn scale_temperature(
        logits: &T,
        temperature: f32,
        ctx: &T::Context,
    ) -> Result<T, BackendFailure> {
        Self::transform(logits, ctx, |row| {
            HostSampling::scale_temperature(&row, temperature, &())
        })
    }
    fn apply_penalties(
        logits: &T,
        history: &[u32],
        penalties: PenaltyConfig,
        ctx: &T::Context,
    ) -> Result<T, BackendFailure> {
        Self::transform(logits, ctx, |row| {
            HostSampling::apply_penalties(&row, history, penalties, &())
        })
    }
    fn apply_top_k(logits: T, k: i32, ctx: &T::Context) -> Result<T, BackendFailure> {
        Self::transform(&logits, ctx, |row| HostSampling::apply_top_k(row, k, &()))
    }
    fn apply_top_p(logits: T, p: f32, ctx: &T::Context) -> Result<T, BackendFailure> {
        Self::transform(&logits, ctx, |row| HostSampling::apply_top_p(row, p, &()))
    }
    fn apply_min_p(logits: T, p: f32, ctx: &T::Context) -> Result<T, BackendFailure> {
        Self::transform(&logits, ctx, |row| HostSampling::apply_min_p(row, p, &()))
    }
    fn apply_token_filter(
        logits: &T,
        filter: &TokenFilter,
        ctx: &T::Context,
    ) -> Result<T, BackendFailure> {
        Self::transform(logits, ctx, |row| {
            HostSampling::apply_token_filter(&row, filter, &())
        })
    }
    fn apply_mirostat(
        logits: &T,
        history: &[u32],
        penalties: PenaltyConfig,
        temperature: f32,
        mu: f32,
        ctx: &T::Context,
    ) -> Result<T, BackendFailure> {
        Self::transform(logits, ctx, |row| {
            HostSampling::apply_mirostat(&row, history, penalties, temperature, mu, &())
        })
    }
    fn sample_raw(
        logits: &T,
        temperature: f32,
        random: Option<&mut HostRandomState>,
        ctx: &T::Context,
    ) -> Result<T, BackendFailure> {
        Self::select(logits, random, ctx, |row, random| {
            HostSampling::sample_raw(row, temperature, random, &())
        })
    }
    fn sample_processed(
        logits: &T,
        temperature: f32,
        random: Option<&mut HostRandomState>,
        ctx: &T::Context,
    ) -> Result<T, BackendFailure> {
        Self::select(logits, random, ctx, |row, random| {
            HostSampling::sample_processed(row, temperature, random, &())
        })
    }
    fn token_id(token: &T, ctx: &T::Context) -> Result<u32, BackendFailure> {
        match Self::tokens(token, ctx)?.as_slice() {
            [id] => u32::try_from(*id).map_err(BackendFailure::from_error),
            _ => Err(Self::error("sampling requires one selected token".into())),
        }
    }
    fn token_probability(logits: &T, token: u32, ctx: &T::Context) -> Result<f32, BackendFailure> {
        let (scores, width) = Self::rows(logits, ctx)?;
        if scores.len() != width {
            return Err(Self::error(
                "token probability requires one vocabulary row".into(),
            ));
        }
        HostSampling::token_probability(&HostScores::from_row(scores)?, token, &())
    }
}

impl<T: Tensor> DenseSamplingBackend for HostTensorSampling<T> {
    fn uniform_f32(
        shape: &[i32],
        random: &mut HostRandomState,
        ctx: &T::Context,
    ) -> Result<T, BackendFailure> {
        Self::draw(shape, random, ctx, HostRandomState::unit_interval_f32)
    }
    fn standard_normal_f32(
        shape: &[i32],
        random: &mut HostRandomState,
        ctx: &T::Context,
    ) -> Result<T, BackendFailure> {
        Self::draw(shape, random, ctx, |random| {
            let radius = (-2.0 * random.open_unit_interval().ln()).sqrt();
            (radius * (std::f64::consts::TAU * random.unit_interval()).cos()) as f32
        })
    }
    fn dense_shape(value: &T) -> &[i32] {
        value.shape()
    }
    fn gumbel_max(
        logits: &T,
        uniform: &T,
        truncation: GumbelTruncation,
        epsilon: f32,
        ctx: &T::Context,
    ) -> Result<T, BackendFailure> {
        truncation.validate().map_err(BackendFailure::from_error)?;
        validate_gumbel_epsilon(epsilon).map_err(BackendFailure::from_error)?;
        if logits.shape() != uniform.shape()
            || logits.element_type() != Some(TensorElementType::F32)
            || uniform.element_type() != Some(TensorElementType::F32)
        {
            return Err(Self::error(
                "Gumbel selection requires matching F32 tensors".into(),
            ));
        }
        let (scores, width) = Self::rows(logits, ctx)?;
        let (uniform, _) = Self::rows(uniform, ctx)?;
        if uniform.iter().any(|u| !(0.0..1.0).contains(u)) {
            return Err(Self::error("Gumbel uniforms must lie in [0, 1)".into()));
        }
        let ids = scores
            .chunks_exact(width)
            .zip(uniform.chunks_exact(width))
            .map(|(row, uniform)| gumbel_row(row, uniform, truncation, epsilon))
            .collect::<Result<Vec<_>, _>>()?;
        T::from_i32_slice(&ids, &logits.shape()[..logits.shape().len() - 1], ctx)
            .map_err(BackendFailure::from_error)
    }
}

fn gumbel_row(
    row: &[f32],
    uniforms: &[f32],
    truncation: GumbelTruncation,
    epsilon: f32,
) -> Result<i32, BackendFailure> {
    let mut scores = HostScores::from_row(row.to_vec())?;
    let (max, _) = scores.normalization()?;
    if let GumbelTruncation::TopK(k) = truncation {
        let k = i32::try_from(k).unwrap_or(i32::MAX);
        scores = HostSampling::apply_top_k(scores, k, &())?;
    }
    let mut filtered = scores.values().to_vec();
    if let GumbelTruncation::TopP(p) = truncation {
        let mut indices = (0..row.len()).collect::<Vec<_>>();
        indices.sort_unstable_by(|&a, &b| row[a].total_cmp(&row[b]).then(a.cmp(&b)));
        let mass: f32 = indices.iter().map(|&i| (row[i] - max).exp()).sum();
        let mut cumulative = 0.0;
        for &i in &indices[..indices.len() - 1] {
            cumulative += (row[i] - max).exp() / mass;
            if cumulative <= (1.0 - p) as f32 {
                filtered[i] = f32::NEG_INFINITY;
            }
        }
    }
    let (max, mass) = HostScores::from_row(filtered.clone())?.normalization()?;
    let normalizer = max + (mass as f32).ln();
    for (score, &uniform) in filtered.iter_mut().zip(uniforms) {
        *score = (*score - normalizer) - (-(uniform + epsilon).ln() + epsilon).ln();
    }
    let id = HostSampling::sample_processed(&HostScores::from_row(filtered)?, 0.0, None, &())?;
    i32::try_from(id).map_err(BackendFailure::from_error)
}
