//! Native score readback and completion handoff for host sampling.

use std::convert::Infallible;

use eredu_core::{
    backend::TextSamplingNative, eager_completion::EagerCompletion, BackendFailure, Submission,
    SubmissionLease,
};
use eredu_nn::{Index, Tensor};

use crate::{CandleContext, CandleTensor, CandleToken};

impl TextSamplingNative<CandleContext> for CandleContext {
    type SamplingCompletion = EagerCompletion<BackendFailure, SubmissionLease>;
    type ScoreRow = Vec<f32>;
    type Sample = u32;
    type SamplingScope = ();
    type Token = CandleToken;

    fn completion_cancellation_mode() -> Option<eredu_core::CompletionCancellationMode> {
        None
    }
    fn begin_sampling(_: &Self::SamplingCompletion) -> Result<(), BackendFailure> {
        Ok(())
    }
    fn last_score_row(
        output: &Option<CandleTensor>,
        context: &Self,
    ) -> Result<Option<Vec<f32>>, BackendFailure> {
        output
            .as_ref()
            .map(|scores| score_row(scores, -1, context))
            .transpose()
    }
    fn sampling_group_coordinates(group: &Infallible) -> (usize, usize) {
        match *group {}
    }
    fn exchange_sampled_token(
        _: Option<u32>,
        _: usize,
        group: &Infallible,
        _: &eredu_core::communication::CommunicationAuthority,
        _: &Self,
    ) -> Result<u32, BackendFailure> {
        match *group {}
    }
    fn finish_sampling(
        _: (),
        completion: Self::SamplingCompletion,
        token: Result<u32, BackendFailure>,
        _: &Self,
    ) -> Result<Submission<CandleToken, Self::SamplingCompletion>, BackendFailure> {
        Ok(Submission {
            output: CandleToken(token?),
            completion,
        })
    }
    fn complete_host_progress(
        lease: SubmissionLease,
    ) -> Result<Self::SamplingCompletion, BackendFailure> {
        Ok(EagerCompletion::completed_with_lease(lease))
    }

    fn complete_host_token(
        token: u32,
        lease: SubmissionLease,
    ) -> Result<Submission<CandleToken, Self::SamplingCompletion>, BackendFailure> {
        Ok(Submission {
            output: CandleToken(token),
            completion: Self::complete_host_progress(lease)?,
        })
    }

    fn sampled_token_value(token: CandleToken) -> u32 {
        token.0
    }
}

impl eredu_core::backend::local_text::HostCausalVerification for CandleContext {
    type Values = crate::text_native::CandleTextValues;
    type Queues = Self;
    type Assistant = crate::prediction::CandleAssistantOperators;
    type Embedded = crate::prediction::CandlePredictionOperators;
    type ContextBinding = Self::Values;

    fn execution_device(&self) -> Result<eredu_core::DevicePlan, BackendFailure> {
        Ok(self.placement().device_plan())
    }

    fn verification_memory(
        context: &Self,
    ) -> Result<eredu_core::capability::ParameterMemoryFacts, BackendFailure> {
        eredu_core::capability::NativePlacementMemory::placement_memory(context)
    }

    type InputTransfer = crate::input_transfer::CandlePromptTransfer;

    fn verification_input_transfer(source: &Self) -> Self::InputTransfer {
        Self::InputTransfer::new(source)
    }

    fn verification_tokens(tokens: &[u32], context: &Self) -> Result<CandleTensor, BackendFailure> {
        eredu_core::input::PlainTextTokens::materialize_tokens(context, tokens)
    }

    fn verification_score(
        scores: &CandleTensor,
        position: usize,
        context: &Self,
    ) -> Result<Vec<f32>, BackendFailure> {
        let position = i32::try_from(position).map_err(BackendFailure::from_error)?;
        score_row(scores, position, context)
    }

    fn verification_storage(scores: &CandleTensor) -> Option<u64> {
        let payload = u64::try_from(scores.value.elem_count())
            .ok()?
            .checked_mul(scores.value.dtype().size_in_bytes() as u64)?;
        payload.checked_add(4096)
    }
}

fn score_row(
    scores: &CandleTensor,
    position: i32,
    context: &CandleContext,
) -> Result<Vec<f32>, BackendFailure> {
    let axes = match scores.shape() {
        [1, _] if matches!(position, 0 | -1) => &[Index::At(0), Index::Full][..],
        [1, _, _] => &[Index::At(0), Index::At(position), Index::Full][..],
        _ => return Err(crate::text::invalid("invalid vocabulary score row")),
    };
    scores
        .index(axes, context)
        .and_then(|row| row.to_f32_vec(context))
        .map_err(BackendFailure::from_error)
}
