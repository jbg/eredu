//! Native bindings for shared-backbone text transactions.
use super::*;
use crate::composition::mlx::speculative::autoregressive::MlxAutoregressiveMechanisms;
use eredu_core::backend::SharedTextNative;

impl eredu_core::backend::CommittedTokenReadback<MlxTextToken, MlxTensor, Stream, Exception>
    for MlxAutoregressiveMechanisms
{
    fn pending_token(token: &MlxTextToken, stream: &Stream) -> Result<u32, Exception> {
        token.value.clone().try_item::<u32>(stream)
    }

    fn committed_block(tokens: &[MlxTensor], stream: &Stream) -> Result<Vec<u32>, Exception> {
        let arrays = tokens.iter().map(MlxTensor::as_array).collect::<Vec<_>>();
        safemlx::ops::concatenate(&arrays, stream)?
            .as_dtype(Dtype::Uint32, stream)?
            .into_evaluated()?
            .try_to_vec::<u32>()
            .map_err(Exception::from_source)
    }
}

impl<Dispatch> SharedTextNative for MlxBackend<'_, Dispatch> {
    type Native = MlxSessionMechanisms;

    fn shared_queue(&self) -> &Stream {
        self.stream()
    }

    fn shared_token_id(&self, token: &MlxTextToken) -> Result<u32, Error> {
        Ok(token.value.clone().try_item::<u32>(self.stream())?)
    }

    fn shared_progress_value() -> Array {
        Array::from_f32(0.0)
    }

    fn finish_shared_progress(
        submission: Submission<Array, MlxSessionCompletion>,
    ) -> Result<MlxTextCompletion, Error> {
        let completion = MlxCompletion::submission(submission.output)?;
        Ok(MlxTextCompletion {
            model: submission.completion,
            token: completion.completion,
            recovery: RefCell::new(None),
        })
    }
}
