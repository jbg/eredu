//! MLX score and token operations under the existing model completion.

use super::*;
use eredu_core::communication::{BroadcastBackend, CommunicationAuthority, CommunicationOperation};
use eredu_core::BackendFailure;

pub(super) fn last_score_row(
    output: &MlxModelOutput,
    stream: &Stream,
) -> Result<Option<MlxTensor>, Error> {
    output
        .logits()
        .map(|logits| {
            let array = logits.as_array();
            match array.shape() {
                [_, sequence, _] if *sequence > 0 => array
                    .try_index_device((.., -1, ..), stream)
                    .map(MlxTensor::from_array)
                    .map_err(Into::into),
                _ => Ok(logits.clone()),
            }
        })
        .transpose()
}

pub(super) fn exchange_sampled_token(
    token: Option<MlxTensor>,
    rank: usize,
    group: &crate::backend::runtime::distributed::Group,
    authority: &CommunicationAuthority,
    stream: &Stream,
) -> Result<MlxTensor, Error> {
    let local = match token {
        Some(token) => token
            .into_array()
            .reshape(&[1, 1], stream)?
            .as_dtype(Dtype::Float32, stream)?,
        None => safemlx::ops::zeros::<f32>(&[1, 1], stream)?,
    };
    let operation = CommunicationOperation::Broadcast;
    let submission = <crate::backend::nn::shared::MlxNeuralBackend as BroadcastBackend>::broadcast(
        MlxTensor::from_array(local),
        rank,
        group,
        stream,
    )
    .map_err(|error| authority.submission_failure(error, operation, None))
    .map_err(BackendFailure::from_error)?;
    let value = authority
        .wait(submission, operation, None)
        .map_err(BackendFailure::from_error)?;
    Ok(MlxTensor::from_array(
        value
            .into_array()
            .as_dtype(Dtype::Uint32, stream)?
            .try_index_device((.., 0), stream)?,
    ))
}

pub(super) fn finish_sampling(
    scope: super::recovery::ResourceOperation,
    model: MlxSessionCompletion,
    token: Result<MlxTensor, Error>,
    stream: &Stream,
) -> Result<Submission<MlxTextToken, MlxTextCompletion>, Error> {
    let submitted = token.and_then(|token| MlxCompletion::submission(token.into_array()));
    let (sampled, recovery) = scope.finish(submitted)?;
    Ok(Submission {
        output: MlxTextToken {
            value: sampled.output,
            stream: stream.clone(),
            owner: std::rc::Rc::clone(model.owner()),
        },
        completion: MlxTextCompletion {
            model,
            token: sampled.completion,
            recovery: std::cell::RefCell::new(Some(recovery)),
        },
    })
}
