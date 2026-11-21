//! MLX tensor and completion bindings for resident block refinement.

use super::*;
use crate::backend::submission_recovery::ResourceOperation;
use eredu_core::refinement::TextRefinementNative;

impl<Dispatch> eredu_core::input::CommittedTokenTensors for MlxBackend<'_, Dispatch> {
    type BlockToken = MlxTensor;
    type TokenTensor = Array;

    fn concatenate_committed_tokens(&self, tokens: &[MlxTensor]) -> Result<Array, Error> {
        let rows = tokens.iter().map(MlxTensor::as_array).collect::<Vec<_>>();
        let stream = self.stream();
        Ok(safemlx::ops::concatenate(&rows, stream)?.reshape(&[1, rows.len() as i32], stream)?)
    }

    fn decode_token_tensor(&self, token: MlxTextToken) -> Result<Array, Error> {
        Ok(token.value.reshape(&[1, 1], self.stream())?)
    }
}

impl<Dispatch> TextRefinementNative for MlxBackend<'_, Dispatch> {
    type BlockScope = ResourceOperation;
    type BlockSampling = MlxSamplingBackend;
    type ScoreTensor = Array;
    type InvocationCompletion = MlxSessionCompletion;
    type ConditioningFormat = Dtype;

    fn block_canvas(
        backend: &Self,
        canvas: &MlxTensor,
        canvas_shape: &[i32],
        feedback: Option<(&MlxTensor, &[i32])>,
    ) -> Result<(MlxTensor, Option<MlxTensor>), Error> {
        let stream = backend.stream();
        let canvas = canvas.as_array().reshape(canvas_shape, stream)?;
        let feedback = feedback
            .map(|(scores, shape)| scores.as_array().reshape(shape, stream))
            .transpose()?
            .map(MlxTensor::from_array);
        Ok((MlxTensor::from_array(canvas), feedback))
    }

    fn block_causal_scores(backend: &Self, scores: &Array) -> Result<MlxTensor, Error> {
        let logits = match scores.shape() {
            [1, sequence, vocabulary] if *sequence > 0 && *vocabulary > 0 => {
                scores.try_index_device((.., -1, ..), backend.stream())?
            }
            [1, vocabulary] if *vocabulary > 0 => scores.clone(),
            _ => {
                return Err(Error::ArchitectureModel(
                    "causal context has invalid next-token score geometry".into(),
                ));
            }
        };
        Ok(MlxTensor::from_array(logits))
    }

    fn block_refinement_scores(
        backend: &Self,
        scores: &Array,
        shape: &[i32],
    ) -> Result<MlxTensor, Error> {
        Ok(MlxTensor::from_array(
            scores.reshape(shape, backend.stream())?,
        ))
    }

    fn read_refinement_scores(
        backend: &Self,
        scores: &Array,
        selections: Vec<eredu_core::refinement::RefinementScoreSelection>,
    ) -> Result<eredu_core::refinement::RefinementScoreRows, Error> {
        Ok(super::super::score_readout::read_selected(
            scores,
            selections,
            backend.stream(),
        )?)
    }

    fn block_conditioning_format(scores: &Array) -> Dtype {
        scores.dtype()
    }

    fn block_token_ids(backend: &Self, tokens: &MlxTensor) -> Result<Vec<u32>, Error> {
        tokens
            .as_array()
            .as_dtype(Dtype::Uint32, backend.stream())?
            .into_evaluated()?
            .try_to_vec::<u32>()
            .map_err(|error| Error::ArchitectureModel(error.to_string()))
    }

    fn open_block_scope(
        submission: &Submission<Array, MlxSessionCompletion>,
    ) -> Result<ResourceOperation, Error> {
        ResourceOperation::begin(submission.completion.owner())
    }

    fn block_output_token(
        backend: &Self,
        token: MlxTensor,
        submission: &Submission<Array, MlxSessionCompletion>,
    ) -> MlxTextToken {
        MlxTextToken {
            value: token.into_array(),
            stream: backend.stream().clone(),
            owner: Rc::clone(submission.completion.owner()),
        }
    }

    fn block_canvas_root(canvas: &MlxTensor) -> Array {
        canvas.as_array().clone()
    }

    fn block_token_root(token: &MlxTextToken) -> Array {
        token.value.clone()
    }

    fn finish_block_scope<P>(
        scope: ResourceOperation,
        submission: Submission<Array, MlxSessionCompletion>,
        result: Result<(P, Array), Error>,
    ) -> Result<Submission<P, MlxTextCompletion>, Error> {
        let completed =
            result.and_then(|(output, root)| Ok((output, MlxCompletion::submission(root)?)));
        let ((output, tail), recovery) = scope.finish(completed)?;
        Ok(Submission {
            output,
            completion: MlxTextCompletion {
                token: tail.completion,
                model: submission.completion,
                recovery: RefCell::new(Some(recovery)),
            },
        })
    }
}
