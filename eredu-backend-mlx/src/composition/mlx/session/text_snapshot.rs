//! Native copying for the shared ordinary-continuation snapshot driver.

use super::*;
use eredu_core::execution_control::{SamplingGrowthCosts, TextSnapshotNative};

fn array_storage(array: &Array) -> Option<u64> {
    u64::try_from(array.nbytes())
        .ok()?
        .checked_mul(2)?
        .checked_add(4096)?
        .checked_add(u64::try_from(array.shape().len()).ok()?.checked_mul(16)?)
}

fn copy_array(array: &Array, stream: &Stream) -> Result<Array, Error> {
    Ok(array.contiguous(false, stream)?.deep_clone()?)
}

impl<Dispatch> TextSnapshotNative for MlxBackend<'_, Dispatch> {
    type CausalScores = Array;
    type RefinementTokens = crate::MlxTensor;
    type RefinementScores = crate::MlxTensor;
    type RefinementMask = crate::MlxTensor;

    fn prompt_tensor_identity(tensor: &Array) -> Option<eredu_core::InputTensorIdentity> {
        eredu_core::input::PreparedInputInspector::identity(&input::MlxInputInspector, tensor).ok()
    }
    fn prompt_tensor_bytes(tensor: &Array) -> Option<u64> {
        array_storage(tensor)
    }
    fn token_bytes(token: &MlxTextToken) -> Option<u64> {
        (token.value.size() == 1)
            .then(|| array_storage(&token.value))
            .flatten()
    }
    fn copy_token(backend: &Self, token: &MlxTextToken) -> Result<MlxTextToken, Error> {
        Ok(MlxTextToken {
            value: copy_array(&token.value, backend.stream())?,
            stream: backend.stream().clone(),
            owner: std::rc::Rc::clone(&token.owner),
        })
    }
    fn copy_prompt_tensor(backend: &Self, tensor: &Array) -> Result<Array, Error> {
        copy_array(tensor, backend.stream())
    }
    fn complete_prompt_tensors<'a>(tensors: impl Iterator<Item = &'a Array>) -> Result<(), Error> {
        Ok(safemlx::transforms::eval(tensors)?)
    }
    fn sampling_growth_costs() -> SamplingGrowthCosts {
        SamplingGrowthCosts {
            random_key_bytes: 4096 + 16 + 2 * 8,
            array_overhead: 4096 + 128,
        }
    }

    fn split_random(
        backend: &Self,
        root: &RandomState,
        [first, second]: [usize; 2],
    ) -> Result<[RandomState; 2], Error> {
        let stream = backend.stream();
        let first = crate::backend::random::split_key_at(root.as_array(), first, stream)?;
        let second = crate::backend::random::split_key_at(root.as_array(), second, stream)?;
        safemlx::transforms::eval([&first, &second])?;
        Ok([RandomState::from_key(first), RandomState::from_key(second)])
    }

    fn child_capture_workspace() -> eredu_core::capture::CaptureWorkspaceModel {
        super::bounded_capture::CAPTURE_WORKSPACE
    }
    fn child_intervention_estimator() -> Result<
        std::sync::Arc<dyn eredu_core::intervention::InterventionEstimator>,
        eredu_core::capture::CaptureError,
    > {
        Ok(std::sync::Arc::new(
            super::intervention::NativeInterventionEstimator,
        ))
    }

    fn random_bytes(random: &RandomState) -> Option<u64> {
        array_storage(random.as_array())
    }

    fn score_bytes(scores: &Array) -> Option<u64> {
        array_storage(scores)
    }

    fn refinement_token_bytes(value: &Self::RefinementTokens) -> Option<u64> {
        array_storage(value.as_array())
    }

    fn refinement_score_bytes(value: &Self::RefinementScores) -> Option<u64> {
        array_storage(value.as_array())
    }

    fn refinement_mask_bytes(value: &Self::RefinementMask) -> Option<u64> {
        array_storage(value.as_array())
    }

    fn proposal_random_bytes(random: &RandomState) -> Option<u64> {
        u64::try_from(random.as_array().nbytes())
            .ok()?
            .checked_add(4096)
    }

    fn copy_random(backend: &Self, random: &RandomState) -> Result<RandomState, Error> {
        copy_array(random.as_array(), backend.stream()).map(RandomState::from_key)
    }

    fn copy_scores(backend: &Self, scores: &Array) -> Result<Array, Error> {
        copy_array(scores, backend.stream())
    }
}
