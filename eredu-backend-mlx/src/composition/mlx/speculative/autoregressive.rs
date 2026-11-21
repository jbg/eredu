//! MLX tensors and completion primitives for portable speculative execution.
use super::*;
use crate::composition::mlx::session::bounded_capture::MlxSpeculativeCaptureMechanisms;
use eredu_core::execution_control::SnapshotEstimate;
use eredu_core::speculative::installed::InstalledSpeculationOperators;

/// MLX graph, state-copy, and random-key operations for speculative execution.
pub struct MlxAutoregressiveMechanisms;

impl InstalledSpeculationOperators for MlxAutoregressiveMechanisms {
    type Text = crate::composition::mlx::replicated_text::MlxTextValues;
    type Memory = crate::composition::mlx::capability::MlxAllocatorObservations;
    type Capture = MlxSpeculativeCaptureMechanisms;
    type Refinement = MlxSamplingBackend;
    type Completion = MlxSpeculativeCompletion;
    type Telemetry = scheduler::SpeculativeComponentTimings;
    type Error = Exception;

    fn causal_tokens(tokens: &[u32]) -> Result<Array, Exception> {
        let width = i32::try_from(tokens.len())
            .map_err(|_| Exception::custom("causal append exceeds native sequence geometry"))?;
        if width == 0 {
            return Err(Exception::custom(
                "causal append requires a nonempty sequence",
            ));
        }
        Ok(Array::from_slice(tokens, &[1, width]))
    }

    fn refinement_inputs(
        canvas: &MlxTensor,
        canvas_shape: &[i32],
        feedback: Option<(&MlxTensor, &[i32])>,
        queue: &Stream,
    ) -> Result<(Array, Option<Array>), Exception> {
        Ok((
            canvas.as_array().reshape(canvas_shape, queue)?,
            feedback
                .map(|(value, shape)| value.as_array().reshape(shape, queue))
                .transpose()?,
        ))
    }

    fn refinement_output(
        scores: Array,
        shape: &[i32],
        queue: &Stream,
    ) -> Result<MlxTensor, Exception> {
        Ok(MlxTensor::from_array(scores.reshape(shape, queue)?))
    }

    fn score_row(scores: &Array, position: usize, queue: &Stream) -> Result<Array, Exception> {
        let row = i32::try_from(position).map_err(Exception::from_source)?;
        scores.try_index_device((.., row, ..), queue)
    }
    fn submit(scores: &Array) -> Result<MlxSpeculativeCompletion, Exception> {
        MlxSpeculativeCompletion::submit([scores])
    }
    fn logits_estimate(scores: &Array) -> Option<SnapshotEstimate> {
        super::state_snapshot::estimate_array(scores)
    }
    fn verification_bytes(scores: &Array) -> Option<u64> {
        (scores.nbytes() as u64).checked_mul(2)?.checked_add(4096)
    }
    fn execution_error(error: impl std::error::Error + Send + Sync + 'static) -> Exception {
        Exception::from_source(error)
    }
    fn observer_error(error: &Exception) -> Error {
        Error::Exception(Exception::custom(error.to_string()))
    }
    fn invalid(message: &'static str) -> Exception {
        Exception::custom(message)
    }

    fn score_format(logits: &Array) -> safemlx::Dtype {
        logits.dtype()
    }
    fn draft_score_format() -> safemlx::Dtype {
        safemlx::Dtype::Float32
    }
    fn refinement_scores(logits: &Array) -> MlxTensor {
        MlxTensor::from_array(logits.clone())
    }
    fn read_refinement_tokens(
        tokens: &MlxTensor,
        range: std::ops::Range<usize>,
        queue: &Stream,
    ) -> Result<Vec<u32>, Exception> {
        tokens
            .as_array()
            .try_index_device(range.start as i32..range.end as i32, queue)?
            .as_dtype(safemlx::Dtype::Uint32, queue)?
            .into_evaluated()?
            .try_to_vec::<u32>()
            .map_err(Exception::from_source)
    }
    fn refinement_score_row(
        scores: &MlxTensor,
        position: usize,
        queue: &Stream,
    ) -> Result<Array, Exception> {
        let row = i32::try_from(position).map_err(Exception::from_source)?;
        let end = row
            .checked_add(1)
            .ok_or_else(|| Exception::custom("score row exceeds native indexing"))?;
        scores.as_array().try_index_device((row..end, ..), queue)
    }

    type RetainedRoot = Array;
    fn refinement_token_root(value: &MlxTensor) -> &Array {
        value.as_array()
    }
    fn refinement_score_root(value: &MlxTensor) -> &Array {
        value.as_array()
    }
    fn refinement_mask_root(value: &MlxTensor) -> &Array {
        value.as_array()
    }
    fn refinement_key_root(value: &RandomState) -> &Array {
        value.as_array()
    }
    fn settle_refinement_roots(roots: Vec<Array>) -> Result<(), Exception> {
        super::state_snapshot::settle(roots.iter().collect())
    }
    fn refinement_root_estimate(root: &Array) -> Option<SnapshotEstimate> {
        super::state_snapshot::estimate_array(root)
    }
    fn randomness_root(seed: u64) -> Result<RandomState, Exception> {
        RandomState::with_seed(seed)
    }
    fn split_randomness(
        root: &RandomState,
        substream: usize,
        queue: &Stream,
    ) -> Result<RandomState, Exception> {
        crate::backend::random::split_key_at(root.as_array(), substream, queue)
            .map(RandomState::from_key)
    }
    fn settle_randomness(randomness: &[&RandomState]) -> Result<(), Exception> {
        super::state_snapshot::settle(randomness.iter().map(|state| state.as_array()).collect())
    }
    fn randomness_bytes(random: &RandomState) -> Option<u64> {
        let survey = safemlx::inspect_storage(&[random.as_array()]).ok()?;
        let backing = &survey.backings[*survey.values.first()?.as_ref()?];
        (backing.allocator_capacity_bytes? as u64)
            .checked_add(std::mem::size_of::<RandomState>() as u64)?
            .checked_add(4096)
    }
}
