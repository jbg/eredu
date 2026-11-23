use eredu_core::{
    capture::{
        host::{estimate_host_transform, transform_f32, transform_scores},
        CaptureBackend, CaptureError, CapturePayload, CaptureSelection, CaptureTokenDomain,
        CaptureTransform, CaptureUsage, CaptureWorkspaceModel, ResolvedCaptureSlice,
    },
    checkpoint::TensorDtype,
    BackendFailure,
};

use eredu_core::sampling::host::HostScores;

pub(super) struct HostScoreCapture<'a>(pub(super) Option<CaptureTokenDomain<'a>>);

impl CaptureBackend for HostScoreCapture<'_> {
    type Tensor = HostScores;
    type Error = BackendFailure;

    fn shape(&self, scores: &HostScores) -> Result<Vec<u64>, BackendFailure> {
        Ok(vec![1, 1, scores.values().len() as u64])
    }

    fn source_dtype(&self, _: &HostScores) -> Option<TensorDtype> {
        // Sampling receives an owned FP32 row after native staging.
        Some(TensorDtype::F32)
    }

    fn estimate(
        &self,
        scores: &HostScores,
        selection: &CaptureSelection,
        slice: &ResolvedCaptureSlice,
    ) -> Result<CaptureUsage, CaptureError> {
        let shape = [1, 1, scores.values().len() as u64];
        estimate_host_transform(
            CaptureWorkspaceModel {
                max_index: i32::MAX as u64,
                reduction_chunk: std::num::NonZeroU64::new(8192).unwrap(),
                source_bytes_per_element: 4,
                selected_bytes_per_element: 4,
                workspace_base_bytes: 2048,
                workspace_bytes_per_element: 16,
                routed: None,
            },
            &shape,
            selection,
            slice,
        )
    }

    fn transform(
        &mut self,
        scores: &HostScores,
        selection: &CaptureSelection,
        slice: &ResolvedCaptureSlice,
        score_row: Option<u64>,
    ) -> Result<CapturePayload, BackendFailure> {
        if matches!(
            selection.transform,
            CaptureTransform::TopCandidates { .. } | CaptureTransform::TokenScores { .. }
        ) {
            if score_row != Some(0) {
                return Err(BackendFailure::from_error(CaptureError::Invalid(
                    "host logits contain one admitted score row".into(),
                )));
            }
            return transform_scores(scores.values(), &selection.transform, self.0)
                .map_err(BackendFailure::from_error);
        }
        let count = match selection.transform {
            CaptureTransform::Preview { max_elements } => slice.shape[2].min(max_elements),
            _ => slice.shape[2],
        } as usize;
        let values = if slice.shape[0] == 0 || slice.shape[1] == 0 {
            Vec::new()
        } else {
            scores.values()[slice.starts[2] as usize..slice.ends[2] as usize]
                .iter()
                .copied()
                .step_by(slice.strides[2] as usize)
                .take(count)
                .collect()
        };
        transform_f32(values, &slice.shape, &selection.transform)
            .map_err(BackendFailure::from_error)
    }
}
