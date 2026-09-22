//! Immutable sampling diagnostics retain the complete admitted saved pair.
use crate::composition::mlx::session::model_session::saved_array_copy::decoder::CopiedTextComponentsOwner;

/// A sampling/input view sharing one saved decoder/sampler pair and its custody.
pub struct MlxSavedSamplingState {
    pair: CopiedTextComponentsOwner,
}

impl MlxSavedSamplingState {
    pub(super) fn paired(pair: CopiedTextComponentsOwner) -> Self {
        Self { pair }
    }

    pub(super) fn paired_control_bytes() -> Option<usize> {
        std::mem::size_of::<Self>().checked_add(std::mem::size_of::<CopiedTextComponentsOwner>())
    }

    pub(super) fn prediction(&self) -> u64 {
        self.pair.sampling().next_prediction()
    }

    pub(super) fn input_tokens(&self, predictions: u64) -> Option<u64> {
        self.pair.sampling().input_tokens(predictions)
    }
}
