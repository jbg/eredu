//! MLX composition conformance for backend-neutral architecture operators.

use crate::backend::ExecutionContext;
use eredu_nn::Tensor;
use eredu_runtime::{DeviceState, LayeredArchitecture, LayeredForwardState};
use safemlx::{transforms::async_eval_with_event, Array, Device, DeviceType};
use std::sync::OnceLock;

use crate::backend::{
    nn::{shared::MlxNeuralBackend, tensor::TokenValidationScope},
    runtime::cache::{
        kv::CompressedLatentCache,
        state::{MlxHybridState, MlxKeyValueState, MlxPoolingAttentionStateFactory},
    },
};
use crate::MlxTensor;

include!("mlx_architecture_conformance/support.rs");
include!("mlx_architecture_conformance/foundations.rs");
include!("mlx_architecture_conformance/text_profiles.rs");
include!("mlx_architecture_conformance/dense_profiles.rs");
include!("mlx_architecture_conformance/multimodal_moe.rs");
include!("mlx_architecture_conformance/recurrent_profiles.rs");
include!("mlx_architecture_conformance/deepseek_profiles.rs");
include!("mlx_architecture_conformance/realtime.rs");
