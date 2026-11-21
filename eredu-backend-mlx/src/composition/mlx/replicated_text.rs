//! MLX mechanisms and generic binding for replicated text composition.

use eredu_checkpoint::lowering::WeightLoweringDescriptor;
use eredu_checkpoint::{LinearFormat, SourceTensorEncoding, StoredDtype};
use eredu_core::preparation::WeightResidencyMechanism;
use eredu_nn::GroupedOperationRequirement;
use eredu_nn::NeuralBackend;
use safemlx::{ops::indexing::TryIndexOp, transforms::async_eval_with_event, Stream};

use crate::{
    backend::{
        error::Error,
        nn::shared::MlxNeuralBackend,
        nn::tensor::{active_token_validation_arrays, validate_active_token_validations},
    },
    native_quantization::NativeQuantizationFormat,
    MlxTensor,
};

use crate::composition::mlx::prepared_speculative;

mod capability;
pub(super) use capability::MlxReplicatedTextSupport;
mod local_component;
mod lowering;
mod prediction;
mod session;

pub(crate) use capability::GROUPED_OPERATION_CAPABILITIES;
pub(super) use capability::*;
pub(super) use lowering::*;
pub(super) use prediction::*;
pub(super) use session::binding::*;
pub(super) use session::retained_session::MlxTextValues;
