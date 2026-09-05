//! Group selection and packed grouped-projection implementations.

use eredu_checkpoint::WeightQuantization;
use eredu_nn::{GatedProductActivation, GatedProductPolicy, TensorParallelGroupedOutput};

use eredu_backend_mlx_macros::PhysicalParameters;
use safemlx::{
    error::Exception,
    ops::{
        arange, argpartition_axis, concatenate_axis, gather_qmm_with_mode,
        indexing::{scatter_single, take_along_axis, topk_axis, NewAxis, TryIndexOp},
        matmul, mean_axis, quantized_matmul_with_mode, quantized_packed_dimension, r#where, rsqrt,
        sigmoid, softmax_axis, sum_axis, zeros_dtype, QuantizationMode,
    },
    Array, Dtype, Stream,
};

use crate::{
    module::PhysicalParam,
    native_quantization::{native_grouped_linear, NativeQuantizedTensor},
};

use super::grouping::{
    gather_grouped_rows, gather_selection_values, grouped_matmul, topk_group_plan,
    GroupedSelectionPlan,
};
use super::layers::{relu2, silu};

/// Applies one affine- or MXFP4-packed group projection to group-major rows.
///
/// The packed weight and its metadata keep the group dimension leading, so
/// this is usable by any checkpoint layout once split groups have been
/// assembled into `[groups, output, input]` banks.
mod gated_product;
mod packed_linear;
mod relu2;
mod selection;

pub use gated_product::PackedGatedProductGroups;
pub use packed_linear::packed_grouped_linear;
pub use relu2::PackedRelu2Groups;
pub(crate) use selection::weighted_group_sum;
pub use selection::{TopKGroupScoring, TopKGroupSelector, TopKGroupSelectorConfig};

#[cfg(test)]
mod tests;
