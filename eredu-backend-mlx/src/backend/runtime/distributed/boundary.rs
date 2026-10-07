//! Native allocation and dtype conversion for architecture-declared pipeline values.

use crate::{backend::nn::shared::MlxNeuralBackend, MlxTensor};
use safemlx::{Dtype, Stream};

#[derive(Clone, Copy, Default)]
pub(crate) struct MlxPartitionTensorAllocator;

impl eredu_runtime::PartitionTensorAllocator<MlxNeuralBackend> for MlxPartitionTensorAllocator {
    fn tensor_to_wire(
        &mut self,
        tensor: MlxTensor,
        logical_dtype: eredu_runtime::BoundaryTensorDtype,
        activation_dtype: eredu_runtime::PipelineActivationDtype,
        context: &Stream,
    ) -> Result<MlxTensor, eredu_nn::Error> {
        let dtype = mlx_boundary_dtype(logical_dtype, activation_dtype)?;
        let source = tensor.as_array().dtype();
        let valid = match logical_dtype {
            eredu_runtime::BoundaryTensorDtype::Activation => {
                matches!(source, Dtype::Float16 | Dtype::Bfloat16 | Dtype::Float32)
            }
            eredu_runtime::BoundaryTensorDtype::Uint32 => source == Dtype::Uint32,
            eredu_runtime::BoundaryTensorDtype::Int32 => source == Dtype::Int32,
            eredu_runtime::BoundaryTensorDtype::Float32 => source == Dtype::Float32,
            _ => false,
        };
        if !valid {
            return Err(eredu_nn::Error::backend(
                "MLX pipeline source tensor does not match its logical boundary dtype",
            ));
        }
        tensor
            .as_array()
            .as_dtype(dtype, context)
            .map(MlxTensor::from_array)
            .map_err(eredu_nn::Error::backend_source)
    }

    fn tensor_placeholder(
        &mut self,
        shape: &[i32],
        logical_dtype: eredu_runtime::BoundaryTensorDtype,
        activation_dtype: eredu_runtime::PipelineActivationDtype,
        context: &Stream,
    ) -> Result<MlxTensor, eredu_nn::Error> {
        let dtype = mlx_boundary_dtype(logical_dtype, activation_dtype)?;
        safemlx::ops::zeros_dtype(shape, dtype, context)
            .map(MlxTensor::from_array)
            .map_err(eredu_nn::Error::backend_source)
    }
}

pub(crate) fn mlx_boundary_dtype(
    logical: eredu_runtime::BoundaryTensorDtype,
    activation: eredu_runtime::PipelineActivationDtype,
) -> Result<Dtype, eredu_nn::Error> {
    match logical {
        eredu_runtime::BoundaryTensorDtype::Activation => mlx_pipeline_activation_dtype(activation),
        eredu_runtime::BoundaryTensorDtype::Uint32 => Ok(Dtype::Uint32),
        eredu_runtime::BoundaryTensorDtype::Int32 => Ok(Dtype::Int32),
        eredu_runtime::BoundaryTensorDtype::Float32 => Ok(Dtype::Float32),
        _ => Err(eredu_nn::Error::backend(
            "MLX pipeline boundary uses an unsupported logical dtype",
        )),
    }
}

pub(crate) fn mlx_pipeline_activation_dtype(
    dtype: eredu_runtime::PipelineActivationDtype,
) -> Result<Dtype, eredu_nn::Error> {
    match dtype {
        eredu_runtime::PipelineActivationDtype::Float16 => Ok(Dtype::Float16),
        eredu_runtime::PipelineActivationDtype::Bfloat16 => Ok(Dtype::Bfloat16),
        eredu_runtime::PipelineActivationDtype::Float32 => Ok(Dtype::Float32),
        _ => Err(eredu_nn::Error::backend(
            "unsupported MLX pipeline wire dtype",
        )),
    }
}
