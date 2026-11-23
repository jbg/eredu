//! Canonical checkpoint bindings for unloaded module parameter trees.
//!
//! These helpers keep checkpoint-name expansion, shape validation, byte
//! accounting, and resident-lease assignment independent of model families.

use eredu_core::checkpoint_mechanisms::bounded_quantization::BoundedQuantizationBackend;
use eredu_core::checkpoint_mechanisms::{
    ParameterBindingTarget, ParameterReadInput, ParameterReadScope,
};
use eredu_core::residency::parameter_binding::WeightBinding;

use safemlx::Stream;

use crate::{
    backend::nn::shared::MlxNeuralBackend,
    backend::runtime::checkpoint::recipe::{
        recipe_dtype_from_mlx, DirectRecipeRead, WeightRecipeError,
    },
    backend::runtime::checkpoint::store::{
        MlxParameterMaterializationContext, WeightMaterialization,
    },
};
use eredu_checkpoint::recipe::DerivedWeightRecipe;

const MODEL_LOAD_MATERIALIZATION_BUFFERS: usize = 2;

impl eredu_core::checkpoint_mechanisms::ExactParameterMaterializer<MlxNeuralBackend>
    for MlxNeuralBackend
{
    fn parameter_target(parameter: &crate::MlxTensor) -> Option<ParameterBindingTarget> {
        mlx_parameter_binding_target(parameter)
    }

    fn lower_mxfp4(
        recipe: DerivedWeightRecipe,
        source: &dyn eredu_checkpoint::store::CheckpointSource,
    ) -> Result<DerivedWeightRecipe, eredu_core::BackendFailure> {
        recipe
            .lower_mxfp4_words(source)
            .map_err(eredu_core::BackendFailure::from_error)
    }

    fn quantizer(
        context: &Stream,
    ) -> Result<Box<dyn BoundedQuantizationBackend + '_>, eredu_core::BackendFailure> {
        Ok(Box::new(
            super::bounded_quantization::MlxBoundedQuantizer::new(context),
        ))
    }
}

impl ParameterReadScope for MlxParameterMaterializationContext {
    type DirectRead = DirectRecipeRead;
    type Value = crate::MlxTensor;
    type Pending = WeightMaterialization;
    type Error = ModuleBindingError;

    fn slots(&self) -> std::num::NonZeroUsize {
        std::num::NonZeroUsize::new(MODEL_LOAD_MATERIALIZATION_BUFFERS).unwrap()
    }

    fn direct_read(
        &self,
        source: &dyn eredu_checkpoint::store::CheckpointSource,
        binding: &WeightBinding,
    ) -> Result<Option<DirectRecipeRead>, ModuleBindingError> {
        DirectRecipeRead::for_binding(source, binding).map_err(Into::into)
    }

    fn submit_direct(
        &mut self,
        reads: Vec<(DirectRecipeRead, eredu_nn::ParameterStorageLayout)>,
    ) -> Result<WeightMaterialization, ModuleBindingError> {
        let (reads, layouts): (Vec<_>, Vec<_>) = reads.into_iter().unzip();
        let inputs = DirectRecipeRead::materialize_many(reads, self.source_stream())?;
        let mut prepared = WeightMaterialization::prepare_retained(inputs, Vec::new())?;
        for (index, layout) in layouts.into_iter().enumerate() {
            let input = &prepared.inputs()[index];
            let output = if self.source_stream() == self.execution_stream() {
                input.clone()
            } else {
                super::recipe::copy_parameter_storage(input, layout, self.execution_stream())
                    .map_err(WeightRecipeError::from)?
            };
            prepared.retain_output(output);
        }
        Ok(prepared.submit_prepared_outputs()?)
    }

    fn submit_ordinary(
        &mut self,
        _: &str,
        input: ParameterReadInput<'_>,
    ) -> Result<WeightMaterialization, ModuleBindingError> {
        match input {
            ParameterReadInput::Recipe(recipe, store) => {
                let (source, sources) =
                    super::recipe::prepare_ordinary_recipe(recipe, store, self, false)?
                        .into_parts();
                let prepared = WeightMaterialization::prepare_retained(vec![source], sources)?;
                let output = if self.source_stream() == self.execution_stream() {
                    prepared.inputs()[0].clone()
                } else {
                    prepared.inputs()[0]
                        .ensure_device_storage(self.execution_stream())
                        .map_err(WeightRecipeError::from)?
                };
                Ok(prepared.submit_outputs(vec![output])?)
            }
            ParameterReadInput::Lease(lease) => Ok(self
                .weight_lease(lease)?
                .materialize(self.source_stream(), self.execution_stream())?),
        }
    }

    fn take_values(
        &mut self,
        pending: WeightMaterialization,
    ) -> Result<Vec<crate::MlxTensor>, ModuleBindingError> {
        Ok(pending
            .synchronize_many()?
            .into_iter()
            .map(crate::MlxTensor::from_array)
            .collect())
    }

    fn checkpoint_capacity_exhausted(error: &ModuleBindingError) -> bool {
        use eredu_checkpoint::store::StoreError::CapacityExhausted;
        matches!(
            error,
            ModuleBindingError::CheckpointStore(CapacityExhausted { .. })
                | ModuleBindingError::WeightRecipe(WeightRecipeError::CheckpointStore(
                    CapacityExhausted { .. }
                ))
        )
    }
}

fn mlx_parameter_binding_target(parameter: &crate::MlxTensor) -> Option<ParameterBindingTarget> {
    let shape = parameter
        .as_array()
        .shape()
        .iter()
        .map(|&dimension| usize::try_from(dimension))
        .collect::<Result<Vec<_>, _>>()
        .ok()?;
    Some(ParameterBindingTarget {
        shape,
        dtype: recipe_dtype_from_mlx(parameter.as_array().dtype()),
        // Floating parameters are unloaded handles whose storage is replaced by
        // the materialized source. Their initial dtype does not request a cast.
        // FP8 values and exponent scales use MLX byte arrays without conversion.
        // Other packed and integer parameters retain exact representation matching.
        permitted_source_dtypes: match parameter.as_array().dtype() {
            safemlx::Dtype::Float16 | safemlx::Dtype::Bfloat16 | safemlx::Dtype::Float32 => vec![
                eredu_checkpoint::recipe::RecipeDtype::F16,
                eredu_checkpoint::recipe::RecipeDtype::BF16,
                eredu_checkpoint::recipe::RecipeDtype::F32,
            ],
            safemlx::Dtype::Uint8 => vec![
                eredu_checkpoint::recipe::RecipeDtype::U8,
                eredu_checkpoint::recipe::RecipeDtype::F8E4M3,
                eredu_checkpoint::recipe::RecipeDtype::F8E8M0,
            ],
            _ => Vec::new(),
        },
    })
}

/// Structured module-to-checkpoint binding failures.
#[derive(Debug, thiserror::Error)]
pub enum ModuleBindingError {
    /// Portable materialization failure.
    #[error(transparent)]
    Operation(#[from] eredu_core::BackendFailure),
    /// Backend-neutral checkpoint inspection or lease acquisition failed.
    #[error(transparent)]
    CheckpointStore(#[from] eredu_checkpoint::store::StoreError),
    /// MLX checkpoint materialization failed.
    #[error(transparent)]
    CheckpointMaterialization(
        #[from] crate::backend::runtime::checkpoint::store::CheckpointMaterializationError,
    ),
    /// Derived-weight metadata validation failed.
    #[error(transparent)]
    WeightRecipe(#[from] crate::backend::runtime::checkpoint::recipe::WeightRecipeError),
}
