//! Cold-path architecture/backend composition selected by public loaders.

use ref_cast::RefCast;
use safemlx::Array;

use eredu_architectures::{BindableStaticParameters, StaticParameterVisitor};
use eredu_checkpoint::{recipe::DerivedWeightRecipe, store::CheckpointSource};
use eredu_nn::Parameterized;
use eredu_runtime::StaticUnitBindings;

use crate::{backend::error::Error, backend::nn::shared::MlxNeuralBackend};

struct StaticBindingVisitor<'a> {
    store: &'a dyn CheckpointSource,
    recipes: std::collections::BTreeMap<String, DerivedWeightRecipe>,
    units: Vec<StaticUnitBindings>,
}

impl StaticParameterVisitor<MlxNeuralBackend> for StaticBindingVisitor<'_> {
    type Error = Error;

    fn visit<M>(&mut self, role: &str, module: &M) -> Result<(), Self::Error>
    where
        M: Parameterized<crate::MlxTensor>,
    {
        let bindings =
            crate::backend::runtime::checkpoint::binding::build_neutral_module_bindings_with_recipes(
                module,
                self.store,
                &mut self.recipes,
            )?;
        self.units.push(StaticUnitBindings::new(role, bindings)?);
        Ok(())
    }
}

pub(crate) fn architecture_static_units<A>(
    architecture: &A,
    store: &dyn CheckpointSource,
) -> Result<Vec<StaticUnitBindings>, Error>
where
    A: BindableStaticParameters<MlxNeuralBackend>,
{
    let recipes = architecture
        .static_parameter_recipes(store)
        .map_err(Error::UnsupportedArchitecture)?;
    let mut visitor = StaticBindingVisitor {
        store,
        recipes,
        units: Vec::new(),
    };
    architecture.visit_static_parameters(&mut visitor)?;
    if !visitor.recipes.is_empty() {
        return Err(Error::UnsupportedArchitecture(format!(
            "architecture declared static recipes for unknown parameters {:?}",
            visitor.recipes.into_keys().collect::<Vec<_>>()
        )));
    }
    Ok(visitor.units)
}

pub(crate) fn architecture_expert_units(
    units: impl IntoIterator<Item = eredu_architectures::ExpertResidencyUnit>,
    store: &dyn CheckpointSource,
    layout: Option<&eredu_runtime::LocalModelLayout>,
) -> Result<Vec<crate::backend::runtime::residency::expert_cache::ExpertCatalogEntry>, Error> {
    use crate::backend::runtime::residency::expert_cache::ExpertCatalogEntry;
    use eredu_runtime::{OffloadUnit, WeightBinding};

    units
        .into_iter()
        .map(|unit| {
            let identity = unit.identity();
            let unit_path = unit.unit_path().to_owned();
            let mut bindings = unit
                .into_parameters()
                .into_iter()
                .map(|parameter| {
                    let (binding_name, logical_target, mut recipe, role) = parameter.into_parts();
                    if recipe.infer(store)?.dtype() == &eredu_checkpoint::recipe::RecipeDtype::F4 {
                        recipe = crate::backend::runtime::checkpoint::recipe::lower_mxfp4_recipe(
                            recipe, store,
                        )?;
                    }
                    let metadata = recipe.infer(store)?;
                    let mut binding =
                        WeightBinding::from_recipe(binding_name, recipe, metadata.byte_len())?
                            .with_logical_target(logical_target)?;
                    if let eredu_architectures::ExpertParameterRole::QuantizableProjection {
                        scales_binding,
                        biases_binding,
                    } = role
                    {
                        binding =
                            binding.with_quantization_companions(scales_binding, biases_binding)?;
                    }
                    Ok(binding)
                })
                .collect::<Result<Vec<_>, Error>>()?;
            if let Some(layout) = layout {
                bindings = crate::backend::runtime::execution::layerwise::shard_layer_bindings(
                    bindings, &unit_path, store, layout,
                )?;
            }
            let bytes = bindings.iter().try_fold(0u64, |total, binding| {
                total.checked_add(binding.expected_bytes()).ok_or_else(|| {
                    Error::UnsupportedArchitecture(format!(
                        "expert {identity:?} byte total overflowed"
                    ))
                })
            })?;
            Ok(ExpertCatalogEntry::new(
                identity,
                OffloadUnit::new(identity.unit_id(), bindings)?,
                bytes,
            )?)
        })
        .collect()
}

pub(crate) fn tensor_ref(array: &Array) -> &crate::MlxTensor {
    crate::MlxTensor::ref_cast(array)
}

pub(crate) fn tensor_opt(array: Option<&Array>) -> Option<&crate::MlxTensor> {
    array.map(tensor_ref)
}

pub mod deepseek;
pub mod deepseek_expert;
pub mod gemma4;
pub mod gemma4_expert;
#[cfg(feature = "media")]
pub mod gemma4_processor;
pub mod gpt_oss;
pub mod inkling;
pub mod inkling_expert;
#[cfg(feature = "media")]
pub mod inkling_processor;
pub mod kimi_linear;
// MLX adapter only; the neutral family is always available from
// `eredu_architectures::lfm2`.
pub mod lfm2;
pub mod llama;
pub mod mlx;
pub mod moshi;
pub mod moshi_parallel;
pub mod muse_glimmer;
pub mod muse_glimmer_expert;
#[cfg(feature = "image")]
pub mod muse_glimmer_processor;
pub mod nemotron_h;
pub mod qwen;

#[cfg(test)]
#[path = "tests/mlx_architecture_conformance.rs"]
mod mlx_architecture_conformance;
