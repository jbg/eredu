//! Cold-path architecture/backend composition selected by public loaders.

use ref_cast::RefCast;
use safemlx::Array;

use eredu_architectures::{BindableStaticParameters, StaticParameterVisitor};
use eredu_checkpoint::{recipe::DerivedWeightRecipe, store::CheckpointSource};
use eredu_nn::Parameterized;
use eredu_runtime::StaticUnitBindings;

use crate::{backend::mlx::error::Error, backend::mlx::nn::shared::MlxBackend};

struct StaticBindingVisitor<'a> {
    store: &'a dyn CheckpointSource,
    recipes: std::collections::BTreeMap<String, DerivedWeightRecipe>,
    units: Vec<StaticUnitBindings>,
}

impl StaticParameterVisitor<MlxBackend> for StaticBindingVisitor<'_> {
    type Error = Error;

    fn visit<M>(&mut self, role: &str, module: &M) -> Result<(), Self::Error>
    where
        M: Parameterized<crate::MlxTensor>,
    {
        let bindings =
            crate::backend::mlx::runtime::checkpoint::binding::build_neutral_module_bindings_with_recipes(
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
    A: BindableStaticParameters<MlxBackend>,
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
