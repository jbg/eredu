//! Thin MLX planner registration for the neutral Moshi model.

use eredu_architectures::moshi::{self, LayeredModel};
use eredu_runtime::LayeredArchitecture;
use safemlx::Stream;

use crate::backend::mlx::{
    error::Error,
    nn::shared::MlxNeuralBackend,
    runtime::{cache::state::MlxKeyValueState, distributed::parallel::ParallelPlanBuilder},
};

/// Registers neutral parameter groups with the general MLX planner.
pub fn register_parallel_parameters(
    architecture: &LayeredModel<MlxNeuralBackend>,
    planner: &mut ParallelPlanBuilder,
    stream: &Stream,
) -> Result<(), Error> {
    for group in moshi::static_parameter_groups(architecture.static_modules())? {
        planner.register(group)?;
    }
    for group_index in 0..2 {
        let count = <LayeredModel<MlxNeuralBackend> as LayeredArchitecture<
            MlxNeuralBackend,
            MlxKeyValueState,
        >>::group_unit_count(architecture, group_index)?;
        for index in 0..count {
            let unit = <LayeredModel<MlxNeuralBackend> as LayeredArchitecture<
                MlxNeuralBackend,
                MlxKeyValueState,
            >>::build_unit(architecture, group_index, index, stream)?;
            for group in
                moshi::unit_parameter_groups(&unit, architecture.config(), group_index, index)?
            {
                planner.register(group)?;
            }
        }
    }
    Ok(())
}
