//! Test checkpoint shapes constructed through neutral architecture types.

use eredu_architectures::qwen::{hybrid, vl};
use eredu_runtime::LayeredArchitecture;
use safemlx::Stream;

use crate::backend::{
    error::Error,
    nn::shared::MlxNeuralBackend,
    runtime::{
        cache::state::MlxHybridState,
        execution::generic::{architecture_execution_layout, construct_architecture_unit},
    },
};

macro_rules! text_checkpoint_template {
    ($name:ident, $model:ty, $unit:ty, $args:ty) => {
        #[derive(eredu_nn::Parameterized)]
        #[parameterized(tensor = "crate::MlxTensor")]
        pub(crate) struct $name {
            pub static_modules: eredu_architectures::decoder::StaticModules<MlxNeuralBackend>,
            pub layers: Vec<$unit>,
        }

        impl $name {
            pub(crate) fn new(args: $args, stream: &Stream) -> Result<Self, Error> {
                let architecture = <$model>::new(args, stream)
                    .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
                let layout = architecture_execution_layout::<_, MlxHybridState>(&architecture)?;
                let layers = (0..layout.len())
                    .map(|ordinal| {
                        construct_architecture_unit(
                            &architecture,
                            &layout,
                            ordinal,
                            stream,
                            std::marker::PhantomData::<MlxHybridState>,
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(Self {
                    static_modules: architecture.static_modules().clone(),
                    layers,
                })
            }
        }
    };
}

text_checkpoint_template!(
    KimiLinearCheckpointTemplate,
    eredu_architectures::kimi_linear::LayeredModel<MlxNeuralBackend>,
    eredu_architectures::kimi_linear::Block<MlxNeuralBackend>,
    eredu_architectures::kimi_linear::ModelArgs
);
text_checkpoint_template!(
    Lfm2CheckpointTemplate,
    eredu_architectures::lfm2::LayeredModel<MlxNeuralBackend>,
    eredu_architectures::lfm2::Block<MlxNeuralBackend>,
    eredu_architectures::lfm2::ModelArgs
);
text_checkpoint_template!(
    NemotronHCheckpointTemplate,
    eredu_architectures::nemotron_h::LayeredModel<MlxNeuralBackend>,
    eredu_architectures::nemotron_h::Unit<MlxNeuralBackend>,
    eredu_architectures::nemotron_h::ModelArgs
);

#[derive(eredu_nn::Parameterized)]
#[parameterized(tensor = "crate::MlxTensor")]
pub(crate) struct QwenHybridCheckpointTemplate {
    pub static_modules: eredu_architectures::decoder::StaticModules<MlxNeuralBackend>,
    pub units: Vec<hybrid::Unit<MlxNeuralBackend>>,
}

impl QwenHybridCheckpointTemplate {
    pub(crate) fn new(config: hybrid::HybridConfig, stream: &Stream) -> Result<Self, Error> {
        let architecture = hybrid::LayeredModel::<MlxNeuralBackend>::new(config, stream)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        let layout = architecture_execution_layout::<_, MlxHybridState>(&architecture)?;
        let units = (0..layout.len())
            .map(|ordinal| {
                construct_architecture_unit(
                    &architecture,
                    &layout,
                    ordinal,
                    stream,
                    std::marker::PhantomData::<MlxHybridState>,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            static_modules: architecture.into_static_modules(),
            units,
        })
    }
}

#[derive(eredu_nn::Parameterized)]
#[parameterized(tensor = "crate::MlxTensor")]
pub(crate) struct QwenConditionalCheckpointTemplate {
    pub static_modules: hybrid::ConditionalStaticModules<MlxNeuralBackend>,
    pub units: Vec<hybrid::ConditionalUnit<MlxNeuralBackend>>,
}

impl QwenConditionalCheckpointTemplate {
    pub(crate) fn new(parsed: hybrid::ParsedHybridConfig, stream: &Stream) -> Result<Self, Error> {
        type Architecture = hybrid::ConditionalLayeredModel<MlxNeuralBackend>;
        let architecture = Architecture::new(parsed, stream)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        let layout = architecture_execution_layout::<_, MlxHybridState>(&architecture)?;
        let units = (0..layout.len())
            .map(|ordinal| {
                construct_architecture_unit(
                    &architecture,
                    &layout,
                    ordinal,
                    stream,
                    std::marker::PhantomData::<MlxHybridState>,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            static_modules: <Architecture as LayeredArchitecture<
                MlxNeuralBackend,
                MlxHybridState,
            >>::static_modules(&architecture)
            .clone(),
            units,
        })
    }
}

#[derive(eredu_nn::Parameterized)]
#[parameterized(tensor = "crate::MlxTensor")]
pub(crate) struct QwenVlCheckpointTemplate {
    pub static_modules: vl::StaticModules<MlxNeuralBackend>,
    pub units: Vec<vl::Unit<MlxNeuralBackend>>,
}

impl QwenVlCheckpointTemplate {
    pub(crate) fn new(args: vl::ModelArgs, stream: &Stream) -> Result<Self, Error> {
        type Architecture = vl::LayeredModel<MlxNeuralBackend>;
        let architecture = Architecture::new(args, stream)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        let layout = architecture_execution_layout::<_, MlxHybridState>(&architecture)?;
        let units = (0..layout.len())
            .map(|ordinal| {
                construct_architecture_unit(
                    &architecture,
                    &layout,
                    ordinal,
                    stream,
                    std::marker::PhantomData::<MlxHybridState>,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            static_modules: <Architecture as LayeredArchitecture<
                MlxNeuralBackend,
                MlxHybridState,
            >>::static_modules(&architecture)
            .clone(),
            units,
        })
    }
}
