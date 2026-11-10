//! Neutral Inkling folded hMLP image tower.

use eredu_nn::{
    DeclareLogicalParameters, Error, LinearOperator, LinearSpec, LogicalParameterDeclaration,
    NeuralBackend, NormalizationConstructionSpec, NormalizationOperator, ParameterSpec,
    Parameterized, Tensor,
};

use super::{VisionConfig, VisionPolicy};

/// One bounded hMLP stage's neural construction and fold geometry.
#[derive(Debug, Clone)]
pub(crate) struct LayerSpec {
    pub projection: LinearSpec,
    pub norm: Option<NormalizationConstructionSpec>,
    temporal_fold: i32,
    spatial_fold: i32,
}
impl LayerSpec {
    fn new(config: &VisionPolicy, layer: usize) -> Result<Self, Error> {
        let second_stage_width = if config.text_hidden_size == 4096 {
            320
        } else {
            512
        };
        let stages = [
            (75, 128, 1, 5),
            (512, second_stage_width, 1, 2),
            (second_stage_width * 16, 4800, 1, 4),
            (9600, config.text_hidden_size, 2, 1),
        ];
        let count = usize::try_from(config.num_hidden_layers).map_err(Error::backend)?;
        let (input, output, temporal_fold, spatial_fold) = stages
            .get(layer)
            .copied()
            .filter(|_| layer < count)
            .ok_or_else(|| {
                Error::backend(format!("Inkling vision layer {layer} is out of range"))
            })?;
        let weight = format!("visual.layers.{layer}.projection.weight");
        let spec = Self {
            projection: LinearSpec {
                input,
                output,
                weight: ParameterSpec::trainable(&weight).map_err(Error::backend)?,
                bias: None,
                format: crate::linear_format::standard_linear_format(
                    &weight,
                    config.linear_format_for(&weight),
                )?,
            },
            norm: (layer + 1 != config.num_hidden_layers as usize)
                .then(|| {
                    Ok::<_, Error>(NormalizationConstructionSpec::learned(
                        output,
                        config.rms_norm_eps,
                        ParameterSpec::trainable(format!(
                            "visual.layers.{layer}.layer_norm.weight"
                        ))
                        .map_err(Error::backend)?,
                    ))
                })
                .transpose()?,
            temporal_fold,
            spatial_fold,
        };
        eredu_nn::validate_logical_parameter_topology(&spec.logical_parameters()?)
            .map_err(Error::backend_source)?;
        Ok(spec)
    }
}
impl DeclareLogicalParameters for LayerSpec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        let mut parameters = self.projection.logical_parameters()?;
        if let Some(norm) = &self.norm {
            parameters.extend(norm.logical_parameters()?);
        }
        Ok(parameters)
    }
}

fn final_norm_spec(config: &VisionPolicy) -> Result<NormalizationConstructionSpec, Error> {
    let spec = NormalizationConstructionSpec::learned(
        config.text_hidden_size,
        config.rms_norm_eps,
        ParameterSpec::trainable("visual.final_norm.weight").map_err(Error::backend)?,
    );
    spec.validate()?;
    Ok(spec)
}

/// Complete immutable hMLP scopes; stages retain independent residency ownership.
#[derive(Debug, Clone)]
pub(crate) struct Parameters {
    pub layers: Vec<LayerSpec>,
    pub final_norm: NormalizationConstructionSpec,
}
impl Parameters {
    pub(super) fn new(config: &VisionPolicy) -> Result<Self, Error> {
        if config.vision_encoder_type != "hmlp"
            || !config.use_vision_norm
            || (
                config.temporal_patch_size,
                config.patch_size,
                config.num_hidden_layers,
                config.num_channels,
            ) != (2, 40, 4, 3)
        {
            return Err(Error::backend(
                "Inkling vision configuration is not the released hMLP tower",
            ));
        }
        let parameters = Self {
            layers: (0..config.num_hidden_layers as usize)
                .map(|layer| LayerSpec::new(config, layer))
                .collect::<Result<_, _>>()?,
            final_norm: final_norm_spec(config)?,
        };
        eredu_nn::validate_logical_parameter_topology(&parameters.logical_parameters()?)
            .map_err(Error::backend_source)?;
        parameters.validate_formats(config)?;
        Ok(parameters)
    }
    pub fn layer(&self, index: usize) -> Result<&LayerSpec, Error> {
        self.layers
            .get(index)
            .ok_or_else(|| Error::backend(format!("Inkling vision layer {index} is out of range")))
    }
    fn validate_formats(&self, config: &VisionPolicy) -> Result<(), Error> {
        crate::format_references::validate_matrices(
            self.logical_parameters()?,
            config
                .quantized_weight_configs
                .iter()
                .flat_map(|formats| formats.iter())
                .map(|(name, format)| (name.as_str(), Some((*format).into()))),
        )
    }
    pub(super) fn project_formats(&mut self, config: &VisionPolicy) -> Result<(), Error> {
        for layer in &mut self.layers {
            let spec = &mut layer.projection;
            let mut format = crate::linear_format::standard_linear_format(
                spec.weight.id.as_str(),
                config.linear_format_for(spec.weight.id.as_str()),
            )?
            .with_arithmetic(spec.format.arithmetic())?;
            if spec.format.exact_zero_inputs() {
                format = format.with_exact_zero_inputs();
            }
            spec.format = format;
        }
        eredu_nn::validate_logical_parameter_topology(&self.logical_parameters()?)
            .map_err(Error::backend_source)?;
        self.validate_formats(config)
    }
}
impl DeclareLogicalParameters for Parameters {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        let mut parameters = self.final_norm.logical_parameters()?;
        for layer in &self.layers {
            parameters.extend(layer.logical_parameters()?);
        }
        Ok(parameters)
    }
}

#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
/// One folded projection stage in the fixed hMLP image tower.
pub struct VisionLayer<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> {
    projection: B::Linear,
    norm: Option<B::Normalization>,
    #[parameter(skip)]
    temporal_fold: i32,
    #[parameter(skip)]
    spatial_fold: i32,
}

impl<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> VisionLayer<B> {
    /// Builds one unloaded folded projection unit.
    pub fn new(
        config: &VisionConfig,
        layer: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let spec = config.parameters().layer(layer)?.clone();
        Ok(Self {
            projection: B::linear(spec.projection, context)?,
            norm: spec
                .norm
                .map(|norm| B::normalization(norm, context))
                .transpose()?,
            temporal_fold: spec.temporal_fold,
            spatial_fold: spec.spatial_fold,
        })
    }

    /// Folds and projects one hMLP stage.
    pub fn forward(
        &mut self,
        input: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let mut hidden = fold(input, self.temporal_fold, self.spatial_fold, context)?;
        hidden = self.projection.forward(&hidden, context)?;
        if let Some(norm) = self.norm.as_mut() {
            hidden = B::Tensor::gelu(&norm.forward(&hidden, context)?, context)?;
        }
        Ok(hidden)
    }
}

/// Pinned final normalization for the hMLP tower.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct VisionStatic<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> {
    final_norm: B::Normalization,
    #[parameter(skip)]
    hidden_size: i32,
}

impl<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> VisionStatic<B> {
    /// Builds the unloaded pinned image normalization.
    pub fn new(
        config: &VisionConfig,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        Ok(Self {
            final_norm: B::normalization(config.parameters().final_norm.clone(), context)?,
            hidden_size: config.text_hidden_size,
        })
    }

    /// Normalizes and flattens the final hMLP activation into decoder embeddings.
    pub fn finish(
        &mut self,
        hidden: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let hidden = self.final_norm.forward(hidden, context)?;
        hidden.reshape(&[1, -1, self.hidden_size], context)
    }
}

/// Fixed four-layer Inkling hMLP image tower.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct VisionTower<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> {
    /// Pinned final normalization.
    pub static_modules: VisionStatic<B>,
    /// Independently streamable folded projection stages.
    pub layers: Vec<VisionLayer<B>>,
}

impl<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> VisionTower<B> {
    /// Builds the released folded hMLP tower.
    pub fn new(
        config: &VisionConfig,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        Ok(Self {
            static_modules: VisionStatic::new(config, context)?,
            layers: (0..config.num_hidden_layers as usize)
                .map(|layer| VisionLayer::new(config, layer, context))
                .collect::<Result<Vec<_>, _>>()?,
        })
    }

    /// Encodes patches shaped `[patches, 2, 40, 40, 3]` into decoder embeddings.
    pub fn forward(
        &mut self,
        patches: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        if patches.shape().len() != 5 || patches.shape()[1..] != [2, 40, 40, 3] {
            return Err(Error::backend("invalid Inkling hMLP patch geometry"));
        }
        let mut hidden = patches.clone();
        for layer in &mut self.layers {
            hidden = layer.forward(&hidden, context)?;
        }
        self.static_modules.finish(&hidden, context)
    }
}

fn fold<T: Tensor>(
    input: &T,
    temporal: i32,
    spatial: i32,
    context: &T::Context,
) -> Result<T, Error> {
    if temporal == 1 && spatial == 1 {
        return Ok(input.clone());
    }
    let shape = input.shape();
    if shape.len() != 5
        || shape[1] % temporal != 0
        || shape[2] % spatial != 0
        || shape[3] % spatial != 0
    {
        return Err(Error::backend("invalid Inkling hMLP fold geometry"));
    }
    let (batch, time, height, width, channels) = (shape[0], shape[1], shape[2], shape[3], shape[4]);
    input
        .reshape(
            &[
                batch,
                time / temporal,
                temporal,
                height / spatial,
                spatial,
                width / spatial,
                spatial,
                channels,
            ],
            context,
        )?
        .transpose_axes(&[0, 1, 3, 5, 2, 4, 6, 7], context)?
        .reshape(
            &[
                batch,
                time / temporal,
                height / spatial,
                width / spatial,
                temporal * spatial * spatial * channels,
            ],
            context,
        )
}
