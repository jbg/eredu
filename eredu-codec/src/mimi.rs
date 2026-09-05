//! Mimi neural audio tokenizer support.
//!
//! Mimi is the neural audio codec used by Moshi-family realtime models. This
//! module implements backend-neutral checkpoint parameters, the split residual
//! vector quantizer, and the non-streaming SEANet/transformer encoder and
//! decoder used to map between PCM and Mimi codebook tokens.

use eredu_checkpoint::{
    recipe::{DerivedWeightRecipe, RecipeCatalog, RecipeDtype, RecipeError, RecipeMetadata},
    safetensors::{SafetensorsMetadataCatalog, SafetensorsShardError},
    schema::{
        CatalogPolicy, CheckpointPlanError, SafetensorsCheckpointPlan, SafetensorsTensorConstraint,
        StoredDtypeConstraint,
    },
    store::{
        PreparedCheckpointSource, PreparedTensorSource, ResolvedCheckpointSource,
        SafetensorsWeightStore, SharedCheckpointSource, StoreError, TensorMetadata,
        TensorSelection, TensorSourceProvenance,
    },
    validation::{resolve_safetensors_plan, CheckpointValidation, ResolvedCheckpointPlan},
    SourceTensorEncoding, StoredDtype,
};
use eredu_nn::{
    AttentionMask, Index, LayerNorm, Linear, LinearSpec, NeuralBackend, PadMode, Parameter,
    ParameterId, ParameterMetadata, ParameterSpec, ParameterVisitor, ParameterVisitorMut,
    Parameterized, Rope, Tensor,
};
use eredu_runtime::{
    bind_materialized_unit, materialize_selected_bindings, select_bindings, ParameterBackend,
    ParameterOrchestrationError, ResidencyDeclarationError, SelectedBindingPlan, WeightBinding,
};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
    sync::Arc,
};

use crate::{AudioTokenizer, AudioTokenizerConfig, Error};

const EPSILON: f32 = 1e-5;

fn parameter_name(prefix: &str, field: &str) -> String {
    if prefix.is_empty() {
        field.to_owned()
    } else {
        format!("{prefix}.{field}")
    }
}

fn parameter_spec(id: &str) -> ParameterSpec {
    ParameterSpec::trainable(id).expect("Mimi parameter identities are non-empty")
}

fn unloaded_parameter<T: Tensor>(
    shape: &[i32],
    context: &T::Context,
) -> Result<Parameter<T>, eredu_nn::Error> {
    Parameter::unloaded(parameter_spec("value"), shape, context)
}

fn unloaded_linear<T: Tensor>(
    input: i32,
    output: i32,
    bias: bool,
    context: &T::Context,
) -> Result<Linear<T>, eredu_nn::Error> {
    Linear::unloaded(
        LinearSpec {
            input,
            output,
            weight: parameter_spec("weight"),
            bias: bias.then(|| parameter_spec("bias")),
            format: eredu_nn::LinearFormatSpec::unscaled(eredu_checkpoint::LinearFormat::Dense)
                .unwrap(),
        },
        context,
    )
}

fn unloaded_layer_norm<T: Tensor>(
    dimensions: i32,
    epsilon: f32,
    context: &T::Context,
) -> Result<LayerNorm<T>, eredu_nn::Error> {
    LayerNorm::unloaded(
        dimensions,
        epsilon,
        Some(parameter_spec("weight")),
        Some(parameter_spec("bias")),
        context,
    )
}

/// Mimi resampling strategy.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum ResampleMethod {
    /// Learned convolutional resampling.
    Conv,
}

/// Mimi codec configuration.
#[derive(Debug, Clone)]
pub struct Config {
    /// Audio channels.
    pub channels: i32,
    /// PCM sample rate.
    pub sample_rate: f64,
    /// Codec frame rate.
    pub frame_rate: f64,
    /// Whether the original training path renormalized audio.
    pub renormalize: bool,
    /// Latent resampling method.
    pub resample_method: ResampleMethod,
    /// Active residual codebooks.
    pub num_codebooks: i32,
    /// Total codebooks available in the released checkpoint.
    pub total_codebooks: i32,
    /// Codebook cardinality.
    pub bins: i32,
    /// Codebook embedding dimension.
    pub quantizer_dim: i32,
    /// Model latent dimension.
    pub latent_dim: i32,
}

impl Config {
    /// Released Mimi v0.1 defaults, with a caller-selected active codebook count.
    pub fn v0_1(num_codebooks: Option<i32>) -> Self {
        Self {
            channels: 1,
            sample_rate: 24_000.0,
            frame_rate: 12.5,
            renormalize: true,
            resample_method: ResampleMethod::Conv,
            num_codebooks: num_codebooks.unwrap_or(16),
            total_codebooks: 32,
            bins: 2_048,
            quantizer_dim: 256,
            latent_dim: 512,
        }
    }

    fn validate(&self) -> Result<(), Error> {
        if self.channels <= 0
            || !self.sample_rate.is_finite()
            || self.sample_rate <= 0.0
            || !self.frame_rate.is_finite()
            || self.frame_rate <= 0.0
            || self.num_codebooks <= 0
            || self.num_codebooks > self.total_codebooks
            || self.bins <= 0
            || self.quantizer_dim <= 0
            || self.latent_dim <= 0
        {
            return Err(Error::InvalidShape(format!(
                "invalid Mimi config: channels={}, sample_rate={}, frame_rate={}, num_codebooks={}, total_codebooks={}, bins={}, quantizer_dim={}, latent_dim={}",
                self.channels,
                self.sample_rate,
                self.frame_rate,
                self.num_codebooks,
                self.total_codebooks,
                self.bins,
                self.quantizer_dim,
                self.latent_dim
            )));
        }
        if self.channels != 1
            || self.sample_rate != 24_000.0
            || self.frame_rate != 12.5
            || !self.renormalize
            || self.resample_method != ResampleMethod::Conv
            || self.total_codebooks != 32
            || self.bins != 2_048
            || self.quantizer_dim != 256
            || self.latent_dim != 512
        {
            return Err(Error::InvalidShape(
                "unsupported Mimi configuration; only the released v0.1 profile is admitted".into(),
            ));
        }
        Ok(())
    }
}

/// Mimi audio tokenizer.
#[derive(Debug, Clone)]
pub struct Mimi<T: Tensor> {
    /// Split residual vector quantizer.
    pub quantizer: SplitResidualVectorQuantizer<T>,
    encoder: SeaNetEncoder<T>,
    encoder_transformer: MimiTransformer<T>,
    downsample: StreamableConv1d<T>,
    upsample: StreamableConvTranspose1d<T>,
    decoder_transformer: MimiTransformer<T>,
    decoder: SeaNetDecoder<T>,
    config: Config,
}

impl<T: Tensor> Mimi<T> {
    /// Creates an unloaded Mimi tokenizer from config.
    pub fn new(config: Config, context: &T::Context) -> Result<Self, Error> {
        config.validate()?;
        Ok(Self {
            quantizer: SplitResidualVectorQuantizer::unloaded(&config, context)?,
            encoder: SeaNetEncoder::unloaded(context)?,
            encoder_transformer: MimiTransformer::unloaded(context)?,
            downsample: StreamableConv1d::unloaded_with_pad_mode(
                config.latent_dim,
                config.latent_dim,
                4,
                2,
                false,
                PadMode::Edge,
                context,
            )?,
            upsample: StreamableConvTranspose1d::unloaded(
                config.latent_dim,
                config.latent_dim,
                4,
                2,
                config.latent_dim,
                false,
                context,
            )?,
            decoder_transformer: MimiTransformer::unloaded(context)?,
            decoder: SeaNetDecoder::unloaded(context)?,
            config,
        })
    }

    /// Returns the Mimi configuration.
    pub fn mimi_config(&self) -> &Config {
        &self.config
    }

    /// Encodes latent frames shaped `[batch, 512, frames]` into Mimi tokens.
    pub fn encode_latent(&mut self, latent: &T, context: &T::Context) -> Result<T, Error> {
        self.quantizer.encode(latent, context)
    }

    /// Encodes PCM shaped `[batch, 1, samples]` into Mimi tokens `[batch, codebooks, frames]`.
    pub fn encode(&mut self, pcm: &T, context: &T::Context) -> Result<T, Error> {
        let latent = self.encoder.forward(pcm, context)?;
        let latent = self.encoder_transformer.forward(&latent, context)?;
        let latent = self.downsample.forward(&latent, context)?;
        self.quantizer.encode(&latent, context)
    }

    /// Resets state used by [`Mimi::encode_step`].
    pub fn reset_encode_state(&mut self) {
        self.encoder.reset_state();
        self.encoder_transformer.reset_state();
        self.downsample.reset_state();
    }

    /// Encodes one PCM frame into the next Mimi token frame.
    ///
    /// Accepts PCM shaped `[batch, 1, samples]`. Returns `None` until the
    /// streaming encoder has enough samples to emit a complete codec frame.
    pub fn encode_step(&mut self, pcm: &T, context: &T::Context) -> Result<Option<T>, Error> {
        let latent = match self.encoder.step(pcm, context)? {
            Some(latent) => latent,
            None => return Ok(None),
        };
        let latent = self.encoder_transformer.step(&latent, context)?;
        let latent = match self.downsample.step(&latent, context)? {
            Some(latent) => latent,
            None => return Ok(None),
        };
        Ok(Some(
            self.quantizer
                .encode(&latent, context)?
                .squeeze_axes(&[2], context)?,
        ))
    }

    /// Decodes Mimi tokens shaped `[batch, codebooks, frames]` into latent frames.
    pub fn decode_latent(&mut self, codes: &T, context: &T::Context) -> Result<T, Error> {
        self.quantizer.decode(codes, context)
    }

    /// Decodes Mimi tokens shaped `[batch, codebooks, frames]` into PCM `[batch, 1, samples]`.
    pub fn decode(&mut self, codes: &T, context: &T::Context) -> Result<T, Error> {
        let latent = self.quantizer.decode(codes, context)?;
        let latent = self.upsample.forward(&latent, context)?;
        let latent = self.decoder_transformer.forward(&latent, context)?;
        self.decoder.forward(&latent, context)
    }

    /// Resets state used by [`Mimi::decode_step`].
    pub fn reset_decode_state(&mut self) {
        self.upsample.reset_state();
        self.decoder_transformer.reset_state();
        self.decoder.reset_state();
    }

    /// Decodes one Mimi token frame into the next PCM chunk.
    ///
    /// Accepts codes shaped `[batch, codebooks]` or `[batch, codebooks, 1]`.
    pub fn decode_step(&mut self, codes: &T, context: &T::Context) -> Result<T, Error> {
        let codes = match codes.shape() {
            [_, _] => codes.expand_dims(2, context)?,
            [_, _, 1] => codes.clone(),
            _ => {
                return Err(Error::InvalidShape(format!(
                    "Mimi decode_step expects [batch, codebooks] or [batch, codebooks, 1], got {:?}",
                    codes.shape()
                )));
            }
        };
        let latent = self.quantizer.decode(&codes, context)?;
        let latent = self.upsample.step(&latent, context)?;
        let latent = self.decoder_transformer.step(&latent, context)?;
        self.decoder.step(&latent, context)
    }
}

/// Exact released-checkpoint requirement for one Mimi parameter.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct MimiParameterRequirement {
    logical_name: String,
    checkpoint_key: String,
    physical_shape: Vec<usize>,
    logical_shape: Vec<usize>,
    source_dtype: StoredDtype,
    output_dtype: RecipeDtype,
    source_encoding: SourceTensorEncoding,
    source_bytes: u64,
    output_bytes: u64,
    recipe: DerivedWeightRecipe,
    active: bool,
}

impl MimiParameterRequirement {
    /// Stable parameter identity in [`Mimi`]'s authoritative traversal.
    pub fn logical_name(&self) -> &str {
        &self.logical_name
    }

    /// Exact physical tensor name in the released SafeTensors artifact.
    pub fn checkpoint_key(&self) -> &str {
        &self.checkpoint_key
    }

    /// Exact physical checkpoint geometry.
    pub fn physical_shape(&self) -> &[usize] {
        &self.physical_shape
    }

    /// Exact logical destination geometry after the neutral recipe.
    pub fn logical_shape(&self) -> &[usize] {
        &self.logical_shape
    }

    /// Exact scalar representation in the released SafeTensors payload.
    pub const fn source_dtype(&self) -> &StoredDtype {
        &self.source_dtype
    }

    /// Exact scalar representation produced by the neutral recipe.
    pub const fn output_dtype(&self) -> &RecipeDtype {
        &self.output_dtype
    }

    /// Exact physical container encoding required from the admitted source.
    pub const fn source_encoding(&self) -> &SourceTensorEncoding {
        &self.source_encoding
    }

    /// Exact selected source byte count.
    pub const fn source_bytes(&self) -> u64 {
        self.source_bytes
    }

    /// Exact logical output byte count.
    pub const fn output_bytes(&self) -> u64 {
        self.output_bytes
    }

    /// Neutral source-to-logical layout recipe.
    pub const fn recipe(&self) -> &DerivedWeightRecipe {
        &self.recipe
    }

    /// Whether this parameter is materialized for the selected active codebooks.
    pub const fn is_active(&self) -> bool {
        self.active
    }
}

/// Header-admitted Mimi artifact with an immutable exact parameter plan.
pub struct PreparedMimiArtifact {
    config: Config,
    source: SharedCheckpointSource,
    requirements: Vec<MimiParameterRequirement>,
    bindings: Vec<WeightBinding>,
}

impl std::fmt::Debug for PreparedMimiArtifact {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PreparedMimiArtifact")
            .field("config", &self.config)
            .field("requirements", &self.requirements)
            .field("bindings", &self.bindings)
            .finish_non_exhaustive()
    }
}

impl PreparedMimiArtifact {
    /// Released configuration fixed by neutral preparation.
    pub const fn config(&self) -> &Config {
        &self.config
    }

    /// Exact full-checkpoint requirements, including inactive codebooks.
    pub fn requirements(&self) -> &[MimiParameterRequirement] {
        &self.requirements
    }

    /// Active generic runtime bindings selected by preparation.
    pub fn bindings(&self) -> &[WeightBinding] {
        &self.bindings
    }

    /// Performs backend mechanism admission without reading payloads or
    /// constructing backend-native Mimi tensors.
    pub fn select<B>(
        self,
    ) -> Result<SelectedMimiArtifact<B>, MimiConstructionError<B::ParameterError>>
    where
        B: ParameterBackend<Parameter = <B as NeuralBackend>::Tensor>,
    {
        let Self {
            config,
            source,
            requirements,
            bindings,
        } = self;
        let selected = select_bindings::<B>(source, bindings)?;
        Ok(SelectedMimiArtifact {
            config,
            requirements,
            selected,
        })
    }
}

/// Backend-admitted Mimi plan ready for native materialization.
pub struct SelectedMimiArtifact<B>
where
    B: ParameterBackend<Parameter = <B as NeuralBackend>::Tensor>,
{
    config: Config,
    requirements: Vec<MimiParameterRequirement>,
    selected: SelectedBindingPlan<B>,
}

impl<B> std::fmt::Debug for SelectedMimiArtifact<B>
where
    B: ParameterBackend<Parameter = <B as NeuralBackend>::Tensor>,
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SelectedMimiArtifact")
            .field("config", &self.config)
            .field("requirements", &self.requirements)
            .field("selected", &self.selected)
            .finish_non_exhaustive()
    }
}

impl<B> SelectedMimiArtifact<B>
where
    B: ParameterBackend<Parameter = <B as NeuralBackend>::Tensor>,
{
    /// Constructs, materializes, validates, and atomically binds the ordinary
    /// backend-neutral Mimi type.
    pub fn construct(
        self,
        tensor_context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
        materialization_context: &B::MaterializationContext,
    ) -> Result<Mimi<<B as NeuralBackend>::Tensor>, MimiConstructionError<B::ParameterError>> {
        let SelectedMimiArtifact {
            config,
            requirements: _,
            selected,
        } = self;
        let mut mimi = Mimi::new(config, tensor_context)?;
        let materialized = materialize_selected_bindings::<B>(selected, materialization_context)?;
        bind_materialized_unit::<B, _>(&mut mimi, materialized)?;
        Ok(mimi)
    }
}

/// Failure while preparing a released Mimi artifact from neutral metadata.
#[derive(Debug, thiserror::Error)]
pub enum MimiArtifactError {
    /// Mimi configuration or pure topology was invalid.
    #[error(transparent)]
    Codec(#[from] Error),
    /// Canonical SafeTensors discovery or header admission failed.
    #[error(transparent)]
    Safetensors(#[from] SafetensorsShardError),
    /// A declarative checkpoint plan was internally invalid.
    #[error(transparent)]
    Plan(#[from] CheckpointPlanError),
    /// The exact released catalog contract was not satisfied.
    #[error("Mimi SafeTensors catalog is not exact: {0:?}")]
    Catalog(CheckpointValidation),
    /// Neutral checkpoint storage could not be opened or inspected.
    #[error(transparent)]
    Store(#[from] StoreError),
    /// A neutral layout recipe was invalid.
    #[error(transparent)]
    Recipe(#[from] RecipeError),
    /// A generic runtime binding declaration was invalid.
    #[error(transparent)]
    Binding(#[from] ResidencyDeclarationError),
    /// Codec-owned parameter topology contradicted its released schema.
    #[error("invalid Mimi parameter topology: {0}")]
    Topology(String),
}

/// Failure while selecting, materializing, or constructing Mimi on a backend.
#[derive(Debug, thiserror::Error)]
pub enum MimiConstructionError<E>
where
    E: std::error::Error + Send + Sync + 'static,
{
    /// Backend-neutral Mimi construction failed.
    #[error(transparent)]
    Codec(#[from] Error),
    /// Generic parameter selection, materialization, validation, or binding failed.
    #[error(transparent)]
    Parameters(#[from] ParameterOrchestrationError<E>),
}

/// Inspects and admits an exact released Mimi SafeTensors artifact without
/// reading tensor payloads.
pub fn prepare_checkpoint(
    path: impl AsRef<Path>,
    config: Config,
) -> Result<PreparedMimiArtifact, MimiArtifactError> {
    let (plan, requirements, bindings) = prepare_catalog(&config)?;
    let catalog = SafetensorsMetadataCatalog::discover(path)?;
    let resolution =
        resolve_safetensors_plan(&catalog, &plan).map_err(MimiArtifactError::Catalog)?;
    let store = Arc::new(SafetensorsWeightStore::open_admitted(
        catalog.into_admitted_shards(),
        1,
    )?);
    prepared_from_resolution(config, store, resolution, requirements, bindings)
}

/// Returns the complete released-checkpoint schema derived from Mimi's
/// authoritative parameter topology without inspecting an artifact.
pub fn released_checkpoint_requirements(
    config: &Config,
) -> Result<Vec<MimiParameterRequirement>, MimiArtifactError> {
    prepare_catalog(config).map(|(_, requirements, _)| requirements)
}

/// Admits an already opened backend-neutral SafeTensors-compatible source.
/// No tensor payload is acquired during preparation.
pub fn prepare_source(
    source: SharedCheckpointSource,
    config: Config,
) -> Result<PreparedMimiArtifact, MimiArtifactError> {
    let (plan, requirements, bindings) = prepare_catalog(&config)?;
    let resolution =
        resolve_safetensors_plan(source.as_ref(), &plan).map_err(MimiArtifactError::Catalog)?;
    prepared_from_resolution(config, source, resolution, requirements, bindings)
}

/// Selects generic backend mechanisms, then constructs Mimi through the exact
/// neutral artifact plan.
pub fn construct<B>(
    prepared: PreparedMimiArtifact,
    tensor_context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
    materialization_context: &B::MaterializationContext,
) -> Result<Mimi<<B as NeuralBackend>::Tensor>, MimiConstructionError<B::ParameterError>>
where
    B: ParameterBackend<Parameter = <B as NeuralBackend>::Tensor>,
{
    prepared
        .select::<B>()?
        .construct(tensor_context, materialization_context)
}

fn checkpoint_parameter_for_key(key: &str) -> Option<String> {
    transform_decoder_key(key)
}

fn prepare_catalog(
    config: &Config,
) -> Result<
    (
        SafetensorsCheckpointPlan,
        Vec<MimiParameterRequirement>,
        Vec<WeightBinding>,
    ),
    MimiArtifactError,
> {
    config.validate()?;
    let active_topology = parameter_topology(config.clone())?;
    let mut full_config = config.clone();
    full_config.num_codebooks = full_config.total_codebooks;
    let full_topology = parameter_topology(full_config)?;
    if active_topology
        .keys()
        .any(|name| !full_topology.contains_key(name))
    {
        return Err(MimiArtifactError::Topology(
            "active codebook topology is not contained by the released topology".into(),
        ));
    }

    let mut physical_names = BTreeSet::new();
    let mut requirements = Vec::with_capacity(full_topology.len());
    let mut bindings = Vec::with_capacity(active_topology.len());
    for (logical_name, logical_shape) in full_topology {
        let checkpoint_key = checkpoint_key_for_parameter(&logical_name).ok_or_else(|| {
            MimiArtifactError::Topology(format!(
                "parameter {logical_name:?} has no released checkpoint identity"
            ))
        })?;
        if !physical_names.insert(checkpoint_key.clone()) {
            return Err(MimiArtifactError::Topology(format!(
                "released checkpoint identity {checkpoint_key:?} is duplicated"
            )));
        }
        if checkpoint_parameter_for_key(&checkpoint_key).as_deref() != Some(&logical_name) {
            return Err(MimiArtifactError::Topology(format!(
                "checkpoint identity {checkpoint_key:?} does not round-trip to {logical_name:?}"
            )));
        }
        let axes = checkpoint_layout_axes(&logical_name);
        let physical_shape = match axes {
            Some(axes) => inverse_permuted_shape(&logical_shape, axes)?,
            None => logical_shape.clone(),
        };
        let source_bytes = f32_bytes(&physical_shape, &checkpoint_key)?;
        let output_bytes = f32_bytes(&logical_shape, &logical_name)?;
        let source = DerivedWeightRecipe::source(&checkpoint_key, TensorSelection::Full);
        let recipe = match axes {
            Some(axes) => DerivedWeightRecipe::Transpose {
                input: Box::new(source),
                axes: axes.to_vec(),
            },
            None => source,
        };
        let active = active_topology.contains_key(&logical_name);
        if active {
            bindings.push(WeightBinding::from_recipe(
                &logical_name,
                recipe.clone(),
                output_bytes,
            )?);
        }
        requirements.push(MimiParameterRequirement {
            logical_name,
            checkpoint_key,
            physical_shape,
            logical_shape,
            source_dtype: StoredDtype::F32,
            output_dtype: RecipeDtype::F32,
            source_encoding: SourceTensorEncoding::Safetensors(StoredDtype::F32),
            source_bytes,
            output_bytes,
            recipe,
            active,
        });
    }
    requirements.sort_by(|left, right| left.checkpoint_key.cmp(&right.checkpoint_key));
    bindings.sort_by(|left, right| left.name().cmp(right.name()));
    let constraints = requirements
        .iter()
        .map(|requirement| {
            SafetensorsTensorConstraint::required(
                &requirement.checkpoint_key,
                requirement.physical_shape.clone(),
                StoredDtypeConstraint::Exact(requirement.source_dtype.clone()),
            )
        })
        .collect();
    let plan = SafetensorsCheckpointPlan::new(
        "mimi-v0.1",
        constraints,
        Vec::new(),
        CatalogPolicy::strict(),
    )?;
    Ok((plan, requirements, bindings))
}

fn prepared_from_resolution(
    config: Config,
    source: SharedCheckpointSource,
    resolution: ResolvedCheckpointPlan,
    requirements: Vec<MimiParameterRequirement>,
    bindings: Vec<WeightBinding>,
) -> Result<PreparedMimiArtifact, MimiArtifactError> {
    let mut exact_catalog = BTreeMap::new();
    for requirement in &requirements {
        let key = requirement.checkpoint_key();
        let actual_metadata = source.source_metadata(key)?;
        let expected_metadata = TensorMetadata {
            name: key.to_owned(),
            logical_shape: requirement.physical_shape.clone(),
            physical_shape: requirement.physical_shape.clone(),
            stored_dtype: requirement.source_dtype.clone(),
            encoded_byte_len: requirement.source_bytes,
            backing_shard: actual_metadata.backing_shard.clone(),
        };
        let actual_provenance = source.source_provenance(key)?;
        let expected_provenance = TensorSourceProvenance {
            catalog_key: key.to_owned(),
            physical_tensor: key.to_owned(),
            output: key.to_owned(),
            backing_shard: expected_metadata.backing_shard.clone(),
            source_encoding: requirement.source_encoding.clone(),
        };
        if actual_metadata != expected_metadata || actual_provenance != expected_provenance {
            return Err(MimiArtifactError::Topology(format!(
                "checkpoint tensor {key:?} does not have the exact released SafeTensors provenance"
            )));
        }
        exact_catalog.insert(
            key.to_owned(),
            PreparedTensorSource {
                metadata: expected_metadata,
                provenance: expected_provenance,
            },
        );
    }
    let source: SharedCheckpointSource =
        Arc::new(PreparedCheckpointSource::new(source, exact_catalog)?);
    let source: SharedCheckpointSource =
        Arc::new(ResolvedCheckpointSource::new(source, resolution));
    validate_requirement_recipes(source.as_ref(), &requirements)?;
    Ok(PreparedMimiArtifact {
        config,
        source,
        requirements,
        bindings,
    })
}

fn validate_requirement_recipes(
    catalog: &(impl RecipeCatalog + ?Sized),
    requirements: &[MimiParameterRequirement],
) -> Result<(), MimiArtifactError> {
    for requirement in requirements {
        let actual = requirement.recipe.infer(catalog)?;
        let expected = RecipeMetadata {
            shape: requirement.logical_shape.clone(),
            dtype: requirement.output_dtype.clone(),
            byte_len: requirement.output_bytes,
        };
        if actual != expected {
            return Err(MimiArtifactError::Topology(format!(
                "recipe for {:?} produced {actual:?}, expected {expected:?}",
                requirement.logical_name
            )));
        }
        let metadata = catalog.tensor_metadata(&requirement.checkpoint_key)?;
        if metadata.encoded_byte_len != requirement.source_bytes {
            return Err(MimiArtifactError::Topology(format!(
                "checkpoint tensor {:?} declares {} source bytes, expected {}",
                requirement.checkpoint_key, metadata.encoded_byte_len, requirement.source_bytes
            )));
        }
    }
    Ok(())
}

#[derive(Debug, Clone)]
struct PlanningTensor(Vec<i32>);

impl PlanningTensor {
    fn unavailable() -> Result<Self, eredu_nn::Error> {
        Err(eredu_nn::Error::backend(
            "Mimi planning tensors cannot execute neural operations",
        ))
    }
}

impl Tensor for PlanningTensor {
    type Context = ();

    fn shape(&self) -> &[i32] {
        &self.0
    }

    fn unloaded_f32(shape: &[i32], _: &Self::Context) -> Result<Self, eredu_nn::Error> {
        Ok(Self(shape.to_vec()))
    }

    fn from_f32_slice(_: &[f32], _: &[i32], _: &Self::Context) -> Result<Self, eredu_nn::Error> {
        Self::unavailable()
    }

    fn add(&self, _: &Self, _: &Self::Context) -> Result<Self, eredu_nn::Error> {
        Self::unavailable()
    }

    fn subtract(&self, _: &Self, _: &Self::Context) -> Result<Self, eredu_nn::Error> {
        Self::unavailable()
    }

    fn multiply(&self, _: &Self, _: &Self::Context) -> Result<Self, eredu_nn::Error> {
        Self::unavailable()
    }

    fn multiply_scalar(&self, _: f32, _: &Self::Context) -> Result<Self, eredu_nn::Error> {
        Self::unavailable()
    }

    fn divide(&self, _: &Self, _: &Self::Context) -> Result<Self, eredu_nn::Error> {
        Self::unavailable()
    }

    fn square(&self, _: &Self::Context) -> Result<Self, eredu_nn::Error> {
        Self::unavailable()
    }

    fn maximum_scalar(&self, _: f32, _: &Self::Context) -> Result<Self, eredu_nn::Error> {
        Self::unavailable()
    }

    fn reshape(&self, _: &[i32], _: &Self::Context) -> Result<Self, eredu_nn::Error> {
        Self::unavailable()
    }

    fn transpose_axes(&self, _: &[i32], _: &Self::Context) -> Result<Self, eredu_nn::Error> {
        Self::unavailable()
    }

    fn swap_axes(&self, _: i32, _: i32, _: &Self::Context) -> Result<Self, eredu_nn::Error> {
        Self::unavailable()
    }

    fn transpose(&self, _: &Self::Context) -> Result<Self, eredu_nn::Error> {
        Self::unavailable()
    }

    fn expand_dims(&self, _: i32, _: &Self::Context) -> Result<Self, eredu_nn::Error> {
        Self::unavailable()
    }

    fn squeeze_axes(&self, _: &[i32], _: &Self::Context) -> Result<Self, eredu_nn::Error> {
        Self::unavailable()
    }

    fn index(&self, _: &[Index], _: &Self::Context) -> Result<Self, eredu_nn::Error> {
        Self::unavailable()
    }

    fn take_axis(&self, _: &Self, _: i32, _: &Self::Context) -> Result<Self, eredu_nn::Error> {
        Self::unavailable()
    }

    fn concatenate(_: &[Self], _: i32, _: &Self::Context) -> Result<Self, eredu_nn::Error> {
        Self::unavailable()
    }

    fn stack(_: &[Self], _: i32, _: &Self::Context) -> Result<Self, eredu_nn::Error> {
        Self::unavailable()
    }

    fn matmul(_: &Self, _: &Self, _: &Self::Context) -> Result<Self, eredu_nn::Error> {
        Self::unavailable()
    }

    fn sum_axis(_: &Self, _: i32, _: bool, _: &Self::Context) -> Result<Self, eredu_nn::Error> {
        Self::unavailable()
    }

    fn argmin_axis(_: &Self, _: i32, _: bool, _: &Self::Context) -> Result<Self, eredu_nn::Error> {
        Self::unavailable()
    }

    fn pad(
        _: &Self,
        _: &[(i32, i32)],
        _: PadMode,
        _: &Self::Context,
    ) -> Result<Self, eredu_nn::Error> {
        Self::unavailable()
    }

    fn conv1d(
        _: &Self,
        _: &Self,
        _: i32,
        _: i32,
        _: i32,
        _: i32,
        _: &Self::Context,
    ) -> Result<Self, eredu_nn::Error> {
        Self::unavailable()
    }

    fn conv_transpose1d(
        _: &Self,
        _: &Self,
        _: i32,
        _: i32,
        _: i32,
        _: i32,
        _: i32,
        _: &Self::Context,
    ) -> Result<Self, eredu_nn::Error> {
        Self::unavailable()
    }

    fn linear(
        _: &Self,
        _: &Self,
        _: Option<&Self>,
        _: &Self::Context,
    ) -> Result<Self, eredu_nn::Error> {
        Self::unavailable()
    }

    fn layer_norm(
        _: &Self,
        _: Option<&Self>,
        _: Option<&Self>,
        _: f32,
        _: &Self::Context,
    ) -> Result<Self, eredu_nn::Error> {
        Self::unavailable()
    }

    fn gelu(_: &Self, _: &Self::Context) -> Result<Self, eredu_nn::Error> {
        Self::unavailable()
    }

    fn elu(_: &Self, _: f32, _: &Self::Context) -> Result<Self, eredu_nn::Error> {
        Self::unavailable()
    }

    fn rope(
        _: &Self,
        _: i32,
        _: bool,
        _: f32,
        _: f32,
        _: i32,
        _: &Self::Context,
    ) -> Result<Self, eredu_nn::Error> {
        Self::unavailable()
    }

    fn scaled_dot_product_attention(
        _: &Self,
        _: &Self,
        _: &Self,
        _: f32,
        _: AttentionMask<'_, Self>,
        _: &Self::Context,
    ) -> Result<Self, eredu_nn::Error> {
        Self::unavailable()
    }
}

fn parameter_topology(config: Config) -> Result<BTreeMap<String, Vec<usize>>, MimiArtifactError> {
    let mimi = Mimi::<PlanningTensor>::new(config, &())?;
    let mut topology = BTreeMap::new();
    let mut duplicate = None;
    mimi.visit_mimi_parameters("", &mut |metadata, parameter| {
        let shape = parameter
            .shape()
            .iter()
            .map(|dimension| {
                usize::try_from(*dimension).map_err(|_| {
                    MimiArtifactError::Topology(format!(
                        "parameter {:?} has invalid shape {:?}",
                        metadata.id,
                        parameter.shape()
                    ))
                })
            })
            .collect::<Result<Vec<_>, _>>();
        match shape {
            Ok(shape) => {
                if topology.insert(metadata.id.to_string(), shape).is_some() {
                    duplicate = Some(metadata.id.to_string());
                }
            }
            Err(error) => duplicate = Some(error.to_string()),
        }
    });
    if let Some(duplicate) = duplicate {
        return Err(MimiArtifactError::Topology(format!(
            "duplicate or invalid parameter identity {duplicate:?}"
        )));
    }
    Ok(topology)
}

fn checkpoint_layout_axes(parameter: &str) -> Option<[usize; 3]> {
    (parameter.ends_with(".weight") && is_conv_weight_key(parameter)).then(|| {
        if parameter.contains(".upsample.") {
            [1, 2, 0]
        } else {
            [0, 2, 1]
        }
    })
}

fn inverse_permuted_shape(
    logical_shape: &[usize],
    axes: [usize; 3],
) -> Result<Vec<usize>, MimiArtifactError> {
    if logical_shape.len() != axes.len() {
        return Err(MimiArtifactError::Topology(format!(
            "rank-{} parameter cannot use transpose axes {axes:?}",
            logical_shape.len()
        )));
    }
    let mut physical = vec![0; axes.len()];
    for (logical_axis, physical_axis) in axes.into_iter().enumerate() {
        physical[physical_axis] = logical_shape[logical_axis];
    }
    Ok(physical)
}

fn f32_bytes(shape: &[usize], name: &str) -> Result<u64, MimiArtifactError> {
    shape.iter().try_fold(4u64, |bytes, dimension| {
        bytes.checked_mul(*dimension as u64).ok_or_else(|| {
            MimiArtifactError::Topology(format!("byte count overflows for {name:?}"))
        })
    })
}

fn checkpoint_key_for_parameter(parameter: &str) -> Option<String> {
    if parameter.starts_with("quantizer.") {
        return Some(parameter.to_owned());
    }
    if parameter == "downsample.weight" {
        return Some("downsample.conv.conv.conv.weight".into());
    }
    if let Some(key) = parameter.strip_prefix("encoder_transformer.") {
        let key = key
            .replace(".self_attn.in_proj.weight", ".self_attn.in_proj_weight")
            .replace(".mlp.linear1.", ".linear1.")
            .replace(".mlp.linear2.", ".linear2.");
        return Some(format!("encoder_transformer.transformer.{key}"));
    }
    if let Some(key) = reverse_seanet_encoder_key(parameter) {
        return Some(format!("encoder.model.{key}"));
    }
    if parameter == "upsample.weight" {
        return Some("upsample.convtr.convtr.convtr.weight".into());
    }
    if let Some(key) = parameter.strip_prefix("decoder_transformer.") {
        let key = key
            .replace(".self_attn.in_proj.weight", ".self_attn.in_proj_weight")
            .replace(".mlp.linear1.", ".linear1.")
            .replace(".mlp.linear2.", ".linear2.");
        return Some(format!("decoder_transformer.transformer.{key}"));
    }
    reverse_seanet_decoder_key(parameter).map(|key| format!("decoder.model.{key}"))
}

const SEANET_ENCODER_KEY_MAPPINGS: &[(&str, &str)] = &[
    ("0.conv.conv.", "encoder.init_conv1d."),
    (
        "1.block.1.conv.conv.",
        "encoder.layers.0.residuals.0.block.0.",
    ),
    (
        "1.block.3.conv.conv.",
        "encoder.layers.0.residuals.0.block.1.",
    ),
    ("3.conv.conv.", "encoder.layers.0.downsample."),
    (
        "4.block.1.conv.conv.",
        "encoder.layers.1.residuals.0.block.0.",
    ),
    (
        "4.block.3.conv.conv.",
        "encoder.layers.1.residuals.0.block.1.",
    ),
    ("6.conv.conv.", "encoder.layers.1.downsample."),
    (
        "7.block.1.conv.conv.",
        "encoder.layers.2.residuals.0.block.0.",
    ),
    (
        "7.block.3.conv.conv.",
        "encoder.layers.2.residuals.0.block.1.",
    ),
    ("9.conv.conv.", "encoder.layers.2.downsample."),
    (
        "10.block.1.conv.conv.",
        "encoder.layers.3.residuals.0.block.0.",
    ),
    (
        "10.block.3.conv.conv.",
        "encoder.layers.3.residuals.0.block.1.",
    ),
    ("12.conv.conv.", "encoder.layers.3.downsample."),
    ("14.conv.conv.", "encoder.final_conv1d."),
];

const SEANET_DECODER_KEY_MAPPINGS: &[(&str, &str)] = &[
    ("0.conv.conv.", "decoder.init_conv1d."),
    ("2.convtr.convtr.", "decoder.layers.0.upsample."),
    (
        "3.block.1.conv.conv.",
        "decoder.layers.0.residuals.0.block.0.",
    ),
    (
        "3.block.3.conv.conv.",
        "decoder.layers.0.residuals.0.block.1.",
    ),
    ("5.convtr.convtr.", "decoder.layers.1.upsample."),
    (
        "6.block.1.conv.conv.",
        "decoder.layers.1.residuals.0.block.0.",
    ),
    (
        "6.block.3.conv.conv.",
        "decoder.layers.1.residuals.0.block.1.",
    ),
    ("8.convtr.convtr.", "decoder.layers.2.upsample."),
    (
        "9.block.1.conv.conv.",
        "decoder.layers.2.residuals.0.block.0.",
    ),
    (
        "9.block.3.conv.conv.",
        "decoder.layers.2.residuals.0.block.1.",
    ),
    ("11.convtr.convtr.", "decoder.layers.3.upsample."),
    (
        "12.block.1.conv.conv.",
        "decoder.layers.3.residuals.0.block.0.",
    ),
    (
        "12.block.3.conv.conv.",
        "decoder.layers.3.residuals.0.block.1.",
    ),
    ("14.conv.conv.", "decoder.final_conv1d."),
];

fn reverse_seanet_key(parameter: &str, mappings: &[(&str, &str)]) -> Option<String> {
    let &(source, target) = mappings
        .iter()
        .find(|(_, target)| parameter.starts_with(target))?;
    Some(format!("{source}{}", &parameter[target.len()..]))
}

fn reverse_seanet_encoder_key(parameter: &str) -> Option<String> {
    reverse_seanet_key(parameter, SEANET_ENCODER_KEY_MAPPINGS)
}

fn reverse_seanet_decoder_key(parameter: &str) -> Option<String> {
    reverse_seanet_key(parameter, SEANET_DECODER_KEY_MAPPINGS)
}

fn transform_decoder_key(key: &str) -> Option<String> {
    if key.starts_with("quantizer.") {
        return Some(key.to_string());
    }
    if key == "downsample.conv.conv.conv.weight" {
        return Some("downsample.weight".to_string());
    }
    if let Some(key) = key.strip_prefix("encoder_transformer.transformer.") {
        let key = key
            .replace(".self_attn.in_proj_weight", ".self_attn.in_proj.weight")
            .replace(".linear1.", ".mlp.linear1.")
            .replace(".linear2.", ".mlp.linear2.");
        return Some(format!("encoder_transformer.{key}"));
    }
    if let Some(key) = key.strip_prefix("encoder.model.") {
        return transform_seanet_encoder_key(key);
    }
    if key == "upsample.convtr.convtr.convtr.weight" {
        return Some("upsample.weight".to_string());
    }
    if let Some(key) = key.strip_prefix("decoder_transformer.transformer.") {
        let key = key
            .replace(".self_attn.in_proj_weight", ".self_attn.in_proj.weight")
            .replace(".linear1.", ".mlp.linear1.")
            .replace(".linear2.", ".mlp.linear2.");
        return Some(format!("decoder_transformer.{key}"));
    }
    if let Some(key) = key.strip_prefix("decoder.model.") {
        return transform_seanet_decoder_key(key);
    }
    None
}

fn transform_seanet_encoder_key(key: &str) -> Option<String> {
    transform_seanet_key(key, SEANET_ENCODER_KEY_MAPPINGS)
}

fn transform_seanet_decoder_key(key: &str) -> Option<String> {
    transform_seanet_key(key, SEANET_DECODER_KEY_MAPPINGS)
}

fn transform_seanet_key(key: &str, mappings: &[(&str, &str)]) -> Option<String> {
    let &(source, target) = mappings
        .iter()
        .find(|(source, _)| key.starts_with(source))?;
    Some(format!("{target}{}", &key[source.len()..]))
}

fn is_conv_weight_key(key: &str) -> bool {
    key.starts_with("upsample.")
        || key.starts_with("downsample.")
        || key.contains(".upsample.")
        || key.contains(".downsample.")
        || key.contains(".init_conv1d.")
        || key.contains(".final_conv1d.")
        || key.contains(".block.")
}

impl<T: Tensor> AudioTokenizer for Mimi<T> {
    type Tensor = T;

    fn config(&self) -> AudioTokenizerConfig {
        AudioTokenizerConfig {
            sample_rate: self.config.sample_rate,
            frame_rate: self.config.frame_rate,
            channels: self.config.channels,
            codebooks: self.config.num_codebooks,
            cardinality: self.config.bins,
        }
    }

    fn encode(&mut self, pcm: &T, context: &T::Context) -> Result<T, Error> {
        self.encode(pcm, context)
    }

    fn decode(&mut self, codes: &T, context: &T::Context) -> Result<T, Error> {
        self.decode(codes, context)
    }
}

#[derive(Debug, Clone)]
struct SeaNetEncoder<T: Tensor> {
    init_conv1d: StreamableConv1d<T>,
    layers: Vec<EncoderLayer<T>>,
    final_conv1d: StreamableConv1d<T>,
}

impl<T: Tensor> SeaNetEncoder<T> {
    fn unloaded(context: &T::Context) -> Result<Self, Error> {
        let ratios = [4, 5, 6, 8];
        let mut channels = 64;
        let mut layers = Vec::with_capacity(ratios.len());
        for ratio in ratios {
            layers.push(EncoderLayer::unloaded(
                channels,
                channels * 2,
                ratio,
                context,
            )?);
            channels *= 2;
        }
        Ok(Self {
            init_conv1d: StreamableConv1d::unloaded(1, 64, 7, 1, context)?,
            layers,
            final_conv1d: StreamableConv1d::unloaded(1024, 512, 3, 1, context)?,
        })
    }

    fn forward(&mut self, pcm: &T, context: &T::Context) -> Result<T, Error> {
        validate_pcm(pcm)?;
        let mut x = self.init_conv1d.forward(pcm, context)?;
        for layer in &mut self.layers {
            x = layer.forward(&x, context)?;
        }
        self.final_conv1d
            .forward(&T::elu(&x, 1.0, context)?, context)
    }

    fn reset_state(&mut self) {
        self.init_conv1d.reset_state();
        for layer in &mut self.layers {
            layer.reset_state();
        }
        self.final_conv1d.reset_state();
    }

    fn step(&mut self, pcm: &T, context: &T::Context) -> Result<Option<T>, Error> {
        validate_pcm(pcm)?;
        let mut x = match self.init_conv1d.step(pcm, context)? {
            Some(x) => x,
            None => return Ok(None),
        };
        for layer in &mut self.layers {
            x = match layer.step(&x, context)? {
                Some(x) => x,
                None => return Ok(None),
            };
        }
        self.final_conv1d.step(&T::elu(&x, 1.0, context)?, context)
    }
}

#[derive(Debug, Clone)]
struct EncoderLayer<T: Tensor> {
    residuals: Vec<SeaNetResnetBlock<T>>,
    downsample: StreamableConv1d<T>,
}

impl<T: Tensor> EncoderLayer<T> {
    fn unloaded(
        in_channels: i32,
        out_channels: i32,
        ratio: i32,
        context: &T::Context,
    ) -> Result<Self, Error> {
        Ok(Self {
            residuals: vec![SeaNetResnetBlock::unloaded(in_channels, context)?],
            downsample: StreamableConv1d::unloaded(
                in_channels,
                out_channels,
                ratio * 2,
                ratio,
                context,
            )?,
        })
    }

    fn forward(&mut self, x: &T, context: &T::Context) -> Result<T, Error> {
        let mut x = x.clone();
        for residual in &mut self.residuals {
            x = residual.forward(&x, context)?;
        }
        self.downsample.forward(&T::elu(&x, 1.0, context)?, context)
    }

    fn reset_state(&mut self) {
        for residual in &mut self.residuals {
            residual.reset_state();
        }
        self.downsample.reset_state();
    }

    fn step(&mut self, x: &T, context: &T::Context) -> Result<Option<T>, Error> {
        let mut x = x.clone();
        for residual in &mut self.residuals {
            x = residual.step(&x, context)?;
        }
        self.downsample.step(&T::elu(&x, 1.0, context)?, context)
    }
}

#[derive(Debug, Clone)]
struct MimiTransformer<T: Tensor> {
    layers: Vec<MimiTransformerLayer<T>>,
}

impl<T: Tensor> MimiTransformer<T> {
    fn unloaded(context: &T::Context) -> Result<Self, Error> {
        Ok(Self {
            layers: (0..8)
                .map(|_| MimiTransformerLayer::unloaded(context))
                .collect::<Result<Vec<_>, _>>()?,
        })
    }

    fn forward(&mut self, latent: &T, context: &T::Context) -> Result<T, Error> {
        let mut x = latent.swap_axes(1, 2, context)?;
        for layer in &mut self.layers {
            x = layer.forward(&x, context)?;
        }
        Ok(x.swap_axes(1, 2, context)?)
    }

    fn reset_state(&mut self) {
        for layer in &mut self.layers {
            layer.reset_state();
        }
    }

    fn step(&mut self, latent: &T, context: &T::Context) -> Result<T, Error> {
        let mut x = latent.swap_axes(1, 2, context)?;
        for layer in &mut self.layers {
            x = layer.step(&x, context)?;
        }
        Ok(x.swap_axes(1, 2, context)?)
    }
}

#[derive(Debug, Clone)]
struct MimiTransformerLayer<T: Tensor> {
    norm1: LayerNorm<T>,
    norm2: LayerNorm<T>,
    self_attn: MimiSelfAttention<T>,
    mlp: MimiMlp<T>,
    layer_scale_1: LayerScale<T>,
    layer_scale_2: LayerScale<T>,
}

impl<T: Tensor> MimiTransformerLayer<T> {
    fn unloaded(context: &T::Context) -> Result<Self, Error> {
        Ok(Self {
            norm1: unloaded_layer_norm(512, 1e-5, context)?,
            norm2: unloaded_layer_norm(512, 1e-5, context)?,
            self_attn: MimiSelfAttention::unloaded(context)?,
            mlp: MimiMlp::unloaded(context)?,
            layer_scale_1: LayerScale::unloaded(512, context)?,
            layer_scale_2: LayerScale::unloaded(512, context)?,
        })
    }

    fn forward(&mut self, x: &T, context: &T::Context) -> Result<T, Error> {
        let normed = self.norm1.forward(x, context)?;
        let attended = self
            .self_attn
            .forward(&normed, context)?
            .multiply(self.layer_scale_1.scale.as_ref(), context)?;
        let x = x.add(&attended, context)?;
        let normed = self.norm2.forward(&x, context)?;
        let mlp = self
            .mlp
            .forward(&normed, context)?
            .multiply(self.layer_scale_2.scale.as_ref(), context)?;
        Ok(x.add(&mlp, context)?)
    }

    fn reset_state(&mut self) {
        self.self_attn.reset_state();
    }

    fn step(&mut self, x: &T, context: &T::Context) -> Result<T, Error> {
        let normed = self.norm1.forward(x, context)?;
        let attended = self
            .self_attn
            .step(&normed, context)?
            .multiply(self.layer_scale_1.scale.as_ref(), context)?;
        let x = x.add(&attended, context)?;
        let normed = self.norm2.forward(&x, context)?;
        let mlp = self
            .mlp
            .forward(&normed, context)?
            .multiply(self.layer_scale_2.scale.as_ref(), context)?;
        Ok(x.add(&mlp, context)?)
    }
}

#[derive(Debug, Clone)]
struct LayerScale<T: Tensor> {
    scale: Parameter<T>,
}

impl<T: Tensor> LayerScale<T> {
    fn unloaded(dim: i32, context: &T::Context) -> Result<Self, Error> {
        Ok(Self {
            scale: unloaded_parameter(&[dim], context)?,
        })
    }
}

#[derive(Debug, Clone)]
struct MimiMlp<T: Tensor> {
    linear1: Linear<T>,
    linear2: Linear<T>,
}

impl<T: Tensor> MimiMlp<T> {
    fn unloaded(context: &T::Context) -> Result<Self, Error> {
        Ok(Self {
            linear1: unloaded_linear(512, 2048, false, context)?,
            linear2: unloaded_linear(2048, 512, false, context)?,
        })
    }

    fn forward(&mut self, x: &T, context: &T::Context) -> Result<T, Error> {
        let x = self.linear1.forward(x, context)?;
        let x = T::gelu(&x, context)?;
        Ok(self.linear2.forward(&x, context)?)
    }
}

#[derive(Debug, Clone)]
struct MimiSelfAttention<T: Tensor> {
    in_proj: Linear<T>,
    out_proj: Linear<T>,
    rope: Rope,
    num_heads: i32,
    head_dim: i32,
    scale: f32,
    context: i32,
    key_cache: Option<T>,
    value_cache: Option<T>,
}

impl<T: Tensor> MimiSelfAttention<T> {
    fn unloaded(context: &T::Context) -> Result<Self, Error> {
        let head_dim = 64;
        Ok(Self {
            in_proj: unloaded_linear(512, 1536, false, context)?,
            out_proj: unloaded_linear(512, 512, false, context)?,
            rope: Rope::new(head_dim, true, 10_000.0, 1.0),
            num_heads: 8,
            head_dim,
            scale: (head_dim as f32).sqrt().recip(),
            context: 250,
            key_cache: None,
            value_cache: None,
        })
    }

    fn forward(&mut self, x: &T, context: &T::Context) -> Result<T, Error> {
        let shape = x.shape();
        if shape.len() != 3 || shape[2] != 512 {
            return Err(Error::InvalidShape(format!(
                "Mimi decoder transformer expects [batch, frames, 512], got {:?}",
                x.shape()
            )));
        }
        let (batch, seq, dim) = (shape[0], shape[1], shape[2]);
        let qkv = self
            .in_proj
            .forward(x, context)?
            .reshape(&[batch, seq, 3, self.num_heads, self.head_dim], context)?;
        let mut q = qkv
            .index(
                &[
                    Index::Full,
                    Index::Full,
                    Index::At(0),
                    Index::Full,
                    Index::Full,
                ],
                context,
            )?
            .transpose_axes(&[0, 2, 1, 3], context)?;
        let mut k = qkv
            .index(
                &[
                    Index::Full,
                    Index::Full,
                    Index::At(1),
                    Index::Full,
                    Index::Full,
                ],
                context,
            )?
            .transpose_axes(&[0, 2, 1, 3], context)?;
        let v = qkv
            .index(
                &[
                    Index::Full,
                    Index::Full,
                    Index::At(2),
                    Index::Full,
                    Index::Full,
                ],
                context,
            )?
            .transpose_axes(&[0, 2, 1, 3], context)?;
        q = self.rope.forward(&q, 0, context)?;
        k = self.rope.forward(&k, 0, context)?;
        let attended = T::scaled_dot_product_attention(
            &q,
            &k,
            &v,
            self.scale,
            AttentionMask::Causal,
            context,
        )?
        .transpose_axes(&[0, 2, 1, 3], context)?
        .reshape(&[batch, seq, dim], context)?;
        Ok(self.out_proj.forward(&attended, context)?)
    }

    fn reset_state(&mut self) {
        self.key_cache = None;
        self.value_cache = None;
    }

    fn step(&mut self, x: &T, context: &T::Context) -> Result<T, Error> {
        let shape = x.shape();
        if shape.len() != 3 || shape[2] != 512 {
            return Err(Error::InvalidShape(format!(
                "Mimi decoder transformer step expects [batch, frames, 512], got {:?}",
                x.shape()
            )));
        }
        let (batch, seq, dim) = (shape[0], shape[1], shape[2]);
        let prev_len = self
            .key_cache
            .as_ref()
            .map(|cache| cache.dim(2))
            .unwrap_or(0);
        let qkv = self
            .in_proj
            .forward(x, context)?
            .reshape(&[batch, seq, 3, self.num_heads, self.head_dim], context)?;
        let mut q = qkv
            .index(
                &[
                    Index::Full,
                    Index::Full,
                    Index::At(0),
                    Index::Full,
                    Index::Full,
                ],
                context,
            )?
            .transpose_axes(&[0, 2, 1, 3], context)?;
        let mut k = qkv
            .index(
                &[
                    Index::Full,
                    Index::Full,
                    Index::At(1),
                    Index::Full,
                    Index::Full,
                ],
                context,
            )?
            .transpose_axes(&[0, 2, 1, 3], context)?;
        let v = qkv
            .index(
                &[
                    Index::Full,
                    Index::Full,
                    Index::At(2),
                    Index::Full,
                    Index::Full,
                ],
                context,
            )?
            .transpose_axes(&[0, 2, 1, 3], context)?;
        q = self.rope.forward(&q, prev_len, context)?;
        k = self.rope.forward(&k, prev_len, context)?;

        let mut keys = match self.key_cache.take() {
            Some(prev) => T::concatenate(&[prev, k], 2, context)?,
            None => k,
        };
        let mut values = match self.value_cache.take() {
            Some(prev) => T::concatenate(&[prev, v], 2, context)?,
            None => v,
        };
        let key_len = keys.dim(2);
        if key_len > self.context + seq {
            let start = key_len - (self.context + seq);
            keys = keys.index(
                &[
                    Index::Full,
                    Index::Full,
                    Index::Range(start, key_len),
                    Index::Full,
                ],
                context,
            )?;
            values = values.index(
                &[
                    Index::Full,
                    Index::Full,
                    Index::Range(start, key_len),
                    Index::Full,
                ],
                context,
            )?;
        }
        let retained_prev_len = keys.dim(2) - seq;
        let mask =
            streaming_attention_mask::<T>(batch, seq, retained_prev_len, self.context, context)?;
        let attended = T::scaled_dot_product_attention(
            &q,
            &keys,
            &values,
            self.scale,
            AttentionMask::Tensor(&mask),
            context,
        )?
        .transpose_axes(&[0, 2, 1, 3], context)?
        .reshape(&[batch, seq, dim], context)?;
        self.key_cache = Some(keys);
        self.value_cache = Some(values);
        Ok(self.out_proj.forward(&attended, context)?)
    }
}

fn streaming_attention_mask<T: Tensor>(
    batch: i32,
    query_len: i32,
    prev_len: i32,
    attention_context: i32,
    execution: &T::Context,
) -> Result<T, Error> {
    let key_len = prev_len + query_len;
    let mut mask = Vec::with_capacity((batch * query_len * key_len) as usize);
    for _ in 0..batch {
        for q in 0..query_len {
            let q_pos = prev_len + q;
            for k in 0..key_len {
                if k <= q_pos && q_pos <= k + attention_context {
                    mask.push(0.0f32);
                } else {
                    mask.push(f32::NEG_INFINITY);
                }
            }
        }
    }
    Ok(T::from_f32_slice(
        &mask,
        &[batch, 1, query_len, key_len],
        execution,
    )?)
}

#[derive(Debug, Clone)]
struct SeaNetDecoder<T: Tensor> {
    init_conv1d: StreamableConv1d<T>,
    layers: Vec<DecoderLayer<T>>,
    final_conv1d: StreamableConv1d<T>,
}

impl<T: Tensor> SeaNetDecoder<T> {
    fn unloaded(context: &T::Context) -> Result<Self, Error> {
        let ratios = [8, 6, 5, 4];
        let mut channels = 1024;
        let mut layers = Vec::with_capacity(ratios.len());
        for ratio in ratios {
            let out_channels = channels / 2;
            layers.push(DecoderLayer::unloaded(
                channels,
                out_channels,
                ratio,
                context,
            )?);
            channels = out_channels;
        }
        Ok(Self {
            init_conv1d: StreamableConv1d::unloaded(512, 1024, 7, 1, context)?,
            layers,
            final_conv1d: StreamableConv1d::unloaded(64, 1, 3, 1, context)?,
        })
    }

    fn forward(&mut self, latent: &T, context: &T::Context) -> Result<T, Error> {
        let mut x = self.init_conv1d.forward(latent, context)?;
        for layer in &mut self.layers {
            x = layer.forward(&T::elu(&x, 1.0, context)?, context)?;
        }
        self.final_conv1d
            .forward(&T::elu(&x, 1.0, context)?, context)
    }

    fn reset_state(&mut self) {
        self.init_conv1d.reset_state();
        for layer in &mut self.layers {
            layer.reset_state();
        }
        self.final_conv1d.reset_state();
    }

    fn step(&mut self, latent: &T, context: &T::Context) -> Result<T, Error> {
        let mut x = self.init_conv1d.step(latent, context)?.ok_or_else(|| {
            Error::InvalidShape("Mimi decoder init conv produced no streaming output".into())
        })?;
        for layer in &mut self.layers {
            x = layer.step(&T::elu(&x, 1.0, context)?, context)?;
        }
        self.final_conv1d
            .step(&T::elu(&x, 1.0, context)?, context)?
            .ok_or_else(|| Error::InvalidShape("Mimi decoder final conv produced no output".into()))
    }
}

#[derive(Debug, Clone)]
struct DecoderLayer<T: Tensor> {
    upsample: StreamableConvTranspose1d<T>,
    residuals: Vec<SeaNetResnetBlock<T>>,
}

impl<T: Tensor> DecoderLayer<T> {
    fn unloaded(
        in_channels: i32,
        out_channels: i32,
        ratio: i32,
        context: &T::Context,
    ) -> Result<Self, Error> {
        Ok(Self {
            upsample: StreamableConvTranspose1d::unloaded(
                in_channels,
                out_channels,
                ratio * 2,
                ratio,
                1,
                true,
                context,
            )?,
            residuals: vec![SeaNetResnetBlock::unloaded(out_channels, context)?],
        })
    }

    fn forward(&mut self, x: &T, context: &T::Context) -> Result<T, Error> {
        let mut x = self.upsample.forward(x, context)?;
        for residual in &mut self.residuals {
            x = residual.forward(&x, context)?;
        }
        Ok(x)
    }

    fn reset_state(&mut self) {
        self.upsample.reset_state();
        for residual in &mut self.residuals {
            residual.reset_state();
        }
    }

    fn step(&mut self, x: &T, context: &T::Context) -> Result<T, Error> {
        let mut x = self.upsample.step(x, context)?;
        for residual in &mut self.residuals {
            x = residual.step(&x, context)?;
        }
        Ok(x)
    }
}

#[derive(Debug, Clone)]
struct SeaNetResnetBlock<T: Tensor> {
    block: Vec<StreamableConv1d<T>>,
}

impl<T: Tensor> SeaNetResnetBlock<T> {
    fn unloaded(channels: i32, context: &T::Context) -> Result<Self, Error> {
        Ok(Self {
            block: vec![
                StreamableConv1d::unloaded(channels, channels / 2, 3, 1, context)?,
                StreamableConv1d::unloaded(channels / 2, channels, 1, 1, context)?,
            ],
        })
    }

    fn forward(&mut self, x: &T, context: &T::Context) -> Result<T, Error> {
        let mut y = x.clone();
        for conv in &mut self.block {
            y = conv.forward(&T::elu(&y, 1.0, context)?, context)?;
        }
        Ok(y.add(x, context)?)
    }

    fn reset_state(&mut self) {
        for conv in &mut self.block {
            conv.reset_state();
        }
    }

    fn step(&mut self, x: &T, context: &T::Context) -> Result<T, Error> {
        let mut y = x.clone();
        for conv in &mut self.block {
            y = conv
                .step(&T::elu(&y, 1.0, context)?, context)?
                .ok_or_else(|| {
                    Error::InvalidShape("Mimi residual conv produced no output".into())
                })?;
        }
        Ok(y.add(x, context)?)
    }
}

#[derive(Debug, Clone)]
struct StreamableConv1d<T: Tensor> {
    weight: Parameter<T>,
    bias: Option<Parameter<T>>,
    stride: i32,
    dilation: i32,
    groups: i32,
    pad_mode: PadMode,
    state_prev_xs: Option<T>,
    left_pad_applied: bool,
}

impl<T: Tensor> StreamableConv1d<T> {
    fn unloaded(
        in_channels: i32,
        out_channels: i32,
        kernel_size: i32,
        stride: i32,
        context: &T::Context,
    ) -> Result<Self, Error> {
        Self::unloaded_with_pad_mode(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            true,
            PadMode::Constant,
            context,
        )
    }

    fn unloaded_with_pad_mode(
        in_channels: i32,
        out_channels: i32,
        kernel_size: i32,
        stride: i32,
        bias: bool,
        pad_mode: PadMode,
        context: &T::Context,
    ) -> Result<Self, Error> {
        Ok(Self {
            weight: unloaded_parameter(&[out_channels, kernel_size, in_channels], context)?,
            bias: bias
                .then(|| unloaded_parameter(&[out_channels], context))
                .transpose()?,
            stride,
            dilation: 1,
            groups: 1,
            pad_mode,
            state_prev_xs: None,
            left_pad_applied: false,
        })
    }

    fn forward(&mut self, x: &T, context: &T::Context) -> Result<T, Error> {
        let kernel_size = self.weight.as_ref().dim(1);
        let effective_kernel = (kernel_size - 1) * self.dilation + 1;
        let padding_total = effective_kernel - self.stride;
        let extra_padding =
            extra_padding_for_conv1d(x.dim(2), effective_kernel, self.stride, padding_total);
        let x = pad_bct(x, padding_total, extra_padding, self.pad_mode, context)?;
        let x = x.swap_axes(1, 2, context)?;
        let mut y = T::conv1d(
            &x,
            self.weight.as_ref(),
            self.stride,
            0,
            self.dilation,
            self.groups,
            context,
        )?;
        if let Some(bias) = &self.bias {
            y = y.add(bias.as_ref(), context)?;
        }
        Ok(y.swap_axes(1, 2, context)?)
    }

    fn reset_state(&mut self) {
        self.state_prev_xs = None;
        self.left_pad_applied = false;
    }

    fn step(&mut self, x: &T, context: &T::Context) -> Result<Option<T>, Error> {
        let kernel_size = self.weight.as_ref().dim(1);
        let effective_kernel = (kernel_size - 1) * self.dilation + 1;
        let padding_total = effective_kernel - self.stride;
        let x = if self.left_pad_applied {
            x.clone()
        } else {
            self.left_pad_applied = true;
            pad_bct(x, padding_total, 0, self.pad_mode, context)?
        };
        let x = match self.state_prev_xs.take() {
            Some(prev) => T::concatenate(&[prev, x], 2, context)?,
            None => x,
        };
        let seq_len = x.dim(2);
        let num_frames = (seq_len + self.stride).saturating_sub(effective_kernel) / self.stride;
        if num_frames <= 0 {
            self.state_prev_xs = Some(x);
            return Ok(None);
        }
        let offset = num_frames * self.stride;
        self.state_prev_xs = Some(x.index(
            &[Index::Full, Index::Full, Index::Range(offset, seq_len)],
            context,
        )?);
        let in_len = (num_frames - 1) * self.stride + effective_kernel;
        let x = x.index(
            &[Index::Full, Index::Full, Index::Range(0, in_len)],
            context,
        )?;
        self.forward_unpadded(&x, context).map(Some)
    }

    fn forward_unpadded(&mut self, x: &T, context: &T::Context) -> Result<T, Error> {
        let x = x.swap_axes(1, 2, context)?;
        let mut y = T::conv1d(
            &x,
            self.weight.as_ref(),
            self.stride,
            0,
            self.dilation,
            self.groups,
            context,
        )?;
        if let Some(bias) = &self.bias {
            y = y.add(bias.as_ref(), context)?;
        }
        Ok(y.swap_axes(1, 2, context)?)
    }
}

#[derive(Debug, Clone)]
struct StreamableConvTranspose1d<T: Tensor> {
    weight: Parameter<T>,
    bias: Option<Parameter<T>>,
    kernel_size: i32,
    stride: i32,
    groups: i32,
    state_prev_ys: Option<T>,
}

impl<T: Tensor> StreamableConvTranspose1d<T> {
    fn unloaded(
        in_channels: i32,
        out_channels: i32,
        kernel_size: i32,
        stride: i32,
        groups: i32,
        bias: bool,
        context: &T::Context,
    ) -> Result<Self, Error> {
        Ok(Self {
            weight: unloaded_parameter(
                &[out_channels, kernel_size, in_channels / groups],
                context,
            )?,
            bias: bias
                .then(|| unloaded_parameter(&[out_channels], context))
                .transpose()?,
            kernel_size,
            stride,
            groups,
            state_prev_ys: None,
        })
    }

    fn forward(&mut self, x: &T, context: &T::Context) -> Result<T, Error> {
        let y = self.forward_untrimmed(x, context)?;
        let padding_total = self.kernel_size.saturating_sub(self.stride);
        unpad_bct(&y, 0, padding_total, context)
    }

    fn reset_state(&mut self) {
        self.state_prev_ys = None;
    }

    fn step(&mut self, x: &T, context: &T::Context) -> Result<T, Error> {
        let y = self.forward_untrimmed(x, context)?;
        let out_len = y.dim(2);
        let y = match self.state_prev_ys.take() {
            None => y,
            Some(prev) => {
                let prev_len = prev.dim(2);
                let prev = match &self.bias {
                    None => prev,
                    Some(bias) => prev.subtract(
                        &bias
                            .as_ref()
                            .reshape(&[1, bias.as_ref().dim(0), 1], context)?,
                        context,
                    )?,
                };
                let y1 = y
                    .index(
                        &[Index::Full, Index::Full, Index::Range(0, prev_len)],
                        context,
                    )?
                    .add(&prev, context)?;
                let y2 = y.index(
                    &[Index::Full, Index::Full, Index::Range(prev_len, out_len)],
                    context,
                )?;
                T::concatenate(&[y1, y2], 2, context)?
            }
        };
        let invalid_steps = self.kernel_size - self.stride;
        let split = out_len - invalid_steps;
        let out = y.index(&[Index::Full, Index::Full, Index::Range(0, split)], context)?;
        self.state_prev_ys = Some(y.index(
            &[Index::Full, Index::Full, Index::Range(split, out_len)],
            context,
        )?);
        Ok(out)
    }

    fn forward_untrimmed(&mut self, x: &T, context: &T::Context) -> Result<T, Error> {
        let x = x.swap_axes(1, 2, context)?;
        let mut y = T::conv_transpose1d(
            &x,
            self.weight.as_ref(),
            self.stride,
            0,
            1,
            0,
            self.groups,
            context,
        )?;
        if let Some(bias) = &self.bias {
            y = y.add(bias.as_ref(), context)?;
        }
        Ok(y.swap_axes(1, 2, context)?)
    }
}

fn extra_padding_for_conv1d(len: i32, kernel_size: i32, stride: i32, padding_total: i32) -> i32 {
    let n_frames = (len + padding_total - kernel_size) as f64 / stride as f64 + 1.0;
    let ideal_len = ((n_frames.ceil() as i32 - 1) * stride + kernel_size) - padding_total;
    ideal_len.saturating_sub(len)
}

fn pad_bct<T: Tensor>(
    x: &T,
    left: i32,
    right: i32,
    mode: PadMode,
    context: &T::Context,
) -> Result<T, Error> {
    Ok(T::pad(x, &[(0, 0), (0, 0), (left, right)], mode, context)?)
}

fn unpad_bct<T: Tensor>(x: &T, left: i32, right: i32, context: &T::Context) -> Result<T, Error> {
    let len = x.dim(2);
    if len < left + right {
        return Err(Error::InvalidShape(format!(
            "cannot unpad Mimi tensor of length {len} by {left}+{right}"
        )));
    }
    Ok(x.index(
        &[Index::Full, Index::Full, Index::Range(left, len - right)],
        context,
    )?)
}

/// Split residual vector quantizer used by Mimi.
#[derive(Debug, Clone)]
pub struct SplitResidualVectorQuantizer<T: Tensor> {
    /// First semantic codebook branch.
    pub rvq_first: ResidualVectorQuantizer<T>,
    /// Remaining acoustic codebook branch.
    pub rvq_rest: ResidualVectorQuantizer<T>,
    n_q: i32,
}

impl<T: Tensor> SplitResidualVectorQuantizer<T> {
    fn unloaded(config: &Config, context: &T::Context) -> Result<Self, Error> {
        Ok(Self {
            rvq_first: ResidualVectorQuantizer::unloaded(
                config.latent_dim,
                config.quantizer_dim,
                1,
                config.bins,
                context,
            )?,
            rvq_rest: ResidualVectorQuantizer::unloaded(
                config.latent_dim,
                config.quantizer_dim,
                config.num_codebooks - 1,
                config.bins,
                context,
            )?,
            n_q: config.num_codebooks,
        })
    }

    /// Encodes latent frames shaped `[batch, 512, frames]`.
    pub fn encode(&mut self, latent: &T, context: &T::Context) -> Result<T, Error> {
        validate_latent(latent)?;
        let first = self.rvq_first.encode(latent, context)?;
        if self.n_q == 1 {
            Ok(first)
        } else {
            let rest = self.rvq_rest.encode(latent, context)?;
            Ok(T::concatenate(&[first, rest], 1, context)?)
        }
    }

    /// Decodes tokens shaped `[batch, codebooks, frames]`.
    pub fn decode(&mut self, codes: &T, context: &T::Context) -> Result<T, Error> {
        validate_codes(codes, self.n_q)?;
        let first_codes = codes.index(&[Index::Full, Index::Range(0, 1), Index::Full], context)?;
        let mut quantized = self.rvq_first.decode(&first_codes, context)?;
        if codes.dim(1) > 1 {
            let rest_codes = codes.index(
                &[Index::Full, Index::Range(1, codes.dim(1)), Index::Full],
                context,
            )?;
            quantized = quantized.add(&self.rvq_rest.decode(&rest_codes, context)?, context)?;
        }
        Ok(quantized)
    }
}

/// Residual vector quantizer branch.
#[derive(Debug, Clone)]
pub struct ResidualVectorQuantizer<T: Tensor> {
    /// Input projection from Mimi latent dimension into codebook dimension.
    pub input_proj: Conv1x1NoBias<T>,
    /// Output projection from codebook dimension back into Mimi latent dimension.
    pub output_proj: Conv1x1NoBias<T>,
    /// Residual codebook layers.
    pub vq: ResidualVectorQuantization<T>,
}

impl<T: Tensor> ResidualVectorQuantizer<T> {
    fn unloaded(
        latent_dim: i32,
        codebook_dim: i32,
        layers: i32,
        bins: i32,
        context: &T::Context,
    ) -> Result<Self, Error> {
        Ok(Self {
            input_proj: Conv1x1NoBias::unloaded(latent_dim, codebook_dim, context)?,
            output_proj: Conv1x1NoBias::unloaded(codebook_dim, latent_dim, context)?,
            vq: ResidualVectorQuantization::unloaded(layers, codebook_dim, bins, context)?,
        })
    }

    fn encode(&mut self, latent: &T, context: &T::Context) -> Result<T, Error> {
        self.vq
            .encode(&self.input_proj.forward(latent, context)?, context)
    }

    fn decode(&mut self, codes: &T, context: &T::Context) -> Result<T, Error> {
        self.output_proj
            .forward(&self.vq.decode(codes, context)?, context)
    }
}

/// Residual vector quantization layers.
#[derive(Debug, Clone)]
pub struct ResidualVectorQuantization<T: Tensor> {
    /// Ordered residual quantization layers.
    pub layers: Vec<VectorQuantization<T>>,
}

impl<T: Tensor> ResidualVectorQuantization<T> {
    fn unloaded(layers: i32, dim: i32, bins: i32, context: &T::Context) -> Result<Self, Error> {
        Ok(Self {
            layers: (0..layers)
                .map(|_| VectorQuantization::unloaded(dim, bins, context))
                .collect::<Result<Vec<_>, _>>()?,
        })
    }

    fn encode(&mut self, latent: &T, context: &T::Context) -> Result<T, Error> {
        if self.layers.is_empty() {
            return Err(Error::InvalidShape("Mimi RVQ has no layers".into()));
        }
        let mut residual = latent.clone();
        let mut codes = Vec::with_capacity(self.layers.len());
        for layer in &mut self.layers {
            let indices = layer.encode(&residual, context)?;
            let quantized = layer.decode_one(&indices, context)?;
            residual = residual.subtract(&quantized, context)?;
            codes.push(indices);
        }
        Ok(T::stack(&codes, 1, context)?)
    }

    fn decode(&mut self, codes: &T, context: &T::Context) -> Result<T, Error> {
        if codes.dim(1) != self.layers.len() as i32 {
            return Err(Error::InvalidShape(format!(
                "Mimi RVQ expected {} codebooks, got {:?}",
                self.layers.len(),
                codes.shape()
            )));
        }
        let mut out: Option<T> = None;
        for (index, layer) in self.layers.iter_mut().enumerate() {
            let code = codes.index(
                &[Index::Full, Index::At(index as i32), Index::Full],
                context,
            )?;
            let quantized = layer.decode_one(&code, context)?;
            out = Some(match out {
                None => quantized,
                Some(prev) => prev.add(&quantized, context)?,
            });
        }
        out.ok_or_else(|| Error::InvalidShape("Mimi RVQ has no layers".into()))
    }
}

/// Single vector-quantization layer.
#[derive(Debug, Clone)]
pub struct VectorQuantization<T: Tensor> {
    /// Euclidean codebook.
    pub _codebook: EuclideanCodebook<T>,
}

impl<T: Tensor> VectorQuantization<T> {
    fn unloaded(dim: i32, bins: i32, context: &T::Context) -> Result<Self, Error> {
        Ok(Self {
            _codebook: EuclideanCodebook::unloaded(dim, bins, context)?,
        })
    }

    fn encode(&mut self, latent: &T, context: &T::Context) -> Result<T, Error> {
        let latent = latent.swap_axes(1, 2, context)?;
        self._codebook.encode(&latent, context)
    }

    fn decode_one(&mut self, codes: &T, context: &T::Context) -> Result<T, Error> {
        self._codebook
            .decode(codes, context)?
            .swap_axes(1, 2, context)
            .map_err(Into::into)
    }
}

/// Euclidean codebook backed by EMA cluster statistics.
#[derive(Debug, Clone)]
pub struct EuclideanCodebook<T: Tensor> {
    /// Checkpoint initialization flag.
    pub _initialized: Parameter<T>,
    /// EMA cluster usage.
    pub cluster_usage: Parameter<T>,
    /// EMA embedding sum.
    pub embedding_sum: Parameter<T>,
}

impl<T: Tensor> EuclideanCodebook<T> {
    fn unloaded(dim: i32, bins: i32, context: &T::Context) -> Result<Self, Error> {
        Ok(Self {
            _initialized: unloaded_parameter(&[1], context)?,
            cluster_usage: unloaded_parameter(&[bins], context)?,
            embedding_sum: unloaded_parameter(&[bins, dim], context)?,
        })
    }

    fn embedding(&self, context: &T::Context) -> Result<T, Error> {
        let usage = self
            .cluster_usage
            .as_ref()
            .maximum_scalar(EPSILON, context)?
            .expand_dims(1, context)?;
        Ok(self.embedding_sum.as_ref().divide(&usage, context)?)
    }

    fn encode(&self, latent_btd: &T, context: &T::Context) -> Result<T, Error> {
        if latent_btd.shape().len() != 3 {
            return Err(Error::InvalidShape(format!(
                "Mimi codebook encode expects [batch, frames, dim], got {:?}",
                latent_btd.shape()
            )));
        }
        let batch = latent_btd.dim(0);
        let frames = latent_btd.dim(1);
        let dim = latent_btd.dim(2);
        let flat = latent_btd.reshape(&[batch * frames, dim], context)?;
        let embedding = self.embedding(context)?;
        let x2 = T::sum_axis(&flat.square(context)?, -1, true, context)?;
        let e2 = T::sum_axis(&embedding.square(context)?, -1, false, context)?
            .expand_dims(0, context)?;
        let dot = T::matmul(&flat, &embedding.transpose(context)?, context)?;
        let dists = x2
            .add(&e2, context)?
            .subtract(&dot.multiply_scalar(2.0, context)?, context)?;
        Ok(T::argmin_axis(&dists, -1, false, context)?.reshape(&[batch, frames], context)?)
    }

    fn decode(&self, codes: &T, context: &T::Context) -> Result<T, Error> {
        if codes.shape().len() != 2 {
            return Err(Error::InvalidShape(format!(
                "Mimi codebook decode expects [batch, frames], got {:?}",
                codes.shape()
            )));
        }
        let batch = codes.dim(0);
        let frames = codes.dim(1);
        let embedding = self.embedding(context)?;
        let flat = codes.reshape(&[batch * frames], context)?;
        Ok(embedding
            .take_axis(&flat, 0, context)?
            .reshape(&[batch, frames, embedding.dim(1)], context)?)
    }
}

/// Bias-free 1x1 convolution over `[batch, channels, frames]` tensors.
#[derive(Debug, Clone)]
pub struct Conv1x1NoBias<T: Tensor> {
    /// Weight shaped `[out_channels, in_channels, 1]`.
    pub weight: Parameter<T>,
}

impl<T: Tensor> Conv1x1NoBias<T> {
    fn unloaded(in_channels: i32, out_channels: i32, context: &T::Context) -> Result<Self, Error> {
        Ok(Self {
            weight: unloaded_parameter(&[out_channels, in_channels, 1], context)?,
        })
    }

    fn forward(&self, latent: &T, context: &T::Context) -> Result<T, Error> {
        if latent.shape().len() != 3 {
            return Err(Error::InvalidShape(format!(
                "Mimi 1x1 projection expects [batch, channels, frames], got {:?}",
                latent.shape()
            )));
        }
        let x = latent.swap_axes(1, 2, context)?;
        let weight = self.weight.as_ref().squeeze_axes(&[-1], context)?;
        Ok(T::matmul(&x, &weight.transpose(context)?, context)?.swap_axes(1, 2, context)?)
    }
}

trait MimiModuleParameters<T: Tensor> {
    fn visit_mimi_parameters<'a>(
        &'a self,
        prefix: &str,
        visitor: &mut dyn FnMut(ParameterMetadata, &'a T),
    );
    fn visit_mimi_parameters_mut<'a>(
        &'a mut self,
        prefix: &str,
        visitor: &mut dyn FnMut(ParameterMetadata, &'a mut T),
    );
    fn set_mimi_trainable(&mut self, trainable: bool);
}

struct PrefixVisitor<'a, F: ?Sized> {
    prefix: &'a str,
    visitor: &'a mut F,
    exact: bool,
}

impl<'a, 'value, T, F: ?Sized> ParameterVisitor<'value, T> for PrefixVisitor<'a, F>
where
    T: 'value,
    F: FnMut(ParameterMetadata, &'value T),
{
    fn visit(&mut self, mut metadata: ParameterMetadata, value: &'value T) {
        let id = if self.exact {
            self.prefix.to_owned()
        } else {
            parameter_name(self.prefix, metadata.id.as_str())
        };
        metadata.id = ParameterId::new(id).expect("Mimi parameter identities are non-empty");
        (self.visitor)(metadata, value);
    }
}

struct PrefixVisitorMut<'a, F: ?Sized> {
    prefix: &'a str,
    visitor: &'a mut F,
    exact: bool,
}

impl<'a, 'value, T, F: ?Sized> ParameterVisitorMut<'value, T> for PrefixVisitorMut<'a, F>
where
    T: 'value,
    F: FnMut(ParameterMetadata, &'value mut T),
{
    fn visit_mut(&mut self, mut metadata: ParameterMetadata, value: &'value mut T) {
        let id = if self.exact {
            self.prefix.to_owned()
        } else {
            parameter_name(self.prefix, metadata.id.as_str())
        };
        metadata.id = ParameterId::new(id).expect("Mimi parameter identities are non-empty");
        (self.visitor)(metadata, value);
    }
}

impl<T: Tensor> MimiModuleParameters<T> for Parameter<T> {
    fn visit_mimi_parameters<'a>(
        &'a self,
        prefix: &str,
        visitor: &mut dyn FnMut(ParameterMetadata, &'a T),
    ) {
        self.visit_parameters(&mut PrefixVisitor {
            prefix,
            visitor,
            exact: true,
        });
    }

    fn visit_mimi_parameters_mut<'a>(
        &'a mut self,
        prefix: &str,
        visitor: &mut dyn FnMut(ParameterMetadata, &'a mut T),
    ) {
        self.visit_parameters_mut(&mut PrefixVisitorMut {
            prefix,
            visitor,
            exact: true,
        });
    }

    fn set_mimi_trainable(&mut self, trainable: bool) {
        self.set_trainable(trainable);
    }
}

macro_rules! structured_leaf_parameters {
    ($type:ty) => {
        impl<T: Tensor> MimiModuleParameters<T> for $type {
            fn visit_mimi_parameters<'a>(
                &'a self,
                prefix: &str,
                visitor: &mut dyn FnMut(ParameterMetadata, &'a T),
            ) {
                self.visit_parameters(&mut PrefixVisitor {
                    prefix,
                    visitor,
                    exact: false,
                });
            }

            fn visit_mimi_parameters_mut<'a>(
                &'a mut self,
                prefix: &str,
                visitor: &mut dyn FnMut(ParameterMetadata, &'a mut T),
            ) {
                self.visit_parameters_mut(&mut PrefixVisitorMut {
                    prefix,
                    visitor,
                    exact: false,
                });
            }

            fn set_mimi_trainable(&mut self, trainable: bool) {
                self.set_trainable(trainable);
            }
        }
    };
}

structured_leaf_parameters!(Linear<T>);
structured_leaf_parameters!(LayerNorm<T>);

impl<T: Tensor, M: MimiModuleParameters<T>> MimiModuleParameters<T> for Vec<M> {
    fn visit_mimi_parameters<'a>(
        &'a self,
        prefix: &str,
        visitor: &mut dyn FnMut(ParameterMetadata, &'a T),
    ) {
        for (index, module) in self.iter().enumerate() {
            module.visit_mimi_parameters(&parameter_name(prefix, &index.to_string()), visitor);
        }
    }

    fn visit_mimi_parameters_mut<'a>(
        &'a mut self,
        prefix: &str,
        visitor: &mut dyn FnMut(ParameterMetadata, &'a mut T),
    ) {
        for (index, module) in self.iter_mut().enumerate() {
            module.visit_mimi_parameters_mut(&parameter_name(prefix, &index.to_string()), visitor);
        }
    }

    fn set_mimi_trainable(&mut self, trainable: bool) {
        for module in self {
            module.set_mimi_trainable(trainable);
        }
    }
}

impl<T: Tensor, M: MimiModuleParameters<T>> MimiModuleParameters<T> for Option<M> {
    fn visit_mimi_parameters<'a>(
        &'a self,
        prefix: &str,
        visitor: &mut dyn FnMut(ParameterMetadata, &'a T),
    ) {
        if let Some(module) = self {
            module.visit_mimi_parameters(prefix, visitor);
        }
    }

    fn visit_mimi_parameters_mut<'a>(
        &'a mut self,
        prefix: &str,
        visitor: &mut dyn FnMut(ParameterMetadata, &'a mut T),
    ) {
        if let Some(module) = self {
            module.visit_mimi_parameters_mut(prefix, visitor);
        }
    }

    fn set_mimi_trainable(&mut self, trainable: bool) {
        if let Some(module) = self {
            module.set_mimi_trainable(trainable);
        }
    }
}

macro_rules! module_parameters {
    ($module:ident { $($field:ident),+ $(,)? }) => {
        impl<T: Tensor> MimiModuleParameters<T> for $module<T> {
            fn visit_mimi_parameters<'a>(
                &'a self,
                prefix: &str,
                visitor: &mut dyn FnMut(ParameterMetadata, &'a T),
            ) {
                $(
                    self.$field.visit_mimi_parameters(
                        &parameter_name(prefix, stringify!($field)),
                        visitor,
                    );
                )+
            }

            fn visit_mimi_parameters_mut<'a>(
                &'a mut self,
                prefix: &str,
                visitor: &mut dyn FnMut(ParameterMetadata, &'a mut T),
            ) {
                $(
                    self.$field.visit_mimi_parameters_mut(
                        &parameter_name(prefix, stringify!($field)),
                        visitor,
                    );
                )+
            }

            fn set_mimi_trainable(&mut self, trainable: bool) {
                $(self.$field.set_mimi_trainable(trainable);)+
            }
        }
    };
}

module_parameters!(Mimi {
    quantizer,
    encoder,
    encoder_transformer,
    downsample,
    upsample,
    decoder_transformer,
    decoder,
});
module_parameters!(SeaNetEncoder {
    init_conv1d,
    layers,
    final_conv1d,
});
module_parameters!(EncoderLayer {
    residuals,
    downsample,
});
module_parameters!(MimiTransformer { layers });
module_parameters!(MimiTransformerLayer {
    norm1,
    norm2,
    self_attn,
    mlp,
    layer_scale_1,
    layer_scale_2,
});
module_parameters!(LayerScale { scale });
module_parameters!(MimiMlp { linear1, linear2 });
module_parameters!(MimiSelfAttention { in_proj, out_proj });
module_parameters!(SeaNetDecoder {
    init_conv1d,
    layers,
    final_conv1d,
});
module_parameters!(DecoderLayer {
    upsample,
    residuals,
});
module_parameters!(SeaNetResnetBlock { block });
module_parameters!(StreamableConv1d { weight, bias });
module_parameters!(StreamableConvTranspose1d { weight, bias });
module_parameters!(SplitResidualVectorQuantizer {
    rvq_first,
    rvq_rest,
});
module_parameters!(ResidualVectorQuantizer {
    input_proj,
    output_proj,
    vq,
});
module_parameters!(ResidualVectorQuantization { layers });
module_parameters!(VectorQuantization { _codebook });
module_parameters!(EuclideanCodebook {
    _initialized,
    cluster_usage,
    embedding_sum,
});
module_parameters!(Conv1x1NoBias { weight });

impl<T: Tensor> Parameterized<T> for Mimi<T> {
    fn visit_parameters<'a, V>(&'a self, visitor: &mut V)
    where
        V: ParameterVisitor<'a, T>,
    {
        self.visit_mimi_parameters("", &mut |metadata, value| {
            visitor.visit(metadata, value);
        });
    }

    fn visit_parameters_mut<'a, V>(&'a mut self, visitor: &mut V)
    where
        V: ParameterVisitorMut<'a, T>,
    {
        self.visit_mimi_parameters_mut("", &mut |metadata, value| {
            visitor.visit_mut(metadata, value);
        });
    }

    fn set_trainable(&mut self, trainable: bool) {
        self.set_mimi_trainable(trainable);
    }
}

fn validate_latent<T: Tensor>(latent: &T) -> Result<(), Error> {
    if latent.shape().len() != 3 || latent.dim(1) != 512 {
        return Err(Error::InvalidShape(format!(
            "Mimi latent frames must have shape [batch, 512, frames], got {:?}",
            latent.shape()
        )));
    }
    Ok(())
}

fn validate_pcm<T: Tensor>(pcm: &T) -> Result<(), Error> {
    if pcm.shape().len() != 3 || pcm.dim(1) != 1 {
        return Err(Error::InvalidShape(format!(
            "Mimi PCM must have shape [batch, 1, samples], got {:?}",
            pcm.shape()
        )));
    }
    Ok(())
}

fn validate_codes<T: Tensor>(codes: &T, max_codebooks: i32) -> Result<(), Error> {
    if codes.shape().len() != 3 || codes.dim(1) <= 0 || codes.dim(1) > max_codebooks {
        return Err(Error::InvalidShape(format!(
            "Mimi codes must have shape [batch, 1..={max_codebooks}, frames], got {:?}",
            codes.shape()
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::{
        collections::BTreeMap,
        sync::{
            atomic::{AtomicUsize, Ordering},
            Arc,
        },
    };

    use super::{
        checkpoint_key_for_parameter, checkpoint_layout_axes, checkpoint_parameter_for_key,
        parameter_topology, prepare_catalog, prepare_source, released_checkpoint_requirements,
        Config, Mimi, MimiArtifactError, MimiParameterRequirement, RecipeDtype,
    };
    use eredu_checkpoint::store::{
        CheckpointLease, CheckpointSource, TensorMetadata, TensorReadRequest,
        TensorSourceProvenance, WeightStoreBackend, WeightStoreDiagnostics,
    };
    use eredu_checkpoint::{SourceTensorEncoding, StoredDtype};

    struct MetadataSource {
        tensors: BTreeMap<String, TensorMetadata>,
        payload_reads: AtomicUsize,
        encoding: SourceTensorEncoding,
    }

    impl MetadataSource {
        fn exact(requirements: &[MimiParameterRequirement]) -> Self {
            Self {
                tensors: requirements
                    .iter()
                    .map(|requirement| {
                        (
                            requirement.checkpoint_key().to_owned(),
                            TensorMetadata {
                                name: requirement.checkpoint_key().to_owned(),
                                logical_shape: requirement.physical_shape().to_vec(),
                                physical_shape: requirement.physical_shape().to_vec(),
                                stored_dtype: requirement.source_dtype().clone(),
                                encoded_byte_len: requirement.source_bytes(),
                                backing_shard: None,
                            },
                        )
                    })
                    .collect(),
                payload_reads: AtomicUsize::new(0),
                encoding: SourceTensorEncoding::Safetensors(StoredDtype::F32),
            }
        }
    }

    impl CheckpointSource for MetadataSource {
        fn source_keys(&self) -> Vec<String> {
            self.tensors.keys().cloned().collect()
        }

        fn source_metadata(
            &self,
            key: &str,
        ) -> Result<TensorMetadata, eredu_checkpoint::store::StoreError> {
            self.tensors.get(key).cloned().ok_or_else(|| {
                eredu_checkpoint::store::StoreError::UnknownTensor { key: key.into() }
            })
        }

        fn acquire_lease(
            &self,
            _: TensorReadRequest,
        ) -> Result<CheckpointLease, eredu_checkpoint::store::StoreError> {
            self.payload_reads.fetch_add(1, Ordering::Relaxed);
            Err(eredu_checkpoint::store::StoreError::Internal(
                "metadata-only test source cannot read payloads".into(),
            ))
        }

        fn source_diagnostics(
            &self,
        ) -> Result<WeightStoreDiagnostics, eredu_checkpoint::store::StoreError> {
            Ok(WeightStoreDiagnostics {
                backend: WeightStoreBackend::Memory,
                cache_hits: 0,
                cache_misses: 0,
                evictions: 0,
                currently_cached_shards: 0,
                touched_shard_paths: Vec::new(),
                payload_shard_paths: Vec::new(),
                physical_reads: self.payload_reads.load(Ordering::Relaxed) as u64,
                physical_read_bytes: 0,
                coalesced_group_hits: 0,
            })
        }

        fn source_provenance(
            &self,
            key: &str,
        ) -> Result<TensorSourceProvenance, eredu_checkpoint::store::StoreError> {
            let metadata = self.source_metadata(key)?;
            Ok(TensorSourceProvenance {
                catalog_key: key.to_owned(),
                physical_tensor: key.to_owned(),
                output: key.to_owned(),
                backing_shard: metadata.backing_shard,
                source_encoding: self.encoding.clone(),
            })
        }
    }

    fn exact_source() -> (Arc<MetadataSource>, Vec<MimiParameterRequirement>) {
        let (_, requirements, _) = prepare_catalog(&Config::v0_1(Some(8))).unwrap();
        (Arc::new(MetadataSource::exact(&requirements)), requirements)
    }

    #[test]
    fn checkpoint_quantizer_keys_keep_the_model_root() {
        let key = "quantizer.rvq_first.vq.layers.0._codebook.embedding_sum";
        assert_eq!(checkpoint_parameter_for_key(key).as_deref(), Some(key));
        assert_eq!(checkpoint_key_for_parameter(key).as_deref(), Some(key));
        assert_eq!(checkpoint_layout_axes(key), None);
    }

    #[test]
    fn checkpoint_plan_declares_canonical_convolution_layouts() {
        assert_eq!(
            checkpoint_layout_axes("encoder.init_conv1d.weight"),
            Some([0, 2, 1])
        );
        assert_eq!(
            checkpoint_layout_axes("decoder.layers.0.upsample.weight"),
            Some([1, 2, 0])
        );
        assert_eq!(checkpoint_layout_axes("upsample.weight"), Some([0, 2, 1]));
        assert!(checkpoint_parameter_for_key("optimizer.state").is_none());
    }

    #[test]
    fn parameter_names_are_unique_and_cover_checkpoint_mapping() {
        let active = parameter_topology(Config::v0_1(Some(8))).unwrap();
        let full = parameter_topology(Config::v0_1(Some(32))).unwrap();
        assert_eq!(active.len(), 246);
        assert_eq!(full.len(), 318);
        for model_name in full.keys() {
            let checkpoint_name = checkpoint_key_for_parameter(model_name)
                .unwrap_or_else(|| panic!("model parameter was not mapped: {model_name}"));
            assert_eq!(
                checkpoint_parameter_for_key(&checkpoint_name).as_deref(),
                Some(model_name.as_str()),
                "checkpoint mapping did not round-trip"
            );
        }
    }

    #[test]
    fn exact_catalog_preparation_validates_total_topology_without_payload_reads() {
        for active in 1..=32 {
            let (_, requirements) = exact_source();
            let source = Arc::new(MetadataSource::exact(&requirements));
            let prepared = prepare_source(source.clone(), Config::v0_1(Some(active))).unwrap();
            assert_eq!(prepared.requirements().len(), 318);
            assert_eq!(prepared.bindings().len(), 3 * active as usize + 222);
            assert_eq!(
                prepared
                    .requirements()
                    .iter()
                    .filter(|requirement| requirement.is_active())
                    .count(),
                3 * active as usize + 222
            );
            assert!(prepared.requirements().iter().all(|requirement| {
                requirement.source_dtype() == &StoredDtype::F32
                    && requirement.output_dtype() == &RecipeDtype::F32
                    && requirement.source_encoding()
                        == &SourceTensorEncoding::Safetensors(StoredDtype::F32)
            }));
            assert_eq!(
                prepared
                    .requirements()
                    .iter()
                    .filter(|requirement| matches!(
                        requirement.recipe(),
                        eredu_checkpoint::recipe::DerivedWeightRecipe::Transpose { .. }
                    ))
                    .count(),
                30
            );
            assert_eq!(
                prepared
                    .requirements()
                    .iter()
                    .filter(|requirement| requirement.recipe().source_keys().len() == 1)
                    .count(),
                318
            );
            assert_eq!(source.payload_reads.load(Ordering::Relaxed), 0);
        }
    }

    #[test]
    fn corrupt_catalogs_fail_before_payload_reads() {
        let (source, requirements) = exact_source();
        let missing_key = requirements[0].checkpoint_key().to_owned();
        let mut missing = MetadataSource::exact(&requirements);
        missing.tensors.remove(&missing_key);
        let missing = Arc::new(missing);
        assert!(matches!(
            prepare_source(missing.clone(), Config::v0_1(Some(8))),
            Err(MimiArtifactError::Catalog(_))
        ));
        assert_eq!(missing.payload_reads.load(Ordering::Relaxed), 0);

        let mut extra = MetadataSource::exact(&requirements);
        extra.tensors.insert(
            "optimizer.state".into(),
            TensorMetadata {
                name: "optimizer.state".into(),
                logical_shape: vec![1],
                physical_shape: vec![1],
                stored_dtype: StoredDtype::F32,
                encoded_byte_len: 4,
                backing_shard: None,
            },
        );
        let extra = Arc::new(extra);
        assert!(matches!(
            prepare_source(extra.clone(), Config::v0_1(Some(8))),
            Err(MimiArtifactError::Catalog(_))
        ));
        assert_eq!(extra.payload_reads.load(Ordering::Relaxed), 0);

        let corrupt = requirements
            .iter()
            .find(|requirement| requirement.physical_shape().len() == 3)
            .unwrap();
        let mut wrong_shape = MetadataSource::exact(&requirements);
        wrong_shape
            .tensors
            .get_mut(corrupt.checkpoint_key())
            .unwrap()
            .logical_shape[0] += 1;
        let wrong_shape = Arc::new(wrong_shape);
        assert!(matches!(
            prepare_source(wrong_shape.clone(), Config::v0_1(Some(8))),
            Err(MimiArtifactError::Catalog(_))
        ));
        assert_eq!(wrong_shape.payload_reads.load(Ordering::Relaxed), 0);

        let mut wrong_dtype = MetadataSource::exact(&requirements);
        wrong_dtype
            .tensors
            .get_mut(corrupt.checkpoint_key())
            .unwrap()
            .stored_dtype = StoredDtype::F16;
        let wrong_dtype = Arc::new(wrong_dtype);
        assert!(matches!(
            prepare_source(wrong_dtype.clone(), Config::v0_1(Some(8))),
            Err(MimiArtifactError::Catalog(_))
        ));
        assert_eq!(wrong_dtype.payload_reads.load(Ordering::Relaxed), 0);

        let mut wrong_bytes = MetadataSource::exact(&requirements);
        wrong_bytes
            .tensors
            .get_mut(corrupt.checkpoint_key())
            .unwrap()
            .encoded_byte_len -= 4;
        let wrong_bytes = Arc::new(wrong_bytes);
        assert!(matches!(
            prepare_source(wrong_bytes.clone(), Config::v0_1(Some(8))),
            Err(MimiArtifactError::Topology(_))
        ));
        assert_eq!(wrong_bytes.payload_reads.load(Ordering::Relaxed), 0);

        let mut wrong_encoding = MetadataSource::exact(&requirements);
        wrong_encoding.encoding = SourceTensorEncoding::RecipeOutput(StoredDtype::F32);
        let wrong_encoding = Arc::new(wrong_encoding);
        assert!(matches!(
            prepare_source(wrong_encoding.clone(), Config::v0_1(Some(8))),
            Err(MimiArtifactError::Topology(_))
        ));
        assert_eq!(wrong_encoding.payload_reads.load(Ordering::Relaxed), 0);
        assert_eq!(source.payload_reads.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn v0_1_config_defaults_to_moshi_active_codebooks() {
        let cfg = Config::v0_1(None);
        assert_eq!(cfg.sample_rate, 24_000.0);
        assert_eq!(cfg.frame_rate, 12.5);
        assert_eq!(cfg.num_codebooks, 16);
        assert_eq!(cfg.total_codebooks, 32);
        assert_eq!(cfg.bins, 2_048);
    }

    #[test]
    fn unsupported_or_non_finite_profiles_fail_before_tensor_construction() {
        let mut invalid = Config::v0_1(Some(8));
        invalid.sample_rate = f64::NAN;
        assert!(Mimi::<super::PlanningTensor>::new(invalid, &()).is_err());

        let mut unsupported = Config::v0_1(Some(8));
        unsupported.total_codebooks = 16;
        assert!(Mimi::<super::PlanningTensor>::new(unsupported, &()).is_err());

        for active in [-1, 0, 33] {
            assert!(released_checkpoint_requirements(&Config::v0_1(Some(active))).is_err());
        }
    }
}
