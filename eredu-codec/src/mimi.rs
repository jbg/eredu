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
    AttentionMask, Index, LayerNorm, Linear, NeuralBackend, Parameter, Parameterized, Rope, Tensor,
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

use crate::{
    streaming::{StreamableConv1d, StreamableConvTranspose1d},
    AudioTokenizer, AudioTokenizerConfig, Error,
};

const EPSILON: f32 = 1e-5;

/// Shared backend-free Mimi construction and logical parameter declarations.
#[path = "mimi/specs.rs"]
pub mod specs;
use specs::{DeclareCheckpointParameters, MimiSpec};

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
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "T")]
pub struct Mimi<T: Tensor> {
    /// Split residual vector quantizer.
    pub quantizer: SplitResidualVectorQuantizer<T>,
    encoder: SeaNetEncoder<T>,
    encoder_transformer: MimiTransformer<T>,
    downsample: StreamableConv1d<T>,
    upsample: StreamableConvTranspose1d<T>,
    decoder_transformer: MimiTransformer<T>,
    decoder: SeaNetDecoder<T>,
    #[parameter(skip)]
    config: Config,
}

impl<T: Tensor> Mimi<T> {
    /// Creates an unloaded Mimi tokenizer from config.
    pub fn new(config: Config, context: &T::Context) -> Result<Self, Error> {
        Self::from_spec(MimiSpec::new(config)?, context)
    }
    fn from_spec(spec: MimiSpec, context: &T::Context) -> Result<Self, Error> {
        Ok(Self {
            quantizer: SplitResidualVectorQuantizer::unloaded(spec.quantizer, context)?,
            encoder: SeaNetEncoder::unloaded(spec.encoder, context)?,
            encoder_transformer: MimiTransformer::unloaded(spec.encoder_transformer, context)?,
            downsample: spec.downsample.construct(context)?,
            upsample: spec.upsample.construct(context)?,
            decoder_transformer: MimiTransformer::unloaded(spec.decoder_transformer, context)?,
            decoder: SeaNetDecoder::unloaded(spec.decoder, context)?,
            config: spec.config,
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
    declaration: MimiSpec,
    source: SharedCheckpointSource,
    requirements: Vec<MimiParameterRequirement>,
    bindings: Vec<WeightBinding>,
}

impl std::fmt::Debug for PreparedMimiArtifact {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PreparedMimiArtifact")
            .field("config", &self.declaration.config)
            .field("requirements", &self.requirements)
            .field("bindings", &self.bindings)
            .finish_non_exhaustive()
    }
}

impl PreparedMimiArtifact {
    /// Released configuration fixed by neutral preparation.
    pub const fn config(&self) -> &Config {
        &self.declaration.config
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
            declaration,
            source,
            requirements,
            bindings,
        } = self;
        let selected = select_bindings::<B>(source, bindings)?;
        Ok(SelectedMimiArtifact {
            declaration,
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
    declaration: MimiSpec,
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
            .field("config", &self.declaration.config)
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
            declaration,
            requirements: _,
            selected,
        } = self;
        let mut mimi = Mimi::from_spec(declaration, tensor_context)?;
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
    let (declaration, plan, requirements, bindings) = prepare_catalog(&config)?;
    let catalog = SafetensorsMetadataCatalog::discover(path)?;
    let resolution =
        resolve_safetensors_plan(&catalog, &plan).map_err(MimiArtifactError::Catalog)?;
    let store = Arc::new(SafetensorsWeightStore::open_admitted(
        catalog.into_admitted_shards(),
        1,
    )?);
    prepared_from_resolution(declaration, store, resolution, requirements, bindings)
}

/// Returns the complete released-checkpoint schema derived from Mimi's
/// authoritative parameter topology without inspecting an artifact.
pub fn released_checkpoint_requirements(
    config: &Config,
) -> Result<Vec<MimiParameterRequirement>, MimiArtifactError> {
    prepare_catalog(config).map(|(_, _, requirements, _)| requirements)
}

/// Admits an already opened backend-neutral SafeTensors-compatible source.
/// No tensor payload is acquired during preparation.
pub fn prepare_source(
    source: SharedCheckpointSource,
    config: Config,
) -> Result<PreparedMimiArtifact, MimiArtifactError> {
    let (declaration, plan, requirements, bindings) = prepare_catalog(&config)?;
    let resolution =
        resolve_safetensors_plan(source.as_ref(), &plan).map_err(MimiArtifactError::Catalog)?;
    prepared_from_resolution(declaration, source, resolution, requirements, bindings)
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
        MimiSpec,
        SafetensorsCheckpointPlan,
        Vec<MimiParameterRequirement>,
        Vec<WeightBinding>,
    ),
    MimiArtifactError,
> {
    config.validate()?;
    let mut full_config = config.clone();
    full_config.num_codebooks = full_config.total_codebooks;
    let full_declaration = MimiSpec::new(full_config)?;
    let declaration = full_declaration.select_codebooks(config.num_codebooks)?;
    let active_topology = parameter_topology(&declaration)?;
    let full_topology = full_declaration
        .checkpoint_parameters()
        .map_err(Error::from)?
        .into_iter()
        .map(|parameter| {
            (
                parameter.logical.parameter.id.to_string(),
                (parameter.logical.shape, parameter.axes),
            )
        })
        .collect::<BTreeMap<_, _>>();
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
    for (logical_name, (logical_shape, axes)) in full_topology {
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
    Ok((declaration, plan, requirements, bindings))
}

fn prepared_from_resolution(
    declaration: MimiSpec,
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
        declaration,
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

fn parameter_topology(
    declaration: &MimiSpec,
) -> Result<BTreeMap<String, Vec<usize>>, MimiArtifactError> {
    use eredu_nn::DeclareLogicalParameters;
    Ok(declaration
        .logical_parameters()
        .map_err(Error::from)?
        .into_iter()
        .map(|p| (p.parameter.id.to_string(), p.shape))
        .collect())
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

#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "T")]
struct SeaNetEncoder<T: Tensor> {
    init_conv1d: StreamableConv1d<T>,
    layers: Vec<EncoderLayer<T>>,
    final_conv1d: StreamableConv1d<T>,
}

impl<T: Tensor> SeaNetEncoder<T> {
    fn unloaded(spec: specs::Encoder, context: &T::Context) -> Result<Self, Error> {
        Ok(Self {
            init_conv1d: spec.init.construct(context)?,
            layers: spec
                .layers
                .into_iter()
                .map(|s| EncoderLayer::unloaded(s, context))
                .collect::<Result<_, _>>()?,
            final_conv1d: spec.final_conv.construct(context)?,
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

#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "T")]
struct EncoderLayer<T: Tensor> {
    residuals: Vec<SeaNetResnetBlock<T>>,
    downsample: StreamableConv1d<T>,
}

impl<T: Tensor> EncoderLayer<T> {
    fn unloaded(spec: specs::EncoderLayer, context: &T::Context) -> Result<Self, Error> {
        Ok(Self {
            residuals: spec
                .residuals
                .into_iter()
                .map(|s| SeaNetResnetBlock::unloaded(s, context))
                .collect::<Result<_, _>>()?,
            downsample: spec.downsample.construct(context)?,
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

#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "T")]
struct MimiTransformer<T: Tensor> {
    layers: Vec<MimiTransformerLayer<T>>,
}

impl<T: Tensor> MimiTransformer<T> {
    fn unloaded(spec: specs::Transformer, context: &T::Context) -> Result<Self, Error> {
        Ok(Self {
            layers: spec
                .layers
                .into_iter()
                .map(|s| MimiTransformerLayer::unloaded(s, context))
                .collect::<Result<_, _>>()?,
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

#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "T")]
struct MimiTransformerLayer<T: Tensor> {
    norm1: LayerNorm<T>,
    norm2: LayerNorm<T>,
    self_attn: MimiSelfAttention<T>,
    mlp: MimiMlp<T>,
    layer_scale_1: LayerScale<T>,
    layer_scale_2: LayerScale<T>,
}

impl<T: Tensor> MimiTransformerLayer<T> {
    fn unloaded(spec: specs::TransformerLayer, context: &T::Context) -> Result<Self, Error> {
        Ok(Self {
            norm1: LayerNorm::new(spec.norm1, context)?,
            norm2: LayerNorm::new(spec.norm2, context)?,
            self_attn: MimiSelfAttention::unloaded(spec.attention, context)?,
            mlp: MimiMlp::unloaded(spec.mlp, context)?,
            layer_scale_1: LayerScale::unloaded(spec.scale1, context)?,
            layer_scale_2: LayerScale::unloaded(spec.scale2, context)?,
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

#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "T")]
struct LayerScale<T: Tensor> {
    scale: Parameter<T>,
}

impl<T: Tensor> LayerScale<T> {
    fn unloaded(spec: specs::Slot, context: &T::Context) -> Result<Self, Error> {
        Ok(Self {
            scale: spec.construct(context)?,
        })
    }
}

#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "T")]
struct MimiMlp<T: Tensor> {
    linear1: Linear<T>,
    linear2: Linear<T>,
}

impl<T: Tensor> MimiMlp<T> {
    fn unloaded(spec: specs::Mlp, context: &T::Context) -> Result<Self, Error> {
        Ok(Self {
            linear1: Linear::unloaded(spec.up, context)?,
            linear2: Linear::unloaded(spec.down, context)?,
        })
    }

    fn forward(&mut self, x: &T, context: &T::Context) -> Result<T, Error> {
        let x = self.linear1.forward(x, context)?;
        let x = T::gelu(&x, context)?;
        Ok(self.linear2.forward(&x, context)?)
    }
}

#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "T")]
struct MimiSelfAttention<T: Tensor> {
    in_proj: Linear<T>,
    out_proj: Linear<T>,
    #[parameter(skip)]
    rope: Rope,
    #[parameter(skip)]
    num_heads: i32,
    #[parameter(skip)]
    head_dim: i32,
    #[parameter(skip)]
    scale: f32,
    #[parameter(skip)]
    context: i32,
    #[parameter(skip)]
    key_cache: Option<T>,
    #[parameter(skip)]
    value_cache: Option<T>,
}

impl<T: Tensor> MimiSelfAttention<T> {
    fn unloaded(spec: specs::Attention, context: &T::Context) -> Result<Self, Error> {
        Ok(Self {
            in_proj: Linear::unloaded(spec.input, context)?,
            out_proj: Linear::unloaded(spec.output, context)?,
            rope: Rope::new(spec.head_dim, true, 10_000.0, 1.0),
            num_heads: spec.heads,
            head_dim: spec.head_dim,
            scale: (spec.head_dim as f32).sqrt().recip(),
            context: spec.context,
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

#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "T")]
struct SeaNetDecoder<T: Tensor> {
    init_conv1d: StreamableConv1d<T>,
    layers: Vec<DecoderLayer<T>>,
    final_conv1d: StreamableConv1d<T>,
}

impl<T: Tensor> SeaNetDecoder<T> {
    fn unloaded(spec: specs::Decoder, context: &T::Context) -> Result<Self, Error> {
        Ok(Self {
            init_conv1d: spec.init.construct(context)?,
            layers: spec
                .layers
                .into_iter()
                .map(|s| DecoderLayer::unloaded(s, context))
                .collect::<Result<_, _>>()?,
            final_conv1d: spec.final_conv.construct(context)?,
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

#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "T")]
struct DecoderLayer<T: Tensor> {
    upsample: StreamableConvTranspose1d<T>,
    residuals: Vec<SeaNetResnetBlock<T>>,
}

impl<T: Tensor> DecoderLayer<T> {
    fn unloaded(spec: specs::DecoderLayer, context: &T::Context) -> Result<Self, Error> {
        Ok(Self {
            upsample: spec.upsample.construct(context)?,
            residuals: spec
                .residuals
                .into_iter()
                .map(|s| SeaNetResnetBlock::unloaded(s, context))
                .collect::<Result<_, _>>()?,
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

#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "T")]
struct SeaNetResnetBlock<T: Tensor> {
    block: Vec<StreamableConv1d<T>>,
}

impl<T: Tensor> SeaNetResnetBlock<T> {
    fn unloaded(spec: specs::Residual, context: &T::Context) -> Result<Self, Error> {
        Ok(Self {
            block: spec
                .block
                .into_iter()
                .map(|s| s.construct(context))
                .collect::<Result<_, _>>()?,
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

/// Split residual vector quantizer used by Mimi.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "T")]
pub struct SplitResidualVectorQuantizer<T: Tensor> {
    /// First semantic codebook branch.
    pub rvq_first: ResidualVectorQuantizer<T>,
    /// Remaining acoustic codebook branch.
    pub rvq_rest: ResidualVectorQuantizer<T>,
    #[parameter(skip)]
    n_q: i32,
}

impl<T: Tensor> SplitResidualVectorQuantizer<T> {
    fn unloaded(spec: specs::Quantizer, context: &T::Context) -> Result<Self, Error> {
        Ok(Self {
            rvq_first: ResidualVectorQuantizer::unloaded(spec.first, context)?,
            rvq_rest: ResidualVectorQuantizer::unloaded(spec.rest, context)?,
            n_q: spec.count,
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
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "T")]
pub struct ResidualVectorQuantizer<T: Tensor> {
    /// Input projection from Mimi latent dimension into codebook dimension.
    pub input_proj: Conv1x1NoBias<T>,
    /// Output projection from codebook dimension back into Mimi latent dimension.
    pub output_proj: Conv1x1NoBias<T>,
    /// Residual codebook layers.
    pub vq: ResidualVectorQuantization<T>,
}

impl<T: Tensor> ResidualVectorQuantizer<T> {
    fn unloaded(spec: specs::QuantizerBranch, context: &T::Context) -> Result<Self, Error> {
        Ok(Self {
            input_proj: Conv1x1NoBias::unloaded(spec.input, context)?,
            output_proj: Conv1x1NoBias::unloaded(spec.output, context)?,
            vq: ResidualVectorQuantization::unloaded(spec.books, context)?,
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
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "T")]
pub struct ResidualVectorQuantization<T: Tensor> {
    /// Ordered residual quantization layers.
    pub layers: Vec<VectorQuantization<T>>,
}

impl<T: Tensor> ResidualVectorQuantization<T> {
    fn unloaded(specs: Vec<specs::Codebook>, context: &T::Context) -> Result<Self, Error> {
        Ok(Self {
            layers: specs
                .into_iter()
                .map(|s| VectorQuantization::unloaded(s, context))
                .collect::<Result<_, _>>()?,
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
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "T")]
pub struct VectorQuantization<T: Tensor> {
    /// Euclidean codebook.
    pub _codebook: EuclideanCodebook<T>,
}

impl<T: Tensor> VectorQuantization<T> {
    fn unloaded(spec: specs::Codebook, context: &T::Context) -> Result<Self, Error> {
        Ok(Self {
            _codebook: EuclideanCodebook::unloaded(spec, context)?,
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
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "T")]
pub struct EuclideanCodebook<T: Tensor> {
    /// Checkpoint initialization flag.
    pub _initialized: Parameter<T>,
    /// EMA cluster usage.
    pub cluster_usage: Parameter<T>,
    /// EMA embedding sum.
    pub embedding_sum: Parameter<T>,
}

impl<T: Tensor> EuclideanCodebook<T> {
    fn unloaded(spec: specs::Codebook, context: &T::Context) -> Result<Self, Error> {
        Ok(Self {
            _initialized: spec.initialized.construct(context)?,
            cluster_usage: spec.usage.construct(context)?,
            embedding_sum: spec.sum.construct(context)?,
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
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "T")]
pub struct Conv1x1NoBias<T: Tensor> {
    /// Weight shaped `[out_channels, in_channels, 1]`.
    pub weight: Parameter<T>,
}

impl<T: Tensor> Conv1x1NoBias<T> {
    fn unloaded(spec: specs::Slot, context: &T::Context) -> Result<Self, Error> {
        Ok(Self {
            weight: spec.construct(context)?,
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
