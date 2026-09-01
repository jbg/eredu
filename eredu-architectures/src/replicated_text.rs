//! Architecture-owned admission for replicated text execution.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use eredu_checkpoint::{
    AffineQuantization, LinearFormat, SourceTensorEncoding, StoredDtype, WeightQuantization,
};
use eredu_core::{
    checkpoint::{TensorCatalog, TensorDtype},
    ArtifactInspection,
};
use eredu_nn::{AttentionCache, NeuralBackend, NeuralOperatorCapabilities, Tensor};
use eredu_runtime::{
    LayerRuntimeState, ParameterTransformConstraint, ReplicatedTextArchitecture,
    ReplicatedTextParameterOwner, ReplicatedTextParameterPresence,
    ReplicatedTextParameterRequirement, ReplicatedTextParameterRole, ReplicatedTextPhysicalSource,
    ReplicatedTextRequirements, RuntimeState, SelectedReplicatedTextRealization,
};

use crate::{
    configuration::{GgufModelConfig, SafetensorsModelConfig},
    processor_plan::ArtifactArchitecturePlan,
    GgufArchitecture,
};

/// Architecture-owned reason that an admitted artifact cannot use replicated text composition.
#[derive(Debug, Clone, Copy, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum ReplicatedTextIneligibility {
    /// The architecture uses routed computation.
    #[error("architecture requires routed execution")]
    Routed,
    /// The architecture uses heterogeneous recurrent or convolutional state.
    #[error("architecture requires hybrid state execution")]
    HybridState,
    /// The architecture owns composite or multimodal ingress.
    #[error("architecture requires composite input execution")]
    CompositeInput,
    /// The architecture owns embedded prediction groups.
    #[error("architecture requires embedded prediction execution")]
    EmbeddedPrediction,
    /// The architecture uses a frame-oriented realtime lifecycle.
    #[error("architecture requires realtime execution")]
    Realtime,
    /// The architecture is outside the replicated text contract.
    #[error("architecture is not an ordinary replicated text decoder")]
    Unrelated,
}

/// Checked architecture value passed across the architecture/backend boundary.
pub struct PreparedReplicatedTextArchitecture<A> {
    architecture: A,
    source_architecture: Option<A>,
    requirements: ReplicatedTextRequirements,
    selected: SelectedReplicatedTextRealization,
    capability_estimate: crate::capability::CapabilityEstimate,
    effective_model_type: String,
}

impl<A> PreparedReplicatedTextArchitecture<A> {
    /// Returns the exact architecture and artifact requirements.
    pub const fn requirements(&self) -> &ReplicatedTextRequirements {
        &self.requirements
    }

    /// Returns the authoritative selected realization.
    pub const fn selected(&self) -> &SelectedReplicatedTextRealization {
        &self.selected
    }

    /// Returns the architecture capability estimate presented by the session.
    pub const fn capability_estimate(&self) -> &crate::capability::CapabilityEstimate {
        &self.capability_estimate
    }

    /// Returns the normalized model-type label presented by the session.
    pub fn effective_model_type(&self) -> &str {
        &self.effective_model_type
    }

    /// Consumes the handoff into opaque architecture module ownership.
    pub fn into_modules(self) -> PreparedReplicatedTextModules<A> {
        PreparedReplicatedTextModules {
            architecture: Some(self.architecture),
            source_architecture: self.source_architecture,
        }
    }
}

/// Opaque ownership of selected-format and optional source-format modules.
pub struct PreparedReplicatedTextModules<A> {
    architecture: Option<A>,
    source_architecture: Option<A>,
}

impl<A> PreparedReplicatedTextModules<A> {
    /// Takes the selected-format architecture exactly once.
    pub fn take_architecture(&mut self) -> A {
        self.architecture
            .take()
            .expect("prepared architecture was already taken")
    }

    /// Takes the source-format architecture used by a selected transform.
    pub fn take_source_architecture(&mut self) -> Option<A> {
        self.source_architecture.take()
    }
}

/// Backend-generic visitor for one admitted replicated text architecture.
pub trait ReplicatedTextArchitectureVisitor<B, S>: Sized
where
    B: NeuralBackend,
    S: RuntimeState<B>,
{
    /// Visitor output.
    type Output;
    /// Backend binding failure.
    type Error;
    /// Records that validated dispatch is about to invoke architecture constructors.
    fn construction_started(&mut self);
    /// Binds one opaque selected-format-aware architecture to backend mechanisms.
    fn visit<A>(
        self,
        prepared: PreparedReplicatedTextArchitecture<A>,
    ) -> Result<Self::Output, Self::Error>
    where
        A: ReplicatedTextArchitecture<B, S, Error = eredu_nn::Error> + 'static,
        A::StaticModules: Clone,
        A::Error: std::fmt::Display;
}

/// Failure before or during the checked architecture/backend handoff.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ReplicatedTextDispatchError<E> {
    /// The normalized architecture belongs to a different execution class.
    #[error(transparent)]
    Ineligible(#[from] ReplicatedTextIneligibility),
    /// Selected formats could not construct the normalized architecture.
    #[error("replicated text architecture construction failed: {0}")]
    Architecture(String),
    /// The backend visitor rejected the checked architecture.
    #[error("replicated text backend binding failed: {0}")]
    Backend(E),
}

/// Constructs and visits the architecture using one authoritative realization.
pub fn visit_replicated_text_architecture<B, S, V>(
    plan: &ArtifactArchitecturePlan,
    requirements: ReplicatedTextRequirements,
    selected: SelectedReplicatedTextRealization,
    context: &<B::Tensor as Tensor>::Context,
    mut visitor: V,
) -> Result<V::Output, ReplicatedTextDispatchError<V::Error>>
where
    B: NeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor>,
    V: ReplicatedTextArchitectureVisitor<B, S>,
{
    validate_selected_handoff(&requirements, &selected)
        .map_err(ReplicatedTextDispatchError::Architecture)?;
    let eligible = eligible_config(plan)?;
    visitor.construction_started();
    match eligible {
        EligibleConfig::Llama(args) => {
            let capability_estimate = crate::capability::llama(args)
                .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
            let effective_model_type = args.model_type.clone();
            let source_architecture = selected_uses_transform(&selected)
                .then(|| crate::llama::LayeredModel::<B>::new(args.clone(), context))
                .transpose()
                .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
            let args = selected_llama_args(args, &selected)
                .map_err(ReplicatedTextDispatchError::Architecture)?;
            let architecture = crate::llama::LayeredModel::<B>::new(args, context)
                .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
            visitor
                .visit(PreparedReplicatedTextArchitecture {
                    architecture,
                    source_architecture,
                    requirements,
                    selected,
                    capability_estimate,
                    effective_model_type,
                })
                .map_err(ReplicatedTextDispatchError::Backend)
        }
        EligibleConfig::Qwen(args) => {
            let capability_estimate = crate::capability::qwen(args)
                .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
            let effective_model_type = args.model_type.clone();
            let source_architecture = selected_uses_transform(&selected)
                .then(|| crate::qwen::LayeredModel::<B>::new(args.clone(), context))
                .transpose()
                .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
            let args = selected_qwen_args(args, &selected)
                .map_err(ReplicatedTextDispatchError::Architecture)?;
            let architecture = crate::qwen::LayeredModel::<B>::new(args, context)
                .map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))?;
            visitor
                .visit(PreparedReplicatedTextArchitecture {
                    architecture,
                    source_architecture,
                    requirements,
                    selected,
                    capability_estimate,
                    effective_model_type,
                })
                .map_err(ReplicatedTextDispatchError::Backend)
        }
    }
}

fn selected_uses_transform(selected: &SelectedReplicatedTextRealization) -> bool {
    selected
        .parameters()
        .iter()
        .any(|parameter| parameter.lowering() == eredu_runtime::WeightLoweringKind::Transform)
}

fn validate_selected_handoff(
    requirements: &ReplicatedTextRequirements,
    selected: &SelectedReplicatedTextRealization,
) -> Result<(), String> {
    let required = requirements
        .parameters()
        .iter()
        .filter(|parameter| parameter.presence().has_physical_source())
        .map(|parameter| parameter.name())
        .collect::<BTreeSet<_>>();
    let realized = selected
        .parameters()
        .iter()
        .map(|parameter| parameter.name())
        .collect::<BTreeSet<_>>();
    if required != realized || selected.parameters().len() != required.len() {
        return Err("selected parameter realization does not match exact requirements".into());
    }
    for requirement in requirements.parameters() {
        if !requirement.presence().has_physical_source() {
            continue;
        }
        let realization = selected
            .parameters()
            .iter()
            .find(|parameter| parameter.name() == requirement.name())
            .expect("parameter identity sets were compared above");
        if realization.sources() != requirement.sources()
            || realization.physical_sources() != requirement.physical_sources()
            || Some(realization.source_encoding()) != requirement.source_encoding()
        {
            return Err(format!(
                "selected source facts for {:?} differ from exact artifact requirements",
                requirement.name()
            ));
        }
        let valid_format = match realization.lowering() {
            eredu_runtime::WeightLoweringKind::Direct => {
                realization.executable() == requirement.native_executable()
            }
            eredu_runtime::WeightLoweringKind::Transform => {
                let request = match realization.executable() {
                    LinearFormat::Affine(format) => Some(eredu_core::QuantizationRequest::Affine {
                        group_size: u32::try_from(format.group_size)
                            .map_err(|_| "negative selected affine group size".to_owned())?,
                        bits: u8::try_from(format.bits)
                            .map_err(|_| "invalid selected affine bit width".to_owned())?,
                    }),
                    LinearFormat::MxFp4 => Some(eredu_core::QuantizationRequest::MxFp4),
                    _ => None,
                };
                match request {
                    Some(request) => requirement
                        .transform_target(request)
                        .map_err(|error| error.to_string())?
                        .is_some_and(|target| target.executable() == realization.executable()),
                    None => false,
                }
            }
            _ => false,
        };
        if !valid_format {
            return Err(format!(
                "selected executable format for {:?} is not architecture-admitted",
                requirement.name()
            ));
        }
    }
    Ok(())
}

fn selected_formats(
    selected: &SelectedReplicatedTextRealization,
) -> HashMap<String, WeightQuantization> {
    selected
        .parameters()
        .iter()
        .filter_map(|parameter| {
            parameter
                .executable()
                .weight_quantization()
                .map(|format| (parameter.name().to_owned(), format))
        })
        .collect()
}

fn selected_llama_args(
    args: &crate::llama::ModelArgs,
    selected: &SelectedReplicatedTextRealization,
) -> Result<crate::llama::ModelArgs, String> {
    let formats = selected_formats(selected);
    if formats.is_empty() {
        Ok(args.clone())
    } else {
        crate::llama::with_checkpoint_formats(args, formats)
    }
}

fn selected_qwen_args(
    args: &crate::qwen::ModelArgs,
    selected: &SelectedReplicatedTextRealization,
) -> Result<crate::qwen::ModelArgs, String> {
    let formats = selected_formats(selected);
    if formats.is_empty() {
        Ok(args.clone())
    } else {
        crate::qwen::with_checkpoint_formats(args, formats)
    }
}

enum EligibleConfig<'a> {
    Llama(&'a crate::llama::ModelArgs),
    Qwen(&'a crate::qwen::ModelArgs),
}

impl EligibleConfig<'_> {
    fn unit_count(&self) -> Result<usize, String> {
        let count = match self {
            Self::Llama(args) => args.num_hidden_layers,
            Self::Qwen(args) => args.num_hidden_layers,
        };
        usize::try_from(count).map_err(|_| format!("invalid replicated layer count {count}"))
    }

    fn state_layout(&self) -> Result<eredu_runtime::StateLayout, String> {
        match self {
            Self::Llama(args) => crate::llama::state_layout(*args),
            Self::Qwen(args) => crate::qwen::state_layout(*args),
        }
        .map_err(|error| error.to_string())
    }

    fn native_format(&self, name: &str) -> LinearFormat {
        match self {
            Self::Llama(args) => args.weight_quantization_for(name),
            Self::Qwen(args) => args.weight_quantization_for(name),
        }
        .map_or(LinearFormat::Dense, LinearFormat::from)
    }

    fn linear_parameter_shapes(&self) -> Result<BTreeMap<String, Vec<usize>>, String> {
        match self {
            Self::Llama(args) => decoder_linear_parameter_shapes(*args),
            Self::Qwen(args) => decoder_linear_parameter_shapes(*args),
        }
    }

    fn parameter_root(&self) -> &str {
        match self {
            Self::Llama(args) => crate::decoder::Config::parameter_root(*args),
            Self::Qwen(args) => crate::decoder::Config::parameter_root(*args),
        }
    }

    fn tied_embeddings(&self) -> bool {
        match self {
            Self::Llama(args) => crate::decoder::Config::tie_word_embeddings(*args),
            Self::Qwen(args) => crate::decoder::Config::tie_word_embeddings(*args),
        }
    }

    fn embedding_shape(&self) -> Result<Vec<usize>, String> {
        let (vocabulary, hidden) = match self {
            Self::Llama(args) => (args.vocab_size, args.hidden_size),
            Self::Qwen(args) => (args.vocab_size, args.hidden_size),
        };
        Ok(vec![
            positive(vocabulary, "vocabulary size")?,
            positive(hidden, "hidden size")?,
        ])
    }

    fn parameter_role(
        &self,
        name: &str,
        companion: bool,
        linear_shapes: &BTreeMap<String, Vec<usize>>,
    ) -> ReplicatedTextParameterRole {
        match self {
            Self::Llama(args) => decoder_parameter_role(*args, name, companion, linear_shapes),
            Self::Qwen(args) => decoder_parameter_role(*args, name, companion, linear_shapes),
        }
    }
}

fn positive(value: i32, label: &str) -> Result<usize, String> {
    usize::try_from(value)
        .ok()
        .filter(|value| *value > 0)
        .ok_or_else(|| format!("decoder {label} must be positive"))
}

/// Returns the exact affine-projection identities and logical geometry declared
/// by a normalized decoder. Artifact rank and name suffixes do not determine
/// transform eligibility.
fn decoder_linear_parameter_shapes<C: crate::decoder::Config>(
    config: &C,
) -> Result<BTreeMap<String, Vec<usize>>, String> {
    config
        .validate_config()
        .map_err(|error| error.to_string())?;
    let root = config.parameter_root();
    let fields = config.block_parameter_fields();
    let hidden = positive(config.hidden_size(), "hidden size")?;
    let intermediate = positive(config.intermediate_size(), "intermediate size")?;
    let head = positive(config.head_dim(), "head dimension")?;
    let query = positive(config.num_attention_heads(), "query-head count")?
        .checked_mul(head)
        .ok_or_else(|| "decoder query width overflowed".to_string())?;
    let key_value = positive(config.num_key_value_heads(), "key/value-head count")?
        .checked_mul(head)
        .ok_or_else(|| "decoder key/value width overflowed".to_string())?;
    let layers = positive(config.num_hidden_layers(), "layer count")?;
    let mut result = BTreeMap::new();
    for layer in 0..layers {
        let layer_prefix = format!("{root}.layers.{layer}");
        let attention = format!("{layer_prefix}.{}", fields.attention);
        match config.attention_projection_layout() {
            crate::decoder::AttentionProjectionLayout::Split => {
                result.insert(
                    format!("{attention}.{}.weight", fields.attention_query),
                    vec![query, hidden],
                );
                result.insert(
                    format!("{attention}.{}.weight", fields.attention_key),
                    vec![key_value, hidden],
                );
                result.insert(
                    format!("{attention}.{}.weight", fields.attention_value),
                    vec![key_value, hidden],
                );
            }
            crate::decoder::AttentionProjectionLayout::Fused { field } => {
                let output =
                    query
                        .checked_add(key_value.checked_mul(2).ok_or_else(|| {
                            "decoder fused attention width overflowed".to_string()
                        })?)
                        .ok_or_else(|| "decoder fused attention width overflowed".to_string())?;
                result.insert(format!("{attention}.{field}.weight"), vec![output, hidden]);
            }
        }
        result.insert(
            format!("{attention}.{}.weight", fields.attention_output),
            vec![hidden, query],
        );
        let feed_forward = format!("{layer_prefix}.{}", fields.feed_forward);
        match config.gated_projection_layout() {
            crate::decoder::GatedProjectionLayout::Split => {
                result.insert(
                    format!("{feed_forward}.{}.weight", fields.feed_forward_gate),
                    vec![intermediate, hidden],
                );
                result.insert(
                    format!("{feed_forward}.{}.weight", fields.feed_forward_up),
                    vec![intermediate, hidden],
                );
            }
            crate::decoder::GatedProjectionLayout::Fused { field } => {
                result.insert(
                    format!("{feed_forward}.{field}.weight"),
                    vec![
                        intermediate.checked_mul(2).ok_or_else(|| {
                            "decoder fused feed-forward width overflowed".to_string()
                        })?,
                        hidden,
                    ],
                );
            }
        }
        result.insert(
            format!("{feed_forward}.{}.weight", fields.feed_forward_output),
            vec![hidden, intermediate],
        );
    }
    if !config.tie_word_embeddings() {
        result.insert(
            "lm_head.weight".into(),
            vec![
                positive(config.vocabulary_size(), "vocabulary size")?,
                hidden,
            ],
        );
    }
    Ok(result)
}

fn decoder_parameter_role<C: crate::decoder::Config>(
    config: &C,
    name: &str,
    companion: bool,
    linear_shapes: &BTreeMap<String, Vec<usize>>,
) -> ReplicatedTextParameterRole {
    if companion {
        return ReplicatedTextParameterRole::FormatCompanion;
    }
    if linear_shapes.contains_key(name)
        || (config.tie_word_embeddings() && name == "lm_head.weight")
    {
        return ReplicatedTextParameterRole::LinearWeight;
    }
    if name == format!("{}.embed_tokens.weight", config.parameter_root()) {
        return ReplicatedTextParameterRole::Embedding;
    }
    if linear_shapes.keys().any(|weight| {
        weight
            .strip_suffix(".weight")
            .is_some_and(|prefix| name == format!("{prefix}.bias"))
    }) {
        return ReplicatedTextParameterRole::LinearBias;
    }
    let fields = config.block_parameter_fields();
    if name == format!("{}.norm.weight", config.parameter_root())
        || (0..usize::try_from(config.num_hidden_layers()).unwrap_or_default()).any(|layer| {
            let prefix = format!("{}.layers.{layer}", config.parameter_root());
            name == format!("{prefix}.{}.weight", fields.input_norm)
                || name == format!("{prefix}.{}.weight", fields.post_attention_norm)
                || name
                    == format!(
                        "{prefix}.{}.{}.weight",
                        fields.attention, fields.attention_query_norm
                    )
                || name
                    == format!(
                        "{prefix}.{}.{}.weight",
                        fields.attention, fields.attention_key_norm
                    )
        })
    {
        return ReplicatedTextParameterRole::Normalization;
    }
    ReplicatedTextParameterRole::Other
}

/// Derives exact replicated text requirements from an admitted artifact.
pub fn replicated_text_requirements(
    inspection: &ArtifactInspection<ArtifactArchitecturePlan>,
) -> Result<ReplicatedTextRequirements, ReplicatedTextRequirementsError> {
    let plan = inspection.architecture_plan();
    let config = eligible_config(plan)?;
    let parameters = match (
        plan.safetensors_architecture(),
        plan.gguf_plan(),
        inspection.gguf_checkpoint(),
    ) {
        (Some(architecture), None, None) => safetensors_parameters(
            architecture,
            inspection.tensors(),
            inspection.safetensors_shards().ok_or_else(|| {
                ReplicatedTextRequirementsError::InvalidArtifact(
                    "SafeTensors inspection omitted its admitted shard set".into(),
                )
            })?,
            &config,
        )?,
        (None, Some(architecture), Some(checkpoint)) => {
            gguf_parameters(architecture, checkpoint, &config)?
        }
        _ => {
            return Err(ReplicatedTextRequirementsError::InvalidArtifact(
                "artifact container and admitted architecture plan disagree".into(),
            ))
        }
    };
    let execution_graph = eredu_runtime::ExecutionGraph::chain(["text_decoder"])
        .map_err(|error| ReplicatedTextRequirementsError::InvalidArchitecture(error.to_string()))?;
    let execution_units = eredu_runtime::ExecutionUnitLayout::new(
        &execution_graph,
        [config
            .unit_count()
            .map_err(ReplicatedTextRequirementsError::InvalidArchitecture)?],
    )
    .map_err(|error| ReplicatedTextRequirementsError::InvalidArchitecture(error.to_string()))?;
    ReplicatedTextRequirements::new(
        NeuralOperatorCapabilities::NONE,
        execution_graph,
        execution_units,
        vec![crate::transport::decoder()],
        config
            .state_layout()
            .map_err(ReplicatedTextRequirementsError::InvalidArchitecture)?,
        parameters,
    )
    .map_err(|error| ReplicatedTextRequirementsError::InvalidArchitecture(error.to_string()))
}

fn eligible_config(
    plan: &ArtifactArchitecturePlan,
) -> Result<EligibleConfig<'_>, ReplicatedTextIneligibility> {
    if plan.has_processor() || plan.gguf_media_projector().is_some() {
        return Err(ReplicatedTextIneligibility::CompositeInput);
    }
    match (plan.safetensors_architecture(), plan.gguf_plan()) {
        (Some(architecture), None) => match architecture.model() {
            SafetensorsModelConfig::Llama(args) => Ok(EligibleConfig::Llama(args)),
            SafetensorsModelConfig::Qwen(args) if !args.is_moe() => Ok(EligibleConfig::Qwen(args)),
            SafetensorsModelConfig::Qwen(_) => Err(ReplicatedTextIneligibility::Routed),
            SafetensorsModelConfig::QwenHybrid(args) if args.vision.is_some() => {
                Err(ReplicatedTextIneligibility::CompositeInput)
            }
            SafetensorsModelConfig::QwenHybrid(args) if args.text.mtp_num_hidden_layers > 0 => {
                Err(ReplicatedTextIneligibility::EmbeddedPrediction)
            }
            SafetensorsModelConfig::QwenHybrid(_)
            | SafetensorsModelConfig::KimiLinear(_)
            | SafetensorsModelConfig::Lfm2(_) => Err(ReplicatedTextIneligibility::HybridState),
            SafetensorsModelConfig::NemotronH(args) if args.num_nextn_predict_layers > 0 => {
                Err(ReplicatedTextIneligibility::EmbeddedPrediction)
            }
            SafetensorsModelConfig::NemotronH(_) => Err(ReplicatedTextIneligibility::HybridState),
            SafetensorsModelConfig::Gemma4(_)
            | SafetensorsModelConfig::Inkling(_)
            | SafetensorsModelConfig::MuseGlimmer(_)
            | SafetensorsModelConfig::QwenVl(_) => Err(ReplicatedTextIneligibility::CompositeInput),
            SafetensorsModelConfig::Moshi(_) => Err(ReplicatedTextIneligibility::Realtime),
            SafetensorsModelConfig::DeepSeekV3(args) if args.num_nextn_predict_layers > 0 => {
                Err(ReplicatedTextIneligibility::EmbeddedPrediction)
            }
            SafetensorsModelConfig::DeepSeekV4(args) if args.num_nextn_predict_layers > 0 => {
                Err(ReplicatedTextIneligibility::EmbeddedPrediction)
            }
            SafetensorsModelConfig::DeepSeekV3(args) if args.has_sparse_moe_layers() => {
                Err(ReplicatedTextIneligibility::Routed)
            }
            SafetensorsModelConfig::DeepSeekV3(_)
            | SafetensorsModelConfig::DeepSeekV4(_)
            | SafetensorsModelConfig::GptOss(_) => Err(ReplicatedTextIneligibility::Unrelated),
        },
        (None, Some(architecture)) => match architecture.model() {
            GgufModelConfig::Llama(args) => Ok(EligibleConfig::Llama(args)),
            GgufModelConfig::Qwen(args)
                if matches!(
                    architecture.architecture(),
                    GgufArchitecture::Qwen2 | GgufArchitecture::Qwen3
                ) && !args.is_moe() =>
            {
                Ok(EligibleConfig::Qwen(args))
            }
            GgufModelConfig::Qwen(_) => Err(ReplicatedTextIneligibility::Routed),
            GgufModelConfig::QwenHybrid(args) if args.vision.is_some() => {
                Err(ReplicatedTextIneligibility::CompositeInput)
            }
            GgufModelConfig::QwenHybrid(args) if args.text.mtp_num_hidden_layers > 0 => {
                Err(ReplicatedTextIneligibility::EmbeddedPrediction)
            }
            GgufModelConfig::QwenHybrid(_)
            | GgufModelConfig::KimiLinear(_)
            | GgufModelConfig::Lfm2(_) => Err(ReplicatedTextIneligibility::HybridState),
            GgufModelConfig::NemotronH(args) if args.num_nextn_predict_layers > 0 => {
                Err(ReplicatedTextIneligibility::EmbeddedPrediction)
            }
            GgufModelConfig::NemotronH(_) => Err(ReplicatedTextIneligibility::HybridState),
            GgufModelConfig::Gemma4(_)
            | GgufModelConfig::Inkling(_)
            | GgufModelConfig::MuseGlimmer(_) => Err(ReplicatedTextIneligibility::CompositeInput),
            GgufModelConfig::DeepSeekV3(args) if args.num_nextn_predict_layers > 0 => {
                Err(ReplicatedTextIneligibility::EmbeddedPrediction)
            }
            GgufModelConfig::DeepSeekV4(args) if args.num_nextn_predict_layers > 0 => {
                Err(ReplicatedTextIneligibility::EmbeddedPrediction)
            }
            GgufModelConfig::DeepSeekV3(args) if args.has_sparse_moe_layers() => {
                Err(ReplicatedTextIneligibility::Routed)
            }
            GgufModelConfig::DeepSeekV3(_)
            | GgufModelConfig::DeepSeekV4(_)
            | GgufModelConfig::GptOss(_) => Err(ReplicatedTextIneligibility::Unrelated),
        },
        _ => Err(ReplicatedTextIneligibility::Unrelated),
    }
}

fn safetensors_parameters(
    architecture: &crate::configuration::SafetensorsArchitecturePlan,
    catalog: &TensorCatalog,
    shards: &eredu_checkpoint::safetensors::SafetensorsShards,
    config: &EligibleConfig<'_>,
) -> Result<Vec<ReplicatedTextParameterRequirement>, ReplicatedTextRequirementsError> {
    let linear_shapes = config
        .linear_parameter_shapes()
        .map_err(ReplicatedTextRequirementsError::InvalidArchitecture)?;
    let selected = architecture.checkpoint_resolution().ok_or_else(|| {
        ReplicatedTextRequirementsError::InvalidArtifact(
            "SafeTensors architecture omitted exact catalog admission".into(),
        )
    })?;
    let plan = architecture.checkpoint();
    let mut constraints = plan.common_tensors.iter().collect::<Vec<_>>();
    for group in &plan.layout_groups {
        for variant in &group.variants {
            if variant.tensors.iter().all(|constraint| {
                selected.source_keys().contains(&constraint.key)
                    || constraint
                        .aliases
                        .iter()
                        .any(|alias| selected.source_keys().contains(alias))
                    || constraint.requirement
                        == eredu_checkpoint::schema::TensorRequirement::Optional
            }) {
                constraints.extend(variant.tensors.iter());
                break;
            }
        }
    }
    let mut parameters = Vec::new();
    for constraint in constraints {
        let source = std::iter::once(&constraint.key)
            .chain(constraint.aliases.iter())
            .find(|name| selected.source_keys().contains(*name))
            .cloned();
        let descriptor = source
            .as_deref()
            .map(|source| {
                catalog.get(source).ok_or_else(|| {
                    ReplicatedTextRequirementsError::InvalidArtifact(format!(
                        "admitted SafeTensors source {source:?} is absent from its catalog"
                    ))
                })
            })
            .transpose()?;
        let presence = match (constraint.requirement, source.is_some()) {
            (eredu_checkpoint::schema::TensorRequirement::Required, true) => {
                ReplicatedTextParameterPresence::Required
            }
            (eredu_checkpoint::schema::TensorRequirement::Optional, true) => {
                ReplicatedTextParameterPresence::OptionalPresent
            }
            (eredu_checkpoint::schema::TensorRequirement::Optional, false) => {
                ReplicatedTextParameterPresence::OptionalAbsent
            }
            (eredu_checkpoint::schema::TensorRequirement::Required, false) => {
                return Err(ReplicatedTextRequirementsError::InvalidArtifact(format!(
                    "required admitted SafeTensors parameter {:?} has no source",
                    constraint.key
                )))
            }
        };
        let role = config.parameter_role(
            &constraint.key,
            constraint.role == eredu_checkpoint::schema::TensorRole::Companion,
            &linear_shapes,
        );
        let logical_shape = if role == ReplicatedTextParameterRole::Embedding {
            config
                .embedding_shape()
                .map_err(ReplicatedTextRequirementsError::InvalidArchitecture)?
        } else {
            linear_shapes
                .get(&constraint.key)
                .cloned()
                .unwrap_or_else(|| constraint.shape.clone())
        };
        parameters.push(parameter_requirement(
            constraint.key.clone(),
            source.clone().into_iter().collect(),
            source
                .as_ref()
                .map(|source| {
                    let shard = shards
                        .tensor_locations()
                        .and_then(|locations| locations.get(source))
                        .or_else(|| {
                            (shards.payload_paths().len() == 1)
                                .then(|| &shards.payload_paths()[0])
                        })
                        .ok_or_else(|| {
                            ReplicatedTextRequirementsError::InvalidArtifact(format!(
                                "admitted SafeTensors source {source:?} has no exact shard membership"
                            ))
                        })?;
                    ReplicatedTextPhysicalSource::new(source, shard, source).map_err(|error| {
                        ReplicatedTextRequirementsError::InvalidArtifact(error.to_string())
                    })
                })
                .transpose()?
                .into_iter()
                .collect(),
            constraint.aliases.clone(),
            descriptor
                .map(|descriptor| stored_dtype(&descriptor.dtype))
                .transpose()?
                .map(SourceTensorEncoding::Safetensors),
            descriptor.map(|descriptor| descriptor.shape.clone()),
            logical_shape,
            config.native_format(&constraint.key),
            linear_shapes.contains_key(&constraint.key),
            role,
            parameter_owner(config, &constraint.key),
            presence,
        )?);
    }
    if config.tied_embeddings() {
        parameters.retain(|parameter| parameter.name() != "lm_head.weight");
        parameters.push(parameter_requirement(
            "lm_head.weight".into(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            None,
            None,
            vec![
                positive(
                    match config {
                        EligibleConfig::Llama(args) => args.vocab_size,
                        EligibleConfig::Qwen(args) => args.vocab_size,
                    },
                    "vocabulary size",
                )
                .map_err(ReplicatedTextRequirementsError::InvalidArchitecture)?,
                positive(
                    match config {
                        EligibleConfig::Llama(args) => args.hidden_size,
                        EligibleConfig::Qwen(args) => args.hidden_size,
                    },
                    "hidden size",
                )
                .map_err(ReplicatedTextRequirementsError::InvalidArchitecture)?,
            ],
            LinearFormat::Dense,
            true,
            ReplicatedTextParameterRole::LinearWeight,
            ReplicatedTextParameterOwner::StaticRole("output".into()),
            ReplicatedTextParameterPresence::Tied {
                target: format!("{}.embed_tokens.weight", config.parameter_root()),
            },
        )?);
    }
    finish_parameters(parameters)
}

fn gguf_parameters(
    architecture: &crate::configuration::GgufArchitecturePlan,
    checkpoint: &eredu_gguf::Checkpoint,
    config: &EligibleConfig<'_>,
) -> Result<Vec<ReplicatedTextParameterRequirement>, ReplicatedTextRequirementsError> {
    let linear_shapes = config
        .linear_parameter_shapes()
        .map_err(ReplicatedTextRequirementsError::InvalidArchitecture)?;
    let mut physical = BTreeMap::new();
    for shard in checkpoint.shards() {
        for tensor in shard.tensors() {
            physical.insert(
                tensor.descriptor().name.as_str(),
                (
                    tensor,
                    shard.path(),
                    SourceTensorEncoding::Gguf {
                        ggml_type: tensor.descriptor().ggml_type,
                        endian: shard.endian(),
                    },
                ),
            );
        }
    }
    let mut parameters = Vec::new();
    for mapping in architecture.tensor_mapping() {
        let Some((tensor, shard, source_encoding)) = physical.get(mapping.physical_name.as_str())
        else {
            return Err(ReplicatedTextRequirementsError::InvalidArtifact(format!(
                "admitted GGUF mapping references absent tensor {:?}",
                mapping.physical_name
            )));
        };
        let native = if let Some((bits, group)) = tensor.affine() {
            LinearFormat::Affine(AffineQuantization::new(
                i32::try_from(group).map_err(|_| {
                    ReplicatedTextRequirementsError::InvalidArtifact(format!(
                        "GGUF group size {group} exceeds i32"
                    ))
                })?,
                i32::from(bits),
            )?)
        } else if tensor.is_mxfp4() {
            LinearFormat::MxFp4
        } else if tensor.descriptor().ggml_type.block_and_bytes().is_ok()
            && !matches!(
                tensor.descriptor().ggml_type,
                eredu_gguf::GgmlType::F16 | eredu_gguf::GgmlType::F32 | eredu_gguf::GgmlType::Bf16
            )
        {
            let SourceTensorEncoding::Gguf { ggml_type, endian } = source_encoding else {
                unreachable!("GGUF catalog produces GGUF source encodings")
            };
            LinearFormat::GgufIQuant {
                ggml_type: *ggml_type,
                endian: *endian,
            }
        } else {
            LinearFormat::Dense
        };
        let companion = mapping.original_name != mapping.physical_name;
        let presence = if companion {
            ReplicatedTextParameterPresence::Derived {
                recipe: format!(
                    "gguf-output:{}:{}",
                    mapping.physical_name, mapping.original_name
                ),
            }
        } else {
            ReplicatedTextParameterPresence::Required
        };
        let role = config.parameter_role(&mapping.layout.name, companion, &linear_shapes);
        let logical_shape = if role == ReplicatedTextParameterRole::Embedding {
            config
                .embedding_shape()
                .map_err(ReplicatedTextRequirementsError::InvalidArchitecture)?
        } else {
            linear_shapes
                .get(&mapping.layout.name)
                .cloned()
                .map(Ok)
                .unwrap_or_else(|| {
                    mapping
                        .layout
                        .shape
                        .iter()
                        .map(|dimension| {
                            usize::try_from(*dimension).map_err(|_| {
                                ReplicatedTextRequirementsError::InvalidArtifact(format!(
                                    "GGUF logical shape for {:?} exceeds usize",
                                    mapping.layout.name
                                ))
                            })
                        })
                        .collect()
                })?
        };
        parameters.push(parameter_requirement(
            mapping.layout.name.clone(),
            (!companion)
                .then(|| mapping.physical_name.clone())
                .into_iter()
                .collect(),
            vec![ReplicatedTextPhysicalSource::new(
                mapping.physical_name.clone(),
                *shard,
                mapping.original_name.clone(),
            )
            .map_err(|error| {
                ReplicatedTextRequirementsError::InvalidArtifact(error.to_string())
            })?],
            vec![mapping.original_name.clone()],
            (!companion).then(|| source_encoding.clone()),
            (!companion)
                .then(|| {
                    tensor
                        .descriptor()
                        .row_major_shape()
                        .into_iter()
                        .map(usize::try_from)
                        .collect::<Result<Vec<_>, _>>()
                        .map_err(|_| {
                            ReplicatedTextRequirementsError::InvalidArtifact(format!(
                                "GGUF physical shape for {:?} exceeds usize",
                                mapping.physical_name
                            ))
                        })
                })
                .transpose()?,
            logical_shape,
            if companion {
                LinearFormat::Dense
            } else {
                native
            },
            !companion && linear_shapes.contains_key(&mapping.layout.name),
            role,
            parameter_owner(config, &mapping.layout.name),
            presence,
        )?);
    }
    finish_parameters(parameters)
}

fn parameter_requirement(
    name: String,
    sources: Vec<String>,
    physical_sources: Vec<ReplicatedTextPhysicalSource>,
    aliases: Vec<String>,
    source_encoding: Option<SourceTensorEncoding>,
    physical_shape: Option<Vec<usize>>,
    logical_shape: Vec<usize>,
    native_executable: LinearFormat,
    transformable: bool,
    role: ReplicatedTextParameterRole,
    owner: ReplicatedTextParameterOwner,
    presence: ReplicatedTextParameterPresence,
) -> Result<ReplicatedTextParameterRequirement, ReplicatedTextRequirementsError> {
    let transform = if transformable {
        ParameterTransformConstraint::Linear {
            packed_axis: logical_shape.len().checked_sub(1).ok_or_else(|| {
                ReplicatedTextRequirementsError::InvalidArchitecture(
                    "linear parameter has a scalar logical shape".into(),
                )
            })?,
        }
    } else {
        ParameterTransformConstraint::None
    };
    ReplicatedTextParameterRequirement::new(
        name,
        sources,
        physical_sources,
        aliases,
        source_encoding,
        physical_shape,
        logical_shape,
        native_executable,
        role,
        owner,
        presence,
        transform,
    )
    .map_err(|error| ReplicatedTextRequirementsError::InvalidArchitecture(error.to_string()))
}

fn parameter_owner(config: &EligibleConfig<'_>, name: &str) -> ReplicatedTextParameterOwner {
    let layer_prefix = format!("{}.layers.", config.parameter_root());
    if let Some(rest) = name.strip_prefix(&layer_prefix) {
        if let Some(layer) = rest
            .split('.')
            .next()
            .and_then(|layer| layer.parse::<usize>().ok())
        {
            return ReplicatedTextParameterOwner::ExecutionUnit {
                group: crate::decoder::TEXT_DECODER_EXECUTION_GROUP.into(),
                unit: layer,
            };
        }
    }
    let role = if name.starts_with(&format!("{}.embed_tokens", config.parameter_root())) {
        "embedding"
    } else if name == "lm_head.weight" || name == "lm_head.bias" {
        "output"
    } else {
        "norm"
    };
    ReplicatedTextParameterOwner::StaticRole(role.into())
}

fn finish_parameters(
    mut parameters: Vec<ReplicatedTextParameterRequirement>,
) -> Result<Vec<ReplicatedTextParameterRequirement>, ReplicatedTextRequirementsError> {
    parameters.sort_by(|left, right| left.name().cmp(right.name()));
    let mut names = BTreeSet::new();
    if let Some(duplicate) = parameters
        .iter()
        .find(|parameter| !names.insert(parameter.name()))
    {
        return Err(ReplicatedTextRequirementsError::InvalidArtifact(format!(
            "logical parameter {:?} is mapped more than once",
            duplicate.name()
        )));
    }
    if parameters.is_empty() {
        return Err(ReplicatedTextRequirementsError::InvalidArtifact(
            "replicated text artifact contains no admitted linear parameters".into(),
        ));
    }
    Ok(parameters)
}

fn stored_dtype(dtype: &TensorDtype) -> Result<StoredDtype, ReplicatedTextRequirementsError> {
    Ok(match dtype {
        TensorDtype::Bool => StoredDtype::Bool,
        TensorDtype::F32 => StoredDtype::F32,
        TensorDtype::F16 => StoredDtype::F16,
        TensorDtype::Bf16 => StoredDtype::BF16,
        TensorDtype::I8 => StoredDtype::I8,
        TensorDtype::U8 => StoredDtype::U8,
        TensorDtype::U16 => StoredDtype::U16,
        TensorDtype::U32 => StoredDtype::U32,
        TensorDtype::U64 => StoredDtype::U64,
        TensorDtype::I16 => StoredDtype::I16,
        TensorDtype::I32 => StoredDtype::I32,
        TensorDtype::I64 => StoredDtype::I64,
        TensorDtype::F64 => StoredDtype::F64,
        TensorDtype::Complex64 => StoredDtype::C64,
        TensorDtype::Encoded(name) => match name.as_str() {
            "F8_E4M3" => StoredDtype::F8E4M3,
            "F8_E5M2" => StoredDtype::F8E5M2,
            "F4" => StoredDtype::F4,
            "F8_E8M0" => StoredDtype::F8E8M0,
            _ => StoredDtype::Other(name.clone()),
        },
    })
}

/// Failure while deriving replicated text requirements from an admitted artifact.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ReplicatedTextRequirementsError {
    /// The architecture belongs to a different execution class.
    #[error(transparent)]
    Ineligible(#[from] ReplicatedTextIneligibility),
    /// Admitted artifact facts are internally inconsistent.
    #[error("invalid replicated text artifact: {0}")]
    InvalidArtifact(String),
    /// Architecture geometry or a requested transform is invalid.
    #[error("invalid replicated text architecture: {0}")]
    InvalidArchitecture(String),
}

impl From<eredu_checkpoint::Error> for ReplicatedTextRequirementsError {
    fn from(error: eredu_checkpoint::Error) -> Self {
        Self::InvalidArchitecture(error.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eredu_core::ModelConfigurationResolver;
    use safetensors::{tensor::serialize_to_file, tensor::TensorView, Dtype};

    fn config(model_type: &str) -> serde_json::Value {
        let architecture = match model_type {
            "llama" => "LlamaForCausalLM",
            "mistral" => "MistralForCausalLM",
            "qwen2" => "Qwen2ForCausalLM",
            "qwen3" => "Qwen3ForCausalLM",
            "qwen3_moe" => "Qwen3MoeForCausalLM",
            _ => unreachable!("test model type"),
        };
        serde_json::json!({
            "model_type": model_type,
            "architectures": [architecture],
            "hidden_size": 8,
            "num_hidden_layers": 1,
            "intermediate_size": 16,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "rms_norm_eps": 0.00001,
            "vocab_size": 16,
            "max_position_embeddings": 32,
            "rope_theta": 10000.0,
            "tie_word_embeddings": false
        })
    }

    fn inspected(
        model_type: &str,
    ) -> (
        tempfile::TempDir,
        ArtifactInspection<ArtifactArchitecturePlan>,
    ) {
        inspected_config(config(model_type))
    }

    fn inspected_config(
        config: serde_json::Value,
    ) -> (
        tempfile::TempDir,
        ArtifactInspection<ArtifactArchitecturePlan>,
    ) {
        let root = tempfile::tempdir().unwrap();
        std::fs::write(
            root.path().join("config.json"),
            serde_json::to_vec(&config).unwrap(),
        )
        .unwrap();
        let resolved = crate::configuration::MODEL_CONFIGURATIONS
            .resolve_safetensors(&config)
            .unwrap();
        let architecture = resolved
            .architecture_plan()
            .safetensors_architecture()
            .unwrap();
        let plan = architecture.checkpoint();
        let mut constraints = plan.common_tensors.iter().collect::<Vec<_>>();
        constraints.extend(
            plan.layout_groups
                .iter()
                .filter_map(|group| group.variants.first())
                .flat_map(|variant| variant.tensors.iter()),
        );
        let tensors = constraints
            .into_iter()
            .map(|constraint| {
                let elements = constraint.shape.iter().product::<usize>();
                (
                    constraint.key.clone(),
                    constraint.shape.clone(),
                    vec![0_u8; elements * 4],
                )
            })
            .collect::<Vec<_>>();
        let views = tensors
            .iter()
            .map(|(name, shape, bytes)| {
                (
                    name.as_str(),
                    TensorView::new(Dtype::F32, shape.clone(), bytes.as_slice()).unwrap(),
                )
            })
            .collect::<Vec<_>>();
        serialize_to_file(views, None, &root.path().join("model.safetensors")).unwrap();
        let inspection = crate::configuration::inspect_artifact(root.path()).unwrap();
        (root, inspection)
    }

    #[test]
    fn exact_llama_and_dense_qwen_artifacts_derive_policy_independent_requirements() {
        for model_type in ["llama", "mistral", "qwen2", "qwen3"] {
            let (_root, inspection) = inspected(model_type);
            let requirements = replicated_text_requirements(&inspection).unwrap();
            let requests = [
                eredu_runtime::ReplicatedTextSelectionRequest::new(
                    eredu_runtime::LayerWeightResidency::FullyResident,
                    eredu_runtime::CacheResidencyPolicy::Device,
                ),
                eredu_runtime::ReplicatedTextSelectionRequest::new(
                    eredu_runtime::LayerWeightResidency::LayerwiseHost(
                        eredu_runtime::LayerwiseLoadOptions::default(),
                    ),
                    eredu_runtime::CacheResidencyPolicy::Device,
                )
                .with_quantization(eredu_core::QuantizationRequest::Affine {
                    group_size: 16,
                    bits: 4,
                }),
                eredu_runtime::ReplicatedTextSelectionRequest::new(
                    eredu_runtime::LayerWeightResidency::DenseDiskStream(
                        eredu_runtime::DenseDiskStreamLoadOptions::default(),
                    ),
                    eredu_runtime::CacheResidencyPolicy::Paged(
                        eredu_runtime::PagedCacheOptions::new(4, 4096, 4096, 1).unwrap(),
                    ),
                )
                .with_topology(eredu_core::ParallelTopology::new(2, 1, 1, 1).unwrap())
                .with_quantization(eredu_core::QuantizationRequest::MxFp4)
                .with_session(eredu_core::SessionCapabilities::new(true, true, true))
                .with_prompt_cache(true)
                .with_exact_completion(true),
            ];
            for request in requests {
                assert_eq!(
                    requirements,
                    replicated_text_requirements(&inspection).unwrap()
                );
                assert!(matches!(
                    request.residency(),
                    eredu_runtime::LayerWeightResidency::FullyResident
                        | eredu_runtime::LayerWeightResidency::LayerwiseHost(_)
                        | eredu_runtime::LayerWeightResidency::DenseDiskStream(_)
                ));
            }
            assert_eq!(requirements.execution_graph().groups().len(), 1);
            assert_eq!(requirements.state_layout().len(), 1);
            assert!(!requirements.parameters().is_empty());
            assert!(requirements.parameters().iter().all(|parameter| matches!(
                parameter.source_encoding(),
                Some(SourceTensorEncoding::Safetensors(StoredDtype::F32)) | None
            )));
            assert!(requirements.parameters().iter().any(|parameter| parameter
                .logical_shape()
                .len()
                == 1
                && parameter
                    .transform_target(eredu_core::QuantizationRequest::Affine {
                        group_size: 16,
                        bits: 4,
                    })
                    .unwrap()
                    .is_none()));
        }
    }

    #[test]
    fn exact_parameter_topology_preserves_geometry_ownership_aliases_and_ties() {
        let (_root, inspection) = inspected("qwen3");
        let requirements = replicated_text_requirements(&inspection).unwrap();
        let mut identities = BTreeSet::new();
        for parameter in requirements.parameters() {
            assert!(identities.insert(parameter.name()));
            assert!(!parameter.logical_shape().is_empty());
            match parameter.presence() {
                ReplicatedTextParameterPresence::Required
                | ReplicatedTextParameterPresence::OptionalPresent => {
                    let source = parameter.sources().first().unwrap();
                    let descriptor = inspection.tensors().get(source).unwrap();
                    assert_eq!(
                        parameter.physical_shape(),
                        Some(descriptor.shape.as_slice())
                    );
                    assert!(parameter.source_encoding().is_some());
                    let physical = parameter.physical_sources();
                    assert_eq!(physical.len(), 1);
                    assert_eq!(physical[0].tensor(), source);
                    assert_eq!(physical[0].output(), source);
                    let shards = inspection.safetensors_shards().unwrap();
                    let expected_shard = shards
                        .tensor_locations()
                        .and_then(|locations| locations.get(source))
                        .unwrap_or(&shards.payload_paths()[0]);
                    assert_eq!(physical[0].shard(), expected_shard);
                }
                ReplicatedTextParameterPresence::OptionalAbsent => {
                    assert!(parameter.sources().is_empty());
                    assert!(parameter.physical_sources().is_empty());
                    assert!(parameter.source_encoding().is_none());
                }
                ReplicatedTextParameterPresence::Tied { .. }
                | ReplicatedTextParameterPresence::Derived { .. } => {}
                _ => unreachable!("test covers every current presence category"),
            }
            assert_eq!(
                matches!(
                    parameter.transform_constraint(),
                    ParameterTransformConstraint::Linear { .. }
                ),
                parameter.role() == ReplicatedTextParameterRole::LinearWeight
            );
            if let ReplicatedTextParameterOwner::ExecutionUnit { group, unit } = parameter.owner() {
                assert_eq!(group, crate::decoder::TEXT_DECODER_EXECUTION_GROUP);
                assert!(*unit < requirements.execution_units().len());
            }
        }
        assert!(requirements.parameters().iter().any(|parameter| {
            parameter.logical_shape().len() == 1
                && parameter.role() == ReplicatedTextParameterRole::Normalization
                && parameter.transform_constraint() == ParameterTransformConstraint::None
        }));
        let mut tied_config = config("llama");
        tied_config["tie_word_embeddings"] = serde_json::json!(true);
        let (_root, tied_inspection) = inspected_config(tied_config);
        let tied = replicated_text_requirements(&tied_inspection).unwrap();
        let output = tied
            .parameters()
            .iter()
            .find(|parameter| parameter.name() == "lm_head.weight")
            .unwrap();
        assert!(matches!(
            output.presence(),
            ReplicatedTextParameterPresence::Tied { target }
                if target == "model.embed_tokens.weight"
        ));
        assert!(output.sources().is_empty());
        assert!(output.physical_sources().is_empty());
    }

    #[test]
    fn routed_qwen_is_rejected_before_backend_construction() {
        let mut config = config("qwen3_moe");
        config["architectures"] = serde_json::json!(["Qwen3MoeForCausalLM"]);
        config["num_experts"] = serde_json::json!(2);
        config["num_experts_per_tok"] = serde_json::json!(1);
        config["moe_intermediate_size"] = serde_json::json!(8);
        let resolved = crate::configuration::MODEL_CONFIGURATIONS
            .resolve_safetensors(&config)
            .unwrap();
        assert!(matches!(
            eligible_config(resolved.architecture_plan()),
            Err(ReplicatedTextIneligibility::Routed)
        ));
    }

    #[test]
    fn embedded_prediction_is_a_distinct_ineligibility() {
        let config = serde_json::json!({
            "architectures": ["DeepseekV3ForCausalLM"],
            "model_type": "deepseek_v3",
            "hidden_size": 16,
            "intermediate_size": 32,
            "moe_intermediate_size": 8,
            "num_hidden_layers": 4,
            "num_attention_heads": 2,
            "vocab_size": 128,
            "max_position_embeddings": 4096,
            "q_lora_rank": 4,
            "kv_lora_rank": 4,
            "qk_nope_head_dim": 6,
            "qk_rope_head_dim": 2,
            "v_head_dim": 8,
            "first_k_dense_replace": 1,
            "moe_layer_freq": 2,
            "n_routed_experts": 8,
            "n_shared_experts": 1,
            "num_experts_per_tok": 2,
            "n_group": 2,
            "topk_group": 1,
            "topk_method": "noaux_tc",
            "scoring_func": "sigmoid",
            "norm_topk_prob": true,
            "routed_scaling_factor": 1.0,
            "tie_word_embeddings": false,
            "attention_dropout": 0.0,
            "hidden_act": "silu",
            "num_nextn_predict_layers": 1
        });
        let resolved = crate::configuration::MODEL_CONFIGURATIONS
            .resolve_safetensors(&config)
            .unwrap();
        assert!(matches!(
            eligible_config(resolved.architecture_plan()),
            Err(ReplicatedTextIneligibility::EmbeddedPrediction)
        ));
    }

    #[test]
    fn partitioned_topology_remains_a_caller_selection_choice() {
        let (_root, inspection) = inspected("llama");
        let requirements = replicated_text_requirements(&inspection).unwrap();
        let request = eredu_runtime::ReplicatedTextSelectionRequest::new(
            eredu_runtime::LayerWeightResidency::FullyResident,
            eredu_runtime::CacheResidencyPolicy::Device,
        )
        .with_topology(eredu_core::ParallelTopology::new(2, 1, 1, 1).unwrap());
        assert!(!request.topology().unwrap().is_replicated());
        assert_eq!(
            requirements,
            replicated_text_requirements(&inspection).unwrap()
        );
    }
}
