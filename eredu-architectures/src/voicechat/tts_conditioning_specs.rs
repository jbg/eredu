//! Shared character, prompt-ingress and fusion parameter declarations.
use super::{
    tts_conditioning::CharacterDimensions,
    tts_conditioning_layered::ConditioningDimensions,
    tts_mog::{linear_spec, parameter, TtsError},
};
use eredu_nn::{
    DeclareLogicalParameters, EmbeddingSpec, Error, LinearSpec, LogicalParameterDeclaration,
    LogicalParameterRole, NormalizationConstructionSpec, NormalizationScale, Parameter,
    ParameterSpec, Tensor,
};

/// Alphabet cardinality required by the released checkpoint's character tables.
pub(crate) const RELEASED_CHARACTERS: i32 = 256;
#[derive(Debug, Clone)]
pub(crate) struct SlotSpec {
    pub parameter: ParameterSpec,
    pub shape: Vec<i32>,
    pub role: LogicalParameterRole,
}
impl SlotSpec {
    pub fn into_parameter<T: Tensor>(self, context: &T::Context) -> Result<Parameter<T>, Error> {
        Parameter::unloaded(self.parameter, &self.shape, context)
    }
}
impl DeclareLogicalParameters for SlotSpec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        Ok(vec![LogicalParameterDeclaration::new(
            &self.parameter,
            &self.shape,
            self.role,
        )?])
    }
}
fn validate(spec: &impl DeclareLogicalParameters) -> Result<(), Error> {
    eredu_nn::validate_logical_parameter_topology(&spec.logical_parameters()?)
        .map_err(Error::backend_source)
}
fn embedding(name: &str, rows: i32, width: i32) -> Result<EmbeddingSpec, Error> {
    Ok(EmbeddingSpec {
        vocabulary: rows,
        dimensions: width,
        weight: parameter(name, "voicechat.tts.character.static")?,
        format: eredu_nn::LinearFormatSpec::unscaled(eredu_checkpoint::LinearFormat::Dense)?,
    })
}
/// Character lookup/projection and complete optional continuation/special tables.
#[derive(Debug, Clone)]
pub struct CharacterSpec {
    pub(crate) dimensions: CharacterDimensions,
    pub(crate) tokens: EmbeddingSpec,
    pub(crate) projection: LinearSpec,
    pub(crate) continuation: Option<EmbeddingSpec>,
    pub(crate) special: Option<EmbeddingSpec>,
}
impl CharacterSpec {
    /// Declares global table ownership without constructing a bidirectional backbone.
    pub fn new(d: CharacterDimensions) -> Result<Self, TtsError> {
        d.validate()?;
        let spec = Self {
            dimensions: d,
            tokens: embedding(
                "embed_subword.embed_tokens.weight",
                d.characters + 1,
                d.width,
            )?,
            projection: linear_spec(
                "embed_subword.proj_embedding",
                "voicechat.tts.character.static",
                d.width,
                d.output_width,
            )?,
            continuation: d
                .use_continuation
                .then(|| embedding("embed_subword.subword_flag_emb.cont_emb.weight", 2, d.width))
                .transpose()?,
            special: d
                .use_special
                .then(|| embedding("embed_subword.bos_eos_emb.special_emb.weight", 3, d.width))
                .transpose()?,
        };
        validate(&spec)?;
        Ok(spec)
    }
    /// Projects independent table-feature and output-feature slices; flags remain replicated.
    pub fn local(mut self, width: i32, output: i32) -> Result<Self, TtsError> {
        if width <= 0
            || width > self.dimensions.width
            || output <= 0
            || output > self.dimensions.output_width
        {
            return Err(TtsError::Configuration("invalid character partition"));
        }
        self.tokens.dimensions = width;
        self.projection.output = output;
        validate(&self)?;
        Ok(self)
    }
}
impl DeclareLogicalParameters for CharacterSpec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        let mut p = self.tokens.logical_parameters()?;
        p.extend(self.projection.logical_parameters()?);
        for flag in [&self.continuation, &self.special].into_iter().flatten() {
            p.extend(flag.logical_parameters()?);
        }
        Ok(p)
    }
}
/// BOS/null features, frozen speaker sequence, and the right-multiplied prompt matrix.
#[derive(Debug, Clone)]
pub struct IngressSpec {
    pub(crate) bos: SlotSpec,
    pub(crate) null: SlotSpec,
    pub(crate) aria: SlotSpec,
    pub(crate) projection: SlotSpec,
}
impl IngressSpec {
    /// Declares frozen speaker ownership and the matrix's stored input-by-output orientation.
    pub fn new(hidden: i32, frames: i32) -> Result<Self, TtsError> {
        if hidden <= 0 || frames <= 0 {
            return Err(TtsError::Configuration("invalid TTS ingress geometry"));
        }
        let mut aria = ParameterSpec::trainable("tts_model.audio_prompt_latents.Aria")
            .map_err(Error::backend)?;
        aria.trainable = false;
        aria.group = Some("voicechat.tts.speaker.aria".into());
        let mut projection = parameter(
            "audio_prompt_projection_W",
            "voicechat.tts.speaker.projection",
        )?;
        projection.trainable = false;
        let spec = Self {
            bos: SlotSpec {
                parameter: parameter("bos_emb", "voicechat.tts.ingress")?,
                shape: vec![hidden],
                role: LogicalParameterRole::ConditioningEmbedding,
            },
            null: SlotSpec {
                parameter: parameter("null_emb", "voicechat.tts.ingress")?,
                shape: vec![hidden],
                role: LogicalParameterRole::ConditioningEmbedding,
            },
            aria: SlotSpec {
                parameter: aria,
                shape: vec![1, frames, hidden],
                role: LogicalParameterRole::ConditioningEmbedding,
            },
            projection: SlotSpec {
                parameter: projection,
                shape: vec![hidden, hidden],
                role: LogicalParameterRole::LinearWeight,
            },
        };
        validate(&spec)?;
        Ok(spec)
    }
    /// Selects stored output columns while retaining complete prompt and control features.
    pub fn local(mut self, output: i32) -> Result<Self, TtsError> {
        if output <= 0 || output > self.bos.shape[0] {
            return Err(TtsError::Configuration("invalid TTS ingress output shard"));
        }
        self.projection.shape[1] = output;
        validate(&self)?;
        Ok(self)
    }
}
impl DeclareLogicalParameters for IngressSpec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        let mut p = Vec::new();
        for slot in [&self.bos, &self.null, &self.aria, &self.projection] {
            p.extend(slot.logical_parameters()?);
        }
        Ok(p)
    }
}
/// Audio/text affine projections, sigmoid controls and offset RMS gain.
#[derive(Debug, Clone)]
pub struct FusionSpec {
    pub(crate) audio: LinearSpec,
    pub(crate) text: LinearSpec,
    pub(crate) gate: SlotSpec,
    pub(crate) residual_scale: SlotSpec,
    pub(crate) norm: NormalizationConstructionSpec,
}
impl FusionSpec {
    /// Declares the released scalar/vector distinction and gain arithmetic.
    pub fn new(width: i32) -> Result<Self, TtsError> {
        if width <= 0 {
            return Err(TtsError::Configuration("invalid gated fusion width"));
        }
        let root = "gated_fusion_audio_text";
        let group = "voicechat.tts.fusion";
        let linear = |field: &str| -> Result<LinearSpec, Error> {
            let mut p = linear_spec(&format!("{root}.{field}"), group, width, width)?;
            p.bias = Some(parameter(&format!("{root}.{field}.bias"), group)?);
            Ok(p)
        };
        let spec = Self {
            audio: linear("audio_proj")?,
            text: linear("text_proj")?,
            gate: SlotSpec {
                parameter: parameter(&format!("{root}.gate"), group)?,
                shape: vec![width],
                role: LogicalParameterRole::FeatureGate,
            },
            residual_scale: SlotSpec {
                parameter: parameter(&format!("{root}.residual_scale"), group)?,
                shape: vec![],
                role: LogicalParameterRole::ResidualScale,
            },
            norm: NormalizationConstructionSpec {
                reduction: eredu_nn::NormalizationReduction::Native,
                dimensions: width,
                epsilon: 1e-6,
                groups: Some(1),
                scale: NormalizationScale::LearnedOffset {
                    weight: parameter(&format!("{root}.final_norm.weight"), group)?,
                    offset: 1.0,
                },
            },
        };
        validate(&spec)?;
        Ok(spec)
    }
    /// Selects projection output rows while keeping gates and normalized features replicated.
    pub fn local(mut self, output: i32) -> Result<Self, TtsError> {
        if output <= 0 || output > self.gate.shape[0] {
            return Err(TtsError::Configuration("invalid fusion output shard"));
        }
        self.audio.output = output;
        self.text.output = output;
        validate(&self)?;
        Ok(self)
    }
}
impl DeclareLogicalParameters for FusionSpec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        let mut p = self.audio.logical_parameters()?;
        p.extend(self.text.logical_parameters()?);
        p.extend(self.gate.logical_parameters()?);
        p.extend(self.residual_scale.logical_parameters()?);
        p.extend(self.norm.logical_parameters()?);
        Ok(p)
    }
}
#[derive(Debug, Clone)]
pub(crate) struct Declaration {
    pub character: CharacterSpec,
    pub ingress: IngressSpec,
    pub fusion: FusionSpec,
}
impl Declaration {
    pub fn new(d: ConditioningDimensions) -> Result<Self, TtsError> {
        ConditioningDimensions::new(d.character, d.speaker_frames)?;
        let spec = Self {
            character: CharacterSpec::new(d.character)?,
            ingress: IngressSpec::new(d.character.output_width, d.speaker_frames)?,
            fusion: FusionSpec::new(d.character.output_width)?,
        };
        validate(&spec)?;
        Ok(spec)
    }
}
impl DeclareLogicalParameters for Declaration {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        let mut p = self.character.logical_parameters()?;
        p.extend(self.ingress.logical_parameters()?);
        p.extend(self.fusion.logical_parameters()?);
        Ok(p)
    }
}

/// Retained character, ingress and fusion scopes with their execution projection.
/// Parameter ownership is independent of conditioning phase invocation and mutable state.
#[derive(Debug, Clone)]
pub struct ConditioningDeclaration {
    spec: Declaration,
    definition: eredu_runtime::ComponentDefinition,
}
impl ConditioningDeclaration {
    /// Declares complete conditioning geometry without native allocation.
    pub fn new(dimensions: ConditioningDimensions) -> Result<Self, TtsError> {
        let spec = Declaration::new(dimensions)?;
        let definition = super::tts_conditioning_layered::declared_component_definition(&spec)?;
        Ok(Self { spec, definition })
    }
    /// Character and fixed prompt geometry from the retained neural slots.
    pub fn dimensions(&self) -> ConditioningDimensions {
        ConditioningDimensions {
            character: self.spec.character.dimensions,
            speaker_frames: self.spec.ingress.aria.shape[1],
        }
    }
    /// Parameter ownership, placement semantics and independently leased boundaries.
    pub fn definition(&self) -> &eredu_runtime::ComponentDefinition {
        &self.definition
    }
    /// Resolves coherent character and speech slices from canonical parameters.
    pub fn local_dimensions(
        &self,
        layout: &eredu_runtime::LocalModelLayout,
    ) -> Result<super::tts_conditioning_layered::ConditioningLocalDimensions, TtsError> {
        super::tts_conditioning_layered::declared_local_dimensions(&self.spec, layout)
    }
    pub(crate) fn spec(&self) -> &Declaration {
        &self.spec
    }
    pub(crate) fn into_parts(self) -> (Declaration, eredu_runtime::ComponentDefinition) {
        (self.spec, self.definition)
    }
}
impl eredu_runtime::ComponentPreparationDeclaration for ConditioningDeclaration {
    fn parameters(&self) -> &eredu_runtime::ArchitectureParameterDescription {
        self.definition.parameters()
    }
}

/// Every admitted configuration validates against this immutable published profile.
/// The scope carries no artifacts, mutable invocation state or backend resources.
pub(crate) fn released_declaration() -> Result<&'static ConditioningDeclaration, TtsError> {
    static DECLARATION: std::sync::OnceLock<Result<ConditioningDeclaration, String>> =
        std::sync::OnceLock::new();
    DECLARATION
        .get_or_init(|| {
            let config = super::config::released_configuration().map_err(|e| e.to_string())?;
            let frames = super::tts_prompt::AriaPromptPlan::from_config(&config, usize::MAX)
                .map_err(|e| e.to_string())?
                .frames();
            let dimensions =
                ConditioningDimensions::from_config(&config.tts, RELEASED_CHARACTERS, frames)
                    .map_err(|e| e.to_string())?;
            ConditioningDeclaration::new(dimensions).map_err(|e| e.to_string())
        })
        .as_ref()
        .map_err(|reason| eredu_nn::Error::backend(reason.clone()).into())
}
