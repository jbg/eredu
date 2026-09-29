//! Bound vision owners derived from one retained source-free declaration.
use super::*;
use crate::qwen::vision::VisionConfig;
use crate::qwen4_exp::config::MediaTokens;

/// Prepared image/video tower retaining its source-free contract and exact source.
#[derive(Clone)]
pub struct PreparedVision {
    pub(super) plan: Arc<VisionPlan>,
    pub(super) source: SharedCheckpointSource,
    pub(super) static_parameters: PreparedParameters,
    pub(super) blocks: Vec<PreparedParameters>,
}
impl PreparedTarget {
    /// Couples shared media ingress to retained target geometry and a prepared tower.
    pub fn media_ingress(
        &self,
        vision: PreparedVision,
    ) -> Result<super::super::media::MediaIngress, super::super::media::MediaInputError> {
        use super::super::media::MediaInputError;
        let validate = |token| {
            vision
                .plan()
                .validate_target(&self.spec.config, token)
                .map_err(|error| match error {
                    PreparationError::VisionMismatch { field } => MediaInputError::Geometry(field),
                    _ => MediaInputError::Geometry("invalid retained vision contract"),
                })
        };
        validate(None)?;
        for table in self.tables.values() {
            if let checkpoint::NGramControls::Gguf { metadata, .. } = &table.controls {
                if let Some(value) = metadata.get("qwen4exp.ple.image_token_id") {
                    let token = value
                        .as_i64()
                        .and_then(|value| u32::try_from(value).ok())
                        .ok_or(MediaInputError::Geometry("invalid n-gram image token ID"))?;
                    validate(Some(token))?;
                }
            }
        }
        super::super::media::MediaIngress::new(&self.spec, vision)
    }

    /// Acquires the integrated tower using already retained exact source identities.
    pub fn vision(&self) -> Result<PreparedVision, PreparationError> {
        let physical = self
            .artifact
            .source_keys()
            .into_iter()
            .filter(|key| key.starts_with("model.visual."))
            .map(|key| {
                crate::replicated_text::exact_physical_source(self.artifact.as_ref(), &key)
                    .map(|physical| (key, physical))
                    .map_err(|error| PreparationError::Contract(error.to_string()))
            })
            .collect::<Result<BTreeMap<_, _>, _>>()?;
        VisionPlan::safetensors(&self.spec.config, self.artifact.as_ref(), physical)?
            .bind(self.artifact.clone())
    }
}
impl PreparedVision {
    /// Retained source-free declaration used to prepare this exact role.
    pub fn plan(&self) -> &VisionPlan {
        &self.plan
    }
    /// Retains optional SafeTensors processor artifacts without reopening files.
    pub fn with_processor_json(
        mut self,
        image: Option<&[u8]>,
        video: Option<&[u8]>,
    ) -> Result<Self, PreparationError> {
        self.plan = Arc::new((*self.plan).clone().with_processor_json(image, video)?);
        Ok(self)
    }
    /// Optional retained raw-media processing policy.
    pub fn processor(&self) -> Option<&crate::processor_plan::QwenProcessorPlan> {
        self.plan.processor()
    }
    /// Exact shared vision policy and per-matrix encodings.
    pub fn config(&self) -> &VisionConfig {
        self.plan.config()
    }
    /// Token identities supplied by target/tokenizer policy.
    pub fn media_tokens(&self) -> &MediaTokens {
        self.plan.media_tokens()
    }
    /// Exact retained projector source with original physical provenance.
    pub fn artifact(&self) -> &SharedCheckpointSource {
        &self.source
    }
    /// Patch/position ingress and output merger parameters.
    pub fn static_parameters(&self) -> &PreparedParameters {
        &self.static_parameters
    }
    /// One independently materializable shared vision transformer block.
    pub fn block(&self, layer: usize) -> Result<&PreparedParameters, PreparationError> {
        self.blocks.get(layer).ok_or_else(|| {
            PreparationError::Contract("vision block outside declared schedule".into())
        })
    }
}
