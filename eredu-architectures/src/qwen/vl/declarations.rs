//! Composite policy and retained text and vision parameter authority.
use super::ModelArgs;
use eredu_nn::Error;
use std::{ops::Deref, sync::Arc};

/// Validated VL invocation policy with immutable text construction scopes.
/// Vision slots use the shared vision component contracts.
#[derive(Debug, Clone)]
pub struct Declaration {
    args: ModelArgs,
    text_scope: Arc<crate::qwen::Declaration>,
    vision_scope: Arc<crate::qwen::vision::Declaration>,
    state: eredu_runtime::StateLayout,
}
impl Declaration {
    /// Declares the text component once at its composite architecture boundary.
    pub fn new(args: ModelArgs) -> Result<Self, Error> {
        let text = Arc::new(crate::qwen::Declaration::new(args.text.clone())?);
        let vision = Arc::new(crate::qwen::vision::Declaration::new(
            args.vision.clone(),
            "model.visual",
        )?);
        Self::from_components(args, text, vision)
    }
    pub(crate) fn from_components(
        args: ModelArgs,
        text: Arc<crate::qwen::Declaration>,
        vision: Arc<crate::qwen::vision::Declaration>,
    ) -> Result<Self, Error> {
        text.common().validate_for(&args.text)?;
        if vision.config() != &args.vision || vision.root() != "model.visual" {
            return Err(Error::backend(
                "VL vision declaration differs from component policy",
            ));
        }
        args.vision
            .validate_for(crate::qwen::vision::VisionMode::DeepStack)
            .map_err(Error::backend_source)?;
        if args.vision.out_hidden_size != args.text.hidden_size {
            return Err(Error::backend("VL vision output differs from text width"));
        }
        let state = super::state_layout_with_text_state(&text, text.common().state_layout())
            .map_err(Error::backend_source)?;
        Ok(Self {
            args,
            text_scope: text,
            vision_scope: vision,
            state,
        })
    }
    /// Complete normalized invocation and media policy.
    pub fn args(&self) -> &ModelArgs {
        &self.args
    }
    /// Exact common and provider slots of the text component.
    pub fn text_declaration(&self) -> &Arc<crate::qwen::Declaration> {
        &self.text_scope
    }
    /// Shared patch, position, block and merger construction authority.
    pub fn vision_declaration(&self) -> &Arc<crate::qwen::vision::Declaration> {
        &self.vision_scope
    }
    /// Global cache and persisted media-position state.
    pub fn state_layout(&self) -> &eredu_runtime::StateLayout {
        &self.state
    }
    /// Projects encodings after rejecting changes to non-format composite policy.
    pub fn with_formats(&self, args: ModelArgs) -> Result<Self, Error> {
        let mut policy = args.clone();
        policy.text.quantization = self.args.text.quantization;
        policy.text.quantized_weights = self.args.text.quantized_weights.clone();
        policy.text.quantized_weight_configs = self.args.text.quantized_weight_configs.clone();
        policy.vision.linear_formats = self.args.vision.linear_formats.clone();
        if super::prompt_cache_architecture_fingerprint(&policy)
            != super::prompt_cache_architecture_fingerprint(&self.args)
        {
            return Err(Error::backend(
                "VL format projection changes invocation policy",
            ));
        }
        Self::from_components(
            args.clone(),
            Arc::new(self.text_scope.with_formats(args.text)?),
            Arc::new(self.vision_scope.with_formats(args.vision)?),
        )
    }
}
impl Deref for Declaration {
    type Target = ModelArgs;
    fn deref(&self) -> &Self::Target {
        &self.args
    }
}
