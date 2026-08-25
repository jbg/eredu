//! Architecture-erased model, cache, and generation dispatch.

use std::path::Path;

use eredu_core::cache::{PromptCacheDescriptor, PromptCacheManifest, PromptCacheOptions};
use eredu_core::MtpCapability;
use safemlx::{error::Exception, Array, Stream};

use crate::backend::error::Error;
use crate::backend::runtime::media::input;
use crate::composition::gpt_oss;
use eredu_architectures::{kimi_linear, ModelKind};
use eredu_core::cache::LayerCachePolicy;
use eredu_core::LayerSchedule;
use eredu_runtime::ActivationObserver as RuntimeActivationObserver;
use eredu_runtime::CacheResidencyReport;
use eredu_runtime::{CacheResidencyPolicy, PagedCacheOptions};

/// Admitted architecture identity checked against the concrete model variant.
#[derive(Clone, Copy)]
pub struct AdmittedModelKind(ModelKind);

impl AdmittedModelKind {
    fn new(artifact_kind: ModelKind, supported_kinds: &[ModelKind]) -> Result<Self, Error> {
        if !supported_kinds.contains(&artifact_kind) {
            return Err(Error::ArchitectureModel(format!(
                "complete model identity mismatch: artifact admitted {}, but the concrete model variant supports {}",
                artifact_kind.canonical_name(),
                supported_kinds
                    .iter()
                    .map(|kind| kind.canonical_name())
                    .collect::<Vec<_>>()
                    .join(" or ")
            )));
        }
        Ok(Self(artifact_kind))
    }

    const fn get(self) -> ModelKind {
        self.0
    }
}

/// Loaded model value for any architecture supported by this crate.
///
/// Construction is restricted to the checked family-specific constructors so
/// reporting and input dispatch cannot disagree with the concrete variant.
pub enum Model {
    /// Neutral DeepSeek-V3/V4 architecture with policy-selected residency.
    DeepSeek(
        AdmittedModelKind,
        Box<crate::composition::deepseek::DeepSeekModel>,
    ),
    /// Gemma 4 text and multimodal model.
    Gemma4(AdmittedModelKind, crate::composition::gemma4::Gemma4Model),
    /// OpenAI GPT-OSS model.
    GptOss(AdmittedModelKind, crate::composition::gpt_oss::GptOssModel),
    /// Moonshot Kimi Linear hybrid KDA/MLA sparse decoder.
    KimiLinear(
        AdmittedModelKind,
        crate::composition::kimi_linear::KimiLinearModel,
    ),
    /// Thinking Machines Lab Inkling multimodal model.
    Inkling(AdmittedModelKind, crate::composition::inkling::InklingModel),
    /// Llama-compatible dense model.
    Llama(AdmittedModelKind, crate::composition::llama::LlamaModel),
    /// Meta Muse-Glimmer dense multimodal model.
    MuseGlimmer(
        AdmittedModelKind,
        crate::composition::muse_glimmer::MuseGlimmerModel,
    ),
    /// Liquid AI LFM2/LFM2.5 model.
    Lfm2(AdmittedModelKind, crate::composition::lfm2::Lfm2Model),
    /// Nemotron-H hybrid model.
    NemotronH(
        AdmittedModelKind,
        crate::composition::nemotron_h::NemotronHModel,
    ),
    /// Neutral Qwen2/Qwen2.5/Qwen3/Qwen3-MoE model.
    Qwen(AdmittedModelKind, crate::composition::qwen::QwenModel),
    /// Qwen3-Next model.
    Qwen3Next(
        AdmittedModelKind,
        crate::composition::qwen::hybrid::QwenHybridModel,
    ),
    /// Qwen3-VL multimodal model.
    Qwen3Vl(AdmittedModelKind, crate::composition::qwen::vl::QwenVlModel),
    /// Qwen3-VL-MoE multimodal model.
    Qwen3VlMoe(AdmittedModelKind, crate::composition::qwen::vl::QwenVlModel),
    /// Qwen3.5 dense or MoE model, optionally multimodal.
    Qwen35(
        AdmittedModelKind,
        crate::composition::qwen::hybrid::QwenHybridModel,
    ),
}

impl Model {
    pub(super) fn deepseek(
        kind: ModelKind,
        model: Box<crate::composition::deepseek::DeepSeekModel>,
    ) -> Result<Self, Error> {
        let identity =
            AdmittedModelKind::new(kind, &[ModelKind::DeepSeekV3, ModelKind::DeepSeekV4])?;
        Ok(Self::DeepSeek(identity, model))
    }

    pub(super) fn gemma4(
        kind: ModelKind,
        model: crate::composition::gemma4::Gemma4Model,
    ) -> Result<Self, Error> {
        let identity = AdmittedModelKind::new(kind, &[ModelKind::Gemma4])?;
        Ok(Self::Gemma4(identity, model))
    }

    pub(super) fn gpt_oss(
        kind: ModelKind,
        model: crate::composition::gpt_oss::GptOssModel,
    ) -> Result<Self, Error> {
        let identity = AdmittedModelKind::new(kind, &[ModelKind::GptOss])?;
        Ok(Self::GptOss(identity, model))
    }

    pub(super) fn kimi_linear(
        kind: ModelKind,
        model: crate::composition::kimi_linear::KimiLinearModel,
    ) -> Result<Self, Error> {
        let identity = AdmittedModelKind::new(kind, &[ModelKind::KimiLinear])?;
        Ok(Self::KimiLinear(identity, model))
    }

    pub(super) fn inkling(
        kind: ModelKind,
        model: crate::composition::inkling::InklingModel,
    ) -> Result<Self, Error> {
        let identity = AdmittedModelKind::new(kind, &[ModelKind::Inkling])?;
        Ok(Self::Inkling(identity, model))
    }

    pub(super) fn llama(
        kind: ModelKind,
        model: crate::composition::llama::LlamaModel,
    ) -> Result<Self, Error> {
        let identity = AdmittedModelKind::new(kind, &[ModelKind::Llama])?;
        Ok(Self::Llama(identity, model))
    }

    pub(super) fn muse_glimmer(
        kind: ModelKind,
        model: crate::composition::muse_glimmer::MuseGlimmerModel,
    ) -> Result<Self, Error> {
        let identity = AdmittedModelKind::new(kind, &[ModelKind::MuseGlimmer])?;
        Ok(Self::MuseGlimmer(identity, model))
    }

    pub(super) fn lfm2(
        kind: ModelKind,
        model: crate::composition::lfm2::Lfm2Model,
    ) -> Result<Self, Error> {
        let identity = AdmittedModelKind::new(kind, &[ModelKind::Lfm2])?;
        Ok(Self::Lfm2(identity, model))
    }

    pub(super) fn nemotron_h(
        kind: ModelKind,
        model: crate::composition::nemotron_h::NemotronHModel,
    ) -> Result<Self, Error> {
        let identity = AdmittedModelKind::new(kind, &[ModelKind::NemotronH])?;
        Ok(Self::NemotronH(identity, model))
    }

    pub(super) fn qwen(
        kind: ModelKind,
        model: crate::composition::qwen::QwenModel,
    ) -> Result<Self, Error> {
        let identity = AdmittedModelKind::new(kind, &[ModelKind::Qwen2, ModelKind::Qwen3])?;
        Ok(Self::Qwen(identity, model))
    }

    pub(super) fn qwen3_next(
        kind: ModelKind,
        model: crate::composition::qwen::hybrid::QwenHybridModel,
    ) -> Result<Self, Error> {
        let identity = AdmittedModelKind::new(kind, &[ModelKind::Qwen3Next])?;
        Ok(Self::Qwen3Next(identity, model))
    }

    pub(super) fn qwen3_vl(
        kind: ModelKind,
        model: crate::composition::qwen::vl::QwenVlModel,
    ) -> Result<Self, Error> {
        let identity = AdmittedModelKind::new(kind, &[ModelKind::Qwen3Vl])?;
        Ok(Self::Qwen3Vl(identity, model))
    }

    pub(super) fn qwen3_vl_moe(
        kind: ModelKind,
        model: crate::composition::qwen::vl::QwenVlModel,
    ) -> Result<Self, Error> {
        let identity = AdmittedModelKind::new(kind, &[ModelKind::Qwen3VlMoe])?;
        Ok(Self::Qwen3VlMoe(identity, model))
    }

    pub(super) fn qwen35(
        kind: ModelKind,
        model: crate::composition::qwen::hybrid::QwenHybridModel,
    ) -> Result<Self, Error> {
        let identity = AdmittedModelKind::new(kind, &[ModelKind::Qwen35])?;
        Ok(Self::Qwen35(identity, model))
    }

    /// Returns architecture-neutral rank-local placement information when this
    /// model was loaded through generalized parallel execution groups.
    pub fn parallel_info(
        &self,
    ) -> Option<&eredu_runtime::ParallelModelInfo<crate::backend::MlxParallelContext>> {
        match self {
            Self::DeepSeek(_, _) => None,
            Self::Llama(_, model) => model.parallel_info(),
            Self::MuseGlimmer(_, model) => model.parallel_info(),
            Self::GptOss(_, model) => model.parallel_info(),
            Self::Qwen(_, model) => model.parallel_info(),
            Self::KimiLinear(_, model) => model.parallel_info(),
            Self::Lfm2(_, model) => model.parallel_info(),
            Self::NemotronH(_, model) => model.parallel_info(),
            Self::Qwen3Next(_, model) | Self::Qwen35(_, model) => model.parallel_info(),
            Self::Qwen3Vl(_, model) | Self::Qwen3VlMoe(_, model) => model.parallel_info(),
            Self::Gemma4(_, model) => model.parallel_info(),
            Self::Inkling(_, model) => model.parallel_info(),
        }
    }

    /// Reports how this model architecture exposes MTP weights.
    pub fn mtp_capability(&self) -> MtpCapability {
        self.architecture_capability_estimate()
            .ok()
            .and_then(|estimate| estimate.mtp_checkpoint_kind())
            .map_or(MtpCapability::Unavailable, |checkpoint| {
                MtpCapability::Ready { checkpoint }
            })
    }

    /// Returns residency telemetry when this model uses bounded layer execution.
    pub fn residency_report(&self) -> Result<Option<eredu_runtime::ResidencyReport>, Error> {
        match self {
            Self::DeepSeek(_, model) => Ok(Some(model.residency_report()?)),
            Self::Gemma4(_, model) => model.residency_report(),
            Self::Inkling(_, model) => model.residency_report(),
            Self::KimiLinear(_, model) => Ok(Some(model.residency_report()?)),
            Self::Llama(_, model) => model.residency_report(),
            Self::GptOss(_, model) => Ok(Some(model.residency_report()?)),
            Self::Lfm2(_, model) => Ok(Some(model.residency_report()?)),
            Self::NemotronH(_, model) => Ok(Some(model.residency_report()?)),
            Self::Qwen3Next(_, model) | Self::Qwen35(_, model) => {
                Ok(Some(model.residency_report()?))
            }
            Self::Qwen(_, model) => model.residency_report(),
            Self::MuseGlimmer(_, model) => model.residency_report(),
            Self::Qwen3Vl(_, model) | Self::Qwen3VlMoe(_, model) => {
                Ok(Some(model.residency_report()?))
            }
        }
    }

    /// Returns experimental dense-stream telemetry when enabled.
    pub fn dense_stream_report(
        &self,
    ) -> Result<Option<eredu_runtime::DenseDiskStreamReport>, Error> {
        match self {
            Self::DeepSeek(_, model) => model.dense_stream_report(),
            Self::Gemma4(_, model) => model.dense_stream_report(),
            Self::Inkling(_, model) => model.dense_stream_report(),
            Self::KimiLinear(_, model) => model.dense_stream_report(),
            Self::Llama(_, model) => model.dense_stream_report(),
            Self::GptOss(_, model) => model.dense_stream_report(),
            Self::Lfm2(_, model) => model.dense_stream_report(),
            Self::NemotronH(_, model) => model.dense_stream_report(),
            Self::Qwen3Next(_, model) | Self::Qwen35(_, model) => model.dense_stream_report(),
            Self::Qwen(_, model) => model.dense_stream_report(),
            Self::MuseGlimmer(_, model) => model.dense_stream_report(),
            Self::Qwen3Vl(_, model) | Self::Qwen3VlMoe(_, model) => model.dense_stream_report(),
        }
    }

    /// Returns sparse routed-expert cache telemetry when enabled.
    pub fn expert_cache_report(
        &self,
    ) -> Result<Option<crate::backend::runtime::residency::expert_cache::ExpertCacheReport>, Error>
    {
        match self {
            Self::DeepSeek(_, model) => model.expert_cache_report(),
            Self::Gemma4(_, model) => model.expert_cache_report(),
            Self::KimiLinear(_, model) => model.expert_cache_report(),
            Self::GptOss(_, model) => model.expert_cache_report(),
            Self::Inkling(_, model) => model.expert_cache_report(),
            Self::Lfm2(_, model) => model.expert_cache_report(),
            Self::NemotronH(_, model) => model.expert_cache_report(),
            Self::Qwen(_, model) => model.expert_cache_report(),
            Self::Qwen3Next(_, model) | Self::Qwen35(_, model) => model.expert_cache_report(),
            Self::Qwen3VlMoe(_, model) => model.expert_cache_report(),
            Self::MuseGlimmer(_, model) => model.expert_cache_report(),
            _ => Ok(None),
        }
    }

    /// Returns the canonical architecture family for this loaded model.
    pub const fn model_family(&self) -> ModelKind {
        match self {
            Self::DeepSeek(kind, _)
            | Self::Gemma4(kind, _)
            | Self::GptOss(kind, _)
            | Self::KimiLinear(kind, _)
            | Self::Inkling(kind, _)
            | Self::Llama(kind, _)
            | Self::MuseGlimmer(kind, _)
            | Self::Lfm2(kind, _)
            | Self::NemotronH(kind, _)
            | Self::Qwen(kind, _)
            | Self::Qwen3Next(kind, _)
            | Self::Qwen3Vl(kind, _)
            | Self::Qwen3VlMoe(kind, _)
            | Self::Qwen35(kind, _) => kind.get(),
        }
    }

    /// Returns the effective model type preserved from the parsed configuration.
    pub fn effective_model_type(&self) -> &str {
        match self {
            Self::DeepSeek(_, model) => model.model_type(),
            Self::Gemma4(_, model) => &model.args().model_type,
            Self::GptOss(_, model) => &model.args().model_type,
            Self::Inkling(_, model) => &model.args().model_type,
            Self::KimiLinear(_, model) => &model.args().model_type,
            Self::Llama(_, model) => &model.args().model_type,
            Self::Lfm2(_, model) => &model.args().model_type,
            Self::NemotronH(_, model) => &model.args().model_type,
            Self::Qwen(_, model) => &model.args().model_type,
            Self::MuseGlimmer(_, model) => &model.args().model_type,
            Self::Qwen3Next(_, model) => &model.args().model_type,
            Self::Qwen3Vl(_, model) => model.model_type(),
            Self::Qwen3VlMoe(_, model) => model.model_type(),
            Self::Qwen35(_, model) => &model.args().model_type,
        }
    }

    /// Returns the canonical cache-relevant architecture identity derived from the loaded model.
    pub fn prompt_cache_architecture_fingerprint(&self) -> Result<String, Exception> {
        match self {
            Self::DeepSeek(_, model) => model
                .prompt_cache_identity()
                .map(|identity| identity.architecture_fingerprint)
                .map_err(|error| Exception::custom(error.to_string())),
            Self::Gemma4(_, model) => model
                .prompt_cache_model_identity()
                .map(|identity| identity.architecture_fingerprint)
                .map_err(|error| Exception::custom(error.to_string())),
            Self::Llama(_, model) => {
                Ok(eredu_architectures::llama::prompt_cache_architecture_fingerprint(model.args()))
            }
            Self::GptOss(_, model) => Ok(model.prompt_cache_architecture_fingerprint()),
            Self::Inkling(_, model) => Ok(model.args().architecture_fingerprint()),
            Self::KimiLinear(_, model) => Ok(kimi_linear::prompt_cache_architecture_fingerprint(
                model.args(),
            )),
            Self::Lfm2(_, model) => model
                .prompt_cache_architecture_fingerprint()
                .map_err(|error| Exception::custom(error.to_string())),
            Self::Qwen(_, model) => Ok(model.prompt_cache_architecture_fingerprint()),
            Self::MuseGlimmer(_, model) => model
                .prompt_cache_model_identity()
                .map(|identity| identity.architecture_fingerprint)
                .map_err(|error| Exception::custom(error.to_string())),
            Self::NemotronH(_, model) => model
                .prompt_cache_architecture_fingerprint()
                .map_err(|error| Exception::custom(error.to_string())),
            Self::Qwen3Next(_, model) | Self::Qwen35(_, model) => {
                Ok(model.prompt_cache_architecture_fingerprint())
            }
            Self::Qwen3Vl(_, model) | Self::Qwen3VlMoe(_, model) => {
                Ok(model.prompt_cache_architecture_fingerprint())
            }
        }
    }

    /// Returns the complete architecture-derived prompt-cache model identity.
    pub fn prompt_cache_model_identity(
        &self,
    ) -> Result<eredu_core::cache::PromptCacheModelIdentity, Exception> {
        match self {
            Self::DeepSeek(_, model) => model.prompt_cache_identity(),
            Self::Gemma4(_, model) => model.prompt_cache_model_identity(),
            Self::GptOss(_, model) => model.prompt_cache_model_identity(),
            Self::Inkling(_, model) => model.prompt_identity(),
            Self::KimiLinear(_, model) => model.prompt_cache_model_identity(),
            Self::Llama(_, model) => model.prompt_cache_model_identity(),
            Self::Lfm2(_, model) => model.prompt_cache_model_identity(),
            Self::NemotronH(_, model) => model.prompt_cache_model_identity(),
            Self::Qwen(_, model) => model.prompt_cache_model_identity(),
            Self::MuseGlimmer(_, model) => model.prompt_cache_model_identity(),
            Self::Qwen3Next(_, model) | Self::Qwen35(_, model) => {
                model.prompt_cache_model_identity()
            }
            Self::Qwen3Vl(_, model) | Self::Qwen3VlMoe(_, model) => {
                model.prompt_cache_model_identity()
            }
        }
        .map_err(|error| Exception::custom(error.to_string()))
    }

    /// Returns the exact ordered prompt-cache state and attention layout.
    pub fn prompt_cache_layer_layout(&self) -> Result<LayerSchedule<LayerCachePolicy>, Exception> {
        match self {
            Self::DeepSeek(_, model) => model
                .prompt_cache_identity()
                .map(|identity| identity.layer_layout),
            Self::Llama(_, model) => model.prompt_cache_layer_layout(),
            Self::GptOss(_, model) => model.prompt_cache_layer_layout(),
            Self::Qwen(_, model) => model.prompt_cache_layer_layout(),
            Self::MuseGlimmer(_, model) => model
                .prompt_cache_model_identity()
                .map(|identity| identity.layer_layout),
            Self::KimiLinear(_, model) => model.prompt_cache_layer_layout(),
            Self::Lfm2(_, model) => model.prompt_cache_layer_layout(),
            Self::NemotronH(_, model) => model.prompt_cache_layer_layout(),
            Self::Qwen3Next(_, model) | Self::Qwen35(_, model) => model.prompt_cache_layer_layout(),
            Self::Qwen3Vl(_, model) | Self::Qwen3VlMoe(_, model) => {
                model.prompt_cache_layer_layout()
            }
            Self::Gemma4(_, model) => model
                .prompt_cache_model_identity()
                .map(|identity| identity.layer_layout),
            Self::Inkling(_, model) => model.prompt_cache_layer_layout(),
        }
        .map_err(|error| Exception::custom(error.to_string()))
    }

    /// Returns each owned layer's processed-token delta relative to the
    /// persisted prefix. Ordinary decoder layers use zero; speculative layers
    /// may trail the target frontier.
    pub fn prompt_cache_layer_prefix_offsets(&self) -> Result<Vec<i32>, Exception> {
        match self {
            Self::DeepSeek(_, model) => model
                .prompt_cache_identity()
                .map(|identity| identity.layer_prefix_offsets)
                .map_err(|error| Exception::custom(error.to_string())),
            Self::Gemma4(_, model) => model
                .prompt_cache_model_identity()
                .map(|identity| identity.layer_prefix_offsets)
                .map_err(|error| Exception::custom(error.to_string())),
            Self::MuseGlimmer(_, model) => model
                .prompt_cache_model_identity()
                .map(|identity| identity.layer_prefix_offsets)
                .map_err(|error| Exception::custom(error.to_string())),
            Self::Inkling(_, model) => model
                .prompt_cache_layer_prefix_offsets()
                .map_err(|error| Exception::custom(error.to_string())),
            Self::NemotronH(_, model) => model
                .prompt_cache_model_identity()
                .map(|identity| identity.layer_prefix_offsets)
                .map_err(|error| Exception::custom(error.to_string())),
            Self::Qwen3Next(_, model) | Self::Qwen35(_, model) => model
                .prompt_cache_model_identity()
                .map(|identity| identity.layer_prefix_offsets)
                .map_err(|error| Exception::custom(error.to_string())),
            _ => Ok(vec![0; self.prompt_cache_layer_layout()?.len()]),
        }
    }

    /// Returns the architecture-declared named prompt-cache state ranges.
    pub fn prompt_cache_state_segments(
        &self,
    ) -> Result<Vec<eredu_core::cache::PromptCacheStateSegment>, Exception> {
        Ok(self.prompt_cache_model_identity()?.state_segments)
    }

    /// Runs an instrumented pass through the canonical generalized executor.
    pub fn forward_with_observer(
        &mut self,
        input_tokens: &Array,
        mask: Option<&Array>,
        cache: &mut ModelCache,
        stream: &Stream,
        observer: &mut impl RuntimeActivationObserver<crate::MlxTensor, Exception>,
    ) -> Result<Array, Exception> {
        crate::composition::mlx::MlxModelSession::forward_with_observer(
            self,
            input_tokens,
            mask,
            cache,
            stream,
            observer,
        )
        .map_err(|error| Exception::custom(error.to_string()))
    }

    /// Submits prompt prefill while reporting detailed activations.
    pub fn submit_prefill_with_observer(
        &mut self,
        input: input::ModelInput<'_>,
        cache: &mut ModelCache,
        stream: &Stream,
        observer: &mut impl RuntimeActivationObserver<crate::MlxTensor, Exception>,
    ) -> Result<eredu_core::Submission<Array, crate::backend::MlxCompletion>, Error> {
        crate::composition::mlx::MlxModelSession::submit_complete_prefill_with_observer(
            self,
            input.into(),
            cache,
            stream,
            observer,
        )
    }

    /// Creates an empty cache value appropriate for this model.
    pub fn new_cache(&self) -> ModelCache {
        match self {
            Self::DeepSeek(_, model) => ModelCache::DeepSeek(
                model
                    .new_state()
                    .expect("validated DeepSeek state geometry"),
            ),
            Self::Gemma4(_, model) => ModelCache::Hybrid(model.new_cache()),
            Self::GptOss(_, model) => ModelCache::GptOss(model.new_cache()),
            Self::Inkling(_, model) => ModelCache::Inkling(model.new_cache()),
            Self::KimiLinear(_, model) => ModelCache::Hybrid(model.new_cache()),
            Self::Llama(_, model) => ModelCache::Llama(model.new_cache()),
            Self::Lfm2(_, model) => ModelCache::Hybrid(model.new_cache()),
            Self::Qwen(_, model) => ModelCache::Qwen(model.new_cache()),
            Self::MuseGlimmer(_, model) => ModelCache::MuseGlimmer(model.new_cache()),
            Self::Qwen3Next(_, model) => ModelCache::Qwen3Next(model.new_cache()),
            Self::Qwen3Vl(_, model) => ModelCache::Qwen3Vl(model.new_cache()),
            Self::Qwen3VlMoe(_, model) => ModelCache::Qwen3VlMoe(model.new_cache()),
            Self::NemotronH(_, model) => ModelCache::Hybrid(model.new_cache()),
            Self::Qwen35(_, model) => ModelCache::Qwen35(model.new_cache()),
        }
    }

    /// Creates ordinary cache state or an explicitly bounded paged cache.
    pub fn new_cache_with_options(
        &self,
        policy: CacheResidencyPolicy,
    ) -> Result<ModelCache, Exception> {
        match policy {
            CacheResidencyPolicy::Device => Ok(self.new_cache()),
            CacheResidencyPolicy::Paged(options) => match self {
                Self::DeepSeek(_, model) => model
                    .new_state_with_options(CacheResidencyPolicy::Paged(options))
                    .map(ModelCache::DeepSeek)
                    .map_err(|error| Exception::custom(error.to_string())),
                Self::Llama(_, model) => model
                    .new_cache_with_options(CacheResidencyPolicy::Paged(options))
                    .map(ModelCache::Llama)
                    .map_err(|error| Exception::custom(error.to_string())),
                Self::KimiLinear(_, model) => model
                    .new_cache_with_options(CacheResidencyPolicy::Paged(options))
                    .map(ModelCache::Hybrid)
                    .map_err(|error| Exception::custom(error.to_string())),
                Self::GptOss(_, model) => model
                    .new_cache_with_options(CacheResidencyPolicy::Paged(options))
                    .map(ModelCache::GptOss)
                    .map_err(|error| Exception::custom(error.to_string())),
                Self::Qwen(_, model) => model
                    .new_cache_with_options(CacheResidencyPolicy::Paged(options))
                    .map(ModelCache::Qwen)
                    .map_err(|error| Exception::custom(error.to_string())),
                Self::MuseGlimmer(_, model) => model
                    .new_cache_with_options(CacheResidencyPolicy::Paged(options))
                    .map(ModelCache::MuseGlimmer)
                    .map_err(|error| Exception::custom(error.to_string())),
                Self::Inkling(_, model) => model
                    .new_cache_with_options(CacheResidencyPolicy::Paged(options))
                    .map(ModelCache::Inkling)
                    .map_err(|error| Exception::custom(error.to_string())),
                Self::Gemma4(_, model) => model
                    .new_cache_with_options(CacheResidencyPolicy::Paged(options))
                    .map(ModelCache::Hybrid)
                    .map_err(|error| Exception::custom(error.to_string())),
                Self::NemotronH(_, model) => model
                    .new_cache_with_options(CacheResidencyPolicy::Paged(options))
                    .map(ModelCache::Hybrid)
                    .map_err(|error| Exception::custom(error.to_string())),
                Self::Lfm2(_, model) => model
                    .new_cache_with_options(CacheResidencyPolicy::Paged(options))
                    .map(ModelCache::Hybrid)
                    .map_err(|error| Exception::custom(error.to_string())),
                Self::Qwen3Next(_, model) => model
                    .new_cache_with_options(CacheResidencyPolicy::Paged(options))
                    .map(ModelCache::Qwen3Next)
                    .map_err(|error| Exception::custom(error.to_string())),
                Self::Qwen35(_, model) => model
                    .new_cache_with_options(CacheResidencyPolicy::Paged(options))
                    .map(ModelCache::Qwen35)
                    .map_err(|error| Exception::custom(error.to_string())),
                _ => Err(Exception::custom(format!(
                    "paged cache residency is unsupported for model type {}",
                    self.effective_model_type()
                ))),
            },
        }
    }

    /// Lazily catalogs a compatible persisted text prefix for a fresh cache.
    pub fn load_prompt_cache(
        &self,
        directory: impl AsRef<Path>,
        expected: &PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: PagedCacheOptions,
        stream: &Stream,
    ) -> Result<(ModelCache, PromptCacheManifest), Exception> {
        macro_rules! load {
            ($model:expr, $variant:path) => {
                $model
                    .load_prompt_cache(directory, expected, prefix_token_ids, options, stream)
                    .map(|(cache, manifest)| ($variant(cache), manifest))
                    .map_err(|error| Exception::custom(error.to_string()))
            };
        }
        match self {
            Self::DeepSeek(_, model) => model
                .load_prompt_cache(directory, expected, prefix_token_ids, options, stream)
                .map(|(state, manifest)| (ModelCache::DeepSeek(state), manifest))
                .map_err(|error| Exception::custom(error.to_string())),
            Self::Llama(_, model) => load!(model, ModelCache::Llama),
            Self::GptOss(_, model) => load!(model, ModelCache::GptOss),
            Self::Qwen(_, model) => load!(model, ModelCache::Qwen),
            Self::MuseGlimmer(_, model) => load!(model, ModelCache::MuseGlimmer),
            Self::KimiLinear(_, model) => load!(model, ModelCache::Hybrid),
            Self::Qwen3Next(_, model) => load!(model, ModelCache::Qwen3Next),
            Self::Qwen35(_, model) => load!(model, ModelCache::Qwen35),
            Self::Qwen3Vl(_, model) => load!(model, ModelCache::Qwen3Vl),
            Self::Qwen3VlMoe(_, model) => load!(model, ModelCache::Qwen3VlMoe),
            Self::Gemma4(_, model) => load!(model, ModelCache::Hybrid),
            Self::Inkling(_, model) => load!(model, ModelCache::Inkling),
            Self::Lfm2(_, model) => load!(model, ModelCache::Hybrid),
            Self::NemotronH(_, model) => load!(model, ModelCache::Hybrid),
        }
    }

    /// Atomically saves a completed immutable prefix with model-owned state validation.
    pub fn save_prompt_cache(
        &self,
        cache: &mut ModelCache,
        destination: impl AsRef<Path>,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &PromptCacheOptions,
        stream: &Stream,
    ) -> Result<PromptCacheManifest, Exception> {
        match (self, &mut *cache) {
            (Self::DeepSeek(_, model), ModelCache::DeepSeek(state)) => {
                return model
                    .save_prompt_cache(state, &destination, descriptor, prefix_token_ids, options)
                    .map_err(|error| Exception::custom(error.to_string()));
            }
            (Self::Llama(_, model), ModelCache::Llama(cache)) => {
                return model
                    .save_prompt_cache(
                        cache,
                        &destination,
                        descriptor,
                        prefix_token_ids,
                        options,
                        stream,
                    )
                    .map_err(|error| Exception::custom(error.to_string()));
            }
            (Self::GptOss(_, model), ModelCache::GptOss(cache)) => {
                return model
                    .save_prompt_cache(
                        cache,
                        &destination,
                        descriptor,
                        prefix_token_ids,
                        options,
                        stream,
                    )
                    .map_err(|error| Exception::custom(error.to_string()));
            }
            (Self::Qwen(_, model), ModelCache::Qwen(cache)) => {
                return model
                    .save_prompt_cache(
                        cache,
                        &destination,
                        descriptor,
                        prefix_token_ids,
                        options,
                        stream,
                    )
                    .map_err(|error| Exception::custom(error.to_string()));
            }
            (Self::MuseGlimmer(_, model), ModelCache::MuseGlimmer(cache)) => {
                return model
                    .save_prompt_cache(
                        cache,
                        &destination,
                        descriptor,
                        prefix_token_ids,
                        options,
                        stream,
                    )
                    .map_err(|error| Exception::custom(error.to_string()));
            }
            (Self::KimiLinear(_, model), ModelCache::Hybrid(cache)) => {
                return model
                    .save_prompt_cache(
                        cache,
                        &destination,
                        descriptor,
                        prefix_token_ids,
                        options,
                        stream,
                    )
                    .map_err(|error| Exception::custom(error.to_string()));
            }
            (Self::Lfm2(_, model), ModelCache::Hybrid(cache)) => {
                return model
                    .save_prompt_cache(
                        cache,
                        &destination,
                        descriptor,
                        prefix_token_ids,
                        options,
                        stream,
                    )
                    .map_err(|error| Exception::custom(error.to_string()));
            }
            (Self::NemotronH(_, model), ModelCache::Hybrid(cache)) => {
                return model
                    .save_prompt_cache(
                        cache,
                        &destination,
                        descriptor,
                        prefix_token_ids,
                        options,
                        stream,
                    )
                    .map_err(|error| Exception::custom(error.to_string()));
            }
            (Self::Qwen3Next(_, model), ModelCache::Qwen3Next(cache))
            | (Self::Qwen35(_, model), ModelCache::Qwen35(cache)) => {
                return model
                    .save_prompt_cache(
                        cache,
                        &destination,
                        descriptor,
                        prefix_token_ids,
                        options,
                        stream,
                    )
                    .map_err(|error| Exception::custom(error.to_string()));
            }
            (Self::Gemma4(_, model), ModelCache::Hybrid(cache)) => {
                return model
                    .save_prompt_cache(
                        cache,
                        &destination,
                        descriptor,
                        prefix_token_ids,
                        options,
                        stream,
                    )
                    .map_err(|error| Exception::custom(error.to_string()));
            }
            (Self::Qwen3Vl(_, model), ModelCache::Qwen3Vl(cache))
            | (Self::Qwen3VlMoe(_, model), ModelCache::Qwen3VlMoe(cache)) => {
                return model
                    .save_prompt_cache(
                        cache,
                        &destination,
                        descriptor,
                        prefix_token_ids,
                        options,
                        stream,
                    )
                    .map_err(|error| Exception::custom(error.to_string()));
            }
            (Self::Inkling(_, model), ModelCache::Inkling(cache)) => {
                return model
                    .save_prompt_cache(
                        cache,
                        &destination,
                        descriptor,
                        prefix_token_ids,
                        options,
                        stream,
                    )
                    .map_err(|error| Exception::custom(error.to_string()));
            }
            _ => {}
        }
        Err(Exception::custom(
            "model and cache representations do not match for prompt-cache publication",
        ))
    }

    /// Submits prompt prefill through the selected MLX model session.
    pub fn submit_prefill(
        &mut self,
        input: input::ModelInput<'_>,
        cache: &mut ModelCache,
        stream: &Stream,
    ) -> Result<eredu_core::Submission<Array, crate::backend::MlxCompletion>, Error> {
        crate::composition::mlx::submit_prefill_with_cache(self, cache, input.into(), stream)
    }

    /// Submits cached decode through the selected MLX model session.
    pub fn submit_decode(
        &mut self,
        input: Array,
        cache: &mut ModelCache,
        stream: &Stream,
    ) -> Result<eredu_core::Submission<Array, crate::backend::MlxCompletion>, Error> {
        crate::composition::mlx::submit_decode_with_cache(self, cache, input, stream)
    }
}

/// Cache value matching a [`Model`] variant.
#[derive(Clone)]
pub enum ModelCache {
    /// Architecture-declared DeepSeek state.
    DeepSeek(crate::composition::deepseek::DeepSeekState),
    /// GPT-OSS cache following its canonical per-layer attention schedule.
    GptOss(gpt_oss::Cache),
    /// Neutral Muse-Glimmer key/value state.
    MuseGlimmer(crate::backend::runtime::cache::state::MlxKeyValueState),
    /// Runtime-policy-selected MLX key/value state.
    Llama(crate::backend::runtime::cache::state::MlxKeyValueState),
    /// Runtime-policy-selected Qwen key/value state.
    Qwen(crate::backend::runtime::cache::state::MlxKeyValueState),
    /// Qwen3-VL key/value cache and multimodal position state.
    Qwen3Vl(crate::backend::runtime::cache::state::MlxHybridState),
    /// Qwen3-VL-MoE key/value cache and multimodal position state.
    Qwen3VlMoe(crate::backend::runtime::cache::state::MlxHybridState),
    /// Architecture-declared heterogeneous append-only and fixed state.
    Hybrid(crate::backend::runtime::cache::state::MlxHybridState),
    /// Neutral Inkling target and checkpoint-embedded predictor state.
    Inkling(crate::composition::inkling::InklingState),
    /// Heterogeneous Qwen3.5 MoE cache.
    Qwen35(crate::backend::runtime::cache::state::MlxHybridState),
    /// Heterogeneous Qwen3-Next cache.
    Qwen3Next(crate::backend::runtime::cache::state::MlxHybridState),
}

impl ModelCache {
    /// Returns aggregate cache-residency telemetry when paging is active.
    pub fn residency_report(&self) -> Result<Option<CacheResidencyReport>, Exception> {
        match self {
            Self::DeepSeek(cache) => cache.residency_report(),
            Self::GptOss(cache) => cache.residency_report(),
            Self::Inkling(cache) => cache.target().residency_report(),
            Self::MuseGlimmer(cache) | Self::Llama(cache) | Self::Qwen(cache) => {
                cache.residency_report()
            }
            Self::Qwen3Vl(cache)
            | Self::Qwen3VlMoe(cache)
            | Self::Hybrid(cache)
            | Self::Qwen35(cache)
            | Self::Qwen3Next(cache) => cache.residency_report(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::runtime::cache::{
        residency::CacheResidencyManager,
        state::{MlxHybridState, MlxKeyValueState},
    };
    use eredu_core::{cache::LayerCachePolicy, AttentionPolicy};
    use eredu_runtime::StateLayout;

    fn paged_state_layout() -> StateLayout {
        StateLayout::new(
            LayerSchedule::new(
                1,
                vec![LayerCachePolicy::key_value(AttentionPolicy::Full, 1, 8).unwrap()],
            )
            .unwrap(),
        )
        .unwrap()
    }

    fn cache_residency_manager() -> CacheResidencyManager {
        CacheResidencyManager::new(
            PagedCacheOptions::new(4, 1 << 20, 1 << 20, 1)
                .unwrap()
                .with_full_attention(true),
        )
        .unwrap()
    }

    #[test]
    fn complete_model_identity_trusts_admission_and_checks_the_concrete_variant() {
        let identity =
            AdmittedModelKind::new(ModelKind::Qwen3, &[ModelKind::Qwen2, ModelKind::Qwen3])
                .unwrap();
        assert_eq!(identity.get(), ModelKind::Qwen3);

        let aliased_identity =
            AdmittedModelKind::new(ModelKind::Qwen2, &[ModelKind::Qwen2, ModelKind::Qwen3])
                .unwrap();
        assert_eq!(aliased_identity.get(), ModelKind::Qwen2);

        assert!(AdmittedModelKind::new(ModelKind::Qwen35, &[ModelKind::Qwen3Next]).is_err());
    }

    #[test]
    fn qwen_cache_variants_forward_paged_residency_reports() {
        let layout = paged_state_layout();
        let manager = cache_residency_manager();
        let caches = [
            (
                "Qwen",
                ModelCache::Qwen(
                    MlxKeyValueState::paged(layout.clone(), manager.clone(), None).unwrap(),
                ),
            ),
            (
                "Qwen3-Next",
                ModelCache::Qwen3Next(
                    MlxHybridState::paged(layout.clone(), manager.clone(), None).unwrap(),
                ),
            ),
            (
                "Qwen3.5",
                ModelCache::Qwen35(MlxHybridState::paged(layout, manager, None).unwrap()),
            ),
        ];

        for (family, cache) in caches {
            assert!(
                cache.residency_report().unwrap().is_some(),
                "{family} paged-cache telemetry should be available"
            );
        }
    }
}
