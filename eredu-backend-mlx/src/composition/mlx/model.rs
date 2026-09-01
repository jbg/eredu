//! Architecture-erased executable and generation dispatch.

use std::path::Path;

use eredu_core::cache::{PromptCacheDescriptor, PromptCacheManifest, PromptCacheOptions};
use eredu_core::SpeculativeCapability;
use safemlx::{error::Exception, Stream};

use crate::backend::error::Error;
use crate::composition::gpt_oss;
use eredu_architectures::ModelKind;
use eredu_runtime::CacheResidencyReport;
use eredu_runtime::{CacheResidencyPolicy, PagedCacheOptions};

/// Admitted architecture identity checked against the concrete model variant.
#[derive(Clone, Copy)]
#[cfg_attr(not(test), allow(dead_code))]
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

    #[cfg(test)]
    const fn get(self) -> ModelKind {
        self.0
    }
}

/// Loaded executable for any architecture supported by this crate.
///
/// Construction is restricted to the checked family-specific constructors so
/// reporting, input dispatch, and mutable state cannot disagree. Each variant
/// owns the concrete model and its correctly typed cache together; there is no
/// independently extensible erased cache type to re-pair at operation sites.
#[cfg_attr(not(test), allow(dead_code))]
pub enum Executable {
    /// Neutral DeepSeek-V3/V4 architecture with policy-selected residency.
    DeepSeek(
        AdmittedModelKind,
        Box<crate::composition::deepseek::DeepSeekModel>,
        crate::composition::deepseek::DeepSeekState,
    ),
    /// Gemma 4 text and multimodal model.
    Gemma4(
        AdmittedModelKind,
        crate::composition::gemma4::Gemma4Model,
        crate::backend::runtime::cache::state::MlxHybridState,
    ),
    /// OpenAI GPT-OSS model.
    GptOss(
        AdmittedModelKind,
        crate::composition::gpt_oss::GptOssModel,
        gpt_oss::Cache,
    ),
    /// Moonshot Kimi Linear hybrid KDA/MLA sparse decoder.
    KimiLinear(
        AdmittedModelKind,
        crate::composition::kimi_linear::KimiLinearModel,
        crate::backend::runtime::cache::state::MlxHybridState,
    ),
    /// Thinking Machines Lab Inkling multimodal model.
    Inkling(
        AdmittedModelKind,
        crate::composition::inkling::InklingModel,
        crate::composition::inkling::InklingState,
    ),
    /// Tensor-parallel Llama model.
    PartitionedLlama(
        AdmittedModelKind,
        crate::composition::llama::PartitionedLlamaModel,
        crate::backend::runtime::cache::state::MlxKeyValueState,
    ),
    /// Ordinary replicated text model bound through the family-neutral contract.
    ReplicatedText(
        AdmittedModelKind,
        Box<dyn super::replicated_text::ErasedReplicatedTextExecutable>,
    ),
    /// Meta Muse-Glimmer dense multimodal model.
    MuseGlimmer(
        AdmittedModelKind,
        crate::composition::muse_glimmer::MuseGlimmerModel,
        crate::backend::runtime::cache::state::MlxKeyValueState,
    ),
    /// Liquid AI LFM2/LFM2.5 model.
    Lfm2(
        AdmittedModelKind,
        crate::composition::lfm2::Lfm2Model,
        crate::backend::runtime::cache::state::MlxHybridState,
    ),
    /// Nemotron-H hybrid model.
    NemotronH(
        AdmittedModelKind,
        crate::composition::nemotron_h::NemotronHModel,
        crate::backend::runtime::cache::state::MlxHybridState,
    ),
    /// Neutral Qwen2/Qwen2.5/Qwen3/Qwen3-MoE model.
    Qwen(
        AdmittedModelKind,
        crate::composition::qwen::QwenModel,
        crate::backend::runtime::cache::state::MlxKeyValueState,
    ),
    /// Qwen3-Next model.
    Qwen3Next(
        AdmittedModelKind,
        crate::composition::qwen::hybrid::QwenHybridModel,
        crate::backend::runtime::cache::state::MlxHybridState,
    ),
    /// Qwen3-VL multimodal model.
    Qwen3Vl(
        AdmittedModelKind,
        crate::composition::qwen::vl::QwenVlModel,
        crate::backend::runtime::cache::state::MlxHybridState,
    ),
    /// Qwen3-VL-MoE multimodal model.
    Qwen3VlMoe(
        AdmittedModelKind,
        crate::composition::qwen::vl::QwenVlModel,
        crate::backend::runtime::cache::state::MlxHybridState,
    ),
    /// Qwen3.5 dense or MoE model, optionally multimodal.
    Qwen35(
        AdmittedModelKind,
        crate::composition::qwen::hybrid::QwenHybridModel,
        crate::backend::runtime::cache::state::MlxHybridState,
    ),
}

impl Executable {
    pub(super) fn selected_session_binding(
        &self,
    ) -> Option<&super::replicated_text::SelectedSessionBinding> {
        match self {
            Self::ReplicatedText(_, executable) => Some(executable.selected_session_binding()),
            _ => None,
        }
    }

    pub(super) fn deepseek(
        kind: ModelKind,
        model: Box<crate::composition::deepseek::DeepSeekModel>,
    ) -> Result<Self, Error> {
        let identity =
            AdmittedModelKind::new(kind, &[ModelKind::DeepSeekV3, ModelKind::DeepSeekV4])?;
        let state = model
            .new_state()
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        Ok(Self::DeepSeek(identity, model, state))
    }

    pub(super) fn gemma4(
        kind: ModelKind,
        model: crate::composition::gemma4::Gemma4Model,
    ) -> Result<Self, Error> {
        let identity = AdmittedModelKind::new(kind, &[ModelKind::Gemma4])?;
        let cache = model.new_cache();
        Ok(Self::Gemma4(identity, model, cache))
    }

    pub(super) fn gpt_oss(
        kind: ModelKind,
        model: crate::composition::gpt_oss::GptOssModel,
    ) -> Result<Self, Error> {
        let identity = AdmittedModelKind::new(kind, &[ModelKind::GptOss])?;
        let cache = model.new_cache();
        Ok(Self::GptOss(identity, model, cache))
    }

    pub(super) fn kimi_linear(
        kind: ModelKind,
        model: crate::composition::kimi_linear::KimiLinearModel,
    ) -> Result<Self, Error> {
        let identity = AdmittedModelKind::new(kind, &[ModelKind::KimiLinear])?;
        let cache = model.new_cache();
        Ok(Self::KimiLinear(identity, model, cache))
    }

    pub(super) fn inkling(
        kind: ModelKind,
        model: crate::composition::inkling::InklingModel,
    ) -> Result<Self, Error> {
        let identity = AdmittedModelKind::new(kind, &[ModelKind::Inkling])?;
        let cache = model.new_cache();
        Ok(Self::Inkling(identity, model, cache))
    }

    pub(super) fn partitioned_llama(
        kind: ModelKind,
        model: crate::composition::llama::PartitionedLlamaModel,
    ) -> Result<Self, Error> {
        let identity = AdmittedModelKind::new(kind, &[ModelKind::Llama])?;
        let cache = model.new_cache();
        Ok(Self::PartitionedLlama(identity, model, cache))
    }

    pub(super) fn replicated_text(
        kind: ModelKind,
        model: Box<dyn super::replicated_text::ErasedReplicatedTextExecutable>,
    ) -> Result<Self, Error> {
        let identity = AdmittedModelKind::new(
            kind,
            &[ModelKind::Llama, ModelKind::Qwen2, ModelKind::Qwen3],
        )?;
        Ok(Self::ReplicatedText(identity, model))
    }

    pub(super) fn muse_glimmer(
        kind: ModelKind,
        model: crate::composition::muse_glimmer::MuseGlimmerModel,
    ) -> Result<Self, Error> {
        let identity = AdmittedModelKind::new(kind, &[ModelKind::MuseGlimmer])?;
        let cache = model.new_cache();
        Ok(Self::MuseGlimmer(identity, model, cache))
    }

    pub(super) fn lfm2(
        kind: ModelKind,
        model: crate::composition::lfm2::Lfm2Model,
    ) -> Result<Self, Error> {
        let identity = AdmittedModelKind::new(kind, &[ModelKind::Lfm2])?;
        let cache = model.new_cache();
        Ok(Self::Lfm2(identity, model, cache))
    }

    pub(super) fn nemotron_h(
        kind: ModelKind,
        model: crate::composition::nemotron_h::NemotronHModel,
    ) -> Result<Self, Error> {
        let identity = AdmittedModelKind::new(kind, &[ModelKind::NemotronH])?;
        let cache = model.new_cache();
        Ok(Self::NemotronH(identity, model, cache))
    }

    pub(super) fn qwen(
        kind: ModelKind,
        model: crate::composition::qwen::QwenModel,
    ) -> Result<Self, Error> {
        let identity = AdmittedModelKind::new(kind, &[ModelKind::Qwen2, ModelKind::Qwen3])?;
        let cache = model.new_cache();
        Ok(Self::Qwen(identity, model, cache))
    }

    pub(super) fn qwen3_next(
        kind: ModelKind,
        model: crate::composition::qwen::hybrid::QwenHybridModel,
    ) -> Result<Self, Error> {
        let identity = AdmittedModelKind::new(kind, &[ModelKind::Qwen3Next])?;
        let cache = model.new_cache();
        Ok(Self::Qwen3Next(identity, model, cache))
    }

    pub(super) fn qwen3_vl(
        kind: ModelKind,
        model: crate::composition::qwen::vl::QwenVlModel,
    ) -> Result<Self, Error> {
        let identity = AdmittedModelKind::new(kind, &[ModelKind::Qwen3Vl])?;
        let cache = model.new_cache();
        Ok(Self::Qwen3Vl(identity, model, cache))
    }

    pub(super) fn qwen3_vl_moe(
        kind: ModelKind,
        model: crate::composition::qwen::vl::QwenVlModel,
    ) -> Result<Self, Error> {
        let identity = AdmittedModelKind::new(kind, &[ModelKind::Qwen3VlMoe])?;
        let cache = model.new_cache();
        Ok(Self::Qwen3VlMoe(identity, model, cache))
    }

    pub(super) fn qwen35(
        kind: ModelKind,
        model: crate::composition::qwen::hybrid::QwenHybridModel,
    ) -> Result<Self, Error> {
        let identity = AdmittedModelKind::new(kind, &[ModelKind::Qwen35])?;
        let cache = model.new_cache();
        Ok(Self::Qwen35(identity, model, cache))
    }

    /// Returns architecture-neutral rank-local placement information when this
    /// model was loaded through generalized parallel execution groups.
    pub fn parallel_info(
        &self,
    ) -> Option<
        &eredu_runtime::ParallelModelInfo<
            crate::composition::mlx::distributed::topology::MlxParallelPlan,
        >,
    > {
        match self {
            Self::DeepSeek(_, _, _) => None,
            Self::PartitionedLlama(_, model, _) => model.parallel_info(),
            Self::ReplicatedText(_, _) => None,
            Self::MuseGlimmer(_, model, _) => model.parallel_info(),
            Self::GptOss(_, model, _) => model.parallel_info(),
            Self::Qwen(_, model, _) => model.parallel_info(),
            Self::KimiLinear(_, model, _) => model.parallel_info(),
            Self::Lfm2(_, model, _) => model.parallel_info(),
            Self::NemotronH(_, model, _) => model.parallel_info(),
            Self::Qwen3Next(_, model, _) | Self::Qwen35(_, model, _) => model.parallel_info(),
            Self::Qwen3Vl(_, model, _) | Self::Qwen3VlMoe(_, model, _) => model.parallel_info(),
            Self::Gemma4(_, model, _) => model.parallel_info(),
            Self::Inkling(_, model, _) => model.parallel_info(),
        }
    }

    /// Reports how this model architecture exposes speculative drafting weights.
    pub fn speculative_capability(&self) -> SpeculativeCapability {
        self.architecture_capability_estimate()
            .ok()
            .and_then(|estimate| estimate.speculative_draft_source())
            .map_or(SpeculativeCapability::Unavailable, |draft_source| {
                SpeculativeCapability::Ready { draft_source }
            })
    }

    /// Returns residency telemetry when this model uses bounded layer execution.
    pub fn residency_report(&self) -> Result<Option<eredu_runtime::ResidencyReport>, Error> {
        match self {
            Self::DeepSeek(_, model, _) => Ok(Some(model.residency_report()?)),
            Self::Gemma4(_, model, _) => model.residency_report(),
            Self::Inkling(_, model, _) => model.residency_report(),
            Self::KimiLinear(_, model, _) => Ok(Some(model.residency_report()?)),
            Self::PartitionedLlama(_, model, _) => model.residency_report(),
            Self::ReplicatedText(_, model) => model.residency_report(),
            Self::GptOss(_, model, _) => Ok(Some(model.residency_report()?)),
            Self::Lfm2(_, model, _) => Ok(Some(model.residency_report()?)),
            Self::NemotronH(_, model, _) => Ok(Some(model.residency_report()?)),
            Self::Qwen3Next(_, model, _) | Self::Qwen35(_, model, _) => {
                Ok(Some(model.residency_report()?))
            }
            Self::Qwen(_, model, _) => model.residency_report(),
            Self::MuseGlimmer(_, model, _) => model.residency_report(),
            Self::Qwen3Vl(_, model, _) | Self::Qwen3VlMoe(_, model, _) => {
                Ok(Some(model.residency_report()?))
            }
        }
    }

    /// Returns experimental dense-stream telemetry when enabled.
    pub fn dense_stream_report(
        &self,
    ) -> Result<Option<eredu_runtime::DenseDiskStreamReport>, Error> {
        match self {
            Self::DeepSeek(_, model, _) => model.dense_stream_report(),
            Self::Gemma4(_, model, _) => model.dense_stream_report(),
            Self::Inkling(_, model, _) => model.dense_stream_report(),
            Self::KimiLinear(_, model, _) => model.dense_stream_report(),
            Self::PartitionedLlama(_, model, _) => model.dense_stream_report(),
            Self::ReplicatedText(_, model) => model.dense_stream_report(),
            Self::GptOss(_, model, _) => model.dense_stream_report(),
            Self::Lfm2(_, model, _) => model.dense_stream_report(),
            Self::NemotronH(_, model, _) => model.dense_stream_report(),
            Self::Qwen3Next(_, model, _) | Self::Qwen35(_, model, _) => model.dense_stream_report(),
            Self::Qwen(_, model, _) => model.dense_stream_report(),
            Self::MuseGlimmer(_, model, _) => model.dense_stream_report(),
            Self::Qwen3Vl(_, model, _) | Self::Qwen3VlMoe(_, model, _) => {
                model.dense_stream_report()
            }
        }
    }

    /// Returns load-time weight transformation telemetry when materialization transformed weights.
    pub fn materialization_report(&self) -> Option<&eredu_runtime::WeightMaterializationReport> {
        match self {
            Self::ReplicatedText(_, model) => model.materialization_report(),
            Self::DeepSeek(_, _, _)
            | Self::Gemma4(_, _, _)
            | Self::GptOss(_, _, _)
            | Self::KimiLinear(_, _, _)
            | Self::Inkling(_, _, _)
            | Self::PartitionedLlama(_, _, _)
            | Self::MuseGlimmer(_, _, _)
            | Self::Lfm2(_, _, _)
            | Self::NemotronH(_, _, _)
            | Self::Qwen(_, _, _)
            | Self::Qwen3Next(_, _, _)
            | Self::Qwen3Vl(_, _, _)
            | Self::Qwen3VlMoe(_, _, _)
            | Self::Qwen35(_, _, _) => None,
        }
    }

    /// Returns sparse routed-expert cache telemetry when enabled.
    pub fn parameter_bank_report(
        &self,
    ) -> Result<
        Option<crate::backend::runtime::residency::parameter_bank::ParameterBankResidencyReport>,
        Error,
    > {
        match self {
            Self::DeepSeek(_, model, _) => model.parameter_bank_report(),
            Self::Gemma4(_, model, _) => model.parameter_bank_report(),
            Self::KimiLinear(_, model, _) => model.parameter_bank_report(),
            Self::GptOss(_, model, _) => model.parameter_bank_report(),
            Self::Inkling(_, model, _) => model.parameter_bank_report(),
            Self::Lfm2(_, model, _) => model.parameter_bank_report(),
            Self::NemotronH(_, model, _) => model.parameter_bank_report(),
            Self::Qwen(_, model, _) => model.parameter_bank_report(),
            Self::Qwen3Next(_, model, _) | Self::Qwen35(_, model, _) => {
                model.parameter_bank_report()
            }
            Self::Qwen3VlMoe(_, model, _) => model.parameter_bank_report(),
            Self::MuseGlimmer(_, model, _) => model.parameter_bank_report(),
            Self::PartitionedLlama(_, _, _)
            | Self::ReplicatedText(_, _)
            | Self::Qwen3Vl(_, _, _) => Ok(None),
        }
    }

    #[cfg(test)]
    pub(crate) const fn model_family(&self) -> ModelKind {
        match self {
            Self::DeepSeek(kind, _, _)
            | Self::Gemma4(kind, _, _)
            | Self::GptOss(kind, _, _)
            | Self::KimiLinear(kind, _, _)
            | Self::Inkling(kind, _, _)
            | Self::PartitionedLlama(kind, _, _)
            | Self::MuseGlimmer(kind, _, _)
            | Self::Lfm2(kind, _, _)
            | Self::NemotronH(kind, _, _)
            | Self::Qwen(kind, _, _)
            | Self::Qwen3Next(kind, _, _)
            | Self::Qwen3Vl(kind, _, _)
            | Self::Qwen3VlMoe(kind, _, _)
            | Self::Qwen35(kind, _, _) => kind.get(),
            Self::ReplicatedText(kind, _) => kind.get(),
        }
    }

    /// Returns the effective model type preserved from the parsed configuration.
    pub fn effective_model_type(&self) -> &str {
        match self {
            Self::DeepSeek(_, model, _) => model.model_type(),
            Self::Gemma4(_, model, _) => model.args().effective_model_type(),
            Self::GptOss(_, model, _) => &model.args().model_type,
            Self::Inkling(_, model, _) => &model.args().model_type,
            Self::KimiLinear(_, model, _) => &model.args().model_type,
            Self::PartitionedLlama(_, model, _) => &model.args().model_type,
            Self::ReplicatedText(_, model) => model.effective_model_type(),
            Self::Lfm2(_, model, _) => &model.args().model_type,
            Self::NemotronH(_, model, _) => &model.args().model_type,
            Self::Qwen(_, model, _) => &model.args().model_type,
            Self::MuseGlimmer(_, model, _) => &model.args().model_type,
            Self::Qwen3Next(_, model, _) => &model.args().model_type,
            Self::Qwen3Vl(_, model, _) => model.effective_model_type(),
            Self::Qwen3VlMoe(_, model, _) => model.effective_model_type(),
            Self::Qwen35(_, model, _) => &model.args().model_type,
        }
    }

    /// Returns the complete architecture-derived prompt-cache model identity.
    pub fn prompt_cache_model_identity(
        &self,
    ) -> Result<eredu_core::cache::PromptCacheModelIdentity, Exception> {
        match self {
            Self::DeepSeek(_, model, _) => model.prompt_cache_identity(),
            Self::Gemma4(_, model, _) => model.prompt_cache_model_identity(),
            Self::GptOss(_, model, _) => model.prompt_cache_model_identity(),
            Self::Inkling(_, model, _) => model.prompt_identity(),
            Self::KimiLinear(_, model, _) => model.prompt_cache_model_identity(),
            Self::PartitionedLlama(_, model, _) => model.prompt_cache_model_identity(),
            Self::ReplicatedText(_, model) => Ok(model.prompt_cache_model_identity().clone()),
            Self::Lfm2(_, model, _) => model.prompt_cache_model_identity(),
            Self::NemotronH(_, model, _) => model.prompt_cache_model_identity(),
            Self::Qwen(_, model, _) => model.prompt_cache_model_identity(),
            Self::MuseGlimmer(_, model, _) => model.prompt_cache_model_identity(),
            Self::Qwen3Next(_, model, _) | Self::Qwen35(_, model, _) => {
                model.prompt_cache_model_identity()
            }
            Self::Qwen3Vl(_, model, _) | Self::Qwen3VlMoe(_, model, _) => {
                model.prompt_cache_model_identity()
            }
        }
        .map_err(|error| Exception::custom(error.to_string()))
    }

    /// Replaces the executable's state with an empty cache using `policy`.
    pub fn reset_cache_with_options(
        &mut self,
        policy: CacheResidencyPolicy,
    ) -> Result<(), Exception> {
        macro_rules! reset_paged {
            ($model:expr, $cache:expr, $options:expr) => {{
                *$cache = $model
                    .new_cache_with_options(CacheResidencyPolicy::Paged($options))
                    .map_err(|error| Exception::custom(error.to_string()))?;
                Ok(())
            }};
        }
        match policy {
            CacheResidencyPolicy::Device => self.reset_cache(),
            CacheResidencyPolicy::Paged(options) => match self {
                Self::DeepSeek(_, model, state) => {
                    *state = model
                        .new_state_with_options(CacheResidencyPolicy::Paged(options))
                        .map_err(|error| Exception::custom(error.to_string()))?;
                    Ok(())
                }
                Self::Gemma4(_, model, cache) => reset_paged!(model, cache, options),
                Self::GptOss(_, model, cache) => reset_paged!(model, cache, options),
                Self::Inkling(_, model, cache) => reset_paged!(model, cache, options),
                Self::KimiLinear(_, model, cache) => reset_paged!(model, cache, options),
                Self::Lfm2(_, model, cache) => reset_paged!(model, cache, options),
                Self::PartitionedLlama(_, model, cache) => reset_paged!(model, cache, options),
                Self::ReplicatedText(_, model) => model.reset_cache(),
                Self::MuseGlimmer(_, model, cache) => reset_paged!(model, cache, options),
                Self::NemotronH(_, model, cache) => reset_paged!(model, cache, options),
                Self::Qwen(_, model, cache) => reset_paged!(model, cache, options),
                Self::Qwen3Next(_, model, cache) => reset_paged!(model, cache, options),
                Self::Qwen3Vl(_, model, cache) | Self::Qwen3VlMoe(_, model, cache) => {
                    reset_paged!(model, cache, options)
                }
                Self::Qwen35(_, model, cache) => reset_paged!(model, cache, options),
            },
        }
    }

    /// Clears all executable-owned cache state.
    pub fn reset_cache(&mut self) -> Result<(), Exception> {
        match self {
            Self::DeepSeek(_, model, state) => {
                *state = model
                    .new_state()
                    .map_err(|error| Exception::custom(error.to_string()))?
            }
            Self::Gemma4(_, model, cache) => *cache = model.new_cache(),
            Self::GptOss(_, model, cache) => *cache = model.new_cache(),
            Self::Inkling(_, model, cache) => *cache = model.new_cache(),
            Self::KimiLinear(_, model, cache) => *cache = model.new_cache(),
            Self::Lfm2(_, model, cache) => *cache = model.new_cache(),
            Self::PartitionedLlama(_, model, cache) => *cache = model.new_cache(),
            Self::ReplicatedText(_, model) => return model.reset_cache(),
            Self::MuseGlimmer(_, model, cache) => *cache = model.new_cache(),
            Self::NemotronH(_, model, cache) => *cache = model.new_cache(),
            Self::Qwen(_, model, cache) => *cache = model.new_cache(),
            Self::Qwen3Next(_, model, cache) => *cache = model.new_cache(),
            Self::Qwen3Vl(_, model, cache) | Self::Qwen3VlMoe(_, model, cache) => {
                *cache = model.new_cache()
            }
            Self::Qwen35(_, model, cache) => *cache = model.new_cache(),
        }
        Ok(())
    }

    /// Lazily catalogs a compatible persisted text prefix for a fresh cache.
    pub fn load_prompt_cache(
        &mut self,
        directory: impl AsRef<Path>,
        expected: &PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: PagedCacheOptions,
        stream: &Stream,
    ) -> Result<PromptCacheManifest, Exception> {
        macro_rules! load_into {
            ($model:expr, $cache:expr) => {{
                let (loaded, manifest) = $model
                    .load_prompt_cache(directory, expected, prefix_token_ids, options, stream)
                    .map_err(|error| Exception::custom(error.to_string()))?;
                *$cache = loaded;
                Ok(manifest)
            }};
        }
        match self {
            Self::DeepSeek(_, model, state) => load_into!(model, state),
            Self::Gemma4(_, model, cache) => load_into!(model, cache),
            Self::GptOss(_, model, cache) => load_into!(model, cache),
            Self::Inkling(_, model, cache) => load_into!(model, cache),
            Self::KimiLinear(_, model, cache) => load_into!(model, cache),
            Self::Lfm2(_, model, cache) => load_into!(model, cache),
            Self::PartitionedLlama(_, model, cache) => load_into!(model, cache),
            Self::ReplicatedText(_, model) => model
                .load_prompt_cache(directory.as_ref(), expected, prefix_token_ids)
                .map_err(|error| Exception::custom(error.to_string())),
            Self::MuseGlimmer(_, model, cache) => load_into!(model, cache),
            Self::NemotronH(_, model, cache) => load_into!(model, cache),
            Self::Qwen(_, model, cache) => load_into!(model, cache),
            Self::Qwen3Next(_, model, cache) => load_into!(model, cache),
            Self::Qwen3Vl(_, model, cache) | Self::Qwen3VlMoe(_, model, cache) => {
                load_into!(model, cache)
            }
            Self::Qwen35(_, model, cache) => load_into!(model, cache),
        }
    }

    /// Atomically saves a completed immutable prefix with model-owned state validation.
    pub fn save_prompt_cache(
        &mut self,
        destination: impl AsRef<Path>,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &PromptCacheOptions,
        stream: &Stream,
    ) -> Result<PromptCacheManifest, Exception> {
        macro_rules! save_from {
            ($model:expr, $cache:expr) => {
                $model
                    .save_prompt_cache(
                        $cache,
                        &destination,
                        descriptor,
                        prefix_token_ids,
                        options,
                        stream,
                    )
                    .map_err(|error| Exception::custom(error.to_string()))
            };
        }
        match self {
            Self::DeepSeek(_, model, state) => model
                .save_prompt_cache(state, &destination, descriptor, prefix_token_ids, options)
                .map_err(|error| Exception::custom(error.to_string())),
            Self::Gemma4(_, model, cache) => save_from!(model, cache),
            Self::GptOss(_, model, cache) => save_from!(model, cache),
            Self::Inkling(_, model, cache) => save_from!(model, cache),
            Self::KimiLinear(_, model, cache) => save_from!(model, cache),
            Self::Lfm2(_, model, cache) => save_from!(model, cache),
            Self::PartitionedLlama(_, model, cache) => save_from!(model, cache),
            Self::ReplicatedText(_, model) => model
                .save_prompt_cache(destination.as_ref(), descriptor, prefix_token_ids, options)
                .map_err(|error| Exception::custom(error.to_string())),
            Self::MuseGlimmer(_, model, cache) => save_from!(model, cache),
            Self::NemotronH(_, model, cache) => save_from!(model, cache),
            Self::Qwen(_, model, cache) => save_from!(model, cache),
            Self::Qwen3Next(_, model, cache) => save_from!(model, cache),
            Self::Qwen3Vl(_, model, cache) | Self::Qwen3VlMoe(_, model, cache) => {
                save_from!(model, cache)
            }
            Self::Qwen35(_, model, cache) => save_from!(model, cache),
        }
    }

    /// Returns aggregate cache-residency telemetry when paging is active.
    pub fn cache_residency_report(&self) -> Result<Option<CacheResidencyReport>, Exception> {
        match self {
            Self::DeepSeek(_, _, cache) => cache.residency_report(),
            Self::GptOss(_, _, cache) => cache.residency_report(),
            Self::Inkling(_, _, cache) => cache.target().residency_report(),
            Self::PartitionedLlama(_, _, cache)
            | Self::MuseGlimmer(_, _, cache)
            | Self::Qwen(_, _, cache) => cache.residency_report(),
            Self::ReplicatedText(_, model) => model.cache_residency_report(),
            Self::Gemma4(_, _, cache)
            | Self::KimiLinear(_, _, cache)
            | Self::Lfm2(_, _, cache)
            | Self::NemotronH(_, _, cache)
            | Self::Qwen3Next(_, _, cache)
            | Self::Qwen3Vl(_, _, cache)
            | Self::Qwen3VlMoe(_, _, cache)
            | Self::Qwen35(_, _, cache) => cache.residency_report(),
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
    use eredu_core::{cache::LayerCachePolicy, AttentionPolicy, LayerSchedule};
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
    fn qwen_cache_states_forward_paged_residency_reports() {
        let layout = paged_state_layout();
        let manager = cache_residency_manager();
        let qwen = MlxKeyValueState::paged(layout.clone(), manager.clone(), None).unwrap();
        let qwen_next = MlxHybridState::paged(layout.clone(), manager.clone(), None).unwrap();
        let qwen35 = MlxHybridState::paged(layout, manager, None).unwrap();

        assert!(qwen.residency_report().unwrap().is_some());
        assert!(qwen_next.residency_report().unwrap().is_some());
        assert!(qwen35.residency_report().unwrap().is_some());
    }
}
