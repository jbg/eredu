//! Architecture-erased model, cache, and generation dispatch.

use super::*;

/// Loaded model value for any architecture supported by this crate.
pub enum Model {
    /// DeepSeek-V3/R1 model.
    DeepSeekV3(deepseek_v3::Model),
    /// DeepSeek-V3/R1 model using bounded layer execution.
    DeepSeekV3Layerwise(crate::deepseek_v3::DeepSeekV3LayerwiseModel),
    /// Gemma 4 text model.
    Gemma4(gemma4::Model),
    /// Gemma 4 multimodal model using bounded layer execution.
    Gemma4Layerwise(crate::gemma4::Gemma4LayerwiseModel),
    /// OpenAI GPT-OSS model.
    GptOss(gpt_oss::Model),
    /// OpenAI GPT-OSS model using bounded layer execution.
    GptOssLayerwise(crate::gpt_oss::GptOssLayerwiseModel),
    /// Thinking Machines Lab Inkling model.
    Inkling(inkling::Model),
    /// Inkling multimodal model using bounded layer execution.
    InklingLayerwise(crate::inkling::InklingLayerwiseModel),
    /// Llama-compatible dense model.
    Llama(llama::ResidentModel),
    /// Llama-compatible model using the unified bounded layer API.
    LlamaLayerwise(crate::llama::LlamaModel),
    /// Liquid AI LFM2/LFM2.5 model.
    Lfm2(lfm2::Model),
    /// Liquid AI LFM2/LFM2.5 model using bounded layer execution.
    Lfm2Layerwise(crate::lfm2::Lfm2LayerwiseModel),
    /// Nemotron-H hybrid model.
    NemotronH(nemotron_h::Model),
    /// Nemotron-H hybrid model using bounded layer execution.
    NemotronHLayerwise(crate::nemotron_h::NemotronHLayerwiseModel),
    /// Qwen3 model.
    Qwen3(qwen3::Model),
    /// Qwen3 dense or sparse-MoE model using bounded layer execution.
    Qwen3Layerwise(crate::qwen3::Qwen3LayerwiseModel),
    /// Qwen3-Next model.
    Qwen3Next(qwen3_next::Model),
    /// Qwen3-Next model using shared hybrid bounded layer execution.
    Qwen3NextLayerwise(crate::qwen_hybrid::QwenHybridLayerwiseModel),
    /// Qwen3-VL multimodal model.
    Qwen3Vl(qwen3_vl::Model),
    /// Qwen3-VL multimodal model using vision/text bounded layer execution.
    Qwen3VlLayerwise(crate::qwen3_vl::Qwen3VlLayerwiseModel),
    /// Qwen3-VL-MoE multimodal model.
    Qwen3VlMoe(qwen3_vl_moe::Model),
    /// Qwen3-VL-MoE multimodal model using vision/text bounded layer execution.
    Qwen3VlMoeLayerwise(crate::qwen3_vl::Qwen3VlLayerwiseModel),
    /// Qwen3.5 dense or MoE model, optionally multimodal.
    Qwen35Moe(qwen3_5_moe::Model),
    /// Qwen3.5 model using shared vision/hybrid bounded layer execution.
    Qwen35MoeLayerwise(crate::qwen_hybrid::QwenHybridLayerwiseModel),
}

impl Model {
    /// Reports how this model architecture exposes MTP weights.
    pub fn mtp_capability(&self) -> MtpCapability {
        match self {
            Self::Gemma4(_) | Self::Gemma4Layerwise(_) => MtpCapability::Ready {
                checkpoint: MtpCheckpointKind::Separate,
            },
            Self::DeepSeekV3(model) if model.args.num_nextn_predict_layers > 0 => {
                MtpCapability::Unsupported {
                    checkpoint: MtpCheckpointKind::Embedded,
                    architecture: "deepseek_v3".into(),
                }
            }
            Self::DeepSeekV3Layerwise(model) if model.args().num_nextn_predict_layers > 0 => {
                MtpCapability::Unsupported {
                    checkpoint: MtpCheckpointKind::Embedded,
                    architecture: "deepseek_v3".into(),
                }
            }
            Self::Inkling(_) | Self::InklingLayerwise(_) => MtpCapability::Unsupported {
                checkpoint: MtpCheckpointKind::Embedded,
                architecture: "inkling".into(),
            },
            Self::Qwen3Next(model) if model.mtp_len() > 0 => MtpCapability::Ready {
                checkpoint: MtpCheckpointKind::Embedded,
            },
            Self::Qwen3NextLayerwise(model) if model.mtp_len() > 0 => MtpCapability::Ready {
                checkpoint: MtpCheckpointKind::Embedded,
            },
            Self::Qwen35Moe(model) if model.mtp_len() > 0 => MtpCapability::Ready {
                checkpoint: MtpCheckpointKind::Embedded,
            },
            Self::Qwen35MoeLayerwise(model) if model.mtp_len() > 0 => MtpCapability::Ready {
                checkpoint: MtpCheckpointKind::Embedded,
            },
            Self::NemotronH(_) | Self::NemotronHLayerwise(_) => MtpCapability::Unsupported {
                checkpoint: MtpCheckpointKind::Embedded,
                architecture: "nemotron_h".into(),
            },
            _ => MtpCapability::Unavailable,
        }
    }

    /// Generates with MTP using the default lossless sampling policy.
    pub fn generate_mtp_input(
        &mut self,
        drafter: &mut LoadedDrafter,
        cache: &mut ModelCache,
        input: input::ModelInput<'_>,
        config: &MtpConfig,
        prng_key: Option<Array>,
        stream: &Stream,
    ) -> Result<(Vec<u32>, MtpStats), Exception> {
        self.generate_mtp_input_with_sampler(
            drafter,
            cache,
            input,
            config,
            prng_key,
            &mut DefaultSampler,
            stream,
        )
    }

    /// Generates with MTP using a caller-provided lossless sampling policy.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_mtp_input_with_sampler<S: SpeculativeSampler + Clone>(
        &mut self,
        drafter: &mut LoadedDrafter,
        cache: &mut ModelCache,
        input: input::ModelInput<'_>,
        config: &MtpConfig,
        prng_key: Option<Array>,
        sampler: &mut S,
        stream: &Stream,
    ) -> Result<(Vec<u32>, MtpStats), Exception> {
        self.generate_mtp_input_with_sampler_callback_and_streams(
            drafter,
            cache,
            input,
            config,
            prng_key,
            sampler,
            MtpExecutionStreams::single(stream),
            |_| Ok(()),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn generate_mtp_input_with_sampler_callback_and_streams<S, F>(
        &mut self,
        drafter: &mut LoadedDrafter,
        cache: &mut ModelCache,
        input: input::ModelInput<'_>,
        config: &MtpConfig,
        prng_key: Option<Array>,
        sampler: &mut S,
        streams: MtpExecutionStreams<'_>,
        on_token: F,
    ) -> Result<(Vec<u32>, MtpStats), Exception>
    where
        S: SpeculativeSampler + Clone,
        F: FnMut(u32) -> Result<(), Exception>,
    {
        self.generate_mtp_input_with_sampler_callback_and_streams_and_options(
            drafter,
            cache,
            input,
            config,
            prng_key,
            sampler,
            streams,
            MtpSchedulerOptions::default(),
            on_token,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn generate_mtp_input_with_sampler_callback_and_streams_and_options<S, F>(
        &mut self,
        drafter: &mut LoadedDrafter,
        cache: &mut ModelCache,
        input: input::ModelInput<'_>,
        config: &MtpConfig,
        prng_key: Option<Array>,
        sampler: &mut S,
        streams: MtpExecutionStreams<'_>,
        scheduler_options: MtpSchedulerOptions,
        on_token: F,
    ) -> Result<(Vec<u32>, MtpStats), Exception>
    where
        S: SpeculativeSampler + Clone,
        F: FnMut(u32) -> Result<(), Exception>,
    {
        let assistant = drafter.gemma4_mut();
        match (self, cache) {
            (Self::Gemma4(target), ModelCache::Gemma4(cache)) => {
                validate_gemma4_drafter(&target.args, assistant)?;
                crate::gemma4_mtp::generate_with_streams_and_callback_and_options(
                    target,
                    assistant,
                    cache,
                    input,
                    config,
                    prng_key,
                    sampler,
                    streams,
                    scheduler_options,
                    on_token,
                )
            }
            (Self::Gemma4Layerwise(target), ModelCache::Gemma4(cache)) => {
                validate_gemma4_drafter(target.args(), assistant)?;
                crate::gemma4_mtp::generate_with_streams_and_callback_and_options(
                    target,
                    assistant,
                    cache,
                    input,
                    config,
                    prng_key,
                    sampler,
                    streams,
                    scheduler_options,
                    on_token,
                )
            }
            (model, _) => Err(Exception::custom(format!(
                "MTP runtime adapter is unavailable for model type {} ({:?})",
                model.model_type(),
                model.mtp_capability()
            ))),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn generate_mtp_input_with_semantics_and_options<S, F>(
        &mut self,
        drafter: &mut LoadedDrafter,
        cache: &mut ModelCache,
        input: input::ModelInput<'_>,
        config: &MtpConfig,
        prng_key: Option<Array>,
        sampler: &mut S,
        semantic: Box<dyn MtpSemanticState>,
        streams: MtpExecutionStreams<'_>,
        scheduler_options: MtpSchedulerOptions,
        on_event: F,
    ) -> Result<(Vec<u32>, MtpStats, FinishReason), Exception>
    where
        S: SpeculativeSampler + Clone,
        F: FnMut(SemanticEvent),
    {
        let assistant = drafter.gemma4_mut();
        match (self, cache) {
            (Self::Gemma4(target), ModelCache::Gemma4(cache)) => {
                validate_gemma4_drafter(&target.args, assistant)?;
                crate::gemma4_mtp::generate_with_semantics_and_options(
                    target,
                    assistant,
                    cache,
                    input,
                    config,
                    prng_key,
                    sampler,
                    semantic,
                    streams,
                    scheduler_options,
                    on_event,
                )
            }
            (Self::Gemma4Layerwise(target), ModelCache::Gemma4(cache)) => {
                validate_gemma4_drafter(target.args(), assistant)?;
                crate::gemma4_mtp::generate_with_semantics_and_options(
                    target,
                    assistant,
                    cache,
                    input,
                    config,
                    prng_key,
                    sampler,
                    semantic,
                    streams,
                    scheduler_options,
                    on_event,
                )
            }
            (model, _) => Err(Exception::custom(format!(
                "MTP runtime adapter is unavailable for model type {} ({:?})",
                model.model_type(),
                model.mtp_capability()
            ))),
        }
    }

    /// Generates with MTP weights embedded in the target checkpoint.
    pub fn generate_embedded_mtp_input(
        &mut self,
        cache: &mut ModelCache,
        input: input::ModelInput<'_>,
        config: &MtpConfig,
        prng_key: Option<Array>,
        stream: &Stream,
    ) -> Result<(Vec<u32>, MtpStats), Exception> {
        self.generate_embedded_mtp_input_with_sampler(
            cache,
            input,
            config,
            prng_key,
            &mut DefaultSampler,
            stream,
        )
    }

    /// Generates with embedded MTP weights and a caller-provided sampler.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_embedded_mtp_input_with_sampler<S: SpeculativeSampler + Clone>(
        &mut self,
        cache: &mut ModelCache,
        input: input::ModelInput<'_>,
        config: &MtpConfig,
        prng_key: Option<Array>,
        sampler: &mut S,
        stream: &Stream,
    ) -> Result<(Vec<u32>, MtpStats), Exception> {
        self.generate_embedded_mtp_input_with_sampler_callback(
            cache,
            input,
            config,
            prng_key,
            sampler,
            stream,
            |_| Ok(()),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn generate_embedded_mtp_input_with_sampler_callback<S, F>(
        &mut self,
        cache: &mut ModelCache,
        input: input::ModelInput<'_>,
        config: &MtpConfig,
        prng_key: Option<Array>,
        sampler: &mut S,
        stream: &Stream,
        on_token: F,
    ) -> Result<(Vec<u32>, MtpStats), Exception>
    where
        S: SpeculativeSampler + Clone,
        F: FnMut(u32) -> Result<(), Exception>,
    {
        match (self, cache) {
            (Self::Qwen3Next(target), ModelCache::Qwen3Next(cache))
            | (Self::Qwen35Moe(target), ModelCache::Qwen35Moe(cache)) => {
                crate::qwen_mtp::generate_with_callback(
                    target, cache, input, config, prng_key, sampler, stream, on_token,
                )
            }
            (Self::Qwen3NextLayerwise(target), ModelCache::Qwen3Next(cache))
            | (Self::Qwen35MoeLayerwise(target), ModelCache::Qwen35Moe(cache)) => {
                crate::qwen_mtp::generate_with_callback(
                    target, cache, input, config, prng_key, sampler, stream, on_token,
                )
            }
            (model, _) => Err(Exception::custom(format!(
                "embedded MTP runtime adapter is unavailable for model type {} ({:?})",
                model.model_type(),
                model.mtp_capability()
            ))),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn generate_embedded_mtp_input_with_semantics_and_options<S, F>(
        &mut self,
        cache: &mut ModelCache,
        input: input::ModelInput<'_>,
        config: &MtpConfig,
        prng_key: Option<Array>,
        sampler: &mut S,
        semantic: Box<dyn MtpSemanticState>,
        stream: &Stream,
        scheduler_options: MtpSchedulerOptions,
        on_event: F,
    ) -> Result<(Vec<u32>, MtpStats, FinishReason), Exception>
    where
        S: SpeculativeSampler + Clone,
        F: FnMut(SemanticEvent),
    {
        match (self, cache) {
            (Self::Qwen3Next(target), ModelCache::Qwen3Next(cache))
            | (Self::Qwen35Moe(target), ModelCache::Qwen35Moe(cache)) => {
                crate::qwen_mtp::generate_with_semantics_and_options(
                    target,
                    cache,
                    input,
                    config,
                    prng_key,
                    sampler,
                    semantic,
                    stream,
                    scheduler_options,
                    on_event,
                )
            }
            (Self::Qwen3NextLayerwise(target), ModelCache::Qwen3Next(cache))
            | (Self::Qwen35MoeLayerwise(target), ModelCache::Qwen35Moe(cache)) => {
                crate::qwen_mtp::generate_with_semantics_and_options(
                    target,
                    cache,
                    input,
                    config,
                    prng_key,
                    sampler,
                    semantic,
                    stream,
                    scheduler_options,
                    on_event,
                )
            }
            (model, _) => Err(Exception::custom(format!(
                "embedded MTP runtime adapter is unavailable for model type {} ({:?})",
                model.model_type(),
                model.mtp_capability()
            ))),
        }
    }

    /// Returns residency telemetry when this model uses bounded layer execution.
    pub fn residency_report(
        &self,
    ) -> Result<Option<crate::runtime::residency::manager::ResidencyReport>, Error> {
        match self {
            Self::DeepSeekV3Layerwise(model) => Ok(Some(model.residency_report()?)),
            Self::Gemma4Layerwise(model) => Ok(Some(model.residency_report()?)),
            Self::InklingLayerwise(model) => Ok(Some(model.residency_report()?)),
            Self::LlamaLayerwise(model) => model.residency_report(),
            Self::GptOssLayerwise(model) => Ok(Some(model.residency_report()?)),
            Self::Lfm2Layerwise(model) => Ok(Some(model.residency_report()?)),
            Self::NemotronHLayerwise(model) => Ok(Some(model.residency_report()?)),
            Self::Qwen3NextLayerwise(model) | Self::Qwen35MoeLayerwise(model) => {
                Ok(Some(model.residency_report()?))
            }
            Self::Qwen3Layerwise(model) => Ok(Some(model.residency_report()?)),
            Self::Qwen3VlLayerwise(model) | Self::Qwen3VlMoeLayerwise(model) => {
                Ok(Some(model.residency_report()?))
            }
            _ => Ok(None),
        }
    }

    /// Returns experimental dense-stream telemetry when enabled.
    pub fn dense_stream_report(
        &self,
    ) -> Result<Option<crate::runtime::execution::layerwise::DenseDiskStreamReport>, Error> {
        match self {
            Self::DeepSeekV3Layerwise(model) => model.dense_stream_report(),
            Self::Gemma4Layerwise(model) => model.dense_stream_report(),
            Self::InklingLayerwise(model) => model.dense_stream_report(),
            Self::LlamaLayerwise(model) => model.dense_stream_report(),
            Self::GptOssLayerwise(model) => model.dense_stream_report(),
            Self::Lfm2Layerwise(model) => model.dense_stream_report(),
            Self::NemotronHLayerwise(model) => model.dense_stream_report(),
            Self::Qwen3NextLayerwise(model) | Self::Qwen35MoeLayerwise(model) => {
                model.dense_stream_report()
            }
            Self::Qwen3Layerwise(model) => model.dense_stream_report(),
            Self::Qwen3VlLayerwise(model) | Self::Qwen3VlMoeLayerwise(model) => {
                model.dense_stream_report()
            }
            _ => Ok(None),
        }
    }

    /// Returns sparse routed-expert cache telemetry when enabled.
    pub fn expert_cache_report(
        &self,
    ) -> Result<Option<crate::runtime::residency::expert_cache::ExpertCacheReport>, Error> {
        match self {
            Self::DeepSeekV3Layerwise(model) => model.expert_cache_report(),
            Self::GptOssLayerwise(model) => model.expert_cache_report(),
            Self::InklingLayerwise(model) => model.expert_cache_report(),
            Self::Lfm2Layerwise(model) => model.expert_cache_report(),
            Self::NemotronHLayerwise(model) => model.expert_cache_report(),
            Self::Qwen3Layerwise(model) => model.expert_cache_report(),
            Self::Qwen3NextLayerwise(model) | Self::Qwen35MoeLayerwise(model) => {
                model.expert_cache_report()
            }
            Self::Qwen3VlMoeLayerwise(model) => model.expert_cache_report(),
            _ => Ok(None),
        }
    }

    /// Returns the effective model type used for dispatch.
    pub fn model_type(&self) -> &str {
        match self {
            Self::DeepSeekV3(model) => model.model_type(),
            Self::DeepSeekV3Layerwise(model) => &model.args().model_type,
            Self::Gemma4(model) => model.model_type(),
            Self::Gemma4Layerwise(model) => &model.args().model_type,
            Self::GptOss(model) => model.model_type(),
            Self::GptOssLayerwise(model) => &model.args().model_type,
            Self::Inkling(model) => model.model_type(),
            Self::InklingLayerwise(model) => &model.args().model_type,
            Self::Llama(model) => model.model_type(),
            Self::LlamaLayerwise(model) => &model.args().model_type,
            Self::Lfm2(model) => model.model_type(),
            Self::Lfm2Layerwise(model) => &model.args().model_type,
            Self::NemotronH(model) => model.model_type(),
            Self::NemotronHLayerwise(model) => &model.args().model_type,
            Self::Qwen3(model) => model.model_type(),
            Self::Qwen3Layerwise(model) => &model.args().model_type,
            Self::Qwen3Next(model) => model.model_type(),
            Self::Qwen3NextLayerwise(model) => &model.args().model_type,
            Self::Qwen3Vl(model) => model.model_type(),
            Self::Qwen3VlLayerwise(model) => model.model_type(),
            Self::Qwen3VlMoe(model) => model.model_type(),
            Self::Qwen3VlMoeLayerwise(model) => model.model_type(),
            Self::Qwen35Moe(model) => model.model_type(),
            Self::Qwen35MoeLayerwise(model) => &model.args().model_type,
        }
    }

    /// Returns checkpoint-native quantization storage statistics when available.
    pub fn native_quantization_stats(
        &self,
    ) -> Option<&safemlx::native_quantization::NativeQuantizationStats> {
        match self {
            Self::Gemma4(model) => Some(&model.native_quantization_stats),
            _ => None,
        }
    }

    /// Returns the canonical cache-relevant architecture identity derived from the loaded model.
    pub fn prompt_cache_architecture_fingerprint(&self) -> Result<String, Exception> {
        match self {
            Self::Llama(model) => Ok(llama::prompt_cache_architecture_fingerprint(&model.args)),
            Self::LlamaLayerwise(model) => {
                Ok(llama::prompt_cache_architecture_fingerprint(model.args()))
            }
            Self::DeepSeekV3(model) => Ok(deepseek_v3::prompt_cache_architecture_fingerprint(
                &model.args,
            )),
            Self::DeepSeekV3Layerwise(model) => Ok(
                deepseek_v3::prompt_cache_architecture_fingerprint(model.args()),
            ),
            Self::GptOss(model) => Ok(gpt_oss::prompt_cache_architecture_fingerprint(&model.args)),
            Self::GptOssLayerwise(model) => {
                Ok(gpt_oss::prompt_cache_architecture_fingerprint(model.args()))
            }
            _ => Err(Exception::custom(format!(
                "prompt-cache architecture identity is unsupported for model type {}",
                self.model_type()
            ))),
        }
    }

    /// Runs a detailed instrumented forward pass for supported model families.
    ///
    /// DeepSeek-V3/R1, Llama, Qwen3, Qwen3.5 MoE, and Gemma4 currently report
    /// detailed layer activations. Other families return an error until their
    /// family-specific inspection paths are wired.
    pub fn forward_with_observer(
        &mut self,
        input_tokens: &Array,
        mask: Option<&Array>,
        cache: &mut ModelCache,
        stream: &Stream,
        observer: &mut impl ActivationObserver,
    ) -> Result<Array, Exception> {
        match (self, cache) {
            (Self::DeepSeekV3(model), ModelCache::DeepSeekV3(cache)) => model
                .forward_with_observer(
                    deepseek_v3::ModelInput {
                        inputs: input_tokens,
                        mask,
                        cache: Some(cache),
                    },
                    stream,
                    observer,
                ),
            (Self::DeepSeekV3Layerwise(_), ModelCache::DeepSeekV3(_)) => Err(Exception::custom(
                "detailed activation inspection is unavailable for bounded-layer DeepSeek-V3 execution",
            )),
            (Self::Llama(model), ModelCache::KeyValue(cache)) => model.forward_with_observer(
                llama::ModelInput {
                    inputs: input_tokens,
                    mask,
                    cache,
                },
                stream,
                observer,
            ),
            (Self::Llama(model), ModelCache::SlidingKeyValue(cache)) => model
                .forward_with_observer(
                    llama::ModelInput {
                        inputs: input_tokens,
                        mask,
                        cache,
                    },
                    stream,
                    observer,
                ),
            (Self::Llama(_), ModelCache::PagedKeyValue(_)) => Err(Exception::custom(
                "detailed attention inspection is unavailable for paged key/value caches",
            )),
            (Self::LlamaLayerwise(_), ModelCache::LlamaLayerwise(_)) => Err(Exception::custom(
                "detailed activation inspection is unavailable for bounded-layer Llama execution",
            )),
            (Self::Qwen3(model), ModelCache::KeyValue(cache)) => model.forward_with_observer(
                qwen3::ModelInput {
                    inputs: input_tokens,
                    mask,
                    cache,
                },
                stream,
                observer,
            ),
            (Self::Qwen3Layerwise(_), ModelCache::KeyValue(_)) => Err(Exception::custom(
                "detailed activation inspection is unavailable for bounded-layer Qwen3 execution",
            )),
            (Self::Qwen35Moe(model), ModelCache::Qwen35Moe(cache)) => model.forward_with_observer(
                qwen3_5_moe::ModelInput {
                    inputs: input_tokens,
                    inputs_embeds: None,
                    mask,
                    cache: Some(cache),
                },
                stream,
                observer,
            ),
            (Self::Qwen35MoeLayerwise(_), ModelCache::Qwen35Moe(_)) => Err(Exception::custom(
                "detailed activation inspection is unavailable for bounded-layer Qwen3.5 execution",
            )),
            (Self::Qwen3Next(model), ModelCache::Qwen3Next(cache)) => model.forward_with_observer(
                qwen3_next::ModelInput {
                    inputs: input_tokens,
                    inputs_embeds: None,
                    mask,
                    cache: Some(cache),
                },
                stream,
                observer,
            ),
            (Self::Qwen3NextLayerwise(_), ModelCache::Qwen3Next(_)) => Err(Exception::custom(
                "detailed activation inspection is unavailable for bounded-layer Qwen3-Next execution",
            )),
            (Self::Gemma4(model), ModelCache::Gemma4(cache)) => model.forward_with_observer(
                gemma4::ModelInput {
                    inputs: input_tokens,
                    inputs_embeds: None,
                    per_layer_input_ids: None,
                    mask,
                    sliding_mask: None,
                    cache: &mut cache.kv,
                },
                stream,
                observer,
            ),
            (Self::Gemma4Layerwise(_), ModelCache::Gemma4(_)) => Err(Exception::custom(
                "detailed activation inspection is unavailable for bounded-layer Gemma 4 execution",
            )),
            (Self::NemotronH(_) | Self::NemotronHLayerwise(_), _) => Err(Exception::custom(
                "detailed activation inspection is not implemented for nemotron_h yet",
            )),
            (Self::Lfm2(_) | Self::Lfm2Layerwise(_), _) => Err(Exception::custom(
                "detailed activation inspection is not implemented for lfm2 yet",
            )),
            (Self::Qwen3Vl(_) | Self::Qwen3VlLayerwise(_), _) => Err(Exception::custom(
                "detailed activation inspection is not implemented for qwen3_vl yet",
            )),
            (Self::Qwen3VlMoe(_) | Self::Qwen3VlMoeLayerwise(_), _) => Err(Exception::custom(
                "detailed activation inspection is not implemented for qwen3_vl_moe yet",
            )),
            (Self::GptOss(_) | Self::GptOssLayerwise(_), _) => Err(Exception::custom(
                "detailed activation inspection is not implemented for gpt_oss yet",
            )),
            (Self::Inkling(_) | Self::InklingLayerwise(_), _) => Err(Exception::custom(
                "detailed activation inspection is not implemented for Inkling yet",
            )),
            _ => Err(Exception::custom(
                "model cache type does not match model kind",
            )),
        }
    }

    /// Computes initial prompt logits while reporting detailed activations.
    ///
    /// This mirrors each model family's prefill semantics and returns logits for
    /// the final prompt token with shape `[batch, vocab]`. Gemma4 uses a split
    /// prefill internally, so callers that want faithful instrumented generation
    /// should use this instead of calling [`Model::forward_with_observer`]
    /// directly on the whole prompt.
    pub fn prefill_input_with_observer(
        &mut self,
        input: input::ModelInput<'_>,
        cache: &mut ModelCache,
        stream: &Stream,
        observer: &mut impl ActivationObserver,
    ) -> Result<Array, Exception> {
        match (self, cache) {
            (Self::DeepSeekV3(model), ModelCache::DeepSeekV3(cache)) => {
                let prompt_tokens = input::text_token_ids(input, stream)?;
                let logits = model.forward_with_observer(
                    deepseek_v3::ModelInput {
                        inputs: &prompt_tokens,
                        mask: None,
                        cache: Some(cache),
                    },
                    stream,
                    observer,
                )?;
                final_token_logits(&logits, stream)
            }
            (Self::Gemma4(model), ModelCache::Gemma4(cache)) => {
                model.prefill_typed_with_observer(input, cache, stream, observer)
            }
            (Self::Gemma4Layerwise(_), ModelCache::Gemma4(_)) => Err(Exception::custom(
                "detailed activation inspection is unavailable for bounded-layer Gemma 4 execution",
            )),
            (Self::Llama(model), ModelCache::KeyValue(cache)) => {
                let prompt_tokens = input::text_token_ids(input, stream)?;
                let logits = model.forward_with_observer(
                    llama::ModelInput {
                        inputs: &prompt_tokens,
                        mask: None,
                        cache,
                    },
                    stream,
                    observer,
                )?;
                final_token_logits(&logits, stream)
            }
            (Self::Llama(model), ModelCache::SlidingKeyValue(cache)) => {
                let prompt_tokens = input::text_token_ids(input, stream)?;
                let logits = model.forward_with_observer(
                    llama::ModelInput {
                        inputs: &prompt_tokens,
                        mask: None,
                        cache,
                    },
                    stream,
                    observer,
                )?;
                final_token_logits(&logits, stream)
            }
            (Self::Llama(_), ModelCache::PagedKeyValue(_)) => Err(Exception::custom(
                "detailed attention inspection is unavailable for paged key/value caches",
            )),
            (Self::LlamaLayerwise(_), ModelCache::LlamaLayerwise(_)) => Err(Exception::custom(
                "detailed activation inspection is unavailable for bounded-layer Llama execution",
            )),
            (Self::Qwen3(model), ModelCache::KeyValue(cache)) => {
                let prompt_tokens = input::text_token_ids(input, stream)?;
                let logits = model.forward_with_observer(
                    qwen3::ModelInput {
                        inputs: &prompt_tokens,
                        mask: None,
                        cache,
                    },
                    stream,
                    observer,
                )?;
                final_token_logits(&logits, stream)
            }
            (Self::Qwen3Layerwise(_), ModelCache::KeyValue(_)) => Err(Exception::custom(
                "detailed activation inspection is unavailable for bounded-layer Qwen3 execution",
            )),
            (Self::Qwen35Moe(model), ModelCache::Qwen35Moe(cache)) => {
                model.prefill_typed_with_observer(input, cache, stream, observer)
            }
            (Self::Qwen35MoeLayerwise(_), ModelCache::Qwen35Moe(_)) => Err(Exception::custom(
                "detailed activation inspection is unavailable for bounded-layer Qwen3.5 execution",
            )),
            (Self::Qwen3Next(model), ModelCache::Qwen3Next(cache)) => {
                model.prefill_typed_with_observer(input, cache, stream, observer)
            }
            (Self::Qwen3NextLayerwise(_), ModelCache::Qwen3Next(_)) => Err(Exception::custom(
                "detailed activation inspection is unavailable for bounded-layer Qwen3-Next execution",
            )),
            (Self::NemotronH(_) | Self::NemotronHLayerwise(_), _) => Err(Exception::custom(
                "detailed activation inspection is not implemented for nemotron_h yet",
            )),
            (Self::Lfm2(_) | Self::Lfm2Layerwise(_), _) => Err(Exception::custom(
                "detailed activation inspection is not implemented for lfm2 yet",
            )),
            (Self::Inkling(_), _) => Err(Exception::custom(
                "detailed activation inspection is not implemented for Inkling yet",
            )),
            (Self::Qwen3Vl(_) | Self::Qwen3VlLayerwise(_), _) => Err(Exception::custom(
                "detailed activation inspection is not implemented for qwen3_vl yet",
            )),
            (Self::Qwen3VlMoe(_) | Self::Qwen3VlMoeLayerwise(_), _) => Err(Exception::custom(
                "detailed activation inspection is not implemented for qwen3_vl_moe yet",
            )),
            _ => Err(Exception::custom(
                "model cache type does not match model kind",
            )),
        }
    }

    /// Creates an empty cache value appropriate for this model.
    pub fn new_cache(&self) -> ModelCache {
        match self {
            Self::DeepSeekV3(model) => ModelCache::DeepSeekV3(model.new_cache()),
            Self::DeepSeekV3Layerwise(model) => ModelCache::DeepSeekV3(model.new_cache()),
            Self::Gemma4(model) => ModelCache::Gemma4(model.new_cache()),
            Self::Gemma4Layerwise(model) => ModelCache::Gemma4(model.new_cache()),
            Self::GptOss(model) => ModelCache::GptOss(model.new_cache()),
            Self::GptOssLayerwise(model) => ModelCache::GptOss(model.new_cache()),
            Self::Inkling(model) => ModelCache::Inkling(model.new_cache()),
            Self::InklingLayerwise(model) => ModelCache::Inkling(model.new_cache()),
            Self::Llama(model) => match model.sliding_window() {
                Some(_) => ModelCache::SlidingKeyValue(model.new_sliding_cache()),
                None => ModelCache::KeyValue(Vec::new()),
            },
            Self::LlamaLayerwise(model) => ModelCache::LlamaLayerwise(model.new_cache()),
            Self::Lfm2(model) => ModelCache::Lfm2(model.new_cache()),
            Self::Lfm2Layerwise(model) => ModelCache::Lfm2(model.new_cache()),
            Self::Qwen3(_) => ModelCache::KeyValue(Vec::new()),
            Self::Qwen3Layerwise(model) => ModelCache::KeyValue(model.new_cache()),
            Self::Qwen3Next(model) => ModelCache::Qwen3Next(model.new_cache()),
            Self::Qwen3NextLayerwise(model) => ModelCache::Qwen3Next(model.new_cache()),
            Self::Qwen3Vl(model) => ModelCache::Qwen3Vl(model.new_cache()),
            Self::Qwen3VlLayerwise(model) => ModelCache::Qwen3Vl(model.new_cache()),
            Self::Qwen3VlMoe(model) => ModelCache::Qwen3VlMoe(model.new_cache()),
            Self::Qwen3VlMoeLayerwise(model) => ModelCache::Qwen3VlMoe(model.new_cache()),
            Self::NemotronH(model) => ModelCache::NemotronH(model.new_cache()),
            Self::NemotronHLayerwise(model) => ModelCache::NemotronH(model.new_cache()),
            Self::Qwen35Moe(model) => ModelCache::Qwen35Moe(model.new_cache()),
            Self::Qwen35MoeLayerwise(model) => ModelCache::Qwen35Moe(model.new_cache()),
        }
    }

    /// Creates ordinary cache state or an explicitly bounded paged cache.
    ///
    /// Paged construction is currently supported for Llama-compatible text
    /// attention, DeepSeek compressed-latent attention, GPT-OSS, Inkling
    /// relative-position attention, and the corresponding bounded
    /// weight-execution wrappers. Other cache representations return a precise
    /// unsupported error and retain their device-resident defaults.
    pub fn new_cache_with_options(
        &self,
        policy: CacheResidencyPolicy,
    ) -> Result<ModelCache, Exception> {
        match policy {
            CacheResidencyPolicy::Device => Ok(self.new_cache()),
            CacheResidencyPolicy::Paged(options) => match self {
                Self::Llama(model) => {
                    let manager = CacheResidencyManager::new(options)
                        .map_err(|error| Exception::custom(error.to_string()))?;
                    let layer_count = usize::try_from(model.args.num_hidden_layers)
                        .map_err(|_| Exception::custom("invalid Llama cache layer count"))?;
                    let caches = (0..layer_count)
                        .map(|layer| {
                            PagedKeyValueCache::new(manager.clone(), layer, model.sliding_window())
                                .map(Some)
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    Ok(ModelCache::PagedKeyValue(caches))
                }
                Self::LlamaLayerwise(model) => model
                    .new_cache_with_options(CacheResidencyPolicy::Paged(options))
                    .map(ModelCache::LlamaLayerwise)
                    .map_err(|error| Exception::custom(error.to_string())),
                Self::DeepSeekV3(model) => model
                    .new_cache_with_options(CacheResidencyPolicy::Paged(options))
                    .map(ModelCache::DeepSeekV3),
                Self::DeepSeekV3Layerwise(model) => model
                    .new_cache_with_options(CacheResidencyPolicy::Paged(options))
                    .map(ModelCache::DeepSeekV3)
                    .map_err(|error| Exception::custom(error.to_string())),
                Self::GptOss(model) => model
                    .new_cache_with_options(CacheResidencyPolicy::Paged(options))
                    .map(ModelCache::GptOss),
                Self::GptOssLayerwise(model) => model
                    .new_cache_with_options(CacheResidencyPolicy::Paged(options))
                    .map(ModelCache::GptOss)
                    .map_err(|error| Exception::custom(error.to_string())),
                Self::Inkling(model) => model.new_paged_cache(options).map(ModelCache::Inkling),
                Self::InklingLayerwise(model) => model
                    .new_cache_with_options(CacheResidencyPolicy::Paged(options))
                    .map(ModelCache::Inkling)
                    .map_err(|error| Exception::custom(error.to_string())),
                _ => Err(Exception::custom(format!(
                    "paged cache residency is unsupported for model type {}",
                    self.model_type()
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
    ) -> Result<(ModelCache, PromptCacheManifest), Exception> {
        match self {
            Self::Llama(model) => {
                let layer_count = usize::try_from(model.args.num_hidden_layers)
                    .map_err(|_| Exception::custom("invalid Llama cache layer count"))?;
                let identity = PromptCacheModelIdentity {
                        model_family: "llama".into(),
                        effective_model_type: model.args.model_type.clone(),
                        architecture_fingerprint:
                            llama::prompt_cache_architecture_fingerprint(&model.args),
                        layer_count,
                        global_layer_start: 0,
                        global_layer_end: layer_count,
                        sliding_window: model.sliding_window(),
                        sink_tokens: 0,
                        topology: Default::default(),
                        layer_layouts: PromptCacheModelIdentity::key_value_layouts(
                            layer_count,
                            model.args.num_key_value_heads,
                            model.args.head_dim,
                        ),
                    };
                validate_prompt_cache_model_identity(expected, &identity)
                .map_err(|error| Exception::custom(error.to_string()))?;
                let (manager, manifest) = open_prompt_cache(
                    directory,
                    expected,
                    &identity,
                    prefix_token_ids,
                    options,
                )
                .map_err(|error| Exception::custom(error.to_string()))?;
                let caches = (0..layer_count)
                    .map(|layer| {
                        PagedKeyValueCache::new(
                            manager.clone(),
                            layer,
                            model.sliding_window(),
                        )
                        .map(Some)
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok((ModelCache::PagedKeyValue(caches), manifest))
            }
            Self::LlamaLayerwise(model) => model
                .load_prompt_cache(
                    directory,
                    expected,
                    prefix_token_ids,
                    options,
                )
                .map(|(cache, manifest)| (ModelCache::LlamaLayerwise(cache), manifest))
                .map_err(|error| Exception::custom(error.to_string())),
            Self::DeepSeekV3(model) => model
                .load_prompt_cache(directory, expected, prefix_token_ids, options)
                .map(|(cache, manifest)| (ModelCache::DeepSeekV3(cache), manifest)),
            Self::DeepSeekV3Layerwise(model) => model
                .load_prompt_cache(directory, expected, prefix_token_ids, options)
                .map(|(cache, manifest)| (ModelCache::DeepSeekV3(cache), manifest))
                .map_err(|error| Exception::custom(error.to_string())),
            Self::GptOss(model) => model
                .load_prompt_cache(directory, expected, prefix_token_ids, options)
                .map(|(cache, manifest)| (ModelCache::GptOss(cache), manifest)),
            Self::GptOssLayerwise(model) => model
                .load_prompt_cache(directory, expected, prefix_token_ids, options)
                .map(|(cache, manifest)| (ModelCache::GptOss(cache), manifest))
                .map_err(|error| Exception::custom(error.to_string())),
            _ => Err(Exception::custom(
                "prompt-cache loading is unsupported for this model cache representation; multimodal and recurrent prefixes require additional identity state",
            )),
        }
    }

    /// Computes logits for an initial typed input using a cache returned by [`Model::new_cache`].
    pub fn prefill_input_with_cache(
        &mut self,
        input: input::ModelInput<'_>,
        cache: &mut ModelCache,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        match (self, cache) {
            (Self::Gemma4(model), ModelCache::Gemma4(cache)) => {
                model.prefill_input_logits(input, cache, stream)
            }
            (Self::Gemma4Layerwise(model), ModelCache::Gemma4(cache)) => {
                model.prefill_input_logits(input, cache, stream)
            }
            (Self::GptOss(model), ModelCache::GptOss(cache)) => {
                model.prefill_input_logits(input, cache, stream)
            }
            (Self::GptOssLayerwise(model), ModelCache::GptOss(cache)) => {
                model.prefill_input_logits(input, cache, stream)
            }
            (Self::Inkling(model), ModelCache::Inkling(cache)) => {
                model.prefill_input_logits(input, cache, stream)
            }
            (Self::InklingLayerwise(model), ModelCache::Inkling(cache)) => {
                model.prefill_input_logits(input, cache, stream)
            }
            (Self::Llama(model), ModelCache::KeyValue(cache)) => {
                model.prefill_input_logits(input, cache, stream)
            }
            (Self::Llama(model), ModelCache::SlidingKeyValue(cache)) => {
                model.prefill_input_logits(input, cache, stream)
            }
            (Self::Llama(model), ModelCache::PagedKeyValue(cache)) => {
                model.prefill_input_logits(input, cache, stream)
            }
            (Self::LlamaLayerwise(model), ModelCache::LlamaLayerwise(cache)) => {
                model.prefill_input_logits(input, cache, stream)
            }
            (Self::Lfm2(model), ModelCache::Lfm2(cache)) => {
                model.prefill_input_logits(input, cache, stream)
            }
            (Self::Lfm2Layerwise(model), ModelCache::Lfm2(cache)) => {
                model.prefill_input_logits(input, cache, stream)
            }
            (Self::NemotronH(model), ModelCache::NemotronH(cache)) => {
                model.prefill_input_logits(input, cache, stream)
            }
            (Self::NemotronHLayerwise(model), ModelCache::NemotronH(cache)) => {
                model.prefill_input_logits(input, cache, stream)
            }
            (Self::Qwen3(model), ModelCache::KeyValue(cache)) => {
                model.prefill_input_logits(input, cache, stream)
            }
            (Self::Qwen3Layerwise(model), ModelCache::KeyValue(cache)) => {
                model.prefill_input_logits(input, cache, stream)
            }
            (Self::Qwen3Vl(model), ModelCache::Qwen3Vl(cache)) => {
                model.prefill_input_logits(input, cache, stream)
            }
            (Self::Qwen3VlLayerwise(model), ModelCache::Qwen3Vl(cache)) => {
                model.prefill_input_logits(input, cache, stream)
            }
            (Self::Qwen3VlMoe(model), ModelCache::Qwen3VlMoe(cache)) => {
                model.prefill_input_logits(input, cache, stream)
            }
            (Self::Qwen3VlMoeLayerwise(model), ModelCache::Qwen3VlMoe(cache)) => {
                model.prefill_input_logits(input, cache, stream)
            }
            (Self::Qwen3Next(model), ModelCache::Qwen3Next(cache)) => {
                model.prefill_input_logits(input, cache, stream)
            }
            (Self::Qwen3NextLayerwise(model), ModelCache::Qwen3Next(cache)) => {
                model.prefill_input_logits(input, cache, stream)
            }
            (Self::Qwen35Moe(model), ModelCache::Qwen35Moe(cache)) => {
                model.prefill_input_logits(input, cache, stream)
            }
            (Self::Qwen35MoeLayerwise(model), ModelCache::Qwen35Moe(cache)) => {
                model.prefill_input_logits(input, cache, stream)
            }
            (Self::DeepSeekV3(model), ModelCache::DeepSeekV3(cache)) => {
                model.prefill_input_logits(input, cache, stream)
            }
            (Self::DeepSeekV3Layerwise(model), ModelCache::DeepSeekV3(cache)) => {
                model.prefill_input_logits(input, cache, stream)
            }
            _ => Err(Exception::custom(
                "model cache type does not match model kind",
            )),
        }
    }

    /// Creates a token iterator from typed input using a cache returned by [`Model::new_cache`].
    pub fn generate_input_with_cache<'a>(
        &'a mut self,
        cache: &'a mut ModelCache,
        temp: f32,
        input: input::ModelInput<'a>,
        prng_key: Option<Array>,
        stream: &'a Stream,
    ) -> ModelGenerate<'a> {
        self.generate_input_with_cache_sampler(cache, temp, input, prng_key, stream, DefaultSampler)
    }

    /// Creates a token iterator from typed input with a caller-provided sampler.
    pub fn generate_input_with_cache_sampler<'a, S>(
        &'a mut self,
        cache: &'a mut ModelCache,
        temp: f32,
        input: input::ModelInput<'a>,
        prng_key: Option<Array>,
        stream: &'a Stream,
        sampler: S,
    ) -> ModelGenerate<'a, S>
    where
        S: Sampler,
    {
        match (self, cache) {
            (Self::Gemma4(model), ModelCache::Gemma4(cache)) => {
                ModelGenerate::Gemma4(gemma4::Generate::with_sampler(
                    model, cache, temp, input, prng_key, stream, sampler,
                ))
            }
            (Self::Gemma4Layerwise(model), ModelCache::Gemma4(cache)) => {
                ModelGenerate::Gemma4Layerwise(crate::gemma4::Generate::with_sampler(
                    model, cache, temp, input, prng_key, stream, sampler,
                ))
            }
            (Self::Lfm2(model), ModelCache::Lfm2(cache)) => ModelGenerate::Lfm2(
                lfm2::Generate::with_sampler(model, cache, temp, input, prng_key, stream, sampler),
            ),
            (Self::Lfm2Layerwise(model), ModelCache::Lfm2(cache)) => {
                ModelGenerate::Lfm2Layerwise(crate::lfm2::Generate::with_sampler(
                    model, cache, temp, input, prng_key, stream, sampler,
                ))
            }
            (Self::GptOss(model), ModelCache::GptOss(cache)) => {
                ModelGenerate::GptOss(gpt_oss::Generate::with_sampler(
                    model, cache, temp, input, prng_key, stream, sampler,
                ))
            }
            (Self::GptOssLayerwise(model), ModelCache::GptOss(cache)) => {
                ModelGenerate::GptOssLayerwise(crate::gpt_oss::Generate::with_sampler(
                    model, cache, temp, input, prng_key, stream, sampler,
                ))
            }
            (Self::Inkling(model), ModelCache::Inkling(cache)) => {
                ModelGenerate::Inkling(inkling::Generate::with_sampler(
                    model, cache, temp, input, prng_key, stream, sampler,
                ))
            }
            (Self::InklingLayerwise(model), ModelCache::Inkling(cache)) => {
                ModelGenerate::InklingLayerwise(crate::inkling::Generate::with_sampler(
                    model, cache, temp, input, prng_key, stream, sampler,
                ))
            }
            (Self::Llama(model), ModelCache::KeyValue(cache)) => ModelGenerate::Llama(
                llama::Generate::with_sampler(model, cache, temp, input, prng_key, stream, sampler),
            ),
            (Self::Llama(model), ModelCache::SlidingKeyValue(cache)) => {
                ModelGenerate::LlamaSliding(llama::Generate::with_sampler(
                    model, cache, temp, input, prng_key, stream, sampler,
                ))
            }
            (Self::Llama(model), ModelCache::PagedKeyValue(cache)) => ModelGenerate::LlamaPaged(
                llama::Generate::with_sampler(model, cache, temp, input, prng_key, stream, sampler),
            ),
            (Self::LlamaLayerwise(model), ModelCache::LlamaLayerwise(cache)) => {
                ModelGenerate::LlamaLayerwise(common::generation::Generate::with_sampler(
                    model, cache, temp, input, prng_key, stream, sampler,
                ))
            }
            (Self::Qwen3(model), ModelCache::KeyValue(cache)) => ModelGenerate::Qwen3(
                qwen3::Generate::with_sampler(model, cache, temp, input, prng_key, stream, sampler),
            ),
            (Self::Qwen3Layerwise(model), ModelCache::KeyValue(cache)) => {
                ModelGenerate::Qwen3Layerwise(common::generation::Generate::with_sampler(
                    model, cache, temp, input, prng_key, stream, sampler,
                ))
            }
            (Self::Qwen3Vl(model), ModelCache::Qwen3Vl(cache)) => {
                ModelGenerate::Qwen3Vl(qwen3_vl::Generate::with_sampler(
                    model, cache, temp, input, prng_key, stream, sampler,
                ))
            }
            (Self::Qwen3VlLayerwise(model), ModelCache::Qwen3Vl(cache)) => {
                ModelGenerate::Qwen3VlLayerwise(crate::qwen3_vl::Generate::with_sampler(
                    model, cache, temp, input, prng_key, stream, sampler,
                ))
            }
            (Self::Qwen3VlMoe(model), ModelCache::Qwen3VlMoe(cache)) => {
                ModelGenerate::Qwen3VlMoe(qwen3_vl_moe::Generate::with_sampler(
                    model, cache, temp, input, prng_key, stream, sampler,
                ))
            }
            (Self::Qwen3VlMoeLayerwise(model), ModelCache::Qwen3VlMoe(cache)) => {
                ModelGenerate::Qwen3VlMoeLayerwise(crate::qwen3_vl::Generate::with_sampler(
                    model, cache, temp, input, prng_key, stream, sampler,
                ))
            }
            (Self::NemotronH(model), ModelCache::NemotronH(cache)) => {
                ModelGenerate::NemotronH(nemotron_h::Generate::with_sampler(
                    model, cache, temp, input, prng_key, stream, sampler,
                ))
            }
            (Self::NemotronHLayerwise(model), ModelCache::NemotronH(cache)) => {
                ModelGenerate::NemotronHLayerwise(crate::nemotron_h::Generate::with_sampler(
                    model, cache, temp, input, prng_key, stream, sampler,
                ))
            }
            (Self::Qwen35Moe(model), ModelCache::Qwen35Moe(cache)) => {
                ModelGenerate::Qwen35Moe(qwen3_5_moe::Generate::with_sampler(
                    model, cache, temp, input, prng_key, stream, sampler,
                ))
            }
            (Self::Qwen35MoeLayerwise(model), ModelCache::Qwen35Moe(cache)) => {
                ModelGenerate::Qwen35MoeLayerwise(crate::qwen_hybrid::Generate::with_sampler(
                    model, cache, temp, input, prng_key, stream, sampler,
                ))
            }
            (Self::Qwen3Next(model), ModelCache::Qwen3Next(cache)) => {
                ModelGenerate::Qwen3Next(qwen3_next::Generate::with_sampler(
                    model, cache, temp, input, prng_key, stream, sampler,
                ))
            }
            (Self::Qwen3NextLayerwise(model), ModelCache::Qwen3Next(cache)) => {
                ModelGenerate::Qwen3NextLayerwise(crate::qwen_hybrid::Generate::with_sampler(
                    model, cache, temp, input, prng_key, stream, sampler,
                ))
            }
            (Self::DeepSeekV3(model), ModelCache::DeepSeekV3(cache)) => {
                ModelGenerate::DeepSeekV3(deepseek_v3::Generate::with_sampler(
                    model, cache, temp, input, prng_key, stream, sampler,
                ))
            }
            (Self::DeepSeekV3Layerwise(model), ModelCache::DeepSeekV3(cache)) => {
                ModelGenerate::DeepSeekV3Layerwise(crate::deepseek_v3::Generate::with_sampler(
                    model, cache, temp, input, prng_key, stream, sampler,
                ))
            }
            _ => panic!("model cache type does not match model kind"),
        }
    }
}

/// Cache value matching a [`Model`] variant.
#[derive(Clone)]
pub enum ModelCache {
    /// Compressed latent MLA cache for DeepSeek-V3/R1.
    DeepSeekV3(deepseek_v3::Cache),
    /// Gemma 4 generation cache.
    Gemma4(gemma4::Cache),
    /// Alternating full/sliding GPT-OSS cache.
    GptOss(gpt_oss::Cache),
    /// Alternating global/local attention and short-convolution Inkling cache.
    Inkling(inkling::Cache),
    /// Homogeneous per-layer key/value cache.
    KeyValue(Vec<Option<ConcatKeyValueCache>>),
    /// Unified Llama cache used by bounded layer execution.
    LlamaLayerwise(crate::llama::LlamaCache),
    /// Qwen3-VL key/value cache and multimodal position state.
    Qwen3Vl(qwen3_vl::Cache),
    /// Qwen3-VL-MoE key/value cache and multimodal position state.
    Qwen3VlMoe(qwen3_vl_moe::Cache),
    /// Homogeneous bounded cache for sliding-window attention.
    SlidingKeyValue(Vec<Option<SlidingKeyValueCache>>),
    /// Homogeneous block-addressable key/value cache under one global budget.
    PagedKeyValue(Vec<Option<PagedKeyValueCache>>),
    /// Heterogeneous Nemotron-H cache.
    NemotronH(nemotron_h::Cache),
    /// Heterogeneous LFM2 attention/convolution cache.
    Lfm2(lfm2::Cache),
    /// Heterogeneous Qwen3.5 MoE cache.
    Qwen35Moe(qwen3_5_moe::Cache),
    /// Heterogeneous Qwen3-Next cache.
    Qwen3Next(qwen3_next::Cache),
}

pub(super) fn validate_gemma4_drafter(
    target: &gemma4::ModelArgs,
    assistant: &gemma4_assistant::Gemma4AssistantDraftModel,
) -> Result<(), Exception> {
    if assistant.config.model_type != "gemma4_assistant" {
        return Err(Exception::custom(format!(
            "expected a gemma4_assistant checkpoint, got {:?}",
            assistant.config.model_type
        )));
    }
    if assistant.config.backbone_hidden_size != target.hidden_size {
        return Err(Exception::custom(format!(
            "Gemma 4 assistant backbone hidden size {} does not match target hidden size {}",
            assistant.config.backbone_hidden_size, target.hidden_size
        )));
    }
    if assistant.config.text_config.vocab_size != target.vocab_size {
        return Err(Exception::custom(format!(
            "Gemma 4 assistant vocabulary size {} does not match target vocabulary size {}",
            assistant.config.text_config.vocab_size, target.vocab_size
        )));
    }
    if assistant.block_size() <= 1 {
        return Err(Exception::custom(
            "Gemma 4 assistant block_size must permit at least one draft token",
        ));
    }
    Ok(())
}

impl ModelCache {
    /// Returns aggregate cache-residency telemetry when paging is active.
    pub fn residency_report(&self) -> Result<Option<CacheResidencyReport>, Exception> {
        match self {
            Self::PagedKeyValue(caches) => caches
                .iter()
                .flatten()
                .next()
                .map(PagedKeyValueCache::report)
                .transpose(),
            Self::LlamaLayerwise(cache) => cache
                .residency_report()
                .map_err(|error| Exception::custom(error.to_string())),
            Self::DeepSeekV3(cache) => cache.residency_report(),
            Self::GptOss(cache) => cache.residency_report(),
            Self::Inkling(cache) => cache.residency_report(),
            _ => Ok(None),
        }
    }

    /// Finalizes and atomically saves a completed immutable text prefix.
    pub fn save_prompt_cache(
        &mut self,
        destination: impl AsRef<Path>,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &PromptCacheOptions,
    ) -> Result<PromptCacheManifest, Exception> {
        match self {
            Self::PagedKeyValue(caches) => {
                for cache in caches.iter_mut().flatten() {
                    cache.finalize()?;
                }
                caches
                    .iter()
                    .flatten()
                    .next()
                    .ok_or_else(|| Exception::custom("cannot persist an empty paged cache"))?
                    .manager()
                    .save_prompt_cache(destination, descriptor, prefix_token_ids, options)
                    .map_err(|error| Exception::custom(error.to_string()))
            }
            Self::LlamaLayerwise(cache) => cache
                .save_prompt_cache(destination, descriptor, prefix_token_ids, options)
                .map_err(|error| Exception::custom(error.to_string())),
            Self::DeepSeekV3(cache) => {
                cache.save_prompt_cache(destination, descriptor, prefix_token_ids, options)
            }
            Self::GptOss(cache) => {
                cache.save_prompt_cache(destination, descriptor, prefix_token_ids, options)
            }
            _ => Err(Exception::custom(
                "prompt-cache persistence is unsupported for this model cache representation",
            )),
        }
    }
}

/// Token iterator for any supported model variant.
pub enum ModelGenerate<'a, S = DefaultSampler>
where
    S: Sampler,
{
    /// DeepSeek-V3/R1 generation iterator.
    DeepSeekV3(deepseek_v3::Generate<'a, S>),
    /// DeepSeek-V3/R1 generation using bounded layer execution.
    DeepSeekV3Layerwise(crate::deepseek_v3::Generate<'a, S>),
    /// Gemma 4 generation iterator.
    Gemma4(gemma4::Generate<'a, S>),
    /// Gemma 4 multimodal-prefill generation using bounded layer execution.
    Gemma4Layerwise(crate::gemma4::Generate<'a, S>),
    /// GPT-OSS generation iterator.
    GptOss(gpt_oss::Generate<'a, S>),
    /// GPT-OSS generation using bounded layer execution.
    GptOssLayerwise(crate::gpt_oss::Generate<'a, S>),
    /// Inkling generation iterator.
    Inkling(inkling::Generate<'a, S>),
    /// Inkling multimodal-prefill generation using bounded layer execution.
    InklingLayerwise(crate::inkling::Generate<'a, S>),
    /// Llama generation iterator.
    Llama(llama::Generate<'a, ConcatKeyValueCache, S>),
    /// Llama-compatible generation with bounded sliding-window caches.
    LlamaSliding(llama::Generate<'a, SlidingKeyValueCache, S>),
    /// Llama-compatible generation with block-addressable cache residency.
    LlamaPaged(llama::Generate<'a, PagedKeyValueCache, S>),
    /// Llama-compatible generation using bounded layer execution.
    LlamaLayerwise(
        common::generation::Generate<'a, crate::llama::LlamaModel, crate::llama::LlamaCache, S>,
    ),
    /// Qwen3 generation iterator.
    Qwen3(qwen3::Generate<'a, ConcatKeyValueCache, S>),
    /// Qwen3 generation using bounded layer execution.
    Qwen3Layerwise(
        common::generation::Generate<
            'a,
            crate::qwen3::Qwen3LayerwiseModel,
            Vec<Option<ConcatKeyValueCache>>,
            S,
        >,
    ),
    /// Qwen3-VL generation iterator.
    Qwen3Vl(qwen3_vl::Generate<'a, S>),
    /// Qwen3-VL generation using vision/text bounded layer execution.
    Qwen3VlLayerwise(crate::qwen3_vl::Generate<'a, S>),
    /// Qwen3-VL-MoE generation iterator.
    Qwen3VlMoe(qwen3_vl_moe::Generate<'a, S>),
    /// Qwen3-VL-MoE generation using vision/text bounded layer execution.
    Qwen3VlMoeLayerwise(crate::qwen3_vl::Generate<'a, S>),
    /// Nemotron-H generation iterator.
    NemotronH(nemotron_h::Generate<'a, S>),
    /// Nemotron-H generation using bounded layer execution.
    NemotronHLayerwise(crate::nemotron_h::Generate<'a, S>),
    /// LFM2 generation iterator.
    Lfm2(lfm2::Generate<'a, S>),
    /// LFM2 generation using bounded layer execution.
    Lfm2Layerwise(crate::lfm2::Generate<'a, S>),
    /// Qwen3.5 MoE generation iterator.
    Qwen35Moe(qwen3_5_moe::Generate<'a, S>),
    /// Qwen3.5 multimodal-prefill generation using shared bounded layer execution.
    Qwen35MoeLayerwise(crate::qwen_hybrid::Generate<'a, S>),
    /// Qwen3-Next generation iterator.
    Qwen3Next(qwen3_next::Generate<'a, S>),
    /// Qwen3-Next generation using shared hybrid bounded layer execution.
    Qwen3NextLayerwise(crate::qwen_hybrid::Generate<'a, S>),
}

impl<S> ModelGenerate<'_, S>
where
    S: Sampler,
{
    /// Returns the architecture iterator's sampler at its committed prefix.
    pub fn sampler_mut(&mut self) -> &mut S {
        match self {
            Self::DeepSeekV3(generate) => generate.sampler_mut(),
            Self::DeepSeekV3Layerwise(generate) => generate.sampler_mut(),
            Self::Gemma4(generate) => generate.sampler_mut(),
            Self::Gemma4Layerwise(generate) => generate.sampler_mut(),
            Self::GptOss(generate) => generate.sampler_mut(),
            Self::GptOssLayerwise(generate) => generate.sampler_mut(),
            Self::Inkling(generate) => generate.sampler_mut(),
            Self::InklingLayerwise(generate) => generate.sampler_mut(),
            Self::Llama(generate) => generate.sampler_mut(),
            Self::LlamaSliding(generate) => generate.sampler_mut(),
            Self::LlamaPaged(generate) => generate.sampler_mut(),
            Self::LlamaLayerwise(generate) => generate.sampler_mut(),
            Self::Lfm2(generate) => generate.sampler_mut(),
            Self::Lfm2Layerwise(generate) => generate.sampler_mut(),
            Self::NemotronH(generate) => generate.sampler_mut(),
            Self::NemotronHLayerwise(generate) => generate.sampler_mut(),
            Self::Qwen3(generate) => generate.sampler_mut(),
            Self::Qwen3Layerwise(generate) => generate.sampler_mut(),
            Self::Qwen3Vl(generate) => generate.sampler_mut(),
            Self::Qwen3VlLayerwise(generate) => generate.sampler_mut(),
            Self::Qwen3VlMoe(generate) => generate.sampler_mut(),
            Self::Qwen3VlMoeLayerwise(generate) => generate.sampler_mut(),
            Self::Qwen35Moe(generate) => generate.sampler_mut(),
            Self::Qwen35MoeLayerwise(generate) => generate.sampler_mut(),
            Self::Qwen3Next(generate) => generate.sampler_mut(),
            Self::Qwen3NextLayerwise(generate) => generate.sampler_mut(),
        }
    }
}

impl<S> Iterator for ModelGenerate<'_, S>
where
    S: Sampler,
{
    type Item = Result<Array, Exception>;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::DeepSeekV3(generate) => generate.next(),
            Self::DeepSeekV3Layerwise(generate) => generate.next(),
            Self::Gemma4(generate) => generate.next(),
            Self::Gemma4Layerwise(generate) => generate.next(),
            Self::GptOss(generate) => generate.next(),
            Self::GptOssLayerwise(generate) => generate.next(),
            Self::Inkling(generate) => generate.next(),
            Self::InklingLayerwise(generate) => generate.next(),
            Self::Llama(generate) => generate.next(),
            Self::LlamaSliding(generate) => generate.next(),
            Self::LlamaPaged(generate) => generate.next(),
            Self::LlamaLayerwise(generate) => generate.next(),
            Self::Lfm2(generate) => generate.next(),
            Self::Lfm2Layerwise(generate) => generate.next(),
            Self::NemotronH(generate) => generate.next(),
            Self::NemotronHLayerwise(generate) => generate.next(),
            Self::Qwen3(generate) => generate.next(),
            Self::Qwen3Layerwise(generate) => generate.next(),
            Self::Qwen3Vl(generate) => generate.next(),
            Self::Qwen3VlLayerwise(generate) => generate.next(),
            Self::Qwen3VlMoe(generate) => generate.next(),
            Self::Qwen3VlMoeLayerwise(generate) => generate.next(),
            Self::Qwen35Moe(generate) => generate.next(),
            Self::Qwen35MoeLayerwise(generate) => generate.next(),
            Self::Qwen3Next(generate) => generate.next(),
            Self::Qwen3NextLayerwise(generate) => generate.next(),
        }
    }
}
