//! MLX model capability derivation, resource observation, and admission adapter.

use std::num::NonZeroU8;

use eredu_architectures::media_plan::{
    self, MediaMetadata, MediaModality, MediaShapePlan, PreparedMediaInput,
};
use eredu_core::{
    estimate_runtime_state, AvailableMemory, CapabilityError, InputTokenCount, ModelCapabilities,
    ModelCapabilityBackend, ModelRuntime, ObservationKind, Observed, PhysicalMemorySemantics,
    RuntimeStateEstimate, StateLayout, StaticMemoryReport,
};
use safemlx::{Array, Stream};

use super::{MlxBackend, MlxModelInput, MlxModelSession, Model};
use crate::backend::runtime::media::input::{self, InputPayload, Modality};
use eredu_core::residency::MemoryTier;

fn positive(value: i32, field: &'static str) -> Result<u64, CapabilityError> {
    u64::try_from(value).map_err(|_| CapabilityError::InvalidConfiguration {
        field,
        detail: format!("expected a non-negative value, got {value}"),
    })
}

fn checked_add(left: u64, right: u64, operation: &'static str) -> Result<u64, CapabilityError> {
    left.checked_add(right)
        .ok_or(CapabilityError::ArithmeticOverflow { operation })
}

fn checked_mul(left: u64, right: u64, operation: &'static str) -> Result<u64, CapabilityError> {
    left.checked_mul(right)
        .ok_or(CapabilityError::ArithmeticOverflow { operation })
}

fn estimate_mlx_runtime_state_with_dtype(
    layout: &StateLayout,
    input: InputTokenCount,
    max_output_tokens: u64,
    batch_size: u64,
    state_dtype_bytes: NonZeroU8,
) -> Result<RuntimeStateEstimate, CapabilityError> {
    estimate_runtime_state(
        layout,
        input,
        max_output_tokens,
        batch_size,
        state_dtype_bytes,
    )
}

#[cfg(test)]
fn estimate_mlx_runtime_state(
    layout: &StateLayout,
    input: InputTokenCount,
    max_output_tokens: u64,
    batch_size: u64,
) -> Result<RuntimeStateEstimate, CapabilityError> {
    estimate_mlx_runtime_state_with_dtype(
        layout,
        input,
        max_output_tokens,
        batch_size,
        NonZeroU8::new(4).expect("test dtype width is nonzero"),
    )
}

impl Model {
    pub(super) fn architecture_capability_estimate(
        &self,
    ) -> Result<eredu_architectures::capability::CapabilityEstimate, CapabilityError> {
        use eredu_architectures::capability;

        match self {
            Self::DeepSeek(model) => {
                if let Some(args) = model.v3_args() {
                    capability::deepseek_v3(args)
                } else {
                    capability::deepseek_v4(model.v4_args().expect("DeepSeek family"))
                }
            }
            Self::Llama(model) => capability::llama(model.args()),
            Self::Qwen(model) => capability::qwen(model.args()),
            Self::MuseGlimmer(model) => capability::muse_glimmer(model.args()),
            Self::Qwen3Vl(model) | Self::Qwen3VlMoe(model) => capability::qwen_vl(model.args()),
            Self::GptOss(model) => capability::gpt_oss(model.args()),
            Self::Gemma4(model) => capability::gemma4(model.args()),
            Self::Inkling(model) => capability::inkling(model.args()),
            Self::KimiLinear(model) => capability::kimi_linear(model.args()),
            Self::Lfm2(model) => capability::lfm2(model.args()),
            Self::NemotronH(model) => capability::nemotron_h(model.args()),
            Self::Qwen3Next(model) | Self::Qwen35(model) => {
                capability::qwen_hybrid(model.parsed_args())
            }
        }
    }
}
fn unavailable_counter(error: safemlx::error::Exception) -> Observed<u64> {
    Observed::Unavailable {
        reason: error.to_string(),
    }
}

fn runtime_counter(
    function: fn() -> Result<usize, safemlx::error::Exception>,
    source: &'static str,
) -> Observed<u64> {
    match function() {
        Ok(value) => match u64::try_from(value) {
            Ok(value) => Observed::Available {
                value,
                kind: ObservationKind::Observational,
                source: source.into(),
            },
            Err(_) => Observed::Unavailable {
                reason: "counter does not fit u64".into(),
            },
        },
        Err(error) => unavailable_counter(error),
    }
}

fn array_shape(array: &Array) -> Result<Vec<u64>, CapabilityError> {
    array
        .shape()
        .iter()
        .map(|dimension| {
            u64::try_from(*dimension).map_err(|_| CapabilityError::ArithmeticOverflow {
                operation: "prepared media array dimension",
            })
        })
        .collect()
}

fn i32_metadata(array: Option<&Array>) -> Result<Option<MediaMetadata<i32>>, CapabilityError> {
    array
        .map(|array| {
            let evaluated = array
                .evaluated()
                .map_err(|error| CapabilityError::Observation(error.to_string()))?;
            let values = evaluated
                .try_as_slice::<i32>()
                .map_err(|error| CapabilityError::Observation(error.to_string()))?;
            Ok(MediaMetadata {
                shape: array_shape(array)?,
                values: values.to_vec(),
            })
        })
        .transpose()
}

fn bool_metadata(array: Option<&Array>) -> Result<Option<MediaMetadata<bool>>, CapabilityError> {
    array
        .map(|array| {
            let evaluated = array
                .evaluated()
                .map_err(|error| CapabilityError::Observation(error.to_string()))?;
            let values = evaluated
                .try_as_slice::<bool>()
                .map_err(|error| CapabilityError::Observation(error.to_string()))?;
            Ok(MediaMetadata {
                shape: array_shape(array)?,
                values: values.to_vec(),
            })
        })
        .transpose()
}

fn prepared_media_input(
    modality: Modality,
    payload: &Array,
    metadata: input::InputMetadata<'_>,
) -> Result<PreparedMediaInput, CapabilityError> {
    let modality = match modality {
        Modality::Image => MediaModality::Image,
        Modality::Audio => MediaModality::Audio,
        Modality::Video => MediaModality::Video,
        Modality::Text => unreachable!("text handled separately"),
    };
    Ok(PreparedMediaInput {
        modality,
        payload_shape: array_shape(payload)?,
        patch_grid: i32_metadata(metadata.patch_grid)?,
        patch_positions: i32_metadata(metadata.patch_positions)?,
        audio_mask: bool_metadata(metadata.audio_mask)?,
    })
}

fn array_bytes(array: &Array, operation: &'static str) -> Result<u64, CapabilityError> {
    u64::try_from(array.nbytes()).map_err(|_| CapabilityError::ArithmeticOverflow { operation })
}

fn four_byte_scalars(scalars: u64, operation: &'static str) -> Result<u64, CapabilityError> {
    checked_mul(scalars, 4, operation)
}

impl Model {
    pub(super) fn prepared_media_plan(
        &self,
        input: &PreparedMediaInput,
    ) -> Result<MediaShapePlan, CapabilityError> {
        match self {
            Self::Qwen3Vl(model) | Self::Qwen3VlMoe(model) => {
                media_plan::qwen_vision(&model.args().vision, input, self.effective_model_type())
            }
            Self::Qwen3Next(model) | Self::Qwen35(model) => media_plan::qwen_hybrid_vision(
                model.vision_config(),
                input,
                self.effective_model_type(),
            ),
            Self::Gemma4(model) => media_plan::gemma4(model.args(), input),
            Self::Inkling(model) => media_plan::inkling(model.args(), input),
            Self::MuseGlimmer(model) => media_plan::muse_glimmer(model.args(), input),
            Self::DeepSeek(_)
            | Self::GptOss(_)
            | Self::KimiLinear(_)
            | Self::Llama(_)
            | Self::Lfm2(_)
            | Self::NemotronH(_)
            | Self::Qwen(_) => media_plan::text_only(self.effective_model_type(), input),
        }
    }
}
fn prepared_media_accounting(
    session: &MlxModelSession<'_>,
    modality: Modality,
    payload: &Array,
    metadata: input::InputMetadata<'_>,
) -> Result<(u64, u64), CapabilityError> {
    let input = prepared_media_input(modality, payload, metadata)?;
    let plan = session.prepared_media_plan(&input)?;
    let mut input_bytes = array_bytes(payload, "prepared media payload bytes")?;
    for array in [
        metadata.patch_grid,
        metadata.patch_positions,
        metadata.audio_mask,
    ]
    .into_iter()
    .flatten()
    {
        input_bytes = checked_add(
            input_bytes,
            array_bytes(array, "prepared media metadata bytes")?,
            "prepared media input bytes",
        )?;
    }
    Ok((
        plan.decoder_positions,
        checked_add(
            input_bytes,
            four_byte_scalars(
                plan.execution_workspace_scalars,
                "prepared media execution workspace bytes",
            )?,
            "prepared media total workspace bytes",
        )?,
    ))
}

pub fn model_capabilities(
    session: &MlxModelSession<'_>,
) -> Result<ModelCapabilities, CapabilityError> {
    session
        .capability_estimate()
        .map(|estimate| estimate.into_parts().0)
}

pub fn count_prepared_input(
    session: &MlxModelSession<'_>,
    prepared: input::ModelInput<'_>,
    _stream: &Stream,
) -> Result<InputTokenCount, CapabilityError> {
    let mut text_tokens = 0u64;
    let mut media_positions = 0u64;
    let mut media_execution_workspace_bytes = 0u64;
    let mut media_execution_workspace_kind = ObservationKind::Exact;
    for part in prepared.parts {
        match (part.modality, part.payload) {
            (Modality::Text, InputPayload::TokenIds(tokens)) => {
                if tokens.ndim() != 2 || tokens.dim(0) != 1 {
                    return Err(CapabilityError::UnsupportedInput {
                        architecture: session.effective_model_type().into(),
                        reason: format!(
                            "prepared text token IDs must be [1, sequence], got {:?}",
                            tokens.shape()
                        ),
                    });
                }
                text_tokens = checked_add(
                    text_tokens,
                    positive(tokens.dim(1), "prepared text sequence")?,
                    "prepared text-token total",
                )?;
            }
            (Modality::Text, _) => {
                return Err(CapabilityError::UnsupportedInput {
                    architecture: session.effective_model_type().into(),
                    reason: "prepared text is not represented by tokenizer IDs".into(),
                });
            }
            (_modality, InputPayload::Embeddings(embeddings)) => {
                if embeddings.ndim() != 3 || embeddings.dim(0) != 1 {
                    return Err(CapabilityError::UnsupportedInput {
                        architecture: session.effective_model_type().into(),
                        reason: format!(
                            "prepared media embeddings must be [1, sequence, hidden], got {:?}",
                            embeddings.shape()
                        ),
                    });
                }
                media_positions = checked_add(
                    media_positions,
                    positive(embeddings.dim(1), "prepared embedding sequence")?,
                    "prepared media-position total",
                )?;
            }
            (modality, InputPayload::Tensor(tensor)) => {
                let (positions, workspace_bytes) =
                    prepared_media_accounting(session, modality, tensor, part.metadata)?;
                media_positions =
                    checked_add(media_positions, positions, "prepared media-position total")?;
                media_execution_workspace_bytes = checked_add(
                    media_execution_workspace_bytes,
                    workspace_bytes,
                    "prepared media-workspace total",
                )?;
                media_execution_workspace_kind = ObservationKind::Conservative;
            }
            (_, InputPayload::TokenIds(_)) => {
                return Err(CapabilityError::UnsupportedInput {
                    architecture: session.effective_model_type().into(),
                    reason: "non-text prepared input cannot contain tokenizer IDs".into(),
                });
            }
        }
    }
    Ok(InputTokenCount::prepared(
        text_tokens,
        media_positions,
        checked_add(
            text_tokens,
            media_positions,
            "prepared model-position total",
        )?,
        media_execution_workspace_bytes,
        media_execution_workspace_kind,
    ))
}

pub fn model_runtime_state(
    session: &MlxModelSession<'_>,
    input: InputTokenCount,
    max_output_tokens: u64,
    batch_size: u64,
    state_dtype_bytes: NonZeroU8,
) -> Result<RuntimeStateEstimate, CapabilityError> {
    let estimate = session.capability_estimate()?;
    estimate_mlx_runtime_state_with_dtype(
        estimate.state_layout(),
        input,
        max_output_tokens,
        batch_size,
        state_dtype_bytes,
    )
}

pub fn static_model_memory(
    session: &MlxModelSession<'_>,
) -> Result<StaticMemoryReport, CapabilityError> {
    let residency = session
        .residency_report()
        .map_err(|error| CapabilityError::Observation(error.to_string()))?;
    let (logical, host, device, disk, mappings) = if let Some(report) = residency {
        let planned = report.offload().planned_bytes();
        let resident = report.offload().resident_bytes();
        let logical = checked_add(
            checked_add(
                planned.get(MemoryTier::Host),
                planned.get(MemoryTier::Device),
                "planned host plus device parameters",
            )?,
            planned.get(MemoryTier::Disk),
            "complete planned parameter bytes",
        )?;
        (
            Observed::Available {
                value: logical,
                kind: ObservationKind::Exact,
                source: "validated bounded-residency plan".into(),
            },
            Observed::Available {
                value: resident.get(MemoryTier::Host),
                kind: ObservationKind::Exact,
                source: "bounded-residency manager".into(),
            },
            Observed::Available {
                value: resident.get(MemoryTier::Device),
                kind: ObservationKind::Exact,
                source: "bounded-residency manager".into(),
            },
            Observed::Available {
                value: planned.get(MemoryTier::Disk),
                kind: ObservationKind::Exact,
                source: "bounded-residency plan".into(),
            },
            Observed::Available {
                value: report.weight_store().currently_mapped_shards as u64,
                kind: ObservationKind::Observational,
                source: "checkpoint-store mapping cache".into(),
            },
        )
    } else {
        (
            Observed::Unavailable {
                reason: "loaded model exposes neither resident parameters nor a residency plan"
                    .into(),
            },
            Observed::Unavailable {
                reason: "host residency unavailable".into(),
            },
            Observed::Unavailable {
                reason: "device residency unavailable".into(),
            },
            Observed::Unavailable {
                reason: "disk residency unavailable".into(),
            },
            Observed::Unavailable {
                reason: "mapping information unavailable".into(),
            },
        )
    };
    Ok(StaticMemoryReport {
        logical_parameter_bytes: logical,
        current_host_resident_bytes: host,
        current_device_resident_bytes: device,
        planned_disk_backed_bytes: disk,
        backend_active_allocation_bytes: runtime_counter(
            safemlx::memory::active_memory,
            "process-global MLX active allocation counter",
        ),
        backend_allocator_cache_bytes: runtime_counter(
            safemlx::memory::cache_memory,
            "process-global MLX allocator cache counter",
        ),
        physical_semantics: if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
            PhysicalMemorySemantics::Unified
        } else {
            PhysicalMemorySemantics::Unknown
        },
        currently_mapped_shards: mappings,
    })
}

impl<'a> ModelCapabilityBackend for MlxBackend<'a> {
    fn model_capabilities(
        runtime: &ModelRuntime<Self>,
    ) -> Result<ModelCapabilities, CapabilityError> {
        model_capabilities(runtime.session())
    }

    fn count_prepared_input(
        runtime: &ModelRuntime<Self>,
        prepared: &MlxModelInput,
    ) -> Result<InputTokenCount, CapabilityError> {
        prepared.with_borrowed(|input| {
            count_prepared_input(runtime.session(), input, runtime.backend().stream())
        })
    }

    fn estimate_runtime_state(
        runtime: &ModelRuntime<Self>,
        input: InputTokenCount,
        max_output_tokens: u64,
        batch_size: u64,
    ) -> Result<RuntimeStateEstimate, CapabilityError> {
        model_runtime_state(
            runtime.session(),
            input,
            max_output_tokens,
            batch_size,
            runtime.session().runtime_state_dtype_bytes(),
        )
    }

    fn static_memory(runtime: &ModelRuntime<Self>) -> Result<StaticMemoryReport, CapabilityError> {
        static_model_memory(runtime.session())
    }
}

#[cfg(target_os = "macos")]
fn macos_memory() -> Result<AvailableMemory, CapabilityError> {
    unsafe extern "C" {
        fn os_proc_available_memory() -> usize;
    }

    let name = c"hw.memsize";
    let mut total = 0u64;
    let mut size = std::mem::size_of::<u64>();
    let status = unsafe {
        libc::sysctlbyname(
            name.as_ptr(),
            (&mut total as *mut u64).cast(),
            &mut size,
            std::ptr::null_mut(),
            0,
        )
    };
    let physical_memory_bytes = if status == 0 && size == std::mem::size_of::<u64>() {
        Observed::Available {
            value: total,
            kind: ObservationKind::Exact,
            source: "macOS sysctl hw.memsize".into(),
        }
    } else {
        Observed::Unavailable {
            reason: std::io::Error::last_os_error().to_string(),
        }
    };
    let available = unsafe { os_proc_available_memory() };
    let available_memory_bytes = match u64::try_from(available) {
        Ok(value) if value > 0 => Observed::Available {
            value,
            kind: ObservationKind::Estimated,
            source: "macOS os_proc_available_memory".into(),
        },
        _ => Observed::Unavailable {
            reason: "os_proc_available_memory returned no usable value".into(),
        },
    };
    Ok(AvailableMemory {
        physical_memory_bytes,
        available_memory_bytes,
        physical_semantics: if cfg!(target_arch = "aarch64") {
            PhysicalMemorySemantics::Unified
        } else {
            PhysicalMemorySemantics::Unknown
        },
    })
}

#[cfg(target_os = "linux")]
fn linux_memory() -> Result<AvailableMemory, CapabilityError> {
    let contents = std::fs::read_to_string("/proc/meminfo")
        .map_err(|error| CapabilityError::Observation(error.to_string()))?;
    let value = |name: &str| -> Option<u64> {
        contents.lines().find_map(|line| {
            let (key, rest) = line.split_once(':')?;
            if key != name {
                return None;
            }
            let kib = rest.split_whitespace().next()?.parse::<u64>().ok()?;
            kib.checked_mul(1024)
        })
    };
    Ok(AvailableMemory {
        physical_memory_bytes: value("MemTotal").map_or_else(
            || Observed::Unavailable {
                reason: "/proc/meminfo has no MemTotal".into(),
            },
            |value| Observed::Available {
                value,
                kind: ObservationKind::Exact,
                source: "Linux /proc/meminfo MemTotal".into(),
            },
        ),
        available_memory_bytes: value("MemAvailable").map_or_else(
            || Observed::Unavailable {
                reason: "/proc/meminfo has no MemAvailable".into(),
            },
            |value| Observed::Available {
                value,
                kind: ObservationKind::Estimated,
                source: "Linux /proc/meminfo MemAvailable".into(),
            },
        ),
        physical_semantics: PhysicalMemorySemantics::Unknown,
    })
}

#[cfg(target_os = "windows")]
fn windows_memory() -> Result<AvailableMemory, CapabilityError> {
    #[repr(C)]
    struct MemoryStatusEx {
        length: u32,
        memory_load: u32,
        total_physical: u64,
        available_physical: u64,
        total_page_file: u64,
        available_page_file: u64,
        total_virtual: u64,
        available_virtual: u64,
        available_extended_virtual: u64,
    }

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GlobalMemoryStatusEx(status: *mut MemoryStatusEx) -> i32;
    }

    let mut status = MemoryStatusEx {
        length: std::mem::size_of::<MemoryStatusEx>() as u32,
        memory_load: 0,
        total_physical: 0,
        available_physical: 0,
        total_page_file: 0,
        available_page_file: 0,
        total_virtual: 0,
        available_virtual: 0,
        available_extended_virtual: 0,
    };
    if unsafe { GlobalMemoryStatusEx(&mut status) } == 0 {
        return Ok(AvailableMemory {
            physical_memory_bytes: Observed::Unavailable {
                reason: "GlobalMemoryStatusEx failed".into(),
            },
            available_memory_bytes: Observed::Unavailable {
                reason: "GlobalMemoryStatusEx failed".into(),
            },
            physical_semantics: PhysicalMemorySemantics::Unknown,
        });
    }
    Ok(AvailableMemory {
        physical_memory_bytes: Observed::Available {
            value: status.total_physical,
            kind: ObservationKind::Exact,
            source: "Windows GlobalMemoryStatusEx ullTotalPhys".into(),
        },
        available_memory_bytes: Observed::Available {
            value: status.available_physical,
            kind: ObservationKind::Estimated,
            source: "Windows GlobalMemoryStatusEx ullAvailPhys".into(),
        },
        physical_semantics: PhysicalMemorySemantics::Unknown,
    })
}

/// Queries system memory that can be used as an admission signal.
///
/// Apple Silicon reports one unified physical capacity; logical host/device
/// residency tiers must not be added as independent physical capacities.
pub fn available_memory() -> Result<AvailableMemory, CapabilityError> {
    #[cfg(target_os = "macos")]
    {
        macos_memory()
    }
    #[cfg(target_os = "linux")]
    {
        linux_memory()
    }
    #[cfg(target_os = "windows")]
    {
        windows_memory()
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    {
        Ok(AvailableMemory {
            physical_memory_bytes: Observed::Unavailable {
                reason: "portable physical-memory query is not implemented on this platform".into(),
            },
            available_memory_bytes: Observed::Unavailable {
                reason: "portable available-memory query is not implemented on this platform"
                    .into(),
            },
            physical_semantics: PhysicalMemorySemantics::Unknown,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eredu_architectures::{
        capability as architecture_capability, gemma4, gpt_oss, kimi_linear, lfm2,
        llama::ModelArgs as LlamaModelArgs, nemotron_h,
    };
    use eredu_core::{
        attention::AttentionPolicy, CacheStateStrategy, EstimationCompleteness, GrowingState,
        InputModalities, SlidingWindowLayerCount,
    };
    use serde_json::json;

    #[test]
    fn qwen2_runtime_state_splits_full_and_sliding_gqa_layers() {
        let args = eredu_architectures::qwen::model_args_from_config_value(&json!({
                "model_type": "qwen2", "hidden_size": 16, "num_hidden_layers": 6,
                "intermediate_size": 32, "num_attention_heads": 4,
                "num_key_value_heads": 2, "rms_norm_eps": 1e-6, "vocab_size": 64,
                "max_position_embeddings": 128, "rope_theta": 10000.0,
                "tie_word_embeddings": false, "use_sliding_window": true,
                "sliding_window": 8, "max_window_layers": 4
        }))
        .unwrap();
        let (capabilities, estimate) = architecture_capability::qwen(&args).unwrap().into_parts();
        let strategy = capabilities.state_strategy;
        assert_eq!(
            strategy,
            CacheStateStrategy::MixedKv {
                full_layers: 4,
                sliding: vec![SlidingWindowLayerCount {
                    layers: 2,
                    window: 8,
                }],
            }
        );
        assert_eq!(estimate.growing.len(), 2);
        assert_eq!(estimate.growing[0].layers, 4);
        assert_eq!(estimate.growing[0].window, None);
        assert_eq!(estimate.growing[1].layers, 2);
        assert_eq!(estimate.growing[1].window, Some(8));
        // 2 KV heads x 4 values per head x key/value.
        assert_eq!(estimate.growing[1].scalars_per_position, 16);
    }

    #[test]
    fn qwen2_runtime_state_groups_arbitrary_distinct_windows_exactly() {
        let mut args = eredu_architectures::qwen::model_args_from_config_value(&json!({
                "model_type": "qwen2", "hidden_size": 16, "num_hidden_layers": 4,
                "intermediate_size": 32, "num_attention_heads": 4,
                "num_key_value_heads": 2, "rms_norm_eps": 1e-6, "vocab_size": 64,
                "max_position_embeddings": 128, "rope_theta": 10000.0,
                "tie_word_embeddings": false
        }))
        .unwrap();
        args.attention_schedule = eredu_core::attention::LayerSchedule::new(
            4,
            vec![
                eredu_core::attention::AttentionPolicy::sliding(4).unwrap(),
                eredu_core::attention::AttentionPolicy::Full,
                eredu_core::attention::AttentionPolicy::sliding(8).unwrap(),
                eredu_core::attention::AttentionPolicy::sliding(4).unwrap(),
            ],
        )
        .unwrap();
        let (capabilities, layout) = architecture_capability::qwen(&args).unwrap().into_parts();
        let strategy = capabilities.state_strategy;
        assert_eq!(
            strategy,
            CacheStateStrategy::MixedKv {
                full_layers: 1,
                sliding: vec![
                    SlidingWindowLayerCount {
                        layers: 2,
                        window: 4,
                    },
                    SlidingWindowLayerCount {
                        layers: 1,
                        window: 8,
                    },
                ],
            }
        );
        let state = estimate_mlx_runtime_state(&layout, InputTokenCount::text(10), 0, 2).unwrap();
        assert_eq!(state.assumptions.sliding_window_bounds, vec![4, 8]);
        assert_eq!(state.context_state_bytes, (10 + 2 * 4 + 8) * 16 * 2 * 4);
    }

    #[test]
    fn lfm2_runtime_state_uses_the_normalized_hybrid_schedule() {
        let args = lfm2::model_args_from_config_value(&json!({
            "model_type": "lfm2", "vocab_size": 32, "hidden_size": 16,
            "intermediate_size": 24, "num_hidden_layers": 3,
            "num_attention_heads": 4, "num_key_value_heads": 2,
            "max_position_embeddings": 128, "norm_eps": 1e-5,
            "conv_L_cache": 3, "block_auto_adjust_ff_dim": false,
            "layer_types": ["conv", "full_attention", "conv"]
        }))
        .unwrap();
        let (capabilities, estimate) = architecture_capability::lfm2(&args).unwrap().into_parts();
        let strategy = capabilities.state_strategy;
        assert_eq!(
            strategy,
            CacheStateStrategy::HybridRecurrent {
                full_attention_layers: 1,
                sliding_attention: Vec::new(),
                recurrent_layers: 2,
            }
        );
        assert_eq!(estimate.fixed_scalars_per_batch, 64);
        assert_eq!(estimate.growing.len(), 1);
        assert_eq!(estimate.growing[0].layers, 1);
        assert_eq!(estimate.growing[0].scalars_per_position, 16);
        assert_eq!(estimate.growing[0].window, None);
    }

    #[test]
    fn gpt_oss_runtime_state_uses_exact_schedule_and_distinct_windows() {
        use eredu_core::attention::{AttentionPolicy, LayerSchedule};

        let mut args = gpt_oss::model_args_from_config_value(&json!({
            "model_type": "gpt_oss", "hidden_size": 32,
            "intermediate_size": 32, "num_hidden_layers": 4,
            "num_attention_heads": 2, "num_key_value_heads": 1,
            "head_dim": 16, "vocab_size": 32, "num_local_experts": 2,
            "num_experts_per_tok": 1, "rms_norm_eps": 1e-5,
            "sliding_window": 5, "max_position_embeddings": 128,
            "layer_types": [
                "sliding_attention", "full_attention",
                "sliding_attention", "full_attention"
            ],
            "quantization_config": {"quant_method": "mxfp4"}
        }))
        .unwrap();
        args.attention_schedule = LayerSchedule::new(
            4,
            vec![
                AttentionPolicy::sliding(3).unwrap(),
                AttentionPolicy::Full,
                AttentionPolicy::sliding(5).unwrap(),
                AttentionPolicy::Full,
            ],
        )
        .unwrap();
        let state_layout = gpt_oss::state_layout(&args).unwrap();
        assert_eq!(state_layout.len(), 4);
        for (layer, attention) in args.attention_schedule.iter().copied().enumerate() {
            assert_eq!(
                state_layout.layer(layer),
                Some(
                    &eredu_core::cache::LayerCachePolicy::key_value(
                        attention,
                        args.num_key_value_heads,
                        args.head_dim,
                    )
                    .unwrap()
                )
            );
        }
        let identity = gpt_oss::state_identity(
            &args,
            &state_layout,
            0,
            eredu_core::cache::PromptCacheTopology::default(),
        )
        .unwrap();
        assert_eq!(identity.layer_count, state_layout.len());
        assert_eq!(identity.global_layer_start, 0);
        assert_eq!(identity.layer_prefix_offsets.len(), state_layout.len());
        assert_eq!(
            identity.architecture_fingerprint,
            gpt_oss::prompt_cache_architecture_fingerprint(&args)
        );
        let (capabilities, estimate) = architecture_capability::gpt_oss(&args)
            .unwrap()
            .into_parts();
        let strategy = capabilities.state_strategy;
        assert_eq!(
            strategy,
            CacheStateStrategy::MixedKv {
                full_layers: 2,
                sliding: vec![
                    SlidingWindowLayerCount {
                        layers: 1,
                        window: 3,
                    },
                    SlidingWindowLayerCount {
                        layers: 1,
                        window: 5,
                    },
                ],
            }
        );
        assert_eq!(estimate.growing.len(), 3);
        assert_eq!(estimate.growing[0].window, None);
        assert_eq!(estimate.growing[1].window, Some(3));
        assert_eq!(estimate.growing[2].window, Some(5));
    }

    #[test]
    fn nemotron_runtime_state_tracks_mixed_recurrent_kv_and_stateless_layers() {
        let args = nemotron_h::model_args_from_config_value(&json!({
            "model_type": "nemotron_h", "vocab_size": 32, "hidden_size": 8,
            "intermediate_size": 12, "num_hidden_layers": 4,
            "hybrid_override_pattern": "M*-E", "num_attention_heads": 2,
            "num_key_value_heads": 1, "head_dim": 4,
            "max_position_embeddings": 128, "sliding_window": 5,
            "mamba_num_heads": 2, "mamba_head_dim": 4, "n_groups": 1,
            "ssm_state_size": 4, "conv_kernel": 3, "chunk_size": 2,
            "moe_intermediate_size": 6,
            "moe_shared_expert_intermediate_size": 10,
            "n_routed_experts": 2, "n_shared_experts": 1,
            "num_experts_per_tok": 2, "mlp_hidden_act": "relu2",
            "mamba_hidden_act": "silu"
        }))
        .unwrap();
        let (capabilities, estimate) = architecture_capability::nemotron_h(&args)
            .unwrap()
            .into_parts();
        let strategy = capabilities.state_strategy;
        assert_eq!(
            strategy,
            CacheStateStrategy::HybridRecurrent {
                full_attention_layers: 0,
                sliding_attention: vec![SlidingWindowLayerCount {
                    layers: 1,
                    window: 5,
                }],
                recurrent_layers: 1,
            }
        );
        assert_eq!(estimate.fixed_scalars_per_batch, 64);
        assert_eq!(estimate.growing.len(), 1);
        assert_eq!(estimate.growing[0].layers, 1);
        assert_eq!(estimate.growing[0].scalars_per_position, 8);
        assert_eq!(estimate.growing[0].window, Some(5));
        let state = estimate_mlx_runtime_state(&estimate, InputTokenCount::text(10), 0, 2).unwrap();
        assert_eq!(state.fixed_state_bytes, 512);
        assert_eq!(state.context_state_bytes, 320);
        assert_eq!(state.bytes_per_position_per_batch, 0);
        assert_eq!(state.assumptions.sliding_window_bounds, vec![5]);
    }

    #[test]
    fn qwen_hybrid_runtime_state_uses_the_normalized_schedule() {
        let args = eredu_architectures::qwen::hybrid::model_args_from_config_value(&json!({
            "model_type": "qwen3_next", "vocab_size": 32, "hidden_size": 16,
            "num_hidden_layers": 4, "num_attention_heads": 2,
            "num_key_value_heads": 1, "head_dim": 8,
            "max_position_embeddings": 128, "intermediate_size": 32,
            "num_experts": 0, "linear_conv_kernel_dim": 3,
            "linear_key_head_dim": 4, "linear_value_head_dim": 4,
            "linear_num_key_heads": 2, "linear_num_value_heads": 2,
            "layer_types": [
                "full_attention", "linear_attention",
                "linear_attention", "full_attention"
            ]
        }))
        .unwrap();
        let (capabilities, estimate) = architecture_capability::qwen_hybrid(&args)
            .unwrap()
            .into_parts();
        let strategy = capabilities.state_strategy;
        assert_eq!(
            strategy,
            CacheStateStrategy::HybridRecurrent {
                full_attention_layers: 2,
                sliding_attention: Vec::new(),
                recurrent_layers: 2,
            }
        );
        assert_eq!(estimate.fixed_scalars_per_batch, 160);
        assert_eq!(estimate.growing.len(), 1);
        assert_eq!(estimate.growing[0].layers, 2);
        assert_eq!(estimate.growing[0].scalars_per_position, 16);
    }

    fn tiny_llama(kv_heads: i32, sliding_window: Option<i32>) -> LlamaModelArgs {
        LlamaModelArgs {
            model_type: "mistral".into(),
            hidden_size: 32,
            num_hidden_layers: 2,
            intermediate_size: 64,
            num_attention_heads: 4,
            rms_norm_eps: 1e-5,
            vocab_size: 64,
            num_key_value_heads: kv_heads,
            max_position_embeddings: 128,
            rope_theta: 10_000.0,
            rope_traditional: false,
            head_dim: 8,
            tie_word_embeddings: true,
            attention_bias: false,
            mlp_bias: false,
            rope_scaling: None,
            attention_schedule: match sliding_window {
                Some(window) => eredu_core::attention::LayerSchedule::all_sliding(
                    2,
                    u32::try_from(window).unwrap(),
                )
                .unwrap(),
                None => eredu_core::attention::LayerSchedule::all_full(2).unwrap(),
            },
            quantization: None,
            quantization_config: None,
            quantized_weights: None,
            quantized_weight_configs: None,
        }
    }

    fn tiny_gemma4() -> gemma4::ModelArgs {
        let layer = |attention, key_value| gemma4::LayerPolicy {
            attention,
            head_dim: std::num::NonZeroU32::new(4).unwrap(),
            num_key_value_heads: std::num::NonZeroU32::new(1).unwrap(),
            key_value,
            intermediate_size: std::num::NonZeroU32::new(16).unwrap(),
            feed_forward: gemma4::FeedForwardPolicy::Dense,
        };
        gemma4::ModelArgs {
            model_type: "gemma4_unified".into(),
            hidden_size: 8,
            num_attention_heads: 2,
            rms_norm_eps: 1e-5,
            vocab_size: 32,
            pad_token_id: 0,
            max_position_embeddings: 128,
            rope_theta: 10_000.0,
            tie_word_embeddings: true,
            attention_bias: false,
            weight_quantization: None,
            quantized_weights: None,
            quantized_weight_configs: None,
            hidden_size_per_layer_input: 0,
            vocab_size_per_layer_input: None,
            layer_schedule: eredu_core::attention::LayerSchedule::new(
                4,
                vec![
                    layer(
                        AttentionPolicy::sliding(4).unwrap(),
                        eredu_nn::AttentionStateSource::Local {
                            value: eredu_nn::AttentionValueSource::Projected,
                        },
                    ),
                    layer(
                        AttentionPolicy::Full,
                        eredu_nn::AttentionStateSource::Publish {
                            value: eredu_nn::AttentionValueSource::Projected,
                        },
                    ),
                    layer(
                        AttentionPolicy::sliding(4).unwrap(),
                        eredu_nn::AttentionStateSource::Local {
                            value: eredu_nn::AttentionValueSource::Projected,
                        },
                    ),
                    layer(
                        AttentionPolicy::Full,
                        eredu_nn::AttentionStateSource::Shared,
                    ),
                ],
            )
            .unwrap(),
            final_logit_softcapping: None,
            num_experts: None,
            top_k_experts: None,
            moe_intermediate_size: None,
            rope_scaling: None,
            rope_parameters: None,
        }
    }

    fn gemma4_capability(
        text: gemma4::ModelArgs,
        modalities: InputModalities,
    ) -> eredu_architectures::capability::CapabilityEstimate {
        let model_type = text.model_type.clone();
        architecture_capability::gemma4(&gemma4::FamilyConfig {
            model_type,
            text,
            vision: None,
            image_token_id: modalities.image.then_some(1),
            video_token_id: modalities.video.then_some(2),
            audio: None,
            audio_token_id: modalities.audio.then_some(3),
        })
        .unwrap()
    }

    fn estimate(
        fixed: u64,
        components: Vec<GrowingState>,
        positions: u64,
        batch: u64,
    ) -> Result<RuntimeStateEstimate, CapabilityError> {
        estimate_mlx_runtime_state(
            &StateLayout {
                fixed_scalars_per_batch: fixed,
                growing: components,
                hidden_size: 1,
                allocation_granularity: 1,
                completeness: EstimationCompleteness::Complete,
            },
            InputTokenCount::text(positions),
            0,
            batch,
        )
    }

    #[test]
    fn standard_kv_and_gqa_use_kv_head_count() {
        let (capabilities, llama_layout) = architecture_capability::llama(&tiny_llama(4, None))
            .unwrap()
            .into_parts();
        let strategy = capabilities.state_strategy;
        assert_eq!(strategy, CacheStateStrategy::FullKv);
        let llama =
            estimate_mlx_runtime_state(&llama_layout, InputTokenCount::text(10), 0, 1).unwrap();
        assert_eq!(llama.requested_state_bytes, 2 * 2 * 4 * 8 * 10 * 4);

        let (_, gqa_layout) = architecture_capability::llama(&tiny_llama(1, None))
            .unwrap()
            .into_parts();
        let gqa = estimate_mlx_runtime_state(&gqa_layout, InputTokenCount::text(10), 0, 1).unwrap();
        assert_eq!(gqa.requested_state_bytes, llama.requested_state_bytes / 4);
    }

    #[test]
    fn llama_runtime_state_groups_exact_per_layer_windows() {
        use eredu_core::attention::{AttentionPolicy, LayerSchedule};

        let mut args = tiny_llama(2, None);
        args.attention_schedule = LayerSchedule::new(
            2,
            vec![AttentionPolicy::Full, AttentionPolicy::sliding(3).unwrap()],
        )
        .unwrap();
        let (capabilities, layout) = architecture_capability::llama(&args).unwrap().into_parts();
        let strategy = capabilities.state_strategy;
        assert_eq!(
            strategy,
            CacheStateStrategy::MixedKv {
                full_layers: 1,
                sliding: vec![SlidingWindowLayerCount {
                    layers: 1,
                    window: 3,
                }],
            }
        );
        let estimate =
            estimate_mlx_runtime_state(&layout, InputTokenCount::text(10), 0, 2).unwrap();
        assert_eq!(estimate.context_state_bytes, (10 + 3) * 2 * 8 * 2 * 2 * 4);
        assert_eq!(estimate.assumptions.sliding_window_bounds, vec![3]);
    }

    #[test]
    fn sliding_window_bounds_only_bounded_layers() {
        let estimate = estimate(
            0,
            vec![
                GrowingState {
                    layers: 1,
                    scalars_per_position: 16,
                    window: None,
                },
                GrowingState {
                    layers: 3,
                    scalars_per_position: 16,
                    window: Some(4),
                },
            ],
            10,
            1,
        )
        .unwrap();
        assert_eq!(estimate.context_state_bytes, (10 + 3 * 4) * 16 * 4);
        assert_eq!(estimate.bytes_per_position_per_batch, 16 * 4);
    }

    #[test]
    fn compressed_mla_uses_latent_plus_rotary_width() {
        let estimate = estimate(
            0,
            vec![GrowingState {
                layers: 3,
                scalars_per_position: 12 + 4,
                window: None,
            }],
            5,
            2,
        )
        .unwrap();
        assert_eq!(estimate.requested_state_bytes, 3 * 16 * 5 * 2 * 4);
    }

    #[test]
    fn kimi_linear_accounts_for_bounded_kda_and_growing_mla_state() {
        let args = kimi_linear::model_args_from_config_value(&json!({
            "model_type": "kimi_linear",
            "vocab_size": 64,
            "hidden_size": 8,
            "num_hidden_layers": 4,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "intermediate_size": 16,
            "head_dim": 4,
            "model_max_length": 128,
            "linear_attn_config": {
                "kda_layers": [1, 3],
                "full_attn_layers": [2, 4],
                "num_heads": 2,
                "head_dim": 4,
                "short_conv_kernel_size": 2
            },
            "num_experts": 4,
            "moe_intermediate_size": 8,
            "kv_lora_rank": 4,
            "qk_nope_head_dim": 2,
            "qk_rope_head_dim": 2,
            "v_head_dim": 2,
            "mla_use_nope": true,
            "num_experts_per_token": 2,
            "routed_scaling_factor": 1.0,
            "first_k_dense_replace": 1,
            "num_expert_group": 1,
            "topk_group": 1
        }))
        .unwrap();
        let (capabilities, estimate) = architecture_capability::kimi_linear(&args)
            .unwrap()
            .into_parts();
        let native = capabilities.native_max_context;
        let effective = capabilities.effective_max_context;
        let strategy = capabilities.state_strategy;
        let modalities = capabilities.modalities;
        assert_eq!(native.value(), Some(&128));
        assert_eq!(effective.value(), Some(&128));
        assert_eq!(
            strategy,
            CacheStateStrategy::HybridRecurrent {
                full_attention_layers: 2,
                sliding_attention: Vec::new(),
                recurrent_layers: 2,
            }
        );
        assert_eq!(modalities, InputModalities::TEXT);
        assert_eq!(estimate.fixed_scalars_per_batch, 112);
        assert_eq!(estimate.growing.len(), 1);
        assert_eq!(estimate.growing[0].layers, 2);
        assert_eq!(estimate.growing[0].scalars_per_position, 6);
        assert_eq!(estimate.allocation_granularity, 256);
    }

    #[test]
    fn hybrid_fixed_and_attention_state_are_separate() {
        let estimate = estimate(
            100,
            vec![GrowingState {
                layers: 2,
                scalars_per_position: 8,
                window: None,
            }],
            5,
            3,
        )
        .unwrap();
        assert_eq!(estimate.fixed_state_bytes, 100 * 3 * 4);
        assert_eq!(estimate.context_state_bytes, 2 * 8 * 5 * 3 * 4);
    }

    #[test]
    fn multimodal_positions_are_distinct_from_text_tokens() {
        let count = InputTokenCount::prepared(7, 12, 19, 1_024, ObservationKind::Conservative);
        assert_eq!(
            count.text_tokens + count.media_positions,
            count.model_positions
        );
        assert_eq!(count.media_execution_workspace_bytes(), 1_024);
        assert_eq!(
            count.media_execution_workspace_kind(),
            ObservationKind::Conservative
        );
    }

    fn tiny_inkling() -> eredu_architectures::inkling::ModelArgs {
        eredu_architectures::inkling::ModelArgs::from_hf_json(
            &serde_json::to_vec(&json!({
                "model_type":"inkling_mm_model",
                "text_config":{
                    "hidden_size":32,"num_hidden_layers":3,"vocab_size":64,
                    "num_attention_heads":4,"num_key_value_heads":2,"head_dim":8,
                    "swa_num_attention_heads":4,"swa_num_key_value_heads":2,"swa_head_dim":8,
                    "sliding_window_size":8,"local_layer_ids":[0,1],"dense_mlp_idx":1,
                    "sconv_kernel_size":4,"d_rel":4,"rel_extent":16,
                    "intermediate_size":24,"dense_intermediate_size":48,
                    "n_routed_experts":4,"num_experts_per_tok":2,"n_shared_experts":1,
                    "route_scale":8.0,"use_sconv":true,"use_embed_norm":true,
                    "shared_expert_sink":true,"use_gate_bias":true,"norm_after_topk":true,
                    "use_global_scale":true,"gate_activation":"sigmoid"
                },
                "audio_config":{
                    "decoder_dmodel":32,"n_mel_bins":80,"mel_vocab_size":16
                },
                "vision_config":{
                    "decoder_dmodel":32,"patch_size":40,"temporal_patch_size":2,
                    "n_channels":3,"n_layers":4
                }
            }))
            .unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn inkling_runtime_state_groups_the_exact_ordered_schedule() {
        use eredu_architectures::inkling::{FeedForwardPolicy, LayerPolicy};
        use eredu_core::attention::LayerSchedule;

        let mut args = tiny_inkling();
        args.text_config.num_key_value_heads = 1;
        args.text_config.swa_num_key_value_heads = Some(2);
        args.text_config.layer_schedule = LayerSchedule::new(
            3,
            vec![
                LayerPolicy {
                    attention: AttentionPolicy::Full,
                    feed_forward: FeedForwardPolicy::Dense,
                },
                LayerPolicy {
                    attention: AttentionPolicy::sliding(3).unwrap(),
                    feed_forward: FeedForwardPolicy::SparseMoe,
                },
                LayerPolicy {
                    attention: AttentionPolicy::sliding(5).unwrap(),
                    feed_forward: FeedForwardPolicy::SparseMoe,
                },
            ],
        )
        .unwrap();

        let (capabilities, layout) = architecture_capability::inkling(&args)
            .unwrap()
            .into_parts();
        let strategy = capabilities.state_strategy;
        assert_eq!(
            strategy,
            CacheStateStrategy::Multimodal {
                decoder: Box::new(CacheStateStrategy::MixedKv {
                    full_layers: 1,
                    sliding: vec![
                        SlidingWindowLayerCount {
                            layers: 1,
                            window: 3,
                        },
                        SlidingWindowLayerCount {
                            layers: 1,
                            window: 5,
                        },
                    ],
                }),
                media_consumes_decoder_positions: true,
            }
        );
        let state = estimate_mlx_runtime_state(&layout, InputTokenCount::text(10), 0, 2).unwrap();
        assert_eq!(state.assumptions.sliding_window_bounds, vec![3, 5]);
        // Full KV: 1 x 10 x (1 head x 8 x K/V). Sliding KV: (3 + 5) x
        // (2 heads x 8 x K/V), all for two batches of f32 state.
        assert_eq!(state.context_state_bytes, (10 * 16 + (3 + 5) * 32) * 2 * 4);
    }

    #[test]
    fn gemma4_shared_and_sliding_layers_use_full_chunked_kv_backing() {
        let modalities = InputModalities {
            text: true,
            image: true,
            audio: false,
            video: false,
        };
        let (capabilities, layout) = gemma4_capability(tiny_gemma4(), modalities).into_parts();
        let strategy = capabilities.state_strategy;
        assert_eq!(
            strategy,
            CacheStateStrategy::Multimodal {
                decoder: Box::new(CacheStateStrategy::SharedFullKv {
                    cached_layers: 3,
                    shared_layers: 1,
                    full_attention_layers: 2,
                    sliding_attention: vec![SlidingWindowLayerCount {
                        window: 4,
                        layers: 2,
                    }],
                }),
                media_consumes_decoder_positions: true,
            }
        );
        let estimate = estimate_mlx_runtime_state(
            &layout,
            InputTokenCount::prepared(5, 3, 8, 1_024, ObservationKind::Conservative),
            2,
            2,
        )
        .unwrap();
        assert_eq!(estimate.assumptions.allocation_granularity, 256);
        assert!(estimate.assumptions.sliding_window_bounds.is_empty());
        assert_eq!(estimate.context_state_bytes, 3 * 2 * 2 * 4 * 256 * 4);
        assert_eq!(estimate.multimodal_embedding_bytes, 3 * 8 * 2 * 4);
        assert_eq!(estimate.media_execution_workspace_bytes, 2_048);
        assert_eq!(estimate.completeness, EstimationCompleteness::Conservative);
    }

    #[test]
    fn gemma4_capabilities_report_each_exact_sliding_window() {
        let mut args = tiny_gemma4();
        let attentions = [
            AttentionPolicy::sliding(3).unwrap(),
            AttentionPolicy::Full,
            AttentionPolicy::sliding(5).unwrap(),
            AttentionPolicy::Full,
        ];
        args.layer_schedule = eredu_core::attention::LayerSchedule::new(
            4,
            args.layer_schedule
                .iter()
                .copied()
                .zip(attentions)
                .map(|(policy, attention)| gemma4::LayerPolicy {
                    attention,
                    ..policy
                })
                .collect(),
        )
        .unwrap();
        let (capabilities, _) = gemma4_capability(args, InputModalities::TEXT).into_parts();
        let strategy = capabilities.state_strategy;
        assert_eq!(
            strategy,
            CacheStateStrategy::SharedFullKv {
                cached_layers: 3,
                shared_layers: 1,
                full_attention_layers: 2,
                sliding_attention: vec![
                    SlidingWindowLayerCount {
                        window: 3,
                        layers: 1,
                    },
                    SlidingWindowLayerCount {
                        window: 5,
                        layers: 1,
                    },
                ],
            }
        );
    }

    #[test]
    fn gemma4_runtime_state_uses_each_scheduled_kv_geometry() {
        let mut args = tiny_gemma4();
        let mut policies = args.layer_schedule.iter().copied().collect::<Vec<_>>();
        policies[0].head_dim = std::num::NonZeroU32::new(8).unwrap();
        args.layer_schedule = eredu_core::attention::LayerSchedule::new(4, policies).unwrap();

        let (_, layout) = gemma4_capability(args, InputModalities::TEXT).into_parts();
        assert_eq!(layout.growing.len(), 2);
        assert_eq!(layout.growing[0].layers, 2);
        assert_eq!(layout.growing[0].scalars_per_position, 8);
        assert_eq!(layout.growing[0].window, None);
        assert_eq!(layout.growing[1].layers, 1);
        assert_eq!(layout.growing[1].scalars_per_position, 16);
        assert_eq!(layout.growing[1].window, None);
    }

    #[test]
    fn checked_arithmetic_reports_overflow() {
        assert_eq!(
            checked_mul(u64::MAX, 2, "synthetic overflow"),
            Err(CapabilityError::ArithmeticOverflow {
                operation: "synthetic overflow"
            })
        );

        let layout = StateLayout {
            fixed_scalars_per_batch: 0,
            growing: Vec::new(),
            hidden_size: 1,
            allocation_granularity: 1,
            completeness: EstimationCompleteness::Complete,
        };
        assert!(matches!(
            estimate_mlx_runtime_state(
                &layout,
                InputTokenCount::prepared(0, 1, 1, u64::MAX, ObservationKind::Conservative,),
                0,
                2,
            ),
            Err(CapabilityError::ArithmeticOverflow {
                operation: "media execution workspace times batch"
            })
        ));
    }

    #[test]
    fn unavailable_memory_is_not_zero() {
        let value: Observed<u64> = Observed::Unavailable {
            reason: "synthetic".into(),
        };
        assert_eq!(value.value(), None);
    }

    #[test]
    fn apple_unified_semantics_do_not_create_two_capacities() {
        let report = AvailableMemory {
            physical_memory_bytes: Observed::Available {
                value: 16,
                kind: ObservationKind::Exact,
                source: "test".into(),
            },
            available_memory_bytes: Observed::Available {
                value: 8,
                kind: ObservationKind::Estimated,
                source: "test".into(),
            },
            physical_semantics: PhysicalMemorySemantics::Unified,
        };
        assert_eq!(report.physical_memory_bytes.value(), Some(&16));
        assert_eq!(report.physical_semantics, PhysicalMemorySemantics::Unified);
    }

    #[test]
    fn dtype_assumption_follows_the_session_activation_width() {
        let layout = StateLayout {
            fixed_scalars_per_batch: 0,
            growing: vec![GrowingState {
                layers: 1,
                scalars_per_position: 16,
                window: None,
            }],
            hidden_size: 1,
            allocation_granularity: 1,
            completeness: EstimationCompleteness::Complete,
        };
        let estimate = estimate_mlx_runtime_state_with_dtype(
            &layout,
            InputTokenCount::text(2),
            0,
            1,
            NonZeroU8::new(2).unwrap(),
        )
        .unwrap();
        assert_eq!(estimate.assumptions.state_dtype_bytes.get(), 2);
        assert_eq!(estimate.requested_state_bytes, 2 * 16 * 2);
    }

    #[test]
    fn capability_value_never_invents_default() {
        let unsupported: Observed<u64> = Observed::Unsupported {
            reason: "not supported".into(),
        };
        assert!(unsupported.value().is_none());
    }
}
