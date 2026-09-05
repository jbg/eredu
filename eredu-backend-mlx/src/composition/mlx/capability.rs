//! MLX model capability derivation, resource observation, and admission adapter.

use std::num::NonZeroU8;

use eredu_architectures::media_plan::{self, MediaShapePlan, PreparedInputPartPlan};
use eredu_core::{
    estimate_runtime_state, AvailableMemory, CapabilityError, InputMetadataKey, InputModality,
    InputTokenCount, ModelCapabilities, ModelCapabilityBackend, ModelRuntime, ObservationKind,
    Observed, PhysicalMemorySemantics, RuntimeStateEstimate, StateMemoryLayout, StaticMemoryReport,
};
use safemlx::{Array, Stream};

use super::{Executable, MlxBackend, MlxModelInput, MlxModelSession};
use crate::backend::runtime::media::input::{self, InputPayload};
use eredu_core::residency::MemoryTier;

fn checked_add(left: u64, right: u64, operation: &'static str) -> Result<u64, CapabilityError> {
    left.checked_add(right)
        .ok_or(CapabilityError::ArithmeticOverflow { operation })
}

fn checked_mul(left: u64, right: u64, operation: &'static str) -> Result<u64, CapabilityError> {
    left.checked_mul(right)
        .ok_or(CapabilityError::ArithmeticOverflow { operation })
}

fn estimate_mlx_runtime_state_with_dtype(
    layout: &StateMemoryLayout,
    input: InputTokenCount,
    max_output_tokens: u64,
    batch_size: u64,
    floating_state_dtype_bytes: NonZeroU8,
) -> Result<RuntimeStateEstimate, CapabilityError> {
    estimate_runtime_state(
        layout,
        input,
        max_output_tokens,
        batch_size,
        floating_state_dtype_bytes,
    )
}

#[cfg(test)]
fn estimate_mlx_runtime_state(
    layout: &StateMemoryLayout,
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

impl Executable {
    pub(super) fn prepared_input_part_plan(
        &self,
        input: &input::InputPart,
    ) -> Result<PreparedInputPartPlan, CapabilityError> {
        media_plan::text_only_input_part(
            self.effective_model_type(),
            input,
            &input::MlxInputInspector,
        )
    }

    pub(super) fn architecture_capability_estimate(
        &self,
    ) -> Result<eredu_architectures::capability::CapabilityEstimate, CapabilityError> {
        Ok(self.erased().capability_estimate().clone())
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

fn array_bytes(array: &Array, operation: &'static str) -> Result<u64, CapabilityError> {
    u64::try_from(array.nbytes()).map_err(|_| CapabilityError::ArithmeticOverflow { operation })
}

fn four_byte_scalars(scalars: u64, operation: &'static str) -> Result<u64, CapabilityError> {
    checked_mul(scalars, 4, operation)
}

fn prepared_media_accounting_with_plan(
    payload: &Array,
    part: &input::InputPart,
    plan: &MediaShapePlan,
) -> Result<(u64, u64), CapabilityError> {
    let mut input_bytes = array_bytes(payload, "prepared media payload bytes")?;
    for array in [
        part.metadata_value(InputMetadataKey::PatchGrid),
        part.metadata_value(InputMetadataKey::PatchPositions),
        part.metadata_value(InputMetadataKey::AudioMask),
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

pub fn model_capabilities(session: &MlxModelSession) -> Result<ModelCapabilities, CapabilityError> {
    session
        .capability_estimate()
        .map(|estimate| estimate.into_parts().0)
}

pub fn count_prepared_input(
    session: &MlxModelSession,
    prepared: input::ModelInput<'_>,
    _stream: &Stream,
) -> Result<InputTokenCount, CapabilityError> {
    let mut text_tokens = 0u64;
    let mut media_positions = 0u64;
    let mut media_execution_workspace_bytes = 0u64;
    let mut media_execution_workspace_kind = ObservationKind::Exact;
    for part in prepared.parts {
        match session.prepared_input_part_plan(part)? {
            PreparedInputPartPlan::Text { positions } => {
                text_tokens = checked_add(text_tokens, positions, "prepared text-token total")?;
            }
            PreparedInputPartPlan::Projected {
                modality,
                positions,
            } => {
                if modality == InputModality::Text {
                    text_tokens = checked_add(text_tokens, positions, "prepared text-token total")?;
                } else {
                    media_positions =
                        checked_add(media_positions, positions, "prepared media-position total")?;
                }
            }
            PreparedInputPartPlan::Media { shape } => {
                let InputPayload::Tensor(tensor) = part.payload() else {
                    return Err(CapabilityError::UnsupportedInput {
                        architecture: session.effective_model_type().into(),
                        reason: "architecture media plan requires a tensor payload".into(),
                    });
                };
                let (positions, workspace_bytes) =
                    prepared_media_accounting_with_plan(tensor, part, &shape)?;
                media_positions =
                    checked_add(media_positions, positions, "prepared media-position total")?;
                media_execution_workspace_bytes = checked_add(
                    media_execution_workspace_bytes,
                    workspace_bytes,
                    "prepared media-workspace total",
                )?;
                media_execution_workspace_kind = ObservationKind::Conservative;
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
    session: &MlxModelSession,
    input: InputTokenCount,
    max_output_tokens: u64,
    batch_size: u64,
    floating_state_dtype_bytes: NonZeroU8,
) -> Result<RuntimeStateEstimate, CapabilityError> {
    let estimate = session.capability_estimate()?;
    estimate_mlx_runtime_state_with_dtype(
        estimate.state_layout(),
        input,
        max_output_tokens,
        batch_size,
        floating_state_dtype_bytes,
    )
}

pub fn static_model_memory(
    session: &MlxModelSession,
) -> Result<StaticMemoryReport, CapabilityError> {
    let residency = session
        .residency_report()
        .map_err(|error| CapabilityError::Observation(error.to_string()))?;
    let (logical, host, device, disk, cached_shards) = if let Some(report) = residency {
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
                value: report.weight_store().currently_cached_shards as u64,
                kind: ObservationKind::Observational,
                source: "checkpoint shard cache".into(),
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
                reason: "checkpoint shard-cache information unavailable".into(),
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
        currently_cached_shards: cached_shards,
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
            runtime.session().floating_state_dtype_bytes(),
        )
    }

    fn static_memory(runtime: &ModelRuntime<Self>) -> Result<StaticMemoryReport, CapabilityError> {
        static_model_memory(runtime.session())
    }
}

/// Queries system memory that can be used as an admission signal.
///
/// Apple Silicon reports one unified physical capacity; logical host/device
/// residency tiers must not be added as independent physical capacities.
pub fn available_memory() -> Result<AvailableMemory, CapabilityError> {
    let memory = safemlx::system::system_memory()
        .map_err(|error| CapabilityError::Observation(error.to_string()))?;
    let physical_memory_bytes = memory.total.map_or_else(
        || Observed::Unavailable {
            reason: "physical-memory capacity is unavailable on this platform".into(),
        },
        |value| Observed::Available {
            value,
            kind: ObservationKind::Exact,
            source: "host physical-memory observation".into(),
        },
    );
    let available_memory_bytes = memory.available.map_or_else(
        || Observed::Unavailable {
            reason: "available physical memory is unavailable on this platform".into(),
        },
        |value| Observed::Available {
            value,
            kind: ObservationKind::Estimated,
            source: "host available-memory observation".into(),
        },
    );
    Ok(AvailableMemory {
        physical_memory_bytes,
        available_memory_bytes,
        physical_semantics: if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
            PhysicalMemorySemantics::Unified
        } else {
            PhysicalMemorySemantics::Unknown
        },
    })
}

#[cfg(test)]
mod tests;
