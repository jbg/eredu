//! MLX observations and plan realization for neutral automatic planning.

use std::{fs, path::Path};

use safemlx::{Device, DeviceType, Stream};
use safemlx_lm_core::{
    AutomaticPlanRequest, AutomaticPlanner, AutomaticPlanningBackend, AutomaticPlanningError,
    BackendId, BoundedResidencyRequirement, CandidateAdmission, DevicePlan, DurationSeconds,
    ExecutionPlan, ExecutionPlanReport, ExpertCacheTelemetry, HardwareBackendProfile,
    HardwareDeviceProfile, HardwareMemorySemantics, HardwareProfile, ModelKind,
    ModelResourceProfile, MtpStats, MtpTelemetry, ObservationKind, Observed, ResidencyPlan,
    ResidencyTelemetry, TransferTelemetry, WeightTransformationPlan, AUTOMATIC_SCHEMA_VERSION,
    EXECUTION_PLAN_SCHEMA_VERSION,
};

use super::{MlxBackend, ModelLoadOptions};
use crate::{
    api::{
        available_memory, inspect_model, CapabilityValue, InspectionSeverity, MeasurementKind,
        ModelInspectionOptions, PhysicalMemorySemantics,
    },
    error::Error,
    runtime::{
        checkpoint::quantization::{AffineQuantization, WeightQuantization},
        execution::layerwise::{
            LayerwiseLoadOptions, LayerwiseModelError, NonExpertWeightResidency, WeightResidency,
        },
        residency::{
            dense_stream::DenseDiskStreamLoadOptions, expert_cache::ExpertCacheLoadOptions,
        },
    },
};
use safemlx_lm_core::residency::{MemoryTier, OffloadConfig, TransferDirection};

/// MLX implementation of high-level automatic-planning observations.
#[derive(Debug, Clone, Copy, Default)]
pub struct MlxAutomaticPlanningBackend;

/// Discovers hardware facts visible to the MLX adapter.
pub fn discover_hardware() -> HardwareProfile {
    let logical_cpu_count = std::thread::available_parallelism().map_or_else(
        |error| Observed::unavailable(error.to_string()),
        |count| Observed::exact(count.get() as u64, "std::thread::available_parallelism"),
    );
    let (physical_memory_bytes, available_memory_bytes, semantics) = match available_memory() {
        Ok(memory) => (
            observed_capability(&memory.physical_memory_bytes),
            observed_capability(&memory.available_memory_bytes),
            memory_semantics(memory.physical_semantics),
        ),
        Err(error) => (
            Observed::unavailable(error.to_string()),
            Observed::unavailable(error.to_string()),
            HardwareMemorySemantics::Unknown,
        ),
    };
    let mut devices = vec![HardwareDeviceProfile {
        id: "cpu:0".into(),
        family: "cpu".into(),
        index: 0,
        total_memory_bytes: physical_memory_bytes.clone(),
        available_memory_bytes: available_memory_bytes.clone(),
    }];
    let mut details = Vec::new();

    #[cfg(target_os = "macos")]
    {
        let (available, detail) = match safemlx::metal::is_available() {
            Ok(available) => (available, None),
            Err(error) => (false, Some(error.to_string())),
        };
        if available {
            let (total, available) = if semantics == HardwareMemorySemantics::Unified {
                (
                    physical_memory_bytes.clone(),
                    available_memory_bytes.clone(),
                )
            } else {
                (
                    Observed::unavailable("MLX does not expose discrete Metal capacity"),
                    Observed::unavailable("MLX does not expose discrete Metal availability"),
                )
            };
            devices.push(HardwareDeviceProfile {
                id: "metal:0".into(),
                family: "metal".into(),
                index: 0,
                total_memory_bytes: total,
                available_memory_bytes: available,
            });
        } else if let Some(detail) = detail {
            details.push(format!("Metal: {detail}"));
        }
    }

    #[cfg(feature = "cuda")]
    {
        let (available, detail) = match safemlx::cuda::is_available() {
            Ok(available) => (available, None),
            Err(error) => (false, Some(error.to_string())),
        };
        if available {
            devices.push(HardwareDeviceProfile {
                id: "cuda:0".into(),
                family: "cuda".into(),
                index: 0,
                total_memory_bytes: Observed::unavailable(
                    "MLX does not expose CUDA device capacity",
                ),
                available_memory_bytes: Observed::unavailable(
                    "MLX does not expose CUDA device availability",
                ),
            });
        } else if let Some(detail) = detail {
            details.push(format!("CUDA: {detail}"));
        }
    }

    HardwareProfile {
        schema_version: AUTOMATIC_SCHEMA_VERSION,
        operating_system: std::env::consts::OS.into(),
        architecture: std::env::consts::ARCH.into(),
        logical_cpu_count,
        physical_memory_bytes,
        available_memory_bytes,
        physical_memory_semantics: semantics,
        backends: vec![HardwareBackendProfile {
            backend: BackendId::new("mlx").expect("MLX is a valid backend identifier"),
            available: true,
            detail: (!details.is_empty()).then(|| details.join("; ")),
            devices,
        }],
    }
}

/// Runs the neutral planner with MLX observations and candidate admission.
pub fn plan_automatic_execution(
    request: &AutomaticPlanRequest,
) -> Result<ExecutionPlanReport, AutomaticPlanningError> {
    AutomaticPlanner::default().plan(&MlxAutomaticPlanningBackend, request)
}

/// Converts a portable execution plan into MLX loader options.
pub fn execution_plan_load_options(plan: &ExecutionPlan) -> Result<ModelLoadOptions, Error> {
    if plan.schema_version != EXECUTION_PLAN_SCHEMA_VERSION {
        return Err(Error::AutomaticPlanning(format!(
            "execution plan schema {} does not match supported schema {}",
            plan.schema_version, EXECUTION_PLAN_SCHEMA_VERSION
        )));
    }
    if plan.topology.world_size() != 1 {
        return Err(Error::AutomaticPlanning(
            "single-device automatic plans require a 1x1x1 parallel topology".into(),
        ));
    }
    let mut load = match plan.weight_transformation {
        WeightTransformationPlan::PreserveCheckpoint => ModelLoadOptions::default(),
        WeightTransformationPlan::Affine { bits, group_size } => {
            ModelLoadOptions::with_quantization(AffineQuantization::new(group_size, bits)?)
        }
        WeightTransformationPlan::MxFp4 => {
            ModelLoadOptions::with_quantization(WeightQuantization::MxFp4)
        }
    };
    let residency = match &plan.residency {
        ResidencyPlan::FullyResident => NonExpertWeightResidency::FullyResident,
        ResidencyPlan::LayerwiseHost {
            device_layer_window,
            device_budget_bytes,
            host_budget_bytes,
        } => NonExpertWeightResidency::LayerwiseHost(LayerwiseLoadOptions {
            offload: OffloadConfig::new(
                *device_budget_bytes,
                *host_budget_bytes,
                *device_layer_window,
            )?,
            max_mapped_shards: plan.max_mapped_shards,
            ..LayerwiseLoadOptions::default()
        }),
        ResidencyPlan::DenseDiskStream {
            device_budget_bytes,
            host_budget_bytes,
            host_lookahead,
            background_queue,
        } => {
            let mut options = DenseDiskStreamLoadOptions::new(
                *device_budget_bytes,
                *host_budget_bytes,
                *host_lookahead,
                *background_queue,
            )?;
            options.max_mapped_shards = plan.max_mapped_shards;
            NonExpertWeightResidency::DenseDiskStream(options)
        }
    };
    let residency = if let Some(expert) = &plan.expert_cache {
        WeightResidency::with_expert_cache(
            residency,
            ExpertCacheLoadOptions::new(
                OffloadConfig::new(expert.device_budget_bytes, expert.host_budget_bytes, 1)?,
                expert.scratch_bytes,
                expert.prefill_bank_bytes,
            )?,
        )
    } else {
        match residency {
            NonExpertWeightResidency::FullyResident => WeightResidency::fully_resident(),
            NonExpertWeightResidency::LayerwiseHost(options) => {
                WeightResidency::layerwise_host(options)
            }
            NonExpertWeightResidency::DenseDiskStream(options) => {
                WeightResidency::dense_disk_stream(options)
            }
        }
    };
    load = load.with_weight_residency(residency);
    Ok(load)
}

impl AutomaticPlanningBackend for MlxAutomaticPlanningBackend {
    fn backend_id(&self) -> BackendId {
        BackendId::new("mlx").expect("MLX is a valid backend identifier")
    }

    fn discover_hardware(&self) -> Result<HardwareProfile, AutomaticPlanningError> {
        Ok(discover_hardware())
    }

    fn inspect_resources(
        &self,
        model_path: &Path,
    ) -> Result<ModelResourceProfile, AutomaticPlanningError> {
        inspect_model(model_path, ModelInspectionOptions::default())
            .map(|report| report.resources)
            .map_err(|error| planning_backend_error("inspect_resources", error))
    }

    fn admit_candidate(
        &self,
        model_path: &Path,
        plan: &ExecutionPlan,
    ) -> Result<CandidateAdmission, AutomaticPlanningError> {
        let load = execution_plan_load_options(plan)
            .map_err(|error| planning_backend_error("realize_plan", error))?;
        let report = inspect_model(
            model_path,
            ModelInspectionOptions {
                load,
                chat_request: None,
            },
        )
        .map_err(|error| planning_backend_error("admit_candidate", error))?;
        let supported = report.is_loadable();
        let rejection = (!supported).then(|| {
            report
                .issues
                .iter()
                .find(|issue| issue.severity == InspectionSeverity::Error)
                .map(|issue| issue.detail.clone())
                .unwrap_or_else(|| {
                    "checkpoint inspection did not admit this MLX load policy".into()
                })
        });
        Ok(CandidateAdmission {
            supported,
            rejection,
        })
    }

    fn bounded_residency_requirement(
        &self,
        model_path: &Path,
        plan: &ExecutionPlan,
    ) -> Result<BoundedResidencyRequirement, AutomaticPlanningError> {
        let mut probe = plan.clone();
        probe.expert_cache = None;
        match &mut probe.residency {
            ResidencyPlan::LayerwiseHost {
                device_budget_bytes,
                ..
            } => *device_budget_bytes = Some(1),
            ResidencyPlan::DenseDiskStream {
                device_budget_bytes,
                ..
            } => *device_budget_bytes = 1,
            ResidencyPlan::FullyResident => {
                return Err(AutomaticPlanningError::Invalid(
                    "fully resident execution has no bounded device window".into(),
                ));
            }
        }
        let stream = Stream::new_with_device(&mlx_device(&probe.device)?);
        let weights_stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
        let backend = MlxBackend::new(&stream, &weights_stream);
        let options = execution_plan_load_options(&probe)
            .map_err(|error| planning_backend_error("realize_probe", error))?;
        match crate::load_model(&backend, model_path, options) {
            Err(crate::ModelLoadError::Backend(Error::LayerwiseModel(
                LayerwiseModelError::DeviceBudgetTooSmall {
                    static_bytes,
                    window_bytes,
                    depth,
                    required,
                    ..
                },
            ))) => Ok(BoundedResidencyRequirement {
                static_bytes,
                window_bytes,
                required_bytes: required,
                depth,
            }),
            Err(error) => Err(planning_backend_error("bounded_residency_probe", error)),
            Ok(_) => Ok(BoundedResidencyRequirement {
                static_bytes: 0,
                window_bytes: 0,
                required_bytes: 1,
                depth: 0,
            }),
        }
    }

    fn embedded_draft_layers(
        &self,
        model_path: &Path,
        model_kind: Option<ModelKind>,
    ) -> Result<Option<usize>, AutomaticPlanningError> {
        if !matches!(
            model_kind,
            Some(
                ModelKind::DeepSeekV3
                    | ModelKind::Inkling
                    | ModelKind::NemotronH
                    | ModelKind::Qwen3Next
                    | ModelKind::Qwen35
            )
        ) {
            return Ok(Some(0));
        }
        if !model_path.is_dir() {
            return Ok(None);
        }
        let bytes = fs::read(model_path.join("config.json"))
            .map_err(|error| planning_backend_error("read_drafting_metadata", error))?;
        let config: serde_json::Value = serde_json::from_slice(&bytes)
            .map_err(|error| planning_backend_error("decode_drafting_metadata", error))?;
        embedded_mtp_count(&config)
            .map(|count| {
                usize::try_from(count).map_err(|_| {
                    AutomaticPlanningError::Invalid("embedded MTP layer count exceeds usize".into())
                })
            })
            .transpose()
            .map(|count| Some(count.unwrap_or(0)))
    }
}

/// Collects neutral residency telemetry from an MLX loaded model.
pub fn collect_residency_telemetry(
    model: &crate::api::LoadedModel<MlxBackend<'static>>,
) -> Result<Option<ResidencyTelemetry>, Error> {
    let Some(report) = model.residency_report()? else {
        return Ok(None);
    };
    let offload = report.offload();
    let planned = offload.planned_bytes();
    let current = offload.resident_bytes();
    let peak = offload.peak_resident_bytes();
    let transfers = TransferDirection::ALL
        .into_iter()
        .map(|direction| {
            let metrics = offload.transfer(direction);
            TransferTelemetry {
                direction: transfer_direction_name(direction).into(),
                count: metrics.count(),
                bytes: metrics.bytes(),
                seconds: DurationSeconds(metrics.duration().as_secs_f64()),
            }
        })
        .collect();
    Ok(Some(ResidencyTelemetry {
        planned_disk_bytes: planned.get(MemoryTier::Disk),
        planned_host_bytes: planned.get(MemoryTier::Host),
        planned_device_bytes: planned.get(MemoryTier::Device),
        current_host_bytes: current.get(MemoryTier::Host),
        current_device_bytes: current.get(MemoryTier::Device),
        peak_host_bytes: peak.get(MemoryTier::Host),
        peak_device_bytes: peak.get(MemoryTier::Device),
        transfers,
    }))
}

/// Collects neutral routed-expert cache telemetry from an MLX loaded model.
pub fn collect_expert_cache_telemetry(
    model: &crate::api::LoadedModel<MlxBackend<'static>>,
) -> Result<Option<ExpertCacheTelemetry>, Error> {
    Ok(model
        .expert_cache_report()?
        .map(|report| ExpertCacheTelemetry {
            owned_experts: report.owned_experts,
            owned_bytes: report.owned_bytes,
            host_resident_experts: report.host_resident_experts,
            device_resident_experts: report.device_resident_experts,
            host_resident_bytes: report.host_resident_bytes,
            device_resident_bytes: report.device_resident_bytes,
            peak_host_resident_bytes: report.peak_host_resident_bytes,
            peak_device_resident_bytes: report.peak_device_resident_bytes,
        }))
}

/// Converts neutral speculative statistics into the stable telemetry document.
pub fn mtp_telemetry(stats: &MtpStats) -> MtpTelemetry {
    MtpTelemetry {
        execution_topology: stats.execution_topology.to_string(),
        target_tokens: stats.target_tokens,
        draft_tokens: stats.draft_tokens,
        accepted_tokens: stats.accepted_tokens,
        accept_rate: stats.accept_rate(),
        rounds: stats.rounds,
        accept_lens: stats.accept_lens.clone(),
        emitted_tokens: stats.emitted_tokens,
        optimistic_draft_tokens: stats.optimistic_draft_tokens,
        reused_optimistic_tokens: stats.reused_optimistic_tokens,
        discarded_optimistic_tokens: stats.discarded_optimistic_tokens,
        adaptive_lookahead_disabled: stats.adaptive_lookahead_disabled,
        optimistic_draft_seconds: stats.optimistic_draft_time.as_secs_f64(),
        verification_in_flight_seconds: stats.verification_in_flight_time.as_secs_f64(),
    }
}

fn observed_capability(value: &CapabilityValue<u64>) -> Observed<u64> {
    match value {
        CapabilityValue::Available {
            value,
            kind,
            source,
        } => Observed::Available {
            value: *value,
            kind: match kind {
                MeasurementKind::Exact => ObservationKind::Exact,
                MeasurementKind::Conservative => ObservationKind::Conservative,
                MeasurementKind::Observational => ObservationKind::Observational,
                MeasurementKind::Estimated => ObservationKind::Estimated,
            },
            source: (*source).into(),
        },
        CapabilityValue::Unsupported { reason } => Observed::unsupported(reason.clone()),
        CapabilityValue::Unavailable { reason } => Observed::unavailable(reason.clone()),
    }
}

fn memory_semantics(value: PhysicalMemorySemantics) -> HardwareMemorySemantics {
    match value {
        PhysicalMemorySemantics::Unified => HardwareMemorySemantics::Unified,
        PhysicalMemorySemantics::SeparateTiers => HardwareMemorySemantics::SeparateTiers,
        PhysicalMemorySemantics::Unknown => HardwareMemorySemantics::Unknown,
    }
}

fn mlx_device(device: &DevicePlan) -> Result<Device, AutomaticPlanningError> {
    if device.backend.as_str() != "mlx" {
        return Err(AutomaticPlanningError::Invalid(format!(
            "MLX cannot probe backend {}",
            device.backend
        )));
    }
    let (family, index) = device.device.split_once(':').ok_or_else(|| {
        AutomaticPlanningError::Invalid(format!(
            "MLX device identifier {:?} must be family:index",
            device.device
        ))
    })?;
    let index = i32::try_from(index.parse::<usize>().map_err(|error| {
        AutomaticPlanningError::Invalid(format!("invalid MLX device index: {error}"))
    })?)
    .map_err(|_| AutomaticPlanningError::Invalid("MLX device index exceeds i32".into()))?;
    let kind = match family {
        "cpu" => DeviceType::Cpu,
        "metal" | "cuda" | "gpu" => DeviceType::Gpu,
        other => {
            return Err(AutomaticPlanningError::Invalid(format!(
                "unknown MLX device family {other:?}"
            )))
        }
    };
    Ok(Device::new(kind, index))
}

fn embedded_mtp_count(value: &serde_json::Value) -> Option<u64> {
    match value {
        serde_json::Value::Object(object) => {
            for key in ["mtp_num_hidden_layers", "num_nextn_predict_layers"] {
                if let Some(count) = object.get(key).and_then(serde_json::Value::as_u64) {
                    return Some(count);
                }
            }
            object.values().find_map(embedded_mtp_count)
        }
        serde_json::Value::Array(values) => values.iter().find_map(embedded_mtp_count),
        _ => None,
    }
}

fn transfer_direction_name(direction: TransferDirection) -> &'static str {
    match direction {
        TransferDirection::DeviceToHost => "device_to_host",
        TransferDirection::DeviceToDisk => "device_to_disk",
        TransferDirection::HostToDevice => "host_to_device",
        TransferDirection::HostToDisk => "host_to_disk",
        TransferDirection::DiskToDevice => "disk_to_device",
        TransferDirection::DiskToHost => "disk_to_host",
    }
}

fn planning_backend_error(
    operation: &'static str,
    error: impl std::fmt::Display,
) -> AutomaticPlanningError {
    AutomaticPlanningError::Backend {
        operation,
        message: error.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mlx_discovery_always_reports_cpu() {
        let profile = discover_hardware();
        assert!(profile.backends.iter().any(|backend| {
            backend.backend.as_str() == "mlx"
                && backend.available
                && backend.devices.iter().any(|device| device.id == "cpu:0")
        }));
    }

    #[test]
    fn plan_realization_rejects_distributed_topology() {
        let mut plan = ExecutionPlan::fully_resident(DevicePlan::new("mlx", "cpu:0").unwrap());
        plan.topology = safemlx_lm_core::ParallelTopology::new(2, 1, 1, 1).unwrap();
        assert!(execution_plan_load_options(&plan).is_err());
    }
}
