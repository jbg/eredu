//! MLX observations and plan realization for neutral automatic planning.

use std::path::Path;

use eredu_core::scheduler::SemanticStateTransaction;
use eredu_core::{
    AutomaticPlanningBackend, AutomaticPlanningError, BackendId, BoundedResidencyRequirement,
    CandidateAdmission, DevicePlan, DraftPlacementPlan, DraftingPlan, DurationSeconds,
    ExecutionPlan, ExecutionPlanBackendFactory, ExecutionPlanTarget, ExpertCacheTelemetry,
    ExternalDraftArtifact, HardwareBackendProfile, HardwareDeviceProfile, HardwareMemorySemantics,
    HardwareProfile, InspectionSeverity, ModelResourceProfile, ModelRuntime, MtpCapability,
    MtpCheckpointKind, MtpStats, MtpTelemetry, Observed, PhysicalMemorySemantics,
    QuantizationRequest, RealizedDrafting, RealtimeBackend, RealtimeInputFrame,
    RealtimeModelLoadingBackend, RealtimeOutputFrame, RealtimeSampling, RealtimeSpeechConfig,
    ResidencyPlan, ResidencyTelemetry, SpeculativeGenerationBackend, Submission, TransferTelemetry,
    WeightTransformationPlan, AUTOMATIC_SCHEMA_VERSION,
};
use safemlx::{Device, DeviceType, Stream};

use super::{
    capability::available_memory,
    inspection::{inspect_model, MlxInspectionOptions},
    realtime::MlxRealtimeBackend,
    speculative::MlxDrafter,
    MlxBackend, ModelLoadOptions,
};
use crate::{
    backend::runtime::{
        execution::layerwise::LayerwiseModelError, residency::expert_cache::ExpertCacheReport,
    },
    backend::{error::Error, MlxAcceleratorFamily, MlxDeviceIdentity},
};
use eredu_core::residency::{MemoryTier, OffloadConfig, TransferDirection};
use eredu_runtime::{
    DenseDiskStreamLoadOptions, ExpertCacheLoadOptions, LayerwiseLoadOptions,
    NonExpertWeightResidency, ResidencyReport, WeightResidency,
};

/// MLX automatic-planning adapter and whole-session backend factory.
#[derive(Debug, Clone, Copy, Default)]
pub struct MlxBackendFactory {
    sample_mlx_memory: bool,
    sample_process_memory: bool,
}

impl MlxBackendFactory {
    /// Enables backend allocator and process-memory sampling for bounded residency.
    pub const fn with_residency_diagnostics(
        mut self,
        sample_mlx_memory: bool,
        sample_process_memory: bool,
    ) -> Self {
        self.sample_mlx_memory = sample_mlx_memory;
        self.sample_process_memory = sample_process_memory;
        self
    }
}

/// Realtime MLX adapter without native device or stream accessors.
///
/// The facade stores this concrete adapter while applications operate it only
/// through facade-owned model and scheduler wrappers. Backend-author tools use
/// [`crate::native::MlxRealtimeBackend`] when they need explicit native handles.
#[derive(Clone)]
pub struct MlxRealtimeAdapter {
    backend: MlxRealtimeBackend,
}

impl RealtimeModelLoadingBackend for MlxRealtimeAdapter {
    type Preparation = eredu_architectures::moshi::RealtimePreparationPlan;
    type LoadOptions = ModelLoadOptions;

    fn materialize_realtime_model(
        &self,
        preparation: Self::Preparation,
        options: Self::LoadOptions,
    ) -> Result<Self::Model, Self::Error> {
        self.backend
            .materialize_realtime_model(preparation, options)
    }
}

impl RealtimeBackend for MlxRealtimeAdapter {
    type Model = <MlxRealtimeBackend as RealtimeBackend>::Model;
    type ModelIdentity = <MlxRealtimeBackend as RealtimeBackend>::ModelIdentity;
    type Input = <MlxRealtimeBackend as RealtimeBackend>::Input;
    type Output = <MlxRealtimeBackend as RealtimeBackend>::Output;
    type Session = <MlxRealtimeBackend as RealtimeBackend>::Session;
    type Completion = <MlxRealtimeBackend as RealtimeBackend>::Completion;
    type Error = Error;

    fn name(&self) -> &str {
        self.backend.name()
    }

    fn model_identity(&self, model: &Self::Model) -> Self::ModelIdentity {
        self.backend.model_identity(model)
    }

    fn session_capabilities(&self, model: &Self::Model) -> eredu_core::SessionCapabilities {
        self.backend.session_capabilities(model)
    }

    fn model_identity_mismatch(
        &self,
        expected: &Self::ModelIdentity,
        actual: &Self::ModelIdentity,
    ) -> Option<String> {
        self.backend.model_identity_mismatch(expected, actual)
    }

    fn speech_config(&self, model: &Self::Model) -> RealtimeSpeechConfig {
        self.backend.speech_config(model)
    }

    fn materialize_input(
        &self,
        model: &Self::Model,
        frame: &RealtimeInputFrame,
    ) -> Result<Self::Input, Self::Error> {
        self.backend.materialize_input(model, frame)
    }

    fn observe_output(&self, output: &Self::Output) -> Result<RealtimeOutputFrame, Self::Error> {
        self.backend.observe_output(output)
    }

    fn create_session(
        &self,
        model: &Self::Model,
        sampling: RealtimeSampling,
    ) -> Result<Self::Session, Self::Error> {
        self.backend.create_session(model, sampling)
    }

    fn validate_session(
        &self,
        model: &Self::Model,
        session: &Self::Session,
    ) -> Result<(), Self::Error> {
        self.backend.validate_session(model, session)
    }

    fn validate_input(&self, model: &Self::Model, input: &Self::Input) -> Result<(), Self::Error> {
        self.backend.validate_input(model, input)
    }

    fn input_batch_size(&self, input: &Self::Input) -> usize {
        self.backend.input_batch_size(input)
    }

    fn set_sampling(
        &self,
        session: &mut Self::Session,
        sampling: RealtimeSampling,
    ) -> Result<(), Self::Error> {
        self.backend.set_sampling(session, sampling)
    }

    fn submit_step(
        &self,
        model: &mut Self::Model,
        session: &mut <Self::Session as SemanticStateTransaction>::Branch,
        input: &Self::Input,
    ) -> Result<Submission<Self::Output, Self::Completion>, Self::Error> {
        self.backend.submit_step(model, session, input)
    }

    fn retained_resources(&self, completion: &Self::Completion) -> usize {
        self.backend.retained_resources(completion)
    }
}

/// Creates a single-device realtime adapter from a portable device plan.
///
/// Application facades use this factory boundary to keep native MLX streams
/// out of their public API. Backend-author code that needs explicit streams or
/// collective groups constructs [`crate::native::MlxRealtimeBackend`] directly instead.
pub fn create_realtime_backend(device: &DevicePlan) -> Result<MlxRealtimeAdapter, Error> {
    let realized =
        mlx_device(device).map_err(|error| Error::AutomaticPlanning(error.to_string()))?;
    let stream = Stream::new_with_device(&realized.device);
    let weights_stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    Ok(MlxRealtimeAdapter {
        backend: MlxRealtimeBackend::new(&stream, &weights_stream),
    })
}

/// Discovers hardware facts visible to the MLX adapter.
pub fn discover_hardware() -> HardwareProfile {
    let logical_cpu_count = std::thread::available_parallelism().map_or_else(
        |error| Observed::unavailable(error.to_string()),
        |count| Observed::exact(count.get() as u64, "std::thread::available_parallelism"),
    );
    let (physical_memory_bytes, available_memory_bytes, semantics) = match available_memory() {
        Ok(memory) => (
            memory.physical_memory_bytes,
            memory.available_memory_bytes,
            memory_semantics(memory.physical_semantics),
        ),
        Err(error) => (
            Observed::unavailable(error.to_string()),
            Observed::unavailable(error.to_string()),
            HardwareMemorySemantics::Unknown,
        ),
    };
    #[allow(unused_mut)] // Native-device probes are target and feature gated.
    let mut devices = vec![HardwareDeviceProfile {
        id: "cpu:0".into(),
        family: "cpu".into(),
        index: 0,
        total_memory_bytes: physical_memory_bytes.clone(),
        available_memory_bytes: available_memory_bytes.clone(),
    }];
    #[allow(unused_mut)] // Native-device probe diagnostics are target and feature gated.
    let mut details: Vec<String> = Vec::new();

    #[cfg(all(target_os = "macos", feature = "metal"))]
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

fn mlx_load_options(
    factory: &MlxBackendFactory,
    plan: &ExecutionPlan,
) -> Result<ModelLoadOptions, Error> {
    if plan.topology.world_size() != 1 {
        return Err(Error::AutomaticPlanning(
            "single-device automatic plans require a 1x1x1 parallel topology".into(),
        ));
    }
    let mut load = match plan.weight_transformation {
        WeightTransformationPlan::PreserveCheckpoint => ModelLoadOptions::default(),
        WeightTransformationPlan::Affine { bits, group_size } => {
            ModelLoadOptions::with_quantization(QuantizationRequest::Affine {
                group_size: u32::try_from(group_size).map_err(|_| {
                    Error::Quantization(format!(
                        "group_size must be non-negative, got {group_size}"
                    ))
                })?,
                bits: u8::try_from(bits)
                    .map_err(|_| Error::Quantization(format!("bits must fit in u8, got {bits}")))?,
            })
        }
        WeightTransformationPlan::MxFp4 => {
            ModelLoadOptions::with_quantization(QuantizationRequest::MxFp4)
        }
    };
    load.required_session_capabilities = plan.required_session_capabilities;
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
            sample_backend_memory: factory.sample_mlx_memory,
            sample_process_memory: factory.sample_process_memory,
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
            options.sample_backend_memory = factory.sample_mlx_memory;
            options.sample_process_memory = factory.sample_process_memory;
            NonExpertWeightResidency::DenseDiskStream(options)
        }
    };
    let residency = if let Some(expert) = &plan.expert_cache {
        WeightResidency::with_expert_cache(
            residency,
            ExpertCacheLoadOptions::new(
                OffloadConfig::new(expert.device_budget_bytes, expert.host_budget_bytes, 1)?
                    .with_eviction_policy(expert.eviction_policy),
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

fn mlx_drafter_load_options(plan: &ExecutionPlan) -> Result<ModelLoadOptions, Error> {
    match plan.weight_transformation {
        WeightTransformationPlan::PreserveCheckpoint => Ok(ModelLoadOptions::default()),
        WeightTransformationPlan::Affine { bits, group_size } => Ok(
            ModelLoadOptions::with_quantization(QuantizationRequest::Affine {
                group_size: u32::try_from(group_size).map_err(|_| {
                    Error::Quantization(format!(
                        "group_size must be non-negative, got {group_size}"
                    ))
                })?,
                bits: u8::try_from(bits)
                    .map_err(|_| Error::Quantization(format!("bits must fit in u8, got {bits}")))?,
            }),
        ),
        WeightTransformationPlan::MxFp4 => Ok(ModelLoadOptions::with_quantization(
            QuantizationRequest::MxFp4,
        )),
    }
}

impl AutomaticPlanningBackend for MlxBackendFactory {
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
        inspect_model(model_path, MlxInspectionOptions::default())
            .map(|report| report.resources)
            .map_err(|error| planning_backend_error("inspect_resources", error))
    }

    fn admit_candidate(
        &self,
        model_path: &Path,
        plan: &ExecutionPlan,
    ) -> Result<CandidateAdmission, AutomaticPlanningError> {
        let load = mlx_load_options(self, plan)
            .map_err(|error| planning_backend_error("realize_plan", error))?;
        let report = inspect_model(model_path, MlxInspectionOptions { load })
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
        let realized = mlx_device(&probe.device)?;
        let stream = Stream::new_with_device(&realized.device);
        let weights_stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
        let backend = MlxBackend::for_execution_plan(&stream, &weights_stream, realized.identity);
        let options = mlx_load_options(self, &probe)
            .map_err(|error| planning_backend_error("realize_probe", error))?;
        match eredu_core::load_model(&backend, model_path, options) {
            Err(eredu_core::ModelLoadError::Backend(Error::LayerwiseModel(
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
}

impl ExecutionPlanBackendFactory for MlxBackendFactory {
    type Backend = MlxBackend<'static>;
    type DrafterPreparation = eredu_architectures::ExternalAssistantPreparationPlan;
    type Drafter = MlxDrafter;

    fn realize_target(
        &self,
        plan: &ExecutionPlan,
    ) -> Result<ExecutionPlanTarget<Self::Backend>, AutomaticPlanningError> {
        let realized = mlx_device(&plan.device)?;
        let stream = Stream::new_with_device(&realized.device);
        let weights_stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
        let options = mlx_load_options(self, plan)
            .map_err(|error| planning_backend_error("realize_execution_plan_target", error))?;
        Ok(ExecutionPlanTarget::new(
            MlxBackend::for_execution_plan(&stream, &weights_stream, realized.identity),
            options,
        ))
    }

    fn realize_drafting(
        &self,
        plan: &ExecutionPlan,
        target: &ModelRuntime<Self::Backend>,
        external_artifact: Option<ExternalDraftArtifact<Self::DrafterPreparation>>,
    ) -> Result<RealizedDrafting<MlxDrafter>, AutomaticPlanningError> {
        let capability =
            <MlxBackend<'static> as SpeculativeGenerationBackend>::mtp_capability(target);
        match &plan.drafting {
            DraftingPlan::Disabled => Ok(RealizedDrafting::Disabled),
            DraftingPlan::Embedded { .. } => {
                if capability
                    != (MtpCapability::Ready {
                        checkpoint: MtpCheckpointKind::Embedded,
                    })
                {
                    return Err(AutomaticPlanningError::Invalid(format!(
                        "execution plan selects embedded drafting but target capability is {capability:?}"
                    )));
                }
                Ok(RealizedDrafting::Embedded)
            }
            DraftingPlan::External { placement, .. } => {
                if capability
                    != (MtpCapability::Ready {
                        checkpoint: MtpCheckpointKind::Separate,
                    })
                {
                    return Err(AutomaticPlanningError::Invalid(format!(
                        "execution plan selects external drafting but target capability is {capability:?}"
                    )));
                }
                let artifact = external_artifact.ok_or_else(|| {
                    AutomaticPlanningError::Invalid(
                        "external drafting is missing proven tokenizer compatibility".into(),
                    )
                })?;
                let draft_stream = match placement {
                    DraftPlacementPlan::Target => target.backend().stream().clone(),
                    DraftPlacementPlan::Device { device } => {
                        let realized = mlx_device(device)?;
                        Stream::new_with_device(&realized.device)
                    }
                };
                let options = mlx_drafter_load_options(plan)
                    .map_err(|error| planning_backend_error("realize_external_drafter", error))?;
                let drafter = MlxDrafter::materialize_with_compatibility(
                    artifact.preparation,
                    artifact.tokenizer_compatibility,
                    options,
                    &draft_stream,
                    target.backend().weights_stream(),
                )
                .map_err(|error| planning_backend_error("realize_external_drafter", error))?;
                target
                    .session()
                    .validate_external_drafter(&drafter)
                    .map_err(|error| planning_backend_error("validate_external_drafter", error))?;
                draft_stream.synchronize().map_err(|error| {
                    planning_backend_error("complete_external_drafter_load", error)
                })?;
                Ok(RealizedDrafting::External(drafter))
            }
        }
    }
}

/// Converts an MLX residency snapshot into neutral telemetry.
pub fn residency_telemetry(report: &ResidencyReport) -> ResidencyTelemetry {
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
    ResidencyTelemetry {
        planned_disk_bytes: planned.get(MemoryTier::Disk),
        planned_host_bytes: planned.get(MemoryTier::Host),
        planned_device_bytes: planned.get(MemoryTier::Device),
        current_host_bytes: current.get(MemoryTier::Host),
        current_device_bytes: current.get(MemoryTier::Device),
        peak_host_bytes: peak.get(MemoryTier::Host),
        peak_device_bytes: peak.get(MemoryTier::Device),
        transfers,
    }
}

/// Converts an MLX routed-expert cache snapshot into neutral telemetry.
pub fn expert_cache_telemetry(report: &ExpertCacheReport) -> ExpertCacheTelemetry {
    ExpertCacheTelemetry {
        owned_experts: report.owned_experts,
        owned_bytes: report.owned_bytes,
        host_resident_experts: report.host_resident_experts,
        device_resident_experts: report.device_resident_experts,
        host_resident_bytes: report.host_resident_bytes,
        device_resident_bytes: report.device_resident_bytes,
        peak_host_resident_bytes: report.peak_host_resident_bytes,
        peak_device_resident_bytes: report.peak_device_resident_bytes,
    }
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

fn memory_semantics(value: PhysicalMemorySemantics) -> HardwareMemorySemantics {
    match value {
        PhysicalMemorySemantics::Unified => HardwareMemorySemantics::Unified,
        PhysicalMemorySemantics::SeparateTiers => HardwareMemorySemantics::SeparateTiers,
        PhysicalMemorySemantics::Unknown => HardwareMemorySemantics::Unknown,
    }
}

#[derive(Debug)]
struct RealizedMlxDevice {
    device: Device,
    identity: MlxDeviceIdentity,
}

fn mlx_device(device: &DevicePlan) -> Result<RealizedMlxDevice, AutomaticPlanningError> {
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
    let accelerator_family = match family {
        "cpu" => None,
        "metal" => Some(MlxAcceleratorFamily::Metal),
        "cuda" => Some(MlxAcceleratorFamily::Cuda),
        other => {
            return Err(AutomaticPlanningError::Invalid(format!(
                "unknown MLX device family {other:?}"
            )))
        }
    };
    let canonical_id = format!("{family}:{index}");
    if device.device != canonical_id {
        return Err(AutomaticPlanningError::Invalid(format!(
            "MLX device identifier {:?} is not canonical; use {canonical_id:?}",
            device.device
        )));
    }
    if let Some(family) = accelerator_family {
        if !family.is_compiled() {
            return Err(AutomaticPlanningError::Invalid(format!(
                "MLX {} device family is not compiled for this target",
                family.as_str()
            )));
        }
        let available = family.is_available().map_err(|error| {
            planning_backend_error("discover_accelerator_family_availability", error)
        })?;
        if !available {
            return Err(AutomaticPlanningError::Invalid(format!(
                "MLX {} device family is not available on the discovered hardware",
                family.as_str()
            )));
        }
    }
    let kind = if accelerator_family.is_some() {
        DeviceType::Gpu
    } else {
        DeviceType::Cpu
    };
    let device = Device::new(kind, index);
    let identity = MlxDeviceIdentity::from_realized_device(&device, accelerator_family)
        .map_err(|error| planning_backend_error("derive_realized_device_identity", error))?;
    Ok(RealizedMlxDevice { device, identity })
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
    use eredu_core::BackendProvider as _;

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
        plan.topology = eredu_core::ParallelTopology::new(2, 1, 1, 1).unwrap();
        assert!(mlx_load_options(&MlxBackendFactory::default(), &plan).is_err());
    }

    #[test]
    fn realized_cpu_identity_is_derived_from_the_native_device() {
        let plan = DevicePlan::new("mlx", "cpu:0").unwrap();
        let realized = mlx_device(&plan).unwrap();
        let stream = Stream::new_with_device(&realized.device);
        let backend = MlxBackend::for_execution_plan(&stream, &stream, realized.identity);

        let devices = backend.devices().unwrap();
        assert_eq!(devices[0].0.id, "cpu:0");
        assert_eq!(devices[0].0.family, "cpu");
    }

    #[test]
    fn realization_rejects_generic_gpu_family() {
        let plan = DevicePlan::new("mlx", "gpu:0").unwrap();
        let error = mlx_device(&plan).unwrap_err();
        assert!(error
            .to_string()
            .contains("unknown MLX device family \"gpu\""));
    }

    #[test]
    fn realization_rejects_noncanonical_device_identity() {
        let plan = DevicePlan::new("mlx", "cpu:00").unwrap();
        let error = mlx_device(&plan).unwrap_err();
        assert!(error.to_string().contains("is not canonical"));
    }

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn realization_rejects_cuda_without_compiled_support() {
        let plan = DevicePlan::new("mlx", "cuda:0").unwrap();
        let error = mlx_device(&plan).unwrap_err();
        assert!(error
            .to_string()
            .contains("cuda device family is not compiled"));
    }

    #[cfg(not(all(feature = "metal", target_vendor = "apple")))]
    #[test]
    fn realization_rejects_metal_without_compiled_support() {
        let plan = DevicePlan::new("mlx", "metal:0").unwrap();
        let error = mlx_device(&plan).unwrap_err();
        assert!(error
            .to_string()
            .contains("metal device family is not compiled"));
    }

    #[cfg(all(feature = "metal", target_vendor = "apple"))]
    #[test]
    fn realized_metal_plan_reports_metal_from_the_native_binding() {
        if !safemlx::metal::is_available().unwrap() {
            return;
        }
        let plan = ExecutionPlan::fully_resident(DevicePlan::new("mlx", "metal:0").unwrap());
        let target =
            eredu_core::realize_execution_plan_target(&MlxBackendFactory::default(), &plan)
                .unwrap();
        let devices = target.backend().devices().unwrap();

        assert_eq!(devices[0].0.id, "metal:0");
        assert_eq!(devices[0].0.family, "metal");
    }
}
