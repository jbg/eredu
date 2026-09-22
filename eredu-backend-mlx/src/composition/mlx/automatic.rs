//! MLX observations and plan realization for neutral automatic planning.

use std::path::Path;

use eredu_core::{
    AutomaticPlanningBackend, AutomaticPlanningError, BackendId, BoundedResidencyRequirement,
    CandidateAdmission, DevicePlan, DraftPlacementPlan, DraftingPlan, ExecutionPlan,
    ExecutionPlanBackendFactory, ExecutionPlanTarget, ExecutionPlanTargetSelection,
    ExpertCacheTelemetry, ExternalDraftArtifact, HardwareBackendProfile, HardwareDeviceProfile,
    HardwareMemorySemantics, HardwareProfile, ModelResourceProfile, ModelRuntime, Observed,
    RealizedDrafting, ResidencyPlan, SelectedExecutionPlanTarget, SpeculativeDraftSource,
    SpeculativeGenerationBackend,
};
use safemlx::{Device, DeviceType, Stream};

use super::{
    capability::available_memory, inspection::MlxInspectionOptions,
    realtime::MlxRealtimeExecutionContext, speculative::MlxDrafter, MlxBackend, MlxLoadRequest,
};
use crate::{
    backend::runtime::residency::parameter_bank::ParameterBanksResidencyReport,
    backend::{error::Error, MlxAcceleratorFamily, MlxDeviceIdentity},
};
use eredu_runtime::selected_text_bounded_requirement;

mod state;
pub use state::MlxStateBackendFactory;

/// MLX automatic-planning adapter and whole-session backend factory.
#[derive(Debug, Clone, Copy, Default)]
pub struct MlxBackendFactory {
    sample_mlx_memory: bool,
    sample_process_memory: bool,
}

impl MlxBackendFactory {
    /// Selects mutable-cache residency before ordinary inspection and native
    /// realization. The configured factory reuses the same prepared resources.
    pub fn with_state_residency(
        self,
        policy: eredu_runtime::CacheResidencyPolicy,
    ) -> MlxStateBackendFactory {
        MlxStateBackendFactory::new(self, policy)
    }

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

    /// Translates a portable execution plan into its backend load request.
    ///
    /// This performs no native device or stream realization and is suitable
    /// for inspection and admission before an executable target is created.
    pub fn load_request_for_plan(
        &self,
        plan: &ExecutionPlan,
    ) -> Result<MlxLoadRequest, AutomaticPlanningError> {
        eredu_runtime::NormalizedLoadRequest::from_execution_plan(
            plan,
            eredu_runtime::ResidencyDiagnostics::new(
                self.sample_mlx_memory,
                self.sample_process_memory,
            ),
            None,
        )
        .map(MlxLoadRequest::from_normalized)
        .map_err(|error| AutomaticPlanningError::Invalid(error.to_string()))
    }
}

/// Selects and materializes one single-device realtime execution route.
///
/// Architecture inspection and exact capability selection complete before
/// this function realizes a native device or creates execution and weight
/// streams. The returned context exposes mechanism operations only; the model
/// remains in its architecture-owned selected wrapper.
pub fn create_realtime_execution(
    preparation: eredu_architectures::moshi::RealtimePreparationPlan,
    device: &DevicePlan,
    options: MlxLoadRequest,
) -> Result<
    (
        MlxRealtimeExecutionContext,
        eredu_architectures::moshi::MoshiRealtimeExecution<
            crate::composition::moshi::MlxRealtimeExecution,
        >,
    ),
    Error,
> {
    let selected =
        MlxRealtimeExecutionContext::select_realtime_execution(preparation, &options, false)?;
    #[cfg(test)]
    crate::tests::support::path_instrumentation::target_native_resource_realization_attempt();
    let backend =
        realize_backend(device).map_err(|error| Error::AutomaticPlanning(error.to_string()))?;
    let context = MlxRealtimeExecutionContext::from_backend(backend);
    let execution = context.materialize_realtime_execution(selected)?;
    Ok((context, execution))
}

/// Discovers hardware facts visible to the MLX adapter.
pub fn discover_hardware() -> HardwareProfile {
    let (physical_memory_bytes, available_memory_bytes, semantics) = match available_memory() {
        Ok(memory) => (
            memory.physical_memory_bytes,
            memory.available_memory_bytes,
            memory.physical_semantics.into(),
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

    HardwareProfile::observe_host(
        physical_memory_bytes,
        available_memory_bytes,
        semantics,
        vec![HardwareBackendProfile {
            backend: BackendId::new("mlx").expect("MLX is a valid backend identifier"),
            available: true,
            detail: (!details.is_empty()).then(|| details.join("; ")),
            devices,
        }],
    )
}

impl AutomaticPlanningBackend for MlxBackendFactory {
    type Inspection = eredu_core::ArtifactInspection<
        eredu_architectures::processor_plan::ArtifactArchitecturePlan,
    >;

    fn backend_id(&self) -> BackendId {
        BackendId::new("mlx").expect("MLX is a valid backend identifier")
    }

    fn discover_hardware(&self) -> Result<HardwareProfile, AutomaticPlanningError> {
        Ok(discover_hardware())
    }

    fn inspect_resources(
        &self,
        model_path: &Path,
    ) -> Result<(ModelResourceProfile, Self::Inspection), AutomaticPlanningError> {
        state::inspect_resources(model_path, MlxInspectionOptions::default())
    }

    fn admit_candidate(
        &self,
        inspection: &Self::Inspection,
        plan: &ExecutionPlan,
    ) -> Result<CandidateAdmission, AutomaticPlanningError> {
        state::admit_candidate(inspection, self.load_request_for_plan(plan)?)
    }

    fn bounded_residency_requirement(
        &self,
        inspection: &Self::Inspection,
        plan: &ExecutionPlan,
    ) -> Result<BoundedResidencyRequirement, AutomaticPlanningError> {
        state::bounded_residency_requirement(inspection, plan, || self.load_request_for_plan(plan))
    }
}

impl ExecutionPlanBackendFactory for MlxBackendFactory {
    type Backend = MlxBackend<'static>;
    type DrafterPreparation = eredu_architectures::ExternalDraftPreparation;
    type SelectedDrafterPreparation = eredu_architectures::PreparedExternalDraft;
    type Drafter = MlxDrafter;

    fn inspect_loading_artifact<R: eredu_core::ModelConfigurationResolver>(
        &self,
        path: &Path,
        resolver: &R,
    ) -> Result<eredu_core::ArtifactInspection<R::ArtifactPlan>, AutomaticPlanningError>
    where
        R::ArtifactPlan: Send + Sync + 'static,
    {
        crate::backend::managed_memory::ledger()
            .inspect_artifact_for_loading(
                path,
                resolver,
                eredu_checkpoint::safetensors::SafetensorsDiscoveryLimits::default(),
                eredu_runtime::working_memory::DependencyMemoryPolicy::default(),
            )
            .map_err(|error| AutomaticPlanningError::backend("inspect_loading_artifact", error))
    }

    fn select_target(
        &self,
        inspection: &eredu_core::ArtifactInspection<
            eredu_architectures::processor_plan::ArtifactArchitecturePlan,
        >,
        plan: &ExecutionPlan,
    ) -> Result<ExecutionPlanTargetSelection<Self::Backend>, AutomaticPlanningError> {
        state::select_target(inspection, self.load_request_for_plan(plan)?)
    }

    fn realize_target(
        &self,
        selected: SelectedExecutionPlanTarget<Self::Backend>,
    ) -> Result<ExecutionPlanTarget<Self::Backend>, AutomaticPlanningError> {
        #[cfg(test)]
        crate::tests::support::path_instrumentation::target_native_resource_realization_attempt();
        let backend = realize_backend(selected.execution_plan().device())?;
        Ok(ExecutionPlanTarget::new(backend, selected))
    }

    fn select_drafting(
        &self,
        plan: &ExecutionPlan,
        target: &SelectedExecutionPlanTarget<Self::Backend>,
        external_artifact: Option<ExternalDraftArtifact<Self::DrafterPreparation>>,
    ) -> Result<
        Option<ExternalDraftArtifact<Self::SelectedDrafterPreparation>>,
        AutomaticPlanningError,
    > {
        let Some(artifact) = external_artifact else {
            return Ok(None);
        };
        eredu_architectures::prepare_execution_plan_draft(
            plan,
            target.inspection(),
            artifact,
            &super::loading::MlxPreparationMechanisms::new(
                &super::replicated_text::GROUPED_OPERATION_CAPABILITIES,
            ),
            |descriptor, transforms| {
                if transforms && super::replicated_text::supports_transform(descriptor) {
                    Some(eredu_runtime::WeightLoweringKind::Transform)
                } else if !transforms && super::replicated_text::supports_direct(descriptor) {
                    Some(eredu_runtime::WeightLoweringKind::Direct)
                } else {
                    None
                }
            },
        )
        .map(Some)
    }

    fn realize_drafting(
        &self,
        plan: &ExecutionPlan,
        target: &ModelRuntime<Self::Backend>,
        selected: eredu_core::SelectedExecutionPlanDrafting<Self::SelectedDrafterPreparation>,
    ) -> Result<RealizedDrafting<MlxDrafter>, AutomaticPlanningError> {
        let external_artifact = selected.into_external_artifact(plan, target)?;
        match plan.drafting() {
            DraftingPlan::Disabled => Ok(RealizedDrafting::Disabled),
            DraftingPlan::Embedded { .. } => {
                let capability =
                    <MlxBackend<'static> as SpeculativeGenerationBackend>::speculative_capability(
                        target,
                    );
                if !capability.is_ready_for(SpeculativeDraftSource::Embedded) {
                    return Err(AutomaticPlanningError::Invalid(format!(
                        "execution plan selects embedded drafting but target capability is {capability:?}"
                    )));
                }
                Ok(RealizedDrafting::Embedded)
            }
            DraftingPlan::External { placement, .. } => {
                let artifact = external_artifact.ok_or_else(|| {
                    AutomaticPlanningError::Invalid(
                        "external drafting is missing proven tokenizer compatibility".into(),
                    )
                })?;
                let drafter = match placement {
                    DraftPlacementPlan::Target => MlxDrafter::materialize_with_source_pool(
                        artifact.preparation,
                        target.backend().memory_ledger(),
                        target.backend().stream(),
                        target.backend().weights_stream(),
                    ),
                    DraftPlacementPlan::Device { device } => MlxDrafter::materialize_with_backend(
                        artifact.preparation,
                        realize_backend(device)?,
                    ),
                    _ => {
                        return Err(AutomaticPlanningError::Invalid(
                            "unsupported draft placement".into(),
                        ));
                    }
                }
                .map_err(|error| planning_backend_error("realize_external_drafter", error))?;
                drafter.stream().synchronize().map_err(|error| {
                    planning_backend_error("complete_external_drafter_load", error)
                })?;
                Ok(RealizedDrafting::External(drafter))
            }
            _ => Err(AutomaticPlanningError::Invalid(
                "unsupported speculative drafting plan".into(),
            )),
        }
    }
}

// Both selected placements use these exact native constructors. Ordinary
// fallback is preserved where no admitted stream layout exists; it supplies
// no original execution/source authority to later request admission.
fn realize_backend(device: &DevicePlan) -> Result<MlxBackend<'static>, AutomaticPlanningError> {
    let realized = mlx_device(device)?;
    // The factory owns these native constructors. Exact prepared wrappers
    // and source accounts remain in MlxBackend; no ordinary clone is needed.
    let kind = realized
        .device
        .get_type()
        .map_err(|error| planning_backend_error("execution_device_type", error))?;
    let index = realized
        .device
        .get_index()
        .map_err(|error| planning_backend_error("execution_device_index", error))?;
    if index == 0 {
        let pool = crate::backend::managed_memory::ledger();
        let streams = crate::backend::managed_memory::gpu_stream::PreparedExecutionStreams::for_device_factory(&pool, kind)
            .map_err(|error| planning_backend_error("create_admitted_execution_stream", error.into_backend_failure()))?;
        if let Some(streams) = streams {
            return Ok(MlxBackend::for_prepared_execution_plan(
                streams,
                realized.identity,
            ));
        }
    }
    let stream = Stream::try_new_with_device(&realized.device)
        .map_err(|error| planning_backend_error("create_execution_stream", error))?;
    let weights_stream = Stream::try_new_with_device(&Device::new(DeviceType::Cpu, 0))
        .map_err(|error| planning_backend_error("create_weights_stream", error))?;
    Ok(MlxBackend::for_execution_plan(
        &stream,
        &weights_stream,
        realized.identity,
    ))
}

/// Converts an MLX routed-expert cache snapshot into neutral telemetry.
pub fn parameter_bank_telemetry(report: &ParameterBanksResidencyReport) -> ExpertCacheTelemetry {
    ExpertCacheTelemetry {
        owned_experts: report.owned_entries(),
        owned_bytes: report.owned_bytes(),
        host_resident_experts: report.host_resident_entries(),
        device_resident_experts: report.device_resident_entries(),
        host_resident_bytes: report.host_resident_bytes(),
        device_resident_bytes: report.device_resident_bytes(),
        peak_host_resident_bytes: report.peak_host_resident_bytes(),
        peak_device_resident_bytes: report.peak_device_resident_bytes(),
    }
}

#[derive(Debug)]
struct RealizedMlxDevice {
    device: Device,
    identity: MlxDeviceIdentity,
}

fn mlx_device(device: &DevicePlan) -> Result<RealizedMlxDevice, AutomaticPlanningError> {
    if device.backend().as_str() != "mlx" {
        return Err(AutomaticPlanningError::Invalid(format!(
            "MLX cannot probe backend {}",
            device.backend()
        )));
    }
    let (family, index) = device.device().split_once(':').ok_or_else(|| {
        AutomaticPlanningError::Invalid(format!(
            "MLX device identifier {:?} must be family:index",
            device.device()
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
            )));
        }
    };
    let canonical_id = format!("{family}:{index}");
    if device.device() != canonical_id {
        return Err(AutomaticPlanningError::Invalid(format!(
            "MLX device identifier {:?} is not canonical; use {canonical_id:?}",
            device.device()
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

fn planning_backend_error(
    operation: &'static str,
    error: impl std::error::Error + Send + Sync + 'static,
) -> AutomaticPlanningError {
    AutomaticPlanningError::backend(operation, error)
}

#[cfg(test)]
mod tests;
