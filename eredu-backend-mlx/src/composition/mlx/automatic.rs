//! MLX observations and plan realization for neutral automatic planning.

use eredu_core::automatic::{ColdPlanPlacement, ExecutionPlanNativeFactory};
use eredu_core::{
    AutomaticPlanningError, BackendFailure, BackendId, DevicePlan, DraftPlacementPlan,
    HardwareBackendProfile, HardwareDeviceProfile, HardwareProfile, Observed,
};
use safemlx::{Device, DeviceType, Stream};

use super::{
    capability::available_memory,
    speculative::{MlxAssistantBinder, MlxDraftBinding},
    MlxBackend,
};
use crate::backend::{MlxAcceleratorFamily, MlxDeviceIdentity};

/// MLX automatic-planning adapter and whole-session backend factory.
#[derive(Debug, Clone, Copy, Default)]
pub struct MlxBackendFactory {
    sample_mlx_memory: bool,
    sample_process_memory: bool,
    world_backend: Option<safemlx::distributed::Backend>,
}

impl eredu_core::preparation::BackendInspection for MlxBackendFactory {
    fn device_uses_host_memory(&self, device: &DevicePlan) -> Result<bool, BackendFailure> {
        self.device_uses_host_memory(device)
            .map_err(BackendFailure::from_error)
    }

    fn reset_allocator_peak(&self) -> Result<bool, BackendFailure> {
        crate::reset_allocator_peak().map_err(|error| {
            BackendFailure::from_error(error).with_operation("allocator peak reset")
        })?;
        Ok(true)
    }

    fn proposed_allocator_cache_limit(&self, bytes: u64) -> Result<u64, BackendFailure> {
        native_cache_limit(bytes)?;
        Ok(bytes)
    }

    fn set_allocator_cache_limit(&self, bytes: u64) -> Result<(), BackendFailure> {
        crate::set_allocator_cache_limit(native_cache_limit(bytes)?)
            .map(|_| ())
            .map_err(|error| {
                BackendFailure::from_error(error).with_operation("allocator cache policy")
            })
    }

    fn allocator_observation(
        &self,
    ) -> Result<Option<eredu_core::AllocatorObservation>, BackendFailure> {
        Ok(Some(eredu_core::AllocatorObservation {
            cache_policy: Some(
                crate::allocator_cache_policy().map_err(BackendFailure::from_error)?,
            ),
            telemetry: allocator_telemetry()?,
        }))
    }
}

fn native_cache_limit(bytes: u64) -> Result<usize, BackendFailure> {
    usize::try_from(bytes).map_err(|error| {
        BackendFailure::new(eredu_core::BackendFailureKind::InvalidInput, error)
            .with_operation("allocator cache policy")
    })
}

fn allocator_telemetry() -> Result<eredu_core::AllocatorTelemetry, BackendFailure> {
    let memory = crate::allocator_memory()
        .map_err(|error| BackendFailure::from_error(error).with_operation("allocator telemetry"))?;
    Ok(memory.into())
}

impl eredu_core::decision::DecisionBindingFactory for MlxBackendFactory {
    type Binding = super::component_binding::MlxComponentBinder;

    fn decision_binding(&self, device: &DevicePlan) -> Result<Self::Binding, BackendFailure> {
        use crate::backend::ExecutionContext;

        let weights = ExecutionContext::cpu().map_err(BackendFailure::from_error)?;
        let execution = self.execution_context(device)?;
        Ok(Self::Binding::new(weights.stream(), execution.stream()))
    }
}

impl MlxBackendFactory {
    /// Realizes an execution context from a canonical MLX device plan.
    pub(crate) fn execution_context(
        &self,
        plan: &DevicePlan,
    ) -> Result<crate::native::ExecutionContext, eredu_core::BackendFailure> {
        let device = mlx_device(plan).map_err(eredu_core::BackendFailure::from_error)?;
        crate::native::ExecutionContext::try_new(device.device)
            .map_err(eredu_core::BackendFailure::from_error)
    }

    /// Classifies a canonical plan device without realizing native resources.
    /// Hardware discovery must separately confirm that the device is available.
    pub fn device_uses_host_memory(
        &self,
        device: &DevicePlan,
    ) -> Result<bool, AutomaticPlanningError> {
        let (family, _) = parse_mlx_device(device)?;
        Ok(family.is_none())
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

    /// Chooses the native world initializer used when a load supplies a rank.
    /// The world is initialized during realization using the launcher's environment.
    pub const fn with_world_backend(mut self, backend: safemlx::distributed::Backend) -> Self {
        self.world_backend = Some(backend);
        self
    }
}

/// Discovers hardware facts visible to the MLX adapter.
pub fn discover_hardware() -> HardwareProfile {
    let memory = available_memory().unwrap_or_else(|error| eredu_core::AvailableMemory {
        physical_memory_bytes: Observed::unavailable(error.to_string()),
        available_memory_bytes: Observed::unavailable(error.to_string()),
        device: None,
        physical_semantics: eredu_core::PhysicalMemorySemantics::Unknown,
    });
    let cpu_memory = memory.selected_capacity(true);
    #[allow(unused_mut)] // Native-device probes are target and feature gated.
    let mut devices = vec![HardwareDeviceProfile {
        id: "cpu:0".into(),
        kind: eredu_core::HardwareDeviceKind::Cpu,
        family: "cpu".into(),
        index: 0,
        total_memory_bytes: cpu_memory.total_bytes,
        available_memory_bytes: cpu_memory.available_bytes,
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
            let capacity = memory.selected_capacity(false);
            devices.push(HardwareDeviceProfile {
                id: "metal:0".into(),
                kind: eredu_core::HardwareDeviceKind::Gpu,
                family: "metal".into(),
                index: 0,
                total_memory_bytes: capacity.total_bytes,
                available_memory_bytes: capacity.available_bytes,
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
                kind: eredu_core::HardwareDeviceKind::Gpu,
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
        memory.physical_memory_bytes,
        memory.available_memory_bytes,
        memory.physical_semantics.into(),
        vec![HardwareBackendProfile {
            backend: BackendId::new("mlx").expect("MLX is a valid backend identifier"),
            available: true,
            detail: (!details.is_empty()).then(|| details.join("; ")),
            devices,
        }],
    )
}

impl eredu_core::BackendHardware for MlxBackendFactory {
    fn backend_id(&self) -> BackendId {
        BackendId::new("mlx").expect("MLX is a valid backend identifier")
    }

    fn discover_hardware(&self) -> Result<HardwareProfile, AutomaticPlanningError> {
        Ok(discover_hardware())
    }
}

impl eredu_core::preparation::PlanMechanismFacts for MlxBackendFactory {
    type Preparation = super::loading::MlxPreparationMechanisms;

    fn preparation_mechanisms(&self) -> Self::Preparation {
        super::loading::MlxPreparationMechanisms
    }

    fn residency_diagnostics(&self) -> eredu_core::preparation::ResidencyDiagnostics {
        eredu_core::preparation::ResidencyDiagnostics::new(
            self.sample_mlx_memory,
            self.sample_process_memory,
        )
    }

    fn media_features(&self) -> eredu_core::MediaFeatureAvailability {
        super::inspection::inspection_media_features()
    }
}

impl ColdPlanPlacement for MlxBackendFactory {
    type Placement = super::loading::MlxModelPlacement;

    fn plan_placement(
        &self,
        device: &DevicePlan,
        rank: Option<(usize, usize)>,
    ) -> Result<Self::Placement, AutomaticPlanningError> {
        let (family, index) = parse_mlx_device(device)?;
        let Some((world_size, global_rank)) = rank else {
            return Ok(Self::Placement::direct(None));
        };
        if !cfg!(feature = "distributed") {
            return Err(planning_backend_error(
                "bind_parallel_device",
                super::Error::DistributedModelExecutionDisabled,
            ));
        }
        let backend = self.world_backend.ok_or_else(|| {
            AutomaticPlanningError::Invalid(
                "parallel placement requires an MLX world initializer".into(),
            )
        })?;
        if !safemlx::distributed::is_available(backend) {
            return Err(AutomaticPlanningError::Invalid(format!(
                "MLX communication backend {} is not compiled",
                backend.as_str(),
            )));
        }
        let assignment = crate::backend::DeviceAssignment::new(
            if family.is_some() {
                DeviceType::Gpu
            } else {
                DeviceType::Cpu
            },
            index as usize,
        );
        let rank_context = crate::backend::MlxRankContext::new(world_size, global_rank, assignment)
            .map_err(|error| planning_backend_error("bind_parallel_device", error))?;
        Ok(Self::Placement {
            rank_context: Some(rank_context),
            world_backend: Some(backend),
        })
    }
}

impl ExecutionPlanNativeFactory for MlxBackendFactory {
    type Backend<D: 'static> = MlxBackend<'static, D>;
    type AssistantBinder = MlxAssistantBinder;
    type DraftBinding<D: 'static, P: 'static> = MlxDraftBinding<D, P>;

    fn realize_placement<D: 'static>(
        &self,
        device: &DevicePlan,
        placement: &super::loading::MlxModelPlacement,
    ) -> Result<Self::Backend<D>, AutomaticPlanningError> {
        let realized = mlx_device(device)?;
        let world = match placement.world_backend {
            Some(backend) => {
                let rank = placement.rank_context.ok_or_else(|| {
                    AutomaticPlanningError::Invalid(
                        "selected native world has no rank authority".into(),
                    )
                })?;
                let world = safemlx::distributed::init(true, backend).map_err(|error| {
                    planning_backend_error("initialize_distributed_world", error)
                })?;
                eredu_core::automatic::validate_realized_world(
                    (rank.world_size(), rank.global_rank()),
                    (world.size(), world.rank()),
                )?;
                Some(world)
            }
            None => None,
        };
        let stream = Stream::try_new_with_device(&realized.device)
            .map_err(|error| planning_backend_error("create_execution_stream", error))?;
        let weights_stream = Stream::try_new_with_device(&Device::new(DeviceType::Cpu, 0))
            .map_err(|error| planning_backend_error("create_weights_stream", error))?;
        Ok(MlxBackend::for_execution_plan(
            &stream,
            &weights_stream,
            realized.identity,
            world,
        ))
    }

    fn open_draft<D: 'static, P: 'static>(
        &self,
        binding: eredu_core::ExternalDraftBinding<'_, Self::Backend<D>>,
    ) -> Result<Self::DraftBinding<D, P>, AutomaticPlanningError> {
        let (placement, backend) = binding.into_parts();
        let draft_stream = match placement {
            DraftPlacementPlan::Target => backend.stream().clone(),
            DraftPlacementPlan::Device { device } => {
                let realized = mlx_device(device)?;
                Stream::try_new_with_device(&realized.device).map_err(|error| {
                    planning_backend_error("create_draft_execution_stream", error)
                })?
            }
            _ => {
                return Err(AutomaticPlanningError::Invalid(
                    "unsupported draft placement".into(),
                ))
            }
        };
        MlxDraftBinding::begin(&draft_stream, backend.weights_stream())
            .map_err(|error| planning_backend_error("realize_external_drafter", error))
    }
}

#[derive(Debug)]
struct RealizedMlxDevice {
    device: Device,
    identity: MlxDeviceIdentity,
}

fn parse_mlx_device(
    device: &DevicePlan,
) -> Result<(Option<MlxAcceleratorFamily>, i32), AutomaticPlanningError> {
    let (family, index) = eredu_core::automatic::parse_indexed_device(device, "mlx")?;
    let index = i32::try_from(index)
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
    if let Some(family) = accelerator_family {
        if !family.is_compiled() {
            return Err(AutomaticPlanningError::Invalid(format!(
                "MLX {} device family is not compiled for this target",
                family.as_str()
            )));
        }
    }
    Ok((accelerator_family, index))
}

fn mlx_device(device: &DevicePlan) -> Result<RealizedMlxDevice, AutomaticPlanningError> {
    let (accelerator_family, index) = parse_mlx_device(device)?;
    if let Some(family) = accelerator_family {
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
    error: impl std::fmt::Display,
) -> AutomaticPlanningError {
    AutomaticPlanningError::Backend {
        operation,
        message: error.to_string(),
    }
}
