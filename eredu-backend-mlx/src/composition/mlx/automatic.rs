//! MLX observations and plan realization for neutral automatic planning.

use std::{num::NonZeroUsize, path::Path};

use eredu_core::{
    AutomaticPlanningBackend, AutomaticPlanningError, BackendId, BoundedResidencyRequirement,
    CandidateAdmission, DevicePlan, DraftPlacementPlan, DraftingPlan, DurationSeconds,
    ExecutionPlan, ExecutionPlanBackendFactory, ExecutionPlanTarget, ExecutionPlanTargetSelection,
    ExpertCacheTelemetry, ExternalDraftArtifact, HardwareBackendProfile, HardwareDeviceProfile,
    HardwareMemorySemantics, HardwareProfile, ModelResourceProfile, ModelRuntime, Observed,
    PhysicalMemorySemantics, QuantizationRequest, RealizedDrafting, ResidencyPlan,
    ResidencyTelemetry, SelectedExecutionPlanTarget, SpeculativeDecodingTelemetry,
    SpeculativeDraftSource, SpeculativeGenerationBackend, SpeculativeStats, TransferTelemetry,
    WeightTransformationPlan, AUTOMATIC_SCHEMA_VERSION,
};
use safemlx::{Device, DeviceType, Stream};

use super::{
    capability::available_memory, inspection::MlxInspectionOptions,
    realtime::MlxRealtimeExecutionContext, speculative::MlxDrafter, MlxBackend, MlxLoadRequest,
};
use crate::{
    backend::runtime::residency::parameter_bank::ParameterBankResidencyReport,
    backend::{error::Error, MlxAcceleratorFamily, MlxDeviceIdentity},
};
use eredu_core::residency::{MemoryTier, OffloadConfig, TransferDirection};
use eredu_runtime::{
    DenseDiskStreamLoadOptions, LayerwiseLoadOptions, OrdinaryWeightResidency,
    ParameterBankLoadOptions, ResidencyReport, WeightResidency,
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

    /// Translates a portable execution plan into its backend load request.
    ///
    /// This performs no native device or stream realization and is suitable
    /// for inspection and admission before an executable target is created.
    pub fn load_request_for_plan(
        &self,
        plan: &ExecutionPlan,
    ) -> Result<MlxLoadRequest, AutomaticPlanningError> {
        mlx_load_options(self, plan)
            .map_err(|error| planning_backend_error("select_execution_plan_target", error))
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
    super::path_instrumentation::target_native_resource_realization_attempt();
    let realized =
        mlx_device(device).map_err(|error| Error::AutomaticPlanning(error.to_string()))?;
    let stream = Stream::try_new_with_device(&realized.device)?;
    let weights_stream = Stream::try_new_with_device(&Device::new(DeviceType::Cpu, 0))?;
    let context = MlxRealtimeExecutionContext::new(&stream, &weights_stream);
    let execution = context.materialize_realtime_execution(selected, options)?;
    Ok((context, execution))
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
) -> Result<MlxLoadRequest, Error> {
    if plan.topology().world_size() != 1 {
        return Err(Error::AutomaticPlanning(
            "single-device automatic plans require a 1x1x1 parallel topology".into(),
        ));
    }
    let mut load = match plan.weight_transformation() {
        WeightTransformationPlan::PreserveCheckpoint => MlxLoadRequest::default(),
        WeightTransformationPlan::Affine { bits, group_size } => {
            MlxLoadRequest::with_quantization(QuantizationRequest::Affine {
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
            MlxLoadRequest::with_quantization(QuantizationRequest::MxFp4)
        }
        _ => {
            return Err(Error::AutomaticPlanning(
                "unsupported weight transformation".into(),
            ))
        }
    };
    load.required_session_capabilities = *plan.required_session_capabilities();
    let residency = match plan.residency() {
        ResidencyPlan::FullyResident => OrdinaryWeightResidency::FullyResident,
        ResidencyPlan::LayerwiseHost {
            device_layer_window,
            device_budget_bytes,
            host_budget_bytes,
        } => OrdinaryWeightResidency::LayerwiseHost(
            LayerwiseLoadOptions::new(OffloadConfig::new(
                *device_budget_bytes,
                *host_budget_bytes,
                *device_layer_window,
            )?)
            .with_max_cached_shards(plan.max_cached_shards())
            .with_memory_sampling(factory.sample_mlx_memory, factory.sample_process_memory),
        ),
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
            options = options
                .with_max_cached_shards(plan.max_cached_shards())
                .with_memory_sampling(factory.sample_mlx_memory, factory.sample_process_memory);
            OrdinaryWeightResidency::DenseDiskStream(options)
        }
        _ => {
            return Err(Error::AutomaticPlanning(
                "unsupported residency plan".into(),
            ))
        }
    };
    let residency = if let Some(expert) = plan.expert_cache() {
        WeightResidency::with_independent_parameter_banks(
            residency,
            ParameterBankLoadOptions::new(
                OffloadConfig::new(expert.device_budget_bytes(), expert.host_budget_bytes(), 1)?
                    .with_eviction_policy(expert.eviction_policy()),
                expert.scratch_bytes(),
                expert.prefill_bank_bytes(),
            )?,
        )
    } else {
        match residency {
            OrdinaryWeightResidency::FullyResident => WeightResidency::fully_resident(),
            OrdinaryWeightResidency::LayerwiseHost(options) => {
                WeightResidency::layerwise_host(options)
            }
            OrdinaryWeightResidency::DenseDiskStream(options) => {
                WeightResidency::dense_disk_stream(options)
            }
            _ => {
                return Err(Error::Parallel(
                    "automatic MLX planning selected an unsupported ordinary weight residency"
                        .into(),
                ));
            }
        }
    };
    load = load
        .with_weight_residency(residency)
        .with_drafting_plan(plan.drafting())?;
    Ok(load)
}

fn mlx_drafter_load_options(plan: &ExecutionPlan) -> Result<MlxLoadRequest, Error> {
    match plan.weight_transformation() {
        WeightTransformationPlan::PreserveCheckpoint => Ok(MlxLoadRequest::default()),
        WeightTransformationPlan::Affine { bits, group_size } => Ok(
            MlxLoadRequest::with_quantization(QuantizationRequest::Affine {
                group_size: u32::try_from(group_size).map_err(|_| {
                    Error::Quantization(format!(
                        "group_size must be non-negative, got {group_size}"
                    ))
                })?,
                bits: u8::try_from(bits)
                    .map_err(|_| Error::Quantization(format!("bits must fit in u8, got {bits}")))?,
            }),
        ),
        WeightTransformationPlan::MxFp4 => Ok(MlxLoadRequest::with_quantization(
            QuantizationRequest::MxFp4,
        )),
        _ => Err(Error::AutomaticPlanning(
            "unsupported weight transformation".into(),
        )),
    }
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
        let inspection = eredu_architectures::configuration::inspect_artifact(model_path)
            .map_err(|error| planning_backend_error("inspect_resources", error))?;
        let report = super::inspection::inspect_selected_artifact(
            &inspection,
            MlxInspectionOptions::default(),
        );
        Ok((report.resources, inspection))
    }

    fn admit_candidate(
        &self,
        inspection: &Self::Inspection,
        plan: &ExecutionPlan,
    ) -> Result<CandidateAdmission, AutomaticPlanningError> {
        let load = mlx_load_options(self, plan)
            .map_err(|error| planning_backend_error("realize_plan", error))?;
        let policy = load
            .preparation_policy()
            .map_err(|error| planning_backend_error("admit_candidate_policy", error))?;
        match super::loading::select_preparation(inspection, load, policy) {
            Ok(_) => Ok(CandidateAdmission {
                supported: true,
                rejection: None,
            }),
            Err(error) => Ok(CandidateAdmission {
                supported: false,
                rejection: Some(error.to_string()),
            }),
        }
    }

    fn bounded_residency_requirement(
        &self,
        inspection: &Self::Inspection,
        plan: &ExecutionPlan,
    ) -> Result<BoundedResidencyRequirement, AutomaticPlanningError> {
        if matches!(plan.residency(), ResidencyPlan::FullyResident) {
            return Err(AutomaticPlanningError::Invalid(
                "fully resident execution has no bounded device window".into(),
            ));
        }
        let options = self
            .load_request_for_plan(plan)
            .map_err(|error| planning_backend_error("bounded_residency_options", error))?;
        let policy = options
            .preparation_policy()
            .map_err(|error| planning_backend_error("bounded_residency_policy", error))?;
        let selected = super::loading::select_preparation(inspection, options, policy)
            .map_err(|error| planning_backend_error("select_model_preparation", error))?;
        let (text, excluded) = selected
            .selected_bounded_residency()
            .map_err(|error| planning_backend_error("selected_text_residency", error))?;
        selected_text_bounded_requirement(&text, &excluded)
            .map_err(|error| planning_backend_error("selected_text_residency", error))
    }
}

fn selected_text_bounded_requirement(
    selected: &eredu_runtime::SelectedReplicatedTextRealization,
    excluded: &std::collections::BTreeSet<String>,
) -> Result<BoundedResidencyRequirement, String> {
    let tasks = eredu_runtime::replicated_text_materialization_tasks(selected)
        .map_err(|error| error.to_string())?;
    let mut static_bytes = 0u64;
    let mut groups =
        std::collections::BTreeMap::<String, std::collections::BTreeMap<usize, u64>>::new();
    for task in tasks {
        if excluded.contains(task.name()) {
            continue;
        }
        let bytes = eredu_runtime::selected_materialization_task_bytes(&task)
            .map_err(|error| error.to_string())?;
        match task.owner() {
            eredu_runtime::ReplicatedTextParameterOwner::StaticRole(_) => {
                static_bytes = static_bytes
                    .checked_add(bytes)
                    .ok_or_else(|| "selected static parameter bytes overflowed".to_owned())?;
            }
            eredu_runtime::ReplicatedTextParameterOwner::ExecutionUnit { group, unit } => {
                let total = groups
                    .entry(group.clone())
                    .or_default()
                    .entry(*unit)
                    .or_default();
                *total = total
                    .checked_add(bytes)
                    .ok_or_else(|| "selected execution-unit bytes overflowed".to_owned())?;
            }
            _ => return Err("selected parameter has an unsupported residency owner".into()),
        }
    }
    let unit_count = groups
        .values()
        .map(|units| units.keys().next_back().map_or(0, |unit| unit + 1))
        .sum::<usize>();
    let depth = selected.residency().device_depth(unit_count);
    let mut window_bytes = 0u64;
    for units in groups.values() {
        let count = units.keys().next_back().map_or(0, |unit| unit + 1);
        let bytes = (0..count)
            .map(|unit| units.get(&unit).copied().unwrap_or(0))
            .collect::<Vec<_>>();
        for start in 0..bytes.len() {
            let current = bytes
                .iter()
                .skip(start)
                .take(depth)
                .try_fold(0u64, |total, bytes| total.checked_add(*bytes))
                .ok_or_else(|| "selected device-window bytes overflowed".to_owned())?;
            window_bytes = window_bytes.max(current);
        }
    }
    let required_bytes = static_bytes
        .checked_add(window_bytes)
        .ok_or_else(|| "selected bounded-residency bytes overflowed".to_owned())?;
    Ok(BoundedResidencyRequirement {
        static_bytes,
        window_bytes,
        required_bytes,
        depth,
    })
}

/// Cold-selected external assistant plus MLX reader-cache mechanism policy.
pub struct SelectedMlxExternalAssistantPreparation {
    preparation: eredu_architectures::CompatibleExternalAssistantPreparation,
    speculative: eredu_runtime::SelectedSpeculativeRealization,
    max_cached_shards: usize,
}

impl ExecutionPlanBackendFactory for MlxBackendFactory {
    type Backend = MlxBackend<'static>;
    type DrafterPreparation = eredu_architectures::ExternalAssistantPreparation;
    type SelectedDrafterPreparation = SelectedMlxExternalAssistantPreparation;
    type Drafter = MlxDrafter;

    fn select_target(
        &self,
        inspection: &eredu_core::ArtifactInspection<
            eredu_architectures::processor_plan::ArtifactArchitecturePlan,
        >,
        plan: &ExecutionPlan,
    ) -> Result<ExecutionPlanTargetSelection<Self::Backend>, AutomaticPlanningError> {
        let options = self.load_request_for_plan(plan)?;
        let policy = options
            .preparation_policy()
            .map_err(|error| planning_backend_error("select_preparation_policy", error))?;
        let selected = super::loading::select_preparation(inspection, options, policy)
            .map_err(|error| planning_backend_error("select_model_preparation", error))?;
        let capabilities = super::structural::inspected_session_capabilities(inspection, policy)
            .map_err(|error| planning_backend_error("select_session_capabilities", error))?;
        Ok(ExecutionPlanTargetSelection::new(
            policy,
            selected,
            capabilities,
        ))
    }

    fn realize_target(
        &self,
        selected: SelectedExecutionPlanTarget<Self::Backend>,
    ) -> Result<ExecutionPlanTarget<Self::Backend>, AutomaticPlanningError> {
        #[cfg(test)]
        super::path_instrumentation::target_native_resource_realization_attempt();
        let realized = mlx_device(selected.execution_plan().device())?;
        let stream = Stream::try_new_with_device(&realized.device)
            .map_err(|error| planning_backend_error("create_execution_stream", error))?;
        let weights_stream = Stream::try_new_with_device(&Device::new(DeviceType::Cpu, 0))
            .map_err(|error| planning_backend_error("create_weights_stream", error))?;
        Ok(ExecutionPlanTarget::new(
            MlxBackend::for_execution_plan(&stream, &weights_stream, realized.identity),
            selected,
        ))
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
        let options = mlx_drafter_load_options(plan)
            .map_err(|error| planning_backend_error("select_external_drafter", error))?;
        if !options.weight_residency.is_fully_resident() {
            return Err(AutomaticPlanningError::Invalid(
                "external assistants require fully resident weights".into(),
            ));
        }
        if options
            .parallel_topology()
            .is_some_and(|topology| !topology.is_replicated())
        {
            return Err(AutomaticPlanningError::Invalid(
                "external assistants require replicated placement".into(),
            ));
        }
        let max_cached_shards = options.weight_residency.max_cached_shards();
        let preparation = artifact
            .preparation
            .select_materialization(options.quantization, |descriptor, transforms| {
                if transforms && super::replicated_text::supports_transform(descriptor) {
                    Some(eredu_runtime::WeightLoweringKind::Transform)
                } else if !transforms && super::replicated_text::supports_direct(descriptor) {
                    Some(eredu_runtime::WeightLoweringKind::Direct)
                } else {
                    None
                }
            })
            .map_err(|message| AutomaticPlanningError::Backend {
                operation: "select_external_drafter",
                message,
            })?;
        let target_profile = target
            .inspection()
            .architecture_plan()
            .external_assistant_target_profile()
            .ok_or_else(|| {
                AutomaticPlanningError::Invalid(
                    "selected target does not admit an external assistant".into(),
                )
            })?;
        let preparation = preparation
            .prove_target_compatibility(&target_profile)
            .map_err(|error| {
                AutomaticPlanningError::Invalid(format!(
                    "external assistant is incompatible with the selected target: {error}"
                ))
            })?;
        let placement = match plan.drafting() {
            DraftingPlan::External { placement, .. } => placement,
            _ => {
                return Err(AutomaticPlanningError::Invalid(
                    "external assistant selection requires an external drafting plan".into(),
                ))
            }
        };
        let maximum_draft_tokens = match plan.drafting() {
            DraftingPlan::External {
                max_draft_tokens, ..
            } => NonZeroUsize::new(*max_draft_tokens).ok_or_else(|| {
                AutomaticPlanningError::Invalid("external draft capacity must be positive".into())
            })?,
            _ => unreachable!("external drafting plan checked above"),
        };
        let placement_request = eredu_runtime::SpeculativePlacementRequest::from_topology(
            placement.execution_topology(plan.device()),
        )
        .map_err(|error| AutomaticPlanningError::Invalid(error.to_string()))?;
        let rank_topology = eredu_core::ParallelRankTopology::new(*plan.topology(), 0)
            .map_err(|error| AutomaticPlanningError::Invalid(error.to_string()))?;
        let processor = eredu_runtime::SpeculativeIdentity::new("prepared-chat/text-token-ids/v1")
            .map_err(|error| AutomaticPlanningError::Invalid(error.to_string()))?;
        let contract = preparation
            .speculative_contract(
                eredu_architectures::ExternalSpeculativeContractRequest::new(
                    rank_topology,
                    processor,
                    artifact.tokenizer_compatibility,
                    artifact.tokenizer_compatibility.fingerprint(),
                    maximum_draft_tokens,
                ),
            )
            .map_err(|error| {
                AutomaticPlanningError::Invalid(format!(
                    "external speculative contract is invalid: {error}"
                ))
            })?;
        let speculative = eredu_runtime::select_and_prepare_speculative_realization_observed(
            contract.requirements(),
            &contract.selection_request(placement_request),
            &super::speculative::speculative_mechanism_capabilities(),
            &|_| Ok(()),
            |_| Ok::<_, AutomaticPlanningError>(()),
            |_, &()| Ok::<_, AutomaticPlanningError>(()),
            |_, &()| Ok::<_, AutomaticPlanningError>(()),
            |_| Ok::<_, AutomaticPlanningError>(()),
            |_, &()| Ok::<_, AutomaticPlanningError>(()),
        )
        .map_err(|error| AutomaticPlanningError::Invalid(error.to_string()))?
        .into_parts()
        .0;
        Ok(Some(ExternalDraftArtifact {
            preparation: SelectedMlxExternalAssistantPreparation {
                preparation,
                speculative,
                max_cached_shards,
            },
            tokenizer_compatibility: artifact.tokenizer_compatibility,
        }))
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
                let max_cached_shards = artifact.preparation.max_cached_shards;
                let preparation = artifact.preparation.preparation;
                let selected = artifact.preparation.speculative;
                let draft_stream = match placement {
                    DraftPlacementPlan::Target => target.backend().stream().clone(),
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
                let drafter = MlxDrafter::materialize_with_compatibility(
                    preparation,
                    artifact.tokenizer_compatibility,
                    max_cached_shards,
                    &draft_stream,
                    target.backend().weights_stream(),
                    selected,
                )
                .map_err(|error| planning_backend_error("realize_external_drafter", error))?;
                draft_stream.synchronize().map_err(|error| {
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
pub fn parameter_bank_telemetry(report: &ParameterBankResidencyReport) -> ExpertCacheTelemetry {
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

/// Converts neutral speculative statistics into the stable telemetry document.
pub fn speculative_decoding_telemetry(stats: &SpeculativeStats) -> SpeculativeDecodingTelemetry {
    SpeculativeDecodingTelemetry {
        execution_topology: stats.execution_topology().to_string(),
        target_tokens: stats.target_tokens(),
        draft_tokens: stats.draft_tokens(),
        accepted_tokens: stats.accepted_tokens(),
        accept_rate: stats.accept_rate(),
        rounds: stats.rounds(),
        accept_lens: stats.accept_lens().to_vec(),
        emitted_tokens: stats.emitted_tokens(),
        optimistic_draft_tokens: stats.optimistic_draft_tokens(),
        reused_optimistic_tokens: stats.reused_optimistic_tokens(),
        discarded_optimistic_tokens: stats.discarded_optimistic_tokens(),
        adaptive_lookahead_disabled: stats.adaptive_lookahead_disabled(),
        optimistic_draft_seconds: stats.optimistic_draft_time().as_secs_f64(),
        verification_in_flight_seconds: stats.verification_in_flight_time().as_secs_f64(),
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
            )))
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
    fn realtime_capability_rejection_precedes_device_and_stream_realization() {
        super::super::path_instrumentation::reset();
        let directory = tempfile::tempdir().expect("tiny realtime artifact directory");
        super::super::realtime::tests::write_tiny_native_artifact(directory.path(), None);
        let preparation = eredu_architectures::moshi::prepare_realtime_model(directory.path())
            .expect("tiny realtime artifact is valid");
        let device = DevicePlan::new("mlx", "gpu:0").expect("portable device name is valid");
        let options = MlxLoadRequest::default().with_required_session_capabilities(
            eredu_core::SessionCapabilities::default().with_activation_inspection(true),
        );

        let error = match create_realtime_execution(preparation, &device, options) {
            Ok(_) => panic!("unsupported activation observation must reject selection"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("activation_inspection"));
        assert_eq!(
            super::super::path_instrumentation::target_native_resource_realization_attempts(),
            0
        );
    }

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
        let plan = ExecutionPlan::fully_resident(DevicePlan::new("mlx", "cpu:0").unwrap())
            .with_topology(eredu_core::ParallelTopology::new(2, 1, 1, 1).unwrap());
        assert!(mlx_load_options(&MlxBackendFactory::default(), &plan).is_err());
    }

    #[test]
    fn failed_target_selection_creates_no_native_resources() {
        super::super::path_instrumentation::reset();
        let directory = tempfile::tempdir().expect("tiny inspected artifact directory");
        super::super::realtime::tests::write_tiny_native_artifact(directory.path(), None);
        let inspection = eredu_architectures::configuration::inspect_artifact(directory.path())
            .expect("tiny artifact inspection succeeds");
        let plan = ExecutionPlan::fully_resident(DevicePlan::new("mlx", "cpu:0").unwrap())
            .with_topology(eredu_core::ParallelTopology::new(2, 1, 1, 1).unwrap());

        let error = match eredu_core::select_execution_plan_target(
            &MlxBackendFactory::default(),
            &plan,
            inspection,
        ) {
            Ok(_) => panic!("distributed topology must fail before native realization"),
            Err(error) => error,
        };

        assert!(error.to_string().contains("1x1x1"));
        assert_eq!(
            super::super::path_instrumentation::target_native_resource_realization_attempts(),
            0
        );
    }

    #[test]
    fn bounded_probe_selects_before_native_resources() {
        super::super::path_instrumentation::reset();
        let directory = tempfile::tempdir().expect("tiny realtime artifact directory");
        super::super::realtime::tests::write_tiny_native_artifact(directory.path(), None);
        let plan = ExecutionPlan::fully_resident(DevicePlan::new("mlx", "cpu:0").unwrap())
            .with_residency(ResidencyPlan::LayerwiseHost {
                device_layer_window: 1,
                device_budget_bytes: None,
                host_budget_bytes: None,
            });

        let inspection = eredu_architectures::configuration::inspect_artifact(directory.path())
            .expect("realtime artifact inspection");
        let error = MlxBackendFactory::default()
            .bounded_residency_requirement(&inspection, &plan)
            .expect_err("realtime architecture must not enter ordinary bounded probing");

        assert!(matches!(
            error,
            AutomaticPlanningError::Backend {
                operation: "select_model_preparation",
                ref message,
            } if message.contains("Realtime loading protocol")
        ));
        assert_eq!(
            super::super::path_instrumentation::target_native_resource_realization_attempts(),
            0
        );
    }

    #[test]
    fn bounded_requirement_for_ordinary_text_uses_only_the_cold_selected_tasks() {
        super::super::path_instrumentation::reset();
        let directory = super::super::replicated_text::tests::tiny_artifact("llama", false);
        let plan = ExecutionPlan::fully_resident(DevicePlan::new("mlx", "cpu:0").unwrap())
            .with_residency(ResidencyPlan::LayerwiseHost {
                device_layer_window: 1,
                device_budget_bytes: None,
                host_budget_bytes: None,
            });

        let inspection = eredu_architectures::configuration::inspect_artifact(directory.path())
            .expect("ordinary artifact inspection");
        let requirement = MlxBackendFactory::default()
            .bounded_residency_requirement(&inspection, &plan)
            .expect("ordinary selected tasks have an exact bounded requirement");

        assert!(requirement.static_bytes > 0);
        assert!(requirement.window_bytes > 0);
        assert_eq!(
            requirement.required_bytes,
            requirement.static_bytes + requirement.window_bytes
        );
        assert_eq!(requirement.depth, 1);
        assert_eq!(
            super::super::path_instrumentation::target_native_resource_realization_attempts(),
            0
        );
        let counts = super::super::path_instrumentation::snapshot();
        assert_eq!(counts.payload_opens, 0);
        assert_eq!(counts.architecture_constructions, 0);
        assert_eq!(counts.constructors, 0);
        assert_eq!(counts.materializations, 0);
    }

    #[test]
    fn realized_cpu_identity_is_derived_from_the_native_device() {
        let plan = DevicePlan::new("mlx", "cpu:0").unwrap();
        let realized = mlx_device(&plan).unwrap();
        let stream = Stream::try_new_with_device(&realized.device).unwrap();
        let backend = MlxBackend::for_execution_plan(&stream, &stream, realized.identity);

        let devices = backend.devices().unwrap();
        assert_eq!(devices[0].0.id(), "cpu:0");
        assert_eq!(devices[0].0.family(), "cpu");
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
        let plan = DevicePlan::new("mlx", "metal:0").unwrap();
        let realized = mlx_device(&plan).unwrap();
        let stream = Stream::try_new_with_device(&realized.device).unwrap();
        let backend = MlxBackend::for_execution_plan(&stream, &stream, realized.identity);
        let devices = backend.devices().unwrap();

        assert_eq!(devices[0].0.id(), "metal:0");
        assert_eq!(devices[0].0.family(), "metal");
    }
}
