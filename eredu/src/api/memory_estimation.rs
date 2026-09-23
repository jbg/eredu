//! Cold request memory planning composed from retained architecture selection.

use eredu_architectures::{ModelInspectionOutcome, PreparationMechanismProvider};
use eredu_core::{
    CapabilityError, DevicePlan, HardwareMemorySemantics, HardwareProfile, InputTokenCount,
    MediaFeatureAvailability, Observed,
};
pub use eredu_runtime::memory_estimation::*;
use eredu_runtime::{LayerWeightResidency, NormalizedLoadRequest};
use std::path::Path;

/// Physical capacities used by the selected backend.
#[derive(Debug, Clone)]
pub enum GenerationMemoryPlacement {
    /// Host and device allocations consume one shared physical capacity.
    Unified,
    /// Execution is entirely in host memory.
    Host,
    /// Physical capacity relationship is unavailable; fit remains indeterminate.
    Unknown,
    /// Host and the named execution device have distinct capacities.
    Separate {
        /// Stable device label.
        device: String,
    },
}

/// Request geometry and explicitly supplied backend estimation assumptions.
#[derive(Debug, Clone)]
pub struct GenerationMemoryOptions {
    /// Exact text and prepared-media position count.
    pub input: InputTokenCount,
    /// Output allowance; `None` uses a labeled forecast horizon.
    pub max_output_tokens: Option<u64>,
    /// Horizon used only when output length is unspecified.
    pub forecast_output_tokens: u64,
    /// Concurrent sequences in this request.
    pub batch_size: u64,
    /// Maximum positions submitted by one prefill invocation.
    pub prefill_chunk_tokens: u64,
    /// Legacy hint, ignored: cold selection now determines prefix support.
    #[deprecated(
        note = "cold prefix support is derived from SelectedPreparation::prefill_chunking_support"
    )]
    pub chunked_prefill_supported: bool,
    /// Physical backing relationship reported by the backend.
    pub placement: GenerationMemoryPlacement,
    /// Selected attention mechanism or an explicit conservative assumption.
    pub attention: AttentionWorkspace,
    /// Simultaneously retained activation workspaces, including lazy graph overlap.
    pub workspace_overlap: WorkspaceOverlap,
    /// Cache-update mechanism or an explicit conservative assumption.
    pub cache_update: CacheUpdateWorkspace,
    /// Actual prefill output contract.
    pub logits: LogitsWorkspace,
    /// Native allocator, graph and driver overhead allowance.
    pub backend_overhead: MemoryBytes,
    /// Execution-domain capacity and uncertainty reserve.
    pub budget: MemoryBudget,
    /// Host capacity, used only for separate host/device memory.
    pub host_budget: MemoryBudget,
    /// Already-resident bytes included in this forecast's execution-domain costs.
    /// Leave zero for a model that has not been loaded. On unified memory count
    /// host/device parameter backing once: host + device - known shared bytes,
    /// or `max(host, device)` when overlap is unknown. Use
    /// [`static_parameter_placement`]'s lower end, not RSS or global allocator
    /// counters. For separate host/device domains this field applies only to
    /// the device; use explicit domain plans or a loaded-model forecast for
    /// independently resident host bytes. This reduces additional-memory demand
    /// against observed availability, never the total application-budget demand.
    pub already_resident_bytes: u64,
    /// Additional retained prepared input, excluding token IDs counted below.
    pub retained_input: MemoryBytes,
}

impl GenerationMemoryOptions {
    /// Derives physical placement and point-in-time availability from an exact
    /// backend/device observation. This pure constructor performs no discovery.
    ///
    /// `host_execution` is a backend-supplied fact about the selected device;
    /// opaque device identifiers and family names are not interpreted here.
    /// Missing capacity stays unknown, including when installed capacity is
    /// known. Application limits/reserves remain unset and overhead remains
    /// unknown. An absent or unavailable backend/device is a typed error.
    pub fn for_hardware_device(
        input: InputTokenCount,
        hardware: &HardwareProfile,
        device: &DevicePlan,
        host_execution: bool,
    ) -> Result<Self, CapabilityError> {
        let backend = hardware
            .backends
            .iter()
            .find(|backend| &backend.backend == device.backend() && backend.available)
            .ok_or_else(|| CapabilityError::InvalidConfiguration {
                field: "hardware.backend",
                detail: format!(
                    "selected backend {} is absent or unavailable",
                    device.backend()
                ),
            })?;
        let selected = backend
            .devices
            .iter()
            .find(|candidate| candidate.id == device.device())
            .ok_or_else(|| CapabilityError::InvalidConfiguration {
                field: "hardware.device",
                detail: format!(
                    "device {} was not discovered for backend {}",
                    device.device(),
                    device.backend()
                ),
            })?;
        let placement = if host_execution {
            GenerationMemoryPlacement::Host
        } else {
            match hardware.physical_memory_semantics {
                HardwareMemorySemantics::Unified => GenerationMemoryPlacement::Unified,
                HardwareMemorySemantics::SeparateTiers => GenerationMemoryPlacement::Separate {
                    device: selected.id.clone(),
                },
                HardwareMemorySemantics::Unknown => GenerationMemoryPlacement::Unknown,
            }
        };
        let available = match placement {
            GenerationMemoryPlacement::Host | GenerationMemoryPlacement::Unified => {
                hardware.available_memory_bytes.value().copied()
            }
            GenerationMemoryPlacement::Separate { .. } => {
                selected.available_memory_bytes.value().copied()
            }
            GenerationMemoryPlacement::Unknown => None,
        };
        let mut options = Self::new(input, placement);
        options.budget.available_bytes = available;
        options.host_budget.available_bytes = hardware.available_memory_bytes.value().copied();
        Ok(options)
    }

    /// Creates an estimate with explicit unknown backend mechanisms and overhead.
    /// `Self::for_hardware_device` derives placement and available capacities
    /// from supplied observations. With `mlx`, `Self::for_local_device` also
    /// performs local discovery and samples allocator overhead.
    /// With the `mlx` feature, `Self::for_local_backend` instead samples the
    /// current local allocator-cache policy for its overhead allowance.
    #[allow(deprecated)]
    pub fn new(input: InputTokenCount, placement: GenerationMemoryPlacement) -> Self {
        Self {
            input,
            max_output_tokens: None,
            forecast_output_tokens: 256,
            batch_size: 1,
            prefill_chunk_tokens: 512,
            chunked_prefill_supported: false,
            placement,
            attention: AttentionWorkspace::Unknown,
            workspace_overlap: WorkspaceOverlap {
                upper_live_copies: None,
                detail: "backend lazy activation overlap has not been calibrated".into(),
            },
            cache_update: CacheUpdateWorkspace::Unknown,
            logits: LogitsWorkspace::EveryPosition,
            backend_overhead: MemoryBytes::unknown(
                "backend allocator and graph overhead has not been calibrated",
            ),
            budget: MemoryBudget::default(),
            host_budget: MemoryBudget::default(),
            already_resident_bytes: 0,
            retained_input: MemoryBytes::exact(0),
        }
    }
}

fn add(a: u64, b: u64) -> Result<u64, CapabilityError> {
    a.checked_add(b).ok_or(CapabilityError::ArithmeticOverflow {
        operation: "cold memory projection",
    })
}
fn mul(a: u64, b: u64) -> Result<u64, CapabilityError> {
    a.checked_mul(b).ok_or(CapabilityError::ArithmeticOverflow {
        operation: "cold memory projection",
    })
}
fn observed_bytes(value: &Observed<u64>) -> MemoryBytes {
    match value {
        Observed::Available { value, .. } => MemoryBytes::exact(*value),
        Observed::Unavailable { reason } | Observed::Unsupported { reason } => {
            MemoryBytes::unknown(reason)
        }
    }
}

/// Inspects metadata, selects execution from cold backend facts, and estimates the request.
/// No native device or tensor is constructed and checkpoint payloads are not loaded.
pub fn inspect_generation_memory<P: PreparationMechanismProvider>(
    path: impl AsRef<Path>,
    request: &NormalizedLoadRequest,
    mechanisms: &P,
    media: MediaFeatureAvailability,
    options: &GenerationMemoryOptions,
) -> Result<GenerationMemoryEstimate, CapabilityError> {
    let inspection = eredu_architectures::inspect_model(path, request, mechanisms, media);
    estimate_inspected_generation_memory(&inspection, options)
}

/// Estimates an already inspected selection, preserving checkpoint and mechanism choices.
/// Unknown coverage affects fit advice only; it never changes load or generation admission.
pub fn estimate_inspected_generation_memory(
    inspection: &ModelInspectionOutcome,
    options: &GenerationMemoryOptions,
) -> Result<GenerationMemoryEstimate, CapabilityError> {
    let selected_embedded = inspection
        .selected()
        .is_some_and(|s| s.preparation().prediction_realization().is_some());
    let (request, mut report) = if selected_embedded {
        let capacity = inspection
            .selected()
            .unwrap()
            .preparation()
            .prediction_realization()
            .unwrap()
            .requirements()
            .strategy()
            .proposal_capacity()
            .get() as u64;
        let (request, plan) = inspected_speculative_generation_memory_plan(
            inspection, options, capacity,
            eredu_core::generation::SpeculativeSchedulerOptions::default(),
            MemoryBytes::unknown("cold speculative sampling scratch has no backend calibration; supply an explicit sampling bound through inspected_speculative_generation_memory_plan"),
        )?;
        let estimate =
            eredu_runtime::memory_forecast::estimate_speculative_memory(&request, &plan)?;
        (request, estimate)
    } else {
        let request = inspected_generation_memory_request(inspection, options)?;
        let estimate = estimate_generation_memory(&request)?;
        (request, estimate)
    };
    if request.prefill_chunk_tokens != options.prefill_chunk_tokens {
        report.assumptions.push("selected execution does not have modeled chunked-prefill support; full-prompt workspace was used".into());
    }
    if let Some(selected) = inspection.selected() {
        if selected
            .preparation()
            .execution()
            .parallel_topology()
            .is_some()
        {
            report.uncertainties.push("rank-local persistent state and workspace are unavailable; zero persistent bytes is only the known lower contribution".into());
        }
        report.assumptions.extend(
            eredu_architectures::memory_estimation::generation_memory_geometry(
                selected.inspection().architecture_plan(),
            )?
            .assumptions,
        );
    }
    Ok(report)
}

/// Builds a reusable request so applications can recompute supported chunk candidates.
pub fn inspected_generation_memory_request(
    inspection: &ModelInspectionOutcome,
    options: &GenerationMemoryOptions,
) -> Result<GenerationMemoryRequest, CapabilityError> {
    inspected_generation_memory_request_inner(inspection, options, false)
}

/// Builds a recomputable embedded startup plan from an exact cold selection.
/// Sampling scratch is a backend calibration, not a fact inferred from a family.
/// The returned target request must be estimated together with the returned plan.
pub fn inspected_speculative_generation_memory_plan(
    inspection: &ModelInspectionOutcome,
    options: &GenerationMemoryOptions,
    max_draft_tokens: u64,
    scheduler: eredu_core::generation::SpeculativeSchedulerOptions,
    sampling_bytes_per_vocabulary_entry: MemoryBytes,
) -> Result<
    (
        GenerationMemoryRequest,
        eredu_runtime::memory_forecast::SpeculativeMemoryPlan,
    ),
    CapabilityError,
> {
    use eredu_runtime::memory_forecast::{
        EmbeddedPredictionMemoryPlan, ForecastCalibration, SpeculativeMemoryPlan,
    };
    let selected = inspection.selected().ok_or_else(|| {
        CapabilityError::Observation("embedded forecast requires a valid cold selection".into())
    })?;
    let topology = selected
        .preparation()
        .embedded_prediction_topology()
        .map_err(|e| CapabilityError::Observation(e.to_string()))?
        .ok_or_else(|| {
            CapabilityError::Observation("selection has no embedded prediction topology".into())
        })?;
    if max_draft_tokens == 0 || max_draft_tokens > topology.proposal_capacity as u64 {
        return Err(CapabilityError::InvalidConfiguration {
            field: "max_draft_tokens",
            detail: "must be positive and within the selected proposal capacity".into(),
        });
    }
    scheduler
        .validate()
        .map_err(|e| CapabilityError::Observation(e.to_string()))?;
    if sampling_bytes_per_vocabulary_entry
        .upper_bytes
        .is_some_and(|upper| upper < sampling_bytes_per_vocabulary_entry.lower_bytes)
    {
        return Err(CapabilityError::InvalidConfiguration {
            field: "sampling_bytes_per_vocabulary_entry",
            detail: "upper bound is below lower bound".into(),
        });
    }
    let mut request = inspected_generation_memory_request_inner(inspection, options, true)?;
    request.prefill_chunk_tokens = request.input.model_positions.max(1);
    for execution in request.domains.iter_mut().flat_map(|d| &mut d.executions) {
        execution.logits = LogitsWorkspace::EveryPosition;
    }
    let calibration = ForecastCalibration {
        attention: options.attention.clone(),
        cache_update: options.cache_update,
        workspace_overlap: Some(options.workspace_overlap.clone()),
        ..Default::default()
    };
    let embedded = EmbeddedPredictionMemoryPlan::from_topology(&topology, &request, &calibration)?;
    Ok((
        request,
        SpeculativeMemoryPlan {
            embedded: Some(embedded),
            draft: None,
            auxiliary_bytes_per_position: MemoryBytes::exact(0),
            sampling_bytes_per_vocabulary_entry,
            max_draft_tokens,
            scheduler,
            shared_allocator: true,
        },
    ))
}

fn inspected_generation_memory_request_inner(
    inspection: &ModelInspectionOutcome,
    options: &GenerationMemoryOptions,
    embedded_target: bool,
) -> Result<GenerationMemoryRequest, CapabilityError> {
    let selected = inspection.selected().ok_or_else(|| CapabilityError::Observation("memory estimation requires a valid cold execution selection; inspect model issues for the admission failure".into()))?;
    let execution = selected.preparation().execution();
    let text = execution.text_realization();
    let geometry = eredu_architectures::memory_estimation::selected_generation_memory_geometry(
        selected.inspection().architecture_plan(),
        execution,
    )?;
    let scalar_bytes = geometry.scalar_bytes;
    let resources = &inspection.report().resources;
    let partitioned = execution.parallel_topology().is_some();
    let mut workspace = geometry.workspace;
    let mut execution_topology = geometry.execution_topology;
    let state_layout = geometry.state_layout;
    if selected.preparation().prediction_realization().is_some() && !embedded_target {
        workspace = None;
        execution_topology = None;
    }
    let total = if partitioned {
        resources
            .selected_rank
            .value()
            .map(|rank| rank.materialized_parameter_bytes)
    } else {
        resources.materialized_parameter_bytes.value().copied()
    };
    let pinned = resources.pinned_parameter_bytes.value().copied();
    let group = resources.largest_execution_group_bytes.value().copied();
    let mut host_parameters = 0;
    let mut resident_parameters = if partitioned {
        total.map_or_else(
            || MemoryBytes::unknown("rank-local materialized parameters are unavailable"),
            MemoryBytes::exact,
        )
    } else {
        observed_bytes(&resources.materialized_parameter_bytes)
    };
    match (text.residency(), total, pinned, group) {
        (LayerWeightResidency::FullyResident, _, _, _) => {}
        (LayerWeightResidency::LayerwiseHost(load), Some(total), Some(pinned), Some(group))
            if !partitioned =>
        {
            host_parameters = total.saturating_sub(pinned);
            let window = mul(
                group,
                u64::try_from(load.offload().prefetch_depth()).map_err(|_| {
                    CapabilityError::ArithmeticOverflow {
                        operation: "prefetch depth",
                    }
                })?,
            )?;
            let upper = add(pinned, window.min(host_parameters))?;
            resident_parameters = MemoryBytes::estimated(
                pinned,
                upper,
                "selected host-backed layer window; physical host/device copies may overlap",
            );
        }
        (LayerWeightResidency::DenseDiskStream(load), Some(total), Some(pinned), _)
            if !partitioned =>
        {
            host_parameters = load.host_budget_bytes().min(total.saturating_sub(pinned));
            resident_parameters = MemoryBytes::estimated(
                pinned,
                total.min(load.device_budget_bytes()).max(pinned),
                "selected bounded device parameter cache",
            );
        }
        _ => {
            resident_parameters =
                MemoryBytes::unknown("selected bounded rank residency has no resource projection")
        }
    }
    if !execution.bounded_residency_exclusions().is_empty() {
        resident_parameters = total.map_or_else(
            || MemoryBytes::unknown("independent expert cache placement unavailable"),
            |total| {
                MemoryBytes::estimated(
                    0,
                    total,
                    "independently cached expert parameters; exact live bank occupancy unavailable",
                )
            },
        );
        workspace = None;
        execution_topology = None;
    }
    let fully_resident = matches!(text.residency(), LayerWeightResidency::FullyResident);
    let mut staging = if fully_resident {
        MemoryBytes::exact(0)
    } else {
        MemoryBytes::unknown("native transfer and recipe staging is unavailable")
    };
    let mut loading_peak = MemoryBytes::unknown("loading/conversion overlap is unavailable");
    if let (Some(parameters), Some(stored), Some(materialization)) = (
        total,
        resources.stored_tensor_bytes.value(),
        selected.memory_materialization_workspace().value(),
    ) {
        let native = mul(
            materialization.ordinary_native_peak_bytes,
            materialization.ordinary_materializations as u64,
        )?;
        let scratch = add(
            add(native, materialization.expert_member_native_peak_bytes)?,
            materialization.compact_bank_bytes,
        )?;
        if !fully_resident {
            staging = MemoryBytes::estimated(
                0,
                scratch,
                "overlapping selected materialization and transfer buffers",
            );
        }
        let conversion = scratch
            .max(materialization.conversion_workspace_bytes)
            .max(materialization.ordinary_recipe_peak_bytes);
        loading_peak = MemoryBytes::estimated(resident_parameters.lower_bytes, add(add(parameters, *stored)?, conversion)?, "materialized destinations plus checkpoint source payload and conversion scratch; file pages may be reclaimable");
    }
    let chunk_tokens = if selected.preparation().prefill_chunking_support().is_ok()
        && options.input.model_positions == options.input.text_tokens
    {
        options.prefill_chunk_tokens
    } else {
        options.input.model_positions.max(1)
    };
    let token_bytes = mul(mul(options.input.text_tokens, options.batch_size)?, 4)?;
    let mut retained_input = options.retained_input.clone();
    // Input geometry alone cannot infer arbitrary image/audio buffers.
    if options.input.model_positions != options.input.text_tokens {
        retained_input = MemoryBytes::unknown(
            "prepared media retention requires its actual host/tensor buffer sizes",
        );
    }
    let execution_plan = ExecutionMemoryPlan {
        state_layout,
        workspace,
        execution_topology,
        input_score_attention_mechanism: geometry.input_score_attention_mechanism,
        attention: options.attention.clone(),
        workspace_overlap: options.workspace_overlap.clone(),
        cache_update: options.cache_update,
        logits: options.logits,
    };
    let mut domain = DomainMemoryPlan {
        domain: MemoryDomain::Unified,
        resident_parameters,
        already_resident_bytes: options.already_resident_bytes,
        retained_input,
        staging,
        backend_overhead: options.backend_overhead.clone(),
        loading_peak,
        executions: vec![execution_plan],
        budget: options.budget.clone(),
    };
    let host_bytes = add(host_parameters, token_bytes)?;
    let domains = match &options.placement {
        GenerationMemoryPlacement::Unified
        | GenerationMemoryPlacement::Host
        | GenerationMemoryPlacement::Unknown => {
            if matches!(options.placement, GenerationMemoryPlacement::Host) {
                domain.domain = MemoryDomain::Host;
            }
            if matches!(options.placement, GenerationMemoryPlacement::Unknown) {
                domain.backend_overhead =
                    MemoryBytes::unknown("physical host/device memory relationship unavailable");
            }
            // Host parameter backing and retained token IDs share this capacity.
            if host_parameters > 0 {
                domain.resident_parameters = MemoryBytes {
                    lower_bytes: domain.resident_parameters.lower_bytes.max(host_parameters),
                    upper_bytes: domain.resident_parameters.upper_bytes.map(|upper| add(upper, host_parameters)).transpose()?,
                    kind: eredu_core::ObservationKind::Estimated,
                    detail: "one shared backing at lower end; distinct host and execution parameter copies at upper end".into(),
                };
            }
            domain.retained_input = sum_bytes(domain.retained_input, token_bytes)?;
            vec![domain]
        }
        GenerationMemoryPlacement::Separate { device } => {
            domain.domain = MemoryDomain::Device(device.clone());
            let host_loading = match resources.stored_tensor_bytes.value() {
                Some(stored) => MemoryBytes::estimated(
                    host_bytes,
                    add(host_bytes, *stored)?,
                    "host source mapping and retained parameter store",
                ),
                None => MemoryBytes::unknown("host loading source bytes unavailable"),
            };
            let host = DomainMemoryPlan {
                domain: MemoryDomain::Host,
                resident_parameters: MemoryBytes::exact(host_parameters),
                already_resident_bytes: 0,
                retained_input: MemoryBytes::exact(token_bytes),
                staging: MemoryBytes::unknown(
                    "host-side native conversion and transfer scratch is not separately attributed",
                ),
                backend_overhead: MemoryBytes::exact(0),
                loading_peak: host_loading,
                executions: Vec::new(),
                budget: options.host_budget.clone(),
            };
            vec![host, domain]
        }
    };
    Ok(GenerationMemoryRequest {
        input: options.input,
        max_output_tokens: options.max_output_tokens,
        forecast_output_tokens: options.forecast_output_tokens,
        batch_size: options.batch_size,
        prefill_chunk_tokens: chunk_tokens,
        scalar_bytes,
        domains,
    })
}

fn sum_bytes(bytes: MemoryBytes, extra: u64) -> Result<MemoryBytes, CapabilityError> {
    Ok(MemoryBytes {
        lower_bytes: add(bytes.lower_bytes, extra)?,
        upper_bytes: bytes
            .upper_bytes
            .map(|upper| add(upper, extra))
            .transpose()?,
        kind: bytes.kind,
        detail: format!("{}; includes retained token IDs", bytes.detail),
    })
}
