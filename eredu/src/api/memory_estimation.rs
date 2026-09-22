//! Cold request memory planning composed from retained architecture selection.

use eredu_architectures::{ModelInspectionOutcome, PreparationMechanismProvider};
use eredu_core::{CapabilityError, InputTokenCount, MediaFeatureAvailability, Observed};
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
    /// Creates an estimate with explicit unknown backend mechanisms and overhead.
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
    let request = inspected_generation_memory_request(inspection, options)?;
    let mut report = estimate_generation_memory(&request)?;
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
    let state_layout = geometry.state_layout;
    if selected.preparation().prediction_realization().is_some() {
        workspace = None;
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
        attention: options.attention.clone(),
        workspace_overlap: options.workspace_overlap.clone(),
        cache_update: options.cache_update.clone(),
        logits: options.logits.clone(),
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
