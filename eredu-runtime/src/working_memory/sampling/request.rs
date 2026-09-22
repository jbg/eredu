//! Complete output-allowance sampling inspection using the execution policy.

use super::*;
use crate::ConfiguredTextSampler;
use crate::generation::SamplerWorkspaceProjection;
use eredu_core::WorkspaceBound;

/// Score geometry and an upper bound on all backing reachable from one score
/// value. Each output step introduces a fresh score identity: escaped aliases
/// must not make independently produced later scores look like shared storage.
/// Current scores belong to the enclosing equation quote; prior scores retained
/// by emitted tokens are counted here. This is no reservation or residual credit.
#[derive(Debug, Clone, Copy)]
pub struct WorkspaceSamplingInput<'a> {
    /// Logical score-row geometry used by the selected sampling mechanisms.
    pub layout: &'a WorkspaceLayout,
    /// Complete provider-described backing envelope, including possible aliases.
    /// Unknown remains unknown when an escaped token retains that backing. A
    /// broadcast view may have fewer physical bytes than its logical geometry.
    pub backing_capacity_bytes: Option<u64>,
}

impl<'a> From<&'a WorkspaceLayout> for WorkspaceSamplingInput<'a> {
    /// Declares packed input storage whose capacity equals its logical bytes.
    /// Native callers with padding, views or uncertain backing must supply an
    /// explicit descriptor instead of relying on this packed-storage contract.
    fn from(layout: &'a WorkspaceLayout) -> Self {
        Self {
            layout,
            backing_capacity_bytes: layout.bytes().ok(),
        }
    }
}

/// An input descriptor plus the maximum backing count of its existing source
/// envelope. This is metadata geometry and never native allocation authority.
/// Legacy byte-only inputs keep their ordinary equation and unknown native count;
/// the layout adapter retains its explicit single packed-backing contract.
#[derive(Clone, Copy, Debug)]
pub struct WorkspaceSamplingSource<'a> {
    input: WorkspaceSamplingInput<'a>,
    maximum_allocations: Option<usize>,
    physical_domains: Option<&'a eredu_core::DomainMemoryRequirements>,
}
impl<'a> WorkspaceSamplingSource<'a> {
    /// Placement established by the actual score producer's backing inventory.
    /// It applies to every allocation in this complete backing envelope.
    pub fn with_physical_domains(
        mut self,
        physical_domains: Option<&'a eredu_core::DomainMemoryRequirements>,
    ) -> Self {
        self.physical_domains = physical_domains;
        self
    }
}
impl<'a> WorkspaceSamplingInput<'a> {
    /// Preserve the actual source's complete possible backing count when its
    /// byte envelope collapses multiple aliases into one metadata input.
    pub fn with_backing_population(
        self,
        maximum_allocations: usize,
    ) -> WorkspaceSamplingSource<'a> {
        WorkspaceSamplingSource {
            input: self,
            maximum_allocations: Some(maximum_allocations),
            physical_domains: None,
        }
    }
}
impl<'a> From<WorkspaceSamplingInput<'a>> for WorkspaceSamplingSource<'a> {
    fn from(input: WorkspaceSamplingInput<'a>) -> Self {
        // The legacy descriptor supplies total bytes, not an allocation count.
        // Preserve its ordinary byte equation while keeping new native rounding
        // evidence unknown unless the input is an observed empty population.
        let count = (input.backing_capacity_bytes == Some(0)).then_some(0);
        Self {
            input,
            maximum_allocations: count,
            physical_domains: None,
        }
    }
}
impl<'a> From<&'a WorkspaceLayout> for WorkspaceSamplingSource<'a> {
    fn from(layout: &'a WorkspaceLayout) -> Self {
        WorkspaceSamplingInput::from(layout)
            .with_backing_population(usize::from(layout.bytes().ok() != Some(0)))
    }
}

/// Exact score descriptor supplied to the shared sampling worker. Its checked
/// one-row rank is at most three, so retention needs no allocated shape buffer.
/// These descriptive facts supply neither source custody nor execution rights.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SamplingWorkspaceInputPlan {
    shape: [i32; 3],
    rank: usize,
    representation: Option<eredu_nn::workspace::WorkspaceRepresentation>,
    backing_capacity_bytes: Option<u64>,
    maximum_allocations: Option<usize>,
    physical_domains: Option<std::sync::Arc<eredu_core::DomainMemoryRequirements>>,
}
impl SamplingWorkspaceInputPlan {
    /// Actual source dimensions, without normalizing or guessing its rank.
    pub fn shape(&self) -> &[i32] {
        &self.shape[..self.rank]
    }
    /// Reconstructs the observed geometry and physical evidence under the
    /// caller's metadata account. Unknown representation stays unknown.
    pub fn layout(&self, context: &WorkspaceContext) -> Result<WorkspaceLayout, Error> {
        Ok(context
            .layout(self.shape(), WorkspaceDtype::Float32)?
            .with_representation(self.representation))
    }
    /// Reconstructs the same source descriptor over a separately paid layout.
    pub fn source<'a>(
        &'a self,
        layout: &'a WorkspaceLayout,
    ) -> Result<WorkspaceSamplingSource<'a>, Error> {
        if layout.shape() != self.shape()
            || layout.dtype() != WorkspaceDtype::Float32
            || layout.representation() != self.representation
        {
            return Err(WorkspaceMetadataError::Unqualified.into());
        }
        Ok(WorkspaceSamplingSource {
            input: WorkspaceSamplingInput {
                layout,
                backing_capacity_bytes: self.backing_capacity_bytes,
            },
            maximum_allocations: self.maximum_allocations,
            physical_domains: self.physical_domains.as_deref(),
        })
    }
}

/// Sampling contribution across an entire output allowance. This includes
/// retained random keys, every emitted token buffer, history replacement overlap
/// and the supplied filter payload. Input logits, constraint-controller state,
/// speculative proposals and retained observations belong to enclosing quotes.
#[derive(Debug, Clone)]
pub struct SamplingWorkspaceReport {
    input: SamplingWorkspaceInputPlan,
    /// Exact selected score-row width used to validate and price the filter.
    pub output_width: usize,
    /// Number of ordinary committed-token sampling invocations inspected.
    pub steps: u64,
    /// Sum of the separate native-buffer and host peaks, or an explicit gap.
    /// Holding the host envelope for the sampler's lifetime therefore still
    /// leaves room for later emitted buffers, even when the peaks differ.
    pub peak: WorkspaceBound,
    /// Complete physical envelope from each phase's original allocation walk.
    /// The held host peak is added after reducing native lifetimes by domain.
    pub physical_domains: Option<eredu_core::DomainMemoryRequirements>,
    /// Native buffer peak, including both old and newly split random keys.
    pub tensor_peak_bytes: Option<u64>,
    /// Managed host peak, including all live and replacement history payloads.
    pub host_peak_bytes: Option<u64>,
    /// First invocation whose native or host mechanism lacks a bound. Zero also
    /// identifies missing initialization coverage, including an unused seed key.
    pub first_gap: Option<u64>,
    /// Maximum backing population of the actual closing random state and all
    /// emitted outputs. Missing score-source population remains unknown.
    pub maximum_closing_storage_allocations: Option<usize>,
    /// Final retained history payload, including spare capacity.
    pub final_history_bytes: u64,
}

/// Actual phase of the shared sampling quotation, including an empty
/// deterministic preparation and every permitted committed prediction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SamplingWorkspacePhase {
    /// Initial random-key creation or existing-state inspection.
    Preparation,
    /// Zero-based committed-token invocation.
    Step { index: u64 },
}

/// Borrows the actual sampling trace before the shared driver releases it.
/// Mechanisms may reduce this trace to native recipes; observation grants no
/// allocation authority or completeness beyond the operations actually seen.
pub trait SamplingWorkspaceObserver {
    /// Observes the actual score source before preparation/step traces. The
    /// default is appropriate for consumers that do not later reprice sampling.
    fn observe_input(&mut self, _input: SamplingWorkspaceInputPlan) -> Result<(), Error> {
        Ok(())
    }
    /// The report includes existing roots, retained prior outputs and the
    /// operations for this phase. Distinguish new producers from those roots
    /// when computing cumulative generations or replacement contributions.
    fn observe(
        &mut self,
        phase: SamplingWorkspacePhase,
        report: &WorkspaceTraceReport,
    ) -> Result<(), Error>;

    /// The same phase with its actual closing random/output backing union.
    /// Existing callbacks remain compatible; this is not native liveness proof.
    fn observe_with_storage(
        &mut self,
        phase: SamplingWorkspacePhase,
        report: &WorkspaceTraceReport,
        _closing: eredu_nn::workspace::WorkspaceStoragePopulation,
    ) -> Result<(), Error> {
        self.observe(phase, report)
    }
}

/// Traces the configured sampling policy for every possible committed output
/// position. The source is borrowed; only history extents and adaptive scalar
/// witnesses advance. No history payload is copied or allocated. The random
/// state clone contains workspace metadata only, never a native key. `logits` describes a single score row already priced by the
/// architecture trace. `filter` describes exact or optional filtering without
/// constructing a mask. A changing controller still needs its enclosing state
/// and decision-copy bound; this description grants no mutation authority.
///
/// Existing random keys require a projection with observed backing capacity;
/// fresh keys can use [`WorkspaceSamplingRandomState::from_seed`]. Unknown
/// backing, native operations or host payloads make the result unknown. No
/// native tensor, stream or completion object is constructed or evaluated.
/// Emitted outputs remain live through every later span because callers may
/// retain all returned token handles. Shared backing is charged once by identity.
pub fn quote_sampling_workspace<'a, 'b>(
    sampler: &ConfiguredTextSampler,
    temperature: f32,
    random: Option<&WorkspaceSamplingRandomState>,
    logits: impl Into<WorkspaceSamplingSource<'a>>,
    filter: impl Into<TextFilterWorkspace<'b>>,
    steps: u64,
    context: &WorkspaceContext,
) -> Result<SamplingWorkspaceReport, Error> {
    quote_sampling_workspace_with_observer(
        sampler,
        temperature,
        random,
        logits,
        filter,
        steps,
        context,
        None,
    )
}

/// Runs the same sampling quote while optionally borrowing each completed trace.
/// The observer receives the real configured/borrowed policy's operations, so a
/// greedy temperature alone never certifies absence of filters or penalties.
#[allow(clippy::too_many_arguments)]
pub fn quote_sampling_workspace_with_observer<'a, 'b>(
    sampler: &ConfiguredTextSampler,
    temperature: f32,
    random: Option<&WorkspaceSamplingRandomState>,
    logits: impl Into<WorkspaceSamplingSource<'a>>,
    filter: impl Into<TextFilterWorkspace<'b>>,
    steps: u64,
    context: &WorkspaceContext,
    observer: Option<&mut dyn SamplingWorkspaceObserver>,
) -> Result<SamplingWorkspaceReport, Error> {
    quote_with_sampler(
        sampler.workspace_projection(),
        temperature,
        random,
        logits.into(),
        filter.into(),
        steps,
        context,
        observer,
    )
}

#[allow(clippy::too_many_arguments)]
fn quote_with_sampler(
    mut sampler: impl QuoteSampler,
    temperature: f32,
    random: Option<&WorkspaceSamplingRandomState>,
    source: WorkspaceSamplingSource<'_>,
    filter: TextFilterWorkspace<'_>,
    steps: u64,
    context: &WorkspaceContext,
    mut observer: Option<&mut dyn SamplingWorkspaceObserver>,
) -> Result<SamplingWorkspaceReport, Error> {
    let logits = source.input;
    let vocabulary = logits_layout_width(logits.layout)?;
    if logits.layout.elements()? != vocabulary as u64 || logits.layout.shape().len() > 3 {
        return Err(context.metadata_error(format_args!(
            "configured sampling requires one score row of rank 1, 2 or 3",
        )));
    }
    context.charge_metadata(
        std::mem::size_of::<SamplingWorkspaceInputPlan>() + std::mem::size_of::<[i32; 3]>(),
    )?;
    let mut shape = [0; 3];
    shape[..logits.layout.shape().len()].copy_from_slice(logits.layout.shape());
    let input_plan = SamplingWorkspaceInputPlan {
        shape,
        rank: logits.layout.shape().len(),
        representation: logits.layout.representation(),
        backing_capacity_bytes: logits.backing_capacity_bytes,
        maximum_allocations: source.maximum_allocations,
        physical_domains: source
            .physical_domains
            .map(|requirements| {
                context.charge_metadata(2 * std::mem::size_of::<usize>())?;
                let metadata = super::super::WorkspaceReportMetadata::new(context);
                let retained = metadata
                    .clone_domain_requirements(requirements)
                    .map_err(|e| metadata.error(e))?;
                Ok::<_, Error>(std::sync::Arc::new(retained))
            })
            .transpose()?,
    };
    if let Some(observer) = observer.as_deref_mut() {
        observer.observe_input(input_plan.clone())?;
    }
    if !temperature.is_finite() || temperature < 0.0 {
        return Err(context.metadata_error(format_args!(
            "sampling temperature must be finite and nonnegative",
        )));
    }
    filter
        .validate_output_width(vocabulary)
        .map_err(|cause| context.metadata_source(cause))?;
    context.validate_values(random.into_iter().map(|state| state.key()))?;
    sampler.validate_steps(steps, context)?;
    let filter_bytes = filter
        .mask_capacity_bytes()
        .map_err(|cause| context.metadata_source(cause))?;
    let fixed = bytes_add(
        context,
        std::mem::size_of::<ConfiguredTextSampler>() as u64,
        filter_bytes,
    )?;
    let initial_fixed = match filter {
        TextFilterWorkspace::Exact(_) => fixed,
        // No optional mask exists during initialization, including a request
        // with zero output allowance. Only emitted decisions need this buffer.
        TextFilterWorkspace::OptionalMask { .. } => {
            std::mem::size_of::<ConfiguredTextSampler>() as u64
        }
    };
    let mut random = random.cloned();
    let fresh = random.as_ref().is_some_and(|state| state.fresh);
    context.begin_state_span(random.iter().filter(|_| !fresh).map(|state| state.key()))?;
    if fresh {
        // Inspect initialization as its own completed span, preserving its
        // scratch and host facts as well as the retained key capacity. The
        // caller's metadata key is untouched.
        random = Some(WorkspaceSamplingRandomState::from_seed(context)?);
    }
    let initial_closing = context
        .report_scalars(
            random
                .as_ref()
                .map(|state| std::slice::from_ref(state.key()))
                .unwrap_or(&[]),
        )?
        .closing_storage;
    let initial = context.finish_report(&[])?;
    if let Some(observer) = observer.as_deref_mut() {
        observer.observe_with_storage(
            SamplingWorkspacePhase::Preparation,
            &initial,
            initial_closing,
        )?;
    }
    let initial_tensor = initial
        .tensor_buffers
        .total_bytes
        .zip(
            initial
                .state
                .as_ref()
                .and_then(|state| state.displaced_bytes),
        )
        .map(|(new, old)| diagnostic_add(context, new, old, initial.physical_domains.is_some()))
        .transpose()?
        .flatten();
    let initial_host = initial
        .host_workspace_bytes
        .map(|bytes| {
            bytes_add(
                context,
                bytes,
                bytes_add(
                    context,
                    initial_fixed,
                    history_bytes(context, sampler.history_capacity())?,
                )?,
            )
        })
        .transpose()?;
    let initial_peak = initial_tensor
        .zip(initial_host)
        .map(|(tensor, host)| {
            diagnostic_add(context, tensor, host, initial.physical_domains.is_some())
        })
        .transpose()?
        .flatten();
    let mut native_domains = sampling_native_domains(&initial, context)?;
    let mut report = SamplingWorkspaceReport {
        input: input_plan,
        output_width: vocabulary,
        steps,
        peak: match initial_peak {
            Some(bytes) => WorkspaceBound::bounded(
                bytes,
                context.metadata_string(format_args!(
                    "initial sampling state and supplied filter payload"
                ))?,
            ),
            None => WorkspaceBound::Unknown {
                reason: context.metadata_string(format_args!(
                    "initial sampling key, scratch or host payload is unknown"
                ))?,
            },
        },
        physical_domains: None,
        tensor_peak_bytes: initial_tensor,
        host_peak_bytes: initial_host,
        first_gap: (initial_peak.is_none() && native_domains.is_none()).then_some(0),
        maximum_closing_storage_allocations: Some(initial_closing.maximum_allocations),
        final_history_bytes: history_bytes(context, sampler.history_capacity())?,
    };
    let mut known_peak = initial_peak.unwrap_or(0);
    let mut emitted = Vec::new();
    for index in 0..steps {
        let old_history = history_bytes(context, sampler.history_capacity())?;
        context.begin_state_span(random.iter().map(|state| state.key()).chain(emitted.iter()))?;
        let input = sampling_score_input(source, context)?;
        let filtered = apply_workspace_token_filter(&input, filter, context)?;
        let token = sampler.sample(&filtered, temperature, random.as_mut(), context)?;
        // No roots are excluded: this component prices its complete retained
        // state as well as its working buffers. Opening roots include all prior
        // token outputs and the old key, deduplicated by backing identity. Old
        // keys remain live through completion even when the next key replaces
        // them; an emitted alias can keep that backing alive for later spans.
        context.reserve_metadata_vec(&mut emitted, 1)?;
        emitted.push(token);
        // This is the same closing-root lifetime walk used by native recipe
        // observers. Publication population remains available without an observer.
        if let Some(random) = &random {
            context.reserve_metadata_vec(&mut emitted, 1)?;
            emitted.push(random.key().clone());
        }
        let closing = context.report_scalars(&emitted);
        if random.is_some() {
            emitted.pop();
        }
        let mut closing = closing?.closing_storage;
        if source.maximum_allocations.is_none() {
            closing.bytes = None;
            report.maximum_closing_storage_allocations = None;
        } else {
            report.maximum_closing_storage_allocations = report
                .maximum_closing_storage_allocations
                .map(|count| count.max(closing.maximum_allocations));
        }
        let trace = context.finish_report(&[])?;
        native_domains = match (
            native_domains.as_ref(),
            sampling_native_domains(&trace, context)?,
        ) {
            (Some(previous), Some(next)) => Some(
                super::super::WorkspaceReportMetadata::new(context)
                    .combine_domain_requirements(previous, &next, false)
                    .map_err(|e| super::super::WorkspaceReportMetadata::new(context).error(e))?,
            ),
            _ => None,
        };
        if let Some(observer) = observer.as_deref_mut() {
            observer.observe_with_storage(
                SamplingWorkspacePhase::Step { index },
                &trace,
                closing,
            )?;
        }
        let new_history = history_bytes(context, sampler.history_capacity())?;
        let history = if old_history == new_history {
            new_history
        } else {
            bytes_add(context, old_history, new_history)?
        };
        let host = trace
            .host_workspace_bytes
            .map(|bytes| bytes_add(context, bytes, bytes_add(context, fixed, history)?))
            .transpose()?;
        let tensor = trace
            .tensor_buffers
            .total_bytes
            .zip(trace.state.as_ref().and_then(|state| state.displaced_bytes))
            .map(|(new, old)| diagnostic_add(context, new, old, trace.physical_domains.is_some()))
            .transpose()?
            .flatten();
        report.tensor_peak_bytes = maximum(report.tensor_peak_bytes, tensor);
        report.host_peak_bytes = maximum(report.host_peak_bytes, host);
        match tensor
            .zip(host)
            .map(|(tensor, host)| {
                diagnostic_add(context, tensor, host, trace.physical_domains.is_some())
            })
            .transpose()?
            .flatten()
        {
            Some(bytes) if report.first_gap.is_none() && bytes >= known_peak => {
                known_peak = bytes;
                report.peak = WorkspaceBound::bounded(
                    bytes,
                    context.metadata_string(format_args!(
                        "inspected {steps} completed configured sampling invocations; includes random-key replacement, all previously emitted token backing retained by identity, current emitted token, exact boxed history growth overlap and supplied filter payload; excludes input logits and enclosing controllers/observations; {}",
                        Assumptions(&trace.assumptions),
                    ))?,
                );
            }
            None if report.first_gap.is_none() && native_domains.is_none() => {
                report.first_gap = Some(index);
                report.peak = WorkspaceBound::Unknown {
                    reason: context.metadata_string(format_args!(
                        "sampling invocation {index} lacks complete key, tensor or host bounds"
                    ))?,
                };
            }
            None if native_domains.is_some() => {
                report.peak = WorkspaceBound::PerDomain { assumptions: context.metadata_string(format_args!(
                    "aggregate sampling diagnostic is unavailable; independently checked physical-domain requirements remain complete"))? };
            }
            _ => {}
        }
        report.final_history_bytes = new_history;
    }
    report.physical_domains = match (
        native_domains,
        report.host_peak_bytes,
        context.memory_topology(),
    ) {
        (Some(mut native), Some(host), Some(topology)) => {
            native
                .add_allocation(
                    host,
                    &eredu_core::MemoryPlacement::fixed(topology, topology.host_domain())
                        .map_err(|e| context.metadata_source(e))?,
                )
                .map_err(|e| context.metadata_source(e))?;
            Some(native)
        }
        _ => None,
    };
    if let (None, Some(tensor), Some(host)) = (
        report.first_gap,
        report.tensor_peak_bytes,
        report.host_peak_bytes,
    ) {
        let combined = diagnostic_add(context, tensor, host, report.physical_domains.is_some())?;
        if let (Some(combined), WorkspaceBound::Bounded { bytes, assumptions }) =
            (combined, &mut report.peak)
        {
            *bytes = combined;
            *assumptions = context.metadata_string(format_args!(
                "{}; separate tensor and host maxima are added conservatively so the complete host envelope, including boxed history replacement overlap, can stay held while later emitted buffers accumulate", assumptions,
            ))?;
        } else if combined.is_none() {
            report.peak = WorkspaceBound::PerDomain { assumptions: context.metadata_string(format_args!(
                "aggregate sampling diagnostic exceeds u64; physical requirements remain separately attributed"))? };
        }
    }
    Ok(report)
}

fn sampling_score_input(
    source: WorkspaceSamplingSource<'_>,
    context: &WorkspaceContext,
) -> Result<WorkspaceTensor, Error> {
    let Some(requirements) = source.physical_domains else {
        let storage = match source.maximum_allocations {
            Some(maximum_allocations) => WorkspaceExistingStorage::try_new_population(
                eredu_nn::workspace::WorkspaceStoragePopulation {
                    bytes: source.input.backing_capacity_bytes,
                    maximum_allocations,
                },
                context,
            )?,
            None => {
                WorkspaceExistingStorage::try_new(source.input.backing_capacity_bytes, context)?
            }
        };
        return WorkspaceTensor::existing_with_storage(
            source.input.layout.clone(),
            &storage,
            context,
        );
    };
    let topology = context
        .memory_topology()
        .ok_or(WorkspaceMetadataError::Unqualified)?;
    requirements
        .validate(topology)
        .map_err(|e| context.metadata_source(e))?;
    let maximum_allocations = source
        .maximum_allocations
        .ok_or(WorkspaceMetadataError::Unqualified)?;
    let count = requirements
        .iter()
        .len()
        .checked_mul(2)
        .ok_or(WorkspaceMetadataError::Overflow)?;
    let mut roots = context.metadata_vec(count)?;
    for (domain, charge) in requirements.iter() {
        if charge.estimated_overhead_bytes != 0 || charge.headroom_bytes != 0 {
            return Err(WorkspaceMetadataError::Unqualified.into());
        }
        if charge.accounted_bytes != 0 {
            let placement = eredu_core::MemoryPlacement::fixed(topology, domain)
                .map_err(|e| context.metadata_source(e))?;
            roots.push(WorkspaceExistingStorage::try_new_population_placed(
                eredu_nn::workspace::WorkspaceStoragePopulation {
                    bytes: Some(charge.accounted_bytes),
                    maximum_allocations,
                },
                &placement,
                context,
            )?);
        }
        if charge.placement_allowance_bytes != 0 {
            let mut candidates = context.metadata_vec(1)?;
            candidates.push(domain);
            let basis = context.metadata_string(format_args!(
                "prospective score-backing envelope from the actual completed-span per-domain reducer; original candidate-placement evidence: {:?}",
                requirements.placement_allowances()))?;
            let placement = eredu_core::MemoryPlacement::possible(topology, candidates, basis)
                .map_err(|e| context.metadata_source(e))?;
            roots.push(WorkspaceExistingStorage::try_new_population_placed(
                eredu_nn::workspace::WorkspaceStoragePopulation {
                    bytes: Some(charge.placement_allowance_bytes),
                    maximum_allocations,
                },
                &placement,
                context,
            )?);
        }
    }
    WorkspaceTensor::existing_with_storages(source.input.layout.clone(), &roots, context)
}

fn sampling_native_domains(
    trace: &WorkspaceTraceReport,
    context: &WorkspaceContext,
) -> Result<Option<eredu_core::DomainMemoryRequirements>, Error> {
    let (Some(domains), Some(topology), Some(host)) = (
        &trace.physical_domains,
        context.memory_topology(),
        trace.host_workspace_bytes,
    ) else {
        return Ok(None);
    };
    let Some(transient) = &domains.state_transient else {
        return Ok(None);
    };
    let metadata = super::super::WorkspaceReportMetadata::new(context);
    let mut native = metadata
        .clone_domain_requirements(transient)
        .map_err(|e| metadata.error(e))?;
    native
        .subtract_accounted(topology.host_domain(), host)
        .map_err(|e| context.metadata_source(e))?;
    Ok(Some(native))
}

fn maximum(old: Option<u64>, next: Option<u64>) -> Option<u64> {
    old.zip(next).map(|(old, next)| old.max(next))
}
fn history_bytes(context: &WorkspaceContext, capacity: usize) -> Result<u64, Error> {
    (capacity as u64)
        .checked_mul(4)
        .ok_or_else(|| context.metadata_error(format_args!("sampling history byte overflow")))
}
fn bytes_add(context: &WorkspaceContext, left: u64, right: u64) -> Result<u64, Error> {
    left.checked_add(right)
        .ok_or_else(|| context.metadata_error(format_args!("sampling workspace byte overflow")))
}
fn diagnostic_add(
    context: &WorkspaceContext,
    left: u64,
    right: u64,
    attributed: bool,
) -> Result<Option<u64>, Error> {
    match left.checked_add(right) {
        Some(bytes) => Ok(Some(bytes)),
        None if attributed => Ok(None),
        None => Err(context.metadata_source(super::super::WorkingMemoryError::Overflow)),
    }
}

// The production cursor holds no numerical history. A test-only real sampler
// cursor exercises the established mechanism as an independent parity oracle.
trait QuoteSampler {
    fn history_capacity(&self) -> usize;
    fn validate_steps(&self, steps: u64, context: &WorkspaceContext) -> Result<(), Error>;
    fn sample(
        &mut self,
        logits: &WorkspaceTensor,
        temperature: f32,
        random: Option<&mut WorkspaceSamplingRandomState>,
        context: &WorkspaceContext,
    ) -> Result<WorkspaceTensor, Error>;
}

impl QuoteSampler for SamplerWorkspaceProjection<'_> {
    fn history_capacity(&self) -> usize {
        SamplerWorkspaceProjection::history_capacity(self)
    }
    fn validate_steps(&self, steps: u64, context: &WorkspaceContext) -> Result<(), Error> {
        SamplerWorkspaceProjection::validate_steps(self, steps)
            .map_err(|cause| context.metadata_source(cause))
    }
    fn sample(
        &mut self,
        logits: &WorkspaceTensor,
        temperature: f32,
        random: Option<&mut WorkspaceSamplingRandomState>,
        context: &WorkspaceContext,
    ) -> Result<WorkspaceTensor, Error> {
        let (token, probability) = match self.source() {
            ConfiguredTextSampler::Standard(policy) => {
                let processed = policy.process_with::<WorkspaceSamplingBackend>(
                    logits,
                    context,
                    |logits, penalties, context| {
                        apply_penalties_extent(logits, self.history_len(), penalties, context)
                    },
                )?;
                let token =
                    WorkspaceSamplingBackend::sample_raw(&processed, temperature, random, context)?;
                let _ = WorkspaceSamplingBackend::token_id(&token, context)?;
                (token, None)
            }
            ConfiguredTextSampler::MirostatV2(policy) => {
                let processed = policy.process_with::<WorkspaceSamplingBackend>(
                    logits,
                    temperature,
                    self.mirostat_mu().expect("adaptive projection"),
                    context,
                    |logits, penalties, temperature, mu, context| {
                        apply_mirostat_extent(
                            logits,
                            self.history_len(),
                            penalties,
                            temperature,
                            mu,
                            context,
                        )
                    },
                )?;
                let token = WorkspaceSamplingBackend::sample_processed(
                    &processed,
                    temperature,
                    random,
                    context,
                )?;
                let token_id = WorkspaceSamplingBackend::token_id(&token, context)?;
                let probability =
                    WorkspaceSamplingBackend::token_probability(&processed, token_id, context)?;
                (token, Some(probability))
            }
        };
        self.advance(probability)
            .map_err(|cause| context.metadata_source(cause))?;
        Ok(token)
    }
}

#[cfg(test)]
impl QuoteSampler for ConfiguredTextSampler {
    fn history_capacity(&self) -> usize {
        ConfiguredTextSampler::history_capacity(self)
    }
    fn validate_steps(&self, steps: u64, context: &WorkspaceContext) -> Result<(), Error> {
        self.workspace_projection()
            .validate_steps(steps)
            .map_err(|cause| context.metadata_source(cause))
    }
    fn sample(
        &mut self,
        logits: &WorkspaceTensor,
        temperature: f32,
        random: Option<&mut WorkspaceSamplingRandomState>,
        context: &WorkspaceContext,
    ) -> Result<WorkspaceTensor, Error> {
        crate::Sampler::<WorkspaceSamplingBackend>::sample(
            self,
            logits,
            temperature,
            random,
            context,
        )
    }
}

#[cfg(test)]
pub(super) fn quote_sampling_workspace_legacy<'a, 'b>(
    sampler: &ConfiguredTextSampler,
    temperature: f32,
    random: Option<&WorkspaceSamplingRandomState>,
    logits: impl Into<WorkspaceSamplingSource<'a>>,
    filter: impl Into<TextFilterWorkspace<'b>>,
    steps: u64,
    context: &WorkspaceContext,
) -> Result<SamplingWorkspaceReport, Error> {
    quote_with_sampler(
        sampler.clone(),
        temperature,
        random,
        logits.into(),
        filter.into(),
        steps,
        context,
        None,
    )
}

// Pure borrowed formatting: no intermediate joined String or source clone.
struct Assumptions<'a>(&'a [String]);
impl std::fmt::Display for Assumptions<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (index, value) in self.0.iter().enumerate() {
            if index != 0 {
                f.write_str("; ")?;
            }
            f.write_str(value)?;
        }
        Ok(())
    }
}

impl SamplingWorkspaceReport {
    /// Exact original score geometry/backing facts retained by this report.
    pub fn input_plan(&self) -> SamplingWorkspaceInputPlan {
        self.input.clone()
    }
    /// Adds this actual sampling report to the enclosing vocabulary contribution.
    /// Equations, prompt preparation and controller storage remain separate.
    /// This is the same composition used by text and prepared-media consumers.
    pub fn enclosing_workspace_metadata(
        &self,
        mut outside: eredu_core::ExecutionWorkspaceEstimate,
        metadata: super::super::WorkspaceReportMetadata<'_>,
    ) -> Result<eredu_core::ExecutionWorkspaceEstimate, super::super::WorkspaceReportError> {
        use eredu_core::{AdmissionPolicyError, WorkspaceBound};
        metadata.admit::<eredu_core::ExecutionWorkspaceEstimate>()?;
        outside.physical_domains = match (outside.physical_domains.take(), &self.physical_domains) {
            (Some(mut outside), Some(sampling)) => {
                outside.vocabulary =
                    metadata.combine_domain_requirements(&outside.vocabulary, sampling, true)?;
                Some(outside)
            }
            _ => None,
        };
        outside.vocabulary = match (&outside.vocabulary, &self.peak) {
            (
                WorkspaceBound::Bounded {
                    bytes: old,
                    assumptions,
                },
                WorkspaceBound::Bounded {
                    bytes: sampling,
                    assumptions: sampling_assumptions,
                },
            ) => match old.checked_add(*sampling) {
                Some(bytes) => metadata
                    .bounded(bytes, format_args!("{assumptions}; {sampling_assumptions}"))?,
                None if outside.physical_domains.is_some() => {
                    metadata.per_domain(format_args!("{assumptions}; {sampling_assumptions}"))?
                }
                None => {
                    return Err(AdmissionPolicyError::ArithmeticOverflow {
                        operation: "sampling and enclosing vocabulary workspace",
                    }
                    .into());
                }
            },
            (WorkspaceBound::Unknown { .. }, _) => outside.vocabulary,
            (_, unknown @ WorkspaceBound::Unknown { .. }) => metadata.clone_bound(unknown)?,
            (bound @ WorkspaceBound::PerDomain { .. }, _) => metadata.clone_bound(bound)?,
            (_, bound @ WorkspaceBound::PerDomain { .. }) => metadata.clone_bound(bound)?,
        };
        Ok(outside)
    }
}
