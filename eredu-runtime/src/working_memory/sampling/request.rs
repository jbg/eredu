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
        }
    }
}
impl<'a> From<WorkspaceSamplingInput<'a>> for WorkspaceSamplingSource<'a> {
    fn from(input: WorkspaceSamplingInput<'a>) -> Self {
        // The legacy descriptor supplies total bytes, not an allocation count.
        // Preserve its ordinary byte equation while keeping new native rounding
        // evidence unknown unless the input is an observed empty population.
        let count = (input.backing_capacity_bytes == Some(0)).then_some(0);
        Self { input, maximum_allocations: count }
    }
}
impl<'a> From<&'a WorkspaceLayout> for WorkspaceSamplingSource<'a> {
    fn from(layout: &'a WorkspaceLayout) -> Self {
        WorkspaceSamplingInput::from(layout)
            .with_backing_population(usize::from(layout.bytes().ok() != Some(0)))
    }
}

/// Sampling contribution across an entire output allowance. This includes
/// retained random keys, every emitted token buffer, history replacement overlap
/// and the supplied filter payload. Input logits, constraint-controller state,
/// speculative proposals and retained observations belong to enclosing quotes.
#[derive(Debug, Clone)]
pub struct SamplingWorkspaceReport {
    /// Exact selected score-row width used to validate and price the filter.
    pub output_width: usize,
    /// Number of ordinary committed-token sampling invocations inspected.
    pub steps: u64,
    /// Sum of the separate native-buffer and host peaks, or an explicit gap.
    /// Holding the host envelope for the sampler's lifetime therefore still
    /// leaves room for later emitted buffers, even when the peaks differ.
    pub peak: WorkspaceBound,
    /// Native buffer peak, including both old and newly split random keys.
    pub tensor_peak_bytes: Option<u64>,
    /// Managed host peak, including all live and replacement history payloads.
    pub host_peak_bytes: Option<u64>,
    /// First invocation whose native or host mechanism lacks a bound. Zero also
    /// identifies missing initialization coverage, including an unused seed key.
    pub first_gap: Option<u64>,
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
        return Err(Error::backend(
            "configured sampling requires one score row of rank 1, 2 or 3",
        ));
    }
    if !temperature.is_finite() || temperature < 0.0 {
        return Err(Error::backend(
            "sampling temperature must be finite and nonnegative",
        ));
    }
    filter
        .validate_output_width(vocabulary)
        .map_err(Error::backend)?;
    context.validate_values(random.into_iter().map(|state| state.key()))?;
    sampler.validate_steps(steps)?;
    let filter_bytes = filter.mask_capacity_bytes().map_err(Error::backend)?;
    let fixed = bytes_add(
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
    let initial_closing = if observer.is_some() {
        Some(
            context
                .report_scalars(
                    random
                        .as_ref()
                        .map(|state| std::slice::from_ref(state.key()))
                        .unwrap_or(&[]),
                )?
                .closing_storage,
        )
    } else {
        None
    };
    let initial = context.finish_report(&[])?;
    if let Some(observer) = observer.as_deref_mut() {
        observer.observe_with_storage(
            SamplingWorkspacePhase::Preparation,
            &initial,
            initial_closing.expect("observed closing roots"),
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
        .map(|(new, old)| bytes_add(new, old))
        .transpose()?;
    let initial_host = initial
        .host_workspace_bytes
        .map(|bytes| {
            bytes_add(
                bytes,
                bytes_add(initial_fixed, history_bytes(sampler.history_capacity())?)?,
            )
        })
        .transpose()?;
    let initial_peak = initial_tensor
        .zip(initial_host)
        .map(|(tensor, host)| bytes_add(tensor, host))
        .transpose()?;
    let mut report = SamplingWorkspaceReport {
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
        tensor_peak_bytes: initial_tensor,
        host_peak_bytes: initial_host,
        first_gap: initial_peak.is_none().then_some(0),
        final_history_bytes: history_bytes(sampler.history_capacity())?,
    };
    let mut known_peak = initial_peak.unwrap_or(0);
    let mut emitted = Vec::new();
    for index in 0..steps {
        let old_history = history_bytes(sampler.history_capacity())?;
        context.begin_state_span(random.iter().map(|state| state.key()).chain(emitted.iter()))?;
        let storage = match source.maximum_allocations {
            Some(maximum_allocations) => WorkspaceExistingStorage::try_new_population(
                eredu_nn::workspace::WorkspaceStoragePopulation {
                    bytes: logits.backing_capacity_bytes, maximum_allocations,
                }, context)?,
            None => WorkspaceExistingStorage::try_new(logits.backing_capacity_bytes, context)?,
        };
        let input = WorkspaceTensor::existing_with_storage(logits.layout.clone(), &storage, context)?;
        let filtered = apply_workspace_token_filter(&input, filter, context)?;
        let token = sampler.sample(&filtered, temperature, random.as_mut(), context)?;
        // No roots are excluded: this component prices its complete retained
        // state as well as its working buffers. Opening roots include all prior
        // token outputs and the old key, deduplicated by backing identity. Old
        // keys remain live through completion even when the next key replaces
        // them; an emitted alias can keep that backing alive for later spans.
        context.reserve_metadata_vec(&mut emitted, 1)?;
        emitted.push(token);
        let closing = if observer.is_some() {
            // Temporarily borrow the same existing output directory for the
            // optional RNG root; no numerical source or history is copied.
            if let Some(random) = &random {
                context.reserve_metadata_vec(&mut emitted, 1)?;
                emitted.push(random.key().clone());
            }
            let closing = context.report_scalars(&emitted);
            if random.is_some() {
                emitted.pop();
            }
            let mut closing = closing?.closing_storage;
            if source.maximum_allocations.is_none() { closing.bytes = None; }
            Some(closing)
        } else {
            None
        };
        let trace = context.finish_report(&[])?;
        if let Some(observer) = observer.as_deref_mut() {
            observer.observe_with_storage(
                SamplingWorkspacePhase::Step { index },
                &trace,
                closing.expect("observed closing roots"),
            )?;
        }
        let new_history = history_bytes(sampler.history_capacity())?;
        let history = if old_history == new_history {
            new_history
        } else {
            bytes_add(old_history, new_history)?
        };
        let host = trace
            .host_workspace_bytes
            .map(|bytes| bytes_add(bytes, bytes_add(fixed, history)?))
            .transpose()?;
        let tensor = trace
            .tensor_buffers
            .total_bytes
            .zip(trace.state.as_ref().and_then(|state| state.displaced_bytes))
            .map(|(new, old)| bytes_add(new, old))
            .transpose()?;
        report.tensor_peak_bytes = maximum(report.tensor_peak_bytes, tensor);
        report.host_peak_bytes = maximum(report.host_peak_bytes, host);
        match tensor
            .zip(host)
            .map(|(tensor, host)| bytes_add(tensor, host))
            .transpose()?
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
            None if report.first_gap.is_none() => {
                report.first_gap = Some(index);
                report.peak = WorkspaceBound::Unknown {
                    reason: context.metadata_string(format_args!(
                        "sampling invocation {index} lacks complete key, tensor or host bounds"
                    ))?,
                };
            }
            _ => {}
        }
        report.final_history_bytes = new_history;
    }
    if let (None, Some(tensor), Some(host)) = (
        report.first_gap,
        report.tensor_peak_bytes,
        report.host_peak_bytes,
    ) {
        let combined = tensor.checked_add(host).ok_or_else(|| {
            Error::backend_source(crate::working_memory::WorkingMemoryError::Overflow)
        })?;
        if let WorkspaceBound::Bounded { bytes, assumptions } = &mut report.peak {
            *bytes = combined;
            *assumptions = context.metadata_string(format_args!(
                "{}; separate tensor and host maxima are added conservatively so the complete host envelope, including boxed history replacement overlap, can stay held while later emitted buffers accumulate", assumptions,
            ))?;
        }
    }
    Ok(report)
}

fn maximum(old: Option<u64>, next: Option<u64>) -> Option<u64> {
    old.zip(next).map(|(old, next)| old.max(next))
}
fn history_bytes(capacity: usize) -> Result<u64, Error> {
    (capacity as u64)
        .checked_mul(4)
        .ok_or_else(|| Error::backend("sampling history byte overflow"))
}
fn bytes_add(left: u64, right: u64) -> Result<u64, Error> {
    left.checked_add(right)
        .ok_or_else(|| Error::backend("sampling workspace byte overflow"))
}

// The production cursor holds no numerical history. A test-only real sampler
// cursor exercises the established mechanism as an independent parity oracle.
trait QuoteSampler {
    fn history_capacity(&self) -> usize;
    fn validate_steps(&self, steps: u64) -> Result<(), Error>;
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
    fn validate_steps(&self, steps: u64) -> Result<(), Error> {
        SamplerWorkspaceProjection::validate_steps(self, steps).map_err(Error::backend_source)
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
        self.advance(probability).map_err(Error::backend_source)?;
        Ok(token)
    }
}

#[cfg(test)]
impl QuoteSampler for ConfiguredTextSampler {
    fn history_capacity(&self) -> usize {
        ConfiguredTextSampler::history_capacity(self)
    }
    fn validate_steps(&self, steps: u64) -> Result<(), Error> {
        self.workspace_projection()
            .validate_steps(steps)
            .map_err(Error::backend_source)
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
            ) => metadata.bounded(
                old.checked_add(*sampling)
                    .ok_or(AdmissionPolicyError::ArithmeticOverflow {
                        operation: "sampling and enclosing vocabulary workspace",
                    })?,
                format_args!("{assumptions}; {sampling_assumptions}"),
            )?,
            (WorkspaceBound::Unknown { .. }, _) => outside.vocabulary,
            (_, unknown @ WorkspaceBound::Unknown { .. }) => metadata.clone_bound(unknown)?,
        };
        Ok(outside)
    }
}
