//! Published source C and the existing numerical phase's shared capture workers.
use super::*;
use crate::backend::array_copy::{
    CandidateExtraction, CaptureTensorNativeError, CaptureTensorSelection,
    PreparedCaptureHistogram, PreparedCaptureSummary, PreparedCaptureTensor, TokenScoreProgram,
};
use eredu_core::{capture::*, checkpoint::TensorDtype};
use eredu_nn::{
    Tensor,
    workspace::{WorkspaceContext, WorkspaceTensor},
};
use eredu_runtime::capture::{
    CaptureObservationStep, FundedSpeculativeCaptureInvocation, ScheduledCaptureBackend,
};
use eredu_runtime::working_memory::{
    CaptureTensorClaim, ClaimedCaptureTensor, RegisteredInferenceSourceWitness,
    SpeculativeCaptureHostPlan,
};
use safemlx::{Array, OriginalScopeObserver, Stream};
use std::cell::RefCell;
mod interventions;
use interventions::Edits;

fn cold(
    context: &WorkspaceContext,
    cause: impl std::error::Error + Send + Sync + 'static,
) -> Error {
    Error::Neural(context.metadata_source(cause))
}

/// A borrowed exact previously published C source. No new source payload,
/// registration, callback or account is adopted by this adapter.
#[derive(Clone, Copy)]
pub(crate) struct CaptureSource<'a> {
    source: &'a SharedCapturePlan,
    proof: Proof<'a>,
    pool: &'a eredu_runtime::working_memory::WorkingMemoryPool,
    domain: Option<CaptureTokenDomain<'a>>,
    interventions: Option<&'a eredu_runtime::working_memory::OriginalInterventionSource>,
}
#[derive(Clone, Copy)]
enum Proof<'a> {
    Registered(&'a RegisteredInferenceSourceWitness),
    Original(&'a eredu_runtime::working_memory::OriginalCaptureSource),
}
enum OwnedProof {
    Registered(RegisteredInferenceSourceWitness),
    Original(eredu_runtime::working_memory::OriginalCaptureSource),
}
impl<'a> CaptureSource<'a> {
    pub(crate) fn new(
        source: &'a SharedCapturePlan,
        witness: &'a RegisteredInferenceSourceWitness,
        sources: &OriginalSpeculativeNumericalSources,
        pool: &'a eredu_runtime::working_memory::WorkingMemoryPool,
    ) -> Result<Self, Error> {
        sources
            .metadata_funding()
            .reserve_metadata(size_of::<Self>() + size_of::<Result<Self, Error>>())
            .map_err(Error::WorkspacePlanning)?;
        let value = Self {
            source,
            proof: Proof::Registered(witness),
            pool,
            domain: None,
            interventions: None,
        };
        value.validate(sources)?;
        Ok(value)
    }
    pub(crate) fn original(
        source: &'a eredu_runtime::working_memory::OriginalCaptureSource,
        sources: &OriginalSpeculativeNumericalSources,
        pool: &'a eredu_runtime::working_memory::WorkingMemoryPool,
    ) -> Result<Self, Error> {
        sources
            .metadata_funding()
            .reserve_metadata(size_of::<Self>() + size_of::<Result<Self, Error>>())
            .map_err(Error::WorkspacePlanning)?;
        let value = Self {
            source: source.plan(),
            proof: Proof::Original(source),
            pool,
            domain: None,
            interventions: None,
        };
        value.validate(sources)?;
        Ok(value)
    }
    /// The validated fixed controller supplies its actual pre-force filters.
    /// This borrow cannot outlive that decision and grants no source authority.
    pub(super) fn with_domain<'b>(self, domain: CaptureTokenDomain<'b>) -> CaptureSource<'b>
    where
        'a: 'b,
    {
        CaptureSource {
            source: self.source,
            proof: self.proof,
            pool: self.pool,
            domain: Some(domain),
            interventions: self.interventions,
        }
    }
    pub(crate) fn with_interventions(
        self,
        source: &'a eredu_runtime::working_memory::OriginalInterventionSource,
    ) -> Self {
        Self {
            interventions: Some(source),
            ..self
        }
    }
    pub(super) fn domain(self) -> Option<CaptureTokenDomain<'a>> {
        self.domain
    }
    pub(super) fn validate(self, sources: &OriginalSpeculativeNumericalSources) -> Result<(), Error> {
        let controls = match self.proof {
            Proof::Registered(_) => {
                RegisteredInferenceSourceWitness::capture_validation_control_bytes()
            }
            Proof::Original(_) => {
                eredu_runtime::working_memory::OriginalCaptureSource::validation_control_bytes()
            }
        }
        .ok_or(Error::WorkspacePlanning(
            WorkspaceMetadataFundingError::Overflow,
        ))?;
        sources
            .metadata_funding()
            .reserve_metadata(controls)
            .map_err(Error::WorkspacePlanning)?;
        sources
            .request()
            .validate_pool(self.pool)
            .map_err(|cause| sources.retain_startup_error(cause))?;
        if let Some(source) = self.interventions {
            let controls=eredu_runtime::working_memory::OriginalInterventionSource::validation_control_bytes()
                .ok_or(Error::WorkspacePlanning(WorkspaceMetadataFundingError::Overflow))?;
            sources
                .metadata_funding()
                .reserve_metadata(controls)
                .map_err(Error::WorkspacePlanning)?;
            source
                .validate_pool(self.pool)
                .map_err(|cause| sources.retain_startup_error(cause))?;
        }
        match self.proof {
            Proof::Registered(witness) => witness.validate_capture_source(self.source, self.pool),
            Proof::Original(source) => source.validate_pool(self.pool),
        }
        .map_err(|cause| sources.retain_startup_error(cause))
    }
    pub(super) fn host(
        self,
        program: program::SpeculativeNumericalProgram,
    ) -> Result<SpeculativeCaptureHostPlan<'a>, eredu_runtime::working_memory::CaptureRunHostError>
    {
        let program::SpeculativeNumericalKind::ProcessLogits(policy) = program.kind() else {
            return Err(eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch.into());
        };
        let host =
            SpeculativeCaptureHostPlan::prepare(self.source, policy.history_len() as u64, program)?;
        match self.interventions {
            Some(source) => host.with_interventions(source),
            None => Ok(host),
        }
    }
}

enum Selection<'a> {
    Raw(CaptureTensorGeometry<'a>),
    Candidates(CandidateExtraction),
    Scores(TokenScoreProgram<'a>),
    Summary(PreparedCaptureSummary),
    Histogram(PreparedCaptureHistogram<'a>),
}
impl<'a> Selection<'a> {
    fn prepare(
        admission: &'a AdmittedCapturePlan,
        index: usize,
        prediction: u64,
        context: &WorkspaceContext,
    ) -> Result<Self, Error> {
        let phase = if prediction == 0 {
            CapturePhase::Prefill
        } else {
            CapturePhase::Decode
        };
        Ok(match admission.plan().selections[index].transform {
            CaptureTransform::Histogram { .. } => {
                let geometry =
                    CaptureHistogramGeometry::prepare(admission, index, phase, prediction, None)
                        .map_err(|cause| cold(context, cause))?;
                Self::Histogram(
                    PreparedCaptureHistogram::from_geometry(&geometry)
                        .map_err(|cause| cold(context, cause))?,
                )
            }
            CaptureTransform::Summary => {
                let geometry =
                    CaptureSummaryGeometry::prepare(admission, index, phase, prediction, None)
                        .map_err(|cause| cold(context, cause))?;
                Self::Summary(
                    PreparedCaptureSummary::from_geometry(&geometry)
                        .map_err(|cause| cold(context, cause))?,
                )
            }
            CaptureTransform::TopCandidates { .. } => {
                let geometry =
                    CaptureCandidateGeometry::prepare(admission, index, phase, prediction, None)
                        .map_err(|cause| cold(context, cause))?;
                Self::Candidates(
                    CandidateExtraction::from_geometry(&geometry)
                        .map_err(|cause| cold(context, cause))?,
                )
            }
            CaptureTransform::TokenScores { .. } => {
                let geometry =
                    CaptureTokenScoreGeometry::prepare(admission, index, phase, prediction, None)
                        .map_err(|cause| cold(context, cause))?;
                Self::Scores(
                    TokenScoreProgram::from_geometry(&geometry)
                        .map_err(|cause| cold(context, cause))?,
                )
            }
            _ => Self::Raw(
                CaptureTensorGeometry::prepare(admission, index, phase, prediction, None)
                    .map_err(|cause| cold(context, cause))?,
            ),
        })
    }
    fn population(&self) -> Option<(usize, usize, usize)> {
        use crate::backend::array_copy;
        match self {
            Self::Raw(_) => Some((5, 1, array_copy::speculative_capture_control_bytes()?)),
            Self::Histogram(program) => {
                let population = program.population()?;
                // The numerical source is completed already; flatten and every
                // scalar read remain in the exact same Graph/Record phase.
                Some((
                    population.retained_roots,
                    population.completions.checked_sub(1)?,
                    array_copy::speculative_histogram_control_bytes(program)?,
                ))
            }
            Self::Summary(program) => {
                let population = program.population()?;
                Some((
                    population.retained_roots,
                    population.completions.checked_sub(1)?,
                    array_copy::speculative_summary_control_bytes(program)?,
                ))
            }
            Self::Candidates(_) => Some((
                1 + CandidateExtraction::ROOTS,
                CandidateExtraction::COMPLETIONS,
                array_copy::speculative_candidate_control_bytes()?,
            )),
            Self::Scores(program) => {
                let population = program.population().ok()?;
                Some((
                    1usize.checked_add(population.retained_outputs)?,
                    population.scalar_completions,
                    array_copy::speculative_token_score_control_bytes(*program)?,
                ))
            }
        }
    }
    fn trace(
        &self,
        row: &WorkspaceTensor,
        context: &WorkspaceContext,
        retained: &mut Vec<WorkspaceTensor>,
    ) -> Result<(), Error> {
        match self {
            Self::Raw(geometry) => {
                CaptureTensorSelection::from_geometry(geometry)
                    .map_err(|cause| cold(context, cause))?
                    .trace_retained_within(row, context, retained)
                    .map_err(|cause| cold(context, cause))?;
            }
            Self::Histogram(program) => program
                .trace(row, context, retained)
                .map_err(|cause| cold(context, cause))?,
            Self::Summary(program) => program
                .trace(row, context, retained)
                .map_err(|cause| cold(context, cause))?,
            Self::Candidates(program) => retained.extend(program.trace(row, context)?),
            Self::Scores(program) => program
                .trace(row, context, retained)
                .map_err(|cause| cold(context, cause))?,
        }
        Ok(())
    }
}

/// Cold borrowed geometry and query frames are paid by the source-pair H
/// before trace construction; native/destination controls remain phase-owned.
pub(super) fn planning_control_bytes() -> Option<usize> {
    let frames = [
        size_of::<Selection<'static>>(),
        size_of::<Result<Selection<'static>, Error>>(),
        size_of::<CaptureCandidateGeometry<'static>>(),
        size_of::<Result<CaptureCandidateGeometry<'static>, CaptureTensorGeometryError>>(),
        size_of::<CaptureTokenScoreGeometry<'static>>(),
        size_of::<Result<CaptureTokenScoreGeometry<'static>, CaptureTensorGeometryError>>(),
        size_of::<CaptureTensorGeometry<'static>>(),
        size_of::<CaptureHistogramGeometry<'static>>(),
        size_of::<Result<CaptureHistogramGeometry<'static>, CaptureTensorGeometryError>>(),
        CaptureHistogramGeometry::preparation_control_bytes()?,
        size_of::<CaptureSummaryGeometry<'static>>(),
        size_of::<Result<CaptureSummaryGeometry<'static>, CaptureTensorGeometryError>>(),
        size_of::<CaptureObservationStep<'static>>(),
        size_of::<Result<bool, Error>>(),
        size_of::<Option<(usize, usize, usize)>>(),
        size_of::<Option<CaptureTokenDomain<'static>>>(),
        size_of::<(&CaptureObservationStep<'static>, &WorkspaceContext)>(),
    ];
    frames
        .into_iter()
        .try_fold(size_of_val(&frames), usize::checked_add)
        .and_then(|n| n.checked_add(interventions::planning_control_bytes()?))
}

pub(super) struct Plan {
    // Actual immutable source owners survive the entire native recovery cut.
    source: SharedCapturePlan,
    proof: OwnedProof,
    pool: eredu_runtime::working_memory::WorkingMemoryPool,
    _trace_roots: Vec<WorkspaceTensor>,
    edits: Option<Edits>,
    pub(super) effective: Option<WorkspaceTensor>,
    // Descriptive applicability only. The complete CPU trace still supplies
    // all operation/layout, source and completion evidence before admission.
    pub(super) cpu_readouts: bool,
    pub(super) completions: usize,
    pub(super) roots: usize,
    pub(super) controls: usize,
}
impl Plan {
    pub(super) fn trace(
        source: CaptureSource<'_>,
        host: &SpeculativeCaptureHostPlan<'_>,
        input: &WorkspaceTensor,
        context: &WorkspaceContext,
    ) -> Result<Self, Error> {
        let policy = CaptureObservationStep::new(
            source.source.admission(),
            if host.prediction() == 0 {
                CapturePhase::Prefill
            } else {
                CapturePhase::Decode
            },
            host.prediction(),
        )
        .map_err(|cause| cold(context, cause))?;
        let selected = |index| -> Result<bool, Error> {
            policy
                .select(
                    index,
                    policy
                        .initial_status(index)
                        .map_err(|cause| cold(context, cause))?,
                    eredu_core::MODEL_LOGITS_OBSERVATION_PATH,
                )
                .map_err(|cause| cold(context, cause))
        };
        // CPU edits and their evidence use the same source-qualified trace and
        // typed recipe loan. Every action still needs a complete CPU plan.
        let mut cpu_readouts=true;
        let mut roots = 1usize; // Normalized row exists even when logical quotas skip all hooks.
        let mut completions = 0usize;
        let mut controls = 0usize;
        for index in 0..source.source.admission().plan().selections.len() {
            if !selected(index)? {
                continue;
            }
            let worker =
                Selection::prepare(source.source.admission(), index, host.prediction(), context)?;
            cpu_readouts &= matches!(&worker,Selection::Raw(_) | Selection::Summary(_) | Selection::Histogram(_) | Selection::Scores(_) | Selection::Candidates(_));
            let (worker_roots, worker_completions, worker_controls) = worker
                .population()
                .ok_or_else(|| cold(context, CaptureTensorNativeError::GeometryOverflow))?;
            roots = roots
                .checked_add(worker_roots)
                .ok_or_else(|| cold(context, CaptureTensorNativeError::GeometryOverflow))?;
            completions = completions
                .checked_add(worker_completions)
                .ok_or_else(|| cold(context, CaptureTensorNativeError::GeometryOverflow))?;
            controls = controls
                .checked_add(worker_controls)
                .ok_or_else(|| cold(context, CaptureTensorNativeError::GeometryOverflow))?;
        }
        let vocabulary = *input
            .shape()
            .last()
            .ok_or_else(|| cold(context, CaptureTensorNativeError::ShapeMismatch))?;
        let edits = source
            .interventions
            .map(|source| Edits::prepare(source, host.prediction(), vocabulary, context))
            .transpose()?;
        if let Some(edits) = &edits {
            roots = roots
                .checked_add(edits.roots)
                .ok_or_else(|| cold(context, CaptureTensorNativeError::GeometryOverflow))?;
            completions = completions
                .checked_add(edits.completions)
                .ok_or_else(|| cold(context, CaptureTensorNativeError::GeometryOverflow))?;
            controls = controls
                .checked_add(edits.controls)
                .ok_or_else(|| cold(context, CaptureTensorNativeError::GeometryOverflow))?;
        }
        let mut retained = context.metadata_vec(roots)?;
        let vocabulary = *input
            .shape()
            .last()
            .ok_or_else(|| cold(context, CaptureTensorNativeError::ShapeMismatch))?;
        let row = input.reshape(&[1, 1, vocabulary], context)?;
        retained.push(row.clone());
        for index in 0..source.source.admission().plan().selections.len() {
            if !selected(index)? {
                continue;
            }
            let worker =
                Selection::prepare(source.source.admission(), index, host.prediction(), context)?;
            // Every numerical adapter retains the actual completed source alias.
            retained.push(row.clone());
            worker.trace(&row, context, &mut retained)?;
        }
        let effective = match &edits {
            Some(edits) => edits
                .trace(&row, context, &mut retained)?
                .map(|value| -> Result<WorkspaceTensor, Error> {
                    let output = value.reshape(input.shape(), context)?;
                    retained.push(output.clone());
                    Ok(output)
                })
                .transpose()?,
            None => None,
        };
        let controls = controls
            .checked_add(
                Layout::array::<Array>(roots)
                    .ok()
                    .ok_or_else(|| cold(context, CaptureTensorNativeError::GeometryOverflow))?
                    .size(),
            )
            .and_then(|n| n.checked_add(rc_bytes::<RefCell<Vec<Array>>>()?))
            .and_then(|n| n.checked_add(safemlx::PreparedArrayClone::control_bytes()?))
            .and_then(|n| n.checked_add(Array::inspection_clone_handle_bytes()))
            .and_then(|n| {
                n.checked_add(
                    size_of::<Backend<'static>>()
                        + size_of::<Roots>()
                        + size_of::<std::collections::TryReserveError>(),
                )
            })
            .ok_or_else(|| cold(context, CaptureTensorNativeError::GeometryOverflow))?;
        Ok(Self {
            source: source.source.clone(),
            proof: match source.proof {
                Proof::Registered(witness) => OwnedProof::Registered(witness.clone()),
                Proof::Original(source) => OwnedProof::Original(source.clone()),
            },
            pool: source.pool.clone(),
            _trace_roots: retained,
            edits,
            effective,
            cpu_readouts,
            completions,
            roots,
            controls,
        })
    }
    pub(super) fn validate(&self, sources: &OriginalSpeculativeNumericalSources) -> Result<(), Error> {
        let controls = match &self.proof {
            OwnedProof::Registered(_) => {
                RegisteredInferenceSourceWitness::capture_validation_control_bytes()
            }
            OwnedProof::Original(_) => {
                eredu_runtime::working_memory::OriginalCaptureSource::validation_control_bytes()
            }
        }
        .ok_or(Error::WorkspacePlanning(
            WorkspaceMetadataFundingError::Overflow,
        ))?;
        sources
            .metadata_funding()
            .reserve_metadata(controls)
            .map_err(Error::WorkspacePlanning)?;
        sources
            .request()
            .validate_pool(&self.pool)
            .map_err(|cause| sources.retain_startup_error(cause))?;
        if let Some(edits) = &self.edits {
            let controls=eredu_runtime::working_memory::OriginalInterventionSource::validation_control_bytes()
                .ok_or(Error::WorkspacePlanning(WorkspaceMetadataFundingError::Overflow))?;
            sources
                .metadata_funding()
                .reserve_metadata(controls)
                .map_err(Error::WorkspacePlanning)?;
            edits
                .validate(&self.pool)
                .map_err(|cause| sources.retain_startup_error(cause))?;
        }
        match &self.proof {
            OwnedProof::Registered(witness) => {
                witness.validate_capture_source(&self.source, &self.pool)
            }
            OwnedProof::Original(source) => source.validate_pool(&self.pool),
        }
        .map_err(|cause| sources.retain_startup_error(cause))
    }
}

/// Closed Rc retirement; every native recovery alias retains the same phase
/// account externally and frees this concrete shell before that account drops.
pub(super) struct Roots(Option<Rc<RefCell<Vec<Array>>>>);
impl Clone for Roots {
    fn clone(&self) -> Self {
        Self(Some(self.0.as_ref().expect("live roots").clone()))
    }
}
impl Drop for Roots {
    fn drop(&mut self) {
        if let Some(value) = self.0.take() {
            drop(Rc::into_inner(value));
        }
    }
}
impl Roots {
    pub(super) fn prepare(count: usize) -> Result<Self, CaptureTensorNativeError> {
        let mut values = Vec::new();
        values.try_reserve_exact(count)?;
        if values.capacity() != count {
            return Err(CaptureTensorNativeError::GeometryOverflow);
        }
        Ok(Self(Some(Rc::new(RefCell::new(values)))))
    }
    fn values(&self) -> &RefCell<Vec<Array>> {
        self.0.as_ref().expect("live roots")
    }
    pub(super) fn append_to(
        &self,
        roots: &mut safemlx::PrefillRoots,
    ) -> Result<(), safemlx::PrefillRootsCause> {
        for value in self.values().borrow().iter() {
            roots.append(value)?;
        }
        Ok(())
    }
}
pub(super) fn observe(
    plan: &Plan,
    recipe:&crate::backend::nn::workspace::SpeculativeNumericalRecipe,
    invocation: &mut FundedSpeculativeCaptureInvocation,
    source: &Array,
    stream: &Stream,
    roots: &Roots,
    custody: &OriginalSpeculativeNumericalBudgetCustody,
    observer: &OriginalScopeObserver,
    domain: Option<CaptureTokenDomain<'_>>,
) -> Result<Option<Array>, eredu_runtime::capture::FundedCaptureError<CaptureTensorNativeError>> {
    if !invocation.source().same_storage(&plan.source) {
        return Err(eredu_runtime::capture::FundedCaptureError::Backend(
            CaptureTensorNativeError::ClaimMismatch,
        ));
    }
    let cpu=recipe.cpu_capture_loan(stream,observer)
        .map_err(eredu_runtime::capture::FundedCaptureError::Backend)?;
    let row = source
        .reshape(&[1, 1, -1], stream)
        .map_err(|cause| eredu_runtime::capture::FundedCaptureError::Backend(cause.into()))?;
    let mut alias = safemlx::PreparedArrayClone::try_prepare_for_inspection()
        .map_err(|cause| eredu_runtime::capture::FundedCaptureError::Backend(cause.into()))?;
    let alias = alias
        .fill_in_original_scope(&row, observer)
        .map_err(|cause| eredu_runtime::capture::FundedCaptureError::Backend(cause.into()))?;
    roots.values().borrow_mut().push(alias);
    let effective = invocation.observe_and_intervene(
        &mut Backend {
            stream,
            roots: roots.values(),
            custody,
            observer,
            domain,
            edits: plan.edits.as_ref(),
            cpu,
        },
        &row,
    )?;
    effective
        .map(
            |value| -> Result<
                Array,
                eredu_runtime::capture::FundedCaptureError<CaptureTensorNativeError>,
            > {
                let output = value.reshape(source.shape(), stream).map_err(|cause| {
                    eredu_runtime::capture::FundedCaptureError::Backend(cause.into())
                })?;
                let mut clone =
                    safemlx::PreparedArrayClone::try_prepare_for_inspection().map_err(|cause| {
                        eredu_runtime::capture::FundedCaptureError::Backend(cause.into())
                    })?;
                let alias = clone
                    .fill_in_original_scope(&output, observer)
                    .map_err(|cause| {
                        eredu_runtime::capture::FundedCaptureError::Backend(cause.into())
                    })?;
                let mut values = roots.values().try_borrow_mut().map_err(|_| {
                    eredu_runtime::capture::FundedCaptureError::Backend(
                        CaptureTensorNativeError::CollectorBusy,
                    )
                })?;
                if values.len() == values.capacity() {
                    return Err(eredu_runtime::capture::FundedCaptureError::Backend(
                        CaptureTensorNativeError::GeometryOverflow,
                    ));
                }
                values.push(alias);
                Ok(output)
            },
        )
        .transpose()
}
struct Backend<'a> {
    stream: &'a Stream,
    roots: &'a RefCell<Vec<Array>>,
    custody: &'a OriginalSpeculativeNumericalBudgetCustody,
    observer: &'a OriginalScopeObserver,
    domain: Option<CaptureTokenDomain<'a>>,
    edits: Option<&'a Edits>,
    cpu:Option<crate::backend::nn::workspace::CpuCaptureLoan<'a>>,
}
impl ScheduledCaptureBackend for Backend<'_> {
    type Tensor = Array;
    type Error = CaptureTensorNativeError;
    fn intervention_evidence_usage(
        &self,
        source: &Array,
        claim: &eredu_runtime::working_memory::CaptureInterventionEvidenceClaim<'_, '_>,
    ) -> Result<(TensorDtype, CaptureUsage), eredu_runtime::capture::FundedCaptureError<Self::Error>>
    {
        self.edits
            .ok_or(CaptureTensorNativeError::ClaimMismatch)
            .and_then(|edits| edits.evidence_usage(source, claim, self.custody))
            .map_err(eredu_runtime::capture::FundedCaptureError::Backend)
    }
    fn capture_intervention_evidence<'a>(
        &mut self,
        source: &Array,
        claim: eredu_runtime::working_memory::CaptureInterventionEvidenceClaim<'a, '_>,
    ) -> Result<
        eredu_runtime::working_memory::ClaimedInterventionEvidence<'a>,
        eredu_runtime::capture::FundedCaptureError<Self::Error>,
    > {
        let edits = self
            .edits
            .ok_or(eredu_runtime::capture::FundedCaptureError::Backend(
                CaptureTensorNativeError::ClaimMismatch,
            ))?;
        edits.capture_evidence(
            source,
            claim,
            self.stream,
            self.roots,
            self.custody,
            self.observer,
            self.cpu.as_ref(),
        )
    }
    fn intervention_usage(
        &self,
        source: &Array,
        claim: &eredu_runtime::working_memory::CaptureInterventionClaim<'_>,
    ) -> Result<CaptureUsage, eredu_runtime::capture::FundedCaptureError<Self::Error>> {
        self.edits
            .ok_or(CaptureTensorNativeError::ClaimMismatch)
            .and_then(|edits| edits.usage(source, claim, self.custody))
            .map_err(eredu_runtime::capture::FundedCaptureError::Backend)
    }
    fn apply_intervention(
        &mut self,
        source: &Array,
        claim: eredu_runtime::working_memory::CaptureInterventionClaim<'_>,
        charged: CaptureUsage,
    ) -> Result<
        (Array, eredu_runtime::working_memory::ClaimedIntervention),
        eredu_runtime::capture::FundedCaptureError<Self::Error>,
    > {
        let edits = self
            .edits
            .ok_or(eredu_runtime::capture::FundedCaptureError::Backend(
                CaptureTensorNativeError::ClaimMismatch,
            ))?;
        edits.execute(
            source,
            claim,
            charged,
            self.stream,
            self.roots,
            self.custody,
            self.observer,
            self.cpu.as_ref(),
        )
    }
    fn validate_source(
        &self,
        source: &Array,
        geometry: &CaptureTensorGeometry<'_>,
    ) -> Result<TensorDtype, Self::Error> {
        Ok(
            match PreparedCaptureTensor::validate_borrowed_source(source, geometry)? {
                safemlx::Dtype::Float32 => TensorDtype::F32,
                safemlx::Dtype::Float16 => TensorDtype::F16,
                safemlx::Dtype::Bfloat16 => TensorDtype::Bf16,
                _ => unreachable!("audited floating source"),
            },
        )
    }
    fn estimate(
        &self,
        _: &Array,
        geometry: &CaptureTensorGeometry<'_>,
    ) -> Result<CaptureUsage, CaptureError> {
        crate::composition::mlx::session::bounded_capture::estimate_tensor_geometry(geometry)
    }
    fn validate_histogram_source(
        &self,
        source: &Array,
        geometry: &CaptureHistogramGeometry<'_>,
    ) -> Result<TensorDtype, eredu_runtime::capture::FundedCaptureError<Self::Error>> {
        let checked = || {
            PreparedCaptureHistogram::from_geometry(geometry)?.validate_source(source)?;
            floating_dtype(source)
        };
        checked().map_err(eredu_runtime::capture::FundedCaptureError::Backend)
    }
    fn estimate_histogram(
        &self,
        geometry: &CaptureHistogramGeometry<'_>,
    ) -> Result<CaptureUsage, CaptureError> {
        crate::composition::mlx::session::bounded_capture::estimate_histogram(geometry)
    }
    fn transform_histogram(
        &mut self,
        source: &Array,
        claim: eredu_runtime::working_memory::CaptureHistogramClaim<'_, '_>,
    ) -> Result<
        eredu_runtime::working_memory::ClaimedCaptureHistogram,
        eredu_runtime::capture::FundedCaptureError<Self::Error>,
    > {
        match self.cpu.as_ref() {
            Some(loan) => crate::backend::array_copy::execute_speculative_histogram_cpu(
                source, claim, self.roots, self.custody, loan),
            None => crate::backend::array_copy::execute_speculative_histogram(
                source, claim, self.stream, self.roots, self.custody, self.observer),
        }
        .map_err(eredu_runtime::capture::FundedCaptureError::Backend)
    }
    fn validate_summary_source(
        &self,
        source: &Array,
        geometry: &CaptureSummaryGeometry<'_>,
    ) -> Result<TensorDtype, eredu_runtime::capture::FundedCaptureError<Self::Error>> {
        let checked = || {
            PreparedCaptureSummary::from_geometry(geometry)?.validate_source(source)?;
            floating_dtype(source)
        };
        checked().map_err(eredu_runtime::capture::FundedCaptureError::Backend)
    }
    fn estimate_summary(
        &self,
        geometry: &CaptureSummaryGeometry<'_>,
    ) -> Result<CaptureUsage, CaptureError> {
        crate::composition::mlx::session::bounded_capture::estimate_summary(geometry)
    }
    fn transform_summary(
        &mut self,
        source: &Array,
        claim: eredu_runtime::working_memory::CaptureSummaryClaim<'_, '_>,
    ) -> Result<
        eredu_runtime::working_memory::ClaimedCaptureSummary,
        eredu_runtime::capture::FundedCaptureError<Self::Error>,
    > {
        match self.cpu.as_ref() {
            Some(loan) => crate::backend::array_copy::execute_speculative_summary_cpu(
                source, claim, self.roots, self.custody, loan),
            None => crate::backend::array_copy::execute_speculative_summary(
                source, claim, self.stream, self.roots, self.custody, self.observer),
        }
        .map_err(eredu_runtime::capture::FundedCaptureError::Backend)
    }
    fn validate_candidate_source(
        &self,
        source: &Array,
        geometry: &CaptureCandidateGeometry<'_>,
    ) -> Result<TensorDtype, eredu_runtime::capture::FundedCaptureError<Self::Error>> {
        let checked = || {
            CandidateExtraction::from_geometry(geometry)?.validate_source(source)?;
            floating_dtype(source)
        };
        checked().map_err(eredu_runtime::capture::FundedCaptureError::Backend)
    }
    fn estimate_candidates(
        &self,
        geometry: &CaptureCandidateGeometry<'_>,
    ) -> Result<CaptureUsage, CaptureError> {
        crate::composition::mlx::session::bounded_capture::estimate_candidates(geometry)
    }
    fn transform_candidates(
        &mut self,
        source: &Array,
        claim: eredu_runtime::working_memory::CaptureCandidateClaim<'_, '_>,
    ) -> Result<
        eredu_runtime::working_memory::ClaimedCaptureCandidates,
        eredu_runtime::capture::FundedCaptureError<Self::Error>,
    > {
        match self.cpu.as_ref() {
            Some(loan) => crate::backend::array_copy::execute_speculative_candidates_cpu(
                source, claim, self.roots, self.custody, loan, self.domain),
            None => crate::backend::array_copy::execute_speculative_candidates(
                source, claim, self.stream, self.roots, self.custody, self.observer, self.domain),
        }
        .map_err(eredu_runtime::capture::FundedCaptureError::Backend)
    }
    fn validate_token_score_source(
        &self,
        source: &Array,
        geometry: &CaptureTokenScoreGeometry<'_>,
    ) -> Result<TensorDtype, eredu_runtime::capture::FundedCaptureError<Self::Error>> {
        let checked = || {
            TokenScoreProgram::from_geometry(geometry)?.validate_source(source)?;
            floating_dtype(source)
        };
        checked().map_err(eredu_runtime::capture::FundedCaptureError::Backend)
    }
    fn estimate_token_scores(
        &self,
        geometry: &CaptureTokenScoreGeometry<'_>,
    ) -> Result<CaptureUsage, CaptureError> {
        crate::composition::mlx::session::bounded_capture::estimate_token_scores(geometry)
    }
    fn transform_token_scores(
        &mut self,
        source: &Array,
        claim: eredu_runtime::working_memory::CaptureTokenScoreClaim<'_, '_>,
    ) -> Result<
        eredu_runtime::working_memory::ClaimedCaptureTokenScores,
        eredu_runtime::capture::FundedCaptureError<Self::Error>,
    > {
        match self.cpu.as_ref() {
            Some(loan) => crate::backend::array_copy::execute_speculative_token_scores_cpu(
                source, claim, self.roots, self.custody, loan, self.domain),
            None => crate::backend::array_copy::execute_speculative_token_scores(
                source, claim, self.stream, self.roots, self.custody, self.observer, self.domain),
        }
        .map_err(eredu_runtime::capture::FundedCaptureError::Backend)
    }
    fn transform(
        &mut self,
        source: &Array,
        claim: CaptureTensorClaim<'_, '_>,
    ) -> Result<ClaimedCaptureTensor, Self::Error> {
        if let Some(loan)=&self.cpu {
            return crate::backend::array_copy::execute_speculative_capture_cpu(
                source,claim,self.roots,self.custody,loan);
        }
        crate::backend::array_copy::execute_speculative_capture(
            source,
            claim,
            self.stream,
            self.roots,
            self.custody,
            self.observer,
        )
    }
}

// A borrowed caller destination, with no new record owner, queue or allowance.
// Success drains explicitly after full recovery; early returns move only sealed
// host evidence. The existing error/Recovery path keeps every native resource.
pub(super) struct Invocation<'a> {
    value: Option<FundedSpeculativeCaptureInvocation>,
    failed: Option<&'a mut Option<SharedCapturedStep>>,
}
impl<'a> Invocation<'a> {
    pub(super) fn new(
        value: Option<FundedSpeculativeCaptureInvocation>,
        failed: Option<&'a mut Option<SharedCapturedStep>>,
    ) -> Self {
        Self { value, failed }
    }
    pub(super) fn as_mut(&mut self) -> Option<&mut FundedSpeculativeCaptureInvocation> {
        self.value.as_mut()
    }
}
impl Drop for Invocation<'_> {
    fn drop(&mut self) {
        if let (Some(value), Some(destination)) = (&mut self.value, &mut self.failed) {
            // Preserve the primary execution error if best-effort host sealing
            // also fails. The original pending owner then retires without refund.
            if let Ok(Some(frame)) = value.take_failed_evidence() {
                **destination = Some(frame);
            }
        }
    }
}

fn floating_dtype(source: &Array) -> Result<TensorDtype, CaptureTensorNativeError> {
    match source.dtype() {
        safemlx::Dtype::Float32 => Ok(TensorDtype::F32),
        safemlx::Dtype::Float16 => Ok(TensorDtype::F16),
        safemlx::Dtype::Bfloat16 => Ok(TensorDtype::Bf16),
        dtype => Err(CaptureTensorNativeError::UnsupportedDtype(dtype)),
    }
}
