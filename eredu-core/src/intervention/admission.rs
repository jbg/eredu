//! One declaration validator and exact destination producer for edit admission.
use super::*;
use crate::{HostPreparationAuthority, capture::plan_copy::Worker};
use std::mem::{size_of, size_of_val};

/// Fixed declaration diagnostic, including borrowed-geometry validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum InterventionDeclarationError {
    /// A declaration violates a structural or payload constraint.
    #[error("invalid intervention declaration: {0}")]
    Invalid(&'static str),
    /// The selected declaration cannot perform the requested operation.
    #[error("intervention declaration unsupported: {0}")]
    Unsupported(&'static str),
    /// No retained declaration matches this operation's target.
    #[error("intervention target absent for operation {operation}")]
    MissingPath { /// Index in the immutable request.
        operation: usize },
    /// Exact invocation/axis mismatch.
    #[error(transparent)]
    Axes(CaptureAxisError),
    /// Exact selected-region mismatch.
    #[error(transparent)]
    Slice(CaptureSliceDestinationError),
    /// The closed declaration could not be canonically serialized.
    #[error("intervention canonical serialization failed")]
    Encoding,
}

/// Admission refusal before a source can be published.
#[derive(Debug, thiserror::Error)]
pub enum InterventionAdmissionError {
    /// The same declaration validation used by ordinary admission.
    #[error(transparent)]
    Declaration(#[from] CaptureError),
    /// A checked destination or actual allocation could not be constructed.
    #[error(transparent)]
    Copy(#[from] CapturePlanCopyError),
}

/// Exact borrowed construction program. Inspection creates no destination or
/// admission. The caller pays its fixed inspection controls first and reserves
/// the complete returned extent before invoking construction.
#[derive(Debug)]
pub struct PreparedInterventionAdmission<'a> {
    source: &'a InterventionPlan,
    discovery: &'a InterventionDiscovery,
    request: CaptureRequestShape,
    invocation_bounds: Option<CaptureInvocationBounds>,
    origin: CaptureTextOrigin,
    session: &'a str,
    bytes: usize,
}
impl<'a> PreparedInterventionAdmission<'a> {
    /// Actual fixed inspection and construction frames, including the shared
    /// validator's bounded rank destinations and canonical digest writer.
    pub fn inspection_control_bytes() -> Option<usize> {
        let frames = [
            size_of::<Self>() * 2, size_of::<Worker>(),
            size_of::<InterventionPlan>(), size_of::<InterventionOperation>(),
            size_of::<InterventionPoint>(), size_of::<AdmittedInterventionPlan>() * 2,
            size_of::<HostPreparationAuthority>(), size_of::<CaptureError>(),
            size_of::<InterventionAdmissionError>(),
            size_of::<Result<AdmittedInterventionPlan, InterventionAdmissionError>>(),
            size_of::<Result<Self, InterventionAdmissionError>>(),
            size_of::<[[u64; 32]; 5]>(),
            size_of::<[u8; IDENTITY_PREFIX.len() + 65]>() * 2,
            size_of::<[u8; INTENT_PREFIX.len() + 65]>() * 2,
            size_of::<DigestWriter>(), size_of::<PlanCounter>(),
            size_of::<serde_json::Error>(), size_of::<CaptureInvocationShape>(),
            size_of::<[std::slice::Iter<'a, InterventionOperation>; 3]>(),
            size_of::<[std::slice::Iter<'a, InterventionPoint>; 2]>(),
            size_of::<[std::slice::Iter<'a, TensorAxis>; 3]>(),
            size_of::<[std::slice::Iter<'a, CaptureSlice>; 2]>(),
            size_of::<[&InterventionOperation; 3]>(),
            size_of::<[&InterventionPoint; 3]>(),
            size_of::<[u64; 12]>(), size_of::<[usize; 8]>(),
        ];
        frames.into_iter().try_fold(size_of_val(&frames), usize::checked_add)
    }

    /// Describes a fresh source from actual loaded declarations. The ordinary
    /// and invocation geometry modes use the same complete validation worker.
    pub fn inspect(
        source: &'a InterventionPlan, discovery: &'a InterventionDiscovery,
        request: CaptureRequestShape, invocation_bounds: Option<CaptureInvocationBounds>,
        origin: CaptureTextOrigin, session: &'a str,
    ) -> Result<Self, InterventionAdmissionError> {
        require(source.operations.len() <= MAX_INTERVENTION_OPERATIONS,
            "too many intervention operations")?;
        for (index, operation) in source.operations.iter().enumerate() {
            let mut found = discovery.points.iter().filter(|point| point.path == operation.target);
            if found.next().is_none() {
                return Err(CaptureError::from(InterventionDeclarationError::MissingPath { operation: index }).into());
            }
            require(found.next().is_none(), "ambiguous intervention target declaration")?;
        }
        let mut prepared = Self { source, discovery, request, invocation_bounds, origin, session, bytes: 0 };
        let mut worker = Worker::with_controls(false,
            Self::inspection_control_bytes().ok_or(CapturePlanCopyError::Overflow)?);
        drop(prepared.build(&mut worker, false)?);
        prepared.bytes = worker.bytes();
        Ok(prepared)
    }

    /// Exact fresh DTO destinations, routing scratch and constructor controls.
    pub const fn required_bytes(&self) -> usize { self.bytes }

    /// Construct only after paying `required_bytes`. The result is an admission
    /// intermediate: its enclosing producer must retain `destination` through
    /// payload, partial failure and final source/error retirement. No caller
    /// allocation is adopted; the existing immutable source compiler can borrow
    /// this result while that construction account remains alive.
    pub fn construct(self, _destination: &HostPreparationAuthority)
        -> Result<AdmittedInterventionPlan, InterventionAdmissionError> {
        let mut worker = Worker::with_controls(true,
            Self::inspection_control_bytes().ok_or(CapturePlanCopyError::Overflow)?);
        let result = self.build(&mut worker, true)?;
        if worker.bytes() != self.bytes { return Err(CapturePlanCopyError::Capacity.into()); }
        Ok(result)
    }

    fn build(&self, worker: &mut Worker, emit: bool)
        -> Result<AdmittedInterventionPlan, InterventionAdmissionError> {
        let selected = |operation: &InterventionOperation| self.discovery.points.iter()
            .find(|point| point.path == operation.target).expect("inspected immutable declaration");
        let groups = self.source.operations.iter().filter_map(|operation| {
            matches!(operation.action, InterventionAction::ExcludeExperts { .. })
                .then(|| selected(operation).routing.as_ref().map_or(0, |policy| policy.groups as usize))
        }).max().unwrap_or(0);
        let mut scratch = worker.repeated(groups, 0u32)?;
        if emit {
            self.source.validate_declarations(self.discovery, self.request,
                self.invocation_bounds, self.origin, self.session, &mut scratch)?;
        }
        let plan = source::copy::request(worker, self.source)?;
        let points = worker.vector(self.source.operations.as_slice(), |worker, operation|
            source::copy::point(worker, selected(operation)))?;
        let mut identity = [b'0'; IDENTITY_PREFIX.len() + 65];
        let mut intent = [b'0'; INTENT_PREFIX.len() + 65];
        if emit {
            let mut encoded = PlanCounter(0);
            serde_json::to_writer(&mut encoded, self.source)
                .map_err(|_| CaptureError::from(InterventionDeclarationError::Encoding))?;
            require(encoded.0 <= MAX_INTERVENTION_PLAN_BYTES, "intervention plan exceeds encoded bound")?;
            digest_into(IDENTITY_PREFIX, &(&plan, &points, self.request,
                &self.discovery.artifact_identity, &self.discovery.session_identity, self.session), &mut identity)?;
            digest_into(INTENT_PREFIX, &(&plan, &points, self.request,
                &self.discovery.artifact_identity), &mut intent)?;
            if let Some(bounds) = self.invocation_bounds {
                let prior = identity;
                digest_into(IDENTITY_PREFIX, &("invocation", ascii(&prior), bounds), &mut identity)?;
                let prior = intent;
                digest_into(INTENT_PREFIX, &("invocation", ascii(&prior), bounds), &mut intent)?;
            } else if self.origin != CaptureTextOrigin::default() {
                let prior = identity;
                digest_into(IDENTITY_PREFIX, &("text_origin", ascii(&prior), self.origin), &mut identity)?;
                let prior = intent;
                digest_into(INTENT_PREFIX, &("text_origin", ascii(&prior), self.origin), &mut intent)?;
            }
        }
        Ok(AdmittedInterventionPlan {
            plan, points, request: self.request, invocation_bounds: self.invocation_bounds,
            text_origin: self.origin, identity: worker.text(ascii(&identity))?,
            intent_identity: worker.text(ascii(&intent))?,
            artifact_identity: worker.text(&self.discovery.artifact_identity)?,
            session_id: worker.text(self.session)?,
        })
    }
}
fn ascii(bytes: &[u8]) -> &str { std::str::from_utf8(bytes).expect("canonical ASCII identity") }
fn digest_into(prefix: &str, value: &impl Serialize, output: &mut [u8]) -> Result<(), CaptureError> {
    let digest = intervention_digest_bytes(value)
        .map_err(|_| InterventionDeclarationError::Encoding)?;
    output[..prefix.len()].copy_from_slice(prefix.as_bytes());
    output[prefix.len()] = b'-';
    source::write_hex(&mut output[prefix.len()+1..], &digest);
    Ok(())
}

/// Borrowed final support chosen by the ordinary runtime projection.
#[derive(Clone, Copy, Debug)]
pub enum InterventionPhaseView<'a> {
    /// Exact support already retained by discovery.
    Retained(&'a ObservationSupportStatus),
    /// A fixed missing-mechanism reason selected by that projection.
    Unavailable(&'static str),
}

/// Exact point-copy producer for a runtime-selected mechanism/phase projection.
/// It changes no semantic field besides the explicit capability filters/phases.
#[derive(Debug)]
pub struct PreparedInterventionPointCopy<'a> {
    source: &'a InterventionPoint,
    facts: InterventionMechanismFacts<'a>,
    prefill: InterventionPhaseView<'a>,
    decode: InterventionPhaseView<'a>,
    bytes: usize,
}
impl<'a> PreparedInterventionPointCopy<'a> {
    /// Fixed query and construction controls for this exhaustive field copy.
    pub fn inspection_control_bytes() -> Option<usize> {
        let frames = [size_of::<Self>() * 2, size_of::<Worker>(),
            size_of::<InterventionPoint>() * 2, size_of::<ObservationSupportStatus>() * 2,
            size_of::<CapturePlanCopyError>(), size_of::<HostPreparationAuthority>(),
            size_of::<Result<Self, CapturePlanCopyError>>(),
            size_of::<Result<InterventionPoint, CapturePlanCopyError>>()];
        frames.into_iter().try_fold(size_of_val(&frames), usize::checked_add)
    }
    /// Count the same producer without allocating a destination.
    pub fn inspect(source: &'a InterventionPoint, facts: InterventionMechanismFacts<'a>,
        prefill: InterventionPhaseView<'a>, decode: InterventionPhaseView<'a>) -> Result<Self, CapturePlanCopyError> {
        let mut result = Self { source, facts, prefill, decode, bytes: 0 };
        let mut worker = Worker::with_controls(false,
            Self::inspection_control_bytes().ok_or(CapturePlanCopyError::Overflow)?);
        drop(result.build(&mut worker)?);
        result.bytes = worker.bytes();
        Ok(result)
    }
    /// Complete selected destination and all intermediate source-copy storage.
    pub const fn required_bytes(&self) -> usize { self.bytes }
    /// Construct under a previously paid enclosing discovery account. The caller
    /// retains that account with the destination and any partial error ownership.
    pub fn copy(self, _destination: &HostPreparationAuthority) -> Result<InterventionPoint, CapturePlanCopyError> {
        let mut worker = Worker::with_controls(true,
            Self::inspection_control_bytes().ok_or(CapturePlanCopyError::Overflow)?);
        let point = self.build(&mut worker)?;
        if worker.bytes() != self.bytes { return Err(CapturePlanCopyError::Capacity); }
        Ok(point)
    }
    fn build(&self, worker: &mut Worker) -> Result<InterventionPoint, CapturePlanCopyError> {
        let mut point = source::copy::point(worker, self.source)?;
        point.operations.retain(|kind| self.facts.operations.contains(kind));
        point.dtypes.retain(|dtype| self.facts.dtypes.contains(dtype));
        point.score_stages.retain(|stage| self.facts.score_stages.contains(stage));
        let mut support = |view| match view {
            InterventionPhaseView::Retained(status) => source::copy::support(worker, status),
            InterventionPhaseView::Unavailable(reason) => Ok(ObservationSupportStatus::Unsupported(worker.text(reason)?)),
        };
        point.prefill = support(self.prefill)?;
        point.decode = support(self.decode)?;
        Ok(point)
    }
}
