//! One support projection with owned and borrowed declaration destinations.
use eredu_core::{ObservationSupport, ObservationSupportStatus as Status, intervention::*};
mod funded;
pub use funded::{prepare_intervention_discovery, intervention_discovery_preparation_bytes,
    FundedInterventionDiscovery, InterventionDiscoveryPreparationError};
const UNAVAILABLE: &str = "required intervention geometry or native mechanism is not declared";
pub(super) fn support_for<'a>(point: &InterventionPoint, support: &'a [ObservationSupport]) -> Option<&'a ObservationSupport> {
    support.iter().find(|row| {
        if point.routing.is_some() {
            row.path.strip_prefix(&point.path).and_then(|suffix| suffix.strip_prefix(".routing."))
                == Some(eredu_core::RoutingObservationField::SelectedExperts.suffix())
        } else { row.path == point.path }
    })
}
enum ProjectedStatus<'a> {
    Borrowed(&'a Status),
    Unavailable,
}
impl ProjectedStatus<'_> {
    fn owned(&self) -> Status {
        match self {
            Self::Borrowed(value) => (*value).clone(),
            Self::Unavailable => Status::Unsupported(UNAVAILABLE.into()),
        }
    }
    fn matches(&self, value: &Status) -> bool {
        match self {
            Self::Borrowed(actual) => *actual == value,
            Self::Unavailable => matches!(value,Status::Unsupported(reason) if reason==UNAVAILABLE),
        }
    }
}
struct Projection<'a> {
    source: &'a InterventionPoint,
    facts: InterventionMechanismFacts<'a>,
    prefill: ProjectedStatus<'a>,
    decode: ProjectedStatus<'a>,
}
impl<'a> Projection<'a> {
    fn new(
        source: &'a InterventionPoint,
        support: Option<&'a ObservationSupport>,
        facts: InterventionMechanismFacts<'a>,
    ) -> Self {
        let unavailable = source.axes.is_empty()
            || (source.routed_units.is_some() && !facts.routed_units)
            || !source
                .operations
                .iter()
                .any(|kind| facts.operations.contains(kind))
            || (source.routing.is_none()
                && !source
                    .dtypes
                    .iter()
                    .any(|dtype| facts.dtypes.contains(dtype)));
        let (prefill, decode) = if unavailable {
            (ProjectedStatus::Unavailable, ProjectedStatus::Unavailable)
        } else {
            (
                ProjectedStatus::Borrowed(support.map_or(&source.prefill, |s| &s.prefill)),
                ProjectedStatus::Borrowed(support.map_or(&source.decode, |s| &s.decode)),
            )
        };
        Self {
            source,
            facts,
            prefill,
            decode,
        }
    }
    fn matches_fields(&self, value: &InterventionPoint) -> bool {
        let s = self.source;
        s.path == value.path
            && s.node_id == value.node_id
            && s.stage == value.stage
            && s.axes == value.axes
            && s.conditions == value.conditions
            && s.routing == value.routing
            && s.routed_units == value.routed_units
            && s.operations
                .iter()
                .filter(|k| self.facts.operations.contains(k))
                .eq(value.operations.iter())
            && s.dtypes
                .iter()
                .filter(|k| self.facts.dtypes.contains(k))
                .eq(value.dtypes.iter())
            && s.score_stages
                .iter()
                .filter(|k| self.facts.score_stages.contains(k))
                .eq(value.score_stages.iter())
    }
}
pub(super) fn apply(
    point: &mut InterventionPoint,
    support: Option<&ObservationSupport>,
    facts: InterventionMechanismFacts<'_>,
) {
    let projected = Projection::new(point, support, facts);
    let prefill = projected.prefill.owned();
    let decode = projected.decode.owned();
    point
        .operations
        .retain(|kind| facts.operations.contains(kind));
    point.dtypes.retain(|dtype| facts.dtypes.contains(dtype));
    point
        .score_stages
        .retain(|stage| facts.score_stages.contains(stage));
    point.prefill = prefill;
    point.decode = decode;
}
/// Compare selected static activation declarations using the same mechanism and
/// loaded observation-support projection as ordinary intervention discovery.
/// The retained source owner must separately authenticate artifact/session identity.
pub fn validate_static_intervention_declarations(
    admitted: &AdmittedInterventionPlan,
    points: &[InterventionPoint],
    support: &[ObservationSupport],
    facts: InterventionMechanismFacts<'_>,
) -> Result<(), InterventionSourceError> {
    validate_static_intervention_declarations_with_phases(
        admitted,
        points,
        facts,
        |actual, expected| {
            let support = support.iter().find(|row| row.path == actual.path);
            let projection = Projection::new(actual, support, facts);
            projection.prefill.matches(&expected.prefill)
                && projection.decode.matches(&expected.decode)
        },
    )
}
/// Reuse the exact declaration/mechanism comparison with an architecture's
/// borrowed selected-hook phase projection. No declaration/status DTO is copied.
pub fn validate_static_intervention_declarations_with_phases(
    admitted: &AdmittedInterventionPlan,
    points: &[InterventionPoint],
    facts: InterventionMechanismFacts<'_>,
    phases: impl FnMut(&InterventionPoint, &InterventionPoint) -> bool,
) -> Result<(), InterventionSourceError> {
    validate_declarations(admitted, points, facts, false, phases)
}
/// Same comparison for dense and sparse activation declarations. The caller must
/// also validate its original producer's operation profile; this only checks the
/// immutable declaration against ordinary discovery and actual mechanism facts.
pub fn validate_activation_intervention_declarations_with_phases(
    admitted: &AdmittedInterventionPlan, points: &[InterventionPoint],
    facts: InterventionMechanismFacts<'_>,
    phases: impl FnMut(&InterventionPoint, &InterventionPoint) -> bool,
) -> Result<(), InterventionSourceError> {
    validate_declarations(admitted, points, facts, true, phases)
}
fn validate_declarations(
    admitted: &AdmittedInterventionPlan, points: &[InterventionPoint],
    facts: InterventionMechanismFacts<'_>, routed: bool,
    mut phases: impl FnMut(&InterventionPoint, &InterventionPoint) -> bool,
) -> Result<(), InterventionSourceError> {
    if admitted.plan().operations.len() != admitted.points().len() {
        return Err(InterventionSourceError::Identity);
    }
    for (operation, expected) in admitted.plan().operations.iter().zip(admitted.points()) {
        let mut found = points.iter().filter(|point| point.path == operation.target);
        let actual = found.next().ok_or(InterventionSourceError::Identity)?;
        if found.next().is_some() || actual.routing.is_some() || (actual.routed_units.is_some() && !routed) {
            return Err(InterventionSourceError::Identity);
        }
        let projected = Projection::new(actual, None, facts);
        let unavailable = matches!(projected.prefill, ProjectedStatus::Unavailable);
        if !projected.matches_fields(expected)
            || if unavailable {
                !projected.prefill.matches(&expected.prefill)
                    || !projected.decode.matches(&expected.decode)
            } else {
                !phases(actual, expected)
            }
        {
            return Err(InterventionSourceError::Identity);
        }
    }
    Ok(())
}
/// Actual fixed projection and equality iterator controls. No source allocation.
pub fn static_intervention_validation_control_bytes() -> Option<usize> {
    use std::mem::{size_of, size_of_val};
    let frames = [
        size_of::<Projection<'static>>(),
        size_of::<bool>(),
        size_of::<InterventionMechanismFacts<'static>>(),
        size_of::<[std::slice::Iter<'static, InterventionPoint>; 2]>(),
        size_of::<[&InterventionPoint; 3]>(),
        size_of::<&InterventionOperation>(),
        size_of::<std::slice::Iter<'static, ObservationSupport>>(),
        size_of::<[std::slice::Iter<'static, InterventionKind>; 2]>(),
        size_of::<[std::slice::Iter<'static, InterventionDtype>; 2]>(),
        size_of::<[std::slice::Iter<'static, RoutingScoreStage>; 2]>(),
        size_of::<[&Projection<'static>; 3]>(),
        size_of::<(&InterventionOperation, &InterventionPoint)>(),
        size_of::<Result<(), InterventionSourceError>>(),
    ];
    frames
        .into_iter()
        .try_fold(size_of_val(&frames), usize::checked_add)
}
