//! Immutable evidence geometry copied inside the actual intervention source.
use super::plan_copy::Worker;
use super::*;
use crate::intervention::{
    AdmittedInterventionPlan, InterventionEvidence, InterventionOperation, InterventionPoint,
};
use crate::{ObservationDtype, ObservationPosition, ObservationValueType, TensorAxis};

/// Exact activation evidence side, independent of a capture selection ordinal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterventionEvidenceSide {
    /// Values immediately before this operation in composition order.
    Before,
    /// Values immediately after this operation in composition order.
    After,
}
impl InterventionEvidenceSide {
    /// Fixed activation companion ordinal.
    pub const fn index(self) -> usize {
        match self {
            Self::Before => 0,
            Self::After => 1,
        }
    }
}
/// Allocation-free ordinary/original evidence layout from an actual operation.
/// It supplies declaration semantics only, without a capture or source grant.
#[derive(Debug, Clone, Copy)]
pub struct InterventionEvidenceLayout<'a> {
    operation: &'a InterventionOperation,
    point: &'a InterventionPoint,
}
impl<'a> InterventionEvidenceLayout<'a> {
    /// Borrow the existing operation and its selected semantic point.
    pub const fn new(operation: &'a InterventionOperation, point: &'a InterventionPoint) -> Self {
        Self { operation, point }
    }
    /// Exact number of before/after fields; no implicit extra observation.
    pub fn len(self) -> usize {
        if self.operation.evidence == InterventionEvidence::None {
            0
        } else if self.point.routing.is_some() {
            4
        } else {
            2
        }
    }
    /// Whether this operation requested no evidence.
    pub fn is_empty(self) -> bool {
        self.len() == 0
    }
    /// Borrow one field in the existing before-fields, after-fields order.
    pub fn descriptor(self, index: usize) -> Option<InterventionEvidenceDescriptor<'a>> {
        if index >= self.len() {
            return None;
        }
        let fields = if self.point.routing.is_some() { 2 } else { 1 };
        Some(InterventionEvidenceDescriptor {
            layout: self,
            side: if index / fields == 0 {
                InterventionEvidenceSide::Before
            } else {
                InterventionEvidenceSide::After
            },
            field: if fields == 1 {
                None
            } else if index % fields == 0 {
                Some(crate::RoutingObservationField::SelectedExperts)
            } else {
                Some(crate::RoutingObservationField::Coefficients)
            },
        })
    }
}
/// Borrowed declaration fields consumed by ordinary and paid source builders.
#[derive(Debug, Clone, Copy)]
pub struct InterventionEvidenceDescriptor<'a> {
    layout: InterventionEvidenceLayout<'a>,
    side: InterventionEvidenceSide,
    field: Option<crate::RoutingObservationField>,
}
impl<'a> InterventionEvidenceDescriptor<'a> {
    /// Existing stable evidence ID, without a temporary formatted String.
    pub fn selection_id_parts(self) -> [&'a str; 5] {
        [
            &self.layout.operation.id,
            ":",
            match self.side {
                InterventionEvidenceSide::Before => "before",
                InterventionEvidenceSide::After => "after",
            },
            ":",
            match self.field {
                None => "None",
                Some(crate::RoutingObservationField::SelectedExperts) => "Some(SelectedExperts)",
                Some(crate::RoutingObservationField::Coefficients) => "Some(Coefficients)",
                _ => unreachable!("closed evidence field"),
            },
        ]
    }
    /// Exact discovered path parts, shared with native routing field naming.
    pub fn path_parts(self) -> [&'a str; 3] {
        match self.field {
            None => [&self.layout.point.path, "", ""],
            Some(field) => [&self.layout.point.path, ".routing.", field.suffix()],
        }
    }
    /// Existing before/after attribution.
    pub const fn position(self) -> ObservationPosition {
        match self.side {
            InterventionEvidenceSide::Before => ObservationPosition::BeforeIntervention,
            InterventionEvidenceSide::After => ObservationPosition::AfterIntervention,
        }
    }
    /// Actual semantic value category of this field.
    pub fn dtype(self) -> ObservationDtype {
        if self.field == Some(crate::RoutingObservationField::SelectedExperts) {
            ObservationDtype::Integer
        } else {
            ObservationDtype::Floating
        }
    }
    /// The closed Preview/Summary policy requested by the original operation.
    pub fn transform(self) -> CaptureTransform {
        match self.layout.operation.evidence {
            InterventionEvidence::Preview { max_elements } => {
                CaptureTransform::Preview { max_elements }
            }
            InterventionEvidence::Summary => CaptureTransform::Summary,
            InterventionEvidence::None => unreachable!("empty evidence layout"),
        }
    }
}

/// Geometry-only companion owned by the freshly copied intervention declaration.
/// It has zero logical budgets; the enclosing capture ledger pays every record
/// and payload. Borrowing it establishes no source/native/callback authority.
#[derive(Debug)]
pub struct InterventionEvidenceCompanion {
    operation: usize,
    geometry: EvidenceGeometry,
}
// The copying worker uses the first variant only while constructing/counting
// its original DTOs. A publicly returned immutable intervention source has only
// Shared variants, with no copied payload hidden behind a separate allowance.
#[derive(Debug)]
enum EvidenceGeometry {
    Copied(AdmittedCapturePlan),
    Shared(SharedCapturePlan),
}
impl InterventionEvidenceCompanion {
    /// Actual operation ordinal in the owning immutable intervention source.
    pub const fn operation(&self) -> usize {
        self.operation
    }
    /// Existing capture geometry and static before/after declarations. This
    /// zero-budget plan cannot create an independent evidence allowance.
    pub fn geometry_source(&self) -> &AdmittedCapturePlan {
        match &self.geometry {
            EvidenceGeometry::Copied(plan) => plan,
            EvidenceGeometry::Shared(plan) => plan.admission(),
        }
    }
    /// Alias the same geometry copied and retained by this original source.
    /// Its zero logical budgets issue no evidence allowance. The source's host
    /// custody follows every alias; callers still need their original operation,
    /// side, window and frame authority before constructing a record.
    pub fn shared_geometry_source(&self) -> &SharedCapturePlan {
        match &self.geometry {
            EvidenceGeometry::Shared(plan) => plan,
            EvidenceGeometry::Copied(_) => unreachable!("unpublished evidence copy"),
        }
    }
    pub(crate) fn into_shared(self, host: crate::HostPreparationAuthority) -> Self {
        let geometry = match self.geometry {
            EvidenceGeometry::Copied(plan) =>
                EvidenceGeometry::Shared(SharedCapturePlan::from_prepared_copy(plan, host, None)),
            EvidenceGeometry::Shared(plan) => EvidenceGeometry::Shared(plan),
        };
        Self { operation: self.operation, geometry }
    }
    pub(crate) fn control_bytes() -> Option<usize> {
        use std::mem::{size_of, size_of_val};
        let frames = [
            size_of::<Self>(),
            size_of::<Option<Self>>(),
            size_of::<EvidenceGeometry>(),
            size_of::<SharedCapturePlan>(),
            size_of::<crate::HostPreparationAuthority>(),
            size_of::<(Self, crate::HostPreparationAuthority)>(),
            size_of::<AdmittedCapturePlan>(),
            size_of::<CaptureSelection>(),
            size_of::<crate::ObservationPoint>(),
            size_of::<Result<Option<Self>, CapturePlanCopyError>>(),
            size_of::<InterventionEvidenceLayout>(),
            size_of::<InterventionEvidenceDescriptor>(),
            size_of::<[usize; 4]>(),
            size_of::<[Option<crate::RoutingObservationField>; 2]>(),
        ];
        frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)
    }
    pub(crate) fn construct(
        w: &mut Worker,
        source: &AdmittedInterventionPlan,
        index: usize,
        operation: &InterventionOperation,
        point: &InterventionPoint,
    ) -> Result<Option<Self>, CapturePlanCopyError> {
        let layout = InterventionEvidenceLayout::new(operation, point);
        if layout.is_empty() {
            return Ok(None);
        }
        // The same actual copied admission moves into a shared owner after the
        // count/copy comparison. Count each real owner and its independent host
        // token here; the inline DTO remains in this worker's/companion's enum
        // storage while the final shared allocation also owns that DTO inline.
        let owner = SharedCapturePlan::new_owner_control_bytes()
            .and_then(|bytes| bytes.checked_add(SharedCapturePlan::ordinary_host_control_bytes()?))
            .and_then(|bytes| bytes.checked_add(std::mem::size_of::<AdmittedCapturePlan>() as u64))
            .ok_or(CapturePlanCopyError::Overflow)?;
        w.add(usize::try_from(owner).map_err(|_| CapturePlanCopyError::Overflow)?)?;
        let slots = [0usize, 1, 2, 3];
        let slots = &slots[..layout.len()];
        let selections = w.vector(slots, |w, &ordinal| {
            let descriptor = layout
                .descriptor(ordinal)
                .expect("bounded evidence ordinal");
            Ok(CaptureSelection {
                id: w.text_parts(&descriptor.selection_id_parts())?,
                path: w.text_parts(&descriptor.path_parts())?,
                schedule: operation.schedule.clone(),
                slices: w.vector(&operation.slices, |w, s| {
                    Ok(CaptureSlice {
                        axis: w.text(&s.axis)?,
                        start: s.start,
                        end: s.end,
                        stride: s.stride,
                    })
                })?,
                transform: descriptor.transform(),
            })
        })?;
        let points = w.vector(slots, |w, &ordinal| {
            let descriptor = layout
                .descriptor(ordinal)
                .expect("bounded evidence ordinal");
            Ok(crate::ObservationPoint {
                path: w.text_parts(&descriptor.path_parts())?,
                node_id: w.text(&point.node_id)?,
                meaning: String::new(),
                value_type: ObservationValueType::Tensor,
                dtype: descriptor.dtype(),
                axes: Some(w.vector(&point.axes, |w, a| {
                    Ok(TensorAxis {
                        name: w.text(&a.name)?,
                        dimension: a.dimension.clone(),
                    })
                })?),
                prefill: true,
                decode: true,
                requirements: Vec::new(),
                position: descriptor.position(),
                retained_bytes: None,
                host_bytes: None,
            })
        })?;
        Ok(Some(Self {
            operation: index,
            geometry: EvidenceGeometry::Copied(AdmittedCapturePlan {
                plan: CapturePlan {
                    schema_version: CAPTURE_SCHEMA_VERSION,
                    selections,
                    limits: CapturePlan::none().limits,
                },
                points,
                request: source.request(),
                invocation_bounds: source.invocation_bounds(),
                text_origin: source.text_origin().unwrap_or_default(),
                // Geometry participates in cross-rank evidence receipts. Its
                // descriptive identity excludes each rank's loaded session;
                // original operation and shared-storage authority remain local.
                identity: w.text(source.intent_identity())?,
            }),
        }))
    }
}
