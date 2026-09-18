//! Actual original operation and paid local geometry, without native authority.
use super::*;
use crate::working_memory::OriginalInterventionSource;
use eredu_nn::workspace::{HostMetadataFunding, HostMetadataFundingError};
use std::mem::{size_of, size_of_val};

#[derive(Debug, thiserror::Error)]
enum Cause {
    #[error("{0}")]
    Coordinates(#[from] eredu_core::component::ComponentCoordinateConstructionError),

    #[error(transparent)] Geometry(#[from] CaptureError),
    #[error(transparent)] SourceGeometry(#[from] eredu_core::intervention::InterventionGeometryError),
    #[error(transparent)] Projection(#[from] CaptureContiguousProjectionError),
    #[error(transparent)] Funding(#[from] HostMetadataFundingError),
    #[error(transparent)] Metadata(#[from] eredu_nn::Error),
    #[error(transparent)] Payload(#[from] PreparedWindowInterventionPayloadError),
}
/// A failed local source keeps the original admission and actual metadata owner.
#[derive(Debug, thiserror::Error)]
#[error("original partition intervention projection: {cause}")]
pub struct PartitionInterventionProjectionSourceError {
    #[source] cause: Cause,
    _source: OriginalInterventionSource,
    _funding: HostMetadataFunding,
}
#[derive(Debug)]
enum Geometry {
    Empty,
    Columns(ResolvedCaptureSlice),
    Fragments(CaptureSlicePartition),
}
impl Geometry {
    fn len(&self) -> usize { match self { Self::Empty => 0, Self::Columns(_) => 1,
        Self::Fragments(projection) => projection.fragments().len() } }
    fn region(&self, index: usize) -> (&ResolvedCaptureSlice, Option<&ResolvedCaptureSlice>) {
        match self {
            Self::Columns(slice) if index == 0 => (slice, None),
            Self::Fragments(projection) => { let fragment = &projection.fragments()[index];
                (fragment.local(), Some(fragment.destination())) },
            _ => unreachable!("validated projected region ordinal"),
        }
    }
}
#[derive(Debug)]
struct Update { region: usize, payload: Option<PreparedWindowInterventionPayload> }
/// Borrowed concrete local update. It cannot mint a claim or native budget.
pub struct PartitionInterventionUpdate<'a> {
    /// Actual local tensor slice, preserving the original selected coordinates.
    pub slice: &'a ResolvedCaptureSlice,
    /// Original or exactly projected payload, with additive offset semantics.
    pub action: &'a InterventionAction,
}
/// Move-only original source geometry. Payloads and coordinate copies are funded
/// before allocation; claims, member votes and completion remain separate.
#[derive(Debug)]
pub struct PreparedPartitionInterventionProjection {
    operation: usize,
    phase: CapturePhase,
    prediction: u64,
    invocation: Option<CaptureInvocationShape>,
    axis: usize,
    global: Vec<u64>,
    local: Vec<u64>,
    coordinates: ComponentCoordinateMap,
    local_columns: bool,
    sum_offset_owner: Option<bool>,
    geometry: Geometry,
    updates: Vec<Update>,
    clear: Option<InterventionAction>,
    source: OriginalInterventionSource,
    _funding: HostMetadataFunding,
}
impl PreparedPartitionInterventionProjection {
    /// Construct from the retained architecture axis/map and original schedule.
    /// No model, tensor, capture quota or numerical source is synthesized here.
    pub fn prepare(source: &OriginalInterventionSource, operation: usize, phase: CapturePhase,
        prediction: u64, invocation: Option<CaptureInvocationShape>, global: &[u64], axis: usize,
        coordinates: &ComponentCoordinateMap, sum_offset_owner: Option<bool>, maximum_regions: usize,
        funding: HostMetadataFunding) -> Result<Self, PartitionInterventionProjectionSourceError> {
        let result: Result<Self, Cause> = (|| {
            funding.reserve_metadata(Self::control_bytes().ok_or(CaptureError::Overflow)?)?;
            let plan = source.plan().admission();
            let action = &plan.plan().operations.get(operation).ok_or_else(|| invalid("unknown original intervention"))?.action;
            let dtype = action.dtype().ok_or_else(|| invalid("original partition projection requires a dense action"))?;
            if maximum_regions == 0 || global.get(axis).copied() != Some(coordinates.global_count() as u64) {
                return Err(invalid("original partition axis or region limit differs").into());
            }
            if sum_offset_owner.is_some() { validate_sum_coordinates(coordinates, action)?; }
            let mut selected = slice(global.len(), &funding)?;
            match invocation {
                Some(invocation) => plan.resolve_prepared_invocation_at(operation, phase, prediction,
                    invocation, global, dtype, &mut selected),
                None => plan.resolve_prepared_at(operation, phase, prediction, global, dtype, &mut selected),
            }?;
            let local_columns = validate_partition_column_region(action, global, &selected)
                .map_err(|_| invalid("column intervention requires the complete global final axis"))? && axis + 1 == global.len();
            let global = copy(global, &funding)?;
            let mut local = copy(&global, &funding)?;
            local[axis] = coordinates.local_count() as u64;
            let coordinates = coordinates.try_clone_with_funding(&funding)?;
            let geometry = if sum_offset_owner == Some(false) && matches!(action, InterventionAction::Add { .. }) {
                Geometry::Empty
            } else if local_columns {
                if coordinates.local_count() == 0 { Geometry::Empty } else {
                    selected.ends[axis] = local[axis];
                    selected.shape[axis] = local[axis];
                    Geometry::Columns(selected)
                }
            } else {
                let projection = CaptureCoordinateProjectionPlan::prepare(&global, &selected, axis, &coordinates, maximum_regions)?;
                funding.reserve_metadata(projection.requested_bytes())?;
                Geometry::Fragments(projection.construct())
            };
            let clear = (sum_offset_owner == Some(false) && matches!(action, InterventionAction::Replace { .. }))
                .then_some(InterventionAction::Zero { dtype });
            let effective = clear.as_ref().unwrap_or(action);
            let mut updates = funding.metadata_vec(geometry.len())?;
            for region in 0..geometry.len() {
                let (_, destination) = geometry.region(region);
                let payload = if local_columns {
                    Some(PreparedWindowInterventionPayload::prepare_columns(effective, &coordinates, funding.clone())?)
                } else {
                    Some(PreparedWindowInterventionPayload::prepare(effective,
                        destination.expect("ordinary scalar fragment destination"), funding.clone())?)
                };
                let projected = payload.as_ref().and_then(PreparedWindowInterventionPayload::projected_action).unwrap_or(effective);
                if matches!(projected, InterventionAction::MaskLogits { token_ids, .. } if token_ids.is_empty()) { continue; }
                // The immutable admitted action was resolved above; the shared
                // payload worker validates every copied destination and the
                // coordinate worker validates the complete ID set before local
                // selection. Re-running the ordinary allocating set validator
                // here would create a second, unfunded source construction.
                updates.push(Update { region, payload });
            }
            Ok(Self { operation, phase, prediction, invocation, axis, global, local, coordinates,
                local_columns, sum_offset_owner, geometry, updates, clear, source: source.clone(), _funding: funding.clone() })
        })();
        result.map_err(|cause| PartitionInterventionProjectionSourceError { cause, _source: source.clone(), _funding: funding })
    }
    /// Same ordinary component projection charge, before filtered native updates.
    /// Compact masks include their original ID preparation even if no local IDs survive.
    pub fn projection_usage(&self) -> Result<CaptureUsage, CaptureError> {
        let plan = self.source.plan().admission();
        let original = &plan.plan().operations[self.operation].action;
        let action = self.clear.as_ref().unwrap_or(original);
        let mut cost = PartitionInterventionProjectionCost::new(plan, self.local.len())?;
        for region in 0..self.geometry.len() {
            cost.include(action, &self.geometry.region(region).0.shape)?;
        }
        Ok(cost.usage())
    }
    /// Retained global component axis, before row-window composition.
    pub fn component_axis(&self) -> usize { self.axis }
    /// The same paid immutable architecture map used by the operation projector.
    /// Borrowing it does not issue a capture fragment, native source or quota.
    pub fn coordinates(&self) -> &ComponentCoordinateMap { &self.coordinates }
    /// Actual local shape, including real empty shards.
    pub fn local_shape(&self) -> &[u64] { &self.local }
    /// Original global source shape, before coordinate projection.
    pub fn global_shape(&self) -> &[u64] { &self.global }
    /// Actual projected update count. An empty member still joins source votes.
    pub fn update_count(&self) -> usize { self.updates.len() }
    /// Original source, required again when a native claim is spent.
    pub fn source(&self) -> &OriginalInterventionSource { &self.source }
    /// Original scheduled operation and physical invocation coordinate.
    pub fn coordinate(&self) -> (usize, CapturePhase, u64, Option<CaptureInvocationShape>) {
        (self.operation, self.phase, self.prediction, self.invocation)
    }
    /// Borrow one finite ordinary local update, with no detached payload copy.
    pub fn update(&self, index: usize) -> Option<PartitionInterventionUpdate<'_>> {
        let update = self.updates.get(index)?;
        let (slice, _) = self.geometry.region(update.region);
        let original = &self.source.plan().admission().plan().operations[self.operation].action;
        let action = update.payload.as_ref().and_then(PreparedWindowInterventionPayload::projected_action)
            .or(self.clear.as_ref()).unwrap_or(original);
        Some(PartitionInterventionUpdate { slice, action })
    }
    /// Bind the same architecture source; equal extents cannot replace a map.
    pub fn matches_layout(&self, axis: usize, coordinates: &ComponentCoordinateMap,
        sum_offset_owner: Option<bool>) -> bool {
        self.axis == axis && self.coordinates == *coordinates && self.sum_offset_owner == sum_offset_owner
    }
    /// Exact ordinary digest consumed by existing member/world coordination.
    pub fn geometry_identity(&self) -> [u8; 32] {
        geometry_identity::identity(self.source.plan().admission(), self.operation, self.local_columns,
            self.sum_offset_owner, &self.local, &self.coordinates, self.geometry.len(), |index| self.geometry.region(index))
    }
    /// Fixed source controls; every variable destination uses actual metadata_vec
    /// or the existing checked coordinate projection constructor source.
    pub fn control_bytes() -> Option<usize> {
        let frames = [size_of::<Self>(), size_of::<Cause>(), size_of::<PartitionInterventionProjectionSourceError>(),
            size_of::<OriginalInterventionSource>(), size_of::<HostMetadataFunding>(), size_of::<Geometry>(),
            size_of::<Update>(), size_of::<ResolvedCaptureSlice>(), size_of::<[Vec<u64>; 6]>(),
            size_of::<Result<Self, Cause>>(), size_of::<Result<Self, PartitionInterventionProjectionSourceError>>(),
            size_of::<(usize, CapturePhase, u64, Option<CaptureInvocationShape>, usize, Option<bool>, usize)>(),
            size_of::<(&OriginalInterventionSource, &[u64], &ComponentCoordinateMap, &HostMetadataFunding)>(),
            size_of::<Option<InterventionAction>>(), size_of::<Option<PreparedWindowInterventionPayload>>(),
            size_of::<Sha256>(), size_of::<(&Geometry, usize)>(), size_of::<(usize, &[u64], &ComponentCoordinateMap)>(),
            size_of::<Result<Vec<u64>, eredu_nn::Error>>(), size_of::<Result<Vec<Update>, eredu_nn::Error>>(),
            PartitionInterventionProjectionCost::control_bytes()?];
        frames.into_iter().try_fold(size_of_val(&frames), usize::checked_add)
    }
}
fn copy(values: &[u64], funding: &HostMetadataFunding) -> Result<Vec<u64>, Cause> {
    let mut copy = funding.metadata_vec(values.len())?;
    copy.extend_from_slice(values);
    Ok(copy)
}
fn slice(rank: usize, funding: &HostMetadataFunding) -> Result<ResolvedCaptureSlice, Cause> {
    fn axis(rank: usize, funding: &HostMetadataFunding) -> Result<Vec<u64>, Cause> {
        let mut values = funding.metadata_vec(rank)?; values.resize(rank, 0); Ok(values)
    }
    Ok(ResolvedCaptureSlice { starts: axis(rank, funding)?, ends: axis(rank, funding)?,
        strides: axis(rank, funding)?, shape: axis(rank, funding)? })
}

use eredu_nn::workspace::WorkspaceMetadataAllocation;
