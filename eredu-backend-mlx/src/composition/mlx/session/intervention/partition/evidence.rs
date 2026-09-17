//! Original companion geometry and actual local before/after scalar witnesses.
use super::*;
use eredu_architectures::component_partition::ComponentPartitionCaptureSource;
use eredu_nn::workspace::WorkspaceFloatingType;
use eredu_runtime::capture::partition::{PartitionInvocationCaptureGeometry,PartitionPrefillCaptureGeometry};
use crate::composition::mlx::session::capture_workspace;

/// A descriptive peer row becomes a native source only through both actual
/// local value traces. The receipt still owns export, budget and spent state.
pub(in crate::composition::mlx::session) struct PreparedPartitionEvidenceSource {
    source: OriginalInterventionSource,
    companion: SharedCapturePlan,
    operation: usize,
    projection_identity: [u8;32],
    phase: CapturePhase,
    prediction: u64,
    rank: usize,
    produces: bool,
    combination: PartitionCaptureCombination,
    projections: [CaptureSlicePartition;2],
    shape: Vec<u64>,
    window: Option<InterventionPrefillWindow>,
    attempted: [bool;2],
    scalar: [Option<WorkspaceFloatingType>;2],
    funding: WorkspaceMetadataFunding,
}
impl PreparedPartitionEvidenceSource {
    /// Borrow the architecture's ordinary exporter decision, independently of
    /// edit membership. All owning projection vectors are paid before copying.
    pub(in crate::composition::mlx::session) fn prepare(
        projection:&PreparedPartitionInterventionProjection,
        selected:&ComponentPartitionCaptureSource<'_>,rank:usize,
        shape:&[u64],window:Option<InterventionPrefillWindow>,context:&WorkspaceContext,
    )->Result<Self> {
        context.charge_metadata(Self::control_bytes().ok_or(WorkspaceMetadataError::Overflow)?)?;
        let funding=context.metadata_funding().ok_or(WorkspaceMetadataError::Unqualified)?;
        let (operation,phase,prediction,invocation)=projection.coordinate();
        let original=projection.source();
        let point=original.plan().admission().points().get(operation)
            .ok_or_else(||context.metadata_source(Failure::ClaimMismatch))?;
        let companion=original.plan().evidence(operation)
            .ok_or_else(||context.metadata_source(Failure::ClaimMismatch))?.shared_geometry_source();
        let source=selected.rank(rank).ok_or_else(||context.metadata_source(Failure::SourceChanged))?;
        if selected.path()!=point.path || invocation.is_some() || companion.admission().plan().selections.len()!=2
            || source.coordinates!=projection.coordinates()
            || point.axes.get(projection.component_axis()).is_none_or(|axis|axis.name!=selected.axis())
            || projection.global_shape().get(projection.component_axis()).copied()!=u64::try_from(selected.width()).ok()
            || shape.len()!=projection.local_shape().len() {
            return Err(context.metadata_source(Failure::ShapeMismatch));
        }
        if let Some(window)=window {
            window.validate(original.plan().admission()).map_err(|e|context.metadata_source(e))?;
            if (phase,prediction)!=(CapturePhase::Prefill,0) {return Err(context.metadata_source(Failure::ClaimMismatch));}
        }
        let before=Self::project(companion,0,projection,context)?;
        let after=Self::project(companion,1,projection,context)?;
        let mut physical=context.metadata_vec(shape.len())?;physical.extend_from_slice(shape);
        Ok(Self {source:original.clone(),companion:companion.clone(),operation,phase,prediction,
            rank,projection_identity:projection.geometry_identity(),produces:source.produces,combination:selected.combination(),
            projections:[before,after],shape:physical,window,attempted:[false;2],scalar:[None;2],funding})
    }
    fn project(companion:&SharedCapturePlan,side:usize,projection:&PreparedPartitionInterventionProjection,
        context:&WorkspaceContext)->Result<CaptureSlicePartition> {
        let admission=companion.admission();let (_,phase,prediction,invocation)=projection.coordinate();
        let point=admission.points().get(side).ok_or_else(||context.metadata_source(Failure::ClaimMismatch))?;
        let selection=&admission.plan().selections[side];
        let axes=point.axes.as_deref().ok_or_else(||context.metadata_source(Failure::ShapeMismatch))?;
        let logical=admission.geometry_at(phase,prediction,invocation).map_err(|e|context.metadata_source(e))?;
        let mut global=vector(axes.len(),context)?;
        if !logical.resolve_axes_into(axes,&mut global).map_err(|e|context.metadata_source(e))?
            || global!=projection.global_shape() {return Err(context.metadata_source(Failure::ShapeMismatch));}
        let mut selected=slice(global.len(),context)?;
        resolve_slice_into(Some(axes),&selection.slices,&global,&mut selected.starts,&mut selected.ends,
            &mut selected.strides,&mut selected.shape).map_err(|e|context.metadata_source(e))?;
        let plan=CaptureCoordinateProjectionPlan::prepare(&global,&selected,projection.component_axis(),
            projection.coordinates(),projection.coordinates().local_count().max(1)).map_err(|e|context.metadata_source(e))?;
        context.charge_metadata(plan.requested_bytes())?;
        Ok(plan.construct())
    }
    pub(super) fn matches(&self,projection:&PreparedPartitionInterventionProjection,
        shape:&[u64],window:Option<InterventionPrefillWindow>)->bool {
        self.source.same_source(projection.source()) && self.projection_identity==projection.geometry_identity() && self.shape==shape && self.window==window
            && projection.coordinate()==(self.operation,self.phase,self.prediction,None)
            && self.projections.iter().all(|source|source.global_shape()==projection.global_shape()
                && source.local_shape()==projection.local_shape() && source.axis()==projection.component_axis())
    }
    /// Use every actual projection fragment and the same ordinary raw/reducer
    /// equation as its eventual writer. No peer tensor or dtype is fabricated.
    pub(super) fn trace(&mut self,side:InterventionEvidenceSide,value:&WorkspaceTensor,
        context:&WorkspaceContext,roots:&mut Vec<WorkspaceTensor>)->Result<CaptureNativePopulation> {
        context.charge_metadata(Self::control_bytes().ok_or(WorkspaceMetadataError::Overflow)?)?;
        let index=side.index();
        if self.attempted[index] || (index==1 && self.scalar[0].is_none())
            || context.metadata_funding().is_none_or(|source|!source.same_account(&self.funding)) {
            return Err(context.metadata_source(Failure::SourceChanged));
        }
        self.attempted[index]=true;
        context.validate_values([value])?;
        if value.shape().len()!=self.shape.len() || value.shape().iter().zip(&self.shape)
            .any(|(&a,&b)|u64::try_from(a).ok()!=Some(b)) {return Err(context.metadata_source(Failure::ShapeMismatch));}
        let scalar=value.layout().representation().map(|source|source.dtype())
            .ok_or_else(||context.metadata_source(Failure::SourceChanged))?;
        let projection=&self.projections[index];
        let mut total=CaptureNativePopulation::default();
        if !self.produces || projection.fragments().is_empty() {
            total=capture_workspace::trace_partition_replica(value,context,roots)?;
        } else {
            for fragment in 0..projection.fragments().len() {
                let population=if let Some(window)=self.window {
                    let inference=window.inference();
                    let ordinal=window.range()[0].checked_div(inference.prefill_chunk_positions)
                        .ok_or(WorkspaceMetadataError::Overflow)?;
                    let source=PartitionPrefillCaptureGeometry::prepare(self.companion.admission(),index,
                        projection,fragment,self.combination,inference).map_err(|e|context.metadata_source(e))?;
                    capture_workspace::trace_partition_prefill_geometry(source,self.rank,ordinal,value,context,roots)?
                } else {
                    let source=PartitionInvocationCaptureGeometry::prepare(self.companion.admission(),index,
                        self.phase,self.prediction,projection,fragment,self.combination).map_err(|e|context.metadata_source(e))?;
                    capture_workspace::trace_partition_invocation_geometry(source,self.rank,value,context,roots)?
                };
                total=total.checked_add(population).ok_or(WorkspaceMetadataError::Overflow)?;
            }
        }
        self.scalar[index]=Some(scalar);
        Ok(total)
    }
    /// Validate the actual native value against the original operation and the
    /// specific cold side witness before any partition writer is lent.
    pub(in crate::composition::mlx::session) fn validate_native(&self,value:&Array,
        claim:&CaptureInterventionClaim<'_>,side:InterventionEvidenceSide,
        window:Option<InterventionPrefillWindow>,scope:&WorkingMemoryFundingScope,
    )->std::result::Result<(),NativeFailure> {
        claim.validate_native_custody(scope)?;claim.validate_source(&self.source)?;
        if claim.index()!=self.operation || claim.coordinate()!=(self.phase,self.prediction)
            || claim.invocation().is_some() || claim.invocation_window().is_some() || window!=self.window {
            return Err(Failure::ClaimMismatch.into());
        }
        let scalar=match value.dtype() {
            safemlx::Dtype::Float32=>WorkspaceFloatingType::Float32,
            safemlx::Dtype::Float16=>WorkspaceFloatingType::Float16,
            safemlx::Dtype::Bfloat16=>WorkspaceFloatingType::Bfloat16,
            other=>return Err(Failure::UnsupportedDtype(other).into()),
        };
        if self.scalar[side.index()]!=Some(scalar) || value.shape().len()!=self.shape.len()
            || value.shape().iter().zip(&self.shape).any(|(&a,&b)|u64::try_from(a).ok()!=Some(b)) {
            return Err(Failure::ShapeMismatch.into());
        }
        Ok(())
    }
    pub(in crate::composition::mlx::session) fn source(&self)->&SharedCapturePlan {&self.companion}
    pub(in crate::composition::mlx::session) fn projection(&self,side:InterventionEvidenceSide)->&CaptureSlicePartition {&self.projections[side.index()]}
    pub(in crate::composition::mlx::session) fn rank(&self)->usize {self.rank}
    pub(in crate::composition::mlx::session) fn produces(&self)->bool {self.produces}
    pub(in crate::composition::mlx::session) fn combination(&self)->PartitionCaptureCombination {self.combination}
    pub(in crate::composition::mlx::session) fn scalar(&self,side:InterventionEvidenceSide)->Option<WorkspaceFloatingType> {self.scalar[side.index()]}
    pub(super) fn control_bytes()->Option<usize> {
        let frames=[size_of::<Self>()*2,size_of::<Result<Self>>(),size_of::<Result<CaptureSlicePartition>>(),
            size_of::<[CaptureSlicePartition;2]>(),size_of::<CaptureCoordinateProjectionPlan<'_>>(),
            size_of::<(&PreparedPartitionInterventionProjection,&ComponentPartitionCaptureSource<'_>,usize,&[u64],Option<InterventionPrefillWindow>,&WorkspaceContext)>(),
            size_of::<(&SharedCapturePlan,usize,&PreparedPartitionInterventionProjection,&WorkspaceContext)>(),
            size_of::<(&mut Self,InterventionEvidenceSide,&WorkspaceTensor,&WorkspaceContext,&mut Vec<WorkspaceTensor>)>(),
            size_of::<(&Self,&Array,&CaptureInterventionClaim<'_>,InterventionEvidenceSide,Option<InterventionPrefillWindow>,&WorkingMemoryFundingScope)>(),
            size_of::<std::result::Result<(),NativeFailure>>(),size_of::<Result<CaptureNativePopulation>>(),
            size_of::<[CaptureNativePopulation;2]>(),size_of::<[ResolvedCaptureSlice;2]>(),
            size_of::<[Vec<u64>;2]>(),size_of::<Option<WorkspaceFloatingType>>(),
            ComponentPartitionCaptureSource::control_bytes()?,
            PartitionInvocationCaptureGeometry::control_bytes()?,PartitionPrefillCaptureGeometry::control_bytes()?];
        frames.into_iter().try_fold(size_of_val(&frames),usize::checked_add)
    }
}
