//! Paid component-then-window updates over actual local native source geometry.
use super::{NativeInterventionEstimator, PreparedStaticActivation};
use crate::backend::array_copy::{
    CaptureCompletion, CaptureNativePopulation, CaptureTensorNativeError as Failure,
};
use eredu_architectures::component_partition::PartitionInterventionMemberLayout;
use eredu_core::{capture::*, intervention::*};
use eredu_nn::{
    workspace::{
        WorkspaceContext, WorkspaceMetadataError, HostMetadataFunding, WorkspaceTensor,
    },
    Tensor,
};
use eredu_runtime::{
    capture::partition::PartitionInterventionLocalAllowance,
    working_memory::{CaptureInterventionClaim, WorkingMemoryFundingScope},
};
use eredu_runtime::{
    capture::CaptureProtocolError,
    intervention::{
        intervention_window_metadata, InterventionPrefillWindow,
        PartitionInterventionProjectionCost, PreparedPartitionInterventionProjection,
        PreparedWindowInterventionPayload,
    },
    working_memory::OriginalInterventionSource,
};
use safemlx::{Array, OriginalScopeObserver, Stream};
use std::{
    cell::RefCell,
    mem::{size_of, size_of_val},
};
mod identity;
mod evidence;
pub(in crate::composition::mlx::session) use evidence::PreparedPartitionEvidenceSource;

type Result<T> = std::result::Result<T, eredu_nn::Error>;
struct Update {
    region: usize,
    slice: ResolvedCaptureSlice,
    payload: Option<PreparedWindowInterventionPayload>,
}
/// One actual executing member of an original operation. Claims, global quota,
/// peer votes and final outcome remain with the runtime partition program.
pub(crate) struct PreparedPartitionModelIntervention {
    projection: PreparedPartitionInterventionProjection,
    shape: Vec<u64>,
    dtype: InterventionDtype,
    window: Option<InterventionPrefillWindow>,
    updates: Vec<Update>,
    usage: CaptureUsage,
    projection_usage: [CaptureUsage; 2],
    source_usage: CaptureUsage,
    execution_identity: [u8; 32],
    funding: HostMetadataFunding,
    traced: bool,
    evidence: Option<PreparedPartitionEvidenceSource>,
}
#[derive(Debug, thiserror::Error)]
#[error("original partition intervention native source: {cause}")]
struct SourceFailure {
    #[source]
    cause: eredu_nn::Error,
    _source: OriginalInterventionSource,
    _funding: HostMetadataFunding,
}
impl PreparedPartitionModelIntervention {
    /// Trace the ordinary static worker for every actual projected update. The
    /// component map is applied before intersecting the canonical token window.
    /// Logical costs are descriptive; only the shared runtime program spends them.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prepare_trace(
        source: &OriginalInterventionSource,
        operation: usize,
        phase: CapturePhase,
        prediction: u64,
        invocation: Option<CaptureInvocationShape>,
        member: PartitionInterventionMemberLayout<'_>,
        window: Option<InterventionPrefillWindow>,
        maximum_regions: usize,
        input: &WorkspaceTensor,
        wait: eredu_core::BoundedCompletionWait,
        context: &WorkspaceContext,
        retained: &mut Vec<WorkspaceTensor>,
    ) -> Result<(Self, Option<WorkspaceTensor>, CaptureNativePopulation)> {
        context.charge_metadata(Self::control_bytes().ok_or(WorkspaceMetadataError::Overflow)?)?;
        let funding = context
            .metadata_funding()
            .ok_or(WorkspaceMetadataError::Unqualified)?;
        let attempted = (|| {
            context.validate_values([input])?;
            let plan = source.plan().admission();
            let point = plan
                .points()
                .get(operation)
                .ok_or_else(|| context.metadata_source(CaptureProtocolError::Geometry))?;
            let declaration = &plan.plan().operations[operation];
            if point.routing.is_some()
                || point.routed_units.is_some()
                || declaration.evidence != InterventionEvidence::None
            {
                return Err(context.metadata_source(CaptureProtocolError::Geometry));
            }
            let dtype = declaration
                .action
                .dtype()
                .ok_or_else(|| context.metadata_source(Failure::ShapeMismatch))?;
            PreparedStaticActivation::validate_workspace_source(input, dtype)
                .map_err(|e| context.metadata_source(e))?;
            let logical = plan
                .geometry_at(phase, prediction, invocation)
                .map_err(|e| context.metadata_source(e))?;
            let mut global = vector(point.axes.len(), context)?;
            if !logical
                .resolve_axes_into(&point.axes, &mut global)
                .map_err(|e| context.metadata_source(e))?
            {
                return Err(context.metadata_source(CaptureProtocolError::Geometry));
            }
            let projection = PreparedPartitionInterventionProjection::prepare(
                source,
                operation,
                phase,
                prediction,
                invocation,
                &global,
                member.axis(),
                member.coordinates(),
                member.sum_offset_owner(),
                maximum_regions,
                funding.clone(),
            )
            .map_err(|e| context.metadata_source(e))?;
            Self::trace_projection(projection,window,input,wait,context,retained)
        })();
        attempted.map_err(|cause| {
            context.metadata_source(SourceFailure {
                cause,
                _source: source.clone(),
                _funding: funding,
            })
        })
    }
    /// Consume the previously bound paid component source at its actual hook.
    /// The pending owner is never reconstructed from a caller's local shape.
    pub(crate) fn prepare_projection(projection: PreparedPartitionInterventionProjection,
        window: Option<InterventionPrefillWindow>, physical_shape: &[u64],
        wait: eredu_core::BoundedCompletionWait, context: &WorkspaceContext,
        ) -> Result<Self> {
        Self::prepare_projection_with_evidence(projection,window,physical_shape,wait,None,context)
    }
    pub(in crate::composition::mlx::session) fn prepare_projection_with_evidence(
        projection:PreparedPartitionInterventionProjection,window:Option<InterventionPrefillWindow>,
        physical_shape:&[u64],wait:eredu_core::BoundedCompletionWait,
        evidence:Option<PreparedPartitionEvidenceSource>,context:&WorkspaceContext,
    )->Result<Self> {
        context.charge_metadata(Self::control_bytes().ok_or(WorkspaceMetadataError::Overflow)?)?;
        let funding=context.metadata_funding().ok_or(WorkspaceMetadataError::Unqualified)?;
        let source=projection.source().clone();
        let attempted=(|| {
            let (operation,phase,prediction,invocation)=projection.coordinate();
            let plan=source.plan().admission();
            let point=&plan.points()[operation];
            let declaration=&plan.plan().operations[operation];
            if point.routing.is_some() || point.routed_units.is_some()
                || (declaration.evidence!=InterventionEvidence::None)!=evidence.is_some()
                || evidence.as_ref().is_some_and(|source|!source.matches(&projection,physical_shape,window))
                {return Err(context.metadata_source(CaptureProtocolError::Geometry));}
            let dtype=declaration.action.dtype().ok_or_else(||context.metadata_source(Failure::ShapeMismatch))?;
            let mut shape=context.metadata_vec(physical_shape.len())?;
            shape.extend_from_slice(physical_shape);
            let row_axis = if let Some(window) = window {
                if invocation.is_some() || (phase, prediction) != (CapturePhase::Prefill, 0) {
                    return Err(context.metadata_source(CaptureProtocolError::Invocation));
                }
                window
                    .validate(plan)
                    .map_err(|e| context.metadata_source(e))?;
                let valid=if evidence.is_some() {
                    InterventionPrefillWindow::validate_partition_evidence_operation(&source,operation)
                }else {InterventionPrefillWindow::validate_operation(plan,operation)};
                valid.map_err(|e|context.metadata_source(e))?;
                if shape.len() != projection.local_shape().len()
                    || shape.get(projection.component_axis()) != projection.local_shape().get(projection.component_axis())
                {
                    return Err(context.metadata_source(Failure::ShapeMismatch));
                }
                // Validate the actual local component extent first. The shared
                // row validator then receives its retained global-axis geometry,
                // not a fabricated WorkspaceTensor or completed-value witness.
                let mut global_physical = context.metadata_vec(shape.len())?;
                global_physical.extend_from_slice(&shape);
                global_physical[projection.component_axis()] = projection.global_shape()[projection.component_axis()];
                let mut global_logical = vector(shape.len(), context)?;
                let (_, axis) = window
                    .window()
                    .source_axes_into(
                        window.physical(),
                        Some(&point.axes),
                        &global_physical,
                        &mut global_logical,
                    )
                    .map_err(|e| context.metadata_source(e))?;
                if global_logical != projection.global_shape() || axis == projection.component_axis() {
                    return Err(context.metadata_source(Failure::ShapeMismatch));
                }
                Some(axis)
            } else {
                if shape != projection.local_shape() {
                    return Err(context.metadata_source(Failure::ShapeMismatch));
                }
                None
            };
            let mut window_cost = PartitionInterventionProjectionCost::new(plan, shape.len())
                .map_err(|e| context.metadata_source(e))?;
            let mut updates = context.metadata_vec(projection.update_count())?;
            let mut usage = CaptureUsage::default();
            for region in 0..projection.update_count() {
                let original = projection
                    .update(region)
                    .ok_or_else(|| context.metadata_source(CaptureProtocolError::Geometry))?;
                let (slice, payload) = if let Some(axis) = row_axis {
                    let span = window.expect("validated row window");
                    let mut local = slice(shape.len(), context)?;
                    let mut destination = slice(shape.len(), context)?;
                    let [start, end] = span.range();
                    if !CaptureSlicePartition::contiguous_fragment_into(
                        projection.local_shape(),
                        original.slice,
                        axis,
                        start..end,
                        &mut local,
                        &mut destination,
                    )
                    .map_err(|e| context.metadata_source(e))?
                    {
                        continue;
                    }
                    window_cost
                        .include(original.action, &local.shape)
                        .map_err(|e| context.metadata_source(e))?;
                    let payload = PreparedWindowInterventionPayload::prepare(
                        original.action,
                        &destination,
                        funding.clone(),
                    )
                    .map_err(|e| context.metadata_source(e))?;
                    (local, Some(payload))
                } else {
                    (copy_slice(original.slice, context)?, None)
                };
                let action = payload
                    .as_ref()
                    .and_then(PreparedWindowInterventionPayload::projected_action)
                    .unwrap_or(original.action);
                let program = PreparedStaticActivation::new(action, &slice, &shape, dtype)
                    .map_err(|e| context.metadata_source(e))?;
                let cost = NativeInterventionEstimator
                    .activation_usage(&shape, &slice, action)
                    .map_err(|e| context.metadata_source(e))?;
                if cost.captures != 0 || cost.encoded_bytes != 0 {
                    return Err(context.metadata_source(CaptureProtocolError::Geometry));
                }
                // Validate the actual ordinary numerical source before retaining it.
                program
                    .population()
                    .map_err(|e| context.metadata_source(e))?;
                usage = usage
                    .checked_add(cost)
                    .map_err(|e| context.metadata_source(e))?;
                updates.push(Update {
                    region,
                    slice,
                    payload,
                });
            }
            let projection_usage = [
                projection
                    .projection_usage()
                    .map_err(|e| context.metadata_source(e))?,
                if window.is_some() {
                    intervention_window_metadata(declaration, point)
                        .and_then(|m| m.checked_add(window_cost.usage()))
                        .map_err(|e| context.metadata_source(e))?
                } else {
                    CaptureUsage::default()
                },
            ];
            let boundaries = 1 + u64::from(!updates.is_empty());
            let source_usage =
                super::super::bounded_capture::estimate_partition_source(&shape, wait)
                    .and_then(|cost| cost.checked_mul(boundaries))
                    .map_err(|e| context.metadata_source(e))?;
            let mut owner = Self {
                projection,
                shape,
                dtype,
                window,
                updates,
                usage,
                projection_usage,
                source_usage,
                execution_identity: [0; 32],
                funding: funding.clone(),
                traced:false,
                evidence,
            };
            owner.execution_identity =
                identity::identity(&owner).map_err(|e| context.metadata_source(e))?;
            Ok(owner)
        })();
        attempted.map_err(|cause|context.metadata_source(SourceFailure {cause,_source:source,_funding:funding}))
    }
    /// Bind only the actual local tensor to a prepared immutable member source.
    pub(crate) fn trace_projection(projection:PreparedPartitionInterventionProjection,
        window:Option<InterventionPrefillWindow>,input:&WorkspaceTensor,wait:eredu_core::BoundedCompletionWait,
        context:&WorkspaceContext,retained:&mut Vec<WorkspaceTensor>)
        ->Result<(Self,Option<WorkspaceTensor>,CaptureNativePopulation)> {
        if input.shape().len()>32 {return Err(context.metadata_source(Failure::ShapeMismatch));}
        let mut shape=[0u64;32];
        for (out,&actual) in shape.iter_mut().zip(input.shape()) {*out=u64::try_from(actual).map_err(|e|context.metadata_source(e))?;}
        let mut owner=Self::prepare_projection(projection,window,&shape[..input.shape().len()],wait,context)?;
        let (output,population)=owner.trace_bound(input,context,retained)?;
        Ok((owner,output,population))
    }
    /// A peer descriptor has no native binding. Only this actual local source
    /// trace can make the retained program usable by a native allowance.
    pub(crate) fn trace_bound(&mut self,input:&WorkspaceTensor,context:&WorkspaceContext,
        retained:&mut Vec<WorkspaceTensor>)->Result<(Option<WorkspaceTensor>,CaptureNativePopulation)> {
        context.charge_metadata(Self::control_bytes().ok_or(WorkspaceMetadataError::Overflow)?)?;
        if self.traced || context.metadata_funding().is_none_or(|funding|!funding.same_account(&self.funding)) {
            return Err(context.metadata_source(Failure::SourceChanged));
        }
        context.validate_values([input])?;
        PreparedStaticActivation::validate_workspace_source(input,self.dtype).map_err(|e|context.metadata_source(e))?;
        if input.shape().len()!=self.shape.len() || input.shape().iter().zip(&self.shape)
            .any(|(&a,&b)|u64::try_from(a).ok()!=Some(b)) {return Err(context.metadata_source(Failure::ShapeMismatch));}
            let mut population = self.trace_boundary(input, context, retained)?;
            population.controls = population
                .controls
                .checked_add(
                    Self::execution_control_bytes().ok_or(WorkspaceMetadataError::Overflow)?,
                )
                .ok_or(WorkspaceMetadataError::Overflow)?;
            if let Some(evidence)=&mut self.evidence {
                population=population.checked_add(evidence.trace(InterventionEvidenceSide::Before,input,context,retained)?)
                    .ok_or(WorkspaceMetadataError::Overflow)?;
            }
            let mut output = None;
            for index in 0..self.updates.len() {
                let program = self
                    .program(index)
                    .map_err(|e| context.metadata_source(e))?;
                let native = program
                    .population()
                    .map_err(|e| context.metadata_source(e))?;
                let next = program
                    .trace(output.as_ref().unwrap_or(input), context, retained)
                    .map_err(|e| context.metadata_source(e))?;
                population = population
                    .checked_add(CaptureNativePopulation {
                        publications: 0,
                        completions: native.completions,
                        retained_roots: native.retained_roots,
                        controls: native
                            .controls
                            .checked_add(native.host_bytes)
                            .ok_or(WorkspaceMetadataError::Overflow)?,
                    })
                    .ok_or(WorkspaceMetadataError::Overflow)?;
                output = Some(next);
            }
            if let Some(output) = &output {
                population = population
                    .checked_add(self.trace_boundary(output, context, retained)?)
                    .ok_or(WorkspaceMetadataError::Overflow)?;
            }
            if let Some(evidence)=&mut self.evidence {
                population=population.checked_add(evidence.trace(InterventionEvidenceSide::After,
                    output.as_ref().unwrap_or(input),context,retained)?)
                    .ok_or(WorkspaceMetadataError::Overflow)?;
            }
            self.traced=true;
            Ok((output,population))
    }
    fn program(&self, index: usize) -> std::result::Result<PreparedStaticActivation<'_>, Failure> {
        let update = self.updates.get(index).ok_or(Failure::ClaimMismatch)?;
        let original = self
            .projection
            .update(update.region)
            .ok_or(Failure::ClaimMismatch)?;
        let action = update
            .payload
            .as_ref()
            .and_then(PreparedWindowInterventionPayload::projected_action)
            .unwrap_or(original.action);
        PreparedStaticActivation::new(action, &update.slice, &self.shape, self.dtype)
    }
    fn trace_boundary(
        &self,
        input: &WorkspaceTensor,
        context: &WorkspaceContext,
        retained: &mut Vec<WorkspaceTensor>,
    ) -> Result<CaptureNativePopulation> {
        context.reserve_metadata_vec(retained, 1)?;
        retained.push(input.clone());
        Ok(CaptureNativePopulation {
            publications: 0,
            completions: 1,
            retained_roots: 1,
            controls: CaptureCompletion::retained_settlement_control_bytes::<1>()
                .ok_or(WorkspaceMetadataError::Overflow)?,
        })
    }
    /// Side-effect-free local preflight before the first shared member vote.
    pub(crate) fn validate_source(
        &self,
        value: &Array,
        claim: &CaptureInterventionClaim<'_>,
        scope: &WorkingMemoryFundingScope,
    ) -> std::result::Result<(), NativeFailure> {
        if !self.traced {return Err(Failure::SourceChanged.into());}
        claim.validate_native_custody(scope)?;
        claim.validate_source(self.projection.source())?;
        let (operation, phase, prediction, invocation) = self.projection.coordinate();
        if claim.index() != operation
            || claim.coordinate() != (phase, prediction)
            || claim.invocation() != invocation
            || claim.invocation_window().is_some()
        {
            return Err(Failure::ClaimMismatch.into());
        }
        let dtype = match value.dtype() {
            safemlx::Dtype::Float32 => InterventionDtype::Float32,
            safemlx::Dtype::Float16 => InterventionDtype::Float16,
            safemlx::Dtype::Bfloat16 => InterventionDtype::Bfloat16,
            other => return Err(Failure::UnsupportedDtype(other).into()),
        };
        if dtype != self.dtype
            || value.shape().len() != self.shape.len()
            || value
                .shape()
                .iter()
                .zip(&self.shape)
                .any(|(&actual, &expected)| u64::try_from(actual).ok() != Some(expected))
        {
            return Err(Failure::ShapeMismatch.into());
        }
        Ok(())
    }
    /// Consume only the exact local allowance after its first member vote.
    /// The original claim is borrowed: the runtime still owns final completion,
    /// the second member vote and the global receipt, including failed prefixes.
    pub(crate) fn execute(
        &self,
        value: &Array,
        claim: &CaptureInterventionClaim<'_>,
        allowance: &mut PartitionInterventionLocalAllowance,
        scope: &WorkingMemoryFundingScope,
        stream: &Stream,
        observer: &OriginalScopeObserver,
        roots: &RefCell<Vec<Array>>,
    ) -> std::result::Result<Option<Array>, NativeFailure> {
        self.validate_source(value, claim, scope)?;
        allowance.validate(
            claim,
            &self.projection,
            self.window,
            self.execution_identity,
            &self.shape,
            self.usage,
            self.projection_usage,
            self.source_usage,
        )?;
        crate::backend::array_copy::PreparedCaptureTensor::validate_stream(stream)?;
        let completion = CaptureCompletion::Original(observer);
        completion.validate_identity()?;
        allowance.charge_projection(self.projection_usage)?;
        allowance.charge_source(self.source_usage)?;
        allowance.charge_execution(self.usage)?;
        completion.settle_retained([value], stream, roots)?;
        let mut output = None;
        for index in 0..self.updates.len() {
            let program = self.program(index)?;
            output = Some(program.execute_array(
                output.as_ref().unwrap_or(value),
                stream,
                observer,
                roots,
            )?);
        }
        if let Some(output) = &output {
            completion.settle_retained([output], stream, roots)?;
        }
        Ok(output)
    }
    fn execution_control_bytes() -> Option<usize> {
        let frames = [
            size_of::<NativeFailure>(),
            size_of::<std::result::Result<Option<Array>, NativeFailure>>(),
            size_of::<std::result::Result<(), NativeFailure>>(),
            size_of::<[Option<Array>; 2]>(),
            size_of::<(
                &Self,
                &Array,
                &CaptureInterventionClaim<'_>,
                &mut PartitionInterventionLocalAllowance,
                &WorkingMemoryFundingScope,
                &Stream,
                &OriginalScopeObserver,
                &RefCell<Vec<Array>>,
            )>(),
            size_of::<PreparedStaticActivation<'_>>(),
            size_of::<CaptureCompletion<'_>>(),
            PartitionInterventionLocalAllowance::control_bytes()?,
            safemlx::Stream::device_type_control_bytes()?,
            OriginalScopeObserver::control_bytes()?,
        ];
        frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)
    }
    pub(in crate::composition::mlx::session) fn evidence(&self)->Option<&PreparedPartitionEvidenceSource> {
        self.evidence.as_ref()
    }
    pub(crate) fn projection(&self) -> &PreparedPartitionInterventionProjection {
        &self.projection
    }
    pub(crate) fn shape(&self) -> &[u64] {
        &self.shape
    }
    pub(crate) fn window(&self) -> Option<InterventionPrefillWindow> {
        self.window
    }
    pub(crate) fn usage(&self) -> CaptureUsage {
        self.usage
    }
    pub(crate) fn projection_usage(&self) -> [CaptureUsage; 2] {
        self.projection_usage
    }
    pub(crate) fn source_usage(&self) -> CaptureUsage {
        self.source_usage
    }
    pub(crate) fn execution_identity(&self) -> [u8; 32] {
        self.execution_identity
    }
    pub(crate) fn control_bytes() -> Option<usize> {
        let frames = [
            size_of::<Self>(),
            size_of::<Update>(),
            size_of::<SourceFailure>(),
            size_of::<Result<(Self, Option<WorkspaceTensor>, CaptureNativePopulation)>>(),
            size_of::<Vec<Update>>(),
            size_of::<[CaptureUsage; 4]>(),
            size_of::<[Vec<u64>; 4]>(),
            size_of::<[ResolvedCaptureSlice; 2]>(),
            size_of::<Option<PreparedWindowInterventionPayload>>(),
            size_of::<PartitionInterventionMemberLayout<'_>>(),
            identity::control_bytes()?,
            PreparedPartitionEvidenceSource::control_bytes()?,
            size_of::<(PreparedPartitionInterventionProjection,Option<InterventionPrefillWindow>,&[u64],
                eredu_core::BoundedCompletionWait,Option<PreparedPartitionEvidenceSource>,&WorkspaceContext)>(),
            size_of::<(
                &OriginalInterventionSource,
                usize,
                CapturePhase,
                u64,
                Option<CaptureInvocationShape>,
                PartitionInterventionMemberLayout<'_>,
                Option<InterventionPrefillWindow>,
                usize,
                &WorkspaceTensor,
                eredu_core::BoundedCompletionWait,
                &WorkspaceContext,
                &mut Vec<WorkspaceTensor>,
            )>(),
            PreparedStaticActivation::inspection_control_bytes()?,
            NativeInterventionEstimator::prepared_preflight_control_bytes()?,
            PartitionInterventionProjectionCost::control_bytes()?,
            CaptureInvocationWindow::source_axes_control_bytes()?,
            CaptureSlicePartition::contiguous_projection_control_bytes()?,
            InterventionPrefillWindow::control_bytes()?,
            CaptureCompletion::retained_settlement_control_bytes::<1>()?,
        ];
        frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)
    }
}
fn vector(rank: usize, context: &WorkspaceContext) -> Result<Vec<u64>> {
    let mut values = context.metadata_vec(rank)?;
    values.resize(rank, 0);
    Ok(values)
}
fn slice(rank: usize, context: &WorkspaceContext) -> Result<ResolvedCaptureSlice> {
    Ok(ResolvedCaptureSlice {
        starts: vector(rank, context)?,
        ends: vector(rank, context)?,
        strides: vector(rank, context)?,
        shape: vector(rank, context)?,
    })
}
fn copy_slice(
    source: &ResolvedCaptureSlice,
    context: &WorkspaceContext,
) -> Result<ResolvedCaptureSlice> {
    let mut target = slice(source.shape.len(), context)?;
    target.starts.copy_from_slice(&source.starts);
    target.ends.copy_from_slice(&source.ends);
    target.strides.copy_from_slice(&source.strides);
    target.shape.copy_from_slice(&source.shape);
    Ok(target)
}

/// Native failures preserve the original portable source/allowance cause.
#[derive(Debug, thiserror::Error)]
pub(crate) enum NativeFailure {
    #[error(transparent)]
    Native(#[from] Failure),
    #[error(transparent)]
    Custody(#[from] eredu_runtime::working_memory::WorkingMemoryError),
    #[error(transparent)]
    Allowance(#[from] eredu_runtime::capture::partition::PartitionInterventionSourceError),
}

use eredu_nn::workspace::WorkspaceMetadataAllocation;
