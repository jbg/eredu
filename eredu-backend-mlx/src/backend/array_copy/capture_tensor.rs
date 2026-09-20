//! CPU and Metal static selection, half-to-F32 conversion, and settled host iteration.
use super::ExistingArrayProjection;
mod fragment;
mod generated;
pub(crate) use generated::GeneratedCaptureRetention;
mod histogram;
mod reduction;
mod summary;
pub(crate) use histogram::PreparedCaptureHistogram;
pub(crate) use summary::PreparedCaptureSummary;
mod original;
mod speculative;
use crate::backend::runtime::residency::storage::StorageIdentity;
use eredu_core::{
    SharedTensorObservation,
    capture::{CapturePrefillFragment, CaptureTensorGeometry, CaptureTransform},
};
use eredu_nn::{
    Tensor,
    workspace::{
        WorkspaceContext, WorkspaceDtype, WorkspaceLayout, WorkspaceOperationKind, WorkspaceTensor,
    },
};
use eredu_runtime::working_memory::{
    CaptureRunHostError, CaptureTensorClaim, CaptureTensorConstructionError, CaptureTensorHostPlan,
    CaptureTensorTransferFinishError, ClaimedCaptureTensor, PreparedCaptureTensorTransfer,
    ScheduledCaptureTensorTransfer, ScheduledCaptureTensorTransferFinishError, WorkingMemoryError,
    WorkingMemoryFundingRun, WorkingMemoryFundingScope, WorkingMemoryPool,
    WorkingMemoryReservation, WorkingMemoryStorage,
};
pub(crate) use fragment::PreparedCaptureFragment;
pub(crate) use original::{
    CaptureCompletion, control_bytes as capture_original_control_bytes,
    error_control_bytes as capture_original_error_control_bytes,
    retain_error as capture_original_error,
};
use safemlx::{Array, ArrayMetadataError, ArrayMetadataSnapshot, Dtype, Stream, error::Exception};
pub(crate) use speculative::{
    candidate_control_bytes as speculative_candidate_control_bytes,
    control_bytes as speculative_capture_control_bytes, execute as execute_speculative_capture, execute_cpu as execute_speculative_capture_cpu,
    execute_candidates as execute_speculative_candidates,
    execute_candidates_cpu as execute_speculative_candidates_cpu,
    execute_histogram as execute_speculative_histogram,
    execute_histogram_cpu as execute_speculative_histogram_cpu,
    execute_summary as execute_speculative_summary,
    execute_summary_cpu as execute_speculative_summary_cpu,
    execute_token_scores as execute_speculative_token_scores,
    execute_token_scores_cpu as execute_speculative_token_scores_cpu,
    histogram_control_bytes as speculative_histogram_control_bytes,
    summary_control_bytes as speculative_summary_control_bytes,
    token_score_control_bytes as speculative_token_score_control_bytes,
};
use std::{cell::RefCell, collections::TryReserveError, fmt};

#[derive(Debug, thiserror::Error)]
pub(crate) enum CaptureTensorNativeError {
    #[error("direct capture host reads require the audited Apple CPU or Metal allocator")]
    UnsupportedMechanism,
    #[error("capture stream is unavailable in this build: {0:?}")]
    UnsupportedStream(safemlx::DeviceType),
    #[error("selected capture requires F32, F16 or BF16 source, got {0:?}")]
    UnsupportedDtype(Dtype),
    #[error("capture source does not match admitted source geometry")]
    ShapeMismatch,
    #[error("candidate capture requires finite raw logits")]
    NonFiniteScores,
    #[error("capture claim does not match the prepared admission and coordinate")]
    ClaimMismatch,
    #[error(transparent)]
    Claim(#[from] CaptureRunHostError),
    #[error("capture workspace source must use floating metadata, got {0:?}")]
    UnsupportedWorkspaceDtype(WorkspaceDtype),
    #[error("capture source backing is not settled and certified")]
    UnsettledSource,
    #[error("full-vocabulary scoring requires finite raw logits")]
    NonfiniteScores,
    #[error("capture source changed after cold preparation")]
    SourceChanged,
    #[error("capture selection cannot be represented by static positive-stride slicing")]
    GeometryOverflow,
    #[error("capture source metadata: {0}")]
    Metadata(#[from] ArrayMetadataError),
    #[error("capture native work: {0}")]
    Native(#[from] Exception),
    #[error("capture host read: {0}")]
    HostRead(#[from] safemlx::error::AsSliceError),
    #[error("capture workspace: {0}")]
    Workspace(#[from] eredu_nn::Error),
    #[error(transparent)]
    Memory(#[from] WorkingMemoryError),
    #[error(transparent)]
    Host(#[from] CaptureTensorConstructionError),
    #[error("capture prepared clone: {0}")]
    Clone(#[from] safemlx::PreparedArrayCloneCause),
    #[error(transparent)]
    SpeculativeFinish(#[from] eredu_runtime::working_memory::ScheduledCaptureTensorFailure),
    #[error(transparent)]
    CandidateFinish(#[from] eredu_runtime::working_memory::CaptureCandidateFailure),
    #[error(transparent)]
    TokenScoreFinish(#[from] eredu_runtime::working_memory::CaptureTokenScoreFailure),
    #[error(transparent)]
    SummaryFinish(#[from] eredu_runtime::working_memory::CaptureSummaryFailure),
    #[error(transparent)]
    HistogramFinish(#[from] eredu_runtime::working_memory::CaptureHistogramFailure),
    #[error("activation policy: {0}")]
    ActivationPolicy(#[source] eredu_core::capture::CaptureError),
    #[error("capture recovery collector is borrowed")]
    CollectorBusy,
    #[error("capture recovery collector capacity: {0}")]
    Allocation(#[from] TryReserveError),
}

/// One actual borrowed source and immutable admitted selection. This is a leaf
/// mechanism, not capture admission or a whole-request quote. The enclosing
/// caller must authenticate invocation/quota and include its full trace plus H
/// in the original request. Source aliases remain settled/exclusive throughout.
/// F32, F16 and BF16 use CPU or shared Metal storage. Half conversion is the actual
/// native AsType, preserving its exceptional-value semantics. Empty half outputs
/// skip conversion/iteration. F64 and other transfer mechanisms are unfinished.
/// This leaf does not lower any managed capture gate.
pub(crate) struct PreparedCaptureTensor<'a> {
    source: &'a Array,
    observed: ArrayMetadataSnapshot,
    host: CaptureTensorHostPlan<'a>,
    program: Selection,
}
/// Fixed selection derived from one admitted geometry, with no source allocation
/// or admission authority. Future traces conservatively include a nonempty F32
/// conversion because floating workspace metadata erases actual precision.
pub(crate) struct Selection {
    source_shape: [i32; 32],
    starts: [i32; 32],
    ends: [i32; 32],
    strides: [i32; 32],
    selected_shape: [i32; 32],
    rank: usize,
    preview: Option<(i32, i32)>,
    cast_f32: bool,
    read_f32: bool,
}
impl<'a> PreparedCaptureTensor<'a> {
    pub(crate) fn new(
        source: &'a Array,
        host: CaptureTensorHostPlan<'a>,
    ) -> Result<Self, CaptureTensorNativeError> {
        let geometry = host.geometry();
        let observed = Self::validate_source_geometry(source, geometry)?;
        if observed.allocation().is_none() {
            return Err(CaptureTensorNativeError::UnsettledSource);
        }
        let program = Selection::with_conversion(
            geometry,
            if observed.dtype() == Dtype::Float32 {
                ConversionMode::ActualF32
            } else {
                ConversionMode::ActualHalf
            },
        )?;
        Ok(Self {
            source,
            observed,
            host,
            program,
        })
    }
    /// Validate the audited mechanism, actual dtype and admitted source axes
    /// without evaluating or requiring a physical allocation. The caller must
    /// retain and settle a lazy source under its existing native authority, then
    /// use `new` to bind the complete backing before transfer. This observation
    /// supplies neither source provenance nor completion/admission authority.
    pub(crate) fn validate_source_geometry(
        source: &Array,
        geometry: &CaptureTensorGeometry<'_>,
    ) -> Result<ArrayMetadataSnapshot, CaptureTensorNativeError> {
        Self::validate_mechanism()?;
        if source.shape().len() > 32 {
            return Err(CaptureTensorNativeError::ShapeMismatch);
        }
        let observed = source.try_metadata_snapshot()?;
        Self::validate_source_axes(observed.shape(), observed.dtype(), geometry.source_shape())?;
        Ok(observed)
    }

    /// Allocation-free descriptor precheck for a scheduled observer before its
    /// logical quota is charged. Borrows the actual axes without cloning a native
    /// handle, entering housekeeping, evaluating, or requiring a known backing.
    /// The funded worker still performs its snapshot/source/stream revalidation.
    pub(crate) fn validate_borrowed_source(
        source: &Array,
        geometry: &CaptureTensorGeometry<'_>,
    ) -> Result<Dtype, CaptureTensorNativeError> {
        Self::validate_borrowed_shape(source,geometry.source_shape())
    }
    /// Same borrowed descriptor worker for a real local source with no selected
    /// destination. The owning runtime plan supplies the exact physical axes.
    pub(crate) fn validate_borrowed_shape(source:&Array,shape:&[usize])->Result<Dtype,CaptureTensorNativeError> {
        Self::validate_mechanism()?;
        let dtype=source.dtype();Self::validate_source_axes(source.shape(),dtype,shape)?;Ok(dtype)
    }

    fn validate_mechanism() -> Result<(), CaptureTensorNativeError> {
        if !cfg!(all(
            target_vendor = "apple",
            not(feature = "cuda")
        )) {
            return Err(CaptureTensorNativeError::UnsupportedMechanism);
        }
        Ok(())
    }

    fn validate_source_axes(
        shape: &[i32],
        dtype: Dtype,
        source_shape: &[usize],
    ) -> Result<(), CaptureTensorNativeError> {
        if !matches!(dtype, Dtype::Float32 | Dtype::Float16 | Dtype::Bfloat16) {
            return Err(CaptureTensorNativeError::UnsupportedDtype(dtype));
        }
        if shape.len() != source_shape.len()
            || shape
                .iter()
                .zip(source_shape)
                .any(|(&a, &b)| usize::try_from(a).ok() != Some(b))
        {
            return Err(CaptureTensorNativeError::ShapeMismatch);
        }
        Ok(())
    }

    /// Device descriptor validation only; this does not evaluate arrays or
    /// authorize a stream from its compile-time availability alone.
    pub(crate) fn validate_stream(stream: &Stream) -> Result<(), CaptureTensorNativeError> {
        let device_type = stream.device_type()?;
        Self::validate_mechanism()?;
        if device_type == safemlx::DeviceType::Gpu && !cfg!(feature = "metal") {
            return Err(CaptureTensorNativeError::UnsupportedStream(device_type));
        }
        Ok(())
    }

    pub(crate) fn host_peak_bytes(&self) -> u64 {
        self.host.initialization_peak_bytes()
    }
    /// Source plus every selected intermediate remains in recovery until caller
    /// explicitly settles it. Descriptor bookkeeping is not a numerical grant.
    pub(crate) fn recovery_descriptors(&self) -> usize {
        2 + if self.program.preview.is_some() { 2 } else { 0 } + usize::from(self.program.cast_f32)
    }
    fn validate(&self) -> Result<(), CaptureTensorNativeError> {
        if self.source.try_metadata_snapshot()? != self.observed {
            return Err(CaptureTensorNativeError::SourceChanged);
        }
        Ok(())
    }
    /// Metadata only; use the enclosing shared context/span so the complete
    /// source backing is retained, including a narrow view's larger root.
    pub(crate) fn trace(
        &self,
        projection: &mut ExistingArrayProjection<'a>,
    ) -> Result<WorkspaceTensor, CaptureTensorNativeError> {
        self.validate()?;
        let source = projection.project(self.source)?;
        self.program.trace_within(&source, projection.context())
    }
    /// Bind parent H and complete actual source pins to this same active native
    /// scope before operations. Caller keeps `roots`, source ownership and
    /// scope through every error/unwind, then explicitly settles/certifies.
    /// Successful host finish alone never certifies native completion. No
    /// native array survives solely in the returned host observation.
    pub(crate) fn transfer<'s>(
        self,
        run: &WorkingMemoryFundingRun,
        reservation: &WorkingMemoryReservation,
        native: &'s mut WorkingMemoryFundingScope,
        stream: &Stream,
        roots: &RefCell<Vec<Array>>,
    ) -> Result<SharedTensorObservation, CaptureTensorExecutionError<'a, 's>> {
        let source_pin = self.prepare_transfer(run.pool(), stream, roots)?;
        let mut destination = run
            .prepare_capture_tensor_with_source(reservation, native, self.host, source_pin)
            .map_err(CaptureTensorNativeError::Host)?;
        execute_selected(
            self.source,
            &self.program,
            &mut destination,
            stream,
            roots,
            CaptureCompletion::Ordinary,
        )?;
        destination
            .finish()
            .map_err(CaptureTensorExecutionError::Finish)
    }

    /// Consume one already funded scheduled claim without another hold. Exact
    /// admission identity and coordinate checks precede all native work. A
    /// same-shaped independent admission never substitutes for this source.
    /// Caller retains actual roots and this scope through settlement/recovery;
    /// host completion does not certify it or spend logical capture quota.
    pub(crate) fn transfer_scheduled<'g, 'c, 's>(
        self,
        claim: CaptureTensorClaim<'g, 'c>,
        native: &'s mut WorkingMemoryFundingScope,
        stream: &Stream,
        roots: &RefCell<Vec<Array>>,
    ) -> Result<ClaimedCaptureTensor, ScheduledCaptureTensorExecutionError<'g, 'c, 's>> {
        self.transfer_scheduled_with_completion(
            claim,
            native,
            stream,
            roots,
            CaptureCompletion::Ordinary,
        )
    }

    pub(crate) fn transfer_scheduled_with_completion<'g, 'c, 's>(
        self,
        claim: CaptureTensorClaim<'g, 'c>,
        native: &'s mut WorkingMemoryFundingScope,
        stream: &Stream,
        roots: &RefCell<Vec<Array>>,
        completion: CaptureCompletion<'_>,
    ) -> Result<ClaimedCaptureTensor, ScheduledCaptureTensorExecutionError<'g, 'c, 's>> {
        self.validate_claim(claim.geometry())?;
        Self::validate_stream(stream)?;
        self.validate()?;
        let source_pin = Self::prepare_registered_source_with_completion(
            &self.observed,
            native.pool(),
            roots,
            self.recovery_descriptors(),
            completion,
        )?;
        let mut destination = claim
            .prepare_with_source(native, source_pin)
            .map_err(CaptureTensorNativeError::Claim)?;
        execute_selected(
            self.source,
            &self.program,
            &mut destination,
            stream,
            roots,
            completion,
        )?;
        destination
            .finish()
            .map_err(ScheduledCaptureTensorExecutionError::Finish)
    }

    fn validate_claim(
        &self,
        geometry: &CaptureTensorGeometry<'_>,
    ) -> Result<(), CaptureTensorNativeError> {
        let prepared = self.host.geometry();
        if !std::ptr::eq(prepared.admission(), geometry.admission())
            || prepared.selection_index() != geometry.selection_index()
            || prepared.phase() != geometry.phase()
            || prepared.prediction() != geometry.prediction()
            || prepared.source_shape() != geometry.source_shape()
            || prepared.shape() != geometry.shape()
            || prepared.elements() != geometry.elements()
        {
            return Err(CaptureTensorNativeError::ClaimMismatch);
        }
        Ok(())
    }

    fn prepare_transfer(
        &self,
        pool: &WorkingMemoryPool,
        stream: &Stream,
        roots: &RefCell<Vec<Array>>,
    ) -> Result<WorkingMemoryStorage<StorageIdentity>, CaptureTensorNativeError> {
        Self::validate_stream(stream)?;
        self.validate()?;
        Self::prepare_registered_source(&self.observed, pool, roots, self.recovery_descriptors())
    }

    fn prepare_registered_source(
        observed: &ArrayMetadataSnapshot,
        pool: &WorkingMemoryPool,
        roots: &RefCell<Vec<Array>>,
        descriptors: usize,
    ) -> Result<WorkingMemoryStorage<StorageIdentity>, CaptureTensorNativeError> {
        Self::prepare_registered_source_with_completion(
            observed,
            pool,
            roots,
            descriptors,
            CaptureCompletion::Ordinary,
        )
    }

    fn prepare_registered_source_with_completion(
        observed: &ArrayMetadataSnapshot,
        pool: &WorkingMemoryPool,
        roots: &RefCell<Vec<Array>>,
        descriptors: usize,
        completion: CaptureCompletion<'_>,
    ) -> Result<WorkingMemoryStorage<StorageIdentity>, CaptureTensorNativeError> {
        let allocation = observed
            .allocation()
            .ok_or(CaptureTensorNativeError::UnsettledSource)?;
        let bytes = u64::try_from(allocation.bytes())
            .map_err(|_| CaptureTensorNativeError::GeometryOverflow)?;
        let source_pin = pool
            .pin_registered_storage(
                [(StorageIdentity::Native(allocation.identity()), bytes)]
                    .into_iter()
                    .filter(|(_, bytes)| *bytes != 0),
            )
            .map_err(CaptureTensorNativeError::Memory)?;
        completion.reserve_roots(roots, descriptors)?;
        Ok(source_pin)
    }
}

// All closed host destinations use the identical selected native program,
// settlement and direct F32 iteration. This private surface provides no allocator,
// source substitution, completion certificate or mutable buffer escape.
trait TransferDestination {
    type Error: Into<CaptureTensorNativeError>;
    fn validate(&self) -> Result<(), Self::Error>;
    fn push_f32(&mut self, value: f32) -> Result<(), Self::Error>;
}
impl TransferDestination for PreparedCaptureTensorTransfer<'_, '_, StorageIdentity> {
    type Error = WorkingMemoryError;
    fn validate(&self) -> Result<(), Self::Error> {
        PreparedCaptureTensorTransfer::validate(self)
    }
    fn push_f32(&mut self, value: f32) -> Result<(), Self::Error> {
        PreparedCaptureTensorTransfer::push_f32(self, value)
    }
}
impl TransferDestination for ScheduledCaptureTensorTransfer<'_, '_, '_, StorageIdentity> {
    type Error = WorkingMemoryError;
    fn validate(&self) -> Result<(), Self::Error> {
        ScheduledCaptureTensorTransfer::validate(self)
    }
    fn push_f32(&mut self, value: f32) -> Result<(), Self::Error> {
        ScheduledCaptureTensorTransfer::push_f32(self, value)
    }
}
fn execute_selected(
    source: &Array,
    program: &Selection,
    destination: &mut impl TransferDestination,
    stream: &Stream,
    roots: &RefCell<Vec<Array>>,
    completion: CaptureCompletion<'_>,
) -> Result<(), CaptureTensorNativeError> {
    destination
        .validate()
        .map_err(Into::<CaptureTensorNativeError>::into)?;
    let first_retained = roots.borrow().len();
    let source = completion.clone_array(source)?;
    let retained_source = completion.clone_array(&source)?;
    roots.borrow_mut().push(retained_source);
    let output = selected(
        program,
        &mut Native {
            stream,
            roots,
            completion,
        },
        source,
    )?;
    // Both choices settle the exact output. Original uses the existing finite
    // resident bank; direct readout cannot create another evaluation afterward.
    let evaluated = completion.settle(&output, stream)?;
    if let CaptureCompletion::Original(observer) = completion {
        // The asynchronous nested event covers the complete selected graph,
        // but settlement publishes only the final descriptor's availability.
        // Recovery also retains its intermediate Slice/reshape/AsType outputs.
        // Publish those same completed matching descriptors before later pure
        // inventory queries. This is no new Eval, wait, root or native grant.
        let retained = roots.try_borrow()
            .map_err(|_| CaptureTensorNativeError::CollectorBusy)?;
        for value in &retained[first_retained..] {
            observer.validate_completed_array(value)?;
        }
    }
    destination
        .validate()
        .map_err(Into::<CaptureTensorNativeError>::into)?;
    #[cfg(test)]
    tests::after_evaluation().map_err(CaptureTensorNativeError::Native)?;
    if program.read_f32 {
        for value in evaluated
            .try_iter::<f32>()
            .map_err(CaptureTensorNativeError::HostRead)?
        {
            destination
                .push_f32(value)
                .map_err(Into::<CaptureTensorNativeError>::into)?;
        }
    }
    #[cfg(test)]
    tests::before_finish();
    Ok(())
}

// Actual modes are selected only by the native constructor after it validates
// the observed dtype. A caller tracing future metadata cannot assert F32.
#[derive(Clone, Copy)]
enum ConversionMode {
    ActualF32,
    ActualHalf,
    MayRequireF32,
}
impl Selection {
    /// Prepare a future floating source. Its concrete precision is unknown, so
    /// every nonempty output includes the fixed F32 conversion bound. No native
    /// source, registration, host payload or execution grant is created.
    pub(crate) fn from_geometry(
        geometry: &CaptureTensorGeometry<'_>,
    ) -> Result<Self, CaptureTensorNativeError> {
        Self::with_conversion(geometry, ConversionMode::MayRequireF32)
    }

    /// Prepare the same future selection program from a closed physical chunk
    /// fragment. The source rectangle and global Preview contribution are derived
    /// by core; this constructor repeats no intersection or scatter arithmetic.
    /// It grants no source/claim/funding/readout or chunk-completion authority.
    /// Actual precision is unknown here, so nonempty output conservatively casts.
    pub(crate) fn from_fragment(
        fragment: &CapturePrefillFragment<'_, '_>,
    ) -> Result<Self, CaptureTensorNativeError> {
        let rank = fragment.source_shape().len();
        if rank > 32 {
            return Err(CaptureTensorNativeError::ShapeMismatch);
        }
        let mut program = Self {
            source_shape: [0; 32],
            starts: [0; 32],
            ends: [0; 32],
            strides: [1; 32],
            selected_shape: [0; 32],
            rank,
            preview: None,
            cast_f32: fragment.output_elements() != 0,
            read_f32: fragment.output_elements() != 0,
        };
        let convert = |n| i32::try_from(n).map_err(|_| CaptureTensorNativeError::GeometryOverflow);
        for axis in 0..rank {
            let slice = fragment
                .selection_axis(axis)
                .ok_or(CaptureTensorNativeError::GeometryOverflow)?;
            program.source_shape[axis] = convert(fragment.source_shape()[axis])?;
            program.starts[axis] = convert(slice.start())?;
            program.ends[axis] = convert(slice.end())?;
            program.strides[axis] = convert(slice.stride())?;
            program.selected_shape[axis] = convert(slice.elements())?;
            // Preserve MLX's signed normalize_slice limit, even for a narrow
            // output whose source or stride cannot be represented by that API.
            (program.ends[axis] - program.starts[axis])
                .checked_add(program.strides[axis] - 1)
                .ok_or(CaptureTensorNativeError::GeometryOverflow)?;
        }
        let geometry = fragment.assembly().logical_geometry();
        let transform = geometry.native_transform();
        if matches!(transform, CaptureTransform::Preview { .. }) {
            program.preview = Some((
                convert(fragment.selected_elements())?,
                convert(fragment.output_elements())?,
            ));
        } else if fragment.output_elements() != fragment.selected_elements() {
            return Err(CaptureTensorNativeError::ShapeMismatch);
        }
        Ok(program)
    }

    fn with_conversion(
        geometry: &CaptureTensorGeometry<'_>,
        conversion: ConversionMode,
    ) -> Result<Self, CaptureTensorNativeError> {
        let transform = geometry.native_transform();
        let rank = geometry.source_shape().len();
        if rank > 32 {
            return Err(CaptureTensorNativeError::ShapeMismatch);
        }
        let mut program = Self {
            source_shape: [0; 32],
            starts: [0; 32],
            ends: [0; 32],
            strides: [1; 32],
            selected_shape: [0; 32],
            rank,
            preview: None,
            cast_f32: !matches!(conversion, ConversionMode::ActualF32) && geometry.elements() != 0,
            // Identity Slice can retain an anomalous empty input descriptor.
            // observe_tensor already returns empty before native conversion;
            // do the same, including Preview whose final prefix is empty.
            read_f32: matches!(conversion, ConversionMode::ActualF32) || geometry.elements() != 0,
        };
        let convert = |n| i32::try_from(n).map_err(|_| CaptureTensorNativeError::GeometryOverflow);
        let mut elements = 1usize;
        for i in 0..rank {
            program.source_shape[i] = i32::try_from(geometry.source_shape()[i])
                .map_err(|_| CaptureTensorNativeError::GeometryOverflow)?;
            let (start, end, stride) = (
                geometry.starts()[i],
                geometry.ends()[i],
                geometry.strides()[i],
            );
            program.starts[i] = convert(start)?;
            program.ends[i] = convert(end)?;
            program.strides[i] = convert(stride)?;
            // MLX normalize_slice uses signed ShapeElem arithmetic here.
            (program.ends[i] - program.starts[i])
                .checked_add(program.strides[i] - 1)
                .ok_or(CaptureTensorNativeError::GeometryOverflow)?;
            let count = (end - start).div_ceil(stride);
            program.selected_shape[i] = convert(count)?;
            elements = elements
                .checked_mul(count as usize)
                .ok_or(CaptureTensorNativeError::GeometryOverflow)?;
        }
        if matches!(transform, CaptureTransform::Preview { .. }) {
            program.preview = Some((
                i32::try_from(elements).map_err(|_| CaptureTensorNativeError::GeometryOverflow)?,
                i32::try_from(geometry.elements())
                    .map_err(|_| CaptureTensorNativeError::GeometryOverflow)?,
            ));
        } else if program.selected_shape[..program.rank]
            .iter()
            .map(|&n| n as usize)
            .ne(geometry.shape().iter().copied())
        {
            return Err(CaptureTensorNativeError::ShapeMismatch);
        }
        Ok(program)
    }

    /// Trace in the caller's existing context/span. The source identity and full
    /// backing aliases are preserved; dtype erasure never elides the future
    /// conversion. All validation precedes the first operation. The enclosing
    /// quote still owns source inventory and the separate host destination H.
    pub(crate) fn trace_within(
        &self,
        source: &WorkspaceTensor,
        context: &WorkspaceContext,
    ) -> Result<WorkspaceTensor, CaptureTensorNativeError> {
        self.validate_workspace_source(source, context)?;
        selected(
            self,
            &mut Trace {
                context,
                retained: None,
            },
            source.clone(),
        )
    }
    /// Same program, retaining each output through the caller's original span
    /// just as the native recovery collector does. The source is caller-owned.
    pub(crate) fn trace_retained_within(
        &self,
        source: &WorkspaceTensor,
        context: &WorkspaceContext,
        retained: &mut Vec<WorkspaceTensor>,
    ) -> Result<WorkspaceTensor, CaptureTensorNativeError> {
        self.validate_workspace_source(source, context)?;
        selected(
            self,
            &mut Trace {
                context,
                retained: Some(retained),
            },
            source.clone(),
        )
    }
    /// Same metadata checks as tracing, before a logical quota is consumed.
    /// No operation, source clone or allocation fact is created here.
    pub(crate) fn validate_workspace_source(
        &self,
        source: &WorkspaceTensor,
        context: &WorkspaceContext,
    ) -> Result<(), CaptureTensorNativeError> {
        context.validate_values([source])?;
        if source.layout().dtype() != WorkspaceDtype::Float32 {
            return Err(CaptureTensorNativeError::UnsupportedWorkspaceDtype(
                source.layout().dtype(),
            ));
        }
        if source.shape() != &self.source_shape[..self.rank] {
            return Err(CaptureTensorNativeError::ShapeMismatch);
        }
        Ok(())
    }
}

trait Mechanism {
    type Value;
    fn slice(
        &mut self,
        source: Self::Value,
        starts: &[i32],
        ends: &[i32],
        strides: &[i32],
        shape: &[i32],
    ) -> Result<Self::Value, CaptureTensorNativeError>;
    fn reshape(
        &mut self,
        source: Self::Value,
        elements: i32,
    ) -> Result<Self::Value, CaptureTensorNativeError>;
    fn cast_f32(&mut self, source: Self::Value) -> Result<Self::Value, CaptureTensorNativeError>;
}
fn selected<M: Mechanism>(
    program: &Selection,
    mechanism: &mut M,
    source: M::Value,
) -> Result<M::Value, CaptureTensorNativeError> {
    let n = program.rank;
    let value = mechanism.slice(
        source,
        &program.starts[..n],
        &program.ends[..n],
        &program.strides[..n],
        &program.selected_shape[..n],
    )?;
    let value = if let Some((elements, count)) = program.preview {
        let flat = mechanism.reshape(value, elements)?;
        mechanism.slice(flat, &[0], &[count], &[1], &[count])?
    } else {
        value
    };
    if program.cast_f32 {
        mechanism.cast_f32(value)
    } else {
        Ok(value)
    }
}
struct Trace<'a> {
    context: &'a WorkspaceContext,
    retained: Option<&'a mut Vec<WorkspaceTensor>>,
}
impl Trace<'_> {
    fn retain(
        &mut self,
        value: WorkspaceTensor,
    ) -> Result<WorkspaceTensor, CaptureTensorNativeError> {
        if let Some(retained) = self.retained.as_mut() {
            self.context.reserve_metadata_vec(retained, 1)?;
            retained.push(value.clone());
        }
        Ok(value)
    }
}
impl Mechanism for Trace<'_> {
    type Value = WorkspaceTensor;
    fn slice(
        &mut self,
        source: WorkspaceTensor,
        starts: &[i32],
        ends: &[i32],
        strides: &[i32],
        shape: &[i32],
    ) -> Result<WorkspaceTensor, CaptureTensorNativeError> {
        let value=source.static_slice(starts,ends,strides,self.context)?;
        if value.shape()!=shape {return Err(CaptureTensorNativeError::ShapeMismatch);}
        self.retain(value)
    }
    fn reshape(
        &mut self,
        source: WorkspaceTensor,
        elements: i32,
    ) -> Result<WorkspaceTensor, CaptureTensorNativeError> {
        let value = source.reshape(&[elements], self.context)?;
        self.retain(value)
    }
    fn cast_f32(
        &mut self,
        source: WorkspaceTensor,
    ) -> Result<WorkspaceTensor, CaptureTensorNativeError> {
        // Floating metadata erases actual precision. An actual half source
        // allocates here; a future source prices that possibility even if its
        // eventual F32 dtype permits a no-op. Never use dtype equality here.
        let output = self
            .context
            .layout(source.shape(), WorkspaceDtype::Float32)?;
        let mut outputs = self.context.metadata_vec(1)?;
        outputs.push(output);
        let value = self
            .context
            .execute(
                WorkspaceOperationKind::Elementwise("capture_cast_f32"),
                &[&source],
                outputs,
            )?
            .remove(0);
        self.retain(value)
    }
}
struct Native<'a> {
    stream: &'a Stream,
    roots: &'a RefCell<Vec<Array>>,
    completion: CaptureCompletion<'a>,
}
impl Native<'_> {
    fn retain(&self, value: &Array) -> Result<(), CaptureTensorNativeError> {
        let retained = match self.completion {
            CaptureCompletion::Ordinary => value.clone(),
            original => original.clone_array(value)?,
        };
        self.roots.borrow_mut().push(retained);
        Ok(())
    }
}
impl Mechanism for Native<'_> {
    type Value = Array;
    fn slice(
        &mut self,
        source: Array,
        starts: &[i32],
        ends: &[i32],
        strides: &[i32],
        _: &[i32],
    ) -> Result<Array, CaptureTensorNativeError> {
        let value = source.try_slice(starts, ends, strides, self.stream)?;
        self.retain(&value)?;
        #[cfg(test)]
        tests::after_slice()?;
        Ok(value)
    }
    fn reshape(&mut self, source: Array, elements: i32) -> Result<Array, CaptureTensorNativeError> {
        let value = source.reshape(&[elements], self.stream)?;
        self.retain(&value)?;
        Ok(value)
    }
    fn cast_f32(&mut self, source: Array) -> Result<Array, CaptureTensorNativeError> {
        let value = source.as_type::<f32>(self.stream)?;
        self.retain(&value)?;
        #[cfg(test)]
        tests::after_cast()?;
        Ok(value)
    }
}
/// A failed host finish owns its protected payload; all native failures retain
/// their original typed cause. Native recovery stays with the caller on all paths.
#[derive(Debug)]
pub(crate) enum CaptureTensorExecutionError<'a, 's> {
    Mechanism(CaptureTensorNativeError),
    Finish(CaptureTensorTransferFinishError<'a, 's, StorageIdentity>),
}
impl From<CaptureTensorNativeError> for CaptureTensorExecutionError<'_, '_> {
    fn from(error: CaptureTensorNativeError) -> Self {
        Self::Mechanism(error)
    }
}
impl fmt::Display for CaptureTensorExecutionError<'_, '_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Mechanism(error) => fmt::Display::fmt(error, f),
            Self::Finish(error) => fmt::Display::fmt(error, f),
        }
    }
}
impl std::error::Error for CaptureTensorExecutionError<'_, '_> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Mechanism(error) => Some(error),
            Self::Finish(error) => std::error::Error::source(error),
        }
    }
}
/// Scheduled finish failures retain the original protected owner and exact
/// native scope borrow. Composition must retire it before boxing an owned cause;
/// all mechanism errors preserve their original typed source.
#[derive(Debug)]
pub(crate) enum ScheduledCaptureTensorExecutionError<'a, 'c, 's> {
    Mechanism(CaptureTensorNativeError),
    Finish(ScheduledCaptureTensorTransferFinishError<'a, 'c, 's, StorageIdentity>),
}
impl From<CaptureTensorNativeError> for ScheduledCaptureTensorExecutionError<'_, '_, '_> {
    fn from(error: CaptureTensorNativeError) -> Self {
        Self::Mechanism(error)
    }
}
impl fmt::Display for ScheduledCaptureTensorExecutionError<'_, '_, '_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Mechanism(error) => fmt::Display::fmt(error, f),
            Self::Finish(error) => fmt::Display::fmt(error, f),
        }
    }
}
impl std::error::Error for ScheduledCaptureTensorExecutionError<'_, '_, '_> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Mechanism(error) => Some(error),
            Self::Finish(error) => std::error::Error::source(error),
        }
    }
}
#[cfg(test)]
mod tests;

#[cfg(test)]
mod workspace_tests;
