//! Declared logical request rows mapped to one fixed-schedule physical chunk.
use super::*;
use crate::{InferenceGeometry, OutputDemand};
use std::ops::Range;

/// Rejection by the closed geometry-only row mapping.
#[derive(Debug, thiserror::Error)]
pub enum CapturePrefillGeometryError {
    /// Independent invocations cannot substitute for ordinary request rows.
    #[error("prefill row mapping requires ordinary admitted text geometry")]
    Invocation,
    /// Original batch, prompt, prediction allowance or cached origin differs.
    #[error("prefill row mapping differs from admitted request geometry")]
    RequestMismatch,
    /// A fixed schedule requires a positive chunk size no larger than the prompt.
    #[error("invalid prefill row chunk schedule")]
    Schedule,
    /// Exactly one declared Sequence axis is required.
    #[error("prefill row mapping requires one Sequence axis, found {actual}")]
    SequenceAxes {
        /// Number of declared Sequence axes.
        actual: usize,
    },
    /// Every other dimension must be fixed Known or request Batch.
    #[error("prefill row mapping cannot assemble axis {axis}")]
    UnsupportedAxis {
        /// Declared axis index, never a guessed semantic path.
        axis: usize,
    },
    /// A requested diagnostic chunk is outside the bound fixed schedule.
    #[error("prefill chunk {index} is outside {chunks} chunks")]
    Chunk {
        /// Requested chunk index.
        index: u64,
        /// Bound number of chunks.
        chunks: u64,
    },
    /// Checked frontier, source indexing or selected indexing cannot be represented.
    #[error("prefill row mapping arithmetic overflow")]
    Overflow,
    /// The actual full logical p0 selection rejected its shape or transform.
    #[error(transparent)]
    Tensor(#[from] CaptureTensorGeometryError),
}

type Result<T> = std::result::Result<T, CapturePrefillGeometryError>;
fn host(n: u64) -> Result<usize> {
    usize::try_from(n).map_err(|_| CapturePrefillGeometryError::Overflow)
}
fn product(shape: &[usize]) -> Result<usize> {
    // Match ordinary capture's rule: a zero axis does not conceal overflow
    // among the remaining declared extents.
    let nonzero = shape
        .iter()
        .filter(|&&d| d != 0)
        .try_fold(1usize, |n, &d| {
            n.checked_mul(d)
                .ok_or(CapturePrefillGeometryError::Overflow)
        })?;
    Ok(if shape.contains(&0) { 0 } else { nonzero })
}

fn strides(shape: &[usize]) -> Result<[usize; 32]> {
    product(shape)?;
    let mut result = [0; 32];
    if shape.contains(&0) {
        return Ok(result);
    }
    let mut n = 1usize;
    for i in (0..shape.len()).rev() {
        result[i] = n;
        n = n
            .checked_mul(shape[i])
            .ok_or(CapturePrefillGeometryError::Overflow)?;
    }
    Ok(result)
}

/// Borrowed placement declaration derived only from an actual admitted selection.
///
/// The immutable catalog's single `Sequence` axis declares operation rows;
/// Known/Batch axes remain fixed. This type proves only their logical-to-physical
/// placement. It does not prove causal/full-forward numerical equivalence,
/// executable hook support, actual source values/backing, readout demand, quota,
/// funding, chunk execution or completion. Activation requires the enclosing
/// architecture/runtime to establish those separate contracts.
///
/// Construction and fragment iteration allocate no shape, index or data payload.
/// No caller supplies a shape, source offset, destination offset or scalar count.
/// The actual admission remains borrowed; identity-equivalent independent plans
/// are not substituted. Context, TokenRows, media and multiple Sequence axes
/// require separately specified assembly semantics and are rejected here.
#[derive(Debug)]
pub struct CapturePrefillRowAssembly<'a> {
    logical: CaptureTensorGeometry<'a>,
    inference: InferenceGeometry,
    axis: usize,
    selected_shape: [usize; 32],
    starts: [usize; 32],
    steps: [usize; 32],
    selected_strides: [usize; 32],
    selected_elements: usize,
}
impl<'a> CapturePrefillRowAssembly<'a> {
    /// Bind full logical p0 selection and one exact fixed chunk schedule.
    /// The output cap may be a prefix of the admitted prediction range: it does
    /// not change prompt axes, fragments or source attribution. This borrowed
    /// geometry grants no capture/run permission.
    /// OutputDemand is retained for comparison, never upgraded into readout
    /// permission. No discovery schema, admission identity or schedule changes.
    pub fn prepare(
        source: &'a AdmittedCapturePlan,
        selection: usize,
        inference: InferenceGeometry,
    ) -> Result<Self> {
        let origin = source
            .text_origin()
            .ok_or(CapturePrefillGeometryError::Invocation)?;
        let request = source.request();
        if request.batch != inference.batch_size
            || request.prompt_tokens != inference.input_positions
            || inference.max_output_tokens > request.max_predictions
            || origin.cached_positions != inference.cached_positions
        {
            return Err(CapturePrefillGeometryError::RequestMismatch);
        }
        if inference.prefill_chunk_positions == 0
            || inference.prefill_chunk_positions > inference.input_positions
        {
            return Err(CapturePrefillGeometryError::Schedule);
        }
        inference
            .cached_positions
            .checked_add(inference.input_positions)
            .and_then(|n| n.checked_add(inference.max_output_tokens))
            .ok_or(CapturePrefillGeometryError::Overflow)?;
        let point = source
            .points()
            .get(selection)
            .ok_or(CaptureTensorGeometryError::SelectionMissing { index: selection })?;
        let axes = point
            .axes
            .as_ref()
            .ok_or(CaptureTensorGeometryError::UnknownShape)?;
        if axes.len() > 32 {
            return Err(CaptureTensorGeometryError::RankExceeded {
                rank: axes.len(),
                maximum: 32,
            }
            .into());
        }
        let mut count = 0;
        let mut axis = 0;
        for (index, declared) in axes.iter().enumerate() {
            match &declared.dimension {
                SymbolicDimension::Sequence => {
                    count += 1;
                    axis = index;
                }
                SymbolicDimension::Known(_) | SymbolicDimension::Batch => {}
                _ => return Err(CapturePrefillGeometryError::UnsupportedAxis { axis: index }),
            }
        }
        if count != 1 {
            return Err(CapturePrefillGeometryError::SequenceAxes { actual: count });
        }
        let logical =
            CaptureTensorGeometry::prepare(source, selection, CapturePhase::Prefill, 0, None)?;
        // Full physical source indexing is checked independently of a narrow
        // slice/Preview; small selected output cannot conceal source overflow.
        product(logical.source_shape())?;
        let selected = &source.plan().selections[selection];
        let mut selected_shape = [0; 32];
        let mut starts = [0; 32];
        let mut steps = [1; 32];
        for (i, declared) in axes.iter().enumerate() {
            let extent = logical.source_shape()[i];
            if let Some(slice) = selected.slices.iter().find(|s| s.axis == declared.name) {
                starts[i] = host(slice.start)?;
                steps[i] = host(slice.stride)?;
                selected_shape[i] = host((slice.end - slice.start).div_ceil(slice.stride))?;
            } else {
                selected_shape[i] = extent;
            }
        }
        let selected_elements = product(&selected_shape[..axes.len()])?;
        let selected_strides = strides(&selected_shape[..axes.len()])?;
        Ok(Self {
            logical,
            inference,
            axis,
            selected_shape,
            starts,
            steps,
            selected_strides,
            selected_elements,
        })
    }
    /// Bind the same fixed temporal worker to one actual rank-local raw
    /// fragment. The projection retains the original global admission and
    /// spatial slice. The temporal axis must remain the complete request: a
    /// producer which owns only some prompt rows needs a distinct schedule.
    /// This supplies geometry only, never a final claim or native permission.
    pub fn prepare_partition(
        source: &'a AdmittedCapturePlan, selection: usize, inference: InferenceGeometry,
        projection: &CaptureSlicePartition, fragment: usize, combination: PartitionCaptureCombination,
    ) -> Result<Self> {
        let assembly = Self::prepare(source, selection, inference)?;
        let logical = CaptureTensorGeometry::prepare_partition(source, selection,
            CapturePhase::Prefill, 0, None, projection, fragment, combination)?;
        Self::from_partition_geometry(logical,inference,assembly.axis)
    }
    /// Raw source for an additive Summary/Histogram: every original selected
    /// F32 term is retained across chunks before the final nonlinear reduction.
    /// No shard-local nonlinear statistic is substituted for that raw tensor.
    pub fn prepare_additive_transform_partition(source: &'a AdmittedCapturePlan, selection: usize,
        inference: InferenceGeometry, projection: &CaptureSlicePartition, fragment: usize)
        -> std::result::Result<Self,CapturePrefillPartitionError>
    {
        let plan=CapturePrefillTransformPlan::prepare_partition(source,selection,inference,projection,fragment,
            PartitionCaptureCombination::SumF64ToF32)?;
        let logical=CaptureTensorGeometry::prepare_partition(source,selection,CapturePhase::Prefill,0,None,
            projection,fragment,PartitionCaptureCombination::SumF64ToF32)?;
        Self::from_partition_geometry(logical,inference,plan.window().axis()).map_err(Into::into)
    }
    fn from_partition_geometry(logical:CaptureTensorGeometry<'a>,inference:InferenceGeometry,axis:usize)->Result<Self> {
        if axis>=logical.source_shape().len() || logical.source_shape()[axis]!=host(inference.input_positions)? {
            return Err(CapturePrefillGeometryError::UnsupportedAxis{axis});
        }
        product(logical.source_shape())?;let rank=logical.source_shape().len();
        let mut starts=[0;32];let mut steps=[1;32];let mut selected_shape=[0;32];
        for axis in 0..rank {
            starts[axis]=host(logical.starts()[axis])?;steps[axis]=host(logical.strides()[axis])?;
            selected_shape[axis]=host((logical.ends()[axis]-logical.starts()[axis]).div_ceil(logical.strides()[axis]))?;
        }
        let selected_elements=product(&selected_shape[..rank])?;let selected_strides=strides(&selected_shape[..rank])?;
        Ok(Self{logical,inference,axis,selected_shape,starts,steps,selected_strides,selected_elements})
    }
    /// Fixed transports for the additive transform source and common raw worker.
    pub fn additive_partition_preparation_control_bytes()->Option<usize> {
        use std::mem::size_of;
        Self::partition_preparation_control_bytes()?
            .checked_add(CapturePrefillTransformPlan::partition_preparation_control_bytes()?)?
            .checked_add(size_of::<CapturePrefillTransformPlan<'_>>())?
            .checked_add(size_of::<std::result::Result<Self,CapturePrefillPartitionError>>())
    }
    /// Exact fixed constructor transports in addition to the original geometry
    /// source. A funded caller reserves these before this partition adapter.
    pub fn partition_preparation_control_bytes() -> Option<usize> {
        use std::mem::{size_of,size_of_val};
        let parts = [size_of::<Self>() * 3, size_of::<CaptureTensorGeometry<'_>>() * 2,
            size_of::<Result<Self>>(), size_of::<CapturePrefillGeometryError>(),
            size_of::<(&AdmittedCapturePlan,usize,InferenceGeometry,&CaptureSlicePartition,usize,PartitionCaptureCombination)>(),
            size_of::<(usize,usize)>(), size_of::<[usize;32]>()*4,
            size_of::<(CaptureTensorGeometry<'_>,InferenceGeometry,usize)>(),
            CaptureTensorGeometry::partition_preparation_control_bytes()?,
        ];
        parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)
    }

    /// Exact borrowed full-request output geometry, including global Preview.
    pub fn logical_geometry(&self) -> &CaptureTensorGeometry<'a> {
        &self.logical
    }
    /// Exact request geometry retained for future original-admission comparison.
    pub fn inference_geometry(&self) -> InferenceGeometry {
        self.inference
    }
    /// Declared row axis; all remaining axes are fixed for this request.
    pub fn sequence_axis(&self) -> usize {
        self.axis
    }
    /// Logical selected axes before Preview flattening.
    pub fn selected_shape(&self) -> &[usize] {
        &self.selected_shape[..self.logical.source_shape().len()]
    }
    /// Logical selected scalar count before Preview truncation. This is not a quota.
    pub fn selected_elements(&self) -> usize {
        self.selected_elements
    }
    /// Number of nonempty physical chunks in the bound schedule.
    pub fn chunk_count(&self) -> u64 {
        self.inference
            .input_positions
            .div_ceil(self.inference.prefill_chunk_positions)
    }
    /// Derive one physical chunk and scatter mapping from its schedule index.
    /// This may be inspected repeatedly; it issues no one-use execution claim.
    pub fn fragment(&self, index: u64) -> Result<CapturePrefillFragment<'_, 'a>> {
        let chunks = self.chunk_count();
        if index >= chunks {
            return Err(CapturePrefillGeometryError::Chunk { index, chunks });
        }
        let start = index
            .checked_mul(self.inference.prefill_chunk_positions)
            .ok_or(CapturePrefillGeometryError::Overflow)?;
        let length = self
            .inference
            .prefill_chunk_positions
            .min(self.inference.input_positions - start);
        let end = start
            .checked_add(length)
            .ok_or(CapturePrefillGeometryError::Overflow)?;
        let position = self
            .inference
            .cached_positions
            .checked_add(start)
            .ok_or(CapturePrefillGeometryError::Overflow)?;
        let a = host(start)?;
        let b = host(end)?;
        let row_start = self.starts[self.axis];
        let row_step = self.steps[self.axis];
        let rows = self.selected_shape[self.axis];
        let lo = a.saturating_sub(row_start).div_ceil(row_step).min(rows);
        let hi = b.saturating_sub(row_start).div_ceil(row_step).min(rows);
        let mut source_shape = [0; 32];
        source_shape[..self.logical.source_shape().len()]
            .copy_from_slice(self.logical.source_shape());
        source_shape[self.axis] = host(length)?;
        let rank = self.logical.source_shape().len();
        let source_strides = strides(&source_shape[..rank])?;
        let mut shape = self.selected_shape;
        shape[self.axis] = hi - lo;
        let selected_elements = product(&shape[..rank])?;
        let mut starts = self.starts;
        starts[self.axis] = if hi == lo {
            0
        } else {
            row_start
                .checked_add(
                    lo.checked_mul(row_step)
                        .ok_or(CapturePrefillGeometryError::Overflow)?,
                )
                .and_then(|n| n.checked_sub(a))
                .ok_or(CapturePrefillGeometryError::Overflow)?
        };
        // Each prefix group contributes one contiguous interval in the global
        // selected order. Preview admits only the prefix of that global order.
        let suffix = product(&self.selected_shape[self.axis + 1..rank])?;
        let block = rows
            .checked_mul(suffix)
            .ok_or(CapturePrefillGeometryError::Overflow)?;
        let width = (hi - lo)
            .checked_mul(suffix)
            .ok_or(CapturePrefillGeometryError::Overflow)?;
        let offset = lo
            .checked_mul(suffix)
            .ok_or(CapturePrefillGeometryError::Overflow)?;
        let output = self.logical.elements();
        let elements = if block == 0 || selected_elements == 0 {
            0
        } else {
            let whole = output / block;
            whole
                .checked_mul(width)
                .and_then(|n| n.checked_add((output % block).saturating_sub(offset).min(width)))
                .ok_or(CapturePrefillGeometryError::Overflow)?
        };
        Ok(CapturePrefillFragment {
            assembly: self,
            index,
            input: start..end,
            position,
            source_shape,
            source_strides,
            selected_shape: shape,
            starts,
            row_lo: lo,
            selected_elements,
            elements,
        })
    }
}

/// Derived physical source and logical destination correspondence for one chunk.
/// No offsets/counts can be supplied independently of its borrowed assembly.
#[derive(Debug)]
pub struct CapturePrefillFragment<'p, 'a> {
    assembly: &'p CapturePrefillRowAssembly<'a>,
    index: u64,
    input: Range<u64>,
    position: u64,
    source_shape: [usize; 32],
    source_strides: [usize; 32],
    selected_shape: [usize; 32],
    starts: [usize; 32],
    row_lo: usize,
    selected_elements: usize,
    elements: usize,
}
impl<'p, 'a> CapturePrefillFragment<'p, 'a> {
    /// The actual borrowed assembly, including its original admission and index.
    /// This is identity/geometry access, not a submitted chunk or write claim.
    pub fn assembly(&self) -> &'p CapturePrefillRowAssembly<'a> {
        self.assembly
    }
    /// Bound schedule index, not a submitted/committed chunk identity.
    pub fn chunk_index(&self) -> u64 {
        self.index
    }
    /// Prompt-relative input range derived from the bound schedule.
    pub fn input(&self) -> &Range<u64> {
        &self.input
    }
    /// Absolute decoder position, including the exact admitted cached prefix.
    pub fn position(&self) -> u64 {
        self.position
    }
    /// Original request's per-chunk output contract; no readout is authorized.
    pub fn output_demand(&self) -> OutputDemand {
        self.assembly
            .inference
            .output
            .for_chunk(self.input.end == self.assembly.inference.input_positions)
    }
    /// Comparison seam for an actual runtime chunk, without importing runtime.
    /// Matching geometry is not source, execution, funding or completion proof.
    pub fn matches_chunk(&self, input: &Range<u64>, position: u64, output: OutputDemand) -> bool {
        input == &self.input && position == self.position && output == self.output_demand()
    }
    /// Exact physical source axes before any local slice/conversion.
    pub fn source_shape(&self) -> &[usize] {
        &self.source_shape[..self.assembly.logical.source_shape().len()]
    }
    /// Physical rectangular selection before global Preview truncation.
    pub fn selected_shape(&self) -> &[usize] {
        &self.selected_shape[..self.source_shape().len()]
    }
    /// The derived local rectangular slice for one physical axis. The canonical
    /// exclusive end is one past its last selected coordinate, which can be
    /// smaller than the original slice end without changing selected values.
    /// Empty axes use 0..0 and retain their positive declared stride.
    ///
    /// Returns None for an absent axis or unrepresentable checked arithmetic.
    /// Prepared fragment bounds already guarantee representability; this check
    /// never accepts a caller-supplied offset/count. Preview remains a separate
    /// global-prefix contribution, not a change to this rectangular selection.
    pub fn selection_axis(&self, axis: usize) -> Option<CapturePrefillAxisSlice> {
        if axis >= self.source_shape().len() {
            return None;
        }
        let elements = self.selected_shape[axis];
        let stride = self.assembly.steps[axis];
        let start = if elements == 0 { 0 } else { self.starts[axis] };
        let end = if elements == 0 {
            0
        } else {
            start
                .checked_add((elements - 1).checked_mul(stride)?)?
                .checked_add(1)?
        };
        (end <= self.source_shape[axis]).then_some(CapturePrefillAxisSlice {
            start,
            end,
            stride,
            elements,
        })
    }
    /// Selected physical scalars before global Preview. Empty intersection and
    /// Preview(0) remain distinguishable for a future generated-source policy.
    pub fn selected_elements(&self) -> usize {
        self.selected_elements
    }
    /// Scalars this fragment contributes to the final logical output.
    pub fn output_elements(&self) -> usize {
        self.elements
    }
    /// Iterate actual physical-source and final-destination flat indices in
    /// physical selected order. Every entry is derived, never a write permission.
    /// Fixed non-row dimensions can scatter across batch/head destination blocks.
    pub fn mappings(
        &self,
    ) -> impl ExactSizeIterator<Item = CapturePrefillElement> + std::iter::FusedIterator + '_ {
        (0..self.elements).map(move |selected| self.element(selected))
    }
    /// Look up one contribution in physical selected order. Returns None after
    /// this fragment's global-Preview contribution, even if its rectangular
    /// selection contains more scalars. A future closed writer can retain the
    /// borrowed fragment and a scalar cursor without an allocated iterator.
    /// The index selects geometry only and grants no arbitrary destination write.
    pub fn mapping_at(&self, selected_index: usize) -> Option<CapturePrefillElement> {
        (selected_index < self.elements).then(|| self.element(selected_index))
    }
    fn element(&self, selected: usize) -> CapturePrefillElement {
        let mut remainder = selected;
        let mut source = 0;
        let mut destination = 0;
        for axis in (0..self.source_shape().len()).rev() {
            // Nonempty output proves all these selected dimensions positive.
            let coordinate = remainder % self.selected_shape[axis];
            remainder /= self.selected_shape[axis];
            source += (self.starts[axis] + coordinate * self.assembly.steps[axis])
                * self.source_strides[axis];
            let logical = coordinate
                + if axis == self.assembly.axis {
                    self.row_lo
                } else {
                    0
                };
            destination += logical * self.assembly.selected_strides[axis];
        }
        // Constructor checks full source/selected products and slice bounds;
        // each term and sum is within those extents, with Preview only reducing.
        CapturePrefillElement {
            source,
            selected,
            destination,
        }
    }
}

/// A derived positive-stride physical axis selection, with no public constructor.
/// It describes placement only; no source, write, quota or execution authority.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CapturePrefillAxisSlice {
    start: usize,
    end: usize,
    stride: usize,
    elements: usize,
}
impl CapturePrefillAxisSlice {
    /// First selected physical coordinate, or zero for an empty axis.
    pub fn start(self) -> usize {
        self.start
    }
    /// Canonical exclusive end, or zero for an empty axis.
    pub fn end(self) -> usize {
        self.end
    }
    /// Positive stride derived from the actual admitted selection.
    pub fn stride(self) -> usize {
        self.stride
    }
    /// Number of selected coordinates on this axis before global Preview.
    pub fn elements(self) -> usize {
        self.elements
    }
}

/// One read-only element correspondence. Equality conveys geometry only.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CapturePrefillElement {
    source: usize,
    selected: usize,
    destination: usize,
}
impl CapturePrefillElement {
    /// Flat index in the actual unsliced physical chunk tensor.
    pub fn source_index(self) -> usize {
        self.source
    }
    /// Flat index after its rectangular local selection, before global Preview.
    pub fn selected_index(self) -> usize {
        self.selected
    }
    /// Flat index in the final logical selected/Preview output.
    pub fn destination_index(self) -> usize {
        self.destination
    }
}

#[cfg(test)]
mod tests;
