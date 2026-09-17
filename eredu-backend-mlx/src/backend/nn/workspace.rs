//! Cold facts about the selected native implementation, independent of family.
//!
//! These bounds cover MLX tensor-buffer capacities. They exclude allocator cache
//! residency, Metal heaps/driver bookkeeping, JIT programs and unrelated process
//! memory. Those domains cannot be advertised as covered by these bounds.
//! Operation-owned host staging is a separate mandatory managed-workspace
//! domain. Its facts remain partial, so tensor-buffer estimates alone cannot
//! authorize strict admission through a completed equation trace.

use eredu_nn::{Error, workspace::*};

mod cpu;
pub(crate) use cpu::MlxCpuWorkspaceMechanisms;
mod cpu_matmul;
pub use cpu_matmul::MlxCpuMatmulMechanism;
mod attention;
mod basic;
mod addressable;
pub(crate) use addressable::{AddressableParentSource, AddressableChildSource, AddressableQuote,AddressableSources,AddressableQuoteRef,AddressableInvocation,MlxAddressableWorkspaceMechanisms};
pub(super) mod zero_fill;
mod parallel;
mod pointwise_traversal;
pub(crate) use parallel::numerical as selected_parallel_numerical;
pub(crate) use parallel::{ExpertLocalQuote,ExpertProviderWaveQuote,ExpertInactiveWaveQuote,ExpertCountQuote,ExpertTransportQuote,ExpertRegionAggregate,ExpertProviderQuote,ExpertMovementKind,ExpertTransferProfile,ExpertReorderEnvelope,ExpertLocalStage,ExpertLocalStageBound};
pub(crate) use parallel::{MlxParallelWorkspace, MlxParallelWorkspaceMechanisms,PipelineBoundaryQuote,BoundaryStageCapacity,LogicalCollectiveQuote,LogicalCollectiveKind};
mod resident_recipe;
mod resident_mechanism;
pub(crate) use resident_mechanism::ResidentExecutionMechanisms;
pub(crate) use pointwise_traversal::PointwiseTraversalFactError;
#[cfg(all(
    test,
    target_vendor = "apple",
    feature = "metal",
    not(feature = "cuda")
))]
pub(crate) use resident_recipe::original_component_tests::{
    OriginalAttentionTestPlan, OriginalComponentTestPlan,
};
pub(crate) use resident_recipe::{
    AutoregressiveEquationRecipe, AutoregressiveReadoutRecipe, EmbeddedEquationRecipe,
    IsolatedCopyNativeLayout, ResidentCompletionRecipe, ResidentNativeRecipe,
    ParallelRecipeRecorder, ResidentRecipeRecorder, ResidentSpanRecipe, SpeculativeNumericalRecipe, CpuCaptureLoan, AddressableNumericalPopulation,
};
#[cfg(all(
    test,
    target_vendor = "apple",
    feature = "metal",
    not(feature = "cuda")
))]
mod cache_state_tests;
mod convolution;
mod facts;
pub use facts::{MlxWorkspaceFactCause, MlxWorkspaceFactError};
mod grouped;
mod host;
mod hyper;
mod indexing;
mod matrix;
mod normalization;
mod packed;
mod pooling;
mod projection;
mod projection_observation;
mod representation;
pub(crate) use projection::OwnedArrayProjection;
pub use projection::{
    ExistingArrayProjection, ProjectedNativeStorage, ProjectedResidentState,
    ProjectionInventoryError, ProjectionInventoryLayout, ProjectionSourceError,
    ProjectionSourceLayout, project_existing_arrays,
};
mod candidates;
mod positional;
mod readout;
mod reduction;
mod rotary;
mod routing;
mod sampling;
mod selective_scan;
mod softmax;
mod storage;
#[cfg(all(
    test,
    target_vendor = "apple",
    feature = "metal",
    not(feature = "cuda")
))]
mod text_prompt_tests;

/// Metal buffer-capacity rules in the vendored MLX allocator. This owner has no
/// device, stream, native allocation or mutable allocator setting.
#[derive(Clone, Copy, Debug)]
pub struct MetalAllocationFacts {
    page_size: u64,
}
impl MetalAllocationFacts {
    /// Captures the Apple host page size without initializing Metal. Only a
    /// retained selection of the Metal realization may use these facts.
    #[cfg(target_vendor = "apple")]
    pub fn current_host() -> Result<Self, Error> {
        Ok(Self {
            page_size: safemlx::memory::host_page_size().map_err(Error::backend)? as u64,
        })
    }
    /// Host allocation granularity used by the selected Metal allocator.
    pub const fn page_size(self) -> u64 {
        self.page_size
    }

    /// Same owning capacity rule with its fixed cause, before ordinary error
    /// rendering. Original constructors can refuse without allocating a String.
    pub(crate) fn fixed_buffer_capacity(self, bytes: u64) -> Result<u64, MlxWorkspaceFactError> {
        facts::buffer_capacity(self, bytes)
    }

    /// Upper bound on one active buffer, including page rounding and oversized
    /// cache reuse. MLX rounds sizes above one page, then accepts a cached
    /// buffer strictly below `min(2 * size, size + 2 * page_size)`.
    /// This is an implementation bound, not an allocator observation or limit.
    pub fn buffer_capacity(self, bytes: u64) -> Result<u64, Error> {
        facts::buffer_capacity(self, bytes).map_err(MlxWorkspaceFactError::ordinary)
    }
}

/// Allocation facts for the selected MLX Metal implementation. Unknown
/// primitives remain explicitly unpriced and cannot authorize strict admission.
/// This is not a claim that the complete native inference path is covered.
#[derive(Clone, Copy, Debug)]
pub struct MlxMetalWorkspaceMechanisms {
    allocation: MetalAllocationFacts,
    sdpa_blocks: Option<u32>,
}
impl MlxMetalWorkspaceMechanisms {
    /// Captures cold host facts for a compiled Metal realization.
    #[cfg(all(target_vendor = "apple", feature = "metal", not(feature = "cuda")))]
    pub fn current_host() -> Result<Self, Error> {
        Ok(Self {
            allocation: MetalAllocationFacts::current_host()?,
            sdpa_blocks: safemlx::fast::sdpa_blocks_override().map_err(Error::backend)?,
        })
    }
    /// The allocation rules retained by this mechanism selection.
    pub const fn allocation(self) -> MetalAllocationFacts {
        self.allocation
    }
}
impl WorkspaceMechanisms for MlxMetalWorkspaceMechanisms {
    fn output_representation(
        &self,
        operation: WorkspaceOperationView<'_>,
        output: usize,
    ) -> Option<WorkspaceRepresentation> {
        representation::output(operation, output)
    }

    fn projection_input_observation_mechanism(
        &self,
        format: &eredu_nn::LinearFormatSpec,
    ) -> Result<Option<eredu_nn::ProjectionInputObservationMechanism>, Error> {
        projection_observation::selection(format)
    }

    fn grouped_observation_schedule(
        &self,
        bank: &WorkspaceGroupedBank,
        tokens: u32,
    ) -> Result<Option<WorkspaceGroupedObservationSchedule>, Error> {
        use super::grouped::{GROUPED_PROJECTION_CHUNK_THRESHOLD, GROUPED_PROJECTION_CHUNK_TOKENS};
        if let WorkspaceGroupedBank::GatedProduct(spec) = bank {
            if !matches!(
                spec.layout(),
                eredu_nn::GatedProductGroupLayout::Packed { .. }
            ) {
                return Ok(None);
            }
            if tokens > GROUPED_PROJECTION_CHUNK_THRESHOLD as u32 {
                return Ok(Some(WorkspaceGroupedObservationSchedule::TokenChunks(
                    std::num::NonZeroU32::new(GROUPED_PROJECTION_CHUNK_TOKENS as u32)
                        .expect("native grouped chunk size is positive"),
                )));
            }
        }
        Ok(Some(WorkspaceGroupedObservationSchedule::WholeBatch))
    }

    fn host_workspace_bound(
        &self,
        operation: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceHostBound>, Error> {
        host::operation_bound(operation, self)
    }
    fn operation_bound(
        &self,
        operation: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceOperationBound>, Error> {
        facts::ordinary_with(
            |sink| self.emit(operation.as_view(), sink),
            |error| ordinary_error(operation, error),
        )
    }
}

impl WorkspaceFactMechanisms for MlxMetalWorkspaceMechanisms {
    type Error = MlxWorkspaceFactError;

    fn operation_facts(
        &self,
        operation: WorkspaceOperationView<'_>,
    ) -> Result<Option<WorkspaceOperationFacts>, Self::Error> {
        self.emit(operation, &mut facts::Emitter::count())
    }

    fn write_operation_facts(
        &self,
        operation: WorkspaceOperationView<'_>,
        destination: WorkspaceEffectDestination<'_>,
    ) -> Result<Option<WorkspaceOperationFacts>, Self::Error> {
        facts::write(|sink| self.emit(operation, sink), destination)
    }

    fn host_facts(
        &self,
        operation: WorkspaceOperationView<'_>,
    ) -> Result<Option<WorkspaceHostFacts>, Self::Error> {
        host::emit(operation, self, &mut facts::HostEmitter::count())
    }

    fn write_host_facts(
        &self,
        operation: WorkspaceOperationView<'_>,
        destination: WorkspaceHostDestination<'_>,
    ) -> Result<Option<WorkspaceHostFacts>, Self::Error> {
        facts::write_host(|sink| host::emit(operation, self, sink), destination)
    }
}

impl MlxMetalWorkspaceMechanisms {
    fn emit(
        &self,
        operation: WorkspaceOperationView<'_>,
        sink: &mut facts::Emitter<'_>,
    ) -> facts::FactResult<Option<WorkspaceOperationFacts>> {
        if let Some(bound) = parallel_lookup::emit(operation, self.allocation, sink)? {
            return Ok(Some(bound));
        }
        if let Some(bound) = projection_observation::emit(operation, self.allocation, sink)? {
            return Ok(Some(bound));
        }
        if let Some(bound) = candidates::emit(operation, self.allocation, sink)? {
            return Ok(Some(bound));
        }
        if let Some(bound) = sampling::emit(operation, self.allocation, sink)? {
            return Ok(Some(bound));
        }
        #[cfg(not(feature = "cuda"))]
        if matches!(operation.kind, WorkspaceOperationKindView::GatedDeltaScan) {
            return metal_gated_delta_emit(operation, self.allocation, sink).map(Some);
        }
        if let Some(bound) = selective_scan::emit(operation, self.allocation, sink)? {
            return Ok(Some(bound));
        }
        if let Some(bound) = reduction::emit(operation, self.allocation, sink)? {
            return Ok(Some(bound));
        }
        if let Some(bound) = normalization::emit(operation, self.allocation, sink)? {
            return Ok(Some(bound));
        }
        if let Some(bound) = matrix::emit(operation, self.allocation, sink)? {
            return Ok(Some(bound));
        }
        if let Some(bound) = packed::emit(operation, self.allocation, sink)? {
            return Ok(Some(bound));
        }
        if let Some(bound) = indexing::emit(operation, self.allocation, sink)? {
            return Ok(Some(bound));
        }
        if let Some(bound) = softmax::emit(operation, self.allocation, sink)? {
            return Ok(Some(bound));
        }
        if let Some(bound) = storage::emit(operation, self.allocation, sink)? {
            return Ok(Some(bound));
        }
        if let Some(bound) = convolution::emit(operation, self.allocation, sink)? {
            return Ok(Some(bound));
        }
        if let Some(bound) = rotary::emit(operation, self.allocation, sink)? {
            return Ok(Some(bound));
        }
        if let Some(bound) = positional::emit(operation, self.allocation, sink)? {
            return Ok(Some(bound));
        }
        if let Some(bound) = readout::emit(operation, self.allocation, sink)? {
            return Ok(Some(bound));
        }
        if let Some(bound) = hyper::emit(operation, self.allocation, sink)? {
            return Ok(Some(bound));
        }
        if let Some(bound) = routing::emit(operation, self.allocation, sink)? {
            return Ok(Some(bound));
        }
        if let Some((bound, _)) = grouped::emit(operation, self.allocation, sink)? {
            return Ok(Some(bound));
        }
        if let Some(bound) = attention::emit(operation, self.allocation, self.sdpa_blocks, sink)? {
            return Ok(Some(bound));
        }
        if let Some(bound) = pooling::emit(operation, self.allocation, self.sdpa_blocks, sink)? {
            return Ok(Some(bound));
        }
        basic::emit(operation, self.allocation, sink)
    }
}

fn ordinary_error(operation: &WorkspaceOperation, error: MlxWorkspaceFactError) -> Error {
    match error.cause() {
        MlxWorkspaceFactCause::MaskedOutput(_) => readout::ordinary_error(operation, error),
        MlxWorkspaceFactCause::Selector(_) => routing::ordinary_selector_error(operation, error),
        MlxWorkspaceFactCause::GroupedLinear(_)
        | MlxWorkspaceFactCause::GroupedBank(_)
        | MlxWorkspaceFactCause::GroupedRelu2(_) => grouped::ordinary_error(operation, error),
        MlxWorkspaceFactCause::RotaryAlgorithm(_) => rotary::ordinary_error(operation, error),
        MlxWorkspaceFactCause::RotaryTable(_) => {
            if let WorkspaceOperationKind::MultiAxisRotary(spec)
            | WorkspaceOperationKind::PreparedMultiAxisRotary(spec) = &operation.kind
            {
                positional::ordinary_rotary_error(spec, error)
            } else {
                error.ordinary()
            }
        }
        MlxWorkspaceFactCause::GatedProduct(_) => {
            if let WorkspaceOperationKind::GatedProduct(policy) = operation.kind {
                if let Err(original) = policy.validate() {
                    return original;
                }
            }
            error.ordinary()
        }
        _ => error.ordinary(),
    }
}

fn add(a: u64, b: u64) -> Result<u64, Error> {
    facts::add(a, b).map_err(MlxWorkspaceFactError::ordinary)
}
fn mul(a: u64, b: u64) -> Result<u64, Error> {
    facts::mul(a, b).map_err(MlxWorkspaceFactError::ordinary)
}

#[cfg(not(feature = "cuda"))]
fn metal_gated_delta_emit(
    operation: WorkspaceOperationView<'_>,
    allocation: MetalAllocationFacts,
    sink: &mut facts::Emitter<'_>,
) -> facts::FactResult<WorkspaceOperationFacts> {
    use facts::{add, mul};
    if !matches!(operation.inputs.len(), 5 | 6)
        || operation.outputs.len() != 2
        || operation
            .inputs
            .iter()
            .chain(operation.outputs.iter())
            .any(|layout| layout.dtype() != WorkspaceDtype::Float32)
    {
        return Err(MlxWorkspaceFactError::descriptor(
            "invalid Metal gated-delta workspace descriptor",
        ));
    }
    let q = operation.inputs.get(0).unwrap().shape();
    let v = operation.inputs.get(2).unwrap().shape();
    if q.len() != 4
        || v.len() != 4
        || q.iter().any(|n| *n <= 0)
        || v.iter().any(|n| *n <= 0)
        || q[..3] != v[..3]
        || operation.inputs.get(1).unwrap().shape() != q
        || operation.inputs.get(4).unwrap().shape() != &q[..3]
        || (operation.inputs.get(3).unwrap().shape() != q
            && operation.inputs.get(3).unwrap().shape() != &q[..3])
    {
        return Err(MlxWorkspaceFactError::descriptor(
            "invalid Metal gated-delta workspace geometry",
        ));
    }
    let state_shape = [q[0], q[2], q[3], v[3]];
    if operation.outputs.get(0).unwrap().shape() != state_shape
        || operation.outputs.get(1).unwrap().shape() != v
        || operation
            .inputs
            .get(5)
            .is_some_and(|state| state.shape() != state_shape)
    {
        return Err(MlxWorkspaceFactError::descriptor(
            "Metal gated-delta workspace state/output geometry differs",
        ));
    }
    q[0].checked_mul(q[2])
        .and_then(|n| n.checked_mul(v[3]))
        .ok_or_else(|| {
            MlxWorkspaceFactError::descriptor("Metal gated-delta grid exceeds native i32 geometry")
        })?;
    let length = q[1] as u64;
    let chunk = super::gated_delta::metal_scan_chunk_tokens(q[1]) as u64;
    let full = length / chunk;
    let tail = length % chunk;
    let chunks = add(full, u64::from(tail != 0))?;
    let state = facts::buffer_capacity(allocation, operation.outputs.get(0).unwrap().bytes()?)?;
    let sequence = facts::buffer_capacity(allocation, operation.outputs.get(1).unwrap().bytes()?)?;
    let sliced = |layout: WorkspaceLayoutView<'_>| -> facts::FactResult<u64> {
        let row_bytes = layout.bytes()? / length;
        add(
            mul(
                full,
                facts::buffer_capacity(allocation, mul(row_bytes, chunk)?)?,
            )?,
            facts::buffer_capacity(allocation, mul(row_bytes, tail)?)?,
        )
    };
    let inputs = operation
        .inputs
        .slice(0..5)
        .unwrap()
        .iter()
        .try_fold(0, |total, layout| add(total, sliced(layout)?))?;
    // Per chunk: each of five sequence inputs can have a slice copy, F32 cast,
    // and CustomKernel row-contiguous copy. State can have a cast and contiguous
    // copy, followed by the newly allocated output state. Count every version
    // until completion. Add initial zero state, chunk outputs, concatenation and
    // final dtype cast; subtract only the two separately declared outputs.
    // The initial-state allowance cancels the final-state output subtraction.
    let scratch = add(
        add(mul(inputs, 3)?, mul(mul(chunks, 3)?, state)?)?,
        add(sliced(operation.outputs.get(1).unwrap())?, sequence)?,
    )?;
    sink.output(facts::Output::Allocate(state))?;
    sink.output(facts::Output::Allocate(sequence))?;
    sink.finish(scratch, format_args!("vendored MLX Metal gated-delta; shared native scan chunk policy; at most F32; all slice/cast/contiguous copies and intermediate states retained through completion; page={} bytes with bounded oversized cache reuse; tensor buffers only, excluding cache residency, heap/driver/JIT storage",allocation.page_size()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn audited_pointwise_has_explicit_host_fact_and_unknown_equations_stay_unknown() {
        use eredu_nn::Tensor;
        let context = WorkspaceContext::new(MlxMetalWorkspaceMechanisms {
            allocation: MetalAllocationFacts { page_size: 16_384 },
            sdpa_blocks: None,
        });
        let input = WorkspaceTensor::unloaded_f32(&[2, 3], &context).unwrap();
        let output = input.square(&context).unwrap();
        let report = context.report(&[output]).unwrap();
        assert!(report.tensor_buffers.total_bytes.unwrap() > 0);
        assert_eq!(report.host_workspace_bytes, Some(0));
        assert_eq!(report.total_bytes, report.tensor_buffers.total_bytes);
        assert!(report.unpriced_operations.is_empty());
        assert!(report.unpriced_host_operations.is_empty());
        let mut unknown = report.operations.last().unwrap().clone();
        unknown.kind = WorkspaceOperationKind::Elementwise("unaudited_operation");
        let selected = MlxMetalWorkspaceMechanisms {
            allocation: MetalAllocationFacts { page_size: 16_384 },
            sdpa_blocks: None,
        };
        assert!(selected.operation_bound(&unknown).unwrap().is_none());
        assert!(selected.host_workspace_bound(&unknown).unwrap().is_none());
    }
    #[test]
    fn allocation_capacity_covers_exact_rounding_and_cache_acceptance_boundaries() {
        for page in [4096, 16384] {
            let facts = MetalAllocationFacts { page_size: page };
            for requested in [
                0,
                1,
                4,
                page - 1,
                page,
                page + 1,
                2 * page - 1,
                2 * page,
                2 * page + 1,
                37 * page + 3,
            ] {
                let bound = facts.buffer_capacity(requested).unwrap();
                if requested == 0 {
                    assert_eq!(bound, 0);
                    continue;
                }
                let aligned = if requested > page {
                    requested.div_ceil(page) * page
                } else {
                    requested
                };
                assert!(bound >= aligned);
                for candidate in [aligned, bound, bound + 1] {
                    let reusable = candidate >= aligned
                        && candidate < 2 * aligned
                        && candidate < aligned + 2 * page;
                    assert_eq!(reusable, candidate <= bound);
                }
            }
        }
        assert!(
            MetalAllocationFacts { page_size: 16384 }
                .buffer_capacity(u64::MAX)
                .is_err()
        );
    }
    #[test]
    #[cfg(not(feature = "cuda"))]
    fn recurrent_workspace_prices_all_native_chunk_transitions_and_keeps_other_primitives_unknown()
    {
        let mechanism = MlxMetalWorkspaceMechanisms {
            allocation: MetalAllocationFacts { page_size: 16384 },
            sdpa_blocks: None,
        };
        let layout = |shape: &[i32]| WorkspaceLayout::new(shape, WorkspaceDtype::Float32).unwrap();
        for length in [1, 2, 16, 32, 64, 65, 255, 256, 257, 512, 3000] {
            let input = vec![
                layout(&[2, length, 4, 3]),
                layout(&[2, length, 4, 3]),
                layout(&[2, length, 4, 5]),
                layout(&[2, length, 4]),
                layout(&[2, length, 4]),
            ];
            let operation = WorkspaceOperation {
                kind: WorkspaceOperationKind::GatedDeltaScan,
                inputs: input,
                outputs: vec![layout(&[2, 4, 3, 5]), layout(&[2, length, 4, 5])],
            };
            let bound = mechanism.operation_bound(&operation).unwrap().unwrap();
            assert!(
                bound.scratch_bytes
                    > operation
                        .outputs
                        .iter()
                        .map(|value| value.bytes().unwrap())
                        .sum::<u64>()
            );
            let mut unrelated = operation;
            unrelated.kind = WorkspaceOperationKind::Elementwise("unaudited_operation");
            assert!(mechanism.operation_bound(&unrelated).unwrap().is_none());
        }
    }

    #[test]
    #[cfg(all(target_vendor = "apple", feature = "metal", not(feature = "cuda")))]
    #[ignore = "requires an exclusive Metal allocator measurement; run with --test-threads=1"]
    fn metal_recurrent_observed_peak_fits_cold_workspace_bound() {
        use safemlx::{Array, Device, DeviceType, Dtype, Stream};
        let mechanisms = MlxMetalWorkspaceMechanisms::current_host().unwrap();
        let device = Device::new(DeviceType::Gpu, 0);
        let stream = Stream::new_with_device(&device);
        let layout = |shape: &[i32]| WorkspaceLayout::new(shape, WorkspaceDtype::Float32).unwrap();
        for dtype in [Dtype::Float32, Dtype::Bfloat16] {
            for vector_decay in [false, true] {
                for length in [1, 65, 257, 3000] {
                    let decay_shape = if vector_decay {
                        vec![2, length, 4, 3]
                    } else {
                        vec![2, length, 4]
                    };
                    let operation = WorkspaceOperation {
                        kind: WorkspaceOperationKind::GatedDeltaScan,
                        inputs: vec![
                            layout(&[2, length, 4, 3]),
                            layout(&[2, length, 4, 3]),
                            layout(&[2, length, 4, 5]),
                            layout(&decay_shape),
                            layout(&[2, length, 4]),
                        ],
                        outputs: vec![layout(&[2, 4, 3, 5]), layout(&[2, length, 4, 5])],
                    };
                    let bound = mechanisms.operation_bound(&operation).unwrap().unwrap();
                    let allowed = bound.outputs.iter().fold(
                        bound.scratch_bytes,
                        |total, output| match output {
                            WorkspaceOutputStorage::Allocate(bytes) => total + bytes,
                            _ => panic!("scan returns new allocations"),
                        },
                    );
                    let inputs = operation
                        .inputs
                        .iter()
                        .enumerate()
                        .map(|(index, layout)| {
                            let mut shape = layout.shape().to_vec();
                            shape.swap(0, 1);
                            let count = layout.elements().unwrap() as usize;
                            let values = (0..count)
                                .map(|n| {
                                    if index == 3 {
                                        -0.02 - (n % 5) as f32 * 0.001
                                    } else {
                                        0.01 + (n % 7) as f32 * 0.002
                                    }
                                })
                                .collect::<Vec<_>>();
                            Array::from_slice(&values, &shape)
                                .as_dtype(dtype, &stream)
                                .unwrap()
                                .swap_axes(0, 1, &stream)
                                .unwrap()
                        })
                        .collect::<Vec<_>>();
                    safemlx::transforms::eval(&inputs).unwrap();
                    let before = safemlx::memory::active_memory().unwrap();
                    safemlx::memory::reset_peak_memory().unwrap();
                    let (state, output) = super::super::gated_delta::gated_delta_scan(
                        &inputs[0], &inputs[1], &inputs[2], &inputs[3], &inputs[4], None, &stream,
                    )
                    .unwrap();
                    safemlx::transforms::eval([&state, &output]).unwrap();
                    let incremental = safemlx::memory::peak_memory()
                        .unwrap()
                        .saturating_sub(before) as u64;
                    eprintln!(
                        "gated-delta dtype={dtype:?} vector_decay={vector_decay} positions={length} observed={incremental} bound={allowed}"
                    );
                    assert!(
                        incremental <= allowed,
                        "native peak {incremental} exceeds cold bound {allowed}"
                    );
                }
            }
        }
    }
}

pub(crate) use projection::{
    OriginalPagedAppendClaim, OriginalPagedVisibleClaim, OriginalPagedAttentionBlock, OriginalPagedBlockSource,
    OriginalPagedDiskWriteSource, OriginalPagedHostEviction, OriginalPagedHostReturn,
    OriginalPagedDiscard, OriginalPagedScanClaim, OriginalPagedScanSource, PagedAppendInput, PagedHostStoreDeclaration,
    PagedScanInput, PagedScopeRetention, ProjectedPagedSources,
};

mod parallel_lookup;

pub(crate) mod byte_view;
