//! Cold facts about the selected native implementation, independent of family.
//!
//! These bounds cover MLX tensor-buffer capacities. They exclude allocator cache
//! residency, Metal heaps/driver bookkeeping, JIT programs and unrelated process
//! memory. Those contributions are not covered by tensor-buffer bounds.
//! Operation-owned host staging is a separate mandatory managed-workspace
//! contribution. Its facts remain partial, so tensor-buffer estimates alone
//! cannot authorize admission through a completed equation trace.

use eredu_nn::{Error, workspace::*};

mod cpu;
mod parameter_backings;
pub(crate) use cpu::{MlxCpuWorkspaceMechanisms, OrdinaryCallControls};
pub(crate) use parameter_backings::{
    CompletedParameterSource, CompletedParameterSources, ParameterWorkspaceBackings,
};
mod cpu_matmul;
pub use cpu_matmul::MlxCpuMatmulMechanism;
mod addressable;
mod attention;
mod basic;
pub(crate) use addressable::{
    AddressableChildSource, AddressableEquationSource, AddressableInvocation,
    AddressableParentSource, AddressableQuote, AddressableQuoteRef, AddressableSources,
    MlxAddressableWorkspaceMechanisms, OrdinaryAddressableProgram, OrdinaryAddressableSources,
};
pub(crate) mod host_array;
mod parallel;
mod pointwise_traversal;
pub(super) mod zero_fill;
pub(crate) use parallel::numerical as selected_parallel_numerical;
pub(crate) use parallel::{
    BoundaryStageCapacity, LogicalCollectiveKind, LogicalCollectiveQuote, MlxParallelWorkspace,
    MlxParallelWorkspaceMechanisms, PipelineBoundaryQuote,
};
pub(crate) use parallel::{
    ExpertCountQuote, ExpertInactiveWaveQuote, ExpertLocalQuote, ExpertLocalStage,
    ExpertLocalStageBound, ExpertMovementKind, ExpertProviderQuote, ExpertProviderWaveQuote,
    ExpertRegionAggregate, ExpertReorderEnvelope, ExpertTransferProfile, ExpertTransportQuote,
};
mod resident_mechanism;
mod resident_recipe;
pub(crate) use pointwise_traversal::PointwiseTraversalFactError;
pub use resident_mechanism::MlxWorkspacePreparationError;
pub(crate) use resident_mechanism::{ResidentExecutionMechanisms, ResidentFactError};
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
    AddressableNumericalPopulation, AutoregressiveEquationRecipe, AutoregressiveReadoutRecipe,
    CpuCaptureLoan, EmbeddedEquationRecipe, IsolatedCopyNativeLayout, OrdinaryIndexedPrograms,
    OrdinaryNativeControls, ParallelRecipeRecorder, ResidentCompletionRecipe, ResidentNativeRecipe,
    ResidentRecipeRecorder, ResidentSamplingProgram, ResidentSpanRecipe,
    SpeculativeNumericalRecipe,
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
mod masked_scatter;
mod matrix;
mod normalization;
mod packed;
mod parameter_decode;
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

/// Buffer-capacity rules for the compiled CPU or Metal allocator. This owner has no
/// device, stream, native allocation or mutable allocator setting.
#[derive(Clone, Copy, Debug)]
pub struct NativeAllocationFacts {
    page_size: u64,
    cpu_header: bool,
    original_storage: bool,
}
impl NativeAllocationFacts {
    /// Captures the actual compiled allocator layout and host page size without
    /// initializing a device or allocator. The native layout query distinguishes
    /// CPU storage from shared Metal storage even under Cargo feature unification.
    #[cfg(target_vendor = "apple")]
    pub fn current_host() -> Result<Self, Error> {
        let layout = safemlx::PreparedInputAllocator::<()>::layout()
            .map_err(Error::backend_retained_source)?;
        Ok(Self {
            page_size: safemlx::memory::host_page_size().map_err(Error::backend)? as u64,
            cpu_header: !layout.requires_device,
            original_storage: false,
        })
    }
    fn host_control_bytes(self) -> Option<u64> {
        if self.original_storage {
            Some(0)
        } else {
            u64::try_from(safemlx::physical_backing_control_bytes())
                .ok()?
                .checked_add(crate::backend::managed_memory::ordinary_root_metadata_bytes().ok()?)
        }
    }
    /// Host page granularity used by original physical allocations.
    pub const fn page_size(self) -> u64 {
        self.page_size
    }

    /// Same owning capacity rule with its fixed cause, before ordinary error
    /// rendering. Original constructors can refuse without allocating a String.
    pub(crate) fn fixed_buffer_capacity(self, bytes: u64) -> Result<u64, MlxWorkspaceFactError> {
        facts::buffer_capacity(self, bytes)
    }

    /// Upper bound on ordinary buffer backing, including oversized cache reuse.
    /// CPU bounds also cover page-backed original allocations and their size
    /// header, even for empty payloads. Metal original allocations use their
    /// separately queried physical capacity; ordinary Metal rounds above a page.
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
    allocation: NativeAllocationFacts,
    sdpa_blocks: Option<u32>,
}
impl MlxMetalWorkspaceMechanisms {
    /// Captures shared allocator facts and the selected Metal attention setting.
    /// CPU equation dispatch uses the same allocator facts with its own workers.
    #[cfg(all(target_vendor = "apple", not(feature = "cuda")))]
    pub fn current_host() -> Result<Self, Error> {
        let allocation = NativeAllocationFacts::current_host()?;
        Ok(Self {
            allocation,
            sdpa_blocks: if allocation.cpu_header {
                None
            } else {
                safemlx::fast::sdpa_blocks_override().map_err(Error::backend)?
            },
        })
    }
    /// Selects the direct allocator whose records are prepaid in the native
    /// original buffer arena. This is a quotation fact, never execution authority.
    pub(crate) const fn original_storage(mut self) -> Self {
        self.allocation.original_storage = true;
        self
    }
    /// Selects the ordinary allocator, including its persistent root controls
    /// and publication metadata. The selected operator implementation is unchanged.
    pub(crate) const fn ordinary_storage(mut self) -> Self {
        self.allocation.original_storage = false;
        self
    }
    /// The allocation rules retained by this mechanism selection.
    pub const fn allocation(self) -> NativeAllocationFacts {
        self.allocation
    }
}
impl WorkspaceMechanisms for MlxMetalWorkspaceMechanisms {
    fn prepare_allocation_sources(
        &self,
        operation: WorkspaceOperationView<'_>,
        context: &WorkspaceContext,
    ) -> Result<Option<WorkspaceOperationAllocationSources>, Error> {
        ResidentExecutionMechanisms::Metal(*self).prepare_allocation_sources(operation, context)
    }

    fn scratch_allocation_count(
        &self,
        operation: WorkspaceOperationView<'_>,
    ) -> Result<Option<usize>, Error> {
        self.scratch_births(operation)
            .map_err(|cause| Error::backend_retained_source(cause))
    }

    fn completion_strategy(&self) -> WorkspaceCompletionStrategy {
        if self.allocation.original_storage {
            WorkspaceCompletionStrategy::EnclosingSubmission
        } else {
            WorkspaceCompletionStrategy::OperationSubmissions
        }
    }
    fn memory_topology(&self) -> Option<&eredu_core::MemoryTopology> {
        crate::backend::managed_memory::cold_topology()
    }
    fn allocation_host_control_bytes(
        &self,
        _: WorkspaceOperationView<'_>,
        _: usize,
    ) -> Option<u64> {
        self.allocation.host_control_bytes()
    }
    fn scratch_host_control_bytes(
        &self,
        operation: WorkspaceOperationView<'_>,
    ) -> Result<Option<u64>, Error> {
        self.scratch_controls(operation)
            .map_err(MlxWorkspaceFactError::ordinary)
    }

    fn output_placement(
        &self,
        operation: WorkspaceOperationView<'_>,
        _: usize,
    ) -> Option<&eredu_core::MemoryPlacement> {
        // The typed slice worker adds only Copy, whose qualified GPU producer
        // retains the eager seed's backing and its default-allocator placement.
        if basic::is_unattributed_initializer(operation) {
            None
        } else if self.allocation.original_storage {
            crate::backend::managed_memory::cold_original_allocator_placement()
        } else if host_array::dtype(operation).is_some()
            || host_array::slice_dtype(operation).is_some()
            || basic::is_scalar_f32(operation)
            || basic::is_scalar_u8(operation)
            || matches!(
                operation.kind,
                WorkspaceOperationKindView::GeneratedF32Initialization
            )
        {
            crate::backend::managed_memory::cold_default_placement()
        } else {
            crate::backend::managed_memory::cold_gpu_allocator_placement()
        }
    }
    fn scratch_placement(
        &self,
        operation: WorkspaceOperationView<'_>,
    ) -> Option<&eredu_core::MemoryPlacement> {
        if basic::is_unattributed_initializer(operation) {
            None
        } else if self.allocation.original_storage {
            crate::backend::managed_memory::cold_original_allocator_placement()
        } else if zero_fill::dtype(operation).is_some() {
            // The typed Full worker's only scratch backing is its eager seed.
            // The independently allocated fill result remains on the GPU source.
            crate::backend::managed_memory::cold_default_placement()
        } else {
            crate::backend::managed_memory::cold_gpu_allocator_placement()
        }
    }
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
        grouped::observation_schedule(bank, tokens)
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
    fn memory_topology(&self) -> Option<&eredu_core::MemoryTopology> {
        crate::backend::managed_memory::cold_topology()
    }
    fn allocation_host_control_bytes(
        &self,
        _: WorkspaceOperationView<'_>,
        _: usize,
    ) -> Option<u64> {
        self.allocation.host_control_bytes()
    }
    fn scratch_host_control_bytes(
        &self,
        operation: WorkspaceOperationView<'_>,
    ) -> Result<Option<u64>, Self::Error> {
        self.scratch_controls(operation).map_err(Into::into)
    }

    fn output_placement(
        &self,
        operation: WorkspaceOperationView<'_>,
        _: usize,
    ) -> Option<&eredu_core::MemoryPlacement> {
        if basic::is_unattributed_initializer(operation) {
            None
        } else if self.allocation.original_storage {
            crate::backend::managed_memory::cold_original_allocator_placement()
        } else if host_array::dtype(operation).is_some()
            || host_array::slice_dtype(operation).is_some()
            || basic::is_scalar_f32(operation)
            || basic::is_scalar_u8(operation)
            || matches!(
                operation.kind,
                WorkspaceOperationKindView::GeneratedF32Initialization
            )
        {
            crate::backend::managed_memory::cold_default_placement()
        } else {
            crate::backend::managed_memory::cold_gpu_allocator_placement()
        }
    }
    fn scratch_placement(
        &self,
        operation: WorkspaceOperationView<'_>,
    ) -> Option<&eredu_core::MemoryPlacement> {
        if basic::is_unattributed_initializer(operation) {
            None
        } else if self.allocation.original_storage {
            crate::backend::managed_memory::cold_original_allocator_placement()
        } else if zero_fill::dtype(operation).is_some() {
            // The typed Full worker's only scratch backing is its eager seed.
            // The independently allocated fill result remains on the GPU source.
            crate::backend::managed_memory::cold_default_placement()
        } else {
            crate::backend::managed_memory::cold_gpu_allocator_placement()
        }
    }
    type Error = MlxWorkspacePreparationError;
    fn with_prepared_facts<T>(
        &self,
        operation: WorkspaceOperationView<'_>,
        funding: Option<&HostMetadataFunding>,
        visit: impl FnOnce(&dyn WorkspaceFactMechanisms<Error = Self::Error>) -> T,
    ) -> Result<T, Self::Error> {
        ResidentExecutionMechanisms::Metal(*self).with_prepared_facts(operation, funding, visit)
    }

    fn with_prepared_facts_context<T>(
        &self,
        operation: WorkspaceOperationView<'_>,
        context: &WorkspaceContext,
        visit: impl FnOnce(&dyn WorkspaceFactMechanisms<Error = Self::Error>) -> T,
    ) -> Result<T, Self::Error> {
        ResidentExecutionMechanisms::Metal(*self)
            .with_prepared_facts_context(operation, context, visit)
    }
    fn operation_facts(
        &self,
        operation: WorkspaceOperationView<'_>,
    ) -> Result<Option<WorkspaceOperationFacts>, Self::Error> {
        self.emit(operation, &mut facts::Emitter::count())
            .map_err(Into::into)
    }

    fn write_operation_facts(
        &self,
        operation: WorkspaceOperationView<'_>,
        destination: WorkspaceEffectDestination<'_>,
    ) -> Result<Option<WorkspaceOperationFacts>, Self::Error> {
        facts::write(|sink| self.emit(operation, sink), destination).map_err(Into::into)
    }

    fn host_facts(
        &self,
        operation: WorkspaceOperationView<'_>,
    ) -> Result<Option<WorkspaceHostFacts>, Self::Error> {
        host::emit(operation, self, &mut facts::HostEmitter::count()).map_err(Into::into)
    }

    fn write_host_facts(
        &self,
        operation: WorkspaceOperationView<'_>,
        destination: WorkspaceHostDestination<'_>,
    ) -> Result<Option<WorkspaceHostFacts>, Self::Error> {
        facts::write_host(|sink| host::emit(operation, self, sink), destination).map_err(Into::into)
    }
}

impl MlxMetalWorkspaceMechanisms {
    fn scratch_births(
        &self,
        operation: WorkspaceOperationView<'_>,
    ) -> facts::FactResult<Option<usize>> {
        let mut emitted = facts::Emitter::count();
        let Some(bound) = self.emit(operation, &mut emitted)? else {
            return Ok(None);
        };
        if bound.scratch_bytes == 0 {
            return Ok(Some(0));
        }
        let births = if matches!(operation.kind, WorkspaceOperationKindView::DeepCopy) {
            // The ordinary eager clone may first materialize one contiguous
            // source; its final independent backing is priced as an output.
            1
        } else {
            let Some(births) = resident_recipe::allocation_births(operation) else {
                return Ok(None);
            };
            births
                .checked_sub(emitted.allocated_output_births())
                .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?
        };
        Ok(Some(births))
    }
    fn scratch_controls(
        &self,
        operation: WorkspaceOperationView<'_>,
    ) -> facts::FactResult<Option<u64>> {
        if self.allocation.original_storage {
            return Ok(Some(0));
        }
        let Some(births) = self.scratch_births(operation)? else {
            return Ok(None);
        };
        let Some(control) = self.allocation.host_control_bytes() else {
            return Ok(None);
        };
        Ok(Some(facts::mul(control, u64::try_from(births)?)?))
    }

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
        if let Some(bound) = parameter_decode::emit(operation, self.allocation, sink)? {
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
    allocation: NativeAllocationFacts,
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
            allocation: NativeAllocationFacts {
                page_size: 16_384,
                cpu_header: false,
                original_storage: false,
            },
            sdpa_blocks: None,
        });
        let input = WorkspaceTensor::unloaded_f32(&[2, 3], &context).unwrap();
        let output = input.square(&context).unwrap();
        let report = context.report(&[output]).unwrap();
        assert!(report.tensor_buffers.total_bytes.unwrap() > 0);
        assert_eq!(report.host_workspace_bytes, Some(0));
        assert_eq!(
            report.total_bytes,
            report
                .tensor_buffers
                .total_bytes
                .map(|n| n + test_backing_controls(
                    &MlxMetalWorkspaceMechanisms {
                        allocation: NativeAllocationFacts {
                            page_size: 16_384,
                            cpu_header: false,
                            original_storage: false
                        },
                        sdpa_blocks: None
                    },
                    &report
                ))
        );
        assert!(report.unpriced_operations.is_empty());
        assert!(report.unpriced_host_operations.is_empty());
        let mut unknown = report.operations.last().unwrap().clone();
        unknown.kind = WorkspaceOperationKind::Elementwise("unaudited_operation");
        let selected = MlxMetalWorkspaceMechanisms {
            allocation: NativeAllocationFacts {
                page_size: 16_384,
                cpu_header: false,
                original_storage: false,
            },
            sdpa_blocks: None,
        };
        assert!(selected.operation_bound(&unknown).unwrap().is_none());
        assert!(selected.host_workspace_bound(&unknown).unwrap().is_none());
    }
    #[test]
    fn allocation_capacity_covers_exact_rounding_and_cache_acceptance_boundaries() {
        for page in [4096, 16384] {
            let facts = NativeAllocationFacts {
                page_size: page,
                cpu_header: false,
                original_storage: false,
            };
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
            NativeAllocationFacts {
                page_size: 16384,
                cpu_header: false,
                original_storage: false
            }
            .buffer_capacity(u64::MAX)
            .is_err()
        );
    }

    #[test]
    fn cpu_allocation_capacity_covers_headers_pages_and_cache_reuse() {
        let header = std::mem::size_of::<usize>() as u64;
        for page in [4096_u64, 16384] {
            let facts = NativeAllocationFacts {
                page_size: page,
                cpu_header: true,
                original_storage: false,
            };
            for requested in [
                0,
                1,
                page - header,
                page - header + 1,
                page,
                page + 1,
                37 * page + 3,
            ] {
                let bound = facts.buffer_capacity(requested).unwrap();
                let physical = ((requested + header - 1) / page + 1) * page;
                assert!(bound >= physical, "original backing for {requested} bytes");
                assert!(bound < physical + 2 * page);
                if requested > 0 {
                    // The ordinary CPU cache uses a 4096-byte search granularity.
                    let largest_cached_payload = (2 * requested).min(requested + 8192) - 1;
                    assert!(bound >= largest_cached_payload + header);
                }
            }
            assert!(facts.buffer_capacity(u64::MAX).is_err());
        }
        let facts = NativeAllocationFacts {
            page_size: 16384,
            cpu_header: true,
            original_storage: false,
        };
        assert_eq!(facts.buffer_capacity(0).unwrap(), 32767);
        assert_eq!(facts.buffer_capacity(16384 - header).unwrap(), 32767);
        assert_eq!(facts.buffer_capacity(16385 - header).unwrap(), 65535);
    }

    #[test]
    #[cfg(target_vendor = "apple")]
    fn selected_allocation_capacity_covers_native_backing() {
        const CASE: &str = "selected_allocation_capacity_covers_native_backing";
        if std::env::var("EREDU_ALLOCATION_FACT_CASE").as_deref() != Ok(CASE) {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    &format!("backend::nn::workspace::tests::{CASE}"),
                    "--nocapture",
                ])
                .env("EREDU_ALLOCATION_FACT_CASE", CASE)
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "{}\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            assert!(String::from_utf8_lossy(&output.stdout).contains("ALLOCATION_FACT_OK"));
            return;
        }
        let facts = NativeAllocationFacts::current_host().unwrap();
        let runtime = safemlx::PreparedInputRuntime::prepare().unwrap();
        let page = facts.page_size() as usize;
        for requested in [0, 1, page - 8, page - 1, page, page + 1, 37 * page + 3] {
            let bound = facts.buffer_capacity(requested as u64).unwrap();
            let array = safemlx::Array::from_slice(&vec![0_u8; requested], &[requested as i32]);
            let allocation = array.allocation_info().unwrap();
            if requested != 0 {
                assert!(allocation.is_some());
            }
            assert!(allocation.map_or(0, |info| info.bytes()) as u64 <= bound);
            if facts.cpu_header {
                let layout =
                    safemlx::OriginalBufferBudget::request_layout(&runtime, requested).unwrap();
                assert!(
                    bound >= layout.capacity() as u64,
                    "original CPU request {requested}"
                );
                assert!(layout.capacity() > 0);
            }
        }
        println!("ALLOCATION_FACT_OK");
    }
    #[test]
    #[cfg(not(feature = "cuda"))]
    fn recurrent_workspace_prices_all_native_chunk_transitions_and_keeps_other_primitives_unknown()
    {
        let mechanism = MlxMetalWorkspaceMechanisms {
            allocation: NativeAllocationFacts {
                page_size: 16384,
                cpu_header: false,
                original_storage: false,
            },
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
    OrdinaryPagedAppend, OrdinaryPagedCause, OrdinaryPagedHostScan, OrdinaryPagedProgram,
    OrdinaryPagedWork, OriginalPagedAppendClaim, OriginalPagedAttentionBlock,
    OriginalPagedBlockSource, OriginalPagedDiscard, OriginalPagedDiskWriteSource,
    OriginalPagedHostEviction, OriginalPagedHostReturn, OriginalPagedScanClaim,
    OriginalPagedScanSource, OriginalPagedVisibleClaim, PagedAppendInput,
    PagedHostStoreDeclaration, PagedMutationCause, PagedScanInput, PagedScopeRetention,
    PreparedOrdinaryPagedScan, ProjectedPagedSources,
};

mod parallel_lookup;

pub(crate) mod byte_view;

/// Converts an original native root into the two authenticated registry rows
/// retained by its one workspace backing identity.
pub(crate) fn registered_storage_row(
    identity: safemlx::AllocationIdentity,
    root: &WorkspaceExistingStorage,
) -> eredu_runtime::working_memory::RegisteredWorkspaceStorageRow<
    crate::backend::runtime::residency::storage::StorageIdentity,
> {
    use crate::backend::runtime::residency::storage::StorageIdentity;
    let row = eredu_runtime::working_memory::RegisteredWorkspaceStorageRow::new(
        StorageIdentity::Native(identity),
        root.clone(),
    );
    if root.host_control_bytes().is_some_and(|bytes| bytes != 0) {
        row.with_host_controls(StorageIdentity::NativeControl(identity))
    } else {
        row
    }
}

#[cfg(test)]
fn test_operation_backing_controls(
    selected: &impl WorkspaceMechanisms,
    operation: &WorkspaceOperation,
) -> u64 {
    let bound = selected.operation_bound(operation).unwrap().unwrap();
    let outputs = bound
        .outputs
        .iter()
        .enumerate()
        .filter(|(_, effect)| {
            matches!(
                effect,
                WorkspaceOutputStorage::Allocate(_)
                    | WorkspaceOutputStorage::AllocateOrAliasInputs { .. }
            )
        })
        .map(|(index, _)| {
            selected
                .allocation_host_control_bytes(operation.as_view(), index)
                .expect("selected backing has a finite native control layout")
        })
        .sum::<u64>();
    outputs
        + if bound.scratch_bytes == 0 {
            0
        } else {
            selected
                .scratch_host_control_bytes(operation.as_view())
                .unwrap()
                .expect("selected scratch population has finite native controls")
        }
}

#[cfg(test)]
fn test_backing_controls(
    selected: &impl WorkspaceMechanisms,
    report: &WorkspaceTraceReport,
) -> u64 {
    report
        .operations
        .iter()
        .map(|operation| test_operation_backing_controls(selected, operation))
        .sum()
}

#[cfg(test)]
fn test_backing_control_completeness(
    selected: &impl WorkspaceMechanisms,
    report: &WorkspaceTraceReport,
) {
    assert!(report.tensor_buffers.total_bytes.is_some());
    assert!(report.unpriced_operations.is_empty());
    let missing = report
        .operations
        .iter()
        .enumerate()
        .filter_map(|(index, operation)| {
            let bound = selected.operation_bound(operation).unwrap().unwrap();
            (bound.scratch_bytes != 0
                && selected
                    .scratch_host_control_bytes(operation.as_view())
                    .unwrap()
                    .is_none())
            .then_some(index)
        })
        .collect::<Vec<_>>();
    assert_eq!(
        report.total_bytes.is_some(),
        missing.is_empty(),
        "missing backing controls: {missing:?}"
    );
    for index in missing {
        assert!(
            report.unpriced_host_operations.contains(&index),
            "missing native backing controls must identify operation {index}: {:?}",
            report.operations[index].kind
        );
    }
}
