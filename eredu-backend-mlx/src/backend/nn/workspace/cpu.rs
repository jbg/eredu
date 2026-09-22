//! CPU model facts consume the selected source-visible worker, never GPU bounds.
use super::*;
use safemlx::{CpuBinaryOperation, CpuCopyEvalLayout, OperationEvent};
use std::mem::{size_of, size_of_val};
mod activation;
mod affine;
pub(super) mod attention;
pub(super) mod blockwise;
mod bool_broadcast;
mod byte_frame;
mod byte_view;
mod candidates;
mod capture_cast;
mod causal_mask;
mod classification;
mod comparison;
mod concatenate;
mod constant_pad;
mod dense;
mod depthwise;
mod embedding;
mod expert_movement;
mod flat_reduction;
mod gated;
mod gelu;
pub(super) mod grouped;
pub(super) mod host_transfer;
mod indexed_bank;
mod indexed_elements;
pub(crate) use indexed_bank::OrdinaryIndexedNumericalFacts;
mod integer_pointwise;
mod integer_storage;
mod masked_readout;
mod normalization;
mod ordinary_calls;
pub(crate) use ordinary_calls::OrdinaryCallControls;
mod numerical_difference;
mod numerical_mask;
mod numerical_random;
mod parameter_decode;
mod pending_input;
mod pointwise;
mod prepared_rotary;
mod program;
mod recipe_validation;
mod rope;
pub(super) mod router;
mod sampling_filters;
mod sampling_mirostat;
mod sampling_penalties;
mod sampling_state;
mod softplus;
mod static_index;
mod static_slice;
mod static_update;
mod storage_copy;
mod strided_capture;
mod summary_selection;
#[cfg(all(test, target_vendor = "apple", not(feature = "cuda")))]
pub(super) mod test_execution;
mod token_scores;
mod views;
mod zero_fill;
pub(super) use numerical_difference::{
    difference as difference_population, logarithm as logarithm_population,
};
pub(super) use numerical_random::{
    categorical as categorical_population, uniform_unit_interval as uniform_population,
};

/// Cold CPU facts paired with their exact recipe selection. Allocation facts
/// describe the compiled native allocator; operator facts are CPU-specific.
#[derive(Clone, Copy, Debug)]
pub(crate) struct MlxCpuWorkspaceMechanisms {
    allocation: NativeAllocationFacts,
    matmul: MlxCpuMatmulMechanism,
}
impl MlxCpuWorkspaceMechanisms {
    pub(crate) const fn ordinary_storage(mut self) -> Self {
        self.allocation.original_storage = false;
        self
    }
    pub(crate) const fn new(
        allocation: NativeAllocationFacts,
        matmul: MlxCpuMatmulMechanism,
    ) -> Self {
        Self { allocation, matmul }
    }
    pub(super) fn plan(
        self,
        operation: WorkspaceOperationView<'_>,
    ) -> facts::FactResult<Option<OperationPlan>> {
        if matches!(
            operation.kind,
            WorkspaceOperationKindView::GroupSelection { .. }
                | WorkspaceOperationKindView::JointGroupSelection(_)
        ) {
            return router::inspect(operation, self);
        }
        if matches!(operation.kind, WorkspaceOperationKindView::Grouped { .. }) {
            return grouped::inspect(operation, self);
        }
        if matches!(
            operation.kind,
            WorkspaceOperationKindView::BlockwiseAttention { .. }
        ) {
            return blockwise::inspect(operation, self);
        }
        if let Some(plan) = host_transfer::inspect(operation, self)? {
            return Ok(Some(plan));
        }
        if let Some(plan) = storage_copy::inspect(operation, self)? {
            return Ok(Some(plan));
        }
        if let Some(plan) = pending_input::inspect(operation, self)? {
            return Ok(Some(plan));
        }
        if let Some(plan) = host_array(operation, self)? {
            return Ok(Some(plan));
        }
        if let Some(plan) = prepared_rotary::inspect(operation, self)? {
            return Ok(Some(plan));
        }
        if let Some(plan) = zero_fill::inspect(operation, self)? {
            return Ok(Some(plan));
        }
        if let Some(plan) = byte_frame::inspect(operation, self)? {
            return Ok(Some(plan));
        }
        if let Some(plan) = affine::inspect(operation, self)? {
            return Ok(Some(plan));
        }
        if let Some(plan) = parameter_decode::inspect(operation, self)? {
            return Ok(Some(plan));
        }
        if let Some(plan) = dense::inspect(operation, self)? {
            return Ok(Some(plan));
        }
        if let Some(plan) = depthwise::inspect(operation, self)? {
            return Ok(Some(plan));
        }
        if let Some(plan) = integer_pointwise::inspect(operation, self)? {
            return Ok(Some(plan));
        }
        if let Some(plan) = indexed_bank::inspect(operation, self)? {
            return Ok(Some(plan));
        }
        if let Some(plan) = integer_storage::inspect(operation, self)? {
            return Ok(Some(plan));
        }
        if let Some(plan) = pointwise::inspect(operation, self)? {
            return Ok(Some(plan));
        }
        if let Some(plan) = embedding::inspect_token_validation(operation, self)? {
            return Ok(Some(plan));
        }
        if let Some(plan) = embedding::inspect(operation, self)? {
            return Ok(Some(plan));
        }
        if let Some(plan) = expert_movement::inspect(operation, self)? {
            return Ok(Some(plan));
        }
        if let Some(plan) = normalization::inspect(operation, self)? {
            return Ok(Some(plan));
        }
        if let Some(plan) = activation::inspect(operation, self)? {
            return Ok(Some(plan));
        }
        if let Some(plan) = softplus::inspect(operation, self)? {
            return Ok(Some(plan));
        }
        if let Some(plan) = gated::inspect(operation, self)? {
            return Ok(Some(plan));
        }
        if let Some(plan) = bool_broadcast::inspect(operation)? {
            return Ok(Some(plan));
        }
        if let Some(plan) = byte_view::inspect(operation, self)? {
            return Ok(Some(plan));
        }
        if let Some(plan) = views::inspect(operation, self)? {
            return Ok(Some(plan));
        }
        if let Some(plan) = concatenate::inspect(operation, self)? {
            return Ok(Some(plan));
        }
        if let Some(plan) = capture_cast::inspect(operation, self)? {
            return Ok(Some(plan));
        }
        if let Some(plan) = classification::inspect(operation, self)? {
            return Ok(Some(plan));
        }
        if let Some(plan) = flat_reduction::inspect(operation, self)? {
            return Ok(Some(plan));
        }
        if let Some(plan) = summary_selection::inspect(operation, self)? {
            return Ok(Some(plan));
        }
        if let Some(plan) = comparison::inspect(operation, self)? {
            return Ok(Some(plan));
        }
        if let Some(plan) = recipe_validation::inspect(operation, self)? {
            return Ok(Some(plan));
        }
        if let Some(plan) = token_scores::inspect(operation, self)? {
            return Ok(Some(plan));
        }
        if let Some(plan) = candidates::inspect(operation, self)? {
            return Ok(Some(plan));
        }
        if let Some(plan) = static_index::inspect(operation, self)? {
            return Ok(Some(plan));
        }
        if let Some(plan) = static_slice::inspect(operation, self)? {
            return Ok(Some(plan));
        }
        if let Some(plan) = static_update::inspect(operation, self)? {
            return Ok(Some(plan));
        }
        if let Some(plan) = constant_pad::inspect(operation, self)? {
            return Ok(Some(plan));
        }
        if let Some(plan) = indexed_elements::inspect(operation, self)? {
            return Ok(Some(plan));
        }
        if let Some(plan) = gelu::inspect(operation, self)? {
            return Ok(Some(plan));
        }
        if let Some(plan) = rope::inspect(operation, self)? {
            return Ok(Some(plan));
        }
        if let Some(plan) = attention::inspect(operation, self)? {
            return Ok(Some(plan));
        }
        if let Some(plan) = causal_mask::inspect(operation, self)? {
            return Ok(Some(plan));
        }
        if let Some(plan) = numerical_mask::inspect(operation, self)? {
            return Ok(Some(plan));
        }
        if let Some(plan) = sampling_filters::inspect(operation, self)? {
            return Ok(Some(plan));
        }
        if let Some(plan) = sampling_mirostat::inspect(operation, self)? {
            return Ok(Some(plan));
        }
        if let Some(plan) = sampling_penalties::inspect(operation, self)? {
            return Ok(Some(plan));
        }
        if let Some(plan) = sampling_state::inspect(operation, self)? {
            return Ok(Some(plan));
        }
        masked_readout::inspect(operation, self)
    }
    fn emit(
        self,
        operation: WorkspaceOperationView<'_>,
        sink: &mut facts::Emitter<'_>,
    ) -> facts::FactResult<Option<WorkspaceOperationFacts>> {
        if matches!(
            operation.kind,
            WorkspaceOperationKindView::CommunicationControl(_)
        ) {
            return basic::emit(operation, self.allocation, sink);
        }
        if matches!(
            operation.kind,
            WorkspaceOperationKindView::ParameterPlaceholder
        ) {
            return basic::emit_parameter_placeholder(operation, self.allocation, sink).map(Some);
        }
        if matches!(
            operation.kind,
            WorkspaceOperationKindView::ValueCompletion
                | WorkspaceOperationKindView::ValueRetention
        ) {
            if value_frontier(operation).is_none() {
                return Ok(None);
            }
            return basic::emit(operation, self.allocation, sink);
        }
        if basic::is_prepared_token_input(operation) {
            sink.output(facts::Output::Allocate(
                self.allocation.fixed_buffer_capacity(
                    operation
                        .outputs
                        .get(0)
                        .expect("validated prepared integer matrix")
                        .bytes()?,
                )?,
            ))?;
            return sink.finish(0, format_args!("one integer matrix from the separately admitted prepared text input; native input/source controls remain external to the model equation")).map(Some);
        }
        let Some(plan) = self.plan(operation)? else {
            return Ok(None);
        };
        if matches!(
            operation.kind,
            WorkspaceOperationKindView::HostStoreFloating(..)
        ) {
            // The accepted Host destination is not a new Device tensor output.
        } else if matches!(
            operation.kind,
            WorkspaceOperationKindView::GroupSelection { .. }
                | WorkspaceOperationKindView::JointGroupSelection(_)
        ) {
            router::emit_outputs(operation, self, sink)?;
        } else if matches!(operation.kind, WorkspaceOperationKindView::Grouped { .. }) {
            grouped::emit_outputs(operation, self, sink)?;
        } else if matches!(
            operation.kind,
            WorkspaceOperationKindView::BlockwiseAttention { .. }
        ) {
            blockwise::emit_outputs(operation, self, sink)?;
        } else if matches!(
            operation.kind,
            WorkspaceOperationKindView::Contiguous
                | WorkspaceOperationKindView::Index { .. }
                | WorkspaceOperationKindView::Sampling(
                    WorkspaceSamplingOperation::OptionalTokenFilter
                )
        ) || (matches!(operation.kind, WorkspaceOperationKindView::View("reshape"))
            && plan.alias_input.is_none()
            && (views::reshape_may_alias(operation)
                || operation.outputs.get(0).is_some_and(|v| {
                    matches!(v.dtype(), WorkspaceDtype::Int32 | WorkspaceDtype::Uint32)
                })))
        {
            // Optional masks may return the full input. Contiguous/reshape
            // workers may also alias or copy. Retain both storage possibilities.
            sink.output(facts::Output::AllocateOrAliasInputs {
                bytes: plan.output_bytes,
                inputs: facts::Aliases::Slice(&[0]),
            })?;
        } else if matches!(
            operation.kind,
            WorkspaceOperationKindView::CandidateExtraction { .. }
                | WorkspaceOperationKindView::PreparedMultiAxisRotary(_)
                | WorkspaceOperationKindView::Elementwise("indexed_bank_discovery")
        ) {
            // Each composite result owns its declared destination envelope;
            // intermediate backing remains separately retained in scratch.
            for output in operation.outputs.iter() {
                sink.output(facts::Output::Allocate(
                    self.allocation.fixed_buffer_capacity(output.bytes()?)?,
                ))?;
            }
        } else {
            sink.output(match plan.alias_input {
                Some(input) => facts::Output::AliasInput(input),
                None => facts::Output::Allocate(plan.output_bytes),
            })?;
        }
        sink.finish(plan.scratch_bytes, format_args!("selected source-visible CPU equation; exact native alias/arithmetic producers and unchanged rounding; F32 physical envelope; fixed worker controls separate; no GPU or BLAS allowance"))
            .map(Some)
    }
    fn host(
        self,
        operation: WorkspaceOperationView<'_>,
        sink: &mut facts::HostEmitter<'_>,
    ) -> facts::FactResult<Option<WorkspaceHostFacts>> {
        if matches!(
            operation.kind,
            WorkspaceOperationKindView::CommunicationControl(_)
        ) {
            basic::emit(operation, self.allocation, &mut facts::Emitter::count())?;
            return sink.finish(0, format_args!("descriptive model control event has no host numerical payload; its source and agreement child are quoted separately")).map(Some);
        }
        if matches!(
            operation.kind,
            WorkspaceOperationKindView::ParameterPlaceholder
        ) {
            return basic::emit_parameter_placeholder_host(operation, sink).map(Some);
        }
        if matches!(
            operation.kind,
            WorkspaceOperationKindView::ValueCompletion
                | WorkspaceOperationKindView::ValueRetention
        ) {
            if value_frontier(operation).is_none() {
                return Ok(None);
            }
            return sink.finish(0,format_args!("borrowed value completion/retention has no host numerical payload; actual CPU root and traversal controls are separately priced")).map(Some);
        }
        if basic::is_prepared_token_input(operation) {
            return sink.finish(0, format_args!("prepared token payload is owned and priced by the existing input producer; model equations borrow it")).map(Some);
        }
        if matches!(
            operation.kind,
            WorkspaceOperationKindView::GeneratedF32Initialization
        ) {
            let plan = basic::generated_f32_plan_view(operation)?;
            return sink.finish(plan.host_buffer_bytes(),
                format_args!("shared fixed F32 initializer retains its Rust vector through the eager native upload and Copy alias"))
                .map(Some);
        }
        if self.plan(operation)?.is_none() {
            return Ok(None);
        }
        if let WorkspaceOperationKindView::PreparedMultiAxisRotary(spec) = operation.kind {
            let input = operation.inputs.get(0).expect("validated rotary source");
            let profile = crate::tensor::PreparedRotaryProfile::inspect(spec, input.shape().len())
                .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?;
            let bytes = if self.allocation.original_storage {
                0
            } else {
                u64::try_from(
                    profile
                        .ordinary_row_bytes()
                        .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?,
                )?
            };
            return sink.finish(bytes, format_args!("prepared rotary borrows its frequency payload; original rows use the separately counted Graph bank, ordinary rows retain the selected SmallVec capacity")).map(Some);
        }
        if matches!(
            operation.kind,
            WorkspaceOperationKindView::ParameterDecode(
                eredu_nn::parameter_values::ParameterDecoding {
                    format: eredu_checkpoint::LinearFormat::GgufIQuant { .. },
                    ..
                }
            )
        ) {
            return super::parameter_decode::gguf::emit_host(operation, sink);
        }
        if matches!(
            operation.kind,
            WorkspaceOperationKindView::GroupSelection { .. }
        ) {
            // The same intervention worker constructs correction/keep vectors
            // on either device; native tensor backing remains separately priced.
            return routing::emit_selector_host(operation, self.allocation, sink);
        }
        if matches!(
            operation.kind,
            WorkspaceOperationKindView::Sampling(
                WorkspaceSamplingOperation::TokenFilter
                    | WorkspaceSamplingOperation::OptionalTokenFilter
                    | WorkspaceSamplingOperation::Penalties { .. }
            )
        ) {
            return sampling::emit_host(operation, sink);
        }
        sink.finish(0, format_args!("CPU equation uses shared native backing; fixed SIMD tiles and alias/worker controls are independently priced by the same CPU source plan"))
            .map(Some)
    }
}
impl MlxCpuWorkspaceMechanisms {
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
        let Some(plan) = self.plan(operation)? else {
            return Ok(None);
        };
        let births = plan
            .population
            .births
            .checked_add(plan.seeds)
            .and_then(|births| births.checked_sub(emitted.allocated_output_births()))
            .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?;
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
}
fn host_array(
    operation: WorkspaceOperationView<'_>,
    mechanism: MlxCpuWorkspaceMechanisms,
) -> facts::FactResult<Option<OperationPlan>> {
    let generated = if matches!(
        operation.kind,
        WorkspaceOperationKindView::GeneratedF32Initialization
    ) {
        Some(basic::generated_f32_plan_view(operation)?)
    } else {
        None
    };
    let slice = super::host_array::slice_dtype(operation);
    let Some((_, floating)) = slice.or_else(|| super::host_array::dtype(operation)) else {
        return Ok(None);
    };
    let output = operation
        .outputs
        .get(0)
        .expect("validated host-array output");
    if output.elements()? > i32::MAX as u64 {
        return Ok(None);
    }
    let mut population = CpuPopulation::default();
    if slice.is_some() {
        let Some(copy) = OperationEvent::cpu_copy_alias_layout(output.shape().len(), false) else {
            return Ok(None);
        };
        if copy.backing_births() != 0 || population.copy(copy, 1).is_none() {
            return Ok(None);
        }
    }
    let frames = [
        generated
            .map_or(Some(0), |plan| {
                plan.worker_control_bytes::<crate::MlxTensor>()
            })
            .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?,
        size_of::<Option<eredu_nn::F32InitializationPlan<'_>>>(),
        (if slice.is_some() {
            super::host_array::slice_control_bytes()
        } else {
            super::host_array::control_bytes()
        })
        .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?,
        size_of::<WorkspaceOperationView<'_>>(),
        size_of::<WorkspaceLayoutView<'_>>(),
        size_of::<MlxCpuWorkspaceMechanisms>(),
        size_of::<CpuPopulation>(),
        size_of::<OperationPlan>(),
        size_of::<facts::FactResult<Option<OperationPlan>>>(),
        size_of::<Option<(safemlx::Dtype, Option<WorkspaceFloatingType>)>>(),
        size_of::<Option<CpuCopyEvalLayout>>(),
        size_of::<CpuCopyEvalLayout>(),
    ];
    population.controls = frames
        .into_iter()
        .try_fold(
            population
                .controls
                .checked_add(size_of_val(&frames))
                .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?,
            usize::checked_add,
        )
        .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?;
    Ok(Some(OperationPlan {
        dtype: floating.unwrap_or(WorkspaceFloatingType::Float32),
        population,
        alias_input: None,
        output_bytes: mechanism
            .allocation
            .fixed_buffer_capacity(output.bytes()?)?,
        scratch_bytes: 0,
        rank: output.shape().len(),
        parameter_shells: 0,
        seeds: 1,
        validations: 0,
    }))
}

impl WorkspaceMechanisms for MlxCpuWorkspaceMechanisms {
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
        if basic::is_unattributed_initializer(operation) {
            None
        } else if self.allocation.original_storage {
            crate::backend::managed_memory::cold_original_allocator_placement()
        } else {
            crate::backend::managed_memory::cold_default_placement()
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
        } else {
            crate::backend::managed_memory::cold_default_placement()
        }
    }
    fn grouped_observation_schedule(
        &self,
        bank: &WorkspaceGroupedBank,
        tokens: u32,
    ) -> Result<Option<WorkspaceGroupedObservationSchedule>, Error> {
        super::grouped::observation_schedule(bank, tokens)
    }

    fn output_representation(
        &self,
        operation: WorkspaceOperationView<'_>,
        output: usize,
    ) -> Option<WorkspaceRepresentation> {
        if output != 0
            && !matches!(
                operation.kind,
                WorkspaceOperationKindView::CandidateExtraction { .. }
                    | WorkspaceOperationKindView::PreparedMultiAxisRotary(_)
                    | WorkspaceOperationKindView::Elementwise("indexed_bank_discovery")
                    | WorkspaceOperationKindView::BlockwiseAttention { .. }
                    | WorkspaceOperationKindView::Grouped { .. }
                    | WorkspaceOperationKindView::GroupSelection { .. }
                    | WorkspaceOperationKindView::JointGroupSelection(_)
            )
        {
            return None;
        }
        let plan = match self.plan(operation) {
            Ok(Some(plan)) => plan,
            _ => return None,
        };
        if matches!(
            operation.kind,
            WorkspaceOperationKindView::BlockwiseAttention { .. }
        ) {
            return blockwise::representation(operation, output);
        }
        if matches!(
            operation.kind,
            WorkspaceOperationKindView::GroupSelection { .. }
                | WorkspaceOperationKindView::JointGroupSelection(_)
        ) {
            return router::representation(operation, output);
        }
        if matches!(operation.kind,WorkspaceOperationKindView::View(name) if super::byte_view::selected(name).is_some())
        {
            return byte_view::representation(operation);
        }
        if matches!(operation.kind, WorkspaceOperationKindView::Index { .. }) {
            return static_index::representation(operation);
        }
        if let Some(input) = plan.alias_input {
            let representation = operation.inputs.get(input)?.representation()?;
            return Some(
                if matches!(operation.kind, WorkspaceOperationKindView::Transpose(_)) {
                    views::transpose_representation(operation, representation)
                } else if matches!(
                    operation.kind,
                    WorkspaceOperationKindView::View("squeeze" | "expand_dims")
                ) {
                    views::unit_axis_representation(operation, representation)
                } else if matches!(operation.kind, WorkspaceOperationKindView::View("reshape"))
                    && !representation.row_contiguous()
                {
                    views::reshape_representation(operation, representation)
                } else if matches!(
                    operation.kind,
                    WorkspaceOperationKindView::StaticSlice { .. }
                ) {
                    static_slice::representation(operation, representation)
                } else {
                    representation
                },
            );
        }
        if views::reshape_may_alias(operation) {
            // The actual branch may preserve gapped rows or create a dense
            // destination. Only their common scalar fact is retained.
            return Some(WorkspaceRepresentation::new(plan.dtype, false));
        }
        // Integer/Bool source plans carry physical envelopes without claiming
        // a floating output representation for their exact declared dtype.
        (operation.outputs.get(output)?.dtype() == WorkspaceDtype::Float32)
            .then_some(WorkspaceRepresentation::new(plan.dtype, true))
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
    fn host_workspace_bound(
        &self,
        operation: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceHostBound>, Error> {
        facts::ordinary_host_with(
            |sink| self.host(operation.as_view(), sink),
            |error| ordinary_error(operation, error),
        )
    }
}
impl WorkspaceFactMechanisms for MlxCpuWorkspaceMechanisms {
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
        self.scratch_controls(operation)
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
        } else {
            crate::backend::managed_memory::cold_default_placement()
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
        } else {
            crate::backend::managed_memory::cold_default_placement()
        }
    }
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
        self.host(operation, &mut facts::HostEmitter::count())
    }
    fn write_host_facts(
        &self,
        operation: WorkspaceOperationView<'_>,
        destination: WorkspaceHostDestination<'_>,
    ) -> Result<Option<WorkspaceHostFacts>, Self::Error> {
        facts::write_host(|sink| self.host(operation, sink), destination)
    }
}

/// Additive frontend construction and actual CPU Eval populations, excluding
/// the final Synchronizer. Every source contributes its owning bank once per
/// quoted completion, including possible worker scratch; no donation or
/// early-retirement credit is used.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct CpuPopulation {
    /// Frontend constructor candidates, including identity calls whose temporary
    /// vectors still use the graph bank although no Eval is constructed.
    pub(super) construction_entries: usize,
    /// Actual native Eval producers used by tape and traversal storage.
    pub(super) primitives: usize,
    pub(super) input_edges: usize,
    /// Retained native operands hidden inside a composite equation. These
    /// extend traversal storage, but are never new backing births or seeds.
    pub(super) hidden_leaves: usize,
    pub(super) maximum_operands: usize,
    /// Peak Data-publication captures inside one actual native Eval. The
    /// record clears these between primitives; repeated Evals reuse capacity.
    pub(super) maximum_captures: usize,
    pub(super) births: usize,
    pub(super) extents: usize,
    pub(super) controls: usize,
}
impl CpuPopulation {
    pub(super) fn copy(&mut self, source: CpuCopyEvalLayout, inputs: usize) -> Option<()> {
        if source.signal_graph_allocation_extents() != 0 {
            return None;
        }
        self.add(Self {
            construction_entries: 1,
            primitives: 1,
            input_edges: inputs,
            hidden_leaves: 0,
            maximum_operands: inputs,
            maximum_captures: 0,
            births: source.backing_births(),
            extents: source
                .graph_allocation_extents()
                .checked_add(source.worker_graph_allocation_extents())?,
            controls: source.control_bytes()?,
        })
    }
    fn concatenate(&mut self, source: CpuCopyEvalLayout, inputs: usize) -> Option<()> {
        if inputs < 2 || source.backing_births() > 1 {
            return None;
        }
        // Concatenate::eval_cpu publishes one output Data, including an empty
        // output with no physical birth, then each unchanged
        // copy_cpu_inplace job makes two unsafe_weak_copy Data publications.
        // out_slice.copy_shared_buffer aliases directly and adds no capture.
        let captures = inputs.checked_mul(2)?.checked_add(1)?;
        let controls = size_of::<(
            &mut Self,
            CpuCopyEvalLayout,
            usize,
            usize,
            Option<usize>,
            Option<()>,
        )>();
        self.copy(source, inputs)?;
        self.maximum_captures = self.maximum_captures.max(captures);
        self.controls = self.controls.checked_add(controls)?;
        Some(())
    }
    fn binary(&mut self, source: safemlx::CpuBinaryEvalLayout) -> Option<()> {
        self.add(Self {
            construction_entries: 1,
            primitives: 1,
            input_edges: 2,
            hidden_leaves: 0,
            maximum_operands: 2,
            maximum_captures: 0,
            births: source.backing_births(),
            extents: source
                .graph_allocation_extents()
                .checked_add(source.worker_graph_allocation_extents())?,
            controls: source.control_bytes()?,
        })
    }
    fn unary(&mut self, source: safemlx::CpuUnaryEvalLayout) -> Option<()> {
        self.add(Self {
            construction_entries: 1,
            primitives: 1,
            input_edges: 1,
            hidden_leaves: 0,
            maximum_operands: 1,
            maximum_captures: 0,
            births: source.backing_births(),
            extents: source
                .graph_allocation_extents()
                .checked_add(source.worker_graph_allocation_extents())?,
            controls: source.control_bytes()?,
        })
    }
    pub(super) fn add(&mut self, other: Self) -> Option<()> {
        let combined = Self {
            construction_entries: self
                .construction_entries
                .checked_add(other.construction_entries)?,
            primitives: self.primitives.checked_add(other.primitives)?,
            input_edges: self.input_edges.checked_add(other.input_edges)?,
            hidden_leaves: self.hidden_leaves.checked_add(other.hidden_leaves)?,
            maximum_operands: self.maximum_operands.max(other.maximum_operands),
            maximum_captures: self.maximum_captures.max(other.maximum_captures),
            births: self.births.checked_add(other.births)?,
            extents: self.extents.checked_add(other.extents)?,
            controls: self.controls.checked_add(other.controls)?,
        };
        *self = combined;
        Some(())
    }
}
#[derive(Clone, Copy, Debug)]
pub(super) struct OperationPlan {
    pub(super) dtype: WorkspaceFloatingType,
    pub(super) population: CpuPopulation,
    /// Existing source alias or independent allocated result. Native source
    /// validation still authenticates the actual backing at execution.
    pub(super) alias_input: Option<usize>,
    pub(super) output_bytes: u64,
    pub(super) scratch_bytes: u64,
    pub(super) rank: usize,
    pub(super) parameter_shells: usize,
    pub(super) seeds: usize,
    pub(super) validations: usize,
}

/// Existing non-numerical value worker: no output array and no new arithmetic
/// primitive. Return its exact fixed controls and actual nested frontier count.
pub(super) fn value_frontier(operation: WorkspaceOperationView<'_>) -> Option<(usize, usize)> {
    if operation.inputs.is_empty() || !operation.outputs.is_empty() {
        return None;
    }
    let (controls, nested) = match operation.kind {
        WorkspaceOperationKindView::ValueCompletion => (
            crate::backend::nn::shared::value_completion_control_bytes(operation.inputs.len())?,
            1,
        ),
        WorkspaceOperationKindView::ValueRetention => (
            crate::backend::submission_recovery::prefill::TransientRootsProjection::control_bytes(
            )?
            .checked_mul(operation.inputs.len())?,
            0,
        ),
        _ => return None,
    };
    let frames = [
        size_of::<WorkspaceOperationView<'_>>(),
        size_of::<Option<(usize, usize)>>(),
        size_of::<(usize, usize)>(),
        size_of::<usize>() * 2,
    ];
    Some((
        frames.into_iter().try_fold(
            controls.checked_add(size_of_val(&frames))?,
            usize::checked_add,
        )?,
        nested,
    ))
}
