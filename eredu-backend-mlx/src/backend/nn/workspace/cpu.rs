//! CPU model facts consume the selected source-visible worker, never GPU bounds.
use super::*;
use safemlx::{CpuBinaryOperation, CpuCopyEvalLayout, OperationEvent};
use std::mem::{size_of, size_of_val};
mod dense;
mod storage_copy;
mod pending_input;
mod pointwise;
mod integer_pointwise;
mod integer_storage;
mod embedding;
mod normalization;
mod activation;
mod gated;
mod views;
mod byte_view;
mod byte_frame;
mod zero_fill;
mod concatenate;
mod capture_cast;
mod classification;
mod strided_capture;
mod flat_reduction;
mod summary_selection;
mod comparison;
mod token_scores;
mod candidates;
mod static_slice;
mod static_update;
mod indexed_elements;
mod gelu;
mod rope;
mod attention;
mod causal_mask;
mod masked_readout;
mod numerical_random;
mod numerical_difference;
mod numerical_mask;
mod sampling_filters;
mod sampling_state;
pub(super) use numerical_difference::{difference as difference_population,logarithm as logarithm_population};
pub(super) use numerical_random::{uniform_unit_interval as uniform_population,categorical as categorical_population};

/// Cold CPU facts paired with their exact recipe selection. The allocator is
/// shared with the Metal-backed native runtime; operator facts are CPU-specific.
#[derive(Clone, Copy, Debug)]
pub(crate) struct MlxCpuWorkspaceMechanisms {
    allocation: MetalAllocationFacts,
    matmul: MlxCpuMatmulMechanism,
}
impl MlxCpuWorkspaceMechanisms {
    pub(crate) const fn new(
        allocation: MetalAllocationFacts,
        matmul: MlxCpuMatmulMechanism,
    ) -> Self {
        Self { allocation, matmul }
    }
    pub(super) fn plan(
        self,
        operation: WorkspaceOperationView<'_>,
    ) -> facts::FactResult<Option<OperationPlan>> {
        if let Some(plan)=storage_copy::inspect(operation,self)? {return Ok(Some(plan));}
        if let Some(plan)=pending_input::inspect(operation,self)? {return Ok(Some(plan));}
        if let Some(plan)=zero_fill::inspect(operation,self)? {return Ok(Some(plan));}
        if let Some(plan)=byte_frame::inspect(operation,self)? {return Ok(Some(plan));}
        if let Some(plan) = dense::inspect(operation, self)? {
            return Ok(Some(plan));
        }
        if let Some(plan)=integer_pointwise::inspect(operation,self)? {return Ok(Some(plan));}
        if let Some(plan)=integer_storage::inspect(operation,self)? {return Ok(Some(plan));}
        if let Some(plan) = pointwise::inspect(operation, self)? {
            return Ok(Some(plan));
        }
        if let Some(plan) = embedding::inspect_token_validation(operation, self)? {
            return Ok(Some(plan));
        }
        if let Some(plan) = embedding::inspect(operation, self)? {
            return Ok(Some(plan));
        }
        if let Some(plan) = normalization::inspect(operation, self)? {
            return Ok(Some(plan));
        }
        if let Some(plan) = activation::inspect(operation, self)? {
            return Ok(Some(plan));
        }
        if let Some(plan) = gated::inspect(operation, self)? {
            return Ok(Some(plan));
        }
        if let Some(plan)=byte_view::inspect(operation,self)? {return Ok(Some(plan));}
        if let Some(plan) = views::inspect(operation, self)? {
            return Ok(Some(plan));
        }
        if let Some(plan)=concatenate::inspect(operation,self)? {return Ok(Some(plan));}
        if let Some(plan)=capture_cast::inspect(operation,self)? {return Ok(Some(plan));}
        if let Some(plan)=classification::inspect(operation,self)? {return Ok(Some(plan));}
        if let Some(plan)=flat_reduction::inspect(operation,self)? {return Ok(Some(plan));}
        if let Some(plan)=summary_selection::inspect(operation,self)? {return Ok(Some(plan));}
        if let Some(plan)=comparison::inspect(operation,self)? {return Ok(Some(plan));}
        if let Some(plan)=token_scores::inspect(operation,self)? {return Ok(Some(plan));}
        if let Some(plan)=candidates::inspect(operation,self)? {return Ok(Some(plan));}
        if let Some(plan)=static_slice::inspect(operation)? {return Ok(Some(plan));}
        if let Some(plan)=static_update::inspect(operation,self)? {return Ok(Some(plan));}
        if let Some(plan)=indexed_elements::inspect(operation,self)? {return Ok(Some(plan));}
        if let Some(plan)=gelu::inspect(operation,self)? {return Ok(Some(plan));}
        if let Some(plan)=rope::inspect(operation,self)? {return Ok(Some(plan));}
        if let Some(plan)=attention::inspect(operation,self)? {return Ok(Some(plan));}
        if let Some(plan)=causal_mask::inspect(operation,self)? {return Ok(Some(plan));}
        if let Some(plan)=numerical_mask::inspect(operation,self)? {return Ok(Some(plan));}
        if let Some(plan)=sampling_filters::inspect(operation,self)? {return Ok(Some(plan));}
        if let Some(plan)=sampling_state::inspect(operation,self)? {return Ok(Some(plan));}
        masked_readout::inspect(operation,self)
    }
    fn emit(
        self,
        operation: WorkspaceOperationView<'_>,
        sink: &mut facts::Emitter<'_>,
    ) -> facts::FactResult<Option<WorkspaceOperationFacts>> {
        if matches!(operation.kind,WorkspaceOperationKindView::ValueCompletion|WorkspaceOperationKindView::ValueRetention) {
            if value_frontier(operation).is_none(){return Ok(None);}
            return basic::emit(operation,self.allocation,sink);
        }
        if basic::is_prepared_token_input(operation) {
            sink.output(facts::Output::Allocate(self.allocation.fixed_buffer_capacity(operation.outputs.get(0).expect("validated prepared integer matrix").bytes()?)?))?;
            return sink.finish(0, format_args!("one integer matrix from the separately admitted prepared text input; native input/source controls remain external to the model equation")).map(Some);
        }
        let Some(plan) = self.plan(operation)? else {
            return Ok(None);
        };
        if matches!(operation.kind, WorkspaceOperationKindView::Contiguous) {
            // Row layout alone does not prove a shape-sized backing. Native
            // Contiguous may reuse the exact input or copy its logical extent.
            sink.output(facts::Output::AllocateOrAliasInputs {
                bytes: plan.output_bytes, inputs: facts::Aliases::Slice(&[0]),
            })?;
        } else if matches!(operation.kind, WorkspaceOperationKindView::CandidateExtraction { .. }) {
            // The composite's ID and score outputs own distinct destination
            // envelopes. The full sort backing is separately retained in scratch.
            for output in operation.outputs.iter() {
                sink.output(facts::Output::Allocate(self.allocation.fixed_buffer_capacity(output.bytes()?)?))?;
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
        if matches!(operation.kind,WorkspaceOperationKindView::ValueCompletion|WorkspaceOperationKindView::ValueRetention) {
            if value_frontier(operation).is_none(){return Ok(None);}
            return sink.finish(0,format_args!("borrowed value completion/retention has no host numerical payload; actual CPU root and traversal controls are separately priced")).map(Some);
        }
        if basic::is_prepared_token_input(operation) {
            return sink.finish(0, format_args!("prepared token payload is owned and priced by the existing input producer; model equations borrow it")).map(Some);
        }
        if self.plan(operation)?.is_none() {
            return Ok(None);
        }
        if matches!(operation.kind,WorkspaceOperationKindView::Sampling(WorkspaceSamplingOperation::TokenFilter)) {
            return sampling::emit_host(operation,sink);
        }
        sink.finish(0, format_args!("CPU equation uses shared native backing; fixed SIMD tiles and alias/worker controls are independently priced by the same CPU source plan"))
            .map(Some)
    }
}
impl WorkspaceMechanisms for MlxCpuWorkspaceMechanisms {
    fn output_representation(
        &self,
        operation: WorkspaceOperationView<'_>,
        output: usize,
    ) -> Option<WorkspaceRepresentation> {
        if output != 0 && !matches!(operation.kind, WorkspaceOperationKindView::CandidateExtraction { .. }) {
            return None;
        }
        let plan = match self.plan(operation) {
            Ok(Some(plan)) => plan,
            refused => {
                // Temporary opt-in attribution for a floating source which
                // becomes Unknown before a collective can finish its trace.
                if operation.outputs.get(output).is_some_and(|value| value.dtype() == WorkspaceDtype::Float32)
                    && std::env::var_os("EREDU_CPU_WORKSPACE_SOURCE_TRACE").is_some() {
                    eprintln!("CPU_WORKSPACE_SOURCE_REFUSAL kind={:?} error={:?}", operation.kind, refused.err());
                    for (index, input) in operation.inputs.iter().enumerate() {
                        eprintln!("CPU_WORKSPACE_SOURCE_INPUT index={index} shape={:?} dtype={:?} representation={:?}",
                            input.shape(), input.dtype(), input.representation());
                    }
                    if let Some(value) = operation.outputs.get(output) {
                        eprintln!("CPU_WORKSPACE_SOURCE_OUTPUT shape={:?} dtype={:?} representation={:?}",
                            value.shape(), value.dtype(), value.representation());
                    }
                }
                return None;
            }
        };
        if matches!(operation.kind,WorkspaceOperationKindView::View(name) if super::byte_view::selected(name).is_some()) {
            return byte_view::representation(operation);
        }
        if let Some(input)=plan.alias_input {
            let representation=operation.inputs.get(input)?.representation()?;
            return Some(if matches!(operation.kind,WorkspaceOperationKindView::View("transpose")|WorkspaceOperationKindView::Transpose(_)) {
                views::transpose_representation(operation,representation)
            } else if matches!(operation.kind,WorkspaceOperationKindView::View("reshape"))
                && !representation.row_contiguous() {
                views::reshape_representation(operation,representation)
            } else if matches!(operation.kind,WorkspaceOperationKindView::StaticSlice {..}) {
                static_slice::representation(operation,representation)
            } else {representation});
        }
        // Integer/Bool source plans carry physical envelopes without claiming
        // a floating output representation for their exact declared dtype.
        (operation.outputs.get(output)?.dtype()==WorkspaceDtype::Float32)
            .then_some(WorkspaceRepresentation::new(plan.dtype,true))
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

/// Additive actual CPU Eval producers excluding the final Synchronizer. Every
/// source contributes its owning bank once per quoted completion, including
/// possible worker scratch; no donation or early-retirement credit is used.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct CpuPopulation {
    pub(super) primitives: usize,
    pub(super) input_edges: usize,
    pub(super) births: usize,
    pub(super) extents: usize,
    pub(super) controls: usize,
}
impl CpuPopulation {
    fn copy(&mut self, source: CpuCopyEvalLayout, inputs: usize) -> Option<()> {
        if source.signal_graph_allocation_extents() != 0 {
            return None;
        }
        self.add(Self {
            primitives: 1,
            input_edges: inputs,
            births: source.backing_births(),
            extents: source
                .graph_allocation_extents()
                .checked_add(source.worker_graph_allocation_extents())?,
            controls: source.control_bytes()?,
        })
    }
    fn binary(&mut self, source: safemlx::CpuBinaryEvalLayout) -> Option<()> {
        self.add(Self {
            primitives: 1,
            input_edges: 2,
            births: source.backing_births(),
            extents: source
                .graph_allocation_extents()
                .checked_add(source.worker_graph_allocation_extents())?,
            controls: source.control_bytes()?,
        })
    }
    fn unary(&mut self, source: safemlx::CpuUnaryEvalLayout) -> Option<()> {
        self.add(Self {
            primitives: 1,
            input_edges: 1,
            births: source.backing_births(),
            extents: source
                .graph_allocation_extents()
                .checked_add(source.worker_graph_allocation_extents())?,
            controls: source.control_bytes()?,
        })
    }
    pub(super) fn add(&mut self, other: Self) -> Option<()> {
        let combined = Self {
            primitives: self.primitives.checked_add(other.primitives)?,
            input_edges: self.input_edges.checked_add(other.input_edges)?,
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
pub(super) fn value_frontier(operation:WorkspaceOperationView<'_>)->Option<(usize,usize)> {
    if operation.inputs.is_empty()||!operation.outputs.is_empty(){return None;}
    let (controls,nested)=match operation.kind {
        WorkspaceOperationKindView::ValueCompletion=>(crate::backend::nn::shared::value_completion_control_bytes(operation.inputs.len())?,1),
        WorkspaceOperationKindView::ValueRetention=>(crate::backend::submission_recovery::prefill::TransientRootsProjection::control_bytes()?
            .checked_mul(operation.inputs.len())?,0),
        _=>return None,
    };
    let frames=[size_of::<WorkspaceOperationView<'_>>(),size_of::<Option<(usize,usize)>>(),
        size_of::<(usize,usize)>(),size_of::<usize>()*2];
    Some((frames.into_iter().try_fold(controls.checked_add(size_of_val(&frames))?,usize::checked_add)?,nested))
}
