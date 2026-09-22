//! Selected resident equations reduced before their ordinary trace retires.
//! Missing lowerings remain explicit; an arena cap supplies no producer fact.

use super::*;
mod addressable_program;
use crate::backend::nn::tensor::GroupedOutputStorage;
use eredu_architectures::prepared_execution::InferenceEquationTraceObserver;
use eredu_core::InferenceGeometry;
use eredu_nn::workspace::WorkspaceStoragePopulation;
use eredu_runtime::working_memory::{InferenceSpanWorkspacePlan, InferenceWorkspaceSpan};
mod activation;
mod attention_direct;
mod attention_tiled;
// These are the fixed frames of the same Rust accumulator, independent of its
// selected numerical backend. CPU primitives retain their own query census.
pub(in crate::backend::nn::workspace) fn blockwise_control_bytes(
    operation: WorkspaceOperationView<'_>,
) -> Option<usize> {
    matches!(
        operation.kind,
        WorkspaceOperationKindView::BlockwiseAttention { .. }
    )
    .then(|| attention_tiled::control_bytes(operation))
    .flatten()
}
mod capture;
mod clip;
mod copy_rank;
mod embedding;
mod fp8;
mod gather;
mod gather_pooled_mask;
mod graph_capacity;
mod grouped_packed;
mod grouped_reduction;
mod initialized_inputs;
pub(super) fn mxfp4_sources(operation: WorkspaceOperationView<'_>) -> Option<usize> {
    grouped_packed::mxfp4_sources(operation)
}
pub(super) fn affine_sources(operation: WorkspaceOperationView<'_>) -> Option<usize> {
    grouped_packed::affine_sources(operation)
}
mod isolated_copy;
pub(crate) use isolated_copy::IsolatedCopyNativeLayout;
pub(crate) use numerical::{
    AddressableNumericalPopulation, CpuCaptureLoan, OrdinaryCpuPopulation,
    SpeculativeNumericalRecipe,
};
mod convolution;
mod cpu;
mod embedded;
mod external;
mod parallel;
mod parallel_population;
pub(crate) use parallel::ParallelRecipeRecorder;
mod host_copies;
mod hyper;
mod indexed_attention;
mod joint_selection;
mod masked_readout;
mod neural_boundaries;
mod numerical;
mod ordinary;
pub(crate) use ordinary::{OrdinaryIndexedPrograms, OrdinaryNativeControls};
#[cfg(all(
    test,
    target_vendor = "apple",
    feature = "metal",
    not(feature = "cuda")
))]
pub(super) mod original_component_tests;
mod parameter_construction;
mod pooled_attention;
mod pooled_positions;
mod pooling_mask;
mod prepared_rotary;
mod record_capacity;
mod relative_attention;
mod rotary;
mod router;
mod sampling_filters;
mod sampling_mirostat;
mod sampling_penalties;
mod sampling_program;
mod sampling_random;
#[cfg(test)]
mod stream_tests;
pub(crate) use sampling_program::ResidentSamplingProgram;
#[cfg(all(
    test,
    target_vendor = "apple",
    feature = "metal",
    not(feature = "cuda")
))]
mod sampling_validation_tests;
mod selective_scan;
mod speculative;
pub(crate) use embedded::EmbeddedEquationRecipe;
mod speculative_io;
pub(crate) use speculative::AutoregressiveEquationRecipe;
pub(crate) use speculative_io::AutoregressiveReadoutRecipe;

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct CertifiedSpanStorage {
    mutable_bytes: u64,
    maximum_births: usize,
}
impl CertifiedSpanStorage {
    pub(crate) fn checked_add(self, bytes: u64, births: usize) -> Option<Self> {
        Some(Self {
            mutable_bytes: self.mutable_bytes.checked_add(bytes)?,
            maximum_births: self.maximum_births.checked_add(births)?,
        })
    }
    pub(crate) fn union(self, other: Self) -> Self {
        Self {
            mutable_bytes: self.mutable_bytes.max(other.mutable_bytes),
            maximum_births: self.maximum_births.max(other.maximum_births),
        }
    }
    pub(crate) fn mutable_bytes(self) -> u64 {
        self.mutable_bytes
    }
    pub(crate) fn maximum_births(self) -> usize {
        self.maximum_births
    }
}
/// Actual device population; absence never supplies a complete Graph bound.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ResidentDispatchPopulation {
    cpu_model: Option<super::cpu::CpuPopulation>,
    gpu_entries: usize,
    gpu_input_edges: usize,
    gpu_siblings: usize,
    gpu_births: usize,
    additional_sort_kernels: usize,
    cpu_entries: usize,
    cpu_input_edges: usize,
    cpu_siblings: usize,
    parallel_entries: usize,
    parallel_graph_extents: usize,
    worker_graph_extents: usize,
    // Ordinary numerical-worker rank is separate from descriptor/host shape rank.
    worker_rank: usize,
    // Additive actual reshape/concat metadata, retained across DAG re-queries.
    copy_rank_extents: usize,
    kernel_attempts: usize,
}
impl ResidentDispatchPopulation {
    /// Count the native sources, not device kinds. A model, the CPU router,
    /// and CPU collectives use independently retained streams. The homogeneous
    /// CPU model has no crossed router; its collective stream is still distinct.
    fn completion_streams(&self) -> Option<usize> {
        if let Some(source) = self.cpu_model.as_ref() {
            if self.gpu_entries != 0
                || self.gpu_input_edges != 0
                || self.gpu_siblings != 0
                || self.gpu_births != 0
                || self.worker_graph_extents != 0
                || self.copy_rank_extents != 0
                || self.kernel_attempts != 0
                || self.additional_sort_kernels != 0
                || self.cpu_entries
                    != source
                        .primitives
                        .checked_add(self.parallel_entries)?
                        .checked_add(1)?
            {
                return None;
            }
            Some(1 + usize::from(self.parallel_entries != 0))
        } else {
            if self.gpu_entries == 0 {
                return None;
            }
            let routers = self.cpu_entries.checked_sub(self.parallel_entries)?;
            Some(1 + usize::from(routers != 0) + usize::from(self.parallel_entries != 0))
        }
    }
    fn completion_stream_control_bytes() -> usize {
        std::mem::size_of::<(
            &Self,
            Option<&super::cpu::CpuPopulation>,
            usize,
            usize,
            Option<usize>,
            bool,
        )>()
    }
}
#[derive(Clone, Copy, Debug)]
pub(crate) struct ResidentCompletionRecipe {
    pub(crate) validation_roots: usize,
    pub(crate) grouped_outputs: GroupedOutputStorage,
    pub(crate) traversal: safemlx::OperationEvalTraversalLayout,
    pub(crate) graph: safemlx::ResidentGraphLayout,
    dispatch: Option<ResidentDispatchPopulation>,
    pub(crate) nested_completions: usize,
    nested_root_capacity: usize,
}
impl ResidentCompletionRecipe {
    pub(crate) fn nested_traversal(self) -> Option<safemlx::OperationEvalTraversalLayout> {
        nested_traversal_with_roots(self.traversal, self.nested_root_capacity.max(3))
    }
}
fn nested_traversal(
    traversal: safemlx::OperationEvalTraversalLayout,
) -> Option<safemlx::OperationEvalTraversalLayout> {
    nested_traversal_with_roots(traversal, 3)
}
fn nested_traversal_with_roots(
    traversal: safemlx::OperationEvalTraversalLayout,
    roots: usize,
) -> Option<safemlx::OperationEvalTraversalLayout> {
    let mut limits = traversal.limits();
    // Late accepted capture callbacks can add nested completion to a row whose
    // original equation had none. Preserve every reachable descriptor and
    // primitive edge, then add the new Synchronizer-to-root frontier. A max on
    // total edges alone would erase the existing primitive-edge contribution.
    // Shrinking a root list keeps the conservative enclosing DAG population.
    let additional = roots.saturating_sub(limits.roots);
    limits.arrays = limits
        .arrays
        .checked_add(additional)?
        .max(roots.checked_add(1)?);
    limits.input_edges = limits.input_edges.checked_add(additional)?;
    limits.roots = roots;
    safemlx::OperationEvent::eval_traversal_layout(limits)
}
/// Actual nested root lists, separate from the existing fixed-width native
/// producers. Iteration allocates nothing and preserves repeated call entries.
#[derive(Clone, Copy)]
struct NestedCompletionRoots<'a> {
    fixed_attempts: usize,
    fixed_roots: usize,
    additional: &'a [usize],
}
impl<'a> NestedCompletionRoots<'a> {
    fn uniform(attempts: usize, roots: usize) -> Self {
        Self {
            fixed_attempts: attempts,
            fixed_roots: roots,
            additional: &[],
        }
    }
    fn for_row(row: &'a ResidentSpanRecipe) -> Self {
        Self {
            fixed_attempts: row.nested_completions,
            fixed_roots: row.nested_root_capacity.max(3),
            additional: row.prediction_roots.as_deref().unwrap_or(&[]),
        }
    }
    fn iter(self) -> impl Iterator<Item = usize> + Clone + 'a {
        std::iter::repeat_n(self.fixed_roots, self.fixed_attempts)
            .chain(self.additional.iter().copied())
    }
    fn count(self) -> Option<usize> {
        self.fixed_attempts.checked_add(self.additional.len())
    }
    fn total_roots(self) -> Option<usize> {
        self.additional.iter().try_fold(
            self.fixed_attempts.checked_mul(self.fixed_roots)?,
            |sum, roots| sum.checked_add(*roots),
        )
    }
}
#[derive(Debug)]
pub(crate) struct ResidentSpanRecipe {
    span: InferenceWorkspaceSpan,
    layerwise_ordinals: Option<Vec<usize>>,
    closing_state: WorkspaceStoragePopulation,
    opening_state: Option<WorkspaceStoragePopulation>,
    current_output: Option<WorkspaceStoragePopulation>,
    validation_producers: Option<CertifiedSpanStorage>,
    traversal: Option<safemlx::OperationEvalTraversalLayout>,
    mutable_storage: Option<CertifiedSpanStorage>,
    first_missing_operation: Option<usize>,
    missing_operation_detail: Option<String>,
    unqualified_kernel_owner: Option<CustomKernelOwner>,
    maximum_rank: usize,
    host_primitive_nodes: usize,
    validation_roots: usize,
    grouped_outputs: GroupedOutputStorage,
    query_controls: Option<usize>,
    graph: Option<safemlx::ResidentGraphLayout>,
    dispatch: Option<ResidentDispatchPopulation>,
    pub(crate) nested_completions: usize,
    nested_root_capacity: usize,
    capture_publications: usize,
    capture_roots: usize,
    capture_scalars: Option<Vec<Option<eredu_nn::workspace::WorkspaceFloatingType>>>,
    prediction_roots: Option<Vec<usize>>,
    parallel: Option<parallel::OriginalParallelInvocation>,
    addressable: Option<AddressableInvocation>,
    ordinary_addressable: Option<std::rc::Rc<OrdinaryAddressableProgram>>,
    ordinary_parallel: Option<super::parallel::OrdinaryParallelControls>,
    ordinary_calls: Option<OrdinaryCallControls>,
    ordinary_call_failure: Option<ordinary::OrdinaryCallFailure>,
    ordinary_paged_source: bool,
    model_controls: Option<eredu_runtime::replicated_session::SessionModelControlPlan>,
}
impl ResidentSpanRecipe {
    /// Paid descriptions retained from the same model traversal. These do not
    /// grant communication or inference authority; the real owner admits them.
    pub(crate) fn model_controls(
        &self,
    ) -> Option<&eredu_runtime::replicated_session::SessionModelControlPlan> {
        self.model_controls.as_ref()
    }
    /// Exact shared traversal visits, including spans which skip encoder units.
    pub(crate) fn layerwise_ordinals(&self) -> Option<&[usize]> {
        self.layerwise_ordinals.as_deref()
    }
    pub(crate) fn missing_operation_detail(&self) -> Option<&str> {
        self.missing_operation_detail.as_deref()
    }
    pub(crate) fn take_missing_operation_detail(&mut self) -> Option<(usize, String)> {
        Some((
            self.first_missing_operation?,
            self.missing_operation_detail.take()?,
        ))
    }
    pub(crate) fn addressable(&self) -> Option<&AddressableInvocation> {
        self.addressable.as_ref()
    }
    pub(crate) fn parallel(&self) -> Option<&parallel::OriginalParallelInvocation> {
        self.parallel.as_ref()
    }
    pub(crate) fn capture_publications(&self) -> usize {
        self.capture_publications
    }
    pub(crate) fn capture_roots(&self) -> usize {
        self.capture_roots
    }
    pub(crate) fn opening_state(&self) -> Option<WorkspaceStoragePopulation> {
        self.opening_state
    }
    pub(crate) fn closing_state(&self) -> WorkspaceStoragePopulation {
        self.closing_state
    }
    pub(crate) fn current_output(&self) -> Option<WorkspaceStoragePopulation> {
        self.current_output
    }
    pub(crate) fn validation_producers(&self) -> Option<CertifiedSpanStorage> {
        self.validation_producers
    }
    pub(crate) fn span(&self) -> &InferenceWorkspaceSpan {
        &self.span
    }
    pub(crate) fn mutable_storage(&self) -> Option<CertifiedSpanStorage> {
        self.mutable_storage
    }
    pub(crate) fn traversal(&self) -> Option<safemlx::OperationEvalTraversalLayout> {
        self.traversal
    }
    pub(crate) fn grouped_outputs(&self) -> GroupedOutputStorage {
        self.grouped_outputs
    }
    pub(crate) fn validation_roots(&self) -> Option<usize> {
        self.first_missing_operation
            .is_none()
            .then_some(self.validation_roots)
    }
    pub(crate) fn first_missing_operation(&self) -> Option<usize> {
        self.first_missing_operation
    }
}
#[derive(Clone, Copy, Debug)]
enum ResidentSamplingPreparation {
    Empty,
    EagerKey(safemlx::ResidentGraphLayout),
}
#[derive(Debug)]
pub(crate) struct ResidentSamplingRecipe {
    closing_roots: Option<WorkspaceStoragePopulation>,
    phase: eredu_runtime::working_memory::SamplingWorkspacePhase,
    preparation: Option<ResidentSamplingPreparation>,
    first_missing_operation: Option<usize>,
    missing_operation_detail: Option<String>,
    mutable_storage: Option<CertifiedSpanStorage>,
    completion: Option<ResidentCompletionRecipe>,
    query_controls: Option<usize>,
    ordinary_calls: Option<OrdinaryCallControls>,
    ordinary_call_failure: Option<ordinary::OrdinaryCallFailure>,
    ordinary_paged_source: bool,
}
impl ResidentSamplingRecipe {
    pub(crate) fn first_missing_operation(&self) -> Option<usize> {
        self.first_missing_operation
    }
    pub(crate) fn closing_roots(&self) -> Option<WorkspaceStoragePopulation> {
        self.closing_roots
    }
    pub(crate) fn phase(&self) -> eredu_runtime::working_memory::SamplingWorkspacePhase {
        self.phase
    }
    pub(crate) fn mutable_storage(&self) -> Option<CertifiedSpanStorage> {
        self.mutable_storage
    }
    pub(crate) fn completion(&self) -> Option<ResidentCompletionRecipe> {
        self.completion
    }
}
#[derive(Debug)]
pub(crate) struct ResidentNativeRecipe {
    pub(crate) quote_components: crate::backend::error::WorkspaceQuoteComponents,
    plan: InferenceSpanWorkspacePlan,
    records: Vec<ResidentSpanRecipe>,
    sampling: ResidentSamplingProgram,
    neural: Option<neural_boundaries::NeuralBoundaries>,
    ordinary_neural: Option<ordinary::OrdinaryNeuralCalls>,
    ordinary_model_completion_bound: bool,
    layerwise_constructors: Option<parameter_construction::ParameterConstructors>,
    host_copies: Option<host_copies::HostCopies>,
    host_transfers: Option<host_copies::HostTransfers>,
    foreground_copies: Option<host_copies::ForegroundCopies>,
    resume_copy: Option<crate::backend::array_copy::OriginalResumeCopyPopulation>,
    parallel_source: Option<parallel::OriginalParallelSource>,
    // Cold recorder destinations have already reserved this independent host
    // account. Keep it after every recipe payload; Q funds later native work.
    addressable_program: addressable_program::RetainedProgram,
    planning_metadata: Option<eredu_nn::workspace::HostMetadataFunding>,
}
impl ResidentNativeRecipe {
    pub(crate) fn ordinary_requires_paged_source(&self) -> bool {
        self.records.iter().any(|row| row.ordinary_paged_source)
            || self
                .sampling
                .rows()
                .iter()
                .any(|row| row.ordinary_paged_source)
    }
    pub(crate) fn planning_metadata(&self) -> Option<&HostMetadataFunding> {
        self.planning_metadata.as_ref()
    }
    pub(crate) fn record_quote_components(
        &mut self,
        mut value: crate::backend::error::WorkspaceQuoteComponents,
    ) -> Result<(), crate::backend::error::Error> {
        let metadata = self.planning_metadata.as_ref();
        if let Some(funding) = metadata {
            funding
                .reserve_metadata(std::mem::size_of::<(
                    &mut Self,
                    crate::backend::error::WorkspaceQuoteComponents,
                    std::slice::Iter<'_, ResidentSpanRecipe>,
                    std::slice::Iter<'_, ResidentSamplingRecipe>,
                    Option<ResidentDispatchPopulation>,
                    Option<ResidentCompletionRecipe>,
                    NestedCompletionRoots<'_>,
                    Option<usize>,
                    usize,
                    Result<(), crate::backend::error::Error>,
                )>())
                .map_err(crate::backend::error::Error::WorkspacePlanning)?;
        }
        value.kernel_attempts = self.kernel_attempts();
        value.equation_rows = self.records.len();
        value.gpu_entries = Some(0);
        value.cpu_entries = Some(0);
        value.nested_frontiers = Some(0);
        for row in &self.records {
            value.mixed_rows += usize::from(
                row.dispatch
                    .is_some_and(|dispatch| dispatch.cpu_entries != 0),
            );
            value.gpu_entries = value
                .gpu_entries
                .and_then(|total| total.checked_add(row.dispatch?.gpu_entries));
            value.cpu_entries = value
                .cpu_entries
                .and_then(|total| total.checked_add(row.dispatch?.cpu_entries));
            value.nested_frontiers = value
                .nested_frontiers
                .and_then(|total| total.checked_add(NestedCompletionRoots::for_row(row).count()?));
        }
        value.sampling_gpu_entries = Some(0);
        value.sampling_cpu_entries = Some(0);
        value.sampling_nested_frontiers = Some(0);
        for row in self.sampling.rows() {
            if let Some(completion) = row.completion() {
                value.sampling_gpu_entries = value
                    .sampling_gpu_entries
                    .and_then(|total| total.checked_add(completion.dispatch?.gpu_entries));
                value.sampling_cpu_entries = value
                    .sampling_cpu_entries
                    .and_then(|total| total.checked_add(completion.dispatch?.cpu_entries));
                value.sampling_nested_frontiers = value
                    .sampling_nested_frontiers
                    .and_then(|total| total.checked_add(completion.nested_completions));
            }
        }
        self.quote_components = value;
        Ok(())
    }
    pub(crate) fn bind_resume_copy(
        &mut self,
        population: crate::backend::array_copy::OriginalResumeCopyPopulation,
    ) -> Result<(), Error> {
        if self.resume_copy.is_some()
            || self.plan.geometry().input_positions != population.pending_positions()
            || (self.plan.geometry().prefill_chunk_positions == 0
                && !(self.plan.geometry().input_positions == 0
                    && self.plan.geometry().max_output_tokens == 0
                    && self.plan.geometry().output == eredu_core::OutputDemand::StateOnly))
            || self.plan.geometry().prefill_chunk_positions > population.pending_positions()
        {
            return Err(Error::backend_retained_source(
                eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
            ));
        }
        self.resume_copy = Some(population);
        Ok(())
    }
    // A completed copy-only restore has a fully known empty equation list.
    // It still owns its actual copy and sampler-preparation programs; absence
    // of model rows does not make those independently counted programs unknown.
    fn is_terminal_resume(&self) -> bool {
        let geometry = self.plan.geometry();
        self.resume_copy.is_some()
            && self.records.is_empty()
            && geometry.input_positions == 0
            && geometry.max_output_tokens == 0
            && geometry.prefill_chunk_positions == 0
            && geometry.output == eredu_core::OutputDemand::StateOnly
    }
    pub(crate) fn resume_copy(
        &self,
    ) -> Option<crate::backend::array_copy::OriginalResumeCopyPopulation> {
        self.resume_copy
    }
    pub(crate) fn plan(&self) -> &InferenceSpanWorkspacePlan {
        &self.plan
    }
    pub(crate) fn sampling_program(&self) -> &ResidentSamplingProgram {
        &self.sampling
    }
    pub(crate) fn sampling_records(&self) -> &[ResidentSamplingRecipe] {
        &self.sampling.rows
    }
    pub(crate) fn records(&self) -> &[ResidentSpanRecipe] {
        &self.records
    }
    /// A failed enclosing quote may move the already funded equation detail
    /// into its retained diagnostic. This never supplies a missing source.
    pub(crate) fn take_first_missing_equation(
        &mut self,
    ) -> Result<Option<(usize, usize, Option<String>)>, crate::backend::error::Error> {
        let frames = [
            size_of::<&mut Self>(),
            size_of::<std::iter::Enumerate<std::slice::IterMut<'_, ResidentSpanRecipe>>>(),
            size_of::<(usize, &mut ResidentSpanRecipe)>(),
            size_of::<Option<(usize, &mut ResidentSpanRecipe)>>(),
            size_of::<std::iter::Enumerate<std::slice::IterMut<'_, ResidentSamplingRecipe>>>(),
            size_of::<(usize, &mut ResidentSamplingRecipe)>(),
            size_of::<Option<(usize, &mut ResidentSamplingRecipe)>>(),
            size_of::<usize>(),
            size_of::<Option<(usize, usize, Option<String>)>>(),
            size_of::<Result<Option<(usize, usize, Option<String>)>, crate::backend::error::Error>>(
            ),
        ];
        if let Some(funding) = &self.planning_metadata {
            let bytes = frames
                .into_iter()
                .try_fold(size_of_val(&frames), usize::checked_add)
                .ok_or(crate::backend::error::Error::WorkspacePlanning(
                    eredu_nn::workspace::HostMetadataFundingError::Overflow,
                ))?;
            funding
                .reserve_metadata(bytes)
                .map_err(crate::backend::error::Error::WorkspacePlanning)?;
        }
        for (record, row) in self.records.iter_mut().enumerate() {
            if let Some(operation) = row.first_missing_operation {
                return Ok(Some((
                    record,
                    operation,
                    row.missing_operation_detail.take(),
                )));
            }
        }
        for (sampling, row) in self.sampling.rows.iter_mut().enumerate() {
            if let Some(operation) = row.first_missing_operation {
                let record = self.records.len().checked_add(sampling).ok_or(
                    crate::backend::error::Error::WorkspacePlanning(
                        eredu_nn::workspace::HostMetadataFundingError::Overflow,
                    ),
                )?;
                return Ok(Some((
                    record,
                    operation,
                    row.missing_operation_detail.take(),
                )));
            }
        }
        Ok(None)
    }
    /// Count every single-GPU equation/sampling DAG once. Nested completions
    /// visit subsets of those descriptors: the fixed native Eval skips any
    /// non-unscheduled input and marks each primitive and sibling evaluated
    /// before the next frontier. Its new Synchronizer has no shader lookup.
    /// The row already includes all temporary/copy/sort kernel attempts; no
    /// descriptor or request-cache row is reused to reduce that population.
    /// Mixed CPU/GPU frontiers retain the full per-completion ceiling: a fresh
    /// GPU root may first-consume an unscheduled CPU result and need another
    /// fence shader beyond the original final-root edges.
    /// An unclosed worker keeps the population absent. Observed empty sampler
    /// setup has no native worker and contributes zero.
    pub(crate) fn kernel_attempts(&self) -> Option<usize> {
        self.kernel_attempts_with_rows(&self.records)
    }
    fn kernel_attempts_with_rows(&self, records: &[ResidentSpanRecipe]) -> Option<usize> {
        let total = records.iter().try_fold(0usize, |n, row| {
            let dispatch = row.dispatch?;
            let frontiers = if dispatch.cpu_entries == 0 {
                1
            } else {
                NestedCompletionRoots::for_row(row)
                    .count()?
                    .checked_add(1)?
            };
            n.checked_add(dispatch.kernel_attempts.checked_mul(frontiers)?)
        })?;
        // Materialization creates new copy descriptors per load/transfer.
        // These are separate producers, not another visit to the equation DAG.
        let total = total.checked_add(self.host_copies.map_or(Some(0), |copies| {
            copies
                .dispatch
                .kernel_attempts
                .checked_mul(copies.per_forward)?
                .checked_mul(records.len())
        })?)?;
        let total = total.checked_add(self.host_copies.map_or(Some(0), |copies| {
            copies
                .aggregate_dispatch
                .kernel_attempts
                .checked_mul(self.host_transfers?.per_forward)?
                .checked_mul(records.len())
        })?)?;
        let total = total.checked_add(
            self.resume_copy
                .map_or(0, |copy| copy.layout().kernel_attempts),
        )?;
        total.checked_add(self.sampling.kernel_attempts()?)
    }
    /// The actual eager key constructor has no Eval, Record or shader lookup.
    pub(crate) fn sampling_preparation_graph(
        &self,
    ) -> Result<Option<safemlx::ResidentGraphLayout>, crate::backend::error::Error> {
        self.sampling
            .preparation_graph()
            .ok_or(crate::backend::error::Error::PrefillControl(
                eredu_runtime::working_memory::WorkingMemoryError::UnknownBound,
            ))
    }
    pub(crate) fn maximum_roots(&self) -> Result<u64, crate::backend::error::Error> {
        self.records
            .iter()
            .try_fold(0usize, |maximum, row| {
                row.traversal.map(|layout| maximum.max(layout.roots()))
            })
            .and_then(|n| u64::try_from(n).ok())
            .ok_or(crate::backend::error::Error::PrefillControl(
                eredu_runtime::working_memory::WorkingMemoryError::UnknownBound,
            ))
    }
    pub(crate) fn unqualified_host_owner(&self) -> Option<&'static str> {
        self.records
            .iter()
            .find_map(|row| row.unqualified_kernel_owner.map(CustomKernelOwner::reason))
    }
    pub(crate) fn control_bytes(&self) -> Result<u64, Error> {
        if let Some(reason) = self.unqualified_host_owner() {
            return Err(Error::backend(reason));
        }
        use std::mem::size_of;
        let retained = if self.planning_metadata.is_some() {
            // These exact Vec destinations and recipe shell were paid by the
            // recorder before construction. Their account is still live in
            // the pool; they are not new native-request destinations.
            0
        } else {
            self.records
                .capacity()
                .checked_mul(size_of::<ResidentSpanRecipe>())
                .and_then(|n| n.checked_add(size_of::<Self>()))
                .and_then(|n| {
                    n.checked_add(
                        self.sampling
                            .rows
                            .capacity()
                            .checked_mul(size_of::<ResidentSamplingRecipe>())?,
                    )
                })
                .and_then(|n| {
                    self.records.iter().try_fold(n, |sum, row| {
                        sum.checked_add(
                            row.layerwise_ordinals
                                .as_ref()
                                .map_or(Some(0), |ordinals| {
                                    ordinals.capacity().checked_mul(size_of::<usize>())
                                })?,
                        )
                    })
                })
                .ok_or_else(|| Error::backend("resident recipe control overflow"))?
        };
        let controls = self
            .records
            .iter()
            .try_fold(retained, |n, row| n.checked_add(row.query_controls?))
            .ok_or_else(|| Error::backend("resident recipe query controls are unqualified"))?;
        let controls = self
            .sampling
            .rows
            .iter()
            .try_fold(controls, |n, row| n.checked_add(row.query_controls?))
            .ok_or_else(|| Error::backend("resident sampling query controls are unqualified"))?;
        let controls = controls
            .checked_add(self.resume_copy.map_or(0, |copy| copy.layout().controls))
            .ok_or_else(|| Error::backend("resident preparation query control overflow"))?;
        let fixed = [
            if self.planning_metadata.is_some() {
                0
            } else {
                size_of::<ResidentRecipeRecorder>()
            },
            if self.planning_metadata.is_some() {
                0
            } else {
                size_of::<(
                    Option<WorkspaceStoragePopulation>,
                    WorkspaceStoragePopulation,
                    bool,
                )>()
            },
            size_of::<record_capacity::ResidentRecordStorage>(),
            size_of::<ResidentSpanRecipe>(),
            size_of::<ResidentSamplingRecipe>(),
            if self.planning_metadata.is_some() {
                0
            } else {
                size_of::<ReducedTrace>()
            },
            size_of::<facts::Emitter<'static>>(),
            size_of::<Lowering>(),
            size_of::<safemlx::OperationEvalTraversalLimits>(),
            size_of::<Option<safemlx::OperationEvalRecordLayout>>(),
            size_of::<Option<safemlx::OperationEvalTraversalLayout>>(),
            if self.planning_metadata.is_some() {
                0
            } else {
                size_of::<Result<Self, Error>>()
            },
        ]
        .into_iter()
        .try_fold(controls, usize::checked_add)
        .ok_or_else(|| Error::backend("resident recipe control overflow"))?;
        u64::try_from(fixed).map_err(Error::backend_retained_source)
    }
    fn parallel_control_sources(&self) -> impl Iterator<Item = &parallel::OriginalParallelSource> {
        self.parallel_source.iter().chain(
            self.records
                .iter()
                .filter_map(|row| row.parallel().map(|value| value.source())),
        )
    }
    fn parallel_source_scan_control_bytes<I: Iterator>(sources: &I) -> Option<usize> {
        std::mem::size_of_val(sources)
            .checked_add(std::mem::size_of::<Option<&parallel::OriginalParallelSource>>())?
            .checked_add(std::mem::size_of::<
                Result<Option<&parallel::OriginalParallelSource>, crate::backend::error::Error>,
            >())
    }
    /// Prospective frames of the same immutable source scan, without a debit or
    /// a source grant. An empty source range performs no funded scan.
    pub(crate) fn parallel_control_source_control_bytes(&self) -> Option<usize> {
        let mut sources = self.parallel_control_sources();
        if sources.next().is_none() {
            return Some(0);
        }
        Self::parallel_source_scan_control_bytes(&sources)
    }
    /// The same immutable model quote source for every collective span. No
    /// occurrence or control request is created by this borrowed projection.
    pub(crate) fn parallel_control_source(
        &self,
    ) -> Result<Option<&parallel::OriginalParallelSource>, crate::backend::error::Error> {
        let mut sources = self.parallel_control_sources();
        let Some(source) = sources.next() else {
            return Ok(None);
        };
        source
            .funding()
            .reserve_metadata(Self::parallel_source_scan_control_bytes(&sources).ok_or(
                crate::backend::error::Error::WorkspacePlanning(
                    eredu_nn::workspace::HostMetadataFundingError::Overflow,
                ),
            )?)
            .map_err(crate::backend::error::Error::WorkspacePlanning)?;
        if sources.any(|other| !source.same_source(other)) {
            return Err(crate::backend::error::Error::PrefillControl(
                eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
            ));
        }
        Ok(Some(source))
    }

    pub(crate) fn parallel_for_prefill(
        &self,
        role: eredu_runtime::prefill::PrefillControlRole,
    ) -> Result<Option<parallel::OriginalParallelInvocation>, crate::backend::error::Error> {
        let eredu_runtime::prefill::PrefillControlRole::Span {
            input_start,
            input_end,
            position,
            output,
            ..
        } = role
        else {
            return Err(crate::backend::error::Error::PrefillScopeUnavailable);
        };
        self.records.iter().find(|row|matches!(&row.span,InferenceWorkspaceSpan::Prefill(chunk)
            if chunk.input.start==input_start && chunk.input.end==input_end && chunk.position==position && chunk.output==output))
            .ok_or(crate::backend::error::Error::PrefillScopeUnavailable)?
            .parallel().map(parallel::OriginalParallelInvocation::try_clone_for_retention).transpose()
    }
    pub(crate) fn parallel_for_step(
        &self,
        step: &eredu_runtime::working_memory::InferenceTextStep,
    ) -> Result<Option<parallel::OriginalParallelInvocation>, crate::backend::error::Error> {
        if step.request().geometry() != self.plan.geometry() {
            return Err(crate::backend::error::Error::PrefillScopeUnavailable);
        }
        let index = step
            .attempt()
            .checked_sub(1)
            .ok_or(crate::backend::error::Error::PrefillScopeUnavailable)?;
        self.records.iter().find(|row|matches!(&row.span,InferenceWorkspaceSpan::Decode{index:actual,..} if *actual==index))
            .ok_or(crate::backend::error::Error::PrefillScopeUnavailable)?
            .parallel().map(parallel::OriginalParallelInvocation::try_clone_for_retention).transpose()
    }
    pub(crate) fn completion_for_prefill(
        &self,
        role: eredu_runtime::prefill::PrefillControlRole,
    ) -> Result<ResidentCompletionRecipe, crate::backend::error::Error> {
        let eredu_runtime::prefill::PrefillControlRole::Span {
            input_start,
            input_end,
            position,
            output,
            ..
        } = role
        else {
            return Err(crate::backend::error::Error::PrefillScopeUnavailable);
        };
        self.completion_for(|span| {
            matches!(span, InferenceWorkspaceSpan::Prefill(chunk)
            if chunk.input.start==input_start && chunk.input.end==input_end
                && chunk.position==position && chunk.output==output)
        })
    }
    /// The issuing text step, including prefill at zero, selects its sampling
    /// phase directly. Decode equation indices intentionally use attempt - 1.
    pub(crate) fn completion_for_sampling_step(
        &self,
        step: &eredu_runtime::working_memory::InferenceTextStep,
    ) -> Result<ResidentCompletionRecipe, crate::backend::error::Error> {
        if step.request().geometry() != self.plan.geometry() {
            return Err(crate::backend::error::Error::PrefillControl(
                eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
            ));
        }
        usize::try_from(step.attempt())
            .ok()
            .and_then(|index| self.sampling.completion(index))
            .ok_or(crate::backend::error::Error::PrefillScopeUnavailable)
    }
    pub(crate) fn completion_for_step(
        &self,
        step: &eredu_runtime::working_memory::InferenceTextStep,
    ) -> Result<ResidentCompletionRecipe, crate::backend::error::Error> {
        if step.request().geometry() != self.plan.geometry() {
            return Err(crate::backend::error::Error::PrefillControl(
                eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
            ));
        }
        let index = step
            .attempt()
            .checked_sub(1)
            .ok_or(crate::backend::error::Error::PrefillScopeUnavailable)?;
        self.completion_for(|span| matches!(span, InferenceWorkspaceSpan::Decode { index:actual,.. } if *actual==index))
    }
    fn completion_for(
        &self,
        matches: impl Fn(&InferenceWorkspaceSpan) -> bool,
    ) -> Result<ResidentCompletionRecipe, crate::backend::error::Error> {
        self.records
            .iter()
            .find(|row| matches(&row.span))
            .and_then(|row| self.completion_for_row(row))
            .ok_or(crate::backend::error::Error::PrefillControl(
                eredu_runtime::working_memory::WorkingMemoryError::UnknownBound,
            ))
    }
    fn completion_for_row(&self, row: &ResidentSpanRecipe) -> Option<ResidentCompletionRecipe> {
        if row.unqualified_kernel_owner.is_some() {
            return None;
        }
        Some(ResidentCompletionRecipe {
            validation_roots: row.validation_roots,
            grouped_outputs: row.grouped_outputs,
            traversal: row.traversal?,
            graph: row.graph?,
            dispatch: row.dispatch,
            nested_completions: NestedCompletionRoots::for_row(row)
                .count()?
                .checked_add(self.host_copies.map_or(0, |copies| copies.per_forward))?
                .checked_add(
                    self.host_transfers
                        .map_or(0, |transfers| transfers.per_forward),
                )?,
            nested_root_capacity: self
                .host_transfers
                .map_or(3, |transfers| transfers.roots)
                .max(row.nested_root_capacity)
                .max(
                    row.prediction_roots
                        .as_deref()
                        .unwrap_or(&[])
                        .iter()
                        .copied()
                        .max()
                        .unwrap_or(0),
                ),
        })
    }
}
pub(crate) struct ResidentRecipeRecorder {
    geometry: InferenceGeometry,
    mechanism: MlxMetalWorkspaceMechanisms,
    cpu: Option<MlxCpuWorkspaceMechanisms>,
    layerwise_constructors: Option<parameter_construction::ParameterConstructors>,
    layerwise_span_constructors:
        Option<crate::backend::runtime::execution::generic::LayerwiseConstructorTrace>,
    addressable_sources: Option<AddressableSources>,
    ordinary_addressable_sources: Option<OrdinaryAddressableSources>,
    records: Vec<ResidentSpanRecipe>,
    sampling: Vec<ResidentSamplingRecipe>,
    sampling_input: Option<eredu_runtime::working_memory::SamplingWorkspaceInputPlan>,
    prefill_validation_roots: usize,
    context: Option<eredu_nn::workspace::WorkspaceContext>,
}
impl ResidentRecipeRecorder {
    pub(crate) fn new(geometry: InferenceGeometry, mechanism: MlxMetalWorkspaceMechanisms) -> Self {
        Self {
            geometry,
            mechanism,
            cpu: None,
            layerwise_constructors: None,
            layerwise_span_constructors: None,
            addressable_sources: None,
            ordinary_addressable_sources: None,
            records: Vec::new(),
            sampling: Vec::new(),
            sampling_input: None,
            prefill_validation_roots: 0,
            context: None,
        }
    }
    /// Uses the same recorder with the caller's participating Context ledger.
    /// Cloning the Context loans its existing shared counters, not a new grant.
    pub(crate) fn with_context(
        geometry: InferenceGeometry,
        mechanism: MlxMetalWorkspaceMechanisms,
        context: &eredu_nn::workspace::WorkspaceContext,
    ) -> Result<Self, Error> {
        let controls = [
            std::mem::size_of::<Self>(),
            std::mem::size_of::<Result<Self, Error>>(),
            std::mem::size_of::<ReducedTrace>(),
            std::mem::size_of::<Result<ReducedTrace, Error>>(),
            std::mem::size_of::<(
                Option<eredu_runtime::replicated_session::SessionModelControlPlan>,
                Option<HostMetadataFunding>,
                &WorkspaceTraceReport,
            )>(),
            std::mem::size_of::<(
                Option<WorkspaceStoragePopulation>,
                WorkspaceStoragePopulation,
                bool,
            )>(),
            std::mem::size_of::<ResidentNativeRecipe>(),
            std::mem::size_of::<Result<ResidentNativeRecipe, Error>>(),
            std::mem::size_of::<(
                InferenceGeometry,
                MlxMetalWorkspaceMechanisms,
                &eredu_nn::workspace::WorkspaceContext,
            )>(),
        ];
        let bytes = controls
            .into_iter()
            .try_fold(std::mem::size_of_val(&controls), usize::checked_add)
            .ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?;
        context.charge_metadata(bytes)?;
        let mut recorder = Self::new(geometry, mechanism);
        recorder.context = Some(context.clone());
        Ok(recorder)
    }
    pub(crate) fn bind_addressable_sources(
        &mut self,
        source: AddressableSources,
    ) -> Result<(), Error> {
        let context = self
            .context
            .as_ref()
            .ok_or_else(|| self.metadata_error("addressable recorder requires original funding"))?;
        if self.addressable_sources.is_some()
            || self.ordinary_addressable_sources.is_some()
            || context
                .metadata_funding()
                .is_none_or(|funding| !funding.same_account(source.funding()))
        {
            return Err(self.metadata_error("addressable recorder source or account differs"));
        }
        context.charge_metadata(std::mem::size_of::<(AddressableSources, Result<(), Error>)>())?;
        self.addressable_sources = Some(source);
        Ok(())
    }
    /// Binds descriptive ordinary source rows under this recorder's planning
    /// account. These rows never install original arenas or execution authority.
    pub(crate) fn bind_ordinary_addressable_sources(
        &mut self,
        source: OrdinaryAddressableSources,
    ) -> Result<(), Error> {
        let context = self.context.as_ref().ok_or_else(|| {
            self.metadata_error("ordinary addressable recorder requires planning funding")
        })?;
        if self.addressable_sources.is_some()
            || self.ordinary_addressable_sources.is_some()
            || context
                .metadata_funding()
                .is_none_or(|funding| !funding.same_account(source.funding()))
        {
            return Err(self.metadata_error("ordinary addressable source or account differs"));
        }
        context.charge_metadata(std::mem::size_of::<(
            OrdinaryAddressableSources,
            Result<(), Error>,
        )>())?;
        self.ordinary_addressable_sources = Some(source);
        Ok(())
    }
    fn metadata_error(&self, message: &'static str) -> Error {
        match &self.context {
            Some(context) => context.metadata_error(format_args!("{message}")),
            None => Error::backend(message),
        }
    }
    fn metadata_source<E: std::error::Error + Send + Sync + 'static>(&self, cause: E) -> Error {
        match &self.context {
            Some(context) => context.metadata_source(cause),
            None => Error::backend_retained_source(cause),
        }
    }
    fn add(&self, a: usize, b: usize) -> Result<usize, Error> {
        a.checked_add(b)
            .ok_or_else(|| self.metadata_error("resident lowering population overflow"))
    }
    pub(crate) fn finish(
        self,
        plan: &InferenceSpanWorkspacePlan,
    ) -> Result<ResidentNativeRecipe, Error> {
        let expected_sampling = usize::try_from(self.geometry.max_output_tokens)
            .ok()
            .and_then(|n| n.checked_add(1));
        self.finish_with_sampling(plan, expected_sampling)
    }

    fn finish_with_sampling(
        mut self,
        plan: &InferenceSpanWorkspacePlan,
        expected_sampling: Option<usize>,
    ) -> Result<ResidentNativeRecipe, Error> {
        if plan.geometry() != self.geometry
            || plan.records().len() != self.records.len()
            || plan
                .records()
                .iter()
                .zip(&self.records)
                .any(|(a, b)| a.span() != b.span())
        {
            return Err(self
                .metadata_error("resident native recipe differs from its actual equation source"));
        }
        let sampling = self.take_sampling_program(expected_sampling)?;
        Ok(ResidentNativeRecipe {
            quote_components: Default::default(),
            plan: plan.clone(),
            records: self.records,
            sampling,
            neural: None,
            ordinary_neural: None,
            ordinary_model_completion_bound: false,
            layerwise_constructors: self.layerwise_constructors,
            host_copies: None,
            foreground_copies: None,
            resume_copy: None,
            parallel_source: None,
            host_transfers: None,
            addressable_program: addressable_program::RetainedProgram::default(),
            planning_metadata: self
                .context
                .as_ref()
                .and_then(|context| context.metadata_funding()),
        })
    }
}
#[derive(Clone, Copy)]
struct Lowering {
    primitives: usize,
    edges: usize,
    seeds: usize,
    validations: usize,
    maximum_births: usize,
    streams: usize,
    hidden_leaves: usize,
    maximum_operands: usize,
    intermediate_rank: usize,
    grouped_output_chunks: usize,
    grouped_output_calls: usize,
    grouped_unit_observers: usize,
    bf16_projection_calls: usize,
    pointwise_calls: usize,
    row_rms_calls: usize,
    row_sum_calls: usize,
    recurrent_calls: usize,
    router_cpu_partitions: usize,
    additional_sort_kernels: Option<usize>,
    unqualified_kernel_owner: Option<CustomKernelOwner>,
    nested_completions: usize,
    backend_shells: usize,
    helper_controls: usize,
}
impl Lowering {
    const fn plain(primitives: usize, edges: usize, seeds: usize) -> Self {
        Self {
            primitives,
            edges,
            seeds,
            validations: 0,
            maximum_births: primitives + seeds,
            streams: 1,
            hidden_leaves: 0,
            maximum_operands: 4,
            intermediate_rank: 0,
            grouped_output_chunks: 0,
            grouped_output_calls: 1,
            grouped_unit_observers: 0,
            bf16_projection_calls: 0,
            pointwise_calls: 0,
            row_rms_calls: 0,
            row_sum_calls: 0,
            recurrent_calls: 0,
            router_cpu_partitions: 0,
            additional_sort_kernels: Some(0),
            nested_completions: 0,
            backend_shells: 0,
            helper_controls: 0,
            unqualified_kernel_owner: None,
        }
    }
    const fn product(primitives: usize, edges: usize, seeds: usize, streams: usize) -> Self {
        Self {
            primitives,
            edges,
            seeds,
            validations: 0,
            streams,
            hidden_leaves: 0,
            maximum_operands: 4,
            intermediate_rank: 0,
            grouped_output_chunks: 0,
            grouped_output_calls: 1,
            grouped_unit_observers: 0,
            bf16_projection_calls: 0,
            pointwise_calls: 0,
            row_rms_calls: 0,
            row_sum_calls: 0,
            recurrent_calls: 0,
            router_cpu_partitions: 0,
            additional_sort_kernels: Some(0),
            nested_completions: 0,
            backend_shells: 0,
            helper_controls: 0,
            unqualified_kernel_owner: None,
            // At most four operand compactions and two split-K/reduction
            // intermediates, from the selected dense/quantized worker bodies.
            maximum_births: primitives + seeds + 6,
        }
    }
}
fn reduction_lowering(primitives: usize, edges: usize, seeds: usize, extra: usize) -> Lowering {
    let mut value = Lowering::plain(primitives, edges, seeds);
    // Actual worker compactions and two-pass accumulators, not arena slack.
    value.maximum_births += extra;
    value
}
fn normalization_lowering(operation: WorkspaceOperationView<'_>) -> Option<Lowering> {
    use eredu_nn::NormalizationScale;
    use WorkspaceOperationKindView as K;
    let mut value = match operation.kind {
        K::Normalization("rms", _) if operation.inputs.len() == 1 => {
            reduction_lowering(21, 24, 2, 3)
        }
        K::Normalization("rms", _) => reduction_lowering(10, 13, 1, 3),
        K::Normalization("l2", _) => reduction_lowering(15, 17, 1, 2),
        K::LayerNorm { .. } => reduction_lowering(4, 6, 2, 0),
        K::Normalization("gated_group_rms_norm" | "silu_gated_group_rms_norm", _) => {
            let mut value = reduction_lowering(42, 48, 2, 2);
            value.intermediate_rank = 3;
            value
        }
        K::ConstructedNormalization(spec) => {
            let (p, e, seeds) = match (spec.groups.is_some(), &spec.scale) {
                (false, NormalizationScale::Unit) => (21, 24, 2),
                (false, NormalizationScale::Learned(_)) => (9, 12, 1),
                (false, NormalizationScale::LearnedOffset { .. }) => (14, 18, 2),
                (true, NormalizationScale::Unit) => (25, 28, 2),
                (true, NormalizationScale::Learned(_)) => (31, 35, 2),
                (true, NormalizationScale::LearnedOffset { .. }) => (36, 41, 3),
            };
            let mut value = reduction_lowering(p, e, seeds, 3);
            if spec.groups.is_some() {
                value.intermediate_rank = operation.inputs.get(0)?.shape().len().checked_add(1)?;
            }
            value
        }
        _ => return None,
    };
    // Selected Metal fast norms and the custom/F32 fallback equations all
    // use the supplied stream. Ordinary axes/configuration owners are separate.
    value.maximum_operands = 4;
    // The selected scale/dtype may take one reproducible row RMS invocation.
    // Both unit and learned RMS call it at most once, including grouped views;
    // layer/l2 and the separate gated-group equations do not call that helper.
    value.row_rms_calls = usize::from(matches!(
        operation.kind,
        K::Normalization("rms", _) | K::ConstructedNormalization(_)
    ));
    Some(value)
}
// The metadata Float32 class also represents FP16/BF16. It is not a native
// dtype witness. Preserve the independently known graph/payload recipe while
// refusing host completeness where an unowned custom worker may execute.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CustomKernelOwner {
    Bf16Projection,
    BlockFp8,
    GroupedIndexed,
    RouterIndexed,
    Pointwise,
    RowRms,
    RowSum,
    RowSoftmax,
    RecurrentScan,
    SamplingFilters,
    SamplingMirostat,
    SamplingRandom,
}
impl CustomKernelOwner {
    const fn reason(self) -> &'static str {
        match self {
            Self::Bf16Projection => {
                "BF16 projection finite source/cache or fixed invocation is unqualified"
            }
            Self::BlockFp8 => {
                "block-FP8 finite source family or invocation controls are unqualified"
            }
            Self::GroupedIndexed => {
                "grouped sort/indexed projection selector and general scatter library owners are unqualified"
            }
            Self::RouterIndexed => "unqualified router indexed/precompiled source owner",
            Self::Pointwise => {
                "F32 pointwise custom-kernel, TLS, generated-source and library-cache owners are unqualified"
            }
            Self::RowSoftmax => {
                "explicit attention row-softmax source/cache or fixed invocation is unqualified"
            }
            Self::RowSum => "reproducible row sum source/cache or fixed invocation is unqualified",
            Self::RowRms => "reproducible RMS source/cache or fixed invocation is unqualified",
            Self::RecurrentScan => "recurrent scan source/cache or fixed invocation is unqualified",
            Self::SamplingRandom => {
                "explicit random sampling precompiled source or selector controls are unqualified"
            }
            Self::SamplingMirostat => {
                "adaptive sampling precompiled source or selector controls are unqualified"
            }
            Self::SamplingFilters => {
                "sampling filter precompiled source or selector controls are unqualified"
            }
        }
    }
}
fn pointwise_source_requirement() -> Option<CustomKernelOwner> {
    (!crate::backend::managed_memory::pointwise_kernel::source_qualified())
        .then_some(CustomKernelOwner::Pointwise)
}
fn grouped_has_selected_rows(operation: WorkspaceOperationView<'_>) -> Option<bool> {
    Some(
        operation.inputs.get(0)?.elements().ok()? != 0
            && operation.inputs.get(1)?.elements().ok()? != 0,
    )
}
fn bf16_projection_source_requirement() -> Option<CustomKernelOwner> {
    (!crate::backend::managed_memory::bf16_projection_kernel::source_qualified())
        .then_some(CustomKernelOwner::Bf16Projection)
}
fn bf16_grouped_width(width: i32) -> bool {
    width > 0 && width % 32 == 0
}
fn grouped_indexed_source_requirement() -> Option<CustomKernelOwner> {
    // The same native cache layout qualifies the compiled standard library and
    // selector implementation; no Device, library or source is constructed here.
    // This source revision includes the exact one-index gather/scatter variants.
    safemlx::PreparedPipelineCachePlan::new(0)
        .layout::<()>()
        .is_err()
        .then_some(CustomKernelOwner::GroupedIndexed)
}
fn grouped_sort_kernels(selections: usize) -> Option<usize> {
    safemlx::OperationEvent::resident_grouped_sort_additional_kernels(selections)
}
fn chunked_grouped_sort_kernels(tokens: usize, top_k: usize, chunk: usize) -> Option<usize> {
    let whole = tokens / chunk;
    let tail = tokens % chunk;
    let full = if whole == 0 {
        0
    } else {
        grouped_sort_kernels(chunk.checked_mul(top_k)?)?.checked_mul(whole)?
    };
    full.checked_add(grouped_sort_kernels(tail.checked_mul(top_k)?)?)
}
fn grouped_linear_lowering(operation: WorkspaceOperationView<'_>) -> Option<Lowering> {
    let WorkspaceOperationKindView::Grouped {
        bank: WorkspaceGroupedBank::Linear(spec),
        phase: WorkspaceGroupedPhase::Whole,
        partitions: None,
    } = operation.kind
    else {
        return None;
    };
    // Each physical projection supplies its actual worker population. The
    // sorting, coefficients and reduction remain the same shared equation.
    let fp8 = matches!(
        spec.projection().format().encoding(),
        eredu_checkpoint::LinearFormat::E4M3BlockFp8(_)
    );
    if !fp8 && spec.projection().format().encoding() != eredu_checkpoint::LinearFormat::Dense {
        return None;
    }
    // Safe ID validation 31/36/3; group_by_id 15/17/1; input take 4/5;
    // projection max 36/43/3 (including BF16 safe-ID validation); coefficient
    // gather/expand/multiply 13/15; zeros/scatter/reshape 10/12/1; sum 2/2.
    // All are potential constructors, including casts/broadcasts that alias.
    let mut value = if fp8 {
        fp8::grouped_linear(operation)?
    } else {
        let mut value = Lowering::plain(111, 130, 8);
        value.validations = 2;
        value.bf16_projection_calls = 1;
        value
    };
    value.additional_sort_kernels = usize::try_from(operation.inputs.get(1)?.elements().ok()?)
        .ok()
        .and_then(grouped_sort_kernels);
    if grouped_has_selected_rows(operation)? {
        let projection_owner = value.unqualified_kernel_owner;
        value.unqualified_kernel_owner = projection_owner
            .or_else(|| match spec.activation() {
                _ if !fp8 && bf16_grouped_width(spec.input_dimensions()) => {
                    bf16_projection_source_requirement()
                }
                eredu_nn::GroupedLinearActivation::Silu => pointwise_source_requirement(),
                _ => None,
            })
            .or_else(|| {
                (spec.activation() == eredu_nn::GroupedLinearActivation::Silu)
                    .then(pointwise_source_requirement)
                    .flatten()
            })
            .or_else(grouped_indexed_source_requirement);
    }
    // Five ArgSort temporaries; either the quantizer's one and grouped
    // projector's five compactions or four GatherMM compactions; two scratch
    // buffers per actual validation and the final weighted sum.
    value.maximum_births += 5 + (if fp8 { 6 } else { 4 }) + 2 * value.validations + 2;
    value.intermediate_rank = value.intermediate_rank.max(3);
    // Native gather/scatter forward the supplied stream to index broadcasts.
    // All closed grouped constructors therefore stay on this selected GPU.
    value.streams = 1;
    if spec.projection().bias().is_some() {
        value.primitives += 9;
        value.edges += 11;
        value.maximum_births += 9;
    }
    match spec.activation() {
        eredu_nn::GroupedLinearActivation::Identity => {}
        eredu_nn::GroupedLinearActivation::Silu => {
            value.pointwise_calls = 1;
            value.primitives += 15;
            value.edges += 17;
            value.seeds += 1;
            value.maximum_births += 15 + 1 + 1; // fallback, seed, custom compaction
        }
        _ => return None,
    }
    let top_k = usize::try_from(*operation.inputs.get(1)?.shape().last()?).ok()?;
    grouped_reduction::extend(&mut value, spec.reduction(), top_k, 1)?;
    Some(value)
}

fn gated_product_lowering(policy: eredu_nn::GatedProductPolicy) -> Option<Lowering> {
    use eredu_nn::GatedProductActivation;
    // core_backend::gated_product uses the shared activation equations and
    // finishes with one binary Multiply (two casts, two broadcasts, result).
    let (mut primitives, mut edges, mut seeds, copies) = match policy.activation() {
        GatedProductActivation::Silu if policy.sigmoid_multiplier() == 1.0 => {
            // Widen, Negative, Exp(+cast), Add and Divide, final cast:
            // 15/17/1 dominates CustomKernel plus its two surrounding casts.
            // The custom worker can additionally compact its one input.
            (20, 23, 1, 1)
        }
        GatedProductActivation::Silu => {
            // Scalar scale, native Sigmoid(+cast), original-gate Multiply,
            // final up Multiply. This branch calls the native unary sigmoid.
            (17, 20, 1, 0)
        }
        GatedProductActivation::GeluApproximate => {
            // The same activation body used by standalone GELU, then the
            // ordinary up Multiply (two casts, two broadcasts and result).
            let activation = activation::gelu_approximate();
            (
                activation.primitives + 5,
                activation.edges + 6,
                activation.seeds,
                0,
            )
        }
        _ => return None,
    };
    if policy.gate_upper_bound().is_some() {
        primitives += 5;
        edges += 6;
        seeds += 1;
    }
    if policy.up_absolute_bound().is_some() {
        // Native clip is Maximum followed by Minimum, with two eager bounds.
        primitives += 10;
        edges += 12;
        seeds += 2;
    }
    if policy.up_offset() != 0.0 {
        primitives += 5;
        edges += 6;
        seeds += 1;
    }
    let mut value = reduction_lowering(primitives, edges, seeds, copies);
    if policy.activation() == GatedProductActivation::Silu && policy.sigmoid_multiplier() == 1.0 {
        value.pointwise_calls = 1;
    }
    Some(value)
}
fn gated_delta_lowering(operation: WorkspaceOperationView<'_>) -> Option<Lowering> {
    #[cfg(feature = "cuda")]
    {
        let _ = operation;
        None
    }
    #[cfg(not(feature = "cuda"))]
    {
        let length = *operation.inputs.get(0)?.shape().get(1)?;
        if length <= 0 || !matches!(operation.inputs.len(), 5 | 6) || operation.outputs.len() != 2 {
            return None;
        }
        // The existing Metal emitter validates every shape/dtype and the grid.
        // Reuse the actual worker's chunk policy, including its decode branch.
        let chunk =
            usize::try_from(super::super::gated_delta::metal_scan_chunk_tokens(length)).ok()?;
        let chunks = usize::try_from(length).ok()?.div_ceil(chunk);
        let slices = if length == 1 { 0usize } else { 5 };
        // Six cast calls and two output descriptors sharing one CustomKernel.
        // Count both descriptors as node slots: each carries six input edges,
        // and the sibling pair adds two references. This also funds both C
        // result shells and both Fixed Eval output slots without a fake seed.
        let mut primitives = chunks.checked_mul(8usize.checked_add(slices)?)?;
        let mut edges = chunks.checked_mul(20usize.checked_add(slices)?)?;
        let mut seeds = 0;
        if operation.inputs.len() == 5 {
            // The shared worker constructs zeros only when no state was given:
            // Full, two Broadcasts, cast, and one eager scalar descriptor.
            primitives = primitives.checked_add(4)?;
            edges = edges.checked_add(4)?;
            seeds = 1;
        }
        if length != 1 {
            // The call exists even for one chunk (where concatenate aliases).
            // Each input cast plus one result bounds its C result shell as well.
            primitives = primitives.checked_add(chunks.checked_add(1)?)?;
            edges = edges.checked_add(chunks.checked_mul(2)?)?;
        }
        // Returned sequence cast. State always remains F32.
        primitives = primitives.checked_add(1)?;
        edges = edges.checked_add(1)?;
        let maximum_births = primitives
            .checked_add(seeds)?
            .checked_add(chunks.checked_mul(6)?)?;
        let mut value = Lowering::plain(primitives, edges, seeds);
        // The custom worker may compact all six inputs; each chunk has two
        // real output Data owners. Their payloads are already in the emitter.
        value.maximum_births = maximum_births;
        value.maximum_operands = chunks.max(6);
        value.intermediate_rank = 4;
        value.recurrent_calls = chunks;
        // Reuse the existing preconstructed chunk-output destination bank.
        value.grouped_output_chunks = if length == 1 { 0 } else { chunks };
        Some(value)
    }
}
fn candidate_lowering(operation: WorkspaceOperationView<'_>) -> Option<Lowering> {
    let WorkspaceOperationKindView::CandidateExtraction { vocabulary, count } = operation.kind
    else {
        return None;
    };
    // The concrete physical emitter validates the exact input/output geometry.
    if vocabulary == 0 || count == 0 || count > vocabulary || vocabulary > i32::MAX as u32 {
        return None;
    }
    // Terminal Slice/Reshape 2/2, F32 cast 1/1, isfinite 27/32 + two
    // F32 infinity seeds, U32 cast 1/1, Reduce/Squeeze 2/2, flatten+ArgSort
    // 2/2, tail Slice/Reshape 2/2, Contiguous 1/1, and flatten+single-index
    // Gather/astype/Squeeze 4/5. Every primitive candidate is counted even
    // when its exact shape/dtype makes it an alias. IDs come from ArgSort.
    let mut value = reduction_lowering(42, 48, 2, 2 + 5);
    value.additional_sort_kernels = grouped_sort_kernels(vocabulary as usize);
    value.intermediate_rank = 3;
    // The composite operation's two visible outputs do not describe these
    // nine explicit retained C handles; its source clone is in the real trace.
    value.backend_shells = crate::backend::array_copy::CandidateExtraction::ROOTS;
    Some(value)
}
// Concrete safe-wrapper aliases which do not emit a neutral operation. The
// generic architecture's semantic copies are counted independently by the
// allocation-free WorkspaceTensor Clone producer.
fn backend_handle_shells(
    operation: WorkspaceOperationView<'_>,
    lowering: Lowering,
) -> Option<usize> {
    use WorkspaceOperationKindView as K;
    (match operation.kind {
        // Int64 token normalization and empty take-index validation each clone.
        K::Embedding(..) => 2,
        K::VocabularyParallelLookup { .. } => 6,
        K::Gather { .. } => 1,
        // fast::rope aliases its single batch; explicit cosine/sine adaptation
        // aliases two inputs. Every batch alias corresponds to a potential
        // fused RoPE node already counted by this exact lowering.
        K::TensorRotary(..) | K::RotaryFrequencies(..) | K::Rotary(..) => {
            lowering.primitives.checked_add(2)?
        }
        K::Attention { .. } => 1, // boolean mask promotion keeps an alias
        K::GatedDeltaScan => 1,   // optional initial state handoff
        K::Projection(_) => 1,    // direct packed weight's owning helper
        K::Grouped { .. } => lowering.bf16_projection_calls.checked_mul(2)?,
        // scores-for-choice, selected scores, optional sum input, packed weight
        K::GroupSelection { .. } => 4,
        K::Sampling(eredu_nn::workspace::WorkspaceSamplingOperation::OptionalTokenFilter) => 1,
        // The shared penalty worker owns one initial adjusted-logits alias.
        K::Sampling(eredu_nn::workspace::WorkspaceSamplingOperation::Penalties { .. }) => 1,
        _ => 0,
    })
    .checked_add(lowering.backend_shells)
}
fn resident_worker_profile(operation: WorkspaceOperationKindView<'_>, lowering: Lowering) -> bool {
    use WorkspaceOperationKindView as K;
    // Called only after the concrete lowering succeeds: recurrent counts both
    // sibling outputs and six possible compactions per chunk, depthwise conv1d
    // counts its output and two copies, and grouped counts five sort backings,
    // projection compactions and the actual extra merge/copy dispatches.
    // They share the priced descriptor/Data, temporary and resource classes.
    // Unqualified convolution algorithms and other unsupported lowerings still return None;
    // source/configuration ownership remains an independent prerequisite.
    // Only this exact multi-stream lowering declares all its CPU entries.
    // Another stream-two lowering cannot inherit router partition authority.
    lowering.streams == 1
        || (matches!(operation, K::GroupSelection { .. }) && lowering.router_cpu_partitions != 0)
}

// Same actual optional-compaction/store sequence for Model and saved-copy roles.
fn host_store_lowering() -> Lowering {
    let mut value = Lowering::plain(2, 2, 0);
    value.maximum_births = 1;
    value.nested_completions = 1;
    value.grouped_output_calls = 0;
    value
}

/// The same selected lowering supplies the maximum backing population for
/// ordinary allocator controls; output roots are subtracted by their effects.
pub(super) fn allocation_births(operation: WorkspaceOperationView<'_>) -> Option<usize> {
    if matches!(
        operation.kind,
        WorkspaceOperationKindView::GroupSelection { .. }
    ) {
        // Original lowering additionally authenticates its prepared CPU stream.
        // Its branch population bounds ordinary ranking without granting that
        // original execution source or replacing ordinary host-side controls.
        router::allocation_births(operation)
    } else {
        lowering(operation).map(|value| value.maximum_births)
    }
}

fn lowering(operation: WorkspaceOperationView<'_>) -> Option<Lowering> {
    use WorkspaceOperationKindView as K;
    match operation.kind {
        // Scalar/host initialization can build two Broadcasts, a cast and
        // Full, plus one eager source. The exact driver placeholder is removed
        // by ordinal before reaching this branch, never by operation kind.
        K::Initialize | K::InitializeFloating(_) => Some(Lowering::plain(4, 4, 1)),
        K::GeneratedF32Initialization => {
            // Shared F32InitializationPlan fills exactly one Rust host buffer.
            // MlxTensor::from_f32_slice copies it into one eager native seed,
            // then Array::copy constructs exactly one Copy with that one input.
            // The host payload is already charged by the same operation's host
            // fact; fixed worker frames are included by copy_rank below.
            basic::generated_f32_plan_view(operation).ok()?;
            let mut source = Lowering::plain(1, 1, 1);
            source.maximum_births = 1;
            Some(source)
        }
        K::Elementwise(_) if super::host_array::slice_dtype(operation).is_some() => {
            let mut source = Lowering::plain(1, 1, 1);
            source.maximum_births = 1;
            Some(source)
        }
        K::Elementwise(_) if super::host_array::dtype(operation).is_some() => {
            Some(Lowering::plain(0, 0, 1))
        }
        K::Elementwise(_) if super::zero_fill::dtype(operation).is_some() => {
            let primitives = 1 + usize::from(!operation.outputs.get(0)?.shape().is_empty());
            Some(Lowering::plain(primitives, primitives, 1))
        }
        K::Elementwise("scalar_u8") if basic::is_scalar_u8(operation) => {
            Some(Lowering::plain(0, 0, 1))
        }
        K::Elementwise("scalar_f32") if basic::is_scalar_f32(operation) => {
            Some(Lowering::plain(0, 0, 1))
        }
        K::HostStoreFloating(..) => {
            let dtype = basic::host_transfer_dtype(operation)?;
            safemlx::PreparedHostCopyDestination::original_layout(
                operation.inputs.get(0)?.shape().len(),
                dtype,
            )?;
            // The ordinary shared worker may compact once, then copies into
            // the separately accepted Host allocation. Only the optional
            // Device compaction is an allocator birth in this recipe. The
            // internal Host Array is retained in the final completion union.
            Some(host_store_lowering())
        }
        K::HostTransferFloating(_) | K::HostLoadStoredFloating(..) => {
            let dtype = basic::host_transfer_dtype(operation)?;
            safemlx::ImmutableHostTransferBuffer::original_copy_layout(
                operation.outputs.get(0)?.shape().len(),
                dtype,
            )?;
            // The existing copy_from_host worker creates one retained host leaf
            // and CopyFromHostTransfer, then settles one nested frontier. Its
            // source is existing backing; only the device output is a birth.
            let mut value = Lowering::plain(1, 1, 1);
            value.maximum_births = 1;
            value.nested_completions = 1;
            Some(value)
        }
        K::ValueRetention => {
            if operation.inputs.is_empty() || !operation.outputs.is_empty() {
                return None;
            }
            let mut lowering = Lowering::plain(0, 0, 0);
            lowering.grouped_output_calls = 0;
            lowering.maximum_operands = operation.inputs.len();
            Some(lowering)
        }
        K::ValueCompletion => {
            if !operation.outputs.is_empty() {
                return None;
            }
            crate::backend::nn::shared::value_completion_control_bytes(operation.inputs.len())?;
            let mut lowering = Lowering::plain(0, 0, 0);
            lowering.grouped_output_calls = 0;
            lowering.nested_completions = 1;
            lowering.maximum_operands = operation.inputs.len();
            Some(lowering)
        }

        K::View(name) if super::byte_view::selected(name).is_some() => {
            super::byte_view::inspect(operation)?;
            // The graph has one View. The generic GPU worker census includes
            // its temporary descriptor/Data and shared general-copy branch.
            Some(Lowering::plain(1, 1, 0))
        }
        K::View("broadcast" | "squeeze" | "expand_dims" | "reshape")
        | K::Transpose(_)
        | K::Contiguous => Some(Lowering::plain(1, 1, 0)),
        K::DeepCopy => {
            // The actual isolated-copy worker completes its contiguous source,
            // then invokes the eager typed constructor for independent data.
            let mut value = Lowering::plain(0, 0, 1);
            value.helper_controls = safemlx::original_scoped_deep_copy_control_bytes(
                operation.inputs.get(0)?.shape().len(),
            )?;
            Some(value)
        }
        // Tensor's static indexing route builds Slice and optional Reshape.
        // Array-valued gathers use K::Gather and do not enter this branch.
        K::Index { .. } => Some(Lowering::plain(2, 2, 0)),
        // This worker preserves rank: there is no indexing-wrapper Reshape.
        K::StaticSlice { .. } if basic::is_static_slice(operation) => {
            Some(Lowering::plain(1, 1, 0))
        }
        K::Pad(eredu_nn::PadMode::Constant) => {
            safemlx::ops::constant_pad_control_bytes(operation.inputs.get(0)?.shape().len())?;
            // The shared adapter creates an I32 zero and a dtype cast; native
            // pad adds a cast candidate and the two-input Pad. The fill/copy
            // worker writes directly into the output without a slice descriptor.
            Some(Lowering::plain(3, 4, 1))
        }
        // The shared static update adapter strips/reinserts singleton axes,
        // then native slice_update casts, broadcasts and creates SliceUpdate.
        K::SliceUpdate { .. } => Some(Lowering::plain(5, 6, 0)),
        // Direct exact-rank update: native cast, broadcast, and SliceUpdate only.
        K::StaticSliceUpdate { .. } => Some(Lowering::plain(3, 4, 0)),
        K::BlockwiseAttention { .. } => attention_tiled::lowering(operation),
        K::CausalMask(geometry) => {
            // The shared no-lengths mask worker creates two I32 aranges and
            // two fixed reshapes, then one comparison. Each binary candidate
            // has two casts, two broadcasts and one result (five primitives,
            // six edges). Windowed masks add subtraction with one eager I32
            // seed, a second comparison and logical_and; no host mask is built.
            let (primitives, edges, seeds) = if geometry.max_past().is_some() {
                (2 + 2 + 4 * 5, 2 + 4 * 6, 1)
            } else {
                (2 + 2 + 5, 2 + 6, 0)
            };
            Some(Lowering::plain(primitives, edges, seeds))
        }
        K::Concatenate => {
            let count = operation.inputs.len();
            // One cast candidate per actual input and one variadic primitive.
            // Empty and mismatched inputs are still rejected by the emitter.
            let mut value = Lowering::plain(count.checked_add(1)?, count.checked_mul(2)?, 0);
            value.maximum_operands = count.max(4);
            Some(value)
        }
        K::PooledAttention { .. } => pooled_attention::lowering(operation),
        K::RelativeAttention { .. } => relative_attention::lowering(operation),
        K::Attention {
            window: None,
            softcap: false,
            arithmetic: eredu_nn::AttentionArithmetic::Fused,
            ..
        } => {
            // Actual native fast.cpp fallback: Q/K/V/mask/sink casts,
            // scale, grouped-head views, two matmuls, optional causal mask,
            // mask select, sink broadcast/concat/slice and final softmax.
            // It dominates the selected fused primitive branch. The backend
            // attention adapter's additive-mask cast is included as well.
            let mut value = reduction_lowering(59, 64, 2, 12);
            value.intermediate_rank = 5;
            value.maximum_operands = 5;
            Some(value)
        }
        K::Attention { .. } => {
            attention_tiled::lowering(operation).or_else(|| attention_direct::lowering(operation))
        }
        // Two casts, two Broadcasts, one binary result. Repeated edges count.
        K::Elementwise(
            "add" | "subtract" | "multiply" | "divide" | "logical_or" | "greater" | "less"
            | "greater_equal" | "less_equal" | "logical_and",
        ) => Some(Lowering::plain(5, 6, 0)),
        // Bool cast (possibly alias) and the ordinary unary Not descriptor.
        K::Elementwise("logical_not") => Some(Lowering::plain(2, 2, 0)),
        K::Elementwise("multiply_scalar" | "maximum_scalar" | "maximum_i32" | "equal_i32") => {
            Some(Lowering::plain(5, 6, 1))
        }
        K::Elementwise("clip") => clip::lowering(operation),
        K::Elementwise("where") => Some(Lowering::plain(7, 9, 0)),
        K::Elementwise("masked_scatter") => {
            super::masked_scatter::geometry(operation).ok()??;
            // One source cast, mask reshape/broadcast, source reshape/broadcast,
            // three leading-axis expansions, MaskedScatter and final squeeze.
            // The ternary primitive has three edges; the other nine are unary.
            let mut value = Lowering::plain(10, 12, 0);
            // Eval-only flatten/contiguous-mask and offsets descriptors are
            // outside the lazy graph. Scan writes the existing offsets buffer.
            value.maximum_births += 3;
            value.intermediate_rank = operation
                .inputs
                .iter()
                .map(|input| input.shape().len())
                .max()?
                .checked_add(1)?;
            Some(value)
        }
        // The shared activation adapters try one custom F32 kernel, then the
        // ordinary scalar equation. Include widening/final cast and that exact
        // fallback rather than assuming the kernel cache is already populated.
        K::Elementwise("silu" | "sigmoid") => {
            let mut value = if matches!(operation.kind, K::Elementwise("silu")) {
                reduction_lowering(15, 17, 1, 1)
            } else {
                reduction_lowering(4, 4, 0, 1)
            };
            value.pointwise_calls = 1;
            Some(value)
        }
        // The shared beta-scaled softplus worker widens, then calls Multiply,
        // Exp, Log1p, Divide, Greater and Select before the final dtype cast.
        // Three binary candidates each have two casts, two broadcasts and the
        // result; Select has three of each input operation. Include both
        // threshold branches and the three actual eager F32 scalar descriptors.
        // The physical emitter already accounts for all nineteen buffers.
        K::Elementwise("softplus") => Some(Lowering::plain(28, 33, 3)),
        K::Elementwise("gelu") => Some(activation::gelu()),
        K::Elementwise("gelu_approximate") => Some(activation::gelu_approximate()),
        // Five binary candidates (two infinity comparisons, isnan's
        // NotEqual(a,a), and two Ors), each two casts/two broadcasts/result;
        // final Bool AsType/Not plus two actual floating infinity seeds.
        K::Elementwise("is_finite") => Some(Lowering::plain(27, 32, 2)),
        // Native floating NotEqual(a,a) and Equal(a,+/-Inf), each the same
        // two-cast/two-broadcast/binary worker; infinity creates one F32 seed.
        K::Elementwise("is_nan") => Some(Lowering::plain(5, 6, 0)),
        K::Elementwise("is_positive_infinity" | "is_negative_infinity") => {
            Some(Lowering::plain(5, 6, 1))
        }
        K::Elementwise("square") => Some(Lowering::plain(1, 1, 0)),
        K::GatedProduct(policy) => gated_product_lowering(policy),
        K::GatedDeltaScan => gated_delta_lowering(operation),
        K::HyperCollapse(..) | K::HyperExpand | K::HyperHeadCoefficients(_) | K::HyperHeadSum => {
            hyper::lowering(operation)
        }
        K::SelectiveStateSpaceScan(..) => selective_scan::lowering(operation),
        K::GroupSelection { .. } => router::lowering(operation),
        K::JointGroupSelection(_) => joint_selection::lowering(operation),
        K::MaskedOutputProjection { .. } => masked_readout::lowering(operation),
        K::Convolution { .. } => convolution::lowering(operation),
        K::Grouped { .. } => {
            grouped_linear_lowering(operation).or_else(|| grouped_packed::lower(operation))
        }
        // The shared TokenFilter worker first realizes the executable mask:
        // truncate an oversized mask, forbid missing IDs, then replicate rows.
        // It uploads that Bool array and one F32 -infinity scalar, followed by
        // native where (three casts, three Broadcasts, Select). Membership and
        // holes change values, never this finite producer population. Exact All
        // emits no filter operation; Optional also permits that input alias and
        // is bounded by the same masked branch. The sampling emitter validates
        // F32/nonempty geometry and prices both seeds, where and host expansion.
        K::Sampling(
            eredu_nn::workspace::WorkspaceSamplingOperation::TokenFilter
            | eredu_nn::workspace::WorkspaceSamplingOperation::OptionalTokenFilter,
        ) => Some(Lowering::plain(7, 9, 2)),
        K::Sampling(
            eredu_nn::workspace::WorkspaceSamplingOperation::TopK { .. }
            | eredu_nn::workspace::WorkspaceSamplingOperation::TopP
            | eredu_nn::workspace::WorkspaceSamplingOperation::MinP,
        ) => sampling_filters::lowering(operation),
        K::Sampling(
            eredu_nn::workspace::WorkspaceSamplingOperation::CreateRandomKey
            | eredu_nn::workspace::WorkspaceSamplingOperation::SelectRandomKey { .. }
            | eredu_nn::workspace::WorkspaceSamplingOperation::SplitRandomKey
            | eredu_nn::workspace::WorkspaceSamplingOperation::UniformUnitInterval
            | eredu_nn::workspace::WorkspaceSamplingOperation::Categorical,
        ) => sampling_random::lowering(operation),
        K::Sampling(
            eredu_nn::workspace::WorkspaceSamplingOperation::MirostatCutoff
            | eredu_nn::workspace::WorkspaceSamplingOperation::TokenProbability,
        ) => sampling_mirostat::lowering(operation),
        K::Sampling(eredu_nn::workspace::WorkspaceSamplingOperation::Penalties { .. }) => {
            sampling_penalties::lowering(operation)
        }
        // Final-axis argmax normally creates ArgReduce + Squeeze. A unit
        // reduced axis instead uses zeros/Full: two Broadcasts, a cast, Full,
        // Squeeze, and its eager scalar seed. Retain that real alternate path.
        K::Sampling(eredu_nn::workspace::WorkspaceSamplingOperation::Greedy) => {
            Some(Lowering::plain(5, 5, 1))
        }
        K::Sampling(eredu_nn::workspace::WorkspaceSamplingOperation::ValidateToken { .. }) => {
            embedding::token_validation_lowering(operation)
        }
        // Direct completion/read of the existing scalar creates no tensor.
        K::Sampling(eredu_nn::workspace::WorkspaceSamplingOperation::ReadToken) => {
            Some(Lowering::plain(0, 0, 0))
        }
        K::Elementwise("exp" | "tanh" | "log") => Some(Lowering::plain(2, 2, 0)),
        // Exact target precision changes only AsType's scalar conversion, not
        // its one-input descriptor population. Equal types can alias instead.
        K::CastFloating(_)
        | K::Elementwise("cast_u32" | "capture_cast_f32" | "cast_f32" | "bool_to_u32") => {
            Some(Lowering::plain(1, 1, 0))
        }
        K::Normalization(..) | K::LayerNorm { .. } | K::ConstructedNormalization(_) => {
            normalization_lowering(operation)
        }
        K::Reduction("sum" | "sum_all" | "min" | "max", _, _) => {
            Some(reduction_lowering(2, 2, 0, 2))
        }
        K::Reduction("mean", _, _) => Some(reduction_lowering(7, 8, 1, 2)),
        K::Reduction("softmax", axis, _) => {
            let last = usize::try_from(axis).ok()?.checked_add(1)?
                == operation.inputs.get(0)?.shape().len();
            Some(if last {
                Lowering::plain(2, 2, 0)
            } else {
                reduction_lowering(19, 21, 0, 4)
            })
        }
        K::TensorRotary(..) | K::RotaryFrequencies(..) | K::Rotary(..) => {
            rotary::lowering(operation)
        }
        K::PreparedMultiAxisRotary(..) => prepared_rotary::lowering(operation),
        // Actual ops.cpp matmul: two rank expansions, two casts, either
        // flatten/unflatten or two batch Broadcasts, Matmul and Squeeze.
        K::Matmul => Some(Lowering::product(8, 9, 0, 1)),
        // Tensor::linear adds the weight Transpose. The biased branch uses
        // addmm's three casts, up to three broadcasts, input/output reshape
        // and AddMM; the explicit zero-K adapter contributes two reshapes.
        K::DenseLinear => Some(Lowering::product(14, 16, 0, 1)),
        K::ParameterDecode(_) => {
            if let Some(g) = super::parameter_decode::gguf::inspect(operation).ok()? {
                let mut value = Lowering::plain(1, 1, 1);
                value.maximum_births = 1;
                value.intermediate_rank = g.rank;
                value.nested_completions = 2;
                value.backend_shells = 1;
                value.helper_controls = super::parameter_decode::gguf::control_bytes()?;
                return Some(value);
            }
            if let Some(g) = super::parameter_decode::fp8::inspect(operation).ok()? {
                // Two three-node repeat programs, ConvertFP8, Slice and the
                // five-node mixed multiply. E8M0 adds the table seed plus
                // cast/flatten/index-cast/Gather/Squeeze; partitions add the
                // two source reshapes and final shape restoration.
                let primitives = 13usize
                    .checked_add(if g.encoded_scale { 5 } else { 0 })?
                    .checked_add(if g.partitioned { 3 } else { 0 })?;
                let edges = 14usize
                    .checked_add(if g.encoded_scale { 6 } else { 0 })?
                    .checked_add(if g.partitioned { 3 } else { 0 })?;
                let mut value = Lowering::plain(primitives, edges, usize::from(g.encoded_scale));
                value.maximum_births = g.births()?;
                value.intermediate_rank = g.work_rank + 1;
                value.maximum_operands = 2;
                value.helper_controls = super::parameter_decode::fp8::control_bytes(g)?;
                return Some(value);
            }
            let _source = super::parameter_decode::inspect(operation).ok()??;
            // One fast::Quantize and the enclosing AsType candidate. The
            // kernel owns one possible compaction for each retained input.
            let mut value = Lowering::plain(2, operation.inputs.len().checked_add(1)?, 0);
            value.maximum_births = operation.inputs.len().checked_add(1)?;
            value.maximum_operands = operation.inputs.len();
            value.helper_controls = super::parameter_decode::control_bytes()?;
            Some(value)
        }
        K::Projection(format) => match format.encoding() {
            // PhysicalLinear permits floating replacements for these exact
            // descriptors. Retain that shared dense alternative, including its
            // BF16 custom source/controls, without selecting by family.
            eredu_checkpoint::LinearFormat::Dense
            | eredu_checkpoint::LinearFormat::Affine(_)
            | eredu_checkpoint::LinearFormat::MxFp4 => {
                // Dense: transpose/matmul plus separate bias Add. Packed:
                // three supplied-stream casts, up to four Broadcasts,
                // QuantizedMatmul and optional bias Add. MXFP4 omits casts.
                // Keep four compactions and both split/reduction temporaries;
                // the existing physical emitter retains every alternate birth.
                let edges = if format.encoding() == eredu_checkpoint::LinearFormat::Dense {
                    16
                } else {
                    17
                };
                let mut value = Lowering::product(14, edges, 1, 1);
                let input = operation.inputs.get(0)?;
                if input.elements().ok()? != 0 && bf16_grouped_width(*input.shape().last()?) {
                    value.bf16_projection_calls = 1;
                    value.unqualified_kernel_owner = bf16_projection_source_requirement();
                }
                Some(value)
            }
            // Direct GGUF Metal rows: flatten, one two-input CustomKernel,
            // result reshape, optional bias Add. Include the same dense
            // replacement alternative without choosing a family or format
            // outside the existing physical realization.
            eredu_checkpoint::LinearFormat::GgufIQuant { .. } => {
                Some(Lowering::product(14, 16, 1, 1))
            }
            eredu_checkpoint::LinearFormat::E4M3BlockFp8(_) => fp8::lowering(operation),
        },
        K::ProjectionPrepare(_) | K::ProjectionFinish(_) => fp8::observed(operation),
        K::BlockFp8ActivationDecode => fp8::decode(operation),
        K::IndexedRowAdd if super::basic::indexed_row_add(operation) => {
            // scatter_axis: update cast, five broadcast candidates, one
            // three-input ScatterAxis::Sum. Empty output may alias the base.
            let mut value = Lowering::plain(7, 9, 0);
            value.intermediate_rank = 2;
            value.maximum_operands = 3;
            if safemlx::PreparedPipelineCachePlan::new(0)
                .layout::<()>()
                .is_err()
            {
                value.unqualified_kernel_owner = Some(CustomKernelOwner::GroupedIndexed);
            }
            Some(value)
        }
        K::IndexedElementSelect if super::basic::indexed_elements(operation) => {
            // Already-flat take: one index cast candidate, Gather and Squeeze.
            // Index values are supplied by the paid sparse producer, so there
            // is no tensor-trait validation/readback graph to duplicate here.
            let mut value = Lowering::plain(3, 4, 0);
            value.intermediate_rank = 2;
            value.maximum_operands = 2;
            value.unqualified_kernel_owner = grouped_indexed_source_requirement();
            Some(value)
        }
        K::IndexedElementUpdate if super::basic::indexed_elements(operation) => {
            // Shared rank-one argument worker: exact-shape Broadcast and
            // Reshape; native index/update cast candidates; three-input Scatter.
            // The singleton-trimming helper has no work for rank-one updates.
            let mut value = Lowering::plain(5, 7, 0);
            value.intermediate_rank = 2;
            value.maximum_operands = 3;
            value.helper_controls = safemlx::Array::flat_index_update_control_bytes()?;
            value.unqualified_kernel_owner = grouped_indexed_source_requirement();
            Some(value)
        }
        K::Gather { .. } => gather::lowering(operation),
        K::GatherPooledMask => gather_pooled_mask::lowering(operation),
        K::CandidateExtraction { .. } => candidate_lowering(operation),
        K::PoolingMask(_) => pooling_mask::lowering(operation),
        K::PooledPositions { .. } => pooled_positions::lowering(operation),
        K::IndexedAttention { .. } => indexed_attention::lowering(operation),
        K::Embedding(..) | K::VocabularyParallelLookup { .. } => embedding::lowering(operation),
        _ => None,
    }
}
fn add(a: usize, b: usize) -> Result<usize, Error> {
    a.checked_add(b)
        .ok_or_else(|| Error::backend("resident lowering population overflow"))
}
impl ResidentRecipeRecorder {
    fn record_equation(
        &mut self,
        span: &InferenceWorkspaceSpan,
        report: &WorkspaceTraceReport,
        retained_roots: usize,
        output_roots: usize,
        input_operation: Option<usize>,
        output: Option<WorkspaceStoragePopulation>,
        prepared_source: bool,
    ) -> Result<(), Error> {
        self.record_equation_with_parallel(
            span,
            report,
            retained_roots,
            output_roots,
            input_operation,
            output,
            prepared_source,
            None,
        )
    }
    fn record_equation_with_parallel(
        &mut self,
        span: &InferenceWorkspaceSpan,
        report: &WorkspaceTraceReport,
        retained_roots: usize,
        output_roots: usize,
        input_operation: Option<usize>,
        output: Option<WorkspaceStoragePopulation>,
        prepared_source: bool,
        parallel: Option<parallel::OriginalParallelInvocation>,
    ) -> Result<(), Error> {
        let (opening_state, closing_state) = if prepared_source {
            let residual = report.residual.as_ref().ok_or_else(|| {
                self.metadata_error("prepared media recipe is missing its exact source selection")
            })?;
            let opening = residual.opening_storage.ok_or_else(|| {
                self.metadata_error("prepared media recipe is missing seeded opening source roots")
            })?;
            let closing = residual.closing_storage.ok_or_else(|| {
                self.metadata_error("prepared media recipe is missing seeded closing source roots")
            })?;
            (Some(opening), closing)
        } else {
            (report.opening_storage, report.closing_storage)
        };
        if let Some(input_operation) = input_operation {
            // The shared driver declares the exact placeholder ordinal. Plain
            // generation uses I32; original speculative spans preserve U32.
            // Both are token IDs, later replaced by their actual input receipt.
            let input = report.operations.get(input_operation).ok_or_else(|| {
                self.metadata_error("resident input producer ordinal is outside its trace")
            })?;
            if !(matches!(input.kind, WorkspaceOperationKind::Initialize)
                || super::basic::is_prepared_token_input(input.as_view())
                || matches!(
                    input.kind,
                    WorkspaceOperationKind::Elementwise("full_i32" | "full_u32")
                ))
                || !input.inputs.is_empty()
                || input.outputs.len() != 1
                || !matches!(
                    input.outputs[0].dtype(),
                    WorkspaceDtype::Int32 | WorkspaceDtype::Uint32
                )
            {
                return Err(self.metadata_error(
                    "resident input producer is not the driver's token placeholder",
                ));
            }
        }
        let layerwise_ordinals = self.validate_layerwise_constructor_trace(report)?;
        // One TokenValidationScope owns the entire prefill. Each completion
        // appends that accumulated batch, including earlier completed results.
        // Those detached predecessors need root/cache/edge slots, but neither
        // new primitives nor new buffer births. Decode has a fresh batch.
        let prefill = matches!(span, InferenceWorkspaceSpan::Prefill(_));
        let retained_roots = self.add(
            retained_roots,
            if prefill {
                self.prefill_validation_roots
            } else {
                0
            },
        )?;
        let publications = report
            .operations
            .iter()
            .filter(|op| {
                matches!(
                    op.kind,
                    WorkspaceOperationKind::Collective(
                        eredu_nn::workspace::WorkspaceCollective::Broadcast { .. }
                    )
                )
            })
            .count();
        // The completed contribution remains in the actual root collector until
        // final model completion; it is distinct from the published output.
        let reduced = self.reduce_trace_with_parallel(
            report,
            input_operation,
            retained_roots,
            self.add(output_roots, publications)?,
            parallel.as_ref(),
        )?;
        let addressable = self
            .addressable_sources
            .as_ref()
            .map(|source| source.invocation(&report.operations))
            .transpose()?
            .filter(|invocation| !invocation.occurrences().is_empty());
        let ordinary_addressable = self
            .ordinary_addressable_sources
            .as_ref()
            .map(|source| source.program(&report.operations))
            .transpose()?
            .filter(|program| crate::backend::runtime::residency::parameter_bank::OrdinaryIndexedRequestProgram::len(&**program) != 0);
        let ordinary_calls = self.ordinary_trace_call_controls(
            report,
            input_operation,
            layerwise_ordinals.as_ref().map(Vec::len),
            ordinary_addressable.as_deref(),
            parallel.as_ref(),
        )?;
        let next_prefill_validations = if prefill {
            self.add(self.prefill_validation_roots, reduced.validation_roots)?
        } else {
            self.prefill_validation_roots
        };
        if let Some(context) = &self.context {
            context.reserve_metadata_vec(&mut self.records, 1)?;
        } else {
            self.records
                .try_reserve(1)
                .map_err(Error::backend_retained_source)?;
        }
        let model_controls = self
            .context
            .as_ref()
            .and_then(|context| context.metadata_funding())
            .map(|funding| {
                eredu_runtime::replicated_session::SessionModelControlPlan::from_operations(
                    &report.operations,
                    &funding,
                )
            })
            .transpose()?;
        self.records.push(ResidentSpanRecipe {
            model_controls,
            span: span.clone(),
            layerwise_ordinals,
            closing_state,
            opening_state,
            current_output: reduced.carryover_complete.then_some(output).flatten(),
            validation_producers: reduced.validation_producers,
            traversal: reduced.traversal,
            mutable_storage: reduced.mutable_storage,
            first_missing_operation: reduced.first_missing_operation,
            missing_operation_detail: reduced.missing_operation_detail,
            unqualified_kernel_owner: reduced.unqualified_kernel_owner,
            maximum_rank: reduced.maximum_rank,
            host_primitive_nodes: reduced.host_primitive_nodes,
            validation_roots: reduced.validation_roots,
            grouped_outputs: reduced.grouped_outputs,
            query_controls: reduced.query_controls,
            graph: reduced.graph,
            dispatch: reduced.dispatch,
            nested_completions: reduced.nested_completions,
            nested_root_capacity: reduced.nested_root_capacity,
            capture_publications: 0,
            capture_roots: 0,
            capture_scalars: None,
            prediction_roots: None,
            parallel,
            addressable,
            ordinary_addressable,
            ordinary_parallel: ordinary_calls.parallel,
            ordinary_call_failure: ordinary_calls.failure,
            ordinary_paged_source: ordinary_calls.paged_source,
            ordinary_calls: ordinary_calls.controls,
        });
        let capture_publications = self
            .records
            .last()
            .and_then(|row| row.addressable.as_ref())
            .map(|invocation| {
                invocation
                    .occurrences()
                    .iter()
                    .try_fold(0usize, |sum, (_, quote)| {
                        sum.checked_add(quote.capture_publications)
                    })
            })
            .unwrap_or(Some(0))
            .ok_or_else(|| {
                self.metadata_error("addressable capture publication population overflow")
            })?;
        let ordinary_publications = self
            .records
            .last()
            .and_then(|row| row.ordinary_addressable.as_ref())
            .map(|program| program.capture_publications())
            .unwrap_or(Some(0))
            .ok_or_else(|| {
                self.metadata_error("ordinary addressable capture population overflow")
            })?;
        let capture_publications = self.add(capture_publications, ordinary_publications)?;
        self.record_capture_population(crate::backend::array_copy::CaptureNativePopulation {
            publications: capture_publications,
            ..Default::default()
        })?;
        // Per-row validation_roots stays local: its sum sizes the one batch.
        // Advance the prefix only after this complete row was retained.
        self.prefill_validation_roots = next_prefill_validations;
        Ok(())
    }

    fn record_sampling(
        &mut self,
        phase: eredu_runtime::working_memory::SamplingWorkspacePhase,
        report: &WorkspaceTraceReport,
        closing_roots: Option<WorkspaceStoragePopulation>,
    ) -> Result<(), Error> {
        use eredu_runtime::working_memory::SamplingWorkspacePhase as Phase;
        let expected = if self.sampling.is_empty() {
            Phase::Preparation
        } else {
            Phase::Step {
                index: u64::try_from(self.sampling.len() - 1)
                    .map_err(|cause| self.metadata_source(cause))?,
            }
        };
        if phase != expected {
            return Err(self.metadata_error("sampling recipe differs from actual phase order"));
        }
        // This is an observed empty preparation, not an inference from
        // temperature or policy. Existing sources are priced separately.
        let mut preparation = None;
        let (
            mutable_storage,
            completion,
            query_controls,
            first_missing_operation,
            missing_operation_detail,
        ) = if phase == Phase::Preparation && report.operations.is_empty() {
            preparation = Some(ResidentSamplingPreparation::Empty);
            (
                Some(CertifiedSpanStorage {
                    mutable_bytes: 0,
                    maximum_births: 0,
                }),
                None,
                Some(0),
                None,
                None,
            )
        } else if phase == Phase::Preparation && sampling_random::is_eager_key_trace(report) {
            match sampling_random::eager_preparation(report, self.mechanism.allocation())? {
                Some((storage, graph, controls)) => {
                    preparation = Some(ResidentSamplingPreparation::EagerKey(graph));
                    (Some(storage), None, Some(controls), None, None)
                }
                None => (None, None, None, Some(0), None),
            }
        } else {
            // One final token root. Prior completed tokens and logits enter as
            // leaves through actual operation inputs, never as new producers.
            let read = |operation: &&eredu_nn::workspace::WorkspaceOperation| {
                matches!(
                    operation.kind,
                    WorkspaceOperationKind::Sampling(
                        eredu_nn::workspace::WorkspaceSamplingOperation::ReadToken
                    )
                )
            };
            let adaptive_read = sampling_mirostat::is_adaptive_step(report);
            let final_read = report
                .operations
                .last()
                .is_some_and(|operation| read(&operation))
                && report.operations.iter().filter(read).count() == 1;
            let final_read = final_read || adaptive_read;
            let random_root = report.operations.iter().any(|operation| {
                matches!(
                    operation.kind,
                    WorkspaceOperationKind::Sampling(
                        eredu_nn::workspace::WorkspaceSamplingOperation::SplitRandomKey
                    )
                )
            });
            let output_roots = if matches!(phase, Phase::Step { .. }) {
                1 + usize::from(random_root)
            } else {
                0
            };
            let reduced = self.reduce_trace(report, None, 0, output_roots)?;
            let final_read = final_read && reduced.nested_completions == usize::from(adaptive_read);
            (
                if matches!(phase, Phase::Step { .. }) && !final_read {
                    None
                } else {
                    reduced.mutable_storage
                },
                final_read
                    .then_some(reduced.traversal.zip(reduced.graph))
                    .flatten()
                    .map(|(traversal, graph)| ResidentCompletionRecipe {
                        traversal,
                        graph,
                        dispatch: reduced.dispatch,
                        nested_completions: reduced.nested_completions,
                        nested_root_capacity: reduced.nested_root_capacity.max(3),
                        validation_roots: reduced.validation_roots,
                        grouped_outputs: reduced.grouped_outputs,
                    }),
                reduced.query_controls,
                reduced.first_missing_operation,
                reduced.missing_operation_detail,
            )
        };
        if let Some(context) = &self.context {
            context.reserve_metadata_vec(&mut self.sampling, 1)?;
        } else {
            self.sampling
                .try_reserve(1)
                .map_err(Error::backend_retained_source)?;
        }
        let ordinary_calls = self.ordinary_trace_call_controls(report, None, None, None, None)?;
        self.sampling.push(ResidentSamplingRecipe {
            closing_roots,
            phase,
            preparation,
            first_missing_operation,
            missing_operation_detail,
            mutable_storage,
            completion,
            query_controls,
            ordinary_call_failure: ordinary_calls.failure,
            ordinary_paged_source: ordinary_calls.paged_source,
            ordinary_calls: ordinary_calls.controls,
        });
        Ok(())
    }
}

impl InferenceEquationTraceObserver for ResidentRecipeRecorder {
    fn observe(
        &mut self,
        span: &InferenceWorkspaceSpan,
        report: &WorkspaceTraceReport,
        retained_roots: usize,
        output_roots: usize,
        input_operation: usize,
    ) -> Result<(), Error> {
        self.record_equation(
            span,
            report,
            retained_roots,
            output_roots,
            Some(input_operation),
            None,
            false,
        )
    }
    fn observe_with_storage(
        &mut self,
        span: &InferenceWorkspaceSpan,
        report: &WorkspaceTraceReport,
        retained_roots: usize,
        output_roots: usize,
        input_operation: usize,
        output: Option<WorkspaceStoragePopulation>,
    ) -> Result<(), Error> {
        self.record_equation(
            span,
            report,
            retained_roots,
            output_roots,
            Some(input_operation),
            output,
            false,
        )
    }
    fn observe_prepared_with_storage(
        &mut self,
        span: &InferenceWorkspaceSpan,
        report: &WorkspaceTraceReport,
        retained_roots: usize,
        output_roots: usize,
        input_operation: Option<usize>,
        output: Option<WorkspaceStoragePopulation>,
    ) -> Result<(), Error> {
        self.record_equation(
            span,
            report,
            retained_roots,
            output_roots,
            input_operation,
            output,
            true,
        )
    }
    fn observe_sampling_input(
        &mut self,
        input: eredu_runtime::working_memory::SamplingWorkspaceInputPlan,
    ) -> Result<(), Error> {
        self.record_sampling_input(input)
    }
    fn observe_sampling(
        &mut self,
        phase: eredu_runtime::working_memory::SamplingWorkspacePhase,
        report: &WorkspaceTraceReport,
    ) -> Result<(), Error> {
        self.record_sampling(phase, report, None)
    }
    fn observe_sampling_with_storage(
        &mut self,
        phase: eredu_runtime::working_memory::SamplingWorkspacePhase,
        report: &WorkspaceTraceReport,
        closing: WorkspaceStoragePopulation,
    ) -> Result<(), Error> {
        self.record_sampling(phase, report, Some(closing))
    }
}

struct ReducedTrace {
    carryover_complete: bool,
    validation_producers: Option<CertifiedSpanStorage>,
    traversal: Option<safemlx::OperationEvalTraversalLayout>,
    mutable_storage: Option<CertifiedSpanStorage>,
    first_missing_operation: Option<usize>,
    missing_operation_detail: Option<String>,
    unqualified_kernel_owner: Option<CustomKernelOwner>,
    maximum_rank: usize,
    host_primitive_nodes: usize,
    validation_roots: usize,
    grouped_outputs: GroupedOutputStorage,
    query_controls: Option<usize>,
    graph: Option<safemlx::ResidentGraphLayout>,
    dispatch: Option<ResidentDispatchPopulation>,
    pub(crate) nested_completions: usize,
    nested_root_capacity: usize,
}
impl ResidentRecipeRecorder {
    fn projection_population_bytes(
        &self,
        operation: WorkspaceOperationView<'_>,
        emitted: u64,
    ) -> Result<u64, Error> {
        let WorkspaceOperationKindView::Projection(format) = operation.kind else {
            return Ok(emitted);
        };
        let parameters = match format.encoding() {
            eredu_checkpoint::LinearFormat::Affine(_) => 4,
            eredu_checkpoint::LinearFormat::MxFp4 => 3,
            eredu_checkpoint::LinearFormat::GgufIQuant { .. } => 2,
            _ => return Ok(emitted),
        };
        // PhysicalLinear chooses its floating branch before inspecting packed
        // companions. A published floating replacement can therefore use dense
        // matmul/custom row projection while its neutral parameter topology is
        // retained. Bound that actual alternate using its logical [N,K] shape;
        // never reinterpret packed physical bytes as a dense copy capacity.
        let input = operation.inputs.get(0).expect("validated projection input");
        let output = operation
            .outputs
            .get(0)
            .expect("validated projection output");
        let k = *input.shape().last().expect("validated projection rank");
        let n = *output.shape().last().expect("validated projection rank");
        let allocation = self.mechanism.allocation();
        let capacity = |elements: u64| -> Result<u64, Error> {
            allocation.buffer_capacity(
                elements
                    .checked_mul(4)
                    .ok_or_else(|| self.metadata_error("dense replacement capacity overflow"))?,
            )
        };
        let sum = |a: u64, b: u64| {
            a.checked_add(b)
                .ok_or_else(|| self.metadata_error("dense replacement population overflow"))
        };
        let result = capacity(output.elements()?)?;
        let ordinary = matrix::matmul_cost(input.shape(), &[k, n], allocation)?;
        let weight_elements = (k as u64)
            .checked_mul(n as u64)
            .ok_or_else(|| self.metadata_error("dense replacement weight overflow"))?;
        let custom = sum(
            sum(capacity(input.elements()?)?, capacity(weight_elements)?)?,
            sum(sum(result, result)?, capacity(1)?)?,
        )?;
        let mut alternate = ordinary.max(custom);
        if operation.inputs.len() > parameters {
            alternate = sum(alternate, sum(sum(result, result)?, capacity(n as u64)?)?)?;
        }
        Ok(emitted.max(alternate))
    }

    fn reduce_trace(
        &self,
        report: &WorkspaceTraceReport,
        input_operation: Option<usize>,
        retained_roots: usize,
        output_roots: usize,
    ) -> Result<ReducedTrace, Error> {
        self.reduce_trace_with_parallel(report, input_operation, retained_roots, output_roots, None)
    }
    fn reduce_trace_with_parallel(
        &self,
        report: &WorkspaceTraceReport,
        input_operation: Option<usize>,
        retained_roots: usize,
        output_roots: usize,
        parallel: Option<&parallel::OriginalParallelInvocation>,
    ) -> Result<ReducedTrace, Error> {
        if let Some(cpu) = self.cpu {
            return self.reduce_cpu_trace_with_parallel(
                report,
                input_operation,
                retained_roots,
                output_roots,
                cpu,
                parallel,
            );
        }
        let (
            mut arrays,
            mut tape,
            mut edges,
            mut validations,
            mut maximum_rank,
            mut births,
            mut seeds,
        ) = (1usize, 1usize, 0usize, 0usize, 0usize, 0usize, 0usize);
        let mut streams = 1usize;
        let mut additional_shells = report.tensor_handle_clones;
        // Component fixtures contribute an actual borrowed cold trace before
        // the ordinary first-row arena/P/cache selection and original admission.
        #[cfg(all(
            test,
            target_vendor = "apple",
            feature = "metal",
            not(feature = "cuda")
        ))]
        let component =
            original_component_tests::current(input_operation.is_some() && self.records.is_empty());
        // Injected component traces contribute real construction but their
        // separately owned roots are absent from the enclosing state/output
        // census. Preserve the all-generation fallback for that source.
        #[cfg(all(
            test,
            target_vendor = "apple",
            feature = "metal",
            not(feature = "cuda")
        ))]
        let carryover_complete = component.is_none();
        #[cfg(not(all(
            test,
            target_vendor = "apple",
            feature = "metal",
            not(feature = "cuda")
        )))]
        let carryover_complete = true;
        #[cfg(all(
            test,
            target_vendor = "apple",
            feature = "metal",
            not(feature = "cuda")
        ))]
        if let Some(component) = &component {
            additional_shells =
                additional_shells.and_then(|n| n.checked_add(component.tensor_handle_clones?));
        }
        let operations = report.operations.iter();
        #[cfg(all(
            test,
            target_vendor = "apple",
            feature = "metal",
            not(feature = "cuda")
        ))]
        let operations = operations.chain(component.iter().flat_map(|r| r.operations.iter()));
        // Only the complete adaptive step performs the intermediate token
        // observation. Cutoff itself is a pure graph constructor, also reused
        // by deterministic speculative processing without any scalar read.
        let mut nested_completions = usize::from(sampling_mirostat::is_adaptive_step(report));
        let mut nested_root_capacity = report
            .operations
            .iter()
            .filter(|op| {
                matches!(
                    op.kind,
                    WorkspaceOperationKind::ValueCompletion
                        | WorkspaceOperationKind::CommunicationDependencies
                        | WorkspaceOperationKind::CachePublicationCompletion
                        | WorkspaceOperationKind::CacheScanCompletion
                )
            })
            .map(|op| op.inputs.len())
            .max()
            .unwrap_or(0);

        #[cfg(all(
            test,
            target_vendor = "apple",
            feature = "metal",
            not(feature = "cuda")
        ))]
        if component.is_some() {
            if let Some(program) = original_component_tests::summary() {
                nested_completions = add(
                    nested_completions,
                    program
                        .population()
                        .map_err(Error::backend_retained_source)?
                        .scalar_completions,
                )?;
            }
        }
        let mut complete_worker_profile = true;
        let mut cpu_partitions = 0usize;
        let mut parallel_entries = 0usize;
        let mut parallel_births = 0usize;
        let mut boundary_births = 0usize;
        let mut parallel_graph_extents = 0usize;
        let mut additional_sort_kernels = Some(0usize);
        let mut maximum_operands = 4usize;
        let mut worker_rank = 0usize;
        let mut copy_rank_extents = 0usize;
        let mut bytes = 0u64;
        let mut validation_bytes = 0u64;
        let mut validation_births = 0usize;
        let mut operation_controls = 0usize;
        let mut grouped_outputs = GroupedOutputStorage::default();
        let mut missing = None;
        let mut missing_operation_detail = None;
        let mut unqualified_kernel_owner = None;
        for (index, op) in operations.enumerate() {
            if matches!(op.kind, WorkspaceOperationKind::CommunicationControl(_)) {
                // The retained control source prices the agreement child; this
                // trace marker creates no model graph node or numerical root.
                if !op.inputs.is_empty() || !op.outputs.is_empty() {
                    missing.get_or_insert(index);
                }
                continue;
            }
            maximum_rank = maximum_rank.max(
                op.inputs
                    .iter()
                    .chain(&op.outputs)
                    .map(|x| x.shape().len())
                    .max()
                    .unwrap_or(0),
            );
            if matches!(
                op.kind,
                WorkspaceOperationKind::Matmul
                    | WorkspaceOperationKind::DenseLinear
                    | WorkspaceOperationKind::Projection(_)
            ) {
                // Vector matmul promotes both operands to rank two even when
                // the final result is scalar. Every product wrapper uses at
                // least that intermediate rank before final squeeze/reshape.
                maximum_rank = maximum_rank.max(2);
            }
            if matches!(
                op.kind,
                WorkspaceOperationKind::Embedding(..)
                    | WorkspaceOperationKind::VocabularyParallelLookup { .. }
            ) {
                // Native gather retains its indexed singleton axis until squeeze.
                maximum_rank = maximum_rank.max(add(op.outputs[0].shape().len(), 1)?);
            }
            // Only the actual architecture-created placeholder represents the
            // caller's prepared input. Other Initialize operations are not skipped.
            if Some(index) == input_operation {
                arrays = add(arrays, 1)?;
                continue;
            }
            if self.layerwise_constructors.is_some()
                && matches!(op.kind, WorkspaceOperationKind::ParameterPlaceholder)
            {
                // This exact trace was compared with the retained source above.
                // Replaced constructor nodes never enter the neural Eval DAG.
                // The same source adds their seeds/Full graph once later, through
                // bind_layerwise_parameter_constructors, before native admission.
                continue;
            }
            if matches!(op.kind, WorkspaceOperationKind::AddressableRegion(_)) {
                let Some(source) = self.addressable_sources.as_ref() else {
                    missing.get_or_insert(index);
                    continue;
                };
                let quote = source.quote(op.as_view())?;
                bytes = bytes
                    .checked_add(
                        u64::try_from(quote.capacity.backing)
                            .map_err(|_| self.metadata_error("addressable backing overflow"))?,
                    )
                    .ok_or_else(|| self.metadata_error("addressable backing overflow"))?;
                boundary_births = add(boundary_births, quote.numerical.storage.maximum_births())?;
                arrays = add(arrays, add(op.inputs.len(), op.outputs.len())?)?;
                // The shared native parent worker completes the four original
                // operands together before opening this one distinct role.
                nested_completions = add(nested_completions, 1)?;
                nested_root_capacity = nested_root_capacity.max(4);
                operation_controls = add(
                    operation_controls,
                    crate::backend::runtime::cache::value_completion_control_bytes(4).ok_or_else(
                        || self.metadata_error("addressable parent completion has no source"),
                    )?,
                )?;
                continue;
            }
            if matches!(op.kind, WorkspaceOperationKind::ExpertProviderWave(_)) {
                let Some(wave) = parallel.and_then(|p| p.expert_provider_wave_occurrence(index))
                else {
                    missing.get_or_insert(index);
                    continue;
                };
                bytes = bytes
                    .checked_add(wave.provider.backing)
                    .ok_or_else(|| self.metadata_error("provider wave backing overflow"))?;
                boundary_births = add(boundary_births, wave.provider.births)?;
                continue;
            }
            if matches!(
                op.kind,
                WorkspaceOperationKind::ExpertRegion(_)
                    | WorkspaceOperationKind::ExpertInactiveWave(_)
            ) {
                let Some(region) = parallel.and_then(|p| p.expert_aggregate(index)) else {
                    missing.get_or_insert(index);
                    continue;
                };
                bytes = bytes
                    .checked_add(region.child_bytes)
                    .ok_or_else(|| self.metadata_error("expert region backing overflow"))?;
                boundary_births = add(boundary_births, region.child_births)?;
                nested_completions = add(nested_completions, region.parent_completions)?;
                if region.indexed_parent {
                    nested_root_capacity = nested_root_capacity.max(4);
                    arrays = add(arrays, 4)?;
                    operation_controls = add(
                        operation_controls,
                        crate::backend::runtime::cache::value_completion_control_bytes(4)
                            .ok_or_else(|| {
                                self.metadata_error(
                                    "indexed expert parent completion has no source",
                                )
                            })?,
                    )?;
                }
                arrays = add(arrays, op.outputs.len())?;
                for (slice, n) in std::iter::once((&region.empty_slice, region.empty_slices))
                    .chain(region.extra_parents.iter().map(|operation| (operation, 1)))
                {
                    let Some(lowering) = lowering(slice.as_view()) else {
                        missing.get_or_insert(index);
                        continue;
                    };
                    let repeated = |value: usize| {
                        value.checked_mul(n).ok_or_else(|| {
                            self.metadata_error("expert empty source population overflow")
                        })
                    };
                    additional_shells = additional_shells.and_then(|v| {
                        v.checked_add(
                            backend_handle_shells(slice.as_view(), lowering)?.checked_mul(n)?,
                        )
                    });
                    complete_worker_profile &=
                        resident_worker_profile(slice.as_view().kind, lowering);
                    let Some(copy) =
                        copy_rank::inspect(slice.as_view(), lowering.intermediate_rank)
                    else {
                        missing.get_or_insert(index);
                        continue;
                    };
                    worker_rank = worker_rank.max(copy.worker_rank);
                    copy_rank_extents = add(copy_rank_extents, repeated(copy.extra_extents)?)?;
                    operation_controls = add(
                        operation_controls,
                        repeated(add(copy.controls, lowering.helper_controls)?)?,
                    )?;
                    let mut sink = facts::Emitter::count();
                    let Some(fact) = self
                        .mechanism
                        .emit(slice.as_view(), &mut sink)
                        .map_err(MlxWorkspaceFactError::ordinary)?
                    else {
                        missing.get_or_insert(index);
                        continue;
                    };
                    let slice_bytes = sink
                        .allocated_output_bytes()
                        .and_then(|v| v.checked_add(fact.scratch_bytes))
                        .and_then(|v| v.checked_mul(n as u64))
                        .ok_or_else(|| {
                            self.metadata_error("expert empty slice backing overflow")
                        })?;
                    bytes = bytes
                        .checked_add(slice_bytes)
                        .ok_or_else(|| self.metadata_error("expert parent backing overflow"))?;
                    arrays = add(
                        arrays,
                        repeated(add(
                            add(
                                add(lowering.primitives, lowering.seeds)?,
                                lowering.hidden_leaves,
                            )?,
                            slice.inputs.len(),
                        )?)?,
                    )?;
                    tape = add(tape, repeated(lowering.primitives)?)?;
                    edges = add(edges, repeated(lowering.edges)?)?;
                    births = add(births, repeated(lowering.maximum_births)?)?;
                    seeds = add(seeds, repeated(lowering.seeds)?)?;
                    maximum_rank = maximum_rank.max(lowering.intermediate_rank).max(2);
                    maximum_operands = maximum_operands.max(lowering.maximum_operands);
                    streams = streams.max(lowering.streams);
                    if lowering.validations != 0
                        || lowering.nested_completions != 0
                        || lowering.router_cpu_partitions != 0
                        || lowering.grouped_output_chunks != 0
                        || lowering.unqualified_kernel_owner.is_some()
                    {
                        missing.get_or_insert(index);
                    }
                    additional_sort_kernels = additional_sort_kernels.and_then(|v| {
                        v.checked_add(lowering.additional_sort_kernels?.checked_mul(n)?)
                    });
                }
                continue;
            }
            if matches!(op.kind, WorkspaceOperationKind::Collective(_)) {
                let Some(profile) = self.parallel_population(index, op, parallel)? else {
                    missing.get_or_insert(index);
                    continue;
                };
                bytes = bytes
                    .checked_add(profile.bytes)
                    .ok_or_else(|| self.metadata_error("parallel backing source overflow"))?;
                boundary_births = add(boundary_births, profile.child_births)?;
                nested_completions = add(nested_completions, profile.nested)?;
                parallel_entries = add(parallel_entries, profile.primitives)?;
                parallel_births = add(parallel_births, profile.births)?;
                parallel_graph_extents = add(parallel_graph_extents, profile.graph_extents)?;
                operation_controls = add(operation_controls, profile.controls)?;
                arrays = add(arrays, profile.arrays)?;
                tape = add(tape, profile.primitives)?;
                edges = add(edges, profile.edges)?;
                births = add(births, profile.births)?;
                streams = streams.max(profile.streams);
                continue;
            }
            let Some(mut lowering) = lowering(op.as_view()) else {
                if missing.is_none() {
                    let arguments = format_args!(
                        "{:?}; inputs: {:?}; router source ready: {}",
                        op.kind,
                        op.inputs,
                        crate::backend::managed_memory::router::is_ready()
                    );
                    missing_operation_detail = Some(match &self.context {
                        Some(context) => context.metadata_string(arguments)?,
                        None => arguments.to_string(),
                    });
                    missing = Some(index);
                }
                continue;
            };
            if !self.mechanism.allocation().original_storage {
                lowering.nested_completions = add(
                    lowering.nested_completions,
                    router::ordinary_predicate_completions(op.as_view()),
                )?;
            }
            additional_shells = additional_shells
                .and_then(|n| n.checked_add(backend_handle_shells(op.as_view(), lowering)?));
            nested_completions = add(nested_completions, lowering.nested_completions)?;
            operation_controls = add(operation_controls, lowering.helper_controls)?;
            complete_worker_profile &= resident_worker_profile(op.as_view().kind, lowering);
            if let Some(profile) = copy_rank::inspect(op.as_view(), lowering.intermediate_rank) {
                worker_rank = worker_rank.max(profile.worker_rank);
                copy_rank_extents = add(copy_rank_extents, profile.extra_extents)?;
                operation_controls = add(operation_controls, profile.controls)?;
            } else {
                complete_worker_profile = false;
            }
            cpu_partitions = add(cpu_partitions, lowering.router_cpu_partitions)?;
            additional_sort_kernels = additional_sort_kernels
                .and_then(|n| n.checked_add(lowering.additional_sort_kernels?));
            if matches!(
                op.kind,
                WorkspaceOperationKind::Matmul
                    | WorkspaceOperationKind::DenseLinear
                    | WorkspaceOperationKind::Projection(_)
            ) {
                worker_rank = worker_rank.max(2);
            }
            if matches!(
                op.kind,
                WorkspaceOperationKind::Embedding(..)
                    | WorkspaceOperationKind::VocabularyParallelLookup { .. }
            ) {
                worker_rank = worker_rank.max(add(op.outputs[0].shape().len(), 1)?);
            }
            if matches!(
                op.kind,
                WorkspaceOperationKind::Pad(eredu_nn::PadMode::Constant)
            ) {
                operation_controls = add(
                    operation_controls,
                    safemlx::ops::constant_pad_control_bytes(op.inputs[0].shape().len())
                        .expect("same closed inline pad lowering"),
                )?;
            }
            if matches!(op.kind, WorkspaceOperationKind::StaticSlice { .. }) {
                operation_controls = add(
                    operation_controls,
                    crate::tensor::narrow::control_bytes(op.inputs[0].shape().len()).ok_or_else(
                        || self.metadata_error("static slice caller controls overflow"),
                    )?,
                )?;
            }
            if super::host_array::dtype(op.as_view()).is_some()
                || super::host_array::slice_dtype(op.as_view()).is_some()
            {
                operation_controls = add(
                    operation_controls,
                    (if super::host_array::slice_dtype(op.as_view()).is_some() {
                        super::host_array::slice_control_bytes()
                    } else {
                        super::host_array::control_bytes()
                    })
                    .ok_or_else(|| {
                        self.metadata_error("typed host-array caller controls overflow")
                    })?,
                )?;
            }
            if super::zero_fill::dtype(op.as_view()).is_some() {
                operation_controls = add(
                    operation_controls,
                    super::zero_fill::operation_control_bytes(op.as_view()).ok_or_else(|| {
                        self.metadata_error("typed scalar fill caller controls overflow")
                    })?,
                )?;
            }
            if basic::is_scalar_u8(op.as_view()) {
                operation_controls = add(
                    operation_controls,
                    basic::scalar_u8_control_bytes().ok_or_else(|| {
                        self.metadata_error("eager byte scalar controls overflow")
                    })?,
                )?;
            }
            if basic::is_scalar_f32(op.as_view()) {
                operation_controls = add(
                    operation_controls,
                    basic::scalar_f32_control_bytes()
                        .ok_or_else(|| self.metadata_error("eager scalar controls overflow"))?,
                )?;
            }
            if matches!(op.kind, WorkspaceOperationKind::ValueRetention) {
                let bytes = crate::backend::submission_recovery::prefill::TransientRootsProjection::control_bytes()
                    .and_then(|bytes| bytes.checked_mul(op.inputs.len()));
                if let Some(bytes) = bytes {
                    operation_controls = add(operation_controls, bytes)?;
                } else {
                    missing.get_or_insert(index);
                }
            }
            if matches!(op.kind, WorkspaceOperationKind::HostStoreFloating(..)) {
                if let Some(controls) = crate::backend::submission_recovery::prefill::TransientRootsProjection::control_bytes() {
                    operation_controls = add(operation_controls, controls)?;
                } else {
                    missing.get_or_insert(index);
                }
            }
            if matches!(
                op.kind,
                WorkspaceOperationKind::ValueCompletion
                    | WorkspaceOperationKind::CommunicationDependencies
                    | WorkspaceOperationKind::CachePublicationCompletion
                    | WorkspaceOperationKind::CacheScanCompletion
            ) {
                let Some(controls) =
                    crate::backend::nn::shared::value_completion_control_bytes(op.inputs.len())
                else {
                    missing.get_or_insert(index);
                    continue;
                };
                operation_controls = add(operation_controls, controls)?;
            }
            if matches!(op.kind, WorkspaceOperationKind::Rotary(..)) {
                let Some(controls) = rotary::control_bytes(op.as_view()) else {
                    missing.get_or_insert(index);
                    continue;
                };
                operation_controls = add(operation_controls, controls)?;
            }
            if matches!(op.kind, WorkspaceOperationKind::PreparedMultiAxisRotary(_)) {
                let Some(controls) = prepared_rotary::control_bytes(op.as_view()) else {
                    missing.get_or_insert(index);
                    continue;
                };
                operation_controls = add(operation_controls, controls)?;
            }
            if matches!(&op.kind, WorkspaceOperationKind::Projection(format)
                | WorkspaceOperationKind::ProjectionPrepare(format)
                | WorkspaceOperationKind::ProjectionFinish(format)
                if matches!(format.encoding(), eredu_checkpoint::LinearFormat::E4M3BlockFp8(_)))
            {
                let Some(controls) = fp8::control_bytes(op.as_view()) else {
                    missing.get_or_insert(index);
                    continue;
                };
                operation_controls = add(operation_controls, controls)?;
            }
            if matches!(op.as_view().kind, WorkspaceOperationKindView::Grouped {
                bank: WorkspaceGroupedBank::Linear(spec), ..
            } if matches!(spec.projection().format().encoding(), eredu_checkpoint::LinearFormat::E4M3BlockFp8(_)))
            {
                let Some(controls) = fp8::grouped_control_bytes() else {
                    missing.get_or_insert(index);
                    continue;
                };
                operation_controls = add(operation_controls, controls)?;
            }
            if grouped_packed::uses_affine_source(op.as_view()) {
                let Some(controls) = grouped_packed::affine_control_bytes(op.as_view()) else {
                    missing.get_or_insert(index);
                    continue;
                };
                operation_controls = add(operation_controls, controls)?;
            }
            if matches!(op.as_view().kind, WorkspaceOperationKindView::Grouped {
                bank: WorkspaceGroupedBank::GatedProduct(spec), ..
            } if matches!(spec.layout(), eredu_nn::GatedProductGroupLayout::Packed { gate_up, down }
                if [gate_up, down].iter().any(|p| p.format().encoding() == eredu_checkpoint::LinearFormat::MxFp4)))
            {
                let Some(controls) = grouped_packed::mxfp4_control_bytes(op.as_view()) else {
                    missing.get_or_insert(index);
                    continue;
                };
                operation_controls = add(operation_controls, controls)?;
            }
            if matches!(op.as_view().kind, WorkspaceOperationKindView::Grouped {
                bank: WorkspaceGroupedBank::GatedProduct(spec), ..
            } if matches!(spec.layout(), eredu_nn::GatedProductGroupLayout::Packed { gate_up, .. }
                if matches!(gate_up.format().encoding(), eredu_checkpoint::LinearFormat::E4M3BlockFp8(_))))
            {
                let Some(controls) = fp8::packed_control_bytes(op.as_view()) else {
                    missing.get_or_insert(index);
                    continue;
                };
                operation_controls = add(operation_controls, controls)?;
            }
            if matches!(
                op.as_view().kind,
                WorkspaceOperationKindView::Grouped {
                    bank: WorkspaceGroupedBank::Relu2(_),
                    ..
                }
            ) {
                let Some(controls) =
                    super::super::shared::MlxGroupedRelu2::original_control_bytes()
                else {
                    missing.get_or_insert(index);
                    continue;
                };
                operation_controls = add(operation_controls, controls)?;
            }
            if unqualified_kernel_owner.is_none() {
                unqualified_kernel_owner = lowering.unqualified_kernel_owner;
            }
            if matches!(
                op.kind,
                WorkspaceOperationKind::Sampling(
                    eredu_nn::workspace::WorkspaceSamplingOperation::CreateRandomKey
                        | eredu_nn::workspace::WorkspaceSamplingOperation::SelectRandomKey { .. }
                        | eredu_nn::workspace::WorkspaceSamplingOperation::SplitRandomKey
                        | eredu_nn::workspace::WorkspaceSamplingOperation::UniformUnitInterval
                        | eredu_nn::workspace::WorkspaceSamplingOperation::Categorical
                )
            ) {
                let Some(controls) = crate::backend::random::standard_sampling_control_bytes(
                    safemlx::DeviceType::Gpu,
                ) else {
                    missing.get_or_insert(index);
                    continue;
                };
                operation_controls = add(operation_controls, controls)?;
            }
            if matches!(
                op.kind,
                WorkspaceOperationKind::VocabularyParallelLookup { .. }
            ) {
                let Some(controls) =
                    crate::backend::nn::shared::parallel_vocabulary_lookup_control_bytes()
                else {
                    missing.get_or_insert(index);
                    continue;
                };
                operation_controls = add(operation_controls, controls)?;
            }
            if let Some(format) = match &op.kind {
                WorkspaceOperationKind::Embedding(format, _)
                | WorkspaceOperationKind::VocabularyParallelLookup { format, .. } => Some(format),
                _ => None,
            } {
                if matches!(
                    format.encoding(),
                    eredu_checkpoint::LinearFormat::Affine(_)
                        | eredu_checkpoint::LinearFormat::MxFp4
                ) {
                    let Some(controls) = crate::nn::QuantizedEmbedding::forward_control_bytes(
                        op.inputs[0].shape().len(),
                    ) else {
                        missing.get_or_insert(index);
                        continue;
                    };
                    operation_controls = add(operation_controls, controls)?;
                }
            }
            if lowering.pointwise_calls != 0 {
                if op
                    .inputs
                    .first()
                    .is_some_and(|input| input.elements().is_ok_and(|n| n != 0))
                {
                    if let Some(owner) = pointwise_source_requirement() {
                        unqualified_kernel_owner.get_or_insert(owner);
                    }
                }
                // The fixed invocation borrows the once-admitted source family.
                // Its exact slot allocation is independent of request Graphs.
                let Some(per_call) = super::super::arithmetic::f32_pointwise_control_bytes(
                    op.inputs
                        .iter()
                        .chain(&op.outputs)
                        .map(|v| v.shape().len())
                        .max()
                        .unwrap_or(0),
                ) else {
                    missing.get_or_insert(index);
                    continue;
                };
                operation_controls = add(
                    operation_controls,
                    per_call
                        .checked_mul(lowering.pointwise_calls)
                        .ok_or_else(|| Error::backend("pointwise controls overflow"))?,
                )?;
            }
            if attention_direct::selected(op.as_view()) || attention_tiled::selected(op.as_view()) {
                let Some(controls) = attention_tiled::control_bytes(op.as_view())
                    .or_else(|| attention_direct::control_bytes(op.as_view()))
                else {
                    missing.get_or_insert(index);
                    continue;
                };
                operation_controls = add(operation_controls, controls)?;
            }
            if matches!(
                op.as_view().kind,
                WorkspaceOperationKindView::Grouped {
                    bank: WorkspaceGroupedBank::GatedProduct(_),
                    phase: WorkspaceGroupedPhase::Whole | WorkspaceGroupedPhase::Units,
                    partitions: Some(_),
                }
            ) {
                let Some(controls) = super::super::shared::MlxGroupedGatedProduct::original_tensor_parallel_control_bytes() else {
                    missing.get_or_insert(index);
                    continue;
                };
                operation_controls = add(operation_controls, controls)?;
            }
            if lowering.grouped_unit_observers != 0 {
                let Some(controls) = super::super::shared::MlxGroupedGatedProduct::original_unit_observation_control_bytes() else {
                    missing.get_or_insert(index);
                    continue;
                };
                operation_controls = add(operation_controls, controls)?;
            }
            if lowering.row_rms_calls != 0 {
                if !crate::backend::managed_memory::row_kernels::rms_source_qualified() {
                    unqualified_kernel_owner.get_or_insert(CustomKernelOwner::RowRms);
                }
                let rank = maximum_rank.max(lowering.intermediate_rank);
                let Some(per_call) =
                    crate::backend::managed_memory::row_kernels::rms_control_bytes(rank)
                else {
                    missing.get_or_insert(index);
                    continue;
                };
                operation_controls = add(
                    operation_controls,
                    per_call
                        .checked_mul(lowering.row_rms_calls)
                        .ok_or_else(|| Error::backend("row RMS controls overflow"))?,
                )?;
            }
            if lowering.row_sum_calls != 0 {
                if !crate::backend::managed_memory::row_kernels::sum_source_qualified() {
                    unqualified_kernel_owner.get_or_insert(CustomKernelOwner::RowSum);
                }
                let Some(per_call) =
                    crate::backend::managed_memory::row_kernels::sum_control_bytes(2)
                else {
                    missing.get_or_insert(index);
                    continue;
                };
                operation_controls = add(
                    operation_controls,
                    per_call
                        .checked_mul(lowering.row_sum_calls)
                        .ok_or_else(|| Error::backend("row sum controls overflow"))?,
                )?;
            }
            if lowering.recurrent_calls != 0 {
                if !crate::backend::managed_memory::recurrent_kernel::source_qualified() {
                    unqualified_kernel_owner.get_or_insert(CustomKernelOwner::RecurrentScan);
                }
                let Some(per_call) =
                    crate::backend::managed_memory::recurrent_kernel::control_bytes()
                else {
                    missing.get_or_insert(index);
                    continue;
                };
                operation_controls = add(
                    operation_controls,
                    per_call
                        .checked_mul(lowering.recurrent_calls)
                        .ok_or_else(|| Error::backend("recurrent kernel controls overflow"))?,
                )?;
            }
            if lowering.bf16_projection_calls != 0 {
                let Some(per_call) = super::super::matrix::grouped_projection_control_bytes()
                else {
                    missing.get_or_insert(index);
                    continue;
                };
                operation_controls = add(
                    operation_controls,
                    per_call
                        .checked_mul(lowering.bf16_projection_calls)
                        .ok_or_else(|| Error::backend("grouped helper controls overflow"))?,
                )?;
            }
            if lowering.router_cpu_partitions != 0 {
                let controls = safemlx::OperationEvent::cpu_argpartition_layout(false)
                    .and_then(|cpu| cpu.control_bytes())
                    .and_then(|n| {
                        n.checked_add(crate::backend::managed_memory::router::control_bytes()?)
                    });
                let Some(controls) = controls else {
                    missing.get_or_insert(index);
                    continue;
                };
                operation_controls = add(
                    operation_controls,
                    controls
                        .checked_mul(lowering.router_cpu_partitions)
                        .ok_or_else(|| Error::backend("router control population overflow"))?,
                )?;
            }
            let mut sink = facts::Emitter::count();
            let Some(fact) = self
                .mechanism
                .emit(op.as_view(), &mut sink)
                .map_err(MlxWorkspaceFactError::ordinary)?
            else {
                missing.get_or_insert(index);
                continue;
            };
            // These closed operations consume existing values without creating
            // outputs. Their emitter validates nonempty roots and zero outputs;
            // retain their separately counted roots/completions and compare the
            // same exact population instead of inventing a dummy tensor.
            let input_only = matches!(
                op.kind,
                WorkspaceOperationKind::ValueRetention
                    | WorkspaceOperationKind::ValueCompletion
                    | WorkspaceOperationKind::CommunicationDependencies
                    | WorkspaceOperationKind::CachePublicationCompletion
                    | WorkspaceOperationKind::CacheScanCompletion
                    | WorkspaceOperationKind::HostStoreFloating(..)
            ) && !op.inputs.is_empty()
                && op.outputs.is_empty();
            if fact.layout.outputs != op.outputs.len() || (fact.layout.outputs == 0 && !input_only)
            {
                return Err(Error::backend(
                    "closed lowering output population differs from its trace",
                ));
            }
            let allocated = sink
                .allocated_output_bytes()
                .ok_or_else(|| Error::backend("resident native output population overflow"))?;
            let operation_bytes = allocated
                .checked_add(fact.scratch_bytes)
                .ok_or_else(|| Error::backend("resident native byte population overflow"))?;
            let operation_bytes =
                self.projection_population_bytes(op.as_view(), operation_bytes)?;
            // Retained validation arrays are hidden native roots. Conservatively
            // retain this actual producer's entire buffer population; scalar
            // result geometry is never substituted for its full backing.
            if lowering.validations != 0 {
                validation_bytes =
                    validation_bytes
                        .checked_add(operation_bytes)
                        .ok_or_else(|| {
                            Error::backend("validation producer byte population overflow")
                        })?;
                validation_births = add(validation_births, lowering.maximum_births)?;
            }
            bytes = bytes
                .checked_add(operation_bytes)
                .ok_or_else(|| Error::backend("resident native byte population overflow"))?;
            arrays = add(
                arrays,
                add(
                    add(
                        add(lowering.primitives, lowering.seeds)?,
                        lowering.hidden_leaves,
                    )?,
                    op.inputs.len(),
                )?,
            )?;
            tape = add(tape, lowering.primitives)?;
            streams = streams.max(lowering.streams);
            maximum_rank = maximum_rank.max(lowering.intermediate_rank);
            maximum_operands = maximum_operands.max(lowering.maximum_operands);
            edges = add(edges, lowering.edges)?;
            validations = add(validations, lowering.validations)?;
            births = add(births, lowering.maximum_births)?;
            grouped_outputs = grouped_outputs
                .merge(GroupedOutputStorage {
                    calls: if lowering.grouped_output_chunks != 0 {
                        lowering.grouped_output_calls
                    } else {
                        0
                    },
                    chunks: lowering.grouped_output_chunks,
                    unit_observers: lowering.grouped_unit_observers,
                    observer_shape_rank: 0,
                })
                .ok_or_else(|| Error::backend("grouped output storage overflow"))?;
            seeds = add(seeds, lowering.seeds)?;
        }
        if missing_operation_detail.is_none() {
            if let Some(operation) = missing.and_then(|index| report.operations.get(index)) {
                let arguments = format_args!(
                    "{:?}; inputs: {:?}; outputs: {:?}",
                    operation.kind, operation.inputs, operation.outputs
                );
                missing_operation_detail = Some(match &self.context {
                    Some(context) => context.metadata_string(arguments)?,
                    None => arguments.to_string(),
                });
            }
        }
        if grouped_outputs.unit_observers != 0 {
            grouped_outputs.observer_shape_rank = maximum_rank;
        }
        let transient_roots = report
            .operations
            .iter()
            .try_fold(0usize, |count, operation| {
                if matches!(operation.kind, WorkspaceOperationKind::ValueRetention) {
                    add(count, operation.inputs.len())
                } else if matches!(
                    operation.kind,
                    WorkspaceOperationKind::HostStoreFloating(..)
                ) {
                    add(count, 1)
                } else {
                    Ok(count)
                }
            })?;
        let roots = add(
            add(add(retained_roots, output_roots)?, validations)?,
            transient_roots,
        )?;
        // This source-derived ceiling covers both the final root list and every
        // nested 1/3-root list. Native completion tightens roots before reserve.
        let traversal_roots = if nested_completions != 0 {
            roots.max(3).max(nested_root_capacity)
        } else {
            roots
        };
        arrays = add(arrays, traversal_roots)?;
        edges = add(edges, traversal_roots)?;
        let mut query_controls = missing.is_some().then_some(0usize);
        // Router ties use the retained dedicated CPU source stream. Native
        // Ring collectives select the CPU communication stream independently.
        // Their individual two-stream profiles both include the model GPU, so
        // taking only their maximum incorrectly aliases those two CPU sources.
        // Preserve each actual source's table slot when both kinds occur in
        // this trace; worker/primitive populations remain separately counted.
        if cpu_partitions != 0 && parallel_entries != 0 {
            streams = add(streams, 1)?;
        }
        let qualified_streams = add(
            add(1, usize::from(cpu_partitions != 0))?,
            usize::from(parallel_entries != 0),
        )?;
        let traversal = if missing.is_none() {
            safemlx::OperationEvent::eval_record_layout(tape, streams, tape).and_then(|base| {
                query_controls = base.query_control_bytes();
                safemlx::OperationEvent::eval_traversal_layout(
                    safemlx::OperationEvalTraversalLimits {
                        roots,
                        arrays,
                        tape_entries: tape,
                        input_edges: edges,
                        output_slots: tape,
                        streams,
                        captures: base.capture_slots().max(1),
                    },
                )
            })
        } else {
            None
        };
        if let Some(layout) = traversal {
            query_controls =
                query_controls.and_then(|n| n.checked_add(layout.query_control_bytes()?));
        }
        let graph = if missing.is_none() {
            additional_shells.and_then(|shells| {
                safemlx::OperationEvent::resident_graph_layout_with_shells(
                    tape - 1,
                    seeds,
                    maximum_rank,
                    maximum_operands,
                    shells,
                )
            })
        } else {
            None
        };
        if let Some(graph) = graph {
            query_controls = query_controls.and_then(|n| {
                n.checked_add(graph.control_bytes()?)
                    .and_then(|n| n.checked_add(operation_controls))
            });
        } else if missing.is_none() {
            query_controls = None;
        }
        // Each crossed router contributes exactly one CPU ArgPartition with
        // one input and one output. Its fixed bank owns its Data/birth, task,
        // weak views, invocation vector and cleanup; exclude those from GPU.
        let cpu_entries = add(cpu_partitions, parallel_entries)?;
        let cpu_births = add(cpu_partitions, parallel_births)?;
        let gpu_population = tape
            .checked_sub(cpu_entries)
            .zip(edges.checked_sub(cpu_entries))
            .zip(births.checked_sub(cpu_births));
        let worker = if missing.is_none()
            && complete_worker_profile
            && streams == qualified_streams
            && graph.is_some()
        {
            additional_sort_kernels.and_then(|sorts| {
                let ((gpu_entries, gpu_edges), gpu_births) = gpu_population?;
                safemlx::OperationEvent::resident_gpu_worker_layout_with_router(
                    gpu_entries,
                    gpu_edges,
                    gpu_entries,
                    arrays,
                    gpu_births,
                    worker_rank,
                    maximum_operands,
                    sorts,
                    cpu_partitions,
                )
            })
        } else {
            None
        };
        if let Some(worker) = worker {
            query_controls = query_controls.and_then(|n| n.checked_add(worker.control_bytes()?));
        }
        let dispatch = worker.and_then(|worker| {
            Some(ResidentDispatchPopulation {
                cpu_model: None,
                // The Synchronizer remains on the selected GPU. Only the actual
                // source-declared rank-two partitions execute on the admitted CPU.
                gpu_entries: tape - cpu_entries,
                gpu_input_edges: edges - cpu_entries,
                gpu_siblings: tape - cpu_entries,
                gpu_births: births - cpu_births,
                additional_sort_kernels: additional_sort_kernels
                    .expect("qualified worker has a sort population"),
                cpu_entries,
                cpu_input_edges: cpu_entries,
                cpu_siblings: cpu_entries,
                parallel_entries,
                parallel_graph_extents,
                worker_graph_extents: worker
                    .allocation_extents()
                    .checked_add(copy_rank_extents)?
                    .checked_add(parallel_graph_extents)?,
                worker_rank,
                copy_rank_extents,
                kernel_attempts: worker.kernel_attempts(),
            })
        });
        if unqualified_kernel_owner.is_some() {
            // Partial helper controls cannot become a complete host quote.
            query_controls = None;
        }
        Ok(ReducedTrace {
            carryover_complete,
            validation_producers: missing.is_none().then_some(CertifiedSpanStorage {
                mutable_bytes: validation_bytes,
                maximum_births: validation_births,
            }),
            traversal,
            mutable_storage: missing.is_none().then_some(CertifiedSpanStorage {
                mutable_bytes: bytes,
                maximum_births: add(births, boundary_births)?,
            }),
            first_missing_operation: missing,
            missing_operation_detail,
            unqualified_kernel_owner,
            maximum_rank,
            host_primitive_nodes: tape - 1,
            validation_roots: validations,
            grouped_outputs,
            query_controls,
            graph,
            dispatch,
            nested_completions,
            nested_root_capacity,
        })
    }
}

#[cfg(all(
    test,
    target_vendor = "apple",
    feature = "metal",
    not(feature = "cuda")
))]
mod recurrent_tests {
    use super::*;
    use eredu_nn::{GatedDeltaScanInput, NeuralBackend, Tensor};

    #[test]
    fn recurrent_recipe_keeps_both_outputs_across_real_chunk_boundaries() {
        let mechanism = MlxMetalWorkspaceMechanisms::current_host().unwrap();
        let qualified = safemlx::OperationEvent::resident_graph_layout(1, 0, 4).is_some()
            && safemlx::OperationEvent::eval_record_layout(1, 1, 1).is_some();
        if std::env::var_os("EREDU_REQUIRE_QUALIFIED_RECORD_LAYOUT").is_some() {
            assert!(
                qualified,
                "selected validation requires the native layout qualification"
            );
        }
        for length in [1, 2, 64, 65, 256, 257] {
            for vector_decay in [false, true] {
                for supplied_state in [false, true] {
                    let context = WorkspaceContext::new(mechanism);
                    let q = WorkspaceTensor::unloaded_f32(&[2, length, 2, 3], &context).unwrap();
                    let k = WorkspaceTensor::unloaded_f32(&[2, length, 2, 3], &context).unwrap();
                    let v = WorkspaceTensor::unloaded_f32(&[2, length, 2, 5], &context).unwrap();
                    let decay_shape = [2, length, 2, 3];
                    let g = WorkspaceTensor::unloaded_f32(
                        &decay_shape[..if vector_decay { 4 } else { 3 }],
                        &context,
                    )
                    .unwrap();
                    let b = WorkspaceTensor::unloaded_f32(&[2, length, 2], &context).unwrap();
                    let state = WorkspaceTensor::unloaded_f32(&[2, 2, 3, 5], &context).unwrap();
                    context.begin_span();
                    let result = WorkspaceBackend::gated_delta_scan(
                        GatedDeltaScanInput {
                            query: &q,
                            key: &k,
                            value: &v,
                            log_decay: &g,
                            beta: &b,
                            initial_state: supplied_state.then_some(&state),
                        },
                        &context,
                    )
                    .unwrap();
                    let report = context.report(&[result.state, result.output]).unwrap();
                    assert_eq!(report.operations.len(), 1);
                    let operation = &report.operations[0];
                    let bound = mechanism.operation_bound(operation).unwrap().unwrap();
                    assert_eq!(bound.outputs.len(), 2);
                    let output_bytes = bound
                        .outputs
                        .iter()
                        .map(|effect| match effect {
                            WorkspaceOutputStorage::Allocate(bytes) => *bytes,
                            other => {
                                panic!("scan output is not an independent allocation: {other:?}")
                            }
                        })
                        .sum::<u64>();
                    let recorder = ResidentRecipeRecorder::new(
                        InferenceGeometry {
                            batch_size: 2,
                            cached_positions: 0,
                            input_positions: length as u64,
                            max_output_tokens: 1,
                            prefill_chunk_positions: length as u64,
                            output: eredu_core::OutputDemand::Sequence,
                        },
                        mechanism,
                    );
                    let reduced = recorder.reduce_trace(&report, None, 0, 2).unwrap();
                    if crate::backend::managed_memory::recurrent_kernel::control_bytes().is_none() {
                        assert_eq!(reduced.first_missing_operation, Some(0));
                        assert!(reduced.dispatch.is_none());
                        continue;
                    }
                    assert_eq!(reduced.first_missing_operation, None);
                    assert_eq!(
                        reduced.mutable_storage.unwrap().mutable_bytes(),
                        output_bytes + bound.scratch_bytes
                    );
                    if qualified {
                        assert_eq!(reduced.traversal.unwrap().roots(), 2);
                        assert!(reduced.graph.unwrap().maximum_operands() >= 6);
                        assert!(
                            reduced.dispatch.is_some(),
                            "both outputs and six possible compactions have native worker owners"
                        );
                    } else {
                        // Physical facts do not manufacture a native layout witness.
                        assert!(reduced.graph.is_none() || reduced.traversal.is_none());
                    }
                    // A matching geometry/output count cannot certify an unrecognized worker.
                    let mut unknown = report;
                    unknown.operations[0].kind =
                        WorkspaceOperationKind::Elementwise("unrecognized_scan");
                    assert!(recorder
                        .reduce_trace(&unknown, None, 0, 2)
                        .unwrap()
                        .mutable_storage
                        .is_none());
                }
            }
        }
    }
}

#[cfg(all(
    test,
    target_vendor = "apple",
    feature = "metal",
    not(feature = "cuda")
))]
mod depthwise_tests {
    use super::*;
    use eredu_nn::Tensor;
    use safemlx::{Array, Device, DeviceType, Stream};

    #[test]
    fn depthwise_convolution_noncontiguous_inputs_match_scalar_reference() {
        let mechanism = MlxMetalWorkspaceMechanisms::current_host().unwrap();
        let context = WorkspaceContext::new(mechanism);
        let input = WorkspaceTensor::unloaded_f32(&[2, 7, 3], &context).unwrap();
        let weight = WorkspaceTensor::unloaded_f32(&[3, 3, 1], &context).unwrap();
        context.begin_span();
        let output = WorkspaceTensor::conv1d(&input, &weight, 1, 0, 1, 3, &context).unwrap();
        let report = context.report(&[output]).unwrap();
        assert_eq!(report.operations.len(), 1);
        let op = &report.operations[0];
        let lowered = lowering(op.as_view()).unwrap();
        assert_eq!(
            (lowered.primitives, lowered.edges, lowered.maximum_births),
            (3, 4, 5)
        );
        let mut other = op.clone();
        other.kind = WorkspaceOperationKind::Convolution {
            stride: vec![1],
            padding: vec![1],
            dilation: vec![1],
            groups: 3,
            transposed: None,
        };
        assert!(
            lowering(other.as_view()).is_none(),
            "changed padding must not reuse the old output geometry"
        );
        let layout = safemlx::OperationEvent::resident_graph_layout(3, 0, 3);
        if std::env::var_os("EREDU_REQUIRE_QUALIFIED_RECORD_LAYOUT").is_some() {
            assert!(
                layout.is_some(),
                "selected validation requires native control qualification"
            );
        }

        if layout.is_some() {
            let recorder = ResidentRecipeRecorder::new(
                InferenceGeometry {
                    batch_size: 2,
                    cached_positions: 0,
                    input_positions: 7,
                    max_output_tokens: 1,
                    prefill_chunk_positions: 7,
                    output: eredu_core::OutputDemand::Sequence,
                },
                mechanism,
            );
            let reduced = recorder.reduce_trace(&report, None, 0, 1).unwrap();
            assert!(
                reduced.dispatch.is_some(),
                "separable worker includes both copies"
            );
            let mut unsupported = report.clone();
            unsupported.operations[0] = other;
            assert!(recorder
                .reduce_trace(&unsupported, None, 0, 1)
                .unwrap()
                .dispatch
                .is_none());
        }

        let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
        // Both views require the actual worker's independent row compactions.
        let input_values = (0..42)
            .map(|n| (n % 17) as f32 * 0.125 - 1.0)
            .collect::<Vec<_>>();
        let weight_values = (0..9).map(|n| n as f32 * 0.0625 - 0.25).collect::<Vec<_>>();
        let input = Array::from_slice(&input_values, &[2, 3, 7])
            .swap_axes(1, 2, &stream)
            .unwrap();
        let weight = Array::from_slice(&weight_values, &[3, 3, 1])
            .swap_axes(0, 1, &stream)
            .unwrap();
        let output = safemlx::ops::conv1d(&input, &weight, 1, 0, 1, 3, &stream).unwrap();
        let output = output.evaluated().unwrap();
        let values = output.as_slice::<f32>();
        let mut expected = Vec::new();
        for batch in 0..2 {
            for position in 0..5 {
                for channel in 0..3 {
                    expected.push(
                        (0..3)
                            .map(|tap| {
                                input_values[(batch * 3 + channel) * 7 + position + tap]
                                    * weight_values[tap * 3 + channel]
                            })
                            .sum::<f32>(),
                    );
                }
            }
        }
        assert_eq!(values.len(), expected.len());
        assert!(expected.iter().any(|n| n.abs() > 0.1));
        assert!(values
            .iter()
            .zip(expected)
            .all(|(actual, expected)| (actual - expected).abs() < 1e-5));
    }
}

#[cfg(all(
    test,
    target_vendor = "apple",
    feature = "metal",
    not(feature = "cuda")
))]
mod constant_pad_tests {
    use super::*;
    use eredu_nn::{PadMode, Tensor};

    #[test]
    fn constant_pad_recipe_consumes_inline_controls_and_keeps_edge_unknown() {
        let mechanism = MlxMetalWorkspaceMechanisms::current_host().unwrap();
        let context = WorkspaceContext::new(mechanism);
        let input = WorkspaceTensor::unloaded_f32(&[2, 4, 3], &context).unwrap();
        context.begin_span();
        let output = WorkspaceTensor::pad(
            &input,
            &[(1, 0), (2, 1), (0, 2)],
            PadMode::Constant,
            &context,
        )
        .unwrap();
        let report = context.report(&[output]).unwrap();
        assert_eq!(report.operations.len(), 1);
        let lowered = lowering(report.operations[0].as_view()).unwrap();
        assert_eq!(
            (lowered.primitives, lowered.edges, lowered.seeds),
            (3, 4, 1)
        );
        let recorder = ResidentRecipeRecorder::new(
            InferenceGeometry {
                batch_size: 2,
                cached_positions: 0,
                input_positions: 4,
                max_output_tokens: 1,
                prefill_chunk_positions: 4,
                output: eredu_core::OutputDemand::Sequence,
            },
            mechanism,
        );
        let reduced = recorder.reduce_trace(&report, None, 0, 1).unwrap();
        assert_eq!(reduced.first_missing_operation, None);
        assert!(reduced.mutable_storage.unwrap().mutable_bytes() > 0);
        let qualified = safemlx::OperationEvent::resident_graph_layout(3, 1, 3).is_some()
            && safemlx::OperationEvent::eval_record_layout(4, 1, 4).is_some();
        if std::env::var_os("EREDU_REQUIRE_QUALIFIED_RECORD_LAYOUT").is_some() {
            assert!(
                qualified,
                "selected validation requires native layout qualification"
            );
        }
        if qualified {
            assert!(reduced.graph.is_some() && reduced.traversal.is_some());
            assert!(
                reduced.query_controls.unwrap()
                    >= safemlx::ops::constant_pad_control_bytes(3).unwrap()
            );
        }
        let mut edge = report.operations[0].clone();
        edge.kind = WorkspaceOperationKind::Pad(PadMode::Edge);
        assert!(lowering(edge.as_view()).is_none());
        let wide = WorkspaceTensor::unloaded_f32(&[1, 1, 1, 1, 2], &context).unwrap();
        context.begin_span();
        let wide = WorkspaceTensor::pad(&wide, &[(0, 0); 5], PadMode::Constant, &context).unwrap();
        let wide = context.report(&[wide]).unwrap();
        assert!(
            lowering(wide.operations[0].as_view()).is_none(),
            "heap width destinations need their own accepted owner"
        );
    }
}

#[cfg(all(
    test,
    target_vendor = "apple",
    feature = "metal",
    not(feature = "cuda")
))]
mod packed_grouped_tests {
    use super::*;
    use eredu_nn::{
        GatedProductGroupLayout, GatedProductPolicy, GroupSelection, GroupedGatedProductOperator,
        GroupedGatedProductSpec, GroupedNeuralBackend, GroupedProjectionSpec, LinearFormatSpec,
        ParameterSpec, Tensor,
    };

    #[test]
    fn packed_gated_recipe_keeps_every_chunk_and_its_prepared_output_destination() {
        let projection = |name: &str| {
            GroupedProjectionSpec::new(
                ParameterSpec::trainable(name).unwrap(),
                None,
                LinearFormatSpec::unscaled(eredu_checkpoint::LinearFormat::Dense).unwrap(),
            )
            .unwrap()
        };
        let spec = GroupedGatedProductSpec::new(
            3,
            32,
            32,
            32,
            GatedProductPolicy::ordinary_silu(),
            GatedProductGroupLayout::Packed {
                gate_up: projection("read.weight"),
                down: projection("write.weight"),
            },
        )
        .unwrap();
        let mechanism = MlxMetalWorkspaceMechanisms::current_host().unwrap();
        let qualified = safemlx::ops::concatenate_axis_control_bytes().is_some();
        let invocation_qualified = super::super::super::arithmetic::f32_pointwise_control_bytes(3)
            .is_some()
            && super::super::super::matrix::grouped_projection_control_bytes().is_some();
        for tokens in [0, 64, 65, 129] {
            let context = WorkspaceContext::new(mechanism);
            let mut bank = WorkspaceBackend::grouped_gated_product(spec.clone(), &context).unwrap();
            let input = WorkspaceTensor::unloaded_f32(&[tokens, 32], &context).unwrap();
            let ids = WorkspaceTensor::existing(
                WorkspaceLayout::new(&[tokens, 2], WorkspaceDtype::Int32).unwrap(),
                &context,
            )
            .unwrap();
            let weights = WorkspaceTensor::unloaded_f32(&[tokens, 2], &context).unwrap();
            let selections = GroupSelection::new(ids, weights.clone(), weights);
            context.begin_span();
            let output = bank.forward_grouped(&input, &selections, &context).unwrap();
            let report = context.report(&[output]).unwrap();
            let recorder = ResidentRecipeRecorder::new(
                InferenceGeometry {
                    batch_size: 1,
                    cached_positions: 0,
                    input_positions: tokens as u64,
                    max_output_tokens: 1,
                    prefill_chunk_positions: (tokens as u64).max(1),
                    output: eredu_core::OutputDemand::Sequence,
                },
                mechanism,
            );
            let reduced = recorder.reduce_trace(&report, None, 0, 1).unwrap();
            if !invocation_qualified || (!qualified && tokens > 64) {
                assert!(reduced.first_missing_operation.is_some());
                assert!(reduced.graph.is_none());
                continue;
            }
            assert!(reduced.first_missing_operation.is_none());
            assert!(reduced.mutable_storage.unwrap().mutable_bytes() > 0);
            let source_requirement = (tokens != 0)
                .then(|| {
                    bf16_projection_source_requirement()
                        .or_else(pointwise_source_requirement)
                        .or_else(grouped_indexed_source_requirement)
                })
                .flatten();
            assert_eq!(reduced.unqualified_kernel_owner, source_requirement);
            if source_requirement.is_some() {
                assert!(reduced.query_controls.is_none());
            }
            let chunks = if tokens > 64 {
                (tokens as usize).div_ceil(32)
            } else {
                1
            };
            assert_eq!(reduced.validation_roots, 1 + 2 * chunks);
            assert_eq!(reduced.grouped_outputs.calls, usize::from(tokens > 64));
            assert_eq!(
                reduced.grouped_outputs.chunks,
                if tokens > 64 { chunks } else { 0 }
            );
            if qualified {
                assert!(reduced.grouped_outputs.control_bytes().unwrap() > 0);
                assert!(reduced.graph.unwrap().maximum_operands() >= chunks.max(4));
                assert!(
                    reduced.dispatch.is_some(),
                    "grouped storage does not certify kernel sources"
                );
            }
        }
    }
}

#[cfg(all(
    test,
    target_vendor = "apple",
    feature = "metal",
    not(feature = "cuda")
))]
mod pointwise_owner_tests {
    use super::*;
    use eredu_nn::{NeuralBackend, Tensor};

    #[test]
    fn direct_pointwise_trace_consumes_declared_family_qualification() {
        let mechanism = MlxMetalWorkspaceMechanisms::current_host().unwrap();
        let qualified = super::super::super::arithmetic::f32_pointwise_control_bytes(2).is_some();
        if std::env::var_os("EREDU_REQUIRE_QUALIFIED_FIXED_KERNEL").is_some() {
            assert!(qualified, "pinned pointwise invocation must qualify");
        }
        for sigmoid in [false, true] {
            let context = WorkspaceContext::new(mechanism);
            let input = WorkspaceTensor::unloaded_f32(&[2, 7], &context).unwrap();
            context.begin_span();
            let output = if sigmoid {
                WorkspaceBackend::sigmoid(input, &context)
            } else {
                WorkspaceBackend::silu(input, &context)
            }
            .unwrap();
            let report = context.report(&[output]).unwrap();
            let recorder = ResidentRecipeRecorder::new(
                InferenceGeometry {
                    batch_size: 1,
                    cached_positions: 0,
                    input_positions: 2,
                    max_output_tokens: 1,
                    prefill_chunk_positions: 2,
                    output: eredu_core::OutputDemand::Sequence,
                },
                mechanism,
            );
            let reduced = recorder.reduce_trace(&report, None, 0, 1).unwrap();
            assert_eq!(
                reduced.unqualified_kernel_owner,
                pointwise_source_requirement()
            );
            if pointwise_source_requirement().is_some() {
                assert!(reduced.query_controls.is_none());
            } else if qualified {
                assert!(reduced.query_controls.is_some());
            }
            assert_eq!(reduced.mutable_storage.is_some(), qualified);
        }
    }
}

#[cfg(all(
    test,
    target_vendor = "apple",
    feature = "metal",
    not(feature = "cuda")
))]
mod stored_host_effect_tests {
    use super::*;
    #[test]
    fn stored_host_effects_price_hidden_roots_and_preserve_actual_copy_representation() {
        let facts = MlxMetalWorkspaceMechanisms::current_host().unwrap();
        for dtype in [
            WorkspaceFloatingType::Float32,
            WorkspaceFloatingType::Float16,
            WorkspaceFloatingType::Bfloat16,
        ] {
            let context = WorkspaceContext::new(facts);
            let source = WorkspaceTensor::existing(
                context
                    .layout(&[1, 2, 3, 8], WorkspaceDtype::Float32)
                    .unwrap()
                    .with_representation(Some(WorkspaceRepresentation::new(dtype, false))),
                &context,
            )
            .unwrap();
            context
                .begin_state_span(std::iter::empty::<&WorkspaceTensor>())
                .unwrap();
            let stored = context.store_host_value(&source, dtype).unwrap();
            let output = stored.load(&context).unwrap();
            assert_eq!(
                output.layout().representation(),
                Some(WorkspaceRepresentation::new(dtype, true))
            );
            let report = context.report(&[output]).unwrap();
            assert!(report.unpriced_operations.is_empty());
            assert!(report.unpriced_host_operations.is_empty());
            assert!(report.inference_transient_bytes().is_some());
            assert_eq!(report.operations.len(), 2);
            let store = &report.operations[0];
            let load = &report.operations[1];
            assert!(store.outputs.is_empty());
            assert!(load.inputs.is_empty());
            assert_eq!(load.outputs.len(), 1);
            let store_lowering = lowering(store.as_view()).unwrap();
            assert_eq!(store_lowering.maximum_births, 1); // optional Device compaction only
            assert_eq!(store_lowering.grouped_output_calls, 0);
            assert_eq!(store_lowering.nested_completions, 1);
            let load_lowering = lowering(load.as_view()).unwrap();
            assert_eq!(load_lowering.maximum_births, 1);
            assert_eq!(load_lowering.nested_completions, 1);
            assert!(copy_rank::inspect(store.as_view(), 0).unwrap().controls > 0);
            assert!(copy_rank::inspect(load.as_view(), 0).unwrap().controls > 0);
            let (
                WorkspaceOperationKind::HostStoreFloating(a, _),
                WorkspaceOperationKind::HostLoadStoredFloating(b, _),
            ) = (&store.kind, &load.kind)
            else {
                panic!("actual store/load trace");
            };
            assert!(a.same_store(b));
            let mut invalid = store.clone();
            invalid.kind = WorkspaceOperationKind::HostStoreFloating(
                a.clone(),
                if dtype == WorkspaceFloatingType::Float32 {
                    WorkspaceFloatingType::Float16
                } else {
                    WorkspaceFloatingType::Float32
                },
            );
            assert!(lowering(invalid.as_view()).is_none());
        }
    }
}

#[cfg(all(
    test,
    target_vendor = "apple",
    feature = "metal",
    not(feature = "cuda")
))]
mod muse_tests;

use eredu_nn::workspace::WorkspaceMetadataAllocation;
