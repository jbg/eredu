//! Actual deterministic numerical trace, with no text/model occurrence grant.
use super::*;
mod addressable;
mod concatenate;
mod layerwise;
mod owned_child;
mod realtime;
pub(crate) use addressable::{AddressableNumericalPopulation, OrdinaryCpuPopulation};
mod cpu_capture;
pub(crate) use cpu_capture::CpuCaptureLoan;

#[derive(Debug, Clone, Copy)]
pub(super) enum CpuNumericalOperation {
    EagerKey,
    UniformUnitInterval,
    Categorical {
        rank: usize,
        columns: usize,
    },
    Difference {
        rank: usize,
        columns: usize,
        rows: usize,
    },
    Logarithm {
        rank: usize,
    },
    SplitKeys {
        count: usize,
        views: usize,
    },
    Slice {
        rank: usize,
    },
    Normalize {
        rank: usize,
        columns: usize,
        rows: usize,
    },
    Greedy {
        rank: usize,
        columns: usize,
    },
    Index {
        rank: usize,
        output_rank: usize,
    },
}
#[derive(Debug, Clone, Copy)]
pub(crate) struct SpeculativeNumericalRecipe {
    pub(crate) completion: ResidentCompletionRecipe,
    pub(crate) storage: CertifiedSpanStorage,
    pub(crate) graph_capacity: usize,
    pub(crate) record_capacity: usize,
    pub(crate) kernels: usize,
    pub(crate) controls: u64,
    /// Actual safe caller census retained from the same selected equation
    /// report. Specialized native sources require their own caller evidence.
    pub(crate) ordinary_calls: Option<OrdinaryCallControls>,
}
impl SpeculativeNumericalRecipe {
    pub(crate) fn with_ordinary_calls(
        mut self,
        report: &WorkspaceTraceReport,
        mechanism: ResidentExecutionMechanisms,
        context: &WorkspaceContext,
    ) -> Result<Self, Error> {
        if mechanism.allocation().original_storage {
            return Ok(self);
        }
        context.charge_metadata(std::mem::size_of::<(
            Self,
            &WorkspaceTraceReport,
            ResidentExecutionMechanisms,
            &WorkspaceContext,
            OrdinaryCallControls,
            Option<OrdinaryCallControls>,
            Result<Self, Error>,
        )>())?;
        self.ordinary_calls = match mechanism {
            ResidentExecutionMechanisms::Cpu { cpu, .. } => cpu
                .ordinary_report_call_controls(report)
                .map_err(MlxWorkspaceFactError::ordinary)?,
            ResidentExecutionMechanisms::Metal(metal) => {
                let mut total = Some(OrdinaryCallControls::default());
                for operation in &report.operations {
                    let calls = metal
                        .ordinary_call_controls(operation.as_view())
                        .map_err(MlxWorkspaceFactError::ordinary)?;
                    total = match (total, calls) {
                        (Some(prior), Some(calls)) => Some(
                            prior
                                .append(calls)
                                .ok_or(WorkspaceMetadataError::Overflow)?,
                        ),
                        _ => None,
                    };
                }
                match (
                    total,
                    report.tensor_handle_clones,
                    safemlx::Array::ordinary_clone_control_bytes(),
                ) {
                    (Some(total), Some(clones), Some(bytes)) => Some(
                        total
                            .metadata(
                                bytes
                                    .checked_mul(clones)
                                    .ok_or(WorkspaceMetadataError::Overflow)?,
                            )
                            .ok_or(WorkspaceMetadataError::Overflow)?,
                    ),
                    _ => None,
                }
            }
        };
        Ok(self)
    }
    pub(crate) fn inspect(
        report: &WorkspaceTraceReport,
        roots: usize,
        mechanism: MlxMetalWorkspaceMechanisms,
        context: &WorkspaceContext,
    ) -> Result<Self, Error> {
        Self::inspect_with_nested(report, roots, 0, mechanism, context)
    }
    pub(crate) fn inspect_with_nested(
        report: &WorkspaceTraceReport,
        roots: usize,
        nested: usize,
        mechanism: MlxMetalWorkspaceMechanisms,
        context: &WorkspaceContext,
    ) -> Result<Self, Error> {
        Self::inspect_source(report, roots, nested, None, mechanism, context, false, None)
    }
    /// The actual eager token constructor has no lazy operation. Its checked
    /// native source facts supplement the same resident completion reducer.
    pub(crate) fn inspect_token_ids(
        report: &WorkspaceTraceReport,
        source: safemlx::OriginalPromptInputFacts,
        mechanism: MlxMetalWorkspaceMechanisms,
        context: &WorkspaceContext,
    ) -> Result<Self, Error> {
        if !report.operations.is_empty() {
            return Err(
                context.metadata_error(format_args!("eager token source contains a lazy equation"))
            );
        }
        Self::inspect_source(report, 1, 0, Some(source), mechanism, context, false, None)
    }
    /// Same exact eager source and zero-equation trace, with the actual private
    /// CPU completion producer. This grants no CPU neural or sampling equations.
    pub(crate) fn inspect_cpu_token_ids(
        report: &WorkspaceTraceReport,
        source: safemlx::OriginalPromptInputFacts,
        mechanism: MlxMetalWorkspaceMechanisms,
        context: &WorkspaceContext,
    ) -> Result<Self, Error> {
        if !report.operations.is_empty() {
            return Err(context.metadata_error(format_args!(
                "CPU eager token source contains a lazy equation"
            )));
        }
        Self::inspect_source(report, 1, 0, Some(source), mechanism, context, true, None)
    }
    /// Fixed-rank range keeps every axis and invokes the same native Slice
    /// alias worker. The report describes constructors; its GPU worker census
    /// is not used as authority for this CPU invocation.
    pub(crate) fn inspect_cpu_range(
        report: &WorkspaceTraceReport,
        mechanism: MlxMetalWorkspaceMechanisms,
        context: &WorkspaceContext,
    ) -> Result<Self, Error> {
        let invalid = || {
            context.metadata_error(format_args!(
                "CPU range source differs from its exact alias equation"
            ))
        };
        let [operation] = report.operations.as_slice() else {
            return Err(invalid());
        };
        let operation = operation.as_view();
        let eredu_nn::workspace::WorkspaceOperationKindView::StaticSlice { strides, .. } =
            operation.kind
        else {
            return Err(invalid());
        };
        if !super::basic::is_static_slice(operation) || strides.iter().any(|&step| step != 1) {
            return Err(invalid());
        }
        let input = operation.inputs.get(0).ok_or_else(invalid)?;
        let output = operation.outputs.get(0).ok_or_else(invalid)?;
        let rank = input.shape().len();
        if !(2..=4).contains(&rank)
            || output.shape().len() != rank
            || input.dtype() != output.dtype()
            || input
                .shape()
                .iter()
                .zip(output.shape())
                .any(|(&a, &b)| a <= 0 || b <= 0 || b > a)
        {
            return Err(invalid());
        }
        Self::inspect_source(
            report,
            1,
            0,
            None,
            mechanism,
            context,
            true,
            Some(CpuNumericalOperation::Slice { rank }),
        )
    }
    /// F32 conversion is an identity for this exact source; the ordinary
    /// precise last-axis Softmax still owns its real copy/worker populations.
    pub(crate) fn inspect_cpu_normalize(
        report: &WorkspaceTraceReport,
        mechanism: MlxMetalWorkspaceMechanisms,
        context: &WorkspaceContext,
    ) -> Result<Self, Error> {
        let invalid = || {
            context.metadata_error(format_args!(
                "CPU normalization source differs from its F32 row equation"
            ))
        };
        let (rank, columns, rows) =
            cpu_normalization_shape(&report.operations).ok_or_else(invalid)?;
        Self::inspect_source(
            report,
            1,
            0,
            None,
            mechanism,
            context,
            true,
            Some(CpuNumericalOperation::Normalize {
                rank,
                columns,
                rows,
            }),
        )
    }
    /// Exact existing final-axis F32 ArgReduce + Squeeze branch. A unit-width
    /// vocabulary uses a distinct fill source and does not enter this recipe.
    pub(crate) fn inspect_cpu_greedy(
        report: &WorkspaceTraceReport,
        mechanism: MlxMetalWorkspaceMechanisms,
        context: &WorkspaceContext,
    ) -> Result<Self, Error> {
        let invalid = || {
            context.metadata_error(format_args!(
                "CPU greedy source differs from its F32 argmax equation"
            ))
        };
        let [operation] = report.operations.as_slice() else {
            return Err(invalid());
        };
        let operation = operation.as_view();
        if !matches!(
            operation.kind,
            eredu_nn::workspace::WorkspaceOperationKindView::Sampling(
                eredu_nn::workspace::WorkspaceSamplingOperation::Greedy
            )
        ) || operation.inputs.len() != 1
            || operation.outputs.len() != 1
        {
            return Err(invalid());
        }
        let input = operation.inputs.get(0).ok_or_else(invalid)?;
        let output = operation.outputs.get(0).ok_or_else(invalid)?;
        let rank = input.shape().len();
        if !(2..=3).contains(&rank)
            || input.dtype() != eredu_nn::workspace::WorkspaceDtype::Float32
            || input.shape()[rank - 1] <= 1
            || input.shape()[..rank - 1].iter().any(|&n| n != 1)
            || output.shape() != &input.shape()[..rank - 1]
            || output.dtype() != eredu_nn::workspace::WorkspaceDtype::Uint32
        {
            return Err(invalid());
        }
        Self::inspect_source(
            report,
            1,
            0,
            None,
            mechanism,
            context,
            true,
            Some(CpuNumericalOperation::Greedy {
                rank,
                columns: input.shape()[rank - 1] as usize,
            }),
        )
    }
    /// Same static coordinate or logit-row selection as the ordinary worker:
    /// one fixed-rank Slice and its row-contiguous Reshape alias.
    pub(crate) fn inspect_cpu_static_index(
        report: &WorkspaceTraceReport,
        mechanism: MlxMetalWorkspaceMechanisms,
        context: &WorkspaceContext,
    ) -> Result<Self, Error> {
        let invalid = || {
            context.metadata_error(format_args!(
                "CPU static index source differs from its Slice/Reshape equation"
            ))
        };
        let [operation] = report.operations.as_slice() else {
            return Err(invalid());
        };
        let operation = operation.as_view();
        if operation.inputs.len() != 1 || operation.outputs.len() != 1 {
            return Err(invalid());
        }
        let input = operation.inputs.get(0).ok_or_else(invalid)?;
        let output = operation.outputs.get(0).ok_or_else(invalid)?;
        let rank = input.shape().len();
        let eredu_nn::workspace::WorkspaceOperationKindView::Index { selected_axes } =
            operation.kind
        else {
            return Err(invalid());
        };
        if !(2..=3).contains(&rank)
            || input.shape().iter().any(|&n| n <= 0)
            || input.dtype() != eredu_nn::workspace::WorkspaceDtype::Float32
            || output.dtype() != input.dtype()
        {
            return Err(invalid());
        }
        let scalar = selected_axes == rank && output.shape().is_empty();
        let row = rank == 3
            && selected_axes == 1
            && input.shape()[0] == 1
            && output.shape() == [1, input.shape()[2]];
        if !scalar && !row {
            return Err(invalid());
        }
        Self::inspect_source(
            report,
            1,
            0,
            None,
            mechanism,
            context,
            true,
            Some(CpuNumericalOperation::Index {
                rank,
                output_rank: output.shape().len(),
            }),
        )
    }
    /// The existing high/low U32[2] constructor has its actual seed bank and
    /// physical source in the shared sampling reducer; only completion is CPU.
    pub(crate) fn inspect_cpu_key(
        report: &WorkspaceTraceReport,
        mechanism: MlxMetalWorkspaceMechanisms,
        context: &WorkspaceContext,
    ) -> Result<Self, Error> {
        if !super::sampling_random::is_eager_key_trace(report) {
            return Err(context.metadata_error(format_args!(
                "CPU random key source differs from its eager seed constructor"
            )));
        }
        Self::inspect_source(
            report,
            1,
            0,
            None,
            mechanism,
            context,
            true,
            Some(CpuNumericalOperation::EagerKey),
        )
    }
    /// The exact shared split-N plus one/two static row views. The current
    /// key and returned key stay separate roots under the same numerical role.
    pub(crate) fn inspect_cpu_split(
        report: &WorkspaceTraceReport,
        roots: usize,
        mechanism: MlxMetalWorkspaceMechanisms,
        context: &WorkspaceContext,
    ) -> Result<Self, Error> {
        let invalid = || {
            context.metadata_error(format_args!(
                "CPU split source differs from its explicit-key row program"
            ))
        };
        let count = cpu_split_count(&report.operations, roots).ok_or_else(invalid)?;
        Self::inspect_source(
            report,
            roots,
            0,
            None,
            mechanism,
            context,
            true,
            Some(CpuNumericalOperation::SplitKeys {
                count,
                views: roots,
            }),
        )
    }
    /// Same sequential split and F32 [1] uniform draw, with both current key
    /// and draw retained as roots. CPU dispatch is independently source-priced.
    pub(crate) fn inspect_cpu_uniform(
        report: &WorkspaceTraceReport,
        mechanism: MlxMetalWorkspaceMechanisms,
        context: &WorkspaceContext,
    ) -> Result<Self, Error> {
        let invalid = || {
            context.metadata_error(format_args!(
                "CPU uniform source differs from its split-two/F32 draw program"
            ))
        };
        if report.operations.len() != 4 || cpu_split_count(&report.operations[..3], 2) != Some(2) {
            return Err(invalid());
        }
        let operation = report.operations[3].as_view();
        if !matches!(
            operation.kind,
            eredu_nn::workspace::WorkspaceOperationKindView::Sampling(
                eredu_nn::workspace::WorkspaceSamplingOperation::UniformUnitInterval
            )
        ) || super::sampling_random::lowering(operation).is_none()
        {
            return Err(invalid());
        }
        Self::inspect_source(
            report,
            2,
            0,
            None,
            mechanism,
            context,
            true,
            Some(CpuNumericalOperation::UniformUnitInterval),
        )
    }
    /// Same neutral normalized positive difference, including every root of
    /// the completed all-axis mass decision. No arbitrary elementwise graph.
    pub(crate) fn inspect_cpu_difference(
        report: &WorkspaceTraceReport,
        mechanism: MlxMetalWorkspaceMechanisms,
        context: &WorkspaceContext,
    ) -> Result<Self, Error> {
        use eredu_nn::workspace::{WorkspaceDtype, WorkspaceOperationKindView as K};
        let invalid = || {
            context.metadata_error(format_args!(
                "CPU correction source differs from its exact F32 two-input program"
            ))
        };
        let [_, _, _, _, subtract, positive, total] = report.operations.as_slice() else {
            return Err(invalid());
        };
        let (rank, columns, rows) =
            cpu_normalization_shape(&report.operations[..2]).ok_or_else(invalid)?;
        if cpu_normalization_shape(&report.operations[2..4]) != Some((rank, columns, rows))
            || columns
                .checked_mul(rows)
                .is_none_or(|n| n <= 1 || n > i32::MAX as usize)
        {
            return Err(invalid());
        }
        let left_operation = report.operations[0].as_view();
        let right_operation = report.operations[2].as_view();
        let left = left_operation.inputs.get(0).ok_or_else(invalid)?;
        let right = right_operation.inputs.get(0).ok_or_else(invalid)?;
        if left.shape() != right.shape() {
            return Err(invalid());
        }
        for (operation, name, inputs) in [
            (subtract, "subtract", 2usize),
            (positive, "maximum_scalar", 1usize),
        ] {
            let operation = operation.as_view();
            if !matches!(operation.kind,K::Elementwise(actual) if actual==name)
                || operation.inputs.len() != inputs
                || operation.outputs.len() != 1
            {
                return Err(invalid());
            }
            for value in operation.inputs.iter().chain(operation.outputs.iter()) {
                if value.dtype() != WorkspaceDtype::Float32 || value.shape() != left.shape() {
                    return Err(invalid());
                }
            }
        }
        let total = total.as_view();
        if !matches!(total.kind, K::Reduction("sum_all", 0, false))
            || total.inputs.len() != 1
            || total.outputs.len() != 1
        {
            return Err(invalid());
        }
        let input = total.inputs.get(0).ok_or_else(invalid)?;
        let output = total.outputs.get(0).ok_or_else(invalid)?;
        if input.shape() != left.shape()
            || input.dtype() != WorkspaceDtype::Float32
            || !output.shape().is_empty()
            || output.dtype() != WorkspaceDtype::Float32
        {
            return Err(invalid());
        }
        Self::inspect_source(
            report,
            2,
            0,
            None,
            mechanism,
            context,
            true,
            Some(CpuNumericalOperation::Difference {
                rank,
                columns,
                rows,
            }),
        )
    }
    /// The second cut is quoted before issuance and only executed after the
    /// ordinary completed mass predicate selects corrected logarithmic logits.
    pub(crate) fn inspect_cpu_logarithm(
        report: &WorkspaceTraceReport,
        mechanism: MlxMetalWorkspaceMechanisms,
        context: &WorkspaceContext,
    ) -> Result<Self, Error> {
        let invalid =
            || context.metadata_error(format_args!("CPU correction logarithm source differs"));
        let [operation] = report.operations.as_slice() else {
            return Err(invalid());
        };
        let operation = operation.as_view();
        if !matches!(
            operation.kind,
            eredu_nn::workspace::WorkspaceOperationKindView::Elementwise("log")
        ) || operation.inputs.len() != 1
            || operation.outputs.len() != 1
        {
            return Err(invalid());
        }
        let input = operation.inputs.get(0).ok_or_else(invalid)?;
        let output = operation.outputs.get(0).ok_or_else(invalid)?;
        let rank = input.shape().len();
        if !(2..=3).contains(&rank)
            || input.shape().iter().any(|&n| n <= 0)
            || output.shape() != input.shape()
            || input.dtype() != eredu_nn::workspace::WorkspaceDtype::Float32
            || output.dtype() != input.dtype()
        {
            return Err(invalid());
        }
        Self::inspect_source(
            report,
            1,
            0,
            None,
            mechanism,
            context,
            true,
            Some(CpuNumericalOperation::Logarithm { rank }),
        )
    }
    /// Actual explicit-key categorical row: split-two and its two static key
    /// views precede the same selected Gumbel/argmax operation as ordinary.
    pub(crate) fn inspect_cpu_categorical(
        report: &WorkspaceTraceReport,
        mechanism: MlxMetalWorkspaceMechanisms,
        context: &WorkspaceContext,
    ) -> Result<Self, Error> {
        use eredu_nn::workspace::{
            WorkspaceDtype, WorkspaceOperationKindView as K, WorkspaceSamplingOperation as S,
        };
        let invalid = || {
            context.metadata_error(format_args!(
                "CPU categorical source differs from its explicit-key F32 row program"
            ))
        };
        if report.operations.len() != 4 || cpu_split_count(&report.operations[..3], 2) != Some(2) {
            return Err(invalid());
        }
        let operation = report.operations[3].as_view();
        if !matches!(operation.kind, K::Sampling(S::Categorical))
            || operation.inputs.len() != 2
            || operation.outputs.len() != 1
            || super::sampling_random::lowering(operation).is_none()
        {
            return Err(invalid());
        }
        let scores = operation.inputs.get(0).ok_or_else(invalid)?;
        let rank = scores.shape().len();
        let columns =
            usize::try_from(*scores.shape().last().ok_or_else(invalid)?).map_err(|_| invalid())?;
        if !(2..=3).contains(&rank)
            || columns <= 1
            || columns > i32::MAX as usize
            || scores.dtype() != WorkspaceDtype::Float32
            || scores.shape()[..rank - 1].iter().any(|&n| n != 1)
        {
            return Err(invalid());
        }
        Self::inspect_source(
            report,
            2,
            0,
            None,
            mechanism,
            context,
            true,
            Some(CpuNumericalOperation::Categorical { rank, columns }),
        )
    }
    fn inspect_source(
        report: &WorkspaceTraceReport,
        roots: usize,
        nested: usize,
        eager: Option<safemlx::OriginalPromptInputFacts>,
        mechanism: MlxMetalWorkspaceMechanisms,
        context: &WorkspaceContext,
        cpu: bool,
        cpu_operation: Option<CpuNumericalOperation>,
    ) -> Result<Self, Error> {
        let invalid = || {
            context.metadata_error(format_args!(
                "speculative numerical trace has an incomplete native producer"
            ))
        };
        if (report.operations.is_empty() && eager.is_none())
            || roots == 0
            || (cpu
                && (roots
                    != match cpu_operation {
                        Some(CpuNumericalOperation::SplitKeys { views, .. }) => views,
                        Some(
                            CpuNumericalOperation::UniformUnitInterval
                            | CpuNumericalOperation::Difference { .. }
                            | CpuNumericalOperation::Categorical { .. },
                        ) => 2,
                        _ => 1,
                    }
                    || nested != 0
                    || match cpu_operation {
                        Some(
                            CpuNumericalOperation::UniformUnitInterval
                            | CpuNumericalOperation::Categorical { .. },
                        ) => eager.is_some() || report.operations.len() != 4,
                        Some(CpuNumericalOperation::Difference { .. }) => {
                            eager.is_some() || report.operations.len() != 7
                        }
                        Some(CpuNumericalOperation::Logarithm { .. }) => {
                            eager.is_some() || report.operations.len() != 1
                        }
                        Some(CpuNumericalOperation::SplitKeys { views, .. }) => {
                            eager.is_some()
                                || report.operations.len() != views + 1
                                || !(1..=2).contains(&views)
                        }
                        Some(
                            CpuNumericalOperation::EagerKey | CpuNumericalOperation::Slice { .. },
                        ) => eager.is_some() || report.operations.len() != 1,
                        Some(CpuNumericalOperation::Normalize { .. }) => {
                            eager.is_some() || report.operations.len() != 2
                        }
                        Some(
                            CpuNumericalOperation::Greedy { .. }
                            | CpuNumericalOperation::Index { .. },
                        ) => eager.is_some() || report.operations.len() != 1,
                        None => eager.is_none() || !report.operations.is_empty(),
                    }))
        {
            return Err(invalid());
        }
        let recorder = ResidentRecipeRecorder::with_context(
            InferenceGeometry {
                batch_size: 1,
                cached_positions: 0,
                input_positions: 1,
                max_output_tokens: 0,
                prefill_chunk_positions: 1,
                output: eredu_core::OutputDemand::Sequence,
            },
            mechanism,
            context,
        )?;
        let reduced = recorder.reduce_trace(report, None, 0, roots)?;
        if reduced.first_missing_operation.is_some()
            || reduced.unqualified_kernel_owner.is_some()
            || reduced.validation_roots != 0
            || reduced.nested_completions != 0
        {
            return Err(context.metadata_error(format_args!(
                "native numerical source is incomplete: operation={:?}, detail={:?}, kernel={:?}, validation={}, nested={}",
                reduced.first_missing_operation, reduced.missing_operation_detail.as_deref(),
                reduced.unqualified_kernel_owner, reduced.validation_roots, reduced.nested_completions,
            )));
        }
        let dispatch = reduced.dispatch.ok_or_else(invalid)?;
        if dispatch.cpu_entries != 0 {
            return Err(context.metadata_error(format_args!(
                "native numerical source requires a CPU worker: entries={}",
                dispatch.cpu_entries,
            )));
        }
        let completion = ResidentCompletionRecipe {
            validation_roots: 0,
            grouped_outputs: reduced.grouped_outputs,
            traversal: reduced.traversal.ok_or_else(invalid)?,
            graph: reduced.graph.ok_or_else(invalid)?,
            // The cold reducer provides the same generic constructor/DAG
            // population. Its GPU dispatch allowance is not a CPU source.
            dispatch: (!cpu).then_some(dispatch),
            nested_completions: nested,
            nested_root_capacity: usize::from(nested != 0),
        };
        let graph = if cpu {
            graph_capacity::ResidentGraphStorage::for_cpu_completion(completion, cpu_operation)
        } else {
            graph_capacity::ResidentGraphStorage::for_completion(completion)
        }
        .ok_or_else(invalid)?;
        let record = record_capacity::ResidentRecordStorage::for_completion(completion)
            .ok_or_else(invalid)?;
        let source_controls = [
            std::mem::size_of::<Option<eredu_nn::workspace::WorkspaceOperationKindView<'_>>>(),
            std::mem::size_of::<std::slice::Iter<'_, i32>>(),
            std::mem::size_of::<[(&WorkspaceOperation, &str, usize); 2]>(),
            std::mem::size_of::<std::array::IntoIter<(&WorkspaceOperation, &str, usize), 2>>(),
            std::mem::size_of::<Option<(usize, usize, usize)>>(),
            std::mem::size_of::<[&WorkspaceOperation; 7]>(),
            std::mem::size_of::<(&str, usize)>(),
            std::mem::size_of::<
                std::iter::Chain<
                    eredu_nn::workspace::WorkspaceLayoutIter<'_>,
                    eredu_nn::workspace::WorkspaceLayoutIter<'_>,
                >,
            >(),
            std::mem::size_of::<eredu_nn::workspace::WorkspaceOperationView<'_>>() * 2,
            std::mem::size_of::<(&[WorkspaceOperation], usize)>(),
            std::mem::size_of::<Option<usize>>(),
            std::mem::size_of::<bool>(),
            std::mem::size_of::<Option<CpuNumericalOperation>>(),
            std::mem::size_of::<[i32; 2]>() * 3,
            std::mem::size_of::<[bool; 2]>(),
            std::mem::size_of::<std::slice::Iter<eredu_nn::workspace::WorkspaceOperation>>(),
            std::mem::size_of::<eredu_nn::workspace::WorkspaceOperationView<'_>>() * 2,
            std::mem::size_of::<eredu_nn::workspace::WorkspaceLayoutView<'_>>() * 4,
            std::mem::size_of::<[Option<eredu_nn::workspace::WorkspaceLayoutView<'_>>; 3]>(),
            std::mem::size_of::<
                std::array::IntoIter<Option<eredu_nn::workspace::WorkspaceLayoutView<'_>>, 3>,
            >(),
            std::mem::size_of::<[usize; 3]>(),
            std::mem::size_of::<(
                &WorkspaceTraceReport,
                MlxMetalWorkspaceMechanisms,
                &WorkspaceContext,
            )>(),
            std::mem::size_of::<Option<safemlx::OriginalPromptInputFacts>>(),
            std::mem::size_of::<Result<Self, Error>>(),
        ];
        let controls = u64::try_from(
            reduced
                .query_controls
                .ok_or_else(invalid)?
                .checked_add(
                    source_controls
                        .into_iter()
                        .try_fold(std::mem::size_of_val(&source_controls), usize::checked_add)
                        .ok_or_else(invalid)?,
                )
                .ok_or_else(invalid)?,
        )
        .map_err(|_| invalid())?
        .checked_add(graph.control_bytes().ok_or_else(invalid)?)
        .ok_or_else(invalid)?;
        Ok(Self {
            completion,
            storage: reduced.mutable_storage.ok_or_else(invalid)?,
            graph_capacity: usize::try_from(graph.full_capacity.ok_or_else(invalid)?)
                .map_err(|_| invalid())?
                .checked_add(eager.map_or(0, |facts| facts.graph_bytes()))
                .ok_or_else(invalid)?,
            record_capacity: usize::try_from(record.full_capacity.ok_or_else(invalid)?)
                .map_err(|_| invalid())?,
            kernels: if cpu { 0 } else { dispatch.kernel_attempts },
            controls,
            ordinary_calls: None,
        })
    }
}

fn cpu_split_count(operations: &[WorkspaceOperation], views: usize) -> Option<usize> {
    if !(1..=2).contains(&views) || operations.len() != views + 1 {
        return None;
    }
    let split = operations.first()?.as_view();
    if !matches!(
        split.kind,
        eredu_nn::workspace::WorkspaceOperationKindView::Sampling(
            eredu_nn::workspace::WorkspaceSamplingOperation::SplitRandomKey
        )
    ) || split.inputs.len() != 1
        || split.outputs.len() != 1
    {
        return None;
    }
    let input = split.inputs.get(0)?;
    let output = split.outputs.get(0)?;
    if input.shape() != [2]
        || input.dtype() != eredu_nn::workspace::WorkspaceDtype::Uint32
        || output.dtype() != input.dtype()
        || output.shape().len() != 2
        || output.shape()[0] <= 0
        || output.shape()[1] != 2
    {
        return None;
    }
    for operation in &operations[1..] {
        let operation = operation.as_view();
        if !matches!(operation.kind,eredu_nn::workspace::WorkspaceOperationKindView::Sampling(
                eredu_nn::workspace::WorkspaceSamplingOperation::SelectRandomKey { index })
                if u64::from(*index) < output.shape()[0] as u64)
            || operation.inputs.len() != 1
            || operation.outputs.len() != 1
        {
            return None;
        }
        let source = operation.inputs.get(0)?;
        let row = operation.outputs.get(0)?;
        if source.shape() != output.shape()
            || source.dtype() != input.dtype()
            || row.shape() != [2]
            || row.dtype() != input.dtype()
        {
            return None;
        }
    }
    Some(output.shape()[0] as usize)
}

fn cpu_normalization_shape(operations: &[WorkspaceOperation]) -> Option<(usize, usize, usize)> {
    let [cast, softmax] = operations else {
        return None;
    };
    let cast = cast.as_view();
    let softmax = softmax.as_view();
    if !matches!(
        cast.kind,
        eredu_nn::workspace::WorkspaceOperationKindView::Elementwise("cast_f32")
    ) || cast.inputs.len() != 1
        || cast.outputs.len() != 1
        || softmax.inputs.len() != 1
        || softmax.outputs.len() != 1
    {
        return None;
    }
    let input = cast.inputs.get(0)?;
    let rank = input.shape().len();
    if !(2..=3).contains(&rank)
        || input.dtype() != eredu_nn::workspace::WorkspaceDtype::Float32
        || input.shape().iter().any(|&n| n <= 0)
        || !matches!(softmax.kind, eredu_nn::workspace::WorkspaceOperationKindView::Reduction("softmax",axis,true)
                if usize::try_from(axis).ok().and_then(|n|n.checked_add(1))==Some(rank))
    {
        return None;
    }
    for value in [
        cast.outputs.get(0),
        softmax.inputs.get(0),
        softmax.outputs.get(0),
    ] {
        let value = value?;
        if value.shape() != input.shape() || value.dtype() != input.dtype() {
            return None;
        }
    }
    let columns = usize::try_from(input.shape()[rank - 1]).ok()?;
    let rows = input.shape()[..rank - 1]
        .iter()
        .try_fold(1usize, |n, &d| n.checked_mul(d as usize))?;
    Some((rank, columns, rows))
}
