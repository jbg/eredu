//! Accepted source callbacks extend the same recorded span, never a schedule guess.
use super::*;
use std::mem::{size_of, size_of_val};

impl ResidentRecipeRecorder {
    /// Each selected hook records its actual scalar representation separately.
    /// The paid immutable row outlives the temporary observer cells and remains
    /// attached to this exact report, including non-exporting receiver hooks.
    pub(crate) fn record_capture_scalars(
        &mut self, scalars: &[std::cell::Cell<Option<eredu_nn::workspace::WorkspaceFloatingType>>],
    ) -> Result<(), Error> {
        let context = self.context.as_ref().ok_or_else(|| self.metadata_error("capture scalar row requires its planning account"))?;
        context.charge_metadata(size_of::<(&mut Self,
            &[std::cell::Cell<Option<eredu_nn::workspace::WorkspaceFloatingType>>],
            Vec<Option<eredu_nn::workspace::WorkspaceFloatingType>>, Result<(), Error>,
            std::slice::Iter<'_, std::cell::Cell<Option<eredu_nn::workspace::WorkspaceFloatingType>>>,
        )>())?;
        if self.records.last().is_none_or(|row| row.capture_scalars.is_some()) {
            return Err(self.metadata_error("capture scalar row has no unique recorded span"));
        }
        let mut values = context.metadata_vec(scalars.len())?;
        for scalar in scalars { values.push(scalar.get()); }
        self.records.last_mut().expect("checked span").capture_scalars = Some(values);
        Ok(())
    }
    /// Called immediately after the actual observed equation report. The
    /// observer counts accepted tensors (including empty selections) and only
    /// nonempty physical prefill fragments, matching the shared native worker.
    pub(crate) fn record_capture_completions(&mut self, count: usize) -> Result<(), Error> {
        if count == 0 {
            return Ok(());
        }
        let population = crate::backend::array_copy::CaptureNativePopulation::raw(count)
            .ok_or_else(|| match &self.context {
                Some(context) => {
                    context.metadata_error(format_args!("capture population overflow"))
                }
                None => Error::backend_retained_source(crate::backend::error::Error::PrefillControl(
                    eredu_runtime::working_memory::WorkingMemoryError::Overflow,
                )),
            })?;
        self.record_capture_population(population)
    }
    /// Actual callback aggregate; raw and selected-score workers supply their
    /// own exact successful-path frontiers and fixed controls.
    pub(crate) fn record_capture_population(
        &mut self,
        population: crate::backend::array_copy::CaptureNativePopulation,
    ) -> Result<(), Error> {
        if population.publications == 0
            && population.completions == 0
            && population.controls == 0
            && population.retained_roots == 0
        {
            return Ok(());
        }
        let invalid = || match &self.context {
            Some(context) => context.metadata_error(format_args!(
                "capture native completion population is incomplete"
            )),
            None => Error::backend_retained_source(crate::backend::error::Error::PrefillControl(
                eredu_runtime::working_memory::WorkingMemoryError::UnknownBound,
            )),
        };
        let fixed = [
            size_of::<ResidentCompletionRecipe>(),
            size_of::<crate::backend::array_copy::CaptureNativePopulation>(),
            size_of::<graph_capacity::ResidentGraphStorage>(),
            size_of::<record_capacity::ResidentRecordStorage>(),
            size_of::<Option<ResidentCompletionRecipe>>(),
            size_of::<Option<usize>>(),
            size_of::<Result<(), Error>>(),
        ];
        let query = fixed
            .into_iter()
            .try_fold(size_of_val(&fixed), usize::checked_add)
            .ok_or_else(invalid)?;
        if let Some(context) = &self.context {
            context.charge_metadata(query)?;
        }
        let row = self.records.last().ok_or_else(invalid)?;
        let nested = row
            .nested_completions
            .checked_add(population.completions)
            .ok_or_else(invalid)?;
        let publications = row
            .capture_publications
            .checked_add(population.publications)
            .ok_or_else(invalid)?;
        let roots = row
            .capture_roots
            .checked_add(population.retained_roots)
            .ok_or_else(invalid)?;
        let native = population.controls;
        // Capture transforms are already in the equation DAG, but their
        // retained aliases are independent closing roots. Include each possible
        // alias in the same final Synchronizer and its native worker census.
        let frontier = capture_frontier(row, population.retained_roots);
        let completion =
            frontier.map(|(traversal, _, _)| traversal)
                .zip(row.graph)
                .map(|(traversal, graph)| ResidentCompletionRecipe {
                    validation_roots: row.validation_roots,
                    grouped_outputs: row.grouped_outputs,
                    traversal,
                    graph,
                    dispatch: frontier.map(|(_, dispatch, _)| dispatch),
                    nested_completions: nested,
                    nested_root_capacity: 3,
                });
        // Re-query the same source-derived DAG/frontier workers. No whole-DAG
        // multiplier or hand-written native event count replaces their facts.
        let joined = completion.and_then(|completion| {
            let graph = graph_capacity::ResidentGraphStorage::for_completion(completion)?;
            let record = record_capacity::ResidentRecordStorage::for_completion(completion)?;
            record.full_capacity?;
            usize::try_from(graph.control_bytes()?).ok()
        });
        let controls = row.query_controls.zip(joined).and_then(|(old, joined)| {
            old.checked_add(native)?
                .checked_add(query)?
                .checked_add(frontier?.2)?
                .checked_add(joined)
        });
        // Missing prior evidence remains missing. No partial mutation precedes
        // overflow/refusal, and final capacity still consumes this exact row.
        if row.query_controls.is_some()
            && row.dispatch.is_some()
            && row.traversal.is_some()
            && row.graph.is_some()
            && controls.is_none()
        {
            return Err(invalid());
        }
        let row = self.records.last_mut().expect("validated recorded span");
        row.nested_completions = nested;
        row.capture_publications = publications;
        row.capture_roots = roots;
        if let Some((traversal, dispatch, _)) = frontier {
            row.traversal = Some(traversal);
            row.dispatch = Some(dispatch);
        }
        row.query_controls = controls;
        Ok(())
    }
}

/// Extend only the submitted aliases. Their tensor equations, Data births and
/// primitives have already been traced; no second transform is introduced.
fn capture_frontier(
    row: &ResidentSpanRecipe,
    roots: usize,
) -> Option<(safemlx::OperationEvalTraversalLayout, ResidentDispatchPopulation, usize)> {
    extend_frontier(row.traversal?,row.graph?,row.dispatch?,roots)
}

/// The same submitted-root extension serves model and numerical capture.
/// Inputs are their actual reduced DAG and dispatch; this grants no new span.
pub(super) fn extend_frontier(
    traversal:safemlx::OperationEvalTraversalLayout,
    graph:safemlx::ResidentGraphLayout,
    mut dispatch:ResidentDispatchPopulation,
    roots:usize,
) -> Option<(safemlx::OperationEvalTraversalLayout,ResidentDispatchPopulation,usize)> {
    if roots == 0 {
        return Some((traversal, dispatch, 0));
    }
    let mut limits = traversal.limits();
    limits.roots = limits.roots.checked_add(roots)?;
    limits.arrays = limits.arrays.checked_add(roots)?;
    limits.input_edges = limits.input_edges.checked_add(roots)?;
    let traversal = safemlx::OperationEvent::eval_traversal_layout(limits)?;
    if dispatch.cpu_model.is_some() {
        dispatch.cpu_input_edges = dispatch.cpu_input_edges.checked_add(roots)?;
        let source = safemlx::OperationEvent::cpu_completion_layout(limits.roots)?;
        let controls = [size_of::<ResidentDispatchPopulation>(),size_of::<safemlx::OperationEvalTraversalLimits>(),
            size_of::<safemlx::CpuCopyEvalLayout>(),size_of::<Option<safemlx::CpuCopyEvalLayout>>(),
            source.control_bytes()?,traversal.query_control_bytes()?];
        return Some((traversal, dispatch, controls.into_iter().try_fold(size_of_val(&controls),usize::checked_add)?));
    }
    dispatch.gpu_input_edges = dispatch.gpu_input_edges.checked_add(roots)?;
    let worker = safemlx::OperationEvent::resident_gpu_worker_layout_with_router(
        dispatch.gpu_entries,
        dispatch.gpu_input_edges,
        dispatch.gpu_siblings,
        limits.arrays,
        dispatch.gpu_births,
        dispatch.worker_rank,
        graph.maximum_operands(),
        dispatch.additional_sort_kernels,
        dispatch.cpu_entries.checked_sub(dispatch.parallel_entries)?,
    )?;
    dispatch.worker_graph_extents =
        worker.allocation_extents().checked_add(dispatch.copy_rank_extents)?.checked_add(dispatch.parallel_graph_extents)?;
    dispatch.kernel_attempts = worker.kernel_attempts();
    let controls = [
        size_of::<(safemlx::OperationEvalTraversalLayout, ResidentDispatchPopulation, usize)>(),
        size_of::<safemlx::OperationEvalTraversalLimits>(),
        size_of::<usize>(),
        traversal.query_control_bytes()?,
        worker.control_bytes()?,
    ];
    let controls = controls.into_iter().try_fold(size_of_val(&controls), usize::checked_add)?;
    Some((traversal, dispatch, controls))
}

#[cfg(all(
    test,
    target_vendor = "apple",
    feature = "metal",
    not(feature = "cuda")
))]
mod tests {
    use super::*;
    use eredu_nn::Tensor;
    fn recorder() -> ResidentRecipeRecorder {
        let mechanism = MlxMetalWorkspaceMechanisms::current_host().unwrap();
        let context = WorkspaceContext::new(mechanism);
        let source = WorkspaceTensor::unloaded_f32(&[1, 5], &context).unwrap();
        context.begin_span();
        let output = context
            .execute(
                WorkspaceOperationKind::Elementwise("capture_cast_f32"),
                &[&source],
                vec![WorkspaceLayout::new(&[1, 5], WorkspaceDtype::Float32).unwrap()],
            )
            .unwrap()
            .remove(0);
        let report = context.report(&[output]).unwrap();
        let mut recorder = ResidentRecipeRecorder::new(
            InferenceGeometry {
                batch_size: 1,
                cached_positions: 1,
                input_positions: 1,
                max_output_tokens: 2,
                prefill_chunk_positions: 1,
                output: eredu_core::OutputDemand::LastPosition,
            },
            mechanism,
        );
        recorder
            .record_equation(
                &InferenceWorkspaceSpan::Decode {
                    index: 0,
                    position: 1,
                    output: eredu_core::OutputDemand::LastPosition,
                },
                &report,
                0,
                1,
                None,
                None,
                false,
            )
            .unwrap();
        assert!(recorder.records[0].dispatch.is_some());
        recorder
    }
    fn capacities(row: &ResidentSpanRecipe) -> (u64, u64) {
        let completion = ResidentCompletionRecipe {
            validation_roots: row.validation_roots,
            grouped_outputs: row.grouped_outputs,
            traversal: row.traversal.unwrap(),
            graph: row.graph.unwrap(),
            dispatch: row.dispatch,
            nested_completions: row.nested_completions,
            nested_root_capacity: 3,
        };
        (
            graph_capacity::ResidentGraphStorage::for_completion(completion)
                .unwrap()
                .full_capacity
                .unwrap(),
            record_capacity::ResidentRecordStorage::for_completion(completion)
                .unwrap()
                .full_capacity
                .unwrap(),
        )
    }
    #[test]
    fn accepted_capture_frontiers_requery_graph_record_and_keep_tensor_population() {
        let mut recorder = recorder();
        let before = capacities(&recorder.records[0]);
        let storage = recorder.records[0].mutable_storage.unwrap();
        let controls = recorder.records[0].query_controls.unwrap();
        let enclosing = recorder.records[0].traversal.unwrap();
        let nested = nested_traversal(enclosing).unwrap().limits();
        let before_limits = enclosing.limits();
        let added_roots = 3usize.saturating_sub(before_limits.roots);
        assert_eq!(nested.roots, 3);
        assert_eq!(nested.input_edges, before_limits.input_edges + added_roots);
        assert_eq!(nested.arrays, before_limits.arrays + added_roots);
        assert_eq!(nested.tape_entries, before_limits.tape_entries);
        assert_eq!(nested.output_slots, before_limits.output_slots);
        // The execution bank and its Graph/Record queries must consume exactly
        // the same frontier, including the short one-cast/one-root case.
        let execution = ResidentCompletionRecipe {
            validation_roots: 0,
            grouped_outputs: recorder.records[0].grouped_outputs,
            traversal: enclosing,
            graph: recorder.records[0].graph.unwrap(),
            dispatch: recorder.records[0].dispatch,
            nested_completions: 4,
            nested_root_capacity: 3,
        };
        assert_eq!(execution.nested_traversal().unwrap().limits(), nested);

        recorder.record_capture_completions(0).unwrap();
        assert_eq!(capacities(&recorder.records[0]), before);
        recorder.record_capture_completions(2).unwrap();
        let row = &recorder.records[0];
        assert_eq!(row.nested_completions, 4);
        assert_eq!(row.capture_publications(), 2);
        let retained = row.capture_roots();
        let after_limits = row.traversal.unwrap().limits();
        assert_eq!(after_limits.roots, before_limits.roots + retained);
        assert_eq!(after_limits.input_edges, before_limits.input_edges + retained);
        assert_eq!(after_limits.arrays, before_limits.arrays + retained);
        assert_eq!(after_limits.tape_entries, before_limits.tape_entries);
        assert_eq!(after_limits.output_slots, before_limits.output_slots);
        assert_eq!(row.dispatch.unwrap().gpu_input_edges,
            execution.dispatch.unwrap().gpu_input_edges + retained);
        let after = capacities(row);
        assert!(after.0 > before.0 && after.1 > before.1);
        assert!(row.query_controls.unwrap() > controls);
        assert_eq!(
            row.mutable_storage.unwrap().mutable_bytes(),
            storage.mutable_bytes()
        );
        assert_eq!(
            row.mutable_storage.unwrap().maximum_births(),
            storage.maximum_births()
        );
    }
    #[test]
    fn capture_count_overflow_is_atomic_and_missing_dispatch_stays_unknown() {
        let mut recorder = recorder();
        let before = capacities(&recorder.records[0]);
        assert!(recorder.record_capture_completions(usize::MAX).is_err());
        assert_eq!(recorder.records[0].capture_publications(), 0);
        assert_eq!(capacities(&recorder.records[0]), before);
        recorder.records[0].dispatch = None;
        recorder.record_capture_completions(1).unwrap();
        assert!(recorder.records[0].dispatch.is_none());
        assert!(recorder.records[0].query_controls.is_none());
    }
    #[test]
    fn selected_token_score_program_has_complete_cold_receipts_and_scalar_frontiers() {
        use crate::backend::array_copy::{CaptureNativePopulation, TokenScoreProgram};
        for vocabulary in [1, 64, 3001] {
            let mechanism = MlxMetalWorkspaceMechanisms::current_host().unwrap();
            let context = WorkspaceContext::new(mechanism);
            let source = WorkspaceTensor::unloaded_f32(&[1, 2, vocabulary], &context).unwrap();
            context.begin_span();
            let ids = [0, (vocabulary - 1) as u32];
            let program = TokenScoreProgram::new(vocabulary, &ids).unwrap();
            let mut retained = vec![source.clone()];
            program.trace(&source, &context, &mut retained).unwrap();
            let report = context.report(&retained).unwrap();
            let mut recorder = ResidentRecipeRecorder::new(
                InferenceGeometry {
                    batch_size: 1,
                    cached_positions: 1,
                    input_positions: 1,
                    max_output_tokens: 2,
                    prefill_chunk_positions: 1,
                    output: eredu_core::OutputDemand::LastPosition,
                },
                mechanism,
            );
            recorder
                .record_equation(
                    &InferenceWorkspaceSpan::Decode {
                        index: 0,
                        position: 1,
                        output: eredu_core::OutputDemand::LastPosition,
                    },
                    &report,
                    0,
                    1,
                    None,
                    None,
                    false,
                )
                .unwrap();
            let row = &recorder.records[0];
            assert!(
                row.dispatch.is_some(),
                "vocabulary={vocabulary}, missing={:?}",
                row.first_missing_operation
            );
            assert!(row.mutable_storage.is_some());
            let population = CaptureNativePopulation::token_scores(program).unwrap();
            assert_eq!(
                population.completions,
                3 + (vocabulary as usize).div_ceil(1024)
                    + ids.len() * if vocabulary == 1 { 2 } else { 4 }
            );
            let before = capacities(row);
            recorder.record_capture_population(population).unwrap();
            let row = &recorder.records[0];
            assert_eq!(row.capture_publications(), 1);
            assert_eq!(row.capture_roots(), population.retained_roots);
            assert_eq!(row.nested_completions, population.completions);
            assert!(row.query_controls.is_some());
            let after = capacities(row);
            assert!(after.0 >= before.0 && after.1 > before.1);
        }
    }
    #[test]
    fn sorted_candidate_receipt_covers_single_and_multiblock_sort_with_four_frontiers() {
        use crate::backend::array_copy::{CandidateExtraction, CaptureNativePopulation};
        for vocabulary in [1, 64, 2048, 2049, 4097] {
            for count in [1, vocabulary] {
                let mechanism = MlxMetalWorkspaceMechanisms::current_host().unwrap();
                let context = WorkspaceContext::new(mechanism);
                let source = WorkspaceTensor::unloaded_f32(&[1, 3, vocabulary], &context).unwrap();
                let program =
                    CandidateExtraction::borrowed(&[1, 3, vocabulary], count as u64).unwrap();
                context.begin_span();
                let outputs = program.trace(&source, &context).unwrap();
                let report = context.report(&outputs).unwrap();
                let mut recorder = ResidentRecipeRecorder::new(
                    InferenceGeometry {
                        batch_size: 1,
                        cached_positions: 1,
                        input_positions: 1,
                        max_output_tokens: 2,
                        prefill_chunk_positions: 1,
                        output: eredu_core::OutputDemand::LastPosition,
                    },
                    mechanism,
                );
                recorder
                    .record_equation(
                        &InferenceWorkspaceSpan::Decode {
                            index: 0,
                            position: 1,
                            output: eredu_core::OutputDemand::LastPosition,
                        },
                        &report,
                        0,
                        1,
                        None,
                        None,
                        false,
                    )
                    .unwrap();
                let row = &recorder.records[0];
                assert!(
                    row.dispatch.is_some(),
                    "V={vocabulary}, K={count}, missing={:?}",
                    row.first_missing_operation
                );
                assert!(row.mutable_storage.is_some());
                let before = capacities(row);
                let population = CaptureNativePopulation::candidates().unwrap();
                recorder.record_capture_population(population).unwrap();
                let row = &recorder.records[0];
                assert_eq!(row.capture_publications(), 1);
                assert_eq!(row.capture_roots(), 1 + CandidateExtraction::ROOTS);
                assert_eq!(row.nested_completions, 4);
                assert!(row.query_controls.is_some());
                let after = capacities(row);
                assert!(after.0 >= before.0 && after.1 > before.1);
            }
        }
    }
}

impl ResidentNativeRecipe {
    /// Merge only matching reports of the issued step, independently per
    /// immutable selection ordinal. A reached hook retains its actual scalar
    /// even when capture quota skips its transform; absent hooks remain absent.
    pub(crate) fn capture_scalars_for_step(
        &self, step: &eredu_runtime::working_memory::InferenceTextStep, count: usize,
        metadata: &eredu_nn::workspace::HostMetadataFunding,
    ) -> Result<Vec<Option<eredu_nn::workspace::WorkspaceFloatingType>>, crate::backend::error::Error> {
        use crate::backend::error::Error as NativeError;
        use eredu_runtime::working_memory::WorkingMemoryError;
        type Scalars = Vec<Option<eredu_nn::workspace::WorkspaceFloatingType>>;
        metadata.reserve_metadata(size_of::<(&Self,
            &eredu_runtime::working_memory::InferenceTextStep, usize, Scalars,
            Result<Scalars, NativeError>, WorkingMemoryError,
            std::slice::Iter<'_, ResidentSpanRecipe>,
            std::iter::Zip<std::slice::IterMut<'_, Option<eredu_nn::workspace::WorkspaceFloatingType>>,
                std::slice::Iter<'_, Option<eredu_nn::workspace::WorkspaceFloatingType>>>,
        )>()).map_err(NativeError::WorkspacePlanning)?;
        if step.request().geometry() != self.plan.geometry() {
            return Err(NativeError::PrefillControl(WorkingMemoryError::IdentityMismatch));
        }
        let mut scalars = metadata.metadata_vec(count).map_err(NativeError::Neural)?;
        scalars.resize(count, None);
        for row in &self.records {
            let selected = match &row.span {
                InferenceWorkspaceSpan::Sampling(_) => return Err(NativeError::PrefillControl(WorkingMemoryError::IdentityMismatch)),
                InferenceWorkspaceSpan::Prefill(_) => step.attempt() == 0,
                InferenceWorkspaceSpan::Decode { index, .. } => index.checked_add(1) == Some(step.attempt()),
            };
            if !selected { continue; }
            let actual = row.capture_scalars.as_deref()
                .ok_or(NativeError::PrefillControl(WorkingMemoryError::UnknownBound))?;
            if actual.len() != scalars.len() {
                return Err(NativeError::PrefillControl(WorkingMemoryError::IdentityMismatch));
            }
            for (scalar, actual) in scalars.iter_mut().zip(actual) {
                if let Some(actual) = actual {
                    if scalar.is_some_and(|prior| prior != *actual) {
                        return Err(NativeError::PrefillControl(WorkingMemoryError::IdentityMismatch));
                    }
                    *scalar = Some(*actual);
                }
            }
        }
        Ok(scalars)
    }
}

use eredu_nn::workspace::WorkspaceMetadataAllocation;
