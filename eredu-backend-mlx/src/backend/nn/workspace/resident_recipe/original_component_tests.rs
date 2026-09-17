//! Existing Source fixture: borrow a real cold component before original admission.
use super::*;
use eredu_nn::{AttentionRequest, NeuralBackend, Tensor};
use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};
thread_local! {
    static COMPONENT: RefCell<Option<Rc<WorkspaceTraceReport>>> = const { RefCell::new(None) };
    static SUMMARY: Cell<Option<crate::backend::array_copy::SummaryProgram>> = const { Cell::new(None) };
    static QUOTED: Cell<bool> = const { Cell::new(false) };
}
pub(super) fn current(first: bool) -> Option<Rc<WorkspaceTraceReport>> {
    let report = first
        .then(|| COMPONENT.with(|slot| slot.borrow().clone()))
        .flatten();
    if report.is_some() {
        QUOTED.with(|slot| slot.set(true));
    }
    report
}
pub(super) fn summary() -> Option<crate::backend::array_copy::SummaryProgram> {
    SUMMARY.get()
}
pub(crate) struct OriginalComponentTestPlan {
    summary: Option<crate::backend::array_copy::SummaryProgram>,
    report: Rc<WorkspaceTraceReport>,
    pub(crate) completion: ResidentCompletionRecipe,
    controls: u64,
}
pub(crate) struct ComponentQuoteGuard;
impl ComponentQuoteGuard {
    pub(crate) fn was_quoted(&self) -> bool {
        QUOTED.with(Cell::get)
    }
}
impl Drop for ComponentQuoteGuard {
    fn drop(&mut self) {
        COMPONENT.with(|slot| {
            slot.borrow_mut().take();
        });
        SUMMARY.set(None);
    }
}
pub(crate) type OriginalAttentionTestPlan = OriginalComponentTestPlan;
impl OriginalComponentTestPlan {
    pub(crate) fn new() -> Self {
        let mechanism = MlxMetalWorkspaceMechanisms::current_host().unwrap();
        let context = WorkspaceContext::new(mechanism);
        let q = WorkspaceTensor::unloaded_f32(&[1, 1, 1, 4], &context).unwrap();
        let k = WorkspaceTensor::unloaded_f32(&[1, 1, 8193, 4], &context).unwrap();
        let v = WorkspaceTensor::unloaded_f32(&[1, 1, 8193, 3], &context).unwrap();
        let mask = WorkspaceTensor::existing(
            WorkspaceLayout::new(&[1, 1, 1, 8193], WorkspaceDtype::Bool).unwrap(),
            &context,
        )
        .unwrap();
        let sinks = WorkspaceTensor::unloaded_f32(&[1], &context).unwrap();
        context.begin_span();
        let output = WorkspaceBackend::attention_with_sinks(
            AttentionRequest {
                queries: q,
                keys: k,
                values: v,
                scale: 0.5,
                softcap: Some(1.75),
                mask: Some(&mask),
                sinks: Some(&sinks),
                arithmetic: eredu_nn::AttentionArithmetic::InputScores,
            },
            &context,
        )
        .unwrap();
        let result = Self::from_report_with_nested_roots(context.report(&[output]).unwrap(), 3);
        assert_eq!(result.completion.nested_completions, 66);
        result
    }
    pub(crate) fn from_report(report: WorkspaceTraceReport) -> Self {
        let result = Self::from_report_with_nested_roots(report, 0);
        assert_eq!(result.completion.nested_completions, 0);
        result
    }
    fn from_report_with_nested_roots(
        report: WorkspaceTraceReport,
        nested_root_capacity: usize,
    ) -> Self {
        let mechanism = MlxMetalWorkspaceMechanisms::current_host().unwrap();
        let report = Rc::new(report);
        assert!(
            !report.operations.is_empty(),
            "a component needs its actual operation trace"
        );
        let recorder = ResidentRecipeRecorder::new(
            InferenceGeometry {
                batch_size: 1,
                cached_positions: 0,
                input_positions: 1,
                max_output_tokens: 1,
                prefill_chunk_positions: 1,
                output: eredu_core::OutputDemand::Sequence,
            },
            mechanism,
        );
        let reduced = recorder.reduce_trace(&report, None, 0, 1).unwrap();
        assert!(reduced.first_missing_operation.is_none());
        assert!(reduced.unqualified_kernel_owner.is_none());
        assert!(reduced.mutable_storage.is_some());
        assert!(reduced.dispatch.is_some());
        let completion = ResidentCompletionRecipe {
            validation_roots: reduced.validation_roots,
            grouped_outputs: reduced.grouped_outputs,
            traversal: reduced.traversal.unwrap(),
            graph: reduced.graph.unwrap(),
            dispatch: reduced.dispatch,
            nested_completions: reduced.nested_completions,
            nested_root_capacity,
        };
        Self {
            summary: None,
            report,
            completion,
            controls: reduced.query_controls.unwrap() as u64,
        }
    }
    /// Actual Summary program supplies its own maximal scalar frontiers; the
    /// real recorder re-queries those same facts before original Q/Graph/Record.
    pub(crate) fn from_summary_report(
        report: WorkspaceTraceReport,
        program: crate::backend::array_copy::SummaryProgram,
    ) -> Self {
        let mut result = Self::from_report_with_nested_roots(report, 3);
        let population = program.population().unwrap();
        result.completion.nested_completions = result
            .completion
            .nested_completions
            .checked_add(population.scalar_completions)
            .unwrap();
        let collector = population
            .retained_outputs
            .checked_mul(size_of::<safemlx::Array>())
            .unwrap()
            .checked_add(3 * size_of::<Vec<safemlx::Array>>())
            .unwrap();
        result.controls = result
            .controls
            .checked_add(program.control_bytes().unwrap() as u64)
            .unwrap()
            .checked_add(collector as u64)
            .unwrap();
        result.summary = Some(program);
        result
    }
    pub(crate) fn quote(&self) -> ComponentQuoteGuard {
        SUMMARY.set(self.summary);
        QUOTED.with(|slot| slot.set(false));
        COMPONENT.with(|slot| assert!(slot.replace(Some(self.report.clone())).is_none()));
        ComponentQuoteGuard
    }
    pub(crate) fn control_bytes(&self) -> u64 {
        // Actual component query frames are also in the first row. This extra
        // test transport is declared through the fixture's existing Q hook.
        self.controls
            .checked_add(std::mem::size_of::<Self>() as u64)
            .unwrap()
    }
}
