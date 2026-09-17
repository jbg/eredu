//! Shared sampler filters, reduced from their existing native child workers.
use super::*;

pub(super) fn lowering(operation: WorkspaceOperationView<'_>) -> Option<Lowering> {
    use eredu_nn::workspace::{WorkspaceDtype, WorkspaceSamplingOperation as S};
    let WorkspaceOperationKindView::Sampling(kind) = operation.kind else {
        return None;
    };
    let input = operation.inputs.get(0)?;
    let width = usize::try_from(*input.shape().last()?).ok()?;
    // The actual sampling contract accepts one nonempty score row, ranks1–3.
    // No numeric values, temperature or cache warmth stand in for that loan.
    if operation.inputs.len() != 1
        || operation.outputs.len() != 1
        || input.dtype() != WorkspaceDtype::Float32
        || width == 0
        || input.shape().len() > 3
        || input.elements().ok()? != width as u64
        || operation.outputs.get(0)? != input
    {
        return None;
    }
    let mut value = match kind {
        S::TopK { keep } if *keep != 0 && (*keep as usize) < width => {
            // Partition+Slice 2/2; keepdims Min 1/1; Less 5/6;
            // mask where 7/9 and its actual -infinity seed. Partition's GPU
            // worker is the same full merge-sort, including tied/nonfinite data.
            let mut value = Lowering::plain(15, 18, 1);
            value.maximum_births += 5 + 2; // sort ping-pong/table; reduce copy/partial
            value.additional_sort_kernels = grouped_sort_kernels(width);
            value
        }
        S::TopP => {
            // Negative+ArgSort 2/2; GatherAxis 3/4; Softmax 2/2; Scan1/1;
            // Subtract+Greater 10/12; where7/9; fill/cast4/4;
            // ScatterAxis: cast+five broadcasts+three-input primitive7/9.
            // Seeds: cutoff, -infinity mask and original-dtype minimum fill.
            let mut value = Lowering::plain(36, 43, 3);
            value.maximum_births += 5 + 1 + 1; // sort; scan copy; softmax compaction
            value.additional_sort_kernels = grouped_sort_kernels(width);
            value
        }
        S::MinP => {
            // Softmax2/2; keepdims Max1/1; threshold Multiply5/6;
            // Less5/6; where7/9. min_p and -infinity are the two real seeds.
            let mut value = Lowering::plain(20, 24, 2);
            value.maximum_births += 1 + 2; // softmax compaction and reduction temporaries
            value
        }
        _ => return None,
    };
    value.intermediate_rank = input.shape().len();
    // The complete standard library includes Scan/Partition and the same
    // GatherAxis/ScatterAxis float/U32 kernels; actual selectors validate their
    // source route before the request's prepared pipeline rows are consumed.
    if safemlx::PreparedPipelineCachePlan::new(0)
        .layout::<()>()
        .is_err()
    {
        value.unqualified_kernel_owner = Some(CustomKernelOwner::SamplingFilters);
    }
    Some(value)
}

#[cfg(all(
    test,
    target_vendor = "apple",
    feature = "metal",
    not(feature = "cuda")
))]
mod tests {
    use super::*;
    use eredu_runtime::working_memory::{
        SamplingWorkspaceObserver, SamplingWorkspacePhase, quote_sampling_workspace_with_observer,
    };
    use eredu_runtime::{ConfiguredTextSampler, GenerationSampler};

    struct Observer(ResidentRecipeRecorder);
    impl SamplingWorkspaceObserver for Observer {
        fn observe(
            &mut self,
            phase: SamplingWorkspacePhase,
            report: &WorkspaceTraceReport,
        ) -> Result<(), Error> {
            if phase == (SamplingWorkspacePhase::Step { index: 0 }) {
                let mut unknown = report.clone();
                unknown.operations[0].kind =
                    WorkspaceOperationKind::Elementwise("unclosed_sampling_worker");
                let refused = self.0.reduce_trace(&unknown, None, 0, 1)?;
                assert_eq!(refused.first_missing_operation, Some(0));
                assert!(refused.mutable_storage.is_none());
            }
            self.0.observe_sampling(phase, report)
        }
    }
    #[test]
    fn default_filters_have_real_completion_and_preserve_missing_sampling_positions() {
        let _sources = crate::tests::support::test_utils::initialize_original_sources();
        let mechanism = MlxMetalWorkspaceMechanisms::current_host().unwrap();
        let qualified = safemlx::PreparedPipelineCachePlan::new(0)
            .layout::<()>()
            .is_ok();
        if std::env::var_os("EREDU_REQUIRE_QUALIFIED_SAMPLING_FILTERS").is_some() {
            assert!(qualified);
        }
        if !qualified {
            return;
        }
        for width in [64, 2049] {
            let geometry = InferenceGeometry {
                batch_size: 1,
                cached_positions: 0,
                input_positions: 5,
                max_output_tokens: 4,
                prefill_chunk_positions: 2,
                output: eredu_core::OutputDemand::Sequence,
            };
            let context = WorkspaceContext::new(mechanism);
            let layout = WorkspaceLayout::new(&[1, width], WorkspaceDtype::Float32).unwrap();
            let source = ConfiguredTextSampler::Standard(GenerationSampler::default());
            let mut observer = Observer(ResidentRecipeRecorder::new(geometry, mechanism));
            quote_sampling_workspace_with_observer(
                &source,
                0.0,
                None,
                &layout,
                &eredu_core::TokenFilter::All,
                4,
                &context,
                Some(&mut observer),
            )
            .unwrap();
            assert_eq!(observer.0.sampling.len(), 5);
            assert_eq!(observer.0.sampling[0].query_controls, Some(0));
            for row in &observer.0.sampling[1..] {
                assert!(row.first_missing_operation.is_none());
                assert!(row.mutable_storage.unwrap().mutable_bytes() > 0);
                assert!(row.query_controls.unwrap() > 0);
                let completion = row.completion.unwrap();
                assert!(completion.traversal.minimum_record_capacity().unwrap() > 0);
                assert!(completion.graph.allocation_extents() > 0);
                assert!(completion.dispatch.unwrap().kernel_attempts > 0);
            }
        }
    }
}
