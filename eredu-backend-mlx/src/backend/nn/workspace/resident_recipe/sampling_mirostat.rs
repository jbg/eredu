//! Actual adaptive cutoff and post-token probability constructors.
//! One finite nested completion separates the two scalar observations; the
//! sampler's policy, random state, cutoff and mu update remain shared.
use super::*;

pub(super) fn lowering(operation: WorkspaceOperationView<'_>) -> Option<Lowering> {
    use eredu_nn::workspace::{WorkspaceDtype, WorkspaceSamplingOperation as S};
    let WorkspaceOperationKindView::Sampling(kind) = operation.kind else {
        return None;
    };
    let input = operation.inputs.get(0)?;
    let output = operation.outputs.get(0)?;
    let width = usize::try_from(*input.shape().last()?).ok()?;
    if operation.inputs.len() != 1
        || operation.outputs.len() != 1
        || input.dtype() != WorkspaceDtype::Float32
        || width == 0
        || input.shape().len() > 3
        || input.elements().ok()? != width as u64
    {
        return None;
    }
    let mut value = match kind {
        S::MirostatCutoff if output == input => {
            // Softmax2/2, Max1/1, Less5/6, ArgReduce+Squeeze+Expand3/3;
            // two Full constructions8/8; ScatterAxis7/9; Greater5/6;
            // two Select constructions14/18. Actual seeds are cutoff, true,
            // false and negative infinity. Unit-axis argmax uses Full instead.
            let mut value = if width == 1 {
                Lowering::plain(48, 56, 5)
            } else {
                Lowering::plain(45, 53, 4)
            };
            value.maximum_births += 1 + 2; // softmax compaction; reduction copy/partial
            value
        }
        S::TokenProbability
            if output.shape().is_empty() && output.dtype() == WorkspaceDtype::Float32 =>
        {
            // The shared worker constructs Softmax2/2 then static Slice and
            // Reshape2/2. Reading the completed F32 scalar creates no tensor.
            let mut value = Lowering::plain(4, 4, 0);
            value.maximum_births += 1; // possible softmax input compaction
            value
        }
        _ => return None,
    };
    value.intermediate_rank = input.shape().len();
    if safemlx::PreparedPipelineCachePlan::new(0)
        .layout::<()>()
        .is_err()
    {
        value.unqualified_kernel_owner = Some(CustomKernelOwner::SamplingMirostat);
    }
    Some(value)
}

/// Match the actual shared Mirostat trace, including its single intermediate
/// token read. An arbitrary trace containing a probability operation cannot
/// obtain a nested-completion destination.
pub(super) fn is_adaptive_step(report: &WorkspaceTraceReport) -> bool {
    use eredu_nn::workspace::WorkspaceSamplingOperation as S;
    let is = |operation: &eredu_nn::workspace::WorkspaceOperation, kind: S| matches!(&operation.kind, WorkspaceOperationKind::Sampling(actual) if *actual == kind);
    let Some((last, prefix)) = report.operations.split_last() else {
        return false;
    };
    is(last, S::TokenProbability)
        && prefix.last().is_some_and(|op| is(op, S::ReadToken))
        && prefix.iter().filter(|op| is(op, S::ReadToken)).count() == 1
        && prefix.iter().filter(|op| is(op, S::MirostatCutoff)).count() == 1
        && prefix.iter().filter(|op| is(op, S::SplitRandomKey)).count() == 1
        && prefix.iter().filter(|op| is(op, S::Categorical)).count() == 1
        && prefix.iter().all(|op| !is(op, S::TokenProbability))
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
        SamplingWorkspaceObserver, SamplingWorkspacePhase, WorkspaceSamplingRandomState,
        quote_sampling_workspace_with_observer,
    };

    struct Observer(ResidentRecipeRecorder);
    impl SamplingWorkspaceObserver for Observer {
        fn observe(
            &mut self,
            phase: SamplingWorkspacePhase,
            report: &WorkspaceTraceReport,
        ) -> Result<(), Error> {
            self.0.observe_sampling(phase, report)
        }
    }

    #[test]
    fn mirostat_retains_one_graph_and_two_real_scalar_frontiers() {
        let _sources = crate::tests::support::test_utils::initialize_original_sources();
        let qualified = safemlx::PreparedPipelineCachePlan::new(0)
            .layout::<()>()
            .is_ok();
        if std::env::var_os("EREDU_REQUIRE_QUALIFIED_MIROSTAT").is_some() {
            assert!(qualified);
        }
        if !qualified {
            return;
        }
        let mechanism = MlxMetalWorkspaceMechanisms::current_host().unwrap();
        for shape in [&[1][..], &[1, 64][..], &[1, 1, 2049][..]] {
            let geometry = InferenceGeometry {
                batch_size: 1,
                cached_positions: 0,
                input_positions: 3,
                max_output_tokens: 3,
                prefill_chunk_positions: 2,
                output: eredu_core::OutputDemand::Sequence,
            };
            let context = WorkspaceContext::new(mechanism);
            let layout = WorkspaceLayout::new(shape, WorkspaceDtype::Float32).unwrap();
            let random = WorkspaceSamplingRandomState::from_seed(&context).unwrap();
            let sampler = eredu_runtime::ConfiguredTextSampler::MirostatV2(
                eredu_runtime::MirostatV2Sampler::default(),
            );
            let mut observer = Observer(ResidentRecipeRecorder::new(geometry, mechanism));
            quote_sampling_workspace_with_observer(
                &sampler,
                1.0,
                Some(&random),
                &layout,
                &eredu_core::TokenFilter::All,
                3,
                &context,
                Some(&mut observer),
            )
            .unwrap();
            assert_eq!(observer.0.sampling.len(), 4);
            assert_eq!(
                observer.0.sampling[0]
                    .mutable_storage
                    .unwrap()
                    .maximum_births(),
                1
            );
            for row in &observer.0.sampling[1..] {
                assert!(row.first_missing_operation.is_none());
                let completion = row
                    .completion
                    .expect("actual adaptive step has final completion");
                assert_eq!(completion.traversal.roots(), 2);
                assert_eq!(completion.nested_completions, 1);
                assert!(completion.nested_traversal().unwrap().roots() >= 2);
                assert!(completion.graph.primitives() > 80);
                assert!(completion.graph.seeds() >= 9);
                assert!(row.mutable_storage.unwrap().mutable_bytes() > 0);
                assert!(row.query_controls.unwrap() > 0);
                assert!(completion.dispatch.unwrap().kernel_attempts > 0);
            }
        }
    }
}
