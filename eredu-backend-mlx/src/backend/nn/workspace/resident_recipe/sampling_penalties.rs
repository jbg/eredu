//! Penalty child graphs from the shared sampler, independent of token values.
use super::*;

pub(super) fn lowering(operation: WorkspaceOperationView<'_>) -> Option<Lowering> {
    use eredu_nn::workspace::{WorkspaceDtype, WorkspaceSamplingOperation as S};
    let WorkspaceOperationKindView::Sampling(S::Penalties {
        repetition,
        additive,
        ..
    }) = operation.kind
    else {
        return None;
    };
    let input = operation.inputs.get(0)?;
    let width = usize::try_from(*input.shape().last()?).ok()?;
    if operation.inputs.len() != 1
        || operation.outputs.len() != 1
        || input.dtype() != WorkspaceDtype::Float32
        || width == 0
        || input.elements().ok()? == 0
        || operation.outputs.get(0)? != input
        || (!*repetition && !*additive)
    {
        return None;
    }
    let (mut primitives, mut edges, mut seeds) = (0usize, 0usize, 0usize);
    if *repetition {
        // Divide, Multiply and Greater each own two possible Casts, two
        // Broadcasts and their result (5/6). Both Where calls own three Casts,
        // three Broadcasts and Select (7/9). Eager seeds: Bool mask, the two
        // independently constructed repeat scalars and the zero comparison.
        primitives = 3 * 5 + 2 * 7;
        edges = 3 * 6 + 2 * 9;
        seeds = 4;
    }
    if *additive {
        // One real F32 host upload and Subtract's two casts/broadcasts/result.
        primitives = primitives.checked_add(5)?;
        edges = edges.checked_add(6)?;
        seeds = seeds.checked_add(1)?;
    }
    let mut value = Lowering::plain(primitives, edges, seeds);
    value.intermediate_rank = input.shape().len();
    // A zero selected history window still executes the configured native
    // graph. Invalid/repeated token IDs alter buffer contents, never births.
    // These are the existing precompiled binary/ternary/copy workers; the
    // request's normal pipeline-source prerequisite remains mandatory.
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
        quote_sampling_workspace_with_observer,
    };
    use eredu_runtime::{ConfiguredTextSampler, GenerationSampler};

    #[test]
    fn actual_penalty_windows_join_complete_sampling_constructors() {
        let _sources = crate::tests::support::test_utils::initialize_original_sources();
        let mechanism = MlxMetalWorkspaceMechanisms::current_host().unwrap();
        let qualified = safemlx::PreparedPipelineCachePlan::new(0)
            .layout::<()>()
            .is_ok();
        if std::env::var_os("EREDU_REQUIRE_QUALIFIED_SAMPLING_PENALTIES").is_some() {
            assert!(qualified);
        }
        if !qualified {
            return;
        }
        let geometry = InferenceGeometry {
            batch_size: 1,
            cached_positions: 0,
            input_positions: 5,
            max_output_tokens: 2,
            prefill_chunk_positions: 2,
            output: eredu_core::OutputDemand::Sequence,
        };
        for window in [-1, 0, 1, 32] {
            for (repeat, frequency, presence) in [
                (1.25, 0.0, 0.0),
                (1.0, 0.5, 0.0),
                (1.0, 0.0, -0.1),
                (1.25, 0.5, 0.2),
            ] {
                let context = WorkspaceContext::new(mechanism);
                let layout = WorkspaceLayout::new(&[1, 64], WorkspaceDtype::Float32).unwrap();
                let source = ConfiguredTextSampler::Standard(
                    GenerationSampler::default()
                        .penalties(repeat, window, frequency, presence)
                        .with_generated_tokens([1, 1, u32::MAX, 2]),
                );
                let mut observer = ResidentRecipeRecorder::new(geometry, mechanism);
                quote_sampling_workspace_with_observer(
                    &source,
                    0.0,
                    None,
                    &layout,
                    &eredu_core::TokenFilter::All,
                    2,
                    &context,
                    Some(&mut observer),
                )
                .unwrap();
                let program = observer.finish_sampling(2).unwrap();
                assert_eq!(program.steps(), Some(2));
                for row in &program.rows()[1..] {
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
}
