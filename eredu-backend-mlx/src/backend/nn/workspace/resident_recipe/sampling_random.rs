//! Explicit-key Standard sampling, using the ordinary native random equations.
use super::*;
use eredu_nn::workspace::{WorkspaceDtype, WorkspaceSamplingOperation as S};

pub(super) fn lowering(operation: WorkspaceOperationView<'_>) -> Option<Lowering> {
    let WorkspaceOperationKindView::Sampling(kind) = operation.kind else {
        return None;
    };
    let [output] = operation.outputs.array()?;
    let mut value = match kind {
        S::CreateRandomKey
            if operation.inputs.is_empty()
                && output.dtype() == WorkspaceDtype::Uint32
                && output.shape() == [2] =>
        {
            // random::key copies seed high/low words into the final eager U32[2].
            Lowering::plain(0, 0, 1)
        }
        S::SplitRandomKey
            if operation.inputs.len() == 1
                && operation.inputs.get(0)?.dtype() == WorkspaceDtype::Uint32
                && operation.inputs.get(0)?.shape() == [2]
                && output.dtype() == WorkspaceDtype::Uint32
                && output.shape().len() == 2
                && output.shape()[0] > 0 && output.shape()[1] == 2 =>
        {
            // random::split(key,n) is one RandomBits. The real metadata trace
            // separately records each static index/squeeze view of its result.
            Lowering::plain(1, 1, 0)
        }
        S::SelectRandomKey { index } if operation.inputs.len() == 1 => {
            let input = operation.inputs.get(0)?;
            if input.dtype() != WorkspaceDtype::Uint32 || input.shape().len() != 2
                || input.shape()[0] <= 0 || input.shape()[1] != 2
                || u64::from(*index) >= input.shape()[0] as u64
                || output.dtype() != WorkspaceDtype::Uint32 || output.shape() != [2] { return None; }
            // A one-row split table omits the identical full Slice; Reshape
            // still removes its leading unit axis. No seed or backing is born.
            Lowering::plain(1 + usize::from(input.shape()[0] != 1),
                1 + usize::from(input.shape()[0] != 1), 0)
        }
        S::UniformUnitInterval if operation.inputs.len() == 1
            && operation.inputs.get(0)?.dtype() == WorkspaceDtype::Uint32
            && operation.inputs.get(0)?.shape() == [2]
            && output.dtype() == WorkspaceDtype::Float32 && output.shape() == [1] => {
            // Actual random.cpp uniform: two initial casts, five binary
            // workers (5P/6E each), RandomBits and one restoration cast.
            // Four eager F32 sources are low, high, upper and maxval. Split
            // and both static key views have separate metadata operations.
            let mut value = Lowering::plain(29, 34, 4);
            value.intermediate_rank = 1;
            value
        }
        S::Categorical if operation.inputs.len() == 2 => {
            let scores = operation.inputs.get(0)?;
            let key = operation.inputs.get(1)?;
            let rank = scores.shape().len();
            let width = *scores.shape().last()?;
            if !(1..=3).contains(&rank)
                || width <= 0
                || scores.elements().ok()? != width as u64
                || scores.dtype() != WorkspaceDtype::Float32
                || key.dtype() != WorkspaceDtype::Uint32
                || key.shape() != [2]
                || output.dtype() != WorkspaceDtype::Uint32
                || output.shape() != &scores.shape()[..rank - 1]
            {
                return None;
            }
            // uniform: lo/hi casts2; range,divide,min,multiply,add are five
            // binary workers (5P/6E each); RandomBits1; restoration cast1.
            // Gumbel adds Log/Negative/Log/Negative (6P/6E). Add to logits is
            // another binary worker; normal ArgReduce+Squeeze is2P/2E.
            // The four eager uniform sources are low/high/upper/maxval.
            let mut value = if width == 1 {
                // Argmax's real no-op-axis branch uses zeros/Full and Squeeze,
                // with its own eager zero source, instead of ArgReduce.
                Lowering::plain(45, 51, 5)
            } else {
                Lowering::plain(42, 48, 4)
            };
            value.intermediate_rank = rank;
            value
        }
        _ => return None,
    };
    if !matches!(kind, S::CreateRandomKey | S::SelectRandomKey { .. }) {
        // random.metal always supplies both rbits/rbitsc in the actual default
        // library. Unary/binary/ArgReduce use the same complete precompiled path.
        value.unqualified_kernel_owner = safemlx::PreparedPipelineCachePlan::new(0)
            .layout::<()>()
            .is_err()
            .then_some(CustomKernelOwner::SamplingRandom);
    }
    Some(value)
}

pub(super) fn is_eager_key_trace(report: &WorkspaceTraceReport) -> bool {
    report.operations.len() == 1
        && matches!(
            report.operations[0].kind,
            WorkspaceOperationKind::Sampling(S::CreateRandomKey)
        )
        && lowering(report.operations[0].as_view()).is_some()
}

/// Only this observed constructor is eager and issues no native evaluation.
pub(super) fn eager_preparation(
    report: &WorkspaceTraceReport,
    allocation: MetalAllocationFacts,
) -> Result<Option<(CertifiedSpanStorage, safemlx::ResidentGraphLayout, usize)>, Error> {
    if !is_eager_key_trace(report) {
        return Ok(None);
    }
    let Some(shells) = report.tensor_handle_clones else {
        return Ok(None);
    };
    let Some(graph) =
        safemlx::OperationEvent::resident_graph_layout_with_shells(0, 1, 1, 4, shells)
    else {
        return Ok(None);
    };
    let Some(controls) = graph
        .control_bytes()
        .and_then(|n| n.checked_add(crate::backend::random::standard_sampling_control_bytes()?))
    else {
        return Ok(None);
    };
    let bytes = allocation
        .buffer_capacity(8)
        .map_err(Error::backend_retained_source)?;
    Ok(Some((
        CertifiedSpanStorage {
            mutable_bytes: bytes,
            maximum_births: 1,
        },
        graph,
        controls,
    )))
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
    fn standard_seeded_sampling_retains_eager_key_and_complete_step_recipes() {
        let _sources = crate::tests::support::test_utils::initialize_original_sources();
        let qualified = safemlx::PreparedPipelineCachePlan::new(0)
            .layout::<()>()
            .is_ok();
        if std::env::var_os("EREDU_REQUIRE_QUALIFIED_STANDARD_RANDOM").is_some() {
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
            let source = eredu_runtime::ConfiguredTextSampler::Standard(
                eredu_runtime::GenerationSampler::default(),
            );
            let mut observer = Observer(ResidentRecipeRecorder::new(geometry, mechanism));
            quote_sampling_workspace_with_observer(
                &source,
                0.8,
                Some(&random),
                &layout,
                &eredu_core::TokenFilter::All,
                3,
                &context,
                Some(&mut observer),
            )
            .unwrap();
            assert_eq!(observer.0.sampling.len(), 4);
            let preparation = &observer.0.sampling[0];
            assert!(preparation.first_missing_operation.is_none());
            assert!(
                preparation.completion.is_none(),
                "eager key construction submits no Eval"
            );
            assert_eq!(preparation.mutable_storage.unwrap().maximum_births(), 1);
            assert!(preparation.mutable_storage.unwrap().mutable_bytes() >= 8);
            let Some(ResidentSamplingPreparation::EagerKey(graph)) = preparation.preparation else {
                panic!("the real seed must own its preparation Graph");
            };
            assert_eq!((graph.primitives(), graph.seeds()), (0, 1));
            assert!(graph.allocation_extents() > 0);
            assert!(preparation.query_controls.unwrap() > 0);
            for row in &observer.0.sampling[1..] {
                assert!(row.first_missing_operation.is_none());
                assert!(row.mutable_storage.unwrap().maximum_births() > 1);
                assert!(row.query_controls.unwrap() > 0);
                assert_eq!(row.completion.unwrap().traversal.roots(), 2);
                assert!(row.completion.unwrap().dispatch.unwrap().kernel_attempts > 0);
            }
        }
    }
    #[test]
    fn sequential_uniform_recipe_covers_draw_and_advanced_key() {
        let _sources = crate::tests::support::test_utils::initialize_original_sources();
        let mechanism = MlxMetalWorkspaceMechanisms::current_host().unwrap();
        let context = WorkspaceContext::new(mechanism);
        let key = WorkspaceTensor::existing(
            context.layout(&[2], WorkspaceDtype::Uint32).unwrap(), &context).unwrap();
        let mut random = WorkspaceSamplingRandomState::from_key(key).unwrap();
        context.begin_span();
        let draw = random.uniform_unit_interval(&context).unwrap();
        let report = context.finish_report(&[draw, random.into_key()]).unwrap();
        let recipe = super::super::numerical::SpeculativeNumericalRecipe::inspect(
            &report, 2, mechanism, &context).unwrap();
        assert_eq!(recipe.completion.traversal.roots(), 2);
        assert!(recipe.completion.dispatch.is_some());
        assert!(recipe.storage.maximum_births() >= 33);
        assert!(recipe.graph_capacity > 0 && recipe.record_capacity > 0);
        let mut primitive = report.operations.last().unwrap().clone();
        assert!(lowering(primitive.as_view()).is_some());
        primitive.outputs[0] = WorkspaceLayout::new(&[2], WorkspaceDtype::Float32).unwrap();
        assert!(lowering(primitive.as_view()).is_none());
    }

}

#[cfg(all(test,target_vendor="apple",feature="metal",not(feature="cuda")))]
mod cpu_state_tests {
    use super::*;
    use crate::backend::nn::workspace::resident_recipe::ResidentRecipeRecorder;
    use eredu_core::InferenceGeometry;
    use eredu_architectures::prepared_execution::InferenceEquationTraceObserver;
    use eredu_runtime::working_memory::{SamplingWorkspaceObserver, SamplingWorkspacePhase,
        WorkspaceSamplingRandomState, quote_sampling_workspace_with_observer};
    struct Observer(ResidentRecipeRecorder);
    impl SamplingWorkspaceObserver for Observer {
        fn observe(&mut self, phase: SamplingWorkspacePhase, report: &WorkspaceTraceReport) -> Result<(), Error> {
            self.0.observe_sampling(phase, report)
        }
    }
    #[test]
    fn cpu_resident_sampler_joins_filter_state_views_and_final_token_completion() {
        let _sources = crate::tests::support::test_utils::initialize_original_sources();
        let metal = MlxMetalWorkspaceMechanisms::current_host().unwrap();
        let choice = MlxCpuMatmulMechanism::select(eredu_nn::CpuMatmulImplementation::Float32Tiles).unwrap();
        let cpu = MlxCpuWorkspaceMechanisms::new(metal.allocation(), choice);
        for temperature in [0.0, 0.8] { for shape in [&[1, 64][..], &[1, 1, 2049][..]] {
            let context = WorkspaceContext::new(cpu);
            let geometry = InferenceGeometry { batch_size: 1, cached_positions: 0, input_positions: 3,
                max_output_tokens: 3, prefill_chunk_positions: 2, output: eredu_core::OutputDemand::Sequence };
            let layout = context.layout(shape, WorkspaceDtype::Float32).unwrap()
                .with_representation(Some(WorkspaceRepresentation::new(WorkspaceFloatingType::Float32, true)));
            let random = if temperature != 0.0 { Some(WorkspaceSamplingRandomState::from_seed(&context).unwrap()) } else { None };
            let mut observer = Observer(ResidentRecipeRecorder::with_cpu_context(geometry, metal, cpu, &context).unwrap());
            let sampler = eredu_runtime::ConfiguredTextSampler::Standard(eredu_runtime::GenerationSampler::default());
            let report = quote_sampling_workspace_with_observer(&sampler, temperature, random.as_ref(), &layout,
                &eredu_core::TokenFilter::All, 3, &context, Some(&mut observer)).unwrap();
            assert!(report.peak.bytes().is_some()); assert_eq!(report.first_gap, None);
            let rows = &observer.0.sampling; assert_eq!(rows.len(), 4);
            assert_eq!(rows[0].phase(), SamplingWorkspacePhase::Preparation);
            assert!(rows[0].completion().is_none()); assert!(rows[0].mutable_storage().is_some());
            for row in &rows[1..] {
                assert_eq!(row.first_missing_operation(), None);
                let completion = row.completion().expect("actual final ReadToken owns completion");
                assert_eq!(completion.traversal.roots(), if temperature == 0.0 { 1 } else { 2 });
                assert_eq!(completion.nested_completions, 0);
                let dispatch = completion.dispatch.unwrap(); assert_eq!(dispatch.gpu_entries, 0);
                assert!(dispatch.cpu_model.unwrap().primitives > 0);
            }
        }}
    }
}
