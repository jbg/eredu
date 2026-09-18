//! Shared forced/sampled validation trace keeps its independent assertion roots.
use super::*;
use eredu_nn::Tensor;
use eredu_runtime::{working_memory::WorkspaceSamplingBackend, SamplingBackend, TokenDomain};

#[test]
fn sampled_token_source_retains_domain_assertions_in_metal_and_cpu_completion() {
    let metal = MlxMetalWorkspaceMechanisms::current_host().unwrap();
    let matmul =
        MlxCpuMatmulMechanism::select(eredu_nn::CpuMatmulImplementation::Float32Tiles).unwrap();
    let cpu = super::super::cpu::MlxCpuWorkspaceMechanisms::new(metal.allocation(), matmul);
    for on_cpu in [false, true] {
        for dtype in [WorkspaceDtype::Int32, WorkspaceDtype::Uint32] {
            for shape in [&[][..], &[1, 1][..], &[2, 3][..], &[0][..]] {
                let context = if on_cpu {
                    WorkspaceContext::new(cpu)
                } else {
                    WorkspaceContext::new(metal)
                };
                let input =
                    WorkspaceTensor::existing(context.layout(shape, dtype).unwrap(), &context)
                        .unwrap();
                context.begin_state_span([&input]).unwrap();
                let output = WorkspaceSamplingBackend::validate_token(
                    &input,
                    TokenDomain::new(33),
                    &context,
                )
                .unwrap();
                assert_eq!(output.shape(), shape);
                assert_eq!(output.layout().dtype(), WorkspaceDtype::Int32);
                assert_eq!(output.layout().representation(), None);
                let report = context.report(&[output]).unwrap();
                let mut recorder = ResidentRecipeRecorder::new(
                    InferenceGeometry {
                        batch_size: 1,
                        cached_positions: 0,
                        input_positions: 1,
                        max_output_tokens: 0,
                        prefill_chunk_positions: 1,
                        output: eredu_core::OutputDemand::Sequence,
                    },
                    metal,
                );
                if on_cpu {
                    recorder.cpu = Some(cpu);
                }
                let reduced = recorder.reduce_trace(&report, None, 0, 1).unwrap();
                assert_eq!(reduced.first_missing_operation, None);
                let assertions = usize::from(input.layout().elements().unwrap() != 0);
                assert_eq!(reduced.validation_roots, assertions);
                assert_eq!(reduced.traversal.unwrap().roots(), 1 + assertions);
                assert_eq!(reduced.graph.unwrap().seeds(), 2 * assertions);
                if assertions != 0 {
                    let owner = reduced
                        .validation_producers
                        .expect("hidden Boolean producer must retain its source");
                    assert!(owner.mutable_bytes() > 0);
                    assert!(owner.maximum_births() > 0);
                }
                // The shared declaration cannot turn an invalid cardinality
                // into a native primitive or an assertion-free successful path.
                let mut invalid = report;
                invalid.operations[0].kind = WorkspaceOperationKind::Sampling(
                    eredu_nn::workspace::WorkspaceSamplingOperation::ValidateToken {
                        cardinality: 0,
                    },
                );
                if on_cpu {
                    assert!(cpu.plan(invalid.operations[0].as_view()).is_err());
                    assert!(recorder.reduce_trace(&invalid, None, 0, 1).is_err());
                } else {
                    let refused = recorder.reduce_trace(&invalid, None, 0, 1).unwrap();
                    assert_eq!(refused.first_missing_operation, Some(0));
                }
            }
        }
    }
}

#[test]
fn terminal_resume_recipe_has_known_empty_equations_and_no_prediction_completion() {
    use eredu_runtime::working_memory::{quote_inference_workspace_with_context,
        quote_sampling_workspace_with_observer};
    let _sources = crate::tests::support::test_utils::initialize_original_sources();
    let mechanism = MlxMetalWorkspaceMechanisms::current_host().unwrap();
    let context = WorkspaceContext::new(mechanism);
    let geometry = InferenceGeometry { batch_size: 1, cached_positions: 7, input_positions: 0,
        max_output_tokens: 0, prefill_chunk_positions: 0, output: eredu_core::OutputDemand::StateOnly };
    let equations = quote_inference_workspace_with_context(geometry, &context,
        |_| -> Result<WorkspaceTraceReport, Error> { panic!("terminal restore invoked model equations") }).unwrap();
    let mut recorder = ResidentRecipeRecorder::with_context(geometry, mechanism, &context).unwrap();
    let sampler = eredu_runtime::ConfiguredTextSampler::Standard(eredu_runtime::GenerationSampler::default());
    let layout = context.layout(&[1, 33], WorkspaceDtype::Float32).unwrap();
    let sampling = quote_sampling_workspace_with_observer(&sampler, 0.0, None, &layout,
        &eredu_core::TokenFilter::All, 0, &context, Some(&mut recorder)).unwrap();
    assert_eq!(sampling.steps, 0);
    let mut recipe = recorder.finish(equations.span_workspace_plan()).unwrap();
    assert!(recipe.records().is_empty());
    assert_eq!(recipe.sampling_program().steps(), Some(0));
    assert!(recipe.sampling_program().completion(0).is_none());
    let copy = crate::backend::array_copy::OriginalCopyLayoutBuilder::new()
        .finish_resume_completed_input(0).unwrap();
    assert_eq!((copy.births(), copy.roots(), copy.logical_bytes()), (0, 0, 0));
    recipe.bind_resume_copy(copy).unwrap();
    assert!(recipe.resume_graph_storage_requirement().unwrap().full_capacity.is_some());
    assert!(recipe.record_storage_requirement().unwrap().full_capacity.is_some());
}
