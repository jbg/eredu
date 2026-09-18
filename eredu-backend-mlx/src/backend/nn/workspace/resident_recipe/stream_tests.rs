use super::*;

// These private scalar populations test source classification only. They do
// not provide native worker layouts, admitted resources or execution authority.
fn population(cpu: bool, routers: usize, collectives: usize) -> ResidentDispatchPopulation {
    let model = super::super::cpu::CpuPopulation {
        construction_entries: 7,
        primitives: 7,
        ..Default::default()
    };
    ResidentDispatchPopulation {
        cpu_model: cpu.then_some(model),
        gpu_entries: if cpu { 0 } else { 8 },
        gpu_input_edges: if cpu { 0 } else { 7 },
        gpu_siblings: if cpu { 0 } else { 8 },
        gpu_births: if cpu { 0 } else { 7 },
        additional_sort_kernels: 0,
        cpu_entries: if cpu {
            8 + collectives
        } else {
            routers + collectives
        },
        cpu_input_edges: if cpu {
            7 + collectives
        } else {
            routers + collectives
        },
        cpu_siblings: if cpu {
            8 + collectives
        } else {
            routers + collectives
        },
        parallel_entries: collectives,
        parallel_graph_extents: 0,
        worker_graph_extents: 0,
        worker_rank: 2,
        copy_rank_extents: 0,
        kernel_attempts: 0,
    }
}

#[test]
fn completion_stream_sources_keep_cpu_collectives_and_metal_router_distinct() {
    for (cpu, routers, collectives, streams) in [
        (true, 0, 0, 1),
        (true, 0, 1, 2),
        (true, 0, 9, 2),
        (false, 0, 0, 1),
        (false, 4, 0, 2),
        (false, 0, 3, 2),
        (false, 1, 1, 3),
        (false, 4, 9, 3),
    ] {
        let dispatch = population(cpu, routers, collectives);
        assert_eq!(
            dispatch.completion_streams(),
            Some(streams),
            "model CPU={cpu}, router entries={routers}, collective entries={collectives}"
        );
        assert_eq!(
            dispatch.parallel_entries, collectives,
            "sharing a stream never collapses the finite collective occurrence count"
        );
    }
}

#[test]
fn completion_stream_sources_reject_uncertified_or_inconsistent_populations() {
    for change in 0..11 {
        let mut invalid = population(true, 0, 2);
        match change {
            0 => invalid.gpu_entries = 1,
            1 => invalid.gpu_input_edges = 1,
            2 => invalid.gpu_siblings = 1,
            3 => invalid.gpu_births = 1,
            4 => invalid.worker_graph_extents = 1,
            5 => invalid.copy_rank_extents = 1,
            6 => invalid.kernel_attempts = 1,
            7 => invalid.additional_sort_kernels = 1,
            8 => invalid.cpu_entries -= 1,
            9 => invalid.cpu_entries += 1,
            10 => invalid.cpu_model.as_mut().unwrap().primitives = usize::MAX,
            _ => unreachable!(),
        }
        assert_eq!(
            invalid.completion_streams(),
            None,
            "CPU source mutation {change}"
        );
    }
    let mut unknown_cpu = population(true, 0, 2);
    unknown_cpu.cpu_model = None;
    assert_eq!(unknown_cpu.completion_streams(), None);
    let mut missing_model = population(false, 2, 1);
    missing_model.gpu_entries = 0;
    assert_eq!(missing_model.completion_streams(), None);
    let mut underflow = population(false, 0, 2);
    underflow.cpu_entries = 1;
    assert_eq!(
        underflow.completion_streams(),
        None,
        "collective entries cannot exceed the actual CPU entry population"
    );
}

#[cfg(all(target_vendor = "apple", feature = "metal", not(feature = "cuda")))]
fn equation_recipe(cpu: bool) -> ResidentNativeRecipe {
    use eredu_nn::Tensor;
    let ordinary = MlxMetalWorkspaceMechanisms::current_host().unwrap();
    let mechanism = if cpu {
        ResidentExecutionMechanisms::Cpu {
            ordinary,
            cpu: MlxCpuWorkspaceMechanisms::new(
                ordinary.allocation(),
                MlxCpuMatmulMechanism::select(eredu_nn::CpuMatmulImplementation::Float32Tiles)
                    .unwrap(),
            ),
        }
    } else {
        ResidentExecutionMechanisms::Metal(ordinary)
    };
    let context = match mechanism {
        ResidentExecutionMechanisms::Cpu { cpu, .. } => WorkspaceContext::new(cpu),
        ResidentExecutionMechanisms::Metal(metal) => WorkspaceContext::new(metal),
    };
    let geometry = InferenceGeometry {
        batch_size: 1,
        cached_positions: 0,
        input_positions: 1,
        max_output_tokens: 1,
        prefill_chunk_positions: 1,
        output: eredu_core::OutputDemand::Sequence,
    };
    let mut recorder = mechanism.recorder(geometry, &context).unwrap();
    let quoted = eredu_runtime::working_memory::quote_inference_workspace_with_context(
        geometry,
        &context,
        |span| -> Result<WorkspaceTraceReport, Error> {
            context.begin_state_span([])?;
            let layout = context
                .layout(&[1, 4], WorkspaceDtype::Float32)?
                .with_representation(Some(WorkspaceRepresentation::new(
                    WorkspaceFloatingType::Float32,
                    true,
                )));
            let input = WorkspaceTensor::existing(layout, &context)?;
            let output = input.tanh(&context)?;
            let report = context.report(&[output])?;
            recorder.record_equation(span, &report, 0, 1, None, None, false)?;
            Ok(report)
        },
    )
    .unwrap();
    let sampler =
        eredu_runtime::ConfiguredTextSampler::Standard(eredu_runtime::GenerationSampler::default());
    let logits = context
        .layout(&[1, 4], WorkspaceDtype::Float32)
        .unwrap()
        .with_representation(Some(WorkspaceRepresentation::new(
            WorkspaceFloatingType::Float32,
            true,
        )));
    eredu_runtime::working_memory::quote_sampling_workspace_with_observer(
        &sampler,
        0.0,
        None,
        &logits,
        &eredu_core::TokenFilter::All,
        1,
        &context,
        Some(&mut recorder),
    )
    .unwrap();
    recorder.finish(quoted.span_workspace_plan()).unwrap()
}

#[test]
#[cfg(all(target_vendor = "apple", feature = "metal", not(feature = "cuda")))]
fn retained_boundaries_use_source_stream_counts_and_keep_finite_attempts() {
    // Exercise the boundary's consumption of private source counts. Actual
    // native collective layouts and execution are covered by distributed runs;
    // this test never constructs or submits an admitted native operation.
    for (cpu, routers, collectives, streams) in [(true, 0, 2, 2), (false, 1, 2, 3)] {
        for wrong_streams in [false, true] {
            let mut recipe = equation_recipe(cpu);
            assert_eq!(recipe.records.len(), 2, "initial prefill and cached decode");
            let geometry = recipe.plan.geometry();
            let submissions = recipe.plan.generation_forward_count().unwrap().checked_mul(3).unwrap();
            let prior = recipe.records.iter().map(|row| (row.nested_completions, row.query_controls.unwrap())).collect::<Vec<_>>();
            for row in &mut recipe.records {
                let dispatch = row.dispatch.as_mut().unwrap();
                dispatch.parallel_entries += collectives;
                dispatch.cpu_entries += collectives + routers;
                dispatch.cpu_input_edges += collectives + routers;
                dispatch.cpu_siblings += collectives + routers;
                assert_eq!(dispatch.completion_streams(), Some(streams));
                let mut limits = row.traversal.unwrap().limits();
                limits.tape_entries += collectives + routers;
                limits.output_slots += collectives + routers;
                limits.input_edges += collectives + routers;
                limits.arrays += 2 * (collectives + routers);
                limits.streams = streams - usize::from(wrong_streams);
                row.traversal = Some(safemlx::OperationEvent::eval_traversal_layout(limits).unwrap());
            }
            let result = recipe.bind_neural_boundaries(geometry, 3, 2);
            if wrong_streams {
                assert!(matches!(
                    result,
                    Err(crate::backend::error::Error::NeuralBoundarySource {
                        requirement: "completion stream population",
                        ..
                    })
                ));
                assert!(recipe.neural.is_none());
                for (row, (completions, controls)) in recipe.records.iter().zip(&prior) {
                    assert_eq!(row.nested_completions, *completions);
                    assert_eq!(row.query_controls, Some(*controls));
                }
            } else {
                result.unwrap();
                for (row, (completions, controls)) in recipe.records.iter().zip(&prior) {
                    assert_eq!(row.nested_completions, completions + 3);
                    assert_eq!(row.query_controls,
                        Some(controls + ResidentDispatchPopulation::completion_stream_control_bytes()));
                }
                assert_eq!(recipe.neural_waits_per_forward(), 6);
                assert!(recipe.matches_neural_boundaries(geometry, submissions, 2));
                assert!(!recipe.matches_neural_boundaries(geometry, submissions + 1, 2));
                assert!(recipe.bind_neural_boundaries(geometry, 3, 2).is_err());
            }
        }
    }
}
