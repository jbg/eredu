use super::*;
use eredu_nn::NeuralBackend;

fn mechanisms() -> (MlxMetalWorkspaceMechanisms, MlxCpuWorkspaceMechanisms) {
    let ordinary = MlxMetalWorkspaceMechanisms::current_host().unwrap();
    let selected =
        MlxCpuMatmulMechanism::select(eredu_nn::CpuMatmulImplementation::Float32Tiles).unwrap();
    (
        ordinary,
        MlxCpuWorkspaceMechanisms::new(ordinary.allocation(), selected),
    )
}

fn trace(shape: &[i32], dtype: WorkspaceFloatingType) -> SpeculativeNumericalRecipe {
    let (ordinary, cpu) = mechanisms();
    let context = WorkspaceContext::new(cpu);
    let input = WorkspaceTensor::existing(
        context
            .layout(shape, WorkspaceDtype::Float32)
            .unwrap()
            .with_representation(Some(WorkspaceRepresentation::new(dtype, true))),
        &context,
    )
    .unwrap();
    context.begin_state_span([]).unwrap();
    let output = WorkspaceBackend::softplus(input, std::f32::consts::LN_2, &context)
        .unwrap_or_else(|error| panic!("softplus shape={shape:?} dtype={dtype:?}: {error}"));
    assert_eq!(
        output.layout().representation(),
        Some(WorkspaceRepresentation::new(dtype, true))
    );
    let report = context.finish_report(&[output]).unwrap();
    assert!(report.unpriced_operations.is_empty() && report.unpriced_host_operations.is_empty());
    assert!(report.inference_transient_bytes().unwrap() > 0);
    let plan = cpu.plan(report.operations[0].as_view()).unwrap().unwrap();
    assert_eq!(plan.dtype, dtype);
    assert_eq!(
        plan.output_bytes,
        ordinary
            .allocation()
            .fixed_buffer_capacity(report.operations[0].outputs[0].bytes().unwrap())
            .unwrap(),
        "logical output floor for shape={shape:?} dtype={dtype:?}",
    );
    assert_eq!(
        report.tensor_buffers.retained_bytes,
        Some(plan.output_bytes)
    );
    assert_eq!(plan.population.primitives, 28);
    assert_eq!(plan.population.input_edges, 33);
    assert_eq!(plan.population.births, 19);
    assert_eq!(plan.seeds, 3);
    assert_eq!(plan.population.maximum_captures, 5);
    SpeculativeNumericalRecipe::inspect_cpu_equations(&report, ordinary, cpu, &context).unwrap()
}

#[test]
fn cpu_softplus_quotes_actual_casts_predicate_and_original_output_precision() {
    for shape in [&[][..], &[7][..], &[1, 2, 8][..], &[1, 2, 2, 2][..]] {
        for dtype in [
            WorkspaceFloatingType::Float32,
            WorkspaceFloatingType::Bfloat16,
            WorkspaceFloatingType::Float16,
        ] {
            let recipe = trace(shape, dtype);
            assert_eq!(recipe.kernels, 0);
            assert!(recipe.graph_capacity > 0 && recipe.record_capacity > 0);
            assert!(recipe.storage.maximum_births() >= 22);
        }
    }
}

#[test]
fn cpu_softplus_refuses_unknown_source_and_malformed_geometry() {
    let (_, cpu) = mechanisms();
    let shape = [1, 2, 8];
    let known = WorkspaceRepresentation::new(WorkspaceFloatingType::Float32, true);
    let input = WorkspaceLayoutView::new(&shape, WorkspaceDtype::Float32)
        .unwrap()
        .with_representation(Some(known));
    let inputs = [input];
    let outputs = [WorkspaceLayoutView::new(&shape, WorkspaceDtype::Float32).unwrap()];
    let operation = WorkspaceOperationView {
        kind: WorkspaceOperationKindView::Elementwise("softplus"),
        inputs: WorkspaceLayoutList::Views(&inputs),
        outputs: WorkspaceLayoutList::Views(&outputs),
    };
    assert!(cpu.plan(operation).unwrap().is_some());
    for representation in [
        None,
        Some(WorkspaceRepresentation::new(
            WorkspaceFloatingType::Float32,
            false,
        )),
    ] {
        let missing = [input.with_representation(representation)];
        let op = WorkspaceOperationView {
            inputs: WorkspaceLayoutList::Views(&missing),
            ..operation
        };
        assert!(cpu.plan(op).unwrap().is_none());
        assert!(cpu.output_representation(op, 0).is_none());
    }
    let wrong_shape = [WorkspaceLayoutView::new(&[2, 8], WorkspaceDtype::Float32).unwrap()];
    assert!(cpu
        .plan(WorkspaceOperationView {
            outputs: WorkspaceLayoutList::Views(&wrong_shape),
            ..operation
        })
        .is_err());
    assert!(cpu
        .plan(WorkspaceOperationView {
            inputs: WorkspaceLayoutList::Views(&[]),
            ..operation
        })
        .is_err());
    let context = WorkspaceContext::new(cpu);
    for beta in [0.0, -1.0, f32::INFINITY, f32::NAN] {
        let value = WorkspaceTensor::existing(
            context
                .layout(&shape, WorkspaceDtype::Float32)
                .unwrap()
                .with_representation(Some(known)),
            &context,
        )
        .unwrap();
        context.begin_span();
        assert!(WorkspaceBackend::softplus(value, beta, &context).is_err());
        assert!(context.finish_report(&[]).unwrap().operations.is_empty());
    }
}

mod native;
