use super::*;
use eredu_nn::Tensor;

fn mechanism() -> (MlxMetalWorkspaceMechanisms, MlxCpuWorkspaceMechanisms) {
    let ordinary = MlxMetalWorkspaceMechanisms::current_host().unwrap();
    let selected = MlxCpuMatmulMechanism::select(eredu_nn::CpuMatmulImplementation::Float32Tiles).unwrap();
    (ordinary, MlxCpuWorkspaceMechanisms::new(ordinary.allocation(), selected))
}

#[test]
fn cpu_scalar_static_slices_share_exact_source_without_floating_representation() {
    let (ordinary, cpu) = mechanism();
    for dtype in [WorkspaceDtype::Int32, WorkspaceDtype::Uint32, WorkspaceDtype::Uint8, WorkspaceDtype::Bool] {
        for (rows, start, end, step) in [(38, 0, 5, 1), (230, 0, 96, 1),
            (230, 7, 103, 2), (38, 4, 5, 1), (38, 0, 38, 1)] {
            let context = WorkspaceContext::new(cpu);
            let storage = WorkspaceExistingStorage::try_new(Some(4096), &context).unwrap();
            let input = WorkspaceTensor::existing_with_storage(
                context.layout(&[rows, 1], dtype).unwrap(), &storage, &context).unwrap();
            context.begin_state_span([&input]).unwrap();
            let output = input.static_slice(&[start, 0], &[end, 1], &[step, 1], &context).unwrap();
            assert_eq!(output.shape(), [(end-start+step-1)/step, 1]);
            assert_eq!(output.layout().dtype(), dtype);
            assert_eq!(output.layout().representation(), None);
            let report = context.finish_report(&[output]).unwrap();
            assert!(report.unpriced_operations.is_empty());
            assert!(report.unpriced_host_operations.is_empty());
            assert_eq!(report.tensor_buffers.total_bytes, Some(0));
            assert_eq!(report.state.as_ref().unwrap().retained_bytes, Some(4096));
            assert_eq!(report.state.as_ref().unwrap().displaced_bytes, Some(0));
            let operation = report.operations[0].as_view();
            let plan = cpu.plan(operation).unwrap().unwrap();
            assert_eq!((plan.alias_input, plan.population.births, plan.output_bytes, plan.scratch_bytes),
                (Some(0), 0, 0, 0));
            assert_eq!(plan.population.primitives, usize::from(start != 0 || end != rows));
            assert_eq!(plan.parameter_shells, usize::from(start == 0 && end == rows));
            assert_eq!(cpu.output_representation(operation, 0), None);
            let recipe = SpeculativeNumericalRecipe::inspect_cpu_outputs(&report, 1, ordinary, cpu, &context).unwrap();
            assert_eq!(recipe.kernels, 0);
            assert_eq!(recipe.storage.maximum_births(), 0);
        }
        // The same source owns multi-axis rectangles; no contiguous-layout or
        // floating precision is invented to price this alias-only operation.
        let context = WorkspaceContext::new(cpu);
        let input = WorkspaceTensor::existing(context.layout(&[2, 7, 5, 3], dtype).unwrap(), &context).unwrap();
        context.begin_span();
        let output = input.static_slice(&[0, 1, 0, 1], &[2, 7, 5, 3], &[1, 2, 2, 1], &context).unwrap();
        assert_eq!(output.shape(), [2, 3, 3, 2]);
        assert_eq!(output.layout().representation(), None);
        let report = context.finish_report(&[output]).unwrap();
        let plan = cpu.plan(report.operations[0].as_view()).unwrap().unwrap();
        assert_eq!((plan.population.primitives, plan.population.births, plan.rank), (1, 0, 4));
    }
}

#[test]
fn cpu_scalar_static_slice_checks_coordinates_and_normalized_precision() {
    let (_, cpu) = mechanism();
    let source = WorkspaceLayoutView::new(&[230, 1], WorkspaceDtype::Int32).unwrap();
    let output = WorkspaceLayoutView::new(&[96, 1], WorkspaceDtype::Int32).unwrap();
    let inputs = [source]; let outputs = [output];
    let operation = WorkspaceOperationView {
        kind: WorkspaceOperationKindView::StaticSlice {starts: &[0, 0], ends: &[96, 1], strides: &[1, 1]},
        inputs: WorkspaceLayoutList::Views(&inputs), outputs: WorkspaceLayoutList::Views(&outputs),
    };
    assert!(cpu.plan(operation).unwrap().is_some());
    for kind in [
        WorkspaceOperationKindView::StaticSlice {starts: &[-1, 0], ends: &[95, 1], strides: &[1, 1]},
        WorkspaceOperationKindView::StaticSlice {starts: &[0], ends: &[96, 1], strides: &[1, 1]},
        WorkspaceOperationKindView::StaticSlice {starts: &[0, 0], ends: &[231, 1], strides: &[1, 1]},
        WorkspaceOperationKindView::StaticSlice {starts: &[0, 0], ends: &[96, 1], strides: &[0, 1]},
    ] { assert!(cpu.plan(WorkspaceOperationView {kind, ..operation}).is_err()); }
    let wrong_dtype = [WorkspaceLayoutView::new(&[96, 1], WorkspaceDtype::Uint32).unwrap()];
    assert!(cpu.plan(WorkspaceOperationView {outputs: WorkspaceLayoutList::Views(&wrong_dtype), ..operation}).is_err());
    // The canonical layout API removes floating evidence from integer
    // layouts. The normalized slice remains valid and cannot publish that
    // discarded evidence as an output representation.
    let floating = Some(WorkspaceRepresentation::new(WorkspaceFloatingType::Float32, true));
    let normalized_source = [source.with_representation(floating)];
    let normalized_output = [output.with_representation(floating)];
    assert_eq!(normalized_source[0].representation(), None);
    assert_eq!(normalized_output[0].representation(), None);
    for normalized in [
        WorkspaceOperationView {inputs: WorkspaceLayoutList::Views(&normalized_source), ..operation},
        WorkspaceOperationView {outputs: WorkspaceLayoutList::Views(&normalized_output), ..operation},
    ] {
        let plan = cpu.plan(normalized).unwrap().unwrap();
        assert_eq!((plan.alias_input, plan.population.births), (Some(0), 0));
        assert_eq!(cpu.output_representation(normalized, 0), None);
    }
    let float_source = [WorkspaceLayoutView::new(&[230, 1], WorkspaceDtype::Float32).unwrap()];
    let float_output = [WorkspaceLayoutView::new(&[96, 1], WorkspaceDtype::Float32).unwrap()];
    assert!(cpu.plan(WorkspaceOperationView {inputs: WorkspaceLayoutList::Views(&float_source),
        outputs: WorkspaceLayoutList::Views(&float_output), ..operation}).unwrap().is_none());
}
