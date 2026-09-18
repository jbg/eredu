use super::*;
use crate::backend::nn::logical_collective::{self, Operations};
use eredu_nn::Tensor;

fn mechanisms() -> (MlxMetalWorkspaceMechanisms, MlxCpuWorkspaceMechanisms) {
    let ordinary = MlxMetalWorkspaceMechanisms::current_host().unwrap();
    let selected =
        MlxCpuMatmulMechanism::select(eredu_nn::CpuMatmulImplementation::Float32Tiles).unwrap();
    (
        ordinary,
        MlxCpuWorkspaceMechanisms::new(ordinary.allocation(), selected),
    )
}

#[test]
fn cpu_integer_binary_broadcasts_preserve_dtype_and_exact_sources() {
    let (ordinary, cpu) = mechanisms();
    for (left, right) in [
        (&[][..], &[][..]),
        (&[18][..], &[][..]),
        (&[][..], &[18][..]),
        (&[18][..], &[18][..]),
        (&[2, 1][..], &[1, 3][..]),
        (&[2, 1, 3, 1][..], &[1, 4, 1, 5][..]),
    ] {
        for kind in ["add", "subtract", "multiply"] {
            let context = WorkspaceContext::new(cpu);
            let input = |shape| {
                WorkspaceTensor::existing(
                    context.layout(shape, WorkspaceDtype::Int32).unwrap(),
                    &context,
                )
                .unwrap()
            };
            let a = input(left);
            let b = input(right);
            context.begin_state_span([&a, &b]).unwrap();
            let output = match kind {
                "add" => a.add(&b, &context),
                "subtract" => a.subtract(&b, &context),
                _ => a.multiply(&b, &context),
            }
            .unwrap();
            assert_eq!(output.layout().dtype(), WorkspaceDtype::Int32);
            assert_eq!(output.layout().representation(), None);
            let output_shape = output.shape().to_vec();
            let report = context.finish_report(&[output]).unwrap();
            assert!(report.unpriced_operations.is_empty());
            assert!(report.unpriced_host_operations.is_empty());
            let plan = cpu.plan(report.operations[0].as_view()).unwrap().unwrap();
            let aliases = usize::from(left != output_shape) + usize::from(right != output_shape);
            assert_eq!(plan.population.primitives, 1 + aliases);
            assert_eq!(plan.population.input_edges, 2 + aliases);
            assert_eq!(
                (plan.population.births, plan.seeds, plan.scratch_bytes),
                (1, 0, 0)
            );
            let recipe = SpeculativeNumericalRecipe::inspect_cpu_outputs(
                &report, 1, ordinary, cpu, &context,
            )
            .unwrap();
            assert_eq!(recipe.storage.maximum_births(), 1);
            assert_eq!(recipe.kernels, 0);
        }
    }
}

#[test]
fn cpu_integer_binary_refuses_incompatible_or_unbounded_descriptors() {
    let (_, cpu) = mechanisms();
    let inspect = |left: &[i32], right: &[i32], output: &[i32], dtype| {
        let inputs = [
            WorkspaceLayoutView::new(left, WorkspaceDtype::Int32).unwrap(),
            WorkspaceLayoutView::new(right, dtype).unwrap(),
        ];
        let outputs = [WorkspaceLayoutView::new(output, WorkspaceDtype::Int32).unwrap()];
        cpu.plan(WorkspaceOperationView {
            kind: WorkspaceOperationKindView::Elementwise("multiply"),
            inputs: WorkspaceLayoutList::Views(&inputs),
            outputs: WorkspaceLayoutList::Views(&outputs),
        })
    };
    assert!(inspect(&[18], &[], &[17], WorkspaceDtype::Int32).is_err());
    assert!(inspect(&[18], &[17], &[18], WorkspaceDtype::Int32).is_err());
    for (left, right, output, dtype) in [
        (&[0][..], &[][..], &[0][..], WorkspaceDtype::Int32),
        (
            &[i32::MAX, 2][..],
            &[][..],
            &[i32::MAX, 2][..],
            WorkspaceDtype::Int32,
        ),
        (
            &[1, 1, 1, 1, 1][..],
            &[][..],
            &[1, 1, 1, 1, 1][..],
            WorkspaceDtype::Int32,
        ),
        (&[18][..], &[][..], &[18][..], WorkspaceDtype::Uint32),
        (&[18][..], &[][..], &[18][..], WorkspaceDtype::Float32),
    ] {
        assert!(inspect(left, right, output, dtype).unwrap().is_none());
    }
}

fn trace_peer(
    shape: &[i32],
    ordinary: MlxMetalWorkspaceMechanisms,
    cpu: MlxCpuWorkspaceMechanisms,
) -> SpeculativeNumericalRecipe {
    let context = WorkspaceContext::new(cpu);
    let input = |shape| {
        WorkspaceTensor::existing(
            context.layout(shape, WorkspaceDtype::Int32).unwrap(),
            &context,
        )
        .unwrap()
    };
    let sent = input(shape);
    let received = input(shape);
    let factor = input(&[]);
    let offset = input(&[]);
    context
        .begin_state_span([&sent, &received, &factor, &offset])
        .unwrap();
    let ops = logical_collective::Workspace(&context);
    let zero = ops.zero(&sent).unwrap();
    let peer = logical_collective::peer(&ops, &sent, &received, &zero).unwrap();
    let output = peer
        .multiply(&factor, &context)
        .unwrap()
        .subtract(&offset, &context)
        .unwrap();
    assert_eq!(output.layout().representation(), None);
    let report = context.finish_report(&[output]).unwrap();
    assert!(report.unpriced_operations.is_empty());
    assert!(report.unpriced_host_operations.is_empty());
    assert!(report.operations.iter().all(|op| {
        op.outputs
            .iter()
            .all(|v| v.dtype() == WorkspaceDtype::Int32 && v.representation().is_none())
    }));
    SpeculativeNumericalRecipe::inspect_cpu_outputs(&report, 1, ordinary, cpu, &context).unwrap()
}

#[test]
fn cpu_integer_peer_quotes_actual_zero_multiply_add_sequence() {
    let (ordinary, cpu) = mechanisms();
    for shape in [&[][..], &[1][..], &[18][..], &[2, 3][..]] {
        let recipe = trace_peer(shape, ordinary, cpu);
        assert_eq!(recipe.kernels, 0);
        assert!(recipe.storage.maximum_births() >= 5);
        assert!(recipe.graph_capacity > 0 && recipe.record_capacity > 0);
    }
}

#[path = "source_tests/native.rs"]
mod native;
