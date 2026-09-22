//! Composition preserves the actual CPU view workers and their one Eval frontier.
use super::*;
use crate::backend::nn::workspace::{
    MlxCpuMatmulMechanism, MlxCpuWorkspaceMechanisms, ResidentExecutionMechanisms,
};
use eredu_nn::{CpuMatmulImplementation, Index, Tensor};

fn readout(mechanism: ResidentExecutionMechanisms, count: usize) -> ResidentCompletionRecipe {
    let context = WorkspaceContext::new(mechanism);
    let source = WorkspaceTensor::existing(
        WorkspaceLayout::new(&[1, 3, 8], WorkspaceDtype::Float32)
            .unwrap()
            .with_representation(Some(WorkspaceRepresentation::new(
                WorkspaceFloatingType::Float32,
                true,
            ))),
        &context,
    )
    .unwrap();
    context.begin_span();
    let outputs = (0..count)
        .map(|row| {
            source
                .index(&[Index::Full, Index::At(row as i32), Index::Full], &context)
                .unwrap()
        })
        .collect::<Vec<_>>();
    let report = context.finish_report(&outputs).unwrap();
    AutoregressiveReadoutRecipe::inspect(&report, count, 3, mechanism, &context)
        .unwrap()
        .completion()
}

#[test]
fn independent_cpu_view_composition_preserves_workers_and_rejects_mixed_sources() {
    let _pool = crate::tests::support::test_utils::initialize_original_sources();
    let ordinary = MlxMetalWorkspaceMechanisms::current_host().unwrap();
    let matmul = MlxCpuMatmulMechanism::select(CpuMatmulImplementation::Float32Tiles).unwrap();
    let mechanism = ResidentExecutionMechanisms::Cpu {
        ordinary,
        cpu: MlxCpuWorkspaceMechanisms::new(ordinary.allocation(), matmul),
    };
    let input = readout(mechanism, 1);
    let output = readout(mechanism, 3);
    let joined = input.checked_union(output).unwrap();
    let dispatch = joined.dispatch.unwrap();
    let population = dispatch.cpu_model.unwrap();
    assert_eq!(
        population.extents,
        input.dispatch.unwrap().cpu_model.unwrap().extents
            + output.dispatch.unwrap().cpu_model.unwrap().extents
    );
    assert_eq!(dispatch.cpu_entries, population.primitives + 1);
    assert_eq!(joined.traversal.limits().tape_entries, dispatch.cpu_entries);
    assert_eq!(
        joined.traversal.roots(),
        input.traversal.roots() + output.traversal.roots()
    );
    assert_eq!(dispatch.completion_streams(), Some(1));
    assert!(
        graph_capacity::ResidentGraphStorage::for_completion(joined)
            .unwrap()
            .full_capacity
            .is_some()
    );
    assert!(
        record_capacity::ResidentRecordStorage::for_completion(joined)
            .unwrap()
            .full_capacity
            .is_some()
    );
    let gpu = readout(ResidentExecutionMechanisms::Metal(ordinary), 1);
    assert!(input.checked_union(gpu).is_none());
    assert!(gpu.checked_union(input).is_none());
}

#[test]
fn captured_index_readout_prices_a_separate_bank_without_another_completion() {
    let _pool = crate::tests::support::test_utils::initialize_original_sources();
    let ordinary = MlxMetalWorkspaceMechanisms::current_host().unwrap();
    let matmul = MlxCpuMatmulMechanism::select(CpuMatmulImplementation::Float32Tiles).unwrap();
    for mechanism in [
        ResidentExecutionMechanisms::Cpu {
            ordinary,
            cpu: MlxCpuWorkspaceMechanisms::new(ordinary.allocation(), matmul),
        },
        ResidentExecutionMechanisms::Metal(ordinary),
    ] {
        let context = WorkspaceContext::new(mechanism);
        let source = WorkspaceTensor::existing(
            WorkspaceLayout::new(&[1, 3, 8], WorkspaceDtype::Float32)
                .unwrap()
                .with_representation(Some(WorkspaceRepresentation::new(
                    WorkspaceFloatingType::Float32,
                    true,
                ))),
            &context,
        )
        .unwrap();
        context.begin_span();
        let operation = context.operation_count();
        let selected = source
            .index(&[Index::Full, Index::At(2), Index::Full], &context)
            .unwrap();
        let report = context.finish_report(&[selected]).unwrap();
        let readout =
            AutoregressiveReadoutRecipe::inspect(&report, 1, 3, mechanism, &context).unwrap();
        let suffix = AutoregressiveReadoutRecipe::index_construction(
            &report.operations[operation],
            mechanism,
            &context,
        )
        .unwrap();
        let original = readout.completion();
        assert_eq!(
            suffix.allocation_extents(),
            original.graph.allocation_extents()
        );
        assert_eq!(
            suffix.additional_shells(),
            original.graph.additional_shells()
        );
        let mut capacity = graph_capacity::ResidentGraphStorage::for_completion(original).unwrap();
        let before = capacity.known_constructor_bytes;
        capacity.include_construction_bank(suffix).unwrap();
        assert_eq!(
            capacity.known_constructor_bytes,
            before
                .checked_add(suffix.allocation_extents() as u64)
                .unwrap()
        );
        assert_eq!(
            capacity.full_capacity,
            safemlx::SubmissionGraphQuota::fresh_capacity_for_extents(
                usize::try_from(capacity.known_constructor_bytes).unwrap()
            )
            .map(|n| n as u64)
        );
        assert_eq!(
            readout.completion().traversal.limits().tape_entries,
            original.traversal.limits().tape_entries
        );
        assert_eq!(readout.completion().nested_completions, 0);
    }
}
