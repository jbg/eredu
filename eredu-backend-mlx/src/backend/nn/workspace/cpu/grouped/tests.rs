use super::*;
use eredu_nn::{
    CpuMatmulImplementation, GatedProductGroupLayout, GatedProductPolicy, GroupReduction,
    GroupSelection, GroupedGatedProductSpec, GroupedProjectionSpec, GroupedUnitBatch,
    GroupedUnitObserver, LinearFormatSpec, ParameterSpec,
};

struct Observe;
impl GroupedUnitObserver<WorkspaceTensor> for Observe {
    fn observe(
        &mut self,
        _: &GroupedUnitBatch<'_, WorkspaceTensor>,
    ) -> Result<(), eredu_nn::Error> {
        Ok(())
    }
}
fn mechanism(implementation: CpuMatmulImplementation) -> MlxCpuWorkspaceMechanisms {
    let original = MlxMetalWorkspaceMechanisms::current_host()
        .unwrap()
        .original_storage();
    MlxCpuWorkspaceMechanisms::new(
        original.allocation(),
        MlxCpuMatmulMechanism::select(implementation).unwrap(),
    )
}
fn trace(
    tokens: i32,
    routes: i32,
    observed: bool,
    policy: GatedProductPolicy,
    reduction: GroupReduction,
    cpu: MlxCpuWorkspaceMechanisms,
) -> WorkspaceTraceReport {
    let projection = |name| {
        GroupedProjectionSpec::new(
            ParameterSpec::trainable(name).unwrap(),
            None,
            LinearFormatSpec::unscaled(eredu_checkpoint::LinearFormat::Dense).unwrap(),
        )
        .unwrap()
    };
    let bank = WorkspaceGroupedBank::GatedProduct(
        GroupedGatedProductSpec::new(
            3,
            4,
            6,
            4,
            policy,
            GatedProductGroupLayout::Packed {
                gate_up: projection("read"),
                down: projection("write"),
            },
        )
        .unwrap()
        .with_reduction(reduction),
    );
    let context = WorkspaceContext::new(cpu);
    let value = |shape: &[i32], dtype| {
        let layout = WorkspaceLayout::new(shape, dtype).unwrap();
        let layout = if dtype == WorkspaceDtype::Float32 {
            layout.with_representation(Some(WorkspaceRepresentation::new(
                WorkspaceFloatingType::Float32,
                true,
            )))
        } else {
            layout
        };
        WorkspaceTensor::existing(layout, &context).unwrap()
    };
    let input = value(&[tokens, 4], WorkspaceDtype::Float32);
    let ids = value(&[tokens, routes], WorkspaceDtype::Uint32);
    let scores = value(&[tokens, routes], WorkspaceDtype::Float32);
    let weights = value(&[tokens, routes], WorkspaceDtype::Float32);
    let read = value(&[3, 12, 4], WorkspaceDtype::Float32);
    let write = value(&[3, 4, 6], WorkspaceDtype::Float32);
    context.begin_state_span([]).unwrap();
    let result = bank
        .trace_with_parameters(
            &[&read, &write],
            &input,
            &GroupSelection::new(ids, scores, weights),
            None,
            &context,
            observed.then_some(&mut Observe),
        )
        .unwrap();
    let (output, bias) = result.into_parts();
    assert!(bias.is_none());
    context.report(&[output]).unwrap()
}

#[test]
fn cpu_grouped_sources_cover_chunked_sum_and_sequential_equations() {
    let cpu = mechanism(CpuMatmulImplementation::Float32Tiles);
    for policy in [
        GatedProductPolicy::ordinary_silu(),
        GatedProductPolicy::ordinary_gelu_approximate(),
        GatedProductPolicy::bounded_silu(1.7).unwrap(),
        GatedProductPolicy::new(
            eredu_nn::GatedProductActivation::Silu,
            Some(1.3),
            Some(2.1),
            1.7,
            0.2,
        )
        .unwrap(),
    ] {
        for reduction in [GroupReduction::Sum, GroupReduction::SequentialGroupOrder] {
            for rows in [1, 2, 31, 32, 64, 65, 97] {
                for routes in [1, 2] {
                    let report = trace(rows, routes, false, policy, reduction, cpu);
                    assert!(
                        report.unpriced_operations.is_empty(),
                        "{rows} {policy:?} {reduction:?}: {:?}",
                        report.unpriced_operations
                    );
                    assert!(report.unpriced_host_operations.is_empty());
                    let transient = report.inference_transient_bytes().unwrap_or_else(|| panic!(
                        "CPU grouped rows={rows}, routes={routes}, policy={policy:?}, reduction={reduction:?}: state={:?}, tensor={:?}, host={:?}",
                        report.state, report.tensor_buffers, report.host_workspace_bytes));
                    assert!(transient > 0);
                    assert_eq!(report.host_workspace_bytes, Some(0));
                    let op = report
                        .operations
                        .iter()
                        .find(|op| matches!(op.kind, WorkspaceOperationKind::Grouped { .. }))
                        .unwrap();
                    let plan = cpu.plan(op.as_view()).unwrap().unwrap();
                    assert_eq!(plan.validations, 1);
                    let ordinary = cpu.ordinary_storage();
                    let calls = ordinary
                        .ordinary_call_controls(op.as_view())
                        .unwrap()
                        .unwrap_or_else(|| panic!(
                            "ordinary grouped caller: rows={rows}, routes={routes}, policy={policy:?}, reduction={reduction:?}"
                        ));
                    assert!(calls.metadata_bytes > 0);
                    // Packed ordinary execution does not register the Original
                    // mask predicate, despite its conservative native envelope.
                    assert_eq!(ordinary.plan(op.as_view()).unwrap().unwrap().validations, 0);
                    assert!(plan.population.births > 0 && plan.population.controls > 0);
                    assert_eq!(
                        plan.population.maximum_captures,
                        if rows > 64 {
                            (1 + 2 * (rows as usize).div_ceil(32)).max(6)
                        } else {
                            6
                        }
                    );
                    let storage = output_storage(op.as_view()).unwrap();
                    assert_eq!(storage.calls, usize::from(rows > 64));
                    assert_eq!(
                        storage.chunks,
                        if rows > 64 {
                            (rows as usize).div_ceil(32)
                        } else {
                            0
                        }
                    );
                }
            }
        }
    }
}

#[test]
fn cpu_grouped_unit_split_preserves_population_and_actual_callback_storage() {
    let cpu = mechanism(CpuMatmulImplementation::Float32Tiles);
    for rows in [1, 64, 65, 97] {
        let whole = trace(
            rows,
            2,
            false,
            GatedProductPolicy::ordinary_silu(),
            GroupReduction::Sum,
            cpu,
        );
        let split = trace(
            rows,
            2,
            true,
            GatedProductPolicy::ordinary_silu(),
            GroupReduction::Sum,
            cpu,
        );
        assert!(
            split.unpriced_operations.is_empty(),
            "{:?}",
            split.unpriced_operations
        );
        assert!(
            split.inference_transient_bytes().is_some(),
            "CPU grouped split rows={rows}: state={:?}, tensor={:?}, host={:?}",
            split.state,
            split.tensor_buffers,
            split.host_workspace_bytes
        );
        let plans = |report: &WorkspaceTraceReport| {
            report
                .operations
                .iter()
                .filter(|op| matches!(op.kind, WorkspaceOperationKind::Grouped { .. }))
                .map(|op| {
                    (
                        cpu.plan(op.as_view()).unwrap().unwrap(),
                        output_storage(op.as_view()).unwrap(),
                    )
                })
                .collect::<Vec<_>>()
        };
        let full = plans(&whole);
        let phases = plans(&split);
        assert_eq!(full.len(), 1);
        assert_eq!(phases.len(), 2);
        assert_eq!(
            full[0].0.population.primitives,
            phases
                .iter()
                .map(|p| p.0.population.primitives)
                .sum::<usize>()
        );
        assert_eq!(
            full[0].0.population.births,
            phases.iter().map(|p| p.0.population.births).sum::<usize>()
        );
        assert_eq!(
            full[0].0.seeds,
            phases.iter().map(|p| p.0.seeds).sum::<usize>()
        );
        assert_eq!(phases[0].0.validations, 1);
        assert_eq!(phases[1].0.validations, 0);
        assert_eq!(phases[0].1.unit_observers, 1);
        assert_eq!(phases[1].1.unit_observers, 0);
        assert_eq!(phases[0].1.calls, 0);
        assert_eq!(phases[1].1.calls, usize::from(rows > 64));
        let ordinary = cpu.ordinary_storage();
        for operation in split
            .operations
            .iter()
            .filter(|operation| matches!(operation.kind, WorkspaceOperationKind::Grouped { .. }))
        {
            assert!(
                ordinary
                    .ordinary_call_controls(operation.as_view())
                    .unwrap()
                    .is_some()
            );
            assert_eq!(
                ordinary
                    .plan(operation.as_view())
                    .unwrap()
                    .unwrap()
                    .validations,
                0
            );
        }
    }
}

#[test]
fn cpu_grouped_source_requires_selected_worker_and_exact_physical_descriptor() {
    let cpu = mechanism(CpuMatmulImplementation::Float32Tiles);
    let report = trace(
        3,
        2,
        false,
        GatedProductPolicy::ordinary_silu(),
        GroupReduction::Sum,
        cpu,
    );
    let operation = report
        .operations
        .iter()
        .find(|op| matches!(op.kind, WorkspaceOperationKind::Grouped { .. }))
        .unwrap();
    assert!(cpu.plan(operation.as_view()).unwrap().is_some());
    let platform = mechanism(CpuMatmulImplementation::PlatformDefault);
    assert!(platform.plan(operation.as_view()).unwrap().is_none());
    let mut unrepresented = operation.clone();
    unrepresented.inputs[4] = unrepresented.inputs[4].clone().with_representation(None);
    assert!(cpu.plan(unrepresented.as_view()).unwrap().is_none());
    let mut half = operation.clone();
    half.inputs[4] =
        half.inputs[4]
            .clone()
            .with_representation(Some(WorkspaceRepresentation::new(
                WorkspaceFloatingType::Float16,
                true,
            )));
    assert!(cpu.plan(half.as_view()).unwrap().is_none());
    let mut malformed = operation.clone();
    malformed.inputs[4] = WorkspaceLayout::new(&[3, 11, 4], WorkspaceDtype::Float32)
        .unwrap()
        .with_representation(Some(WorkspaceRepresentation::new(
            WorkspaceFloatingType::Float32,
            true,
        )));
    assert!(cpu.plan(malformed.as_view()).is_err());
}

#[test]
fn metal_grouped_callers_retain_unknown_operand_precision_and_require_f32_banks() {
    let cpu = mechanism(CpuMatmulImplementation::Float32Tiles);
    let report = trace(
        3,
        2,
        false,
        GatedProductPolicy::ordinary_silu(),
        GroupReduction::Sum,
        cpu,
    );
    let mut operation = report
        .operations
        .iter()
        .find(|op| matches!(op.kind, WorkspaceOperationKind::Grouped { .. }))
        .unwrap()
        .clone();
    let quoted = ordinary_metal_call_controls(operation.as_view()).unwrap();
    for index in [0, 2, 3] {
        operation.inputs[index] = operation.inputs[index].clone().with_representation(None);
    }
    assert!(cpu.plan(operation.as_view()).unwrap().is_none());
    assert_eq!(
        ordinary_metal_call_controls(operation.as_view()),
        Some(quoted)
    );
    assert!(operation.inputs[0].representation().is_none());
    assert!(operation.inputs[2].representation().is_none());
    assert!(operation.inputs[3].representation().is_none());
    operation.inputs[4] = operation.inputs[4].clone().with_representation(None);
    assert!(ordinary_metal_call_controls(operation.as_view()).is_none());
}

#[test]
fn cpu_grouped_original_unsigned_ids_keep_relu_and_tp_bias_cast_births() {
    let cpu = mechanism(CpuMatmulImplementation::Float32Tiles);
    let report = trace(
        3,
        2,
        false,
        GatedProductPolicy::ordinary_silu(),
        GroupReduction::Sum,
        cpu,
    );
    let source = report
        .operations
        .iter()
        .find(|op| matches!(op.kind, WorkspaceOperationKind::Grouped { .. }))
        .unwrap();
    let projection = |name: &str, bias: bool| {
        GroupedProjectionSpec::new(
            ParameterSpec::trainable(name).unwrap(),
            bias.then(|| ParameterSpec::trainable(format!("{name}_bias")).unwrap()),
            LinearFormatSpec::unscaled(eredu_checkpoint::LinearFormat::Dense).unwrap(),
        )
        .unwrap()
    };
    let floating = |shape: &[i32]| {
        WorkspaceLayout::new(shape, WorkspaceDtype::Float32)
            .unwrap()
            .with_representation(Some(WorkspaceRepresentation::new(
                WorkspaceFloatingType::Float32,
                true,
            )))
    };
    let mut relu = source.clone();
    let WorkspaceOperationKind::Grouped { bank, .. } = &mut relu.kind else {
        unreachable!()
    };
    **bank = WorkspaceGroupedBank::Relu2(
        eredu_nn::GroupedRelu2Spec::new(
            3,
            4,
            6,
            projection("read", false),
            projection("write", false),
        )
        .unwrap(),
    );
    relu.inputs[4] = floating(&[3, 6, 4]);
    let mut biased = source.clone();
    let WorkspaceOperationKind::Grouped {
        bank, partitions, ..
    } = &mut biased.kind
    else {
        unreachable!()
    };
    **bank = WorkspaceGroupedBank::GatedProduct(
        GroupedGatedProductSpec::new(
            3,
            4,
            6,
            4,
            GatedProductPolicy::ordinary_silu(),
            GatedProductGroupLayout::Packed {
                gate_up: projection("read", true),
                down: projection("write", true),
            },
        )
        .unwrap(),
    );
    *partitions = Some(2);
    biased.inputs.insert(5, floating(&[3, 12]));
    biased.inputs.push(floating(&[3, 4]));
    biased.outputs.push(biased.outputs[0].clone());
    // The canonical cast query prices its General-copy destination for both
    // identity and converting calls. Frontend elision/donation never refunds
    // that envelope. UInt32 still must obtain its own positively qualified
    // conversion source; same-width signed IDs cannot stand in for it.
    let identity =
        OperationEvent::cpu_cast_layout(Dtype::Int32, Dtype::Int32, 1, 6, false).unwrap();
    let conversion =
        OperationEvent::cpu_cast_layout(Dtype::Uint32, Dtype::Int32, 1, 6, false).unwrap();
    assert_eq!(identity.backing_births(), 1);
    assert_eq!(conversion.backing_births(), 1);
    let difference = conversion
        .backing_births()
        .checked_sub(identity.backing_births())
        .unwrap();
    for (operation, conversions) in [(relu, 1), (biased, 2)] {
        let unsigned = cpu.plan(operation.as_view()).unwrap().unwrap();
        let mut signed = operation.clone();
        signed.inputs[1] = WorkspaceLayout::new(&[3, 2], WorkspaceDtype::Int32).unwrap();
        let signed = cpu.plan(signed.as_view()).unwrap().unwrap();
        assert_eq!(
            unsigned.population.births - signed.population.births,
            difference * conversions
        );
        assert_eq!(
            unsigned.scratch_bytes - signed.scratch_bytes,
            cpu.allocation.fixed_buffer_capacity(6 * 4).unwrap()
                * (difference * conversions) as u64
        );
    }
}
