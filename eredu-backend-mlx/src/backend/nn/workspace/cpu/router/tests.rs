use super::*;
use eredu_nn::workspace::WorkspaceParameterRepresentation;
use eredu_nn::{
    GroupSelectionOperator, GroupedNeuralBackend, LinearFormatSpec, ParameterSpec,
    RoutingArithmetic, Tensor, TopKGroupSelectionSpec,
};

fn mechanisms() -> (MlxMetalWorkspaceMechanisms, MlxCpuWorkspaceMechanisms) {
    let ordinary = MlxMetalWorkspaceMechanisms::current_host().unwrap();
    let selected =
        MlxCpuMatmulMechanism::select(eredu_nn::CpuMatmulImplementation::Float32Tiles).unwrap();
    (
        ordinary,
        MlxCpuWorkspaceMechanisms::new(ordinary.allocation(), selected),
    )
}
fn spec(scoring: GroupScoring, top_k: i32, normalize: bool, scale: f32) -> TopKGroupSelectorSpec {
    TopKGroupSelectorSpec::new(
        8,
        ParameterSpec::trainable("weight").unwrap(),
        LinearFormatSpec::unscaled(eredu_checkpoint::LinearFormat::Dense).unwrap(),
        TopKGroupSelectionSpec::new(3, top_k, scoring, normalize)
            .unwrap()
            .with_weight_policy(1e-6, scale)
            .unwrap(),
    )
    .unwrap()
    .with_correction_bias(ParameterSpec::trainable("correction").unwrap())
    .unwrap()
    .with_arithmetic(RoutingArithmetic {
        projection: RoutingPrecision::Input,
        scores: RoutingPrecision::Float32,
        coefficients: RoutingPrecision::Input,
    })
}
fn trace(
    spec: TopKGroupSelectorSpec,
    rows: i32,
    supplied: bool,
    dtype: WorkspaceDtype,
    cpu: MlxCpuWorkspaceMechanisms,
) -> (WorkspaceContext, WorkspaceTraceReport) {
    trace_shape(spec, &[1, rows, 8], supplied, dtype, cpu)
}
fn trace_shape(
    spec: TopKGroupSelectorSpec,
    shape: &[i32],
    supplied: bool,
    dtype: WorkspaceDtype,
    cpu: MlxCpuWorkspaceMechanisms,
) -> (WorkspaceContext, WorkspaceTraceReport) {
    let rows = shape.iter().product::<i32>() / 8;
    let context = WorkspaceContext::new(cpu);
    let repr = Some(WorkspaceRepresentation::new(
        WorkspaceFloatingType::Float32,
        true,
    ));
    let parameter = |name: &str, shape: &[i32]| {
        WorkspaceParameterRepresentation::new(
            eredu_nn::ParameterId::new(name).unwrap(),
            context
                .layout(shape, WorkspaceDtype::Float32)
                .unwrap()
                .with_representation(repr),
        )
    };
    let mut parameters = vec![parameter("weight", &[3, 8]), parameter("correction", &[3])];
    if spec.bias().is_some() {
        parameters.push(parameter("bias", &[3]));
    }
    if spec.input_transform().is_some() {
        parameters.push(parameter("input_scale", &[8]));
    }
    if spec.coefficient_scale().is_some() {
        parameters.push(parameter("coefficient_scale", &[3]));
    }
    context
        .install_parameter_representations(parameters)
        .unwrap();
    let k = spec.selection().top_k();
    let mut selector = WorkspaceBackend::top_k_group_selector(spec, &context).unwrap();
    let input = WorkspaceTensor::existing(
        context
            .layout(shape, WorkspaceDtype::Float32)
            .unwrap()
            .with_representation(repr),
        &context,
    )
    .unwrap();
    let ids =
        WorkspaceTensor::existing(context.layout(&[rows, k], dtype).unwrap(), &context).unwrap();
    context.begin_state_span([&input, &ids]).unwrap();
    let result = if supplied {
        selector.select_indices(&input, &ids, &context)
    } else {
        selector.select(&input, &context)
    }
    .unwrap();
    for value in [result.selected_scores(), result.coefficients()] {
        assert_eq!(
            value.layout().representation().unwrap().dtype(),
            WorkspaceFloatingType::Float32
        );
    }
    assert_eq!(result.group_indices().layout().representation(), None);
    let report = context
        .finish_report(&[
            result.group_indices().clone(),
            result.selected_scores().clone(),
            result.coefficients().clone(),
        ])
        .unwrap();
    (context, report)
}

#[test]
fn cpu_router_sources_cover_scores_normalization_bias_scale_and_supplied_ids() {
    let _sources = crate::tests::support::test_utils::initialize_original_sources();
    let (ordinary, cpu) = mechanisms();
    for scoring in [
        GroupScoring::Softmax,
        GroupScoring::SelectedSoftmax,
        GroupScoring::Sigmoid,
        GroupScoring::SqrtSoftplus,
    ] {
        for (rows, k) in [(1, 1), (2, 2), (5, 3)] {
            for (supplied, dtype) in [
                (false, WorkspaceDtype::Uint32),
                (true, WorkspaceDtype::Uint32),
                (true, WorkspaceDtype::Int32),
            ] {
                for (normalize, scale) in [(false, 1.0), (true, 1.7)] {
                    let (context, report) = trace(
                        spec(scoring, k, normalize, scale),
                        rows,
                        supplied,
                        dtype,
                        cpu,
                    );
                    assert!(
                        report.unpriced_operations.is_empty(),
                        "{scoring:?}, rows={rows}, k={k}, supplied={supplied}: {:?}",
                        report.unpriced_operations
                    );
                    assert!(report.unpriced_host_operations.is_empty());
                    let operation = report
                        .operations
                        .iter()
                        .find(|op| matches!(op.kind, WorkspaceOperationKind::GroupSelection { .. }))
                        .unwrap();
                    let plan = cpu.plan(operation.as_view()).unwrap().unwrap();
                    let g = geometry(operation.as_view()).unwrap().unwrap();
                    let (_, retained) = storage(g, ordinary.allocation()).unwrap();
                    assert_eq!(plan.output_bytes, retained);
                    assert_eq!(report.tensor_buffers.retained_bytes, Some(retained));
                    let recipe = SpeculativeNumericalRecipe::inspect_cpu_outputs(
                        &report, 3, ordinary, cpu, &context,
                    )
                    .unwrap();
                    assert_eq!(recipe.kernels, 0);
                    assert!(recipe.graph_capacity > 0 && recipe.record_capacity > 0);
                    assert!(recipe.storage.maximum_births() > 0 && recipe.controls > 0);
                }
            }
        }
    }
}

#[test]
fn cpu_router_unknown_or_mismatched_sources_cannot_grant_output_representation() {
    let (_, cpu) = mechanisms();
    let (_, report) = trace(
        spec(GroupScoring::Sigmoid, 2, true, 1.7),
        2,
        false,
        WorkspaceDtype::Uint32,
        cpu,
    );
    let source = report
        .operations
        .into_iter()
        .find(|op| matches!(op.kind, WorkspaceOperationKind::GroupSelection { .. }))
        .unwrap();
    for index in 0..source.inputs.len() {
        let mut op = source.clone();
        op.inputs[index] = op.inputs[index].clone().with_representation(None);
        assert!(cpu.plan(op.as_view()).unwrap().is_none());
        for output in 0..3 {
            assert!(cpu.output_representation(op.as_view(), output).is_none());
        }
    }
    let mut op = source.clone();
    op.outputs[1] = WorkspaceLayout::new(&[2, 1], WorkspaceDtype::Float32).unwrap();
    assert!(cpu.plan(op.as_view()).is_err());
    let mut op = source;
    op.inputs[0] = WorkspaceLayout::new(&[1, 2, 7], WorkspaceDtype::Float32)
        .unwrap()
        .with_representation(Some(WorkspaceRepresentation::new(
            WorkspaceFloatingType::Float32,
            true,
        )));
    assert!(cpu.plan(op.as_view()).is_err());
}

mod joint_native;
mod native;

fn extended_spec(scoring: GroupScoring) -> TopKGroupSelectorSpec {
    spec(scoring, 2, true, 1.7)
        .with_bias(ParameterSpec::trainable("bias").unwrap())
        .unwrap()
        .with_input_transform(
            eredu_nn::SelectorInputTransformSpec::new(
                1e-5,
                ParameterSpec::trainable("input_scale").unwrap(),
                true,
            )
            .unwrap(),
        )
        .with_coefficient_scale(ParameterSpec::trainable("coefficient_scale").unwrap())
        .with_arithmetic(RoutingArithmetic {
            projection: RoutingPrecision::Float32,
            scores: RoutingPrecision::Preserve,
            coefficients: RoutingPrecision::Float32,
        })
}

#[test]
fn cpu_router_optional_transforms_and_joint_views_keep_exact_source_storage() {
    let _sources = crate::tests::support::test_utils::initialize_original_sources();
    let (ordinary, cpu) = mechanisms();
    for scoring in [
        GroupScoring::Softmax,
        GroupScoring::SelectedSoftmax,
        GroupScoring::Sigmoid,
        GroupScoring::SqrtSoftplus,
    ] {
        for supplied in [false, true] {
            let (context, report) = trace_shape(
                extended_spec(scoring),
                &[2, 8],
                supplied,
                WorkspaceDtype::Int32,
                cpu,
            );
            assert!(
                report.unpriced_operations.is_empty(),
                "{scoring:?}, supplied={supplied}: {:?}",
                report.unpriced_operations
            );
            let op = report
                .operations
                .iter()
                .find(|op| matches!(op.kind, WorkspaceOperationKind::GroupSelection { .. }))
                .unwrap();
            let plan = cpu.plan(op.as_view()).unwrap().unwrap();
            assert_eq!(
                plan.rank, 3,
                "learned scale Gather has rank-three output before Squeeze"
            );
            assert!(plan.seeds >= 4);
            let recipe = SpeculativeNumericalRecipe::inspect_cpu_outputs(
                &report, 3, ordinary, cpu, &context,
            )
            .unwrap();
            assert!(recipe.graph_capacity > 0 && recipe.storage.maximum_births() > 0);
        }
    }
    for (rows, k, shared) in [(1, 1, 1), (2, 2, 2), (5, 3, 1)] {
        let context = WorkspaceContext::new(cpu);
        let value = |shape: &[i32]| {
            WorkspaceTensor::existing(
                context
                    .layout(shape, WorkspaceDtype::Float32)
                    .unwrap()
                    .with_representation(Some(WorkspaceRepresentation::new(
                        WorkspaceFloatingType::Float32,
                        true,
                    ))),
                &context,
            )
            .unwrap()
        };
        let inputs = [
            value(&[rows, 8]),
            value(&[3 + shared, 8]),
            value(&[3]),
            value(&[1]),
        ];
        context.begin_state_span(inputs.iter()).unwrap();
        let spec = eredu_nn::JointGroupSelectionSpec::new(3, shared, k, 1.7).unwrap();
        let output = WorkspaceBackend::joint_group_selection(
            eredu_nn::JointGroupSelectionInput::new(
                &inputs[0], &inputs[1], &inputs[2], &inputs[3], spec,
            )
            .unwrap(),
            &context,
        )
        .unwrap();
        let outputs = [
            output.primary_indices().clone(),
            output.primary_coefficients().clone(),
            output.always_on_coefficients().clone(),
        ];
        let report = context.finish_report(&outputs).unwrap();
        assert!(report.unpriced_operations.is_empty());
        let expected = ordinary
            .allocation()
            .fixed_buffer_capacity(rows as u64 * 3 * 4)
            .unwrap()
            + ordinary
                .allocation()
                .fixed_buffer_capacity(rows as u64 * (k + shared) as u64 * 4)
                .unwrap();
        assert_eq!(report.tensor_buffers.retained_bytes, Some(expected));
        for output in &outputs[1..] {
            let repr = output.layout().representation().unwrap();
            assert_eq!(repr.row_contiguous(), rows == 1);
            assert!(repr.last_axis_contiguous());
            if rows == 1 {
                // The outer stride is unobservable for one row. Canonical
                // row-major facts deliberately omit redundant stride slots.
                assert_eq!(
                    repr,
                    WorkspaceRepresentation::new(WorkspaceFloatingType::Float32, true)
                );
                assert_eq!(repr.element_stride_at(2, 0), None);
            } else {
                assert_eq!(repr.element_stride_at(2, 0), Some((k + shared) as u64));
                assert_eq!(repr.element_stride_at(2, 1), Some(1));
            }
            let strides = super::super::views::physical_strides(output.layout().as_view(), repr)
                .expect("positive coefficient-view stride evidence");
            for row in 0..rows {
                for column in 0..output.shape()[1] {
                    assert_eq!(
                        i64::from(row) * strides[0] + i64::from(column) * strides[1],
                        i64::from(row * (k + shared) + column),
                        "coefficient view must address its shared backing exactly"
                    );
                }
            }
        }
        let recipe =
            SpeculativeNumericalRecipe::inspect_cpu_outputs(&report, 3, ordinary, cpu, &context)
                .unwrap();
        assert_eq!(recipe.kernels, 0);
    }
}
