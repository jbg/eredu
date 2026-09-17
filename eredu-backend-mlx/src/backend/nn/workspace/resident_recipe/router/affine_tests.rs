use super::*;
use eredu_nn::{GroupSelectionOperator, GroupedNeuralBackend, LinearFormatSpec,
    ParameterSpec, RoutingArithmetic, TopKGroupSelectionSpec, TopKGroupSelectorSpec};

fn source(group_size: i32, supplied: bool) -> WorkspaceOperation {
    let context = WorkspaceContext::new(MlxMetalWorkspaceMechanisms::current_host().unwrap());
    let parameter = |name| ParameterSpec::trainable(name).unwrap();
    let spec = TopKGroupSelectorSpec::new(64, parameter("weight"),
        LinearFormatSpec::affine(eredu_checkpoint::LinearFormat::Affine(
            eredu_checkpoint::AffineQuantization::new(group_size, 4).unwrap()),
            parameter("scales"), parameter("biases")).unwrap(),
        TopKGroupSelectionSpec::new(4, 1, GroupScoring::Sigmoid, true).unwrap()
            .with_weight_policy(1e-6, 1.0).unwrap()).unwrap()
        .with_bias(parameter("bias")).unwrap()
        .with_correction_bias(parameter("correction")).unwrap()
        .with_arithmetic(RoutingArithmetic::uniform(RoutingPrecision::Input));
    let mut selector = WorkspaceBackend::top_k_group_selector(spec, &context).unwrap();
    let input = WorkspaceTensor::existing(WorkspaceLayout::new(&[1, 2, 64],
        WorkspaceDtype::Float32).unwrap(), &context).unwrap();
    let ids = WorkspaceTensor::existing(WorkspaceLayout::new(&[2, 1],
        WorkspaceDtype::Uint32).unwrap(), &context).unwrap();
    context.begin_span();
    let result = if supplied { selector.select_indices(&input, &ids, &context).unwrap() }
        else { selector.select(&input, &context).unwrap() };
    let mut operation = context.report(&[result.group_indices().clone(),
        result.selected_scores().clone(), result.coefficients().clone()]).unwrap()
        .operations.into_iter().find(|op|
            matches!(op.kind, WorkspaceOperationKind::GroupSelection { .. })).unwrap();
    for input in &mut operation.inputs {
        if input.dtype() == WorkspaceDtype::Float32 {
            *input = input.clone().with_representation(Some(WorkspaceRepresentation::new(
                WorkspaceFloatingType::Float32, false)));
        }
    }
    operation
}

#[test]
fn affine_selector_reuses_projection_source_and_preserves_replacement_branch() {
    let _sources = crate::tests::support::test_utils::initialize_original_sources();
    crate::backend::managed_memory::router::prepare_before_native_construction();
    let mechanism = MlxMetalWorkspaceMechanisms::current_host().unwrap();
    for group_size in [16, 64] {
        for supplied in [false, true] {
            let operation = source(group_size, supplied);
            let packed = lowering(operation.as_view()).unwrap();
            assert_eq!(packed.helper_controls, 0, "packed QMM does not probe BF16 matmul");
            assert_eq!(packed.bf16_projection_calls, 0);
            assert_eq!(packed.router_cpu_partitions, usize::from(!supplied));
            assert!(mechanism.operation_bound(&operation).unwrap().is_some());
            let weight = 1 + usize::from(supplied);
            let mut missing = operation.clone();
            missing.inputs[weight + 1] = missing.inputs[weight + 1].clone().with_representation(None);
            assert!(lowering(missing.as_view()).is_none(), "packed scales need actual precision");
            let mut malformed = operation.clone();
            malformed.inputs[weight + 2] = WorkspaceLayout::new(&[4, 3], WorkspaceDtype::Float32).unwrap()
                .with_representation(Some(WorkspaceRepresentation::new(WorkspaceFloatingType::Float32, true)));
            assert!(lowering(malformed.as_view()).is_none(), "affine bias is a physical companion");
            let mut replacement = missing;
            replacement.inputs[weight] = WorkspaceLayout::new(&[4, 64], WorkspaceDtype::Float32).unwrap()
                .with_representation(Some(WorkspaceRepresentation::new(WorkspaceFloatingType::Float32, false)));
            replacement.inputs[weight + 2] = replacement.inputs[weight + 2].clone().with_representation(None);
            let dense = lowering(replacement.as_view()).unwrap();
            assert_eq!(dense.helper_controls,
                crate::backend::nn::matrix::row_projection_probe_control_bytes().unwrap());
            assert!(mechanism.operation_bound(&replacement).unwrap().is_some(),
                "floating replacement must use matrix storage, ignoring unused packed companions");
            super::super::super::routing::with_selector_projection(replacement.as_view(), |projection| {
                assert_eq!(projection.inputs.len(), 3, "input, actual weight and ordinary bias");
                assert_eq!(projection.inputs.get(0).unwrap().shape(), [2, 64]);
                assert_eq!(projection.inputs.get(0).unwrap().representation().unwrap().dtype(),
                    WorkspaceFloatingType::Float32);
                assert!(matches!(projection.kind, WorkspaceOperationKindView::Projection(format)
                    if format.encoding() == eredu_checkpoint::LinearFormat::Dense));
            }).unwrap().unwrap();
        }
    }
}
