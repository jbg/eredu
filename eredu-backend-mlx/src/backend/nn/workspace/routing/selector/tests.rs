use super::*;
use eredu_checkpoint::AffineQuantization;
use eredu_nn::{
    routing_intervention::{
        GroupScoreStage, GroupSelectionAction, GroupSelectionControl, IntervenedGroupSelection,
    },
    GroupSelectionOperator, GroupedNeuralBackend, LinearFormatSpec, ParameterSpec,
    RoutingArithmetic, SelectorInputTransformSpec, Tensor,
};

fn mechanisms() -> MlxMetalWorkspaceMechanisms {
    MlxMetalWorkspaceMechanisms {
        allocation: MetalAllocationFacts { page_size: 16384 },
        sdpa_blocks: None,
    }
}
fn parameter(name: &str) -> ParameterSpec {
    ParameterSpec::trainable(name).unwrap()
}
fn spec(
    d: i32,
    groups: i32,
    k: i32,
    partitions: i32,
    scoring: GroupScoring,
    rich: bool,
    encoding: LinearFormat,
) -> TopKGroupSelectorSpec {
    let format = match encoding {
        LinearFormat::Affine(_) => {
            LinearFormatSpec::affine(encoding, parameter("scales"), parameter("biases"))
        }
        LinearFormat::MxFp4 => LinearFormatSpec::scaled(encoding, parameter("scales")),
        _ => LinearFormatSpec::unscaled(encoding),
    }
    .unwrap();
    let selection = TopKGroupSelectionSpec::new(groups, k, scoring, rich)
        .unwrap()
        .with_groups(partitions, if partitions >= 4 { 2 } else { 1 })
        .unwrap()
        .with_weight_policy(if rich { 0.1 } else { 0. }, if rich { 1.5 } else { 1. })
        .unwrap();
    let mut spec = TopKGroupSelectorSpec::new(d, parameter("weight"), format, selection)
        .unwrap()
        .with_arithmetic(if rich {
            RoutingArithmetic {
                projection: RoutingPrecision::Float32,
                scores: RoutingPrecision::Input,
                coefficients: RoutingPrecision::Input,
            }
        } else {
            RoutingArithmetic::uniform(RoutingPrecision::Preserve)
        });
    if rich {
        spec = spec
            .with_bias(parameter("bias"))
            .unwrap()
            .with_correction_bias(parameter("correction"))
            .unwrap()
            .with_input_transform(
                SelectorInputTransformSpec::new(1e-4, parameter("input_scale"), true).unwrap(),
            )
            .with_coefficient_scale(parameter("coefficient_scale"));
    }
    spec
}
fn controls(spec: &TopKGroupSelectorSpec, capture: bool) -> Vec<GroupSelectionControl> {
    [
        GroupSelectionAction::Exclude(vec![0]),
        GroupSelectionAction::ZeroContribution(
            (0..spec.selection().group_count() as u32).collect(),
        ),
        GroupSelectionAction::Bias {
            stage: GroupScoreStage::RawLogits,
            ids: vec![0, 2],
            values: vec![0.3, 0.2],
        },
        GroupSelectionAction::Bias {
            stage: GroupScoreStage::TransformedScores,
            ids: vec![0, 2],
            values: vec![0.3, 0.2],
        },
        GroupSelectionAction::Bias {
            stage: GroupScoreStage::RankingScores,
            ids: vec![0, 2],
            values: vec![0.3, 0.2],
        },
        GroupSelectionAction::Force(vec![0, 1, 2, 3]),
    ]
    .into_iter()
    .map(|action| GroupSelectionControl {
        expected: spec.selection(),
        learned_coefficient_scale: spec.coefficient_scale().is_some(),
        first_row: 1,
        end_row: 4,
        row_stride: 2,
        action,
        capture_original: capture,
    })
    .collect()
}
fn run<T: Tensor, S: GroupSelectionOperator<T>>(
    selector: &mut S,
    input: &T,
    supplied: Option<&T>,
    control: Option<&GroupSelectionControl>,
    c: &T::Context,
) -> IntervenedGroupSelection<T> {
    if let Some(control) = control {
        selector.select_intervened(input, control, c).unwrap()
    } else {
        IntervenedGroupSelection {
            original: None,
            effective: if let Some(ids) = supplied {
                selector.select_indices(input, ids, c).unwrap()
            } else {
                selector.select(input, c).unwrap()
            },
        }
    }
}
fn outputs<T: Clone>(value: &IntervenedGroupSelection<T>) -> Vec<T> {
    value
        .original
        .iter()
        .chain(std::iter::once(&value.effective))
        .flat_map(|v| {
            [
                v.group_indices().clone(),
                v.selected_scores().clone(),
                v.coefficients().clone(),
            ]
        })
        .collect()
}
fn quote(
    spec: &TopKGroupSelectorSpec,
    shape: &[i32],
    supplied: bool,
    control: Option<&GroupSelectionControl>,
) -> (WorkspaceOperation, WorkspaceTraceReport) {
    let c = WorkspaceContext::new(mechanisms());
    let mut selector = WorkspaceBackend::top_k_group_selector(spec.clone(), &c).unwrap();
    let input = WorkspaceTensor::existing(
        WorkspaceLayout::new(shape, WorkspaceDtype::Float32).unwrap(),
        &c,
    )
    .unwrap();
    let rows = shape.iter().product::<i32>() / spec.input_dimensions();
    let ids = supplied.then(|| {
        WorkspaceTensor::existing(
            WorkspaceLayout::new(&[rows, spec.selection().top_k()], WorkspaceDtype::Uint32)
                .unwrap(),
            &c,
        )
        .unwrap()
    });
    c.begin_span();
    let out = run(&mut selector, &input, ids.as_ref(), control, &c);
    let report = c.report(&outputs(&out)).unwrap();
    (report.operations[0].clone(), report)
}

#[test]
fn topk_routing_workspace_prices_all_encodings_policies_and_control_stages() {
    let mut encodings = vec![LinearFormat::Dense, LinearFormat::MxFp4];
    for bits in [2, 3, 4, 5, 6, 8] {
        encodings.push(LinearFormat::Affine(
            AffineQuantization::new(32, bits).unwrap(),
        ));
    }
    for ggml_type in [
        eredu_gguf::GgmlType::Q4K,
        eredu_gguf::GgmlType::Q5K,
        eredu_gguf::GgmlType::Q6K,
        eredu_gguf::GgmlType::Q5_1,
        eredu_gguf::GgmlType::Q8_0,
        eredu_gguf::GgmlType::IQ2XXS,
        eredu_gguf::GgmlType::IQ2XS,
        eredu_gguf::GgmlType::IQ3XXS,
        eredu_gguf::GgmlType::IQ1S,
        eredu_gguf::GgmlType::IQ4NL,
        eredu_gguf::GgmlType::IQ3S,
        eredu_gguf::GgmlType::IQ2S,
        eredu_gguf::GgmlType::IQ4XS,
        eredu_gguf::GgmlType::IQ1M,
    ] {
        for endian in [eredu_gguf::Endian::Little, eredu_gguf::Endian::Big] {
            encodings.push(LinearFormat::GgufIQuant { ggml_type, endian });
        }
    }
    for encoding in encodings {
        for scoring in [
            GroupScoring::Softmax,
            GroupScoring::SelectedSoftmax,
            GroupScoring::Sigmoid,
            GroupScoring::SqrtSoftplus,
        ] {
            let spec = spec(256, 8, 2, 2, scoring, true, encoding);
            for shape in [&[0, 256][..], &[2, 2, 256][..], &[256][..]] {
                for supplied in [false, true] {
                    let (_, report) = quote(&spec, shape, supplied, None);
                    assert!(
                        report.total_bytes.is_some(),
                        "{encoding:?}: {:?}",
                        report.unpriced_operations
                    );
                    let expected_host = if supplied || shape[0] == 0 { 0 } else { 8 * 4 };
                    assert_eq!(report.host_workspace_bytes, Some(expected_host));
                }
            }
            for capture in [false, true] {
                for control in controls(&spec, capture) {
                    let (_, report) = quote(&spec, &[2, 2, 256], false, Some(&control));
                    assert!(report.total_bytes.is_some());
                    assert_eq!(report.host_workspace_bytes, Some(32));
                }
            }
        }
    }
}

#[test]
fn topk_routing_workspace_preserves_shared_coefficients_and_supplied_index_roots() {
    let spec = spec(
        8,
        8,
        2,
        1,
        GroupScoring::SelectedSoftmax,
        false,
        LinearFormat::Dense,
    );
    let (op, report) = quote(&spec, &[2, 8], false, None);
    let b = mechanisms().operation_bound(&op).unwrap().unwrap();
    assert_eq!(b.outputs[2], WorkspaceOutputStorage::AliasOutput(1));
    assert_eq!(
        report.retained_bytes,
        Some(
            capacity(mechanisms().allocation, 16).unwrap()
                + capacity(mechanisms().allocation, 4).unwrap()
        )
    );
    let c = WorkspaceContext::new(mechanisms());
    let mut selector = WorkspaceBackend::top_k_group_selector(spec, &c).unwrap();
    let x = WorkspaceTensor::unloaded_f32(&[2, 8], &c).unwrap();
    let storage = WorkspaceExistingStorage::new(Some(4096), &c);
    let ids = WorkspaceTensor::existing_with_storage(
        WorkspaceLayout::new(&[2, 2], WorkspaceDtype::Uint32).unwrap(),
        &storage,
        &c,
    )
    .unwrap();
    c.begin_state_span([&ids]).unwrap();
    let out = selector.select_indices(&x, &ids, &c).unwrap();
    let report = c.report(&[out.group_indices().clone()]).unwrap();
    assert_eq!(
        report.state.as_ref().unwrap().retained_bytes,
        Some(4096 + capacity(mechanisms().allocation, 4).unwrap())
    );
    assert_eq!(
        report.retained_bytes,
        Some(capacity(mechanisms().allocation, 4).unwrap())
    );
}

#[test]
fn topk_routing_workspace_rejects_malformed_geometry_and_unselected_formats() {
    let spec = spec(8, 8, 2, 1, GroupScoring::Softmax, true, LinearFormat::Dense);
    let (mut op, _) = quote(&spec, &[2, 8], false, None);
    op.inputs.pop();
    assert!(mechanisms().operation_bound(&op).is_err());
    let (mut op, _) = quote(&spec, &[2, 8], false, None);
    op.inputs[0] = WorkspaceLayout::new(&[2, 8], WorkspaceDtype::Int32).unwrap();
    assert!(mechanisms().operation_bound(&op).unwrap().is_none());
    let (mut op, _) = quote(&spec, &[2, 8], false, None);
    op.outputs[0] = WorkspaceLayout::new(&[2, 3], WorkspaceDtype::Uint32).unwrap();
    assert!(mechanisms().operation_bound(&op).is_err());
    let (mut op, _) = quote(&spec, &[2, 8], false, None);
    op.inputs[0] = WorkspaceLayout::new(&[i32::MAX, 2, 8], WorkspaceDtype::Float32).unwrap();
    assert!(mechanisms().operation_bound(&op).is_err());
    let block = TopKGroupSelectorSpec::new(
        128,
        parameter("weight"),
        LinearFormatSpec::scaled(
            LinearFormat::E4M3BlockFp8(
                eredu_checkpoint::BlockFp8Format::new(
                    128,
                    128,
                    eredu_checkpoint::BlockFp8ScaleEncoding::FloatingPoint,
                )
                .unwrap(),
            ),
            parameter("scales"),
        )
        .unwrap(),
        spec.selection(),
    )
    .unwrap();
    let (op, report) = quote(&block, &[2, 128], false, None);
    assert!(report.total_bytes.is_none());
    assert!(mechanisms().operation_bound(&op).unwrap().is_none());
    assert!(mechanisms().host_workspace_bound(&op).unwrap().is_none());
}

#[cfg(all(target_vendor = "apple", feature = "metal", not(feature = "cuda")))]
mod native;

#[test]
fn shared_routing_failure_preserves_one_wrapper_and_ordinary_source_behavior() {
    let spec = spec(
        4,
        4,
        1,
        1,
        GroupScoring::SelectedSoftmax,
        false,
        LinearFormat::Dense,
    );
    let control = GroupSelectionControl {
        expected: spec.selection(),
        learned_coefficient_scale: false,
        first_row: 0,
        end_row: 1,
        row_stride: 1,
        action: GroupSelectionAction::Force(vec![0]),
        capture_original: false,
    };
    // Exercise the real shared driver's rows primitive and the same closed
    // Counter used by inspection. Arithmetic fails before projection; its
    // unused layout is not an admission or source-construction witness.
    let make = || Counter {
        a: MetalAllocationFacts { page_size: 4096 },
        spec: &spec,
        rows: u64::MAX,
        projection: WorkspaceOperationFacts {
            layout: WorkspaceEffectLayout {
                outputs: 0,
                aliases: 0,
                assumption_bytes: 0,
            },
            scratch_bytes: 0,
        },
        projection_output: 0,
        tensor: Cell::new(0),
        host: Cell::new(0),
        next: Cell::new(0),
    };
    let mut fixed_counter = make();
    let input = Value {
        id: 0,
        elements: 4,
        storage: Output::AliasInput(0),
    };
    let fixed = match execute_routing_intervention_fixed(&mut fixed_counter, &input, &control) {
        Err(error) => intervention_error(error),
        Ok(_) => panic!("the native row-buffer arithmetic must fail"),
    };
    let mut ordinary_counter = make();
    let original = match eredu_nn::routing_intervention::execute_routing_intervention(
        &mut ordinary_counter,
        &input,
        &control,
    ) {
        Err(error) => Error::backend(error),
        Ok(_) => panic!("the same native arithmetic must fail"),
    };
    assert_eq!(fixed_counter.next.get(), 0);
    assert!(matches!(
        fixed.cause(),
        MlxWorkspaceFactCause::Descriptor(_)
    ));
    assert_eq!(
        fixed
            .to_string()
            .matches("native routing operation failed: ")
            .count(),
        1
    );
    assert!(std::error::Error::source(&fixed).is_some());
    let mapped = fixed.ordinary();
    assert_eq!(mapped.to_string(), original.to_string());
    assert!(std::error::Error::source(&mapped).is_none());
    assert!(std::error::Error::source(&original).is_none());
}
