use super::*;
mod scheduled;
use crate::{
    routing_intervention::{GroupSelectionAction, GroupSelectionControl},
    GatedProductGroupLayout, GatedProductGroupParameters, GatedProductPolicy, GroupReduction,
    GroupScoring, GroupSelection, GroupSelectionOperator, GroupedGatedProductOperator,
    GroupedGatedProductSpec, GroupedLinearActivation, GroupedLinearOperator, GroupedLinearSpec,
    GroupedNeuralBackend, GroupedProjectionSpec, GroupedRelu2Operator, GroupedRelu2Spec,
    GroupedUnitBatch, GroupedUnitObserver, JointGroupSelectionInput, JointGroupSelectionSpec,
    SelectorInputTransformSpec, TensorParallelGroupedGatedProductOperator, TopKGroupSelectionSpec,
    TopKGroupSelectorSpec,
};

fn projection(name: &str, packed: bool, bias: bool) -> GroupedProjectionSpec {
    GroupedProjectionSpec::new(
        ParameterSpec::trainable(format!("{name}.weight")).unwrap(),
        bias.then(|| ParameterSpec::trainable(format!("{name}.bias")).unwrap()),
        if packed {
            LinearFormatSpec::affine(
                LinearFormat::Affine(eredu_checkpoint::AffineQuantization::new(32, 3).unwrap()),
                ParameterSpec::trainable(format!("{name}.scales")).unwrap(),
                ParameterSpec::trainable(format!("{name}.biases")).unwrap(),
            )
            .unwrap()
        } else {
            LinearFormatSpec::unscaled(LinearFormat::Dense).unwrap()
        },
    )
    .unwrap()
}
fn selector(packed: bool, scoring: GroupScoring) -> TopKGroupSelectorSpec {
    let p = projection("router", packed, true);
    TopKGroupSelectorSpec::new(
        64,
        p.weight().clone(),
        p.format().clone(),
        TopKGroupSelectionSpec::new(8, 2, scoring, true)
            .unwrap()
            .with_groups(2, 1)
            .unwrap()
            .with_weight_policy(0.1, 1.5)
            .unwrap(),
    )
    .unwrap()
    .with_bias(p.bias().unwrap().clone())
    .unwrap()
    .with_correction_bias(ParameterSpec::trainable("correction").unwrap())
    .unwrap()
    .with_input_transform(
        SelectorInputTransformSpec::new(
            1e-5,
            ParameterSpec::trainable("input_scale").unwrap(),
            true,
        )
        .unwrap(),
    )
    .with_coefficient_scale(ParameterSpec::trainable("coefficient_scale").unwrap())
}
fn routes(context: &WorkspaceContext, count: i32) -> GroupSelection<WorkspaceTensor> {
    let ids = WorkspaceTensor::existing(
        WorkspaceLayout::new(&[count, 2], WorkspaceDtype::Uint32).unwrap(),
        context,
    )
    .unwrap();
    let scores = existing_f32(&[count, 2], context).unwrap();
    GroupSelection::new(ids, scores.clone(), scores)
}
fn bank(packed: bool, independent: bool, bias: bool) -> GroupedGatedProductSpec {
    GroupedGatedProductSpec::new(
        3,
        64,
        32,
        64,
        GatedProductPolicy::bounded_silu(2.5).unwrap(),
        if independent {
            GatedProductGroupLayout::Independent(
                (0..3)
                    .map(|i| {
                        GatedProductGroupParameters::new(
                            projection(&format!("g{i}.gate"), packed, bias),
                            projection(&format!("g{i}.up"), packed, bias),
                            projection(&format!("g{i}.down"), packed, bias),
                        )
                    })
                    .collect(),
            )
        } else {
            GatedProductGroupLayout::Packed {
                gate_up: projection("gate_up", packed, bias),
                down: projection("down", packed, bias),
            }
        },
    )
    .unwrap()
    .with_reduction(GroupReduction::SequentialGroupOrder)
}

#[test]
fn selector_retains_every_parameter_policy_and_intervention_output() {
    for packed in [false, true] {
        for scoring in [
            GroupScoring::Softmax,
            GroupScoring::SelectedSoftmax,
            GroupScoring::Sigmoid,
            GroupScoring::SqrtSoftplus,
        ] {
            let context = context();
            let spec = selector(packed, scoring);
            let mut router =
                WorkspaceBackend::top_k_group_selector(spec.clone(), &context).unwrap();
            let input = existing_f32(&[2, 3, 64], &context).unwrap();
            context.begin_span();
            let ordinary = router.select(&input, &context).unwrap();
            assert_eq!(ordinary.group_indices().shape(), [6, 2]);
            let supplied = WorkspaceTensor::existing(
                WorkspaceLayout::new(&[2, 3, 2], WorkspaceDtype::Int32).unwrap(),
                &context,
            )
            .unwrap();
            let selected = router.select_indices(&input, &supplied, &context).unwrap();
            assert_eq!(
                selected.group_indices().layout().dtype(),
                WorkspaceDtype::Int32
            );
            let control = GroupSelectionControl {
                expected: spec.selection(),
                learned_coefficient_scale: true,
                first_row: 1,
                end_row: 6,
                row_stride: 2,
                action: GroupSelectionAction::Force(vec![0, 1, 4, 5, 2, 3]),
                capture_original: true,
            };
            let controlled = router
                .select_intervened(&input, &control, &context)
                .unwrap();
            assert_eq!(controlled.original.unwrap().coefficients().shape(), [6, 2]);
            assert_eq!(controlled.effective.coefficients().shape(), [6, 2]);
            let report = context.report(&[]).unwrap();
            assert_eq!(report.operations.len(), 3);
            for (i, op) in report.operations.iter().enumerate() {
                let WorkspaceOperationKind::GroupSelection {
                    spec: actual,
                    supplied_indices,
                    control,
                } = &op.kind
                else {
                    panic!("selector descriptor")
                };
                assert_eq!(actual.selection(), spec.selection());
                assert_eq!(actual.arithmetic(), spec.arithmetic());
                assert_eq!(*supplied_indices, i == 1);
                assert_eq!(control.is_some(), i == 2);
                assert_eq!(op.outputs.len(), if i == 2 { 6 } else { 3 });
                let start = if i == 1 { 2 } else { 1 };
                let mut expected = if packed {
                    vec![vec![8, 6], vec![8, 2], vec![8, 2]]
                } else {
                    vec![vec![8, 64]]
                };
                expected.extend([vec![8], vec![8], vec![64], vec![8]]);
                assert_eq!(
                    op.inputs[start..]
                        .iter()
                        .map(|l| l.shape().to_vec())
                        .collect::<Vec<_>>(),
                    expected
                );
            }
            let mut invalid = control;
            invalid.learned_coefficient_scale = false;
            assert!(router
                .select_intervened(&input, &invalid, &context)
                .is_err());
            assert!(router.select_indices(&input, &input, &context).is_err());
            assert_eq!(context.report(&[]).unwrap().operations.len(), 3);
        }
    }
}

#[test]
fn grouped_banks_preserve_packed_and_independent_geometry_and_post_reduction_bias() {
    for packed in [false, true] {
        for independent in [false, true] {
            for bias in [false, true] {
                let context = context();
                let spec = bank(packed, independent, bias);
                let mut bank =
                    WorkspaceBackend::grouped_gated_product(spec.clone(), &context).unwrap();
                let input = existing_f32(&[2, 5, 64], &context).unwrap();
                let selection = routes(&context, 10);
                for partitions in [None, Some(1), Some(3)] {
                    let (output, post) = if let Some(n) = partitions {
                        bank.forward_grouped_tensor_parallel(&input, &selection, n, &context)
                            .unwrap()
                            .into_parts()
                    } else {
                        (
                            bank.forward_grouped(&input, &selection, &context).unwrap(),
                            None,
                        )
                    };
                    assert_eq!(output.shape(), input.shape());
                    assert_eq!(post.is_some(), partitions.is_some() && bias);
                    let report = context.report(&[]).unwrap();
                    let op = report.operations.last().unwrap();
                    let WorkspaceOperationKind::Grouped {
                        bank,
                        phase,
                        partitions: actual,
                    } = &op.kind
                    else {
                        panic!("grouped descriptor")
                    };
                    assert_eq!(*phase, WorkspaceGroupedPhase::Whole);
                    assert_eq!(*actual, partitions);
                    assert!(
                        matches!(bank.as_ref(), WorkspaceGroupedBank::GatedProduct(s) if s == &spec)
                    );
                    let shapes = op.inputs[4..]
                        .iter()
                        .map(|p| p.shape().to_vec())
                        .collect::<Vec<_>>();
                    let mut expected = Vec::new();
                    let dimensions = if independent {
                        vec![(32, 64), (32, 64), (64, 32)].repeat(3)
                    } else {
                        vec![(64, 64), (64, 32)]
                    };
                    for (rows, columns) in dimensions {
                        let mut matrix = if packed {
                            vec![
                                vec![rows, columns / 32 * 3],
                                vec![rows, columns / 32],
                                vec![rows, columns / 32],
                            ]
                        } else {
                            vec![vec![rows, columns]]
                        };
                        if bias {
                            matrix.push(vec![rows]);
                        }
                        for mut shape in matrix {
                            if !independent {
                                shape.insert(0, 3);
                            }
                            expected.push(shape);
                        }
                    }
                    assert_eq!(shapes, expected);
                }
            }
        }
    }
}

#[test]
fn selected_linear_output_ownership_and_relu2_stay_distinct() {
    for packed in [false, true] {
        for groups in [0, 3] {
            let context = context();
            let spec = GroupedLinearSpec::new(
                3,
                64,
                17,
                GroupedLinearActivation::Silu,
                projection("selected", packed, true),
            )
            .unwrap()
            .partition_output(9..17)
            .unwrap()
            .with_group_count(groups)
            .unwrap();
            let mut bank = WorkspaceBackend::grouped_linear_bank(spec.clone(), &context).unwrap();
            let input = existing_f32(&[2, 3, 64], &context).unwrap();
            let output = bank
                .forward_grouped(&input, &routes(&context, 6), &context)
                .unwrap();
            assert_eq!(output.shape(), [2, 3, 8]);
            let report = context.report(&[]).unwrap();
            let op = report.operations.last().unwrap();
            assert_eq!(
                op.inputs[4].shape(),
                if packed {
                    vec![groups, 8, 6]
                } else {
                    vec![groups, 8, 64]
                }
            );
            assert!(
                matches!(&op.kind, WorkspaceOperationKind::Grouped {bank,partitions:None,..} if matches!(bank.as_ref(), WorkspaceGroupedBank::Linear(s) if s==&spec))
            );
        }
        let context = context();
        let spec = GroupedRelu2Spec::new(
            3,
            64,
            32,
            projection("relu.up", packed, false),
            projection("relu.down", packed, false),
        )
        .unwrap();
        let mut bank = WorkspaceBackend::grouped_relu2(spec.clone(), &context).unwrap();
        context.begin_span();
        let input = existing_f32(&[2, 3, 64], &context).unwrap();
        assert_eq!(
            bank.forward_grouped(&input, &routes(&context, 6), &context)
                .unwrap()
                .shape(),
            input.shape()
        );
        assert!(
            matches!(&context.report(&[]).unwrap().operations[0].kind, WorkspaceOperationKind::Grouped {bank,..} if matches!(bank.as_ref(), WorkspaceGroupedBank::Relu2(s) if s==&spec))
        );
    }
}

struct Observer<'a> {
    context: &'a WorkspaceContext,
    events: Vec<&'static str>,
    invalid: u8,
}
impl GroupedUnitObserver<WorkspaceTensor> for Observer<'_> {
    fn observe(&mut self, batch: &GroupedUnitBatch<'_, WorkspaceTensor>) -> Result<(), Error> {
        self.events.push("original");
        assert_eq!(batch.values.shape(), [12, 32]);
        for ids in [
            batch.group_indices,
            batch.selection_indices,
            batch.token_indices,
        ] {
            assert_eq!(ids.shape(), [12]);
        }
        assert_eq!(batch.coefficients.shape(), [6, 2]);
        assert_eq!(
            (
                batch.total_token_count,
                batch.token_offset,
                batch.group_count
            ),
            (6, 0, 3)
        );
        Ok(())
    }
    fn intervene(
        &mut self,
        batch: &GroupedUnitBatch<'_, WorkspaceTensor>,
    ) -> Result<Option<WorkspaceTensor>, Error> {
        self.events.push("intervene");
        match self.invalid {
            1 => existing_f32(&[12, 31], self.context).map(Some),
            2 => WorkspaceTensor::existing(
                WorkspaceLayout::new(&[12, 32], WorkspaceDtype::Int32)?,
                self.context,
            )
            .map(Some),
            3 => existing_f32(&[12, 32], &context()).map(Some),
            4 => Err(Error::backend("admitted observer failed")),
            _ => batch.values.square(self.context).map(Some),
        }
    }
    fn observe_effective(
        &mut self,
        batch: &GroupedUnitBatch<'_, WorkspaceTensor>,
    ) -> Result<(), Error> {
        self.events.push("effective");
        assert_eq!(batch.values.shape(), [12, 32]);
        Ok(())
    }
}
#[test]
fn unit_observation_and_replacement_stop_before_down_projection_on_failure() {
    for invalid in 0..=4 {
        let context = context();
        let mut bank =
            WorkspaceBackend::grouped_gated_product(bank(true, false, true), &context).unwrap();
        let input = existing_f32(&[2, 3, 64], &context).unwrap();
        let mut observer = Observer {
            context: &context,
            events: vec![],
            invalid,
        };
        let result = bank.forward_grouped_tensor_parallel_with_unit_observer(
            &input,
            &routes(&context, 6),
            2,
            &context,
            Some(&mut observer),
        );
        assert_eq!(result.is_ok(), invalid == 0);
        assert_eq!(
            observer.events,
            if invalid == 0 {
                vec!["original", "intervene", "effective"]
            } else {
                vec!["original", "intervene"]
            }
        );
        let report = context.report(&[]).unwrap();
        let finish = report
            .operations
            .iter()
            .filter(|op| {
                matches!(
                    op.kind,
                    WorkspaceOperationKind::Grouped {
                        phase: WorkspaceGroupedPhase::Finish,
                        ..
                    }
                )
            })
            .count();
        assert_eq!(finish, usize::from(invalid == 0));
        assert!(report.total_bytes.unwrap() > 0); // failure never refunds preceding work
        if invalid == 0 {
            assert!(result.unwrap().post_reduce().is_some());
        }
    }
}

#[test]
fn cross_context_or_bad_selection_fails_before_grouped_work() {
    let context = context();
    let mut bank =
        WorkspaceBackend::grouped_gated_product(bank(false, false, false), &context).unwrap();
    let input = existing_f32(&[2, 3, 64], &context).unwrap();
    context.begin_span();
    let other = WorkspaceContext::new(AllocatingMechanism);
    assert!(bank
        .forward_grouped(&input, &routes(&other, 6), &context)
        .is_err());
    assert!(bank
        .forward_grouped(&input, &routes(&context, 5), &context)
        .is_err());
    assert!(bank
        .forward_grouped_tensor_parallel(&input, &routes(&context, 6), 0, &context)
        .is_err());
    assert!(context.report(&[]).unwrap().operations.is_empty());
}

#[test]
fn joint_selection_keeps_always_on_outputs_and_unknown_routing_is_not_priced() {
    #[derive(Debug)]
    struct MissingRouting;
    impl WorkspaceMechanisms for MissingRouting {
        fn host_workspace_bound(
            &self,
            _: &WorkspaceOperation,
        ) -> Result<Option<WorkspaceHostBound>, Error> {
            Ok(Some(WorkspaceHostBound {
                bytes: 0,
                assumptions: "test mechanism has no disjoint host workspace".into(),
            }))
        }

        fn operation_bound(
            &self,
            op: &WorkspaceOperation,
        ) -> Result<Option<WorkspaceOperationBound>, Error> {
            if matches!(
                op.kind,
                WorkspaceOperationKind::GroupSelection { .. }
                    | WorkspaceOperationKind::JointGroupSelection(_)
                    | WorkspaceOperationKind::Grouped { .. }
            ) {
                Ok(None)
            } else {
                AllocatingMechanism.operation_bound(op)
            }
        }
    }
    let context = WorkspaceContext::new(MissingRouting);
    let hidden = existing_f32(&[2, 3, 64], &context).unwrap();
    let weight = existing_f32(&[11, 64], &context).unwrap();
    let bias = existing_f32(&[8], &context).unwrap();
    let scale = existing_f32(&[1], &context).unwrap();
    let mut router =
        WorkspaceBackend::top_k_group_selector(selector(false, GroupScoring::Sigmoid), &context)
            .unwrap();
    let mut bank =
        WorkspaceBackend::grouped_gated_product(bank(false, false, false), &context).unwrap();
    context.begin_span();
    let selected = WorkspaceBackend::joint_group_selection(
        JointGroupSelectionInput::new(
            &hidden,
            &weight,
            &bias,
            &scale,
            JointGroupSelectionSpec::new(8, 3, 2, 1.5).unwrap(),
        )
        .unwrap(),
        &context,
    )
    .unwrap();
    assert_eq!(selected.primary_indices().shape(), [6, 2]);
    assert_eq!(selected.always_on_coefficients().shape(), [6, 3]);
    let selection = router.select(&hidden, &context).unwrap();
    let output = bank.forward_grouped(&hidden, &selection, &context).unwrap();
    let report = context.report(&[output]).unwrap();
    assert_eq!(report.unpriced_operations, [0, 1, 2]);
    assert_eq!(
        (
            report.total_bytes,
            report.retained_bytes,
            report.transient_bytes
        ),
        (None, None, None)
    );
}

#[test]
fn explicit_group_axis_traces_full_projection_then_diagonal_selection() {
    struct Input {
        calls: usize,
    }
    impl crate::TensorValueObserver<WorkspaceTensor> for Input {
        fn observe(&mut self, input: &WorkspaceTensor) -> Result<(), Error> {
            assert_eq!(input.shape(), [2, 3, 5, 64]);
            self.calls += 1;
            Ok(())
        }
        fn observe_generated(
            &mut self,
            _: &WorkspaceTensor,
            _: &crate::GeneratedTensorSource,
            _: &mut dyn FnMut() -> Result<WorkspaceTensor, Error>,
        ) -> Result<(), Error> {
            panic!("ordinary multiply input")
        }
    }
    let context = context();
    let mut linear = WorkspaceBackend::linear(
        crate::LinearSpec {
            input: 64,
            output: 12,
            weight: ParameterSpec::trainable("block.weight").unwrap(),
            bias: None,
            format: LinearFormatSpec::unscaled(LinearFormat::Dense).unwrap(),
        },
        &context,
    )
    .unwrap();
    let input = existing_f32(&[2, 3, 5, 64], &context).unwrap();
    context.begin_span();
    let mut observer = Input { calls: 0 };
    let output = WorkspaceBackend::grouped_linear_with_input_observer(
        &mut linear,
        &input,
        3,
        4,
        &context,
        Some(&mut observer),
    )
    .unwrap();
    assert_eq!(output.shape(), [2, 3, 5, 4]);
    assert_eq!(observer.calls, 1);
    let report = context.report(&[]).unwrap();
    assert_eq!(report.operations[0].outputs[0].shape(), [2, 3, 5, 12]);
    assert_eq!(
        report
            .operations
            .iter()
            .filter(|op| matches!(op.kind, WorkspaceOperationKind::Index { selected_axes: 1 }))
            .count(),
        3
    );
    assert!(matches!(
        report.operations.last().unwrap().kind,
        WorkspaceOperationKind::Concatenate
    ));
    context.begin_span();
    assert!(WorkspaceBackend::grouped_linear_with_input_observer(
        &mut linear,
        &input,
        3,
        5,
        &context,
        Some(&mut observer)
    )
    .is_err());
    assert_eq!(observer.calls, 1);
    assert!(context.report(&[]).unwrap().operations.is_empty());
}

#[test]
fn grouped_borrowed_sources_match_actual_workspace_parameter_slots() {
    fn check(value: &impl crate::Parameterized<WorkspaceTensor>) {
        struct Rows<'a> {
            values: Vec<&'a WorkspaceTensor>,
            metadata: Vec<crate::ParameterMetadata>,
        }
        impl<'a> crate::ParameterSourceVisitor<'a, WorkspaceTensor> for Rows<'a> {
            fn parameter(&mut self, m: crate::ParameterMetadataView<'a>, v: &'a WorkspaceTensor) {
                self.values.push(v);
                self.metadata.push(m.to_owned());
            }
            fn retained(&mut self, _: &'a WorkspaceTensor) {
                panic!("concrete grouped spec has no auxiliary tensor");
            }
        }
        let mut rows = Rows {
            values: Vec::new(),
            metadata: Vec::new(),
        };
        value.visit_parameter_sources(&mut rows).unwrap();
        let mut actual = Vec::new();
        assert!(value.visit_retained_values(&mut |v| actual.push(v as *const WorkspaceTensor)));
        assert_eq!(
            rows.values
                .iter()
                .map(|v| *v as *const WorkspaceTensor)
                .collect::<Vec<_>>(),
            actual
        );
        rows.metadata.sort_by(|a, b| a.id.cmp(&b.id));
        let mut ordinary = crate::validate_parameter_topology(value).unwrap();
        ordinary.sort_by(|a, b| a.id.cmp(&b.id));
        assert_eq!(rows.metadata, ordinary);
    }
    let context = context();
    for packed in [false, true] {
        let linear = WorkspaceBackend::grouped_linear_bank(
            GroupedLinearSpec::new(
                3,
                64,
                32,
                GroupedLinearActivation::Silu,
                projection("linear", packed, true),
            )
            .unwrap(),
            &context,
        )
        .unwrap();
        let gated =
            WorkspaceBackend::grouped_gated_product(bank(packed, true, true), &context).unwrap();
        let relu = WorkspaceBackend::grouped_relu2(
            GroupedRelu2Spec::new(
                3,
                64,
                32,
                projection("up", packed, false),
                projection("down", packed, false),
            )
            .unwrap(),
            &context,
        )
        .unwrap();
        check(&linear);
        check(&gated);
        check(&relu);
    }
}
