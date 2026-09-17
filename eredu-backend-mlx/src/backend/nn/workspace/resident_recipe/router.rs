//! Descriptor/worker population for the existing unpartitioned selector.
use super::*;
use eredu_nn::{GroupScoring, RoutingPrecision};

pub(super) fn lowering(operation: WorkspaceOperationView<'_>) -> Option<Lowering> {
    let WorkspaceOperationKindView::GroupSelection {
        spec,
        supplied_indices,
        control,
    } = operation.kind
    else {
        return None;
    };
    let selection = spec.selection();
    spec.validate_fixed().ok()?;
    let extended = supplied_indices
        || selection.normalize_selected()
        || matches!(
            selection.scoring(),
            GroupScoring::Sigmoid | GroupScoring::SqrtSoftplus
        );
    let arithmetic = spec.arithmetic();
    let explicit_f32 = arithmetic.projection == RoutingPrecision::Float32
        && arithmetic.scores == RoutingPrecision::Float32
        && arithmetic.coefficients == RoutingPrecision::Float32;
    // Preserve/Input have the same F32 result only when each floating operand
    // has that actual retained representation. A logical floating layout alone
    // does not establish native precision.
    let retained_f32 = operation.inputs.iter().enumerate().all(|(index, input)| {
        supplied_indices && index == 1
            || input.representation().is_some_and(|actual|
                actual.dtype() == WorkspaceFloatingType::Float32)
    });
    let affine = matches!(spec.format().encoding(), eredu_checkpoint::LinearFormat::Affine(_));
    if affine {
        super::super::representation::affine_selector_source(operation)?;
    }
    // This worker still requires the existing unpartitioned arithmetic recipe.
    // Affine precision and role checks come from its actual retained operands.
    if control.is_some()
        || !matches!(spec.format().encoding(), eredu_checkpoint::LinearFormat::Dense
            | eredu_checkpoint::LinearFormat::Affine(_))
        || selection.selection_partitions() != 1
        || selection.selected_groups() != 1
        || spec.input_transform().is_some()
        || spec.coefficient_scale().is_some()
        || !affine && extended && !explicit_f32 && !retained_f32
        || !affine && operation.inputs.iter().enumerate().any(|(index, input)| {
            if supplied_indices && index == 1 {
                !matches!(input.dtype(), WorkspaceDtype::Int32 | WorkspaceDtype::Uint32)
            } else { input.dtype() != WorkspaceDtype::Float32 }
        })
        || operation.outputs.len() != 3
        || operation.inputs.get(0)?.elements().ok()? == 0
    { return None; }
    let projection = if affine {
        Some(super::super::routing::with_selector_projection(operation, super::lowering).ok().flatten().flatten()?)
    } else { None };
    // Reuse the ordinary affine/dense replacement Projection's casts,
    // broadcasts, QMM/matmul, bias, compactions and split/reduction source.
    // The selector additionally flattens, may cast its input to F32, then
    // restores projection precision and applies the separate score cast.
    let (mut p, mut e, mut seeds) = match projection {
        Some(source) => (source.primitives.checked_add(4)?, source.edges.checked_add(4)?, source.seeds),
        // Existing dense: flatten, two projection casts, transpose, matmul,
        // output precision boundary, then the separate score cast.
        None => (14usize, 15usize, 0usize),
    };
    let pointwise = match selection.scoring() {
        GroupScoring::Softmax => {
            p += 2;
            e += 2;
            false
        }
        GroupScoring::SelectedSoftmax => false,
        // Native sigmoid cast+unary dominates CustomKernel; final dtype cast.
        GroupScoring::Sigmoid => {
            p += 3;
            e += 3;
            true
        }
        // The shared softplus is LogAddExp(x,I32 zero), then sqrt cast+unary.
        GroupScoring::SqrtSoftplus => {
            p += 7;
            e += 8;
            seeds += 1;
            false
        }
        _ => return None,
    };
    if spec.bias().is_some() && !affine {
        p += 5;
        e += 6;
    }
    let crossed = !supplied_indices && selection.top_k() < selection.group_count();
    if crossed && !crate::backend::managed_memory::router::is_ready() {
        return None;
    }
    if supplied_indices {
        // Actual supplied IDs are only reshaped. No ranking, bias or sort runs.
        p += 1;
        e += 1;
    } else {
        if spec.correction_bias().is_some() {
            p += 5;
            e += 6;
        }
        // Descending Multiply+ArgPartition+Slice8/9. Crossing ties additionally
        // gather chosen scores, reduce cutoff, compare/reduce both tie counts,
        // global Any, CPU partition/GPU slice and lazy Where:46/53 in total.
        p += if crossed { 46 } else { 8 };
        e += if crossed { 53 } else { 9 };
        seeds += 1;
    }
    p += 3;
    e += 4; // selected-score GatherAxis
    if selection.scoring() == GroupScoring::SelectedSoftmax {
        p += 2;
        e += 2;
    }
    if selection.normalize_selected() {
        // Widen candidate, native Reduce/Squeeze or one row-Sum CustomKernel,
        // and result restore. Include possible input compaction separately.
        p += 4;
        e += 4;
        if selection.normalization_epsilon() != 0.0 {
            p += 5;
            e += 6;
            seeds += 1;
        }
        p += 5;
        e += 6; // normalization Divide
    }
    if selection.coefficient_scale() != 1.0 {
        p += 5;
        e += 6;
        seeds += 1;
    }
    p += 1;
    e += 1; // final coefficient precision boundary
    let mut value = Lowering::plain(p, e, seeds);
    // Preserve/Input projection probes the BF16 helper before Matmul. On the
    // retained F32 branch that probe returns before creating native values.
    // Its fixed caller controls still belong to the admitted operation.
    let floating_weight = operation.inputs.get(1 + usize::from(supplied_indices))?
        .dtype() == WorkspaceDtype::Float32;
    if (retained_f32 || affine && floating_weight) && arithmetic.projection != RoutingPrecision::Float32 {
        value.helper_controls = crate::backend::nn::matrix::row_projection_probe_control_bytes()?;
    }
    value.intermediate_rank = 2;
    value.streams = if crossed { 2 } else { 1 };
    value.router_cpu_partitions = usize::from(crossed);
    value.pointwise_calls = usize::from(pointwise);
    value.row_sum_calls = usize::from(selection.normalize_selected());
    value.additional_sort_kernels = if supplied_indices {
        Some(0)
    } else {
        usize::try_from(selection.group_count())
            .ok()
            .and_then(grouped_sort_kernels)
    };
    value.unqualified_kernel_owner =
        grouped_indexed_source_requirement().map(|_| CustomKernelOwner::RouterIndexed);
    // Matmul compactions/partials6; crossing reductions8, sort ping-pong/table5,
    // two actual fence births; normalization two-pass/compaction2; pointwise1.
    let selection_births = if crossed {
        8 + 5 + 2
    } else if supplied_indices {
        0
    } else {
        5
    };
    let projection_births = match projection {
        Some(source) => source.maximum_births.checked_sub(source.primitives)?.checked_sub(source.seeds)?,
        None => 6,
    };
    value.maximum_births = p
        .checked_add(seeds)?
        .checked_add(projection_births + selection_births)?
        .checked_add(usize::from(selection.normalize_selected()) * 2)?
        .checked_add(usize::from(pointwise))?;
    Some(value)
}

#[cfg(all(
    test,
    target_vendor = "apple",
    feature = "metal",
    not(feature = "cuda")
))]
mod tests {
    use super::*;
    use eredu_nn::{
        GroupSelectionOperator, GroupedNeuralBackend, LinearFormatSpec, ParameterSpec,
        RoutingArithmetic, TopKGroupSelectionSpec, TopKGroupSelectorSpec,
    };

    fn quote(scoring: GroupScoring, supplied: bool, normalize: bool) -> WorkspaceOperation {
        let context = WorkspaceContext::new(MlxMetalWorkspaceMechanisms::current_host().unwrap());
        let selection = TopKGroupSelectionSpec::new(4, 1, scoring, normalize)
            .unwrap()
            .with_weight_policy(1e-20, 1.0)
            .unwrap();
        let spec = TopKGroupSelectorSpec::new(
            8,
            ParameterSpec::trainable("router.weight").unwrap(),
            LinearFormatSpec::unscaled(eredu_checkpoint::LinearFormat::Dense).unwrap(),
            selection,
        )
        .unwrap()
        .with_correction_bias(ParameterSpec::trainable("router.bias").unwrap())
        .unwrap()
        .with_arithmetic(RoutingArithmetic::uniform(RoutingPrecision::Float32));
        let mut selector = WorkspaceBackend::top_k_group_selector(spec, &context).unwrap();
        let input = WorkspaceTensor::existing(
            WorkspaceLayout::new(&[1, 2, 8], WorkspaceDtype::Float32).unwrap(),
            &context,
        )
        .unwrap();
        let ids = WorkspaceTensor::existing(
            WorkspaceLayout::new(&[2, 1], WorkspaceDtype::Uint32).unwrap(),
            &context,
        )
        .unwrap();
        context.begin_span();
        let result = if supplied {
            selector.select_indices(&input, &ids, &context).unwrap()
        } else {
            selector.select(&input, &context).unwrap()
        };
        let report = context
            .report(&[
                result.group_indices().clone(),
                result.selected_scores().clone(),
                result.coefficients().clone(),
            ])
            .unwrap();
        assert!(report.total_bytes.is_some());
        report
            .operations
            .into_iter()
            .find(|op| matches!(op.kind, WorkspaceOperationKind::GroupSelection { .. }))
            .unwrap()
    }

    #[test]
    fn normalized_router_recipe_retains_actual_source_and_skips_supplied_ranking() {
        let _sources = crate::tests::support::test_utils::initialize_original_sources();
        // This fixture creates only a neutral selector. Prepare the same actual
        // native stream/worker owner ordinary native selector construction owns.
        crate::backend::managed_memory::router::prepare_before_native_construction();
        assert!(crate::backend::managed_memory::router::is_ready());
        for scoring in [GroupScoring::Sigmoid, GroupScoring::SqrtSoftplus] {
            let ranked = quote(scoring, false, true);
            let supplied = quote(scoring, true, true);
            let unnormalized = quote(scoring, true, false);
            let a = lowering(ranked.as_view()).unwrap();
            let b = lowering(supplied.as_view()).unwrap();
            let c = lowering(unnormalized.as_view()).unwrap();
            assert_eq!((a.router_cpu_partitions, b.router_cpu_partitions), (1, 0));
            assert_eq!((a.streams, b.streams), (2, 1));
            assert_eq!((b.row_sum_calls, c.row_sum_calls), (1, 0));
            assert_eq!(
                b.pointwise_calls,
                usize::from(scoring == GroupScoring::Sigmoid)
            );
            assert!(a.primitives > b.primitives && b.primitives > c.primitives);
            assert!(a.maximum_births > b.maximum_births);
            assert_eq!(b.additional_sort_kernels, Some(0));
            assert!(crate::backend::managed_memory::row_kernels::sum_source_qualified());
            assert!(crate::backend::managed_memory::row_kernels::sum_control_bytes(2).is_some());
        }
    }

    #[test]
    fn normalized_router_uses_retained_f32_precision_and_accounts_rejected_probe() {
        let _sources = crate::tests::support::test_utils::initialize_original_sources();
        crate::backend::managed_memory::router::prepare_before_native_construction();
        for supplied in [false, true] {
            let original = quote(GroupScoring::Softmax, supplied, true);
            let explicit = lowering(original.as_view()).unwrap();
            for projection in [RoutingPrecision::Preserve, RoutingPrecision::Input,
                RoutingPrecision::Float32] {
                for scores in [RoutingPrecision::Preserve, RoutingPrecision::Input,
                    RoutingPrecision::Float32] {
                    for coefficients in [RoutingPrecision::Preserve, RoutingPrecision::Input,
                        RoutingPrecision::Float32] {
                        let mut operation = original.clone();
                        let WorkspaceOperationKind::GroupSelection { spec, .. } = &mut operation.kind
                            else { unreachable!() };
                        **spec = (**spec).clone().with_arithmetic(RoutingArithmetic {
                            projection, scores, coefficients });
                        for (index, input) in operation.inputs.iter_mut().enumerate() {
                            if supplied && index == 1 { continue; }
                            *input = input.clone().with_representation(Some(
                                WorkspaceRepresentation::new(WorkspaceFloatingType::Float32, false)));
                        }
                        let actual = lowering(operation.as_view()).unwrap();
                        assert_eq!((actual.primitives, actual.edges, actual.maximum_births),
                            (explicit.primitives, explicit.edges, explicit.maximum_births));
                        assert_eq!(actual.bf16_projection_calls, 0);
                        assert_eq!(actual.helper_controls, if projection == RoutingPrecision::Float32 {
                            0
                        } else { crate::backend::nn::matrix::row_projection_probe_control_bytes().unwrap() });
                        if projection == RoutingPrecision::Float32 && scores == RoutingPrecision::Float32
                            && coefficients == RoutingPrecision::Float32 { continue; }
                        let weight = if supplied { 2 } else { 1 };
                        operation.inputs[weight] = operation.inputs[weight].clone().with_representation(None);
                        assert!(lowering(operation.as_view()).is_none());
                        for dtype in [WorkspaceFloatingType::Float16, WorkspaceFloatingType::Bfloat16] {
                            operation.inputs[weight] = operation.inputs[weight].clone().with_representation(
                                Some(WorkspaceRepresentation::new(dtype, true)));
                            assert!(lowering(operation.as_view()).is_none());
                        }
                    }
                }
            }
        }
    }
}

#[cfg(all(test, target_vendor = "apple", feature = "metal", not(feature = "cuda")))]
#[path = "router/affine_tests.rs"]
mod affine_tests;
