//! Descriptor/worker population for the existing unpartitioned selector.
use super::*;
use eredu_nn::{GroupScoring, RoutingPrecision};

mod control;

#[derive(Clone, Copy)]
struct Phases {
    summary: Lowering,
    projection: Lowering,
    transform: Lowering,
    ranking: Lowering,
    selection: Lowering,
    weights: Lowering,
}
fn phase(
    before: (usize, usize, usize),
    after: (usize, usize, usize),
    extra: usize,
) -> Option<Lowering> {
    let mut value = Lowering::plain(
        after.0.checked_sub(before.0)?,
        after.1.checked_sub(before.1)?,
        after.2.checked_sub(before.2)?,
    );
    value.maximum_births = value.maximum_births.checked_add(extra)?;
    value.grouped_output_calls = 0;
    value.intermediate_rank = 2;
    Some(value)
}

/// Ordinary routing observes its real cutoff predicate before an optional CPU
/// fallback. Original routing instead keeps the predicate in its lazy Where DAG.
pub(super) fn ordinary_predicate_completions(operation: WorkspaceOperationView<'_>) -> usize {
    match operation.kind {
        WorkspaceOperationKindView::GroupSelection {
            spec,
            supplied_indices: false,
            control: None,
        } if spec.selection().top_k() < spec.selection().group_count() => 1,
        _ => 0,
    }
}

pub(super) fn lowering(operation: WorkspaceOperationView<'_>) -> Option<Lowering> {
    population(operation, true)
}

pub(super) fn allocation_births(operation: WorkspaceOperationView<'_>) -> Option<usize> {
    population(operation, false).map(|value| value.maximum_births)
}

fn population(
    operation: WorkspaceOperationView<'_>,
    require_original_source: bool,
) -> Option<Lowering> {
    if matches!(
        operation.kind,
        WorkspaceOperationKindView::GroupSelection {
            control: Some(_),
            ..
        }
    ) {
        return control::population(operation, require_original_source);
    }
    phases(operation, require_original_source).map(|phases| phases.summary)
}
fn phases(operation: WorkspaceOperationView<'_>, require_original_source: bool) -> Option<Phases> {
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
            || input
                .representation()
                .is_some_and(|actual| actual.dtype() == WorkspaceFloatingType::Float32)
    });
    let affine = matches!(
        spec.format().encoding(),
        eredu_checkpoint::LinearFormat::Affine(_)
    );
    if affine {
        super::super::representation::affine_selector_source(operation)?;
    }
    // This worker still requires the existing unpartitioned arithmetic recipe.
    // Affine precision and role checks come from its actual retained operands.
    if control.is_some()
        || !matches!(
            spec.format().encoding(),
            eredu_checkpoint::LinearFormat::Dense | eredu_checkpoint::LinearFormat::Affine(_)
        )
        || selection.selection_partitions() != 1
        || selection.selected_groups() != 1
        || spec.input_transform().is_some()
        || spec.coefficient_scale().is_some()
        || !affine && extended && !explicit_f32 && !retained_f32
        || !affine
            && operation.inputs.iter().enumerate().any(|(index, input)| {
                if supplied_indices && index == 1 {
                    !matches!(
                        input.dtype(),
                        WorkspaceDtype::Int32 | WorkspaceDtype::Uint32
                    )
                } else {
                    input.dtype() != WorkspaceDtype::Float32
                }
            })
        || operation.outputs.len() != 3
        || operation.inputs.get(0)?.elements().ok()? == 0
    {
        return None;
    }
    let projection = if affine {
        Some(
            super::super::routing::with_selector_projection(operation, super::lowering)
                .ok()
                .flatten()
                .flatten()?,
        )
    } else {
        None
    };
    // Reuse the ordinary affine/dense replacement Projection's casts,
    // broadcasts, QMM/matmul, bias, compactions and split/reduction source.
    // The selector additionally flattens, may cast its input to F32, then
    // restores projection precision and applies the separate score cast.
    let (mut p, mut e, mut seeds) = match projection {
        Some(source) => (
            source.primitives.checked_add(4)?,
            source.edges.checked_add(4)?,
            source.seeds,
        ),
        // Existing dense: flatten, two projection casts, transpose, matmul,
        // output precision boundary, then the separate score cast.
        None => (14usize, 15usize, 0usize),
    };
    if spec.bias().is_some() && !affine {
        p += 5;
        e += 6;
    }
    let projection_end = (p, e, seeds);
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
    let transform_end = (p, e, seeds);
    let crossed = !supplied_indices && selection.top_k() < selection.group_count();
    if require_original_source && crossed && !crate::backend::managed_memory::router::is_ready() {
        return None;
    }
    if !supplied_indices && spec.correction_bias().is_some() {
        p += 5;
        e += 6;
    }
    let ranking_end = (p, e, seeds);
    if supplied_indices {
        // Actual supplied IDs are only reshaped. No ranking, bias or sort runs.
        p += 1;
        e += 1;
    } else {
        // Descending Multiply+ArgPartition+Slice8/9. Crossing ties additionally
        // gather chosen scores, reduce cutoff, compare/reduce both tie counts,
        // global Any, CPU partition/GPU slice and lazy Where:46/53 in total.
        p += if crossed { 46 } else { 8 };
        e += if crossed { 53 } else { 9 };
        seeds += 1;
    }
    let selection_end = (p, e, seeds);
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
    let floating_weight = operation
        .inputs
        .get(1 + usize::from(supplied_indices))?
        .dtype()
        == WorkspaceDtype::Float32;
    if (retained_f32 || affine && floating_weight)
        && arithmetic.projection != RoutingPrecision::Float32
    {
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
        Some(source) => source
            .maximum_births
            .checked_sub(source.primitives)?
            .checked_sub(source.seeds)?,
        None => 6,
    };
    value.maximum_births = p
        .checked_add(seeds)?
        .checked_add(projection_births + selection_births)?
        .checked_add(usize::from(selection.normalize_selected()) * 2)?
        .checked_add(usize::from(pointwise))?;
    let mut projected = phase((0, 0, 0), projection_end, projection_births)?;
    projected.helper_controls = value.helper_controls;
    let mut transformed = phase(projection_end, transform_end, usize::from(pointwise))?;
    transformed.pointwise_calls = usize::from(pointwise);
    let ranked = phase(transform_end, ranking_end, 0)?;
    let mut selected = phase(ranking_end, selection_end, selection_births)?;
    selected.streams = value.streams;
    selected.router_cpu_partitions = value.router_cpu_partitions;
    selected.additional_sort_kernels = value.additional_sort_kernels;
    selected.unqualified_kernel_owner = value.unqualified_kernel_owner;
    let mut weighted = phase(
        selection_end,
        (p, e, seeds),
        usize::from(selection.normalize_selected()) * 2,
    )?;
    weighted.row_sum_calls = value.row_sum_calls;
    Some(Phases {
        summary: value,
        projection: projected,
        transform: transformed,
        ranking: ranked,
        selection: selected,
        weights: weighted,
    })
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
        quote_report(scoring, supplied, normalize)
            .operations
            .into_iter()
            .find(|op| matches!(op.kind, WorkspaceOperationKind::GroupSelection { .. }))
            .unwrap()
    }
    fn quote_report(
        scoring: GroupScoring,
        supplied: bool,
        normalize: bool,
    ) -> WorkspaceTraceReport {
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
    }

    #[test]
    fn ordinary_metal_router_population_retains_cpu_stream_and_completion_frontiers() {
        if !crate::tests::support::native_process::enter("ordinary-mixed-router-population") {
            return;
        }
        crate::tests::support::test_utils::initialize_original_sources();
        crate::backend::managed_memory::router::prepare_before_native_construction();
        let mechanism = MlxMetalWorkspaceMechanisms::current_host().unwrap();
        for supplied in [false, true] {
            let report = quote_report(GroupScoring::SqrtSoftplus, supplied, true);
            let recorder = ResidentRecipeRecorder::new(
                InferenceGeometry {
                    batch_size: 1,
                    cached_positions: 0,
                    input_positions: 2,
                    max_output_tokens: 1,
                    prefill_chunk_positions: 2,
                    output: eredu_core::OutputDemand::Sequence,
                },
                mechanism,
            );
            let reduced = recorder.reduce_trace(&report, None, 0, 3).unwrap();
            assert!(reduced.first_missing_operation.is_none());
            assert!(reduced.unqualified_kernel_owner.is_none());
            let completion = ResidentCompletionRecipe {
                validation_roots: reduced.validation_roots,
                grouped_outputs: reduced.grouped_outputs,
                traversal: reduced.traversal.unwrap(),
                graph: reduced.graph.unwrap(),
                dispatch: reduced.dispatch,
                nested_completions: reduced.nested_completions,
                nested_root_capacity: reduced.nested_root_capacity,
            };
            let dispatch = completion.dispatch.unwrap();
            assert_eq!(dispatch.cpu_entries, usize::from(!supplied));
            assert_eq!(completion.nested_completions, usize::from(!supplied));
            assert_eq!(
                dispatch.completion_streams(),
                Some(1 + usize::from(!supplied))
            );
            let base = completion.ordinary_metal_controls(0).unwrap();
            let waiting = completion.ordinary_metal_controls(2).unwrap();
            assert!(waiting.observed_host_bytes > base.observed_host_bytes);
            assert!(waiting.control_allocations > base.control_allocations);
            let mut foreign = completion;
            foreign.dispatch.as_mut().unwrap().parallel_entries = 1;
            assert!(foreign.ordinary_metal_controls(0).is_none());
        }
    }

    #[test]
    fn ordinary_router_population_does_not_require_an_original_stream_source() {
        if !crate::tests::support::native_process::enter("ordinary-router-population") {
            return;
        }
        assert!(!crate::backend::managed_memory::router::is_ready());
        let operation = quote(GroupScoring::SqrtSoftplus, false, true);
        let before = allocation_births(operation.as_view()).unwrap();
        assert!(before > operation.outputs.len());
        assert!(lowering(operation.as_view()).is_none());
        crate::tests::support::test_utils::initialize_original_sources();
        crate::backend::managed_memory::router::prepare_before_native_construction();
        assert_eq!(
            before,
            lowering(operation.as_view()).unwrap().maximum_births
        );
    }

    #[test]
    fn normalized_router_recipe_retains_actual_source_and_skips_supplied_ranking() {
        if !crate::tests::support::native_process::enter("qualified-recipe") {
            return;
        }
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
        if !crate::tests::support::native_process::enter("qualified-recipe") {
            return;
        }
        let _sources = crate::tests::support::test_utils::initialize_original_sources();
        crate::backend::managed_memory::router::prepare_before_native_construction();
        for supplied in [false, true] {
            let original = quote(GroupScoring::Softmax, supplied, true);
            let explicit = lowering(original.as_view()).unwrap();
            for projection in [
                RoutingPrecision::Preserve,
                RoutingPrecision::Input,
                RoutingPrecision::Float32,
            ] {
                for scores in [
                    RoutingPrecision::Preserve,
                    RoutingPrecision::Input,
                    RoutingPrecision::Float32,
                ] {
                    for coefficients in [
                        RoutingPrecision::Preserve,
                        RoutingPrecision::Input,
                        RoutingPrecision::Float32,
                    ] {
                        let mut operation = original.clone();
                        let WorkspaceOperationKind::GroupSelection { spec, .. } =
                            &mut operation.kind
                        else {
                            unreachable!()
                        };
                        **spec = (**spec).clone().with_arithmetic(RoutingArithmetic {
                            projection,
                            scores,
                            coefficients,
                        });
                        for (index, input) in operation.inputs.iter_mut().enumerate() {
                            if supplied && index == 1 {
                                continue;
                            }
                            *input = input.clone().with_representation(Some(
                                WorkspaceRepresentation::new(WorkspaceFloatingType::Float32, false),
                            ));
                        }
                        let actual = lowering(operation.as_view()).unwrap();
                        assert_eq!(
                            (actual.primitives, actual.edges, actual.maximum_births),
                            (explicit.primitives, explicit.edges, explicit.maximum_births)
                        );
                        assert_eq!(actual.bf16_projection_calls, 0);
                        assert_eq!(
                            actual.helper_controls,
                            if projection == RoutingPrecision::Float32 {
                                0
                            } else {
                                crate::backend::nn::matrix::row_projection_probe_control_bytes()
                                    .unwrap()
                            }
                        );
                        if projection == RoutingPrecision::Float32
                            && scores == RoutingPrecision::Float32
                            && coefficients == RoutingPrecision::Float32
                        {
                            continue;
                        }
                        let weight = if supplied { 2 } else { 1 };
                        operation.inputs[weight] =
                            operation.inputs[weight].clone().with_representation(None);
                        assert!(lowering(operation.as_view()).is_none());
                        for dtype in [
                            WorkspaceFloatingType::Float16,
                            WorkspaceFloatingType::Bfloat16,
                        ] {
                            operation.inputs[weight] =
                                operation.inputs[weight].clone().with_representation(Some(
                                    WorkspaceRepresentation::new(dtype, true),
                                ));
                            assert!(lowering(operation.as_view()).is_none());
                        }
                    }
                }
            }
        }
    }
}

#[cfg(all(
    test,
    target_vendor = "apple",
    feature = "metal",
    not(feature = "cuda")
))]
#[path = "router/affine_tests.rs"]
mod affine_tests;
