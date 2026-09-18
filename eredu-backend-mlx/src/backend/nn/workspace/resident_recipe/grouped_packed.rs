//! The two actual packed projection banks share sorting, projections and Sum.
//! Source facts distinguish their real activation, slices and chunk protocol.
use super::*;
mod mxfp4;
mod sources;
mod affine;
pub(super) use affine::{control_bytes as affine_control_bytes, uses_source as uses_affine_source};
pub(super) fn affine_sources(operation: WorkspaceOperationView<'_>) -> Option<usize> {
    if !affine::uses_source(operation) { return None; }
    let counts = sources::counts(operation)?;
    counts[1].checked_add(counts[2])
}
pub(super) use mxfp4::control_bytes as mxfp4_control_bytes;
pub(super) fn mxfp4_sources(operation: WorkspaceOperationView<'_>) -> Option<usize> {
    mxfp4::packed_sources(operation)
}
struct Bank<'a> {
    first: &'a eredu_nn::GroupedProjectionSpec,
    reduction: eredu_nn::GroupReduction,
    down: &'a eredu_nn::GroupedProjectionSpec,
    input: i32,
    units: i32,
    activation: Lowering,
    pointwise: bool,
    outer_validation: bool,
    slices: usize,
    chunked: bool,
}
impl<'a> Bank<'a> {
    fn new(bank: &'a WorkspaceGroupedBank) -> Option<Self> {
        Some(match bank {
            WorkspaceGroupedBank::GatedProduct(spec) => {
                spec.validate_fixed().ok()?;
                let eredu_nn::GatedProductGroupLayout::Packed { gate_up, down } = spec.layout()
                else {
                    return None;
                };
                Self {
                    first: gate_up,
                    reduction: spec.reduction(),
                    down,
                    input: spec.input_dimensions(),
                    units: spec.intermediate_dimensions(),
                    activation: gated_product_lowering(spec.policy())?,
                    pointwise: spec.policy().activation() == eredu_nn::GatedProductActivation::Silu
                        && spec.policy().sigmoid_multiplier() == 1.0,
                    outer_validation: true,
                    slices: 2,
                    chunked: true,
                }
            }
            WorkspaceGroupedBank::Relu2(spec) => {
                spec.validate_fixed().ok()?;
                if spec.up().bias().is_some()
                    || spec.down().bias().is_some()
                    || spec.up().format().encoding() != eredu_checkpoint::LinearFormat::Dense
                    || spec.down().format().encoding() != eredu_checkpoint::LinearFormat::Dense
                {
                    return None;
                }
                Self {
                    first: spec.up(),
                    reduction: eredu_nn::GroupReduction::Sum,
                    down: spec.down(),
                    input: spec.hidden_dimensions(),
                    units: spec.intermediate_dimensions(),
                    // Existing layers::relu2: Maximum (two casts, two broadcasts,
                    // result) and Square, with one actual eager zero scalar.
                    activation: Lowering::plain(6, 7, 1),
                    pointwise: false,
                    outer_validation: false,
                    slices: 0,
                    chunked: false,
                }
            }
            _ => return None,
        })
    }
}
// The packed bank uses this shared loop for every architecture. Dense here
// describes the actual two projection realizations, not checkpoint family.
pub(super) fn lower(operation: WorkspaceOperationView<'_>) -> Option<Lowering> {
    let WorkspaceOperationKindView::Grouped {
        bank,
        phase,
        partitions,
    } = operation.kind
    else {
        return None;
    };
    if let Some(partitions) = partitions {
        // Positive TP adapters call this same packed forward worker. A down
        // bias then executes the actual full-route suffix below; the caller
        // separately adds that post-reduce value once after its collective.
        let WorkspaceGroupedBank::GatedProduct(spec) = bank else { return None; };
        let eredu_nn::GatedProductGroupLayout::Packed { down, .. } = spec.layout()
            else { return None; };
        if partitions == 0 { return None; }
        if down.bias().is_some()
            && operation.outputs.len() != if phase == WorkspaceGroupedPhase::Units { 4 } else { 2 }
        { return None; }
    }
    let bank = Bank::new(bank)?;
    let gate_up = bank.first;
    let down = bank.down;
    let fp8 = matches!(
        gate_up.format().encoding(),
        eredu_checkpoint::LinearFormat::E4M3BlockFp8(_)
    );
    let mxfp4 = mxfp4::uses_source(operation);
    let affine = affine::uses_source(operation);
    let affine_counts = if affine { Some(sources::counts(operation)?) } else { None };
    if affine {
        // The exact parameter roles and active per-phase workers were checked.
    } else if mxfp4 {
        mxfp4::packed_sources(operation)?;
    } else if fp8 {
        fp8::packed_sources(operation)?;
    } else if gate_up.format().encoding() != eredu_checkpoint::LinearFormat::Dense
        || down.format().encoding() != eredu_checkpoint::LinearFormat::Dense
    {
        return None;
    }
    use crate::backend::nn::grouped::{
        GROUPED_PROJECTION_CHUNK_THRESHOLD as THRESHOLD, GROUPED_PROJECTION_CHUNK_TOKENS as CHUNK,
    };
    let input = operation.inputs.get(0)?;
    let width = usize::try_from(*input.shape().last()?).ok()?;
    if width == 0 {
        return None;
    }
    let tokens = usize::try_from(input.elements().ok()?)
        .ok()?
        .checked_div(width)?;
    let chunked = bank.chunked && tokens > THRESHOLD as usize;
    let chunks = if chunked {
        tokens.div_ceil(CHUNK as usize)
    } else {
        1
    };
    let before = phase != WorkspaceGroupedPhase::Finish;
    let after = phase != WorkspaceGroupedPhase::Units;
    let activation = bank.activation;
    let (projection_nodes, projection_edges, projection_seeds) = if fp8 {
        (12usize, 17usize, 1usize)
    } else {
        (36, 43, 3)
    };
    // MXFP4 executes arange, input reshape, signed-route cast, five-input
    // GatherQMM and result reshape (5 nodes/8 edges/no seeds). The same packed
    // helper selects the existing dense grouped worker for a published floating
    // weight; its 36/43/3 envelope above covers either actual branch, without
    // summing mutually exclusive projection populations.
    // Affine explicit-index GatherQMM has 11 nodes / 15 edges: arange,
    // input reshape, two index broadcasts/casts, three floating casts,
    // six-input GatherQMM, output reshape. Group16 has three selected-bank
    // gathers (9/12), input/output reshape (2/2), three casts and four batch
    // broadcasts (7/7), four-input QMM (1/4): 19/25. The same dense 36/43/3
    // alternative dominates each without summing mutually exclusive workers.
    // The same forward_chunk crosses the callback after its plan, input gather,
    // first projection, any gate/up slices and activation; the second projection
    // and weighted reduction follow it. No concatenation is executed at Units.
    let mut p = 0usize;
    let mut e = 0usize;
    let mut seeds = 0usize;
    if before {
        // plan 15/17/1 + input gather 4/5 + actual gate/up slices.
        p = (19usize.checked_add(bank.slices)?)
            .checked_add(projection_nodes)?
            .checked_add(activation.primitives)?
            .checked_add(9 * usize::from(gate_up.bias().is_some()))?
            .checked_add(if chunked { 3 } else { 0 })?;
        e = (22usize.checked_add(bank.slices)?)
            .checked_add(projection_edges)?
            .checked_add(activation.edges)?
            .checked_add(11 * usize::from(gate_up.bias().is_some()))?
            .checked_add(if chunked { 3 } else { 0 })?;
        seeds = 1usize
            .checked_add(projection_seeds)?
            .checked_add(activation.seeds)?;
    }
    if after {
        p = p
            .checked_add(25)?
            .checked_add(projection_nodes)?
            .checked_add(9 * usize::from(down.bias().is_some()))?;
        e = e
            .checked_add(29)?
            .checked_add(projection_edges)?
            .checked_add(11 * usize::from(down.bias().is_some()))?;
        seeds = seeds.checked_add(1)?.checked_add(projection_seeds)?;
    }
    // Gated banks have flatten and outer safe IDs 32/37/3; Relu2 has
    // just its actual flatten 1/1/0. Final reshape is
    // 1/1; the actual output concatenation exists only after the last chunk.
    let primitives = chunks
        .checked_mul(p)?
        .checked_add(if before {
            if bank.outer_validation { 32 } else { 1 }
        } else {
            0
        })?
        .checked_add(usize::from(after))?
        .checked_add(if after && chunked {
            chunks.checked_add(1)?
        } else {
            0
        })?;
    let edges = chunks
        .checked_mul(e)?
        .checked_add(if before {
            if bank.outer_validation { 37 } else { 1 }
        } else {
            0
        })?
        .checked_add(usize::from(after))?
        .checked_add(if after && chunked {
            chunks.checked_mul(2)?
        } else {
            0
        })?;
    let seeds = chunks
        .checked_mul(seeds)?
        .checked_add(if before && bank.outer_validation {
            3
        } else {
            0
        })?;
    primitives.checked_add(seeds)?;
    let mut value = Lowering::plain(primitives, edges, seeds);
    let projections = usize::from(before).checked_add(usize::from(after))?;
    value.bf16_projection_calls = if fp8 {
        0
    } else {
        chunks.checked_mul(projections)?
    };
    let projection_validations = if fp8 {
        0
    } else if phase == WorkspaceGroupedPhase::Whole {
        value.bf16_projection_calls
    } else {
        // The shared BF16 helper exits before validation unless both scalar
        // inputs are BF16 and its row width applies. Known F32/F16 evidence on
        // the actual phase operand excludes that branch; absent evidence does
        // not. Finish borrows the observer's effective value, not original input.
        let candidate = |index: usize, width: i32| -> Option<usize> {
            let input = operation.inputs.get(index)?;
            Some(usize::from(
                bf16_grouped_width(width)
                    && input.representation().is_none_or(|value| {
                        value.dtype() == eredu_nn::workspace::WorkspaceFloatingType::Bfloat16
                    }),
            ))
        };
        let first = if before { candidate(0, bank.input)? } else { 0 };
        let second = if after {
            candidate(operation.inputs.len().checked_sub(4)?, bank.units)?
        } else {
            0
        };
        chunks.checked_mul(first.checked_add(second)?)?
    };
    value.validations =
        projection_validations.checked_add(usize::from(before && bank.outer_validation))?;
    let top_k = usize::try_from(*operation.inputs.get(1)?.shape().last()?).ok()?;
    if before {
        value.additional_sort_kernels = if chunked {
            chunked_grouped_sort_kernels(tokens, top_k, CHUNK as usize)
        } else {
            tokens.checked_mul(top_k).and_then(grouped_sort_kernels)
        };
        value.pointwise_calls = chunks.checked_mul(activation.pointwise_calls)?;
    }
    if grouped_has_selected_rows(operation)? {
        value.unqualified_kernel_owner = if fp8 {
            (!crate::backend::nn::fp8::kernel::source_qualified())
                .then_some(CustomKernelOwner::BlockFp8)
        } else if (before && bf16_grouped_width(bank.input))
            || (after && bf16_grouped_width(bank.units))
        {
            bf16_projection_source_requirement()
        } else {
            None
        }
        .or_else(|| {
            (before && bank.pointwise)
                .then(pointwise_source_requirement)
                .flatten()
        })
        .or_else(grouped_indexed_source_requirement);
    }
    let activation_extra = activation
        .maximum_births
        .checked_sub(activation.primitives)?
        .checked_sub(activation.seeds)?;
    // Sort has five buffers. Each dense projection has four compactions plus
    // two validation scratch; each FP8 projection has six compactions. Weighted
    // reduction has two buffers. Partition-view copies are already node births.
    let extra = (if before {
        11usize.checked_add(activation_extra)?
    } else {
        0
    })
    .checked_add(if after { 8 } else { 0 })?;
    value.maximum_births = value
        .maximum_births
        .checked_add(if before && bank.outer_validation {
            2
        } else {
            0
        })?
        .checked_add(chunks.checked_mul(extra)?)?;
    value.streams = 1;
    value.intermediate_rank = if fp8 || affine_counts.is_some_and(|counts| counts[2] != 0) { 4 } else { 3 };
    value.maximum_operands = if after { chunks } else { 0 }.max(if affine_counts.is_some_and(|counts| counts[1] != 0) { 6 } else if fp8 || mxfp4 { 5 } else { 4 });
    if fp8 {
        value.backend_shells = chunks.checked_mul(projections)?.checked_mul(3)?;
    }
    if after && chunked {
        safemlx::ops::concatenate_axis_control_bytes()?;
        value.grouped_output_chunks = chunks;
    }
    if phase == WorkspaceGroupedPhase::Units {
        value.grouped_unit_observers = 1;
    }
    if after && partitions.is_some() && down.bias().is_some() {
        // PackedGatedProductGroups::forward_tensor_parallel runs this once
        // AFTER the full forward/chunk concatenation: group_by_id 15/17/1,
        // selected bias take 4/5, weighted sum 25/29/1, subtract 5/6, and the
        // second final adapter reshape 1/1. The original reducible reshape is
        // already present above. No collective or post-reduce addition is here.
        value.primitives = value.primitives.checked_add(50)?;
        value.edges = value.edges.checked_add(58)?;
        value.seeds = value.seeds.checked_add(2)?;
        // Existing source populations: five sort temporaries, two reduction
        // buffers, and the ordinary per-node/seed births of this suffix.
        value.maximum_births = value.maximum_births.checked_add(59)?;
        // Bias ordering is over ALL selected rows, independent of forward's
        // per-chunk sorts; use the existing native sort geometry query.
        value.additional_sort_kernels = value.additional_sort_kernels?
            .checked_add(grouped_sort_kernels(tokens.checked_mul(top_k)?)?);
    }
    // Every completed chunk uses the same selected arithmetic. The TP bias
    // correction, when present, invokes it once more over the full route table.
    let reductions = if after {
        chunks.checked_add(usize::from(partitions.is_some() && down.bias().is_some()))?
    } else { 0 };
    grouped_reduction::extend(&mut value, bank.reduction, top_k, reductions)?;
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
        GroupSelection, GroupedNeuralBackend, GroupedProjectionSpec, GroupedRelu2Operator,
        GroupedRelu2Spec, GroupedUnitBatch, GroupedUnitObserver, LinearFormatSpec, ParameterSpec,
    };
    struct Observe;
    impl GroupedUnitObserver<WorkspaceTensor> for Observe {
        fn observe(&mut self, _: &GroupedUnitBatch<'_, WorkspaceTensor>) -> Result<(), Error> {
            Ok(())
        }
    }
    fn spec() -> GroupedRelu2Spec {
        let projection = |name| {
            GroupedProjectionSpec::new(
                ParameterSpec::trainable(name).unwrap(),
                None,
                LinearFormatSpec::unscaled(eredu_checkpoint::LinearFormat::Dense).unwrap(),
            )
            .unwrap()
        };
        GroupedRelu2Spec::new(4, 8, 16, projection("up"), projection("down")).unwrap()
    }
    fn report(
        tokens: i32,
        observed: bool,
        mechanism: MlxMetalWorkspaceMechanisms,
    ) -> WorkspaceTraceReport {
        let context = WorkspaceContext::new(mechanism);
        let mut module = WorkspaceBackend::grouped_relu2(spec(), &context).unwrap();
        let input = WorkspaceTensor::existing(
            WorkspaceLayout::new(&[tokens, 8], WorkspaceDtype::Float32).unwrap(),
            &context,
        )
        .unwrap();
        let ids = WorkspaceTensor::existing(
            WorkspaceLayout::new(&[tokens, 2], WorkspaceDtype::Uint32).unwrap(),
            &context,
        )
        .unwrap();
        let coefficients = WorkspaceTensor::existing(
            WorkspaceLayout::new(&[tokens, 2], WorkspaceDtype::Float32).unwrap(),
            &context,
        )
        .unwrap();
        let selections = GroupSelection::new(ids, coefficients.clone(), coefficients);
        context.begin_span();
        let output = module
            .forward_grouped_with_unit_observer(
                &input,
                &selections,
                &context,
                observed.then_some(&mut Observe),
            )
            .unwrap();
        context.report(&[output]).unwrap()
    }
    #[test]
    fn relu2_whole_and_unit_phases_use_the_same_two_projection_recipe() {
        let mechanism = MlxMetalWorkspaceMechanisms::current_host().unwrap();
        for tokens in [0, 3, 129] {
            let whole = report(tokens, false, mechanism);
            let observed = report(tokens, true, mechanism);
            let whole_op = whole
                .operations
                .iter()
                .find(|op| matches!(op.kind, WorkspaceOperationKind::Grouped { .. }))
                .unwrap();
            let full = lower(whole_op.as_view()).unwrap();
            let phases: Vec<_> = observed
                .operations
                .iter()
                .filter(|op| matches!(op.kind, WorkspaceOperationKind::Grouped { .. }))
                .collect();
            assert_eq!(phases.len(), 2);
            let first = lower(phases[0].as_view()).unwrap();
            let last = lower(phases[1].as_view()).unwrap();
            assert_eq!(phases[0].outputs.len(), 4);
            assert_eq!(phases[1].inputs.len(), whole_op.inputs.len() + 4);
            assert_eq!(full.primitives, first.primitives + last.primitives);
            assert_eq!(full.edges, first.edges + last.edges);
            assert_eq!(full.seeds, first.seeds + last.seeds);
            assert_eq!(
                full.maximum_births,
                first.maximum_births + last.maximum_births
            );
            assert_eq!(first.grouped_unit_observers, 1);
            assert_eq!(last.grouped_unit_observers, 0);
            assert_eq!(
                full.pointwise_calls, 0,
                "Maximum/Square does not require the gated pointwise source"
            );
            assert_eq!(
                full.grouped_output_chunks, 0,
                "the native Relu2 bank has no chunk loop"
            );
            assert_eq!(full.bf16_projection_calls, 2);
            let recorder = ResidentRecipeRecorder::new(
                InferenceGeometry {
                    batch_size: 1,
                    cached_positions: 0,
                    input_positions: tokens as u64,
                    max_output_tokens: 1,
                    prefill_chunk_positions: tokens.max(1) as u64,
                    output: eredu_core::OutputDemand::Sequence,
                },
                mechanism,
            );
            let reduced = recorder.reduce_trace(&observed, None, 0, 1).unwrap();
            assert_eq!(reduced.first_missing_operation, None);
            assert_eq!(reduced.grouped_outputs.calls, 0);
            assert_eq!(reduced.grouped_outputs.unit_observers, 1);
            assert!(reduced.mutable_storage.is_some());
            let view = whole_op.as_view();
            let refused = WorkspaceOperationView {
                kind: WorkspaceOperationKindView::Grouped {
                    bank: match view.kind {
                        WorkspaceOperationKindView::Grouped { bank, .. } => bank,
                        _ => unreachable!(),
                    },
                    phase: WorkspaceGroupedPhase::Whole,
                    partitions: Some(2),
                },
                ..view
            };
            assert!(
                lower(refused).is_none(),
                "no partition authority follows from a resident recipe"
            );
        }
    }
}

#[cfg(all(test, target_vendor = "apple", feature = "metal", not(feature = "cuda")))]
mod tensor_parallel_tests {
    use super::*;
    use eredu_nn::Tensor;
    use eredu_nn::{GatedProductGroupLayout, GatedProductPolicy, GroupSelection,
        GroupedGatedProductOperator, GroupedGatedProductSpec, GroupedNeuralBackend,
        GroupedProjectionSpec, LinearFormatSpec, ParameterSpec, TensorParallelGroupedGatedProductOperator};

    #[test]
    fn packed_tp_bias_quotes_full_route_suffix_and_two_actual_outputs() {
        let projection = |name, bias| GroupedProjectionSpec::new(
            ParameterSpec::trainable(name).unwrap(), bias,
            LinearFormatSpec::unscaled(eredu_checkpoint::LinearFormat::Dense).unwrap()).unwrap();
        let spec = GroupedGatedProductSpec::new(2,8,2,8,GatedProductPolicy::ordinary_silu(),
            GatedProductGroupLayout::Packed {gate_up:projection("read",None),
                down:projection("write",Some(ParameterSpec::trainable("bias").unwrap()))}).unwrap();
        let mechanism = MlxMetalWorkspaceMechanisms::current_host().unwrap();
        for positions in [2,65] {
            let context = WorkspaceContext::new(mechanism);
            let mut bank = WorkspaceBackend::grouped_gated_product(spec.clone(),&context).unwrap();
            let input = WorkspaceTensor::unloaded_f32(&[1,positions,8],&context).unwrap();
            let ids = WorkspaceTensor::existing(WorkspaceLayout::new(&[positions,2],WorkspaceDtype::Uint32).unwrap(),&context).unwrap();
            let coefficients = WorkspaceTensor::unloaded_f32(&[positions,2],&context).unwrap();
            let selection=GroupSelection::new(ids,coefficients.clone(),coefficients);
            context.begin_span();
            let (reducible,bias)=bank.forward_grouped_tensor_parallel(&input,&selection,2,&context).unwrap().into_parts();
            let bias=bias.expect("actual independent post-reduce bias");
            assert_eq!(bias.shape(),reducible.shape());
            let report=context.report(&[reducible,bias]).unwrap();
            let op=report.operations.iter().find(|op|matches!(op.kind,WorkspaceOperationKind::Grouped{..})).unwrap();
            assert_eq!(op.outputs.len(),2);
            let lowering=lower(op.as_view()).unwrap();
            let routes=(positions as usize)*2;
            assert!(lowering.additional_sort_kernels.unwrap()>=grouped_sort_kernels(routes).unwrap());
            let recorder=ResidentRecipeRecorder::new(InferenceGeometry{batch_size:1,cached_positions:0,
                input_positions:positions as u64,max_output_tokens:1,prefill_chunk_positions:positions as u64,
                output:eredu_core::OutputDemand::Sequence},mechanism);
            let reduced=recorder.reduce_trace(&report,None,0,2).unwrap();
            assert_eq!(reduced.first_missing_operation,None);
            assert!(reduced.graph.is_some()&&reduced.dispatch.is_some()&&reduced.mutable_storage.is_some());
        }
    }

    #[test]
    fn bias_free_packed_tp_keeps_the_actual_ordinary_worker_and_rejects_incomplete_bias_source() {
        let projection = |name, bias| GroupedProjectionSpec::new(
            ParameterSpec::trainable(name).unwrap(), bias,
            LinearFormatSpec::unscaled(eredu_checkpoint::LinearFormat::Dense).unwrap()).unwrap();
        let spec = |bias| GroupedGatedProductSpec::new(2,8,2,8,GatedProductPolicy::ordinary_silu(),
            GatedProductGroupLayout::Packed {gate_up:projection("read",None),down:projection("write",bias)}).unwrap();
        let mechanism = MlxMetalWorkspaceMechanisms::current_host().unwrap();
        for positions in [1,2] {
            let trace = |partitions| {
                let context = WorkspaceContext::new(mechanism);
                let mut bank = WorkspaceBackend::grouped_gated_product(spec(None),&context).unwrap();
                let input = WorkspaceTensor::unloaded_f32(&[1,positions,8],&context).unwrap();
                let ids = WorkspaceTensor::existing(WorkspaceLayout::new(&[positions,1],WorkspaceDtype::Uint32).unwrap(),&context).unwrap();
                let coefficients = WorkspaceTensor::unloaded_f32(&[positions,1],&context).unwrap();
                let selection=GroupSelection::new(ids,coefficients.clone(),coefficients);
                context.begin_span();
                let output=match partitions {
                    Some(count)=>{let (value,bias)=bank.forward_grouped_tensor_parallel(&input,&selection,count,&context).unwrap().into_parts();assert!(bias.is_none());value},
                    None=>bank.forward_grouped(&input,&selection,&context).unwrap(),
                };
                context.report(&[output]).unwrap()
            };
            let ordinary=trace(None);
            let ordinary_op=ordinary.operations.iter().find(|op|matches!(op.kind,WorkspaceOperationKind::Grouped{..})).unwrap();
            let expected=lower(ordinary_op.as_view()).unwrap();
            for partitions in [1,2] {
                let report=trace(Some(partitions));
                let op=report.operations.iter().find(|op|matches!(op.kind,WorkspaceOperationKind::Grouped{..})).unwrap();
                let actual=lower(op.as_view()).unwrap();
                assert_eq!((actual.primitives,actual.edges,actual.seeds,actual.maximum_births),
                    (expected.primitives,expected.edges,expected.seeds,expected.maximum_births));
                let recorder=ResidentRecipeRecorder::new(InferenceGeometry {batch_size:1,cached_positions:0,
                    input_positions:positions as u64,max_output_tokens:1,prefill_chunk_positions:positions as u64,
                    output:eredu_core::OutputDemand::Sequence},mechanism);
                let reduced=recorder.reduce_trace(&report,None,0,1).unwrap();
                assert_eq!(reduced.first_missing_operation,None);assert!(reduced.dispatch.is_some());
                let view=op.as_view();
                let WorkspaceOperationKindView::Grouped{bank,phase,..}=view.kind else{unreachable!()};
                assert!(lower(WorkspaceOperationView{kind:WorkspaceOperationKindView::Grouped{bank,phase,partitions:Some(0)},..view}).is_none());
                let biased=WorkspaceGroupedBank::GatedProduct(spec(Some(ParameterSpec::trainable("down.bias").unwrap())));
                assert!(lower(WorkspaceOperationView{kind:WorkspaceOperationKindView::Grouped{bank:&biased,phase,partitions:Some(partitions)},..view}).is_none());
            }
        }
    }
}
