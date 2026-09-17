use super::*;
use crate::{multimodal::*, operation_geometry::*, routing_intervention::*, workspace::*, *};

fn parameter(name: &str) -> ParameterSpec {
    ParameterSpec::trainable(name).unwrap()
}

#[test]
fn every_owning_operation_preserves_options_and_borrows_actual_source_storage() {
    use WorkspaceOperationKind as K;
    use WorkspaceOperationKindView as V;
    let dense = LinearFormatSpec::unscaled(eredu_checkpoint::LinearFormat::Dense).unwrap();
    let selection = TopKGroupSelectionSpec::new(8, 2, GroupScoring::Sigmoid, true).unwrap();
    let selector =
        TopKGroupSelectorSpec::new(7, parameter("selector.weight"), dense.clone(), selection)
            .unwrap()
            .with_bias(parameter("selector.bias"))
            .unwrap();
    let control = GroupSelectionControl {
        expected: selection,
        learned_coefficient_scale: false,
        first_row: 1,
        end_row: 5,
        row_stride: 2,
        action: GroupSelectionAction::Bias {
            stage: GroupScoreStage::RawLogits,
            ids: vec![3, 7],
            values: vec![1.25, -0.75],
        },
        capture_original: true,
    };
    let bank = WorkspaceGroupedBank::Linear(
        GroupedLinearSpec::new(
            8,
            7,
            11,
            GroupedLinearActivation::Silu,
            GroupedProjectionSpec::new(
                parameter("bank.weight"),
                Some(parameter("bank.bias")),
                dense.clone(),
            )
            .unwrap(),
        )
        .unwrap(),
    );
    let multi = MultiAxisRotarySpec {
        axes: vec![
            RotaryAxisSpec {
                dimensions: 4,
                position_offset: -3,
            },
            RotaryAxisSpec {
                dimensions: 8,
                position_offset: 5,
            },
        ],
        base: 1024.0,
        minimum_position: 2,
        layout: MultiAxisRotaryLayout::SplitHalves,
    };
    let kinds = vec![
        K::ParameterPlaceholder,
        K::Initialize,
        K::GeneratedF32Initialization,
        K::Sampling(WorkspaceSamplingOperation::Penalties {
            history_positions: 9,
            repetition: true,
            additive: false,
        }),
        K::CandidateExtraction {
            vocabulary: 31,
            count: 7,
        },
        K::Elementwise("multiply"),
        K::View("transpose"),
        K::Transpose(vec![0, 2, 1, 3]),
        K::Index { selected_axes: 3 },
        K::StaticSlice {starts:vec![0,1,2],ends:vec![2,5,8],strides:vec![1,2,3]},
        K::SliceUpdate {
            starts: vec![0, 2, 4],
        },
        K::Contiguous,
        K::DeepCopy,
        K::Gather { axis: 2 },
        K::Concatenate,
        K::Matmul,
        K::DenseLinear,
        K::Projection(dense.clone()),
        K::ProjectionPrepare(dense.clone()),
        K::ProjectionFinish(dense.clone()),
        K::BlockFp8ActivationDecode,
        K::GroupSelection {
            spec: Box::new(selector),
            supplied_indices: true,
            control: Some(Box::new(control)),
        },
        K::JointGroupSelection(JointGroupSelectionSpec::new(8, 3, 2, 1.25).unwrap()),
        K::Grouped {
            bank: Box::new(bank),
            phase: WorkspaceGroupedPhase::Finish,
            partitions: Some(3),
        },
        K::HyperCollapse(
            Box::new(HyperConnectionSpec {
                streams: 3,
                hidden_size: 7,
                sinkhorn_iterations: 5,
                epsilon: 0.002,
                function: parameter("hyper.function"),
                base: parameter("hyper.base"),
                scale: parameter("hyper.scale"),
            }),
            0.75,
        ),
        K::HyperExpand,
        K::HyperHeadCoefficients(Box::new(HyperHeadSpec {
            streams: 3,
            hidden_size: 7,
            norm_epsilon: 0.001,
            epsilon: 0.002,
            function: parameter("head.function"),
            base: parameter("head.base"),
            scale: parameter("head.scale"),
        })),
        K::HyperHeadSum,
        K::RowParallelProjection(dense.clone(), 3),
        K::Embedding(dense.clone(), EmbeddingLookupPolicy::Strict),
        K::VocabularyParallelLookup {
            format: dense,
            range: VocabularyParallelRange {
                global_vocabulary: 31,
                local: 7..19,
            },
            policy: EmbeddingLookupPolicy::Strict,
        },
        K::Reduction("sum", -2, true),
        K::Normalization("rms", Some(3)),
        K::ConstructedNormalization(NormalizationConstructionSpec {
            groups: Some(3),
            dimensions: 12,
            epsilon: 0.001,
            scale: NormalizationScale::LearnedOffset {
                weight: parameter("norm.weight"),
                offset: 1.25,
            },
        }),
        K::GatedProduct(
            GatedProductPolicy::new(
                GatedProductActivation::Silu,
                Some(3.0),
                Some(5.0),
                1.25,
                -0.5,
            )
            .unwrap(),
        ),
        K::GatedDeltaScan,
        K::SelectiveStateSpaceScan(17, 0.003),
        K::Rotary(
            RotarySpec {
                dimensions: 8,
                traditional: true,
                base: 1024.0,
                arithmetic: RotaryArithmetic::InputProducts,
                algorithm: RotaryAlgorithm::Linear { factor: 2.5 },
            },
            Some(3),
        ),
        K::RotaryFrequencies(8, true, 3),
        K::TensorRotary(8, true, 1024.0, 0.5, 3),
        K::MultiAxisRotary(multi.clone()),
        K::PreparedMultiAxisRotary(multi),
        K::MaskedOutputProjection {
            top_centroids: 3,
            mask_margin: 0.125,
        },
        K::IndexedAttention {
            scale: 0.25,
            local_mask: true,
            pooled_mask: false,
            sinks: true,
        },
        K::PooledAttention {
            scale: 0.75,
            local_mask: false,
            pooled_mask: true,
            sinks: false,
        },
        K::PooledPositions {
            top_k: 3,
            scale: 0.5,
            head_scale: 1.25,
            masked: true,
        },
        K::GatherPooledMask,
        K::RelativeAttention {
            query_offset: 7,
            key_offset: 3,
            window: Some(11),
            log_scaling_floor: Some(5),
            log_scaling_alpha: 0.75,
        },
        K::Attention {
            causal: true,
            window: Some((11, 7)),
            sinks: true,
            softcap: false,
            arithmetic: AttentionArithmetic::InputScores,
        },
        K::BlockwiseAttention {
            policy: WorkspaceBlockwisePolicy {
                query_start: 7,
                context_end: 19,
                scale: 0.25,
                sliding_window: Some(11),
                prefix_tokens: 3,
                mask_origin: Some(5),
                output_type: None,
                options: BlockwiseAttentionOptions {
                    arithmetic: AttentionArithmetic::InputScores,
                    softcap: Some(1.5),
                },
            },
            stage: WorkspaceBlockwiseStage::Accumulate {
                start: 9,
                end: 13,
                previous: true,
                value_pass: true,
                bias: true,
            },
        },
        K::CausalMask(CausalMaskGeometry::new(3, 7, Some(5)).unwrap()),
        K::PoolingMask(PoolingMaskGeometry::new(3, 5, 7, 2).unwrap()),
        K::Convolution {
            stride: vec![2, 3],
            padding: vec![1, 2],
            dilation: vec![3, 2],
            groups: 5,
            transposed: Some(vec![0, 1]),
        },
        K::Pad(PadMode::Edge),
        K::Collective(WorkspaceCollective::GatherFirstAxis {
            axis: 2,
            rank: 1,
            peer_widths: vec![4, 3],
        }),
        K::Collective(WorkspaceCollective::Gather {
            axis: 2,
            rank: 1,
            peer_widths: vec![3, 5, 7],
        }),
    ];
    assert_eq!(kinds.len(), 56);
    for kind in kinds {
        let view = kind.as_view();
        // Derived diagnostics expose every scalar/option, nested source identity,
        // and slice element. Only the explicit borrowed rotary type name differs.
        assert_eq!(
            format!("{kind:?}"),
            format!("{view:?}").replace("MultiAxisRotarySpecRef", "MultiAxisRotarySpec")
        );
        match (&kind, view) {
            (K::Transpose(a), V::Transpose(b)) => assert!(std::ptr::eq(a.as_ptr(), b.as_ptr())),
            (K::Sampling(a), V::Sampling(b)) => assert!(std::ptr::eq(a, b)),
            (K::Projection(a), V::Projection(b))
            | (K::ProjectionPrepare(a), V::ProjectionPrepare(b))
            | (K::ProjectionFinish(a), V::ProjectionFinish(b))
            | (K::RowParallelProjection(a, _), V::RowParallelProjection(b, _))
            | (K::Embedding(a, _), V::Embedding(b, _)) => assert!(std::ptr::eq(a, b)),
            (
                K::VocabularyParallelLookup {
                    format: a,
                    range: r,
                    ..
                },
                V::VocabularyParallelLookup {
                    format: b,
                    range: s,
                    ..
                },
            ) => {
                assert!(std::ptr::eq(a, b));
                assert!(std::ptr::eq(r, s));
            }
            (
                K::GroupSelection {
                    spec: a,
                    control: Some(c),
                    ..
                },
                V::GroupSelection {
                    spec: b,
                    control: Some(d),
                    ..
                },
            ) => {
                assert!(std::ptr::eq(&**a, b));
                assert!(std::ptr::eq(&**c, d));
                let GroupSelectionAction::Bias { ids, values, .. } = &d.action else {
                    unreachable!()
                };
                let GroupSelectionAction::Bias {
                    ids: old_ids,
                    values: old_values,
                    ..
                } = &c.action
                else {
                    unreachable!()
                };
                assert!(std::ptr::eq(ids.as_ptr(), old_ids.as_ptr()));
                assert!(std::ptr::eq(values.as_ptr(), old_values.as_ptr()));
            }
            (K::JointGroupSelection(a), V::JointGroupSelection(b)) => assert!(std::ptr::eq(a, b)),
            (K::Grouped { bank: a, .. }, V::Grouped { bank: b, .. }) => {
                assert!(std::ptr::eq(&**a, b))
            }
            (K::HyperCollapse(a, _), V::HyperCollapse(b, _)) => assert!(std::ptr::eq(&**a, b)),
            (K::HyperHeadCoefficients(a), V::HyperHeadCoefficients(b)) => {
                assert!(std::ptr::eq(&**a, b))
            }
            (K::ConstructedNormalization(a), V::ConstructedNormalization(b)) => {
                assert!(std::ptr::eq(a, b))
            }
            (K::StaticSlice {starts:a,ends:b,strides:c}, V::StaticSlice {starts:d,ends:e,strides:f}) => {
                assert_eq!(a.as_slice(),d);assert_eq!(b.as_slice(),e);assert_eq!(c.as_slice(),f);
                assert!(std::ptr::eq(a.as_ptr(),d.as_ptr()));
                assert!(std::ptr::eq(b.as_ptr(),e.as_ptr()));
                assert!(std::ptr::eq(c.as_ptr(),f.as_ptr()));
            }
            (K::SliceUpdate { starts: a }, V::SliceUpdate { starts: b }) => {
                assert!(std::ptr::eq(a.as_ptr(), b.as_ptr()))
            }
            (K::MultiAxisRotary(a), V::MultiAxisRotary(b))
            | (K::PreparedMultiAxisRotary(a), V::PreparedMultiAxisRotary(b)) => {
                assert!(std::ptr::eq(a.axes.as_ptr(), b.axes.as_ptr()))
            }
            (
                K::Convolution {
                    stride: a,
                    padding: b,
                    dilation: c,
                    transposed: Some(d),
                    ..
                },
                V::Convolution {
                    stride: e,
                    padding: f,
                    dilation: g,
                    transposed: Some(h),
                    ..
                },
            ) => {
                assert!(std::ptr::eq(a.as_ptr(), e.as_ptr()));
                assert!(std::ptr::eq(b.as_ptr(), f.as_ptr()));
                assert!(std::ptr::eq(c.as_ptr(), g.as_ptr()));
                assert!(std::ptr::eq(d.as_ptr(), h.as_ptr()));
            }
            (
                K::Collective(WorkspaceCollective::GatherFirstAxis { peer_widths: a, .. }),
                V::Collective(WorkspaceCollectiveView::GatherFirstAxis { peer_widths: b, .. }),
            ) => assert!(std::ptr::eq(a.as_slice(), b)),
            (
                K::Collective(WorkspaceCollective::Gather { peer_widths: a, .. }),
                V::Collective(WorkspaceCollectiveView::Gather { peer_widths: b, .. }),
            ) => assert!(std::ptr::eq(a.as_ptr(), b.as_ptr())),
            _ => {}
        }
    }
    let sum = WorkspaceCollective::Sum {
        partitions: 5,
        rank: 3,
    };
    assert_eq!(
        sum.as_view(),
        WorkspaceCollectiveView::Sum {
            partitions: 5,
            rank: 3
        }
    );
}

#[test]
fn borrowed_layout_lists_preserve_source_identity_order_and_unbounded_rank() {
    let mut shape = vec![1; 513];
    shape[256] = 7;
    let layouts = [
        WorkspaceLayout::new(&shape, WorkspaceDtype::Float32).unwrap(),
        WorkspaceLayout::new(&[], WorkspaceDtype::Int32).unwrap(),
    ];
    let views = [layouts[0].as_view(), layouts[1].as_view()];
    for list in [
        WorkspaceLayoutList::Owned(&layouts),
        WorkspaceLayoutList::Views(&views),
    ] {
        assert_eq!(list.len(), 2);
        assert_eq!(list.get(0).unwrap().shape().len(), 513);
        for (index, actual) in list.iter().enumerate() {
            assert_eq!(actual, views[index]);
            assert!(std::ptr::eq(
                actual.shape().as_ptr(),
                layouts[index].shape().as_ptr()
            ));
        }
        assert_eq!(list.slice(1..2).unwrap().get(0), Some(views[1]));
        assert_eq!(list.slice(2..2).unwrap().len(), 0);
        assert!(list.slice(1..3).is_none());
        assert!(list.slice(2..1).is_none());
        assert!(list.get(2).is_none());
        assert!(list.array::<3>().is_none());
        assert_eq!(list.array::<2>().unwrap(), views);
    }
}
