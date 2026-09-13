use super::*;
use eredu_checkpoint::{LinearFormat, SourceTensorEncoding, StoredDtype};
use eredu_core::capture::{CaptureError, CaptureSkipReason};
use eredu_runtime::{
    ParameterRole, ReplicatedTextParameterOwner, ReplicatedTextParameterRole,
    ReplicatedTextPhysicalSource, WeightLoweringDescriptor, WeightLoweringKind,
};

struct Budget {
    usage: CaptureUsage,
    limit: CaptureUsage,
}
impl Budget {
    fn new(host: u64) -> Self {
        Self {
            usage: CaptureUsage::default(),
            limit: CaptureUsage {
                host_bytes: host,
                ..Default::default()
            },
        }
    }
}
impl CaptureReservation for Budget {
    fn reserve(&mut self, cost: CaptureUsage) -> Result<Option<CaptureSkipReason>, CaptureError> {
        eredu_runtime::parameter_operations::reserve_parameter_work(
            &mut self.usage,
            cost,
            self.limit,
        )
        .map_err(|error| match error {
            ParameterError::Budget(error) => error,
            other => panic!("{other}"),
        })?;
        Ok(None)
    }
}
fn task(
    format: LinearFormat,
    physical: Vec<usize>,
    logical: Vec<usize>,
) -> ReplicatedTextMaterializationTask {
    let source = SourceTensorEncoding::Safetensors(StoredDtype::U8);
    let descriptor = WeightLoweringDescriptor::new(
        source.clone(),
        format,
        physical.clone(),
        logical.clone(),
        Some(logical.len() - 1),
    )
    .unwrap();
    ReplicatedTextMaterializationTask::from_exact_source(
        "bank.weight",
        ReplicatedTextPhysicalSource::new(
            "source.weight",
            "source.weight",
            "fixture.safetensors",
            "source.weight",
            source,
            1,
        )
        .unwrap(),
        vec![],
        physical,
        logical,
        ReplicatedTextParameterRole::LinearWeight,
        ReplicatedTextParameterOwner::StaticRole("bank".into()),
        format,
        WeightLoweringKind::Direct,
        descriptor,
    )
    .unwrap()
}

#[test]
fn packed_axes_preserve_exact_blocks_expert_permutations_and_fused_rows() {
    let affine = LinearFormat::Affine(eredu_checkpoint::AffineQuantization::new(32, 4).unwrap());
    let iq = LinearFormat::GgufIQuant {
        ggml_type: eredu_gguf::GgmlType::IQ4NL,
        endian: eredu_gguf::Endian::Little,
    };
    for (format, stored) in [
        (LinearFormat::Dense, 64),
        (affine, 8),
        (LinearFormat::MxFp4, 8),
        (LinearFormat::MxFp4, 32),
        (iq, 36),
    ] {
        let task = task(format, vec![5, 8, stored], vec![5, 8, 64]);
        let layout = LocalTensorLayout::new(
            "bank",
            ParameterRole::ExpertIntermediate,
            vec![5, 8, stored],
            vec![2, 4, stored / 2],
            TensorPlacement::Range {
                axis: 2,
                start: stored / 2,
                end: stored,
            },
            None,
            None,
            false,
        )
        .with_additional_placement(TensorPlacement::Indices {
            axis: 0,
            indices: vec![4, 1],
        })
        .with_additional_placement(TensorPlacement::Indices {
            axis: 1,
            indices: vec![2, 3, 6, 7],
        });
        let mut budget = Budget::new(1 << 20);
        let map = derive_parameter_coordinates(&task, &layout, 0, &mut budget)
            .unwrap()
            .unwrap();
        assert_eq!(map.global_shape(), [5, 8, 64]);
        assert_eq!(map.local_shape(), [2, 4, 32]);
        assert_eq!(map.axes()[0].local_to_global(0), Some(4));
        assert_eq!(map.axes()[0].local_to_global(1), Some(1));
        assert_eq!(
            (0..4)
                .map(|n| map.axes()[1].local_to_global(n).unwrap())
                .collect::<Vec<_>>(),
            [2, 3, 6, 7]
        );
        assert_eq!(map.axes()[2].contiguous_range(), Some(32..64));
        let region = eredu_core::parameters::ParameterRegion {
            starts: vec![1, 3, 30],
            shape: vec![4, 4, 6],
        };
        let fragments = map.project_region(&region, 8, &mut budget).unwrap();
        assert_eq!(fragments.fragments().len(), 4);
        assert!(fragments
            .fragments()
            .iter()
            .all(|piece| piece.local().shape == [1, 1, 4]));
        assert_eq!(budget.usage.retained_bytes, 0);
    }
}

#[test]
fn indexed_packed_blocks_cannot_reorder_bytes_or_split_quantization_groups() {
    let format = LinearFormat::GgufIQuant {
        ggml_type: eredu_gguf::GgmlType::IQ4NL,
        endian: eredu_gguf::Endian::Little,
    };
    let task = task(format, vec![2, 54], vec![2, 96]);
    let make = |indices: Vec<usize>| {
        LocalTensorLayout::new(
            "weight",
            ParameterRole::RowProjection,
            vec![2, 54],
            vec![2, indices.len()],
            TensorPlacement::Indices { axis: 1, indices },
            None,
            None,
            false,
        )
    };
    let exact = make((36..54).chain(0..18).collect());
    let mut budget = Budget::new(1 << 20);
    let coordinates = derive_parameter_coordinates(&task, &exact, 0, &mut budget)
        .unwrap()
        .unwrap();
    assert_eq!(
        (0..64)
            .map(|n| coordinates.axes()[1].local_to_global(n).unwrap())
            .collect::<Vec<_>>(),
        (64..96).chain(0..32).collect::<Vec<_>>()
    );
    for indices in [
        (9..27).collect(),
        (0..9).collect(),
        (0..18).rev().collect(),
        (0..18).chain(0..18).collect(),
        (54..72).collect(),
    ] {
        assert!(derive_parameter_coordinates(&task, &make(indices), 0, &mut budget).is_err());
    }
    let mut no_budget = Budget::new(0);
    assert!(matches!(
        derive_parameter_coordinates(&task, &exact, 0, &mut no_budget),
        Err(ParameterError::Budget(_))
    ));
    assert_eq!(no_budget.usage, CaptureUsage::default());
    let bad_shape = LocalTensorLayout::new(
        "weight",
        ParameterRole::RowProjection,
        vec![2, 54],
        vec![2, 17],
        TensorPlacement::Range {
            axis: 1,
            start: 0,
            end: 18,
        },
        None,
        None,
        false,
    );
    assert!(derive_parameter_coordinates(&task, &bad_shape, 0, &mut budget).is_err());
    let repeated = exact.with_additional_placement(TensorPlacement::Range {
        axis: 1,
        start: 0,
        end: 18,
    });
    assert!(derive_parameter_coordinates(&task, &repeated, 0, &mut budget).is_err());
}

#[test]
fn uncompressed_fp8_axes_keep_uneven_selections_and_absent_owners_explicit() {
    let format = LinearFormat::E4M3BlockFp8(
        eredu_checkpoint::BlockFp8Format::new(
            128,
            128,
            eredu_checkpoint::BlockFp8ScaleEncoding::FloatingPoint,
        )
        .unwrap(),
    );
    let task = task(format, vec![259, 257], vec![259, 257]);
    let make = |shape, placement| {
        LocalTensorLayout::new(
            "weight",
            ParameterRole::ColumnProjection,
            vec![259, 257],
            shape,
            placement,
            None,
            None,
            false,
        )
    };
    let mut budget = Budget::new(1 << 20);
    let layout = make(
        vec![130, 257],
        TensorPlacement::Range {
            axis: 0,
            start: 129,
            end: 259,
        },
    );
    let map = derive_parameter_coordinates(&task, &layout, 0, &mut budget)
        .unwrap()
        .unwrap();
    assert_eq!(map.axes()[0].contiguous_range(), Some(129..259));
    assert_eq!(map.local_shape(), [130, 257]);
    let empty = make(
        vec![0, 257],
        TensorPlacement::Range {
            axis: 0,
            start: 259,
            end: 259,
        },
    );
    let map = derive_parameter_coordinates(&task, &empty, 0, &mut budget)
        .unwrap()
        .unwrap();
    assert_eq!(map.local_shape(), [0, 257]);
    for placement in [TensorPlacement::Omit, TensorPlacement::Rank { rank: 1 }] {
        assert!(derive_parameter_coordinates(
            &task,
            &make(vec![259, 257], placement),
            0,
            &mut budget
        )
        .unwrap()
        .is_none());
    }
    assert!(derive_parameter_coordinates(
        &task,
        &make(vec![259, 257], TensorPlacement::Rank { rank: 0 }),
        0,
        &mut budget
    )
    .unwrap()
    .is_some());
}

#[test]
fn retained_dense_selection_resolves_every_rank_without_resolving_artifact_content() {
    use crate::preparation_selection::tests::{inspected_config, BoundedIndependentAdapter};
    use eredu_core::{CompletionCancellationMode, ModelPreparationPlan, ParallelTopology};
    use eredu_runtime::{
        CommunicationCompletionPolicy, NormalizedLoadRequest, ParallelLoadRequest,
        PipelineActivationDtype, PipelineWireContract,
    };
    for tied in [false, true] {
        let config = serde_json::json!({
            "model_type":"llama", "architectures":["LlamaForCausalLM"], "hidden_size":8,
            "intermediate_size":64, "num_hidden_layers":4, "num_attention_heads":4,
            "num_key_value_heads":2, "head_dim":2, "vocab_size":15,
            "rms_norm_eps":1e-5, "max_position_embeddings":32, "tie_word_embeddings":tied
        });
        let args = crate::llama::model_args_from_config_value(&config).unwrap();
        let parameters =
            std::sync::Arc::new(crate::decoder::dense_parameter_description(&args).unwrap());
        let (_root, inspection) = inspected_config(config);
        let topology = ParallelTopology::new(2, 2, 1, 1).unwrap();
        let rank = ParallelRankTopology::new(topology, 0).unwrap();
        let completion = CommunicationCompletionPolicy::new(
            std::time::Duration::from_secs(1),
            CompletionCancellationMode::QuarantineUntilComplete,
        )
        .unwrap();
        let request = NormalizedLoadRequest::default()
            .with_parallel_execution(
                ParallelLoadRequest::new(
                    rank,
                    PipelineWireContract::new(PipelineActivationDtype::Float32),
                    1,
                    32,
                    completion,
                )
                .unwrap(),
            )
            .unwrap();
        let selected = crate::preparation_selection::select_preparation(
            &inspection,
            &request,
            &BoundedIndependentAdapter::default(),
        )
        .unwrap();
        let plan = ModelPreparationPlan::from_retained_admission(inspection, selected.admission())
            .unwrap();
        let sources = crate::prepared_sources::prepare_model_sources(plan, selected).unwrap();
        let discovery = sources.prepare_discovery(Default::default(), Default::default());
        let mut budget = Budget::new(1 << 26);
        assert!(discovery
            .parameter_partition_layout_for_rank("model.embed_tokens.weight", 0, &mut budget)
            .is_err());
        let discovery = discovery
            .bind_partition_parameters(Some(parameters.clone()))
            .unwrap();
        let tasks = sources
            .selected()
            .text_realization()
            .materialization_tasks();
        // Retained discovery uses its immutable index. Compare it against the
        // independent direct projection, including identical budget charges.
        let duplicated = vec![tasks[0].clone(), tasks[0].clone()];
        let ambiguous = ParameterMemberIndex::new(&duplicated, &parameters);
        assert!(matches!(
            ambiguous.output(&duplicated, tasks[0].name()),
            Err(ParameterError::Invalid(_))
        ));
        let mut inspected = 0;
        for task in tasks {
            let mut seen = vec![0usize; task.logical_shape().iter().product()];
            let mut owners = 0;
            for rank in 0..topology.world_size() {
                let before = budget.usage;
                let layout = discovery
                    .parameter_partition_layout_for_rank(task.name(), rank, &mut budget)
                    .unwrap()
                    .unwrap();
                let mut direct_budget = Budget::new(1 << 26);
                direct_budget.usage = before;
                let direct = sources
                    .selected()
                    .execution()
                    .parameter_partition_layout_for_rank(
                        &parameters,
                        task.name(),
                        rank,
                        &mut direct_budget,
                    )
                    .unwrap()
                    .unwrap();
                assert_eq!(layout, direct);
                assert_eq!(budget.usage, direct_budget.usage);
                assert_eq!(
                    layout.global_shape(),
                    task.logical_shape()
                        .iter()
                        .map(|n| *n as u64)
                        .collect::<Vec<_>>()
                );
                let Some(map) = layout.coordinates() else {
                    continue;
                };
                owners += 1;
                let local_count: u64 = map.local_shape().iter().product();
                for index in 0..local_count {
                    let mut cursor = index;
                    let mut global = 0usize;
                    let mut stride = 1;
                    for axis in (0..map.axes().len()).rev() {
                        let local = cursor % map.local_shape()[axis];
                        cursor /= map.local_shape()[axis];
                        global +=
                            map.axes()[axis].local_to_global(local as usize).unwrap() * stride;
                        stride *= map.global_shape()[axis] as usize;
                    }
                    seen[global] += 1;
                }
                for alias in task.aliases() {
                    assert_eq!(
                        layout,
                        discovery
                            .parameter_partition_layout_for_rank(alias, rank, &mut budget)
                            .unwrap()
                            .unwrap()
                    );
                }
            }
            assert!(
                seen.iter().all(|count| *count > 0),
                "complete selected parameter {}",
                task.name()
            );
            assert!(owners >= 2);
            if task.name().contains("layers.0.") || task.name().contains("layers.3.") {
                assert_eq!(owners, 2);
            }
            if task.name() == "model.embed_tokens.weight" {
                assert_eq!(owners, if tied { 4 } else { 2 });
            }
            inspected += 1;
        }
        assert!(inspected > 30);
        assert!(!discovery.identity_is_resolved());
        assert!(!sources.graph().source_identity().is_resolved());
        assert!(discovery
            .parameter_partition_layout_for_rank(
                tasks[0].name(),
                topology.world_size(),
                &mut budget
            )
            .is_err());
        assert!(matches!(
            discovery.parameter_partition_layout_for_rank("absent", 0, &mut budget),
            Err(ParameterError::Missing(_))
        ));
        let mut zero = Budget::new(0);
        assert!(matches!(
            discovery.parameter_partition_layout_for_rank(tasks[0].name(), 0, &mut zero),
            Err(ParameterError::Budget(_))
        ));
        assert_eq!(zero.usage, CaptureUsage::default());
    }
}

#[test]
fn companion_coordinates_keep_scalar_blocks_across_expert_and_fused_row_permutations() {
    let layout = LocalTensorLayout::new(
        "bank.scales",
        ParameterRole::ExpertIntermediate,
        vec![5, 8, 3],
        vec![2, 4, 1],
        TensorPlacement::Range {
            axis: 2,
            start: 2,
            end: 3,
        },
        None,
        None,
        false,
    )
    .with_additional_placement(TensorPlacement::Indices {
        axis: 0,
        indices: vec![4, 1],
    })
    .with_additional_placement(TensorPlacement::Indices {
        axis: 1,
        indices: vec![2, 3, 6, 7],
    });
    let mut budget = Budget::new(1 << 20);
    let map = derive_companion_coordinates(&[5, 8, 3], &layout, 0, &mut budget)
        .unwrap()
        .unwrap();
    assert_eq!(map.local_shape(), [2, 4, 1]);
    assert_eq!(map.axes()[0].local_to_global(0), Some(4));
    assert_eq!(map.axes()[0].local_to_global(1), Some(1));
    assert_eq!(map.axes()[1].local_to_global(2), Some(6));
    assert_eq!(map.axes()[2].contiguous_range(), Some(2..3));
    // A scale cell cannot be expanded as if it were the packed weight block.
    assert!(derive_companion_coordinates(&[5, 8, 96], &layout, 0, &mut budget).is_err());
    assert!(derive_companion_coordinates(&[5, 8, 3], &layout, 0, &mut Budget::new(0)).is_err());
}

#[test]
fn prepared_prediction_parameter_maps_preserve_tp_shards_and_pp_ep_replicas() {
    use crate::preparation_selection::tests::{inspected_config, BoundedIndependentAdapter};
    use eredu_core::{CompletionCancellationMode, ModelPreparationPlan, ParallelTopology};
    use eredu_runtime::{
        CommunicationCompletionPolicy, NormalizedLoadRequest, ParallelLoadRequest,
        PipelineActivationDtype, PipelineWireContract,
    };
    use std::sync::Arc;
    let config = serde_json::json!({
        "model_type": "deepseek_v3", "hidden_size": 8, "intermediate_size": 16,
        "moe_intermediate_size": 8, "num_hidden_layers": 2, "num_attention_heads": 2,
        "vocab_size": 16, "max_position_embeddings": 64, "kv_lora_rank": 4,
        "qk_nope_head_dim": 2, "qk_rope_head_dim": 2, "v_head_dim": 2,
        "first_k_dense_replace": 1, "n_routed_experts": 4, "n_shared_experts": 1,
        "num_experts_per_tok": 2, "n_group": 2, "topk_group": 1,
        "num_nextn_predict_layers": 2, "tie_word_embeddings": false
    });
    let args = crate::deepseek::parse_v3_config(&config).unwrap();
    let parameters = Arc::new(crate::deepseek::parallel::v3_parameter_description(&args).unwrap());
    let target = Arc::new(
        crate::deepseek::parallel::v3_parameter_description(&args.prediction_target().unwrap())
            .unwrap(),
    );
    let (_root, inspection) = inspected_config(config);
    let topology = ParallelTopology::new(2, 2, 2, 1).unwrap();
    let rank = ParallelRankTopology::new(topology, 0).unwrap();
    let completion = CommunicationCompletionPolicy::new(
        std::time::Duration::from_secs(1),
        CompletionCancellationMode::QuarantineUntilComplete,
    )
    .unwrap();
    let request = NormalizedLoadRequest::default()
        .with_parallel_execution(
            ParallelLoadRequest::new(
                rank,
                PipelineWireContract::new(PipelineActivationDtype::Float32),
                1,
                64,
                completion,
            )
            .unwrap(),
        )
        .unwrap();
    let selected = crate::preparation_selection::select_preparation(
        &inspection,
        &request,
        &BoundedIndependentAdapter::default(),
    )
    .unwrap();
    let plan =
        ModelPreparationPlan::from_retained_admission(inspection, selected.admission()).unwrap();
    let sources = crate::prepared_sources::prepare_model_sources(plan, selected).unwrap();
    let discovery = sources
        .prepare_discovery(Default::default(), Default::default())
        .bind_partition_parameters(Some(target.clone()))
        .unwrap();
    let tasks = discovery
        .partition_parameter_tasks()
        .unwrap()
        .collect::<Vec<_>>();
    let prediction_tasks = tasks
        .iter()
        .filter(|task| {
            !target.groups().iter().any(|group| {
                group.members().iter().any(|member| {
                    member.target() == task.name()
                        || task.aliases().iter().any(|alias| alias == member.target())
                })
            })
        })
        .collect::<Vec<_>>();
    assert!(prediction_tasks.len() > 20);
    let mut budget = Budget::new(1 << 28);
    assert!(matches!(
        discovery.parameter_partition_layout_for_rank(prediction_tasks[0].name(), 0, &mut budget),
        Err(ParameterError::Missing(_))
    ));
    let tensor = ParallelTopology::new(2, 1, 1, 1).unwrap();
    let layouts = (0..2)
        .map(|rank| {
            Arc::new(
                crate::partitioned_execution::derive_partitioned_local_layout(
                    &parameters,
                    ParallelRankTopology::new(tensor, rank).unwrap(),
                )
                .unwrap(),
            )
        })
        .collect::<Vec<_>>();
    sources
        .graph()
        .prediction_placement
        .set(Arc::new(
            crate::prediction_extension::PreparedPredictionPlacement::from_prepared(
                rank,
                Some(parameters.clone()),
                Some(layouts[0].clone()),
            ),
        ))
        .unwrap();
    for task in prediction_tasks {
        for rank in 0..topology.world_size() {
            let placement = discovery
                .parameter_partition_layout_for_rank(task.name(), rank, &mut budget)
                .unwrap()
                .unwrap();
            let tensor_rank = ParallelRankTopology::new(topology, rank)
                .unwrap()
                .tensor_parallel_rank();
            let tensor = layouts[tensor_rank].tensor(placement.target()).unwrap();
            let expected = derive_parameter_coordinates(task, tensor, rank, &mut budget).unwrap();
            assert_eq!(
                placement.coordinates(),
                expected.as_ref(),
                "{} rank {rank}",
                task.name()
            );
            assert!(placement.coordinates().is_some());
        }
    }
    // The ordinary embedding retains its actual storage replicas on every stage;
    // it must not be reclassified as a prediction-owned parameter.
    for rank in 0..topology.world_size() {
        let embedding = discovery
            .parameter_partition_layout_for_rank("model.embed_tokens.weight", rank, &mut budget)
            .unwrap()
            .unwrap();
        assert!(embedding.coordinates().is_some());
    }
    assert!(!discovery.identity_is_resolved());
    assert!(!sources.graph().source_identity().is_resolved());
    assert!(matches!(
        discovery.parameter_partition_layout_for_rank("absent", 0, &mut budget),
        Err(ParameterError::Missing(_))
    ));
    let first = tasks
        .iter()
        .find(|task| task.name().contains("eh_proj"))
        .unwrap();
    assert!(discovery
        .parameter_partition_layout_for_rank(first.name(), topology.world_size(), &mut budget)
        .is_err());
    let mut zero = Budget::new(0);
    assert!(matches!(
        discovery.parameter_partition_layout_for_rank(first.name(), 0, &mut zero),
        Err(ParameterError::Budget(_))
    ));
    assert_eq!(zero.usage, CaptureUsage::default());
}
