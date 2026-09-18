use super::*;
use eredu_core::{ModelConfigurationResolver, ParallelRankTopology, ParallelTopology};
use eredu_runtime::{
    ExecutionGraph, ExecutionUnitLayout, OwnedParameterGroupSpec, ParameterGroupOwner,
    ParameterGroupSpec,
};

fn describe(value: &serde_json::Value) -> ArchitectureDescriptor {
    crate::configuration::MODEL_CONFIGURATIONS
        .resolve_safetensors(value)
        .unwrap()
        .architecture_plan()
        .architecture_descriptor()
}

// Actual family unit declarations, with their logical execution owners. Static
// modules are irrelevant to this resolver and are deliberately not fabricated.
fn unit_parameters(
    group: &str,
    count: usize,
    parameters: impl Fn(usize) -> Vec<ParameterGroupSpec>,
) -> ArchitectureParameterDescription {
    let graph = ExecutionGraph::chain([group]).unwrap();
    let layout = ExecutionUnitLayout::new(&graph, [count]).unwrap();
    let owned = (0..count)
        .flat_map(|index| {
            let owner =
                ParameterGroupOwner::execution_unit(layout.group_id(0).unwrap().clone(), index);
            parameters(index)
                .into_iter()
                .map(move |p| OwnedParameterGroupSpec::new(owner.clone(), p))
        })
        .collect::<Vec<_>>();
    ArchitectureParameterDescription::new(
        &graph,
        &layout,
        owned.iter().map(|p| p.group().clone()),
        owned.clone(),
    )
    .unwrap()
}
fn local(owner: &ParameterGroupOwner, stage: usize, per_stage: usize) -> bool {
    match owner {
        ParameterGroupOwner::ExecutionUnit { global_unit, .. } => *global_unit / per_stage == stage,
        _ => panic!("complete decoder values require an execution unit"),
    }
}
fn gemma() -> (ArchitectureDescriptor, ArchitectureParameterDescription) {
    let value = serde_json::json!({
        "model_type":"gemma4_unified", "text_config": {
            "model_type":"gemma4_text", "hidden_size":8,"num_hidden_layers":4,
            "intermediate_size":12,"num_attention_heads":2,"num_key_value_heads":2,
            "head_dim":4,"rms_norm_eps":0.00001,"vocab_size":19,
            "max_position_embeddings":64,"num_kv_shared_layers":2,
            "layer_types":["sliding_attention","full_attention","sliding_attention","full_attention"],
            "sliding_window":4,"enable_moe_block":true,"num_experts":4,
            "top_k_experts":2,"moe_intermediate_size":6
        }
    });
    let args =
        crate::gemma4::FamilyConfig::from_hf_json(&serde_json::to_vec(&value).unwrap()).unwrap();
    let parameters = unit_parameters(crate::gemma4::TEXT_EXECUTION_GROUP, 4, |i| {
        crate::gemma4::layer_parameter_groups(&args.text, i).unwrap()
    });
    (describe(&value), parameters)
}
fn inkling() -> (ArchitectureDescriptor, ArchitectureParameterDescription) {
    let value = serde_json::json!({
        "model_type":"inkling_mm_model", "image_token_id":5,
        "text_config":{"hidden_size":8,"num_hidden_layers":2,"vocab_size":16,
            "num_attention_heads":4,"num_key_value_heads":2,"head_dim":2,
            "sliding_window_size":4,"layer_types":["full_attention","sliding_attention"],
            "mlp_layer_types":["dense","dense"],"sconv_kernel_size":3,"d_rel":2,
            "rel_extent":8,"intermediate_size":12,"dense_intermediate_size":12,
            "n_routed_experts":4,"num_experts_per_tok":2,"n_shared_experts":1,"moe_intermediate_size":6},
        "mtp_config":{"num_nextn_predict_layers":2,"local_layer_ids":[1],"chain_hidden_post_norm":true}
    });
    let args =
        crate::inkling::ModelArgs::from_hf_json(&serde_json::to_vec(&value).unwrap()).unwrap();
    let parameters = unit_parameters(crate::inkling::TEXT_EXECUTION_GROUP, 2, |i| {
        crate::inkling::layer_parameter_groups(&args, i).unwrap()
    });
    (describe(&value), parameters)
}

#[test]
fn complete_decoder_rows_follow_composite_invocations_and_all_replicas() {
    for (descriptor, parameters, prefix, count) in [
        {
            let (d, p) = gemma();
            (d, p, "model.language_model", 4)
        },
        {
            let (d, p) = inkling();
            (d, p, "model", 2)
        },
    ] {
        let topology = ParallelTopology::new(2, 2, 2, 1).unwrap();
        let mut layouts = Vec::new();
        for rank in 0..topology.world_size() {
            let topology = ParallelRankTopology::new(topology, rank).unwrap();
            let mut observations = SourceMap::new();
            register(&mut observations, &descriptor, &parameters, |owner| {
                local(owner, topology.pipeline_parallel_rank(), count / 2)
            })
            .unwrap();
            for layer in 0..count {
                for boundary in ["input", "input.effective", "output", "output.effective"] {
                    let path = format!("{prefix}.layers.{layer}.{boundary}");
                    let point = &observations[&path];
                    assert_eq!(point.axis(), "hidden");
                    assert_eq!(point.site(), ObservationHookSite::Unit);
                    assert_eq!(point.combination(), PartitionCaptureCombination::Disjoint);
                    let expected = layer / (count / 2) == topology.pipeline_parallel_rank();
                    assert_eq!(point.exports(), expected);
                    assert_eq!(
                        point.coordinates().map(|m| m.contiguous_range()),
                        expected.then_some(Some(0..8))
                    );
                }
            }
            for point in descriptor.observations.points.iter().filter(|point| {
                descriptor
                    .node(&point.node_id)
                    .is_some_and(|node| node.kind == ArchitectureNodeKind::DecoderBlock)
                    && point
                        .axes
                        .as_ref()
                        .is_some_and(|axes| axes.iter().any(|a| a.name == "hidden"))
            }) {
                let node = descriptor.node(&point.node_id).unwrap();
                if !node.observation_paths.contains(&point.path)
                    || crate::speculative_execution::speculative_capture_scope(
                        &descriptor,
                        &node.id,
                    )
                    .unwrap()
                        != SpeculativeCaptureScope::Target
                {
                    continue;
                }
                let owner = node_invocation_owner(&descriptor, &parameters, &node.id).unwrap();
                let expected = local(owner, topology.pipeline_parallel_rank(), count / 2);
                assert_eq!(
                    observations[&point.path]
                        .coordinates()
                        .map(|m| m.contiguous_range()),
                    expected.then_some(Some(0..8)),
                    "complete unit evidence {}",
                    point.path
                );
            }
            // No traversal into child operators or prediction subtrees.
            assert!(
                descriptor
                    .components
                    .iter()
                    .all(|c| !observations.contains_key(&c.activation))
            );
            assert!(observations.keys().all(|path| {
                let point = descriptor.observations.get(path).unwrap();
                descriptor.node(&point.node_id).unwrap().kind == ArchitectureNodeKind::DecoderBlock
                    && crate::speculative_execution::speculative_capture_scope(
                        &descriptor,
                        &point.node_id,
                    )
                    .unwrap()
                        == SpeculativeCaptureScope::Target
            }));
            layouts.push(ComponentPartitionLayout {
                topology,
                groups: SourceMap::new(),
                paths: SourceMap::new(),
                observations,
                routed: SourceMap::new(),
            });
        }
        let layouts = ComponentPartitionLayouts::new(topology, layouts).unwrap();
        for layer in 0..count {
            let expected = (0..topology.world_size())
                .filter(|rank| {
                    ParallelRankTopology::new(topology, *rank)
                        .unwrap()
                        .pipeline_parallel_rank()
                        == layer / (count / 2)
                })
                .collect::<Vec<_>>();
            assert_eq!(
                expected.len(),
                4,
                "all TP/EP replicas execute the complete value"
            );
            assert_eq!(
                layouts.capture_hook_members(&format!("{prefix}.layers.{layer}.input")),
                Some(expected)
            );
        }
    }
}

#[test]
fn complete_decoder_rows_keep_shared_physical_passes_as_distinct_logical_invocations() {
    let value = serde_json::json!({"model_type":"nanbeige","hidden_size":8,"intermediate_size":12,
        "num_hidden_layers":2,"num_attention_heads":2,"num_key_value_heads":2,"head_dim":4,
        "vocab_size":16,"max_position_embeddings":64,"rms_norm_eps":0.00001,"num_loops":2});
    let args = crate::nanbeige::model_args_from_config_value(&value).unwrap();
    let parameters = crate::decoder::dense_parameter_description(&args).unwrap();
    let mut descriptor = describe(&value);
    assert_eq!(descriptor.layer_groups[0].physical_layer_count, 2);
    assert_eq!(descriptor.layer_groups[0].passes.len(), 2);
    assert!(
        descriptor
            .parameter_groups
            .iter()
            .any(|g| g.shared_with.is_some())
    );
    // A descriptor-only node, even with plausible geometry and an actual source
    // prefix, is not an executed layer merely because it is a DecoderBlock.
    let mut unexecuted = descriptor.node("decoder.layers.0").unwrap().clone();
    unexecuted.id = "unexecuted".into();
    let mut point = descriptor
        .observations
        .get("model.layers.0.input")
        .unwrap()
        .clone();
    point.path = "unexecuted.value".into();
    point.node_id = unexecuted.id.clone();
    unexecuted.observation_paths = vec![point.path.clone()];
    descriptor.nodes.push(unexecuted);
    descriptor.observations.points.push(point);
    for stage in 0..2 {
        let mut observations = SourceMap::new();
        register(&mut observations, &descriptor, &parameters, |o| {
            local(o, stage, 2)
        })
        .unwrap();
        assert!(!observations.contains_key("unexecuted.value"));
        for layer in 0..4 {
            for suffix in ["input", "input.effective", "output", "output.effective"] {
                assert_eq!(
                    observations[&format!("model.layers.{layer}.{suffix}")]
                        .coordinates()
                        .is_some(),
                    layer / 2 == stage
                );
            }
        }
    }
}

#[test]
fn complete_decoder_rows_preserve_v4_stream_axes_and_existing_stream_placements() {
    let value = serde_json::json!({"model_type":"deepseek_v4","hidden_size":8,"moe_intermediate_size":8,
        "num_hidden_layers":2,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":4,
        "qk_rope_head_dim":2,"q_lora_rank":4,"o_lora_rank":2,"o_groups":2,"vocab_size":16,
        "max_position_embeddings":64,"sliding_window":4,"compress_ratios":[0,4],
        "index_n_heads":2,"index_head_dim":4,"index_topk":1,"hc_mult":2,"hc_sinkhorn_iters":2,
        "n_routed_experts":2,"n_shared_experts":1,"num_experts_per_tok":1,"num_hash_layers":1});
    let args = crate::deepseek::parse_v4_config(&value).unwrap();
    let parameters = crate::deepseek::parallel::v4_parameter_description(&args).unwrap();
    let descriptor = describe(&value);
    let stream = descriptor
        .component_readout
        .as_ref()
        .unwrap()
        .stream_residual
        .as_ref()
        .unwrap();
    for stage in 0..2 {
        let mut observations = SourceMap::new();
        streams::register(
            &mut observations,
            &descriptor,
            stream,
            &|name| {
                let owner = parameters
                    .groups()
                    .iter()
                    .find(|g| g.members().iter().any(|m| m.target() == name))
                    .unwrap()
                    .owner();
                Ok(match owner {
                    ParameterGroupOwner::ExecutionUnit { .. } => local(owner, stage, 1),
                    _ => stage == 1,
                })
            },
            (stage == 0, ObservationHookSite::Input),
            (stage == 1, ObservationHookSite::Readout),
        )
        .unwrap();
        let before = observations.clone();
        register(&mut observations, &descriptor, &parameters, |o| {
            local(o, stage, 1)
        })
        .unwrap();
        for (path, placement) in before {
            assert_eq!(observations[&path], placement, "{path}");
        }
        for layer in 0..2 {
            for suffix in ["input", "input.effective", "output", "output.effective"] {
                let path = format!("layers.{layer}.{suffix}");
                let point = descriptor.observations.get(&path).unwrap();
                let axes = point.axes.as_ref().unwrap();
                assert_eq!(
                    axes.iter().map(|a| a.name.as_str()).collect::<Vec<_>>(),
                    ["batch", "sequence", "stream", "hidden"]
                );
                let placement = &observations[&path];
                assert_eq!(
                    placement.coordinates().map(|m| m.contiguous_range()),
                    (layer == stage).then_some(Some(0..8))
                );
                let shape = [1, 3, 2, 8];
                if let Some(coordinates) = placement.coordinates() {
                    use eredu_core::capture::{
                        CaptureSchedule, CaptureSelection, CaptureSlicePartition, CaptureTransform,
                        resolve_slice,
                    };
                    let selection = CaptureSelection {
                        id: path.clone(),
                        path: path.clone(),
                        schedule: CaptureSchedule::default(),
                        slices: vec![],
                        transform: CaptureTransform::Slice,
                    };
                    let resolved = resolve_slice(point, &selection, &shape).unwrap();
                    let projected =
                        CaptureSlicePartition::new(&shape, &resolved, 3, coordinates, 1).unwrap();
                    assert_eq!(projected.local_shape(), shape);
                    assert_eq!(projected.fragments()[0].local().shape, shape);
                }
            }
        }
    }
}

#[test]
fn complete_decoder_rows_reject_conflicting_placements_and_foreign_node_or_owner() {
    let (descriptor, parameters) = gemma();
    let path = "model.language_model.layers.0.input";
    let mut observations = SourceMap::new();
    register(&mut observations, &descriptor, &parameters, |_| true).unwrap();
    observations.get_mut(path).unwrap().site = ObservationHookSite::Readout;
    assert!(
        matches!(register(&mut observations,&descriptor,&parameters,|_|true),Err(ComponentPartitionError::DuplicateIdentity(p)) if p==path)
    );
    let mut foreign = descriptor.clone();
    foreign
        .observations
        .points
        .iter_mut()
        .find(|p| p.path == path)
        .unwrap()
        .node_id = "decoder.1".into();
    assert!(matches!(
        register(&mut SourceMap::new(), &foreign, &parameters, |_| true),
        Err(ComponentPartitionError::Capture(CaptureError::Invalid(_)))
    ));
    let mut owned = parameters.groups().to_vec();
    let first = owned[0].group().clone();
    owned[0] = OwnedParameterGroupSpec::new(
        ParameterGroupOwner::execution_unit(
            parameters.unit_layout().group_id(0).unwrap().clone(),
            1,
        ),
        first,
    );
    let foreign = ArchitectureParameterDescription::new(
        parameters.graph(),
        parameters.unit_layout(),
        owned.iter().map(|g| g.group().clone()),
        owned.clone(),
    )
    .unwrap();
    assert!(matches!(
        register(&mut SourceMap::new(), &descriptor, &foreign, |_| true),
        Err(ComponentPartitionError::Capture(CaptureError::Invalid(_)))
    ));
    let mut unbound = descriptor.clone();
    unbound
        .nodes
        .iter_mut()
        .find(|n| n.id == "decoder.0")
        .unwrap()
        .observation_paths
        .retain(|p| p != "model.language_model.layers.0.input.effective");
    let mut exact = SourceMap::new();
    register(&mut exact, &unbound, &parameters, |_| true).unwrap();
    assert!(exact.contains_key(path));
    assert!(
        !exact.contains_key("model.language_model.layers.0.input.effective"),
        "catalog presence alone does not manufacture a node binding"
    );
    let mut scoped = descriptor.clone();
    // Even an erroneously listed prediction invocation cannot gain target placement.
    scoped
        .speculative_invocations
        .push(eredu_core::speculative::SpeculativeCaptureBinding {
            node_id: "decoder.0".into(),
            scope: SpeculativeCaptureScope::Prediction { depth: 0 },
        });
    let mut observations = SourceMap::new();
    register(&mut observations, &scoped, &parameters, |_| true).unwrap();
    assert!(!observations.contains_key(path));
}

#[test]
fn decoder_source_refuses_each_scratch_node_and_observation_destination() {
    let (descriptor, parameters) = gemma();
    construction::tests::verify(|allocation| {
        let mut result = SourceMap::new();
        worker(&mut result, &descriptor, &parameters, |_| true, allocation)?;
        Ok(result)
    });
}
