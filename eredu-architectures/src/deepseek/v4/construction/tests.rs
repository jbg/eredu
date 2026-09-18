use super::*;
use eredu_nn::{ParameterMetadata, ParameterVisitor, Parameterized, workspace::*};
use std::{
    convert::Infallible,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
};
#[derive(Debug)]
struct Facts;
impl WorkspaceMechanisms for Facts {
    fn operation_bound(
        &self,
        _: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceOperationBound>, Error> {
        Ok(None)
    }
}
impl WorkspaceFactMechanisms for Facts {
    type Error = Infallible;
    fn operation_facts(
        &self,
        _: WorkspaceOperationView<'_>,
    ) -> Result<Option<WorkspaceOperationFacts>, Infallible> {
        Ok(None)
    }
    fn write_operation_facts(
        &self,
        _: WorkspaceOperationView<'_>,
        _: WorkspaceEffectDestination<'_>,
    ) -> Result<Option<WorkspaceOperationFacts>, Infallible> {
        panic!("absent facts")
    }
    fn host_facts(
        &self,
        _: WorkspaceOperationView<'_>,
    ) -> Result<Option<WorkspaceHostFacts>, Infallible> {
        Ok(None)
    }
    fn write_host_facts(
        &self,
        _: WorkspaceOperationView<'_>,
        _: WorkspaceHostDestination<'_>,
    ) -> Result<Option<WorkspaceHostFacts>, Infallible> {
        panic!("absent host facts")
    }
}
#[derive(Debug)]
struct State {
    remaining: AtomicUsize,
    retired: AtomicBool,
}
#[derive(Debug)]
struct Account(Arc<State>);
impl HostMetadataAccount for Account {
    fn reserve_metadata(&self, bytes: usize) -> Result<(), HostMetadataFundingError> {
        self.0
            .remaining
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |left| {
                left.checked_sub(bytes)
            })
            .map(|_| ())
            .map_err(|left| HostMetadataFundingError::Capacity {
                required: bytes as u64,
                available: left as u64,
            })
    }
}
impl Drop for Account {
    fn drop(&mut self) {
        self.0.retired.store(true, Ordering::SeqCst);
    }
}
struct Retained<T> {
    value: T,
    _funding: HostMetadataFunding,
}
fn rows(value: &impl Parameterized<WorkspaceTensor>) -> Vec<(String, Vec<i32>, WorkspaceDtype)> {
    struct Rows(Vec<(String, Vec<i32>, WorkspaceDtype)>);
    impl<'a> ParameterVisitor<'a, WorkspaceTensor> for Rows {
        fn visit(&mut self, metadata: eredu_nn::ParameterMetadataView<'_>, value: &'a WorkspaceTensor) {
            self.0.push((
                metadata.id().as_str().to_owned(),
                value.shape().to_vec(),
                value.layout().dtype(),
            ));
        }
    }
    let mut rows = Rows(Vec::new());
    value.visit_parameters(&mut rows);
    rows.0
}

fn args(fused: bool) -> V4Args {
    let mut value = serde_json::json!({"model_type":"deepseek_v4","hidden_size":16,"moe_intermediate_size":16,
        "num_hidden_layers":3,"num_nextn_predict_layers":3,"num_attention_heads":2,"num_key_value_heads":1,
        "head_dim":8,"qk_rope_head_dim":4,"q_lora_rank":16,"o_lora_rank":8,"o_groups":2,"vocab_size":32,
        "max_position_embeddings":512,"sliding_window":8,"compress_ratios":[0,4,128,0,0,0],
        "index_n_heads":2,"index_head_dim":8,"index_topk":2,"hc_mult":2,"hc_sinkhorn_iters":2,
        "n_routed_experts":4,"n_shared_experts":1,"num_experts_per_tok":2,"num_hash_layers":1,
        "scoring_func":"sqrtsoftplus","topk_method":"noaux_tc","norm_topk_prob":true,"routed_scaling_factor":1.0,"swiglu_limit":4.0});
    if fused {
        value["dspark_block_size"] = serde_json::json!(3);
        value["dspark_noise_token_id"] = serde_json::json!(0);
        value["dspark_target_layer_ids"] = serde_json::json!([2, 0]);
        value["dspark_markov_rank"] = serde_json::json!(8);
    }
    let mut args = crate::deepseek::parse_v4_config(&value).unwrap();
    let format = eredu_checkpoint::LinearFormat::Affine(eredu_checkpoint::AffineQuantization {
        group_size: 16,
        ..Default::default()
    });
    args.linear_formats
        .insert("mtp.0.attn.wq_a.weight".into(), format);
    args.linear_formats
        .insert("mtp.0.ffn.switch_mlp.gate_up_proj".into(), format);
    args
}
fn auxiliary_count(module: &impl Parameterized<WorkspaceTensor>) -> usize {
    struct P(Vec<*const WorkspaceTensor>);
    impl<'a> ParameterVisitor<'a, WorkspaceTensor> for P {
        fn visit(&mut self, _: eredu_nn::ParameterMetadataView<'_>, v: &'a WorkspaceTensor) {
            self.0.push(v);
        }
    }
    let mut parameters = P(Vec::new());
    module.visit_parameters(&mut parameters);
    let mut count = 0;
    assert!(module.visit_retained_values(&mut |v| {
        if !parameters.0.contains(&(v as *const WorkspaceTensor)) {
            assert_eq!(v.shape(), [2]);
            count += 1;
        }
    }));
    count
}
#[test]
fn retained_v4_factories_preserve_pooling_hyper_dspark_and_original_copy_retirement() {
    for fused in [false, true] {
        let args = args(fused);
        let ordinary_context = WorkspaceContext::new(Facts);
        let model = Model::<WorkspaceBackend>::new(args.clone(), &ordinary_context).unwrap();
        let ordinary = (0..3)
            .map(|depth| {
                model
                    .construct_unit(depth + 1, 0, &ordinary_context)
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let expected = ordinary.iter().map(rows).collect::<Vec<_>>();
        assert!(
            expected[0]
                .iter()
                .any(|(name, _, _)| name == "mtp.0.attn.wq_a.scales")
        );
        assert!(
            expected[0]
                .iter()
                .any(|(name, _, _)| name == "mtp.0.hc_attn_fn")
        );
        assert!(
            expected[0]
                .iter()
                .any(|(name, _, _)| name == "mtp.0.ffn.switch_mlp.gate_up_proj_scales")
        );
        // Compression belongs to actual target layers; all prediction layers
        // are local by the shared released configuration contract.
        let ordinary_targets = (0..3)
            .map(|layer| model.construct_unit(0, layer, &ordinary_context).unwrap())
            .collect::<Vec<_>>();
        let expected_targets = ordinary_targets.iter().map(rows).collect::<Vec<_>>();
        assert!(expected_targets[1].iter().any(|(name, shape, _)| name
            == "layers.1.attn.indexer.compressor.ape"
            && shape == &[4, 16]));
        assert!(
            expected_targets[2].iter().any(|(name, shape, _)| name
                == "layers.2.attn.compressor.ape"
                && shape == &[128, 8])
        );
        let target_specs = (0..3)
            .map(|layer| {
                crate::deepseek::block::V4BlockSpec::new(
                    &args,
                    layer,
                    &format!("layers.{layer}"),
                    None,
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let specs = (0..3)
            .map(|depth| model.prediction_unit_spec(depth).unwrap())
            .collect::<Vec<_>>();
        let shared = model.dspark_static_spec().unwrap();
        let shared_expected = model.static_modules().dspark.as_ref().map(rows);
        let strategy = args.target_capture_policy.clone();
        let layout = state_layout(&args).unwrap();
        let policies = (0..6)
            .map(|i| layout.layer(i).unwrap().clone())
            .collect::<Vec<_>>();
        drop((
            args,
            model,
            ordinary,
            ordinary_targets,
            ordinary_context,
            layout,
        ));
        let state = Arc::new(State {
            remaining: AtomicUsize::new(usize::MAX),
            retired: AtomicBool::new(false),
        });
        let funding = HostMetadataFunding::new(Account(state.clone())).unwrap();
        let context = WorkspaceContext::new_with_metadata_funding(Facts, funding).unwrap();
        let units = specs
            .iter()
            .map(|spec| spec.instantiate::<WorkspaceBackend>(&context).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(units.iter().map(rows).collect::<Vec<_>>(), expected);
        assert_eq!(
            units.iter().map(auxiliary_count).collect::<Vec<_>>(),
            vec![1, 1, 1]
        );
        let targets = target_specs
            .iter()
            .map(|spec| spec.instantiate::<WorkspaceBackend>(&context).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(
            targets.iter().map(rows).collect::<Vec<_>>(),
            expected_targets
        );
        assert_eq!(
            targets.iter().map(auxiliary_count).collect::<Vec<_>>(),
            vec![1, 3, 2]
        );
        let shared = shared
            .as_ref()
            .map(|spec| spec.instantiate::<WorkspaceBackend>(&context).unwrap());
        assert_eq!(shared.as_ref().map(rows), shared_expected);
        if let Some(shared) = &shared {
            assert!(
                rows(shared)
                    .iter()
                    .any(|(name, shape, _)| name == "mtp.0.main_proj.weight" && shape == &[16, 32])
            );
        }
        if let Some(shared) = &shared {
            let actual = rows(shared);
            for (name, shape) in [
                ("mtp.2.markov_head.markov_w1.weight", vec![32, 8]),
                ("mtp.2.markov_head.markov_w2.weight", vec![32, 8]),
                ("mtp.2.confidence_head.proj.weight", vec![1, 24]),
            ] {
                assert!(
                    actual
                        .iter()
                        .any(|(id, actual, _)| id == name && actual == &shape)
                );
            }
        }
        let copied = policies
            .iter()
            .map(|p| StateLayout::clone_policy_workspace(p, &context).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(copied, policies);
        // Exact ordered target IDs are paid through the same boxed-policy clone.
        let capture = if let Some(strategy) = &strategy {
            Some(strategy.clone_workspace(&context).unwrap())
        } else {
            None
        };
        if let Some(capture) = &capture {
            assert_eq!(capture.layer_ids(), [2, 0]);
        }
        let retained = Retained {
            value: (units, targets, shared, copied, capture),
            _funding: context.metadata_funding().unwrap(),
        };
        let consumed = context.metadata_census().unwrap().context_bytes();
        assert!(consumed > 0);
        state.remaining.store(0, Ordering::SeqCst);
        assert!(matches!(
            specs[0]
                .instantiate::<WorkspaceBackend>(&context)
                .err()
                .unwrap()
                .into_metadata_funding_error(),
            Ok(HostMetadataFundingError::Capacity { .. })
        ));
        assert_eq!(context.metadata_census().unwrap().context_bytes(), consumed);
        drop((specs, target_specs, strategy, context));
        assert!(!state.retired.load(Ordering::SeqCst));
        drop(retained);
        assert!(state.retired.load(Ordering::SeqCst));
    }
}

#[test]
fn routed_v4_metadata_preserves_pooling_segments_identity_and_group_order() {
    for fused in [false, true] {
        let args = args(fused);
        let expected = state_layout(&args).unwrap();
        let expected_identity = state_identity(&args, &expected, 0, Default::default()).unwrap();
        let expected_graph = super::super::prediction_groups(&args, None).unwrap().execution_graph().unwrap().into_owned();
        let state = Arc::new(State {
            remaining: AtomicUsize::new(usize::MAX),
            retired: AtomicBool::new(false),
        });
        let funding = HostMetadataFunding::new(Account(state.clone())).unwrap();
        let context = WorkspaceContext::new_with_metadata_funding(Facts, funding.clone()).unwrap();
        let before = state.remaining.load(Ordering::SeqCst);
        let actual = super::super::state_layout_with_metadata(&args, &context).unwrap();
        let actual_identity = super::super::state_identity_with_metadata(
            &args, &actual, 0, Default::default(), &context).unwrap();
        let groups = super::super::prediction_groups(&args, Some(&context)).unwrap();
        let declaration_budget = state.remaining.load(Ordering::SeqCst);
        let declaration = groups.execution_graph().unwrap();
        let second_declaration = groups.execution_graph().unwrap();
        assert!(std::ptr::eq(declaration.group_id(0).unwrap(), second_declaration.group_id(0).unwrap()));
        assert_eq!(state.remaining.load(Ordering::SeqCst), declaration_budget);
        let graph = declaration.into_owned_with_metadata(&context).unwrap();
        assert!(state.remaining.load(Ordering::SeqCst) < declaration_budget);
        assert_eq!(actual, expected);
        assert_eq!(actual_identity, expected_identity);
        assert_eq!(graph, expected_graph);
        assert_eq!(groups.unit_path(0, 2, None).unwrap(), "layers.2");
        assert_eq!(groups.unit_path(3, 0, None).unwrap(), "mtp.2");
        assert!(state.remaining.load(Ordering::SeqCst) < before);
        let retained = Retained { value: (actual, actual_identity, groups, graph), _funding: funding };
        drop(context);
        assert!(!state.retired.load(Ordering::SeqCst));
        assert_eq!(retained.value.0, expected);
        drop(retained);
        assert!(state.retired.load(Ordering::SeqCst));
    }
    let state = Arc::new(State {
        remaining: AtomicUsize::new(usize::MAX), retired: AtomicBool::new(false),
    });
    let context = WorkspaceContext::new_with_metadata_funding(Facts,
        HostMetadataFunding::new(Account(state.clone())).unwrap()).unwrap();
    state.remaining.store(0, Ordering::SeqCst);
    assert!(super::super::state_layout_with_metadata(&args(true), &context).is_err());
    assert_eq!(state.remaining.load(Ordering::SeqCst), 0);
}

#[test]
fn retained_v4_target_source_preserves_local_pooling_hyper_and_companions() {
    use crate::routed_text::RoutedConstructionParameters;
    let mut args = args(true).prediction_target().unwrap();
    args.linear_formats.insert("layers.1.ffn.switch_mlp.gate_up_proj".into(),
        eredu_checkpoint::LinearFormat::Affine(eredu_checkpoint::AffineQuantization {
            group_size: 16, ..Default::default()
        }));
    let ordinary = WorkspaceContext::new(Facts);
    let mut model = Model::<WorkspaceBackend>::new(args, &ordinary).unwrap();
    let expected = (0..3).map(|index| rows(&model.construct_unit(0,index,&ordinary).unwrap())).collect::<Vec<_>>();
    assert!(expected[1].iter().any(|(name,_,_)| name == "layers.1.ffn.switch_mlp.gate_up_proj_biases"));
    let source = model.prepare_construction_units(None, &ordinary).unwrap().unwrap();
    model.install_construction_units(Some(source)).unwrap();
    let state = Arc::new(State { remaining: AtomicUsize::new(usize::MAX), retired: AtomicBool::new(false) });
    let context = WorkspaceContext::new_with_metadata_funding(Facts,
        HostMetadataFunding::new(Account(state.clone())).unwrap()).unwrap();
    let units = (0..3).map(|index| model.construct_unit(0,index,&context).unwrap()).collect::<Vec<_>>();
    assert_eq!(units.iter().map(rows).collect::<Vec<_>>(), expected);
    assert_eq!(units.iter().map(auxiliary_count).collect::<Vec<_>>(), vec![1,3,2]);
    let retained = Retained { value: units, _funding: context.metadata_funding().unwrap() };
    drop((model, ordinary, context));
    assert!(!state.retired.load(Ordering::SeqCst));
    assert_eq!(retained.value.iter().map(rows).collect::<Vec<_>>(), expected);
    drop(retained);
    assert!(state.retired.load(Ordering::SeqCst));
}
