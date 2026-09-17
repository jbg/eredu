use super::*;
use eredu_nn::{workspace::*, ParameterMetadata, ParameterVisitor, Parameterized};
use std::{
    convert::Infallible,
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc,
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
impl WorkspaceMetadataAccount for Account {
    fn reserve_metadata(&self, bytes: usize) -> Result<(), WorkspaceMetadataFundingError> {
        self.0
            .remaining
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |left| {
                left.checked_sub(bytes)
            })
            .map(|_| ())
            .map_err(|left| WorkspaceMetadataFundingError::Capacity {
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
    _funding: WorkspaceMetadataFunding,
}
fn rows(value: &impl Parameterized<WorkspaceTensor>) -> Vec<(String, Vec<i32>, WorkspaceDtype)> {
    struct Rows(Vec<(String, Vec<i32>, WorkspaceDtype)>);
    impl<'a> ParameterVisitor<'a, WorkspaceTensor> for Rows {
        fn visit(&mut self, metadata: ParameterMetadata, value: &'a WorkspaceTensor) {
            self.0.push((
                metadata.id.as_str().to_owned(),
                value.shape().to_vec(),
                value.layout().dtype(),
            ));
        }
    }
    let mut rows = Rows(Vec::new());
    value.visit_parameters(&mut rows);
    rows.0
}

#[test]
fn retained_nemotron_keeps_physical_groups_companions_and_state_custody() {
    for pattern in ["*E", "E*"] {
        let mut args = crate::nemotron_h::model_args_from_config_value(&serde_json::json!({
            "model_type":"nemotron_h","vocab_size":32,"hidden_size":16,"intermediate_size":32,
            "num_hidden_layers":1,"hybrid_override_pattern":"*","num_attention_heads":2,
            "num_key_value_heads":2,"head_dim":8,"mamba_num_heads":2,"n_groups":2,
            "mamba_head_dim":8,"ssm_state_size":2,"conv_kernel":3,"chunk_size":2,
            "n_routed_experts":4,"n_shared_experts":1,"moe_intermediate_size":32,
            "moe_shared_expert_intermediate_size":32,"num_experts_per_tok":2,"n_group":2,
            "topk_group":1,"num_nextn_predict_layers":2,"mtp_hybrid_override_pattern":pattern,
            "tie_word_embeddings":false,"mlp_bias":true,"attention_bias":true
        }))
        .unwrap();
        let policies = args.mtp_policies().unwrap();
        let mut formats = std::collections::HashMap::new();
        let affine =
            eredu_checkpoint::WeightQuantization::Affine(eredu_checkpoint::AffineQuantization {
                group_size: 16,
                ..Default::default()
            });
        for (physical, policy) in policies.iter().enumerate() {
            let name = match policy {
                LayerPolicy::SelfAttention(_) => {
                    format!("model.mtp.layers.{physical}.mixer.q_proj.weight")
                }
                LayerPolicy::SparseMoe => {
                    format!("model.mtp.layers.{physical}.mixer.experts.up_proj")
                }
                _ => unreachable!(),
            };
            formats.insert(name, affine);
        }
        args.quantized_weight_configs = Some(formats);
        let specs = policies
            .iter()
            .enumerate()
            .map(|(physical, policy)| {
                let geometry = match policy {
                    LayerPolicy::SelfAttention(_) => LayerGeometry::Attention {
                        query_heads: 1,
                        kv_heads: 1,
                    },
                    LayerPolicy::SparseMoe => LayerGeometry::SparseMoe {
                        routed: 16,
                        shared: 16,
                    },
                    _ => unreachable!(),
                };
                PredictionUnitSpec::with_geometry(
                    &args,
                    physical / 2,
                    physical % 2,
                    *policy,
                    geometry,
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let state_layout = crate::nemotron_h::state_layout(&args)
            .unwrap()
            .slice(1..5)
            .unwrap();
        drop(args);
        let account = Arc::new(State {
            remaining: AtomicUsize::new(usize::MAX),
            retired: AtomicBool::new(false),
        });
        let funding = WorkspaceMetadataFunding::new(Account(account.clone())).unwrap();
        let context = WorkspaceContext::new_with_metadata_funding(Facts, funding).unwrap();
        let units = specs
            .iter()
            .map(|spec| spec.instantiate::<WorkspaceBackend>(&context).unwrap())
            .collect::<Vec<_>>();
        for (physical, unit) in units.iter().enumerate() {
            assert_eq!(unit.embedding_norm.is_some(), physical % 2 == 0);
            assert_eq!(unit.hidden_norm.is_some(), physical % 2 == 0);
            assert_eq!(unit.fusion.is_some(), physical % 2 == 0);
            assert_eq!(unit.final_norm.is_some(), physical % 2 == 1);
            let parameters = rows(unit);
            let root = format!("model.mtp.layers.{physical}.");
            assert!(parameters
                .iter()
                .all(|(name, _, _)| name.starts_with(&root)));
            match &unit.block.operator {
                crate::nemotron_h::Operator::Attention(_) => {
                    assert!(parameters.iter().any(|(name, shape, _)| name
                        == &format!("{root}mixer.k_proj.weight")
                        && shape == &[8, 16]));
                    assert!(parameters
                        .iter()
                        .any(|(name, _, _)| name == &format!("{root}mixer.q_proj.scales")));
                }
                crate::nemotron_h::Operator::Sparse(_) => {
                    assert!(parameters.iter().any(|(name, shape, _)| name
                        == &format!("{root}mixer.shared_experts.up_proj.weight")
                        && shape == &[16, 16]));
                    assert!(
                        parameters
                            .iter()
                            .any(|(name, _, _)| name
                                == &format!("{root}mixer.experts.up_proj_biases"))
                    );
                    assert!(parameters
                        .iter()
                        .any(|(name, _, _)| name
                            == &format!("{root}mixer.gate.e_score_correction_bias")));
                }
                _ => panic!("prediction operator changed"),
            }
        }
        let copied = state_layout.clone_workspace(&context).unwrap();
        assert_eq!(copied, state_layout);
        assert_eq!(copied.len(), 4);
        let retained = Retained {
            value: (units, copied),
            _funding: context.metadata_funding().unwrap(),
        };
        let consumed = context.metadata_census().unwrap().context_bytes();
        account.remaining.store(0, Ordering::SeqCst);
        let refusal = specs[0]
            .instantiate::<WorkspaceBackend>(&context)
            .err()
            .unwrap();
        assert!(matches!(
            refusal.into_metadata_funding_error(),
            Ok(WorkspaceMetadataFundingError::Capacity { .. })
        ));
        assert_eq!(context.metadata_census().unwrap().context_bytes(), consumed);
        assert_eq!(retained.value.0.len(), 4);
        drop((context, specs, state_layout));
        assert!(!account.retired.load(Ordering::SeqCst));
        drop(retained);
        assert!(account.retired.load(Ordering::SeqCst));
    }
}

#[test]
fn retained_nemotron_target_and_fixed_sources_preserve_all_operators_and_paid_outputs() {
    use crate::routed_text::RoutedConstructionParameters;
    let mut args = crate::nemotron_h::model_args_from_config_value(&serde_json::json!({
        "model_type":"nemotron_h","vocab_size":32,"hidden_size":16,"intermediate_size":32,
        "num_hidden_layers":4,"hybrid_override_pattern":"M*-E","num_attention_heads":2,
        "num_key_value_heads":2,"head_dim":8,"mamba_num_heads":2,"n_groups":2,
        "mamba_head_dim":8,"ssm_state_size":2,"conv_kernel":3,"chunk_size":2,
        "n_routed_experts":4,"n_shared_experts":1,"moe_intermediate_size":32,
        "moe_shared_expert_intermediate_size":32,"num_experts_per_tok":2,"n_group":2,
        "topk_group":1,"num_nextn_predict_layers":0,"tie_word_embeddings":false,
        "mlp_bias":true,"attention_bias":true,"use_bias":true,"use_conv_bias":true
    })).unwrap();
    let affine = eredu_checkpoint::WeightQuantization::Affine(eredu_checkpoint::AffineQuantization {
        group_size: 16, ..Default::default()
    });
    args.quantized_weight_configs = Some([
        ("model.layers.0.mamba.in_proj.weight".into(), affine),
        ("model.layers.1.attention.q_proj.weight".into(), affine),
        ("model.layers.2.mlp.up_proj.weight".into(), affine),
        ("model.layers.3.moe.experts.up_proj".into(), affine),
    ].into_iter().collect());
    let ordinary_context = WorkspaceContext::new(Facts);
    let mut model = crate::nemotron_h::LayeredModel::<WorkspaceBackend>::new(args, &ordinary_context).unwrap();
    let expected = [0, 1, 2, 3].map(|index| rows(&model.construct_unit(0, index, &ordinary_context).unwrap()));
    for (index, name) in ["model.layers.0.mamba.in_proj.scales", "model.layers.1.attention.q_proj.scales", "model.layers.2.mlp.up_proj.biases", "model.layers.3.moe.experts.up_proj_biases"].into_iter().enumerate() {
        assert!(expected[index].iter().any(|(actual, _, _)| actual == name));
    }
    assert!(expected[0].iter().any(|(name, shape, _)| name == "model.layers.0.mamba.conv1d.weight" && shape == &[24, 1, 3]));
    let source = model.prepare_construction_units(None, &ordinary_context).unwrap();
    assert!(model.install_construction_units(Some(crate::routed_text::RetainedRoutedUnits::nemotron_source(Vec::new()))).is_err());
    model.install_construction_units(source).unwrap();
    let account = Arc::new(State { remaining: AtomicUsize::new(usize::MAX), retired: AtomicBool::new(false) });
    let funding = WorkspaceMetadataFunding::new(Account(account.clone())).unwrap();
    let context = WorkspaceContext::new_with_metadata_funding(Facts, funding).unwrap();
    let units = [0, 1, 2, 3].map(|index| model.construct_unit(0, index, &context).unwrap());
    for (index, unit) in units.iter().enumerate() { assert_eq!(rows(unit), expected[index]); }
    // Fixed non-routed construction reaches the same three paid leaf builders.
    let fixed = [0, 1, 2].map(|index| crate::nemotron_h::block::ReplicatedBlock::<WorkspaceBackend>::new(model.args(), index, &context).unwrap());
    for (index, unit) in fixed.iter().enumerate() { assert_eq!(rows(unit), expected[index]); }
    let retained = Retained { value: (units, fixed), _funding: context.metadata_funding().unwrap() };
    let consumed = context.metadata_census().unwrap().context_bytes();
    account.remaining.store(0, Ordering::SeqCst);
    let refusal = model.construct_unit(0, 0, &context).err().unwrap();
    assert!(matches!(refusal.into_metadata_funding_error(), Ok(WorkspaceMetadataFundingError::Capacity { .. })));
    let fixed_refusal = crate::nemotron_h::block::ReplicatedBlock::<WorkspaceBackend>::new(model.args(), 0, &context).err().unwrap();
    assert!(matches!(fixed_refusal.into_metadata_funding_error(), Ok(WorkspaceMetadataFundingError::Capacity { .. })));
    assert_eq!(context.metadata_census().unwrap().context_bytes(), consumed);
    drop((model, context, ordinary_context));
    assert_eq!(rows(&retained.value.0[0]), expected[0]);
    assert!(!account.retired.load(Ordering::SeqCst));
    drop(retained);
    assert!(account.retired.load(Ordering::SeqCst));
}
