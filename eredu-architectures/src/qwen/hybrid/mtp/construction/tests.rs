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
#[test]
fn retained_qwen_units_share_names_and_copy_exact_state_before_refusal() {
    for routed in [false, true] {
        let mut config=crate::qwen::hybrid::model_args_from_config_value(&serde_json::json!({
            "model_type":if routed {"qwen3_next"} else {"qwen3_5_text"},
            "vocab_size":32,"hidden_size":16,"num_hidden_layers":2,"mtp_num_hidden_layers":2,
            "num_attention_heads":2,"num_key_value_heads":1,"head_dim":8,"max_position_embeddings":64,
            "intermediate_size":32,"moe_intermediate_size":16,"shared_expert_intermediate_size":16,
            "num_experts":if routed {4} else {0},"num_experts_per_tok":if routed {2} else {0},
            "tie_word_embeddings":true,"layer_types":["full_attention","full_attention"]
        })).unwrap().text;
        let affine = eredu_checkpoint::LinearFormat::Affine(eredu_checkpoint::AffineQuantization {
            group_size: 16,
            ..Default::default()
        });
        config
            .linear_formats
            .insert("mtp.layers.0.self_attn.q_proj.weight".into(), affine);
        let companion = if routed {
            "mtp.layers.0.mlp.experts.gate_up_proj_biases"
        } else {
            "mtp.layers.0.mlp.gate_proj.biases"
        };
        config.linear_formats.insert(
            if routed {
                "mtp.layers.0.mlp.experts.gate_up_proj"
            } else {
                "mtp.layers.0.mlp.gate_proj.weight"
            }
            .into(),
            affine,
        );
        let ordinary_context = WorkspaceContext::new(Facts);
        let ordinary =
            PredictionUnit::<WorkspaceBackend>::new(&config, 0, &ordinary_context).unwrap();
        let expected = rows(&ordinary);
        assert!(
            expected
                .iter()
                .any(|(name, _, _)| name == "mtp.layers.0.self_attn.q_proj.scales")
        );
        assert!(expected.iter().any(|(name, _, _)| name == companion));
        let specs = [
            PredictionUnitSpec::new(&config, 0).unwrap(),
            PredictionUnitSpec::new(&config, 1).unwrap(),
        ];
        let shared_spec = PredictionSharedSpec::new(config.hidden_size, config.rms_norm_eps);
        let shared_expected =
            rows(&PredictionShared::<WorkspaceBackend>::new(&config, &ordinary_context).unwrap());
        let state_layout = crate::qwen::hybrid::state_layout(&config)
            .unwrap()
            .slice(2..4)
            .unwrap();
        drop((config, ordinary, ordinary_context));
        let state = Arc::new(State {
            remaining: AtomicUsize::new(usize::MAX),
            retired: AtomicBool::new(false),
        });
        let funding = HostMetadataFunding::new(Account(state.clone())).unwrap();
        let context = WorkspaceContext::new_with_metadata_funding(Facts, funding).unwrap();
        let first = specs[0].instantiate::<WorkspaceBackend>(&context).unwrap();
        let second = specs[1].instantiate::<WorkspaceBackend>(&context).unwrap();
        let shared = shared_spec
            .instantiate::<WorkspaceBackend>(&context)
            .unwrap();
        let layout = state_layout.clone_workspace(&context).unwrap();
        assert_eq!(rows(&first), expected);
        assert!(
            rows(&second)
                .iter()
                .all(|(name, _, _)| name.starts_with("mtp.layers.1."))
        );
        assert_eq!(rows(&shared), shared_expected);
        assert_eq!(layout, state_layout);
        let retained = Retained {
            value: (first, second, shared, layout),
            _funding: context.metadata_funding().unwrap(),
        };
        let consumed = context.metadata_census().unwrap().context_bytes();
        state.remaining.store(0, Ordering::SeqCst);
        let refusal = specs[0]
            .instantiate::<WorkspaceBackend>(&context)
            .err()
            .unwrap();
        assert!(matches!(
            refusal.into_metadata_funding_error(),
            Ok(HostMetadataFundingError::Capacity { .. })
        ));
        assert_eq!(context.metadata_census().unwrap().context_bytes(), consumed);
        assert_eq!(rows(&retained.value.0), expected);
        drop((context, specs, state_layout));
        assert!(!state.retired.load(Ordering::SeqCst));
        drop(retained);
        assert!(state.retired.load(Ordering::SeqCst));
    }
}

#[test]
fn retained_qwen_target_source_preserves_mixed_mixers_companions_and_paid_outputs() {
    use crate::routed_text::RoutedConstructionParameters;
    for routed in [false, true] {
        let mut config = crate::qwen::hybrid::model_args_from_config_value(&serde_json::json!({
            "model_type": if routed { "qwen3_next" } else { "qwen3_5_text" },
            "vocab_size":32,"hidden_size":16,"num_hidden_layers":2,"mtp_num_hidden_layers":0,
            "num_attention_heads":2,"num_key_value_heads":1,"head_dim":8,"max_position_embeddings":64,
            "linear_num_key_heads":1,"linear_num_value_heads":2,
            "linear_key_head_dim":8,"linear_value_head_dim":8,"linear_conv_kernel_dim":4,
            "intermediate_size":32,"moe_intermediate_size":16,"shared_expert_intermediate_size":16,
            "num_experts":if routed {4} else {0},"num_experts_per_tok":if routed {2} else {0},
            "tie_word_embeddings":true,"layer_types":["linear_attention","full_attention"]
        })).unwrap().text;
        let affine = eredu_checkpoint::LinearFormat::Affine(eredu_checkpoint::AffineQuantization {
            group_size: 16, ..Default::default()
        });
        config.linear_formats.insert("model.layers.1.self_attn.q_proj.weight".into(), affine);
        let (projection, companion) = if routed {
            ("model.layers.0.mlp.experts.gate_up_proj", "model.layers.0.mlp.experts.gate_up_proj_biases")
        } else {
            ("model.layers.0.mlp.gate_proj.weight", "model.layers.0.mlp.gate_proj.biases")
        };
        config.linear_formats.insert(projection.into(), affine);
        let ordinary_context = WorkspaceContext::new(Facts);
        let mut model = crate::qwen::hybrid::LayeredModel::<WorkspaceBackend>::new(config, &ordinary_context).unwrap();
        let expected = [0, 1].map(|index| rows(&model.construct_unit(0, index, &ordinary_context).unwrap()));
        assert!(expected[0].iter().any(|(name, _, _)| name == companion));
        assert!(expected[0].iter().any(|(name, _, _)| name == "model.layers.0.linear_attn.conv1d.weight"));
        assert!(expected[1].iter().any(|(name, _, _)| name == "model.layers.1.self_attn.q_proj.scales"));
        let source = model.prepare_construction_units(None, &ordinary_context).unwrap();
        assert!(model.install_construction_units(Some(crate::routed_text::RetainedRoutedUnits::qwen_source(Vec::new()))).is_err());
        model.install_construction_units(source).unwrap();
        let state = Arc::new(State { remaining: AtomicUsize::new(usize::MAX), retired: AtomicBool::new(false) });
        let funding = HostMetadataFunding::new(Account(state.clone())).unwrap();
        let context = WorkspaceContext::new_with_metadata_funding(Facts, funding).unwrap();
        let first = model.construct_unit(0, 0, &context).unwrap();
        let second = model.construct_unit(0, 1, &context).unwrap();
        assert_eq!(rows(&first), expected[0]);
        assert_eq!(rows(&second), expected[1]);
        let retained = Retained { value: (first, second), _funding: context.metadata_funding().unwrap() };
        let consumed = context.metadata_census().unwrap().context_bytes();
        state.remaining.store(0, Ordering::SeqCst);
        let refusal = model.construct_unit(0, 0, &context).err().unwrap();
        assert!(matches!(refusal.into_metadata_funding_error(), Ok(HostMetadataFundingError::Capacity { .. })));
        assert_eq!(context.metadata_census().unwrap().context_bytes(), consumed);
        drop((model, context, ordinary_context));
        assert_eq!(rows(&retained.value.0), expected[0]);
        assert!(!state.retired.load(Ordering::SeqCst));
        drop(retained);
        assert!(state.retired.load(Ordering::SeqCst));
    }
}
