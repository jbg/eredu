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
fn retained_inkling_preserves_optional_shared_depths_convolution_and_relative_state() {
    for shared in [false, true] {
        let mut args=ModelArgs::from_hf_json(&serde_json::to_vec(&serde_json::json!({
            "model_type":"inkling_mm_model","text_config":{
                "hidden_size":16,"num_hidden_layers":2,"vocab_size":32,
                "num_attention_heads":2,"num_key_value_heads":2,"head_dim":8,
                "sliding_window_size":4,"layer_types":["sliding_attention","full_attention"],
                "mlp_layer_types":["dense","dense"],"sconv_kernel_size":3,
                "d_rel":4,"rel_extent":8,"intermediate_size":32,"dense_intermediate_size":32,
                "moe_intermediate_size":16,"n_routed_experts":2,"num_experts_per_tok":1,"n_shared_experts":1,"unpadded_vocab_size":32
            },"mtp_config":{"num_nextn_predict_layers":2,"local_layer_ids":[1],"chain_hidden_post_norm":shared}
        })).unwrap()).unwrap();
        let format =
            eredu_checkpoint::WeightQuantization::Affine(eredu_checkpoint::AffineQuantization {
                group_size: 16,
                ..Default::default()
            });
        args.text_config
            .quantized_weight_configs
            .get_or_insert_with(Default::default)
            .insert("model.mtp.layers.0.input_proj.weight".into(), format);
        let ordinary_context = WorkspaceContext::new(Facts);
        let ordinary = MtpModel::<WorkspaceBackend>::new(&args, &ordinary_context)
            .unwrap()
            .unwrap();
        let expected = rows(&ordinary);
        assert_eq!(ordinary.chain_norm.is_some(), shared);
        assert_eq!(ordinary.policy(0), Some(AttentionPolicy::Full));
        assert!(ordinary.policy(1).unwrap().window().is_some());
        assert!(
            expected
                .iter()
                .any(|(name, _, _)| name == "model.mtp.layers.0.input_proj.scales")
        );
        assert!(expected.iter().any(|(name, shape, _)| name
            == "model.mtp.layers.0.transformer_block.self_attn.rel_proj"
            && shape == &[4, 8]));
        assert!(expected.iter().any(|(name, shape, _)| name
            == "model.mtp.layers.1.transformer_block.self_attn.rel_proj"
            && shape == &[4, 4]));
        assert!(expected.iter().any(|(name, shape, _)| name
            == "model.mtp.layers.0.transformer_block.attn_sconv.weight"
            && shape == &[16, 1, 3]));
        let spec = MtpModelSpec::new(&args).unwrap().unwrap();
        let layout = crate::inkling::mtp_state_layout(&args).unwrap().unwrap();
        drop((args, ordinary, ordinary_context));
        let state = Arc::new(State {
            remaining: AtomicUsize::new(usize::MAX),
            retired: AtomicBool::new(false),
        });
        let funding = HostMetadataFunding::new(Account(state.clone())).unwrap();
        let context = WorkspaceContext::new_with_metadata_funding(Facts, funding).unwrap();
        let model = spec.instantiate::<WorkspaceBackend>(&context).unwrap();
        assert_eq!(model.chain_norm.is_some(), shared);
        assert_eq!(rows(&model), expected);
        let copy = layout.clone_workspace(&context).unwrap();
        assert_eq!(copy, layout);
        let retained = Retained {
            value: (model, copy),
            _funding: context.metadata_funding().unwrap(),
        };
        let consumed = context.metadata_census().unwrap().context_bytes();
        state.remaining.store(0, Ordering::SeqCst);
        let refusal = spec
            .depth(1)
            .unwrap()
            .instantiate::<WorkspaceBackend>(&context)
            .err()
            .unwrap();
        assert!(matches!(
            refusal.into_metadata_funding_error(),
            Ok(HostMetadataFundingError::Capacity { .. })
        ));
        assert_eq!(context.metadata_census().unwrap().context_bytes(), consumed);
        drop((spec, context, layout));
        assert!(!state.retired.load(Ordering::SeqCst));
        drop(retained);
        assert!(state.retired.load(Ordering::SeqCst));
    }
}
