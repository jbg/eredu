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
fn retained_composite_inkling_reuses_static_media_and_mtp_declarations_with_paid_failure_custody() {
    let mut args = ModelArgs::from_hf_json(&serde_json::to_vec(&serde_json::json!({
        "model_type":"inkling_mm_model", "image_token_id":60, "audio_token_id":61,
        "text_config": {"hidden_size":16,"num_hidden_layers":2,"vocab_size":64,
            "num_attention_heads":2,"num_key_value_heads":2,"head_dim":8,
            "sliding_window_size":4,"layer_types":["sliding_attention","full_attention"],
            "mlp_layer_types":["dense","dense"],"sconv_kernel_size":3,"d_rel":4,"rel_extent":8,
            "intermediate_size":32,"dense_intermediate_size":32,"moe_intermediate_size":16,
            "n_routed_experts":2,"num_experts_per_tok":1,"n_shared_experts":1,"unpadded_vocab_size":64},
        "mtp_config":{"num_nextn_predict_layers":2,"local_layer_ids":[1],"chain_hidden_post_norm":true},
        "audio_config":{"text_hidden_size":16,"num_codebooks":4,"codebook_size":8},
        "vision_config":{"text_hidden_size":16,"patch_size":40,"temporal_patch_size":2,"num_channels":3,"num_hidden_layers":4}
    })).unwrap()).unwrap();
    let format =
        eredu_checkpoint::WeightQuantization::Affine(eredu_checkpoint::AffineQuantization {
            group_size: 16,
            ..Default::default()
        });
    let formats = args
        .text_config
        .quantized_weight_configs
        .get_or_insert_with(Default::default);
    formats.insert("model.mtp.layers.0.input_proj.weight".into(), format);
    formats.insert("lm_head.weight".into(), format);
    let ordinary_context = WorkspaceContext::new(Facts);
    let ordinary = LayeredModel::<WorkspaceBackend>::new(args, &ordinary_context).unwrap();
    let expected = rows(&ordinary.static_modules);
    let fingerprint = ordinary.args().architecture_fingerprint();
    let source = ordinary.construction_source().clone();
    assert!(expected.iter().any(|(name, _, _)| name == "lm_head.scales"));
    assert!(
        expected
            .iter()
            .any(|(name, _, _)| name == "audio.encoder.weight")
    );
    assert!(
        expected
            .iter()
            .any(|(name, _, _)| name == "visual.final_norm.weight")
    );
    assert!(
        expected
            .iter()
            .any(|(name, _, _)| name == "model.mtp.chain_norm.weight")
    );
    assert!(
        expected
            .iter()
            .any(|(name, _, _)| name == "model.mtp.layers.0.input_proj.scales")
    );
    drop((ordinary, ordinary_context));
    let state = Arc::new(State {
        remaining: AtomicUsize::new(usize::MAX),
        retired: AtomicBool::new(false),
    });
    let funding = HostMetadataFunding::new(Account(state.clone())).unwrap();
    let context = WorkspaceContext::new_with_metadata_funding(Facts, funding).unwrap();
    let first =
        LayeredModel::<WorkspaceBackend>::new_with_source(source.clone(), &context).unwrap();
    let second =
        LayeredModel::<WorkspaceBackend>::new_with_source(source.clone(), &context).unwrap();
    assert_eq!(rows(&first.static_modules), expected);
    assert_eq!(rows(&second.static_modules), expected);
    assert!(std::ptr::eq(first.args(), second.args()));
    assert_eq!(
        first
            .args()
            .architecture_fingerprint_with_metadata(crate::decoder::identity::Metadata::new(Some(
                &context
            )))
            .unwrap(),
        fingerprint
    );
    let retained = Retained {
        value: (first, second),
        _funding: context.metadata_funding().unwrap(),
    };
    let consumed = context.metadata_census().unwrap().context_bytes();
    state.remaining.store(0, Ordering::SeqCst);
    let refusal = LayeredModel::<WorkspaceBackend>::new_with_source(source.clone(), &context)
        .err()
        .unwrap();
    assert!(matches!(
        refusal.into_metadata_funding_error(),
        Ok(HostMetadataFundingError::Capacity { .. })
    ));
    assert_eq!(context.metadata_census().unwrap().context_bytes(), consumed);
    drop((source, context));
    assert!(!state.retired.load(Ordering::SeqCst));
    drop(retained);
    assert!(state.retired.load(Ordering::SeqCst));
}
