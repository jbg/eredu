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
fn retained_prediction_specs_preserve_companions_and_refuse_before_an_unfunded_copy() {
    let mut args=crate::deepseek::parse_v3_config(&serde_json::json!({
  "hidden_size":16,"intermediate_size":32,"moe_intermediate_size":16,"num_hidden_layers":2,"num_attention_heads":2,"vocab_size":32,
  "max_position_embeddings":64,"kv_lora_rank":4,"qk_nope_head_dim":2,"qk_rope_head_dim":2,"v_head_dim":2,"first_k_dense_replace":1,
  "q_lora_rank":4,"n_routed_experts":4,"n_shared_experts":1,"num_experts_per_tok":2,"n_group":2,"topk_group":1,"num_nextn_predict_layers":1,"tie_word_embeddings":false
 })).unwrap();
    let format = eredu_checkpoint::LinearFormat::Affine(eredu_checkpoint::AffineQuantization {
        group_size: 16,
        ..Default::default()
    });
    args.linear_formats
        .insert("model.layers.2.eh_proj.weight".into(), format);
    args.linear_formats
        .insert("model.layers.2.mlp.experts.gate_up_proj".into(), format);
    let ordinary_context = WorkspaceContext::new(Facts);
    let ordinary = V3PredictionLayer::<WorkspaceBackend>::new(&args, 0, &ordinary_context).unwrap();
    let expected = rows(&ordinary);
    assert!(
        expected
            .iter()
            .any(|(name, _, _)| name == "model.layers.2.eh_proj.scales")
    );
    assert!(
        expected
            .iter()
            .any(|(name, _, _)| name == "model.layers.2.mlp.experts.gate_up_proj_biases")
    );
    let spec = V3PredictionLayerSpec::new(&args, 0).unwrap();
    drop((args, ordinary, ordinary_context));
    let state = Arc::new(State {
        remaining: AtomicUsize::new(usize::MAX),
        retired: AtomicBool::new(false),
    });
    let funding = HostMetadataFunding::new(Account(state.clone())).unwrap();
    let context = WorkspaceContext::new_with_metadata_funding(Facts, funding).unwrap();
    let value = spec.instantiate::<WorkspaceBackend>(&context).unwrap();
    let retained = Retained {
        value,
        _funding: context.metadata_funding().unwrap(),
    };
    assert_eq!(rows(&retained.value), expected);
    let consumed = context.metadata_census().unwrap().context_bytes();
    assert!(consumed > 0);
    state.remaining.store(0, Ordering::SeqCst);
    let error = spec
        .instantiate::<WorkspaceBackend>(&context)
        .err()
        .unwrap();
    assert!(matches!(
        error.into_metadata_funding_error(),
        Ok(HostMetadataFundingError::Capacity { .. })
    ));
    assert_eq!(context.metadata_census().unwrap().context_bytes(), consumed);
    assert_eq!(rows(&retained.value), expected);
    drop((spec, context));
    assert!(!state.retired.load(Ordering::SeqCst));
    drop(retained);
    assert!(state.retired.load(Ordering::SeqCst));
}

#[test]
fn retained_v3_target_source_preserves_dense_routed_companions_and_paid_outputs() {
    use crate::routed_text::RoutedConstructionParameters;
    let mut args = crate::deepseek::parse_v3_config(&serde_json::json!({
        "hidden_size":16,"intermediate_size":32,"moe_intermediate_size":16,"num_hidden_layers":2,"num_attention_heads":2,"vocab_size":32,
        "max_position_embeddings":64,"kv_lora_rank":4,"qk_nope_head_dim":2,"qk_rope_head_dim":2,"v_head_dim":2,"first_k_dense_replace":1,
        "q_lora_rank":4,"n_routed_experts":4,"n_shared_experts":1,"num_experts_per_tok":2,"n_group":2,"topk_group":1,"num_nextn_predict_layers":0,"tie_word_embeddings":false
    })).unwrap();
    let format = eredu_checkpoint::LinearFormat::Affine(eredu_checkpoint::AffineQuantization {
        group_size: 16, ..Default::default()
    });
    args.linear_formats.insert("model.layers.0.mlp.gate_proj.weight".into(), format);
    args.linear_formats.insert("model.layers.1.mlp.experts.gate_up_proj".into(), format);
    let ordinary = WorkspaceContext::new(Facts);
    let mut model = crate::deepseek::v3::Model::<WorkspaceBackend>::new(args, &ordinary).unwrap();
    let expected = (0..2).map(|index| rows(&model.construct_unit(0, index, &ordinary).unwrap())).collect::<Vec<_>>();
    assert!(expected[0].iter().any(|(name,_,_)| name == "model.layers.0.mlp.gate_proj.scales"));
    assert!(expected[1].iter().any(|(name,_,_)| name == "model.layers.1.mlp.experts.gate_up_proj_biases"));
    let source = model.prepare_construction_units(None, &ordinary).unwrap().unwrap();
    assert!(model.install_construction_units(Some(crate::routed_text::RetainedRoutedUnits::v3_source(Vec::new()))).is_err());
    model.install_construction_units(Some(source)).unwrap();
    let state = Arc::new(State { remaining: AtomicUsize::new(usize::MAX), retired: AtomicBool::new(false) });
    let context = WorkspaceContext::new_with_metadata_funding(Facts,
        HostMetadataFunding::new(Account(state.clone())).unwrap()).unwrap();
    let units = (0..2).map(|index| model.construct_unit(0,index,&context).unwrap()).collect::<Vec<_>>();
    assert_eq!(units.iter().map(rows).collect::<Vec<_>>(), expected);
    let consumed = context.metadata_census().unwrap().context_bytes();
    state.remaining.store(0, Ordering::SeqCst);
    assert!(model.construct_unit(0,0,&context).is_err());
    assert_eq!(context.metadata_census().unwrap().context_bytes(), consumed);
    let retained = Retained { value: units, _funding: context.metadata_funding().unwrap() };
    drop((model, ordinary, context));
    assert!(!state.retired.load(Ordering::SeqCst));
    assert_eq!(retained.value.iter().map(rows).collect::<Vec<_>>(), expected);
    drop(retained);
    assert!(state.retired.load(Ordering::SeqCst));
}
