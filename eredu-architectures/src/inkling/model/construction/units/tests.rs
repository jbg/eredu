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
    _value: T,
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
type Model = LayeredModel<WorkspaceBackend>;
type StateType = eredu_runtime::DeviceState<
    WorkspaceBackend,
    eredu_runtime::working_memory::WorkspaceResidentLayerState,
>;
fn unit(
    model: &Model,
    group: usize,
    index: usize,
    context: &WorkspaceContext,
) -> Result<Unit<WorkspaceBackend>, Error> {
    <Model as LayeredArchitecture<WorkspaceBackend, StateType>>::build_unit(
        model, group, index, context,
    )
}
#[test]
fn retained_inkling_graph_units_preserve_sparse_shared_media_state_and_source_retirement() {
    let mut args = ModelArgs::from_hf_json(&serde_json::to_vec(&serde_json::json!({
        "model_type":"inkling_mm_model","image_token_id":60,"audio_token_id":61,
        "text_config":{"hidden_size":16,"num_hidden_layers":2,"vocab_size":64,
            "num_attention_heads":2,"num_key_value_heads":2,"head_dim":8,
            "sliding_window_size":4,"layer_types":["sliding_attention","full_attention"],
            "mlp_layer_types":["dense","moe"],"sconv_kernel_size":3,"d_rel":4,"rel_extent":8,
            "intermediate_size":32,"dense_intermediate_size":32,"moe_intermediate_size":16,
            "n_routed_experts":4,"num_experts_per_tok":2,"n_shared_experts":1,"unpadded_vocab_size":64},
        "mtp_config":{"num_nextn_predict_layers":2,"local_layer_ids":[1],"chain_hidden_post_norm":true},
        "audio_config":{"text_hidden_size":16,"num_codebooks":4,"codebook_size":8},
        "vision_config":{"text_hidden_size":16,"patch_size":40,"temporal_patch_size":2,"num_channels":3,"num_hidden_layers":4}
    })).unwrap()).unwrap();
    args.text_config
        .quantized_weight_configs
        .get_or_insert_with(Default::default)
        .insert(
            "model.layers.0.self_attn.q_proj.weight".into(),
            eredu_checkpoint::WeightQuantization::Affine(eredu_checkpoint::AffineQuantization {
                group_size: 16,
                ..Default::default()
            }),
        );
    let unrelated = args.clone();
    let ordinary_context = WorkspaceContext::new(Facts);
    let mut ordinary = Model::new(args, &ordinary_context).unwrap();
    let coordinates = [(0, 0), (0, 1), (0, 2), (0, 3), (2, 0), (2, 1)];
    let expected = coordinates
        .map(|(group, index)| rows(&unit(&ordinary, group, index, &ordinary_context).unwrap()));
    let layout = eredu_runtime::ArchitectureParameters::state_layout(&ordinary).unwrap();
    let state_geometry = eredu_runtime::PartitionState::new(layout.clone(), 0).unwrap();
    let identity = eredu_runtime::ArchitectureParameters::state_identity(
        &ordinary,
        &state_geometry,
        PromptCacheTopology::default(),
    )
    .unwrap();
    let graph =
        <Model as LayeredArchitecture<WorkspaceBackend, StateType>>::execution_graph(&ordinary)
            .unwrap();
    let transports = std::array::from_fn::<_, 3, _>(|group| {
        <Model as LayeredArchitecture<WorkspaceBackend, StateType>>::group_transport(
            &ordinary, group,
        )
    });
    ordinary
        .prepare_graph_units(None, &ordinary_context)
        .unwrap();
    let source = ordinary.construction_source().clone();
    assert!(source.has_units());
    assert!(
        expected[4]
            .iter()
            .any(|(name, _, _)| name == "model.layers.0.self_attn.q_proj.scales")
    );
    assert!(
        expected[5].iter().any(
            |(name, shape, _)| name == "model.layers.1.moe.router.weight" && shape == &[5, 16]
        )
    );
    assert_eq!(
        source
            .units()
            .unwrap()
            .layouts
            .prediction()
            .unwrap()
            .global_layer_offset(),
        2
    );
    let wrong = Model::new(unrelated, &ordinary_context).unwrap();
    drop((ordinary, ordinary_context));
    let state = Arc::new(State {
        remaining: AtomicUsize::new(usize::MAX),
        retired: AtomicBool::new(false),
    });
    let funding = WorkspaceMetadataFunding::new(Account(state.clone())).unwrap();
    let context = WorkspaceContext::new_with_metadata_funding(Facts, funding).unwrap();
    assert!(
        source
            .units()
            .unwrap()
            .validate(
                &wrong,
                crate::decoder::identity::Metadata::new(Some(&context))
            )
            .is_err()
    );
    drop(wrong);
    let first = Model::new_with_source(source.clone(), &context).unwrap();
    let second = Model::new_with_source(source.clone(), &context).unwrap();
    assert_eq!(
        eredu_runtime::ArchitectureParameters::state_layout_with_metadata(&first, &context)
            .unwrap(),
        layout
    );
    assert!(
        <Model as LayeredArchitecture<WorkspaceBackend, StateType>>::execution_graph_with_metadata(
            &first, &context
        )
        .unwrap()
        .matches(&graph)
    );
    for (group, transport) in transports.iter().enumerate() {
        assert!(<Model as LayeredArchitecture<
            WorkspaceBackend,
            StateType,
        >>::group_transport_matches(&first, group, transport));
    }
    assert_eq!(
        eredu_runtime::ArchitectureParameters::state_identity_with_metadata(
            &first,
            &state_geometry,
            PromptCacheTopology::default(),
            &context
        )
        .unwrap(),
        identity
    );
    let description = eredu_runtime::ArchitectureParameters::parameter_description_with_metadata(
        &first, &context,
    )
    .unwrap();
    assert!(matches!(description, std::borrow::Cow::Borrowed(_)));
    let outputs = coordinates.map(|(group, index)| unit(&first, group, index, &context).unwrap());
    for (index, (group, position)) in coordinates.into_iter().enumerate() {
        assert_eq!(rows(&outputs[index]), expected[index]);
        assert_eq!(
            rows(&unit(&second, group, position, &context).unwrap()),
            expected[index]
        );
    }
    drop(description);
    let retained = Retained {
        _value: outputs,
        _funding: context.metadata_funding().unwrap(),
    };
    let consumed = context.metadata_census().unwrap().context_bytes();
    state.remaining.store(0, Ordering::SeqCst);
    let refusal = unit(&first, 2, 1, &context).err().unwrap();
    assert!(matches!(
        refusal.into_metadata_funding_error(),
        Ok(WorkspaceMetadataFundingError::Capacity { .. })
    ));
    assert_eq!(context.metadata_census().unwrap().context_bytes(), consumed);
    drop((first, second, source, context));
    assert!(!state.retired.load(Ordering::SeqCst));
    drop(retained);
    assert!(state.retired.load(Ordering::SeqCst));
}
