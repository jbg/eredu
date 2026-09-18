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
fn retained_conditional_units_bind_real_config_and_preserve_paid_branch_retirement() {
    let mut config = fixture_config();
    let affine = eredu_checkpoint::LinearFormat::Affine(eredu_checkpoint::AffineQuantization {
        group_size: 16,
        ..Default::default()
    });
    config
        .text
        .linear_formats
        .insert("model.layers.1.self_attn.q_proj.weight".into(), affine);
    config
        .text
        .linear_formats
        .insert("mtp.layers.0.self_attn.q_proj.weight".into(), affine);
    let wrong_config = config.clone();
    let ordinary_context = WorkspaceContext::new(Facts);
    let mut ordinary =
        ConditionalLayeredModel::<WorkspaceBackend>::new(config, &ordinary_context).unwrap();
    let coordinates = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (3, 0)];
    let expected = coordinates.map(|(group, index)| {
        rows(
            &ordinary
                .construct_unit(group, index, &ordinary_context)
                .unwrap(),
        )
    });
    ordinary
        .prepare_construction_units(None, &ordinary_context)
        .unwrap();
    let units = ordinary.construction_units().unwrap().clone();
    let config = ordinary.config_owner().clone();
    assert!(
        expected[3]
            .iter()
            .any(|(name, _, _)| name == "model.layers.1.self_attn.q_proj.scales")
    );
    assert!(
        expected[4]
            .iter()
            .any(|(name, _, _)| name == "mtp.layers.0.self_attn.q_proj.scales")
    );
    let mut wrong =
        ConditionalLayeredModel::<WorkspaceBackend>::new(wrong_config, &ordinary_context).unwrap();
    drop(ordinary);
    let state = Arc::new(State {
        remaining: AtomicUsize::new(usize::MAX),
        retired: AtomicBool::new(false),
    });
    let funding = HostMetadataFunding::new(Account(state.clone())).unwrap();
    let context = WorkspaceContext::new_with_metadata_funding(Facts, funding).unwrap();
    assert!(
        wrong
            .install_construction_units(units.clone(), &context)
            .is_err()
    );
    assert!(wrong.construction_units().is_none());
    drop((wrong, ordinary_context));
    let mut first = ConditionalLayeredModel::<WorkspaceBackend>::new_with_config(
        crate::replicated_text::CompositeModelConfig::Retained(config.clone()),
        &context,
    )
    .unwrap();
    let mut second = ConditionalLayeredModel::<WorkspaceBackend>::new_with_config(
        crate::replicated_text::CompositeModelConfig::Retained(config),
        &context,
    )
    .unwrap();
    first
        .install_construction_units(units.clone(), &context)
        .unwrap();
    second
        .install_construction_units(units.clone(), &context)
        .unwrap();
    let outputs =
        coordinates.map(|(group, index)| first.construct_unit(group, index, &context).unwrap());
    for (index, (group, unit)) in coordinates.into_iter().enumerate() {
        assert_eq!(rows(&outputs[index]), expected[index]);
        assert_eq!(
            rows(&second.construct_unit(group, unit, &context).unwrap()),
            expected[index]
        );
    }
    let retained = Retained {
        value: outputs,
        _funding: context.metadata_funding().unwrap(),
    };
    let consumed = context.metadata_census().unwrap().context_bytes();
    state.remaining.store(0, Ordering::SeqCst);
    let refusal = first.construct_unit(1, 0, &context).err().unwrap();
    assert!(matches!(
        refusal.into_metadata_funding_error(),
        Ok(HostMetadataFundingError::Capacity { .. })
    ));
    assert_eq!(context.metadata_census().unwrap().context_bytes(), consumed);
    drop((first, second, units, context));
    assert!(!state.retired.load(Ordering::SeqCst));
    drop(retained);
    assert!(state.retired.load(Ordering::SeqCst));
}

fn fixture_config() -> ParsedHybridConfig {
    crate::qwen::hybrid::model_args_from_config_value(&serde_json::json!({
        "model_type":"qwen3_5", "image_token_id":60,"video_token_id":61,
        "text_config":{"model_type":"qwen3_5_text","vocab_size":64,"hidden_size":32,
            "num_hidden_layers":2,"num_attention_heads":4,"num_key_value_heads":2,"head_dim":8,
            "max_position_embeddings":128,"linear_conv_kernel_dim":4,"linear_key_head_dim":8,
            "linear_value_head_dim":8,"linear_num_key_heads":2,"linear_num_value_heads":4,
            "intermediate_size":64,"layer_types":["linear_attention","full_attention"],
            "mtp_num_hidden_layers":2,"tie_word_embeddings":false},
        "vision_config":{"depth":2,"hidden_size":32,"intermediate_size":64,"num_heads":4,
            "num_position_embeddings":16,"in_channels":3,"patch_size":2,"spatial_merge_size":2,
            "temporal_patch_size":2,"out_hidden_size":32}
    }))
    .unwrap()
}

#[test]
fn canonical_conditional_boundary_preserves_uneven_rows_and_every_reached_refusal() {
    use crate::composite_execution::CompositeArchitecture;
    use eredu_runtime::{ArchitectureBoundary, DeviceState};
    type Model = ConditionalLayeredModel<WorkspaceBackend>;
    type RuntimeState = DeviceState<WorkspaceBackend, eredu_runtime::working_memory::WorkspaceResidentLayerState>;
    let ordinary = WorkspaceContext::new(Facts);
    let model = Model::new(fixture_config(), &ordinary).unwrap();
    for count in [0, 2] {
        let selected = super::super::ConditionalPipelineBoundarySchema {
            hidden_size: 32, deepstack_count: count,
        }.wire_schema().unwrap().resolve(2, 3).unwrap();
        for (from, to, continuation) in [(0, 0, Some((12, 32))), (0, 1, None), (1, 1, None)] {
            crate::architecture_parameter_metadata_tests::boundary(|metadata| {
                <Model as CompositeArchitecture<WorkspaceBackend, RuntimeState>>::partition_boundary_schema(
                    &model, from, to, &selected, 2, 3, &[7, 3], continuation, metadata)
            });
            let actual = <Model as CompositeArchitecture<WorkspaceBackend, RuntimeState>>::partition_boundary_schema(
                &model, from, to, &selected, 2, 3, &[7, 3], continuation, None).unwrap().unwrap();
            assert_eq!(actual.primary().shape(), if from == 0 && to == 0 { vec![12, 32] } else { vec![2, 3, 32] });
            for field in actual.auxiliary() {
                assert_eq!(field.shape(), if from == 1 { [2, 3, 32] } else { [2, 7, 32] });
            }
        }
    }
}

#[test]
fn canonical_conditional_collective_waves_preserve_segmented_shapes_and_reached_refusals() {
    use crate::composite_execution::{CompositeArchitecture, CompositeTensorCollective, PreparedCompositeInput};
    use eredu_nn::Tensor;
    use eredu_runtime::{PreparedInputInspector, PreparedInputPart, PreparedInputPayload, PreparedModelInput};
    struct Inspector;
    impl PreparedInputInspector<WorkspaceTensor> for Inspector {
        fn identity(&self, tensor: &WorkspaceTensor) -> Result<eredu_core::InputTensorIdentity, eredu_core::PreparedInputError> {
            eredu_core::InputTensorIdentity::new(match tensor.layout().dtype() {
                WorkspaceDtype::Uint32 => eredu_core::checkpoint::TensorDtype::U32,
                WorkspaceDtype::Int32 => eredu_core::checkpoint::TensorDtype::I32,
                _ => eredu_core::checkpoint::TensorDtype::F32,
            }, tensor.shape().iter().map(|&n|n as usize).collect())
        }
        fn i32_values(&self, value: &WorkspaceTensor) -> Result<Vec<i32>, eredu_core::CapabilityError> {
            assert_eq!(value.shape(), [1,3]); Ok(vec![1,4,4])
        }
        fn bool_values(&self, _: &WorkspaceTensor) -> Result<Vec<bool>, eredu_core::CapabilityError> { panic!("no Boolean metadata") }
    }
    type Model = ConditionalLayeredModel<WorkspaceBackend>;
    type RuntimeState = eredu_runtime::DeviceState<WorkspaceBackend, eredu_runtime::working_memory::WorkspaceResidentLayerState>;
    let ordinary = WorkspaceContext::new(Facts);
    let config = fixture_config();
    let model = Model::new(config.clone(), &ordinary).unwrap();
    let text = |n| PreparedInputPart::new(eredu_core::InputModality::Text,
        PreparedInputPayload::TokenIds(WorkspaceTensor::full_u32(7,&[1,n],&ordinary).unwrap()),[]).unwrap();
    let input = PreparedModelInput::new(vec![text(2),
        PreparedInputPart::new_with_extents(eredu_core::InputModality::Image,
            PreparedInputPayload::Tensor(WorkspaceTensor::full_f32(0.5,&[16,24],&ordinary).unwrap()),
            [(eredu_core::InputMetadataKey::PatchGrid,WorkspaceTensor::full_i32(0,&[1,3],&ordinary).unwrap())],
            [eredu_core::InputExtent::PatchGrid { time:1,height:4,width:4 }]).unwrap(), text(3)],
        |tensor| Inspector.identity(tensor)).unwrap();
    let admitted=crate::media_plan::admit_qwen_hybrid_input(&config,&input,&Inspector).unwrap();
    let prepared=PreparedCompositeInput::new(&input,&admitted).unwrap();
    let operation=|metadata: Option<&WorkspaceContext>| <Model as CompositeArchitecture<WorkspaceBackend,RuntimeState>>::prepared_group_collective_waves(
        &model,0,prepared,2,2,metadata);
    crate::architecture_parameter_metadata_tests::boundary(operation);
    let waves=operation(None).unwrap().unwrap();
    let shapes=waves.iter().map(|wave|wave.iter().map(|op|match op {
        CompositeTensorCollective::Sum {shape}=>shape.clone(),
    }).collect::<Vec<_>>()).collect::<Vec<_>>();
    assert_eq!(shapes, [vec![vec![1,2,32],vec![1,3,32],vec![16,32],vec![16,32]],
        vec![vec![1,2,32],vec![1,3,32],vec![16,32],vec![16,32],vec![4,32]]]);
}
