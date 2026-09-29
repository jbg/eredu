//! Shared DeepStack/final mergers preserve shuffle, dtype and rank-reduction equations.
use eredu_architectures::qwen::vision::{
    VisionBlock, VisionConfigSource, VisionInput, VisionState, VisionStatic,
};
use eredu_backend_mlx::{backend::nn::shared::MlxNeuralBackend as Backend, MlxTensor};
use eredu_nn::{ParameterMetadata, ParameterVisitorMut, Parameterized, Tensor, TensorElementType};
use safemlx::{Array, Device, DeviceType, Dtype, Stream};
struct Bind<'a>(&'a Stream);
impl<'a> ParameterVisitorMut<'a, MlxTensor> for Bind<'_> {
    fn visit_mut(&mut self, metadata: ParameterMetadata, tensor: &'a mut MlxTensor) {
        let name = metadata.id.as_str();
        let shape = tensor.shape().to_vec();
        let count: usize = shape.iter().map(|n| *n as usize).product();
        let seed: usize = name.bytes().map(usize::from).sum();
        let data: Vec<f32> = (0..count)
            .map(|i| {
                let delta = ((i * 7 + seed) % 19) as f32 - 9.;
                if name.contains("norm") && name.ends_with("weight") {
                    1. + delta * 0.01
                } else {
                    delta * 0.025
                }
            })
            .collect();
        let dtype = if name.contains("merger") && name.contains("norm") || name.ends_with("bias") {
            Dtype::Float32
        } else {
            Dtype::Bfloat16
        };
        *tensor = MlxTensor::from_array(
            Array::from_slice(&data, &shape)
                .as_dtype(dtype, self.0)
                .unwrap(),
        );
    }
}

fn setup(
    stream: &Stream,
    approximate: bool,
) -> (
    VisionStatic<Backend>,
    VisionBlock<Backend>,
    MlxTensor,
    VisionState<MlxTensor>,
) {
    let config = serde_json::from_value::<VisionConfigSource>(serde_json::json!({
        "depth": 1, "hidden_size": 8, "intermediate_size": 12,
        "num_heads": 2, "num_position_embeddings": 16, "in_channels": 3,
        "patch_size": 2, "spatial_merge_size": 2, "temporal_patch_size": 1,
        "out_hidden_size": 6, "hidden_act": "gelu_pytorch_tanh",
        "deepstack_visual_indexes": [0]
    }))
    .unwrap();
    let config = if approximate {
        config.normalize_qwen3_5()
    } else {
        config.normalize_qwen3_vl()
    }
    .unwrap();
    let mut block = VisionBlock::<Backend>::new(&config, 0, stream).unwrap();
    let mut tower = VisionStatic::<Backend>::new(config, stream).unwrap();
    block.visit_parameters_mut(&mut Bind(stream));
    tower.visit_parameters_mut(&mut Bind(stream));
    let pixels = MlxTensor::from_array(Array::from_slice(
        &(0..96)
            .map(|i| ((i * 11 % 23) as f32 - 11.) * 0.07)
            .collect::<Vec<_>>(),
        &[8, 12],
    ));
    let (hidden, state) = tower
        .begin(
            VisionInput {
                pixels: &pixels,
                grid: &[(1, 2, 4)],
            },
            stream,
        )
        .unwrap();
    (tower, block, hidden, state)
}

fn values(tensor: &MlxTensor) -> Vec<f32> {
    assert_eq!(tensor.shape(), &[1, 2, 6]);
    assert_eq!(tensor.element_type(), Some(TensorElementType::F32));
    let values = tensor
        .as_array()
        .evaluated()
        .unwrap()
        .as_slice::<f32>()
        .to_vec();
    assert!(values.iter().all(|v| v.is_finite()));
    assert!(values.iter().any(|v| v.abs() > 0.01));
    values
}

fn exercise(device: DeviceType, approximate: bool) {
    let stream = Stream::new_with_device(&Device::new(device, 0));
    let (mut tower, mut block, hidden, mut state) = setup(&stream, approximate);
    let mut ordinary_tower = tower.clone();
    let mut ordinary_block = block.clone();
    let mut ordinary_state = state.clone();
    let expected_hidden = ordinary_tower
        .forward_block(
            &mut ordinary_block,
            0,
            &hidden,
            &mut ordinary_state,
            &stream,
        )
        .unwrap();
    let expected_hidden = MlxTensor::from_array(
        expected_hidden
            .as_array()
            .as_dtype(Dtype::Bfloat16, &stream)
            .unwrap(),
    );
    let expected = ordinary_tower
        .finish(&expected_hidden, &mut ordinary_state, &stream)
        .unwrap();
    let native = safemlx::distributed::init(false, safemlx::distributed::Backend::Ring).unwrap();
    let group = <Backend as eredu_nn::NeuralBackend>::ParallelContext::uncontracted(&native);
    assert_eq!(group.size(), 1);
    let hidden = tower
        .forward_block_parallel(&mut block, 0, &hidden, &mut state, &group, &stream)
        .unwrap();
    assert_eq!(state.deepstack_features().len(), 1);
    let hidden = MlxTensor::from_array(
        hidden
            .as_array()
            .as_dtype(Dtype::Bfloat16, &stream)
            .unwrap(),
    );
    let actual = tower
        .finish_parallel(&hidden, &mut state, &group, &stream)
        .unwrap();
    assert!(state.deepstack_features().is_empty());
    assert_eq!(values(&actual.embeddings), values(&expected.embeddings));
    assert_eq!(
        values(&actual.deepstack_features[0]),
        values(&expected.deepstack_features[0])
    );
}
#[test]
fn shared_mergers_preserve_rank_reduction_and_single_bias() {
    for approximate in [false, true] {
        exercise(DeviceType::Cpu, approximate);
    }
}
#[cfg(all(feature = "metal", not(feature = "cuda")))]
#[test]
#[ignore = "requires an accessible Metal device"]
fn metal_shared_mergers_preserve_rank_reduction_and_single_bias() {
    for approximate in [false, true] {
        exercise(DeviceType::Gpu, approximate);
    }
}
