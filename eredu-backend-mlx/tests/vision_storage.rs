//! Actual shared vision products retain native backing and alias ownership.
use eredu_architectures::qwen::vision::{
    VisionBlock, VisionConfigSource, VisionOutput, VisionOutputTensorRole,
};
use eredu_backend_mlx::{backend::nn::shared::MlxNeuralBackend, MlxTensor};
use eredu_nn::{
    tensor_storage::TensorStorageSnapshot, ParameterMetadata, ParameterVisitorMut, Parameterized,
    Tensor, TensorElementType,
};
use safemlx::{Array, Device, DeviceType, Dtype, Stream};
type Block = VisionBlock<MlxNeuralBackend>;

struct Bind<'a>(&'a Stream);
impl<'a> ParameterVisitorMut<'a, MlxTensor> for Bind<'_> {
    fn visit_mut(&mut self, metadata: ParameterMetadata, value: &'a mut MlxTensor) {
        let name = metadata.id.as_str();
        let shape = value.as_array().shape();
        let count = shape.iter().map(|&n| n as usize).product();
        let norm = name.contains(".norm");
        let weight = name.ends_with(".weight");
        let seed = name.bytes().map(usize::from).sum::<usize>();
        let data: Vec<f32> = (0..count)
            .map(|i| {
                let delta = ((i * 7 + seed) % 19) as f32 - 9.0;
                if norm && weight {
                    1.0 + delta * 0.01
                } else {
                    delta * 0.025
                }
            })
            .collect();
        // The first normalization remains BF16. The QKV bias promotes its
        // result to F32, so subsequent projections do not consume block ingress.
        let dtype = if weight || norm {
            Dtype::Bfloat16
        } else {
            Dtype::Float32
        };
        *value = MlxTensor::from_array(
            Array::from_slice(&data, &shape)
                .as_dtype(dtype, self.0)
                .unwrap(),
        );
    }
}

fn block(stream: &Stream) -> Block {
    let config = serde_json::from_value::<VisionConfigSource>(serde_json::json!({
        "depth": 1, "hidden_size": 8, "intermediate_size": 12,
        "num_heads": 2, "num_position_embeddings": 16, "in_channels": 3,
        "patch_size": 2, "spatial_merge_size": 2, "temporal_patch_size": 1,
        "out_hidden_size": 8, "hidden_act": "gelu_pytorch_tanh",
        "deepstack_visual_indexes": []
    }))
    .unwrap()
    .normalize_qwen3_vl()
    .unwrap();
    let mut block = Block::new(&config, 0, stream).unwrap();
    block.visit_parameters_mut(&mut Bind(stream));
    block
}

fn inputs(stream: &Stream) -> (MlxTensor, MlxTensor, MlxTensor) {
    let data: Vec<f32> = (0..40)
        .map(|i| ((i * 11 % 23) as f32 - 11.0) * 0.07)
        .collect();
    let hidden = MlxTensor::from_array(
        Array::from_slice(&data, &[5, 8])
            .as_dtype(Dtype::Bfloat16, stream)
            .unwrap(),
    );
    let angles: Vec<f32> = (0..20)
        .map(|i| (i / 4) as f32 * 0.13 + (i % 2) as f32 * 0.07)
        .collect();
    let cos = MlxTensor::from_array(Array::from_slice(
        &angles.iter().map(|x| x.cos()).collect::<Vec<_>>(),
        &[5, 4],
    ));
    let sin = MlxTensor::from_array(Array::from_slice(
        &angles.iter().map(|x| x.sin()).collect::<Vec<_>>(),
        &[5, 4],
    ));
    (hidden, cos, sin)
}

fn values(value: MlxTensor) -> Vec<f32> {
    assert_eq!(value.as_array().shape(), &[5, 8]);
    assert_eq!(value.as_array().dtype(), Dtype::Float32);
    let values = value
        .as_array()
        .evaluated()
        .unwrap()
        .as_slice::<f32>()
        .to_vec();
    assert!(values.iter().all(|x| x.is_finite()));
    assert!(values.iter().any(|x| x.abs() > 0.01));
    values
}

#[test]
fn completed_vision_storage_survey_identifies_backing_larger_than_a_view() {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let mut model = block(&stream);
    let (hidden, cos, sin) = inputs(&stream);
    let output = model
        .forward(&hidden, &[2, 3], &cos, &sin, &stream)
        .unwrap();
    assert!(!output.as_array().is_available().unwrap());
    let lazy = MlxTensor::inspect_storage(&[&output]).unwrap();
    assert_eq!(lazy.values, [None]);
    assert!(!output.as_array().is_available().unwrap());
    let expected = values(output.clone());
    let alias = output.clone();
    let slice = output
        .index(
            &[eredu_nn::Index::Range(1, 3), eredu_nn::Index::Full],
            &stream,
        )
        .unwrap();
    slice.as_array().evaluated().unwrap();
    assert_eq!(slice.as_array().nbytes(), 2 * 8 * 4);
    let survey = MlxTensor::inspect_storage(&[&output, &alias, &slice]).unwrap();
    survey.validate(3).unwrap();
    assert_eq!(survey.values, [Some(0), Some(0), Some(0)]);
    assert_eq!(survey.backings.len(), 1);
    assert!(survey.backings[0].allocator_owned);
    assert!(survey.backings[0].allocator_capacity_bytes.unwrap() >= 5 * 8 * 4);
    println!(
        "VISION_STORAGE_METRICS output_logical_bytes={} view_logical_bytes={} unique_backings={} allocator_capacity_bytes={}",
        output.as_array().nbytes(), slice.as_array().nbytes(), survey.backings.len(),
        survey.backings[0].allocator_capacity_bytes.unwrap(),
    );
    assert_eq!(values(output.clone()), expected);
    drop((model, output, alias, slice, hidden, cos, sin));
    // The report holds plain data after every native root is gone. Its index
    // must never be used to join another survey or as a retirement authority.
    survey.validate(3).unwrap();
    assert_eq!(survey.backings.len(), 1);
}

#[test]
fn vision_owner_snapshot_keeps_mixed_logical_roots_and_shared_native_backing() {
    let snapshot = {
        let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
        let mut model = block(&stream);
        let (hidden, cos, sin) = inputs(&stream);
        let output = model
            .forward(&hidden, &[2, 3], &cos, &sin, &stream)
            .unwrap();
        let embeddings = output.reshape(&[1, 5, 8], &stream).unwrap();
        let sliced_feature = embeddings
            .index(
                &[
                    eredu_nn::Index::Full,
                    eredu_nn::Index::Range(1, 3),
                    eredu_nn::Index::Full,
                ],
                &stream,
            )
            .unwrap();
        // Exercise the shared output-root traversal with actual nonzero block
        // products. The BF16 feature intentionally differs from F32 embeddings.
        let first = VisionOutput {
            embeddings,
            deepstack_features: vec![sliced_feature, hidden.reshape(&[1, 5, 8], &stream).unwrap()],
        };
        let second = VisionOutput {
            embeddings: first.embeddings.clone(),
            deepstack_features: vec![],
        };
        let lazy =
            TensorStorageSnapshot::inspect(first.storage_values().chain(second.storage_values()))
                .unwrap();
        lazy.validate().unwrap();
        assert_eq!(lazy.values.len(), 4);
        assert_eq!(lazy.storage.values[0], None);
        assert_eq!(lazy.storage.values[3], None);
        assert!(!output.as_array().is_available().unwrap());

        let expected = values(output.clone());
        for (_, value) in first.storage_values().chain(second.storage_values()) {
            value.as_array().evaluated().unwrap();
        }
        let snapshot =
            TensorStorageSnapshot::inspect(first.storage_values().chain(second.storage_values()))
                .unwrap();
        snapshot.validate().unwrap();
        assert_eq!(
            snapshot
                .values
                .iter()
                .map(|value| value.role)
                .collect::<Vec<_>>(),
            [
                VisionOutputTensorRole::Embeddings,
                VisionOutputTensorRole::DeepStack(0),
                VisionOutputTensorRole::DeepStack(1),
                VisionOutputTensorRole::Embeddings,
            ]
        );
        assert_eq!(
            snapshot
                .values
                .iter()
                .map(|value| value.logical_bytes)
                .collect::<Vec<_>>(),
            [Some(160), Some(64), Some(80), Some(160)]
        );
        assert_eq!(snapshot.values[1].shape, [1, 2, 8]);
        assert_eq!(snapshot.values[0].element, Some(TensorElementType::F32));
        assert_eq!(snapshot.values[2].element, Some(TensorElementType::Bf16));
        assert_eq!(
            snapshot.storage.values,
            [Some(0), Some(0), Some(1), Some(0)]
        );
        assert_eq!(snapshot.storage.backings.len(), 2);
        assert!(snapshot
            .storage
            .backings
            .iter()
            .all(|backing| backing.allocator_owned));
        assert!(
            snapshot.storage.backings[0]
                .allocator_capacity_bytes
                .unwrap()
                >= 160
        );
        assert!(
            snapshot.storage.backings[1]
                .allocator_capacity_bytes
                .unwrap()
                >= 80
        );
        assert_eq!(values(output), expected);
        snapshot
    };
    // Repeated semantic roles remain separate logical roots, even when their
    // backing is identical. The report owns metadata after all native roots die.
    snapshot.validate().unwrap();
    assert_eq!(snapshot.values.len(), 4);
    assert_eq!(snapshot.storage.backings.len(), 2);
}
