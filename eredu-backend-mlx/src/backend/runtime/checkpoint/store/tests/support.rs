use super::*;
use safemlx::{Device, DeviceType, Dtype as MlxDtype};
use safetensors::tensor::{serialize_to_file, TensorView};

fn cpu_stream() -> Stream {
    Stream::new_with_device(&Device::new(DeviceType::Cpu, 0))
}

fn write_index(dir: &Path, mappings: &[(&str, &str)]) {
    let weight_map = mappings
        .iter()
        .map(|(key, shard)| ((*key).to_string(), serde_json::json!(shard)))
        .collect::<serde_json::Map<_, _>>();
    std::fs::write(
        dir.join("model.safetensors.index.json"),
        serde_json::to_vec(&serde_json::json!({ "weight_map": weight_map })).unwrap(),
    )
    .unwrap();
}

fn write_i32(path: &Path, name: &str, values: &[i32], shape: Vec<usize>) {
    let bytes = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    let view = TensorView::new(Dtype::I32, shape, &bytes).unwrap();
    serialize_to_file([(name, view)], None, path).unwrap();
}

trait AcquireBoundedForTest {
    fn acquire(
        &self,
        key: &str,
        selection: TensorSelection,
    ) -> Result<WeightLease, CheckpointMaterializationError>;
}

impl<T: CheckpointSource> AcquireBoundedForTest for T {
    fn acquire(
        &self,
        key: &str,
        selection: TensorSelection,
    ) -> Result<WeightLease, CheckpointMaterializationError> {
        let lease = self
            .acquire_lease(TensorReadRequest {
                key: key.into(),
                selection,
                policy: WeightReadPolicy::RequireBounded,
            })
            .map_err(CheckpointMaterializationError::from)?;
        let stream = cpu_stream();
        MlxParameterMaterializationContext::new(&stream, &stream).weight_lease(lease)
    }
}
