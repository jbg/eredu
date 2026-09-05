use super::test_support::{gguf_test_plan, open_gguf_checkpoint_source_for_test};
use super::*;
use eredu_gguf::{GgmlType, TensorInput, Writer};
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

fn write_two_i32(path: &Path) {
    let left_bytes = [1i32, 2]
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    let right_bytes = [3i32, 4]
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    let left = TensorView::new(Dtype::I32, vec![2], &left_bytes).unwrap();
    let right = TensorView::new(Dtype::I32, vec![2], &right_bytes).unwrap();
    serialize_to_file([("z_tensor", left), ("a_tensor", right)], None, path).unwrap();
}

fn write_affine_gguf(path: &Path) {
    let bytes = [0u8; 36];
    Writer::default()
        .write(
            std::fs::File::create(path).unwrap(),
            &BTreeMap::new(),
            &[TensorInput {
                name: "bank.weight",
                dimensions: &[32, 2],
                ggml_type: GgmlType::Q4_0,
                data: &bytes,
            }],
        )
        .unwrap();
}

fn write_dense_bank_gguf(path: &Path) -> Vec<f32> {
    let values = (0..24).map(|value| value as f32).collect::<Vec<_>>();
    let bytes = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    Writer::default()
        .write(
            std::fs::File::create(path).unwrap(),
            &BTreeMap::new(),
            &[TensorInput {
                name: "bank.weight",
                dimensions: &[4, 3, 2],
                ggml_type: GgmlType::F32,
                data: &bytes,
            }],
        )
        .unwrap();
    values
}

fn write_wide_affine_gguf(path: &Path) {
    let blocks = [[0x00u8; 18], [0x11u8; 18], [0x22u8; 18], [0x33u8; 18]];
    let bytes = blocks.into_iter().flatten().collect::<Vec<_>>();
    Writer::default()
        .write(
            std::fs::File::create(path).unwrap(),
            &BTreeMap::new(),
            &[TensorInput {
                name: "bank.weight",
                dimensions: &[64, 2],
                ggml_type: GgmlType::Q4_0,
                data: &bytes,
            }],
        )
        .unwrap();
}

fn write_dense_gguf(path: &Path, name: &str, value: f32) {
    let bytes = value.to_le_bytes();
    Writer::default()
        .write(
            std::fs::File::create(path).unwrap(),
            &BTreeMap::new(),
            &[TensorInput {
                name,
                dimensions: &[1],
                ggml_type: GgmlType::F32,
                data: &bytes,
            }],
        )
        .unwrap();
}

fn write_two_dense_gguf(path: &Path) {
    let selected = 1.0f32.to_le_bytes();
    let unselected = 2.0f32.to_le_bytes();
    Writer::default()
        .write(
            std::fs::File::create(path).unwrap(),
            &BTreeMap::new(),
            &[
                TensorInput {
                    name: "selected.weight",
                    dimensions: &[1],
                    ggml_type: GgmlType::F32,
                    data: &selected,
                },
                TensorInput {
                    name: "unselected.weight",
                    dimensions: &[1],
                    ggml_type: GgmlType::F32,
                    data: &unselected,
                },
            ],
        )
        .unwrap();
}

fn write_block_gguf(path: &Path, ty: GgmlType, byte_len: usize) {
    let bytes = vec![0u8; byte_len];
    Writer::default()
        .write(
            std::fs::File::create(path).unwrap(),
            &BTreeMap::new(),
            &[TensorInput {
                name: "bank.weight",
                dimensions: &[64, 2],
                ggml_type: ty,
                data: &bytes,
            }],
        )
        .unwrap();
}

fn gguf_physical_selection(lease: &WeightLease) -> Option<GgufTensorSelection> {
    match &lease.source {
        WeightLeaseSource::Gguf(source) => {
            source
                .lease
                .identity()
                .physical_selection()
                .map(|selection| match selection {
                    GgufPhysicalSelection::Axis(selection) => selection.clone(),
                    GgufPhysicalSelection::DenseSpan(_) => {
                        panic!("expected single-axis GGUF selection")
                    }
                })
        }
        WeightLeaseSource::Safetensors(_) | WeightLeaseSource::Memory(_) => {
            panic!("expected GGUF lease")
        }
    }
}
