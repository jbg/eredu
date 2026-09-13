use super::quantized_parameters::{
    packed_residencies, verify_parameter_fixture, write_tensors, ParameterFixture,
};
use super::*;
use eredu_gguf::{GgmlType, MetadataValue as V, TensorInput, Writer};
use std::collections::BTreeMap;

fn gguf_fixture(ggml_type: GgmlType) -> ParameterFixture {
    let (root, reference, mut expected_values) = super::quantized_parameters::packed_fixture(false);
    let data = std::fs::read(reference.0.join("model.safetensors")).unwrap();
    let header_len = u64::from_le_bytes(data[..8].try_into().unwrap()) as usize;
    let header: serde_json::Value = serde_json::from_slice(&data[8..8 + header_len]).unwrap();
    let mut dense = BTreeMap::new();
    let mut packed = Vec::new();
    for (name, tensor) in header.as_object().unwrap() {
        let shape: Vec<usize> = tensor["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let mut encoded = Vec::new();
        let mut decoded = Vec::new();
        let ty = if shape.len() == 2 {
            let width = shape[1];
            assert_eq!(width % 32, 0);
            for row in 0..shape[0] {
                for block in 0..width / 32 {
                    let exponent = ((row + block + name.len()) % 3) as i32;
                    let scale_exponent =
                        exponent - if ggml_type == GgmlType::Q8_0 { 7 } else { 10 };
                    let scale = 2.0f32.powi(scale_exponent);
                    // All scales are normal FP16 powers of two, encoded independently.
                    encoded.extend((((scale_exponent + 15) as u16) << 10).to_le_bytes());
                    if ggml_type == GgmlType::Q8_0 {
                        for column in 0..32 {
                            let code =
                                ((row * 7 + block * 3 + column * 5 + name.len()) % 31) as i8 - 15;
                            encoded.push(code as u8);
                            decoded.push(code as f32 * scale);
                        }
                    } else {
                        let table = [
                            -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89,
                            113,
                        ];
                        let code = |c: usize| (row * 7 + block * 3 + c * 5 + name.len()) % 16;
                        for c in 0..16 {
                            encoded.push(code(c) as u8 | (code(c + 16) as u8) << 4);
                        }
                        for c in 0..32 {
                            decoded.push(table[code(c)] as f32 * scale);
                        }
                    }
                }
            }
            ggml_type
        } else {
            decoded = expected_values[name].clone();
            encoded = decoded.iter().flat_map(|v| v.to_le_bytes()).collect();
            GgmlType::F32
        };
        expected_values.insert(name.clone(), decoded.clone());
        dense.insert(
            name.clone(),
            (
                "F32".into(),
                shape.clone(),
                decoded.iter().flat_map(|v| v.to_le_bytes()).collect(),
            ),
        );
        let gguf_name = name
            .replace("model.layers.", "blk.")
            .replace("self_attn.q_proj", "attn_q")
            .replace("self_attn.k_proj", "attn_k")
            .replace("self_attn.v_proj", "attn_v")
            .replace("self_attn.o_proj", "attn_output")
            .replace("input_layernorm", "attn_norm")
            .replace("post_attention_layernorm", "ffn_norm")
            .replace("mlp.gate_proj", "ffn_gate")
            .replace("mlp.up_proj", "ffn_up")
            .replace("mlp.down_proj", "ffn_down")
            .replace("model.embed_tokens", "token_embd")
            .replace("model.norm", "output_norm")
            .replace("lm_head", "output");
        packed.push((
            gguf_name,
            shape.iter().rev().map(|v| *v as u64).collect::<Vec<_>>(),
            ty,
            encoded,
        ));
    }
    write_tensors(&reference.0, &dense);
    let mut config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(reference.0.join("config.json")).unwrap()).unwrap();
    config["rope_theta"] = serde_json::json!(10000.0);
    for directory in [&root.0, &reference.0] {
        std::fs::write(
            directory.join("config.json"),
            serde_json::to_vec(&config).unwrap(),
        )
        .unwrap();
    }
    let metadata = BTreeMap::from([
        ("general.architecture".into(), V::String("qwen2".into())),
        ("qwen2.embedding_length".into(), V::Uint32(32)),
        ("qwen2.block_count".into(), V::Uint32(2)),
        ("qwen2.feed_forward_length".into(), V::Uint32(64)),
        ("qwen2.attention.head_count".into(), V::Uint32(4)),
        ("qwen2.attention.head_count_kv".into(), V::Uint32(2)),
        ("qwen2.attention.key_length".into(), V::Uint32(8)),
        (
            "qwen2.attention.layer_norm_rms_epsilon".into(),
            V::Float32(1e-5),
        ),
        ("qwen2.rope.freq_base".into(), V::Float32(10000.0)),
        ("qwen2.context_length".into(), V::Uint32(1024)),
        ("qwen2.vocab_size".into(), V::Uint32(64)),
    ]);
    let tensors: Vec<_> = packed
        .iter()
        .map(|(name, dimensions, ty, bytes)| TensorInput {
            name,
            dimensions,
            ggml_type: *ty,
            data: bytes,
        })
        .collect();
    Writer::default()
        .write(
            std::fs::File::create(root.0.join("model.gguf")).unwrap(),
            &metadata,
            &tensors,
        )
        .unwrap();
    std::fs::remove_file(root.0.join("model.safetensors")).unwrap();
    ParameterFixture {
        root,
        reference,
        expected_values,
        source_file: "model.gguf",
        promote_reference: false,
    }
}

fn verify_gguf_parameters(device: LocalDevice) {
    for ty in [GgmlType::Q8_0, GgmlType::IQ4NL] {
        for residency in packed_residencies() {
            verify_parameter_fixture(gguf_fixture(ty), residency, device);
        }
    }
}

#[test]
#[ignore = "requires local native MLX execution"]
fn native_gguf_parameter_lifecycle_cpu() {
    verify_gguf_parameters(LocalDevice::Cpu);
}

#[test]
#[cfg(all(feature = "metal", target_vendor = "apple"))]
#[ignore = "requires local MLX Metal execution"]
fn native_gguf_parameter_lifecycle_metal() {
    verify_gguf_parameters(LocalDevice::Accelerator(0));
}
