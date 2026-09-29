//! Encoded native trajectories against independently decoded identical weights.
use super::*;
use eredu_evaluation::qwen4_exp::Fixture;

pub(super) fn is_fixture(checkpoint: &Path) -> bool {
    checkpoint
        .file_name()
        .is_some_and(|name| name == "encoded.gguf")
}

pub(super) fn baseline_path(checkpoint: &Path) -> std::path::PathBuf {
    if is_fixture(checkpoint) {
        checkpoint.parent().unwrap().join("safetensors")
    } else {
        checkpoint.to_path_buf()
    }
}

fn encode(fixture: &mut Fixture, physical: &str, canonical: &str) {
    let tensor = fixture
        .tensors
        .iter()
        .find(|tensor| tensor.name == physical)
        .unwrap();
    assert_eq!(tensor.shape.last(), Some(&32));
    let count = tensor.shape.iter().product::<usize>();
    let mut encoded = Vec::with_capacity(count / 32 * 34);
    let mut decoded = Vec::with_capacity(count);
    for row in 0..count / 32 {
        // Exact binary scales and independently authored signed quants. Decode
        // with scalar arithmetic, without the GGUF or native decoding kernels.
        let scale = (1 + row % 4) as f32 / 128.;
        encoded.extend(half::f16::from_f32(scale).to_bits().to_le_bytes());
        for column in 0..32 {
            let quant = ((row * 11 + column * 7 + 3) % 63) as i8 - 31;
            encoded.push(quant as u8);
            decoded.push(scale * f32::from(quant));
        }
    }
    assert!(decoded.iter().any(|value| *value != 0.));
    assert_eq!(fixture.expected[canonical].len(), decoded.len());
    fixture.expected.insert(canonical.into(), decoded);
    fixture.replace_encoding(physical, eredu_gguf::GgmlType::Q8_0, encoded);
}

fn run(axes: &'static str, residency: WorkerResidency) {
    assert!(distributed::is_available(Backend::Ring));
    let directory = tempfile::tempdir().unwrap();
    let mut fixture = Fixture::new();
    encode(
        &mut fixture,
        "blk.1.attn_q.weight",
        "model.layers.1.self_attn.q_proj.weight",
    );
    for layer in 0..2 {
        // Gate/up split output rows; the fixture's down projection would split
        // 32-value encoding blocks across TP, so it remains F32.
        for projection in ["gate", "up"] {
            encode(
                &mut fixture,
                &format!("blk.{layer}.ffn_{projection}_exps.weight"),
                &format!("model.layers.{layer}.mlp.experts.{projection}_proj"),
            );
        }
    }
    // prepare writes canonical F32 SafeTensors from `expected`, independently
    // from encoded GGUF. Both artifacts have exactly the same decoded values.
    let prepared = fixture.prepare(directory.path()).unwrap();
    let encoded = directory.path().join("encoded.gguf");
    std::fs::rename(&prepared.gguf_path, &encoded).unwrap();
    drop(prepared);
    run_ring_pipeline_processes(
        residency,
        FixtureFamily::Qwen4Exp,
        WorkerMode::OpaqueSession,
        directory,
        encoded,
        Some(axes),
    );
}

#[test]
#[ignore = "requires two local MLX Ring ranks and loopback sockets"]
fn ring_encoded_q8_query_and_experts_tp2_resident() {
    run("tp", WorkerResidency::FullyResident);
}

#[test]
#[ignore = "requires four local MLX Ring ranks and loopback sockets"]
fn ring_encoded_q8_query_and_experts_tp2_ep2_host() {
    run("tp-ep", WorkerResidency::LayerwiseHost);
}

#[test]
#[ignore = "requires eight local MLX Ring ranks and loopback sockets"]
fn ring_encoded_q8_query_and_experts_tp2_pp2_ep2_disk() {
    run("tp-pp-ep", WorkerResidency::DenseDiskStream);
}
