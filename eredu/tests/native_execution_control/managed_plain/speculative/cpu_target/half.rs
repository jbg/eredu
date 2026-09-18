//! Real half checkpoint payloads through the existing public placement driver.
use super::*;
use std::collections::BTreeMap;

const CASE: &str = "managed_plain::speculative::cpu_target::half::native_original_half_cpu_target_matches_ordinary_and_controlled";
const MODE: &str = "EREDU_PUBLIC_HALF_CPU_TARGET_MODE";
const DTYPE: &str = "EREDU_PUBLIC_HALF_CPU_TARGET_DTYPE";
const RESULT: &str = "PUBLIC_HALF_CPU_TARGET_RESULT:";

#[derive(Clone, Copy, Debug)]
enum StoredHalf {
    Bfloat16,
    Float16,
}
impl StoredHalf {
    fn name(self) -> &'static str {
        match self {
            Self::Bfloat16 => "BF16",
            Self::Float16 => "F16",
        }
    }
    fn encode(self, value: f32) -> u16 {
        match self {
            Self::Bfloat16 => ::half::bf16::from_f32(value).to_bits(),
            Self::Float16 => ::half::f16::from_f32(value).to_bits(),
        }
    }
    fn decode(self, bits: u16) -> f32 {
        match self {
            Self::Bfloat16 => ::half::bf16::from_bits(bits).to_f32(),
            Self::Float16 => ::half::f16::from_bits(bits).to_f32(),
        }
    }
}

fn target_payload(target: &Fixture, dtype: StoredHalf) -> usize {
    let path = target.0.join("model.safetensors");
    let original = std::fs::read(&path).unwrap();
    let length = usize::try_from(u64::from_le_bytes(original[..8].try_into().unwrap())).unwrap();
    let header: serde_json::Map<String, serde_json::Value> =
        serde_json::from_slice(&original[8..8 + length]).unwrap();
    let data = &original[8 + length..];
    let mut tensors = BTreeMap::new();
    let mut converted = 0;
    for (name, entry) in header {
        if name == "__metadata__" {
            continue;
        }
        let shape: Vec<usize> = entry["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|n| usize::try_from(n.as_u64().unwrap()).unwrap())
            .collect();
        let count = shape
            .iter()
            .copied()
            .try_fold(1usize, usize::checked_mul)
            .unwrap();
        let start = usize::try_from(entry["data_offsets"][0].as_u64().unwrap()).unwrap();
        let end = usize::try_from(entry["data_offsets"][1].as_u64().unwrap()).unwrap();
        let stored = entry["dtype"].as_str().unwrap();
        let payload = if stored == "F32" {
            assert_eq!(end - start, count * 4);
            let mut nonzero = false;
            let mut encoded = Vec::with_capacity(count * 2);
            for bytes in data[start..end].chunks_exact(4) {
                let value = f32::from_le_bytes(bytes.try_into().unwrap());
                assert!(value.is_finite());
                let bits = dtype.encode(value);
                let rounded = dtype.decode(bits);
                assert!(rounded.is_finite());
                nonzero |= rounded != 0.0;
                encoded.extend_from_slice(&bits.to_le_bytes());
            }
            assert!(nonzero, "nonzero fixture tensor became empty: {name}");
            converted += 1;
            (dtype.name().to_owned(), shape, encoded)
        } else {
            assert_eq!(stored, "I32", "unexpected fixture payload: {name}");
            assert_eq!(end - start, count * 4);
            (stored.to_owned(), shape, data[start..end].to_vec())
        };
        tensors.insert(name, payload);
    }
    assert!(converted > 1);
    // Preserve the fixture's ordinary serialization/schema; only actual stored
    // floating bytes and their exact dtype/offset metadata change.
    super::super::super::super::quantized_parameters::write_tensors(&target.0, &tensors);
    let written = std::fs::read(path).unwrap();
    let length = usize::try_from(u64::from_le_bytes(written[..8].try_into().unwrap())).unwrap();
    let header: serde_json::Map<String, serde_json::Value> =
        serde_json::from_slice(&written[8..8 + length]).unwrap();
    for (name, (stored, shape, bytes)) in &tensors {
        let entry = &header[name];
        assert_eq!(entry["dtype"], stored.as_str());
        assert_eq!(entry["shape"], serde_json::json!(shape));
        let start = usize::try_from(entry["data_offsets"][0].as_u64().unwrap()).unwrap();
        let end = usize::try_from(entry["data_offsets"][1].as_u64().unwrap()).unwrap();
        assert_eq!(&written[8 + length + start..8 + length + end], bytes);
    }
    converted
}

#[test]
#[ignore = "requires retained half CPU target sources and an accessible Metal assistant"]
fn native_original_half_cpu_target_matches_ordinary_and_controlled() {
    if let Ok(mode) = std::env::var(MODE) {
        let dtype = match std::env::var(DTYPE).unwrap().as_str() {
            "BF16" => StoredHalf::Bfloat16,
            "F16" => StoredHalf::Float16,
            _ => panic!("unknown half fixture"),
        };
        let (target, draft) = super::super::super::super::speculative::artifacts();
        let tensors = target_payload(&target, dtype);
        let (output, rounds) = run_artifacts_on_with_rounds(
            &mode,
            target,
            draft,
            target_device(),
            draft_placement(),
            cpu_assistant::cpu_settings(),
        );
        // max_draft_tokens=1 and four committed outputs exercise cached target
        // verification beyond the first complete prompt/assistant invocation.
        assert!(
            rounds >= 2,
            "half checkpoint did not reach cached verification"
        );
        println!(
            "\n{RESULT}{}",
            serde_json::json!({"dtype":dtype.name(),"tensors":tensors,"rounds":rounds,"output":output})
        );
        return;
    }
    for dtype in [StoredHalf::Bfloat16, StoredHalf::Float16] {
        let mut expected = None;
        for mode in ["ordinary", "managed", "controlled"] {
            let result = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", CASE, "--ignored", "--nocapture"])
                .env(MODE, mode)
                .env(DTYPE, dtype.name())
                .output()
                .unwrap();
            let stdout = String::from_utf8_lossy(&result.stdout);
            assert!(
                result.status.success(),
                "{dtype:?} {mode}: {stdout}\n{}",
                String::from_utf8_lossy(&result.stderr)
            );
            let output: serde_json::Value = serde_json::from_str(
                stdout
                    .lines()
                    .find_map(|line| line.strip_prefix(RESULT))
                    .expect("completed half target result"),
            )
            .unwrap();
            assert_eq!(output["dtype"], dtype.name());
            assert_eq!(output["output"]["ids"].as_array().unwrap().len(), 4);
            if let Some(expected) = &expected {
                assert_eq!(&output, expected, "{dtype:?} {mode}");
            } else {
                expected = Some(output);
            }
        }
    }
}
