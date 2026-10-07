use super::*;

use eredu_architectures::voicechat::{
    tts_mog::TtsError,
    tts_prompt::{AriaPromptConfig, AriaPromptPlan},
};
use serde_json::Value;

fn integers(value: &Value) -> Vec<i32> {
    value["values"]
        .as_array()
        .unwrap()
        .iter()
        .map(|n| n.as_i64().unwrap() as i32)
        .collect()
}
fn shape(value: &Value) -> Vec<i32> {
    value["shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|n| n.as_i64().unwrap() as i32)
        .collect()
}

#[test]
fn voicechat_aria_prompt_matches_original_initialization_and_batch_expansion() {
    let fixture: Value =
        serde_json::from_str(eredu_evaluation::voicechat::ARIA_PROMPT_REFERENCE_JSON).unwrap();
    let context = NumericContext::default();
    for case in fixture["cases"].as_array().unwrap() {
        let config = AriaPromptConfig {
            sample_rate: case["sample_rate"].as_u64().unwrap() as usize,
            samples_per_frame: case["samples_per_frame"].as_u64().unwrap() as usize,
            audio_prompt_duration: case["prompt_duration"].as_f64().unwrap(),
            codebooks: case["codebooks"].as_i64().unwrap() as i32,
            speech_pad: case["speech_pad_id"].as_i64().unwrap() as i32,
        };
        let carrier_samples = case["carrier"]["shape"][2].as_u64().unwrap() as usize;
        let plan = AriaPromptPlan::new(config, carrier_samples).unwrap();
        assert_eq!(plan.carrier_samples() as usize, carrier_samples);
        assert!(matches!(
            AriaPromptPlan::new(config, carrier_samples - 1),
            Err(TtsError::Capacity { resource: "carrier samples", required, maximum })
                if required == carrier_samples && maximum == carrier_samples - 1
        ));
        let carrier = plan.carrier::<NumericTensor>(&context).unwrap();
        assert_eq!(carrier.shape(), &[1, 1, carrier_samples as i32]);
        assert!(carrier
            .to_f32_vec(&context)
            .unwrap()
            .iter()
            .all(|&x| x == 0.0));
        assert_eq!(case["carrier"]["nonzero_samples"].as_u64().unwrap(), 0);

        let original = integers(&case["supplied_codes"]);
        let encoded =
            NumericTensor::from_i32_slice(&original, &shape(&case["supplied_codes"]), &context)
                .unwrap();
        for batch in case["batches"].as_array().unwrap() {
            let rows = batch["batch"].as_i64().unwrap() as i32;
            let expected = &batch["tensors"];
            let positions = integers(&expected["subword_ids"]).len();
            assert!(matches!(
                plan.prepare_encoded(&encoded, rows, positions - 1, &context),
                Err(TtsError::Capacity { resource: "prompt positions", required, maximum })
                    if required == positions && maximum == positions - 1
            ));
            let prepared = plan
                .prepare_encoded(&encoded, rows, positions, &context)
                .unwrap();
            assert_eq!(prepared.codes().shape(), shape(&expected["code"]));
            assert_eq!(
                prepared.codes().to_i32_vec(&context).unwrap(),
                integers(&expected["code"])
            );
            assert_eq!(prepared.subword_ids(), integers(&expected["subword_ids"]));
            let subword_mask = expected["subword_mask"]["values"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_bool().unwrap())
                .collect::<Vec<_>>();
            assert_eq!(prepared.subword_mask(), subword_mask);
            assert_eq!(prepared.layout().batch(), rows);
            assert_eq!(
                prepared.layout().frames(),
                expected["audio_mask"]["shape"][1].as_i64().unwrap() as i32
            );

            let feedback = prepared.initial_feedback(&context).unwrap();
            assert_eq!(feedback.shape(), &[rows, 1, config.codebooks]);
            let expected_codes = integers(&expected["code"]);
            let row_stride = prepared.layout().frames() as usize * config.codebooks as usize;
            let final_codes = expected_codes
                .chunks_exact(row_stride)
                .flat_map(|row| {
                    row[row_stride - config.codebooks as usize..]
                        .iter()
                        .copied()
                })
                .collect::<Vec<_>>();
            assert_eq!(feedback.to_i32_vec(&context).unwrap(), final_codes);
        }
        // Preparation and expansion preserve the supplied finite encoder output.
        assert_eq!(encoded.to_i32_vec(&context).unwrap(), original);
    }
}
