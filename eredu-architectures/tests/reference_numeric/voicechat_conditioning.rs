//! Original static character conditioning and gated fusion with a disclosed affine backbone oracle.
use super::*;
use eredu_architectures::voicechat::tts_conditioning::{
    CharacterDimensions, CharacterEmbedding, CharacterFlags, CharacterMapping, GatedFusion,
};
use serde_json::Value;

fn tensor(value: &Value) -> NumericTensor {
    NumericTensor::new(
        value["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap() as i32)
            .collect::<Vec<_>>(),
        value["values"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect(),
    )
}
pub(super) fn integers(value: &Value) -> Vec<i32> {
    value["values"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap() as i32)
        .collect()
}
pub(super) fn mapping(fixture: &Value) -> CharacterMapping {
    let vocabulary = fixture["vocabulary"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
        .map(|(id, token)| (id as u32, token.as_str().unwrap().to_owned()))
        .collect();
    CharacterMapping::new(
        &vocabulary,
        CharacterFlags {
            continuation: integers(&fixture["continuation_flags"]),
            continuation_pad: fixture["continuation_pad"].as_i64().unwrap() as i32,
            special: integers(&fixture["special_flags"]),
            special_pad: fixture["special_pad"].as_i64().unwrap() as i32,
        },
    )
    .unwrap()
}
struct Bind<'v>(&'v Value);
impl<'a> ParameterVisitorMut<'a, NumericTensor> for Bind<'_> {
    fn visit_mut(&mut self, metadata: ParameterMetadata, output: &'a mut NumericTensor) {
        let replacement = tensor(&self.0[metadata.id.as_str()]);
        assert_eq!(output.shape(), replacement.shape());
        *output = replacement;
    }
}
pub(super) fn compare(actual: &NumericTensor, expected: &Value, fixture: &Value) {
    let expected = tensor(expected);
    assert_eq!(actual.shape(), expected.shape());
    let abs = fixture["tolerance"]["absolute"].as_f64().unwrap() as f32;
    let rel = fixture["tolerance"]["relative"].as_f64().unwrap() as f32;
    for (actual, expected) in actual.data.iter().zip(expected.data) {
        assert!(
            (actual - expected).abs() <= abs + rel * expected.abs(),
            "{actual} versus NVIDIA {expected}"
        );
    }
}

#[test]
fn voicechat_character_mapping_pooling_flags_and_fusion_match_nvidia() {
    let fixture: Value =
        serde_json::from_str(eredu_evaluation::voicechat::CONDITIONING_REFERENCE_JSON).unwrap();
    let context = NumericContext::default();
    let mapping = mapping(&fixture);
    let tokens = integers(&fixture["tokens"]);
    let mask = fixture["selected"]["values"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_bool().unwrap())
        .collect::<Vec<_>>();
    let batch = mapping.prepare(&tokens, Some(&mask), 2, 4, 1024).unwrap();
    assert_eq!(batch.ids(), integers(&fixture["character_ids"]));
    assert_eq!(batch.lengths(), integers(&fixture["character_lengths"]));
    let expected_mask = fixture["attention_mask"]["values"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| if v.as_bool().unwrap() { 1.0 } else { 0.0 })
        .collect::<Vec<_>>();
    assert_eq!(
        batch
            .attention_mask::<NumericTensor>(&context)
            .unwrap()
            .data,
        expected_mask
    );
    let mut embedding = CharacterEmbedding::<NumericBackend>::new(
        CharacterDimensions {
            characters: mapping.characters().len() as i32,
            width: 4,
            output_width: 4,
            use_continuation: true,
            use_special: true,
        },
        &context,
    )
    .unwrap();
    embedding.visit_parameters_mut(&mut Bind(&fixture["parameters"]));
    let output = embedding
        .forward(
            &batch,
            |embedded, _| {
                compare(embedded, &fixture["embedded"], &fixture);
                Ok(tensor(&fixture["hidden"]))
            },
            &context,
        )
        .unwrap();
    compare(&output, &fixture["character_output"], &fixture);
    let mut fusion = GatedFusion::<NumericBackend>::new(4, &context).unwrap();
    fusion.visit_parameters_mut(&mut Bind(&fixture["parameters"]));
    compare(
        &fusion
            .forward(&tensor(&fixture["audio"]), &output, &context)
            .unwrap(),
        &fixture["fusion_output"],
        &fixture,
    );

    // Capacity is checked before constructing rectangular native tensors.
    assert!(mapping
        .prepare(&tokens, Some(&mask), 2, 4, batch.ids().len() - 1)
        .is_err());
    assert!(mapping.prepare(&[-1], None, 1, 1, 8).is_err());
    let empty = mapping
        .prepare(&tokens, Some(&[false; 8]), 2, 4, 0)
        .unwrap();
    assert!(empty.is_empty());
    assert!(embedding.embed(&empty, &context).is_err());
}
