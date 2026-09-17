use super::*;

fn auxiliary_values(module: &impl Parameterized<NumericTensor>) -> Vec<NumericTensor> {
    #[derive(Default)]
    struct Parameters(Vec<(String, *const NumericTensor)>);
    impl<'a> ParameterVisitor<'a, NumericTensor> for Parameters {
        fn visit(&mut self, metadata: ParameterMetadata, value: &'a NumericTensor) {
            self.0.push((metadata.id.as_str().into(), value));
        }
    }
    let mut before = Parameters::default();
    module.visit_parameters(&mut before);
    let mut auxiliary = Vec::new();
    let mut retained = Vec::new();
    assert!(module.visit_retained_values(&mut |value| {
        let identity = value as *const NumericTensor;
        retained.push(identity);
        if !before.0.iter().any(|(_, parameter)| *parameter == identity) {
            auxiliary.push(value.clone());
        }
    }));
    assert!(before.0.iter().all(|(_, value)| retained.contains(value)));
    let mut after = Parameters::default();
    module.visit_parameters(&mut after);
    assert_eq!(
        before.0, after.0,
        "retained inspection preserves editable slots"
    );
    auxiliary
}

#[test]
fn v4_retained_values_include_local_compressor_and_indexer_rotary_frequencies() {
    let args = tiny_v4_args();
    let context = NumericContext::default();
    for (layer, mut expected) in [
        (0, vec![1.0_f32]),
        (1, vec![1.0, 0.25, 0.25]),
        (2, vec![1.0, 1.0 / 128.0]),
    ] {
        let attention =
            deepseek::attention::v4::Attention::<NumericBackend>::new(&args, layer, &context)
                .unwrap();
        let auxiliary = auxiliary_values(&attention);
        let mut actual = auxiliary
            .iter()
            .map(|value| {
                assert_eq!(value.shape, vec![args.qk_rope_head_dim / 2]);
                assert!(value
                    .data
                    .iter()
                    .all(|value| value.is_finite() && *value > 0.0));
                value.data[0]
            })
            .collect::<Vec<_>>();
        actual.sort_by(f32::total_cmp);
        expected.sort_by(f32::total_cmp);
        assert_eq!(actual, expected, "layer {layer} retained rotary buffers");
    }
}

#[test]
fn muse_optional_query_scale_is_retained_without_an_editable_parameter_identity() {
    let mut args =
        muse_glimmer::DecoderConfig::from_hf_value(&dense_muse_partition_fixture()).unwrap();
    args.qk_scale_factor = 0.375;
    let context = NumericContext::default();
    for gguf in [false, true] {
        args.weight_convention = if gguf {
            muse_glimmer::WeightConvention::Gguf
        } else {
            muse_glimmer::WeightConvention::HuggingFace
        };
        let attention =
            muse_glimmer::text::Attention::<NumericBackend>::new(&args, 0, &context).unwrap();
        let auxiliary = auxiliary_values(&attention);
        if gguf {
            assert!(auxiliary.is_empty());
        } else {
            assert_eq!(auxiliary.len(), 1);
            assert_eq!(auxiliary[0].shape, vec![args.head_dim]);
            assert_eq!(auxiliary[0].data, vec![0.375; args.head_dim as usize]);
        }
    }
}
