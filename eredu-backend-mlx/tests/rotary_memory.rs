//! Generic rotary graph descriptions and native numerical behavior.
use eredu_backend_mlx::{backend::nn::shared::MlxNeuralBackend, MlxTensor};
use eredu_nn::{
    mechanism_memory::{MechanismBacking, MechanismBytes, MechanismPlacement},
    multimodal::{
        multi_axis_rotary_embeddings, MultiAxisRotaryLayout, MultiAxisRotarySpec, RotaryAxisSpec,
    },
    NeuralBackend, TensorElementType,
};
use safemlx::{Array, Device, DeviceType, Stream};

fn spec(layout: MultiAxisRotaryLayout) -> MultiAxisRotarySpec {
    MultiAxisRotarySpec {
        axes: vec![
            RotaryAxisSpec {
                dimensions: 6,
                position_offset: -2,
            },
            RotaryAxisSpec {
                dimensions: 2,
                position_offset: 1,
            },
        ],
        base: 81.0,
        minimum_position: 0,
        layout,
    }
}

#[test]
fn rotary_cpu_layouts_match_independent_equations_and_described_payloads() {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let positions = [-3_i32, 2, 3, 0, 7, 5, 9, -4];
    let input = MlxTensor::from_array(Array::from_slice(&positions, &[2, 2, 2]));
    for layout in [
        MultiAxisRotaryLayout::IndependentAxes,
        MultiAxisRotaryLayout::SplitHalves,
        MultiAxisRotaryLayout::RoundRobinSections,
    ] {
        let spec = spec(layout);
        let invocation = spec
            .memory_invocation(&[2, 2, 2], TensorElementType::I32)
            .unwrap();
        let report = MlxNeuralBackend::mechanism_memory_in_context(&invocation, &stream).unwrap();
        report.validate().unwrap();
        let storage = |name: &str| report.storage.iter().find(|s| s.name == name).unwrap();
        for name in ["cosine", "sine", "angles"] {
            assert_eq!(storage(name).payload, MechanismBytes::exact(4 * 8 * 4));
            assert_eq!(storage(name).capacity.upper, None);
        }
        assert_eq!(
            storage("axis_0_host_inverse_frequencies").placement,
            MechanismPlacement::Host
        );
        assert_eq!(
            storage("axis_0_host_inverse_frequencies").payload,
            MechanismBytes::exact(3 * 4)
        );
        assert_eq!(
            storage("axis_1_angles").payload,
            MechanismBytes::exact(4 * 4)
        );
        assert_eq!(
            storage("axis_0_float_positions").backing,
            MechanismBacking::Unknown
        );
        assert!(storage("native_workspace").payload.upper.is_none());
        assert!(!report.missing.is_empty());
        let (cosine, sine) = multi_axis_rotary_embeddings(&input, &spec, &stream).unwrap();
        for (actual, sine_output) in [(cosine, false), (sine, true)] {
            assert_eq!(actual.as_array().shape(), &[2, 2, 8]);
            let evaluated = actual.as_array().evaluated().unwrap();
            let actual = evaluated.as_slice::<f32>();
            for row in 0..4 {
                let p = [
                    (positions[row * 2] - 2).max(0) as f64,
                    (positions[row * 2 + 1] + 1).max(0) as f64,
                ];
                let axis0 = [
                    p[0],
                    p[0] / 81_f64.powf(1.0 / 3.0),
                    p[0] / 81_f64.powf(2.0 / 3.0),
                ];
                let angles = match layout {
                    MultiAxisRotaryLayout::IndependentAxes => vec![
                        axis0[0], axis0[1], axis0[2], axis0[0], axis0[1], axis0[2], p[1], p[1],
                    ],
                    MultiAxisRotaryLayout::SplitHalves => vec![
                        axis0[0], axis0[1], axis0[2], p[1], axis0[0], axis0[1], axis0[2], p[1],
                    ],
                    MultiAxisRotaryLayout::RoundRobinSections => {
                        let half = [p[0], p[1] / 3.0, p[0] / 9.0, p[0] / 27.0];
                        half.into_iter().chain(half).collect()
                    }
                };
                for (column, angle) in angles.into_iter().enumerate() {
                    let expected = if sine_output {
                        angle.sin()
                    } else {
                        angle.cos()
                    } as f32;
                    assert!(
                        (actual[row * 8 + column] - expected).abs() < 2e-6,
                        "{layout:?} row {row} column {column}"
                    );
                }
            }
        }
    }
}

#[test]
fn rotary_description_rejects_native_extent_overflow_and_keeps_promotion_unknown() {
    let spec = spec(MultiAxisRotaryLayout::RoundRobinSections);
    let request = spec
        .memory_invocation(&[i32::MAX as u64 + 1, 2], TensorElementType::I32)
        .unwrap();
    assert!(MlxNeuralBackend::mechanism_memory(&request).is_err());
    let request = spec
        .memory_invocation(&[3, 2], TensorElementType::I64)
        .unwrap();
    let report = MlxNeuralBackend::mechanism_memory(&request).unwrap();
    assert!(report.missing.iter().any(|m| m.contains("promotion")));
    assert!(!report
        .storage
        .iter()
        .any(|s| s.name == "axis_0_offset_positions"));
    assert_eq!(report.values[1].element, TensorElementType::F32);
}
