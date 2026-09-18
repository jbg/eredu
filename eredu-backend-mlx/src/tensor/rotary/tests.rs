use super::*;
use crate::backend::ExecutionContext;
use eredu_nn::multimodal::{MultiAxisRotarySpec, PreparedMultiAxisRotary, RotaryAxisSpec};
use safemlx::{Device, DeviceType};

#[test]
#[ignore = "explicit native CPU prepared rotary parity; root serialized native validation"]
fn native_prepared_rotary_matches_generated_values_for_every_layout() {
    let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = context.stream();
    let positions = MlxTensor::from_array(Array::from_slice(&[-2i32, 3, 8, 5, -1, 7], &[2, 3]));
    for layout in [
        MultiAxisRotaryLayout::IndependentAxes,
        MultiAxisRotaryLayout::SplitHalves,
        MultiAxisRotaryLayout::RoundRobinSections,
    ] {
        let dimensions = if layout == MultiAxisRotaryLayout::RoundRobinSections {
            [0, 4, 8]
        } else {
            [2, 4, 8]
        };
        let spec = MultiAxisRotarySpec {
            axes: dimensions
                .into_iter()
                .enumerate()
                .map(|(i, dimensions)| RotaryAxisSpec {
                    dimensions,
                    position_offset: i as i32 - 1,
                })
                .collect(),
            base: 10000.,
            minimum_position: 0,
            layout,
        };
        let mut frequencies = vec![0.; spec.as_ref().frequency_count().unwrap()];
        spec.as_ref().fill_frequencies(&mut frequencies).unwrap();
        let ordinary = MlxTensor::multi_axis_rotary_embeddings(&positions, &spec, stream).unwrap();
        let prepared = MlxTensor::multi_axis_rotary_embeddings_prepared(
            &positions,
            PreparedMultiAxisRotary::new(spec.as_ref(), &frequencies).unwrap(),
            stream,
        )
        .unwrap();
        for (actual, expected) in [(&prepared.0, &ordinary.0), (&prepared.1, &ordinary.1)] {
            assert_eq!(actual.shape(), [2, spec.dimensions().unwrap()]);
            assert_eq!(actual.as_array().dtype(), safemlx::Dtype::Float32);
            assert_eq!(expected.as_array().dtype(), safemlx::Dtype::Float32);
            let a = actual.as_array().evaluated().unwrap();
            let b = expected.as_array().evaluated().unwrap();
            assert_eq!(
                a.as_slice::<f32>()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                b.as_slice::<f32>()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>()
            );
        }
        let sine = prepared.1.as_array().evaluated().unwrap();
        assert!(sine.as_slice::<f32>().iter().any(|v| v.abs() > 0.01));
    }
}
