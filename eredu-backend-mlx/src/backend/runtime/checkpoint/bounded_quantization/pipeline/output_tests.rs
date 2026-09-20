use super::*;
use safemlx::{Device, DeviceType};

fn prepared() -> WeightMaterialization {
    let values: Vec<_> = (0..128).map(|index| (index % 16) as f32).collect();
    let input = Array::from_slice(&values, &[2, 64]);
    WeightMaterialization::prepare_retained(vec![input], Vec::new()).unwrap()
}

fn target() -> BoundedQuantizationTarget {
    BoundedQuantizationTarget::direct("matrix.weight", "matrix.scales", Some("matrix.biases"))
        .unwrap()
        .with_affine_companion_dtype(RecipeDtype::F16)
        .unwrap()
}

#[test]
fn companion_failure_keeps_native_siblings_and_each_accepted_conversion() {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    for fail_at in [0, 1] {
        let mut owner = prepared();
        let mut calls = 0;
        let result = prepare_quantized_outputs_with(
            &mut owner,
            eredu_checkpoint::AffineQuantization::default().into(),
            &target(),
            &stream,
            None,
            |input, dtype, stream| {
                let ordinal = calls;
                calls += 1;
                if ordinal == fail_at {
                    return Err(safemlx::error::Exception::custom("companion failure"));
                }
                let output = input.as_dtype(dtype, stream)?;
                // A producer may complete eagerly before a later constructor
                // fails. The accepted result must already have an owner.
                output.evaluated()?;
                Ok(output)
            },
        );
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("companion failure")
        );
        assert_eq!(calls, fail_at + 1);
        assert_eq!(owner.inputs().len(), 1);
        assert_eq!(owner.outputs().len(), 3);
        assert_eq!(owner.outputs()[0].dtype(), Dtype::Uint32);
        assert_eq!(
            owner.outputs()[1].dtype(),
            if fail_at == 1 {
                Dtype::Float16
            } else {
                Dtype::Float32
            }
        );
        assert_eq!(owner.outputs()[2].dtype(), Dtype::Float32);
        assert!(
            owner.outputs()[0]
                .evaluated()
                .unwrap()
                .as_slice::<u32>()
                .iter()
                .any(|&value| value != 0)
        );
        assert_eq!(
            owner.outputs()[2].evaluated().unwrap().as_slice::<f32>(),
            &[15.0, 15.0]
        );
        owner.finish().unwrap();
    }
}

#[test]
fn mxfp4_retains_two_roots_without_converting_encoded_scales() {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let mut owner = prepared();
    prepare_quantized_outputs_with(
        &mut owner,
        WeightQuantization::MxFp4,
        &target(),
        &stream,
        None,
        |_, _, _| panic!("MXFP4 scales must retain their encoded dtype"),
    )
    .unwrap();
    assert_eq!(owner.outputs().len(), 2);
    assert_eq!(owner.outputs()[0].dtype(), Dtype::Uint32);
    assert_eq!(owner.outputs()[1].dtype(), Dtype::Uint8);
    let owner = owner.submit_prepared_outputs().unwrap();
    owner.wait().unwrap();
    assert!(owner.completed_outputs().all(|output| output.is_ok()));
    owner.finish().unwrap();
}
