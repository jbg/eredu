use super::*;
use crate::{
    backend::{
        managed_memory::gpu_stream::PreparedExecutionStreams, MlxAcceleratorFamily, MlxBackend,
        MlxDeviceIdentity,
    },
    MlxTensor,
};
use eredu_checkpoint::{BlockFp8Format, BlockFp8ScaleEncoding, LinearFormat};
use eredu_nn::parameter_values::{decode_parameter, ParameterDecoding, WorkspaceParameterDecoding};
use safemlx::{Array, Device, DeviceType};

struct FailureContext(usize, bool);
impl Drop for FailureContext {
    fn drop(&mut self) {
        if std::thread::panicking() {
            eprintln!(
                "standalone block FP8 source: case {}, Metal={}",
                self.0, self.1
            );
        }
    }
}

#[test]
fn original_cpu_block_fp8_decode_preserves_scale_and_partition_origins() {
    if crate::tests::support::native_process::enter("cpu-standalone-block-fp8") {
        run(false);
    }
}
#[test]
fn original_metal_block_fp8_decode_preserves_scale_and_partition_origins() {
    if crate::tests::support::native_process::enter("metal-standalone-block-fp8") {
        run(true);
    }
}
fn run(metal: bool) {
    let startup = crate::tests::support::test_utils::initialize_original_sources();
    let ordinary = MlxMetalWorkspaceMechanisms::current_host().unwrap();
    let choice =
        MlxCpuMatmulMechanism::select(eredu_nn::CpuMatmulImplementation::Float32Tiles).unwrap();
    let cpu = MlxCpuWorkspaceMechanisms::new(ordinary.allocation(), choice);
    let streams = if metal {
        PreparedExecutionStreams::for_factory(&startup)
    } else {
        PreparedExecutionStreams::for_cpu_factory_with_matmul(&startup, choice)
    }
    .unwrap()
    .unwrap();
    let backend = MlxBackend::for_prepared_execution_plan(
        streams,
        MlxDeviceIdentity::from_realized_device(
            &Device::new(
                if metal {
                    DeviceType::Gpu
                } else {
                    DeviceType::Cpu
                },
                0,
            ),
            metal.then_some(MlxAcceleratorFamily::Metal),
        )
        .unwrap(),
    );
    for case in 0..4 {
        let _failure_context = FailureContext(case, metal);
        let grouped = case % 2 != 0;
        let partitioned = case >= 2;
        let groups = if grouped { 2usize } else { 1 };
        let rows = if partitioned { 6usize } else { 3 };
        let columns = 132usize;
        let row_layout = if partitioned {
            eredu_nn::LinearRowLayout::equal_partitions(2).unwrap()
        } else {
            eredu_nn::LinearRowLayout::Contiguous
        };
        let partitions = row_layout.partitions();
        let row_width = rows / partitions;
        let scale_rows = partitions;
        let shape = if grouped {
            vec![groups as i32, rows as i32, columns as i32]
        } else {
            vec![rows as i32, columns as i32]
        };
        let scale_shape = if grouped {
            vec![groups as i32, scale_rows as i32, 2]
        } else {
            vec![scale_rows as i32, 2]
        };
        let codes = [0x38u8, 0x40, 0x30, 0xb8, 0xc0, 0x48];
        let decoded = [1.0f32, 2.0, 0.5, -1.0, -2.0, 4.0];
        let packed = (0..groups * rows * columns)
            .map(|i| codes[i % codes.len()])
            .collect::<Vec<_>>();
        let exponents = (0..groups * scale_rows * 2)
            .map(|i| 124u8 + (i % 6) as u8)
            .collect::<Vec<_>>();
        let floating_scales = exponents
            .iter()
            .map(|&e| 2f32.powi(i32::from(e) - 127))
            .collect::<Vec<_>>();
        let dtype = match case {
            0 => Dtype::Float32,
            1 => Dtype::Bfloat16,
            2 => Dtype::Uint8,
            _ => Dtype::Float16,
        };
        let weight = MlxTensor::from_array(Array::from_slice(&packed, &shape));
        let scale = MlxTensor::from_array(match dtype {
            Dtype::Uint8 => Array::from_slice(&exponents, &scale_shape),
            Dtype::Float16 => Array::from_slice(
                &floating_scales
                    .iter()
                    .copied()
                    .map(half::f16::from_f32)
                    .collect::<Vec<_>>(),
                &scale_shape,
            ),
            Dtype::Bfloat16 => Array::from_slice(
                &floating_scales
                    .iter()
                    .copied()
                    .map(half::bf16::from_f32)
                    .collect::<Vec<_>>(),
                &scale_shape,
            ),
            Dtype::Float32 => Array::from_slice(&floating_scales, &scale_shape),
            _ => unreachable!(),
        });
        weight.as_array().evaluated().unwrap();
        scale.as_array().evaluated().unwrap();
        let expected = (0..packed.len())
            .map(|i| {
                let group = i / (rows * columns);
                let row = i / columns % rows;
                let col = i % columns;
                let si = (group * scale_rows + row / row_width) * 2 + col / 128;
                decoded[i % decoded.len()] * floating_scales[si]
            })
            .collect::<Vec<_>>();
        let context = if metal {
            WorkspaceContext::new(ordinary)
        } else {
            WorkspaceContext::new(cpu)
        };
        let symbolic_weight = WorkspaceTensor::existing(
            context.layout(&shape, WorkspaceDtype::Uint8).unwrap(),
            &context,
        )
        .unwrap();
        let representation = match dtype {
            Dtype::Float32 => Some(WorkspaceFloatingType::Float32),
            Dtype::Float16 => Some(WorkspaceFloatingType::Float16),
            Dtype::Bfloat16 => Some(WorkspaceFloatingType::Bfloat16),
            _ => None,
        };
        let symbolic_scale = WorkspaceTensor::existing(
            context
                .layout(
                    &scale_shape,
                    if dtype == Dtype::Uint8 {
                        WorkspaceDtype::Uint8
                    } else {
                        WorkspaceDtype::Float32
                    },
                )
                .unwrap()
                .with_representation(representation.map(|d| WorkspaceRepresentation::new(d, true))),
            &context,
        )
        .unwrap();
        context
            .begin_state_span([&symbolic_weight, &symbolic_scale])
            .unwrap();
        let decoding = ParameterDecoding {
            format: LinearFormat::E4M3BlockFp8(BlockFp8Format {
                block_rows: 128,
                block_columns: 128,
                scale_encoding: if dtype == Dtype::Uint8 {
                    BlockFp8ScaleEncoding::Ue8m0
                } else {
                    BlockFp8ScaleEncoding::FloatingPoint
                },
            }),
            row_layout,
        };
        let output = decode_parameter(
            WorkspaceParameterDecoding::new(
                &symbolic_weight,
                Some(&symbolic_scale),
                None,
                &shape,
                &context,
            )
            .unwrap(),
            decoding,
        )
        .unwrap();
        let report = context.finish_report(&[output]).unwrap();
        assert!(
            report.unpriced_operations.is_empty(),
            "case {case}, Metal {metal}: {report:?}"
        );
        assert!(
            report.unpriced_host_operations.is_empty(),
            "case {case}, Metal {metal}: {report:?}"
        );
        let recipe = SpeculativeNumericalRecipe::inspect_completed_outputs(
            &report,
            1,
            if metal {
                ResidentExecutionMechanisms::Metal(ordinary)
            } else {
                ResidentExecutionMechanisms::Cpu { ordinary, cpu }
            },
            &context,
        )
        .unwrap();
        super::super::test_execution::run(
            recipe,
            &backend,
            &[&weight, &scale],
            |stream| {
                let result = crate::backend::nn::fp8::dequantize_with_row_layout(
                    weight.as_array(),
                    scale.as_array(),
                    row_layout,
                    stream,
                )
                .unwrap();
                MlxTensor::from_array(result.as_dtype(Dtype::Float32, stream).unwrap())
            },
            |actual| {
                let actual = actual.as_array().evaluated().unwrap();
                let values = actual.as_slice::<f32>();
                assert_eq!(values.len(), expected.len());
                for (index, (&value, &expected)) in values.iter().zip(&expected).enumerate() {
                    assert!(
                        (value - expected).abs() < 1e-6,
                        "case {case}, Metal {metal}, index {index}: {value} != {expected}"
                    );
                }
            },
        );
    }
}
