use super::*;
use crate::{
    MlxTensor,
    backend::{
        MlxBackend, MlxDeviceIdentity, managed_memory::gpu_stream::PreparedExecutionStreams,
        nn::shared::MlxNeuralBackend,
    },
};
use eredu_nn::{GatedProductPolicy, NeuralBackend, Tensor};
use safemlx::{Array, Device, DeviceType};
const TYPES: [WorkspaceFloatingType; 3] = [
    WorkspaceFloatingType::Float32,
    WorkspaceFloatingType::Float16,
    WorkspaceFloatingType::Bfloat16,
];
fn round(x: f32, dtype: WorkspaceFloatingType) -> f32 {
    match dtype {
        WorkspaceFloatingType::Float32 => x,
        WorkspaceFloatingType::Float16 => half::f16::from_f32(x).to_f32(),
        WorkspaceFloatingType::Bfloat16 => half::bf16::from_f32(x).to_f32(),
    }
}
fn tensor(values: &[f32], shape: &[i32], dtype: WorkspaceFloatingType) -> MlxTensor {
    MlxTensor::from_array(match dtype {
        WorkspaceFloatingType::Float32 => Array::from_slice(values, shape),
        WorkspaceFloatingType::Float16 => Array::from_slice(
            &values
                .iter()
                .map(|&x| half::f16::from_f32(x))
                .collect::<Vec<_>>(),
            shape,
        ),
        WorkspaceFloatingType::Bfloat16 => Array::from_slice(
            &values
                .iter()
                .map(|&x| half::bf16::from_f32(x))
                .collect::<Vec<_>>(),
            shape,
        ),
    })
}
fn values(tensor: &MlxTensor) -> Vec<f32> {
    let value = tensor.as_array().evaluated().unwrap();
    match tensor.as_array().dtype() {
        Dtype::Float32 => value.try_to_vec::<f32>().unwrap(),
        Dtype::Float16 => value
            .try_to_vec::<half::f16>()
            .unwrap()
            .into_iter()
            .map(|x| x.to_f32())
            .collect(),
        Dtype::Bfloat16 => value
            .try_to_vec::<half::bf16>()
            .unwrap()
            .into_iter()
            .map(|x| x.to_f32())
            .collect(),
        _ => panic!("unexpected activation precision"),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Operation {
    Gelu,
    GeluProduct,
    SiluProduct,
    Sigmoid,
}
impl Operation {
    fn gated(self) -> bool {
        matches!(self, Self::GeluProduct | Self::SiluProduct)
    }
}
#[test]
fn cpu_gelu_sigmoid_and_mixed_gated_products_match_independent_arithmetic_and_retire() {
    let pool = crate::tests::support::test_utils::initialize_original_sources();
    let ordinary = MlxMetalWorkspaceMechanisms::current_host().unwrap();
    let selected =
        MlxCpuMatmulMechanism::select(eredu_nn::CpuMatmulImplementation::Float32Tiles).unwrap();
    let cpu = MlxCpuWorkspaceMechanisms::new(ordinary.allocation(), selected);
    let streams = PreparedExecutionStreams::for_cpu_factory_with_matmul(&pool, selected)
        .unwrap()
        .unwrap();
    let backend = MlxBackend::for_prepared_execution_plan(
        streams,
        MlxDeviceIdentity::from_realized_device(&Device::new(DeviceType::Cpu, 0), None).unwrap(),
    );
    let mut cases = 0;
    for shape in [&[][..], &[1], &[17], &[2, 3, 8], &[1, 2, 3, 19]] {
        let count: usize = shape.iter().map(|&n| n as usize).product();
        for gate_type in TYPES {
            for up_type in TYPES {
                for operation in [
                    Operation::Gelu,
                    Operation::GeluProduct,
                    Operation::SiluProduct,
                    Operation::Sigmoid,
                ] {
                    if !operation.gated() && up_type != WorkspaceFloatingType::Float32 {
                        continue;
                    }
                    let gate_data = (0..count)
                        .map(|i| round(((i * 11 + 7) % 67) as f32 * 0.113 - 3.729, gate_type))
                        .collect::<Vec<_>>();
                    let up_data = (0..count)
                        .map(|i| round((i % 13) as f32 * 0.17 - 1.02, up_type))
                        .collect::<Vec<_>>();
                    let gate = tensor(&gate_data, shape, gate_type);
                    let up = tensor(&up_data, shape, up_type);
                    let context = WorkspaceContext::new(cpu);
                    let source = |dtype| {
                        WorkspaceTensor::existing(
                            context
                                .layout(shape, WorkspaceDtype::Float32)
                                .unwrap()
                                .with_representation(Some(WorkspaceRepresentation::new(
                                    dtype, true,
                                ))),
                            &context,
                        )
                        .unwrap()
                    };
                    let wg = source(gate_type);
                    let wu = source(up_type);
                    let policy = if operation == Operation::GeluProduct {
                        GatedProductPolicy::ordinary_gelu_approximate()
                    } else {
                        GatedProductPolicy::ordinary_silu()
                    };
                    context.begin_span();
                    let output = match operation {
                        Operation::Gelu => {
                            WorkspaceBackend::gelu_approximate(wg, &context).unwrap()
                        }
                        Operation::Sigmoid => WorkspaceBackend::sigmoid(wg, &context).unwrap(),
                        _ => WorkspaceBackend::gated_product(wg, wu, policy, &context).unwrap(),
                    };
                    let dtype = match operation {
                        Operation::Sigmoid => gate_type,
                        Operation::SiluProduct => {
                            super::super::super::representation::promote(gate_type, up_type)
                        }
                        _ => WorkspaceFloatingType::Float32,
                    };
                    assert_eq!(output.layout().representation().unwrap().dtype(), dtype);
                    let report = context.finish_report(&[output]).unwrap();
                    let recipe = SpeculativeNumericalRecipe::inspect_cpu_equations(
                        &report, ordinary, cpu, &context,
                    )
                    .unwrap();
                    let expected = gate_data
                        .iter()
                        .enumerate()
                        .map(|(i, &x)| {
                            let activated = if operation == Operation::SiluProduct {
                                round(x / (1.0 + (-x).exp()), gate_type)
                            } else if operation == Operation::Sigmoid {
                                round(1.0 / (1.0 + (-x).exp()), gate_type)
                            } else {
                                let cube = round(x.powf(3.0), gate_type);
                                let inner =
                                    (2.0 / std::f32::consts::PI).sqrt() * (x + 0.044715 * cube);
                                (0.5 * x) * (1.0 + inner.tanh())
                            };
                            if !operation.gated() {
                                activated
                            } else {
                                round(activated * up_data[i], dtype)
                            }
                        })
                        .collect::<Vec<_>>();
                    let inputs = if !operation.gated() {
                        vec![&gate]
                    } else {
                        vec![&gate, &up]
                    };
                    super::super::test_execution::run(
                        recipe,
                        &backend,
                        &inputs,
                        |stream| match operation {
                            Operation::Gelu => {
                                MlxNeuralBackend::gelu_approximate(gate.clone(), stream).unwrap()
                            }
                            Operation::Sigmoid => {
                                MlxNeuralBackend::sigmoid(gate.clone(), stream).unwrap()
                            }
                            _ => MlxNeuralBackend::gated_product(
                                gate.clone(),
                                up.clone(),
                                policy,
                                stream,
                            )
                            .unwrap(),
                        },
                        |actual| {
                            let native = match dtype {
                                WorkspaceFloatingType::Float32 => Dtype::Float32,
                                WorkspaceFloatingType::Float16 => Dtype::Float16,
                                WorkspaceFloatingType::Bfloat16 => Dtype::Bfloat16,
                            };
                            assert_eq!(actual.as_array().dtype(), native);
                            let actual = values(actual);
                            for (&actual, &expected) in actual.iter().zip(&expected) {
                                let tolerance = if dtype == WorkspaceFloatingType::Float32 {
                                    2e-5
                                } else {
                                    0.012
                                };
                                assert!(
                                    (actual - expected).abs()
                                        <= tolerance * expected.abs().max(1.0),
                                    "shape={shape:?} gate={gate_type:?} up={up_type:?} operation={operation:?}: {actual} != {expected}"
                                );
                            }
                        },
                    );
                    assert_eq!(values(&gate), gate_data);
                    assert_eq!(values(&up), up_data);
                    let mut missing = report;
                    missing.operations[0].inputs[0] = missing.operations[0].inputs[0]
                        .clone()
                        .with_representation(None);
                    assert!(cpu.plan(missing.operations[0].as_view()).unwrap().is_none());
                    cases += 1;
                }
            }
        }
    }
    assert_eq!(cases, 120);
}
