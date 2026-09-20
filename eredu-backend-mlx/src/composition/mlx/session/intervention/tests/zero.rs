use super::*;
use crate::backend::nn::workspace::{
    MlxCpuMatmulMechanism, MlxCpuWorkspaceMechanisms, MlxMetalWorkspaceMechanisms,
    SpeculativeNumericalRecipe,
};
use eredu_nn::workspace::{
    WorkspaceContext, WorkspaceDtype, WorkspaceFloatingType, WorkspaceMechanisms,
    WorkspaceRepresentation, WorkspaceTensor,
};

#[test]
fn zero_interventions_have_complete_cpu_sources_and_preserve_unselected_values() {
    let _sources = crate::tests::support::test_utils::initialize_original_sources();
    let ordinary = MlxMetalWorkspaceMechanisms::current_host().unwrap();
    let cpu = MlxCpuWorkspaceMechanisms::new(ordinary.allocation(),
        MlxCpuMatmulMechanism::select(eredu_nn::CpuMatmulImplementation::Float32Tiles).unwrap());
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    for (dtype, floating) in [
        (InterventionDtype::Float32, WorkspaceFloatingType::Float32),
        (InterventionDtype::Float16, WorkspaceFloatingType::Float16),
        (InterventionDtype::Bfloat16, WorkspaceFloatingType::Bfloat16),
    ] {
        for shape in [&[7][..], &[2, 7], &[1, 2, 7], &[1, 1, 2, 7]] {
            let source_shape: Vec<u64> = shape.iter().map(|&n| n as u64).collect();
            let count = shape.iter().product::<i32>() as usize;
            for partial in [false, true] {
                let mut slice = ResolvedCaptureSlice {
                    starts: vec![0; shape.len()], ends: source_shape.clone(),
                    strides: vec![1; shape.len()], shape: source_shape.clone(),
                };
                if partial {
                    let last = shape.len() - 1;
                    slice.starts[last] = 1;
                    slice.strides[last] = 2;
                    slice.shape[last] = 3;
                }
                let action = InterventionAction::Zero { dtype };
                let program = PreparedStaticActivation::new(&action, &slice, &source_shape, dtype).unwrap();
                let context = WorkspaceContext::new(cpu);
                let source = WorkspaceTensor::existing(context.layout(shape, WorkspaceDtype::Float32)
                    .unwrap().with_representation(Some(WorkspaceRepresentation::new(floating, true))),
                    &context).unwrap();
                context.begin_state_span([&source]).unwrap();
                let mut roots = Vec::new();
                let output = program.trace(&source, &context, &mut roots).unwrap();
                assert_eq!(output.shape(), shape);
                assert_eq!(output.layout().representation().unwrap().dtype(), floating);
                let report = context.finish_report(&roots).unwrap();
                for operation in &report.operations {
                    assert!(cpu.operation_bound(operation).unwrap().is_some(), "{operation:?}");
                }
                let population = program.population().unwrap();
                SpeculativeNumericalRecipe::inspect_cpu_capture(&report, roots.len(),
                    population.completions, ordinary, cpu, &context).unwrap();

                let values: Vec<f32> = (0..count).map(|i| (i + 1) as f32 * 0.25).collect();
                let array = match dtype {
                    InterventionDtype::Float32 => Array::from_slice(&values, shape),
                    InterventionDtype::Float16 => Array::from_slice(
                        &values.iter().map(|&v| half::f16::from_f32(v)).collect::<Vec<_>>(), shape),
                    InterventionDtype::Bfloat16 => Array::from_slice(
                        &values.iter().map(|&v| half::bf16::from_f32(v)).collect::<Vec<_>>(), shape),
                };
                let input = MlxTensor::from_array(array);
                let mut native = NativeCapture { stream: &stream, domain: None, partition: None };
                let actual = eredu_runtime::intervention::apply_activation(
                    &mut native, &input, &action, &slice).unwrap();
                let expected: Vec<f32> = values.iter().enumerate().map(|(i, &value)|
                    if !partial || [1, 3, 5].contains(&(i % 7)) { 0.0 } else { value }).collect();
                assert_eq!(actual.to_f32_vec(&stream).unwrap(), expected);
                assert_eq!(input.to_f32_vec(&stream).unwrap(), values);
            }
        }
    }
}
