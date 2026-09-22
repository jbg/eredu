use super::*;
use eredu_runtime::{
    input::host::{HostInputPart, PreparedHostInputPlan},
    working_memory::MemoryLedger,
};
use std::convert::Infallible;
#[derive(Default)]
struct Lowering {
    calls: Vec<(TensorDtype, Vec<usize>)>,
}
impl PreparedInputInspector<HostTensor> for Lowering {
    fn identity(&self, value: &HostTensor) -> Result<InputTensorIdentity, PreparedInputError> {
        value.identity()
    }
    fn i32_values(&self, _: &HostTensor) -> Result<Vec<i32>, eredu_core::CapabilityError> {
        panic!("lowering must borrow host values, not read them back")
    }
    fn bool_values(&self, _: &HostTensor) -> Result<Vec<bool>, eredu_core::CapabilityError> {
        panic!("lowering must borrow host values, not read them back")
    }
}
impl ProcessorMechanisms for Lowering {
    type Tensor = HostTensor;
    type Error = Infallible;
    fn tensor_u32(&mut self, v: &[u32], shape: &[usize]) -> Result<HostTensor, Infallible> {
        self.calls.push((TensorDtype::U32, shape.to_vec()));
        Ok(HostTensor::U32 {
            values: v.to_vec(),
            shape: shape.to_vec(),
        })
    }
    fn tensor_i32(&mut self, v: &[i32], shape: &[usize]) -> Result<HostTensor, Infallible> {
        self.calls.push((TensorDtype::I32, shape.to_vec()));
        Ok(HostTensor::I32 {
            values: v.to_vec(),
            shape: shape.to_vec(),
        })
    }
    fn tensor_f32(&mut self, v: &[f32], shape: &[usize]) -> Result<HostTensor, Infallible> {
        self.calls.push((TensorDtype::F32, shape.to_vec()));
        Ok(HostTensor::F32 {
            values: v.to_vec(),
            shape: shape.to_vec(),
        })
    }
    fn tensor_bool(
        &mut self,
        v: &[bool],
        shape: &[usize],
    ) -> Result<HostTensor, OptionalProcessorMechanism<Infallible>> {
        self.calls.push((TensorDtype::Bool, shape.to_vec()));
        Ok(HostTensor::Bool {
            values: v.to_vec(),
            shape: shape.to_vec(),
        })
    }
}
#[test]
fn original_host_source_and_legacy_boolean_processor_share_value_preserving_lowering() {
    let metadata = [(
        InputMetadataKey::PatchGrid,
        HostTensorView {
            shape: &[1, 3],
            values: HostTensorValues::I32(&[1, 2, 2]),
        },
    )];
    let parts = [
        HostInputPart {
            modality: InputModality::Text,
            kind: InputPayloadKind::TokenIds,
            payload: HostTensorView {
                shape: &[1, 2],
                values: HostTensorValues::U32(&[2, 7]),
            },
            metadata: &[],
            extents: &[],
        },
        HostInputPart {
            modality: InputModality::Image,
            kind: InputPayloadKind::Tensor,
            payload: HostTensorView {
                shape: &[4, 1],
                values: HostTensorValues::F32(&[0.5, -2., 1., 3.]),
            },
            metadata: &metadata,
            extents: &[InputExtent::PatchGrid {
                time: 1,
                height: 2,
                width: 2,
            }],
        },
    ];
    let pool = crate::memory_fixture::ledger(u64::MAX, 0).unwrap();
    let source = pool
        .compile_prepared_host_input(PreparedHostInputPlan::prepare(&parts).unwrap())
        .unwrap();
    let mut lower = Lowering::default();
    let result =
        lower_prepared_host_input::<_, Infallible, _, _>(source.parts(), &mut lower).unwrap();
    assert_eq!(
        lower.calls,
        [
            (TensorDtype::U32, vec![1, 2]),
            (TensorDtype::F32, vec![4, 1]),
            (TensorDtype::I32, vec![1, 3])
        ]
    );
    assert_eq!(
        result.parts()[0].payload().value(),
        &HostTensor::U32 {
            values: vec![2, 7],
            shape: vec![1, 2]
        }
    );
    assert_eq!(
        result.parts()[1].payload().value(),
        &HostTensor::F32 {
            values: vec![0.5, -2., 1., 3.],
            shape: vec![4, 1]
        }
    );
    assert_eq!(result.parts()[1].extents(), parts[1].extents);
    drop(source);
    assert_eq!(crate::memory_fixture::used(&pool).unwrap(), 0);
    let legacy = PreparedModelInput::new(
        vec![PreparedInputPart::new(
            InputModality::Audio,
            PreparedInputPayload::Tensor(HostTensor::F32 {
                values: vec![0.25, 1.5],
                shape: vec![1, 2],
            }),
            [(
                InputMetadataKey::AudioMask,
                HostTensor::Bool {
                    values: vec![true, false],
                    shape: vec![1, 2],
                },
            )],
        )
        .unwrap()],
        HostTensor::identity,
    )
    .unwrap();
    let result = lower_host_input::<_, Infallible>(legacy, &mut lower).unwrap();
    assert_eq!(
        result.parts()[0].metadata_value(InputMetadataKey::AudioMask),
        Some(&HostTensor::Bool {
            values: vec![true, false],
            shape: vec![1, 2]
        })
    );
    assert_eq!(lower.calls.last(), Some(&(TensorDtype::Bool, vec![1, 2])));
}
