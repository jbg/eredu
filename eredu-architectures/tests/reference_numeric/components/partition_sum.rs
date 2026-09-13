//! Measure actual TP terms through ordinary receipt encoding and host assembly.
use super::*;
use eredu_core::{capture::*, TensorObservation, TensorObservationData};
use eredu_runtime::capture::partition::*;

struct Collector;
fn selected(value: &NumericTensor, slice: &ResolvedCaptureSlice) -> NumericTensor {
    let count = slice.shape.iter().product::<u64>() as usize;
    let mut data = Vec::with_capacity(count);
    for ordinal in 0..count {
        let mut remaining = ordinal as u64;
        let mut index = 0;
        let mut stride = 1;
        for axis in (0..slice.shape.len()).rev() {
            let coordinate =
                slice.starts[axis] + remaining % slice.shape[axis] * slice.strides[axis];
            remaining /= slice.shape[axis];
            index += coordinate as usize * stride;
            stride *= value.shape[axis] as usize;
        }
        data.push(value.data[index]);
    }
    NumericTensor::new(
        slice.shape.iter().map(|d| *d as i32).collect::<Vec<_>>(),
        data,
    )
}
impl CaptureBackend for Collector {
    type Tensor = NumericTensor;
    type Error = Error;
    fn shape(&self, value: &NumericTensor) -> Result<Vec<u64>, Error> {
        Ok(value.shape.iter().map(|d| *d as u64).collect())
    }
    fn source_dtype(&self, _: &NumericTensor) -> Option<eredu_core::checkpoint::TensorDtype> {
        Some(eredu_core::checkpoint::TensorDtype::F32)
    }
    fn estimate(
        &self,
        _: &NumericTensor,
        _: &CaptureSelection,
        slice: &ResolvedCaptureSlice,
    ) -> Result<CaptureUsage, CaptureError> {
        let count = slice.shape.iter().product::<u64>();
        Ok(CaptureUsage {
            captures: 1,
            retained_bytes: count * 4,
            host_bytes: count * 4,
            encoded_bytes: 4096 + count * 32,
        })
    }
    fn transform(
        &mut self,
        value: &NumericTensor,
        selection: &CaptureSelection,
        slice: &ResolvedCaptureSlice,
    ) -> Result<CapturePayload, Error> {
        assert!(matches!(selection.transform, CaptureTransform::Slice));
        let value = selected(value, slice);
        Ok(CapturePayload::Tensor(
            TensorObservation::new(
                slice.shape.iter().map(|d| *d as usize).collect(),
                TensorObservationData::F32(value.data),
            )
            .unwrap(),
        ))
    }
}

pub(super) fn verify<'a>(
    plan: &AdmittedCapturePlan,
    receipt: PartitionCaptureReceiptPlan,
    source: impl Fn(usize) -> &'a NumericTensor,
    expected: &NumericTensor,
) {
    let context = receipt.context();
    let index = context.selection_index;
    let mut ledger = CaptureLedger::new(plan);
    let mut collector = Collector;
    let mut encoded = Vec::new();
    for (rank, projection) in receipt.producers() {
        let value = source(rank);
        let fragments = (0..projection.fragments().len())
            .map(|fragment_index| {
                capture_sum_fragment(
                    &mut collector,
                    value,
                    PartitionCaptureRequest {
                        invocation: None,
                        plan,
                        selection_index: index,
                        phase: context.phase,
                        prediction: context.prediction,
                        projection,
                        fragment_index,
                        producer_rank: rank,
                    },
                    &mut ledger,
                )
                .unwrap()
            })
            .collect();
        encoded.push((
            rank,
            receipt
                .encode_producer(rank, collector.source_dtype(value), fragments, &mut ledger)
                .unwrap(),
        ));
    }
    let shape = collector.shape(expected).unwrap();
    let slice = resolve_slice(
        &plan.points()[index],
        &plan.plan().selections[index],
        &shape,
    )
    .unwrap();
    let expected = selected(expected, &slice);
    let mut delivery = receipt.into_delivery();
    for (rank, bytes) in encoded.into_iter().rev() {
        delivery.receive(rank, &bytes, &mut ledger).unwrap();
    }
    let complete = delivery.finish(&mut ledger).unwrap();
    assert_eq!(
        complete.capture().combination(),
        PartitionCaptureCombination::SumF64ToF32
    );
    let Some(CapturePayload::Tensor(value)) = &complete.capture().record().payload else {
        panic!("complete summed payload")
    };
    let TensorObservationData::F32(values) = value.data() else {
        panic!("declared F32 assembly")
    };
    let actual = NumericTensor::new(
        value.shape().iter().map(|d| *d as i32).collect::<Vec<_>>(),
        values.clone(),
    );
    assert_tensor_close(
        &actual,
        &expected,
        &format!(
            "summed actual TP writes: {}",
            plan.plan().selections[index].path
        ),
    );
}
