use super::*;
use crate::capture::{FundedCaptureError, ScheduledCaptureBackend};
use crate::working_memory::{
    CaptureHistogramClaim, CapturePrefillFragmentClaim, CaptureSummaryClaim,
    ClaimedCaptureHistogram, ClaimedCaptureSummary,
};
use std::cell::Cell;

pub(super) struct Tensor {
    pub(super) shape: [usize; 3],
    pub(super) values: Vec<f32>,
}
#[derive(Debug, thiserror::Error)]
pub(super) enum NativeError {
    #[error("native source shape differs")]
    Shape,
    #[error("native sentinel after partial write")]
    Partial,
}
#[derive(Default)]
pub(super) struct Native {
    pub(super) validations: Cell<usize>,
    pub(super) raw: usize,
    pub(super) summary: usize,
    pub(super) histogram: usize,
    pub(super) fail: bool,
    pub(super) wrong_usage: bool,
}
type Error = FundedCaptureError<NativeError>;
impl Native {
    fn validate(&self, value: &Tensor, shape: &[usize]) -> Result<TensorDtype, Error> {
        self.validations.set(self.validations.get() + 1);
        if shape != value.shape {
            return Err(FundedCaptureError::Backend(NativeError::Shape));
        }
        Ok(TensorDtype::F16)
    }
    fn usage(&self) -> CaptureUsage {
        let mut usage = estimate(0).capture;
        if self.wrong_usage {
            usage.host_bytes += 1;
        }
        usage
    }
}
fn selected(value: &Tensor, starts: &[u64], ends: &[u64], strides: &[u64]) -> Vec<f32> {
    let mut out = vec![];
    for a in (starts[0]..ends[0]).step_by(strides[0] as usize) {
        for b in (starts[1]..ends[1]).step_by(strides[1] as usize) {
            for c in (starts[2]..ends[2]).step_by(strides[2] as usize) {
                out.push(
                    value.values
                        [(a as usize * value.shape[1] + b as usize) * value.shape[2] + c as usize],
                );
            }
        }
    }
    out
}
impl ScheduledCaptureBackend for Native {
    type Tensor = Tensor;
    type Error = NativeError;
    fn validate_source(
        &self,
        value: &Tensor,
        g: &CaptureTensorGeometry<'_>,
    ) -> Result<TensorDtype, NativeError> {
        self.validations.set(self.validations.get() + 1);
        if g.source_shape() != value.shape {
            return Err(NativeError::Shape);
        }
        Ok(TensorDtype::F16)
    }
    fn estimate(
        &self,
        _: &Tensor,
        _: &CaptureTensorGeometry<'_>,
    ) -> Result<CaptureUsage, CaptureError> {
        Ok(self.usage())
    }
    fn transform(
        &mut self,
        value: &Tensor,
        claim: CaptureTensorClaim<'_, '_>,
    ) -> Result<ClaimedCaptureTensor, NativeError> {
        self.raw += 1;
        let g = claim.geometry();
        let mut values = selected(value, g.starts(), g.ends(), g.strides());
        values.truncate(g.elements());
        let mut writer = claim.prepare().unwrap();
        for value in values {
            writer.push_f32(value).unwrap();
            if self.fail {
                return Err(NativeError::Partial);
            }
        }
        Ok(writer.finish().unwrap())
    }
    fn validate_prefill_source(
        &self,
        value: &Tensor,
        fragment: &CapturePrefillFragment<'_, '_>,
    ) -> Result<TensorDtype, Error> {
        self.validate(value, fragment.source_shape())
    }
    fn estimate_prefill(
        &self,
        _: &CaptureTensorGeometry<'_>,
    ) -> Result<CaptureUsage, CaptureError> {
        Ok(self.usage())
    }
    fn transform_prefill_fragment(
        &mut self,
        value: &Tensor,
        claim: CapturePrefillFragmentClaim<'_, '_, '_, '_>,
    ) -> Result<(), Error> {
        self.raw += 1;
        let fragment = claim.fragment();
        let mut writer = claim.prepare()?;
        for mapping in fragment.mappings() {
            writer.push_f32(value.values[mapping.source_index()])?;
            if self.fail {
                return Err(FundedCaptureError::Backend(NativeError::Partial));
            }
        }
        writer.finish()?;
        Ok(())
    }
    fn validate_summary_source(
        &self,
        value: &Tensor,
        g: &CaptureSummaryGeometry<'_>,
    ) -> Result<TensorDtype, Error> {
        self.validate(value, g.source_shape())
    }
    fn estimate_summary(
        &self,
        _: &CaptureSummaryGeometry<'_>,
    ) -> Result<CaptureUsage, CaptureError> {
        Ok(self.usage())
    }
    fn estimate_prefill_summary(
        &self,
        _: &CapturePrefillTransformPlan<'_>,
    ) -> Result<CaptureUsage, CaptureError> {
        Ok(self.usage())
    }
    fn transform_summary(
        &mut self,
        value: &Tensor,
        claim: CaptureSummaryClaim<'_, '_>,
    ) -> Result<ClaimedCaptureSummary, Error> {
        self.summary += 1;
        let g = claim.geometry();
        let values = selected(value, g.starts(), g.ends(), g.strides());
        Ok(claim.finish_partition(crate::capture::partition::summarize_f32(&values))?)
    }
    fn validate_histogram_source(
        &self,
        value: &Tensor,
        g: &CaptureHistogramGeometry<'_>,
    ) -> Result<TensorDtype, Error> {
        self.validate(value, g.source_shape())
    }
    fn estimate_histogram(
        &self,
        _: &CaptureHistogramGeometry<'_>,
    ) -> Result<CaptureUsage, CaptureError> {
        Ok(self.usage())
    }
    fn estimate_prefill_histogram(
        &self,
        _: &CapturePrefillTransformPlan<'_>,
    ) -> Result<CaptureUsage, CaptureError> {
        Ok(self.usage())
    }
    fn transform_histogram(
        &mut self,
        value: &Tensor,
        claim: CaptureHistogramClaim<'_, '_>,
    ) -> Result<ClaimedCaptureHistogram, Error> {
        self.histogram += 1;
        let g = claim.geometry();
        let values = selected(value, g.starts(), g.ends(), g.strides());
        let mut writer = claim.prepare()?;
        writer.fill_partition_sum(&values)?;
        Ok(writer.finish_partition()?)
    }
}

#[test]
fn local_partition_callback_uses_existing_workers_and_retains_partial_failure() {
    use super::super::prefill::{inference, projected_receipt, projected_source};
    for (transform, combination, mode) in [
        (
            CaptureTransform::Slice,
            PartitionCaptureCombination::Disjoint,
            0,
        ),
        (
            CaptureTransform::Summary,
            PartitionCaptureCombination::Disjoint,
            0,
        ),
        (
            CaptureTransform::Histogram {
                edges: vec![0.0, 15.0, 200.0],
            },
            PartitionCaptureCombination::Disjoint,
            0,
        ),
        (
            CaptureTransform::Summary,
            PartitionCaptureCombination::SumF64ToF32,
            0,
        ),
        (
            CaptureTransform::Slice,
            PartitionCaptureCombination::Disjoint,
            1,
        ),
        (
            CaptureTransform::Slice,
            PartitionCaptureCombination::Disjoint,
            2,
        ),
    ] {
        let sum = combination == PartitionCaptureCombination::SumF64ToF32;
        let source = projected_source(transform.clone());
        let (funding, _, _, _) = funding();
        let transport = Ranks {
            local: 0,
            bytes: RefCell::new(std::array::from_fn(|_| None)),
            funding: funding.clone(),
            reject: false,
            calls: RefCell::new(vec![]),
        };
        let mut quota = CaptureLedger::new(source.admission());
        quota.begin_step();
        let mut receipt = projected_receipt(&source, combination, &funding, &mut quota);
        let geometry: Vec<_> = receipt
            .producers()
            .flat_map(|(rank, p)| {
                p.fragments()
                    .iter()
                    .enumerate()
                    .map(move |(i, g)| (rank, i, p.local_shape().to_vec(), g.local().clone()))
            })
            .collect();
        let raw = CaptureTransform::Slice;
        let sources: Vec<_> = geometry
            .iter()
            .map(
                |(rank, fragment, shape, slice)| PartitionCaptureFragmentSource {
                    producer: *rank,
                    fragment: *fragment,
                    local_shape: shape,
                    local_slice: slice,
                    transform: if sum { &raw } else { &transform },
                    dtype: TensorDtype::F16,
                    estimate: estimate(*rank),
                },
            )
            .collect();
        let allowance = PreparedPartitionFragmentAllowance::prepare(
            &transport,
            &mut receipt,
            &sources,
            &funding,
            &mut quota,
        )
        .unwrap();
        let ranks: Vec<_> = receipt
            .producers()
            .map(|(producer, _)| PartitionCaptureRankSource {
                producer,
                dtype: Some(TensorDtype::F16),
            })
            .collect();
        let host = PartitionFragmentHostPlan::prepare_prefill(&receipt, inference()).unwrap();
        let h = host.initialization_peak_bytes();
        let pool = capture_test_ledger(h, 0).unwrap();
        let (reservation, run) = fresh(&pool, h);
        let bank = run
            .prepare_partition_fragments(&reservation, host, allowance)
            .unwrap();
        let delivery =
            PreparedPartitionFragmentDelivery::prepare(&transport, receipt, bank, &ranks, &funding)
                .unwrap();
        let spent = quota.total();
        let (continuation, hook) = delivery.into_local_hook();
        let mut next = Some(crate::capture::partition::PartitionCaptureLocalHook::new(
            hook,
        ));
        let mut native = Native {
            fail: mode == 1,
            wrong_usage: mode == 2,
            ..Default::default()
        };
        let width = if sum { 7 } else { 4 };
        let offset = if sum { 0 } else { 3 };
        let mut failure = None;
        for k in 0..3 {
            let value = Tensor {
                shape: [2, 1, width],
                values: (0..2)
                    .flat_map(|head| {
                        (0..width).map(move |column| {
                            head as f32 * 100.0 + k as f32 * 10.0 + (column + offset) as f32
                        })
                    })
                    .collect(),
            };
            match next
                .take()
                .unwrap()
                .observe(&mut native, &value, inference(), k)
            {
                Ok(hook) => next = Some(hook),
                Err(error) => {
                    failure = Some(error);
                    break;
                }
            }
        }
        assert_eq!(quota.total(), spent);
        assert!(transport.calls.borrow().is_empty());
        if mode != 0 {
            assert!(failure.is_some());
            assert_eq!(native.raw, usize::from(mode == 1));
            if mode == 1 {
                assert!(
                    failure
                        .as_ref()
                        .unwrap()
                        .to_string()
                        .contains("native sentinel")
                );
            }
            drop(continuation);
            drop(run);
            drop(reservation);
            assert!(ledger(&pool).0 > 0);
            drop(failure);
            assert_eq!(ledger(&pool).0, 0);
            continue;
        }
        assert_eq!(native.validations.get(), 3);
        let mut hook = next.take().unwrap().into_inner();
        let (receipt, bank) = hook.parts();
        bank.finish_local_prefill(receipt, 0).unwrap();
        let values: Vec<_> = (0..2)
            .flat_map(|head| {
                (1..3).flat_map(move |row| {
                    [1, 3, 5]
                        .into_iter()
                        .filter(move |c| sum || *c >= 3)
                        .map(move |column| head as f32 * 100.0 + row as f32 * 10.0 + column as f32)
                })
            })
            .collect();
        match bank.value(0, 0).unwrap() {
            PartitionFragmentValue::Routed(_) => panic!("dense fixture returned sparse assembly"),
            PartitionFragmentValue::Tensor(value) => {
                assert_eq!((native.raw, native.summary, native.histogram), (2, 0, 0));
                assert_eq!(
                    value.observation().data(),
                    &TensorObservationData::F32(values)
                );
            }
            PartitionFragmentValue::Summary(value) => {
                assert_eq!((native.raw, native.summary, native.histogram), (0, 2, 0));
                let expected = crate::capture::partition::summarize_f32(&values);
                let actual = value.observation();
                assert_eq!(
                    (
                        actual.elements,
                        actual.finite,
                        actual.non_finite,
                        actual.min,
                        actual.max
                    ),
                    (
                        expected.elements,
                        expected.finite,
                        expected.non_finite,
                        expected.min,
                        expected.max
                    )
                );
                assert!((actual.mean.unwrap() - expected.mean.unwrap()).abs() < 1e-12);
                assert!((actual.rms.unwrap() - expected.rms.unwrap()).abs() < 1e-12);
            }
            PartitionFragmentValue::Histogram(value) => {
                assert_eq!((native.raw, native.summary, native.histogram), (0, 0, 2));
                let CaptureTransform::Histogram { edges } = &transform else {
                    unreachable!()
                };
                let mut expected = value.observation().clone();
                expected.counts.fill(0);
                expected.below = 0;
                expected.above = 0;
                expected.non_finite = 0;
                assert_eq!(&expected.edges, edges);
                crate::capture::partition::fill_histogram_f32(&values, &mut expected).unwrap();
                assert_eq!(value.observation(), &expected);
            }
        }
        let delivery = continuation.resume(hook).unwrap();
        drop(delivery);
        drop(run);
        drop(reservation);
        assert_eq!(ledger(&pool).0, 0);
    }
}

mod invocation;
