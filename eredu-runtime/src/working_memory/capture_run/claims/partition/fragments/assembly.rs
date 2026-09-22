//! Actual paid fragment assembly into the caller's once-only final host claim.
use super::*;
use crate::capture::reduction::{CaptureSummaryError, Summary};
use crate::capture::{PrefixDestination, PrefixError};
use eredu_core::TensorObservationData;

#[derive(Debug, thiserror::Error)]
enum AssemblyCause {
    #[error("contiguous assembly source, geometry, charge or payload differs")]
    Source,
    #[error(transparent)]
    Memory(#[from] WorkingMemoryError),
    #[error(transparent)]
    Host(#[from] CaptureRunHostError),
    #[error(transparent)]
    Capture(#[from] CaptureError),
    #[error(transparent)]
    Prefix(#[from] PrefixError),
    #[error(transparent)]
    Summary(#[from] CaptureSummaryError),
    #[error(transparent)]
    Tensor(#[from] ScheduledCaptureTensorFailure),
    #[error(transparent)]
    SummaryDestination(#[from] CaptureSummaryFailure),
    #[error(transparent)]
    HistogramDestination(#[from] CaptureHistogramFailure),
    #[error(transparent)]
    Routed(#[from] CaptureRoutedHostError),
    #[error(transparent)]
    RoutedDestination(#[from] CaptureRoutedFailure),
}
/// Failure keeps every completed/partial source and both original host accounts.
/// No rank or destination can reuse a failed assembly attempt.
#[derive(Debug, thiserror::Error)]
#[error("contiguous capture assembly: {cause}")]
pub(crate) struct PartitionFragmentAssemblyError {
    #[source]
    cause: AssemblyCause,
    _bank: PreparedPartitionFragmentDestinations,
    _destination: CaptureTensorCustody,
}

pub(super) fn control_bytes() -> Option<usize> {
    let parts = [
        size_of::<PartitionFragmentAssemblyError>(),
        size_of::<AssemblyCause>(),
        size_of::<Result<(PartitionFragmentValue, CaptureUsage), PartitionFragmentAssemblyError>>(),
        size_of::<Result<(PartitionFragmentValue, CaptureUsage), AssemblyCause>>(),
        size_of::<(
            &mut PreparedPartitionFragmentDestinations,
            &PartitionCaptureReceiptPlan,
            PartitionFragmentDestination<'_, '_>,
            &TensorDtype,
        )>(),
        size_of::<(usize, usize, u64, CaptureUsage)>(),
        size_of::<CaptureUsage>() * 3,
        size_of::<CaptureQuota>(),
        size_of::<(
            &PartitionCaptureReceiptPlan,
            &CaptureSelection,
            &eredu_core::ObservationPoint,
            &[u64],
        )>(),
        size_of::<Option<CaptureUsage>>(),
        size_of::<Result<Option<CaptureUsage>, CaptureError>>(),
        size_of::<PrefixDestination<'_>>(),
        size_of::<Result<PrefixDestination<'_>, PrefixError>>(),
        size_of::<Summary>() * 2,
        size_of::<CaptureSummary>(),
        size_of::<CaptureTensorCustody>(),
        size_of::<Option<(usize, usize, CaptureTensorCustody)>>(),
        size_of::<Option<ClaimedCaptureTensor>>(),
        size_of::<ScheduledCaptureTensor<'_, '_>>(),
        size_of::<ScheduledCaptureHistogram<'_, '_>>(),
        size_of::<Result<ClaimedCaptureTensor, ScheduledCaptureTensorFinishError<'_, '_>>>(),
        size_of::<Result<ClaimedCaptureSummary, CaptureSummaryFailure>>(),
        size_of::<Result<ClaimedCaptureHistogram, CaptureHistogramFailure>>(),
        size_of::<std::slice::Iter<'_, Slot>>(),
        size_of::<std::slice::Iter<'_, f32>>(),
        size_of::<std::slice::Iter<'_, u64>>(),
        size_of::<(&Slot, usize, &[f32])>(),
        // Terms is the concrete iterator consumed by the ordinary sum worker.
        size_of::<Terms<'_>>(),
        size_of::<Option<&[f32]>>(),
        crate::capture::partition::numeric_control_bytes()?,
        routed::control_bytes()?,
    ];
    parts
        .into_iter()
        .try_fold(size_of_val(&parts), usize::checked_add)
}

struct Terms<'a>(std::slice::Iter<'a, Slot>);
impl<'a> Iterator for Terms<'a> {
    type Item = &'a [f32];
    fn next(&mut self) -> Option<Self::Item> {
        let slot = self.0.next()?;
        // validate() established exact completed raw terms before this worker.
        let Some(PartitionFragmentValue::Tensor(value)) = &slot.value else {
            unreachable!("validated raw term")
        };
        let TensorObservationData::F32(values) = value.observation().data() else {
            unreachable!("validated F32 term")
        };
        Some(values)
    }
}
impl PartitionFragmentDestination<'_, '_> {
    pub(super) fn identity(&self) -> &ReceiptIdentity {
        match self {
            Self::Tensor(v) => &v.identity,
            Self::Summary(v) => v.partition_identity(),
            Self::Histogram(v) => v.partition_identity(),
            Self::Routed(v) => v.partition_identity(),
        }
    }
    fn matches_global(
        &self,
        receipt: &PartitionCaptureReceiptPlan,
        source: &SharedCapturePlan,
    ) -> bool {
        let Some((_, projection)) = receipt.producers().next() else {
            return false;
        };
        let context = receipt.context();
        let slice = projection.global_slice();
        macro_rules! check {
            ($value:expr) => {{
                let geometry = $value.geometry();
                std::ptr::eq(geometry.admission(), source.admission())
                    && geometry.selection_index() == context.selection_index
                    && geometry.phase() == context.phase
                    && geometry.prediction() == context.prediction
                    && geometry.source_shape().len() == projection.global_shape().len()
                    && geometry
                        .source_shape()
                        .iter()
                        .zip(projection.global_shape())
                        .all(|(&a, &b)| u64::try_from(a).ok() == Some(b))
                    && geometry.starts() == slice.starts.as_slice()
                    && geometry.strides() == slice.strides.as_slice()
                    // A temporal window closes its exclusive end immediately
                    // after the last selected coordinate. The original slice
                    // may retain a later end within the same stride. Compare
                    // the exact coordinates through start, stride and count.
                    && geometry.ends().len() == slice.shape.len()
                    && geometry
                        .ends()
                        .iter()
                        .zip(geometry.starts())
                        .zip(geometry.strides())
                        .zip(&slice.shape)
                        .all(|(((&end, &start), &stride), &count)| {
                            stride != 0
                                && end.checked_sub(start)
                                    .is_some_and(|width| width.div_ceil(stride) == count)
                        })
            }};
        }
        match (
            self,
            &source.admission().plan().selections[context.selection_index].transform,
        ) {
            (Self::Routed(v), CaptureTransform::RoutedUnits) => check!(v),
            (
                Self::Tensor(v),
                CaptureTransform::FullTensor
                | CaptureTransform::Slice
                | CaptureTransform::Preview { .. },
            ) => check!(v),
            (Self::Summary(v), CaptureTransform::Summary) => check!(v),
            (Self::Histogram(v), CaptureTransform::Histogram { .. }) => check!(v),
            _ => false,
        }
    }
}
impl PreparedPartitionFragmentDestinations {
    /// Finish the actual host equation into its separately paid global claim.
    /// The unique-rank exchange and native completion boundary must finish before
    /// calling this internal worker. It issues no vote or final receipt evidence.
    pub(crate) fn assemble_into(
        mut self,
        receipt: &PartitionCaptureReceiptPlan,
        destination: PartitionFragmentDestination<'_, '_>,
        dtype: &TensorDtype,
    ) -> Result<(PartitionFragmentValue, CaptureUsage), PartitionFragmentAssemblyError> {
        let custody = destination.identity().custody.share_scheduled();
        self.assemble_worker(receipt, destination, dtype)
            .map_err(|cause| PartitionFragmentAssemblyError {
                cause,
                _bank: self,
                _destination: custody,
            })
    }
    fn validate_assembly(
        &self,
        receipt: &PartitionCaptureReceiptPlan,
        dtype: &TensorDtype,
    ) -> Result<(), AssemblyCause> {
        self.custody.validate()?;
        if !self.matches(receipt)
            || !self.complete()
            || !matches!(
                dtype,
                TensorDtype::F16 | TensorDtype::Bf16 | TensorDtype::F32
            )
        {
            return Err(AssemblyCause::Source);
        }
        let selection =
            &self.source.admission().plan().selections[receipt.context().selection_index];
        let raw = receipt.combination() == PartitionCaptureCombination::SumF64ToF32
            || matches!(
                selection.transform,
                CaptureTransform::FullTensor
                    | CaptureTransform::Slice
                    | CaptureTransform::Preview { .. }
            );
        for slot in &self.rows {
            if !self
                .allowance
                .fragment_source(slot.producer, slot.fragment)
                .is_some_and(|(actual, _)| actual == dtype)
            {
                return Err(AssemblyCause::Source);
            }
            let value = slot.value.as_ref().ok_or(AssemblyCause::Source)?;
            value.identity().custody.validate()?;
            match (raw, value, &selection.transform) {
                (true, PartitionFragmentValue::Tensor(value), _)
                    if matches!(value.observation().data(), TensorObservationData::F32(_)) => {}
                (false, PartitionFragmentValue::Summary(_), CaptureTransform::Summary) => {}
                (
                    false,
                    PartitionFragmentValue::Histogram(_),
                    CaptureTransform::Histogram { .. },
                ) => {}
                _ => return Err(AssemblyCause::Source),
            }
        }
        Ok(())
    }
    fn fill_sum(&self, output: &mut ScheduledCaptureTensor<'_, '_>) -> Result<(), AssemblyCause> {
        for index in 0..output.len() {
            let value = crate::capture::partition::sum_f32_at(Terms(self.rows.iter()), index)
                .ok_or(AssemblyCause::Source)?;
            output.push_f32(value)?;
        }
        Ok(())
    }
    fn sum_temporary(
        &mut self,
        receipt: &PartitionCaptureReceiptPlan,
    ) -> Result<Option<ClaimedCaptureTensor>, AssemblyCause> {
        let Some((producer, fragment, custody)) = self.scratch.take() else {
            if !self.rows.is_empty() {
                return Err(AssemblyCause::Source);
            }
            return Ok(None);
        };
        let FragmentHostPlan::Tensor(plan) =
            FragmentHostPlan::prepare(receipt, producer, fragment)?
        else {
            return Err(AssemblyCause::Source);
        };
        let identity = ReceiptIdentity {
            phase: receipt.context().phase,
            prediction: receipt.context().prediction,
            index: receipt.context().selection_index,
            custody,
        };
        let mut output = CaptureTensorClaim {
            plan,
            identity,
            exclusive: PhantomData,
        }
        .prepare()?;
        self.fill_sum(&mut output)?;
        output
            .finish()
            .map(Some)
            .map_err(|e| e.into_owned_error().into())
    }
    fn assemble_worker(
        &mut self,
        receipt: &PartitionCaptureReceiptPlan,
        destination: PartitionFragmentDestination<'_, '_>,
        dtype: &TensorDtype,
    ) -> Result<(PartitionFragmentValue, CaptureUsage), AssemblyCause> {
        if let PartitionFragmentDestination::Routed(claim) = destination {
            return self.assemble_routed(receipt, claim, dtype);
        }
        self.validate_assembly(receipt, dtype)?;
        destination.identity().custody.validate()?;
        if !destination.matches_global(receipt, &self.source) {
            return Err(AssemblyCause::Source);
        }
        let usage = receipt
            .dense_assembly_usage()?
            .ok_or(AssemblyCause::Source)?;
        // These are the original common credits, already charged globally.
        // Spending a child allowance changes no parent total or refund.
        self.allowance.quota_mut().reserve_quota(usage)?;
        let charged = self.allowance.assembly_record_charge(receipt)?;
        let sum = receipt.combination() == PartitionCaptureCombination::SumF64ToF32;
        let value = match destination {
            PartitionFragmentDestination::Routed(_) => return Err(AssemblyCause::Source),
            PartitionFragmentDestination::Tensor(claim) => {
                let mut output = claim.prepare()?;
                if sum {
                    self.fill_sum(&mut output)?;
                } else {
                    let (_, projection) =
                        receipt.producers().next().ok_or(AssemblyCause::Source)?;
                    let global = &projection.global_slice().shape;
                    for _ in 0..output.len() {
                        output.push_f32(0.0)?;
                    }
                    let mut copied = 0u64;
                    for slot in &self.rows {
                        let projection = receipt
                            .producer(slot.producer)
                            .ok_or(AssemblyCause::Source)?;
                        let geometry = projection
                            .fragments()
                            .get(slot.fragment)
                            .ok_or(AssemblyCause::Source)?
                            .destination();
                        let mapping = PrefixDestination::new_fixed(
                            global,
                            &geometry.shape,
                            &geometry.starts,
                            &geometry.strides,
                        )?;
                        let Some(PartitionFragmentValue::Tensor(value)) = &slot.value else {
                            return Err(AssemblyCause::Source);
                        };
                        let TensorObservationData::F32(values) = value.observation().data() else {
                            return Err(AssemblyCause::Source);
                        };
                        for (local, &value) in values.iter().enumerate() {
                            let index =
                                mapping.ordinal(local as u64).ok_or(AssemblyCause::Source)?;
                            if index >= output.len() as u64 {
                                continue;
                            }
                            output.replace_initialized_f32(
                                usize::try_from(index).map_err(|_| AssemblyCause::Source)?,
                                value,
                            )?;
                            copied = copied.checked_add(1).ok_or(WorkingMemoryError::Overflow)?;
                        }
                    }
                    if copied != output.len() as u64 {
                        return Err(AssemblyCause::Source);
                    }
                }
                PartitionFragmentValue::Tensor(output.finish().map_err(|e| e.into_owned_error())?)
            }
            PartitionFragmentDestination::Summary(claim) => {
                let value = if sum {
                    let temporary = self.sum_temporary(receipt)?;
                    let values = match temporary.as_ref().map(|v| v.observation().data()) {
                        Some(TensorObservationData::F32(values)) => values.as_slice(),
                        None => &[],
                        _ => return Err(AssemblyCause::Source),
                    };
                    crate::capture::partition::summarize_f32(values)
                } else {
                    let mut summary = Summary::default();
                    for slot in &self.rows {
                        let Some(PartitionFragmentValue::Summary(value)) = &slot.value else {
                            return Err(AssemblyCause::Source);
                        };
                        summary = summary
                            .appended_fixed(value.observation(), value.observation().elements)?;
                    }
                    summary.value()
                };
                PartitionFragmentValue::Summary(claim.finish_partition(value)?)
            }
            PartitionFragmentDestination::Histogram(claim) => {
                let mut output = claim.prepare()?;
                if sum {
                    let temporary = self.sum_temporary(receipt)?;
                    let values = match temporary.as_ref().map(|v| v.observation().data()) {
                        Some(TensorObservationData::F32(values)) => values.as_slice(),
                        None => &[],
                        _ => return Err(AssemblyCause::Source),
                    };
                    output.fill_partition_sum(values)?;
                } else {
                    for slot in &self.rows {
                        let Some(PartitionFragmentValue::Histogram(value)) = &slot.value else {
                            return Err(AssemblyCause::Source);
                        };
                        output.merge_partition(value.observation())?;
                    }
                }
                PartitionFragmentValue::Histogram(output.finish_partition()?)
            }
        };
        Ok((value, charged))
    }
}

mod routed;
