//! One ordinary projected invocation, shared by Host and native source consumers.
use super::*;
use std::mem::{size_of, size_of_val};

/// Exact descriptive source failure, before receipt or native authority is lent.
#[derive(Debug, thiserror::Error)]
pub enum PartitionInvocationCaptureSourceError {
    /// The original receipt does not bind this dense fragment.
    #[error("partition invocation source differs from its original receipt")]
    Source,
    /// The existing ordinary geometry rejected the actual selection or projection.
    #[error(transparent)]
    Geometry(#[from] CaptureTensorGeometryError),
}
/// The ordinary local worker selected by the original global transform.
#[derive(Debug)]
pub enum PartitionInvocationCaptureKind<'a> {
    /// Raw selected values, including additive terms before global reduction.
    Tensor(CaptureTensorGeometry<'a>),
    /// A disjoint partial summary.
    Summary(CaptureSummaryGeometry<'a>),
    /// Disjoint partial bins with the original edges.
    Histogram(CaptureHistogramGeometry<'a>),
}
/// Borrowed source for one original ordinary or explicitly shaped invocation.
/// The receipt retains the original bounds and exact axes. Chunked prefill uses its
/// separate temporal projection; this source never impersonates a chunk or a
/// native scalar witness, and carries no account or submission authority.
#[derive(Debug)]
pub struct PartitionInvocationCaptureGeometry<'a> {
    admission: &'a AdmittedCapturePlan,
    selection: usize,
    projection: &'a CaptureSlicePartition,
    fragment: usize,
    window: Option<(CaptureInvocationShape, CaptureInvocationWindow)>,
    kind: PartitionInvocationCaptureKind<'a>,
}
impl<'a> PartitionInvocationCaptureGeometry<'a> {
    /// Derive an ordinary invocation's source from the immutable global plan.
    /// Additive Summary/Histogram use raw terms under that same admission;
    /// the existing global assembly alone applies the nonlinear reduction.
    pub fn prepare(
        admission: &'a AdmittedCapturePlan,
        selection: usize,
        phase: CapturePhase,
        prediction: u64,
        projection: &'a CaptureSlicePartition,
        fragment: usize,
        combination: PartitionCaptureCombination,
    ) -> Result<Self, PartitionInvocationCaptureSourceError> {
        Self::prepare_source(
            admission,
            selection,
            phase,
            prediction,
            None,
            projection,
            fragment,
            combination,
        )
    }
    /// Resolve the real independent invocation through the same projected worker.
    /// Geometry remains descriptive and grants no receipt or completion authority.
    #[allow(clippy::too_many_arguments)]
    pub fn prepare_shaped(
        admission: &'a AdmittedCapturePlan,
        selection: usize,
        phase: CapturePhase,
        prediction: u64,
        invocation: CaptureInvocationShape,
        projection: &'a CaptureSlicePartition,
        fragment: usize,
        combination: PartitionCaptureCombination,
    ) -> Result<Self, PartitionInvocationCaptureSourceError> {
        Self::prepare_source(
            admission,
            selection,
            phase,
            prediction,
            Some(invocation),
            projection,
            fragment,
            combination,
        )
    }
    fn prepare_source(
        admission: &'a AdmittedCapturePlan,
        selection: usize,
        phase: CapturePhase,
        prediction: u64,
        invocation: Option<CaptureInvocationShape>,
        projection: &'a CaptureSlicePartition,
        fragment: usize,
        combination: PartitionCaptureCombination,
    ) -> Result<Self, PartitionInvocationCaptureSourceError> {
        use PartitionInvocationCaptureKind as K;
        use PartitionInvocationCaptureSourceError as E;
        let selected = admission
            .plan()
            .selections
            .get(selection)
            .ok_or(E::Source)?;
        let kind = match (&selected.transform, combination) {
            (CaptureTransform::Summary, PartitionCaptureCombination::Disjoint) => {
                K::Summary(CaptureSummaryGeometry::prepare_partition(
                    admission, selection, phase, prediction, invocation, projection, fragment,
                )?)
            }
            (CaptureTransform::Histogram { .. }, PartitionCaptureCombination::Disjoint) => {
                K::Histogram(CaptureHistogramGeometry::prepare_partition(
                    admission, selection, phase, prediction, invocation, projection, fragment,
                )?)
            }
            (
                CaptureTransform::FullTensor
                | CaptureTransform::Slice
                | CaptureTransform::Preview { .. },
                _,
            )
            | (
                CaptureTransform::Summary | CaptureTransform::Histogram { .. },
                PartitionCaptureCombination::SumF64ToF32,
            ) => K::Tensor(CaptureTensorGeometry::prepare_partition(
                admission,
                selection,
                phase,
                prediction,
                invocation,
                projection,
                fragment,
                combination,
            )?),
            _ => return Err(E::Source),
        };
        Ok(Self {
            admission,
            selection,
            projection,
            fragment,
            window: None,
            kind,
        })
    }
    /// Authenticate the same original receipt used by paid destination issuance.
    /// Explicit physical axes come from this receipt and are revalidated against
    /// the same immutable bounds. Routed sources and logical row windows retain
    /// their separate consumers; equal sizes cannot replace those identities.
    pub fn from_receipt(
        receipt: &'a PartitionCaptureReceiptPlan,
        producer: usize,
        fragment: usize,
    ) -> Result<Self, PartitionInvocationCaptureSourceError> {
        use PartitionInvocationCaptureSourceError as E;
        let source = receipt.shared_plan_source();
        let context = receipt.context();
        let projection = receipt.producer(producer).ok_or(E::Source)?;
        if context.capture_plan_identity != source.admission().identity()
            || receipt.routed_producer(producer).is_some()
        {
            return Err(E::Source);
        }
        use PartitionInvocationCaptureKind as K;
        let selected = source
            .admission()
            .plan()
            .selections
            .get(context.selection_index)
            .ok_or(E::Source)?;
        let kind = match (&selected.transform, receipt.combination()) {
            (CaptureTransform::Summary, PartitionCaptureCombination::Disjoint) => {
                K::Summary(CaptureSummaryGeometry::prepare_receipt_partition(
                    source.admission(),
                    context,
                    projection,
                    fragment,
                )?)
            }
            (CaptureTransform::Histogram { .. }, PartitionCaptureCombination::Disjoint) => {
                K::Histogram(CaptureHistogramGeometry::prepare_receipt_partition(
                    source.admission(),
                    context,
                    projection,
                    fragment,
                )?)
            }
            (
                CaptureTransform::FullTensor
                | CaptureTransform::Slice
                | CaptureTransform::Preview { .. },
                _,
            )
            | (
                CaptureTransform::Summary | CaptureTransform::Histogram { .. },
                PartitionCaptureCombination::SumF64ToF32,
            ) => K::Tensor(CaptureTensorGeometry::prepare_receipt_partition(
                source.admission(),
                context,
                projection,
                fragment,
                receipt.combination(),
            )?),
            _ => return Err(E::Source),
        };
        Ok(Self {
            admission: source.admission(),
            selection: context.selection_index,
            projection,
            fragment,
            window: context
                .invocation_window
                .map(|value| (value.physical, value.window)),
            kind,
        })
    }
    /// Authenticate a physical row window of this same logical receipt. The
    /// receipt preserves the complete invocation's identity; the selected native
    /// source retains both coordinates without rewriting its admission.
    pub fn from_window_receipt(
        receipt: &'a PartitionCaptureReceiptPlan,
        producer: usize,
        fragment: usize,
        physical: CaptureInvocationShape,
        window: CaptureInvocationWindow,
    ) -> Result<Self, PartitionInvocationCaptureSourceError> {
        use PartitionInvocationCaptureSourceError as E;
        let source = receipt.shared_plan_source();
        let context = receipt.context();
        if context.invocation_window.is_some() {
            if !context.matches_invocation(Some(physical), Some(window)) {
                return Err(E::Source);
            }
            return Self::from_receipt(receipt, producer, fragment);
        }
        let logical = window
            .validate(physical)
            .map_err(CaptureTensorGeometryError::from)?;
        if context.invocation != Some(logical)
            || context.capture_plan_identity != source.admission().identity()
            || receipt.routed_producer(producer).is_some()
        {
            return Err(E::Source);
        }
        let projection = receipt.producer(producer).ok_or(E::Source)?;
        Self::prepare_window(
            source.admission(),
            context.selection_index,
            context.phase,
            context.prediction,
            physical,
            window,
            projection,
            fragment,
            receipt.combination(),
        )
    }
    /// Resolve exact physical and logical coordinates from the same immutable
    /// admission before a receipt or native execution authority exists.
    pub fn prepare_window(
        admission: &'a AdmittedCapturePlan,
        selection: usize,
        phase: CapturePhase,
        prediction: u64,
        physical: CaptureInvocationShape,
        window: CaptureInvocationWindow,
        projection: &'a CaptureSlicePartition,
        fragment: usize,
        combination: PartitionCaptureCombination,
    ) -> Result<Self, PartitionInvocationCaptureSourceError> {
        use PartitionInvocationCaptureKind as K;
        use PartitionInvocationCaptureSourceError as E;
        let selected = admission
            .plan()
            .selections
            .get(selection)
            .ok_or(E::Source)?;
        let kind = match (&selected.transform, combination) {
            (CaptureTransform::Summary, PartitionCaptureCombination::Disjoint) => {
                K::Summary(CaptureSummaryGeometry::prepare_partition_window(
                    admission, selection, phase, prediction, physical, window, projection, fragment,
                )?)
            }
            (CaptureTransform::Histogram { .. }, PartitionCaptureCombination::Disjoint) => {
                K::Histogram(CaptureHistogramGeometry::prepare_partition_window(
                    admission, selection, phase, prediction, physical, window, projection, fragment,
                )?)
            }
            (
                CaptureTransform::FullTensor
                | CaptureTransform::Slice
                | CaptureTransform::Preview { .. },
                _,
            )
            | (
                CaptureTransform::Summary | CaptureTransform::Histogram { .. },
                PartitionCaptureCombination::SumF64ToF32,
            ) => K::Tensor(CaptureTensorGeometry::prepare_partition_window(
                admission,
                selection,
                phase,
                prediction,
                physical,
                window,
                projection,
                fragment,
                combination,
            )?),
            _ => return Err(E::Source),
        };
        Ok(Self {
            admission,
            selection,
            projection,
            fragment,
            window: Some((physical, window)),
            kind,
        })
    }
    /// Window-only call/result frames in addition to the ordinary source quote.
    pub fn window_control_bytes() -> Option<usize> {
        let parts = [
            Self::control_bytes()?,
            size_of::<Self>() * 2,
            size_of::<Result<Self, PartitionInvocationCaptureSourceError>>(),
            size_of::<(
                &PartitionCaptureReceiptPlan,
                usize,
                usize,
                CaptureInvocationShape,
                CaptureInvocationWindow,
            )>(),
            size_of::<(
                CaptureInvocationShape,
                CaptureInvocationShape,
                CaptureInvocationWindow,
            )>(),
            CaptureTensorGeometry::partition_window_preparation_control_bytes()?,
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
    /// Actual physical axes and logical row placement, when independently split.
    pub const fn invocation_window(
        &self,
    ) -> Option<(CaptureInvocationShape, CaptureInvocationWindow)> {
        self.window
    }
    /// The selected ordinary local worker.
    pub fn kind(&self) -> &PartitionInvocationCaptureKind<'a> {
        &self.kind
    }
    /// Move the same geometry into its typed Host or native consumer.
    pub fn into_kind(self) -> PartitionInvocationCaptureKind<'a> {
        self.kind
    }
    /// Original immutable admission, retained without a local replacement.
    pub const fn admission(&self) -> &'a AdmittedCapturePlan {
        self.admission
    }
    /// Original selection ordinal.
    pub const fn selection_index(&self) -> usize {
        self.selection
    }
    /// Exact selected producer projection.
    pub const fn projection(&self) -> &'a CaptureSlicePartition {
        self.projection
    }
    /// Exact local fragment ordinal.
    pub const fn fragment_index(&self) -> usize {
        self.fragment
    }
    /// Actual transform accepted by the fragment's native allowance.
    pub fn native_transform(&self) -> &CaptureTransform {
        match &self.kind {
            PartitionInvocationCaptureKind::Tensor(geometry) => geometry.native_transform(),
            _ => &self.admission.plan().selections[self.selection].transform,
        }
    }
    /// Fixed construction and error frames; no backing allocation is requested.
    pub fn control_bytes() -> Option<usize> {
        let parts = [
            size_of::<Self>() * 2,
            size_of::<PartitionInvocationCaptureKind<'_>>(),
            size_of::<Result<Self, PartitionInvocationCaptureSourceError>>(),
            size_of::<PartitionInvocationCaptureSourceError>(),
            size_of::<(
                &AdmittedCapturePlan,
                usize,
                CapturePhase,
                u64,
                &CaptureSlicePartition,
                usize,
                PartitionCaptureCombination,
            )>(),
            size_of::<(&PartitionCaptureReceiptPlan, usize, usize)>(),
            size_of::<Option<CaptureInvocationShape>>(),
            size_of::<(
                &AdmittedCapturePlan,
                usize,
                CapturePhase,
                u64,
                Option<CaptureInvocationShape>,
                &CaptureSlicePartition,
                usize,
                PartitionCaptureCombination,
            )>(),
            CaptureTensorGeometry::partition_preparation_control_bytes()?,
            CaptureSummaryGeometry::preparation_control_bytes()?,
            CaptureHistogramGeometry::preparation_control_bytes()?,
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
}
