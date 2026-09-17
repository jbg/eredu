//! Host reduction algebra; geometry, source authority and reservation stay with callers.
use super::*;

fn invalid(message: &str) -> CaptureError {
    CaptureError::Invalid(message.into())
}

/// Allocation-free validation failure for the shared finite-summary merger.
#[derive(Debug, thiserror::Error)]
pub enum CaptureSummaryError {
    /// Counter arithmetic cannot be represented.
    #[error("summary count arithmetic overflow")]
    Overflow,
    /// Selected extent and finite/nonfinite classification counts disagree.
    #[error("summary fragment counts disagree with its selected geometry")]
    Counts,
    /// No finite values may supply finite-only aggregates.
    #[error("summary fragment fabricated finite aggregates")]
    EmptyAggregates,
    /// A nonempty finite contribution needs all four aggregate values.
    #[error("summary fragment is missing finite aggregates")]
    MissingAggregates,
    /// A finite-only aggregate is invalid or ordered inconsistently.
    #[error("summary fragment contains invalid finite aggregates")]
    InvalidAggregates,
    /// Finite moment accumulation overflowed.
    #[error("summary fragment aggregation overflowed finite statistics")]
    Moments,
}
impl From<CaptureSummaryError> for CaptureError {
    fn from(value: CaptureSummaryError) -> Self {
        match value {
            CaptureSummaryError::Overflow => Self::Overflow,
            value => Self::Invalid(value.to_string()),
        }
    }
}
fn summary_add(a: u64, b: u64) -> Result<u64, CaptureSummaryError> {
    a.checked_add(b).ok_or(CaptureSummaryError::Overflow)
}

#[derive(Debug, Clone, Default)]
struct CompensatedSum {
    sum: f64,
    correction: f64,
}
impl CompensatedSum {
    fn add(&mut self, value: f64) -> Result<(), CaptureSummaryError> {
        let next = self.sum + value;
        self.correction += if self.sum.abs() >= value.abs() {
            (self.sum - next) + value
        } else {
            (value - next) + self.sum
        };
        self.sum = next;
        if !self.sum.is_finite() || !self.correction.is_finite() {
            return Err(CaptureSummaryError::Moments);
        }
        Ok(())
    }
    fn value(&self) -> f64 {
        self.sum + self.correction
    }
}

/// Cloning this fixed scalar state allocates nothing. Rejected input cannot
/// partially change counters or moments.
#[derive(Debug, Clone)]
pub(crate) struct Summary {
    value: CaptureSummary,
    sum: CompensatedSum,
    squares: CompensatedSum,
}
impl Default for Summary {
    fn default() -> Self {
        Self {
            value: CaptureSummary {
                elements: 0,
                finite: 0,
                non_finite: 0,
                nan: 0,
                positive_infinity: 0,
                negative_infinity: 0,
                min: None,
                max: None,
                mean: None,
                rms: None,
            },
            sum: CompensatedSum::default(),
            squares: CompensatedSum::default(),
        }
    }
}
impl Summary {
    pub(super) fn appended(
        &self,
        value: &CaptureSummary,
        expected: u64,
    ) -> Result<Self, CaptureError> {
        self.appended_fixed(value, expected).map_err(Into::into)
    }
    pub(crate) fn appended_fixed(
        &self,
        value: &CaptureSummary,
        expected: u64,
    ) -> Result<Self, CaptureSummaryError> {
        if value.elements != expected
            || summary_add(value.finite, value.non_finite)? != value.elements
            || summary_add(
                summary_add(value.nan, value.positive_infinity)?,
                value.negative_infinity,
            )? != value.non_finite
        {
            return Err(CaptureSummaryError::Counts);
        }
        let mut next = self.clone();
        if value.finite == 0 {
            if [value.min, value.max, value.mean, value.rms]
                .iter()
                .any(Option::is_some)
            {
                return Err(CaptureSummaryError::EmptyAggregates);
            }
        } else {
            let (Some(min), Some(max), Some(mean), Some(rms)) =
                (value.min, value.max, value.mean, value.rms)
            else {
                return Err(CaptureSummaryError::MissingAggregates);
            };
            if ![min, max, mean, rms].iter().all(|v| v.is_finite()) || min > max || rms < 0.0 {
                return Err(CaptureSummaryError::InvalidAggregates);
            }
            next.value.min = Some(next.value.min.map_or(min, |current| current.min(min)));
            next.value.max = Some(next.value.max.map_or(max, |current| current.max(max)));
            next.sum.add(mean * value.finite as f64)?;
            next.squares.add(rms * rms * value.finite as f64)?;
        }
        next.value.elements = summary_add(next.value.elements, value.elements)?;
        next.value.finite = summary_add(next.value.finite, value.finite)?;
        next.value.non_finite = summary_add(next.value.non_finite, value.non_finite)?;
        next.value.nan = summary_add(next.value.nan, value.nan)?;
        next.value.positive_infinity =
            summary_add(next.value.positive_infinity, value.positive_infinity)?;
        next.value.negative_infinity =
            summary_add(next.value.negative_infinity, value.negative_infinity)?;
        if next.value.finite != 0 {
            next.value.mean = Some(next.sum.value() / next.value.finite as f64);
            next.value.rms = Some((next.squares.value() / next.value.finite as f64).sqrt());
            if !next.value.mean.unwrap().is_finite() || !next.value.rms.unwrap().is_finite() {
                return Err(CaptureSummaryError::Moments);
            }
        }
        Ok(next)
    }
    pub(crate) fn value(&self) -> CaptureSummary {
        self.value.clone()
    }
}

/// Fixed histogram validation failure, without diagnostic allocation.
#[derive(Clone, Copy, Debug, thiserror::Error)]
pub enum CaptureHistogramError {
    /// The payload does not retain the admitted edges/bin cardinality.
    #[error("histogram payload changed its admitted bin edges")]
    Edges,
    /// Classified counts do not cover exactly the selected values.
    #[error("histogram counts disagree with selected geometry")]
    Counts,
    /// Count addition cannot be represented.
    #[error("histogram count overflow")]
    Overflow,
}
impl From<CaptureHistogramError> for CaptureError {
    fn from(cause: CaptureHistogramError) -> Self {
        match cause {
            CaptureHistogramError::Overflow => Self::Overflow,
            cause => Self::Invalid(cause.to_string()),
        }
    }
}
fn histogram_add(left: u64, right: u64) -> Result<u64, CaptureHistogramError> {
    left.checked_add(right)
        .ok_or(CaptureHistogramError::Overflow)
}
pub(crate) fn validate_histogram_fixed(
    value: &CaptureHistogram,
    edges: &[f32],
    expected: u64,
) -> Result<(), CaptureHistogramError> {
    if value.edges != edges || value.counts.len().checked_add(1) != Some(edges.len()) {
        return Err(CaptureHistogramError::Edges);
    }
    let count = value.counts.iter().try_fold(
        histogram_add(histogram_add(value.below, value.above)?, value.non_finite)?,
        |sum, count| histogram_add(sum, *count),
    )?;
    if count != expected {
        return Err(CaptureHistogramError::Counts);
    }
    Ok(())
}

pub(crate) fn check_histogram_add_fixed(
    out: &CaptureHistogram,
    value: &CaptureHistogram,
) -> Result<(), CaptureHistogramError> {
    if out.edges != value.edges || out.counts.len() != value.counts.len() {
        return Err(CaptureHistogramError::Edges);
    }
    histogram_add(out.below, value.below)?;
    histogram_add(out.above, value.above)?;
    histogram_add(out.non_finite, value.non_finite)?;
    for (left, right) in out.counts.iter().zip(&value.counts) {
        histogram_add(*left, *right)?;
    }
    Ok(())
}

/// Validates the whole update before any mutation; no temporary bin vector.
pub(crate) fn add_histogram_fixed(
    out: &mut CaptureHistogram,
    value: &CaptureHistogram,
) -> Result<(), CaptureHistogramError> {
    check_histogram_add_fixed(out, value)?;
    out.below += value.below;
    out.above += value.above;
    out.non_finite += value.non_finite;
    for (left, right) in out.counts.iter_mut().zip(&value.counts) {
        *left += right;
    }
    Ok(())
}

pub(super) fn validate_histogram(
    value: &CaptureHistogram,
    edges: &[f32],
    expected: u64,
) -> Result<(), CaptureError> {
    validate_histogram_fixed(value, edges, expected).map_err(CaptureError::from)
}
pub(super) fn check_histogram_add(
    out: &CaptureHistogram,
    value: &CaptureHistogram,
) -> Result<(), CaptureError> {
    check_histogram_add_fixed(out, value).map_err(CaptureError::from)
}
pub(super) fn add_histogram(
    out: &mut CaptureHistogram,
    value: &CaptureHistogram,
) -> Result<(), CaptureError> {
    add_histogram_fixed(out, value).map_err(CaptureError::from)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn summary_extremes_nonfinite_identity_and_failed_updates_preserve_prior_state() {
        let empty = Summary::default();
        assert_eq!((empty.value().mean, empty.value().rms), (None, None));
        let maximum = f64::from(f32::MAX);
        let mut positive = empty.value();
        positive.elements = 4;
        positive.finite = 1;
        positive.non_finite = 3;
        positive.nan = 1;
        positive.positive_infinity = 1;
        positive.negative_infinity = 1;
        positive.min = Some(maximum);
        positive.max = Some(maximum);
        positive.mean = Some(maximum);
        positive.rms = Some(maximum);
        let state = empty.appended(&positive, 4).unwrap();
        let mut negative = empty.value();
        negative.elements = 1;
        negative.finite = 1;
        negative.min = Some(-maximum);
        negative.max = Some(-maximum);
        negative.mean = Some(-maximum);
        negative.rms = Some(maximum);
        let combined = state.appended(&negative, 1).unwrap().value();
        assert_eq!(
            (combined.elements, combined.finite, combined.non_finite),
            (5, 2, 3)
        );
        assert_eq!(
            (combined.min, combined.max, combined.mean, combined.rms),
            (Some(-maximum), Some(maximum), Some(0.), Some(maximum))
        );
        negative.finite = 2;
        assert!(state.appended(&negative, 1).is_err());
        assert_eq!(state.value(), positive);
        let nonfinite = CaptureSummary {
            elements: 3,
            finite: 0,
            non_finite: 3,
            nan: 1,
            positive_infinity: 1,
            negative_infinity: 1,
            ..empty.value()
        };
        assert_eq!(empty.appended(&nonfinite, 3).unwrap().value(), nonfinite);
    }

    #[test]
    fn histogram_checks_every_counter_before_any_update() {
        let mut out = CaptureHistogram {
            edges: vec![0., 1., 2.],
            counts: vec![0, u64::MAX],
            below: 0,
            above: 0,
            non_finite: 0,
        };
        let original = out.clone();
        let mut incoming = CaptureHistogram {
            edges: vec![0., 1., 2.],
            counts: vec![1, 1],
            below: 1,
            above: 0,
            non_finite: 0,
        };
        assert!(add_histogram(&mut out, &incoming).is_err());
        assert_eq!(out, original);
        incoming.counts[1] = 0;
        add_histogram(&mut out, &incoming).unwrap();
        assert_eq!(out.below, 1);
        assert_eq!(out.counts, [1, u64::MAX]);
        let before = out.clone();
        incoming.edges[1] = 0.5;
        assert!(add_histogram(&mut out, &incoming).is_err());
        assert_eq!(out, before);
        assert!(validate_histogram(&incoming, &[0., 0.5, 2.], 3).is_err());
    }
}

/// Existing logical summary quota: one final scalar payload plus each actual
/// physical fragment's transform usage. This grants no native/host allocation.
pub fn summary_prefill_usage(
    plan: &eredu_core::capture::CapturePrefillTransformPlan<'_>,
    physical: impl FnMut(
        &eredu_core::capture::CapturePrefillTransformFragment<'_, '_>,
    )
        -> Result<eredu_core::capture::CaptureUsage, eredu_core::capture::CaptureError>,
) -> Result<eredu_core::capture::CaptureUsage, eredu_core::capture::CaptureError> {
    if !matches!(plan.selection().transform, CaptureTransform::Summary) {
        return Err(CaptureError::Unsupported(
            "summary quota requires summary selection".into(),
        ));
    }
    reduction_prefill_usage(plan, physical)
}
/// The existing Histogram logical quota: one final payload plus each actual physical fragment.
pub fn histogram_prefill_usage(
    plan: &eredu_core::capture::CapturePrefillTransformPlan<'_>,
    physical: impl FnMut(
        &eredu_core::capture::CapturePrefillTransformFragment<'_, '_>,
    ) -> Result<CaptureUsage, CaptureError>,
) -> Result<CaptureUsage, CaptureError> {
    if !matches!(
        plan.selection().transform,
        CaptureTransform::Histogram { .. }
    ) {
        return Err(CaptureError::Unsupported(
            "histogram quota requires histogram selection".into(),
        ));
    }
    reduction_prefill_usage(plan, physical)
}
fn reduction_prefill_usage(
    plan: &eredu_core::capture::CapturePrefillTransformPlan<'_>,
    mut physical: impl FnMut(
        &eredu_core::capture::CapturePrefillTransformFragment<'_, '_>,
    ) -> Result<CaptureUsage, CaptureError>,
) -> Result<CaptureUsage, CaptureError> {
    let mut total = CaptureUsage {
        captures: 1,
        host_bytes: (plan.window().rank() as u64)
            .checked_mul(2 * size_of::<u64>() as u64)
            .ok_or(CaptureError::Overflow)?,
        encoded_bytes: 512,
        retained_bytes: 0,
    };
    if let CaptureTransform::Histogram { edges } = &plan.selection().transform {
        let edges = edges.len() as u64;
        total.host_bytes = add(
            total.host_bytes,
            add(
                mul(edges, size_of::<f32>() as u64)?,
                mul(edges - 1, size_of::<u64>() as u64)?,
            )?,
        )?;
        total.encoded_bytes = add(256, mul(edges, 64)?)?;
    }
    for chunk in 0..plan.chunk_count() {
        let mut usage = physical(&plan.fragment(chunk)?)?;
        usage.captures = 0;
        usage.encoded_bytes = 0;
        total = total.checked_add(usage)?;
    }
    Ok(total)
}
