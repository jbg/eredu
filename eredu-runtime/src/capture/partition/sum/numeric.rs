//! The ordinary additive capture arithmetic, shared by paid host destinations.
use super::*;
use crate::capture::reduction::CaptureHistogramError;
use std::mem::{size_of,size_of_val};

#[derive(Default)]
struct Sum {
    value: f64,
    correction: f64,
}
impl Sum {
    fn add(&mut self, value: f64) {
        let next = self.value + value;
        if self.value.is_finite() && value.is_finite() {
            self.correction += if self.value.abs() >= value.abs() {
                (self.value - next) + value
            } else {
                (value - next) + self.value
            };
        } else {
            self.correction = 0.0;
        }
        self.value = next;
    }
    fn finish(self) -> f64 {
        self.value + self.correction
    }
}

/// Consume complete raw terms in canonical producer order. The single F32 cast
/// follows the compensated F64 sum and precedes every nonlinear transform.
pub(crate) fn sum_f32_at<'a>(terms: impl Iterator<Item=&'a [f32]>, index:usize)->Option<f32> {
    let mut sum=Sum::default();
    for term in terms {sum.add(f64::from(*term.get(index)?));}
    Some(sum.finish() as f32)
}
/// Existing finite/nonfinite Summary loop over the materialized F32 assembly.
pub(crate) fn summarize_f32(values:&[f32])->CaptureSummary {
            let mut summary = CaptureSummary {
                elements: values.len() as u64,
                finite: 0,
                non_finite: 0,
                nan: 0,
                positive_infinity: 0,
                negative_infinity: 0,
                min: None,
                max: None,
                mean: None,
                rms: None,
            };
            let mut sum = Sum::default();
            let mut squares = Sum::default();
            for &value in values {
                if value.is_finite() {
                    let value = f64::from(value);
                    summary.finite += 1;
                    summary.min = Some(summary.min.map_or(value, |min| min.min(value)));
                    summary.max = Some(summary.max.map_or(value, |max| max.max(value)));
                    sum.add(value);
                    squares.add(value * value);
                } else {
                    summary.non_finite += 1;
                    summary.nan += u64::from(value.is_nan());
                    summary.positive_infinity += u64::from(value == f32::INFINITY);
                    summary.negative_infinity += u64::from(value == f32::NEG_INFINITY);
                }
            }
            if summary.finite != 0 {
                summary.mean = Some(sum.finish() / summary.finite as f64);
                summary.rms = Some((squares.finish() / summary.finite as f64).sqrt());
            }
    summary
}
/// Existing fixed-edge loop into a fresh exact destination. No vector is born
/// here; the caller's ordinary/paid constructor supplies its edges and zero bins.
pub(crate) fn fill_histogram_f32(values:&[f32],histogram:&mut CaptureHistogram)->Result<(),CaptureHistogramError> {
    if histogram.edges.len()<2 || histogram.counts.len().checked_add(1)!=Some(histogram.edges.len()) {
        return Err(CaptureHistogramError::Edges);
    }
    if histogram.below!=0 || histogram.above!=0 || histogram.non_finite!=0 || histogram.counts.iter().any(|n|*n!=0) {
        return Err(CaptureHistogramError::Counts);
    }
    let edges=&histogram.edges;
            for &value in values {
                if !value.is_finite() {
                    histogram.non_finite += 1;
                } else if value < edges[0] {
                    histogram.below += 1;
                } else if value > *edges.last().expect("admitted histogram") {
                    histogram.above += 1;
                } else {
                    let index = edges
                        .partition_point(|edge| *edge <= value)
                        .saturating_sub(1)
                        .min(histogram.counts.len() - 1);
                    histogram.counts[index] += 1;
                }
            }
    Ok(())
}
/// Actual fixed arithmetic frames. The caller additionally prices its concrete
/// source iterator, argument/result transport and the real F32 assembly buffer.
pub(crate) fn numeric_control_bytes()->Option<usize> {
    let parts=[size_of::<Sum>()*2,size_of::<CaptureSummary>(),size_of::<Option<f32>>(),
        size_of::<std::slice::Iter<'_,f32>>(),size_of::<std::slice::Iter<'_,u64>>(),
        size_of::<(&[f32],&mut CaptureHistogram)>(),size_of::<(&[f32],usize)>(),
        size_of::<Result<(),CaptureHistogramError>>(),size_of::<(f32,f64,usize)>(),
        size_of::<(&[f32],f32)>(),size_of::<CaptureHistogramError>()];
    parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn additive_worker_rounds_once_then_reduces_materialized_f32_and_keeps_nonfinite_classes() {
        let terms=[vec![16_777_216.0,f32::MAX,f32::INFINITY,f32::NAN,-0.0],
            vec![1.0,-f32::MAX,f32::NEG_INFINITY,0.0,0.0],
            vec![-16_777_216.0,1.25,0.0,0.0,-0.0]];
        let values:Vec<_>=(0..5).map(|index|sum_f32_at(terms.iter().map(Vec::as_slice),index).unwrap()).collect();
        assert_eq!(values[0],1.0,"per-term F32 accumulation would lose this residual");
        assert_eq!(values[1],1.25);assert!(values[2].is_nan()&&values[3].is_nan());
        let summary=summarize_f32(&values);
        assert_eq!((summary.elements,summary.finite,summary.nan),(5,3,2));
        assert_eq!((summary.positive_infinity,summary.negative_infinity),(0,0));
        assert_eq!(summary.mean,Some(0.75));assert_eq!(summary.rms,Some((2.5625f64/3.0).sqrt()));
        let mut histogram=CaptureHistogram{edges:vec![0.0,1.0,2.0],counts:vec![0,0],below:0,above:0,non_finite:0};
        fill_histogram_f32(&values,&mut histogram).unwrap();assert_eq!(histogram.counts,[1,2]);assert_eq!(histogram.non_finite,2);
        let before=histogram.clone();assert!(fill_histogram_f32(&values,&mut histogram).is_err());assert_eq!(histogram,before);
        assert!(sum_f32_at([&[1.0][..],&[][..]].into_iter(),0).is_none());
    }
}
