//! Synchronous host sources use the same original claim and summary reduction.
use super::*;

/// Borrowed real F32 host storage, or one fixed value repeated by a host equation.
/// This view grants no native readback or producer allocation permission.
#[derive(Clone, Copy, Debug)]
pub enum CaptureHostF32<'a> {
    /// Complete contiguous row-major source, including unselected values.
    Dense(&'a [f32]),
    /// A host equation's immutable scalar broadcast over the supplied shape.
    Uniform(f32),
}

struct Selected<'a, 'g> {
    source: CaptureHostF32<'a>,
    geometry: &'g CaptureSummaryGeometry<'g>,
    next: usize,
}
impl Iterator for Selected<'_, '_> {
    type Item = f32;
    fn next(&mut self) -> Option<f32> {
        if self.next == self.geometry.elements() {
            return None;
        }
        let mut ordinal = self.next;
        self.next += 1;
        if let CaptureHostF32::Uniform(value) = self.source {
            return Some(value);
        }
        let mut offset = 0usize;
        let mut pitch = 1usize;
        // All products and coordinates were checked by the original geometry
        // and full source extent validation before this iterator was born.
        for axis in (0..self.geometry.shape().len()).rev() {
            let coordinate = ordinal % self.geometry.shape()[axis];
            ordinal /= self.geometry.shape()[axis];
            offset += (self.geometry.starts()[axis] as usize
                + coordinate * self.geometry.strides()[axis] as usize)
                * pitch;
            pitch *= self.geometry.source_shape()[axis];
        }
        match self.source {
            CaptureHostF32::Dense(values) => Some(values[offset]),
            _ => unreachable!(),
        }
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.geometry.elements() - self.next;
        (remaining, Some(remaining))
    }
}
impl ExactSizeIterator for Selected<'_, '_> {}

pub(super) fn control_bytes() -> Option<usize> {
    let parts = [
        size_of::<CaptureHostF32<'_>>(),
        size_of::<Selected<'_, '_>>(),
        size_of::<(&[usize], CaptureHostF32<'_>)>(),
        size_of::<(usize, usize, usize, usize)>(),
        size_of::<std::ops::Range<usize>>(),
        size_of::<std::iter::Rev<std::ops::Range<usize>>>(),
        size_of::<Result<(), CaptureRunHostError>>(),
        size_of::<Result<ClaimedCaptureSummary, CaptureSummaryFailure>>(),
        crate::capture::partition::numeric_control_bytes()?,
    ];
    parts
        .into_iter()
        .try_fold(std::mem::size_of_val(&parts), usize::checked_add)
}

impl CaptureSummaryClaim<'_, '_> {
    /// Complete a synchronous host reduction under this exact spent account.
    ///
    /// The full source shape and backing are checked before any value is read.
    /// Fixed source copies, selected traversal and the existing arithmetic are
    /// included in `CaptureSummaryHostPlan` before original bank construction.
    /// The source stays borrowed through completion; no source alias, callback,
    /// unchecked summary or native completion permission escapes this worker.
    pub fn summarize_host_f32(
        self,
        shape: &[usize],
        source: CaptureHostF32<'_>,
    ) -> Result<ClaimedCaptureSummary, CaptureSummaryFailure> {
        let checked = (|| {
            self.identity.custody.validate()?;
            if shape != self.geometry().source_shape() {
                return Err(WorkingMemoryError::IdentityMismatch.into());
            }
            let extent = shape
                .iter()
                .try_fold(1usize, |n, &d| n.checked_mul(d))
                .ok_or(WorkingMemoryError::Overflow)?;
            if matches!(source, CaptureHostF32::Dense(values) if values.len() != extent) {
                return Err(WorkingMemoryError::IdentityMismatch.into());
            }
            Ok(())
        })();
        if let Err(error) = checked {
            // The zero-initialized scalar destination and original account
            // survive together even when source validation refuses the read.
            return Err(CaptureSummaryFailure {
                value: crate::capture::partition::summarize_f32(&[]),
                error,
                custody: self.identity.custody,
            });
        }
        let value = crate::capture::partition::summarize_f32_values(Selected {
            source,
            geometry: self.geometry(),
            next: 0,
        });
        self.finish(value)
    }
}
