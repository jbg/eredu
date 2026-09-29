//! Exact request positions for chunked prefill records and local selector geometry.
use super::*;

/// Half-open decoder positions in the original admitted request. Media positions
/// use the architecture's assembled token sequence, not its raw patch count.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapturePrefillSpan {
    /// First absolute decoder position in this invocation.
    pub start: u64,
    /// Exclusive final absolute decoder position.
    pub end: u64,
    /// Complete admitted decoder request length.
    pub total: u64,
}
impl CapturePrefillSpan {
    /// Validates a nonempty chunk within the complete request.
    pub fn validate(self) -> Result<(), CaptureError> {
        if self.start >= self.end || self.end > self.total {
            return Err(CaptureError::Invalid(
                "prefill capture span is outside its request".into(),
            ));
        }
        Ok(())
    }
}

impl AdmittedCapturePlan {
    /// Projects this ordinary capture authority onto one actual request chunk.
    /// Fixed-width axes keep their meanings; absolute sequence/token slices are
    /// intersected and translated to physical rows without materializing tensors.
    pub fn for_prefill_span(&self, span: CapturePrefillSpan) -> Result<Self, CaptureError> {
        span.validate()?;
        if self.invocation_bounds.is_some()
            || self.prefill_span.is_some()
            || span.total != self.request.prompt_tokens
        {
            return Err(CaptureError::Invalid(
                "prefill span differs from ordinary capture admission".into(),
            ));
        }
        if self.request.batch != 1 {
            return Err(CaptureError::Unsupported(
                "chunked capture currently requires one decoder sequence".into(),
            ));
        }
        let mut projected = self.clone();
        for (selection, point) in projected.plan.selections.iter_mut().zip(&projected.points) {
            for slice in &mut selection.slices {
                let axis = point
                    .axes
                    .as_ref()
                    .and_then(|axes| axes.iter().find(|axis| axis.name == slice.axis))
                    .ok_or_else(|| {
                        CaptureError::Invalid("prefill selection axis is absent".into())
                    })?;
                let (start, end) = match axis.dimension {
                    SymbolicDimension::Sequence | SymbolicDimension::TokenRows => {
                        (span.start, span.end)
                    }
                    SymbolicDimension::Context => (0, span.end),
                    _ => continue,
                };
                let lower = slice.start.max(start);
                let upper = slice.end.min(end);
                // Preserve the original stride origin even when a chunk starts
                // between selected rows. Empty intersections never touch native data.
                let first = add(
                    slice.start,
                    mul((lower - slice.start).div_ceil(slice.stride), slice.stride)?,
                )?;
                if first >= upper {
                    slice.start = 0;
                    slice.end = 0;
                } else {
                    slice.start = first - start;
                    slice.end = upper - start;
                }
            }
        }
        projected.prefill_span = Some(span);
        let bytes = serde_json::to_vec(&(self.identity(), span))
            .map_err(|error| CaptureError::Invalid(error.to_string()))?;
        projected.identity = Sha256::digest(bytes)
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect();
        Ok(projected)
    }

    /// Absolute attribution of a projected prefill authority.
    pub fn prefill_span(&self) -> Option<CapturePrefillSpan> {
        self.prefill_span
    }
}
