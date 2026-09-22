//! Exact ordinary request windows; descriptors convey no claim or native authority.
use crate::prefill::PrefillChunk;
use eredu_core::{capture::*, intervention::*, InferenceGeometry, SymbolicDimension};
mod projection;
pub use projection::InterventionPrefillProjectionError;
use std::mem::{size_of, size_of_val};
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum InterventionPrefillSourceError {
    #[error("intervention prefill source differs from the admitted ordinary request")]
    Identity,
    #[error("intervention prefill chunk differs from the canonical source schedule")]
    Chunk,
    #[error("intervention prefill requires a single declared row axis and qualified evidence")]
    Profile,
    #[error("intervention prefill source arithmetic overflow")]
    Overflow,
}
/// A checked canonical chunk over one original ordinary request. Its account,
/// operation claim and submission authority remain with their existing owners.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InterventionPrefillWindow {
    inference: InferenceGeometry,
    range: [u64; 2],
}
impl InterventionPrefillWindow {
    pub fn new(
        plan: &AdmittedInterventionPlan,
        inference: InferenceGeometry,
        chunk: &PrefillChunk,
    ) -> Result<Self, InterventionPrefillSourceError> {
        use InterventionPrefillSourceError as E;
        inference.validate_fixed().map_err(|_| E::Identity)?;
        let request = plan.request();
        let origin = plan.text_origin().ok_or(E::Identity)?;
        if plan.invocation_bounds().is_some()
            || request.batch != 1
            || request.prompt_tokens == 0
            || inference.batch_size != request.batch
            || inference.input_positions != request.prompt_tokens
            || inference.cached_positions != origin.cached_positions
            || inference.max_output_tokens > request.max_predictions
            || inference.prefill_chunk_positions == 0
        {
            return Err(E::Identity);
        }
        let [start, end] = [chunk.input.start, chunk.input.end];
        if start >= end
            || end > request.prompt_tokens
            || start % inference.prefill_chunk_positions != 0
            || start
                .checked_add(inference.prefill_chunk_positions)
                .ok_or(E::Overflow)?
                .min(request.prompt_tokens)
                != end
            || chunk.position
                != origin
                    .cached_positions
                    .checked_add(start)
                    .ok_or(E::Overflow)?
            || chunk.output != inference.output.for_chunk(end == request.prompt_tokens)
        {
            return Err(E::Chunk);
        }
        Ok(Self {
            inference,
            range: [start, end],
        })
    }
    pub fn validate(
        self,
        plan: &AdmittedInterventionPlan,
    ) -> Result<(), InterventionPrefillSourceError> {
        Self::new(
            plan,
            self.inference,
            &PrefillChunk {
                input: self.range[0]..self.range[1],
                position: self
                    .inference
                    .cached_positions
                    .checked_add(self.range[0])
                    .ok_or(InterventionPrefillSourceError::Overflow)?,
                output: self.inference.output.for_chunk(self.is_final()),
            },
        )
        .map(|_| ())
    }
    /// The exact admitted schedule retained by this checked window.
    pub fn inference(self) -> InferenceGeometry {
        self.inference
    }
    pub fn range(self) -> [u64; 2] {
        self.range
    }
    pub fn is_final(self) -> bool {
        self.range[1] == self.inference.input_positions
    }
    pub fn logical_positions(self) -> u64 {
        self.inference.input_positions
    }
    pub fn physical(self) -> CaptureInvocationShape {
        CaptureInvocationShape {
            batch: self.inference.batch_size,
            sequence: self.range[1] - self.range[0],
            context: Some(self.inference.cached_positions + self.range[1]),
        }
    }
    pub fn window(self) -> CaptureInvocationWindow {
        CaptureInvocationWindow {
            logical_sequence: self.inference.input_positions,
            start: self.range[0],
        }
    }
    /// Single terminal hooks remain on the existing ordinary row driver.
    pub fn row_axis(point: &InterventionPoint) -> bool {
        point.axes.iter().any(|axis| {
            matches!(
                axis.dimension,
                SymbolicDimension::Sequence | SymbolicDimension::TokenRows
            )
        })
    }
    pub fn validate_operation(
        plan: &AdmittedInterventionPlan,
        index: usize,
    ) -> Result<(), InterventionPrefillSourceError> {
        Self::validate_operation_profile(plan, index)?;
        if plan.plan().operations[index].evidence != InterventionEvidence::None {
            return Err(InterventionPrefillSourceError::Profile);
        }
        Ok(())
    }
    /// Source-only companion qualification. Execution additionally requires the
    /// original prepared companion frame and actual side-specific native loan.
    pub fn validate_evidence_operation(
        source: &crate::working_memory::OriginalInterventionSource,
        index: usize,
    ) -> Result<(), InterventionPrefillSourceError> {
        let plan = source.plan().admission();
        Self::validate_operation_profile(plan, index)?;
        let companion = source
            .plan()
            .evidence(index)
            .ok_or(InterventionPrefillSourceError::Profile)?;
        if plan.plan().operations[index].evidence == InterventionEvidence::None
            || companion
                .shared_geometry_source()
                .admission()
                .plan()
                .selections
                .len()
                != if plan.points()[index].routing.is_some() {
                    4
                } else {
                    2
                }
        {
            return Err(InterventionPrefillSourceError::Profile);
        }
        Ok(())
    }
    /// Descriptive progression for the original before/after companion in this
    /// canonical window. It grants no host claim or native execution authority.
    pub fn evidence_policy<'a>(
        self,
        source: &'a crate::working_memory::OriginalInterventionSource,
        index: usize,
    ) -> Result<
        crate::capture::CapturePrefillObservationPolicy<'a>,
        crate::capture::CapturePrefillProgressError,
    > {
        self.validate(source.plan().admission())
            .and_then(|()| Self::validate_evidence_operation(source, index))
            .map_err(|_| crate::capture::CapturePrefillProgressError::Identity)?;
        crate::capture::CapturePrefillObservationPolicy::new(
            source
                .plan()
                .evidence(index)
                .expect("validated companion")
                .shared_geometry_source(),
            self.inference,
        )
    }
    fn validate_operation_profile(
        plan: &AdmittedInterventionPlan,
        index: usize,
    ) -> Result<(), InterventionPrefillSourceError> {
        use InterventionPrefillSourceError as E;
        let point = plan.points().get(index).ok_or(E::Identity)?;
        if !matches!(
            point.stage,
            InterventionStage::Activation
                | InterventionStage::LogitsBeforeSampling
                | InterventionStage::RoutingBeforeDispatch
        ) || point
            .axes
            .iter()
            .filter(|axis| {
                matches!(
                    axis.dimension,
                    SymbolicDimension::Sequence | SymbolicDimension::TokenRows
                )
            })
            .count()
            != 1
        {
            return Err(E::Profile);
        }
        Ok(())
    }
    pub fn control_bytes() -> Option<usize> {
        let frames = [
            size_of::<Self>(),
            size_of::<[Self; 2]>(),
            size_of::<PrefillChunk>(),
            size_of::<(&AdmittedInterventionPlan, InferenceGeometry, &PrefillChunk)>(),
            size_of::<Result<Self, InterventionPrefillSourceError>>(),
            size_of::<Result<(), InterventionPrefillSourceError>>(),
            size_of::<CaptureInvocationWindow>(),
            size_of::<CaptureInvocationShape>(),
        ];
        frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)
    }
}
