//! Source-authenticated table planning over the original semantic source loan.
use super::*;
use eredu_nn::sequence_layout::{
    PatchEncoderTableError, PatchEncoderTableLayout, SequenceLayoutError,
};

/// Borrowed fixed-table recipe for the actual source admitted by A. Cloning the
/// scalar recipe cannot create a B account or native bind capability.
#[derive(Clone, Copy)]
pub struct PreparedMediaEncoderTablePlan<'a> {
    source: &'a OriginalPreparedHostInput,
    policy: QwenPolicy<'a>,
    layout: PatchEncoderTableLayout,
}
fn failure(error: PatchEncoderTableError) -> MediaSemanticError {
    match error {
        PatchEncoderTableError::Sequence(SequenceLayoutError::Overflow)
        | PatchEncoderTableError::Rotary(eredu_nn::multimodal::RotaryTableError::Overflow) => {
            MediaSemanticError::overflow("original media encoder source tables")
        }
        _ => MediaSemanticError::input("original media encoder source-table geometry"),
    }
}
impl OriginalPreparedMediaSemantics<'_> {
    /// Selected encoders without Qwen's fixed lookup tables still retain the
    /// same original B/source population. No empty table substitute is minted.
    pub fn optional_encoder_table_plan(
        &self,
    ) -> Result<Option<PreparedMediaEncoderTablePlan<'_>>, MediaSemanticError> {
        if matches!(
            Policy::source(self.0.provenance())?,
            Policy::Gemma(_) | Policy::Inkling(_) | Policy::Muse(_)
        ) {
            return Ok(None);
        }
        self.encoder_table_plan().map(Some)
    }
    /// Measures the selected tower's complete source tables before B admission.
    /// The returned loan keeps the same A/I/selection source available to fill.
    pub fn encoder_table_plan(
        &self,
    ) -> Result<PreparedMediaEncoderTablePlan<'_>, MediaSemanticError> {
        let selected = Policy::source(self.0.provenance())?;
        if matches!(
            selected,
            Policy::Gemma(_) | Policy::Inkling(_) | Policy::Muse(_)
        ) {
            return Err(MediaSemanticError::input(
                "selected encoder has no Qwen table source",
            ));
        }
        let policy = selected.qwen();
        let vision = policy.vision.ok_or(MediaSemanticError::input(
            "original media has no selected vision tower",
        ))?;
        let spec = crate::qwen::vision::encoder_table_spec(vision).map_err(failure)?;
        let source = self.source();
        let rows = rows(source, policy);
        let layout = PatchEncoderTableLayout::new(spec, rows, None).map_err(failure)?;
        Ok(PreparedMediaEncoderTablePlan {
            source,
            policy,
            layout,
        })
    }
}
fn rows<'a>(
    source: &'a OriginalPreparedHostInput,
    policy: QwenPolicy<'a>,
) -> impl Iterator<Item = (i32, i32, i32)> + Clone + 'a {
    parts(source, policy).flat_map(|part| {
        // A already validated these immutable source slots and policy. Grid
        // rows borrow aligned I32 chunks; no nested grid collection is made.
        let part = part.expect("immutable original semantic part");
        part.grid
            .iter()
            .map(|[time, height, width]| (*time, *height, *width))
    })
}
impl PreparedMediaEncoderTablePlan<'_> {
    /// Exact I source used to measure every table.
    pub fn source(&self) -> &OriginalPreparedHostInput {
        self.source
    }
    /// Fixed scalar layout; numerical geometry itself grants no authority.
    pub fn layout(&self) -> PatchEncoderTableLayout {
        self.layout
    }
    /// Fills only the actual destinations allocated by the enclosing B worker.
    pub fn fill(
        self,
        integers: &mut [i32],
        floats: &mut [f32],
    ) -> Result<(), PatchEncoderTableError> {
        self.layout
            .fill(rows(self.source, self.policy), integers, floats)
    }
}
