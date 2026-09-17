//! The actual single eager prompt source under this retained native allocator.
use super::*;

impl MlxNativeStorage {
    pub(crate) fn prompt_input_facts(
        &self,
        geometry: eredu_core::InferenceGeometry,
    ) -> Result<Option<safemlx::OriginalPromptInputFacts>, NativeStorageCause> {
        if geometry.batch_size != 1 {
            return Ok(None);
        }
        let Ok(elements) = usize::try_from(geometry.input_positions) else {
            return Ok(None);
        };
        let runtime = match &self.runtime {
            Ok(runtime) => runtime,
            Err(cause) => return Err(NativeStorageCause::Cold(cause.retained())),
        };
        match safemlx::OriginalPromptInputFacts::inspect(runtime, elements) {
            Ok(facts) => Ok(Some(facts)),
            Err(_) => Ok(None),
        }
    }
}
