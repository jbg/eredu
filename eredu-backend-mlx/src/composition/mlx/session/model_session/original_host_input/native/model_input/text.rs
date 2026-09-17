//! Original plain token input through the same full source compiler as media.
use super::*;
use std::num::NonZeroU64;

pub(crate) struct MlxPreparedTextInputPlan<'a>(FullCompiler<'a>);
impl MlxPreparedInputMaterializer {
    /// Plain selection is structural. Family/tokenizer/vocabulary policy stays
    /// in the caller's shared semantic preparation and model invocation.
    pub(crate) fn text_input_plan<'a>(
        &'a self,
        source: &'a OriginalPreparedHostInput,
    ) -> Result<MlxPreparedTextInputPlan<'a>, WorkingMemoryError> {
        let mut parts = source.parts();
        let part = parts.next().ok_or(WorkingMemoryError::IdentityMismatch)?;
        let payload = part.payload();
        if parts.next().is_some()
            || part.modality() != eredu_core::InputModality::Text
            || part.kind() != eredu_core::InputPayloadKind::TokenIds
            || part.metadata().len() != 0
            || !part.extents().is_empty()
            || !matches!(payload.values, HostTensorValues::U32(_))
            || !matches!(payload.shape, [1, n] if *n > 0)
        {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        Ok(MlxPreparedTextInputPlan(
            self.full_source_plan(source, None)?,
        ))
    }
}
impl MlxPreparedTextInputPlan<'_> {
    pub(crate) fn required_bytes(&self) -> Result<u64, WorkingMemoryError> {
        WorkingMemoryPool::prepared_native_input_required_bytes(&self.0)
    }
    pub(crate) fn materialize(
        self,
        pool: &WorkingMemoryPool,
    ) -> Result<MlxOriginalPreparedTextInput, MlxPreparedModelInputError> {
        pool.compile_prepared_native_input(self.0)
            .map(MlxOriginalPreparedTextInput)
            .map_err(MlxPreparedModelInputError)
    }
}
/// Only the completed B compiler can create the original text packet.
pub(crate) struct MlxOriginalPreparedTextInput(OriginalPreparedInputMaterialization<Body>);
impl MlxOriginalPreparedTextInput {
    pub(crate) fn into_prompt(self, chunk: Option<NonZeroU64>) -> MlxModelInput {
        let packet = CompletedOriginalTextInput {
            body: self.0.storage().clone(),
        };
        let prompt = MlxModelInput {
            parts: super::super::super::super::pending_prompt::ModelInputParts::OriginalText(
                packet,
            ),
            controlled_attribution: None,
            prepared_capture: None,
            original_media: None,
            cache_identity: self.0.storage().cache().cloned(),
            prefill_chunk_positions: chunk,
            inference_request: None,
            memory_owner: None,
            quote: None,
        };
        drop(self);
        prompt
    }
}
#[derive(Clone)]
pub(in crate::composition::mlx::session::model_session) struct CompletedOriginalTextInput {
    body: Body,
}
impl std::fmt::Debug for CompletedOriginalTextInput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("CompletedOriginalTextInput")
    }
}
impl CompletedOriginalTextInput {
    pub(in crate::composition::mlx::session::model_session) fn prediction_source(
        &self,
        pool: &WorkingMemoryPool,
        parts: &[input::InputPart],
        cache: &eredu_runtime::SharedPreparedInputCacheIdentity,
    ) -> Result<&eredu_runtime::input::PreparedModelInputOwner<crate::MlxTensor>, WorkingMemoryError>
    {
        self.body.source().validate_pool(pool)?;
        if !std::ptr::eq(self.parts(), parts)
            || !self
                .body
                .cache()
                .is_some_and(|value| value.same_storage(cache))
        {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        let prepared = self
            .body
            .prepared()
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        if !prepared
            .original_source()
            .is_some_and(|source| source.same_source(self.body.source()))
        {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        Ok(prepared)
    }
    pub(in crate::composition::mlx::session::model_session) fn parts(&self) -> &[input::InputPart] {
        self.body.parts().expect("complete text B").as_ref()
    }
    fn copy_source(
        &self,
    ) -> Result<crate::backend::array_copy::OriginalPreparedArrayCopySource<'_>, WorkingMemoryError>
    {
        crate::backend::array_copy::OriginalPreparedArrayCopySource::from_input(&self.body)
    }
}
impl MlxModelInput {
    pub(crate) fn original_text_copy_source(
        &self,
    ) -> Option<
        Result<crate::backend::array_copy::OriginalPreparedArrayCopySource<'_>, WorkingMemoryError>,
    > {
        match &self.parts {
            super::super::super::super::pending_prompt::ModelInputParts::OriginalText(source) => {
                Some(source.copy_source())
            }
            _ => None,
        }
    }
}
pub(super) fn control_bytes() -> usize {
    size_of::<(
        MlxPreparedTextInputPlan<'_>,
        MlxOriginalPreparedTextInput,
        CompletedOriginalTextInput,
        Result<MlxOriginalPreparedTextInput, MlxPreparedModelInputError>,
        Option<NonZeroU64>,
        MlxModelInput,
    )>()
}
