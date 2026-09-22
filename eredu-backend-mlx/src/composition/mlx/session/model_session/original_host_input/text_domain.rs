//! Canonical IDs from the authenticated original host source; no native readback.
use super::*;
use eredu_core::{InputPayloadKind, TokenInputRejection as R};
use eredu_runtime::{
    input::host::{HostInputPartView, HostTensorValues},
    working_memory::OriginalTokenizer,
};

pub(in crate::composition::mlx::session::model_session) fn validate_text_domain(
    runtime: &ModelRuntime<MlxBackend<'_>>,
    prompt: &MlxModelInput,
    tokenizer: &OriginalTokenizer,
) -> Result<(), R> {
    let pool = runtime.backend().memory_ledger();
    tokenizer
        .validate_pool(pool)
        .map_err(|_| R::IdentityMismatch)?;
    let domain = tokenizer.generation_domain().ok_or(R::Unsupported)?;
    let input::OriginalMediaPacket::Original(completed) =
        prompt.original_media.as_ref().ok_or(R::Unsupported)?
    else {
        return Err(R::Unsupported);
    };
    if prompt.memory_owner.is_some() || prompt.quote.is_some() || prompt.inference_request.is_some()
    {
        return Err(R::IdentityMismatch);
    }
    let cache = prompt.cache_identity.as_ref().ok_or(R::IdentityMismatch)?;
    completed
        .validate_request_source(pool, &prompt.parts, cache)
        .map_err(|_| R::IdentityMismatch)?;
    for part in completed.borrowed_semantics().source().parts() {
        if part.kind() != InputPayloadKind::TokenIds {
            continue;
        }
        let valid = match part.payload().values {
            HostTensorValues::U32(ids) => ids.iter().all(|&id| domain.allows(id)),
            HostTensorValues::I32(ids) => ids
                .iter()
                .all(|&id| u32::try_from(id).is_ok_and(|id| domain.allows(id))),
            _ => return Err(R::InvalidToken),
        };
        if !valid {
            return Err(R::InvalidToken);
        }
    }
    Ok(())
}
