//! Compatibility exports for portable speculative execution.

pub use eredu_runtime::speculative::{
    speculative_capture_scope, EmbeddedPredictionCache, EmbeddedPredictionCacheAccessError,
    EmbeddedPredictionCacheError, EmbeddedPredictionCheckpoint, EmbeddedPredictionDraftCache,
    EmbeddedPredictionDraftState, EmbeddedPredictionExecutor, EmbeddedPredictionObservers,
    EmbeddedPredictionOutput, EmbeddedPredictionStrategy, EmbeddedPredictionTargetState,
    ReplicatedMaterializedPredictionStrategy, ReplicatedPredictionInput,
    ReplicatedPredictionNative, SpeculativeActivationExecution, SpeculativeTensorMechanisms,
    EMBEDDED_PREDICTION_OUTPUT_PATH, EMBEDDED_PROPOSAL_LOGITS_PATH, EMBEDDED_TARGET_CAPTURE_PATH,
    EMBEDDED_VERIFICATION_LOGITS_PATH,
};
