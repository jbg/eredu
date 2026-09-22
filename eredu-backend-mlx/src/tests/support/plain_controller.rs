//! Native fixture tokenizer preparation for the reusable neutral controller.
use eredu_core::MemoryLimits;
pub(crate) use eredu_evaluation::execution_control::fixture::PlainController;
use eredu_runtime::working_memory::{
    InferenceExecutionIdentity, MemoryLedger, PreparedSemanticSource,
};
pub(crate) fn source(
    pool: &MemoryLedger,
    execution: &InferenceExecutionIdentity,
) -> (
    eredu_runtime::working_memory::OriginalTokenizer,
    PreparedSemanticSource,
) {
    let tokenizer = tokenizer(pool, 64);
    let prepared = PreparedSemanticSource::new(
        &tokenizer,
        execution,
        MemoryLimits::unlimited(pool.topology()),
    )
    .unwrap();
    (tokenizer, prepared)
}

fn tokenizer(
    pool: &MemoryLedger,
    vocabulary: usize,
) -> eredu_runtime::working_memory::OriginalTokenizer {
    assert!(vocabulary > 0);
    let vocab = (0..vocabulary)
        .map(|i| (format!("t{i}"), serde_json::Value::from(i)))
        .collect::<serde_json::Map<_, _>>();
    let input = serde_json::to_vec(&serde_json::json!({"version":"1.0", "truncation":null,
            "padding":null,"normalizer":null,"pre_tokenizer":null,"post_processor":null,
            "decoder":null,"added_tokens":[],"model":{"type":"BPE","vocab":vocab,"merges":[]}}))
    .unwrap();
    let tokenizer = pool
        .compile_tokenizer(
            eredu_text::tokenizer_storage::TokenizerPlan::prepare_json(&input)
                .unwrap()
                .with_generation_domain()
                .unwrap(),
        )
        .unwrap();
    tokenizer
}

/// Cold immutable sources precede the semantic account that excludes unquoted
/// construction. Aliases reuse these same sources across independent runs.
pub(crate) struct ControllerSource {
    pub(crate) prepared: PreparedSemanticSource,
    pub(crate) validity: eredu_core::SharedTokenFilter,
    pub(crate) stops: eredu_runtime::working_memory::OriginalStopSource,
}

pub(crate) fn controller_source(
    pool: &MemoryLedger,
    execution: &InferenceExecutionIdentity,
    vocabulary: usize,
) -> ControllerSource {
    let tokenizer = tokenizer(pool, vocabulary);
    let validity = pool
        .prepare_shared_token_filter(|| tokenizer.generation_domain().unwrap().clone())
        .unwrap();
    let stops = pool
        .compile_stop_source(eredu_text::stop_storage::StopCompilePlan::prepare_refs(&[]).unwrap())
        .unwrap();
    let prepared = PreparedSemanticSource::new(
        &tokenizer,
        execution,
        MemoryLimits::unlimited(pool.topology()),
    )
    .unwrap();
    ControllerSource {
        prepared,
        validity,
        stops,
    }
}

pub(crate) fn new(
    pool: &MemoryLedger,
    execution: &InferenceExecutionIdentity,
    capacity: usize,
) -> (PlainController, eredu_core::SemanticStateOwner) {
    let ControllerSource {
        prepared,
        validity,
        stops,
    } = controller_source(pool, execution, 64);
    let controller = PlainController::from_prepared(&prepared, validity, capacity);
    let semantic = prepared
        .prepare(
            &stops,
            capacity,
            std::num::NonZeroUsize::new(32).unwrap(),
            false,
        )
        .unwrap();
    (controller, semantic)
}
