//! Optional released-source grammar funding probe; no model/device execution.
use super::*;
use crate::runtime::chat::dialect::DECLARATIVE_DIALECT;
use eredu_core::{HostMetadataAccount, HostMetadataFunding, HostMetadataFundingError};
use eredu_runtime::working_memory::{
    ControllerCompilationOutput, ControllerCompilationSources, InferenceExecutionIdentity,
    OriginalChatProfilePreparation, OriginalControllerCompiler, OriginalTokenizer,
    WorkingMemoryPool,
};
use std::sync::{Mutex, atomic::AtomicU64};

#[derive(Debug, Default)]
struct Counts {
    total: AtomicU64,
    calls: AtomicU64,
    sizes: Mutex<std::collections::BTreeMap<usize, u64>>,
    max_calls: u64,
    max_bytes: u64,
}
#[derive(Debug)]
struct Account(Arc<Counts>);
impl HostMetadataAccount for Account {
    fn reserve_metadata(&self, bytes: usize) -> Result<(), HostMetadataFundingError> {
        let calls = self.0.calls.load(Ordering::Relaxed);
        let available = self
            .0
            .max_bytes
            .saturating_sub(self.0.total.load(Ordering::Relaxed));
        if calls >= self.0.max_calls || bytes as u64 > available {
            return Err(HostMetadataFundingError::Capacity {
                required: bytes as u64,
                available,
            });
        }
        self.0.total.fetch_add(bytes as u64, Ordering::Relaxed);
        self.0.calls.fetch_add(1, Ordering::Relaxed);
        *self.0.sizes.lock().unwrap().entry(bytes).or_default() += 1;
        Ok(())
    }
}
fn report(label: &str, counts: &Counts) {
    let mut rows: Vec<_> = counts
        .sizes
        .lock()
        .unwrap()
        .iter()
        .map(|(&size, &count)| (size as u64 * count, size, count))
        .collect();
    rows.sort_unstable_by(|a, b| b.cmp(a));
    eprintln!(
        "GRAMMAR_FUNDING stage={label} bytes={} calls={} largest={:?}",
        counts.total.load(Ordering::Relaxed),
        counts.calls.load(Ordering::Relaxed),
        &rows[..rows.len().min(12)]
    );
}
struct Output(GenerationRuntimePlan);
impl ControllerCompilationOutput for Output {
    fn controller_sources(&self) -> ControllerCompilationSources<'_> {
        self.0.controller_sources()
    }
}
#[derive(Debug, thiserror::Error)]
enum Failure {
    #[error(transparent)]
    Source(#[from] ConstraintCompilerSourceError),
    #[error(transparent)]
    Plan(#[from] preparation_error::PreparationFailure),
    #[error(transparent)]
    Funding(#[from] HostMetadataFundingError),
}
struct Compile<'a> {
    source: OriginalTokenizer,
    eos: &'a [u32],
    tools: &'a [Value],
}
impl OriginalControllerCompiler for Compile<'_> {
    type Output = Output;
    type Error = Failure;
    fn compile(self, funding: &HostMetadataFunding) -> Result<Output, Failure> {
        let compiler = ConstraintCompiler::from_original_tokenizer(self.source, self.eos, funding)?;
        let marker = "<|im_end|>";
        funding.reserve_metadata(
            2 * size_of::<Vec<String>>()
                + size_of::<Vec<u32>>()
                + 2 * size_of::<String>()
                + size_of::<u32>()
                + 2 * marker.len(),
        )?;
        Ok(Output(compiler.compile_generation_plan(
            &DECLARATIVE_DIALECT,
            DialectParameters::Declarative(
                &crate::runtime::chat::QWEN_TAGGED_TOOL_SPEC_NO_REASONING,
            ),
            self.tools,
            ToolChoice::Required,
            ParallelToolCallPolicy::Disabled,
            vec![marker.into()],
            vec![self.eos[0]],
            vec![marker.into()],
            true,
        )?))
    }
}
#[test]
#[ignore = "requires EREDU_RELEASED_TOKENIZER_JSON pointing to a pinned released tokenizer"]
fn released_required_grammar_funding() {
    run(false);
}

#[test]
#[ignore = "requires EREDU_RELEASED_TOKENIZER_JSON pointing to a pinned released tokenizer"]
fn released_required_prose_prefix_under_finite_funding() {
    run(true);
}

fn run(full: bool) {
    let path = std::env::var("EREDU_RELEASED_TOKENIZER_JSON").unwrap();
    let source = std::fs::read(&path).unwrap();
    let tokenizer =
        ChatTokenizer::from_tokenizer(tokenizers::Tokenizer::from_bytes(&source).unwrap());
    let eos = [tokenizer.token_to_id("<|im_end|>").unwrap()];
    let pool = WorkingMemoryPool::new(2 << 30, 0).unwrap();
    let validity = pool
        .prepare_shared_token_filter(|| eredu_core::TokenFilter::All)
        .unwrap();
    let original = pool
        .compile_tokenizer(
            eredu_text::tokenizer_storage::TokenizerPlan::prepare_json(&source).unwrap(),
        )
        .unwrap();
    let template = pool
        .compile_chat_template(
            eredu_text::chat_storage::ChatTemplatePlan::prepare_utf8("probe", "probe").unwrap(),
        )
        .unwrap();
    let prep = OriginalChatProfilePreparation::new(
        &template,
        &original,
        &InferenceExecutionIdentity::default(),
        2 << 30,
    )
    .unwrap();
    let tools = [
        json!({"type":"function","function":{"name":"reading","parameters":{"type":"object","properties":{"value":{"type":"integer","enum":[17]}},"required":["value"],"additionalProperties":false}}}),
    ];
    let (output, receipt) = prep
        .compile_controller(Compile {
            source: original,
            eos: &eos,
            tools: &tools,
        })
        .unwrap();
    eprintln!("GRAMMAR_FUNDING source_pool={}", pool.used_bytes().unwrap());
    let counts = Arc::new(Counts {
        max_calls: 200_000,
        max_bytes: if full { 2 << 30 } else { 32 << 30 },
        ..Counts::default()
    });
    let funding = HostMetadataFunding::new(Account(counts.clone())).unwrap();
    let reached = |label: &str, error: &dyn std::fmt::Display| {
        report(label, &counts);
        assert!(!full, "released prose-prefix regression: {error}");
        assert_eq!(
            counts.calls.load(Ordering::Relaxed),
            200_000,
            "unexpected failure: {error}"
        );
        eprintln!("GRAMMAR_FUNDING bounded stop: {error}");
    };
    if full {
        use eredu_core::TokenFilterController;
        let mut controller = ConstraintController::from_original_generation_plan_with(
            &output.0,
            &receipt,
            validity,
            48,
            &funding,
            |_| panic!("Required tools must select the active grammar"),
        )
        .unwrap();
        report("startup", &counts);
        let ids = tokenizer.encode("I am unable to", false).unwrap();
        assert_eq!(ids.get_ids().len(), 4);
        for (index, &token) in ids.get_ids().iter().enumerate() {
            let filter = controller.current_filter().unwrap();
            assert!(
                filter.allows(token),
                "Required grammar must preserve legal prose prefixes"
            );
            assert!(!filter.allows(eos[0]), "Required cannot stop before a tool call");
            report(&format!("mask{index}"), &counts);
            controller.commit_token(token).unwrap();
            report("commit", &counts);
            assert!(!controller.is_complete().unwrap());
            report("terminal", &counts);
        }
        let filter = controller.current_filter().unwrap();
        assert!(
            !filter.allows(eos[0]),
            "a prose prefix cannot complete Required tool output"
        );
        report("final mask", &counts);
        return;
    }
    let state = match output
        .0
        .generation_constraint()
        .inner
        .original_grammar_state(&receipt, &funding)
    {
        Ok(state) => state,
        Err(error) => {
            reached("startup refusal", &error);
            return;
        }
    };
    report("startup", &counts);
    match state.compute_mask() {
        Ok(_) => report("mask0", &counts),
        Err(error) => reached("mask0 refusal", &error),
    };
}
use std::mem::size_of;
