//! Finite source-owned controller fixtures. Every constructor and operation is
//! the same paid worker used by public preparation; no fixture runtime exists.
use super::*;
use eredu_runtime::working_memory::{
    ControllerCompilationOutput, ControllerCompilationSources, InferenceExecutionIdentity,
    OriginalChatProfilePreparation, OriginalControllerCompilation, OriginalControllerCompiler,
    OriginalTokenizer, WorkingMemoryPool,
};

const CAPACITY: u64 = 1 << 30;
pub(crate) struct Compiler {
    pool: WorkingMemoryPool,
    execution: InferenceExecutionIdentity,
    validity: SharedTokenFilter,
    tokenizer: OriginalTokenizer,
    profile: OriginalChatProfilePreparation,
    eos: Vec<u32>,
}
pub(crate) struct Plan {
    plan: GenerationRuntimePlan,
    compilation: OriginalControllerCompilation,
    tokenizer: OriginalTokenizer,
    pool: WorkingMemoryPool,
    execution: InferenceExecutionIdentity,
    validity: SharedTokenFilter,
}
impl std::ops::Deref for Plan {
    type Target = GenerationRuntimePlan;
    fn deref(&self) -> &Self::Target {
        &self.plan
    }
}
struct Output(GenerationRuntimePlan);
impl ControllerCompilationOutput for Output {
    fn controller_sources(&self) -> ControllerCompilationSources<'_> {
        self.0.controller_sources()
    }
}
struct Compile<'a, F> {
    tokenizer: OriginalTokenizer,
    eos: &'a [u32],
    run: F,
}
impl<F> OriginalControllerCompiler for Compile<'_, F>
where
    F: FnOnce(&ConstraintCompiler) -> GenerationRuntimePlan,
{
    type Output = Output;
    type Error = std::convert::Infallible;
    fn compile(self, funding: &eredu_core::HostMetadataFunding) -> Result<Output, Self::Error> {
        let compiler =
            ConstraintCompiler::from_original_tokenizer(self.tokenizer, self.eos, funding).unwrap();
        Ok(Output((self.run)(&compiler)))
    }
}
impl Compiler {
    pub(crate) fn new(tokenizer: &ChatTokenizer, eos: &[u32]) -> Self {
        let pool = WorkingMemoryPool::new(CAPACITY, 0).unwrap();
        // Establish the same canonical baseline before source reservations,
        // including sparse IDs, exactly as model loading does.
        let validity = pool
            .prepare_shared_token_filter(|| {
                crate::api::tokenizer_token_filter_for_tests(tokenizer)
            })
            .unwrap();
        let tokenizer = pool
            .compile_tokenizer_source_for_generation(
                eredu_runtime::working_memory::OriginalTokenizerInput::Configuration(tokenizer),
            )
            .unwrap();
        assert!(tokenizer.generation_domain().unwrap() == validity.as_ref(),
            "original source must match ordinary tokenizer validity");
        let execution = InferenceExecutionIdentity::default();
        let template = pool
            .compile_chat_template(
                eredu_text::chat_storage::ChatTemplatePlan::prepare_utf8("fixture", "fixture")
                    .unwrap(),
            )
            .unwrap();
        let profile =
            OriginalChatProfilePreparation::new(&template, &tokenizer, &execution, CAPACITY)
                .unwrap();
        Self {
            pool,
            execution,
            validity,
            tokenizer,
            profile,
            eos: eos.to_vec(),
        }
    }
    pub(crate) fn byte_tokens(eos: &[u32]) -> Self {
        Self::byte_tokens_with_added(eos, &[], &[])
    }
    pub(crate) fn byte_tokens_with_added(eos: &[u32], special: &[&str], extra: &[&[u8]]) -> Self {
        // Explicit fixture alphabet: byte IDs remain 0..255 and the canonical
        // ByteLevel decoder yields exactly those bytes, including non-UTF8 IDs.
        let mut extended = 256u32;
        let vocab: tokenizers::models::bpe::Vocab = (0u32..256)
            .map(|byte| {
                let scalar = if (33..=126).contains(&byte)
                    || (161..=172).contains(&byte)
                    || (174..=255).contains(&byte)
                {
                    byte
                } else {
                    let scalar = extended;
                    extended += 1;
                    scalar
                };
                (char::from_u32(scalar).unwrap().to_string(), byte)
            })
            .collect();
        let model = tokenizers::models::bpe::BPE::builder()
            .vocab_and_merges(vocab, Vec::new())
            .build()
            .unwrap();
        let mut tokenizer = tokenizers::Tokenizer::new(model);
        tokenizer.with_pre_tokenizer(Some(
            tokenizers::pre_tokenizers::byte_level::ByteLevel::new(false, false, false),
        ));
        tokenizer.with_decoder(Some(tokenizers::decoders::byte_level::ByteLevel::default()));
        tokenizer
            .add_special_tokens(
                special
                    .iter()
                    .map(|text| tokenizers::AddedToken::from(*text, true).normalized(false)),
            )
            .unwrap();
        tokenizer
            .add_tokens(extra.iter().map(|bytes| {
                tokenizers::AddedToken::from(std::str::from_utf8(bytes).unwrap(), false)
                    .normalized(false)
            }))
            .unwrap();
        Self::new(&ChatTokenizer::from_tokenizer(tokenizer), eos)
    }
    pub(crate) fn compile(
        &self,
        run: impl FnOnce(&ConstraintCompiler) -> GenerationRuntimePlan,
    ) -> Plan {
        let (output, compilation) = self
            .profile
            .compile_controller(Compile {
                tokenizer: self.tokenizer.clone(),
                eos: &self.eos,
                run,
            })
            .unwrap();
        Plan {
            plan: output.0,
            compilation,
            tokenizer: self.tokenizer.clone(),
            pool: self.pool.clone(),
            execution: self.execution.clone(),
            validity: self.validity.clone(),
        }
    }
    pub(crate) fn compile_like(
        &self,
        source: &GenerationRuntimePlan,
        parallel: ParallelToolCallPolicy,
    ) -> Plan {
        let tools = source.semantic.recipe.tools().unwrap();
        self.compile(|compiler| {
            let funding = &compiler.allocation_funding;
            let mut spellings = Vec::new();
            let mut ids = Vec::new();
            for (id, text) in source.semantic.structural_tokens() {
                funding
                    .try_push(&mut spellings, funding.try_copy_str(text).unwrap())
                    .unwrap();
                funding.try_push(&mut ids, id).unwrap();
            }
            let mut stops = Vec::new();
            for text in source.semantic.recipe.stop_sequences() {
                funding
                    .try_push(&mut stops, funding.try_copy_str(text).unwrap())
                    .unwrap();
            }
            compiler
                .compile_generation_plan(
                    source.semantic.dialect,
                    source.semantic.dialect_parameters,
                    &tools,
                    source.tool_choice(),
                    parallel,
                    spellings,
                    ids,
                    stops,
                    source.has_tool_surface(),
                )
                .unwrap()
        })
    }
    pub(crate) fn compile_tool_plan(
        &self,
        dialect: &'static dyn FormatDialect,
        parameters: DialectParameters,
        tools: &[Value],
        choice: ToolChoice,
        parallel: ParallelToolCallPolicy,
        ids: Vec<u32>,
    ) -> Result<Plan, std::convert::Infallible> {
        Ok(self.compile(|compiler| {
            let funding = &compiler.allocation_funding;
            let strings = |source: &[&str]| {
                let mut result = Vec::new();
                funding.try_grow_vec(&mut result, source.len()).unwrap();
                for value in source {
                    result.push(funding.try_copy_str(value).unwrap());
                }
                result
            };
            compiler
                .compile_generation_plan(
                    dialect,
                    parameters,
                    tools,
                    choice,
                    parallel,
                    strings(dialect.required_structural_tokens(parameters).unwrap()),
                    ids,
                    strings(dialect.stop_sequences(parameters).unwrap()),
                    true,
                )
                .unwrap()
        }))
    }
}
impl Plan {
    pub(crate) fn funding(&self) -> eredu_core::HostMetadataFunding {
        self.pool
            .prepare_workspace_metadata(&self.execution, CAPACITY)
            .unwrap()
    }
    pub(crate) fn controller(&self) -> ConstraintController {
        let preparation = eredu_runtime::working_memory::PreparedSemanticSource::new(
            &self.tokenizer,
            &self.execution,
            CAPACITY,
        )
        .unwrap();
        let validity = self.validity.clone();
        self.controller_with_funding(validity, preparation.metadata_funding())
            .bind_preparation(&preparation)
            .unwrap()
    }
    pub(crate) fn controller_with_validity(
        &self,
        validity: SharedTokenFilter,
    ) -> ConstraintController {
        self.controller_with_funding(validity, &self.funding())
    }
    fn controller_with_funding(
        &self,
        validity: SharedTokenFilter,
        funding: &eredu_core::HostMetadataFunding,
    ) -> ConstraintController {
        ConstraintController::from_original_generation_plan_with(
            &self.plan,
            &self.compilation,
            validity,
            4096,
            &funding,
            |validity| {
                ConstraintController::from_original_forbidden_generation_plan_with(
                    &self.plan,
                    validity,
                    4096,
                    &funding,
                    |trigger| {
                        Ok(self
                            .pool
                            .compile_forbidden_tokenizer_source(&self.tokenizer, trigger)?)
                    },
                )
            },
        )
        .unwrap()
    }
}
