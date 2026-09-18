use super::*;
use eredu_core::{HostPreparationAuthority, TokenSamplingDecision, TextControllerContract,
    speculative::{PlainControllerHistory, ForbiddenControllerSource, NoPreparedGrammar,
        byte_trigger::TriggerPrefix}};
use eredu_text::tokenizer_storage::TokenizerPlan;

#[derive(Clone)]
struct Controller {
    history: PlainControllerHistory,
    source: OriginalForbiddenSource,
    validity: SharedTokenFilter,
    binding: Option<PreparedControllerBinding>,
}
impl TokenFilterController for Controller {
    type Error = std::convert::Infallible;
    fn inference_storage(&self) -> TextControllerStorage<'_> {
        self.binding.as_ref().map_or(TextControllerStorage::Unknown, |b| b.storage(self))
    }
    fn inference_workspace(&self, _: u64) -> Option<TextControllerWorkspace<'_>> {
        self.binding.as_ref()?.workspace()
    }
    fn current_filter(&mut self) -> Result<TokenFilter, Self::Error> { Ok(self.validity.as_ref().clone()) }
    fn commit_token(&mut self, _: u32) -> Result<(), Self::Error> { Ok(()) }
    fn is_complete(&mut self) -> Result<bool, Self::Error> { Ok(false) }
}
impl SpeculativeTokenFilterController for Controller {
    type PreparedGrammar = NoPreparedGrammar;
    fn filter_at(&self, _: &[u32]) -> Result<TokenFilter, Self::Error> { Ok(self.validity.as_ref().clone()) }
    fn prefix_is_complete(&self, _: &[u32]) -> Result<bool, Self::Error> { Ok(false) }
    fn prepared_forbidden_source(&self) -> Option<ForbiddenControllerSource<'_>> {
        Some(ForbiddenControllerSource::new(&self.history, &self.validity, self.source.inputs(),
            TriggerPrefix::default()).unwrap().with_original_storage(OriginalSourceWitness::new(&self.source)))
    }
}
fn tokenizer(pool: &WorkingMemoryPool) -> OriginalTokenizer {
    let json = br#"{"version":"1.0","truncation":null,"padding":null,"normalizer":null,
        "pre_tokenizer":null,"post_processor":null,"decoder":{"type":"ByteLevel",
        "add_prefix_space":false,"trim_offsets":false,"use_regex":false},"added_tokens":[],
        "model":{"type":"BPE","vocab":{"h":0,"i":1},"merges":[]}}"#;
    pool.compile_tokenizer(TokenizerPlan::prepare_json(json).unwrap().with_generation_domain().unwrap()).unwrap()
}
fn history(preparation: &PreparedSemanticSource) -> PlainControllerHistory {
    let funding = preparation.metadata_funding();
    funding.reserve_metadata(PlainControllerHistory::copy_metadata_bytes(3).unwrap()
        + HostPreparationAuthority::retention_bytes::<eredu_core::HostMetadataFunding>().unwrap()).unwrap();
    PlainControllerHistory::default().copy_prepared(3, HostPreparationAuthority::retain(funding.clone())).unwrap()
}

#[test]
fn semantic_admission_authenticates_actual_source_execution_and_history_payer() {
    let capacity = 64 * 1024 * 1024;
    let pool = WorkingMemoryPool::new(capacity, 0).unwrap();
    let tokenizer = tokenizer(&pool);
    let execution = InferenceExecutionIdentity::default();
    let preparation = PreparedSemanticSource::new(&tokenizer, &execution, capacity).unwrap();
    let mut controller = Controller {
        history: history(&preparation),
        source: pool.compile_forbidden_tokenizer_source(&tokenizer, b"hi").unwrap(),
        validity: SharedTokenFilter::new(tokenizer.generation_domain().unwrap().clone()), binding: None,
    };
    let binding = PreparedControllerBinding::new(&preparation, &controller).unwrap();
    let stop_rows: [String; 0] = [];
    let stops = pool.compile_stop_source(eredu_text::stop_storage::StopCompilePlan::prepare(&stop_rows).unwrap()).unwrap();
    let semantic = preparation.prepare(&stops, 3, std::num::NonZeroUsize::new(1).unwrap(), true).unwrap();
    let semantic = semantic.prepared_source().unwrap().downcast_ref::<crate::working_memory::PreparedSemanticState>().unwrap();
    let actual_source = PreparedControllerBinding::source(&controller).unwrap();
    binding.validate_semantic_state(semantic, actual_source, 3).unwrap();
    assert!(binding.validate_semantic_state(semantic, actual_source, 2).is_err());
    let other_preparation = PreparedSemanticSource::new(&tokenizer, &execution, capacity).unwrap();
    let other_semantic = other_preparation.prepare(&stops, 3, std::num::NonZeroUsize::new(1).unwrap(), true).unwrap();
    let other_semantic = other_semantic.prepared_source().unwrap().downcast_ref::<crate::working_memory::PreparedSemanticState>().unwrap();
    assert!(binding.validate_semantic_state(other_semantic, actual_source, 3).is_err());
    controller.binding = Some(binding.clone());
    let workspace = controller.inference_workspace(3).unwrap();
    let storage = ControllerStorageContract::inspect_original_retained(&controller, workspace,
        &pool, &execution, 3).unwrap();
    assert!(ControllerStorageContract::inspect(&controller).is_err());
    assert!(ControllerStorageContract::inspect_original_retained(&controller, workspace,
        &pool, &InferenceExecutionIdentity::default(), 3).is_err());
    let contract = TextControllerContract::from_workspace(workspace, 2).unwrap();
    let domain = tokenizer.generation_domain().unwrap();
    let decision = TokenSamplingDecision::new(TokenFilter::allowed(vec![false, true]).unwrap())
        .with_original_tokenizer_validity(domain, OriginalSourceWitness::new(&tokenizer))
        .with_controller_storage(controller.inference_storage());
    storage.validate_sampling_decision(&contract, &decision, &pool).unwrap();
    let equal_foreign_domain = domain.clone();
    let foreign_decision = TokenSamplingDecision::new(TokenFilter::allowed(vec![false, true]).unwrap())
        .with_original_tokenizer_validity(&equal_foreign_domain, OriginalSourceWitness::new(&tokenizer))
        .with_controller_storage(controller.inference_storage());
    assert!(storage.validate_sampling_decision(&contract, &foreign_decision, &pool).is_err());
    let other = PreparedSemanticSource::new(&tokenizer, &execution, capacity).unwrap();
    let original_history = std::mem::replace(&mut controller.history, history(&other));
    assert!(matches!(binding.storage(&controller), TextControllerStorage::Unknown));
    assert!(storage.validate(&controller).is_err());
    controller.history = original_history;
    let foreign_tokenizer = self::tokenizer(&pool);
    controller.source = pool.compile_forbidden_tokenizer_source(&foreign_tokenizer, b"hi").unwrap();
    assert!(PreparedControllerBinding::new(&preparation, &controller).is_err());
    assert!(matches!(binding.storage(&controller), TextControllerStorage::Unknown));
}
