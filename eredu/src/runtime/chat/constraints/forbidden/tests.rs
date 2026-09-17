use super::super::vocabulary::VocabularyPlan;
use super::*;
use eredu_core::{TokenFilter, TokenFilterController};
use eredu_nn::workspace::WorkspaceMetadataAccount;
use llguidance::toktrie::{ApproximateTokEnv, TokRxInfo, TokTrie};
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};

#[derive(Debug)]
struct Account {
    used: Arc<AtomicUsize>,
    limit: Arc<AtomicUsize>,
    retired: Arc<AtomicBool>,
}
impl WorkspaceMetadataAccount for Account {
    fn reserve_metadata(&self, bytes: usize) -> Result<(), WorkspaceMetadataFundingError> {
        let used = self.used.load(Ordering::SeqCst);
        let next = used
            .checked_add(bytes)
            .ok_or(WorkspaceMetadataFundingError::Overflow)?;
        if next > self.limit.load(Ordering::SeqCst) {
            return Err(WorkspaceMetadataFundingError::Capacity {
                required: bytes as u64,
                available: self.limit.load(Ordering::SeqCst).saturating_sub(used) as u64,
            });
        }
        self.used.store(next, Ordering::SeqCst);
        Ok(())
    }
}
impl Drop for Account {
    fn drop(&mut self) {
        self.retired.store(true, Ordering::SeqCst);
    }
}

fn funding() -> (
    WorkspaceMetadataFunding,
    Arc<AtomicUsize>,
    Arc<AtomicUsize>,
    Arc<AtomicBool>,
) {
    let used = Arc::new(AtomicUsize::new(0));
    let limit = Arc::new(AtomicUsize::new(usize::MAX));
    let retired = Arc::new(AtomicBool::new(false));
    let funding = WorkspaceMetadataFunding::new(Account {
        used: used.clone(),
        limit: limit.clone(),
        retired: retired.clone(),
    })
    .unwrap();
    (funding, used, limit, retired)
}

fn controller() -> ConstraintController {
    let words: Vec<Vec<u8>> = [
        b"<".as_slice(),
        b"ca",
        b"ll>",
        b"safe",
        b"<call>",
        "é".as_bytes(),
        b"\0",
        b"",
        b"llx>",
    ]
    .into_iter()
    .map(<[u8]>::to_vec)
    .collect();
    let trie = TokTrie::from(&TokRxInfo::new(words.len() as u32, 0), &words);
    let vocabulary = VocabularyPlan::new(Arc::new(ApproximateTokEnv::new(trie)))
        .unwrap()
        .unregistered();
    ConstraintController {
        runtime: ConstraintRuntime::Forbidden {
            vocabulary,
            trigger: b"<call>".to_vec(),
            pending: TriggerPrefix::default(),
        },
        committed_tokens: PlainControllerHistory::default(),
        validity: SharedTokenFilter::new(
            TokenFilter::allowed(vec![true, true, true, true, true, true, false, true, true])
                .unwrap(),
        ),
        authority: HostPreparationAuthority::unmanaged(),
    }
}

#[test]
fn forbidden_source_preserves_ordinary_filters_history_and_escaped_custody() {
    let mut controller = controller();
    controller.commit_token(0).unwrap();
    controller.commit_token(1).unwrap();
    let (funding, used, _, retired) = funding();
    let before = used.load(Ordering::SeqCst);
    let source = controller.prepare_forbidden_source(&funding).unwrap();
    assert!(used.load(Ordering::SeqCst) > before);
    assert_eq!(source.history(), &[0, 1]);
    assert_eq!(source.pending(), b"<ca");
    assert_eq!(source.token_bytes(5), Some("é".as_bytes()));
    assert_eq!(source.token_bytes(6), Some(b"\0".as_slice()));
    assert_eq!(source.token_bytes(7), Some(b"".as_slice()));
    assert!(source.validity().same_storage(&controller.validity));
    let ordinary = controller.current_filter().unwrap();
    for token in 0..source.vocabulary_len() as u32 + 1 {
        assert_eq!(source.allows(token), ordinary.allows(token));
    }
    assert!(!source.allows(2));
    assert!(controller.commit_token(2).is_err());
    assert_eq!(&*controller.committed_tokens, &[0, 1]);
    assert_eq!(controller.current_filter().unwrap(), ordinary);
    controller.commit_token(3).unwrap();
    assert!(controller.current_filter().unwrap().allows(2));
    assert!(
        !source.allows(2),
        "source preserves the exact earlier state"
    );
    drop((controller, funding));
    assert!(!retired.load(Ordering::SeqCst));
    assert_eq!(source.history(), &[0, 1]);
    drop(source);
    assert!(retired.load(Ordering::SeqCst));
}

#[test]
fn forbidden_source_refusal_retains_cause_and_rejects_plain_state() {
    let controller = controller();
    let (funding, used, limit, retired) = funding();
    let before = used.load(Ordering::SeqCst);
    limit.store(before + controls().unwrap(), Ordering::SeqCst);
    let failure = controller.prepare_forbidden_source(&funding).unwrap_err();
    assert!(matches!(
        failure.cause,
        Cause::Funding(WorkspaceMetadataFundingError::Capacity { .. })
    ));
    assert_eq!(used.load(Ordering::SeqCst), before + controls().unwrap());
    drop((controller, funding));
    assert!(!retired.load(Ordering::SeqCst));
    assert!(std::error::Error::source(&failure).is_some());
    drop(failure);
    assert!(retired.load(Ordering::SeqCst));

    let (funding, _, _, _) = self::funding();
    let plain = ConstraintController::text(SharedTokenFilter::new(TokenFilter::All));
    assert!(matches!(
        plain.prepare_forbidden_source(&funding).unwrap_err().cause,
        Cause::Source
    ));
}

#[test]
fn forbidden_decision_keeps_error_order_atomic_prefix_and_exact_retained_identity() {
    use eredu_core::speculative::ForbiddenControllerMutation;
    let mut controller = controller();
    controller.commit_token(0).unwrap();
    controller.commit_token(1).unwrap();
    let (funding, _, _, retired) = funding();
    let source = controller.prepare_forbidden_source(&funding).unwrap();
    let borrowed = source.source().unwrap();
    assert!(
        matches!(
            borrowed.decision_at(&[0, 1, 2, 6]),
            Err(ForbiddenControllerError::Forbidden(2))
        ),
        "the first forbidden transition precedes a later invalid tokenizer ID"
    );
    assert!(matches!(
        borrowed.decision_at(&[1]),
        Err(ForbiddenControllerError::History(
            PlainControllerError::History
        ))
    ));
    let provisional = borrowed.decision_at(&[0, 1, 8]).unwrap();
    assert!(provisional.allows(2));
    assert!(!borrowed.decision_at(&[0, 1]).unwrap().allows(2));
    assert_eq!(source.pending(), b"<ca");

    funding
        .reserve_metadata(
            PlainControllerHistory::copy_metadata_bytes(2).unwrap()
                + PlainControllerHistory::copy_metadata_bytes(4).unwrap()
                + ForbiddenControllerInputs::operation_control_bytes().unwrap(),
        )
        .unwrap();
    let host = HostPreparationAuthority::retain(funding.clone());
    let mut full = source.history.copy_prepared(2, host.clone()).unwrap();
    let mut full_prefix = source.prefix;
    let failure = ForbiddenControllerMutation::new(
        &mut full,
        &source.validity,
        &source.inputs,
        &mut full_prefix,
    )
    .unwrap()
    .commit(3)
    .unwrap_err();
    assert!(matches!(
        failure,
        ForbiddenControllerError::History(PlainControllerError::Destination)
    ));
    assert_eq!(&*full, &[0, 1]);
    assert_eq!(full_prefix, source.prefix);

    let mut copied = source.history.copy_prepared(4, host.clone()).unwrap();
    let mut prefix = source.prefix;
    let copied_source =
        ForbiddenControllerSource::new(&copied, &source.validity, &source.inputs, prefix).unwrap();
    assert!(borrowed.matches_copy(copied_source, 4));
    let identity = copied_source.retain_prepared_identity().unwrap();
    assert!(copied_source.matches_prepared_identity(&identity));
    let failure = ForbiddenControllerMutation::new(
        &mut copied,
        &source.validity,
        &source.inputs,
        &mut prefix,
    )
    .unwrap()
    .commit(3)
    .unwrap_err();
    assert!(matches!(
        failure,
        ForbiddenControllerError::History(PlainControllerError::Destination)
    ));
    assert_eq!(prefix, source.prefix);
    drop(identity);
    ForbiddenControllerMutation::new(&mut copied, &source.validity, &source.inputs, &mut prefix)
        .unwrap()
        .commit(3)
        .unwrap();
    assert_eq!(&*copied, &[0, 1, 3]);
    assert_eq!(prefix.bytes(source.trigger()), b"");
    let escaped = ForbiddenControllerSource::new(&copied, &source.validity, &source.inputs, prefix)
        .unwrap()
        .retain_prepared_identity()
        .unwrap();
    funding
        .reserve_metadata(
            ForbiddenControllerInputs::copy_metadata_bytes(
                source.inputs.vocabulary().len(),
                source.trigger().len(),
            )
            .unwrap(),
        )
        .unwrap();
    let foreign = ForbiddenControllerInputs::copy_prepared(
        source.inputs.vocabulary(),
        source.inputs.vocabulary_len(),
        6,
        source.trigger(),
        host.clone(),
    )
    .unwrap();
    let different =
        ForbiddenControllerSource::new(&copied, &source.validity, &foreign, prefix).unwrap();
    assert!(
        !different.matches_prepared_identity(&escaped),
        "equal bytes are not source identity"
    );
    drop((foreign, controller, source, full, copied, host, funding));
    assert!(!retired.load(Ordering::SeqCst));
    drop(escaped);
    assert!(retired.load(Ordering::SeqCst));
}

#[test]
fn forbidden_controller_copy_uses_exact_inputs_and_independent_atomic_history() {
    use eredu_core::SpeculativeTokenFilterController;
    let mut ordinary = controller();
    ordinary.commit_token(0).unwrap();
    ordinary.commit_token(1).unwrap();
    let (funding, _, _, retired) = funding();
    let prepared = ordinary.prepare_forbidden_controller(4, &funding).unwrap();
    assert!(prepared.prepared_plain_source().is_none());
    assert!(prepared.prepared_plain_copy_bytes(4).is_none());
    let source = prepared.prepared_forbidden_source().unwrap();
    assert_eq!(source.history(), &[0, 1]);
    let ordinary_mask = ordinary.current_filter().unwrap();
    for token in 0..10 {
        assert_eq!(
            source.decision_at(&[0, 1]).unwrap().allows(token),
            ordinary_mask.allows(token)
        );
    }
    let bytes = prepared.prepared_forbidden_copy_bytes(3).unwrap()
        + HostPreparationAuthority::retention_bytes::<WorkspaceMetadataFunding>().unwrap();
    funding.reserve_metadata(bytes).unwrap();
    let host = HostPreparationAuthority::retain(funding.clone());
    let mut copied = prepared.copy_prepared_forbidden(3, host.clone()).unwrap();
    assert!(source.matches_copy(copied.prepared_forbidden_source().unwrap(), 3));
    assert!(matches!(
        copied.prepared_forbidden_mutation().unwrap().commit(2),
        Err(ForbiddenControllerError::Forbidden(2))
    ));
    copied
        .prepared_forbidden_mutation()
        .unwrap()
        .commit(3)
        .unwrap();
    ordinary.commit_token(3).unwrap();
    assert_eq!(
        copied.current_filter().unwrap(),
        ordinary.current_filter().unwrap()
    );
    assert_eq!(
        copied.prepared_forbidden_source().unwrap().history(),
        &[0, 1, 3]
    );
    assert_eq!(source.history(), &[0, 1]);
    let prefix = copied.prepared_forbidden_source().unwrap().prefix();
    assert!(matches!(
        copied.prepared_forbidden_mutation().unwrap().commit(0),
        Err(ForbiddenControllerError::History(
            PlainControllerError::Destination
        ))
    ));
    assert_eq!(copied.prepared_forbidden_source().unwrap().prefix(), prefix);
    // Ordinary forcing and the fixed prepared decision share the same domain.
    let mut choice = eredu_runtime::execution_control::TokenChoiceController::new(
        copied,
        eredu_runtime::TokenDomain::new(9),
    );
    choice
        .force_at(2, eredu_runtime::TokenDomain::new(9), 3)
        .unwrap();
    let decision = choice.prepared_forbidden_decision(&[0, 1, 3]).unwrap();
    assert_eq!(decision.forced_token(), Some(2));
    assert!(decision.allows(2));
    assert!(!decision.allows(3));
    assert!(decision.before_forcing().allows(3));
    let shape = [2, 11];
    let plan = eredu_runtime::generation::TokenMaskPlan::forbidden(
        decision.before_forcing(),
        &shape,
        decision.forced_token(),
    )
    .unwrap();
    let mut invalid = Vec::with_capacity(plan.elements());
    plan.fill(&mut invalid).unwrap();
    let expected: Vec<_> = (0..2)
        .flat_map(|_| (0..11).map(|token| !decision.allows(token)))
        .collect();
    assert_eq!(invalid, expected);
    let observed = decision.capture_domain().summary(11);
    assert_eq!(
        observed.allowed_tokens,
        (0..11)
            .filter(|&token| decision.before_forcing().allows(token))
            .count() as u64
    );
    assert!(observed.constrained);
    assert!(
        observed.allowed_tokens > 1,
        "capture excludes the forced-token override"
    );
    let escaped = choice
        .inner()
        .prepared_forbidden_source()
        .unwrap()
        .retain_prepared_identity()
        .unwrap();
    drop((ordinary, prepared, choice, host, funding));
    assert!(!retired.load(Ordering::SeqCst));
    drop(escaped);
    assert!(retired.load(Ordering::SeqCst));
}

#[test]
fn forbidden_shared_sampler_preserves_forcing_provisional_commit_and_choice_retirement() {
    use eredu_core::SpeculativeTokenFilterController;
    use eredu_runtime::{
        execution_control::{PreparedControllerCause, PreparedControllerSource, TokenChoiceError},
        generation::{
            ConstrainedSampler, DefaultSampler, PreparedSpeculativeController, SpeculativeSampler,
        },
        working_memory::WorkspaceSamplingBackend,
    };
    type Sampler = ConstrainedSampler<DefaultSampler, ConstraintController>;
    fn plan(source: &Sampler) -> PreparedSpeculativeController<'_, Sampler> {
        SpeculativeSampler::<WorkspaceSamplingBackend>::prepared_controller(source).unwrap()
    }
    let mut ordinary = controller();
    ordinary.commit_token(0).unwrap();
    ordinary.commit_token(1).unwrap();
    let (funding, _, _, retired) = funding();
    let pool = eredu_runtime::working_memory::WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let foreign_pool = eredu_runtime::working_memory::WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let source = Sampler::new(
        DefaultSampler,
        ordinary
            .prepare_original_forbidden_controller_with(2, &funding, |plan| {
                pool.compile_forbidden_source(plan)
            })
            .unwrap(),
    );
    let original = source.controller().original_forbidden_source().unwrap();
    let actual = source.controller().prepared_forbidden_source().unwrap();
    original.validate_controller(actual, &pool).unwrap();
    assert!(original.validate_controller(actual, &foreign_pool).is_err());
    assert!(
        actual
            .original_storage()
            .unwrap()
            .downcast_ref::<eredu_runtime::working_memory::OriginalForbiddenSource>()
            .is_some()
    );
    assert!(matches!(
        plan(&source).controller_source().unwrap(),
        PreparedControllerSource::Forbidden(_)
    ));
    assert!(matches!(
        plan(&source).validate_commit(2),
        Err(TokenChoiceError::Constraint(
            PreparedControllerCause::Forbidden(ForbiddenControllerError::Forbidden(2))
        ))
    ));
    let (decision, _) = plan(&source).logits(&[0, 1]).unwrap().into_parts();
    let shape = [1, 9];
    let mask = decision.mask_plan(&shape).unwrap();
    let mut invalid = Vec::with_capacity(mask.elements());
    mask.fill(&mut invalid).unwrap();
    let actual = ordinary.current_filter().unwrap();
    assert_eq!(
        invalid,
        (0..9)
            .map(|token| !actual.allows(token))
            .collect::<Vec<_>>()
    );

    funding
        .reserve_metadata(
            plan(&source).copy_metadata_bytes()
                + HostPreparationAuthority::retention_bytes::<WorkspaceMetadataFunding>().unwrap(),
        )
        .unwrap();
    let host = HostPreparationAuthority::retain(funding.clone());
    let forced = plan(&source)
        .force(3, eredu_runtime::TokenDomain::new(9), 2, host.clone())
        .unwrap();
    let (decision, _) = plan(&forced).logits(&[0, 1]).unwrap().into_parts();
    assert_eq!(decision.forced_token(), Some(3));
    assert!(decision.capture_domain().filter.allows(8));
    let forced_mask = decision.mask_plan(&shape).unwrap();
    let mut invalid = Vec::with_capacity(forced_mask.elements());
    forced_mask.fill(&mut invalid).unwrap();
    assert_eq!(invalid, (0..9).map(|token| token != 3).collect::<Vec<_>>());
    let escaped = plan(&forced).choice(0.0).unwrap();
    funding
        .reserve_metadata(plan(&forced).commit_metadata_bytes())
        .unwrap();
    let committed = plan(&forced).commit(3, host.clone()).unwrap();
    assert_eq!(
        plan(&source).controller_source().unwrap().history(),
        &[0, 1]
    );
    assert_eq!(
        plan(&committed).controller_source().unwrap().history(),
        &[0, 1, 3]
    );
    assert!(!plan(&committed).matches_choice(&escaped));
    assert!(plan(&forced).matches_choice(&escaped));
    drop((ordinary, source, forced, committed, host, funding));
    assert!(!retired.load(Ordering::SeqCst));
    assert!(pool.used_bytes().unwrap() > 0);
    drop(escaped);
    assert!(retired.load(Ordering::SeqCst));
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
