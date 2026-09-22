use super::*;
use crate::memory_fixture::{LedgerFixture as _, StorageFixture as _};
use eredu_core::{SpeculativeTokenFilterController, TokenFilter, TokenFilterController};
use eredu_nn::workspace::HostMetadataAccount;
use eredu_runtime::working_memory::MemoryLedger;
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Arc,
};

#[derive(Debug)]
struct Account {
    used: Arc<AtomicUsize>,
    limit: Arc<AtomicUsize>,
    retired: Arc<AtomicBool>,
}
impl HostMetadataAccount for Account {
    fn reserve_metadata(&self, bytes: usize) -> Result<(), HostMetadataFundingError> {
        let used = self.used.load(Ordering::SeqCst);
        let next = used
            .checked_add(bytes)
            .ok_or(HostMetadataFundingError::Overflow)?;
        if next > self.limit.load(Ordering::SeqCst) {
            return Err(HostMetadataFundingError::Capacity {
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
    HostMetadataFunding,
    Arc<AtomicUsize>,
    Arc<AtomicUsize>,
    Arc<AtomicBool>,
) {
    let used = Arc::new(AtomicUsize::new(0));
    let limit = Arc::new(AtomicUsize::new(usize::MAX));
    let retired = Arc::new(AtomicBool::new(false));
    let funding = HostMetadataFunding::new(Account {
        used: used.clone(),
        limit: limit.clone(),
        retired: retired.clone(),
    })
    .unwrap();
    (funding, used, limit, retired)
}

/// Explicit byte fixture compiled by the actual immutable source worker. Its
/// empty, binary and non-ASCII tokens are intentional controller edge cases.
struct Fixture {
    plan: super::super::fixtures::Plan,
    packed: Vec<u8>,
    pool: MemoryLedger,
}
impl Fixture {
    fn new(pool: MemoryLedger) -> Self {
        use super::super::{fixtures, frozen_tests, ParallelToolCallPolicy, ToolChoice};
        let compiler = fixtures::Compiler::byte_tokens(&[255, 254]);
        let plan = compiler
            .compile_tool_plan(
                &crate::runtime::chat::dialect::DECLARATIVE_DIALECT,
                frozen_tests::PARAMETERS,
                &frozen_tests::ordinary_tools(),
                ToolChoice::None,
                ParallelToolCallPolicy::Disabled,
                Vec::new(),
            )
            .unwrap();
        let words: [&[u8]; 9] = [
            b"{",
            b"\"ca",
            b"lls\":",
            b"safe",
            br#"{"calls":"#,
            "é".as_bytes(),
            b"\0",
            b"",
            b"llx\":",
        ];
        let mut packed = Vec::new();
        let mut offset = (words.len() + 1) * 8;
        for bytes in words {
            packed.extend_from_slice(&(offset as u64).to_le_bytes());
            offset += bytes.len();
        }
        packed.extend_from_slice(&(offset as u64).to_le_bytes());
        for bytes in words {
            packed.extend_from_slice(bytes);
        }
        Self { plan, packed, pool }
    }
    fn controller(
        &self,
        capacity: usize,
        funding: &HostMetadataFunding,
    ) -> Result<ConstraintController, ForbiddenSourceError> {
        ConstraintController::from_original_forbidden_generation_plan_with(
            &self.plan,
            SharedTokenFilter::new(
                TokenFilter::allowed(vec![true, true, true, true, true, true, false, true, true])
                    .unwrap(),
            ),
            capacity,
            funding,
            |trigger| {
                Ok(self.pool.compile_forbidden_source(
                    eredu_core::speculative::PreparedForbiddenInputCopy::new(
                        &self.packed,
                        9,
                        9,
                        trigger,
                    )?,
                )?)
            },
        )
    }
}

#[test]
fn forbidden_original_source_preserves_filter_bytes_atomic_history_and_escaped_custody() {
    let pool = crate::memory_fixture::host_ledger(1 << 26, 0).unwrap();
    let fixture = Fixture::new(pool.clone());
    let (funding, _, _, retired) = funding();
    let mut controller = fixture.controller(4, &funding).unwrap();
    controller.commit_token(0).unwrap();
    controller.commit_token(1).unwrap();
    let source = controller.prepared_forbidden_source().unwrap();
    assert_eq!(source.history(), &[0, 1]);
    assert_eq!(source.prefix().bytes(source.inputs().trigger()), b"{\"ca");
    assert_eq!(source.inputs().token_bytes(5), Some("é".as_bytes()));
    assert_eq!(source.inputs().token_bytes(6), Some(b"\0".as_slice()));
    assert_eq!(source.inputs().token_bytes(7), Some(b"".as_slice()));
    let mask = controller.current_filter().unwrap();
    for token in 0..11 {
        assert_eq!(mask.allows(token), token < 9 && ![2, 4, 6].contains(&token));
    }
    assert!(controller.commit_token(2).is_err());
    assert_eq!(&*controller.committed_tokens, &[0, 1]);
    assert_eq!(controller.current_filter().unwrap(), mask);
    let source = controller.prepared_forbidden_source().unwrap();
    assert!(matches!(
        source.decision_at(&[0, 1, 2, 6]),
        Err(ForbiddenControllerError::Forbidden(2))
    ));
    assert!(matches!(
        source.decision_at(&[1]),
        Err(ForbiddenControllerError::History(
            PlainControllerError::History
        ))
    ));
    assert!(source.decision_at(&[0, 1, 8]).unwrap().allows(2));
    let escaped = source.retain_prepared_identity().unwrap();
    // An escaped history alias prevents mutation rather than overwriting it.
    assert!(matches!(
        controller.prepared_forbidden_mutation().unwrap().commit(3),
        Err(ForbiddenControllerError::History(
            PlainControllerError::Destination
        ))
    ));
    drop((controller, funding));
    assert!(!retired.load(Ordering::SeqCst));
    assert!(pool.live_charge_bytes().unwrap() > 0);
    drop(escaped);
    assert!(retired.load(Ordering::SeqCst));
    assert_eq!(pool.live_charge_bytes().unwrap(), 0);
}

#[test]
fn forbidden_startup_refusal_preserves_cause_and_source_budget() {
    let pool = crate::memory_fixture::host_ledger(1 << 26, 0).unwrap();
    let fixture = Fixture::new(pool.clone());
    let (funding, used, limit, retired) = funding();
    let before = used.load(Ordering::SeqCst);
    limit.store(before, Ordering::SeqCst);
    let failure = fixture.controller(4, &funding).err().unwrap();
    assert!(matches!(
        failure.cause,
        Cause::Funding(HostMetadataFundingError::Capacity { .. })
    ));
    assert_eq!(used.load(Ordering::SeqCst), before);
    assert_eq!(pool.live_charge_bytes().unwrap(), 0);
    drop(funding);
    assert!(!retired.load(Ordering::SeqCst));
    assert!(std::error::Error::source(&failure).is_some());
    drop(failure);
    assert!(retired.load(Ordering::SeqCst));
    let plain = ConstraintController::text(SharedTokenFilter::new(TokenFilter::All));
    assert!(plain.prepared_forbidden_source().is_none());
    assert!(plain.prepared_forbidden_copy_bytes(4).is_none());
}

#[test]
fn forbidden_copy_uses_exact_inputs_independent_history_and_shared_forcing() {
    let pool = crate::memory_fixture::host_ledger(1 << 26, 0).unwrap();
    let fixture = Fixture::new(pool.clone());
    let (funding, _, _, retired) = funding();
    let mut prepared = fixture.controller(4, &funding).unwrap();
    prepared.commit_token(0).unwrap();
    prepared.commit_token(1).unwrap();
    assert!(prepared.prepared_plain_source().is_none());
    assert!(prepared.prepared_forbidden_copy_bytes(1).is_none());
    funding
        .reserve_metadata(
            prepared.prepared_forbidden_copy_bytes(3).unwrap()
                + HostPreparationAuthority::retention_bytes::<HostMetadataFunding>().unwrap(),
        )
        .unwrap();
    let host = HostPreparationAuthority::retain(funding.clone());
    let mut copied = prepared.copy_prepared_forbidden(3, host.clone()).unwrap();
    let source = prepared.prepared_forbidden_source().unwrap();
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
    assert_eq!(source.history(), &[0, 1]);
    assert_eq!(
        copied.prepared_forbidden_source().unwrap().history(),
        &[0, 1, 3]
    );
    let prefix = copied.prepared_forbidden_source().unwrap().prefix();
    assert!(matches!(
        copied.prepared_forbidden_mutation().unwrap().commit(0),
        Err(ForbiddenControllerError::History(
            PlainControllerError::Destination
        ))
    ));
    assert_eq!(copied.prepared_forbidden_source().unwrap().prefix(), prefix);
    // Equal source bytes from a separate original compiler do not authenticate.
    let foreign = fixture.controller(4, &funding).unwrap();
    let identity = source.retain_prepared_identity().unwrap();
    assert!(!foreign
        .prepared_forbidden_source()
        .unwrap()
        .matches_prepared_identity(&identity));
    drop(identity);
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
    assert!(observed.constrained && observed.allowed_tokens > 1);
    let escaped = choice
        .inner()
        .prepared_forbidden_source()
        .unwrap()
        .retain_prepared_identity()
        .unwrap();
    drop((prepared, choice, foreign, host, funding));
    assert!(!retired.load(Ordering::SeqCst));
    drop(escaped);
    assert!(retired.load(Ordering::SeqCst));
    assert_eq!(pool.live_charge_bytes().unwrap(), 0);
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
    let (funding, _, _, retired) = funding();
    let pool = crate::memory_fixture::host_ledger(u64::MAX, 0).unwrap();
    let foreign_pool = crate::memory_fixture::host_ledger(u64::MAX, 0).unwrap();
    let fixture = Fixture::new(pool.clone());
    let mut controller = fixture.controller(2, &funding).unwrap();
    controller.commit_token(0).unwrap();
    controller.commit_token(1).unwrap();
    let source = Sampler::new(DefaultSampler, controller);
    let original = source.controller().original_forbidden_source().unwrap();
    let actual = source.controller().prepared_forbidden_source().unwrap();
    original.validate_controller(actual, &pool).unwrap();
    assert!(original.validate_controller(actual, &foreign_pool).is_err());
    assert!(actual
        .original_storage()
        .unwrap()
        .downcast_ref::<eredu_runtime::working_memory::OriginalForbiddenSource>()
        .is_some());
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
    let actual = source.controller().filter_at(&[0, 1]).unwrap();
    assert_eq!(
        invalid,
        (0..9)
            .map(|token| !actual.allows(token))
            .collect::<Vec<_>>()
    );

    funding
        .reserve_metadata(
            plan(&source).copy_metadata_bytes()
                + HostPreparationAuthority::retention_bytes::<HostMetadataFunding>().unwrap(),
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
    drop((source, forced, committed, host, funding));
    assert!(!retired.load(Ordering::SeqCst));
    assert!(pool.live_charge_bytes().unwrap() > 0);
    drop(escaped);
    assert!(retired.load(Ordering::SeqCst));
    assert_eq!(pool.live_charge_bytes().unwrap(), 0);
}
