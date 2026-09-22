use super::*;
use crate::working_memory::{
    InferenceRequest, InferenceTextPreparation, RegisteredWorkspaceStorage, RunOwnedTextSampler,
    SamplerCopyLimits, WorkingMemoryFundingRun, WorkingMemoryReservation,
};
use crate::{ConfiguredTextSampler, PenaltyConfig, SamplingBackend, TokenDomain};
use eredu_core::{
    Admission, EstimationCompleteness, ExecutionWorkspaceEstimate, InferenceGeometry,
    InputTokenCount, LayerSchedule, OutputDemand, ResolvedGenerationConfig, StateMemoryLayout,
    TextGenerationConfig, TokenFilter, WorkspaceBound, cache::LayerCachePolicy,
};
use eredu_nn::workspace::{
    WorkspaceContext, WorkspaceDtype, WorkspaceExistingStorage, WorkspaceHostBound,
    WorkspaceIsolatedCopyPlan, WorkspaceLayout, WorkspaceMechanisms, WorkspaceOperation,
    WorkspaceOperationBound, WorkspaceOperationKind, WorkspaceOutputStorage, WorkspaceTensor,
};
use std::cell::Cell;

thread_local! {
    static COPY_ATTEMPTS: Cell<usize> = const { Cell::new(0) };
    static FAIL_COPY: Cell<bool> = const { Cell::new(false) };
}

pub(super) fn before_copy() {
    COPY_ATTEMPTS.with(|count| count.set(count.get() + 1));
    assert!(
        !FAIL_COPY.with(|flag| flag.replace(false)),
        "injected copy failure"
    );
}

fn attempts() -> usize {
    COPY_ATTEMPTS.with(Cell::get)
}

const HOST_SOURCE_BYTES: u64 = 1024;

fn config(outputs: usize, adaptive: bool) -> TextGenerationConfig {
    let config = TextGenerationConfig::new(ResolvedGenerationConfig {
        do_sample: true,
        temperature: 0.7,
        top_k: 17,
        top_p: 0.83,
        min_p: 0.07,
        repetition_penalty: 1.13,
        repeat_last_n: 23,
        frequency_penalty: 0.17,
        presence_penalty: 0.29,
        max_new_tokens: Some(outputs),
    });
    if adaptive {
        config.with_mirostat_v2(3.7, 0.23).unwrap()
    } else {
        config
    }
}

// A scalar portable sampling backend has no native arrays. Its real boxed
// history is constructed by the one-time stage and grown by the shared sampler.
// The generous source envelope covers this fixture's complete host work; it is
// not a synthetic assertion about native model workspace or snapshot support.
fn prepared_request(
    pool: &MemoryLedger,
    capacity: u64,
    outputs: usize,
    adaptive: bool,
) -> (
    InferenceTextPreparation,
    WorkingMemoryFundingRun,
    TextGenerationConfig,
) {
    prepared_request_with_bytes(pool, capacity, outputs, adaptive, HOST_SOURCE_BYTES)
}

fn prepared_request_with_bytes(
    pool: &MemoryLedger,
    capacity: u64,
    outputs: usize,
    adaptive: bool,
    source_bytes: u64,
) -> (
    InferenceTextPreparation,
    WorkingMemoryFundingRun,
    TextGenerationConfig,
) {
    let execution = InferenceExecutionIdentity::default();
    let geometry = InferenceGeometry {
        batch_size: 1,
        cached_positions: 0,
        input_positions: 1,
        max_output_tokens: outputs as u64,
        prefill_chunk_positions: 1,
        output: OutputDemand::LastPosition,
    };
    let layout = StateMemoryLayout::new(
        LayerSchedule::new(1, vec![LayerCachePolicy::NoState]).unwrap(),
        vec![0],
        1,
        1,
        EstimationCompleteness::Complete,
    )
    .unwrap();
    let state = eredu_core::estimate_runtime_state(
        &layout,
        InputTokenCount::text(1),
        outputs as u64,
        1,
        std::num::NonZeroU8::new(4).unwrap(),
    )
    .unwrap();
    let bound = |bytes| WorkspaceBound::bounded(bytes, "portable scalar sampler host envelope");
    let mut state = state
        .with_execution_workspace(ExecutionWorkspaceEstimate {
            physical_domains: Some(crate::working_memory::memory_fixture::host_workspace(
                pool,
                geometry,
                source_bytes,
            )),
            geometry,
            activations: bound(source_bytes),
            attention: bound(0),
            vocabulary: bound(0),
            state_update: bound(0),
            materialization: bound(0),
            retained: bound(0),
        })
        .unwrap();
    state.physical_domains = Some(crate::working_memory::memory_fixture::empty_state(
        pool, geometry,
    ));
    let reservation: WorkingMemoryReservation = pool
        .reserve_with_capacity(
            &execution,
            &Admission {
                memory_limits: crate::working_memory::memory_fixture::host_limits(capacity),
                additional_headroom: eredu_core::MemoryHeadroomDeclarations::none(),
                requested_positions: 1 + outputs as u64,
                state,
                incremental_required_bytes: Some(source_bytes),
            },
            crate::working_memory::memory_fixture::host_limits(capacity)
                .resolve(pool.topology())
                .unwrap(),
        )
        .unwrap();
    let (reservation, run) = reservation.into_funding().unwrap();
    let request = InferenceRequest::from(reservation);
    let config = config(outputs, adaptive);
    let preparation = request
        .prepare_text(&execution, geometry, config.clone())
        .unwrap();
    (preparation, run, config)
}

fn source(
    pool: &MemoryLedger,
    capacity: u64,
    outputs: usize,
    adaptive: bool,
) -> (
    RunOwnedTextSampler,
    InferenceTextPreparation,
    WorkingMemoryFundingRun,
) {
    let (preparation, run, config) = prepared_request(pool, capacity, outputs, adaptive);
    let (sampler, completion) = preparation
        .claim_sampling(config.clone())
        .unwrap()
        .construct_sampler(run.sampler_scope().unwrap())
        .unwrap();
    completion.finish().unwrap();
    (sampler, preparation, run)
}

#[derive(Default)]
struct Context {
    callbacks: Cell<usize>,
}

struct Scalar;
impl SamplingBackend for Scalar {
    type Logits = u32;
    type Token = u32;
    type RandomState = ();
    type Context = Context;
    type Error = String;

    fn error(message: String) -> String {
        message
    }
    fn validate_token(token: &u32, _: TokenDomain, _: &Context) -> Result<u32, String> {
        Ok(*token)
    }
    fn scale_temperature(logits: &u32, _: f32, _: &Context) -> Result<u32, String> {
        Ok(*logits)
    }
    fn apply_penalties(
        logits: &u32,
        _: &[u32],
        _: PenaltyConfig,
        context: &Context,
    ) -> Result<u32, String> {
        context.callbacks.set(context.callbacks.get() + 1);
        Ok(*logits)
    }
    fn apply_top_k(logits: u32, _: i32, _: &Context) -> Result<u32, String> {
        Ok(logits)
    }
    fn apply_top_p(logits: u32, _: f32, _: &Context) -> Result<u32, String> {
        Ok(logits)
    }
    fn apply_min_p(logits: u32, _: f32, _: &Context) -> Result<u32, String> {
        Ok(logits)
    }
    fn apply_token_filter(logits: &u32, _: &TokenFilter, _: &Context) -> Result<u32, String> {
        Ok(*logits)
    }
    fn apply_mirostat(
        logits: &u32,
        _: &[u32],
        _: PenaltyConfig,
        _: f32,
        _: f32,
        context: &Context,
    ) -> Result<u32, String> {
        context.callbacks.set(context.callbacks.get() + 1);
        Ok(*logits)
    }
    fn sample_raw(logits: &u32, _: f32, _: Option<&mut ()>, _: &Context) -> Result<u32, String> {
        Ok(*logits)
    }
    fn sample_processed(
        logits: &u32,
        _: f32,
        _: Option<&mut ()>,
        _: &Context,
    ) -> Result<u32, String> {
        Ok(*logits)
    }
    fn token_id(token: &u32, _: &Context) -> Result<u32, String> {
        Ok(*token)
    }
    fn token_probability(_: &u32, _: u32, _: &Context) -> Result<f32, String> {
        Ok(0.125)
    }
}

fn grow(sampler: &mut RunOwnedTextSampler, tokens: &[u32]) {
    let context = Context::default();
    for token in tokens {
        assert_eq!(
            sampler
                .prepare_sample()
                .unwrap()
                .sample::<Scalar>(token, 0.7, None, &context)
                .unwrap(),
            *token
        );
    }
    assert_eq!(context.callbacks.get(), tokens.len());
}

fn history(sampler: &ConfiguredTextSampler) -> &[u32] {
    match sampler {
        ConfiguredTextSampler::Standard(sampler) => sampler.generated_tokens(),
        ConfiguredTextSampler::MirostatV2(sampler) => sampler.generated_tokens(),
    }
}

#[derive(Debug, Default)]
struct Facts {
    missing_tensor: bool,
    missing_host: bool,
    physical: Option<(
        std::sync::Arc<eredu_core::MemoryTopology>,
        eredu_core::MemoryPlacement,
    )>,
}

impl Facts {
    pub(super) fn with_pool(mut self, pool: &MemoryLedger) -> Self {
        self.physical = Some((
            pool.topology_handle(),
            (*pool.host_placement_handle()).clone(),
        ));
        self
    }
}

impl WorkspaceMechanisms for Facts {
    fn memory_topology(&self) -> Option<&eredu_core::MemoryTopology> {
        self.physical.as_ref().map(|value| value.0.as_ref())
    }
    fn output_placement(
        &self,
        _: eredu_nn::workspace::WorkspaceOperationView<'_>,
        _: usize,
    ) -> Option<&eredu_core::MemoryPlacement> {
        self.physical.as_ref().map(|value| &value.1)
    }
    fn scratch_placement(
        &self,
        _: eredu_nn::workspace::WorkspaceOperationView<'_>,
    ) -> Option<&eredu_core::MemoryPlacement> {
        self.physical.as_ref().map(|value| &value.1)
    }

    fn operation_bound(
        &self,
        operation: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceOperationBound>, eredu_nn::Error> {
        if self.missing_tensor {
            return Ok(None);
        }
        let [layout] = operation.outputs.as_slice() else {
            panic!("closed copy has one output");
        };
        let capacity = layout.bytes()?.div_ceil(16) * 16;
        let effect = match operation.kind {
            WorkspaceOperationKind::Contiguous => WorkspaceOutputStorage::AllocateOrAliasInputs {
                bytes: capacity,
                inputs: vec![0],
            },
            WorkspaceOperationKind::DeepCopy => WorkspaceOutputStorage::Allocate(capacity),
            _ => panic!("only the closed copy program is allowed"),
        };
        Ok(Some(WorkspaceOperationBound {
            outputs: vec![effect],
            scratch_bytes: 3,
            assumptions: "fixture: sixteen-byte padded result and three scratch bytes".into(),
        }))
    }

    fn host_workspace_bound(
        &self,
        _: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceHostBound>, eredu_nn::Error> {
        Ok((!self.missing_host).then(|| WorkspaceHostBound {
            bytes: 5,
            assumptions: "fixture: five disjoint staging bytes per operation".into(),
        }))
    }
}

const COPY_BYTES: u64 = 48; // two padded outputs, two scratch and two host bounds
const ARRAY_SOURCE_BYTES: u64 = 64;

fn plan_parts(
    pool: &MemoryLedger,
    key: u32,
    source_bytes: u64,
    copies: usize,
    facts: Facts,
) -> (WorkspaceIsolatedCopyPlan, RegisteredWorkspaceStorage<u32>) {
    let context = WorkspaceContext::new(facts.with_pool(pool));
    let root = WorkspaceExistingStorage::try_new_placed(
        Some(source_bytes),
        &pool.host_placement_handle(),
        &context,
    )
    .unwrap();
    let source = RegisteredWorkspaceStorage::bind(pool, &context, [(key, root.clone())]).unwrap();
    let sources = (0..copies)
        .map(|_| {
            WorkspaceTensor::existing_with_storage(
                WorkspaceLayout::new(&[2], WorkspaceDtype::Uint32).unwrap(),
                &root,
                &context,
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let plan =
        WorkspaceIsolatedCopyPlan::prepare(&context, source.borrowed_storage(), &sources).unwrap();
    (plan, source)
}

fn copy_plan(
    pool: &MemoryLedger,
    key: u32,
    source_bytes: u64,
    copies: usize,
) -> RegisteredWorkspaceCopy<u32> {
    let (plan, source) = plan_parts(pool, key, source_bytes, copies, Facts::default());
    assert_eq!(plan.incremental_bytes(), Some(COPY_BYTES * copies as u64));
    RegisteredWorkspaceCopy::bind(plan, source).unwrap()
}

fn usage(pool: &MemoryLedger) -> (u64, u64, u64) {
    (
        pool.payload_used_bytes().unwrap(),
        pool.payload_peak_bytes().unwrap(),
        pool.payload_effective_capacity().unwrap(),
    )
}

fn joint<'a>(
    pool: &MemoryLedger,
    source: BorrowedFundedSampler<'a>,
) -> RegisteredSamplingCopy<'a, u32> {
    RegisteredSamplingCopy::prepare(source, copy_plan(pool, 1, ARRAY_SOURCE_BYTES, 1)).unwrap()
}

fn current(pool: &MemoryLedger) -> u64 {
    let snapshot = pool.snapshot().unwrap();
    let host = snapshot
        .domains
        .iter()
        .find(|d| d.domain == pool.topology().host_domain())
        .unwrap();
    host.current_charge_bytes - host.fixed_baseline.total().unwrap()
}
fn full_copy_bytes(pool: &MemoryLedger, copy: &RegisteredSamplingCopy<'_, u32>) -> u64 {
    pool.sampling_copy_requirements(copy, &WorkspaceCopyLimits::default())
        .unwrap()
        .get(pool.topology().host_domain())
        .unwrap()
        .total()
        .unwrap()
}
fn settle(copied: (FundedSamplerCopy, AdmittedWorkspaceCopy)) {
    let (sampler, arrays) = copied;
    let (custody, scope) = arrays.into_parts();
    scope.certify().unwrap();
    drop((sampler, custody));
}

#[test]
fn exact_joint_admission_copies_real_standard_and_adaptive_history_once() {
    for adaptive in [false, true] {
        let pool = crate::working_memory::memory_fixture::host_ledger((1 << 20), 0).unwrap();
        let physical = pool
            .register_host_storage([(1u32, ARRAY_SOURCE_BYTES)])
            .unwrap();
        let (mut sampler, preparation, run) = source(&pool, (1 << 20), 8, adaptive);
        grow(&mut sampler, &[3, 11, 7, 19, 5]);
        let plan = sampler.as_sampler().prepare_copy().unwrap();
        assert_eq!((plan.history_len(), plan.history_capacity()), (5, 8));
        let host = plan.retained_bytes();
        drop(plan);
        let before = usage(&pool);
        let before_attempts = attempts();
        let bytes = host + COPY_BYTES;
        assert_eq!(
            joint(&pool, sampler.borrow_funded())
                .required_bytes()
                .unwrap(),
            bytes
        );
        let rejected = joint(&pool, sampler.borrow_funded());
        let full = full_copy_bytes(&pool, &rejected);
        let capacity = current(&pool) + full;
        let before = usage(&pool);
        assert!(matches!(
            pool.copy_sampling_components(rejected, WorkspaceCopyLimits::new(crate::working_memory::memory_fixture::host_limits(capacity - 1))),
            Err(SamplingCopyAdmissionError::Memory(WorkingMemoryError::Domain(eredu_core::MemoryDomainError::BudgetExceeded { requested_bytes: required_bytes, limit_bytes, existing_bytes, .. }))) if required_bytes == full && limit_bytes - existing_bytes == full - 1
        ));
        assert_eq!(usage(&pool), before);
        assert_eq!(attempts(), before_attempts);
        let accounts = pool.0.usage.lock().unwrap().funding.len();
        let (copied, arrays) = pool
            .copy_sampling_components(
                joint(&pool, sampler.borrow_funded()),
                WorkspaceCopyLimits::new(crate::working_memory::memory_fixture::host_limits(
                    capacity,
                )),
            )
            .unwrap();
        assert_eq!(pool.0.usage.lock().unwrap().funding.len(), accounts + 1);
        assert_eq!(attempts(), before_attempts + 1);
        assert_eq!(
            (
                copied.bytes(),
                arrays
                    .requirements()
                    .get(pool.topology().host_domain())
                    .unwrap()
                    .total()
                    .unwrap()
            ),
            (host, full)
        );
        assert_eq!(usage(&pool), (before.0 + bytes, capacity, capacity));
        assert_eq!(history(copied.as_sampler()), &[3, 11, 7, 19, 5]);
        assert_ne!(
            history(copied.as_sampler()).as_ptr(),
            history(sampler.as_sampler()).as_ptr()
        );
        if let (
            ConfiguredTextSampler::MirostatV2(original),
            ConfiguredTextSampler::MirostatV2(copy),
        ) = (sampler.as_sampler(), copied.as_sampler())
        {
            assert_ne!(original.mu(), 2.0 * original.tau());
            assert_eq!(
                (copy.mu(), copy.tau(), copy.eta()),
                (original.mu(), original.tau(), original.eta())
            );
        }
        grow(&mut sampler, &[29]);
        assert_eq!(history(copied.as_sampler()), &[3, 11, 7, 19, 5]);
        drop((sampler, preparation, run, physical));
        assert_eq!(
            pool.payload_used_bytes().unwrap(),
            ARRAY_SOURCE_BYTES + bytes
        );
        let (custody, scope) = arrays.into_parts();
        scope.certify().unwrap();
        assert_eq!(pool.payload_used_bytes().unwrap(), bytes);
        drop((copied, custody));
        assert_eq!(pool.payload_used_bytes().unwrap(), 0);
        assert_eq!(pool.payload_effective_capacity().unwrap(), (1 << 20));
    }
}

#[test]
fn domain_limits_and_headroom_price_the_combined_account_before_copying() {
    let pool = crate::working_memory::memory_fixture::host_ledger((1 << 20), 0).unwrap();
    let _physical = pool
        .register_host_storage([(1u32, ARRAY_SOURCE_BYTES)])
        .unwrap();
    let (sampler, _preparation, _run) = source(&pool, (1 << 20), 8, false);
    let copy = joint(&pool, sampler.borrow_funded());
    let payload = copy.required_bytes().unwrap();
    let required = full_copy_bytes(&pool, &copy);
    let initial = current(&pool);
    let before = usage(&pool);
    let before_attempts = attempts();
    let mut limits = WorkspaceCopyLimits::new(crate::working_memory::memory_fixture::host_limits(
        (1 << 20),
    ));
    limits.additional_headroom = eredu_core::MemoryHeadroomDeclarations::new([("host".into(), 7)]);
    limits.memory_limits =
        crate::working_memory::memory_fixture::host_limits(initial + required + 6);
    assert!(matches!(
        pool.copy_sampling_components(copy, limits.clone()),
        Err(SamplingCopyAdmissionError::Memory(WorkingMemoryError::Domain(eredu_core::MemoryDomainError::BudgetExceeded { requested_bytes: required_bytes, limit_bytes, existing_bytes, .. })))
            if required_bytes == required + 7 && limit_bytes - existing_bytes == required + 6
    ));
    limits.additional_headroom =
        eredu_core::MemoryHeadroomDeclarations::new([("host".into(), u64::MAX)]);
    assert!(matches!(
        pool.copy_sampling_components(joint(&pool, sampler.borrow_funded()), limits.clone()),
        Err(SamplingCopyAdmissionError::Memory(
            WorkingMemoryError::Overflow
                | WorkingMemoryError::Domain(eredu_core::MemoryDomainError::Overflow)
        ))
    ));
    assert_eq!(usage(&pool), before);
    assert_eq!(attempts(), before_attempts);
    limits.additional_headroom = eredu_core::MemoryHeadroomDeclarations::new([("host".into(), 7)]);
    limits.memory_limits =
        crate::working_memory::memory_fixture::host_limits(initial + required + 7);
    limits.memory_limits =
        crate::working_memory::memory_fixture::host_limits(initial + required + 7);
    let copied = pool
        .copy_sampling_components(joint(&pool, sampler.borrow_funded()), limits.clone())
        .unwrap();
    assert_eq!(
        copied
            .1
            .requirements()
            .get(pool.topology().host_domain())
            .unwrap()
            .total()
            .unwrap(),
        required + 7
    );
    assert_eq!(copied.0.bytes(), payload - COPY_BYTES);
    settle(copied);
    assert_eq!(pool.payload_used_bytes().unwrap(), before.0);
}

#[test]
fn each_source_domain_is_checked_before_one_joint_commit() {
    let a = crate::working_memory::memory_fixture::host_ledger((1 << 20), 0).unwrap();
    let b = crate::working_memory::memory_fixture::host_ledger((1 << 20), 0).unwrap();
    let _arrays_a = a
        .register_host_storage([(1u32, ARRAY_SOURCE_BYTES)])
        .unwrap();
    let _arrays_b = b
        .register_host_storage([(1u32, ARRAY_SOURCE_BYTES)])
        .unwrap();
    let (sampler, _preparation, _run) = source(&a, (1 << 20), 8, false);
    let before = (usage(&a), usage(&b), attempts());
    // A genuine sampler from A cannot be authenticated by B's array proof,
    // and A cannot admit the independently valid but foreign array proof.
    for destination in [&a, &b] {
        let plan = RegisteredSamplingCopy::prepare(
            sampler.borrow_funded(),
            copy_plan(&b, 1, ARRAY_SOURCE_BYTES, 1),
        )
        .unwrap();
        assert!(matches!(
            destination.copy_sampling_components(
                plan,
                WorkspaceCopyLimits::new(crate::working_memory::memory_fixture::host_limits(
                    (1 << 20)
                ))
            ),
            Err(SamplingCopyAdmissionError::Memory(
                WorkingMemoryError::IdentityMismatch
            ))
        ));
    }
    assert_eq!((usage(&a), usage(&b), attempts()), before);
}

#[test]
fn either_source_quarantined_after_preparation_rejects_without_destination_mutation() {
    let pool = crate::working_memory::memory_fixture::host_ledger((1 << 20), 0).unwrap();
    let _physical = pool
        .register_host_storage([(1u32, ARRAY_SOURCE_BYTES)])
        .unwrap();
    let (sampler, _preparation, run) = source(&pool, (1 << 20), 8, false);
    let plan = joint(&pool, sampler.borrow_funded());
    drop(run.scope().unwrap());
    let before = (usage(&pool), attempts());
    assert!(matches!(
        pool.copy_sampling_components(
            plan,
            WorkspaceCopyLimits::new(crate::working_memory::memory_fixture::host_limits(
                (1 << 20)
            ))
        ),
        Err(SamplingCopyAdmissionError::Memory(
            WorkingMemoryError::ExecutionFenced
        ))
    ));
    assert_eq!((usage(&pool), attempts()), before);

    let pool = crate::working_memory::memory_fixture::host_ledger((1 << 20), 0).unwrap();
    let _physical = pool
        .register_host_storage([(1u32, ARRAY_SOURCE_BYTES)])
        .unwrap();
    let (sampler, _preparation, _run) = source(&pool, (1 << 20), 8, false);
    let origin = pool
        .admit_workspace_copy(
            copy_plan(&pool, 1, ARRAY_SOURCE_BYTES, 1),
            WorkspaceCopyLimits::new(crate::working_memory::memory_fixture::host_limits(
                (1 << 20),
            )),
        )
        .unwrap();
    let (origin, scope) = origin.into_parts();
    let _registered = scope.publish_host_storage_fixture([(2u32, 16)]).unwrap();
    let plan = RegisteredSamplingCopy::prepare(sampler.borrow_funded(), copy_plan(&pool, 2, 16, 1))
        .unwrap();
    // The source was healthy during preparation. Admission must recheck its
    // allocation origin, not merely trust the still-live registry key.
    drop(scope);
    let before = (usage(&pool), attempts());
    assert!(matches!(
        pool.copy_sampling_components(
            plan,
            WorkspaceCopyLimits::new(crate::working_memory::memory_fixture::host_limits(
                (1 << 20)
            ))
        ),
        Err(SamplingCopyAdmissionError::Memory(
            WorkingMemoryError::ExecutionFenced
        ))
    ));
    assert_eq!((usage(&pool), attempts()), before);
    drop(origin);
}

#[test]
fn native_adoption_cannot_spend_history_hold_and_alias_credit_is_not_double_counted() {
    let pool = crate::working_memory::memory_fixture::host_ledger((1 << 20), 0).unwrap();
    let physical = pool
        .register_host_storage([(1u32, ARRAY_SOURCE_BYTES)])
        .unwrap();
    let (mut sampler, preparation, run) = source(&pool, (1 << 20), 8, false);
    grow(&mut sampler, &[3, 11, 7, 19, 5]);
    let host = sampler
        .as_sampler()
        .prepare_copy()
        .unwrap()
        .retained_bytes();
    let (copied, native) = pool
        .copy_sampling_components(
            joint(&pool, sampler.borrow_funded()),
            WorkspaceCopyLimits::new(crate::working_memory::memory_fixture::host_limits(
                (1 << 20),
            )),
        )
        .unwrap();
    let (custody, scope) = native.into_parts();
    drop((sampler, preparation, run, physical));
    let mut outputs = scope
        .publish_host_storage_fixture([(1u32, ARRAY_SOURCE_BYTES), (2, COPY_BYTES)])
        .unwrap();
    let alias = outputs.remove(&1).unwrap();
    let before = usage(&pool);
    assert!(matches!(
        scope.publish_host_storage_fixture([(3u32, 1)]),
        Err(WorkingMemoryError::DomainAllowanceExceeded {
            required_bytes: 1,
            available_bytes: 0,
            ..
        })
    ));
    assert_eq!(usage(&pool), before);
    // Retiring a published key returns its credit to the same envelope. The
    // host hold still blocks over-adoption when the credit is used again.
    drop(outputs);
    let output = scope
        .publish_host_storage_fixture([(2u32, COPY_BYTES)])
        .unwrap();
    assert!(matches!(
        scope.publish_host_storage_fixture([(3u32, 1)]),
        Err(WorkingMemoryError::DomainAllowanceExceeded {
            available_bytes: 0,
            ..
        })
    ));
    drop(alias);
    scope.certify().unwrap();
    assert_eq!(pool.payload_used_bytes().unwrap(), host + COPY_BYTES);
    assert!(matches!(
        pool.pin_registered_storage([(1u32, ARRAY_SOURCE_BYTES)]),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    drop(custody);
    assert_eq!(pool.payload_used_bytes().unwrap(), host + COPY_BYTES);
    drop(copied);
    assert_eq!(pool.payload_used_bytes().unwrap(), COPY_BYTES);
    drop(output);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}

#[test]
fn copied_sampler_keeps_destination_identity_after_native_custody_and_parent_retire() {
    let pool = crate::working_memory::memory_fixture::host_ledger((1 << 20), 0).unwrap();
    let physical = pool
        .register_host_storage([(1u32, ARRAY_SOURCE_BYTES)])
        .unwrap();
    let (mut sampler, preparation, run) = source(&pool, (1 << 20), 8, true);
    grow(&mut sampler, &[3, 11, 7, 19, 5]);
    let (copied, native) = pool
        .copy_sampling_components(
            joint(&pool, sampler.borrow_funded()),
            WorkspaceCopyLimits::new(crate::working_memory::memory_fixture::host_limits(
                (1 << 20),
            )),
        )
        .unwrap();
    let (custody, scope) = native.into_parts();
    let output = scope.publish_host_storage_fixture([(2u32, 16)]).unwrap();
    scope.certify().unwrap();
    drop((custody, sampler, preparation, run, physical));
    // The source funding run is closed, but its immutable host scope is still
    // live. This needs the new account identity, not the original run identity.
    let second = pool
        .copy_sampler(
            copied.borrow_funded(),
            SamplerCopyLimits::new(crate::working_memory::memory_fixture::host_limits(
                (1 << 20),
            )),
        )
        .unwrap();
    let next = RegisteredSamplingCopy::prepare(copied.borrow_funded(), copy_plan(&pool, 2, 16, 1))
        .unwrap();
    let third = pool
        .copy_sampling_components(
            next,
            WorkspaceCopyLimits::new(crate::working_memory::memory_fixture::host_limits(
                (1 << 20),
            )),
        )
        .unwrap();
    assert_eq!(history(second.as_sampler()), &[3, 11, 7, 19, 5]);
    assert_eq!(history(third.0.as_sampler()), history(second.as_sampler()));
    assert_ne!(
        history(third.0.as_sampler()).as_ptr(),
        history(second.as_sampler()).as_ptr()
    );
    drop((copied, output));
    settle(third);
    assert_eq!(pool.payload_used_bytes().unwrap(), second.bytes());
    drop(second);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}

#[test]
fn host_payload_can_retire_first_without_certifying_the_native_scope() {
    let pool = crate::working_memory::memory_fixture::host_ledger((1 << 20), 0).unwrap();
    let physical = pool
        .register_host_storage([(1u32, ARRAY_SOURCE_BYTES)])
        .unwrap();
    let (sampler, preparation, run) = source(&pool, (1 << 20), 8, false);
    let (copied, native) = pool
        .copy_sampling_components(
            joint(&pool, sampler.borrow_funded()),
            WorkspaceCopyLimits::new(crate::working_memory::memory_fixture::host_limits(
                (1 << 20),
            )),
        )
        .unwrap();
    let bytes = copied.bytes() + COPY_BYTES;
    let (custody, scope) = native.into_parts();
    drop((sampler, preparation, run, physical, copied, custody));
    assert_eq!(
        pool.payload_used_bytes().unwrap(),
        ARRAY_SOURCE_BYTES + bytes
    );
    // The payload has retired, so its former hold is spendable. Neither the
    // host-only cleanup nor closing custody has certified this native work.
    let output = scope.publish_host_storage_fixture([(2u32, bytes)]).unwrap();
    assert_eq!(
        pool.payload_used_bytes().unwrap(),
        ARRAY_SOURCE_BYTES + bytes
    );
    scope.certify().unwrap();
    assert_eq!(pool.payload_used_bytes().unwrap(), bytes);
    drop(output);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}

#[test]
fn partial_native_publication_quarantines_source_pin_even_after_host_retirement() {
    let pool = crate::working_memory::memory_fixture::host_ledger((1 << 20), 0).unwrap();
    let physical = pool
        .register_host_storage([(1u32, ARRAY_SOURCE_BYTES)])
        .unwrap();
    let (sampler, preparation, run) = source(&pool, (1 << 20), 8, false);
    let (copied, native) = pool
        .copy_sampling_components(
            joint(&pool, sampler.borrow_funded()),
            WorkspaceCopyLimits::new(crate::working_memory::memory_fixture::host_limits(
                (1 << 20),
            )),
        )
        .unwrap();
    let bytes = copied.bytes() + COPY_BYTES;
    let (custody, scope) = native.into_parts();
    let mut outputs = scope
        .publish_host_storage_fixture([(2u32, 16), (3, 16)])
        .unwrap();
    let attached = outputs.remove(&2).unwrap();
    drop(outputs);
    drop((sampler, preparation, run, physical, custody));
    drop(scope);
    let before = (usage(&pool), attempts());
    assert!(matches!(
        pool.copy_sampler(
            copied.borrow_funded(),
            SamplerCopyLimits::new(crate::working_memory::memory_fixture::host_limits(
                (1 << 20)
            ))
        ),
        Err(crate::working_memory::SamplerCopyAdmissionError::Memory(
            WorkingMemoryError::ExecutionFenced
        ))
    ));
    assert_eq!((usage(&pool), attempts()), before);
    drop((copied, attached));
    assert_eq!(
        pool.payload_used_bytes().unwrap(),
        ARRAY_SOURCE_BYTES + bytes
    );
    drop(
        pool.pin_registered_storage([(1u32, ARRAY_SOURCE_BYTES)])
            .unwrap(),
    );
    assert!(matches!(
        pool.acquire_unquoted(),
        Err(WorkingMemoryError::ReservedWorkActive)
    ));
}

#[test]
fn host_copy_unwind_keeps_committed_account_and_native_source_pin_quarantined() {
    let pool = crate::working_memory::memory_fixture::host_ledger((1 << 20), 0).unwrap();
    let physical = pool
        .register_host_storage([(1u32, ARRAY_SOURCE_BYTES)])
        .unwrap();
    let (sampler, preparation, run) = source(&pool, (1 << 20), 8, false);
    let bytes = joint(&pool, sampler.borrow_funded())
        .required_bytes()
        .unwrap();
    FAIL_COPY.with(|flag| flag.set(true));
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        pool.copy_sampling_components(
            joint(&pool, sampler.borrow_funded()),
            WorkspaceCopyLimits::new(crate::working_memory::memory_fixture::host_limits(
                (1 << 20),
            )),
        )
        .unwrap()
    }));
    assert!(result.is_err());
    drop((sampler, preparation, run, physical));
    assert_eq!(
        pool.payload_used_bytes().unwrap(),
        ARRAY_SOURCE_BYTES + bytes
    );
    drop(
        pool.pin_registered_storage([(1u32, ARRAY_SOURCE_BYTES)])
            .unwrap(),
    );
    assert!(matches!(
        pool.acquire_unquoted(),
        Err(WorkingMemoryError::ReservedWorkActive)
    ));
}

mod complete_source;

mod saved_source;
