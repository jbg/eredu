//! Original-run replacement uses the same native publisher and distinct roles.
use super::*;
use crate::working_memory::*;
use std::convert::Infallible;

#[derive(Debug)]
struct SamplingFacts;
impl WorkspaceMechanisms for SamplingFacts {
    fn memory_topology(&self) -> Option<&eredu_core::MemoryTopology> {
        Some(crate::working_memory::memory_fixture::host_topology_ref())
    }
    fn output_placement(
        &self,
        _: WorkspaceOperationView<'_>,
        _: usize,
    ) -> Option<&eredu_core::MemoryPlacement> {
        Some(crate::working_memory::memory_fixture::host_placement())
    }
    fn scratch_placement(
        &self,
        _: WorkspaceOperationView<'_>,
    ) -> Option<&eredu_core::MemoryPlacement> {
        Some(crate::working_memory::memory_fixture::host_placement())
    }

    fn operation_bound(
        &self,
        _: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceOperationBound>, eredu_nn::Error> {
        panic!("paid sampling fixture uses borrowed facts")
    }
}
impl WorkspaceFactMechanisms for SamplingFacts {
    fn memory_topology(&self) -> Option<&eredu_core::MemoryTopology> {
        Some(crate::working_memory::memory_fixture::host_topology_ref())
    }
    fn output_placement(
        &self,
        _: WorkspaceOperationView<'_>,
        _: usize,
    ) -> Option<&eredu_core::MemoryPlacement> {
        Some(crate::working_memory::memory_fixture::host_placement())
    }
    fn scratch_placement(
        &self,
        _: WorkspaceOperationView<'_>,
    ) -> Option<&eredu_core::MemoryPlacement> {
        Some(crate::working_memory::memory_fixture::host_placement())
    }

    type Error = Infallible;
    fn operation_facts(
        &self,
        operation: WorkspaceOperationView<'_>,
    ) -> Result<Option<WorkspaceOperationFacts>, Infallible> {
        Ok(Some(WorkspaceOperationFacts {
            layout: WorkspaceEffectLayout {
                outputs: operation.outputs.len(),
                aliases: 0,
                assumption_bytes: b"fixture owns exact output layouts".len(),
            },
            scratch_bytes: 0,
        }))
    }
    fn write_operation_facts(
        &self,
        operation: WorkspaceOperationView<'_>,
        destination: WorkspaceEffectDestination<'_>,
    ) -> Result<Option<WorkspaceOperationFacts>, Infallible> {
        let facts = self.operation_facts(operation)?.unwrap();
        destination.validate(facts.layout).unwrap();
        destination
            .assumptions
            .copy_from_slice(b"fixture owns exact output layouts");
        for (out, layout) in destination.outputs.iter_mut().zip(operation.outputs.iter()) {
            *out = WorkspaceOutputEffect::Allocate(layout.bytes().unwrap());
        }
        Ok(Some(facts))
    }
    fn host_facts(
        &self,
        _: WorkspaceOperationView<'_>,
    ) -> Result<Option<WorkspaceHostFacts>, Infallible> {
        // This fixed fixture has no separate host staging or producer payload.
        Ok(Some(WorkspaceHostFacts {
            bytes: 0,
            assumption_bytes: b"fixture has no host staging".len(),
        }))
    }
    fn write_host_facts(
        &self,
        operation: WorkspaceOperationView<'_>,
        destination: WorkspaceHostDestination<'_>,
    ) -> Result<Option<WorkspaceHostFacts>, Infallible> {
        let facts = self.host_facts(operation)?.unwrap();
        destination.validate(facts).unwrap();
        destination
            .assumptions
            .copy_from_slice(b"fixture has no host staging");
        Ok(Some(facts))
    }
}
fn candidate(
    preparation: &InferenceTextPreparation,
    context: &eredu_core::TextStepContext,
    funding: &eredu_core::HostMetadataFunding,
) -> SamplingExtensionQuote {
    let pending = preparation
        .request()
        .begin_sampling_extension(context, funding)
        .unwrap();
    let context =
        WorkspaceContext::new_with_metadata_funding(SamplingFacts, funding.clone()).unwrap();
    let mut observer = SamplingWorkspacePlanCollector::new(
        pending.geometry(),
        pending.remaining_steps(),
        &context,
    )
    .unwrap();
    let layout = context.layout(&[1, 37], WorkspaceDtype::Float32).unwrap();
    let sampler = crate::ConfiguredTextSampler::Standard(crate::GenerationSampler::new());
    let source_requirements = fixture_requirements(layout.bytes().unwrap());
    let source =
        WorkspaceSamplingSource::from(&layout).with_physical_domains(Some(&source_requirements));
    let report = quote_sampling_workspace_with_observer(
        &sampler,
        0.0,
        None,
        source,
        &eredu_core::TokenFilter::All,
        pending.remaining_steps(),
        &context,
        Some(&mut observer),
    )
    .unwrap();
    let plan = observer.finish(&report).unwrap();
    assert!(
        plan.records()
            .iter()
            .all(|record| matches!(record.span(), InferenceWorkspaceSpan::Sampling(_)))
    );
    pending.compose(&report, plan).unwrap()
}
fn native_candidate(
    quote: SamplingExtensionQuote,
    mechanism: &Mechanism,
) -> Result<SamplingExtensionQuote, WorkingMemoryError> {
    let steps = quote.workspace().plan().records().len() - 1;
    let covered = quote
        .workspace()
        .plan()
        .records()
        .iter()
        .map(|row| row.new_tensor_allocation_bytes());
    let capacity = covered.clone().map(Option::unwrap).sum();
    let native = PreparedNativeStoragePlan::prepare_qualified(
        quote.workspace(),
        mechanism,
        Some(capacity),
        Some((steps + 1, 1)),
        covered,
        crate::working_memory::qualified_storage::shared_bytes::<Budget>().ok(),
    )?;
    let native = native.with_retained_equation_generations(
        quote
            .workspace()
            .plan()
            .records()
            .iter()
            .map(|row| row.new_tensor_allocation_bytes()),
    )?;
    let controls = quote
        .prepare_controls(TextHostControlFacts::new(Some(0), Some(0), Some(0)))?
        .with_sampling_prediction_scopes(Some(0), Some(0))?
        .with_sampling_reseed_scope(Some(0))?
        .with_native_storage(native)?;
    quote.with_controls(controls)
}
fn extension(
    preparation: &InferenceTextPreparation,
    context: &eredu_core::TextStepContext,
    funding: &eredu_core::HostMetadataFunding,
    mechanism: &Mechanism,
) -> SamplingExtensionQuote {
    native_candidate(candidate(preparation, context, funding), mechanism).unwrap()
}
fn config() -> eredu_core::TextGenerationConfig {
    eredu_core::TextGenerationConfig::new(
        eredu_core::resolve_generation_config(
            None,
            eredu_core::GenerationConfigOverrides {
                max_new_tokens: Some(geometry().max_output_tokens as usize),
                ..Default::default()
            },
        )
        .unwrap(),
    )
}

#[test]
fn sampling_extension_uses_distinct_native_account_and_replaces_only_sampling_roles() {
    assert!(crate::working_memory::qualified_storage::qualified());
    let pool = crate::working_memory::memory_fixture::host_ledger(1_000_000, 0).unwrap();
    let root = pool.register_host_storage([(1u32, 64)]).unwrap();
    let raw = replacement_quote(&pool, geometry(), 0).into_incremental();
    let controls = PreparedTextControlWorkspace::prepare_controls(
        geometry(),
        raw.span_workspace().plan(),
        TextHostControlFacts::new(Some(0), Some(0), Some(0)),
    )
    .unwrap()
    .with_prediction_scopes(TextPredictionScopeFacts::new(
        Some(0),
        Some(0),
        Some(0),
        Some(0),
        Some(0),
    ))
    .unwrap();
    let (reservation, old_run, raw) = accept(
        &pool,
        raw.with_span_workspace_and_text_controls(controls).unwrap(),
    );
    let (mut old_workspace, _) = raw
        .into_funded_text_span_workspace(&old_run, &reservation)
        .unwrap();
    let mut old_roles = old_workspace.take_prediction_scopes().unwrap().unwrap();
    let preparation = InferenceRequest::from(&reservation)
        .prepare_text(&reservation.0.execution, geometry(), config())
        .unwrap();
    let context = crate::working_memory::original_request::tests::issued_lock_context();
    preparation.bind_run(&context).unwrap();
    preparation.claim_prompt().unwrap().finish().unwrap();
    preparation.bind_prompt().unwrap();
    preparation
        .claim_sampling(config())
        .unwrap()
        .finish()
        .unwrap();
    let funding = pool
        .prepare_workspace_metadata(
            &reservation.0.execution,
            crate::working_memory::memory_fixture::resolved_host_limits(&pool, 1_000_000),
        )
        .unwrap();
    let mut mechanism = Mechanism::new(&pool);
    mechanism.nested_key_bytes = Some(0);
    let quote = extension(&preparation, &context, &funding, &mechanism);
    let short = pool.payload_used_bytes().unwrap() + quote.required_bytes().unwrap() - 1;
    assert!(matches!(
        quote.admit(crate::working_memory::memory_fixture::resolved_host_limits(&pool, short)),
        Err(capacity_error) if matches!(capacity_numbers(&capacity_error), Some((_, _))) || matches!(capacity_error, WorkingMemoryError::MetadataConstruction(eredu_nn::workspace::WorkspaceMetadataError::Funding(eredu_core::HostMetadataFundingError::Capacity { .. })))));
    assert_eq!(
        mechanism.calls.get(),
        0,
        "refusal must precede native construction"
    );
    assert_eq!(
        preparation
            .request()
            .sampling_extension_remaining(&context)
            .unwrap(),
        geometry().max_output_tokens
    );
    let quote = extension(&preparation, &context, &funding, &mechanism);
    let mut owner = quote
        .admit(crate::working_memory::memory_fixture::resolved_host_limits(
            &pool, 1_000_000,
        ))
        .unwrap();
    let guard = owner.control_guard();
    assert!(guard.validate_reservation(&reservation).is_err());
    let reseed = owner.claim_reseed().unwrap();
    assert!(owner.claim_reseed().is_err());
    let mut bank = owner
        .take_native_storage_bank::<Mechanism>(&mechanism.selection)
        .unwrap()
        .unwrap();
    assert!(
        owner
            .take_native_storage_bank::<Mechanism>(&mechanism.selection)
            .is_err()
    );
    bank.install(mechanism.clone()).unwrap();
    let mut new_roles = owner.take_prediction_scopes().unwrap().unwrap();
    assert!(owner.take_prediction_scopes().is_err());
    owner.commit().unwrap();
    assert!(owner.commit().is_err());
    let step = preparation
        .claim_step(&context, eredu_core::PendingTextInput::Prefill(()))
        .unwrap();
    let mut old = old_roles.claim(&step).unwrap();
    assert!(old.take_sampling().is_err());
    assert!(old.take_sampling_event().is_err());
    let model = old.take_model_execution().unwrap();
    let validation = old.take_model_validation().unwrap();
    let scalar = old.take_token_scalar().unwrap();
    let mut new = new_roles.claim(&step).unwrap();
    assert!(new.take_model_execution().is_err());
    assert!(new.take_model_validation().is_err());
    assert!(new.take_token_scalar().is_err());
    let sampling = new.take_sampling().unwrap();
    let event = new.take_sampling_event().unwrap();
    assert!(new.take_sampling().is_err());
    assert!(new_roles.claim(&step).is_err());
    step.finish().unwrap();
    drop((
        owner,
        old_workspace,
        old_run,
        reservation,
        preparation,
        funding,
        root,
        old_roles,
        new_roles,
        bank,
        mechanism,
        guard,
    ));
    assert!(
        pool.payload_used_bytes().unwrap() > 0,
        "escaped role custody retains original accounts"
    );
    drop((old, new, reseed, model, validation, scalar, sampling, event));
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}

#[derive(Debug, Default)]
struct ColdReservations {
    enabled: std::sync::atomic::AtomicBool,
    calls: std::sync::atomic::AtomicUsize,
    fail: std::sync::atomic::AtomicUsize,
}
#[derive(Debug)]
struct AuditedAccount {
    calls: std::sync::Arc<ColdReservations>,
    original: eredu_core::HostMetadataFunding,
}
impl eredu_core::HostMetadataAccount for AuditedAccount {
    fn reserve_metadata(&self, bytes: usize) -> Result<(), eredu_core::HostMetadataFundingError> {
        use std::sync::atomic::Ordering::SeqCst;
        if self.calls.enabled.load(SeqCst) {
            let call = self.calls.calls.fetch_add(1, SeqCst) + 1;
            if call >= self.calls.fail.load(SeqCst) {
                return Err(eredu_core::HostMetadataFundingError::Capacity {
                    required: bytes as u64,
                    available: 0,
                });
            }
        }
        self.original.reserve_metadata(bytes)
    }
}
#[test]
fn sampling_extension_every_cold_native_plan_reservation_refuses_without_publication() {
    use std::sync::atomic::Ordering::SeqCst;
    let pool = crate::working_memory::memory_fixture::host_ledger(1_000_000, 0).unwrap();
    let reservation = crate::working_memory::funding::tests::reservation(&pool, 100, 1_000_000);
    let config = eredu_core::TextGenerationConfig::new(
        eredu_core::resolve_generation_config(
            None,
            eredu_core::GenerationConfigOverrides {
                max_new_tokens: Some(1),
                ..Default::default()
            },
        )
        .unwrap(),
    );
    let preparation = InferenceRequest::from(&reservation)
        .prepare_text(
            &reservation.0.execution,
            reservation.geometry(),
            config.clone(),
        )
        .unwrap();
    let context = crate::working_memory::original_request::tests::issued_lock_context();
    preparation.bind_run(&context).unwrap();
    preparation.claim_prompt().unwrap().finish().unwrap();
    preparation.bind_prompt().unwrap();
    preparation
        .claim_sampling(config)
        .unwrap()
        .finish()
        .unwrap();
    let mut mechanism = Mechanism::new(&pool);
    mechanism.nested_key_bytes = Some(0);
    let baseline = pool.payload_used_bytes().unwrap();
    let mut reached = 0;
    for fail in std::iter::once(usize::MAX).chain(1..=128) {
        if reached != 0 && fail > reached {
            break;
        }
        let original = pool
            .prepare_workspace_metadata(
                &reservation.0.execution,
                crate::working_memory::memory_fixture::resolved_host_limits(&pool, 1_000_000),
            )
            .unwrap();
        let calls = std::sync::Arc::new(ColdReservations::default());
        calls.fail.store(fail, SeqCst);
        let funding = eredu_core::HostMetadataFunding::new(AuditedAccount {
            calls: calls.clone(),
            original,
        })
        .unwrap();
        let quote = candidate(&preparation, &context, &funding);
        calls.enabled.store(true, SeqCst);
        let result = native_candidate(quote, &mechanism);
        if fail == usize::MAX {
            reached = calls.calls.load(SeqCst);
            assert!(reached > 8 && reached <= 128);
            assert!(result.is_ok());
        } else {
            assert!(
                matches!(result, Err(ref capacity_error) if matches!(capacity_numbers(&capacity_error), Some((_, _))) || matches!(capacity_error, WorkingMemoryError::MetadataConstruction(eredu_nn::workspace::WorkspaceMetadataError::Funding(eredu_core::HostMetadataFundingError::Capacity { .. })))),
                "reservation {fail}/{reached}: {result:?}"
            );
        }
        drop(result);
        assert_eq!(mechanism.calls.get(), 0);
        assert_eq!(
            preparation
                .request()
                .sampling_extension_remaining(&context)
                .unwrap(),
            1
        );
        drop(funding);
        assert_eq!(
            pool.payload_used_bytes().unwrap(),
            baseline,
            "reservation {fail} kept no failed prefix"
        );
    }
    drop((preparation, reservation));
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}
