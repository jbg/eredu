//! A real accepted source role exercises the retained operation-bank boundary.
use super::*;
use crate::backend::runtime::execution::generic::OriginalOperationScopeCountProbe;
use crate::backend::submission_recovery::{Retention, Status, prediction};

struct PredictionPayload {
    registration: OriginalOperationScopeCountProbe,
    destroyed_with_live_scope: Rc<Cell<Option<(usize, usize)>>>,
    original: Option<eredu_runtime::working_memory::OriginalPredictionRecoveryCustody>,
}
impl Retention for PredictionPayload {
    fn observe(&self, _: Status) {}
}
impl prediction::PredictionRetention for PredictionPayload {
    fn install_prediction_custody(
        &mut self,
        custody: eredu_runtime::working_memory::OriginalPredictionRecoveryCustody,
    ) {
        self.original = Some(custody);
    }
}
impl Drop for PredictionPayload {
    fn drop(&mut self) {
        self.destroyed_with_live_scope
            .set(Some(self.registration.counts().unwrap()));
    }
}

#[test]
fn bounded_bank_activation_preserves_spent_roles_and_refuses_unfinished_source() {
    let environment = fixture::PreparedResidencyFixture::new();
    for residency in [1, 2] {
        for asynchronous_refusal in [false, true] {
            let (mut runtime, artifact, baseline) = environment.load(residency, None);
            let probe = Probe::new(&runtime, None, false);
            let run = ControlledTextGeneration::from_token_ids_with_sequence(
                &mut runtime,
                eredu_core::TokenIdsInputPlan::new(&[2, 5, 7, 3, 11]).unwrap(),
                original_source_config(),
                disk::Controller::default(),
                None,
                GenerationSequenceRequest::new(4, &[]),
            )
            .unwrap();
            let payload = run.runtime().session().payload.clone();
            let native_runtime = payload.model.erased().prefill_roots_runtime().unwrap();
            let preparation = probe.take();
            let step = preparation
                .request
                .as_ref()
                .unwrap()
                .claim_step(&probe.context(), PendingTextInput::Prefill(()))
                .unwrap();
            let quote = preparation.quote.as_ref().unwrap();
            let original = quote
                .prefill_scopes
                .as_ref()
                .unwrap()
                .borrow_mut()
                .claim(&step)
                .unwrap();
            let controls = quote.original_controls().unwrap();
            let registration = quote.operation_registration.clone().unwrap();
            assert!(matches!(
                registration.test_with_materialization_loan(|_| ()),
                Err(Error::PrefillScopeUnavailable)
            ));
            let host = quote.operation_host_destinations.borrow_mut().take();
            // The same retained source, constructor and admitted authority used by
            // claim_prefill_scopes; no registry or activation flag is synthesized.
            let operations = payload
                .model
                .erased()
                .prepare_original_operation_banks(
                    &payload.memory_ledger,
                    &original,
                    &step,
                    registration.clone(),
                    controls.clone(),
                    host,
                    quote
                        .layerwise
                        .as_ref()
                        .and_then(LayerwiseQuoteSources::retained_workspace),
                    quote.native_recipe.as_ref(),
                    quote.planning_metadata.as_ref(),
                )
                .unwrap()
                .expect("Host/Disk fixture must install an actual unit bank");
            assert_eq!(registration.test_scope_counts().unwrap(), (0, 0));
            for _ in 0..3 {
                let activation = operations.activate().unwrap();
                assert!(matches!(
                    operations.activate(),
                    Err(Error::PrefillScopeReentrant)
                ));
                assert_eq!(registration.test_scope_counts().unwrap(), (0, 0));
                drop(activation);
            }
            let activation = operations.activate().unwrap();
            let (bank, view) = PrefillBankOwner::new(
                original,
                step.request(),
                &native_runtime,
                quote.record_quota.as_ref(),
                quote.graph_quota.as_ref().unwrap(),
                controls.clone(),
            )
            .unwrap();
            let bank = bank.with_operation_registration(registration.clone());
            let (source, roots) = view
                .begin(step.request(), PrefillControlRole::SourcePreparation)
                .unwrap();
            assert!(roots.is_none());
            assert_eq!(registration.test_scope_counts().unwrap(), (1, 1));
            let native_bank = quote.native_storage.as_ref().unwrap();
            let expected_budget = native_bank
                .try_borrow()
                .unwrap()
                .budget_for_controls(&controls)
                .unwrap()
                .clone();
            let before_loan = environment.pool.snapshot().unwrap();
            registration
                .test_with_materialization_loan(|loan| {
                    assert!(loan.budget.same_budget(&expected_budget));
                    assert!(
                        loan.funding
                            .same_account(quote.planning_metadata.as_ref().unwrap())
                    );
                })
                .unwrap();
            let called = Cell::new(false);
            {
                let _borrow = native_bank.try_borrow_mut().unwrap();
                assert!(matches!(
                    registration.test_with_materialization_loan(|_| called.set(true)),
                    Err(Error::PrefillScopeReentrant)
                ));
            }
            assert!(!called.get());
            assert_eq!(environment.pool.snapshot().unwrap(), before_loan);
            drop(expected_budget);
            // An accidentally ended lexical guard cannot certify a still accepted
            // native role. Its actual registered observer is the independent proof.
            drop(activation);
            assert!(matches!(
                registration.test_with_materialization_loan(|_| ()),
                Err(Error::PrefillScopeUnavailable)
            ));
            assert!(matches!(
                operations.activate(),
                Err(Error::PrefillScopeReentrant)
            ));
            assert_eq!(registration.test_scope_counts().unwrap(), (1, 1));
            assert!(
                crate::backend::submission_recovery::prefill::finish(source)
                    .unwrap()
                    .settled
            );
            assert_eq!(registration.test_scope_counts().unwrap(), (1, 0));
            for _ in 0..3 {
                let activation = operations.activate().unwrap();
                assert_eq!(registration.test_scope_counts().unwrap(), (1, 0));
                // Switching activity never reissues the consumed Source role.
                assert!(
                    view.begin(step.request(), PrefillControlRole::SourcePreparation)
                        .is_err()
                );
                assert_eq!(registration.test_scope_counts().unwrap(), (1, 0));
                drop(activation);
            }
            // Prediction roles must use the same positive retirement join. This
            // real payload fits the actual model ticket's admitted concrete slot.
            assert!(
                prediction::control_bytes::<PredictionPayload>().unwrap()
                    <= prediction::control_bytes::<ScopeRetention>().unwrap()
            );
            let mut predictions = quote.claim_prediction_scopes(&step).unwrap().unwrap();
            let activation = operations.activate().unwrap();
            let destroyed = Rc::new(Cell::new(None));
            let mut prediction = prediction::begin(
                Some(predictions.take_model_execution().unwrap()),
                PredictionPayload {
                    registration: registration.test_scope_count_probe().unwrap(),
                    destroyed_with_live_scope: destroyed.clone(),
                    original: None,
                },
            )
            .unwrap();
            assert_eq!(registration.test_scope_counts().unwrap(), (2, 1));
            prediction.seal();
            assert!(prediction.finish().unwrap().settled);
            assert_eq!(
                destroyed.get(),
                Some((2, 1)),
                "payload must retire before its live registration"
            );
            assert_eq!(registration.test_scope_counts().unwrap(), (2, 0));
            drop(activation);
            let activation = operations.activate().unwrap();
            let destroyed = Rc::new(Cell::new(None));
            let mut prediction = prediction::begin(
                Some(predictions.take_sampling().unwrap()),
                PredictionPayload {
                    registration: registration.test_scope_count_probe().unwrap(),
                    destroyed_with_live_scope: destroyed.clone(),
                    original: None,
                },
            )
            .unwrap();
            prediction.seal();
            if asynchronous_refusal {
                registration.test_with_scope_loan(|| drop(prediction));
            } else {
                let cause = registration
                    .test_with_scope_loan(|| prediction.finish())
                    .unwrap_err();
                assert!(matches!(cause, crate::backend::runtime::execution::generic::RegisteredScopeRetirementCause::Reentrant));
            }
            assert_eq!(destroyed.get(), Some((3, 1)));
            assert_eq!(
                registration.test_scope_counts().unwrap(),
                (3, 1),
                "refusal must not remove/refund the accepted role"
            );
            drop(activation);
            if asynchronous_refusal {
                assert!(
                    matches!(operations.activate(), Err(Error::PrefillScopeReentrant)),
                    "abandoned cleanup preserves its first exact cause"
                );
            }
            assert!(
                matches!(operations.activate(), Err(Error::PrefillScopeUnavailable)),
                "retirement refusal permanently fences this bank"
            );
            drop((predictions, destroyed));
            let retired_registration = registration.clone();
            drop((
                bank,
                view,
                operations,
                registration,
                controls,
                step,
                preparation,
                run,
                probe,
                payload,
            ));
            assert!(matches!(
                retired_registration.test_with_materialization_loan(|_| ()),
                Err(Error::PrefillScopeUnavailable)
            ));
            drop(retired_registration);
            runtime.synchronize().unwrap();
            environment.stream().synchronize().unwrap();
            drop((runtime, artifact));
            fixture::settle(&environment.pool, baseline);
        }
    }
}
