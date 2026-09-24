use super::*;
use eredu_core::residency::*;
use eredu_runtime::residency::conversion_retention::*;

pub(super) fn budget(request: &eredu_runtime::NormalizedLoadRequest) -> ConversionRetentionBudget {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let layers = request.weight_residency().layers();
    let eligibility = if layers.offload().unwrap().device_budget_bytes().is_some() {
        ParameterConversionRetentionEligibility::DeviceResidencyLimit
    } else {
        match layers {
            eredu_runtime::LayerWeightResidency::FullyResident => {
                ParameterConversionRetentionEligibility::Eligible
            }
            eredu_runtime::LayerWeightResidency::LayerwiseHost(_) => {
                ParameterConversionRetentionEligibility::HostLayerwise
            }
            eredu_runtime::LayerWeightResidency::DenseDiskStream(_) => {
                ParameterConversionRetentionEligibility::DiskStreamed
            }
            _ => panic!("unsupported fixture residency"),
        }
    };
    ConversionRetentionRegistry::default()
        .budget(
            ParameterConversionRetentionGroup(eredu_core::resources::ResourceIdentity {
                scope: format!("mock-execution-{}", NEXT.fetch_add(1, Ordering::Relaxed)),
                key: "retention".into(),
            }),
            ParameterConversionRetentionPolicyReport::resolve(
                request.parameter_conversion_retention(),
                eligibility,
            ),
        )
        .unwrap()
}

impl ParameterConversionRetentionObserver for MockDrafter {
    fn parameter_conversion_retention(
        &self,
    ) -> Result<Observed<Vec<ParameterConversionRetentionReport>>, eredu_core::BackendFailure> {
        Ok(self.retention.as_ref().map_or_else(
            || Observed::unsupported("fixture auxiliary drafter has no retention ledger"),
            |budget| Observed::exact(vec![budget.report()], "independent draft ledger"),
        ))
    }
}

fn plan(policy: Option<ParameterConversionRetentionPolicy>) -> ExecutionPlan {
    ExecutionPlan::fully_resident(DevicePlan::new("mock", "gpu:0").unwrap())
        .with_parameter_conversion_retention(policy)
}

#[test]
fn loaded_facade_preserves_policy_scope_and_independent_models() {
    let artifact = TestDirectory::new();
    write_loadable_text_artifact(artifact.path());
    let mut first =
        LoadedModel::load_execution_plan(&MockBackend, artifact.path(), &plan(None)).unwrap();
    let before = first.parameter_conversion_retention().unwrap();
    let default = &before.target.value().unwrap()[0];
    assert_eq!(
        default.policy.value().unwrap().effective,
        ParameterConversionRetentionPolicy::MANAGED_DEFAULT
    );
    assert_eq!(
        default.policy.value().unwrap().source,
        ParameterConversionRetentionPolicySource::ManagedDefault
    );
    assert_eq!(default.usage.value().unwrap().retained_payload_bytes, 0);
    assert!(before.external_drafter.is_none());
    for policy in [
        ParameterConversionRetentionPolicy::Disabled,
        ParameterConversionRetentionPolicy::Bounded { max_bytes: 17 },
        ParameterConversionRetentionPolicy::Bounded { max_bytes: 0 },
        ParameterConversionRetentionPolicy::Unlimited,
    ] {
        let other =
            LoadedModel::load_execution_plan(&MockBackend, artifact.path(), &plan(Some(policy)))
                .unwrap();
        let observation = other.parameter_conversion_retention().unwrap();
        let report = &observation.target.value().unwrap()[0];
        assert_ne!(report.group, default.group);
        assert_eq!(report.policy.value().unwrap().requested, policy);
        assert_eq!(
            report.policy.value().unwrap().effective,
            policy.normalized()
        );
        assert_eq!(
            report.policy.value().unwrap().source,
            ParameterConversionRetentionPolicySource::Explicit
        );
        assert_eq!(
            other
                .model()
                .static_memory()
                .unwrap()
                .parameter_conversion_retention,
            observation.target
        );
        let encoded = serde_json::to_string(&observation).unwrap();
        assert_eq!(
            serde_json::from_str::<ExecutionConversionRetentionReport>(&encoded).unwrap(),
            observation
        );
        assert_eq!(first.parameter_conversion_retention().unwrap(), before);
    }
    first.model_mut().reset().unwrap();
    assert_eq!(first.parameter_conversion_retention().unwrap(), before);
}

#[test]
fn loaded_exclusions_preserve_the_explicit_request() {
    let artifact = TestDirectory::new();
    write_loadable_text_artifact(artifact.path());
    for (residency, eligibility) in [
        (
            ResidencyPlan::LayerwiseHost {
                device_budget_bytes: None,
                host_budget_bytes: None,
                device_layer_window: 1,
            },
            ParameterConversionRetentionEligibility::HostLayerwise,
        ),
        (
            ResidencyPlan::LayerwiseHost {
                device_budget_bytes: Some(1024),
                host_budget_bytes: None,
                device_layer_window: 1,
            },
            ParameterConversionRetentionEligibility::DeviceResidencyLimit,
        ),
        (
            ResidencyPlan::DenseDiskStream {
                device_budget_bytes: 1024,
                host_budget_bytes: 2048,
                host_lookahead: 2,
                background_queue: 1,
            },
            ParameterConversionRetentionEligibility::DeviceResidencyLimit,
        ),
    ] {
        let requested =
            plan(Some(ParameterConversionRetentionPolicy::Unlimited)).with_residency(residency);
        let model =
            LoadedModel::load_execution_plan(&MockBackend, artifact.path(), &requested).unwrap();
        let report = model.parameter_conversion_retention().unwrap();
        let selected = report.target.value().unwrap()[0].policy.value().unwrap();
        assert_eq!(
            selected.requested,
            ParameterConversionRetentionPolicy::Unlimited
        );
        assert_eq!(
            selected.effective,
            ParameterConversionRetentionPolicy::Disabled
        );
        assert_eq!(selected.eligibility, eligibility);
    }
}

#[test]
fn telemetry_queries_leave_submission_authority_and_reservations_untouched() {
    let mut runtime = ModelRuntime::prepare(
        MockBackend,
        eredu_runtime::NormalizedLoadRequest::default().with_parameter_conversion_retention(Some(
            ParameterConversionRetentionPolicy::Bounded { max_bytes: 16 },
        )),
    )
    .unwrap();
    let registry = ConversionRetentionRegistry::default();
    let parameter = registry
        .parameter(
            eredu_core::resources::ResourceIdentity {
                scope: "fixture".into(),
                key: "weight".into(),
            },
            16,
        )
        .unwrap();
    let reserved = parameter.reserve(&runtime.session().retention).unwrap();
    let pending = runtime.session_mut().authority.begin_submission().unwrap();
    let before = MockBackend::parameter_conversion_retention(&runtime).unwrap();
    assert_eq!(
        before.value().unwrap()[0]
            .usage
            .value()
            .unwrap()
            .reserved_payload_bytes,
        16
    );
    for _ in 0..3 {
        assert_eq!(
            MockBackend::parameter_conversion_retention(&runtime).unwrap(),
            before
        );
        assert_eq!(runtime.session().cache_positions, 0);
        assert!(runtime.session().authority.require_idle().is_err());
    }
    drop(reserved);
    assert_eq!(
        MockBackend::parameter_conversion_retention(&runtime)
            .unwrap()
            .value()
            .unwrap()[0]
            .usage
            .value()
            .unwrap()
            .reserved_payload_bytes,
        0
    );
    // The query left ordinary execution authority usable.
    drop(pending);
    runtime.prefill(vec![1, 2]).unwrap().wait().unwrap();
}

#[test]
fn old_static_memory_documents_keep_retention_unknown() {
    let runtime = ModelRuntime::prepare(MockBackend, Default::default()).unwrap();
    let mut value = serde_json::to_value(MockBackend::static_memory(&runtime).unwrap()).unwrap();
    value
        .as_object_mut()
        .unwrap()
        .remove("parameter_conversion_retention");
    let old: StaticMemoryReport = serde_json::from_value(value).unwrap();
    assert!(matches!(
        old.parameter_conversion_retention,
        Observed::Unavailable { .. }
    ));
}

#[test]
fn composed_reports_distinguish_independent_drafter_and_embedded_ownership() {
    let artifact = TestDirectory::new();
    write_loadable_text_artifact(artifact.path());
    let assistant = TestDirectory::new();
    write_loadable_assistant_artifact(assistant.path());
    let policy = ParameterConversionRetentionPolicy::Bounded { max_bytes: 32 };
    let external = plan(Some(policy)).with_drafting(DraftingPlan::External {
        model: assistant.path().display().to_string(),
        placement: DraftPlacementPlan::Target,
        max_draft_tokens: 4,
        lookahead: false,
        adaptive_lookahead: false,
    });
    let mut planned =
        LoadedModel::load_execution_plan(&MockBackend, artifact.path(), &external).unwrap();
    let before = planned.parameter_conversion_retention().unwrap();
    let target = &before.target.value().unwrap()[0];
    let draft = &before.external_drafter.as_ref().unwrap().value().unwrap()[0];
    assert_ne!(target.group, draft.group);
    assert_eq!(target.policy, draft.policy);
    assert_eq!(target.policy.value().unwrap().effective, policy);
    assert_eq!(planned.parameter_conversion_retention().unwrap(), before);
    // Unknown observations must not make an existing participant disappear.
    if let RealizedDrafting::External(draft) = planned.parts_mut().1 {
        draft.retention = None;
    }
    let unsupported = planned.parameter_conversion_retention().unwrap();
    assert!(matches!(
        unsupported.external_drafter,
        Some(Observed::Unsupported { .. })
    ));
    assert_eq!(unsupported.target, before.target);
    let embedded = plan(Some(policy)).with_drafting(DraftingPlan::Embedded {
        max_draft_tokens: 4,
        lookahead: false,
        adaptive_lookahead: false,
    });
    let planned =
        LoadedModel::load_execution_plan(&MockBackend, artifact.path(), &embedded).unwrap();
    let report = planned.parameter_conversion_retention().unwrap();
    assert_eq!(report.target.value().unwrap().len(), 1);
    assert!(report.external_drafter.is_none());
}

#[test]
fn retention_query_failure_preserves_native_source_and_pending_authority() {
    let mut runtime = ModelRuntime::prepare(MockBackend, Default::default()).unwrap();
    runtime.session_mut().retention_failure = true;
    let pending = runtime.session_mut().authority.begin_submission().unwrap();
    let error = MockBackend::parameter_conversion_retention(&runtime).unwrap_err();
    let source = std::error::Error::source(&error).unwrap();
    assert!(source.downcast_ref::<MockError>().is_some());
    assert!(runtime.session().authority.require_idle().is_err());
    assert_eq!(runtime.session().cache_positions, 0);
    drop(pending);
}

pub(super) fn retain_fixture(runtime: &ModelRuntime<MockBackend>) {
    let registry = ConversionRetentionRegistry::default();
    let identity = |key: &str| eredu_core::resources::ResourceIdentity {
        scope: runtime.session().retention.group().0.scope.clone(),
        key: key.into(),
    };
    let parameter = registry.parameter(identity("weight"), 16).unwrap();
    let ConversionRetentionAdmission::Reserved(reservation) =
        parameter.reserve(&runtime.session().retention).unwrap()
    else {
        panic!("fresh fixture")
    };
    let claim = reservation
        .publish(registry.allocation(identity("allocation"), 16).unwrap())
        .unwrap();
    runtime
        .session()
        .retention_claims
        .lock()
        .unwrap()
        .push(claim);
}

#[test]
fn ordinary_trim_preserves_request_and_reset_retention_then_allows_readmission() {
    let mut runtime = ModelRuntime::prepare(MockBackend, Default::default()).unwrap();
    runtime.session_mut().cache_positions = 7;
    retain_fixture(&runtime);
    runtime.reset().unwrap();
    assert_eq!(
        runtime
            .session()
            .retention
            .report()
            .usage
            .value()
            .unwrap()
            .retained_payload_bytes,
        16
    );
    runtime.session_mut().cache_positions = 7;
    let identity = runtime.session().intervention_identity.clone();
    let pending = runtime.session_mut().authority.begin_submission().unwrap();
    assert!(matches!(
        MockBackend::trim_parameter_conversions(&mut runtime),
        Err(ParameterConversionTrimError::NotQuiescent)
    ));
    assert!(runtime.trim_parameter_conversions().is_err());
    assert_eq!(
        runtime
            .session()
            .retention
            .report()
            .usage
            .value()
            .unwrap()
            .retained_payload_bytes,
        16
    );
    assert!(runtime.session().authority.require_idle().is_err());
    drop(pending);
    let report = runtime.trim_parameter_conversions().unwrap();
    assert_eq!(report[0].released_claims, 1);
    assert_eq!(report[0].released_payload_bytes, 16);
    assert_eq!(report[0].remaining.retained_payload_bytes, 0);
    assert!(report[0].reclaimed_backing_bytes.value().is_none());
    assert_eq!(runtime.session().cache_positions, 7);
    assert_eq!(runtime.session().intervention_identity, identity);
    assert_eq!(
        runtime.trim_parameter_conversions().unwrap()[0].released_claims,
        0
    );
    retain_fixture(&runtime);
    assert_eq!(
        runtime.trim_parameter_conversions().unwrap()[0].released_payload_bytes,
        16
    );
    runtime.session_mut().retention_failure = true;
    let ParameterConversionTrimError::Backend(error) =
        runtime.trim_parameter_conversions().unwrap_err()
    else {
        panic!("native error")
    };
    assert!(std::error::Error::source(&error)
        .unwrap()
        .downcast_ref::<MockError>()
        .is_some());
}

#[test]
fn controlled_driver_requires_completed_drained_boundary_before_trimming() {
    let mut runtime = ModelRuntime::prepare(MockBackend, Default::default()).unwrap();
    retain_fixture(&runtime);
    struct All;
    impl eredu_core::TokenFilterController for All {
        type Error = std::convert::Infallible;
        fn current_filter(&mut self) -> Result<TokenFilter, Self::Error> {
            Ok(TokenFilter::All)
        }
        fn commit_token(&mut self, _: u32) -> Result<(), Self::Error> {
            Ok(())
        }
        fn is_complete(&mut self) -> Result<bool, Self::Error> {
            Ok(false)
        }
    }
    let config = TextGenerationConfig::new(
        eredu_core::resolve_generation_config(
            None,
            GenerationConfigOverrides {
                max_new_tokens: Some(3),
                ..Default::default()
            },
        )
        .unwrap(),
    )
    .with_seed(17);
    let mut driver = eredu_core::TextGenerationDriver::new(&mut runtime);
    let mut state = driver.start(vec![1, 2], config, All).unwrap();
    let token = driver.advance(&mut state).unwrap().unwrap().token_id();
    assert!(matches!(
        driver.quiescent(&mut state),
        Err(eredu_core::TextContinuationError::NotQuiescent)
    ));
    assert_eq!(
        driver
            .runtime()
            .session()
            .retention
            .report()
            .usage
            .value()
            .unwrap()
            .retained_payload_bytes,
        16
    );
    driver.take_completed_step(&mut state).unwrap();
    let position = driver.runtime().session().cache_positions;
    let remaining = state.remaining_tokens();
    let report = driver
        .quiescent(&mut state)
        .unwrap()
        .trim_parameter_conversions()
        .unwrap();
    assert_eq!(report[0].released_payload_bytes, 16);
    assert_eq!(driver.runtime().session().cache_positions, position);
    assert_eq!(state.remaining_tokens(), remaining);
    assert_eq!(
        driver.advance(&mut state).unwrap().unwrap().token_id(),
        token + 1
    );
}

impl ParameterConversionRetentionTrimmer for MockDrafter {
    fn trim_parameter_conversions(
        &mut self,
    ) -> Result<Vec<ParameterConversionRetentionTrimReport>, ParameterConversionTrimError> {
        match &self.retention {
            Some(budget) => budget
                .release_claims(&mut Vec::new())
                .map(|report| vec![report])
                .map_err(|error| eredu_core::BackendFailure::from_error(error).into()),
            None => Err(ParameterConversionTrimError::Unsupported),
        }
    }
}

#[test]
fn planned_trim_preserves_independent_external_and_shared_embedded_scopes() {
    let artifact = TestDirectory::new();
    write_loadable_text_artifact(artifact.path());
    let assistant = TestDirectory::new();
    write_loadable_assistant_artifact(assistant.path());
    for drafting in [
        DraftingPlan::External {
            model: assistant.path().display().to_string(),
            placement: DraftPlacementPlan::Target,
            max_draft_tokens: 4,
            lookahead: false,
            adaptive_lookahead: false,
        },
        DraftingPlan::Embedded {
            max_draft_tokens: 4,
            lookahead: false,
            adaptive_lookahead: false,
        },
    ] {
        let mut planned = LoadedModel::load_execution_plan(
            &MockBackend,
            artifact.path(),
            &plan(None).with_drafting(drafting),
        )
        .unwrap();
        let before = planned.parameter_conversion_retention().unwrap();
        let trimmed = planned.trim_parameter_conversions().unwrap();
        assert_eq!(
            trimmed.target[0].group,
            before.target.value().unwrap()[0].group
        );
        assert_eq!(trimmed.target[0].released_payload_bytes, 0);
        assert_eq!(
            trimmed.external_drafter.is_some(),
            before.external_drafter.is_some()
        );
        if let Some(draft) = &trimmed.external_drafter {
            assert_eq!(
                draft[0].group,
                before.external_drafter.as_ref().unwrap().value().unwrap()[0].group
            );
            assert_ne!(draft[0].group, trimmed.target[0].group);
        }
        assert_eq!(planned.parameter_conversion_retention().unwrap(), before);
        assert_eq!(
            serde_json::from_str::<ExecutionConversionRetentionTrimReport>(
                &serde_json::to_string(&trimmed).unwrap()
            )
            .unwrap(),
            trimmed
        );
    }
}
