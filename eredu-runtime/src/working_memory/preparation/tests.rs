use super::*;

#[derive(Debug)]
struct Facts {
    host: bool,
    tensor: bool,
}
impl WorkspaceMechanisms for Facts {
    fn operation_bound(
        &self,
        operation: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceOperationBound>, Error> {
        Ok(if !self.tensor {
            None
        } else {
            Some(WorkspaceOperationBound {
                outputs: operation
                    .outputs
                    .iter()
                    .map(|layout| layout.bytes().map(WorkspaceOutputStorage::Allocate))
                    .collect::<Result<_, _>>()?,
                scratch_bytes: 11,
                assumptions: "test eager initializer allocates its complete output".into(),
            })
        })
    }
    fn host_workspace_bound(
        &self,
        _: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceHostBound>, Error> {
        Ok(self.host.then(|| WorkspaceHostBound {
            bytes: 13,
            assumptions: "test operation staging".into(),
        }))
    }
}
fn identity_controls() -> Option<u64> {
    let controls = text_identity_control_bytes();
    if std::env::var_os("EREDU_REQUIRE_QUALIFIED_PROMPT_INPUT") == Some("1".into()) {
        assert!(
            controls.is_some(),
            "pinned prompt identity owner must qualify"
        );
    }
    controls
}
fn geometry(chunk: u64) -> InferenceGeometry {
    InferenceGeometry {
        batch_size: 1,
        cached_positions: 5,
        input_positions: 11,
        max_output_tokens: 9,
        prefill_chunk_positions: chunk,
        output: eredu_core::OutputDemand::LastPosition,
    }
}
fn outside(g: InferenceGeometry) -> ExecutionWorkspaceEstimate {
    let zero = || WorkspaceBound::bounded(0, "test component priced elsewhere");
    ExecutionWorkspaceEstimate {
        geometry: g,
        activations: zero(),
        attention: zero(),
        vocabulary: zero(),
        state_update: zero(),
        materialization: zero(),
        retained: zero(),
    }
}

#[test]
fn preparation_prices_full_input_and_real_host_capacity_independently_of_chunk_size() {
    let Some(identity_controls) = identity_controls() else {
        let context = WorkspaceContext::new(Facts {
            host: true,
            tensor: true,
        });
        assert_eq!(
            quote_text_prompt_workspace(geometry(4), Some(128), &context)
                .unwrap()
                .peak()
                .bytes(),
            None
        );
        return;
    };
    let context = WorkspaceContext::new(Facts {
        host: true,
        tensor: true,
    });
    for chunk in [1, 4, 11] {
        let g = geometry(chunk);
        let identity_peak =
            crate::input::TextInputIdentityPlan::new(g.batch_size, g.input_positions)
                .unwrap()
                .peak_bytes();
        let report = quote_text_prompt_workspace(g, Some(128), &context).unwrap();
        assert_eq!(report.tensor_peak_bytes(), Some(11 * 4 + 11));
        assert_eq!(
            report.host_peak_bytes(),
            Some(128 + 13 + identity_peak + identity_controls)
        );
        assert_eq!(
            report.peak().bytes(),
            Some(196 + identity_peak + identity_controls)
        );
        let combined = report.compose(outside(g)).unwrap();
        assert_eq!(
            combined.peak_bytes().unwrap(),
            Some(196 + identity_peak + identity_controls)
        );
        assert_eq!(
            context.report(&[]).unwrap().operations[0].outputs[0].shape(),
            [1, 11]
        );
    }
}

#[test]
fn preparation_preserves_missing_backing_mechanisms_and_enclosing_bounds() {
    for (host, tensor, capacity) in [
        (false, true, Some(44)),
        (true, false, Some(44)),
        (true, true, None),
    ] {
        let context = WorkspaceContext::new(Facts { host, tensor });
        let report = quote_text_prompt_workspace(geometry(4), capacity, &context).unwrap();
        assert_eq!(report.peak().bytes(), None);
        assert_eq!(
            report
                .compose(outside(geometry(4)))
                .unwrap()
                .peak_bytes()
                .unwrap(),
            None
        );
    }
    let context = WorkspaceContext::new(Facts {
        host: true,
        tensor: true,
    });
    let report = quote_text_prompt_workspace(geometry(4), Some(44), &context).unwrap();
    let mut missing = outside(geometry(4));
    missing.materialization = WorkspaceBound::Unknown {
        reason: "unpriced weight staging".into(),
    };
    assert_eq!(report.compose(missing).unwrap().peak_bytes().unwrap(), None);
}

#[test]
fn preparation_rejects_invalid_capacity_extent_and_request_composition() {
    let context = WorkspaceContext::new(Facts {
        host: true,
        tensor: true,
    });
    assert!(quote_text_prompt_workspace(geometry(4), Some(43), &context).is_err());
    assert!(quote_text_prompt_workspace(
        InferenceGeometry {
            input_positions: i32::MAX as u64 + 1,
            ..geometry(4)
        },
        None,
        &context
    )
    .is_err());
    assert!(context.report(&[]).unwrap().operations.is_empty());
    let report = quote_text_prompt_workspace(geometry(4), Some(44), &context).unwrap();
    assert!(report.compose(outside(geometry(3))).is_err());
    assert!(report
        .compose(outside(InferenceGeometry {
            cached_positions: 6,
            ..geometry(4)
        }))
        .is_err());
}

#[test]
fn complete_prompt_capacity_remains_reserved_when_chunk_search_and_concurrent_requests_compete() {
    let Some(identity_controls) = identity_controls() else {
        let context = WorkspaceContext::new(Facts {
            host: true,
            tensor: true,
        });
        assert_eq!(
            quote_text_prompt_workspace(geometry(4), Some(128), &context)
                .unwrap()
                .peak()
                .bytes(),
            None
        );
        return;
    };
    use crate::working_memory::{
        plan_prefill, InferenceExecutionIdentity, PrefillPlanningError, WorkingMemoryError,
        WorkingMemoryPool,
    };
    use eredu_core::{
        AdmissionRequest, CacheStateStrategy, EstimationCompleteness, InputModalities,
        InputTokenCount, LayerSchedule, ModelCapabilities, Observed, StateMemoryLayout,
    };
    let g = geometry(4);
    let capabilities = ModelCapabilities {
        effective_model_type: "stateless preparation fixture".into(),
        native_max_context: Observed::exact(64, "fixture"),
        effective_max_context: Observed::exact(64, "fixture"),
        state_strategy: CacheStateStrategy::FullKv,
        modalities: InputModalities::TEXT,
        estimation: EstimationCompleteness::Complete,
    };
    let request = AdmissionRequest {
        input: InputTokenCount::text(g.cached_positions + g.input_positions),
        max_output_tokens: g.max_output_tokens,
        batch_size: g.batch_size,
        safety_reserve_bytes: 0,
        application_memory_budget_bytes: None,
        require_complete_estimate: true,
    };
    let layout = StateMemoryLayout::new(
        LayerSchedule::empty(),
        vec![],
        1,
        1,
        EstimationCompleteness::Complete,
    )
    .unwrap();
    let quote = |g, capacity| {
        let state = eredu_core::estimate_runtime_state(
            &layout,
            request.input,
            request.max_output_tokens,
            1,
            std::num::NonZeroU8::new(4).unwrap(),
        )
        .unwrap();
        let prompt = quote_text_prompt_workspace(
            g,
            capacity,
            &WorkspaceContext::new(Facts {
                host: true,
                tensor: true,
            }),
        )
        .unwrap();
        state.with_execution_workspace(prompt.compose(outside(g))?)
    };
    let execution = InferenceExecutionIdentity::default();
    let identity_peak = crate::input::TextInputIdentityPlan::new(g.batch_size, g.input_positions)
        .unwrap()
        .peak_bytes();
    let pool = WorkingMemoryPool::new(196 + identity_peak + identity_controls, 0).unwrap();
    let (_, first) = plan_prefill(&execution, &pool, &capabilities, request, g, |g| {
        quote(g, Some(128))
    })
    .unwrap();
    assert_eq!(
        pool.used_bytes().unwrap(),
        196 + identity_peak + identity_controls
    );
    let mut attempted_chunks = Vec::new();
    let second = plan_prefill(&execution, &pool, &capabilities, request, g, |g| {
        attempted_chunks.push(g.prefill_chunk_positions);
        quote(g, Some(44))
    });
    assert!(matches!(
        second,
        Err(PrefillPlanningError::Reservation(
            WorkingMemoryError::BudgetExceeded { .. }
        ))
    ));
    assert_eq!(attempted_chunks, [4, 3, 2, 1]);
    assert_eq!(
        pool.used_bytes().unwrap(),
        196 + identity_peak + identity_controls
    );
    drop(first);
    assert_eq!(pool.used_bytes().unwrap(), 0);
    let (_, second) = plan_prefill(&execution, &pool, &capabilities, request, g, |g| {
        quote(g, Some(44))
    })
    .unwrap();
    assert_eq!(
        pool.used_bytes().unwrap(),
        112 + identity_peak + identity_controls
    );
    assert!(
        plan_prefill(&execution, &pool, &capabilities, request, g, |g| quote(
            g, None
        ))
        .is_err()
    );
    assert_eq!(
        pool.used_bytes().unwrap(),
        112 + identity_peak + identity_controls
    );
    drop(second);
    assert_eq!(pool.used_bytes().unwrap(), 0);
    assert_eq!(
        pool.peak_bytes().unwrap(),
        196 + identity_peak + identity_controls
    );
}

#[test]
fn original_token_input_report_preserves_native_identity_and_staging_without_legacy_capacity() {
    if identity_controls().is_none() {
        let context = WorkspaceContext::new(Facts {
            host: true,
            tensor: true,
        });
        assert_eq!(
            quote_text_prompt_workspace(geometry(4), Some(128), &context)
                .unwrap()
                .host_peak_bytes(),
            None
        );
        return;
    }
    let ids = [7; 11];
    let input = crate::working_memory::OriginalTokenInputLayout::prepare(
        &eredu_core::TokenIdsInputPlan::new(&ids).unwrap(),
    )
    .unwrap();
    for chunk in [1, 4, 11] {
        let context = WorkspaceContext::new(Facts {
            host: true,
            tensor: true,
        });
        let original =
            quote_original_token_prompt_workspace(geometry(chunk), &input, &context).unwrap();
        let legacy = quote_text_prompt_workspace(geometry(chunk), Some(128), &context).unwrap();
        assert_eq!(original.tensor_peak_bytes(), legacy.tensor_peak_bytes());
        assert_eq!(
            original.host_peak_bytes().unwrap() + 128,
            legacy.host_peak_bytes().unwrap()
        );
        assert_eq!(
            original.peak().bytes().unwrap() + 128,
            legacy.peak().bytes().unwrap()
        );
        assert_eq!(
            context.report(&[]).unwrap().operations[0].outputs[0].shape(),
            [1, 11]
        );
    }
    for (host, tensor) in [(false, true), (true, false)] {
        let context = WorkspaceContext::new(Facts { host, tensor });
        assert_eq!(
            quote_original_token_prompt_workspace(geometry(4), &input, &context)
                .unwrap()
                .peak()
                .bytes(),
            None
        );
    }
    let context = WorkspaceContext::new(Facts {
        host: true,
        tensor: true,
    });
    assert!(quote_original_token_prompt_workspace(
        InferenceGeometry {
            input_positions: 12,
            ..geometry(4)
        },
        &input,
        &context
    )
    .is_err());
}
