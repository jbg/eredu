use super::*;
use crate::capture::OriginalSpeculativeCapture;
use eredu_core::speculative::{SpeculativeActivationOrigin, SpeculativeCaptureScope};

#[test]
fn original_outer_source_preserves_scope_occurrence_and_shared_role_delivery() {
    let selected = selected(SpeculativeStrategyClass::EmbeddedSequential);
    let schedule = plan(&selected, 3, true);
    let (invocation, _) = schedule.prefill_invocations(0).unwrap();
    let workspace = EmbeddedInvocationWorkspace::target(invocation).unwrap();
    let report = report(workspace.geometry());
    let capacity = 1 << 26;
    let pool = crate::working_memory::memory_fixture::host_ledger(capacity, 0).unwrap();
    let execution = InferenceExecutionIdentity::default();
    let source = capture_source(&pool);
    let funding = pool
        .prepare_workspace_metadata(
            &execution,
            crate::working_memory::memory_fixture::resolved_host_limits(&pool, capacity),
        )
        .unwrap();
    let request = OriginalSpeculativeRequest::prepare_embedded(
        &pool,
        &execution,
        &schedule,
        crate::working_memory::memory_fixture::resolved_host_limits(&pool, capacity),
    )
    .unwrap();
    let mut cursor = schedule.into_cursor();
    let origin = SpeculativeActivationOrigin {
        request: SpeculativeRequestId::new(71),
        committed_tokens: 0,
        prediction: 0,
        prefix_digest: [0; 32],
        optimistic: false,
    };
    let mut outer = OriginalSpeculativeCapture::prepare(
        source.clone(),
        &[SpeculativeCaptureScope::Target],
        "actual-loaded-capture",
        origin.request,
        funding.clone(),
    )
    .unwrap();
    assert!(outer.preview().is_none());
    outer.set_origin(Some(origin));
    let preview = outer.preview().unwrap();
    assert!(
        preview
            .prepare_invocation(Phase::TargetPrefill, 0, None, 0)
            .is_err()
    );
    assert!(
        preview
            .prepare_invocation(Phase::TargetPrefill, 3, None, u64::MAX)
            .is_err()
    );
    let prospective = preview
        .prepare_invocation(Phase::TargetPrefill, 3, None, 0)
        .unwrap();
    let future_draft = preview
        .prepare_invocation(Phase::PredictionPrefill, 3, None, 1)
        .unwrap();
    assert!(prospective.invocation().source().same_source(&source));
    assert_eq!(prospective.invocation().selected(), &[true]);
    assert_eq!(future_draft.invocation().selected(), &[false]);
    assert_eq!(future_draft.invocation().invocation(), 1);
    assert!(outer.invocation().is_none());
    assert!(outer.take().is_none());
    // Cold prospect construction neither begins nor spends this occurrence.
    outer.begin(Phase::TargetPrefill, 3).unwrap();
    assert!(outer.preview().is_none());
    assert_eq!(
        outer.invocation().unwrap().invocation(),
        prospective.invocation().invocation()
    );
    drop(future_draft);
    drop(prospective);
    let descriptor = outer.invocation().unwrap();
    assert!(descriptor.source().same_source(&source));
    assert_eq!(descriptor.selected(), &[true]);
    assert_eq!(descriptor.invocation(), 0);
    assert!(descriptor.requires_sequence_readout().unwrap());
    let mut required = requirements(report.span_workspace_plan());
    required.controls = required
        .controls
        .checked_add(descriptor.envelope_control_bytes().unwrap() as u64)
        .unwrap();
    let host = EmbeddedCaptureHostPlan::prepare(
        descriptor.source(),
        CaptureRunHostPlan::prepare_invocation(
            descriptor.source().plan(),
            descriptor.capture_phase(),
            0,
            CaptureInvocationShape {
                batch: 1,
                sequence: 3,
                context: None,
            },
            descriptor.selected(),
        )
        .unwrap(),
        workspace,
        descriptor.origin(),
    )
    .unwrap();
    let (role, pending) = request
        .reserve_embedded_role_with_capture(
            cursor.claim(invocation).unwrap(),
            workspace,
            required,
            host,
        )
        .unwrap();
    // This exact identity String was counted in the accepted model role above.
    let identity = descriptor.admission_identity().to_owned();
    let mut owner = pending.begin().unwrap();
    owner.prepare_envelope(descriptor).unwrap();
    assert!(owner.prepare_envelope(descriptor).is_err());
    let mut backend = Backend {
        custody: role.budget_custody(),
        calls: 0,
    };
    forward(&mut owner, &mut backend, 1).unwrap();
    let usage = owner.usage();
    let envelope_usage = descriptor.envelope_usage().unwrap();
    assert!(usage.encoded_bytes >= 512 + envelope_usage.encoded_bytes);
    assert!(usage.host_bytes >= 24 + envelope_usage.host_bytes);
    let shared = owner.take_shared_step().unwrap().unwrap();
    assert_eq!(owner.usage(), usage);
    let alias = shared.clone();
    let envelope = descriptor.envelope(identity, shared, true).unwrap();
    outer.receive(envelope).unwrap();
    outer.complete().unwrap();
    outer.finish(true);
    outer.begin(Phase::PredictionPrefill, 3).unwrap();
    let second = outer.invocation().unwrap();
    assert_eq!(second.invocation(), 1);
    assert_eq!(second.selected(), &[false]);
    assert!(!second.requires_sequence_readout().unwrap());
    outer.finish(false);
    outer.begin(Phase::TargetPrefill, 3).unwrap();
    assert_eq!(outer.invocation().unwrap().invocation(), 2);
    outer.finish(false);
    let delivered = outer.take().unwrap();
    assert!(delivered.completed);
    assert!(delivered.captures.same_storage(&alias));
    assert!(outer.take().is_none());
    request.close().unwrap();
    drop(outer);
    drop(funding);
    drop(source);
    drop(owner);
    drop(backend);
    drop(role);
    drop(request);
    drop(alias);
    assert!(pool.payload_used_bytes().unwrap() > 0);
    let CapturePayload::SharedTensor(value) = delivered.captures.as_step().records[0]
        .payload
        .as_ref()
        .unwrap()
    else {
        panic!("actual shared model frame")
    };
    let TensorObservationData::F32(values) = value.data() else {
        panic!("F32")
    };
    assert_eq!(values, &[0.5, -1.0, 2.25, 3.5, -4.0, 5.75]);
    drop(delivered);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}

#[test]
fn original_window_aggregate_uses_same_prefix_and_escapes_with_paid_report() {
    window_aggregate(false);
}
#[test]
fn original_internal_checkpoint_preserves_spending_drain_and_escaped_custody() {
    window_aggregate(true);
}
fn window_aggregate(checkpoints: bool) {
    struct WindowBackend {
        custody: OriginalSpeculativeBudgetCustody,
        calls: usize,
    }
    impl ScheduledCaptureBackend for WindowBackend {
        type Tensor = Vec<f32>;
        type Error = Failure;
        fn validate_source(
            &self,
            values: &Vec<f32>,
            geometry: &CaptureTensorGeometry<'_>,
        ) -> Result<eredu_core::checkpoint::TensorDtype, Failure> {
            assert_eq!(geometry.source_shape(), &[values.len() / 2, 2]);
            Ok(eredu_core::checkpoint::TensorDtype::F32)
        }
        fn estimate(
            &self,
            _: &Vec<f32>,
            geometry: &CaptureTensorGeometry<'_>,
        ) -> Result<CaptureUsage, CaptureError> {
            Ok(CaptureUsage {
                captures: 1,
                retained_bytes: geometry.elements() as u64 * 4,
                host_bytes: geometry.elements() as u64 * 4,
                encoded_bytes: 512,
            })
        }
        fn transform(
            &mut self,
            values: &Vec<f32>,
            claim: CaptureTensorClaim<'_, '_>,
        ) -> Result<ClaimedCaptureTensor, Failure> {
            claim.validate_model_custody(&self.custody)?;
            self.calls += 1;
            let mut output = claim.prepare()?;
            for &value in values.iter().take(output.len()) {
                output.push_f32(value).unwrap();
            }
            Ok(output.finish().unwrap())
        }
    }
    let selected = selected(SpeculativeStrategyClass::EmbeddedSequential);
    let schedule = plan(&selected, 3, true);
    let invocations: Vec<_> = (0..3)
        .map(|i| schedule.prefill_invocations(i).unwrap().0)
        .collect();
    let capacity = 1 << 27;
    let pool = crate::working_memory::memory_fixture::host_ledger(capacity, 0).unwrap();
    let execution = InferenceExecutionIdentity::default();
    let source = capture_source_with_transform(
        &pool,
        7,
        CaptureTransform::Preview { max_elements: 5 },
        u64::MAX,
    );
    let funding = pool
        .prepare_workspace_metadata(
            &execution,
            crate::working_memory::memory_fixture::resolved_host_limits(&pool, capacity),
        )
        .unwrap();
    let request = OriginalSpeculativeRequest::prepare_embedded(
        &pool,
        &execution,
        &schedule,
        crate::working_memory::memory_fixture::resolved_host_limits(&pool, capacity),
    )
    .unwrap();
    let lineage = request.prepare_embedded_capture_lineage(&source).unwrap();
    let mut cursor = schedule.into_cursor();
    let origin = SpeculativeActivationOrigin {
        request: SpeculativeRequestId::new(97),
        committed_tokens: 0,
        prediction: 0,
        prefix_digest: [0; 32],
        optimistic: false,
    };
    let mut outer = OriginalSpeculativeCapture::prepare(
        source.clone(),
        &[SpeculativeCaptureScope::Target],
        "actual-window-source",
        origin.request,
        funding.clone(),
    )
    .unwrap()
    .with_lineage(lineage.clone())
    .unwrap();
    let saved = if checkpoints {
        assert!(outer.control_storage_bytes().unwrap() > 0);
        let saved = outer.save_control().unwrap();
        let mut foreign = OriginalSpeculativeCapture::prepare(
            source.clone(),
            &[SpeculativeCaptureScope::Target],
            "actual-window-source",
            origin.request,
            funding.clone(),
        )
        .unwrap()
        .with_lineage(lineage.clone())
        .unwrap();
        assert!(
            foreign.prepare_control(&saved).is_err(),
            "same C and ledger do not substitute the collector"
        );
        Some(saved)
    } else {
        None
    };
    outer.set_origin(Some(origin));
    outer.set_prefill_reduction_geometry(
        eredu_core::speculative::SpeculativePrefillReductionGeometry {
            target_sequence: 7,
            prediction_sequence: 7,
        },
    );
    let mut physical = Vec::new();
    let mut spent = CaptureUsage::default();
    for (i, invocation) in invocations.into_iter().enumerate() {
        outer.set_prefill_span(invocation.prefill_span());
        outer
            .begin(Phase::TargetPrefill, invocation.positions())
            .unwrap();
        if checkpoints {
            assert!(outer.save_control().is_err());
        }
        let descriptor = outer.invocation().unwrap();
        let inherited = request
            .inspect_embedded_capture_lineage_usage(&source, &lineage)
            .unwrap();
        assert!(inherited.host_bytes > spent.host_bytes);
        let prefix = descriptor
            .prepared_prefix()
            .unwrap()
            .resume(source.plan(), inherited)
            .unwrap();
        assert!(prefix.step().host_bytes > 0);
        let workspace = EmbeddedInvocationWorkspace::target(invocation).unwrap();
        let report = report(workspace.geometry());
        let mut required = requirements(report.span_workspace_plan());
        required.controls += descriptor.envelope_control_bytes().unwrap() as u64;
        let host = EmbeddedCaptureHostPlan::prepare(
            &source,
            CaptureRunHostPlan::prepare_invocation_window(
                source.plan(),
                CapturePhase::Prefill,
                0,
                CaptureInvocationShape {
                    batch: 1,
                    sequence: invocation.positions() as u64,
                    context: None,
                },
                descriptor.selected(),
                descriptor.window().unwrap(),
            )
            .unwrap()
            .with_skip_reasons(descriptor.skip_reasons())
            .unwrap(),
            workspace,
            origin,
        )
        .unwrap()
        .with_lineage(&lineage)
        .unwrap()
        .with_quoted_usage(inherited);
        let (role, pending) = request
            .reserve_embedded_role_with_capture(
                cursor.claim(invocation).unwrap(),
                workspace,
                required,
                host,
            )
            .unwrap();
        let identity = descriptor.admission_identity().to_owned();
        let mut owner = pending.begin().unwrap();
        owner.prepare_envelope(descriptor).unwrap();
        let start = invocation.prefill_span().unwrap().input_start as usize;
        let values: Vec<f32> = (start * 2..(start + invocation.positions()) * 2)
            .map(|n| (n as f32 - 3.0) * 0.375)
            .collect();
        let policy = crate::capture::CaptureObservationStep::with_invocation(
            source.plan().admission(),
            CapturePhase::Prefill,
            0,
            Some(CaptureInvocationShape {
                batch: 1,
                sequence: invocation.positions() as u64,
                context: None,
            }),
        )
        .unwrap()
        .with_window(descriptor.window().unwrap())
        .unwrap();
        let expected_usage = inherited
            .checked_add(policy.window_metadata_usage(0).unwrap().unwrap())
            .unwrap()
            .checked_add(CaptureUsage {
                captures: 1,
                retained_bytes: values.len().min(5) as u64 * 4,
                host_bytes: values.len().min(5) as u64 * 4,
                encoded_bytes: 512,
            })
            .unwrap();
        let mut backend = WindowBackend {
            custody: role.budget_custody(),
            calls: 0,
        };
        let epoch = eredu_core::DistributedCommitEpoch::new(i as u64 + 61).unwrap();
        owner
            .with_observer(&mut backend, &|cause| cause, |observer| {
                let guard = crate::inspection::ObservationTransactionGuard::new(observer, epoch);
                guard
                    .observer
                    .prepare_transaction(epoch, crate::ExpertPass::Prefill)?;
                guard.observer.observe("block.output", &values)?;
                guard.observer.complete_transaction(epoch)?;
                guard.finish(true);
                Ok::<_, FundedCaptureError<Failure>>(())
            })
            .unwrap()
            .unwrap();
        assert_eq!(backend.calls, 1);
        spent = owner.usage();
        assert_eq!(
            spent, expected_usage,
            "the source-paid prefix must not be charged again"
        );
        let frame = owner.take_shared_step().unwrap().unwrap();
        outer
            .receive(descriptor.envelope(identity, frame, true).unwrap())
            .unwrap();
        outer.complete().unwrap();
        outer.finish(true);
        if let Some(delivered) = outer.take() {
            physical.push(delivered);
        }
        assert_eq!(
            request
                .inspect_embedded_capture_lineage_usage(&source, &lineage)
                .unwrap(),
            spent
        );
    }
    assert_eq!(physical.len(), 2);
    assert!(
        physical
            .iter()
            .all(|frame| frame.prefill_reductions.is_none())
    );
    if checkpoints {
        assert!(
            outer.save_control().is_err(),
            "held aggregate is not drained"
        );
    }
    outer.complete_prefill_reductions().unwrap();
    outer.finish_prefill_reductions(true);
    if checkpoints {
        assert!(outer.save_control().is_err(), "delivery is still queued");
    }
    let final_frame = outer.take().unwrap();
    if let Some(saved) = &saved {
        assert_eq!(
            outer.control_usage().unwrap(),
            spent,
            "control reads the current request lineage"
        );
        {
            let active = lineage.ledger().borrow().unwrap();
            assert!(
                outer.control_usage().is_err(),
                "an active ledger cannot supply settled readmission usage"
            );
            drop(active);
        }
        drop(outer.prepare_control(saved).unwrap());
        outer.prepare_control(saved).unwrap().commit();
        assert_eq!(
            request
                .inspect_embedded_capture_lineage_usage(&source, &lineage)
                .unwrap(),
            spent
        );
        let post = outer.save_control().unwrap();
        outer.prepare_control(&post).unwrap().commit();
        outer.set_prefill_span(None);
        outer.begin(Phase::Verification, 1).unwrap();
        assert_eq!(
            outer.invocation().unwrap().invocation(),
            3,
            "restore cannot recycle physical IDs"
        );
        outer.finish(false);
        assert!(outer.save_control().is_err());
        assert!(
            outer.prepare_control(saved).is_err(),
            "failed callback cannot become checkpoint-ready by restore"
        );
        assert!(
            outer.control_usage().is_err(),
            "failed callbacks cannot supply readmission usage"
        );
        spent = request
            .inspect_embedded_capture_lineage_usage(&source, &lineage)
            .unwrap();
    }
    let report = final_frame.prefill_reductions.as_ref().unwrap().clone();
    let entry = &report.as_reductions().records[0];
    assert_eq!(
        entry.status,
        eredu_core::speculative::SpeculativePrefillReductionStatus::Complete
    );
    assert_eq!(
        (
            entry.windows,
            entry.covered_sequence,
            entry.logical_sequence
        ),
        (3, 7, 7)
    );
    let payload = entry.record.payload.as_ref().unwrap().as_tensor().unwrap();
    let TensorObservationData::F32(values) = payload.data() else {
        panic!("F32 Preview")
    };
    assert_eq!(values, &[-1.125, -0.75, -0.375, 0.0, 0.375]);
    assert_eq!(
        request
            .inspect_embedded_capture_lineage_usage(&source, &lineage)
            .unwrap(),
        spent
    );
    request.close().unwrap();
    drop(outer);
    drop(physical);
    drop(final_frame);
    drop(funding);
    drop(source);
    drop(lineage);
    drop(request);
    assert!(pool.payload_used_bytes().unwrap() > 0);
    assert_eq!(report.as_reductions().records[0].covered_sequence, 7);
    drop(report);
    if saved.is_some() {
        assert!(
            pool.payload_used_bytes().unwrap() > 0,
            "escaped checkpoint retains actual C and host accounts"
        );
    }
    drop(saved);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}
