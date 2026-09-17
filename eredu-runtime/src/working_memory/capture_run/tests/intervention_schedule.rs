use super::*;
use eredu_core::intervention::*;

fn sources() -> (SharedCapturePlan, AdmittedInterventionPlan) { sources_with_rows(false) }
fn sources_with_rows(rows:bool) -> (SharedCapturePlan, AdmittedInterventionPlan) {
    sources_with_evidence(rows,InterventionEvidence::None)
}
fn sources_with_evidence(rows:bool,evidence:InterventionEvidence) -> (SharedCapturePlan, AdmittedInterventionPlan) {
    let mut raw = raw();
    raw.selections.clear();
    let capture = SharedCapturePlan::new(
        raw.admit_with_text_origin(
            &ObservationCatalog {
                schema_version: 1,
                points: vec![],
                completeness: DescriptionCompleteness::Complete,
            },
            &ObservationSupportReport {
                schema_version: 1,
                capture: Default::default(),
                points: vec![],
            },
            &CaptureCapabilities {
                transformations: vec![],
                max_histogram_bins: 0,
                physical_native_limit: false,
                conditions: vec![],
            },
            CaptureRequestShape {
                batch: 1,
                prompt_tokens: 3,
                max_predictions: 4,
            },
            CaptureTextOrigin {
                cached_positions: 0,
            },
        )
        .unwrap(),
    );
    let mut discovery = InterventionDiscovery {
        schema_version: INTERVENTION_SCHEMA_VERSION,
        artifact_identity: "scheduled-original".into(),
        session_identity: Some("session".into()),
        points: vec![InterventionPoint {
            path: MODEL_LOGITS_OBSERVATION_PATH.into(),
            node_id: "output".into(),
            stage: InterventionStage::LogitsBeforeSampling,
            axes: vec![
                TensorAxis {
                    name: "batch".into(),
                    dimension: SymbolicDimension::Batch,
                },
                TensorAxis {
                    name: "vocabulary".into(),
                    dimension: SymbolicDimension::Known(4),
                },
            ],
            dtypes: vec![InterventionDtype::Float32],
            operations: vec![InterventionKind::Scale],
            score_stages: vec![],
            prefill: ObservationSupportStatus::Supported,
            decode: ObservationSupportStatus::Supported,
            conditions: vec![],
            routing: None,
            routed_units: None,
        }],
    };
    if rows {discovery.points[0].axes.insert(1,TensorAxis{name:"sequence".into(),dimension:SymbolicDimension::Sequence});}
    let mut edits = InterventionPlan {
        schema_version: INTERVENTION_SCHEMA_VERSION,
        operations: vec![InterventionOperation {
            id: "alternate".into(),
            target: MODEL_LOGITS_OBSERVATION_PATH.into(),
            schedule: CaptureSchedule {
                prefill: false,
                first_prediction: 1,
                every: 2,
                ..Default::default()
            },
            slices: vec![],
            action: InterventionAction::Scale {
                dtype: InterventionDtype::Float32,
                factor: 2.0,
            },
            evidence,
        }],
    };
    if rows {
        edits.operations[0].schedule=CaptureSchedule::default();
        let mut second=edits.operations[0].clone();second.id="second".into();
        edits.operations.push(second);
    }
    let edits=edits.admit(&discovery,capture.admission().request(),"session").unwrap();
    (capture, edits)
}

#[test]
fn intervention_only_schedule_prices_every_outcome_and_authenticates_actual_native_account() {
    let (capture, admitted) = sources();
    let pool = WorkingMemoryPool::new(1 << 24, 0).unwrap();
    let source = pool
        .compile_intervention_source(PreparedInterventionPlanCopy::inspect(&admitted).unwrap())
        .unwrap();
    let prepared = || {
        CaptureRunHostPlan::prepare(&capture)
            .unwrap()
            .with_interventions(&source)
            .unwrap()
    };
    assert_eq!(
        CaptureRunHostPlan::prepare(&capture).unwrap().claim_slots(),
        0
    );
    let h = prepared().initialization_peak_bytes();
    let (short_reservation, short_run) = fresh(&pool, h - 1);
    let before = ledger(&pool);
    let allocations = CLAIM_ALLOCATIONS.get();
    assert!(matches!(
        short_run.prepare_capture_run(&short_reservation, prepared()),
        Err(CaptureRunHostError::Memory(
            WorkingMemoryError::BudgetExceeded { .. }
        ))
    ));
    assert_eq!(ledger(&pool), before);
    assert_eq!(CLAIM_ALLOCATIONS.get(), allocations);
    drop((short_reservation, short_run));

    let other_pool = WorkingMemoryPool::new(1 << 24, 0).unwrap();
    let foreign = other_pool
        .compile_intervention_source(PreparedInterventionPlanCopy::inspect(&admitted).unwrap())
        .unwrap();
    let (reservation, run) = fresh(&pool, h);
    let before = ledger(&pool);
    assert!(matches!(
        run.prepare_capture_run(
            &reservation,
            CaptureRunHostPlan::prepare(&capture)
                .unwrap()
                .with_interventions(&foreign)
                .unwrap()
        ),
        Err(CaptureRunHostError::Memory(
            WorkingMemoryError::IdentityMismatch
        ))
    ));
    assert_eq!(ledger(&pool), before);
    assert_eq!(CLAIM_ALLOCATIONS.get(), allocations);
    let native = run.scope().unwrap();
    let (foreign_reservation, foreign_run) = fresh(&other_pool, h);
    let foreign_native = foreign_run.scope().unwrap();
    let mut bank = run.prepare_capture_run(&reservation, prepared()).unwrap();
    let mut frames = Vec::new();
    for prediction in 0..4 {
        let mut step = bank
            .begin_step(plan::phase(prediction), prediction as u64)
            .unwrap()
            .prepare()
            .unwrap();
        assert_eq!(step.interventions().len(), 1);
        if prediction % 2 == 1 {
            let claim = step.take_intervention(0).unwrap();
            claim.validate_source(&source).unwrap();
            claim.validate_native_custody(&native).unwrap();
            assert!(matches!(
                claim.validate_native_custody(&foreign_native),
                Err(WorkingMemoryError::IdentityMismatch)
            ));
            assert!(step.take_intervention(0).is_err());
            step.record_intervention(claim.finish(CaptureUsage::default()).unwrap())
                .unwrap();
            assert_eq!(
                step.interventions()[0].outcome,
                InterventionOutcome::Applied
            );
        } else {
            assert_eq!(
                step.interventions()[0].outcome,
                InterventionOutcome::Inactive
            );
            assert!(step.take_intervention(0).is_err());
        }
        frames.push(finish(step));
    }
    drop((bank, reservation, run, source));
    native.certify().unwrap();
    assert!(
        pool.used_bytes().unwrap() > 0,
        "escaped outcome frames keep their original host hold"
    );
    drop(frames);
    assert_eq!(pool.used_bytes().unwrap(), 0);
    foreign_native.certify().unwrap();
    drop((foreign, foreign_reservation, foreign_run));
    assert_eq!(other_pool.used_bytes().unwrap(), 0);
}


#[test]
fn prefill_interventions_keep_one_claim_and_ledger_across_uneven_chunks_and_poison_abandoned_work() {
    use crate::intervention::InterventionPrefillWindow as Window;
    use crate::prefill::PrefillChunk;
    let (capture,admitted)=sources_with_rows(true);
    let pool=WorkingMemoryPool::new(1<<26,0).unwrap();
    let source=pool.compile_intervention_source(PreparedInterventionPlanCopy::inspect(&admitted).unwrap()).unwrap();
    let host=||CaptureRunHostPlan::prepare(&capture).unwrap().with_interventions(&source).unwrap();
    let h=host().initialization_peak_bytes();
    let geometry=InferenceGeometry{batch_size:1,cached_positions:0,input_positions:3,max_output_tokens:4,
        prefill_chunk_positions:2,output:OutputDemand::Sequence};
    let chunks=[PrefillChunk{input:0..2,position:0,output:OutputDemand::Sequence},
        PrefillChunk{input:2..3,position:2,output:OutputDemand::Sequence}];
    let windows=chunks.each_ref().map(|chunk|Window::new(source.plan().admission(),geometry,chunk).unwrap());
    let (reservation,run)=fresh(&pool,h);
    let native=run.scope().unwrap();
    let mut bank=run.prepare_capture_run(&reservation,host()).unwrap();
    let mut step=bank.begin_step(CapturePhase::Prefill,0).unwrap().prepare_prefill(geometry).unwrap();
    assert!(step.records().is_empty());
    assert_eq!(step.interventions().len(),2);
    let record_metadata: Vec<_> = step.interventions().iter().map(|record|record.charged).collect();
    let mut ledger=CaptureLedger::new(capture.admission());ledger.begin_step();
    source.reserve_capture_metadata(&mut ledger).unwrap();
    let initial=ledger.total();
    let usage=CaptureUsage{host_bytes:7,retained_bytes:11,..CaptureUsage::default()};
    for (chunk,window) in windows.into_iter().enumerate() {
        assert!(step.validate_prefill_intervention_end(window).is_err());
        if chunk==0 {assert!(step.take_prefill_intervention(1,window).is_err());}
        for index in 0..2 {
            let mut cursor=step.take_prefill_intervention(index,window).unwrap();
            let mut fragment=cursor.begin(window).unwrap();
            fragment.claim().validate_native_custody(&native).unwrap();
            fragment.claim().validate_source(&source).unwrap();
            assert_eq!(fragment.claim().coordinate(),(CapturePhase::Prefill,0));
            assert_eq!(fragment.source(),window);
            crate::intervention::reserve_envelope(&mut ledger,usage).unwrap();
            fragment.charge(usage).unwrap();
            fragment.finish().unwrap();
            step.retain_prefill_intervention(cursor).unwrap();
            assert!(step.take_intervention(index).is_err(),"original claim cannot be reissued");
            assert_eq!(step.interventions()[index].outcome,if window.is_final(){InterventionOutcome::Applied}else{InterventionOutcome::Missing});
        }
        step.validate_prefill_intervention_end(window).unwrap();
        step.complete_prefill_chunk(chunk as u64).unwrap();
    }
    step.finish_prefill_targets().unwrap();
    let twice=usage.checked_add(usage).unwrap();
    for (record,metadata) in step.interventions().iter().zip(record_metadata) {
        assert_eq!(record.charged,metadata.checked_add(twice).unwrap());
    }
    assert_eq!(ledger.total(),initial.checked_add(twice).unwrap().checked_add(twice).unwrap());
    let escaped=finish(step);
    drop((bank,reservation,run));native.certify().unwrap();
    assert!(pool.used_bytes().unwrap()>0);
    drop(escaped);

    let (reservation,run)=fresh(&pool,h);
    let native=run.scope().unwrap();
    let mut bank=run.prepare_capture_run(&reservation,host()).unwrap();
    let mut step=bank.begin_step(CapturePhase::Prefill,0).unwrap().prepare_prefill(geometry).unwrap();
    let mut cursor=step.take_prefill_intervention(0,windows[0]).unwrap();
    let mut fragment=cursor.begin(windows[0]).unwrap();
    crate::intervention::reserve_envelope(&mut ledger,usage).unwrap();
    fragment.charge(usage).unwrap();
    drop(fragment);
    let spent=ledger.total();
    assert!(cursor.begin(windows[0]).is_err());
    assert!(cursor.begin(windows[1]).is_err());
    let charged=cursor.charged();drop(cursor);
    assert!(step.take_prefill_intervention(0,windows[0]).is_err());
    assert!(step.validate_prefill_intervention_end(windows[0]).is_err());
    step.record_intervention_failure(0,"failed physical fragment",charged).unwrap();
    assert_eq!(ledger.total(),spent);
    drop(step);drop((bank,reservation,run));native.certify().unwrap();
    drop(source);
    assert_eq!(pool.used_bytes().unwrap(),0);
}

mod partition;

mod evidence_frame;
