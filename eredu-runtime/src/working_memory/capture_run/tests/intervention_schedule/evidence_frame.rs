use super::*;

#[test]
fn partition_evidence_lends_original_frame_and_preserves_spent_sides() {
    let (capture,admitted)=sources_with_evidence(false,InterventionEvidence::Preview{max_elements:4});
    let pool=WorkingMemoryPool::new(1<<25,0).unwrap();
    let source=pool.compile_intervention_source(PreparedInterventionPlanCopy::inspect(&admitted).unwrap()).unwrap();
    let planned=CaptureRunHostPlan::prepare(&capture).unwrap().with_interventions(&source).unwrap();
    let (reservation,run)=fresh(&pool,planned.initialization_peak_bytes());
    let mut bank=run.prepare_capture_run(&reservation,planned).unwrap();
    drop(bank.begin_step(CapturePhase::Prefill,0).unwrap().prepare().unwrap());
    let mut frame=bank.begin_step(CapturePhase::Decode,1).unwrap().prepare().unwrap();
    let claim=frame.take_intervention(0).unwrap();
    frame.record_intervention(claim.finish(CaptureUsage::default()).unwrap()).unwrap();
    let before=ledger(&pool);
    let mut loan=frame.take_partition_intervention_evidence(0).unwrap();
    assert!(loan.source().same_storage(source.plan().evidence(0).unwrap().shared_geometry_source()));
    assert_eq!(loan.frame().records().len(),2);
    assert!(frame.take_partition_intervention_evidence(0).is_err());
    assert!(frame.take_intervention_evidence(0,InterventionEvidenceSide::Before).is_err());
    drop(loan.frame_mut().take_tensor(0).unwrap());
    frame.return_partition_intervention_evidence(loan).unwrap();
    let mut loan=frame.take_partition_intervention_evidence(0).unwrap();
    assert!(loan.frame_mut().take_tensor(0).is_err());
    assert!(loan.frame_mut().take_tensor(1).is_ok());
    assert_eq!(ledger(&pool),before,"the loan grants no new account or Host backing");
    drop(loan);
    assert!(frame.take_partition_intervention_evidence(0).is_err());
    assert!(frame.take_intervention_evidence(0,InterventionEvidenceSide::After).is_err());
    // The original slot remains present and explicitly lent/abandoned. It may
    // not be flushed into an empty list and mistaken for successful evidence.
    assert!(frame.frame.intervention_evidence[0].as_ref().unwrap().frame.is_none());
    assert_eq!(frame.interventions()[0].outcome,InterventionOutcome::Applied);
    assert!(frame.finish(CaptureStepOutcome::Committed,unlimited(),unlimited(),0.0).is_err());
}

#[test]
fn partition_evidence_prefill_reuses_original_targets_without_resetting_progression() {
    let (capture,admitted)=sources_with_evidence(true,InterventionEvidence::Preview{max_elements:4});
    let pool=WorkingMemoryPool::new(1<<25,0).unwrap();
    let source=pool.compile_intervention_source(PreparedInterventionPlanCopy::inspect(&admitted).unwrap()).unwrap();
    let planned=CaptureRunHostPlan::prepare(&capture).unwrap().with_interventions(&source).unwrap();
    let (reservation,run)=fresh(&pool,planned.initialization_peak_bytes());
    let mut bank=run.prepare_capture_run(&reservation,planned).unwrap();
    let mut frame=bank.begin_step(CapturePhase::Prefill,0).unwrap().prepare().unwrap();
    let geometry=eredu_core::InferenceGeometry {batch_size:1,input_positions:3,cached_positions:0,max_output_tokens:4,prefill_chunk_positions:2,output:eredu_core::OutputDemand::Sequence};
    let before=ledger(&pool);
    let window=crate::intervention::InterventionPrefillWindow::new(source.plan().admission(),geometry,
        &crate::prefill::PrefillChunk {input:0..2,position:0,output:geometry.output.for_chunk(false)}).unwrap();
    assert!(frame.take_prefill_intervention(0,window).is_err(),"source-only Host preparation cannot open the execution cursor");
    let mut loan=frame.take_partition_intervention_evidence(0).unwrap();
    assert_eq!(loan.frame().prefill_geometry(),None);
    loan.frame_mut().prepare_existing_prefill_with_progression(geometry).unwrap();
    let targets=loan.frame().frame.prefill.as_ref().unwrap().slots.as_ptr();
    assert!(loan.frame_mut().complete_prefill_chunk(0).is_err(),"active original evidence still requires its real callbacks");
    frame.return_partition_intervention_evidence(loan).unwrap();
    let mut loan=frame.take_partition_intervention_evidence(0).unwrap();
    loan.frame_mut().prepare_existing_prefill_with_progression(geometry).unwrap();
    assert_eq!(loan.frame().frame.prefill.as_ref().unwrap().slots.as_ptr(),targets);
    assert_eq!(loan.frame().prefill_geometry(),Some(geometry));
    assert!(loan.frame_mut().prepare_existing_prefill_with_progression(eredu_core::InferenceGeometry {prefill_chunk_positions:1,..geometry}).is_err());
    assert_eq!(ledger(&pool),before,"companion headers use the protected original Host source");
    frame.return_partition_intervention_evidence(loan).unwrap();
}


#[test]
fn partition_evidence_prefill_defers_completion_until_both_actual_producer_hooks() {
    use crate::capture::{CapturePrefillHookDecision, CapturePrefillObservationPolicy};
    let (capture,admitted)=sources_with_evidence(true,InterventionEvidence::Preview{max_elements:4});
    let pool=WorkingMemoryPool::new(1<<25,0).unwrap();
    let source=pool.compile_intervention_source(PreparedInterventionPlanCopy::inspect(&admitted).unwrap()).unwrap();
    let planned=CaptureRunHostPlan::prepare(&capture).unwrap().with_interventions(&source).unwrap();
    let (reservation,run)=fresh(&pool,planned.initialization_peak_bytes());
    let mut bank=run.prepare_capture_run(&reservation,planned).unwrap();
    let mut frame=bank.begin_step(CapturePhase::Prefill,0).unwrap().prepare().unwrap();
    let geometry=InferenceGeometry {batch_size:1,input_positions:3,cached_positions:0,max_output_tokens:4,
        prefill_chunk_positions:2,output:OutputDemand::Sequence};
    let companion=source.plan().evidence(0).unwrap().shared_geometry_source();
    let policy=CapturePrefillObservationPolicy::new(companion,geometry).unwrap();
    // Companion admission is geometry-only. The existing parent ledger owns
    // evidence quota; neither its zero limit nor its authority is replaced.
    let mut descriptive=CaptureLedger::new(companion.admission());descriptive.begin_step();
    assert!(descriptive.reserve(CaptureUsage {captures:1,..CaptureUsage::default()}).is_err());
    let mut quota=CaptureLedger::new(capture.admission());quota.begin_step();
    let hold=ledger(&pool);
    for ordinal in 0..2 {
        // Coordination can lend and restore the original target before the
        // model runs. It must leave both producer callbacks outstanding.
        let mut loan=frame.take_partition_intervention_evidence(0).unwrap();
        loan.frame_mut().prepare_existing_prefill_with_progression(geometry).unwrap();
        assert!(matches!(loan.frame_mut().complete_prefill_chunk(ordinal),
            Err(CaptureRunHostError::Prefill(CapturePrefillHostError::Progression(
                crate::capture::CapturePrefillProgressError::Missing)))));
        frame.return_partition_intervention_evidence(loan).unwrap();
        for side in 0..2 {
            let row=policy.row(side).unwrap();
            let assembly=row.assembly().unwrap();
            let fragment=assembly.fragment(ordinal).unwrap();
            let chunk=crate::prefill::PrefillChunk {input:fragment.input().clone(),
                position:fragment.position(),output:fragment.output_demand()};
            let path=&companion.admission().plan().selections[side].path;
            let mut loan=frame.take_partition_intervention_evidence(0).unwrap();
            let target=loan.frame_mut();
            let decision=target.begin_prefill_hook(side,&chunk,path).unwrap();
            if ordinal==0 {
                assert_eq!(decision,CapturePrefillHookDecision::First);
                assert!(target.reserve_prefill_hook(side,&mut quota,TensorDtype::F32,
                    CaptureUsage {captures:1,..CaptureUsage::default()}).unwrap().is_none());
            }
            let mut writer=target.take_prefill_fragment(side,&fragment).unwrap().prepare().unwrap();
            for map in fragment.mappings() {
                let before=(fragment.input().start*4+map.source_index() as u64+1) as f32;
                writer.push_f32(before*(side+1) as f32).unwrap();
            }
            writer.finish().unwrap();
            target.finish_prefill_hook(side,&fragment).unwrap();
            if side==0 {
                assert!(matches!(target.complete_prefill_chunk(ordinal),
                    Err(CaptureRunHostError::Prefill(CapturePrefillHostError::Progression(
                        crate::capture::CapturePrefillProgressError::Missing)))),"After is still outstanding");
            }
            frame.return_partition_intervention_evidence(loan).unwrap();
        }
        // The actual chunk completion sees both hooks, including the final
        // zero-preview-contribution chunk, and advances each cursor only once.
        let mut loan=frame.take_partition_intervention_evidence(0).unwrap();
        loan.frame_mut().complete_prefill_chunk(ordinal).unwrap();
        assert!(loan.frame_mut().complete_prefill_chunk(ordinal).is_err());
        if ordinal==1 {
            loan.frame_mut().finish_local_prefill_targets().unwrap();
            for (record,expected) in loan.frame().records().iter().zip([[1.,2.,3.,4.],[2.,4.,6.,8.]]) {
                let Some(CapturePayload::SharedTensor(value))=&record.payload else {panic!("original evidence payload")};
                let TensorObservationData::F32(actual)=value.data() else {panic!("actual Float32 evidence")};
                assert_eq!(&actual[..],&expected);
            }
        }
        frame.return_partition_intervention_evidence(loan).unwrap();
    }
    assert_eq!(quota.total().captures,2,"uneven chunks do not reissue either evidence target");
    assert_eq!(ledger(&pool),hold,"both callback phases retain the original protected Host owner");
}


#[test]
fn original_evidence_segment_accepts_only_its_admitted_companion_and_exact_chunk() {
    let (capture,admitted)=sources_with_evidence(true,InterventionEvidence::Preview{max_elements:4});
    let pool=WorkingMemoryPool::new(1<<25,0).unwrap();
    let source=pool.compile_intervention_source(PreparedInterventionPlanCopy::inspect(&admitted).unwrap()).unwrap();
    let equal=pool.compile_intervention_source(PreparedInterventionPlanCopy::inspect(&admitted).unwrap()).unwrap();
    let planned=CaptureRunHostPlan::prepare(&capture).unwrap().with_interventions(&source).unwrap();
    let plain=CaptureRunHostPlan::prepare(&capture).unwrap();
    let (reservation,run)=fresh(&pool,planned.initialization_peak_bytes()+plain.initialization_peak_bytes());
    let request:InferenceRequest=reservation.into();
    let mut bank=run.prepare_capture_run(request.memory_reservation().unwrap(),planned).unwrap();
    let unbound=run.prepare_capture_run(request.memory_reservation().unwrap(),plain).unwrap();
    let mut native=run.scope().unwrap();let mut other=run.scope().unwrap();
    let geometry=request.geometry();
    let chunk=crate::prefill::PrefillChunk {input:0..geometry.input_positions,
        position:geometry.cached_positions,output:geometry.output};
    let context=crate::inspection::PrefillChunkRetentionContext::new(&request,&chunk,
        DistributedCommitEpoch::new(1).unwrap());
    let before=ledger(&pool);
    let bootstrap=bank.prefill_source_bootstrap().unwrap();
    assert!(bootstrap.intervention_source().unwrap().same_source(&source));
    let (segment,registration)=bootstrap.begin_segment(&mut native,&context).unwrap();
    let (unbound_segment,unbound_registration)=unbound.prefill_source_bootstrap().unwrap()
        .begin_segment(&mut other,&context).unwrap();
    for side in 0..2 {
        let companion=source.plan().evidence(0).unwrap().shared_geometry_source();
        let foreign=equal.plan().evidence(0).unwrap().shared_geometry_source();
        assert_eq!(companion.admission().identity(),foreign.admission().identity());
        let rows=CapturePrefillRowAssembly::prepare(companion.admission(),side,geometry).unwrap();
        let fragment=rows.fragment(0).unwrap();
        segment.validate_prefill_fragment(&fragment).unwrap();
        assert!(unbound_segment.validate_prefill_fragment(&fragment).is_err(),
            "same account and geometry cannot register a missing original intervention owner");
        let foreign_rows=CapturePrefillRowAssembly::prepare(foreign.admission(),side,geometry).unwrap();
        assert!(segment.validate_prefill_fragment(&foreign_rows.fragment(0).unwrap()).is_err(),
            "equal independently prepared companions are not the admitted source");
        let different=CapturePrefillRowAssembly::prepare(companion.admission(),side,
            InferenceGeometry {prefill_chunk_positions:1,..geometry}).unwrap();
        assert!(segment.validate_prefill_fragment(&different.fragment(0).unwrap()).is_err());
    }
    assert!(segment.validate_native_scope(&other).is_err());
    segment.validate_native_scope(&native).unwrap();
    assert_eq!(ledger(&pool),before,"the retained source uses already priced bootstrap/stamp controls");
    let frame=bank.begin_step(CapturePhase::Prefill,0).unwrap().prepare().unwrap();
    assert!(frame.prefill_source_bootstrap().unwrap().intervention_source().unwrap().same_source(&source));
    drop(frame);
    drop((registration,unbound_registration,segment,unbound_segment));
    drop((bank,unbound));
    native.certify().unwrap();other.certify().unwrap();
    drop((request,run,source,equal));
    assert_eq!(pool.used_bytes().unwrap(),0);
}
