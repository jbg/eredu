//! Exact ordered partial edit expectations against the unmodified captured row.
use super::*;
use eredu_core::intervention::*;

const START:usize=3;
const END:usize=11;
const OPERATIONS:[(&str,f32,InterventionEvidence);2]=[
    ("halve-partial",0.5,InterventionEvidence::Preview{max_elements:3}),
    ("negate-partial",-2.0,InterventionEvidence::Summary),
];

pub(super) fn plan()->InterventionPlan {
    InterventionPlan{schema_version:INTERVENTION_SCHEMA_VERSION,operations:OPERATIONS.into_iter()
        .map(|(id,factor,evidence)|InterventionOperation{
            id:id.into(),target:eredu_core::MODEL_LOGITS_OBSERVATION_PATH.into(),schedule:Default::default(),
            slices:vec![CaptureSlice{axis:"vocabulary".into(),start:START as u64,end:END as u64,stride:1}],
            action:InterventionAction::Scale{dtype:InterventionDtype::Float32,factor},evidence,
        }).collect()}
}
fn close(actual:f64,expected:f64){
    assert!((actual-expected).abs()<=2e-5+2e-5*expected.abs(),"{actual} != {expected}");
}

pub(super) fn compare(raw:&[f32],records:&[InterventionRecord],prediction:u64){
    assert_eq!(raw.len(),64);assert_eq!(records.len(),OPERATIONS.len());
    let mut effective=raw.to_vec();
    for (record,(id,factor,evidence)) in records.iter().zip(OPERATIONS) {
        assert_eq!(record.operation_id,id);assert_eq!(record.outcome,InterventionOutcome::Applied);
        assert_eq!(record.prediction_index,prediction);assert_eq!(record.evidence.len(),2);
        assert!(record.charged.retained_bytes>0);
        let before=effective[START..END].to_vec();
        for value in &mut effective[START..END]{*value *= factor;}
        let after=effective[START..END].to_vec();
        for (actual,(side,position,expected)) in record.evidence.iter().zip([
            ("before",eredu_core::ObservationPosition::BeforeIntervention,before),
            ("after",eredu_core::ObservationPosition::AfterIntervention,after),
        ]) {
            assert_eq!(actual.selection_id,format!("{id}:{side}:None"));
            assert_eq!(actual.path,eredu_core::MODEL_LOGITS_OBSERVATION_PATH);
            assert_eq!(actual.position,position);
            assert_eq!(actual.source_shape.as_deref(),Some([1,1,64].as_slice()));
            assert_eq!(actual.selected_shape.as_deref(),Some([1,1,(END-START) as u64].as_slice()));
            let payload=actual.payload.as_ref().expect("ordered partial evidence was not skipped");
            match evidence {
                InterventionEvidence::Preview{max_elements}=>{
                    let tensor=payload.as_tensor().expect("bounded before/after Preview");
                    let eredu_core::TensorObservationData::F32(values)=tensor.data() else{panic!("F32 evidence")};
                    assert_eq!(values.len(),max_elements as usize);
                    for (&a,&e) in values.iter().zip(&expected){close(f64::from(a),f64::from(e));}
                }
                InterventionEvidence::Summary=>{
                    let CapturePayload::Summary(summary)=payload else{panic!("finite before/after Summary")};
                    assert_eq!(summary.elements,(END-START) as u64);
                    assert_eq!(summary.finite,summary.elements);assert_eq!(summary.non_finite,0);
                    assert_eq!(summary.nan,0);assert_eq!(summary.positive_infinity,0);assert_eq!(summary.negative_infinity,0);
                    let n=expected.len() as f64;
                    let sum=expected.iter().map(|&v|f64::from(v)).sum::<f64>();
                    let square=expected.iter().map(|&v|f64::from(v).powi(2)).sum::<f64>();
                    close(summary.min.unwrap(),expected.iter().map(|&v|f64::from(v)).fold(f64::INFINITY,f64::min));
                    close(summary.max.unwrap(),expected.iter().map(|&v|f64::from(v)).fold(f64::NEG_INFINITY,f64::max));
                    close(summary.mean.unwrap(),sum/n);close(summary.rms.unwrap(),(square/n).sqrt());
                }
                _=>unreachable!("closed two-action evidence request"),
            }
        }
    }
    for (index,(&actual,&original)) in effective.iter().zip(raw).enumerate(){
        assert_eq!(actual,if (START..END).contains(&index){-original}else{original});
    }
}
