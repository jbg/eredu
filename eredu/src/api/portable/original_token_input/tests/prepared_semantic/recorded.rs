use super::*;
use crate::api::{ControlledGenerationEvent,ControlledGenerationRecord,ObservedGenerationEvent,TraceLimits};
use eredu_core::execution_control::{GenerationControlHandle,GenerationStatus};
use std::ops::ControlFlow;

fn limits()->TraceLimits {TraceLimits {per_record_bytes:1<<20,total_bytes:16<<20}}
fn collect(records:&mut Vec<ControlledGenerationRecord>)->impl FnMut(ControlledGenerationRecord)->ControlFlow<()>+'_ {
    |record|{records.push(record);ControlFlow::Continue(())}
}
#[test]
fn recorded_session_preserves_ordinary_events_forcing_pause_timing_and_source_custody() {
    for mode in 0..4 {
        let text=if mode<2 {"Seventeen."} else {"<tool_call>\n{\"name\":\"reading\",\"arguments\":{\"value\":17}}\n</tool_call>"};
        let (mut model,source,facts,pool)=fixture(text);
        let mut policy=request(match mode {0|1=>ToolChoice::None,2=>ToolChoice::Required,_=>ToolChoice::Auto});
        policy.tools[0]["function"]["parameters"]["$schema"]=serde_json::json!("http://json-schema.org/draft-07/schema#");
        if mode<2 {policy.tools.clear();}
        let chat=prepare_chat(&model,&source,policy);
        let make_request=|| {
            let mut request=PreparedChatRequest::new(&chat,settings());
            if mode==0 {request.output_mode=crate::api::PreparedChatOutputMode::Text;}
            request
        };
        let cancel=GenerationCancellationToken::new();
        let mut ordinary_events=Vec::new();
        let ordinary=model.start_prepared_chat(make_request(),&cancel).unwrap().unwrap()
            .run(&cancel,&mut |event|ordinary_events.push(event)).unwrap();
        let expected_ids=ordinary.token_ids.to_vec();
        let expected_finish=ordinary.finish_reason;
        let expected_events=serde_json::to_value(&ordinary_events).unwrap();
        drop((ordinary,ordinary_events));
        facts.borrow_mut().order.clear();
        let control=GenerationControlHandle::default();
        let mut records=Vec::new();
        let mut session=model.start_controlled_chat(make_request(),limits(),control.clone(),collect(&mut records)).unwrap().unwrap();
        assert!(matches!(records[0].event,ControlledGenerationEvent::Started {..}));
        assert!(session.token_ids().is_empty());
        control.request_pause();
        assert_eq!(session.run(collect(&mut records)).unwrap(),GenerationStatus::Paused);
        assert!(!facts.borrow().order.contains(&"submit"));
        session.force_next_token(expected_ids[0]).unwrap();
        session.step(collect(&mut records)).unwrap();
        assert_eq!(session.token_ids(),&expected_ids[..1]);
        let first=records.iter().find(|record|matches!(record.event.progress(),Some(ObservedGenerationEvent::Token {..}))).unwrap();
        assert!(matches!(first.event.progress(),Some(ObservedGenerationEvent::Token {forced:true,..})));
        assert_eq!(first.timing,session.timing());
        assert_eq!(session.resume(collect(&mut records)).unwrap(),GenerationStatus::Completed);
        assert_eq!(session.token_ids(),expected_ids);
        assert_eq!(session.finish_reason(),Some(expected_finish));
        let semantic=records.iter().filter_map(|record|match record.event.progress() {
            Some(ObservedGenerationEvent::Semantic {event,..})=>Some(event),_=>None,
        }).collect::<Vec<_>>();
        assert_eq!(serde_json::to_value(semantic).unwrap(),expected_events);
        for (sequence,record) in records.iter().enumerate() {assert_eq!(record.sequence,sequence as u64);}
        let checkpoint=session.output_checkpoint().unwrap();
        assert_eq!(checkpoint.next_prediction,expected_ids.len() as u64);
        drop(session);drop((chat,source,model));
        assert!(pool.used_bytes().unwrap()>0,"escaped records retain their actual producer");
        drop(records);
        assert!(pool.used_bytes().unwrap()>0,"escaped output checkpoint retains its actual producer");
        assert_eq!(checkpoint.next_prediction,expected_ids.len() as u64);
        drop(checkpoint);assert_eq!(pool.used_bytes().unwrap(),0);
    }
}

#[test]
fn recorded_failure_retains_original_prefix_and_cancellation_precedes_prediction() {
    for fail in [false,true] {
        let (mut model,source,facts,pool)=fixture("Seventeen.");
        let mut policy=request(ToolChoice::None);policy.tools.clear();
        let chat=prepare_chat(&model,&source,policy);
        let control=GenerationControlHandle::default();
        let mut records=Vec::new();
        let mut session=model.start_controlled_chat(PreparedChatRequest::new(&chat,settings()),limits(),control.clone(),collect(&mut records)).unwrap().unwrap();
        if fail {
            session.step(collect(&mut records)).unwrap();
            let prefix=session.token_ids().to_vec();
            facts.borrow_mut().fail_step=true;
            let error=session.step(collect(&mut records)).unwrap_err();
            assert_eq!(session.status(),GenerationStatus::Failed);
            assert_eq!(session.token_ids(),prefix);
            assert_eq!(error.session_failure().unwrap().committed_token_ids(),Some(prefix.as_slice()));
            drop(session);drop((records,chat,source,model));
            assert!(pool.used_bytes().unwrap()>0);
            assert_eq!(error.session_failure().unwrap().committed_token_ids(),Some(prefix.as_slice()));
            drop(error);
        } else {
            control.request_pause();control.cancel();
            assert_eq!(session.run(collect(&mut records)).unwrap(),GenerationStatus::Cancelled);
            assert!(session.token_ids().is_empty());
            assert_eq!(session.next_prediction(),0);
            assert!(!facts.borrow().order.contains(&"submit"));
            drop(session);drop((records,chat,source,model));
        }
        assert_eq!(pool.used_bytes().unwrap(),0);
    }
}
