use super::*;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering::SeqCst},
};

#[test]
fn ordinary_and_mode_statistics_keep_default_vec_output_and_clone_api() {
    let timing = GenerationTiming::new(Some(Duration::from_millis(17)));
    let ordinary: GenerationOutput =
        GenerationOutput::new(vec![4, 7], FinishReason::Eos, timing, ());
    let copy = ordinary.clone();
    assert_eq!(copy, ordinary);
    assert_eq!(copy.token_ids(), [4, 7]);
    assert_ne!(copy.token_ids.as_ptr(), ordinary.token_ids.as_ptr());
    assert_eq!(copy.token_ids.into_iter().collect::<Vec<_>>(), [4, 7]);
    let mode: GenerationOutput<Vec<u32>> =
        GenerationOutput::new(vec![8], FinishReason::MaxTokens, timing, vec![2, 5]);
    assert_eq!(mode.clone(), mode);
    assert_eq!(mode.stats(), &[2, 5]);
    assert_eq!(mode.finish_reason(), FinishReason::MaxTokens);
    assert_eq!(*mode.timing(), timing);
}

#[derive(Debug)]
struct Tokens {
    values: Vec<u32>,
    drops: Arc<AtomicUsize>,
}
impl GenerationTokenIdStorage for Tokens {
    fn token_ids(&self) -> &[u32] {
        &self.values
    }
    fn retire(self: Arc<Self>) {
        drop(Arc::into_inner(self));
    }
}
impl Drop for Tokens {
    fn drop(&mut self) {
        self.drops.fetch_add(1, SeqCst);
    }
}

#[test]
fn retained_ordinary_output_moves_owner_and_partial_iteration_preserves_it() {
    fn send_sync<T: Send + Sync>() {}
    send_sync::<GenerationOutput<(), GenerationTokenIds>>();
    let drops = Arc::new(AtomicUsize::new(0));
    let owner = Arc::new(Tokens {
        values: vec![11, 13, 17],
        drops: drops.clone(),
    });
    let pointer = owner.values.as_ptr();
    let timing = GenerationTiming::new(Some(Duration::from_micros(23)));
    let output = GenerationOutput::from_retained(
        GenerationTokenIds::from_owner(owner),
        FinishReason::Eos,
        timing,
    );
    assert_eq!(output.token_ids(), [11, 13, 17]);
    assert_eq!(output.token_ids().as_ptr(), pointer);
    assert_eq!(*output.timing(), timing);
    assert_eq!(output.finish_reason(), FinishReason::Eos);
    assert_eq!(output.stats(), &());
    let mut ids = output.token_ids.into_iter();
    assert_eq!(ids.next(), Some(11));
    assert_eq!(ids.next_back(), Some(17));
    assert_eq!(drops.load(SeqCst), 0);
    drop(ids);
    assert_eq!(drops.load(SeqCst), 1);
}

#[test]
fn output_fields_retire_before_the_final_token_storage_owner() {
    // Exercise the actual record's declaration order with a destructor witness.
    // This test grants no bound for arbitrary mode-specific statistics.
    struct Statistics(Arc<AtomicUsize>);
    impl Drop for Statistics {
        fn drop(&mut self) {
            assert_eq!(self.0.load(SeqCst), 0);
        }
    }
    let drops = Arc::new(AtomicUsize::new(0));
    let output = GenerationOutput {
        finish_reason: FinishReason::Cancelled,
        timing: GenerationTiming::default(),
        stats: Statistics(drops.clone()),
        token_ids: GenerationTokenIds::from_owner(Arc::new(Tokens {
            values: vec![19],
            drops: drops.clone(),
        })),
    };
    drop(output);
    assert_eq!(drops.load(SeqCst), 1);
}

#[test]
fn escaped_reasoning_event_uses_the_same_text_owner_and_ordinary_copy_semantics() {
    #[derive(Debug)]
    struct Funding(Arc<AtomicUsize>);
    impl Drop for Funding {
        fn drop(&mut self) {
            self.0.fetch_add(1, SeqCst);
        }
    }
    let drops = Arc::new(AtomicUsize::new(0));
    let host = crate::HostPreparationAuthority::retain(Funding(drops.clone()));
    let text = SemanticText::try_copy_retained("réason 🦀", host.clone()).unwrap();
    let pointer = text.as_str().as_ptr();
    let events = vec![SemanticEvent::ReasoningDelta(text)];
    let escaped = events[0].clone();
    drop((events, host));
    assert_eq!(drops.load(SeqCst), 0);
    let SemanticEvent::ReasoningDelta(text) = &escaped else {
        panic!("reasoning event");
    };
    assert_eq!(text.as_str(), "réason 🦀");
    assert_eq!(text.as_str().as_ptr(), pointer);
    assert_eq!(text.snapshot_copy_bytes(), 0);
    let ordinary = SemanticEvent::ReasoningDelta("réason 🦀".into());
    assert_eq!(escaped, ordinary);
    let copy = ordinary.clone();
    let (SemanticEvent::ReasoningDelta(a), SemanticEvent::ReasoningDelta(b)) = (&ordinary, &copy)
    else {
        panic!("ordinary reasoning event");
    };
    assert_ne!(a.as_str().as_ptr(), b.as_str().as_ptr());
    assert!(a.snapshot_copy_bytes() >= a.len());
    drop(escaped);
    assert_eq!(drops.load(SeqCst), 1);
}

#[test]
fn escaped_tool_event_fields_retain_their_payer_and_preserve_string_serialization() {
    #[derive(Debug)]
    struct Funding(Arc<AtomicUsize>);
    impl Drop for Funding {
        fn drop(&mut self) { self.0.fetch_add(1, SeqCst); }
    }
    let drops = Arc::new(AtomicUsize::new(0));
    let host = crate::HostPreparationAuthority::retain(Funding(drops.clone()));
    let id = SemanticText::try_copy_retained("call_é", host.clone()).unwrap();
    let name = SemanticText::try_copy_retained("lookup", host.clone()).unwrap();
    let arguments = SemanticText::try_copy_retained(r#"{"place":"Zürich"}"#, host.clone()).unwrap();
    let pointers = [id.as_ptr(), name.as_ptr(), arguments.as_ptr()];
    let events = [
        SemanticEvent::ToolCallStart { index: 1, id, name },
        SemanticEvent::ToolArgumentsDelta { index: 1, json_fragment: arguments },
    ];
    let escaped = events.clone();
    drop((events, host));
    assert_eq!(drops.load(SeqCst), 0);
    let SemanticEvent::ToolCallStart { id, name, .. } = &escaped[0] else { panic!("start") };
    let SemanticEvent::ToolArgumentsDelta { json_fragment, .. } = &escaped[1] else { panic!("arguments") };
    assert_eq!([id.as_ptr(), name.as_ptr(), json_fragment.as_ptr()], pointers);
    assert_eq!(id, "call_é");
    assert_eq!(name, &String::from("lookup"));
    assert_eq!([id.snapshot_copy_bytes(), name.snapshot_copy_bytes(), json_fragment.snapshot_copy_bytes()], [0; 3]);
    let ordinary = [
        SemanticEvent::ToolCallStart { index: 1, id: "call_é".into(), name: "lookup".into() },
        SemanticEvent::ToolArgumentsDelta { index: 1, json_fragment: r#"{"place":"Zürich"}"#.into() },
    ];
    assert_eq!(escaped, ordinary);
    let json = serde_json::to_string(&escaped).unwrap();
    assert_eq!(json, serde_json::to_string(&ordinary).unwrap());
    assert_eq!(serde_json::from_str::<[SemanticEvent; 2]>(&json).unwrap(), ordinary);
    let copy = ordinary.clone();
    let SemanticEvent::ToolCallStart { id: a, name: n, .. } = &ordinary[0] else { panic!("ordinary") };
    let SemanticEvent::ToolCallStart { id: b, name: m, .. } = &copy[0] else { panic!("copy") };
    assert_ne!(a.as_ptr(), b.as_ptr());
    assert_ne!(n.as_ptr(), m.as_ptr());
    assert!(a.snapshot_copy_bytes() >= a.len());
    let [start, delta] = escaped;
    drop(start);
    assert_eq!(drops.load(SeqCst), 0);
    drop(delta);
    assert_eq!(drops.load(SeqCst), 1);
}
