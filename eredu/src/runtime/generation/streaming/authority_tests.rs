use super::*;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

struct Retired(Arc<AtomicUsize>);

impl Drop for Retired {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

fn authority() -> (HostPreparationAuthority, Arc<AtomicUsize>) {
    let retired = Arc::new(AtomicUsize::new(0));
    (
        HostPreparationAuthority::retain(Retired(retired.clone())),
        retired,
    )
}

fn tools() -> Vec<serde_json::Value> {
    vec![serde_json::json!({
        "type": "function",
        "function": {
            "name": "measure",
            "parameters": {
                "type": "object",
                "properties": {"count": {"type": "integer", "minimum": 3}},
                "required": ["count"],
                "additionalProperties": false
            }
        }
    })]
}

fn text(events: &[SemanticEvent]) -> String {
    events
        .iter()
        .filter_map(|event| match event {
            SemanticEvent::TextDelta(delta) => Some(delta.as_str()),
            _ => None,
        })
        .collect()
}

#[test]
fn text_forks_diverge_with_stop_lookbehind_and_keep_custody_until_final_drop() {
    let (authority, retired) = authority();
    let mut original = ToolRuntimeParser::text(["<stop>"]).with_host_preparation(authority);
    original.push("mar").unwrap();
    let mut branch = original.fork().unwrap();
    original.push("ket<sto").unwrap();
    branch.push("ble<sto").unwrap();
    assert_eq!(text(original.events()), "market");
    assert_eq!(text(branch.events()), "marble");
    assert!(original.push("p>discarded").unwrap());
    assert!(branch.push("p>ignored").unwrap());
    assert_eq!(text(original.events()), "market");
    assert_eq!(text(branch.events()), "marble");
    assert!(original.is_finished());
    assert!(branch.is_finished());
    drop(original);
    assert_eq!(retired.load(Ordering::SeqCst), 0);
    drop(branch);
    assert_eq!(retired.load(Ordering::SeqCst), 1);
}

#[test]
fn shared_schemas_keep_new_authority_after_parser_forks_and_prior_domain_retire() {
    let (first, first_retired) = authority();
    let (second, second_retired) = authority();
    let parser = ToolRuntimeParser::text([])
        .with_host_preparation(first)
        .with_tool_schemas_under_authority(&tools(), &second)
        .unwrap();
    let branch = parser.fork().unwrap();
    let schemas = parser.stream.sink.tool_schemas.clone().unwrap();
    drop((parser, second));
    assert_eq!(first_retired.load(Ordering::SeqCst), 0);
    assert_eq!(second_retired.load(Ordering::SeqCst), 0);
    drop(branch);
    assert_eq!(first_retired.load(Ordering::SeqCst), 1);
    assert_eq!(second_retired.load(Ordering::SeqCst), 0);
    schemas.validate("measure", r#"{"count":23}"#).unwrap();
    assert!(schemas.validate("measure", r#"{"count":1}"#).is_err());
    drop(schemas);
    assert_eq!(second_retired.load(Ordering::SeqCst), 1);
}

#[test]
fn existing_schema_builder_inherits_an_already_owned_parsers_authority() {
    let (authority, retired) = authority();
    let parser = ToolRuntimeParser::text([])
        .with_host_preparation(authority)
        .with_tool_schemas(&tools())
        .unwrap();
    let schemas = parser.stream.sink.tool_schemas.clone().unwrap();
    drop(parser);
    assert_eq!(retired.load(Ordering::SeqCst), 0);
    schemas.validate("measure", r#"{"count":17}"#).unwrap();
    drop(schemas);
    assert_eq!(retired.load(Ordering::SeqCst), 1);
}

#[test]
fn rebinding_parser_authority_keeps_both_original_and_incoming_domains() {
    let (first, first_retired) = authority();
    let (second, second_retired) = authority();
    let mut parser = ToolRuntimeParser::text([])
        .with_host_preparation(first)
        .with_host_preparation(second);
    parser.push("owned by both domains").unwrap();
    assert_eq!(text(parser.events()), "owned by both domains");
    assert_eq!(first_retired.load(Ordering::SeqCst), 0);
    assert_eq!(second_retired.load(Ordering::SeqCst), 0);
    drop(parser);
    assert_eq!(first_retired.load(Ordering::SeqCst), 1);
    assert_eq!(second_retired.load(Ordering::SeqCst), 1);
}

struct CheckedProtocol {
    retired: Arc<AtomicUsize>,
    drops: Arc<AtomicUsize>,
    fail_fork: bool,
}

impl Drop for CheckedProtocol {
    fn drop(&mut self) {
        assert_eq!(self.retired.load(Ordering::SeqCst), 0);
        self.drops.fetch_add(1, Ordering::SeqCst);
    }
}

impl ProtocolParser for CheckedProtocol {
    type Error = String;

    fn fork_box(&self) -> Result<Box<dyn ProtocolParser<Error = String>>, String> {
        assert_eq!(self.retired.load(Ordering::SeqCst), 0);
        if self.fail_fork {
            return Err("deliberate protocol copy rejection".into());
        }
        Ok(Box::new(Self {
            retired: self.retired.clone(),
            drops: self.drops.clone(),
            fail_fork: false,
        }))
    }

    fn push(&mut self, text: &str, sink: &mut SemanticEventSink) -> Result<(), String> {
        sink.text(text);
        Ok(())
    }

    fn finish(&mut self, _: &mut SemanticEventSink) -> Result<(), String> {
        Ok(())
    }
}

struct CheckedDecoder {
    retired: Arc<AtomicUsize>,
    drops: Arc<AtomicUsize>,
}

impl Clone for CheckedDecoder {
    fn clone(&self) -> Self {
        assert_eq!(self.retired.load(Ordering::SeqCst), 0);
        Self {
            retired: self.retired.clone(),
            drops: self.drops.clone(),
        }
    }
}

impl Drop for CheckedDecoder {
    fn drop(&mut self) {
        assert_eq!(self.retired.load(Ordering::SeqCst), 0);
        self.drops.fetch_add(1, Ordering::SeqCst);
    }
}

impl TokenDecoderBackend for CheckedDecoder {
    type Error = std::convert::Infallible;

    fn decode_token(&mut self, token: u32, _: bool) -> Result<Vec<u8>, Self::Error> {
        Ok(vec![u8::try_from(token).unwrap()])
    }
}

#[test]
fn decoder_and_protocol_destructors_run_before_final_pipeline_authority() {
    let (authority, retired) = authority();
    let protocol_drops = Arc::new(AtomicUsize::new(0));
    let decoder_drops = Arc::new(AtomicUsize::new(0));
    let parser = ToolRuntimeParser::new(
        Box::new(CheckedProtocol {
            retired: retired.clone(),
            drops: protocol_drops.clone(),
            fail_fork: false,
        }),
        [],
        [],
    )
    .with_host_preparation(authority);
    let decoder = RawTokenDecoder::new(
        CheckedDecoder {
            retired: retired.clone(),
            drops: decoder_drops.clone(),
        },
        [],
    );
    let mut pipeline = CommittedTokenPipeline::new(decoder, parser);
    let mut branch = pipeline.fork().unwrap();
    let mut first_events = Vec::new();
    let mut second_events = Vec::new();
    pipeline
        .push(b'a' as u32, &mut |event| first_events.push(event))
        .unwrap();
    branch
        .push(b'b' as u32, &mut |event| second_events.push(event))
        .unwrap();
    assert_eq!(text(&first_events), "a");
    assert_eq!(text(&second_events), "b");
    drop(pipeline);
    assert_eq!(retired.load(Ordering::SeqCst), 0);
    drop(branch);
    assert_eq!(decoder_drops.load(Ordering::SeqCst), 2);
    assert_eq!(protocol_drops.load(Ordering::SeqCst), 2);
    assert_eq!(retired.load(Ordering::SeqCst), 1);
}

#[test]
fn failed_pipeline_fork_drops_partial_decoder_while_source_authority_is_live() {
    let (authority, retired) = authority();
    let protocol_drops = Arc::new(AtomicUsize::new(0));
    let decoder_drops = Arc::new(AtomicUsize::new(0));
    let parser = ToolRuntimeParser::new(
        Box::new(CheckedProtocol {
            retired: retired.clone(),
            drops: protocol_drops.clone(),
            fail_fork: true,
        }),
        [],
        [],
    )
    .with_host_preparation(authority);
    let decoder = RawTokenDecoder::new(
        CheckedDecoder {
            retired: retired.clone(),
            drops: decoder_drops.clone(),
        },
        [],
    );
    let pipeline = CommittedTokenPipeline::new(decoder, parser);
    let error = match pipeline.fork() {
        Ok(_) => panic!("copy unexpectedly succeeded"),
        Err(error) => error,
    };
    assert_eq!(error, "deliberate protocol copy rejection");
    assert_eq!(decoder_drops.load(Ordering::SeqCst), 1);
    assert_eq!(protocol_drops.load(Ordering::SeqCst), 0);
    assert_eq!(retired.load(Ordering::SeqCst), 0);
    drop(pipeline);
    assert_eq!(decoder_drops.load(Ordering::SeqCst), 2);
    assert_eq!(protocol_drops.load(Ordering::SeqCst), 1);
    assert_eq!(retired.load(Ordering::SeqCst), 1);
}
