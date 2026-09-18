//! Funding probes shared by the grammar producers' behavioral tests.
use super::*;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

#[derive(Debug, thiserror::Error)]
#[error("grammar producer funding refused")]
struct Refused;

pub(crate) fn every_refusal(
    mut render: impl FnMut(&ParserAllocationFunding) -> Result<String, Error>,
) -> String {
    let ordinary = render(&ParserAllocationFunding::unenforced()).unwrap();
    let calls = Arc::new(AtomicUsize::new(0));
    let bytes = Arc::new(AtomicUsize::new(0));
    let tally = calls.clone();
    let total = bytes.clone();
    let funding = ParserAllocationFunding::prepare(move |bytes| {
        tally.fetch_add(1, Ordering::Relaxed);
        total.fetch_add(bytes, Ordering::Relaxed);
        Ok::<_, Refused>(())
    })
    .unwrap();
    let expected = render(&funding).unwrap();
    assert_eq!(ordinary, expected);
    let reached = calls.load(Ordering::Relaxed);
    assert!(reached > 2);
    eprintln!(
        "grammar output {} bytes; {} reservation requests; {} cumulative bytes",
        expected.len(),
        reached,
        bytes.load(Ordering::Relaxed)
    );
    for fail in 1..reached {
        let calls = Arc::new(AtomicUsize::new(0));
        let tally = calls.clone();
        let funding = ParserAllocationFunding::prepare(move |_| {
            if tally.fetch_add(1, Ordering::Relaxed) == fail {
                Err(Refused)
            } else {
                Ok(())
            }
        })
        .unwrap();
        let error = render(&funding).unwrap_err();
        let mut cause: &(dyn std::error::Error + 'static) = &error;
        while let Some(next) = cause.source() {
            cause = next;
        }
        assert!(cause.is::<Refused>(), "failure {fail}: {error:?}");
        assert_eq!(
            calls.load(Ordering::Relaxed),
            fail + 1,
            "request after refusal at {fail}"
        );
    }
    expected
}

#[test]
fn nested_display_escaping_matches_independent_json_serialization() {
    for source in ["", "quotes'\"\\/\n\r\t\u{0}\u{1f}", "mañana 水 🦀"] {
        assert_eq!(
            Quoted(Literal(source)).to_string(),
            serde_json::to_string(&serde_json::to_string(source).unwrap()).unwrap()
        );
        assert_eq!(
            Quoted(format_args!("\r\n{source}\r\n")).to_string(),
            serde_json::to_string(&format!("\r\n{source}\r\n")).unwrap()
        );
    }
}
