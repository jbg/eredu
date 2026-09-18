use super::*;
use crate::{ParserAllocationFunding, prepared_funding::Scope};
use std::sync::{Arc, atomic::{AtomicUsize, Ordering}};

#[derive(Debug)]
struct Refused(usize);
impl fmt::Display for Refused {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "inspection cut {}", self.0) }
}
impl std::error::Error for Refused {}

fn tree(branches: usize, depth: usize) -> RegexAst {
    RegexAst::Or((0..branches).map(|_| {
        let mut text = String::with_capacity(31);
        text.push_str("alpha");
        let mut node = RegexAst::Literal(text);
        for _ in 0..depth { node = RegexAst::Repeat(Box::new(node), 1, 1); }
        node
    }).collect())
}
fn account(cut: usize) -> (ParserAllocationFunding, Arc<AtomicUsize>, Arc<AtomicUsize>, std::sync::Weak<()>) {
    let calls = Arc::new(AtomicUsize::new(0));
    let bytes = Arc::new(AtomicUsize::new(0));
    let owner = Arc::new(());
    let weak = Arc::downgrade(&owner);
    let (c, b) = (calls.clone(), bytes.clone());
    let funding = ParserAllocationFunding::prepare(move |amount| {
        let _keep = &owner;
        let n = c.fetch_add(1, Ordering::SeqCst) + 1;
        if n == cut { Err(Refused(n)) } else { b.fetch_add(amount, Ordering::SeqCst); Ok(()) }
    }).unwrap();
    (funding, calls, bytes, weak)
}

#[test]
fn ast_inspection_pays_live_depth_reuses_sibling_frames_and_preserves_spare_source_capacity() {
    let inspect = |source: &RegexAst| {
        let (funding, calls, bytes, _) = account(usize::MAX);
        let scope = Scope::new(&funding).unwrap();
        let plan = source.source_copy_plan(&scope).unwrap();
        let after = calls.load(Ordering::SeqCst);
        let actual = source.retained_capacity_bytes(&scope).unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), after);
        let again = source.source_copy_plan(&scope).unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), after);
        assert_eq!(again.requirements().retained_bytes(), plan.requirements().retained_bytes());
        let copied = plan.compile().unwrap();
        assert_eq!(copied.retained_capacity_bytes(&scope).unwrap(), again.requirements().retained_bytes());
        assert!(actual > again.requirements().retained_bytes());
        (bytes.load(Ordering::SeqCst), calls.load(Ordering::SeqCst))
    };
    let narrow = inspect(&tree(1, 12));
    let wide = inspect(&tree(16, 12));
    assert_eq!(narrow, wide, "sequential siblings share only the same live frame capacity");
    assert!(inspect(&tree(1, 24)).0 > narrow.0, "simultaneous deeper recursion must pay");
}

#[test]
fn ast_source_and_retained_inspection_refuse_every_reached_frame_before_copy_and_keep_original_cause() {
    let source = tree(3, 12);
    for retained_only in [false, true] {
        let (funding, calls, _, _) = account(usize::MAX);
        let scope = Scope::new(&funding).unwrap();
        if retained_only { source.retained_capacity_bytes(&scope).unwrap(); }
        else { source.source_copy_plan(&scope).unwrap(); }
        let reached = calls.load(Ordering::SeqCst);
        drop(scope);
        drop(funding);
        assert!(reached > 12);
        // The callback owner's construction is separately tested by allocation_funding;
        // these cuts start with the operation-scope shell, then each reached frame.
        for cut in 2..=reached {
            let (funding, calls, _, weak) = account(cut);
            let failure = match Scope::new(&funding) {
                Err(FrameError::Funding(error)) => error,
                Err(FrameError::Overflow) => panic!("finite scope"),
                Ok(scope) => {
                    let error = if retained_only { source.retained_capacity_bytes(&scope).unwrap_err() }
                        else { source.source_copy_plan(&scope).unwrap_err() };
                    assert!(error.destinations.is_empty());
                    assert!(error.completed.is_none());
                    let Cause::Funding(error) = error.cause else { panic!("lost original funding cause") };
                    error
                }
            };
            assert_eq!(calls.load(Ordering::SeqCst), cut);
            use std::error::Error;
            assert_eq!(failure.source().unwrap().downcast_ref::<Refused>().unwrap().0, cut);
            drop(funding);
            assert!(weak.upgrade().is_some());
            drop(failure);
            assert!(weak.upgrade().is_none());
        }
    }
}
