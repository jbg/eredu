use super::*;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Weak,
};

struct PayloadController {
    inner: Controller,
    _payload: Arc<Vec<u32>>,
}

impl TokenFilterController for PayloadController {
    type Error = io::Error;

    fn current_filter(&mut self) -> Result<TokenFilter, Self::Error> {
        self.inner.current_filter()
    }

    fn commit_token(&mut self, token: u32) -> Result<(), Self::Error> {
        self.inner.commit_token(token)
    }

    fn is_complete(&mut self) -> Result<bool, Self::Error> {
        self.inner.is_complete()
    }
}

struct RetirementProbe {
    payloads: Vec<Weak<Vec<u32>>>,
    drops: Arc<AtomicUsize>,
}

impl Drop for RetirementProbe {
    fn drop(&mut self) {
        assert!(
            self.payloads
                .iter()
                .all(|payload| payload.upgrade().is_none()),
            "host custody must outlive actual controller payloads"
        );
        self.drops.fetch_add(1, Ordering::SeqCst);
    }
}

fn custody(payloads: &[&Arc<Vec<u32>>]) -> (HostPreparationAuthority, Arc<AtomicUsize>) {
    let drops = Arc::new(AtomicUsize::new(0));
    let authority = HostPreparationAuthority::retain(RetirementProbe {
        payloads: payloads
            .iter()
            .map(|payload| Arc::downgrade(payload))
            .collect(),
        drops: Arc::clone(&drops),
    });
    (authority, drops)
}

#[test]
fn continuation_host_custody_combines_owners_and_follows_forked_payloads() {
    let (mut runtime, facts) = fixture();
    let root_payload = Arc::new(vec![3, 5, 8]);
    let child_payload = Arc::new(vec![3, 5, 8, 13]);
    let (source, source_drops) = custody(&[&root_payload, &child_payload]);
    let (root_only, root_drops) = custody(&[&root_payload]);
    let (destination, destination_drops) = custody(&[&child_payload]);
    let mut driver = TextGenerationDriver::new(&mut runtime);
    let mut state = driver
        .start(
            Prompt {
                ids: vec![1, 2],
                facts: Rc::clone(&facts),
            },
            config(),
            PayloadController {
                inner: controller(&facts, ControllerFailure::None),
                _payload: root_payload,
            },
        )
        .unwrap();
    state.retain_host_preparation(source);
    let mut child = {
        let boundary = driver.quiescent(&mut state).unwrap();
        boundary.fork_host_state(
            State {
                changes: 0,
                facts: Rc::clone(&facts),
            },
            PayloadController {
                inner: controller(&facts, ControllerFailure::None),
                _payload: child_payload,
            },
            Some(PendingTextInput::Prefill(Prompt {
                ids: vec![1, 2],
                facts: Rc::clone(&facts),
            })),
            Some(8),
        )
    };
    child.retain_host_preparation(destination);
    driver
        .quiescent(&mut state)
        .unwrap()
        .retain_host_preparation(root_only);
    drop(driver);
    drop(runtime);
    drop(state);
    assert_eq!(root_drops.load(Ordering::SeqCst), 1);
    assert_eq!(source_drops.load(Ordering::SeqCst), 0);
    assert_eq!(destination_drops.load(Ordering::SeqCst), 0);
    drop(child);
    assert_eq!(source_drops.load(Ordering::SeqCst), 1);
    assert_eq!(destination_drops.load(Ordering::SeqCst), 1);
}

#[test]
fn retained_host_installation_custody_survives_boundary_unwind() {
    let (mut runtime, facts) = fixture();
    let payload = Arc::new(vec![21, 34, 55]);
    let (authority, drops) = custody(&[&payload]);
    let mut driver = TextGenerationDriver::new(&mut runtime);
    let mut state = driver
        .start(
            Prompt {
                ids: vec![1],
                facts: Rc::clone(&facts),
            },
            config(),
            PayloadController {
                inner: controller(&facts, ControllerFailure::None),
                _payload: Arc::new(vec![1, 2]),
            },
        )
        .unwrap();
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut boundary = driver.quiescent(&mut state).unwrap();
        boundary.retain_host_preparation(authority.clone());
        boundary.install_host_state(
            PayloadController {
                inner: controller(&facts, ControllerFailure::None),
                _payload: payload,
            },
            None,
            Some(4),
        );
        panic!("later composed installation failed");
    }))
    .is_err());
    drop(authority);
    assert!(matches!(
        state.require_quiescent(),
        Err(TextContinuationError::Failed)
    ));
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    drop(state);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}
