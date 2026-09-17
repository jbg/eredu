use super::*;
use std::{
    sync::{mpsc, Arc},
    time::Duration,
};

#[test]
fn terminal_alias_fences_without_waiting_for_poison_mutex_or_allocating_another_owner() {
    let authority = PartitionCommunicationAuthority::new(None);
    let alias = authority.clone();
    assert!(Arc::ptr_eq(&authority.shared, &alias.shared));
    let original = Arc::as_ptr(&authority.shared);
    let held = authority.clone();
    let (entered_tx, entered_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let worker = std::thread::spawn(move || {
        let _poison = held.poison_guard();
        entered_tx.send(()).unwrap();
        release_rx.recv_timeout(Duration::from_secs(2)).is_ok()
    });
    entered_rx.recv_timeout(Duration::from_secs(2)).unwrap();
    authority.mark_terminal();
    assert!(matches!(
        alias.ensure_active(),
        Err(PartitionExecutionError::CommunicationTerminal)
    ));
    assert_eq!(Arc::as_ptr(&alias.shared), original);
    let released = release_tx.send(()).is_ok();
    assert!(
        worker.join().unwrap() && released,
        "terminal path waited for poison mutex"
    );
    assert!(alias.is_poisoned());
    assert_eq!(
        PartitionCommunicationAuthority::shared_control_bytes(),
        std::mem::size_of::<CommunicationAuthorityState>() + 2 * std::mem::size_of::<usize>()
    );
    assert!(
        authority.poison_guard().is_none(),
        "terminal marking must not allocate/format a diagnostic"
    );
}
