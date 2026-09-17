use super::*;
use eredu_core::HostPreparationAuthority;
use std::sync::atomic::{AtomicUsize, Ordering};

fn source(data: &Data) -> ResidentResetSource<'_, State> {
    ResidentResetSource {
        state: &data.state,
        selected: &data.selected,
        execution: &data.execution,
        control: &data.control,
        revision: data.state.retention.initialized_revision(),
    }
}
#[test]
fn empty_fork_preserves_canonical_state_and_keeps_table_alias_custody() {
    let (_runtime, data, _, _) = fixture(None, 3);
    let fills = FILLS.get();
    {
        let data = data.borrow();
        assert!(matches!(
            PreparedResidentEmptyState::inspect(source(&data)),
            Err(WorkingMemoryError::UnknownBound)
        ));
        assert_eq!(
            FILLS.get(),
            fills,
            "populated source refuses before construction"
        );
    }
    for slot in data.borrow_mut().state.layers.slots_mut() {
        slot.position = 0;
        slot.values = [0; 4];
    }
    struct Host(Arc<AtomicUsize>);
    impl Drop for Host {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }
    let retired = Arc::new(AtomicUsize::new(0));
    let host = HostPreparationAuthority::retain(Host(retired.clone()));
    let mut fork = {
        let data = data.borrow();
        let plan = PreparedResidentEmptyState::inspect(source(&data)).unwrap();
        assert!(plan.required_bytes() >= 3 * size_of::<Slot>() as u64);
        let fork = plan.construct(&host).unwrap();
        assert_ne!(
            fork.layers.metadata().identity().registry_key(),
            data.state.layers.metadata().identity().registry_key()
        );
        assert_eq!(fork.global_start, data.state.global_start);
        assert_eq!(fork.layout.layout(), data.state.layout.layout());
        fork
    };
    assert_eq!(FILLS.get(), fills + 3);
    fork.layers.slots_mut()[0].position = 7;
    assert_eq!(data.borrow().state.layers.slots()[0].position, 0);
    let alias = fork.layers.metadata().clone();
    drop((fork, host));
    assert_eq!(retired.load(Ordering::SeqCst), 0);
    drop(alias);
    assert_eq!(retired.load(Ordering::SeqCst), 1);
}
