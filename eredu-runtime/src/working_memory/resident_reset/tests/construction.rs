use super::*;
use eredu_nn::workspace::{WorkspaceMetadataFunding, WorkspaceMetadataFundingError};
pub(super) const PAYLOAD: usize = 37;
pub(super) const BYTES: usize =
    PAYLOAD + std::mem::size_of::<Owner>() + std::mem::size_of::<Result<(), BackendFailure>>();
struct Owner {
    values: Vec<u8>,
    _funding: WorkspaceMetadataFunding,
}
thread_local! {
    static ENABLED: Cell<bool> = const { Cell::new(false) };
    static OWNER: RefCell<Option<Owner>> = const { RefCell::new(None) };
}
pub(super) fn enabled() -> bool {
    ENABLED.get()
}
pub(super) fn prepare(funding: &WorkspaceMetadataFunding) -> Result<(), BackendFailure> {
    funding
        .reserve_metadata(BYTES)
        .map_err(BackendFailure::from_error)?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(PAYLOAD)
        .map_err(BackendFailure::from_error)?;
    values.resize(PAYLOAD, 17);
    OWNER.with_borrow_mut(|owner| {
        *owner = Some(Owner {
            values,
            _funding: funding.clone(),
        })
    });
    assert!(matches!(
        funding.reserve_metadata(1),
        Err(WorkspaceMetadataFundingError::Capacity { .. })
    ));
    Ok(())
}
struct Active;
impl Drop for Active {
    fn drop(&mut self) {
        ENABLED.set(false);
        OWNER.with_borrow_mut(|owner| *owner = None);
        FAIL_AT.set(None);
    }
}
#[test]
fn exact_source_constructor_uses_reset_comparison_and_retains_escaped_partial_custody() {
    ENABLED.set(true);
    let _active = Active;
    let (mut runtime, data, source, required) = fixture(None, 3);
    let pool = data.borrow().pool.clone();
    let error = runtime
        .reset_admitted(SessionResetLimits::new(source + required - 1))
        .unwrap_err();
    assert!(OWNER.with_borrow(Option::is_none));
    assert_eq!(FILLS.get(), 0);
    assert_eq!(pool.used_bytes().unwrap(), source);
    drop(error);
    FAIL_AT.set(Some(1));
    let error = runtime
        .reset_admitted(SessionResetLimits::new(source + required))
        .unwrap_err();
    let typed = std::error::Error::source(&error)
        .unwrap()
        .downcast_ref::<ResidentResetError<State>>()
        .unwrap();
    assert_eq!(typed.initialized_count(), 1);
    assert_eq!(typed.retained_bytes(), required);
    assert!(
        data.borrow()
            .state
            .layers
            .slots()
            .iter()
            .all(|value| value.position == 19)
    );
    drop(runtime);
    drop(data);
    drop(error);
    assert!(pool.used_bytes().unwrap() >= required);
    OWNER.with_borrow(|owner| assert_eq!(owner.as_ref().unwrap().values, vec![17; PAYLOAD]));
    OWNER.with_borrow_mut(|owner| *owner = None);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
