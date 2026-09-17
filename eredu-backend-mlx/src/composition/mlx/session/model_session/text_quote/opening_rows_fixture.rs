//! Test-only route selection for the private original admission entry.
//!
//! The actual core driver still creates/binds its context, prepares prompt and
//! sampler, installs the returned bank and advances native work. This selector
//! supplies no capability, bound, reservation, context or completion evidence.
use super::*;
use eredu_core::capture::SharedCapturePlan;
use std::rc::Weak;

struct Slot {
    session: Weak<Cell<bool>>,
    source: eredu_core::SharedStorageIdentity,
    calls: Cell<usize>,
    quote: RefCell<Option<TextExecutionQuoteOwner>>,
}
thread_local! {
    static CURRENT: RefCell<Option<Rc<Slot>>> = const { RefCell::new(None) };
}
struct Selection(Rc<Slot>);
impl Selection {
    fn new(runtime: &ModelRuntime<MlxBackend<'_>>, source: &SharedCapturePlan) -> Self {
        let slot = Rc::new(Slot {
            session: Rc::downgrade(&runtime.session().poison),
            source: source.storage_identity().clone(),
            calls: Cell::new(0),
            quote: RefCell::new(None),
        });
        CURRENT.with(|current| {
            let mut current = current.borrow_mut();
            assert!(current.is_none(), "private admission fixture cannot nest");
            *current = Some(slot.clone());
        });
        Self(slot)
    }
    fn take_quote(&self) -> TextExecutionQuoteOwner {
        self.0
            .quote
            .borrow_mut()
            .take()
            .expect("actual returned quote")
    }
}
impl Drop for Selection {
    fn drop(&mut self) {
        let removed = CURRENT.with(|current| current.borrow_mut().take());
        debug_assert!(
            removed
                .as_ref()
                .is_some_and(|slot| Rc::ptr_eq(slot, &self.0))
        );
        // Any retained quote and its resources retire outside the TLS loan.
        drop(removed);
    }
}
fn selected(
    runtime: &ModelRuntime<MlxBackend<'_>>,
    source: &SharedCapturePlan,
) -> Option<Rc<Slot>> {
    CURRENT.with(|current| {
        current
            .borrow()
            .as_ref()
            .filter(|slot| {
                slot.source == *source.storage_identity()
                    && slot
                        .session
                        .upgrade()
                        .is_some_and(|session| Rc::ptr_eq(&session, &runtime.session().poison))
            })
            .cloned()
    })
}
/// Select only the genuine private function. All production validation inside
/// that function, and all later shared-machine readiness, remain mandatory.
pub(in crate::composition::mlx::session::model_session) fn admit_if_selected<
    C: TokenFilterController,
>(
    runtime: &ModelRuntime<MlxBackend<'_>>,
    input: &TextPreparationInput<'_, MlxModelInput>,
    config: TextGenerationConfig,
    controller: &C,
    source: &SharedCapturePlan,
) -> Option<Result<(InferenceTextPreparation, TextExecutionQuoteOwner), BackendFailure>> {
    let slot = selected(runtime, source)?;
    slot.calls.set(slot.calls.get().checked_add(1).unwrap());
    assert!(
        slot.quote.borrow().is_none(),
        "consume the previous observed quote first"
    );
    let result = admit_with_capture_opening_rows(runtime, input, config, controller, source);
    if let Ok((_, quote)) = &result {
        *slot.quote.borrow_mut() = Some(quote.clone());
    }
    Some(result)
}

#[cfg(all(feature = "metal", target_vendor = "apple", not(feature = "cuda")))]
mod tests;
