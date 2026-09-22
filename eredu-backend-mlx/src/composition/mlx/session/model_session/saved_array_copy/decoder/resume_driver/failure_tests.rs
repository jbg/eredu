use super::*;
use eredu_core::GenerationCancellationToken;
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::composition::mlx) enum Point {
    AfterPrompt,
    AfterSampling,
    AfterExchange,
    BeforeSamplingReadiness,
}
#[derive(Debug, thiserror::Error)]
#[error("injected native resume failure at {0:?}")]
pub(in crate::composition::mlx) struct InjectedResumeFailure(pub Point);
pub(in crate::composition::mlx) enum Action {
    Error,
    Unwind,
    Cancel(GenerationCancellationToken),
}
thread_local! { static FAULT: RefCell<Option<(Point, Action, Rc<Cell<bool>>)>> = const { RefCell::new(None) }; }
pub(in crate::composition::mlx) struct FaultGuard(Rc<Cell<bool>>);
impl FaultGuard {
    pub(in crate::composition::mlx) fn reached(&self) -> bool {
        self.0.get()
    }
}
impl Drop for FaultGuard {
    fn drop(&mut self) {
        FAULT.with(|slot| {
            slot.borrow_mut().take();
        });
    }
}
pub(in crate::composition::mlx) fn fault(point: Point, action: Action) -> FaultGuard {
    let fired = Rc::new(Cell::new(false));
    FAULT.with(|slot| {
        assert!(slot.borrow().is_none());
        slot.replace(Some((point, action, fired.clone())));
    });
    FaultGuard(fired)
}
pub(in crate::composition::mlx::session) fn checkpoint(
    point: Point,
    runtime: &ModelRuntime<MlxBackend<'_>>,
) -> Result<(), Error> {
    let active = FAULT.with(|slot| {
        let mut slot = slot.borrow_mut();
        if slot.as_ref().is_some_and(|(p, _, _)| *p == point) {
            slot.take()
        } else {
            None
        }
    });
    let Some((_, action, fired)) = active else {
        return Ok(());
    };
    fired.set(true);
    runtime
        .session()
        .authority
        .borrow()
        .require_idle()
        .expect("native lease must retire before the shared readiness boundary");
    match action {
        Action::Error => Err(Error::Other(Box::new(InjectedResumeFailure(point)))),
        Action::Unwind => std::panic::panic_any(InjectedResumeFailure(point)),
        Action::Cancel(token) => {
            token.cancel();
            Ok(())
        }
    }
}

pub(in crate::composition::mlx) fn healthy(runtime: &ModelRuntime<MlxBackend<'_>>) -> bool {
    runtime.session().ensure_healthy().is_ok()
}
pub(in crate::composition::mlx) fn escaped_installed_array(
    runtime: &ModelRuntime<MlxBackend<'_>>,
) -> Array {
    runtime
        .session()
        .payload
        .model
        .erased()
        .retained_decoder_state_storage()
        .unwrap()
        .into_retained_arrays()
        .unwrap()
        .next()
        .expect("installed nonempty decoder backing")
}
