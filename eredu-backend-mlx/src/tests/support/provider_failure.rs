//! Thread-local, one-shot failures inside actual grouped mechanism calls.
use std::{
    cell::{Cell, RefCell},
    marker::PhantomData,
    rc::Rc,
};

use safemlx::{Array, Stream};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Operator {
    Gated,
    Relu2,
    Linear,
}

thread_local! {
    static ARMED: Cell<Option<(Operator, usize)>> = const { Cell::new(None) };
    static HITS: Cell<usize> = const { Cell::new(0) };
    static CAUSE: RefCell<Option<(String, String, u32)>> = const { RefCell::new(None) };
}

pub(crate) struct Guard(PhantomData<Rc<()>>);
impl Drop for Guard {
    fn drop(&mut self) {
        ARMED.set(None);
    }
}

pub(crate) fn arm(operator: Operator, preceding_calls: usize) -> Guard {
    assert!(ARMED.get().is_none(), "provider failure already armed");
    ARMED.set(Some((operator, preceding_calls)));
    HITS.set(0);
    CAUSE.set(None);
    Guard(PhantomData)
}

pub(crate) fn hits() -> usize {
    HITS.get()
}

pub(crate) fn cause() -> (String, String, u32) {
    CAUSE.with_borrow(|cause| cause.clone().expect("native failure was injected"))
}

pub(crate) fn check(operator: Operator, stream: &Stream) -> Result<(), eredu_nn::Error> {
    let Some((selected, remaining)) = ARMED.get() else {
        return Ok(());
    };
    if selected != operator {
        return Ok(());
    }
    if remaining != 0 {
        ARMED.set(Some((selected, remaining - 1)));
        return Ok(());
    }
    ARMED.set(None);
    HITS.set(HITS.get() + 1);
    // Obtain a real MLX exception, before this grouped call submits any work.
    // Earlier units and capture reservations still follow their normal lifecycle.
    let source = Array::from_slice(&[1.0f32, 2.0], &[2])
        .reshape(&[3], stream)
        .expect_err("invalid native reshape must fail");
    CAUSE.set(Some((
        source.what().into(),
        source.location().file().into(),
        source.location().line(),
    )));
    Err(eredu_nn::Error::backend_source(source))
}
