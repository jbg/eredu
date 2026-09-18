//! Scoped fixed-frame capacity beside unchanged cumulative heap funding.
use std::{
    alloc::Layout,
    cell::Cell,
    mem::{size_of, size_of_val},
    rc::Rc,
    marker::PhantomData,
};

mod private {
    pub trait Sealed {}
    impl<E, F: Fn(usize) -> Result<(), E>> Sealed for F {}
    impl Sealed for crate::ParserAllocationFunding {}
    impl<E, F: super::PreparedFunding<Error = E>> Sealed for super::Scope<'_, F> {}
}

/// Original prepared-parser allocation callback. Ordinary callbacks retain
/// their cumulative behavior; only the parser's private operation scope loans
/// reusable fixed frames. This supplies no parser or execution authority.
pub trait PreparedFunding: private::Sealed {
    /// Concrete failure from the exact borrowed allocation callback.
    type Error;
    /// Prospectively pays an actual allocation, without reusable credit.
    fn reserve(&self, bytes: usize) -> Result<(), Self::Error>;
    /// Internal fixed-frame entry; returned guards retain all live overlap.
    #[doc(hidden)]
    fn frame(&self, bytes: usize) -> Result<Frame<'_>, FrameError<Self::Error>>;
}
impl<E, F: Fn(usize) -> Result<(), E>> PreparedFunding for F {
    type Error = E;
    fn reserve(&self, bytes: usize) -> Result<(), E> {
        self(bytes)
    }
    fn frame(&self, bytes: usize) -> Result<Frame<'_>, FrameError<E>> {
        self(frame_control_bytes::<E>(bytes).ok_or(FrameError::Overflow)?).map_err(FrameError::Funding)?;
        Ok(Frame {
            _scope: PhantomData,
            state: None,
            bytes: 0,
        })
    }
}
impl PreparedFunding for crate::ParserAllocationFunding {
    type Error = crate::ParserAllocationFailure;
    fn reserve(&self, bytes: usize) -> Result<(), Self::Error> {
        crate::ParserAllocationFunding::reserve(self, bytes)
    }
    fn frame(&self, bytes: usize) -> Result<Frame<'_>, FrameError<Self::Error>> {
        self.reserve(frame_control_bytes::<Self::Error>(bytes).ok_or(FrameError::Overflow)?)
            .map_err(FrameError::Funding)?;
        Ok(Frame { _scope: PhantomData, state: None, bytes: 0 })
    }
}
/// Fixed failure; the enclosing parser retains its original payer and prefix.
#[doc(hidden)]
#[derive(Debug)]
pub enum FrameError<E> {
    Overflow,
    Funding(E),
}
#[derive(Debug, Default)]
struct State {
    live: Cell<usize>,
    capacity: Cell<usize>,
}
/// A live fixed-frame loan. Private construction prevents external reuse.
/// The guard borrows its scope, which in turn borrows the original payer.
///
/// ```compile_fail
/// use derivre::prepared_funding::{PreparedFunding, Scope};
/// let payer = derivre::ParserAllocationFunding::unenforced();
/// let scope = Scope::new(&payer).unwrap();
/// let guard = scope.frame(64).unwrap();
/// drop(scope); // live guard must not outlive the paid scope shell
/// drop(guard);
/// ```
///
/// ```compile_fail
/// use derivre::prepared_funding::{PreparedFunding, Scope};
/// let payer = derivre::ParserAllocationFunding::unenforced();
/// let scope = Scope::new(&payer).unwrap();
/// let guard = scope.frame(64).unwrap();
/// drop(payer); // the same original payer remains borrowed throughout
/// drop(guard);
/// drop(scope);
/// ```
#[doc(hidden)]
pub struct Frame<'a> {
    _scope: PhantomData<&'a ()>,
    state: Option<Rc<State>>,
    bytes: usize,
}
impl Drop for Frame<'_> {
    fn drop(&mut self) {
        if let Some(state) = &self.state {
            state.live.set(
                state
                    .live
                    .get()
                    .checked_sub(self.bytes)
                    .expect("live frame loan"),
            );
        }
    }
}
/// Exact fixed-frame entry and guard controls; inspection grants no funding.
#[doc(hidden)]
pub fn frame_control_bytes<E>(bytes: usize) -> Option<usize> {
    let parts = [
        bytes,
        size_of::<Frame<'_>>(),
        size_of::<FrameError<E>>(),
        size_of::<Result<Frame<'_>, FrameError<E>>>(),
        size_of::<Option<Rc<State>>>(),
        size_of::<(&State, usize, usize, usize)>(),
        size_of::<Result<(), E>>(),
    ];
    parts
        .into_iter()
        .try_fold(size_of_val(&parts), usize::checked_add)
}
/// Lives only inside one actual parser or source-inspection operation. The callback borrow fixes
/// both its concrete error type and payer; no credits survive parser copies.
#[doc(hidden)]
pub struct Scope<'a, F> {
    funding: &'a F,
    state: Rc<State>,
}
impl<'a, F> Scope<'a, F> {
    /// Opens one paid operation scope; children borrow this same live scope.
    pub fn new<E>(funding: &'a F) -> Result<Self, FrameError<E>>
    where
        F: PreparedFunding<Error = E>,
    {
        let shell = Layout::new::<[Cell<usize>; 2]>()
            .extend(Layout::new::<State>())
            .ok()
            .map(|(layout, _)| layout.pad_to_align().size())
            .ok_or(FrameError::Overflow)?;
        let parts = [
            shell,
            size_of::<Self>(),
            size_of::<State>(),
            size_of::<Rc<State>>(),
            size_of::<Result<Self, FrameError<E>>>(),
            size_of::<FrameError<E>>(),
            size_of::<Result<(), E>>(),
            size_of::<(&F, Layout)>(),
            frame_control_bytes::<E>(0).ok_or(FrameError::Overflow)?,
        ];
        funding
            .reserve(
                parts
                    .into_iter()
                    .try_fold(size_of_val(&parts), usize::checked_add)
                    .ok_or(FrameError::Overflow)?,
            )
            .map_err(FrameError::Funding)?;
        Ok(Self {
            funding,
            state: Rc::new(State::default()),
        })
    }
}
impl<E, F: PreparedFunding<Error = E>> PreparedFunding for Scope<'_, F> {
    type Error = E;
    fn reserve(&self, bytes: usize) -> Result<(), E> {
        self.funding.reserve(bytes)
    }
    fn frame(&self, bytes: usize) -> Result<Frame<'_>, FrameError<E>> {
        let bytes = frame_control_bytes::<E>(bytes).ok_or(FrameError::Overflow)?;
        let live = self
            .state
            .live
            .get()
            .checked_add(bytes)
            .ok_or(FrameError::Overflow)?;
        if live > self.state.capacity.get() {
            self.funding
                .reserve(live - self.state.capacity.get())
                .map_err(FrameError::Funding)?;
            self.state.capacity.set(live);
        }
        self.state.live.set(live);
        Ok(Frame {
            _scope: PhantomData,
            state: Some(Rc::clone(&self.state)),
            bytes,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::panic::{catch_unwind, AssertUnwindSafe};

    #[test]
    fn sequential_frames_reuse_capacity_but_recursive_frames_and_heap_growth_pay() {
        let requests = Cell::new(0usize);
        let spent = Cell::new(0usize);
        let funding = |bytes| {
            requests.set(requests.get() + 1);
            spent.set(spent.get().checked_add(bytes).unwrap());
            Ok::<(), &'static str>(())
        };
        let scope = Scope::new(&funding).unwrap();
        let constructor = requests.get();
        let outer = scope.frame(1_000).unwrap();
        let child = scope.frame(2_000).unwrap();
        assert_eq!(requests.get(), constructor + 2);
        let peak = spent.get();
        drop(child);
        for _ in 0..100 {
            let child = scope.frame(2_000).unwrap();
            assert_eq!(spent.get(), peak);
            drop(child);
        }
        // Heap allocations remain cumulative while an outer frame is live.
        scope.reserve(31).unwrap();
        scope.reserve(31).unwrap();
        assert_eq!(spent.get(), peak + 62);
        drop(outer);
        assert_eq!(scope.state.live.get(), 0);
        assert!(catch_unwind(AssertUnwindSafe(|| {
            let _outer = scope.frame(1_000).unwrap();
            let _child = scope.frame(2_000).unwrap();
            panic!("injected traversal unwind");
        }))
        .is_err());
        assert_eq!(scope.state.live.get(), 0);
        let before = requests.get();
        drop(scope.frame(2_500).unwrap());
        assert_eq!(requests.get(), before);
        // A different operation/payer never inherits this scope's peak.
        let other_calls = Cell::new(0);
        let other = |_| {
            other_calls.set(other_calls.get() + 1);
            Ok::<(), &'static str>(())
        };
        let other_scope = Scope::new(&other).unwrap();
        drop(other_scope.frame(1_000).unwrap());
        assert_eq!(other_calls.get(), 2);
        assert_eq!(requests.get(), before);
    }

    #[test]
    fn refused_frame_growth_keeps_existing_live_loans_and_exact_callback_error() {
        let calls = Cell::new(0usize);
        let refuse = Cell::new(false);
        let funding = |_| {
            calls.set(calls.get() + 1);
            if refuse.get() {
                Err("first refused frame growth")
            } else {
                Ok(())
            }
        };
        let scope = Scope::new(&funding).unwrap();
        let parent = scope.frame(1_000).unwrap();
        let live = scope.state.live.get();
        let capacity = scope.state.capacity.get();
        refuse.set(true);
        assert!(matches!(
            scope.frame(2_000),
            Err(FrameError::Funding("first refused frame growth"))
        ));
        assert_eq!(scope.state.live.get(), live);
        assert_eq!(scope.state.capacity.get(), capacity);
        drop(parent);
        let before = calls.get();
        drop(scope.frame(1_000).unwrap());
        assert_eq!(calls.get(), before);
        assert_eq!(scope.state.live.get(), 0);
        // No new scope can be constructed after its shell's refusal.
        assert!(matches!(
            Scope::new(&funding),
            Err(FrameError::Funding("first refused frame growth"))
        ));
    }
}
