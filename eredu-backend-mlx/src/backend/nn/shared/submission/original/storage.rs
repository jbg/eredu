//! Cold requested layouts and actual reserves; no population/authority issuer.
use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct NeuralSubmissionShape {
    arrays: usize,
    consumers: usize,
}
impl NeuralSubmissionShape {
    pub(crate) fn new(arrays: usize, consumers: usize) -> Option<Self> {
        consumers.checked_add(1)?;
        Layout::array::<Array>(arrays).ok()?;
        Layout::array::<PreparedArrayClone>(arrays).ok()?;
        Layout::array::<Option<OriginalRetirementCleanup>>(consumers + 1).ok()?;
        Some(Self { arrays, consumers })
    }
    pub(crate) fn arrays(self) -> usize {
        self.arrays
    }
    pub(crate) fn consumers(self) -> usize {
        self.consumers
    }
    /// Actual final C handles constructed by `PreparedNeuralSubmission::try_new`.
    /// Already included in its control quote; no descriptor or Graph/Record fit.
    pub(crate) fn outer_array_handle_bytes(self) -> Option<usize> {
        self.arrays
            .checked_mul(Array::inspection_clone_handle_bytes())
    }
    /// The selected local traversal never calls owning host retention. Arbitrary
    /// T requires a separate prepaid typed owner, not a guessed Vec capacity.
    pub(crate) fn host_resources(self) -> usize {
        0
    }
}

#[derive(Debug)]
pub(crate) enum SubmissionPreparationCause {
    Overflow,
    Reserve {
        site: &'static str,
        cause: TryReserveError,
    },
    NativeClone {
        index: usize,
        cause: Exception,
    },
}
pub(crate) struct SubmissionPreparationError<C: 'static, P: Observer> {
    pub(crate) cause: SubmissionPreparationCause,
    pub(crate) pending: Option<PreparedNeuralSubmission<C, P>>,
    pub(crate) controls: C,
}
impl<C: 'static, P: Observer> std::fmt::Debug for SubmissionPreparationError<C, P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SubmissionPreparationError")
            .field("cause", &self.cause)
            .field("has_prepared_prefix", &self.pending.is_some())
            .finish()
    }
}

// Rust 1.98 alloc::rc::RcInner is repr(C, align(2)): two Cell<usize> counts
// followed by T. This is a pinned implementation layout, not a stable-Rust ABI.
fn rc_layout<T>() -> Option<usize> {
    let header = Layout::new::<[Cell<usize>; 2]>()
        .align_to(2)
        .ok()?
        .pad_to_align();
    Some(
        header
            .extend(Layout::new::<T>())
            .ok()?
            .0
            .pad_to_align()
            .size(),
    )
}

impl<C: Clone + 'static, P: Observer> PreparedNeuralSubmission<C, P> {
    /// Exact requested Rust and cold native clone storage plus named controls.
    /// Caller adds native event/root/stream facts, dynamic C/P/error
    /// children, allocator/TLS infrastructure and its actual producer controls.
    pub(crate) fn control_bytes(shape: NeuralSubmissionShape) -> Option<u64> {
        let cleanup_count = shape.consumers.checked_add(1)?;
        let arrays = Layout::array::<Array>(shape.arrays)
            .ok()?
            .size()
            .checked_add(
                Layout::array::<PreparedArrayClone>(shape.arrays)
                    .ok()?
                    .size(),
            )?
            .checked_add(
                shape
                    .arrays
                    .checked_mul(PreparedArrayClone::control_bytes()?)?,
            )?
            .checked_add(shape.outer_array_handle_bytes()?)?
            .checked_add(
                Layout::array::<ConsumerSlot<C, P>>(shape.consumers)
                    .ok()?
                    .size(),
            )?
            .checked_add(
                Layout::array::<Option<OriginalRetirementCleanup>>(cleanup_count)
                    .ok()?
                    .size(),
            )?;
        let fixed = [
            size_of::<Self>(),
            size_of::<Option<Self>>(),
            size_of::<OriginalNeuralSubmissionCompletion<C, P>>(),
            size_of::<Option<OriginalNeuralSubmissionCompletion<C, P>>>(),
            size_of::<Rc<Shared<C>>>(),
            size_of::<SubmissionRetention<C>>(),
            size_of::<Payload<C>>(),
            size_of::<ConsumerSlot<C, P>>(),
            size_of::<Option<Active<C, P>>>(),
            size_of::<C>(),
            size_of::<P>(),
            size_of::<RefCell<Vec<ConsumerSlot<C, P>>>>(),
            size_of::<SubmissionPreparationError<C, P>>(),
            size_of::<Result<Self, SubmissionPreparationError<C, P>>>(),
            size_of::<TryReserveError>(),
            size_of::<Result<(), TryReserveError>>(),
            size_of::<std::cell::RefMut<'static, Vec<ConsumerSlot<C, P>>>>(),
            size_of::<std::cell::RefMut<'static, Option<OrdinaryRetirement<Payload<C>>>>>(),
            size_of::<std::cell::Ref<'static, Option<OrdinaryRetirement<Payload<C>>>>>(),
            size_of::<ConsumerUnwind<'static>>(),
            size_of::<Observation>(),
            size_of::<OriginalSubmissionOwner<C, P>>(),
            size_of::<OriginalSubmissionFailure<C, P>>(),
            size_of::<
                Result<OriginalNeuralSubmissionCompletion<C, P>, OriginalSubmissionFailure<C, P>>,
            >(),
            size_of::<OriginalArraySubmissionCause>(),
            size_of::<OriginalArraySubmissionFailure<C, P>>(),
            size_of::<
                Result<
                    OriginalNeuralSubmissionCompletion<C, P>,
                    OriginalArraySubmissionFailure<C, P>,
                >,
            >(),
            size_of::<OriginalObservationSite>(),
            size_of::<OriginalObservationFailure>(),
            size_of::<Result<bool, OriginalObservationFailure>>(),
            size_of::<usize>(),
            size_of::<Option<&Array>>(),
            size_of::<&[Array]>(),
            size_of::<Result<Observation, P::Error>>(),
        ]
        .into_iter()
        .try_fold(arrays, usize::checked_add)?
        .checked_add(rc_layout::<Shared<C>>()?)?;
        let nodes = u64::try_from(cleanup_count)
            .ok()?
            .checked_mul(Ready::<C, P>::control_bytes::<P::Error>()?)?;
        nodes.checked_add(OrdinaryRetirement::<Payload<C>>::control_bytes()?)?
            .checked_add(CompletedObservedRetention::<SubmissionRetention<C>, C, P>::release_with_control_bytes::<fn(&mut SubmissionRetention<C>)>()?)?
            .checked_add(u64::try_from(fixed).ok()?)
    }

    #[cfg(test)]
    pub(crate) fn fail_next_reserve_for_test(site: usize) {
        assert!(site < 4);
        super::tests::FAIL_RESERVE.with(|failure| {
            assert!(failure.replace(Some(site)).is_none());
        });
    }

    /// Reserve four finite buffers, every final clone handle and every node before activation.
    /// Actual reserve errors own the accepted prefix and original custody. Box
    /// and Rc construction retain their ordinary process-abort OOM contracts.
    pub(crate) fn try_new(
        shape: NeuralSubmissionShape,
        controls: C,
    ) -> Result<Self, SubmissionPreparationError<C, P>> {
        if Self::control_bytes(shape).is_none() {
            return Err(SubmissionPreparationError {
                cause: SubmissionPreparationCause::Overflow,
                pending: None,
                controls,
            });
        }
        let mut prepared = Self {
            shape,
            shared: Rc::new(Shared {
                payload: RefCell::new(Some(OrdinaryRetirement::new(Payload {
                    arrays: Vec::new(),
                    event: None,
                    traversal: None,
                    nested: false,
                    clone_slots: Vec::new(),
                    cleanups: Vec::new(),
                    #[cfg(test)]
                    witness: None,
                    _custody: controls.clone(),
                }))),
                children: Cell::new(0),
                failed: Cell::new(false),
            }),
            primary: Some(Ready::new(controls.clone())),
            consumers: Vec::new(),
            controls: controls.clone(),
        };
        let result = (|| {
            {
                let mut payload = prepared.shared.payload.borrow_mut();
                let payload = payload.as_mut().expect("prepared payload");
                payload
                    .arrays
                    .try_reserve_exact(reserve_count(0, shape.arrays))
                    .map_err(|cause| SubmissionPreparationCause::Reserve {
                        site: "arrays",
                        cause,
                    })?;
                payload
                    .cleanups
                    .try_reserve_exact(reserve_count(1, shape.consumers + 1))
                    .map_err(|cause| SubmissionPreparationCause::Reserve {
                        site: "cleanup slots",
                        cause,
                    })?;
                for _ in 0..=shape.consumers {
                    payload.cleanups.push(None);
                }
                payload
                    .clone_slots
                    .try_reserve_exact(reserve_count(3, shape.arrays))
                    .map_err(|cause| SubmissionPreparationCause::Reserve {
                        site: "clone slots",
                        cause,
                    })?;
                for index in 0..shape.arrays {
                    let slot = PreparedArrayClone::try_new().map_err(|cause| {
                        SubmissionPreparationCause::NativeClone { index, cause }
                    })?;
                    payload.clone_slots.push(slot);
                }
            }
            prepared
                .consumers
                .try_reserve_exact(reserve_count(2, shape.consumers))
                .map_err(|cause| SubmissionPreparationCause::Reserve {
                    site: "consumer slots",
                    cause,
                })?;
            for _ in 0..shape.consumers {
                prepared
                    .consumers
                    .push(ConsumerSlot::Ready(Ready::new(controls.clone())));
            }
            Ok(())
        })();
        match result {
            Ok(()) => Ok(prepared),
            Err(cause) => Err(SubmissionPreparationError {
                cause,
                pending: Some(prepared),
                controls,
            }),
        }
    }
}
fn reserve_count(site: usize, count: usize) -> usize {
    #[cfg(test)]
    if super::tests::FAIL_RESERVE.with(|fail| fail.get() == Some(site)) {
        super::tests::FAIL_RESERVE.with(|fail| fail.set(None));
        // The real existing Vec call returns CapacityOverflow, not simulated
        // allocator exhaustion. Source shape/prefix remains the requested one.
        return usize::MAX;
    }
    let _ = site;
    count
}
