use super::*;
use eredu_runtime::prefill::ChunkedPrefillRequest;

#[derive(Clone)]
struct Request {
    fail: Rc<Cell<bool>>,
}

impl
    ChunkedPrefillRequest<
        OrdinaryTextFixture,
        FakeBackend,
        DeviceState<FakeBackend, FakeLayerState>,
    > for Request
{
    type Continuation = ();
    fn token_count(&self) -> usize {
        3
    }
    fn maximum_chunk_tokens(&self) -> usize {
        1
    }
    fn with_chunk<R>(
        &self,
        range: std::ops::Range<usize>,
        _: Option<&()>,
        operation: impl for<'a> FnOnce(&'a FakeTensor) -> R,
    ) -> Result<R, Error> {
        if self.fail.get() {
            return Err(Error::backend("injected chunk preparation failure"));
        }
        Ok(operation(&FakeTensor(vec![range.start as i32 + 3])))
    }
    fn request_values(&self) -> Vec<&FakeTensor> {
        vec![]
    }
    fn continuation(&self, _: &()) {}
    fn retained_values<'a>(&self, _: &'a ()) -> Vec<&'a FakeTensor> {
        vec![]
    }
}

#[test]
fn distributed_prefill_cursor_rejections_preserve_committed_prefix_and_retry() {
    let coordinator = Arc::new(ConcurrentCheckpointCoordinator::new());
    std::thread::scope(|scope| {
        let workers = (0..2)
            .map(|rank| {
                let coordinator = Arc::clone(&coordinator);
                scope.spawn(move || {
                    let make_session = || {
                        partitioned_agreement_session(
                            rank,
                            LocalPartitionFailure::None,
                            TestPhaseAgreement::Concurrent(ConcurrentCheckpointAgreement {
                                rank,
                                coordinator: Arc::clone(&coordinator),
                            }),
                            true,
                        )
                    };
                    let (mut session, counters, _) = make_session();
                    let fail = Rc::new(Cell::new(false));
                    let request = Request {
                        fail: Rc::clone(&fail),
                    };
                    let rejected = if rank == 0 {
                        Err(Error::backend("request failure"))
                    } else {
                        Ok(request.clone())
                    };
                    assert!(session.start_prefill(rejected, 1, None, &()).is_err());
                    assert_eq!(counters.snapshot().forward_calls, 0);
                    let mut cursor = session
                        .start_prefill(Ok(request.clone()), 1, None, &())
                        .unwrap();
                    let mut stale = cursor.clone();
                    session
                        .advance_prefill(&mut cursor, &(), &mut eredu_runtime::NoopObserver)
                        .unwrap();
                    assert_eq!(cursor.position(), 1);
                    let baseline = session.report().unwrap().state_report().to_vec();
                    let forwards = counters.snapshot().forward_calls;
                    // One rank tries replaying an old cursor after a successful commit.
                    let wrong = if rank == 0 { &mut stale } else { &mut cursor };
                    assert!(session
                        .advance_prefill(wrong, &(), &mut eredu_runtime::NoopObserver)
                        .is_err());
                    assert_eq!(counters.snapshot().forward_calls, forwards);
                    assert_eq!(
                        session.report().unwrap().state_report(),
                        baseline.as_slice()
                    );
                    // A rank-local preparation error must meet the peers' normal input vote.
                    fail.set(rank == 0);
                    assert!(session
                        .advance_prefill(&mut cursor, &(), &mut eredu_runtime::NoopObserver)
                        .is_err());
                    assert_eq!(cursor.position(), 1);
                    assert_eq!(counters.snapshot().forward_calls, forwards);
                    assert_eq!(
                        session.report().unwrap().state_report(),
                        baseline.as_slice()
                    );
                    fail.set(false);
                    // A local post-execution observation failure rolls every rank back.
                    let mut failing = FailingFinalOutputObserver;
                    let mut passing = eredu_runtime::NoopObserver;
                    let observer: &mut dyn eredu_runtime::ActivationObserver<FakeTensor, Error> =
                        if rank == 0 {
                            &mut failing
                        } else {
                            &mut passing
                        };
                    assert!(session.advance_prefill(&mut cursor, &(), observer).is_err());
                    assert_eq!(cursor.position(), 1);
                    assert_eq!(
                        session.report().unwrap().state_report(),
                        baseline.as_slice()
                    );
                    let forwards = counters.snapshot().forward_calls;

                    // A cursor from another session is rejected by all participants.
                    let (mut other, _, _) = make_session();
                    let mut foreign = other.start_prefill(Ok(request), 1, None, &()).unwrap();
                    let wrong = if rank == 0 { &mut foreign } else { &mut cursor };
                    assert!(session
                        .advance_prefill(wrong, &(), &mut eredu_runtime::NoopObserver)
                        .is_err());
                    assert_eq!(cursor.position(), 1);
                    assert_eq!(counters.snapshot().forward_calls, forwards);
                    session
                        .finish_prefill(&mut cursor, &(), &mut eredu_runtime::NoopObserver)
                        .unwrap();
                    assert_eq!(cursor.position(), 3);
                    assert!(cursor.is_complete());
                    assert_eq!(counters.snapshot().forward_calls, forwards + 2);
                })
            })
            .collect::<Vec<_>>();
        for worker in workers {
            worker.join().unwrap();
        }
    });
}
